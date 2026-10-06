# The prognostic SGS TKE closure (Aeolus.PrognosticSGS, Era VI P56-b, #182)
# in the `:sgs` slot, provider `:tke` (the default).
#
# Deardorff (1980) with the Moeng (1984) coefficients: ν_t = c_k l √e,
# κ_t = (1 + 2l/Δ) ν_t, l = min(Δ, 0.76 √e/N, κ d_wall), ε = c_ε e^{3/2}/l.
# It is PALM's urban-LES default and the only closure here whose mixing
# responds prognostically to stratification (the stable nights).
#
# How it is wired:
#
#   * the closure object is the `sgs` of every dynamics step
#     (`step_options`), so every core and time stepper runs it through the
#     P54-a contract: the coefficients from e, one commit of the sources
#     (shear production — the exact dual of the SGS momentum operator —,
#     buoyancy production with the θ_v of the qv tracer, the implicit
#     sinks) per accepted step, and an exact rollback of a rejected
#     adaptive step;
#   * e is the registered tracer `:e` (`advect = true, positive = true`):
#     the `:transport` stage advects it with the case's scheme and
#     positivity limiter (W-weighted on a sound-proof core) and diffuses it
#     with 2ν_t (`diffusivity`), mask-aware; the closure itself runs with
#     `transport = false`, so e is transported once;
#   * the mixing length is capped at κ · `IBM.wall_distance` (floor, roofs
#     and facades) and c_ε gets Moeng's 3.9 at the first level above a
#     surface (`IBM.first_level_cells`); e = 0 in the buildings;
#   * the first air cell of every column (`bus.k_air`) holds the surface
#     value e_s = u*²/√(c_k c_ε) with the friction velocity of the surface
#     exchange (`bus.u_star`, MOST): no second log law here.
#
# A TKE closure has its own stability length scale, so no TurbulenceStability
# stability function (nor the StabilityCorrected wrapper of P55-e, #187) is
# applied on top of it.

"""
    SGSTKE

Slot `:sgs`, provider `:tke` (default): `PrognosticSGS.TKEClosure`, the
Deardorff (1980) / Moeng (1984) prognostic-TKE closure, passed as the `sgs`
of every step, with the SGS TKE as the advected, positive tracer `:e`
(m² s⁻², written to the NetCDF file) diffused with `2ν_t` by the
`:transport` stage.

Options: `wall_cap = true` (cap ``l`` at ``κ d_{wall}``, `IBM.wall_distance`),
`surface = true` (the first air cell takes ``e_s = u_*^2/\\sqrt{c_k c_ε}``
from `bus.u_star`), `e_min = 1e-4` (m² s⁻²: the floor of the initial
local-equilibrium ``e``; ``e = 0`` is a fixed point of the TKE equation).
"""
struct SGSTKE{C, F} <: AbstractCoupling
    closure::C
    K2::F                       # 2ν_t, the diffusivity of :e in the :transport stage
    e_min::Float64
end
coupling_name(::Type{<:SGSTKE}) = :sgs_tke
slot(::Type{<:SGSTKE}) = :sgs
provider(::Type{<:SGSTKE}) = :tke
is_default(::Type{<:SGSTKE}) = true
modules_used(::Type{<:SGSTKE}) = (:PrognosticSGS, :IBM)
default_options(::Type{<:SGSTKE}) = (wall_cap = true, surface = true, e_min = 1e-4)

function build(::Type{<:SGSTKE}, case, o)
    m, cfg, sm = case.mesh, case.cfg, case.geom.sm
    K2 = Field(CellCenter(), m)
    diffusivity = cs -> (K2.data .= 2 .* cs.sim.νt.data; K2)
    e = ensure_tracer!(case, :e; advect = true, positive = true, units = "m2 s-2",
                       long_name = "subgrid-scale turbulent kinetic energy",
                       diffusivity = diffusivity)
    closure = TKEClosure(m; e = e, mask = sm.mask,
                         wall_distance = o.wall_cap ? wall_distance(sm; bottom = true, bcs = case.sim.bcs) : nothing,
                         first_level = first_level_cells(sm; bottom = true),
                         surface = o.surface ? (u_star = case.bus.u_star, k = case.bus.k_air) : nothing,
                         θ₀ = cfg.θ₀, g_accel = cfg.g, qv = tracer(case, :qv),
                         transport = false)
    return SGSTKE(closure, K2, Float64(o.e_min))
end

# the local equilibrium of the initial flow (at least e_min) once the
# geometry and the initial state exist
init!(c::SGSTKE, case) = (initialize_tke!(c.closure, case.sim; e_min = c.e_min); nothing)

# The Laplacian SGS momentum form (the cores' default `sgs_stress`), not the
# stress form of the other providers (#175): the shear production is the
# exact dual of the Laplacian operator, so Σ P_s V balances the resolved-KE sink.
step_options(c::SGSTKE, case) = (sgs = c.closure,)

# The SGS TKE carries no surface energy and no water: nothing for the ledger
# (its dissipation heating is neglected, as with every closure of the case).
budget!(c::SGSTKE, case, l) = nothing

"""
    tke_profile(c::SGSTKE, case) -> (canopy, free, νt_canopy)

Mean SGS TKE (m² s⁻²) over the fluid cells of the canopy levels (more than
5 % of the level solid) and of the levels above them up to the sponge, and
the mean ``ν_t`` of the canopy levels. (A few towers of the flagship reach
the lid, so "below the highest roof" would be the whole domain.)
"""
function tke_profile(c::SGSTKE, case)
    m = case.mesh
    ksp = floor(Int, case.cfg.sponge_frac * m.Nz)
    solid = view(case.geom.sm.mask, m.Hx+1:m.Hx+m.Nx, m.Hy+1:m.Hy+m.Ny, m.Hz+1:m.Hz+m.Nz)
    kr = something(findlast(k -> count(view(solid, :, :, k)) > 0.05 * m.Nx * m.Ny, 1:m.Nz), 0)
    e = interior_view(c.closure.e); ν = interior_view(case.sim.νt)
    lvl(a, ks) = (sel = [(i, j, k) for k in ks for j in 1:m.Ny for i in 1:m.Nx if !solid[i, j, k]];
                  isempty(sel) ? NaN : mean(a[i, j, k] for (i, j, k) in sel))
    return (canopy = lvl(e, 1:kr), free = lvl(e, kr+1:ksp), νt_canopy = lvl(ν, 1:kr))
end

"""
    tke_model_stress(c::SGSTKE, case) -> (τ, e)

The subfilter stress the closure implies at the cell centres,
``τ_{ij} = \\tfrac23 e δ_{ij} − 2ν_t \\bar S_{ij}`` (current ``ν_t``, the
centred resolved strain), as a `SymmetricTensorField`, and the interior
``e`` (P56-e, #192: the a-priori metrics and the `II_sgs` output).
"""
function tke_model_stress(c::SGSTKE, case)
    m = case.mesh
    τ = SymmetricTensorField(m)
    eddy_viscosity_stress!(τ, _sgs_centre_strain(case), Array(interior_view(case.sim.νt)))
    e = Array(interior_view(c.closure.e))
    τ.xx .+= 2 .* e ./ 3; τ.yy .+= 2 .* e ./ 3; τ.zz .+= 2 .* e ./ 3
    return τ, e
end

function report(c::SGSTKE, case)
    p = tke_profile(c, case)
    st = tke_statistics(c.closure)
    τ, e = tke_model_stress(c, case)
    return @sprintf("SGS TKE: mean e canopy %.3f, free stream %.3f m²/s²; mean ν_t canopy %.3f m²/s; %d clips",
                    p.canopy, p.free, p.νt_canopy, st.clips) *
           "\n    SGS TKE " * sgs_apriori_line(sgs_test_filter_metrics(case, τ, e))
end

# e is the registered tracer :e (written already); II of the implied stress
output_fields(c::SGSTKE, case) =
    [OutputField("II_sgs", "1",
                 "second invariant of the anisotropy of the implied SGS stress 2/3 e I - 2 nu_t S (0 in solids)",
                 :xyz, cs -> replace(lumley_invariants(first(tke_model_stress(c, cs)))[1], NaN => 0.0))]

function smoke_check(c::SGSTKE, case)
    sim, m = case.sim, case.mesh
    cl = c.closure
    e = interior_view(cl.e)
    solid = view(case.geom.sm.mask, m.Hx+1:m.Hx+m.Nx, m.Hy+1:m.Hy+m.Ny, m.Hz+1:m.Hz+m.Nz)
    checks = ["e ≥ 0 and ν_t ≥ 0" => minimum(e) >= 0 && minimum(interior_view(sim.νt)) >= 0,
              "e = 0 in the buildings" => all(e[solid] .== 0),
              "κ_t ≥ ν_t" => all(interior_view(sim.κt) .>= interior_view(sim.νt)),
              "e > 0 in the air (the closure is active)" => maximum(e) > 0,
              "TKE equation advanced once per accepted step" => tke_statistics(cl).steps > 0]
    # P56-e (#192): the a-priori metrics and the II output
    τ, ee = tke_model_stress(c, case)
    a = sgs_test_filter_metrics(case, τ, ee)
    II = lumley_invariants(τ)[1]
    push!(checks, "test-filter energy k_test > 0 and e/k_test finite (a priori)" =>
                      a.k_test > 0 && isfinite(a.ratio) && a.ratio > 0)
    push!(checks, "implied-stress II ≥ 0 where e > 0" => all(x -> isnan(x) || x >= -1e-12, II))
    if sim isa BoussinesqSimulation
        # Σ P_s V is the resolved-KE sink of the SGS momentum operator
        b = kinetic_energy_budget(sim; sgs = cl)
        tb = tke_budget(cl, sim)
        push!(checks, "Σ P_s V = −(resolved-KE SGS sink) to 1e-8" =>
                          b.sgs < 0 && abs(tb.shear + b.sgs) <= 1e-8 * abs(b.sgs))
    end
    return checks
end

smoke_variants(::Type{<:SGSTKE}) =
    ((label = "TKE without the wall cap and the surface value",
      options = Dict(:sgs_tke => (wall_cap = false, surface = false))),)

register!(SGSTKE)
