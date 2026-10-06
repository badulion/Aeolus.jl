# The differential SGS Reynolds-stress closure (Aeolus.PrognosticSGS,
# Era VI P56-d, #188) in the `:sgs` slot, provider `:reynolds_stress` — an
# option: a genuine alternative to the default `:tke` (couplings/sgs_tke.jl),
# with the same module, so no coverage list changes.
#
# Six prognostic stresses τᵢⱼ with exact production, Daly–Harlow buoyancy,
# exact Coriolis and the LRR-IP pressure–strain; ε, l and ν_t are the TKE
# closure's functions. How it is wired:
#
#   * the closure object is the `sgs` of every dynamics step
#     (`step_options`): the cores take ν_t, the Daly–Harlow tensor K = c_θ τ
#     (θ through the P55-c tensor operator) and the momentum −∂ⱼτᵈᵉᵛᵢⱼ
#     (the P55-b stress operator) from the committed τ, and commit the
#     source step once per accepted step (`advance_sgs!`: implicit sources,
#     exact Coriolis, realizability projection, booked), rolled back with the
#     state when step_adaptive! rejects a step. That commit is the source
#     stage: a case stage could not be rolled back;
#   * the stresses are the registered tracers `:tau11 … :tau23`
#     (`advect = true`; the diagonals `positive = true`, so flux-limited):
#     the `:transport` stage advects them with the case's scheme and
#     diffuses them with 2ν_t (`diffusivity`), mask-aware; the closure runs
#     with `transport = false`, so they are transported once;
#   * l is capped at κ · `IBM.wall_distance` and c_ε gets Moeng's 3.9 at the
#     first level above a surface, as for `:tke`; τ = 0 in the buildings, and
#     no stress acts on an edge that touches a building or the ground (the
#     wall stress is the surface exchange's).
#
# Stated exceptions: the other tracers (qv, pollutants, …) are diffused by
# the `:transport` stage with the isotropic κ_t = tr K/3 (`sim.κt`), not the
# full tensor; `step_imex!` cannot run the closure (incompatible with
# `timestepper = :imex`).
#
# The first air cell of every column (`bus.k_air`) holds the LRR-IP
# constant-stress-layer value: τ₁₃, τ₂₃ = −u*²(cos α, sin α) against the
# cell's wind, e_s = u*²/0.3395, with the friction velocity of the MOST
# surface exchange (`bus.u_star`) — the counterpart of the `:tke` surface
# value, no second log law here.

const RSM_TRACERS = (:tau11, :tau22, :tau33, :tau12, :tau13, :tau23)

"""
    SGSReynoldsStress

Slot `:sgs`, provider `:reynolds_stress` (an option; the default is `:tke`):
`PrognosticSGS.ReynoldsStressClosure`, the Deardorff (1973) differential
Reynolds-stress closure with the LRR-IP pressure–strain, passed as the
`sgs` of every step, with the six stresses as the advected tracers
`:tau11 … :tau23` (m² s⁻²) transported by the `:transport` stage.

Options: `wall_cap = true` (cap ``l`` at ``κ d_{wall}``), `surface = true`
(the first air cell takes the constant-stress-layer stresses from
`bus.u_star`), `e_min = 1e-4`
(m² s⁻²: the floor of the initial isotropic local-equilibrium ``e``).
"""
struct SGSReynoldsStress{C, F} <: AbstractCoupling
    closure::C
    K2::F                       # 2ν_t, the diffusivity of the stresses in :transport
    e_min::Float64
end
coupling_name(::Type{<:SGSReynoldsStress}) = :sgs_reynolds_stress
slot(::Type{<:SGSReynoldsStress}) = :sgs
provider(::Type{<:SGSReynoldsStress}) = :reynolds_stress
is_default(::Type{<:SGSReynoldsStress}) = false
modules_used(::Type{<:SGSReynoldsStress}) = (:PrognosticSGS,)
default_options(::Type{<:SGSReynoldsStress}) = (wall_cap = true, surface = true, e_min = 1e-4)
incompatible(::Type{<:SGSReynoldsStress}) = (:timestepper => :imex,)
incompatible_reason(::Type{<:SGSReynoldsStress}) =
    "step_imex! treats the SGS momentum implicitly as an eddy viscosity and cannot represent " *
    "the Reynolds stress (Operators.sgs_explicit_stress); use timestepper = :adaptive or :fixed"

function build(::Type{<:SGSReynoldsStress}, case, o)
    m, cfg, sm = case.mesh, case.cfg, case.geom.sm
    K2 = Field(CellCenter(), m)
    diffusivity = cs -> (K2.data .= 2 .* cs.sim.νt.data; K2)
    names = ("11", "22", "33", "12", "13", "23")
    τ = ntuple(k -> ensure_tracer!(case, RSM_TRACERS[k]; advect = true, positive = k <= 3,
                                   units = "m2 s-2",
                                   long_name = "subgrid-scale Reynolds stress tau_$(names[k])",
                                   diffusivity = diffusivity), 6)
    # the solid mask with the case's scalar halos (edges across the periodic
    # seam; mirrored at an open face, #262)
    mf = Field(CellCenter(), m)
    mf.data .= sm.mask
    fill_halos!(mf, case_scalar_bcs(cfg))
    closure = ReynoldsStressClosure(m; τ = τ, mask = mf.data .> 0.5,
                                    wall_distance = o.wall_cap ? wall_distance(sm; bottom = true, bcs = case.sim.bcs) : nothing,
                                    first_level = first_level_cells(sm; bottom = true),
                                    surface = o.surface ? (u_star = case.bus.u_star, k = case.bus.k_air) : nothing,
                                    θ₀ = cfg.θ₀, g_accel = cfg.g, qv = tracer(case, :qv),
                                    transport = false)
    return SGSReynoldsStress(closure, K2, Float64(o.e_min))
end

init!(c::SGSReynoldsStress, case) =
    (initialize_reynolds_stress!(c.closure, case.sim; e_min = c.e_min); nothing)

step_options(c::SGSReynoldsStress, case) = (sgs = c.closure,)

# The stresses carry no surface energy and no water (dissipation heating is
# neglected, as with every closure of the case); realizability corrections
# are booked by the closure (reynolds_stress_statistics) and checked below.
budget!(c::SGSReynoldsStress, case, l) = nothing


"""
    rsm_canopy_profile(c::SGSReynoldsStress, case) -> NamedTuple

Over the fluid cells of the canopy levels (more than 5 % of the level
solid): the mean SGS energy `e` (m² s⁻²), the mean second invariant `II`
of the anisotropy ``b_{ij}`` (cells with ``e > 0``), and the ratio
`τ33_τ11` of the mean vertical to the mean streamwise-x normal stress.
"""
function rsm_canopy_profile(c::SGSReynoldsStress, case)
    cells, _ = sgs_canopy_cells(case)
    τv = map(interior_view, c.closure.τ)
    e = 0.0; II = 0.0; n2 = 0; s11 = 0.0; s33 = 0.0
    for (i, j, k) in cells
        τc = ntuple(q -> τv[q][i, j, k], 6)
        ec = (τc[1] + τc[2] + τc[3]) / 2
        e += ec; s11 += τc[1]; s33 += τc[3]
        if ec > 0
            II += stress_anisotropy(τc)[2]; n2 += 1
        end
    end
    n = max(length(cells), 1)
    return (e = e / n, II = II / max(n2, 1), τ33_τ11 = s33 / s11)
end

"""
    rsm_model_stress(c::SGSReynoldsStress, case) -> (τ, e)

The closure's prognostic stresses as a `SymmetricTensorField` (interior)
and ``e = ½ τ_{kk}`` (P56-e, #192: the a-priori metrics and the outputs).
"""
function rsm_model_stress(c::SGSReynoldsStress, case)
    τ = SymmetricTensorField((Array(interior_view(f)) for f in c.closure.τ)...)
    return τ, (τ.xx .+ τ.yy .+ τ.zz) ./ 2
end

function report(c::SGSReynoldsStress, case)
    p = rsm_canopy_profile(c, case)
    st = reynolds_stress_statistics(c.closure)
    τ, e = rsm_model_stress(c, case)
    return @sprintf("SGS Reynolds stress: canopy e %.4f m²/s², II %.4f, τ33/τ11 %.3f; %d realizability corrections",
                    p.e, p.II, p.τ33_τ11, st.clips) *
           "\n    SGS Reynolds stress " * sgs_apriori_line(sgs_test_filter_metrics(case, τ, e))
end

# the six stresses are tracers (written already); e and II of τ
output_fields(c::SGSReynoldsStress, case) =
    [OutputField("e", "m2 s-2", "subgrid-scale turbulent kinetic energy, tr(tau)/2",
                 :xyz, cs -> last(rsm_model_stress(c, cs))),
     OutputField("II_sgs", "1", "second invariant of the anisotropy of the SGS Reynolds stress (0 in solids)",
                 :xyz, cs -> replace(lumley_invariants(first(rsm_model_stress(c, cs)))[1], NaN => 0.0))]

function smoke_check(c::SGSReynoldsStress, case)
    sim, m = case.sim, case.mesh
    cl = c.closure
    τv = map(interior_view, cl.τ)
    solid = view(case.geom.sm.mask, m.Hx+1:m.Hx+m.Nx, m.Hy+1:m.Hy+m.Ny, m.Hz+1:m.Hz+m.Nz)
    psd = all(I -> is_realizable(ntuple(q -> τv[q][I], 6)), eachindex(τv[1]))
    st = reynolds_stress_statistics(cl)
    V = m.dx * m.dy * m.dz
    esgs = sum(I -> solid[I] ? 0.0 : (τv[1][I] + τv[2][I] + τv[3][I]) / 2, eachindex(τv[1])) * V
    checks = ["τ positive semi-definite in every cell (Schumann 1977)" => psd,
              "τ = 0 in the buildings" => all(q -> all(iszero, τv[q][solid]), 1:6),
              "realizability corrections < 1e-3 of the SGS energy" => st.clipped_energy < 1e-3 * esgs,
              "SGS energy > 0 (the closure is active)" => esgs > 0,
              "ν_t ≥ 0" => minimum(interior_view(sim.νt)) >= 0,
              "stresses advanced once per accepted step" => st.steps > 0]
    # P56-e (#192): the a-priori metrics and the outputs (a realizable τ
    # lies in the Lumley triangle, 0 ≤ II ≤ 2/3)
    τm, em = rsm_model_stress(c, case)
    a = sgs_test_filter_metrics(case, τm, em)
    II = lumley_invariants(τm)[1]
    push!(checks, "test-filter energy k_test > 0 and e/k_test finite (a priori)" =>
                      a.k_test > 0 && isfinite(a.ratio) && a.ratio > 0)
    push!(checks, "II of τ in [0, 2/3] (Lumley triangle)" =>
                      all(x -> isnan(x) || -1e-10 <= x <= 2/3 + 1e-10, II))
    push!(checks, "output e = ½ tr τ = 0 in the buildings" => all(iszero, em[solid]))
    if sim isa BoussinesqSimulation
        # the resolved-KE sink of the stress divergence is the SGS production
        # (the dual gradient); they differ by Σ (2/3) e ∇·u V only
        b = kinetic_energy_budget(sim; sgs = cl)
        rb = reynolds_stress_budget(cl, sim)
        push!(checks, "Σ P V = −(resolved-KE SGS sink) to 1e-6" =>
                          abs(rb.shear + b.sgs) <= 1e-6 * max(abs(b.sgs), abs(rb.shear)))
    end
    return checks
end

smoke_variants(::Type{<:SGSReynoldsStress}) =
    ((label = "Reynolds stress without the wall cap and the surface value",
      physics = Dict(:sgs => :reynolds_stress),
      options = Dict(:sgs_reynolds_stress => (wall_cap = false, surface = false))),)

register!(SGSReynoldsStress)
