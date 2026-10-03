# Turbulence diagnostics (Aeolus.TurbulenceStability, Era VI P58-a, #170).
#
# At the output cadence: the pointwise squared buoyancy frequency N² (of θ_v,
# Operators.buoyancy_frequency_squared!), the strain-rate magnitude |S|² and
# the gradient Richardson number Ri_g (Operators.gradient_richardson!), and
# the stability factor f_m(Ri_g) of a TurbulenceStability family (default
# the LEM / MONC functions of Brown et al. 1994). Their means over the fluid
# cells of each level go to the NetCDF file (as z profiles) and to the log.
#
# A diagnostic: it changes no state, so `budget!` is a no-op. P59-c (#180)
# extends this addon with the transition diagnostics, so there is one
# diagnostic addon.

"""
    turbulence_stability_family(name::Symbol) -> AbstractRiStability

The `stability` option of the `turbulence_diag` coupling: `:brown`
(`BrownEtAl1994`, default), `:lilly` (`LillyCutoff`), `:short_tail`
(`ShortTail`), `:long_tail` (`LongTail`) or `:most`
(`SurfaceExchange.MOSTConsistent` with the Businger–Dyer–Beljaars–Holtslag
functions of the surface exchange).
"""
function turbulence_stability_family(name::Symbol)
    name === :brown      && return BrownEtAl1994()
    name === :lilly      && return LillyCutoff()
    name === :short_tail && return ShortTail()
    name === :long_tail  && return LongTail()
    name === :most       && return MOSTConsistent(BusingerDyerBeljaarsHoltslag())
    error("turbulence_diag: unknown stability family :$name " *
          "(known: :brown, :lilly, :short_tail, :long_tail, :most)")
end

"""
    TurbulenceDiag

Additive coupling (on by default), stage `:output`: the pointwise
``N^2``, ``|S|^2``, ``Ri_g = N^2/\\max(|S|^2, S^2_{\\min})`` and
``f_m(Ri_g)`` of the resolved flow, and their fluid-cell plane means
`N2`, `S2`, `fm` per level. The level's Richardson number `Ri` is the
ratio of the plane means, ``⟨N^2⟩/⟨|S|^2⟩`` (the pointwise ratio is
dominated by the shear-free cells, where the floor makes it huge).
Option `stability` (see [`turbulence_stability_family`](@ref)).

The velocities, θ and q_v are copied into scratch fields whose halos are
filled with the core's boundary conditions, so the diagnostic does not
touch the model state. `ncalls` / `seconds` time the evaluation.
"""
mutable struct TurbulenceDiag{S<:AbstractRiStability, FU, FV, FW, FC, B, BT} <: AbstractCoupling
    stability::S
    u::FU; v::FV; w::FW; θ::FC; qv::FC
    N2f::FC; S2f::FC; Rif::FC
    bcs::B; θ_bcs::BT
    fluid::Array{Bool,3}
    N2::Vector{Float64}
    S2::Vector{Float64}
    Ri::Vector{Float64}
    fm::Vector{Float64}
    t_eval::Float64
    ncalls::Int
    seconds::Float64
end

coupling_name(::Type{<:TurbulenceDiag}) = :turbulence_diag
modules_used(::Type{<:TurbulenceDiag}) = (:TurbulenceStability,)
default_options(::Type{<:TurbulenceDiag}) = (stability = :brown,)
stages(::Type{<:TurbulenceDiag}) = ((:output, 10, Cadence(:output)),)
smoke_variants(::Type{<:TurbulenceDiag}) =
    ((label = "turbulence_diag, MOST-consistent stability",
      options = Dict(:turbulence_diag => (stability = :most,))),)

function build(::Type{<:TurbulenceDiag}, case, o)
    m = case.mesh
    sm = case.geom.sm.mask
    fluid = [!sm[m.Hx+i, m.Hy+j, m.Hz+k] for i in 1:m.Nx, j in 1:m.Ny, k in 1:m.Nz]
    cc() = Field(CellCenter(), m)
    p = core_parameters(case.cfg)
    z() = zeros(m.Nz)
    return TurbulenceDiag(turbulence_stability_family(o.stability),
                          Field(FaceX(), m), Field(FaceY(), m), Field(FaceZ(), m), cc(), cc(),
                          cc(), cc(), cc(), p.bcs, p.θ_bcs, fluid,
                          z(), z(), z(), z(), NaN, 0, 0.0)
end

"""
    turbulence_diagnostics!(c::TurbulenceDiag, case) -> c

Evaluate the fields and the plane-mean profiles for the current state
(once per simulated time; the output getters, `report` and the stage
share one evaluation).
"""
function turbulence_diagnostics!(c::TurbulenceDiag, case)
    sim = case.sim
    c.t_eval == sim.t && return c
    t0 = time()
    copyto!(c.u.data, sim.u.data); copyto!(c.v.data, sim.v.data)
    copyto!(c.w.data, sim.w.data); copyto!(c.θ.data, sim.θ.data)
    copyto!(c.qv.data, tracer(case, :qv).data)
    fill_halos!(c.u, c.bcs); fill_halos!(c.v, c.bcs); fill_halos!(c.w, c.bcs)
    fill_halos!(c.θ, c.θ_bcs); fill_halos!(c.qv, c.θ_bcs)
    cfg = case.cfg
    buoyancy_frequency_squared!(c.N2f, c.θ, c.qv; θ₀ = cfg.θ₀, g = cfg.g)
    gradient_richardson!(c.Rif, c.N2f, c.u, c.v, c.w)
    strain_rate_squared!(c.S2f, c.u, c.v, c.w)
    N2 = interior_view(c.N2f); Ri = interior_view(c.Rif); S2 = interior_view(c.S2f)
    Nz = size(N2, 3)
    @inbounds for k in 1:Nz
        n = 0; sN = 0.0; sS = 0.0; sf = 0.0
        for j in axes(N2, 2), i in axes(N2, 1)
            c.fluid[i, j, k] || continue
            n += 1
            sN += N2[i, j, k]; sS += S2[i, j, k]
            sf += stability_factor_m(c.stability, Ri[i, j, k])
        end
        c.N2[k] = n > 0 ? sN / n : NaN
        c.S2[k] = n > 0 ? sS / n : NaN
        c.fm[k] = n > 0 ? sf / n : NaN
        c.Ri[k] = n > 0 ? gradient_richardson(c.N2[k], c.S2[k]) : NaN
    end
    c.t_eval = sim.t
    c.ncalls += 1
    c.seconds += time() - t0
    return c
end

run_stage!(c::TurbulenceDiag, case, ::Val{:output}, dt) =
    (turbulence_diagnostics!(c, case); nothing)

output_fields(c::TurbulenceDiag, case) = [
    OutputField("N2_mean", "s-2",
                "fluid-cell plane mean of the squared buoyancy frequency of theta_v",
                :z, cs -> turbulence_diagnostics!(c, cs).N2),
    OutputField("Ri_g_mean", "1",
                "gradient Richardson number of the plane means, <N2>/<|S|2>",
                :z, cs -> turbulence_diagnostics!(c, cs).Ri),
    OutputField("f_m_mean", "1",
                "fluid-cell plane mean of the momentum stability factor f_m(Ri_g)",
                :z, cs -> turbulence_diagnostics!(c, cs).fm)]

function report(c::TurbulenceDiag, case)
    turbulence_diagnostics!(c, case)
    dz = case.mesh.dz
    ks = unique(clamp.(round.(Int, [10.0, 30.0, 50.0, 70.0, 90.0, 150.0, 300.0] ./ dz .+ 0.5),
                       1, case.mesh.Nz))
    prof = join([@sprintf("z=%.0f: N2=%.2e Ri=%.3g fm=%.3f", (k - 0.5) * dz, c.N2[k], c.Ri[k], c.fm[k])
                 for k in ks], "; ")
    return @sprintf("  turbulence (%s, %.2f ms/call): %s",
                    nameof(typeof(c.stability)), 1e3 * c.seconds / max(c.ncalls, 1), prof)
end

function smoke_check(c::TurbulenceDiag, case)
    c.t_eval = NaN                                      # evaluate the final state
    turbulence_diagnostics!(c, case)
    Ri = interior_view(c.Rif)
    fl = c.fluid
    fm_ok = all(I -> !fl[I] || Ri[I] < 0 ||
                     0 <= stability_factor_m(c.stability, Ri[I]) <= 1, eachindex(Ri))
    fin = all(I -> !fl[I] || isfinite(Ri[I]), eachindex(Ri))
    # θ = θ_s + (z − z_i) Γ above the capping inversion z_i: stably
    # stratified there (below the sponge), N² > 0, Ri_g of the means > 0 and
    # f_m < 1. The θ_v of the vapour lapse (q_v ∝ e^{−z/H_q}) removes about
    # 2/3 of the dry gΓ/θ₀ in the summer state (3.3e-5 against 1.0e-4 s⁻²),
    # and below z_i, where θ is uniform, it makes N² < 0.
    cfg = case.cfg
    zs = [(k - 0.5) * case.mesh.dz for k in 1:case.mesh.Nz]
    above = [k for k in eachindex(zs)
             if cfg.θ_inversion_z + case.mesh.dz < zs[k] < cfg.sponge_frac * cfg.extent[3]]
    inv_ok = all(k -> c.N2[k] > 0 && c.Ri[k] > 0 && c.fm[k] < 1, above)
    return ["f_m ∈ [0, 1] where Ri_g ≥ 0 (fluid cells)" => fm_ok,
            "Ri_g finite in fluid cells" => fin,
            "stable above the capping inversion: plane-mean N² > 0, Ri_g > 0, f_m < 1" => inv_ok,
            "f_m = 1 at Ri = 0" => stability_factor_m(c.stability, 0.0) == 1]
end

register!(TurbulenceDiag)
