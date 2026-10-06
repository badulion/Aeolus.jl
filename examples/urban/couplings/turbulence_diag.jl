# Turbulence diagnostics (Aeolus.TurbulenceStability, Era VI P58-a, #170).
#
# At the output cadence: the pointwise squared buoyancy frequency N² (of θ_v,
# Operators.buoyancy_frequency_squared!), the strain-rate magnitude |S|² and
# the gradient Richardson number Ri_g (Operators.gradient_richardson!), and
# the stability factor f_m(Ri_g) of a TurbulenceStability family (default
# the LEM / MONC functions of Brown et al. 1994). Their means over the fluid
# cells of each level go to the NetCDF file (as z profiles) and to the log.
#
# P59-c (#180) extends it with the transition diagnostics of
# Aeolus.Transition (there is one diagnostic addon): the turbulent fraction
# γ(z) of the vorticity-fluctuation indicator, the fraction of fluid cells
# with Ri_g > 1/4 (of the same Ri_g field) and the grid Reynolds number
# U Δ/(ν + ⟨ν_t⟩).
#
# P59-e (#195) adds the relaminarised share of the stable layer
# (`Transition.relaminarised_fraction`): among the fluid cells with
# Ri_g > 1/4, those that are γ-laminar and carry ν_t/ν < 1e-2 (ν_t the
# larger of the active closure's horizontal and vertical viscosity). In
# `:winter` before sunrise this is the relaminarisation diagnostic of the
# stable night-time layer. Its smoke check asserts, for a laminar-consistent
# closure (`Operators.laminar_consistent`: the σ-model, `:sigma`), that ν_t
# vanishes identically in a laminar (parallel) flow on the Berlin mesh.
#
# A diagnostic: it changes no state, so `budget!` is a no-op.

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
    error("turbulence_diag: unknown stability family :$name (known: :brown, :lilly, :short_tail, :long_tail, :most)")
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

Relaminarisation (P59-e, #195): `relam` per level and `relam_frac`
over the whole stable layer — the fraction of the fluid cells with
``Ri_g > 1/4`` that are γ-laminar (the `γ` indicator below, normalised
by the option `relam_normalization`, default `:volume`) **and** have
``ν_t/ν <`` `relam_νt_ratio` (default ``10^{-2}``), with ``ν_t`` the larger
of the active SGS closure's ``ν_h`` and ``ν_v`` (`Operators.sgs_fields`)
and ``ν`` = `cfg.ν` (`Transition.relaminarised_fraction`); `relam_nstable`
counts the stable cells.

Transition diagnostics (`Aeolus.Transition`, P59-c #180), per level over
the fluid cells: the turbulent fraction `γ` (`turbulent_indicator!` with
options `gamma_threshold`, default 0.2, and `gamma_normalization`,
default `:plane`; solid cells are never turbulent), the supercritical
fraction `Rsup` of cells with ``Ri_g > 1/4`` (`supercritical_fraction`
of the `Rif` field above; no second Ri) and the grid Reynolds number
`Re_grid` ``= |U_h| Δ/(ν + ⟨ν_t⟩)`` with ``|U_h|`` the plane-mean
horizontal wind speed, ``Δ = (Δx Δy Δz)^{1/3}`` and `cfg.ν`.

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
    gamma_threshold::Float64
    gamma_normalization::Symbol
    ind::Array{Bool,3}
    γ::Vector{Float64}
    Rsup::Vector{Float64}
    Re_grid::Vector{Float64}
    relam_νt_ratio::Float64
    relam_normalization::Symbol
    lam::Array{Bool,3}
    νmax::Array{Float64,3}
    relam::Vector{Float64}
    relam_frac::Float64
    relam_nstable::Int
    t_eval::Float64
    ncalls::Int
    seconds::Float64
end

coupling_name(::Type{<:TurbulenceDiag}) = :turbulence_diag
modules_used(::Type{<:TurbulenceDiag}) = (:TurbulenceStability, :Transition)
default_options(::Type{<:TurbulenceDiag}) =
    (stability = :brown, gamma_threshold = 0.2, gamma_normalization = :plane,
     relam_νt_ratio = 1e-2, relam_normalization = :volume)
stages(::Type{<:TurbulenceDiag}) = ((:output, 10, Cadence(:output)),)
smoke_variants(::Type{<:TurbulenceDiag}) =
    ((label = "turbulence_diag, MOST-consistent stability",
      options = Dict(:turbulence_diag => (stability = :most,))),)

function build(::Type{<:TurbulenceDiag}, case, o)
    m = case.mesh
    sm = case.geom.sm.mask
    fluid = [!sm[m.Hx+i, m.Hy+j, m.Hz+k] for i in 1:m.Nx, j in 1:m.Ny, k in 1:m.Nz]
    cc() = Field(CellCenter(), m)
    p = case.sim                       # the core's own conditions (periodic or open, #262)
    z() = zeros(m.Nz)
    return TurbulenceDiag(turbulence_stability_family(o.stability),
                          Field(FaceX(), m), Field(FaceY(), m), Field(FaceZ(), m), cc(), cc(),
                          cc(), cc(), cc(), p.bcs, p.θ_bcs, fluid,
                          z(), z(), z(), z(), Float64(o.gamma_threshold),
                          Symbol(o.gamma_normalization), zeros(Bool, m.Nx, m.Ny, m.Nz),
                          z(), z(), z(), Float64(o.relam_νt_ratio),
                          Symbol(o.relam_normalization), zeros(Bool, m.Nx, m.Ny, m.Nz), zeros(m.Nx, m.Ny, m.Nz), z(), NaN, 0,
                          NaN, 0, 0.0)
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
    fill_halos!(c.θ, c.θ_bcs)
    fill_halos!(c.qv, Aeolus.BoundaryConditions.passive_scalar_bcs(c.θ_bcs))   # not θ's top gradient
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
    # transition diagnostics (Aeolus.Transition)
    turbulent_indicator!(c.ind, c.u, c.v, c.w; threshold = c.gamma_threshold,
                         fluid = c.fluid, normalization = c.gamma_normalization)
    c.γ .= turbulent_fraction(c.ind; fluid = c.fluid)
    c.Rsup .= supercritical_fraction(c.Rif; fluid = c.fluid)
    transition_grid_reynolds!(c, case)
    relaminarisation!(c, case)
    c.t_eval = sim.t
    c.ncalls += 1
    c.seconds += time() - t0
    return c
end

"""
    transition_grid_reynolds!(c::TurbulenceDiag, case)

Per level, `Re_grid = |U_h| Δ/(ν + ⟨ν_t⟩)` (`grid_reynolds_number`) from
the fluid-cell plane means of the horizontal wind speed at the cell
centre and of `sim.νt`, with `Δ = (Δx Δy Δz)^{1/3}` and `cfg.ν`.
"""
function transition_grid_reynolds!(c::TurbulenceDiag, case)
    m = case.mesh
    Δ = cbrt(m.dx * m.dy * m.dz)
    ud, vd = c.u.data, c.v.data
    νt = interior_view(case.sim.νt)
    @inbounds for k in 1:m.Nz
        n = 0; sU = 0.0; sν = 0.0
        for j in 1:m.Ny, i in 1:m.Nx
            c.fluid[i, j, k] || continue
            I, J, K = m.Hx + i, m.Hy + j, m.Hz + k
            uc = 0.5 * (ud[I, J, K] + ud[I+1, J, K]); vc = 0.5 * (vd[I, J, K] + vd[I, J+1, K])
            n += 1; sU += sqrt(uc^2 + vc^2); sν += νt[i, j, k]
        end
        c.Re_grid[k] = n > 0 ? grid_reynolds_number(sU / n, Δ, case.cfg.ν, sν / n) : NaN
    end
    return c
end

"""
    active_sgs_closure(case) -> closure or nothing

The SGS closure the case hands the core (`sgs` of the step keywords of the
`:sgs` provider), or `nothing` for a provider that uses the legacy
`eddy_viscosity!` hook (`:mlsgs`).
"""
active_sgs_closure(case) = get(case.step_kwargs, :sgs, nothing)

"""
    relaminarisation!(c::TurbulenceDiag, case)

The relaminarised share of the stable layer (`relam`, `relam_frac`,
`relam_nstable`): `Transition.relaminarised_fraction` of the γ-laminar
fluid cells (`laminar_mask!` with the option `relam_normalization`,
default `:volume`: the vorticity fluctuation against the domain-mean
``|ω|``, so that a quiet, shear-free free stream is laminar — with the
plane normalisation of `γ` it is not, as the indicator is scale-free), the
`Rif` field and the
larger of the closure's ``ν_h``, ``ν_v`` (`Operators.sgs_fields`; `sim.νt`
without a contract closure), against `cfg.ν`.
"""
function relaminarisation!(c::TurbulenceDiag, case)
    sim = case.sim
    cl = active_sgs_closure(case)
    νh = interior_view(sim.νt); νv = νh
    if cl isa AbstractSGSClosure
        f = sgs_fields(cl, sim)
        νh = interior_view(f.νh); νv = interior_view(f.νv)
    end
    @inbounds for I in eachindex(c.νmax)
        c.νmax[I] = max(νh[I], νv[I])
    end
    if c.relam_normalization === c.gamma_normalization
        @inbounds for I in eachindex(c.lam)
            c.lam[I] = c.fluid[I] && !c.ind[I]
        end
    else
        laminar_mask!(c.lam, c.u, c.v, c.w; threshold = c.gamma_threshold, fluid = c.fluid,
                      normalization = c.relam_normalization)
    end
    r = relaminarised_fraction(c.lam, c.Rif, c.νmax, case.cfg.ν; fluid = c.fluid,
                               νt_ratio = c.relam_νt_ratio)
    c.relam .= r.profile
    c.relam_frac = r.fraction
    c.relam_nstable = r.nstable
    return c
end

"""
    laminar_flow_νt(closure, case) -> (νh_max, νv_max)

The closure's eddy viscosity in a laminar flow on the Berlin mesh: a copy
of the closure (`deepcopy`, so the case's closure state is untouched) on a
scratch core whose wind is the current plane-mean profile ``(U(z), V(z),
0)`` — a parallel, turning shear flow — and whose θ is uniform. A
laminar-consistent closure gives exactly 0.
"""
function laminar_flow_νt(closure, case)
    m = case.mesh; sim = case.sim; cfg = case.cfg
    # a horizontally uniform flow: a laterally periodic scratch core, also
    # for an open (#262) or nested (#265) case
    pcfg = deepcopy(cfg); pcfg.lateral = :periodic; pcfg.physics[:nesting] = false
    s = boussinesq_core(pcfg, m)
    fill!(s.θ.data, cfg.θ₀)
    ui = interior_view(sim.u); vi = interior_view(sim.v)
    for k in 1:m.Nz
        K = m.Hz + k
        s.u.data[:, :, K] .= mean(view(ui, :, :, k)); s.v.data[:, :, K] .= mean(view(vi, :, :, k))
    end
    fill!(s.w.data, 0.0)
    cl = deepcopy(closure)
    hasproperty(cl, :qv) && cl.qv !== nothing && fill!(cl.qv.data, 0.0)
    f = compute_sgs!(cl, s)
    return maximum(abs, interior_view(f.νh)), maximum(abs, interior_view(f.νv))
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
                :z, cs -> turbulence_diagnostics!(c, cs).fm),
    OutputField("gamma_turb", "1",
                "turbulent fraction of the fluid cells (vorticity-fluctuation indicator)",
                :z, cs -> turbulence_diagnostics!(c, cs).γ),
    OutputField("Ri_supercritical_fraction", "1",
                "fraction of the fluid cells with Ri_g > 1/4",
                :z, cs -> turbulence_diagnostics!(c, cs).Rsup),
    OutputField("Re_grid", "1",
                "grid Reynolds number |U_h| Delta / (nu + <nu_t>)",
                :z, cs -> turbulence_diagnostics!(c, cs).Re_grid),
    OutputField("relam_fraction", "1",
                "fraction of the Ri_g > 1/4 fluid cells that are gamma-laminar with nu_t/nu < 1e-2",
                :z, cs -> turbulence_diagnostics!(c, cs).relam)]

"""
    transition_summary(c::TurbulenceDiag, case) -> NamedTuple

Canopy and free-stream summaries of the transition diagnostics: the mean
`γ`, `Rsup` and `Re_grid` over the levels with buildings (`canopy`) and
over the levels above the highest roof + 1 and below the sponge (`free`).
"""
function transition_summary(c::TurbulenceDiag, case)
    turbulence_diagnostics!(c, case)
    m = case.mesh
    kr = highest_solid_level(case)
    ksp = floor(Int, case.cfg.sponge_frac * m.Nz)
    lvl(x, ks) = (ks = [k for k in ks if isfinite(x[k])]; isempty(ks) ? NaN : sum(x[ks]) / length(ks))
    can = 1:kr; fr = kr+2:ksp
    return (γ_canopy = lvl(c.γ, can), γ_free = lvl(c.γ, fr),
            Rsup_canopy = lvl(c.Rsup, can), Rsup_free = lvl(c.Rsup, fr),
            Re_canopy = lvl(c.Re_grid, can), Re_free = lvl(c.Re_grid, fr),
            Rsup_all = lvl(c.Rsup, 1:m.Nz))
end

function report(c::TurbulenceDiag, case)
    turbulence_diagnostics!(c, case)
    dz = case.mesh.dz
    ks = unique(clamp.(round.(Int, [10.0, 30.0, 50.0, 70.0, 90.0, 150.0, 300.0] ./ dz .+ 0.5),
                       1, case.mesh.Nz))
    prof = join([@sprintf("z=%.0f: N2=%.2e Ri=%.3g fm=%.3f", (k - 0.5) * dz, c.N2[k], c.Ri[k], c.fm[k])
                 for k in ks], "; ")
    t = transition_summary(c, case)
    return @sprintf("  turbulence (%s, %.2f ms/call): %s\n  transition: γ canopy %.3f free %.3f; Ri_g>1/4 fraction canopy %.3f free %.3f; Re_grid canopy %.1f free %.1f\n  relaminarised: %.1f %% of the %d stable cells (Ri_g > 1/4) are γ-laminar with ν_t/ν < %.0e (ν = %.3g m²/s)",
                    nameof(typeof(c.stability)), 1e3 * c.seconds / max(c.ncalls, 1), prof,
                    t.γ_canopy, t.γ_free, t.Rsup_canopy, t.Rsup_free, t.Re_canopy, t.Re_free,
                    100 * c.relam_frac, c.relam_nstable, c.relam_νt_ratio, case.cfg.ν)
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
    # z_i is that of the initial profile (`initial_inversion_height`: the
    # idealised θ_inversion_z, or the parcel z_i of the sounding, #247).
    # When it lies above the box (the observed noon profile is mixed to
    # ~1.2 km) there is no stable layer to test, and the check says so.
    cfg = case.cfg
    zi = initial_inversion_height(cfg)
    zs = [(k - 0.5) * case.mesh.dz for k in 1:case.mesh.Nz]
    above = [k for k in eachindex(zs)
             if zi + case.mesh.dz < zs[k] < cfg.sponge_frac * cfg.extent[3]]
    inv = isempty(above) ?
        (@sprintf("no capping inversion inside the box (initial z_i = %.0f m above %.0f m)",
                  zi, cfg.sponge_frac * cfg.extent[3]) => zi + case.mesh.dz >= cfg.sponge_frac * cfg.extent[3]) :
        ("stable above the capping inversion: plane-mean N² > 0, Ri_g > 0, f_m < 1" =>
             all(k -> c.N2[k] > 0 && c.Ri[k] > 0 && c.fm[k] < 1, above))
    return ["f_m ∈ [0, 1] where Ri_g ≥ 0 (fluid cells)" => fm_ok,
            "Ri_g finite in fluid cells" => fin,
            inv,
            "f_m = 1 at Ri = 0" => stability_factor_m(c.stability, 0.0) == 1,
            "0 ≤ γ ≤ 1 on every level" => all(x -> 0 <= x <= 1, c.γ),
            "γ = 0 in solids (no solid cell is turbulent)" => !any(c.ind .& .!fl),
            "0 ≤ fraction of Ri_g > 1/4 ≤ 1" => all(x -> 0 <= x <= 1, c.Rsup),
            "grid Reynolds number finite and positive" => all(x -> isfinite(x) && x > 0, c.Re_grid),
            relam_checks(c, case)...]
end

# P59-e: the relaminarised fraction is a fraction of the stable cells; and a
# laminar-consistent closure gives ν_t ≡ 0 in a laminar flow on this mesh.
function relam_checks(c::TurbulenceDiag, case)
    ok = (c.relam_nstable == 0 ? isnan(c.relam_frac) : 0 <= c.relam_frac <= 1) &&
         all(x -> isnan(x) || 0 <= x <= 1, c.relam)
    checks = Pair{String,Bool}["0 ≤ relaminarised fraction of the stable layer ≤ 1" => ok]
    cl = active_sgs_closure(case)
    if cl isa AbstractSGSClosure && Aeolus.Operators.laminar_consistent(cl)
        νh, νv = laminar_flow_νt(cl, case)
        push!(checks, "laminar-consistent closure: ν_t ≡ 0 in the parallel plane-mean flow" =>
                      νh == 0 && νv == 0)
    end
    return checks
end

register!(TurbulenceDiag)
