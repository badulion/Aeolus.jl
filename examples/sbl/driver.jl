# =========================================================================
# Shared stable-boundary-layer (SBL) driver — P55-f (#191)
#
# A horizontally homogeneous, periodic Boussinesq LES of a boundary layer
# over a cooled surface, built only from package physics:
#
#   * the geostrophic pressure gradient is `LargeScaleForcing`'s
#     `GeostrophicPressureGradient` (P58-c, #171), applied with
#     `apply_forcing!` from the step's `body_force!` hook, together with the
#     core's own Coriolis term (`f_coriolis`);
#   * the surface temperature is a `SurfaceTemperatureSeries` (#171), read
#     with `surface_temperature(series, t)` — a prescribed cooling rate is a
#     two-point series;
#   * the surface heat flux is `SurfaceExchange.solve_surface_layer` (the
#     Era V Monin–Obukhov solver), column by column, from the first-level
#     wind and θ: w'θ'_0 = -u_* θ_*;
#   * the surface stress is `WallModels.MOSTWallLaw` (P57-a, #169) with the
#     Obukhov length of that same solve, applied implicitly by
#     `apply_wall_stress!` in every substage (`post_predictor!`); with the
#     same stability functions and L the law returns the solver's u_* bit
#     for bit;
#   * the SGS closure is any `AbstractEddyViscosityClosure`, wrapped in
#     `Operators.StabilityCorrected` (P55-e, #187) unless
#     `sgs_stability = :none`, with the stress-form momentum divergence
#     (`sgs_stress = :symmetric`, P55-b), so that the corrected vertical
#     viscosity reaches the momentum equation;
#   * the top Rayleigh sponge is the core's (`sponge_α_max`,
#     `sponge_z_start`), retargeted at (U_g, V_g) with the core's own
#     `add_sponge!(…; f_ref)` profile, as the Berlin `:geostrophic` coupling
#     does.
#
# The wall-resolved stratified channel of the same validation
# (`stratified_channel_run`, at the end) is the P54-d `channel_case` with
# `bottom_bc = :noslip` and θ walls; P59-e (#195) reuses that builder.
#
# There is no private log law, no private geostrophic forcing and no
# private stability function family: the GABLS1 surface-layer functions
# are `SurfaceExchange.BusingerDyerLogLinear(β_m = 4.8, β_h = 7.8)`
# (`SBL_GABLS1_STABILITY`), the same family the ocean warm layer uses.
# This file only defines functions and constants (prefix `sbl_`/`SBL_`); it is
# included by `examples/sbl/gabls1.jl` and `test/era6_gabls1.jl`, and by
# `examples/sbl/cases.jl` (GABLS2/3/4, P58-d #194).
#
# P58-d (#194) extensions, all generic (no case constants live here):
#   * height- and time-dependent U_g(z, t), V_g(z, t) (any LargeScaleForcing
#     quantity: number, ForcingTable, function), initial profiles θ(z),
#     u(z), v(z), a start time `t_start`, and a case-specific perturbation;
#   * the top sponge retargeted level by level at U_g(z, t) or, for
#     baroclinic cases whose mean wind above the BL evolves under advective
#     tendencies, at the plane mean (`sponge_target = :mean`: only the
#     fluctuations are damped);
#   * the θ budget books the advective tendencies (`adv_heat_applied`);
#   * diagnostics: the mean-momentum budget (`SBLMomentumBudget`), screen-
#     level 2 m θ and 10 m wind from the package ψ functions (`sbl_screen`),
#     tower samples of V_TKE (`Transition.turbulence_velocity_scale`) against
#     the wind speed and the hockey-stick threshold fit (`sbl_hockey_threshold`).
# =========================================================================

using Aeolus
using Aeolus.Fields: set!
using Aeolus.Operators: add_sponge!, total_stress_profile, interior_view, sgs_fields
using Random
using Statistics: mean

"""
    SBL_GABLS1_STABILITY

The GABLS1 surface-layer functions, the package's
`SurfaceExchange.BusingerDyerLogLinear` with ``β_m = 4.8``, ``β_h = 7.8``
(Beare et al. 2006; quoted by Basu & Porté-Agel 2006, JAS 63, 2074, §4a:
"the constants b_m and b_h were set to 4.8 and 7.8").
"""
const SBL_GABLS1_STABILITY = BusingerDyerLogLinear(β_m = 4.8, β_h = 7.8)

mutable struct SBLCase
    sim::Any
    closure::Any
    sgs_stability::Symbol
    forcing::Any                      # AbstractLargeScaleForcing or a Tuple of them
    Ts::Any                           # SurfaceTemperatureSeries (or anything with surface_temperature)
    sf::Any                           # SurfaceExchange stability functions
    z0m::Float64
    z0h::Float64
    U_g::Any                          # number, ForcingTable or (z, t) -> value (LargeScaleForcing)
    V_g::Any
    sponge_target::Symbol             # :geostrophic (U_g(z, t)) or :mean (damp fluctuations only)
    ρcp::Float64                      # ρ c_p for the W m⁻² conversion of w'θ'_0 (J m⁻³ K⁻¹)
    laws::Matrix{MOSTWallLaw}         # the per-column wall law of the current step
    wθ::Matrix{Float64}               # per-column kinematic surface heat flux (K m s⁻¹, upward +)
    ustar::Matrix{Float64}            # per-column u_* of the surface-layer solve
    Lob::Matrix{Float64}              # per-column Obukhov length
    zero_u::Any                       # zero fields for the sponge retargeting
    zero_v::Any
    τx::Float64                       # plane-mean kinematic surface stress the last step applied
    τy::Float64
    heat_applied::Float64             # ∫ ⟨w'θ'⟩_0 dt actually applied (K m)
    adv_heat_applied::Float64         # ∫∫ (∂θ/∂t)_adv dz dt of the prescribed advective tendencies (K m)
    nsteps::Int
    wall_time::Float64
    budget::Any                       # nothing, or an SBLMomentumBudget fed in every substage
    t_surf::Float64                   # the time of the last surface-layer solve (s)
end

"""
    sbl_case(; N = (32, 32, 32), extent = (400, 400, 400), U_g = 8, V_g = 0,
             f = 1.39e-4, z0m = 0.1, z0h = z0m, θ_ml = 265, z_ml = 100,
             Γ = 0.01, θ₀ = 263.5, T_s0 = 265, cooling = 0.25/3600,
             T_s = nothing, base = StaticSmagorinsky(0.1),
             sgs_stability = :brown94, stability = SBL_GABLS1_STABILITY,
             sponge_z = 300, sponge_α = 0.01, perturbation = 0.1,
             z_pert = 50, seed = 1, extra_forcing = (),
             θ_init = nothing, u_init = nothing, v_init = nothing,
             perturb = nothing, t_start = 0, sponge_target = :geostrophic,
             ρcp = 1.3 * 1005) -> SBLCase

Build the SBL LES. Defaults are the GABLS1 protocol (Beare et al. 2006,
BLM 118, 247; set-up as quoted by Basu & Porté-Agel 2006, JAS 63, 2074):
400 m cube, U_g = 8 m s⁻¹, f = 1.39e-4 s⁻¹ (73° N), z0 = 0.1 m,
θ = 265 K to 100 m then +0.01 K m⁻¹, the surface cooled at 0.25 K h⁻¹
from 265 K, log-linear surface functions β_m = 4.8, β_h = 7.8, a Rayleigh
damping layer above 300 m, random θ perturbations of ±`perturbation` K
below `z_pert`. `T_s` (a `SurfaceTemperatureSeries`) overrides the linear
cooling. `θ₀` is the Boussinesq reference temperature of the buoyancy.
`sgs_stability` is `:brown94` (Brown et al. 1994, the Berlin default),
`:lilly`, or `:none` (the uncorrected `base`).

The observed cases (GABLS2/3/4, `examples/sbl/cases.jl`, P58-d #194) use
the remaining keywords:

* `U_g`, `V_g` may be any `LargeScaleForcing` quantity — a number, a
  height-time `ForcingTable` or a function `(z, t) -> value`;
* `θ_init`, `u_init`, `v_init` are initial profiles `z -> value`
  (defaults: the GABLS1 θ, and `U_g(z, t_start)`, `V_g(z, t_start)`);
* `perturb(rng, x, y, z) -> (du, dv, dθ)` replaces the GABLS1 θ
  perturbation (it is then the only perturbation); it is called once per
  field and grid point and only that field's component is used, so the
  three perturbations are independent draws;
* `t_start` is the initial model time (s; the forcing tables are read at
  `sim.t`);
* `sponge_target = :mean` damps only the departures from the plane mean
  in the sponge layer, for cases whose mean wind above the boundary layer
  evolves under advective tendencies (GABLS3); `:geostrophic` relaxes
  toward `U_g(z, t)`;
* `ρcp` converts the kinematic surface heat flux to W m⁻².
"""
function sbl_case(; N = (32, 32, 32), extent = (400.0, 400.0, 400.0),
                  U_g = 8.0, V_g = 0.0, f = 1.39e-4, z0m = 0.1, z0h = z0m,
                  θ_ml = 265.0, z_ml = 100.0, Γ = 0.01, θ₀ = 263.5,
                  T_s0 = 265.0, cooling = 0.25 / 3600, T_s = nothing,
                  base = StaticSmagorinsky(0.1), sgs_stability::Symbol = :brown94,
                  stability = SBL_GABLS1_STABILITY, sponge_z = 300.0, sponge_α = 0.01,
                  perturbation = 0.1, z_pert = 50.0, seed = 1, extra_forcing = (),
                  θ_init = nothing, u_init = nothing, v_init = nothing,
                  perturb = nothing, t_start = 0.0, sponge_target::Symbol = :geostrophic,
                  ρcp = 1.3 * 1005)
    sponge_target in (:geostrophic, :mean) ||
        throw(ArgumentError("sponge_target must be :geostrophic or :mean, got :$sponge_target"))
    Nx, Ny, Nz = N
    mesh = RegularGrid(; size = (Nx, Ny, Nz), extent = Float64.(extent))
    bcs = FieldBoundaryConditions(:periodic, :periodic, :wall)
    sim = BoussinesqSimulation(mesh; θ₀ = θ₀, g_accel = 9.81, f_coriolis = f,
                               sponge_α_max = sponge_α, sponge_z_start = sponge_z,
                               bcs = bcs,
                               θ_bcs = FieldBoundaryConditions(:periodic, :periodic, :noflux))
    rng = Random.Xoshiro(seed)
    θprof = θ_init === nothing ? (z -> z <= z_ml ? θ_ml : θ_ml + Γ * (z - z_ml)) : θ_init
    uprof = u_init === nothing ? (z -> forcing_value(U_g, z, t_start)) : u_init
    vprof = v_init === nothing ? (z -> forcing_value(V_g, z, t_start)) : v_init
    if perturb === nothing
        # the GABLS1 initialisation (unchanged from P55-f: the same random stream)
        set!(sim.θ, (x, y, z) -> θprof(z) +
                                 (z < z_pert ? perturbation * (2rand(rng) - 1) : 0.0))
        set!(sim.u, (x, y, z) -> uprof(z))
        set!(sim.v, (x, y, z) -> vprof(z))
    else
        set!(sim.θ, (x, y, z) -> θprof(z) + perturb(rng, x, y, z)[3])
        set!(sim.u, (x, y, z) -> uprof(z) + perturb(rng, x, y, z)[1])
        set!(sim.v, (x, y, z) -> vprof(z) + perturb(rng, x, y, z)[2])
    end
    set!(sim.w, (x, y, z) -> 0.0)
    # random u, v perturbations are not solenoidal; the flux-form scalar
    # advection would turn their divergence into a θ source (θ ∇·u, of order
    # 10 K s⁻¹ for 0.45 m s⁻¹ noise on a 25 m grid), so the initial velocity
    # is projected once (the core's Chorin projection)
    perturb === nothing || Aeolus.TimeIntegration._project!(sim, 1.0)
    sim.t = t_start
    closure = sgs_stability === :none ? base :
              StabilityCorrected(base, mesh; scheme = _sbl_scheme(sgs_stability))
    Ts = T_s === nothing ? SurfaceTemperatureSeries([0.0, 86400.0], [T_s0, T_s0 - cooling * 86400]) : T_s
    F = (GeostrophicPressureGradient(U_g, V_g), extra_forcing...)
    laws = [MOSTWallLaw(stability, z0m, Inf) for _ in 1:Nx, _ in 1:Ny]
    z = zeros(Nx, Ny)
    zu = Aeolus.Fields.Field(Aeolus.Grids.FaceX(), mesh)
    zv = Aeolus.Fields.Field(Aeolus.Grids.FaceY(), mesh)
    return SBLCase(sim, closure, sgs_stability, F, Ts, stability, Float64(z0m), Float64(z0h),
                   U_g, V_g, sponge_target, Float64(ρcp), laws, copy(z), copy(z),
                   fill(Inf, Nx, Ny), zu, zv, 0.0, 0.0, 0.0, 0.0, 0, 0.0, nothing, Float64(t_start))
end

_sbl_scheme(s::Symbol) = s === :brown94 ? BrownEtAl1994() :
                         s === :lilly   ? LillyCutoff() :
                         s === :long_tail ? LongTail() :
                         s === :short_tail ? ShortTail() :
                         throw(ArgumentError("sgs_stability must be :brown94, :lilly, :long_tail, :short_tail or :none, got :$s"))

# cell-centre wind and θ of the first level of column (i, j)
@inline function _sbl_first_level(sim, i, j)
    m = sim.mesh
    I, J, K = m.Hx + i, m.Hy + j, m.Hz + 1
    uc = 0.5 * (sim.u.data[I, J, K] + sim.u.data[I+1, J, K])
    vc = 0.5 * (sim.v.data[I, J, K] + sim.v.data[I, J+1, K])
    return uc, vc, sim.θ.data[I, J, K]
end

"""
    sbl_surface!(c, t) -> c

Solve Monin–Obukhov similarity (`SurfaceExchange.solve_surface_layer`)
in every column between the first-level wind and θ (at Δz/2) and the
surface temperature `surface_temperature(c.Ts, t)`; store the kinematic
heat flux ``w'θ'_0 = -u_*θ_*``, ``u_*``, ``L``, and the column's
`MOSTWallLaw(sf, z0m, L)` for the momentum.
"""
function sbl_surface!(c::SBLCase, t)
    sim = c.sim
    m = sim.mesh
    z1 = m.dz / 2
    θs = surface_temperature(c.Ts, t)
    c.t_surf = t
    @inbounds for j in 1:m.Ny, i in 1:m.Nx
        uc, vc, θ = _sbl_first_level(sim, i, j)
        sol = solve_surface_layer(c.sf; U = sqrt(uc^2 + vc^2), z_m = z1, θ_a = θ, θ_s = θs,
                                  z0m = c.z0m, z0h = c.z0h)
        c.wθ[i, j] = -sol.u_star * sol.θ_star
        c.ustar[i, j] = sol.u_star
        c.Lob[i, j] = sol.L
        c.laws[i, j] = MOSTWallLaw(c.sf, c.z0m, sol.L)
    end
    return c
end

"""
    sbl_dt(c; cfl = 1.2, c_diff = 0.5) -> Δt

A stable step: the advective limit on Σ|u_i|/Δx_i and the diffusive limit
on the largest eddy viscosity of the last evaluation.
"""
function sbl_dt(c::SBLCase; cfl = 1.2, c_diff = 0.5)
    s = c.sim; g = s.mesh
    ii, jj, kk = Aeolus.Grids.interior(g)
    adv = maximum(abs, view(s.u.data, ii, jj, kk)) / g.dx +
          maximum(abs, view(s.v.data, ii, jj, kk)) / g.dy +
          maximum(abs, view(s.w.data, ii, jj, kk)) / g.dz
    νmax = maximum(view(s.νt.data, ii, jj, kk))
    dif = νmax * (1 / g.dx^2 + 1 / g.dy^2 + 1 / g.dz^2)
    dt = cfl / adv
    dif > 0 && (dt = min(dt, c_diff / dif))
    return dt
end

const _SBL_RK3_WEIGHTS = (1 / 6, 1 / 6, 2 / 3)

"""
    sbl_step!(c, dt) -> c

One SSP-RK3 step (`step_rk3!`). The surface layer is solved once from the
start-of-step state at the mid-step surface temperature; the heat flux is
held over the step (its RK3 weights sum to one, so exactly
``⟨w'θ'_0⟩Δt`` enters the column), the wall law is applied implicitly in
every substage with the start-of-step L. The geostrophic forcing is the
exact step mean of `c.forcing`.
"""
function sbl_step!(c::SBLCase, dt::Real)
    s = c.sim; m = s.mesh
    t0 = s.t
    sbl_surface!(c, t0 + dt / 2)
    A = m.Lx * m.Ly
    stage = Ref(0)
    τx = Ref(0.0); τy = Ref(0.0)
    wm = WallModel(c.laws)
    z_top = m.z0 + m.Lz
    fstage = Ref(0)
    force! = function (sim, Δt)
        fstage[] += 1
        # the momentum budget sees the built-in tendencies of this substage
        c.budget === nothing ||
            _sbl_budget_stage!(c.budget, c, sim, _SBL_RK3_WEIGHTS[mod1(fstage[], 3)] * dt, t0, dt)
        apply_forcing!(sim, c.forcing, t0, dt)
        K = m.Hz + 1
        @inbounds for j in 1:m.Ny, i in 1:m.Nx
            sim.tθ.data[m.Hx+i, m.Hy+j, K] += c.wθ[i, j] / m.dz
        end
        if sim.has_sponge
            if c.sponge_target === :geostrophic && c.U_g isa Real && c.V_g isa Real
                # GABLS1 (P55-f): constant U_g, the core's own sponge profile
                add_sponge!(sim.tu, c.zero_u, sim.sponge_α_max, sim.sponge_z_start, z_top; f_ref = c.U_g)
                add_sponge!(sim.tv, c.zero_v, sim.sponge_α_max, sim.sponge_z_start, z_top; f_ref = c.V_g)
            else
                _sbl_sponge_target!(sim, c, t0, dt)
            end
        end
        return nothing
    end
    wall! = function (sim, Δt)
        stage[] += 1
        w = _SBL_RK3_WEIGHTS[mod1(stage[], 3)]
        F = apply_wall_stress!(sim, wm, Δt; side = :bottom)
        τx[] += w * (-F.Fx) / A
        τy[] += w * (-F.Fy) / A
        if c.budget !== nothing           # the surface stress enters level 1
            c.budget.wall_u -= w * dt * (-F.Fx) / A / m.dz
            c.budget.wall_v -= w * dt * (-F.Fy) / A / m.dz
        end
        return nothing
    end
    tw = time()
    step_rk3!(s, dt; sgs = c.closure, sgs_stress = :symmetric,
              body_force! = force!, post_predictor! = wall!)
    c.wall_time += time() - tw
    c.τx = τx[]; c.τy = τy[]
    c.heat_applied += mean(c.wθ) * dt
    c.adv_heat_applied += _sbl_advective_heat(c.forcing, m, t0, dt)
    c.nsteps += 1
    return c
end

# The sponge retargeting for height/time-dependent U_g or the plane mean:
# the core adds -α(z) u in the sponge; this adds +α(z) u_target(z) per level,
# with the core's α(z) = α_max sin²(π/2 (z - z_s)/(z_top - z_s)) at the
# FaceX/FaceY nodes (cell centres in z).
function _sbl_sponge_target!(sim, c::SBLCase, t0, dt)
    m = sim.mesh
    z_s = sim.sponge_z_start; z_top = m.z0 + m.Lz
    n = m.Nx * m.Ny
    @inbounds for k in 1:m.Nz
        z = m.z0 + (k - 0.5) * m.dz
        z >= z_s || continue
        sθ = sin(0.5π * (z - z_s) / (z_top - z_s))
        α = sim.sponge_α_max * sθ * sθ
        K = m.Hz + k
        if c.sponge_target === :mean
            ū = 0.0; v̄ = 0.0
            for j in 1:m.Ny, i in 1:m.Nx
                ū += sim.u.data[m.Hx+i, m.Hy+j, K]; v̄ += sim.v.data[m.Hx+i, m.Hy+j, K]
            end
            ut, vt = ū / n, v̄ / n
        else
            ut = forcing_mean(c.U_g, z, t0, t0 + dt); vt = forcing_mean(c.V_g, z, t0, t0 + dt)
        end
        for j in 1:m.Ny, i in 1:m.Nx
            sim.tu.data[m.Hx+i, m.Hy+j, K] += α * ut
            sim.tv.data[m.Hx+i, m.Hy+j, K] += α * vt
        end
    end
    return sim
end

# ∫ (∂θ/∂t)_adv dz over the step [t0, t0 + dt], times dt: exactly what
# `apply_forcing!` adds (it uses the same step means at the same levels;
# the SSP-RK3 weights of the three substages sum to one).
_sbl_advective_heat(F::Tuple, m, t0, dt) = sum((_sbl_advective_heat(f, m, t0, dt) for f in F); init = 0.0)
_sbl_advective_heat(::Any, m, t0, dt) = 0.0
function _sbl_advective_heat(F::AdvectiveTendencies, m, t0, dt)
    F.θ === nothing && return 0.0
    return dt * m.dz * sum(forcing_mean(F.θ, m.z0 + (k - 0.5) * m.dz, t0, t0 + dt) for k in 1:m.Nz)
end

"""
    sbl_column_heat(c) -> K m

The plane-mean column integral ``∫_0^{L_z} ⟨θ⟩ dz``.
"""
function sbl_column_heat(c::SBLCase)
    s = c.sim; m = s.mesh
    return sum(interior_view(s.θ)) * m.dz / (m.Nx * m.Ny)
end

"""
    sbl_profiles(c) -> NamedTuple

Instantaneous plane-mean profiles: `z` (cell centres), `U`, `V`, speed
`S`, `θ`; at the w-faces `zf`: the total (resolved + SGS) kinematic
momentum fluxes `τx`, `τy` (positive downward, with the surface stress of
the last step at z = 0, from `Operators.total_stress_profile` with the
closure's vertical viscosity) and their magnitude `τ`, and the total
heat flux `wθ` (resolved ⟨w θ⟩ + SGS −⟨κ_v ∂θ/∂z⟩, the surface flux at
z = 0).
"""
function sbl_profiles(c::SBLCase)
    s = c.sim; m = s.mesh
    Nx, Ny, Nz, Hx, Hy, Hz = m.Nx, m.Ny, m.Nz, m.Hx, m.Hy, m.Hz
    fill_halos!(s.u, s.bcs); fill_halos!(s.v, s.bcs); fill_halos!(s.w, s.bcs)
    fill_halos!(s.θ, s.θ_bcs)
    f = compute_sgs!(c.closure, s)
    νv = f.νv; κv = f.κv
    fill_halos!(νv, s.bcs); fill_halos!(κv, s.bcs)
    px = total_stress_profile(s.u, s.v, s.w, νv; ν = 0.0, τ_wall_bottom = c.τx)
    py = total_stress_profile(s.u, s.v, s.w, νv; ν = 0.0, τ_wall_bottom = c.τy, component = :y)
    U = zeros(Nz); V = zeros(Nz); θ = zeros(Nz)
    @inbounds for k in 1:Nz, j in 1:Ny, i in 1:Nx
        I, J, K = Hx + i, Hy + j, Hz + k
        U[k] += 0.5 * (s.u.data[I, J, K] + s.u.data[I+1, J, K])
        V[k] += 0.5 * (s.v.data[I, J, K] + s.v.data[I, J+1, K])
        θ[k] += s.θ.data[I, J, K]
    end
    n = Nx * Ny
    U ./= n; V ./= n; θ ./= n
    wθ = zeros(Nz + 1)
    @inbounds for k in 2:Nz, j in 1:Ny, i in 1:Nx
        I, J, K = Hx + i, Hy + j, Hz + k
        θf = 0.5 * (s.θ.data[I, J, K] + s.θ.data[I, J, K-1])
        κf = 0.5 * (κv.data[I, J, K] + κv.data[I, J, K-1])
        wθ[k] += s.w.data[I, J, K] * θf - κf * (s.θ.data[I, J, K] - s.θ.data[I, J, K-1]) / m.dz
    end
    wθ ./= n
    wθ[1] = mean(c.wθ)
    τ = sqrt.(px.total .^ 2 .+ py.total .^ 2)
    return (z = [(k - 0.5) * m.dz for k in 1:Nz], U = U, V = V, S = sqrt.(U .^ 2 .+ V .^ 2),
            θ = θ, zf = px.z, τx = px.total, τy = py.total, τ = τ, wθ = wθ)
end

"""
    sbl_bl_height(zf, τ; fraction = 0.05) -> h (m)

The SBL height of Kosović & Curry (2000) and Beare et al. (2006): the
height where the mean total stress first falls to `fraction` (5 %) of its
surface value, divided by 0.95 (linear interpolation between faces).
"""
function sbl_bl_height(zf, τ; fraction = 0.05)
    τ0 = τ[1]
    for k in 2:length(τ)
        if τ[k] <= fraction * τ0
            z = zf[k-1] + (fraction * τ0 - τ[k-1]) / (τ[k] - τ[k-1]) * (zf[k] - zf[k-1])
            return z / 0.95
        end
    end
    return zf[end] / 0.95
end

"""
    sbl_run!(c, t_end; cfl = 1.2, average_from = Inf, sample_every = 10,
             callback = nothing) -> stats

Advance to `t_end` (s) with [`sbl_dt`](@ref). From `average_from` on,
every `sample_every`-th step is sampled: the returned `stats` hold the
mean profiles (`U`, `V`, `S`, `θ`, `τ`, `wθ`), the BL height of the mean
stress profile `h`, the mean `u_star` (√ of the mean surface stress
magnitude), the mean surface heat flux `wθ0` (K m s⁻¹) and `H0`
(W m⁻², with the case's `ρcp`; GABLS1: 1.3·1005 J m⁻³ K⁻¹ near 263 K), the low-level-jet maximum
`S_max` and its height `z_jet`, the mean Obukhov length `L`, and
`nsteps`, `wall_time`.
"""
function sbl_run!(c::SBLCase, t_end; cfl = 1.2, average_from = Inf, sample_every = 10,
                  callback = nothing)
    s = c.sim
    acc = nothing; ns = 0; τs = 0.0; wθs = 0.0; Ls = 0.0
    n0 = c.nsteps
    while s.t < t_end - 1e-9
        dt = min(sbl_dt(c; cfl = cfl), t_end - s.t)
        sbl_step!(c, dt)
        callback === nothing || callback(c)
        if s.t >= average_from && (c.nsteps - n0) % sample_every == 0
            p = sbl_profiles(c)
            acc === nothing ? (acc = Dict(k => copy(getfield(p, k)) for k in (:U, :V, :S, :θ, :τ, :wθ, :τx, :τy))) :
                              (for k in keys(acc); acc[k] .+= getfield(p, k); end)
            τs += sqrt(c.τx^2 + c.τy^2); wθs += mean(c.wθ)
            Ls += mean(c.ustar)^3 / (0.4 * 9.81 / s.θ₀ * max(-mean(c.wθ), 1e-12))
            ns += 1
        end
    end
    ns == 0 && return (nsamples = 0, nsteps = c.nsteps - n0, wall_time = c.wall_time)
    for k in keys(acc); acc[k] ./= ns; end
    m = s.mesh
    z = [(k - 0.5) * m.dz for k in 1:m.Nz]
    zf = [(k - 1) * m.dz for k in 1:m.Nz+1]
    τm = sqrt.(acc[:τx] .^ 2 .+ acc[:τy] .^ 2)
    kj = argmax(acc[:S])
    wθ0 = wθs / ns
    return (z = z, zf = zf, U = acc[:U], V = acc[:V], S = acc[:S], θ = acc[:θ],
            τ = τm, wθ = acc[:wθ], h = sbl_bl_height(zf, τm),
            u_star = sqrt(τs / ns), wθ0 = wθ0, H0 = c.ρcp * wθ0,
            S_max = acc[:S][kj], z_jet = z[kj], L = Ls / ns,
            nsamples = ns, nsteps = c.nsteps - n0, wall_time = c.wall_time)
end

"""
    sbl_read_csv(path) -> (; header, columns, rows)

Read a reference-data CSV of `examples/sbl/data/`: the leading `#` lines
are its provenance header (returned verbatim), the first other line the
column names, the rest the rows (numbers parsed where possible).
"""
function sbl_read_csv(path)
    header = String[]; columns = String[]; rows = Vector{Any}[]
    for line in eachline(path)
        isempty(strip(line)) && continue
        if startswith(line, "#")
            push!(header, line)
        elseif isempty(columns)
            columns = strip.(split(line, ","))
        else
            push!(rows, [something(tryparse(Float64, strip(x)), strip(x)) for x in split(line, ",")])
        end
    end
    return (header = header, columns = columns, rows = rows)
end

"The value of column `col` in the row whose first column is `key`."
function sbl_csv_value(tab, key, col)
    j = findfirst(==(col), tab.columns)
    for r in tab.rows
        r[1] == key && return r[j]
    end
    throw(KeyError(key))
end

# =========================================================================
# P58-d (#194) diagnostics: the mean-momentum budget, screen-level values,
# tower samples of V_TKE and the hockey-stick threshold.
# =========================================================================

"""
    SBLMomentumBudget

The plane-mean momentum budget of an SBL run over a window, per level
`k` (cell centres), for `U` and `V`:

```math
ΔU_k = \\int \\Big[\\underbrace{\\frac{τ_{k+1} - τ_k}{Δz}}_{\\text{turb}}
       + \\underbrace{f⟨v⟩}_{\\text{cor}} \\underbrace{- f V_g}_{\\text{pg}}
       + \\underbrace{(∂u/∂t)_{adv}}_{\\text{adv}}
       \\underbrace{- α(z)(⟨u⟩ - u_{target})}_{\\text{sponge}}\\Big]\\,dt ,
```

with ``τ`` the total (resolved ``-⟨uw⟩`` + SGS ``ν_v ∂u/∂z``) downward
momentum flux of `Operators.total_stress_profile` on the interior faces
and the applied wall-model stress on the surface face (`wall_u`, booked
into level 1 by the wall-model hook). Every term is
integrated exactly in time: it is evaluated in each SSP-RK3 substage from
the substage state (the same state the dynamics use) and weighted with the
scheme's weights 1/6, 1/6, 2/3 (the forcing tables with their exact step
means). The plane-mean pressure gradient vanishes (periodic `x, y`).
`turb_model_u` is the core's own advection + SGS tendency (the built-in
tendency minus Coriolis and sponge), for comparison with the diagnosed
flux divergence. Start it with [`sbl_budget_start!`](@ref); read it with
[`sbl_budget`](@ref).
"""
mutable struct SBLMomentumBudget
    U0::Vector{Float64}; V0::Vector{Float64}; t0::Float64
    turb_u::Vector{Float64}; turb_v::Vector{Float64}
    cor_u::Vector{Float64};  cor_v::Vector{Float64}
    pg_u::Vector{Float64};   pg_v::Vector{Float64}
    adv_u::Vector{Float64};  adv_v::Vector{Float64}
    sponge_u::Vector{Float64}; sponge_v::Vector{Float64}
    turb_model_u::Vector{Float64}; turb_model_v::Vector{Float64}
    wall_u::Float64; wall_v::Float64     # ∫ -τ_wall/Δz dt into level 1
end

# plane means of a FaceX/FaceY/CellCenter field's interior, per level
function _sbl_plane_mean(f, m)
    out = zeros(m.Nz)
    @inbounds for k in 1:m.Nz, j in 1:m.Ny, i in 1:m.Nx
        out[k] += f.data[m.Hx+i, m.Hy+j, m.Hz+k]
    end
    return out ./ (m.Nx * m.Ny)
end

"""
    sbl_budget_start!(c) -> SBLMomentumBudget

Start the mean-momentum budget of `c` at the current state; every later
[`sbl_step!`](@ref) adds to it until `c.budget = nothing`.
"""
function sbl_budget_start!(c::SBLCase)
    m = c.sim.mesh; z() = zeros(m.Nz)
    c.budget = SBLMomentumBudget(_sbl_plane_mean(c.sim.u, m), _sbl_plane_mean(c.sim.v, m), c.sim.t,
                                 z(), z(), z(), z(), z(), z(), z(), z(), z(), z(), z(), z(), 0.0, 0.0)
    return c.budget
end

function _sbl_budget_stage!(b::SBLMomentumBudget, c::SBLCase, sim, wdt, t0, dt)
    m = sim.mesh
    ū = _sbl_plane_mean(sim.u, m); v̄ = _sbl_plane_mean(sim.v, m)
    tū = _sbl_plane_mean(sim.tu, m); tv̄ = _sbl_plane_mean(sim.tv, m)
    f = sim.f_coriolis
    νv = sgs_fields(c.closure, sim).νv          # this substage's coefficients, halos filled
    px = total_stress_profile(sim.u, sim.v, sim.w, νv; ν = 0.0)
    py = total_stress_profile(sim.u, sim.v, sim.w, νv; ν = 0.0, component = :y)
    z_s = sim.sponge_z_start; z_top = m.z0 + m.Lz
    for k in 1:m.Nz
        z = m.z0 + (k - 0.5) * m.dz
        # interior faces only (k + 1 ≤ Nz); the surface face is the wall stress (wall!)
        τu_hi = k < m.Nz ? px.total[k+1] : 0.0; τu_lo = k > 1 ? px.total[k] : 0.0
        τv_hi = k < m.Nz ? py.total[k+1] : 0.0; τv_lo = k > 1 ? py.total[k] : 0.0
        b.turb_u[k] += wdt * (τu_hi - τu_lo) / m.dz
        b.turb_v[k] += wdt * (τv_hi - τv_lo) / m.dz
        b.cor_u[k] += wdt * f * v̄[k]
        b.cor_v[k] -= wdt * f * ū[k]
        b.pg_u[k] -= wdt * f * forcing_mean(c.V_g, z, t0, t0 + dt)
        b.pg_v[k] += wdt * f * forcing_mean(c.U_g, z, t0, t0 + dt)
        for F in c.forcing
            F isa AdvectiveTendencies || continue
            F.u === nothing || (b.adv_u[k] += wdt * forcing_mean(F.u, z, t0, t0 + dt))
            F.v === nothing || (b.adv_v[k] += wdt * forcing_mean(F.v, z, t0, t0 + dt))
        end
        α = 0.0
        if sim.has_sponge && z >= z_s
            sθ = sin(0.5π * (z - z_s) / (z_top - z_s)); α = sim.sponge_α_max * sθ * sθ
        end
        ut, vt = c.sponge_target === :mean ? (ū[k], v̄[k]) :
                 (forcing_mean(c.U_g, z, t0, t0 + dt), forcing_mean(c.V_g, z, t0, t0 + dt))
        b.sponge_u[k] -= wdt * α * (ū[k] - ut)
        b.sponge_v[k] -= wdt * α * (v̄[k] - vt)
        # the core's built-in tendency (advection + SGS + Coriolis - α u) minus Coriolis and its sponge
        b.turb_model_u[k] += wdt * (tū[k] - f * v̄[k] + α * ū[k])
        b.turb_model_v[k] += wdt * (tv̄[k] + f * ū[k] + α * v̄[k])
    end
    return b
end

"""
    sbl_budget(c; z_max = Inf) -> NamedTuple

The momentum budget accumulated since [`sbl_budget_start!`](@ref) on the
levels below `z_max`: the change `ΔU`, `ΔV` of the plane means, each term
(`turb`, `cor`, `pg`, `adv`, `sponge`, the `_u`/`_v` vectors of
[`SBLMomentumBudget`](@ref)), the residual `res_u = ΔU - Σ terms` (and
`res_v`), the largest term magnitude `largest` and `rel = max|res| /
largest`; `model_turb_mismatch` is the largest difference between the
diagnosed flux divergence and the core's own advection + SGS tendency,
relative to `largest`.
"""
function sbl_budget(c::SBLCase; z_max = Inf)
    b = c.budget; m = c.sim.mesh
    ks = [k for k in 1:m.Nz if m.z0 + (k - 0.5) * m.dz <= z_max]
    ΔU = _sbl_plane_mean(c.sim.u, m) .- b.U0
    ΔV = _sbl_plane_mean(c.sim.v, m) .- b.V0
    tu = copy(b.turb_u); tv = copy(b.turb_v)
    tu[1] += b.wall_u; tv[1] += b.wall_v                     # the surface stress enters level 1
    ru = ΔU .- (tu .+ b.cor_u .+ b.pg_u .+ b.adv_u .+ b.sponge_u)
    rv = ΔV .- (tv .+ b.cor_v .+ b.pg_v .+ b.adv_v .+ b.sponge_v)
    terms = (tu, b.cor_u, b.pg_u, b.adv_u, b.sponge_u, tv, b.cor_v, b.pg_v, b.adv_v, b.sponge_v)
    largest = maximum(maximum(abs, t[ks]) for t in terms)
    res = max(maximum(abs, ru[ks]), maximum(abs, rv[ks]))
    # the core's built-in tendency has no wall term (the wall model acts after the predictor)
    mism = max(maximum(abs, (b.turb_model_u .- b.turb_u)[ks]), maximum(abs, (b.turb_model_v .- b.turb_v)[ks]))
    return (z = [m.z0 + (k - 0.5) * m.dz for k in ks], ΔU = ΔU[ks], ΔV = ΔV[ks],
            turb_u = tu[ks], cor_u = b.cor_u[ks], pg_u = b.pg_u[ks], adv_u = b.adv_u[ks],
            sponge_u = b.sponge_u[ks], turb_v = tv[ks], cor_v = b.cor_v[ks], pg_v = b.pg_v[ks],
            adv_v = b.adv_v[ks], sponge_v = b.sponge_v[ks], res_u = ru[ks], res_v = rv[ks],
            largest = largest, rel = res / largest, model_turb_mismatch = mism / largest,
            duration = c.sim.t - b.t0)
end

"""
    sbl_screen(c; z_θ = 2, z_U = 10) -> (θ_screen, U_screen)

Plane means of the screen-level potential temperature at `z_θ` (2 m) and
wind speed at `z_U` (10 m). Below the first model level (``Δz/2``) each
column's profile is Monin–Obukhov similarity with the `u_*`, ``θ_* =
-w'θ'_0/u_*`` and `L` of its last surface-layer solve and the
stability-corrected log terms of `SurfaceExchange` (`_Φm`, `_Φh`: the
solver's own functions):

```math
θ(z) = θ_s + \\frac{θ_*}{κ}\\Big[\\ln\\frac{z}{z_{0h}} - ψ_h\\Big(\\frac{z}{L}\\Big) + ψ_h\\Big(\\frac{z_{0h}}{L}\\Big)\\Big],
\\qquad
U(z) = \\frac{u_*}{κ}\\Big[\\ln\\frac{z}{z_{0m}} - ψ_m\\Big(\\frac{z}{L}\\Big) + ψ_m\\Big(\\frac{z_{0m}}{L}\\Big)\\Big];
```

at or above the first level the resolved plane-mean profile is
interpolated linearly. ``θ_s`` is the surface temperature of the last
solve.
"""
function sbl_screen(c::SBLCase; z_θ = 2.0, z_U = 10.0)
    m = c.sim.mesh; z1 = m.dz / 2
    κ = Aeolus.SurfaceExchange.von_karman()
    θs = surface_temperature(c.Ts, c.t_surf)
    SE = Aeolus.SurfaceExchange
    p = nothing
    prof(sym, z) = begin
        p === nothing && (p = sbl_profiles(c))
        zs = p.z; y = getfield(p, sym)
        k = clamp(searchsortedlast(zs, z), 1, length(zs) - 1)
        y[k] + (y[k+1] - y[k]) * (z - zs[k]) / (zs[k+1] - zs[k])
    end
    θscr = if z_θ < z1
        mean(θs + (-c.wθ[i] / c.ustar[i]) / κ * SE._Φh(c.sf, z_θ / c.Lob[i], z_θ, z_θ, c.z0h)
             for i in eachindex(c.wθ))
    else
        prof(:θ, z_θ)
    end
    Uscr = if z_U < z1
        mean(c.ustar[i] / κ * SE._Φm(c.sf, z_U / c.Lob[i], z_U, c.z0m) for i in eachindex(c.ustar))
    else
        prof(:S, z_U)
    end
    return (θ_screen = θscr, U_screen = Uscr)
end

"""
    SBLTowers

Virtual towers: every column of the domain is a tower with instruments
at the cell centres nearest to the requested heights. Over each averaging
`window` (s; Sun et al. 2012 use 5-min segments) the time means and
variances of `u, v, w` are formed per tower and level, and each segment
gives one sample of the mean wind speed ``V = |\\bar{\\mathbf u}|`` and of
the turbulence velocity scale ``V_{TKE} = \\sqrt{e}``
(`Transition.turbulence_velocity_scale`), with ``e = (σ_u^2 + σ_v^2 +
σ_w^2)/2`` plus the mean SGS energy ``e_{sgs} = (ν_t/(C_k Δ))^2``
(Deardorff 1980, ``ν_t = C_k Δ \\sqrt{e_{sgs}}``, ``C_k = 0.1``). The
samples are binned in `V` per level (`bins`: counts and sums).
Start with [`sbl_towers`](@ref), feed with [`sbl_towers_sample!`](@ref)
after every step.
"""
mutable struct SBLTowers
    ks::Vector{Int}; z::Vector{Float64}
    window::Float64; t_win::Float64; τ::Float64
    acc::Array{Float64,4}                 # (Nx, Ny, nlev, 7): u, v, w, uu, vv, ww, e_sgs
    edges::Vector{Float64}
    count::Matrix{Int}; sumV::Matrix{Float64}; sumE::Matrix{Float64}; sumE2::Matrix{Float64}
    nseg::Int
    active::Bool
end

"""
    sbl_towers(c, heights; window = 300, Vmax = 16, dV = 0.25) -> SBLTowers

Virtual towers at `heights` (m) with `window`-second segments, binning
the wind speed in `dV` bins up to `Vmax`.
"""
function sbl_towers(c::SBLCase, heights; window = 300.0, Vmax = 16.0, dV = 0.25)
    m = c.sim.mesh
    ks = [clamp(round(Int, h / m.dz + 0.5), 1, m.Nz) for h in heights]
    edges = collect(0.0:dV:Vmax)
    nb = length(edges) - 1; nl = length(ks)
    return SBLTowers(ks, [(k - 0.5) * m.dz for k in ks], Float64(window), c.sim.t, 0.0,
                     zeros(m.Nx, m.Ny, nl, 7), edges, zeros(Int, nb, nl), zeros(nb, nl),
                     zeros(nb, nl), zeros(nb, nl), 0, true)
end

"""
    sbl_towers_sample!(T, c, dt) -> T

Add the state at the end of a step of length `dt` to the current
segment of the towers `T`; close the segment (one sample per tower and
level) when it reaches `T.window`. Does nothing while `T.active` is false.
"""
function sbl_towers_sample!(T::SBLTowers, c::SBLCase, dt)
    T.active || (T.t_win = c.sim.t; T.τ = 0.0; fill!(T.acc, 0); return T)
    s = c.sim; m = s.mesh
    νt = s.νt                                     # the base (horizontal) eddy viscosity
    Δ = cbrt(m.dx * m.dy * m.dz); Ck = 0.1
    @inbounds for (l, k) in enumerate(T.ks), j in 1:m.Ny, i in 1:m.Nx
        I, J, K = m.Hx + i, m.Hy + j, m.Hz + k
        u = 0.5 * (s.u.data[I, J, K] + s.u.data[I+1, J, K])
        v = 0.5 * (s.v.data[I, J, K] + s.v.data[I, J+1, K])
        w = 0.5 * (s.w.data[I, J, K] + s.w.data[I, J, K+1])
        es = (νt.data[I, J, K] / (Ck * Δ))^2
        a = T.acc
        a[i, j, l, 1] += dt * u; a[i, j, l, 2] += dt * v; a[i, j, l, 3] += dt * w
        a[i, j, l, 4] += dt * u * u; a[i, j, l, 5] += dt * v * v; a[i, j, l, 6] += dt * w * w
        a[i, j, l, 7] += dt * es
    end
    T.τ += dt
    if T.τ >= T.window - 1e-9
        a = T.acc ./ T.τ
        nb = length(T.edges) - 1; dV = T.edges[2] - T.edges[1]
        for l in eachindex(T.ks), j in 1:m.Ny, i in 1:m.Nx
            ū, v̄, w̄ = a[i, j, l, 1], a[i, j, l, 2], a[i, j, l, 3]
            e = 0.5 * (max(a[i, j, l, 4] - ū^2, 0) + max(a[i, j, l, 5] - v̄^2, 0) +
                       max(a[i, j, l, 6] - w̄^2, 0)) + a[i, j, l, 7]
            V = hypot(ū, v̄); E = turbulence_velocity_scale(e)
            b = floor(Int, V / dV) + 1
            1 <= b <= nb || continue
            T.count[b, l] += 1; T.sumV[b, l] += V; T.sumE[b, l] += E; T.sumE2[b, l] += E^2
        end
        T.nseg += 1; T.τ = 0.0; T.t_win = s.t; fill!(T.acc, 0)
    end
    return T
end

"""
    sbl_towers_bins(T, l; min_count = 20) -> (V, V_TKE, σ, n)

The bin means of `V` and `V_TKE` (and the standard deviation and count of
`V_TKE`) at tower level `l`, for the bins with at least `min_count`
samples.
"""
function sbl_towers_bins(T::SBLTowers, l; min_count = 20)
    sel = findall(>=(min_count), T.count[:, l])
    n = T.count[sel, l]
    V = T.sumV[sel, l] ./ n; E = T.sumE[sel, l] ./ n
    σ = sqrt.(max.(T.sumE2[sel, l] ./ n .- E .^ 2, 0))
    return (V = V, V_TKE = E, σ = σ, n = n)
end

"""
    sbl_hockey_threshold(V, E; weights = ones) -> (V_s, slope1, slope2, intercept, sse)

The threshold wind speed of the "hockey stick" (Sun et al. 2012): the
break point `V_s` of the continuous two-segment line
``E = a + s_1 V`` for ``V < V_s``, ``E = a + s_1 V_s + s_2 (V - V_s)``
above, fitted by weighted least squares to the bin means `(V, E)`. The
break point is searched on a fine grid between the second and the
second-to-last bin (each segment keeps at least two bins); for each
candidate the line is linear in ``(a, s_1, s_2)`` and solved exactly.
"""
function sbl_hockey_threshold(V::AbstractVector, E::AbstractVector; weights = ones(length(V)))
    n = length(V); n >= 4 || throw(ArgumentError("sbl_hockey_threshold needs at least 4 bins"))
    p = sortperm(V); V = V[p]; E = E[p]; w = weights[p]
    best = (V_s = NaN, slope1 = NaN, slope2 = NaN, intercept = NaN, sse = Inf)
    for Vs in range(V[2], V[end-1]; length = 400)
        X = hcat(ones(n), min.(V, Vs), max.(V .- Vs, 0))
        W = sqrt.(w)
        β = (W .* X) \ (W .* E)
        r = E .- X * β
        sse = sum(w .* r .^ 2)
        sse < best.sse && (best = (V_s = Vs, slope1 = β[2], slope2 = β[3], intercept = β[1], sse = sse))
    end
    return best
end

# =========================================================================
# The wall-resolved, stably stratified channel (García-Villalba & del
# Álamo 2011) on the P54-d builder — shared by examples/sbl/
# stratified_channel.jl and test/era6_stratified_channel.jl.
# =========================================================================

"""
    stratified_channel_case(Ri, stab; N, seed = 1) -> WallBoundedCase

The wall-resolved channel at Re_τ = 180 on 2πδ × πδ × 2δ: no-slip walls,
θ walls at `stratified_channel_walls(Ri)` (for `Ri = 0` θ is passive: walls
(0, 1) and g = 0), Pr = 0.71, the plane-averaged dynamic Smagorinsky
closure (Germano et al. 1991), wrapped in `StabilityCorrected` (Brown et
al. 1994, anisotropic length scales: the Berlin default) for
`stab = :brown94`, stress-form momentum divergence.
"""
function stratified_channel_case(Ri, stab; N = (32, 32, 48), seed = 1)
    mesh = RegularGrid(; size = N, extent = (2π, π, 2.0))
    base = DynamicSmagorinsky(mesh)
    closure = stab === :none ? base : StabilityCorrected(base, mesh)
    walls = Ri == 0 ? (0.0, 1.0) : stratified_channel_walls(Ri)
    return channel_case(; Re_τ = 180, N = N, closure = closure, bottom_bc = :noslip,
                        sgs_stress = :symmetric, θ_walls = walls, Pr = 0.71,
                        g_accel = Ri == 0 ? 0.0 : 1.0, seed = seed)
end

"""
    stratified_channel_spinup(; N = (32, 32, 48), T_spin = 10, seed = 1) -> snapshot

The neutral spin-up the stratified runs start from (García-Villalba & del
Álamo 2011 switch the stratification on in a developed neutral channel):
the uncorrected dynamic closure with θ walls (0, 1) and no buoyancy, from
the law-of-the-wall profile plus noise, for `T_spin` δ/u_τ. Returns the
fields `u, v, w, θ` (normalised θ, 0 at the bottom wall and 1 at the top)
and the time `t`.
"""
function stratified_channel_spinup(; N = (32, 32, 48), T_spin = 10.0, seed = 1)
    c = stratified_channel_case(0.0, :none; N = N, seed = seed)
    run_channel!(c, T_spin)
    s = c.sim
    return (u = copy(s.u.data), v = copy(s.v.data), w = copy(s.w.data), θ = copy(s.θ.data), t = s.t)
end

"""
    stratified_channel_run(Ri, stab; N = (32, 32, 48), T_end = 40, T_avg = 20, seed = 1,
                           spinup = nothing) -> NamedTuple

One wall-resolved channel ([`stratified_channel_case`](@ref)) at
Re_τ = 180 and `Ri`, run to `T_end` and averaged from `T_avg` (δ/u_τ),
started from `spinup` (a [`stratified_channel_spinup`](@ref) snapshot, θ
rescaled to the walls) or, with `spinup = nothing`, from noise. Returns the time-averaged mean profile `U` (cell
centres `zc`), bulk velocity `Ub` (u_τ units), Nusselt number `Nu` (wall
heat flux over the laminar ``κΔθ/2δ``), mean wall stress, the averaged
total stress profile and the momentum-budget residual
``\\max_z|τ(z) - τ(0) - ∫_0^z(∂⟨U⟩/∂t - G)dz'|``.
"""
function stratified_channel_run(Ri, stab; N = (32, 32, 48), T_end = 40.0, T_avg = 20.0,
                                seed = 1, spinup = nothing)
    c = stratified_channel_case(Ri, stab; N = N, seed = seed)
    walls = c.θ_walls
    if spinup !== nothing          # start from the developed neutral channel
        s = c.sim
        s.u.data .= spinup.u; s.v.data .= spinup.v; s.w.data .= spinup.w
        s.θ.data .= walls[1] .+ (walls[2] - walls[1]) .* spinup.θ
        s.t = spinup.t
    end
    run_channel!(c, T_avg)
    U0 = Aeolus.WallModels._uv_means(c.sim)[1]; t0 = c.sim.t
    q = Ref(0.0); nq = Ref(0)
    cb = c -> (q[] += -(c.q_wall_bottom + c.q_wall_top) / 2; nq[] += 1)
    st = run_channel!(c, T_end; average_from = T_avg, callback = cb)
    U1 = Aeolus.WallModels._uv_means(c.sim)[1]
    dUdt = (U1 .- U0) ./ (c.sim.t - t0)
    C = [0.0; cumsum((dUdt .- c.forcing) .* c.sim.mesh.dz)]
    budget = maximum(abs, st.stress.total .- (st.stress.total[1] .+ C))
    q_lam = (c.sim.ν / c.Pr) * (walls[2] - walls[1]) / 2
    return (Ri = Ri, stab = stab, N = N, U = st.U, zc = st.zc, Ub = mean(st.U),
            Nu = q[] / nq[] / q_lam, τ_wall = st.τ_wall, stress = st.stress, budget = budget,
            ΔUb = mean(U1) - mean(U0), nsteps = c.nsteps, wall_time = st.wall_time)
end

