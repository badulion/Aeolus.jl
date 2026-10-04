# The wall-bounded benchmark harness of Era VI (P54-d, #174): the
# pressure-driven half-channel ABL and the two-wall channel. No new wall
# physics — the wall stress is `apply_wall_stress!` (operator.jl), the SGS
# closure is any `AbstractSGSClosure`, the diagnostics are Operators'.
# Every Era VI wall-bounded validation (LASD log law, log-layer mismatch,
# stratified channel, collapse, relaminarisation) builds its case here,
# with keyword options, instead of copying the set-up.

export WallBoundedCase, half_channel_case, channel_case,
       advance_channel!, run_channel!, channel_dt

# SSP-RK3 weights of the three substage effects in the step's result
# (uⁿ⁺¹ = uⁿ/3 + 2/3 (u⁽²⁾ + Δt L(u⁽²⁾)), u⁽²⁾ = 3/4 uⁿ + 1/4 (…)):
# 1/6, 1/6, 2/3.
const _RK3_WEIGHTS = (1 / 6, 1 / 6, 2 / 3)

"""
    WallBoundedCase

A horizontally periodic, pressure-driven wall-bounded LES built by
[`half_channel_case`](@ref) or [`channel_case`](@ref). Fields:

  * `sim` — the `BoussinesqSimulation` (free-slip `ImpermeableWall`
    z-boundaries; the wall model supplies the wall stress);
  * `closure` — the SGS closure (an `AbstractSGSClosure`, or `nothing`);
  * `wall` — the [`WallModel`](@ref); `walls` — `:bottom` or `:both`;
    `formulation` — `:local`, `:schumann` or `:moeng`;
  * `forcing` — the mean pressure gradient ``-∂p/∂x/ρ = u_*^2/H``
    (m s⁻²), applied as a uniform body force on ``u``;
  * `u_star`, `height` — the friction velocity the forcing balances and
    the height ``H`` over which it does (the half-channel depth, or the
    channel half-height δ);
  * `kind` — `:half_channel` or `:channel`; `stepper` — `:rk3` (SSP-RK3,
    `step_rk3!`, the default) or `:euler` (`step!`);
  * `τ_wall_bottom`, `τ_wall_top` — the plane-mean kinematic wall stress
    (x component, m² s⁻²) the last step applied at each wall, from the
    momentum the wall operator removed (the SSP-RK3 combination of its
    three substages), so the plane-mean momentum budget of a step is
    exact: ``Δ⟨u⟩ H_{tot} = (G H_{tot} - τ_b - τ_t)Δt``;
  * `nsteps` — accepted steps so far.

Advance with [`advance_channel!`](@ref) or [`run_channel!`](@ref).
Instantaneous diagnostics: `phi_m_profile(case)` (Operators'
`phi_m_profile` with the case's `u_star` and κ =
`SurfaceExchange.von_karman()`) and `total_stress_profile(case)` (the
eddy viscosity recomputed from the current state with the case's closure
— pure, nothing is committed — the simulation's ν, and the wall stresses
the last step applied).
"""
mutable struct WallBoundedCase{S, C, W<:WallModel, T<:AbstractFloat}
    sim::S
    closure::C
    wall::W
    walls::Symbol
    formulation::Symbol
    forcing::T
    u_star::T
    height::T
    kind::Symbol
    stepper::Symbol
    τ_wall_bottom::T
    τ_wall_top::T
    nsteps::Int
    _stage::Int
end

_dims(N::Integer) = (Int(N), Int(N), Int(N))
_dims(N::NTuple{3, Integer}) = Int.(N)

function _reserved(name, value, default, issue)
    value == default ||
        throw(ArgumentError("$name: reserved for $issue, not implemented yet " *
                            "(only the default `$name = $(repr(default))` is accepted)"))
end

# The wall-parallel speed at distance d from the wall for which `law` gives
# the friction velocity u_star: friction_velocity is increasing in U, so a
# bracketing bisection inverts any law (smooth, rough or stability-corrected).
function _law_speed(law::AbstractWallLaw, u_star::Real, d::Real, ν::Real)
    f(U) = friction_velocity(law, U, d; ν = ν) - u_star
    lo, hi = 0.0, max(1.0, 10.0 * u_star)
    while f(hi) < 0
        hi *= 2
        hi < 1e12 || throw(ArgumentError("_law_speed: no speed gives u_* = $u_star at d = $d"))
    end
    for _ in 1:200
        mid = (lo + hi) / 2
        f(mid) < 0 ? (lo = mid) : (hi = mid)
        hi - lo <= 4eps(hi) && break
    end
    return (lo + hi) / 2
end

# Mean profile from the law (distance to the nearest wall with a wall
# model) plus seeded uniform noise of amplitude `noise·u*` on u, v, w.
function _initialise!(sim, law, u_star, ν, walls, noise, seed)
    g = sim.mesh
    Nx, Ny, Nz, Hx, Hy, Hz = g.Nx, g.Ny, g.Nz, g.Hx, g.Hy, g.Hz
    rng = Random.Xoshiro(seed)
    for k in 1:Nz
        z = (k - 0.5) * g.dz
        d = walls === :both ? min(z, g.Lz - z) : z
        U = _law_speed(law, u_star, d, ν)
        for j in 1:Ny, i in 1:Nx
            I, J, K = Hx + i, Hy + j, Hz + k
            sim.u.data[I, J, K] = U + noise * u_star * (2rand(rng) - 1)
            sim.v.data[I, J, K] = noise * u_star * (2rand(rng) - 1)
            # w on the bottom face of cell k; the wall face (k = 1) stays 0
            k > 1 && (sim.w.data[I, J, K] = noise * u_star * (2rand(rng) - 1))
        end
    end
    return sim
end

function _build_case(kind, dims, extent, closure, law, ν, walls, formulation,
                     u_star, height, noise, seed, stepper;
                     init_law = law, θ_ref = 0.0, θ_flux_bottom = 0.0,
                     c::SurfaceConstants = DEFAULT_SURFACE_CONSTANTS)
    walls in (:bottom, :both) ||
        throw(ArgumentError("walls must be :bottom or :both, got :$walls"))
    formulation in (:local, :schumann, :moeng) ||
        throw(ArgumentError("formulation must be :local, :schumann or :moeng, got :$formulation"))
    stepper in (:rk3, :euler) ||
        throw(ArgumentError("stepper must be :rk3 or :euler, got :$stepper"))
    closure === nothing || closure isa AbstractSGSClosure ||
        throw(ArgumentError("closure must be an AbstractSGSClosure or nothing, got $(typeof(closure))"))
    mesh = RegularGrid(; size = dims, extent = extent)
    bcs = FieldBoundaryConditions(:periodic, :periodic, :wall)
    sim = θ_ref > 0 ?
          BoussinesqSimulation(mesh; ν = ν, bcs = bcs, θ₀ = θ_ref, g_accel = grav(c),
                               θ_flux_bottom = θ_flux_bottom) :
          BoussinesqSimulation(mesh; ν = ν, bcs = bcs)
    θ_ref > 0 && fill!(sim.θ.data, θ_ref)
    _initialise!(sim, init_law, u_star, ν, walls, noise, seed)
    wm = WallModel(law; ν = ν > 0 ? ν : 1.5e-5)
    T = typeof(float(mesh.dz))
    return WallBoundedCase(sim, closure, wm, walls, formulation,
                           T(u_star^2 / height), T(u_star), T(height), kind, stepper,
                           zero(T), zero(T), 0, 0)
end

"""
    half_channel_case(; N = 24, closure = StaticSmagorinsky(0.1),
                      wall_law = nothing, formulation = :local,
                      z0_over_H = 1e-4, filter = :none, H = 1.0,
                      u_star = 1.0, noise = 3.0, seed = 1,
                      stepper = :rk3,
                      surface_heat_flux = nothing, θ_ref = 300.0,
                      stability = BusingerDyerBeljaarsHoltslag(),
                      bottom_bc = :wall_model,
                      θ_walls = nothing) -> WallBoundedCase

The neutral, pressure-driven **half channel** of Porté-Agel, Meneveau &
Parlange (2000) and Bou-Zeid, Meneveau & Parlange (2005): the canonical
high-Reynolds-number surface layer for testing an SGS closure's log law.

  * Domain ``2πH × 2πH × H`` on `N` cells per direction (`N` an integer
    or a 3-tuple), periodic in x and y.
  * Forcing ``-∂p/∂x/ρ = u_*^2/H``, a uniform body force on ``u``
    (`body_force!`), so the steady total stress is ``u_*^2(1 - z/H)``.
  * Free-slip top (`ImpermeableWall`); at the bottom, the modelled wall
    stress of `wall_law` through [`apply_wall_stress!`](@ref) (implicit,
    `post_predictor!`), in the `formulation` `:local` (Schumann-Grötzbach),
    `:schumann` or `:moeng`. `wall_law = nothing` is
    `RoughLogLaw(z0_over_H * H)`.
  * Inviscid (ν = 0), no Coriolis, no buoyancy. `closure` is any
    `AbstractSGSClosure` (no wall-damping function is applied; that is
    the point of the case).
  * Initial state: the law's profile ``U(z)`` (the speed for which the
    law gives ``u_*`` at height z, at every cell centre), plus seeded
    uniform noise of amplitude `noise·u_star` on all three components
    (rms 0.58·`noise`·u*). The default `noise = 3` (rms
    1.7 u*, the level of the surface-layer turbulence it stands for) makes
    the flow turbulent within about 5 H/u* at 24³ with
    `StaticSmagorinsky(0.1)`; with `noise = 0.3` the same run stays
    laminar (the eddy viscosity damps the noise) and its wall stress sits
    at 0.5-0.7 u*² for 30 H/u*.

Keyword hooks reserved for later Era VI issues — documented, **not
implemented** (any value other than the default raises an
`ArgumentError`):

  * `filter = :none` — the wall-model input filtering / matching height
    of Phase 57 (P57-b, #183);
  * `bottom_bc = :wall_model` and `θ_walls = nothing` — a resolved
    no-slip bottom with θ walls (P55-f, #191; P59-e, #195).

**Surface heat flux** (P58-e, #193). `surface_heat_flux = Q` (K m s⁻¹,
the upward kinematic flux ``\\overline{w'θ'}``; negative cools the
surface) turns the case into the flux-cooled (or heated) half channel of
the stable-boundary-layer collapse studies (Nieuwstadt 2005; Flores &
Riley 2011; Donda et al. 2015):

  * buoyancy on, ``g(θ - θ_{ref})/θ_{ref}`` with `θ_ref` (default
    300 K) and ``g`` of `SurfaceExchange`; θ starts uniform at `θ_ref`;
  * `Q` enters the first cell as the prescribed bottom θ flux
    (`θ_flux_bottom` of the simulation); the top and the bottom are
    otherwise adiabatic;
  * `wall_law = nothing` becomes
    [`MOSTFluxWallLaw`](@ref)`(stability, z0, Q, θ_ref)`: the wall stress
    of Monin–Obukhov similarity with the stability set by `Q` and the
    wall's own ``u_τ``, which drops to zero where ``-Q`` exceeds the
    maximum sustainable heat flux of the local wind
    (`max_sustainable_heat_flux`, Van de Wiel et al. 2012);
  * the initial velocity is the *neutral* log law (`RoughLogLaw(z0)`)
    plus noise, so a neutral and a cooled run start from the same state.

`max_sustainable_heat_flux(stability, U₁, Δz/2, z0, z0, θ_ref)` with the
neutral first-level wind ``U_1 = (u_*/κ)\\ln(Δz/(2z_0))`` is the MSHF the
collapse sweep is normalised by (`examples/sbl/collapse_channel.jl`,
`test/era6_collapse.jl`).

Measured with `StaticSmagorinsky(0.1)` at 24³ (`test/era6_neutral_abl.jl`)
and `StaticSmagorinsky(0.17)` (the Mason & Thomson 1992 overshoot): see
the *Wall models* theory page, section "Wall-bounded benchmark cases".

# References

  * Porté-Agel, F., Meneveau, C. & Parlange, M. B. (2000). A scale-dependent
    dynamic model for large-eddy simulation: application to a neutral
    atmospheric boundary layer. *J. Fluid Mech.*, 415, 261-284.
  * Bou-Zeid, E., Meneveau, C. & Parlange, M. B. (2005). A scale-dependent
    Lagrangian dynamic model for large eddy simulation of complex
    turbulent flows. *Phys. Fluids*, 17, 025105.
  * Mason, P. J. & Thomson, D. J. (1992). Stochastic backscatter in
    large-eddy simulations of boundary layers. *J. Fluid Mech.*, 242,
    51-78.
  * Nieuwstadt, F. T. M. (2005). Direct numerical simulation of stable
    channel flow at large stability. *Boundary-Layer Meteorol.*, 116,
    277-299.
  * Flores, O. & Riley, J. J. (2011). Analysis of turbulence collapse in
    the stably stratified surface layer using direct numerical
    simulation. *Boundary-Layer Meteorol.*, 139, 241-259.
  * Donda, J. M. M. et al. (2015). Collapse of turbulence in stably
    stratified channel flow: a transient phenomenon. *Q. J. R. Meteorol.
    Soc.*, 141, 2137-2147.
"""
function half_channel_case(; N = 24, closure = StaticSmagorinsky(0.1),
                           wall_law = nothing, formulation::Symbol = :local,
                           z0_over_H::Real = 1e-4, filter::Symbol = :none,
                           H::Real = 1.0, u_star::Real = 1.0,
                           noise::Real = 3.0, seed::Integer = 1,
                           stepper::Symbol = :rk3,
                           surface_heat_flux = nothing, θ_ref::Real = 300.0,
                           stability::AbstractStabilityFunctions = BusingerDyerBeljaarsHoltslag(),
                           bottom_bc::Symbol = :wall_model,
                           θ_walls = nothing)
    _reserved("filter", filter, :none, "P57-b (#183)")
    _reserved("bottom_bc", bottom_bc, :wall_model, "P55-f (#191) / P59-e (#195)")
    _reserved("θ_walls", θ_walls, nothing, "P55-f (#191) / P59-e (#195)")
    neutral = RoughLogLaw(z0_over_H * H)
    if surface_heat_flux === nothing
        law = wall_law === nothing ? neutral : wall_law
    else
        surface_heat_flux isa Real ||
            throw(ArgumentError("surface_heat_flux must be a number (K m/s) or nothing"))
        θ_ref > 0 || throw(ArgumentError("θ_ref must be > 0, got $θ_ref"))
        law = wall_law === nothing ?
              MOSTFluxWallLaw(stability, z0_over_H * H, surface_heat_flux, θ_ref) : wall_law
    end
    law isa AbstractWallLaw ||
        throw(ArgumentError("wall_law must be an AbstractWallLaw, got $(typeof(law))"))
    surface_heat_flux === nothing &&
        return _build_case(:half_channel, _dims(N), (2π * H, 2π * H, H), closure, law,
                           0.0, :bottom, formulation, u_star, H, noise, seed, stepper)
    return _build_case(:half_channel, _dims(N), (2π * H, 2π * H, H), closure, law,
                       0.0, :bottom, formulation, u_star, H, noise, seed, stepper;
                       init_law = neutral, θ_ref = θ_ref,
                       θ_flux_bottom = float(surface_heat_flux))
end

"""
    channel_case(; Re_τ = 1000, N = (16, 8, 16), closure = StaticSmagorinsky(0.065),
                 wall_law = SpaldingLaw(), walls = :both, formulation = :local,
                 δ = 1.0, u_τ = 1.0, extent = (2π*δ, π*δ, 2δ), noise = 3.0,
                 seed = 1, stepper = :rk3,
                 bottom_bc = :wall_model, θ_walls = nothing) -> WallBoundedCase

The smooth-wall, pressure-driven **plane channel** of half-height δ at
friction Reynolds number ``Re_τ = u_τ δ/ν`` (``ν = u_τ δ/Re_τ``, used
both as the molecular viscosity of the simulation and by the smooth wall
law), for wall-modelled LES in the sense of Piomelli & Balaras (2002) and
the log-layer-mismatch study (Kawai & Larsson 2012; P57-b, #183).

  * Domain `extent`, default ``2πδ × πδ × 2δ``, on `N` cells (integer or
    3-tuple), periodic in x and y.
  * Forcing ``-∂p/∂x/ρ = u_τ^2/δ`` (body force on ``u``): in steady state
    the mean stress of the two walls is ``u_τ^2`` and the total stress is
    ``u_τ^2(1 - z/δ)``.
  * `walls = :both` — the wall stress of `wall_law` at the bottom and the
    top (free-slip faces carrying the modelled stress); `:bottom` — a
    free-slip top at ``z = L_z`` (an open-channel variant).
  * Initial state: the law's profile in the distance to the nearest
    modelled wall plus seeded noise, as in [`half_channel_case`](@ref).
  * The default closure is Smagorinsky with ``C_s = 0.065``, the channel
    value of Moin & Kim (1982). At 16×8×16 the Porté-Agel ABL value 0.1
    keeps the flow laminar-like for more than 10 δ/u_τ: its mean wall
    stress over 2-10 δ/u_τ was 0.77-0.86 u_τ², against 0.994-1.007 u_τ²
    with 0.065 (four seeds each).

Reserved hooks (not implemented; see [`half_channel_case`](@ref)):
`bottom_bc = :wall_model`, `θ_walls = nothing` (P55-f, #191; P59-e, #195).

# References

  * Piomelli, U. & Balaras, E. (2002). *Annu. Rev. Fluid Mech.*, 34, 349-374.
  * Moin, P. & Kim, J. (1982). Numerical investigation of turbulent channel
    flow. *J. Fluid Mech.*, 118, 341-377.
  * Kawai, S. & Larsson, J. (2012). Wall-modeling in large eddy simulation:
    length scales, grid resolution, and accuracy. *Phys. Fluids*, 24, 015105.
  * Lee, M. & Moser, R. D. (2015). *J. Fluid Mech.*, 774, 395-415.
"""
function channel_case(; Re_τ::Real = 1000, N = (16, 8, 16),
                      closure = StaticSmagorinsky(0.065),
                      wall_law::AbstractWallLaw = SpaldingLaw(), walls::Symbol = :both,
                      formulation::Symbol = :local, δ::Real = 1.0, u_τ::Real = 1.0,
                      extent = (2π * δ, π * δ, 2δ), noise::Real = 3.0,
                      seed::Integer = 1, stepper::Symbol = :rk3,
                      bottom_bc::Symbol = :wall_model, θ_walls = nothing)
    _reserved("bottom_bc", bottom_bc, :wall_model, "P55-f (#191) / P59-e (#195)")
    _reserved("θ_walls", θ_walls, nothing, "P55-f (#191) / P59-e (#195)")
    Re_τ > 0 || throw(ArgumentError("Re_τ must be positive, got $Re_τ"))
    ν = u_τ * δ / Re_τ
    return _build_case(:channel, _dims(N), Tuple(float.(extent)), closure, wall_law,
                       ν, walls, formulation, u_τ, δ, noise, seed, stepper)
end

# ---------------------------------------------------------------------------
# Stepping
# ---------------------------------------------------------------------------

"""
    channel_dt(case; cfl = 0.5, c_diff = 0.3) -> Δt

A stable step for the current state: the advective limit
``Δt ≤ \\mathrm{cfl}/(\\max|u|/Δx + \\max|v|/Δy + \\max|w|/Δz)`` and the
diffusive limit ``Δt ≤ c_{diff}/[(ν + \\max ν_t)(Δx^{-2} + Δy^{-2} + Δz^{-2})]``
with the eddy viscosity of the last tendency evaluation (`sim.νt`).
"""
function channel_dt(case::WallBoundedCase; cfl::Real = 0.5, c_diff::Real = 0.3)
    s = case.sim; g = s.mesh
    ii, jj, kk = Grids.interior(g)
    umax = maximum(abs, view(s.u.data, ii, jj, kk))
    vmax = maximum(abs, view(s.v.data, ii, jj, kk))
    wmax = maximum(abs, view(s.w.data, ii, jj, kk))
    adv = umax / g.dx + vmax / g.dy + wmax / g.dz
    νmax = s.ν + maximum(view(s.νt.data, ii, jj, kk))
    dif = νmax * (1 / g.dx^2 + 1 / g.dy^2 + 1 / g.dz^2)
    dt = Inf
    adv > 0 && (dt = min(dt, cfl / adv))
    dif > 0 && (dt = min(dt, c_diff / dif))
    isfinite(dt) || throw(ArgumentError("channel_dt: the flow is at rest"))
    return dt
end

"""
    advance_channel!(case, dt) -> case

One step of length `dt` with the case's stepper (`:rk3`: `step_rk3!`,
`:euler`: `step!`), its SGS closure (`sgs = case.closure`), the mean
pressure gradient as `body_force!` and the wall stress as
`post_predictor!` ([`apply_wall_stress!`](@ref) at each modelled wall, in
every substage). Updates `case.τ_wall_bottom` / `case.τ_wall_top` with the
plane-mean stress the step applied.
"""
function advance_channel!(case::WallBoundedCase{S, C, W, T}, dt::Real) where {S, C, W, T}
    s = case.sim; g = s.mesh
    G = case.forcing
    A = g.Lx * g.Ly
    case._stage = 0
    τb = Ref(zero(T)); τt = Ref(zero(T))
    rk3 = case.stepper === :rk3
    force! = function (sim, Δt)
        ii, jj, kk = Grids.interior(sim.mesh)
        @inbounds for k in kk, j in jj, i in ii
            sim.tu.data[i, j, k] += G
        end
        return nothing
    end
    wall! = function (sim, Δt)
        case._stage += 1
        wgt = rk3 ? _RK3_WEIGHTS[mod1(case._stage, 3)] : 1.0
        Fb = apply_wall_stress!(sim, case.wall, Δt; side = :bottom,
                                formulation = case.formulation)
        τb[] += wgt * (-Fb.Fx) / A
        if case.walls === :both
            Ft = apply_wall_stress!(sim, case.wall, Δt; side = :top,
                                    formulation = case.formulation)
            τt[] += wgt * (-Ft.Fx) / A
        end
        return nothing
    end
    if rk3
        step_rk3!(s, dt; sgs = case.closure, body_force! = force!, post_predictor! = wall!)
    else
        step!(s, dt; sgs = case.closure, body_force! = force!, post_predictor! = wall!)
    end
    case.τ_wall_bottom = τb[]
    case.τ_wall_top = τt[]
    case.nsteps += 1
    return case
end

# plane means of the interior of u and v
function _uv_means(sim)
    g = sim.mesh
    Nx, Ny, Nz, Hx, Hy, Hz = g.Nx, g.Ny, g.Nz, g.Hx, g.Hy, g.Hz
    um = zeros(Nz); vm = zeros(Nz)
    @inbounds for k in 1:Nz, j in 1:Ny, i in 1:Nx
        um[k] += sim.u.data[Hx+i, Hy+j, Hz+k]
        vm[k] += sim.v.data[Hx+i, Hy+j, Hz+k]
    end
    return um ./ (Nx * Ny), vm ./ (Nx * Ny)
end

# phi_m_profile(case): the instantaneous Φ_m profile, with the case's u_star
# and κ = SurfaceExchange.von_karman() (documented on WallBoundedCase; the
# docstring of the generic is Operators').
phi_m_profile(case::WallBoundedCase) =
    phi_m_profile(case.sim.u, case.sim.v, case.u_star; κ = von_karman())

# total_stress_profile(case): the instantaneous flux profile; νt recomputed
# from the current state with the case's closure (pure, nothing committed),
# the wall stresses of the last step (documented on WallBoundedCase).
function total_stress_profile(case::WallBoundedCase)
    s = case.sim
    fill_halos!(s.u, s.bcs); fill_halos!(s.v, s.bcs); fill_halos!(s.w, s.bcs)
    νt = s.νt
    if case.closure === nothing
        fill!(νt.data, 0)
    else
        νt = compute_sgs!(case.closure, s).νv
    end
    return total_stress_profile(s.u, s.v, s.w, νt; ν = s.ν,
                                τ_wall_bottom = case.τ_wall_bottom,
                                τ_wall_top = case.τ_wall_top)
end

"""
    run_channel!(case, t_end; dt = nothing, cfl = 0.5, average_from = Inf,
                 sample_every = 1, callback = nothing) -> stats

Advance `case` to the time `t_end` (in units of the simulation clock;
``H/u_*`` is the eddy-turnover time of the default cases, which have
``H = u_* = 1``). The step is `dt`, or [`channel_dt`](@ref)`(case; cfl)`
re-evaluated every step (the last step is shortened to land on
`t_end`). `callback(case)` runs after every step.

From `average_from` on, every `sample_every`-th step is sampled and the
returned `stats` hold the averages over those samples:

  * `U` — the mean streamwise velocity profile at the cell centres `zc`;
  * `phi_m` (with heights `zf`) — Φ_m of the averaged profile;
  * `stress` — the averaged `total_stress_profile` (fields `z`,
    `resolved`, `sgs`, `viscous`, `wall`, `total`);
  * `τ_wall` — the mean wall stress per modelled wall,
    ``(τ_b + τ_t)/n_{walls}``, time-weighted over the sampled steps;
    `τ_wall_bottom`, `τ_wall_top` separately;
  * `t`, `τ_series` — the sampling times and the per-step wall stress;
  * `nsamples`, `nsteps`, `wall_time` (s).
"""
function run_channel!(case::WallBoundedCase, t_end::Real; dt = nothing, cfl::Real = 0.5,
                      average_from::Real = Inf, sample_every::Integer = 1,
                      callback = nothing)
    s = case.sim; g = s.mesh; Nz = g.Nz
    Usum = zeros(Nz)
    keys_ = (:resolved, :sgs, :viscous, :wall, :total)
    ssum = Dict(k => zeros(Nz + 1) for k in keys_)
    τb_sum = 0.0; τt_sum = 0.0; tw = 0.0
    ts = Float64[]; τs = Float64[]
    nsamp = 0; n0 = case.nsteps
    nw = case.walls === :both ? 2 : 1
    t0 = time()
    while s.t < t_end - 1e-12 * max(1.0, abs(t_end))
        h = dt === nothing ? channel_dt(case; cfl = cfl) : float(dt)
        h = min(h, t_end - s.t)
        advance_channel!(case, h)
        callback === nothing || callback(case)
        if s.t >= average_from && (case.nsteps - n0) % sample_every == 0
            nsamp += 1
            um, _ = _uv_means(s)
            Usum .+= um
            p = total_stress_profile(case)
            for k in keys_
                ssum[k] .+= getfield(p, k)
            end
            τb_sum += h * case.τ_wall_bottom; τt_sum += h * case.τ_wall_top; tw += h
            push!(ts, s.t); push!(τs, (case.τ_wall_bottom + case.τ_wall_top) / nw)
        end
    end
    zc = [(k - 0.5) * g.dz for k in 1:Nz]
    if nsamp == 0
        return (nsamples = 0, nsteps = case.nsteps - n0, wall_time = time() - t0)
    end
    U = Usum ./ nsamp
    pm = phi_m_profile(U, g.dz, case.u_star; κ = von_karman())
    stress = (z = [(k - 1) * g.dz for k in 1:Nz+1],
              resolved = ssum[:resolved] ./ nsamp, sgs = ssum[:sgs] ./ nsamp,
              viscous = ssum[:viscous] ./ nsamp, wall = ssum[:wall] ./ nsamp,
              total = ssum[:total] ./ nsamp)
    return (U = U, zc = zc, phi_m = pm.Φ, zf = pm.z, stress = stress,
            τ_wall = (τb_sum + τt_sum) / (nw * tw),
            τ_wall_bottom = τb_sum / tw, τ_wall_top = τt_sum / tw,
            t = ts, τ_series = τs, nsamples = nsamp,
            nsteps = case.nsteps - n0, wall_time = time() - t0)
end
