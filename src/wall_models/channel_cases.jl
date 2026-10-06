# The wall-bounded benchmark harness of Era VI (P54-d, #174): the
# pressure-driven half-channel ABL and the two-wall channel. No new wall
# physics — the wall stress is `apply_wall_stress!` (operator.jl), the SGS
# closure is any `AbstractSGSClosure`, the diagnostics are Operators'.
# Every Era VI wall-bounded validation (LASD log law, log-layer mismatch,
# stratified channel, collapse, relaminarisation) builds its case here,
# with keyword options, instead of copying the set-up.

export WallBoundedCase, half_channel_case, channel_case,
       advance_channel!, run_channel!, channel_dt,
       channel_richardson, stratified_channel_walls

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
  * `wall` — the wall model: a [`WallModel`](@ref) (equilibrium law) or
    a [`TBLEWallModel`](@ref) (non-equilibrium, P57-c); `walls` —
    `:bottom` or `:both`;
    `formulation` — `:local`, `:schumann` or `:moeng`;
  * `inputs` — `nothing` (first-cell matching, unfiltered), or a named
    tuple `(bottom, top)` of [`MatchingInput`](@ref)s (`top` is `nothing`
    for one wall): the matching height and input filters of P57-b;
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
  * `nsteps` — accepted steps so far;
  * `roughness` — `nothing`, or the [`DynamicRoughness`](@ref) of a
    partially resolved rough bottom (P57-e, #190): its resolved relief
    exerts the [`surface_gradient_drag`](@ref) on the first cell, its
    sub-filter relief sets the per-column roughness of the wall law,
    updated once per step; `τ_form` — the plane-mean kinematic form drag
    (x component, m² s⁻²) of the resolved relief in the last step (the
    SSP-RK3 combination), so the budget reads
    ``Δ⟨u⟩ H_{tot} = (G H_{tot} - τ_b - τ_t - τ_{form})Δt``;
  * `bottom_bc` — `:wall_model` (free-slip walls carrying the modelled
    stress) or `:noslip` (resolved [`NoSlipWall`](@ref)s, no wall model;
    `τ_wall_bottom`/`τ_wall_top` are then the molecular wall stress
    ``ν ∂⟨u⟩/∂z`` of `wall_shear_stress` after the step);
  * `sgs_stress` — the form of the SGS momentum divergence handed to the
    core (`:laplacian` or `:symmetric`);
  * `θ_walls` — `nothing` (no scalar) or the wall temperatures
    `(θ_bottom, θ_top)`; `Pr` — the molecular Prandtl number; and
    `q_wall_bottom`, `q_wall_top` — the plane-mean kinematic heat flux
    through each wall the last step applied (positive upward, i.e. into
    the fluid at the bottom and out of it at the top), from the step's
    RK3 combination, so ``Δ∫⟨θ⟩dz = (q_b - q_t)Δt`` exactly.

Advance with [`advance_channel!`](@ref) or [`run_channel!`](@ref).
Instantaneous diagnostics: `phi_m_profile(case)` (Operators'
`phi_m_profile` with the case's `u_star` and κ =
`SurfaceExchange.von_karman()`) and `total_stress_profile(case)` (the
eddy viscosity recomputed from the current state with the case's closure
— pure, nothing is committed — the simulation's ν, and the wall stresses
the last step applied).
"""
mutable struct WallBoundedCase{S, C, W<:AbstractWallModel, T<:AbstractFloat}
    sim::S
    closure::C
    wall::W
    walls::Symbol
    formulation::Symbol
    inputs::Any
    forcing::T
    u_star::T
    height::T
    kind::Symbol
    stepper::Symbol
    τ_wall_bottom::T
    τ_wall_top::T
    nsteps::Int
    _stage::Int
    roughness::Any
    τ_form::T
    bottom_bc::Symbol
    sgs_stress::Symbol
    θ_walls::Union{Nothing, NTuple{2, T}}
    Pr::T
    q_wall_bottom::T
    q_wall_top::T
    _lapθ::Any
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

# The wall-model inputs of the case's keywords (P57-b): `nothing` for the
# classical first-cell, unfiltered input (the pre-P57-b path, bit for bit).
function _case_inputs(walls, h_wm, h_wm_cells, filter, T_f, u_star, height)
    filter in (:none, :temporal, :spatial, :both) ||
        throw(ArgumentError("filter must be :none, :temporal, :spatial or :both, got :$filter"))
    temporal = filter in (:temporal, :both)
    T_f !== nothing && !temporal &&
        throw(ArgumentError("T_f is the time scale of filter = :temporal or :both"))
    h_wm === nothing && h_wm_cells == 1 && filter === :none && return nothing
    Tf = temporal ? (T_f === nothing ? _DEFAULT_TF * height / u_star : float(T_f)) : 0.0
    mk() = MatchingInput(; h_wm = h_wm, h_wm_cells = h_wm_cells, T_f = Tf,
                         spatial_filter = filter in (:spatial, :both))
    return (bottom = mk(), top = walls === :both ? mk() : nothing)
end

# Default time scale of the temporal input filter, in units of H/u_* (δ/u_τ).
const _DEFAULT_TF = 0.5

function _build_case(kind, dims, extent, closure, law, ν, walls, formulation,
                     u_star, height, noise, seed, stepper, inputs = nothing,
                     roughness = nothing, wall_model::Symbol = :equilibrium, tble = (;);
                     bottom_bc::Symbol = :wall_model, sgs_stress::Symbol = :laplacian,
                     θ_walls = nothing, Pr::Real = 0.71, θ₀::Real = 1.0, g_accel::Real = 1.0,
                     init_law = law, θ_ref::Real = 0.0, θ_flux_bottom::Real = 0.0,
                     flux_g = nothing, c::SurfaceConstants = DEFAULT_SURFACE_CONSTANTS,
                     inflow = nothing, outflow::Symbol = :convective,
                     lateral::Symbol = :periodic, lateral_outflow::Symbol = outflow)
    walls in (:bottom, :both) ||
        throw(ArgumentError("walls must be :bottom or :both, got :$walls"))
    formulation in (:local, :schumann, :moeng) ||
        throw(ArgumentError("formulation must be :local, :schumann or :moeng, got :$formulation"))
    stepper in (:rk3, :euler) ||
        throw(ArgumentError("stepper must be :rk3 or :euler, got :$stepper"))
    closure === nothing || closure isa AbstractSGSClosure ||
        throw(ArgumentError("closure must be an AbstractSGSClosure or nothing, got $(typeof(closure))"))
    bottom_bc in (:wall_model, :noslip) ||
        throw(ArgumentError("bottom_bc must be :wall_model or :noslip, got :$bottom_bc"))
    sgs_stress in (:laplacian, :symmetric) ||
        throw(ArgumentError("sgs_stress must be :laplacian or :symmetric, got :$sgs_stress"))
    if bottom_bc === :noslip
        ν > 0 || throw(ArgumentError("bottom_bc = :noslip needs a viscous flow (ν > 0)"))
        # a resolved wall has no wall model: its input, TBLE columns and
        # dynamic roughness do not apply
        inputs === nothing ||
            throw(ArgumentError("bottom_bc = :noslip has no wall model: h_wm, h_wm_cells and filter do not apply"))
        wall_model === :equilibrium ||
            throw(ArgumentError("bottom_bc = :noslip has no wall model: wall_model = :nonequilibrium does not apply"))
        roughness === nothing ||
            throw(ArgumentError("bottom_bc = :noslip has no wall model: roughness does not apply"))
    end
    if θ_walls !== nothing
        bottom_bc === :noslip ||
            throw(ArgumentError("θ_walls needs bottom_bc = :noslip (a wall-modelled heat flux " *
                                "is half_channel_case(surface_heat_flux), P58-e, #193)"))
        length(θ_walls) == 2 || throw(ArgumentError("θ_walls must be (θ_bottom, θ_top)"))
        Pr > 0 || throw(ArgumentError("Pr must be positive, got $Pr"))
        θ₀ > 0 || throw(ArgumentError("θ₀ must be positive, got $θ₀"))
    end
    # a prescribed bottom θ flux with buoyancy g(θ - θ_ref)/θ_ref (P58-e, #193)
    θ_ref >= 0 || throw(ArgumentError("θ_ref must be ≥ 0 (0: no flux-driven buoyancy), got $θ_ref"))
    θ_ref > 0 && θ_walls !== nothing &&
        throw(ArgumentError("θ_walls and a surface heat flux (θ_ref > 0) are exclusive"))
    θ_flux_bottom == 0 || θ_ref > 0 ||
        throw(ArgumentError("θ_flux_bottom needs θ_ref > 0 (buoyancy reference)"))
    mesh = RegularGrid(; size = dims, extent = extent)

    T = typeof(float(mesh.dz))
    if inflow !== nothing
        # N-inflow-d (#266): open in x — the inflow data on the west face,
        # a radiation (or Dirichlet) outflow on the east face (N-open-c)
        inflow isa AbstractBoundaryData ||
            throw(ArgumentError("inflow must be an AbstractBoundaryData, got $(typeof(inflow))"))
        bottom_bc === :wall_model && walls === :bottom ||
            throw(ArgumentError("inflow: only the wall-modelled half channel has an open-x variant"))
        lateral in (:periodic, :open) ||
            throw(ArgumentError("lateral must be :periodic or :open, got :$lateral"))
        # open y faces: the same data, the radiation condition where the
        # flow leaves through them, the data where it enters (N-open-c)
        out = lateral === :open ?
              (east = outflow, south = lateral_outflow, north = lateral_outflow) :
              (east = outflow,)
        bcs = open_boundary_conditions(mesh, inflow; x = :open, y = lateral,
                                       bottom = :wall, top = :wall, outflow = out)
    elseif bottom_bc === :noslip
        bcs = FieldBoundaryConditions(:periodic, :periodic; bottom = :noslip,
                                      top = walls === :both ? :noslip : :wall)
    else
        bcs = FieldBoundaryConditions(:periodic, :periodic, :wall)
    end
    θb = θ_walls === nothing ? nothing : (T(θ_walls[1]), T(θ_walls[2]))
    θ_ref > 0 && (Pr > 0 || throw(ArgumentError("Pr must be positive, got $Pr")))
    # flux-driven buoyancy: θ_ref and the surface constants' g, or the
    # g_accel of a flux-cooled channel_case (flux_g, P59-e)
    θ0_sim, g_sim = θ_ref > 0 ? (θ_ref, flux_g === nothing ? grav(c) : flux_g) :
                                (θb === nothing ? 0.0 : θ₀, g_accel)
    sim = BoussinesqSimulation(mesh; ν = ν, bcs = bcs,
                               θ₀ = θ0_sim, g_accel = g_sim, θ_flux_bottom = θ_flux_bottom,
                               θ_bcs = inflow === nothing ?
                                       FieldBoundaryConditions(:periodic, :periodic, :wall) : bcs)
    θ_ref > 0 && fill!(sim.θ.data, θ_ref)
    _initialise!(sim, init_law, u_star, ν, walls, noise, seed)
    θb === nothing || _initialise_θ!(sim, θb, walls)
    wall_model in (:equilibrium, :nonequilibrium) ||
        throw(ArgumentError("wall_model must be :equilibrium or :nonequilibrium, got :$wall_model"))
    roughness === nothing || wall_model === :equilibrium ||
        throw(ArgumentError("roughness (a DynamicRoughness) needs wall_model = :equilibrium"))
    wm = if wall_model === :nonequilibrium
        formulation === :local ||
            throw(ArgumentError("wall_model = :nonequilibrium is local per column (formulation = :local)"))
        # κ_wm starts from the law's κ, so the primed columns are that law's
        κ0 = law isa SpaldingLaw ? law.κ : von_karman()
        TBLEWallModel(; ν = ν > 0 ? ν : 1.5e-5, κ = κ0, tble...)
    elseif roughness === nothing
        WallModel(law; ν = ν > 0 ? ν : 1.5e-5)
    else
        roughness isa DynamicRoughness ||
            throw(ArgumentError("roughness must be a DynamicRoughness, got $(typeof(roughness))"))
        size(roughness.stats) == (mesh.Nx, mesh.Ny) ||
            throw(DimensionMismatch("roughness: surface $(size(roughness.stats)) ≠ grid $((mesh.Nx, mesh.Ny))"))
        (roughness.stats.periodic && roughness.stats.dx ≈ mesh.dx && roughness.stats.dy ≈ mesh.dy) ||
            throw(ArgumentError("roughness: the surface statistics must be periodic with the grid spacing"))
        (law isa RoughLogLaw || law isa MOSTWallLaw) ||
            throw(ArgumentError("roughness: the wall law must be a RoughLogLaw or MOSTWallLaw"))
        WallModel(dynamic_wall_laws(roughness, law); ν = ν > 0 ? ν : 1.5e-5)
    end
    # the (1/Pr − 1)ν∇²θ scratch of a viscous run with an active θ (θ walls,
    # or a prescribed surface flux on a resolved wall, P59-e)
    lapθ = (θb !== nothing || (θ_ref > 0 && ν > 0 && Pr != 1)) ?
           Field(Grids.CellCenter(), mesh) : nothing
    return WallBoundedCase(sim, closure, wm, walls, formulation, inputs,
                           T(u_star^2 / height), T(u_star), T(height), kind, stepper,
                           zero(T), zero(T), 0, 0, roughness, zero(T),
                           bottom_bc, sgs_stress, θb, T(Pr), zero(T), zero(T), lapθ)
end

# The conduction profile between the two θ walls (linear in z), or a
# uniform θ_bottom under a free-slip, insulated top (walls = :bottom).
function _initialise_θ!(sim, θb, walls)
    g = sim.mesh
    for k in 1:g.Nz
        z = (k - 0.5) * g.dz
        θ = walls === :both ? θb[1] + (θb[2] - θb[1]) * z / g.Lz : θb[1]
        @views sim.θ.data[g.Hx+1:g.Hx+g.Nx, g.Hy+1:g.Hy+g.Ny, g.Hz+k] .= θ
    end
    return sim
end

"""
    half_channel_case(; N = 24, closure = StaticSmagorinsky(0.1),
                      wall_law = nothing, formulation = :local,
                      z0_over_H = 1e-4, h_wm = nothing, h_wm_cells = 1,
                      filter = :none, T_f = nothing, H = 1.0,
                      u_star = 1.0, noise = 3.0, seed = 1,
                      stepper = :rk3, roughness = nothing,
                      surface_heat_flux = nothing, θ_ref = 300.0,
                      stability = BusingerDyerBeljaarsHoltslag(),
                      bottom_bc = :wall_model,
                      θ_walls = nothing, extent = nothing,
                      inflow = nothing, outflow = :convective,
                      lateral = :periodic, lateral_outflow = outflow) -> WallBoundedCase

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

**Domain and inflow** (N-inflow-d, #266). `extent = (Lx, Ly)` replaces
the default ``2πH × 2πH`` horizontal extent. `inflow` — an
[`AbstractBoundaryData`](@ref Aeolus.BoundaryConditions.AbstractBoundaryData)
(e.g. a `ProfileBoundaryData` with the mean profile of a periodic
precursor) — opens the x direction: the west face takes the data
(Dirichlet), the east face the `outflow` condition of N-open-c
(`:convective` by default, `:orlanski` or `:dirichlet`), and the walls, forcing, closure and wall model are those of
the periodic case. This is the inflow–outflow half channel on which the
development length of a synthetic inflow is measured against the
periodic precursor (`Transition.development_length`; Xie & Castro 2008;
Kim, Castro & Xie 2013). Neutral only.

`lateral = :periodic` (default) keeps y periodic; `lateral = :open`
opens the south and north faces as well, as in a nest child: they take
the same data, with the `lateral_outflow` condition (default: that of
the east face) at points where the flow leaves the domain and the data
(Dirichlet) where it enters. With a mean-profile `inflow` (``v = 0``)
the lateral faces carry no turbulence of their own, so the side layers
of the domain are quieter than its centre; the fetch diagnostics
(`FetchStatistics(sim; jrange)`) can exclude them.

**Wall-model input** (P57-b, #183; see [`MatchingInput`](@ref)):

  * `h_wm` (m) or `h_wm_cells` — the matching height: the law is
    evaluated with the velocity there and its stress applied at the wall
    face (Kawai & Larsson 2012). Default: the first cell centre;
  * `filter` — `:none`, `:temporal` (the exponential time filter of Yang,
    Park & Moin 2017, time scale `T_f`, default ``0.5\\,H/u_*``),
    `:spatial` (the 3 × 3 wall-parallel box of Bou-Zeid et al. 2005) or
    `:both`.

The defaults reproduce the first-cell, unfiltered input bit for bit.

**Partially resolved roughness** (P57-e, #190): `roughness` — a
[`DynamicRoughness`](@ref) on the `(N, N)` surface grid (periodic, the
grid's spacing). The resolved relief ``h_Δ`` of its statistics exerts
[`surface_gradient_drag`](@ref) on the first cell (a body force
``f_i/Δz`` on the faces, the mean of the two adjacent columns), and the
wall law (`wall_law`, a `RoughLogLaw` or `MOSTWallLaw`, as a template)
gets, per column, the dynamic roughness ``\\sqrt{z_{0,f}^2+α^2σ_Δ^2}``,
with α solved once per step from the first-cell (or matching-height)
velocity (Anderson & Meneveau 2011). `τ_form` holds the form drag.

Keyword hooks reserved for later Era VI issues — documented, **not
implemented** (any value other than the default raises an
`ArgumentError`):

  * `bottom_bc = :wall_model` and `θ_walls = nothing` — the half channel
    is inviscid, so it has no resolved wall; the resolved no-slip,
    θ-walled flow is [`channel_case`](@ref)`(bottom_bc = :noslip,
    θ_walls = …)` (P55-f, #191).

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
                           z0_over_H::Real = 1e-4, h_wm = nothing, h_wm_cells = 1,
                           filter::Symbol = :none, T_f = nothing,
                           H::Real = 1.0, u_star::Real = 1.0,
                           noise::Real = 3.0, seed::Integer = 1,
                           stepper::Symbol = :rk3, roughness = nothing,
                           surface_heat_flux = nothing, θ_ref::Real = 300.0,
                           stability::AbstractStabilityFunctions = BusingerDyerBeljaarsHoltslag(),
                           bottom_bc::Symbol = :wall_model,
                           θ_walls = nothing, extent = nothing,
                           inflow = nothing, outflow::Symbol = :convective,
                           lateral::Symbol = :periodic, lateral_outflow::Symbol = outflow)
    inflow === nothing && lateral !== :periodic &&
        throw(ArgumentError("lateral = :$lateral needs an inflow (the open-x half channel)"))
    _reserved("bottom_bc", bottom_bc, :wall_model, "channel_case (the half channel is inviscid)")
    _reserved("θ_walls", θ_walls, nothing, "channel_case (the half channel is inviscid)")
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
    ext = if extent === nothing
        (2π * H, 2π * H, H)
    else
        length(extent) == 2 && all(>(0), extent) ||
            throw(ArgumentError("extent must be (Lx, Ly) > 0, got $extent"))
        (float(extent[1]), float(extent[2]), H)
    end
    inflow === nothing || surface_heat_flux === nothing ||
        throw(ArgumentError("inflow: the open-x half channel is neutral (no surface_heat_flux)"))
    surface_heat_flux === nothing &&
        return _build_case(:half_channel, _dims(N), ext, closure, law,
                           0.0, :bottom, formulation, u_star, H, noise, seed, stepper,
                           _case_inputs(:bottom, h_wm, h_wm_cells, filter, T_f, u_star, H),
                           roughness; inflow = inflow, outflow = outflow,
                           lateral = lateral, lateral_outflow = lateral_outflow)
    return _build_case(:half_channel, _dims(N), ext, closure, law,
                       0.0, :bottom, formulation, u_star, H, noise, seed, stepper,
                       _case_inputs(:bottom, h_wm, h_wm_cells, filter, T_f, u_star, H),
                       roughness; init_law = neutral, θ_ref = θ_ref,
                       θ_flux_bottom = float(surface_heat_flux))
end

"""
    channel_case(; Re_τ = 1000, N = (16, 8, 16), closure = StaticSmagorinsky(0.065),
                 wall_law = SpaldingLaw(), walls = :both, formulation = :local,
                 δ = 1.0, u_τ = 1.0, extent = (2π*δ, π*δ, 2δ), noise = 3.0,
                 seed = 1, stepper = :rk3, h_wm = nothing, h_wm_cells = 1,
                 filter = :none, T_f = nothing, wall_model = :equilibrium, tble = (;),
                 bottom_bc = :wall_model, θ_walls = nothing,
                 sgs_stress = :laplacian, Pr = 0.71, θ₀ = 1, g_accel = 1,
                 surface_heat_flux = nothing) -> WallBoundedCase

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

The wall-model input keywords `h_wm`, `h_wm_cells`, `filter` and `T_f`
(P57-b) are those of [`half_channel_case`](@ref), applied at each
modelled wall (`T_f` default ``0.5\\,δ/u_τ``).

`wall_model = :nonequilibrium` (P57-c, #189) replaces the equilibrium
law by a [`TBLEWallModel`](@ref) at each modelled wall, built with
`ν`, ``κ_{wm}`` initially the law's κ (`SpaldingLaw`), and the keywords
in the named tuple `tble` (`terms`, `dynamic_kappa`, `T_κ`, `N`, `r`,
…). Its columns take the input of the matching-height and filter
keywords, the case's mean pressure gradient ``∂p/∂x = -u_τ^2/δ`` plus
the resolved one, and are advanced once per step after the projection.
`formulation` must be `:local`.

**Wall-resolved, stratified channel** (P55-f, #191; reused by P59-e,
#195):

  * `bottom_bc = :noslip` — resolved [`NoSlipWall`](@ref)s instead of the
    wall model: no-slip at the bottom and, with `walls = :both`, at the
    top (`walls = :bottom` keeps a free-slip top, an open channel). The
    wall stress is then the viscous flux of the core through the ghost
    cells, plus the SGS flux there (the closures carry no wall damping;
    resolve the viscous sublayer or use a dynamic closure);
    `τ_wall_bottom`/`τ_wall_top` report the molecular part,
    `wall_shear_stress` of P59-a. The wall-model keywords (`h_wm`,
    `h_wm_cells`, `filter`, `wall_model = :nonequilibrium`) do not apply
    and raise an `ArgumentError`.
  * `θ_walls = (θ_bottom, θ_top)` — isothermal walls (needs
    `bottom_bc = :noslip`). θ is an active Boussinesq scalar with
    reference `θ₀` and gravity `g_accel` (the friction Richardson number
    is [`channel_richardson`](@ref); [`stratified_channel_walls`](@ref)
    gives the walls for a target Ri_τ), molecular diffusivity ``ν/Pr``
    (the core diffuses θ with ν; the difference ``(1/Pr - 1)ν∇²θ`` is
    added) and the wall conduction ``(ν/Pr)(θ_w - θ_1)/(Δz/2)`` — the
    Dirichlet condition of a ghost value ``2θ_w - θ_1`` — added to the
    wall-adjacent cells (the θ halos are mirrored, so the SGS scalar
    flux through a wall face is zero, as ``κ_t → 0`` at a resolved
    wall). The initial θ is the conduction profile. The wall heat
    fluxes of a step are `q_wall_bottom`, `q_wall_top`.
  * `sgs_stress = :symmetric` hands the stress form to the core, so that
    an anisotropic closure's vertical viscosity (`StabilityCorrected`)
    reaches the momentum equation; `:laplacian` (default) uses `νh`
    only.

**Flux-cooled open channel** (P59-e, #195): `surface_heat_flux = Q`
(K m s⁻¹, the upward kinematic flux ``\\overline{w'θ'}_0``; negative
cools) with `bottom_bc = :noslip` (and, for the open channel of
Nieuwstadt 2005, `walls = :bottom` and a depth `extent[3] = δ`) is the
stable-channel collapse case: buoyancy on with reference `θ₀` and gravity
`g_accel`, θ uniform at `θ₀` initially, `Q` added to the wall-adjacent
cells as the prescribed bottom θ flux of the simulation (the top and the
bottom are otherwise insulated), molecular diffusivity ``ν/Pr`` as above.
`q_wall_bottom` is then `Q`. The Obukhov length of the flux and the
nominal ``u_τ`` gives ``h/L = κ g |Q| δ/(θ_0 u_τ^3)``
(`relaminarisation_criteria` of `Aeolus.Transition`). Exclusive with
`θ_walls`; a wall-modelled flux-cooled case is
[`half_channel_case`](@ref)`(surface_heat_flux = Q)`.

# References (stratified channel)

  * Nieuwstadt, F. T. M. (2005). Direct numerical simulation of stable
    channel flow at large stability. *Boundary-Layer Meteorol.*, 116,
    277-299.
  * García-Villalba, M. & del Álamo, J. C. (2011). Turbulence modification
    by stable stratification in channel flow. *Phys. Fluids*, 23, 045104.
  * Armenio, V. & Sarkar, S. (2002). An investigation of stably stratified
    turbulent channel flow using large-eddy simulation. *J. Fluid Mech.*,
    459, 1-42.

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
                      h_wm = nothing, h_wm_cells = 1, filter::Symbol = :none, T_f = nothing,
                      wall_model::Symbol = :equilibrium, tble = (;),
                      bottom_bc::Symbol = :wall_model, θ_walls = nothing,
                      sgs_stress::Symbol = :laplacian, Pr::Real = 0.71,
                      θ₀::Real = 1.0, g_accel::Real = 1.0,
                      surface_heat_flux = nothing)
    Re_τ > 0 || throw(ArgumentError("Re_τ must be positive, got $Re_τ"))
    ν = u_τ * δ / Re_τ
    if surface_heat_flux !== nothing
        surface_heat_flux isa Real ||
            throw(ArgumentError("surface_heat_flux must be a number (K m/s) or nothing"))
        bottom_bc === :noslip ||
            throw(ArgumentError("channel_case: surface_heat_flux needs bottom_bc = :noslip " *
                                "(the wall-modelled, flux-cooled case is half_channel_case)"))
        θ_walls === nothing ||
            throw(ArgumentError("channel_case: θ_walls and surface_heat_flux are exclusive"))
        θ₀ > 0 || throw(ArgumentError("θ₀ must be positive, got $θ₀"))
        return _build_case(:channel, _dims(N), Tuple(float.(extent)), closure, wall_law,
                           ν, walls, formulation, u_τ, δ, noise, seed, stepper,
                           _case_inputs(walls, h_wm, h_wm_cells, filter, T_f, u_τ, δ),
                           nothing, wall_model, tble;
                           bottom_bc = bottom_bc, sgs_stress = sgs_stress, Pr = Pr,
                           θ_ref = θ₀, θ_flux_bottom = float(surface_heat_flux),
                           flux_g = g_accel)
    end
    return _build_case(:channel, _dims(N), Tuple(float.(extent)), closure, wall_law,
                       ν, walls, formulation, u_τ, δ, noise, seed, stepper,
                       _case_inputs(walls, h_wm, h_wm_cells, filter, T_f, u_τ, δ),
                       nothing, wall_model, tble;
                       bottom_bc = bottom_bc, sgs_stress = sgs_stress, θ_walls = θ_walls,
                       Pr = Pr, θ₀ = θ₀, g_accel = g_accel)
end

"""
    channel_richardson(case) -> Ri_τ

The friction Richardson number of a channel with θ walls,
``Ri_τ = g\\,(θ_{top} - θ_{bottom})\\,δ/(θ_0 u_τ^2)`` (García-Villalba &
del Álamo 2011, with the density difference of the walls written as the
Boussinesq buoyancy difference ``gΔθ/θ_0``); 0 without θ walls.
"""
function channel_richardson(case::WallBoundedCase)
    case.θ_walls === nothing && return zero(case.u_star)
    s = case.sim
    return s.g_accel * (case.θ_walls[2] - case.θ_walls[1]) * case.height /
           (s.θ₀ * case.u_star^2)
end

"""
    stratified_channel_walls(Ri_τ; θ₀ = 1, g_accel = 1, δ = 1, u_τ = 1, θ_bottom = θ₀)
        -> (θ_bottom, θ_top)

The wall temperatures of a stably stratified channel (hot top) at the
friction Richardson number `Ri_τ` (see [`channel_richardson`](@ref)), for
the `θ_walls` keyword of [`channel_case`](@ref).
"""
stratified_channel_walls(Ri_τ::Real; θ₀::Real = 1.0, g_accel::Real = 1.0, δ::Real = 1.0,
                         u_τ::Real = 1.0, θ_bottom::Real = θ₀) =
    (float(θ_bottom), float(θ_bottom + Ri_τ * θ₀ * u_τ^2 / (g_accel * δ)))

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
every substage, with the `wind` / `height` of the case's
[`MatchingInput`](@ref)s when it has them; their temporal filters are
advanced once, after the step; a [`TBLEWallModel`](@ref)'s columns are
then advanced with [`update_wall_model!`](@ref), with the mean pressure
gradient ``-G`` plus the resolved one). Updates `case.τ_wall_bottom` /
`case.τ_wall_top` with the plane-mean stress the step applied.
"""
function advance_channel!(case::WallBoundedCase{S, C, W, T}, dt::Real) where {S, C, W, T}
    s = case.sim; g = s.mesh
    G = case.forcing
    A = g.Lx * g.Ly
    case._stage = 0
    τb = Ref(zero(T)); τt = Ref(zero(T)); τf = Ref(zero(T))
    rk3 = case.stepper === :rk3
    dr = case.roughness
    dr === nothing || _update_case_roughness!(case, dr)
    nforce = Ref(0)
    force! = function (sim, Δt)
        ii, jj, kk = Grids.interior(sim.mesh)
        @inbounds for k in kk, j in jj, i in ii
            sim.tu.data[i, j, k] += G
        end
        if dr !== nothing
            nforce[] += 1
            wgt = rk3 ? _RK3_WEIGHTS[mod1(nforce[], 3)] : 1.0
            τf[] += wgt * _form_drag!(sim, dr)
        end
        return nothing
    end
    wall! = function (sim, Δt)
        case._stage += 1
        wgt = rk3 ? _RK3_WEIGHTS[mod1(case._stage, 3)] : 1.0
        Fb = _apply_case_wall!(case, sim, Δt, :bottom)
        τb[] += wgt * (-Fb.Fx) / A
        if case.walls === :both
            Ft = _apply_case_wall!(case, sim, Δt, :top)
            τt[] += wgt * (-Ft.Fx) / A
        end
        return nothing
    end
    noslip = case.bottom_bc === :noslip
    qb = Ref(zero(T)); qt = Ref(zero(T))
    hstage = Ref(0)
    if case.θ_walls !== nothing || case._lapθ !== nothing
        force_θ! = function (sim, Δt)
            force!(sim, Δt)
            hstage[] += 1
            wgt = rk3 ? _RK3_WEIGHTS[mod1(hstage[], 3)] : 1.0
            fb, ft = _add_θ_walls!(case, sim)
            qb[] += wgt * fb; qt[] += wgt * ft
            return nothing
        end
    else
        force_θ! = force!
    end
    post! = noslip ? nothing : wall!
    if rk3
        step_rk3!(s, dt; sgs = case.closure, sgs_stress = case.sgs_stress,
                  body_force! = force_θ!, post_predictor! = post!)
    else
        step!(s, dt; sgs = case.closure, sgs_stress = case.sgs_stress,
              body_force! = force_θ!, post_predictor! = post!)
    end
    if case.inputs !== nothing
        update_input_filter!(case.inputs.bottom, s, dt; side = :bottom)
        case.inputs.top === nothing || update_input_filter!(case.inputs.top, s, dt; side = :top)
    end
    if case.wall isa TBLEWallModel
        # the TBLE columns, to the end of the step: the mean pressure
        # gradient −G plus the resolved one
        for side in (case.walls === :both ? (:bottom, :top) : (:bottom,))
            mi = case.inputs === nothing ? nothing : getfield(case.inputs, side)
            update_wall_model!(case.wall, s, dt; side = side, input = mi, dpdx = -G)
        end
    end
    if noslip
        case.τ_wall_bottom = wall_shear_stress(s; side = :bottom)
        case.τ_wall_top = case.walls === :both ? -wall_shear_stress(s; side = :top) : zero(T)
    else
        case.τ_wall_bottom = τb[]
        case.τ_wall_top = τt[]
    end
    case.τ_form = τf[]
    # a prescribed surface flux (P58-e / P59-e) enters exactly as given
    case.q_wall_bottom = case.θ_walls === nothing ? T(s.θ_flux_bottom) : qb[]
    case.q_wall_top = qt[]
    case.nsteps += 1
    return case
end

# The θ walls of a stage: the molecular diffusion beyond the core's ν∇²θ,
# (ν/Pr − ν)∇²θ, and the conduction through each isothermal wall face into
# the wall-adjacent cells (none for a prescribed surface flux, P59-e: the
# mirrored θ halos make the correction's wall flux zero, so the flux
# through the wall stays exactly the prescribed one). Returns the plane-mean kinematic heat fluxes
# (upward positive) through the bottom and top walls. The θ halos were
# filled (mirror) by the core before the body force.
function _add_θ_walls!(case::WallBoundedCase, sim)
    g = sim.mesh
    Nx, Ny, Nz, Hx, Hy, Hz = g.Nx, g.Ny, g.Nz, g.Hx, g.Hy, g.Hz
    κ = sim.ν / case.Pr
    θ = sim.θ.data; tθ = sim.tθ.data
    if κ != sim.ν
        L = case._lapθ
        laplacian(L, sim.θ)
        a = κ - sim.ν
        @inbounds for k in 1:Nz, j in 1:Ny, i in 1:Nx
            tθ[Hx+i, Hy+j, Hz+k] += a * L.data[Hx+i, Hy+j, Hz+k]
        end
    end
    case.θ_walls === nothing && return zero(eltype(θ)), zero(eltype(θ))
    θb, θt = case.θ_walls
    c = 2κ / g.dz                       # conduction across the half cell, per kelvin
    fb = zero(eltype(θ)); ft = zero(eltype(θ))
    top = case.walls === :both
    @inbounds for j in 1:Ny, i in 1:Nx
        I, J = Hx + i, Hy + j
        qb = c * (θb - θ[I, J, Hz+1])                  # into the fluid (upward)
        tθ[I, J, Hz+1] += qb / g.dz
        fb += qb
        if top
            qt = c * (θ[I, J, Hz+Nz] - θt)             # upward, out of the fluid
            tθ[I, J, Hz+Nz] -= qt / g.dz
            ft += qt
        end
    end
    return fb / (Nx * Ny), ft / (Nx * Ny)
end

function _apply_case_wall!(case, sim, Δt, side)
    if case.wall isa TBLEWallModel
        mi = case.inputs === nothing ? nothing : getfield(case.inputs, side)
        return apply_wall_stress!(sim, case.wall, Δt; side = side, input = mi)
    end
    case.inputs === nothing &&
        return apply_wall_stress!(sim, case.wall, Δt; side = side, formulation = case.formulation)
    mi = getfield(case.inputs, side)
    wind, height = matching_input!(mi, sim; side = side)
    return apply_wall_stress!(sim, case.wall, Δt; side = side, formulation = case.formulation,
                              wind = wind, height = height)
end

# P57-e: the dynamic roughness of the bottom from the start-of-step
# velocity at the wall-model input (first cell, or the matching height and
# filters of the case's MatchingInput), and the per-column laws.
function _update_case_roughness!(case, dr::DynamicRoughness)
    s = case.sim; g = s.mesh
    if case.inputs === nothing
        uc, vc = _first_cell_velocity(s)
        h = g.dz / 2
    else
        uc, vc, h = _sample_matching(case.inputs.bottom, s, :bottom, nothing)
    end
    update_dynamic_roughness!(dr, uc, vc, h)
    dynamic_wall_laws!(case.wall.law, dr)
    return dr
end

# Cell-centre (u, v) of the first level of a horizontally periodic sim.
function _first_cell_velocity(sim)
    g = sim.mesh
    Nx, Ny, Hx, Hy, Hz = g.Nx, g.Ny, g.Hx, g.Hy, g.Hz
    u, v = sim.u.data, sim.v.data
    T = eltype(u)
    uc = zeros(T, Nx, Ny); vc = zeros(T, Nx, Ny)
    @inbounds for j in 1:Ny, i in 1:Nx
        ie = i == Nx ? 1 : i + 1; jn = j == Ny ? 1 : j + 1
        uc[i, j] = (u[Hx+i, Hy+j, Hz+1] + u[Hx+ie, Hy+j, Hz+1]) / 2
        vc[i, j] = (v[Hx+i, Hy+j, Hz+1] + v[Hx+i, Hy+jn, Hz+1]) / 2
    end
    return uc, vc
end

# The form drag of the resolved relief, f/Δz on the first-level faces
# (the mean of the two adjacent columns' surface_gradient_drag), added to
# the tendencies; returns its plane-mean kinematic x stress (m² s⁻²).
function _form_drag!(sim, dr::DynamicRoughness)
    g = sim.mesh
    Nx, Ny, Hx, Hy, Hz, dz = g.Nx, g.Ny, g.Hx, g.Hy, g.Hz, g.dz
    st = dr.stats
    uc, vc = _first_cell_velocity(sim)
    T = eltype(uc)
    fx = zeros(T, Nx, Ny); fy = zeros(T, Nx, Ny)
    @inbounds for I in eachindex(uc)
        fx[I], fy[I] = surface_gradient_drag(uc[I], vc[I], st.dhx[I], st.dhy[I]; C_d = dr.C_d)
    end
    τ = zero(T)
    @inbounds for j in 1:Ny, i in 1:Nx
        iw = i == 1 ? Nx : i - 1; js = j == 1 ? Ny : j - 1
        ax = (fx[i, j] + fx[iw, j]) / 2
        sim.tu.data[Hx+i, Hy+j, Hz+1] += ax / dz
        sim.tv.data[Hx+i, Hy+j, Hz+1] += (fy[i, j] + fy[i, js]) / 2 / dz
        τ -= ax
    end
    return τ / (Nx * Ny)
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
    noslip = case.bottom_bc === :noslip      # the wall flux is then the viscous + SGS one
    return total_stress_profile(s.u, s.v, s.w, νt; ν = s.ν,
                                τ_wall_bottom = noslip ? zero(case.u_star) : case.τ_wall_bottom,
                                τ_wall_top = noslip ? zero(case.u_star) : case.τ_wall_top)
end

"""
    run_channel!(case, t_end; dt = nothing, cfl = 0.5, average_from = Inf,
                 sample_every = 1, callback = nothing, checkpoint = nothing,
                 checkpoint_components = (;)) -> stats

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

**Checkpoints** (#245). With `checkpoint` (an `IO.CheckpointSchedule`) the
run resumes from the newest checkpoint of its folder (the simulation, the
case — closure memory, wall model, inputs, roughness, counters — and the
averages above), writes periodic checkpoints, and a final one at `t_end`.
`checkpoint_components` (a `NamedTuple`) adds the state of the caller's
`callback`. A restarted run returns the same `stats`, bit for bit, as an
uninterrupted one. Use one checkpoint `prefix` per call when a driver calls
`run_channel!` more than once.
"""
function run_channel!(case::WallBoundedCase, t_end::Real; dt = nothing, cfl::Real = 0.5,
                      average_from::Real = Inf, sample_every::Integer = 1,
                      callback = nothing, checkpoint = nothing, checkpoint_components = (;))
    s = case.sim; g = s.mesh; Nz = g.Nz
    keys_ = (:resolved, :sgs, :viscous, :wall, :total)
    # the run's accumulators, in one checkpointable Dict (#245)
    R = Dict{Symbol,Any}(:Usum => zeros(Nz), :ssum => Dict(k => zeros(Nz + 1) for k in keys_),
                         :τb => 0.0, :τt => 0.0, :tw => 0.0, :ts => Float64[], :τs => Float64[],
                         :nsamp => 0, :n0 => case.nsteps)
    comps = (case = case, run_channel = R, checkpoint_components...)
    checkpoint === nothing || resume!(checkpoint, s; components = comps)
    nw = case.walls === :both ? 2 : 1
    t0 = time()
    while s.t < t_end - 1e-12 * max(1.0, abs(t_end))
        h = dt === nothing ? channel_dt(case; cfl = cfl) : float(dt)
        h = min(h, t_end - s.t)
        advance_channel!(case, h)
        callback === nothing || callback(case)
        if s.t >= average_from && (case.nsteps - R[:n0]) % sample_every == 0
            R[:nsamp] += 1
            um, _ = _uv_means(s)
            R[:Usum] .+= um
            p = total_stress_profile(case)
            for k in keys_
                R[:ssum][k] .+= getfield(p, k)
            end
            R[:τb] += h * case.τ_wall_bottom; R[:τt] += h * case.τ_wall_top; R[:tw] += h
            push!(R[:ts], s.t); push!(R[:τs], (case.τ_wall_bottom + case.τ_wall_top) / nw)
        end
        checkpoint === nothing || maybe_checkpoint!(checkpoint, s; step = case.nsteps, components = comps)
    end
    # a final checkpoint: a rerun of a finished phase returns at once
    checkpoint === nothing || maybe_checkpoint!(checkpoint, s; step = case.nsteps, force = true,
                                                components = comps)
    Usum, ssum, τb_sum, τt_sum, tw = R[:Usum], R[:ssum], R[:τb], R[:τt], R[:tw]
    ts, τs, nsamp, n0 = R[:ts], R[:τs], R[:nsamp], R[:n0]
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
