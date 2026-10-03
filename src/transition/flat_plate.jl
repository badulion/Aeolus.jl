# P59-d (#185) — the fringe region, Blasius inflow and the boundary-layer
# integral diagnostics of a spatially developing flat-plate boundary layer.
#
# Included inside `module Transition`, after blasius.jl.
#
# A spatially developing boundary layer on the x-periodic FFT grid: the
# fringe region at the end of the domain (Nordström, Nordin & Henningson
# 1999; Bertolotti, Herbert & Spalart 1992) relaxes the outflow back to the
# Blasius inflow profile, so the flow that re-enters at x = x0 is the
# laminar boundary layer at a prescribed distance x_in from the leading
# edge. The relaxation rate λ(x) is the shared `Operators.DampingProfile`
# (NordstromRamp) — the same implementation as the Rayleigh sponge.

# =========================================================================
# Fringe region
# =========================================================================

"""
    FringeRegion(; x_start, x_end, rise, fall = rise, λ_max)

Fringe region for a spatially developing flow in an `x`-periodic domain:
the relaxation term ``-λ(x)(\\mathbf u - \\mathbf u_{\\rm target})`` with

    λ(x) = λ_max [ S((x − x_start)/rise) − S((x − x_end)/fall + 1) ]

(Nordström, Nordin & Henningson 1999, eq. 2.3; `S` is
[`smooth_step`](@ref Aeolus.Operators.smooth_step)). λ is zero outside
`[x_start, x_end]` and `λ_max` on the plateau
`[x_start + rise, x_end − fall]`. Apply it with [`add_fringe!`](@ref) or
through `body_force!` with [`fringe_forcing`](@ref).

The rate is an `Operators.DampingProfile` with a `NordstromRamp`: the
one damping-profile implementation that [`add_sponge!`](@ref
Aeolus.Operators.add_sponge!) also uses.

`λ_max` must make the relaxation fast against advection through the
plateau: Nordström et al. require ``λ_{\\max} L_{\\rm plateau}/U ≫ 1``;
`λ_max · (x_end − x_start − rise − fall)/U ≈ 10–20` recovers the target
to ``< 1\\%`` of ``U``.

Reference: Nordström, J., Nordin, N. & Henningson, D. (1999). The
fringe region technique and the Fourier method used in the direct
numerical simulation of spatially evolving viscous flows. *SIAM J. Sci.
Comput.*, 20(4), 1365-1393.
"""
struct FringeRegion{P<:DampingProfile}
    profile::P
end

function FringeRegion(; x_start::Real, x_end::Real, rise::Real, fall::Real = rise,
                      λ_max::Real)
    x_end > x_start || throw(ArgumentError("FringeRegion: x_end must exceed x_start"))
    rise + fall <= x_end - x_start ||
        throw(ArgumentError("FringeRegion: rise + fall exceeds the fringe length"))
    λ_max >= 0 || throw(ArgumentError("FringeRegion: λ_max must be ≥ 0"))
    p = DampingProfile(NordstromRamp(), float(λ_max), float(x_start), float(rise);
                       stop = float(x_end), fall = float(fall))
    return FringeRegion(p)
end

"""
    fringe_rate(fringe, x) -> λ

The relaxation rate ``λ(x)`` (s⁻¹) of `fringe` at `x`.
"""
fringe_rate(fr::FringeRegion, x::Real) = damping_rate(fr.profile, x)

"""
    add_fringe!(tu, tv, tw, u, v, w, fringe::FringeRegion, target) -> (tu, tv, tw)

Add the fringe relaxation ``-λ(x)(u_i - u_{i,\\rm target})`` to the three
momentum tendencies, with ``λ`` evaluated at each component's own
staggered `x` position (`FaceX` for `u`, cell centres for `v`, `w`).
`target` is a 3-tuple or a `NamedTuple` `(u, v, w)` whose entries are
numbers, `Field`s or arrays of the full (halo-included) field size.
Only cells inside the fringe are touched.

The kernel is `Operators.FiniteVolume._add_damping!`, the one that the
Rayleigh sponge [`add_sponge!`](@ref Aeolus.Operators.add_sponge!) uses.
"""
function add_fringe!(tu::Field{FaceX}, tv::Field{FaceY}, tw::Field{FaceZ},
                     u::Field{FaceX}, v::Field{FaceY}, w::Field{FaceZ},
                     fr::FringeRegion, target)
    g = mesh(u)
    ut, vt, wt = _target3(target)
    _add_damping!(tu, u, fr.profile, collect(xnodes(g, FaceX())), 1, ut, g)
    _add_damping!(tv, v, fr.profile, collect(xnodes(g, FaceY())), 1, vt, g)
    _add_damping!(tw, w, fr.profile, collect(xnodes(g, FaceZ())), 1, wt, g)
    return tu, tv, tw
end

_target3(t::NamedTuple) = (t.u, t.v, t.w)
_target3(t::Tuple) = t

"""
    fringe_forcing(fringe, target; update! = nothing) -> body_force!(sim, Δt)

A `body_force!` callback for `step!` / `step_rk3!` that adds the fringe
relaxation to the momentum tendencies of a `BoussinesqSimulation`. When
`update!` is given, it is called as `update!(target, sim.t)` first (a
time-dependent target, e.g. Blasius + free-stream disturbances advected
through the fringe; see [`FreeStreamDisturbance`](@ref)).
"""
function fringe_forcing(fr::FringeRegion, target; update! = nothing)
    return function (sim, Δt)
        update! === nothing || update!(target, sim.t)
        add_fringe!(sim.tu, sim.tv, sim.tw, sim.u, sim.v, sim.w, fr, target)
        return nothing
    end
end

# =========================================================================
# Blasius inflow
# =========================================================================

"""
    fringe_distance(mesh, fringe, x; x_le, x_min = (mesh.x0 - x_le)/4) -> x_d

Distance from the virtual leading edge `x_le` that the Blasius target and
the free-stream boundary use at `x`. On the plate (upstream of the fringe
start) it is `x − x_le`. Inside the fringe it is the distance of the
*upstream periodic image*, `x − L_x − x_le`, clipped below at `x_min`: the
target is the boundary layer that is developing towards the inflow
(`x_d = x_in` at `x = x0 + L_x`, the fringe exit), so the profile the
fringe leaves behind is already in equilibrium with its own growth and
its `w` is consistent with continuity (a target frozen at `x_in`
releases a layer whose shape factor measured 2.37 instead of 2.59 at the
inflow).
"""
function fringe_distance(g::RegularGrid, fr::FringeRegion, x::Real; x_le::Real,
                         x_min::Real = (g.x0 - x_le) / 4)
    xw = g.x0 + mod(x - g.x0, g.Lx)                # periodic image in [x0, x0 + Lx)
    xw <= fr.profile.start && return xw - x_le
    return max(xw - g.Lx - x_le, x_min)
end

"""
    blasius_target(mesh, fringe; x_le, U∞ = 1, ν, x_min = (mesh.x0 - x_le)/4,
                   z_wall = mesh.z0) -> (u, v, w)

Fringe target fields: the Blasius boundary layer at the distance
[`fringe_distance`](@ref) from the virtual leading edge `x_le` (heights
measured from `z_wall`), on the staggered locations (`u @ FaceX`,
`v = 0 @ FaceY`, `w @ FaceZ`), uniform in `y`. At the fringe exit
(`x = x0 + L_x`, the periodic image of the inflow `x0`) this is the
Blasius layer at `x_in = x0 − x_le`. Returned as a `NamedTuple` of
`Field`s, halos included; outside the fringe the values are not used.
"""
function blasius_target(g::RegularGrid, fr::FringeRegion; x_le::Real, U∞::Real = 1.0,
                        ν::Real, x_min::Real = (g.x0 - x_le) / 4, z_wall::Real = g.z0)
    ut = Field(FaceX(), g); vt = Field(FaceY(), g); wt = Field(FaceZ(), g)
    _fill_blasius!(ut, wt, (i, x) -> fringe_distance(g, fr, x; x_le = x_le, x_min = x_min),
                   U∞, ν, z_wall)
    return (u = ut, v = vt, w = wt)
end

# Fill u (FaceX) and w (FaceZ) with the Blasius solution at distance
# xfun(i_storage, ·) from the leading edge; every storage cell, halos too.
function _fill_blasius!(u::Field{FaceX}, w::Field{FaceZ}, xdist, U∞, ν, z_wall)
    g = mesh(u)
    nx, ny, nz = size(u.data)
    for k in 1:nz
        zc = g.z0 + (k - g.Hz - 0.5) * g.dz - z_wall     # u: cell-centred in z
        zf = g.z0 + (k - g.Hz - 1) * g.dz - z_wall       # w: bottom face of cell k
        for i in 1:nx
            xu = g.x0 + (i - g.Hx - 1) * g.dx            # FaceX x position
            xc = xu + 0.5 * g.dx                          # cell-centre x (w)
            uu, _ = blasius_velocity(xdist(i, xu), abs(zc); U∞ = U∞, ν = ν)
            _, ww = blasius_velocity(xdist(i, xc), max(zf, 0.0); U∞ = U∞, ν = ν)
            zc < 0 && (uu = -uu)                          # odd reflection below the wall
            for j in 1:ny
                u.data[i, j, k] = uu
                w.data[i, j, k] = zf <= 0 ? 0.0 : ww
            end
        end
    end
    return u, w
end

"""
    blasius_freestream(mesh, fringe; x_le, U∞ = 1, ν, x_min = (mesh.x0 - x_le)/4)
        -> FreeStreamBoundary

The top boundary of a spatially developing Blasius layer whose inflow
(the fringe exit, periodic image of `x = x0`) sits at distance
`x_in = x0 − x_le` from the virtual leading edge `x_le`:

* the displacement velocity of the free stream,
  ``W_∞ = U_∞\\,dδ^*/dx = \\tfrac12 δ^*_c U_∞ \\sqrt{ν/(U_∞ x_d)}``
  (0.8604 ``U_∞/\\sqrt{Re_x}``), with ``x_d`` the
  [`fringe_distance`](@ref) — the distance from the leading edge on the
  plate, the developing target layer inside the fringe;
* plus the return of the plate's outflow, an inflow bump
  ``-C\\,4r(1-r)``, ``r = S((x - x_{start})/Δ_{rise})``, confined to the
  fringe rise — upstream of the plateau where the velocity is pinned to
  the target, so the free stream re-enters the plate at ``U_∞``. ``C``
  makes the net transpiration exactly zero (the solvability condition of
  the periodic pressure problem, see
  [`FreeStreamBoundary`](@ref Aeolus.BoundaryConditions.FreeStreamBoundary)).
  With the bump spread over the whole fringe instead, the inflow in the
  fall region raised the free stream entering the plate to 1.08 ``U_∞``.
"""
function blasius_freestream(g::RegularGrid, fr::FringeRegion; x_le::Real,
                            U∞::Real = 1.0, ν::Real, x_min::Real = (g.x0 - x_le) / 4)
    p = fr.profile
    x_start, x_end = p.start, p.stop
    x_in = g.x0 - x_le
    x_in > 0 || throw(ArgumentError("blasius_freestream: the inflow must lie downstream of x_le"))
    xs = [g.x0 + (i - g.Hx - 0.5) * g.dx for i in 1:(g.Nx + 2g.Hx)]   # FaceZ x (cell centres)
    xw = [g.x0 + mod(x - g.x0, g.Lx) for x in xs]       # periodic image in [x0, x0 + Lx)
    δc = default_blasius().delta_star
    W(xd) = xd > 0 ? 0.5 * δc * U∞ * sqrt(ν / (U∞ * xd)) : 0.0
    base = similar(xs); bump = similar(xs)
    for (n, x) in enumerate(xw)
        base[n] = W(fringe_distance(g, fr, x; x_le = x_le, x_min = x_min))
        rr = smooth_step((x - x_start) * p.inv_rise)
        bump[n] = (x_start < x < x_start + 1 / p.inv_rise) ? 4rr * (1 - rr) : 0.0
    end
    ii = (g.Hx + 1):(g.Hx + g.Nx)
    den = sum(bump[i] for i in ii)
    den > 0 || throw(ArgumentError("blasius_freestream: the fringe rise has no cells"))
    C = sum(base[i] for i in ii) / den
    wb = base .- C .* bump
    wb .-= sum(wb[i] for i in ii) / g.Nx          # round-off of the mean
    return FreeStreamBoundary(wb)
end

"""
    set_blasius!(sim; x_le, U∞ = 1, ν = sim.ν, x_min, z_wall = sim.mesh.z0) -> sim

Initial condition: the Blasius boundary layer of a plate whose leading
edge sits at `x = x_le`, evaluated at distance `max(x − x_le, x_min)`
(so the fringe region, downstream of the plate end, and anything upstream
of `x_le + x_min` carry the profile at `x_min`). Sets `u` and `w`
(halos included); `v` is zeroed. Project once (one `step!`) before
measuring divergence: the sampled similarity solution is divergence-free
only to truncation error.
"""
function set_blasius!(sim::BoussinesqSimulation; x_le::Real, U∞::Real = 1.0,
                      ν::Real = sim.ν, x_min::Real, z_wall::Real = sim.mesh.z0)
    fill!(sim.v.data, 0)
    _fill_blasius!(sim.u, sim.w, (i, x) -> max(x - x_le, x_min), U∞, ν, z_wall)
    return sim
end

# =========================================================================
# Boundary-layer integral diagnostics as functions of x
# =========================================================================
#
# All diagnostics take the streamwise velocity as an (Nx × Nz) matrix ū of
# u @ FaceX, averaged over y (and, by the caller, over time), with the wall
# half a cell below the first row. The `sim` methods average the current
# state over y. Quadrature is the midpoint rule on the cell-centred values,
# second order, consistent with the finite-volume state.

"""
    streamwise_profile(sim) -> ū :: Matrix (Nx × Nz)

`u @ FaceX` averaged over `y`: one column per height, one row per
streamwise `FaceX` position (`xnodes(mesh, FaceX())`).
"""
function streamwise_profile(sim::BoussinesqSimulation)
    ui = interior_view(sim.u)
    return dropdims(sum(ui; dims = 2); dims = 2) ./ size(ui, 2)
end

"""
    edge_velocity(ū; level = size(ū, 2)) -> Vector

The local free-stream (edge) velocity ``U_e(x)``: `ū` at the interior
level `level` (default the top one, which in a closed-lid domain lies
outside the boundary layer).
"""
edge_velocity(ū::AbstractMatrix; level::Integer = size(ū, 2)) = ū[:, level]

_edge(ū, U_e::Nothing) = edge_velocity(ū)
_edge(ū, U_e::Real) = fill(float(U_e), size(ū, 1))
_edge(ū, U_e::AbstractVector) = U_e

"""
    displacement_thickness(ū, dz; U_e = edge_velocity(ū)) -> δ*(x)
    displacement_thickness(sim; U_e = nothing) -> δ*(x)

``δ^*(x) = ∫_0^H (1 - ū/U_e)\\,dz`` (midpoint rule on the cell-centred
values, the wall at the bottom face). `U_e` is a number, a vector over
`x`, or `nothing` for the local edge velocity.
"""
function displacement_thickness(ū::AbstractMatrix, dz::Real; U_e = nothing)
    Ue = _edge(ū, U_e)
    return [dz * sum(1 - ū[i, k] / Ue[i] for k in axes(ū, 2)) for i in axes(ū, 1)]
end

"""
    momentum_thickness(ū, dz; U_e = edge_velocity(ū)) -> θ(x)
    momentum_thickness(sim; U_e = nothing) -> θ(x)

``θ(x) = ∫_0^H (ū/U_e)(1 - ū/U_e)\\,dz`` (midpoint rule).
"""
function momentum_thickness(ū::AbstractMatrix, dz::Real; U_e = nothing)
    Ue = _edge(ū, U_e)
    return [dz * sum((ū[i, k] / Ue[i]) * (1 - ū[i, k] / Ue[i]) for k in axes(ū, 2))
            for i in axes(ū, 1)]
end

"""
    shape_factor(ū, dz; U_e = edge_velocity(ū)) -> H(x)
    shape_factor(sim; U_e = nothing) -> H(x)

``H = δ^*/θ``: 2.59 for the Blasius layer, ≈ 1.4 for a turbulent
zero-pressure-gradient layer.
"""
shape_factor(ū::AbstractMatrix, dz::Real; U_e = nothing) =
    displacement_thickness(ū, dz; U_e = U_e) ./ momentum_thickness(ū, dz; U_e = U_e)

"""
    skin_friction(ū, dz, ν; U_e = edge_velocity(ū), u_wall = 0) -> c_f(x)
    skin_friction(sim; U_e = nothing) -> c_f(x)

``c_f(x) = 2τ_w/U_e^2`` with the molecular wall stress
``τ_w = ν\\,2(ū_1 - u_{\\rm wall})/Δz`` — exactly the one-sided gradient
the viscous operator sees across a [`NoSlipWall`](@ref
Aeolus.BoundaryConditions.NoSlipWall) (`(u₁ − u_ghost)/Δz` with the odd
ghost), so the plane mean of ``τ_w(x)`` equals `wall_shear_stress(sim)`.
The `sim` method requires a `NoSlipWall` at the bottom and uses its
`u_wall`.
"""
function skin_friction(ū::AbstractMatrix, dz::Real, ν::Real; U_e = nothing,
                       u_wall::Real = 0)
    Ue = _edge(ū, U_e)
    return [2 * ν * 2 * (ū[i, 1] - u_wall) / dz / Ue[i]^2 for i in axes(ū, 1)]
end

function skin_friction(sim::BoussinesqSimulation; U_e = nothing)
    bc = sim.bcs.bottom
    bc isa NoSlipWall ||
        throw(ArgumentError("skin_friction: the bottom face is a $(nameof(typeof(bc))), not a NoSlipWall"))
    return skin_friction(streamwise_profile(sim), sim.mesh.dz, sim.ν; U_e = U_e,
                         u_wall = bc.u_wall)
end

displacement_thickness(sim::BoussinesqSimulation; U_e = nothing) =
    displacement_thickness(streamwise_profile(sim), sim.mesh.dz; U_e = U_e)
momentum_thickness(sim::BoussinesqSimulation; U_e = nothing) =
    momentum_thickness(streamwise_profile(sim), sim.mesh.dz; U_e = U_e)
shape_factor(sim::BoussinesqSimulation; U_e = nothing) =
    shape_factor(streamwise_profile(sim), sim.mesh.dz; U_e = U_e)

"""
    transition_onset(x, cf, cf_laminar; rise = 0.1) -> (x_onset, index)

Transition onset from a streamwise skin-friction distribution: the
location of the **minimum of c_f** upstream of the first point from which
``c_f`` stays above ``(1 + \\text{rise})\\,c_{f,\\rm lam}`` to the end of
the range — the surface-measurement definition of onset used by
Abu-Ghannam & Shaw (1980) and in the ERCOFTAC T3 series (Roach &
Brierley 1992). `cf_laminar` is a vector (e.g.
`blasius_skin_friction.(Re_x)`). Returns `(NaN, 0)` if ``c_f`` never
departs from laminar.

References: Abu-Ghannam, B. J. & Shaw, R. (1980). Natural transition of
boundary layers — the effects of turbulence, pressure gradient, and flow
history. *J. Mech. Eng. Sci.*, 22, 213-228. Roach, P. E. & Brierley,
D. H. (1992). The influence of a turbulent free-stream on zero pressure
gradient transitional boundary layer development. Part I. In *Numerical
Simulation of Unsteady Flows and Transition to Turbulence*, Cambridge
University Press, 319-347.
"""
function transition_onset(x::AbstractVector, cf::AbstractVector,
                          cf_laminar::AbstractVector; rise::Real = 0.1)
    n = length(cf)
    (length(x) == n && length(cf_laminar) == n) ||
        throw(DimensionMismatch("x, cf and cf_laminar differ in length"))
    above = cf .> (1 + rise) .* cf_laminar
    above[end] || return (NaN, 0)
    i_star = n
    while i_star > 1 && above[i_star-1]
        i_star -= 1
    end
    i_star == 1 && return (x[1], 1)                 # turbulent over the whole range
    imin = argmin(view(cf, 1:i_star))
    return (x[imin], imin)
end

"""
    onset_reynolds_theta_mayle(Tu) -> Re_θ

Mayle (1991) correlation for the momentum-thickness Reynolds number at
transition onset, ``Re_{θ,t} = 400\\,Tu^{-5/8}`` (`Tu` in per cent):
≈ 190 at the T3A level Tu = 3.3 %.

Reference: Mayle, R. E. (1991). The role of laminar-turbulent transition
in gas turbine engines. *J. Turbomach.*, 113, 509-537.
"""
onset_reynolds_theta_mayle(Tu::Real) = 400 * Tu^(-5 / 8)

"""
    onset_reynolds_theta_abu_ghannam_shaw(Tu) -> Re_θ

Abu-Ghannam & Shaw (1980), zero pressure gradient: ``Re_{θ,s} = 163 +
\\exp(6.91 - Tu)`` (`Tu` in per cent): ≈ 200 at Tu = 3.3 %.
"""
onset_reynolds_theta_abu_ghannam_shaw(Tu::Real) = 163 + exp(6.91 - Tu)

# =========================================================================
# Assembled case
# =========================================================================

"""
    flat_plate_setup(; Re_δ0 = 200, Lx = 512, Ly = 1, H = 16, Nx = 256, Ny = 1, Nz = 64,
                     fringe_length = 80, rise = 24, fall = 16, λ_max = 0.4,
                     top = :freestream, halo = (2, 2, 2), sim_kwargs...)
        -> (; sim, fringe, target, forcing!, x_le, x_in, ν)

A spatially developing zero-pressure-gradient flat-plate boundary layer
in units of the inflow displacement thickness ``δ^*_0`` and the free
stream ``U_∞ = 1`` (so ``ν = 1/Re_{δ_0}``): a `NoSlipWall` plate along
the whole bottom, a [`FringeRegion`](@ref) over the last `fringe_length`
of the `x`-periodic domain relaxing to [`blasius_target`](@ref), and the
[`blasius_freestream`](@ref) top (`top = :wall` gives a closed free-slip
lid instead, for comparison). The inflow, at the fringe exit `x = 0`, is
the Blasius layer at ``x_{in} = (Re_{δ_0}/1.7208)^2 ν`` from the virtual
leading edge `x_le = −x_in`, so ``Re_x = (x - x_{le})/ν``. The defaults
span ``Re_x`` = 1.35e4 to 1.0e5 over the plate (`x < Lx − fringe_length`).
`sim_kwargs` (e.g. `C_s`) go to `BoussinesqSimulation`.

Advance with `step_rk3!(sim, dt; body_force! = forcing!)` (centred
advection needs the RK3 stability region); the state is left at rest
(``u = 0``) — set it with [`set_blasius!`](@ref) or a uniform stream.
"""
function flat_plate_setup(; Re_δ0::Real = 200, Lx::Real = 512, Ly::Real = 1, H::Real = 16,
                          Nx::Integer = 256, Ny::Integer = 1, Nz::Integer = 64,
                          fringe_length::Real = 80, rise::Real = 24, fall::Real = 16,
                          λ_max::Real = 0.4, top::Symbol = :freestream,
                          halo = (2, 2, 2), sim_kwargs...)
    ν = 1 / Re_δ0
    x_in = (Re_δ0 / default_blasius().delta_star)^2 * ν
    x_le = -x_in
    g = RegularGrid(size = (Nx, Ny, Nz), extent = (Lx, Ly, H), halo = halo)
    fr = FringeRegion(x_start = Lx - fringe_length, x_end = Lx, rise = rise, fall = fall,
                      λ_max = λ_max)
    topbc = top === :freestream ? blasius_freestream(g, fr; x_le = x_le, ν = ν) :
            top === :wall ? :wall :
            throw(ArgumentError("flat_plate_setup: top must be :freestream or :wall"))
    bcs = FieldBoundaryConditions(:periodic, :periodic; bottom = :noslip, top = topbc)
    sim = BoussinesqSimulation(g; ν = ν, bcs = bcs, sim_kwargs...)
    target = blasius_target(g, fr; x_le = x_le, ν = ν)
    return (sim = sim, fringe = fr, target = target, forcing! = fringe_forcing(fr, target),
            x_le = x_le, x_in = x_in, ν = ν)
end
