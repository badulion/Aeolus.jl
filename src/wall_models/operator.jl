# The wall-stress operator: the stress of a wall law applied, implicitly,
# to the first velocity level next to the z-walls of a RegularGrid.

export AbstractWallModel, WallModel, apply_wall_stress!, wall_stress_term

"""
    AbstractWallModel

A wall model: whatever decides the stress [`apply_wall_stress!`](@ref)
applies at a wall. [`WallModel`](@ref) is the equilibrium (algebraic)
one — a wall law evaluated at the matching point; `TBLEWallModel`
(P57-c, #189) integrates the thin-boundary-layer equations under every
wall column and carries per-column state. Both hand per-column
coefficients ``τ_{w,i} = a\\,u_i + b_i`` to the same implicit,
face-by-face application, so the operator exists once.
"""
abstract type AbstractWallModel end

"""
    WallModel(law; ν = 1.5e-5)

A wall law and the kinematic viscosity it is evaluated with (m² s⁻¹; air
near 15 °C; only smooth-wall laws use it). `law` is one
[`AbstractWallLaw`](@ref) for the whole wall, or an `(Nx, Ny)` matrix of
laws, one per column (a [`MOSTWallLaw`](@ref) per column with its own
roughness and Obukhov length, as the Berlin ground uses).
"""
struct WallModel{L, T<:AbstractFloat} <: AbstractWallModel
    law::L
    ν::T
end
WallModel(law::Union{AbstractWallLaw, AbstractMatrix{<:AbstractWallLaw}}; ν::Real = 1.5e-5) =
    WallModel(law, float(ν))

@inline _law(l::AbstractWallLaw, i, j) = l
@inline _law(l::AbstractMatrix, i, j) = @inbounds l[i, j]

# Wall level of column (i, j), or 0 when the column has no wall there.
@inline _level(::Nothing, i, j, kw, Nz) = kw
@inline _level(m::AbstractMatrix{Bool}, i, j, kw, Nz) = (@inbounds m[i, j]) ? kw : 0
@inline function _level(m::AbstractMatrix{<:Integer}, i, j, kw, Nz)
    k = @inbounds m[i, j]
    return 1 <= k <= Nz ? Int(k) : 0
end

"""
    apply_wall_stress!(sim, wm, Δt; side = :bottom, mask = nothing,
                       formulation = :local, wind = nothing, height = nothing,
                       ustar = nothing, wind_out = nothing) -> (Fx, Fy)

Apply the modelled wall stress of `wm` (a [`WallModel`](@ref), or a bare
[`AbstractWallLaw`](@ref) with the default ν) to the horizontal velocity
of `sim` next to its z-walls, over the step `Δt`. Use it as the
`post_predictor!` hook of a step (on the predicted velocity, before the
projection):

```julia
wm = WallModel(SpaldingLaw(); ν = 1e-5)
step!(sim, dt; post_predictor! = (s, Δt) -> apply_wall_stress!(s, wm, Δt))
```

**Geometry.** The wall-parallel speed of a column is the speed at the
centre of its wall-adjacent cell, ``U = |(\\bar u, \\bar v)|`` (faces
averaged to the centre), at the distance ``h = Δz/2`` from the wall.
`side` is `:bottom` (level 1), `:top` (level `Nz`) or `:both`. `mask`
restricts the wall: `nothing` (the whole plane), an `(Nx, Ny)` `Bool`
matrix (columns with a wall), or an `(Nx, Ny)` integer matrix giving,
per column, the level of the first cell above an immersed floor
(`:bottom` only; a level outside `1:Nz` means no wall — the roof-aware
ground of the Berlin case). A velocity face is shared by two columns; it
gets the mean of the two columns' stresses, a column contributing only
at its own wall level (half of the face's area is each column's).

**Formulations** (Piomelli & Balaras 2002 review all three):

  * `:local` — Schumann (1975) as generalised by Grötzbach (1987): the
    law is applied locally, ``τ_{w,i} = τ_w(U)\\,u_i/U``.
  * `:schumann` — Schumann (1975): the plane-mean stress from the
    plane-mean speed, ``τ_{w,i} = ⟨τ_w⟩\\,u_i/⟨U⟩`` with
    ``⟨τ_w⟩ = τ_w(⟨U⟩)``.
  * `:moeng` — Moeng (1984):
    ``τ_{w,i} = ⟨τ_w⟩\\,[⟨U⟩(u_i - ⟨u_i⟩) + U⟨u_i⟩]/(⟨U⟩\\,|⟨\\mathbf u⟩|)``,
    whose plane mean is ``⟨τ_w⟩`` along the mean wind. With no mean
    wind (``|⟨\\mathbf u⟩| = 0``) it is undefined; the step then uses
    `:local`.

Plane means are over the columns (or faces) with a wall.

**Time stepping.** Each stress is linear in the face velocity,
``τ_{w,i} = a\\,u_i + b`` (``b ≠ 0`` only for `:moeng`), with ``a`` and
``b`` from the velocity at the start of the step, and is applied
implicitly:

```math
u_i ← \\frac{u_i - Δt\\,b/Δz}{1 + Δt\\,a/Δz},
```

so the wall stress is evaluated with the end-of-step velocity: for
`:local` and `:schumann` the velocity decays monotonically and never
changes sign, for any Δt (the explicit form is unstable beyond
``Δt = 2Δz/a``). The momentum the step removes is exactly
``\\sum τ_{w,i}\\,ΔA\\,Δt``.

Halos and the finite-volume operators are not modified: on a free-slip
(`ImpermeableWall`) z-wall, whose eddy-viscosity flux is zero, the wall
model is the only momentum flux through the wall face.

**Keywords.** `wind` — an `(Nx, Ny)` matrix of speeds to evaluate the
law at instead of the resolved ``U`` (a held or filtered wind, or the
speed at a matching height off the first cell: see
[`MatchingInput`](@ref)). The stress then has the magnitude of the law at
that speed and the direction of the resolved first-cell velocity,
``τ_{w,i} = τ_w(U_{in})\\,u_i/U`` (`:local`; `:schumann`:
``τ_w(⟨U_{in}⟩)\\,u_i/⟨U⟩``), so the implicit coefficient is
``a = τ_w(U_{in})/U``. (Before P57-b it was ``τ_w(U_{in})/U_{in}``,
which scaled the magnitude by ``U/U_{in}``.) `:moeng` uses the input
speeds for ``U`` and ``⟨U⟩``. `height` — the distance from the wall
(m) at which the law is evaluated, a number or an `(Nx, Ny)` matrix
(default ``Δz/2``, the first cell centre; the matching height of
[`MatchingInput`](@ref)); the stress is still applied at the wall face,
to the first velocity level (Kawai & Larsson 2012).
`ustar` / `wind_out` — `(Nx, Ny)` matrices that receive, for every
column with a wall, the friction velocity of the law and the speed it
was evaluated at (one side only).

A [`TBLEWallModel`](@ref) (P57-c) is applied by the same face loop
with the stress of its columns (`apply_wall_stress!(sim, wm::TBLEWallModel,
Δt; side, mask, input, ustar, wind_out)`; `input` a
[`MatchingInput`](@ref), `formulation = :local` only).

Returns the volume-integrated kinematic force of the step,
``\\sum (u' - u)/Δt\\,ΔV`` (m⁴ s⁻²) per component, negative for a flow
in the positive direction; divided by the wall area it is the mean
kinematic stress.

# References

  * Schumann, U. (1975). *J. Comput. Phys.*, 18, 376-404.
  * Grötzbach, G. (1987). In *Encyclopedia of Fluid Mechanics*, vol. 6,
    1337-1391.
  * Moeng, C.-H. (1984). *J. Atmos. Sci.*, 41, 2052-2062.
  * Piomelli, U. & Balaras, E. (2002). *Annu. Rev. Fluid Mech.*, 34,
    349-374.
  * Kawai, S. & Larsson, J. (2012). Wall-modeling in large eddy
    simulation: length scales, grid resolution, and accuracy.
    *Phys. Fluids*, 24, 015105.
"""
function apply_wall_stress!(sim, wm::WallModel, Δt::Real; side::Symbol = :bottom,
                            mask = nothing, formulation::Symbol = :local,
                            wind = nothing, height = nothing,
                            ustar = nothing, wind_out = nothing)
    side in (:bottom, :top, :both) ||
        throw(ArgumentError("apply_wall_stress!: side must be :bottom, :top or :both, got :$side"))
    formulation in (:local, :schumann, :moeng) ||
        throw(ArgumentError("apply_wall_stress!: formulation must be :local, :schumann or :moeng, got :$formulation"))
    mask isa AbstractMatrix{<:Integer} && !(mask isa AbstractMatrix{Bool}) && side !== :bottom &&
        throw(ArgumentError("apply_wall_stress!: a level mask is for side = :bottom only"))
    side === :both && (ustar !== nothing || wind_out !== nothing) &&
        throw(ArgumentError("apply_wall_stress!: ustar / wind_out take one side, not :both"))
    mesh = sim.mesh
    fill_halos!(sim.u, sim.bcs)
    fill_halos!(sim.v, sim.bcs)
    Fx = Fy = zero(float(mesh.dz))
    for (s, kw) in ((:bottom, 1), (:top, mesh.Nz))
        (side === s || side === :both) || continue
        fx, fy = _wall_side!(sim, wm, Δt, kw, mask, formulation, wind, height, ustar, wind_out)
        Fx += fx; Fy += fy
    end
    return (Fx = Fx, Fy = Fy)
end
apply_wall_stress!(sim, law::AbstractWallLaw, Δt::Real; kwargs...) =
    apply_wall_stress!(sim, WallModel(law), Δt; kwargs...)

# Implicit coefficient a = τ_w(U_in)/U_res for a law evaluated at an input
# speed other than the resolved one (the stress magnitude is the law's).
@inline _input_drag(L, Uin, Ures, h, ν) =
    Ures > 0 ? wall_stress(L, Uin, h; ν = ν) / Ures : wall_drag(L, Uin, h; ν = ν)

@inline _height(::Nothing, i, j, h0) = h0
@inline _height(h::Real, i, j, h0) = oftype(h0, h)
@inline _height(h::AbstractMatrix, i, j, h0) = oftype(h0, @inbounds h[i, j])

function _wall_side!(sim, wm::WallModel, Δt, kw, mask, formulation, wind, height, ustar, wind_out)
    mesh = sim.mesh
    Nx, Ny, Nz = mesh.Nx, mesh.Ny, mesh.Nz
    Hx, Hy, Hz, dz = mesh.Hx, mesh.Hy, mesh.Hz, mesh.dz
    T = typeof(float(dz))
    h0 = dz / 2
    ν = wm.ν
    u, v = sim.u.data, sim.v.data
    xper = sim.bcs.west isa Periodic
    yper = sim.bcs.south isa Periodic
    lev(i, j) = _level(mask, i, j, kw, Nz)

    # 1. column speeds at the wall level, from the start-of-step velocity
    Uc = zeros(T, Nx, Ny)                       # resolved speed
    Ue = zeros(T, Nx, Ny)                       # speed the law is evaluated at
    nact = 0
    @inbounds for j in 1:Ny, i in 1:Nx
        k = lev(i, j)
        k == 0 && continue
        I, J, K = Hx + i, Hy + j, Hz + k
        uc = (u[I, J, K] + u[I+1, J, K]) / 2
        vc = (v[I, J, K] + v[I, J+1, K]) / 2
        Uc[i, j] = sqrt(uc^2 + vc^2)
        Ue[i, j] = wind === nothing ? Uc[i, j] : T(wind[i, j])
        nact += 1
    end
    nact == 0 && return (zero(T), zero(T))

    # neighbour column across the west / south face, or 0 outside a
    # non-periodic domain
    west(i) = i > 1 ? i - 1 : (xper ? Nx : 0)
    south(j) = j > 1 ? j - 1 : (yper ? Ny : 0)
    levw(i, j) = (iw = west(i); iw == 0 ? 0 : lev(iw, j))
    levs(i, j) = (js = south(j); js == 0 ? 0 : lev(i, js))

    # 2. the plane means a non-local formulation needs
    form = formulation
    Ubar = zero(T); ubar = zero(T); vbar = zero(T); M = zero(T); Ures = zero(T)
    if form !== :local
        Ubar = sum(Ue) / nact
        Ures = sum(Uc) / nact
        if form === :moeng
            # face means, each face weighted by the share of it that has a wall
            su = sv = wu = wv = zero(T)
            @inbounds for j in 1:Ny, i in 1:Nx
                for (k, δ) in _face_levels(lev(i, j), levw(i, j))
                    k == 0 && continue
                    su += δ * u[Hx+i, Hy+j, Hz+k]; wu += δ
                end
                for (k, δ) in _face_levels(lev(i, j), levs(i, j))
                    k == 0 && continue
                    sv += δ * v[Hx+i, Hy+j, Hz+k]; wv += δ
                end
            end
            ubar = su / wu; vbar = sv / wv
            M = sqrt(ubar^2 + vbar^2)
            (M > 0 && Ubar > 0) || (form = :local)
        end
    end

    # 3. per column: τ = a u_i + b_i (b only for Moeng)
    a = zeros(T, Nx, Ny); bx = zeros(T, Nx, Ny); by = zeros(T, Nx, Ny)
    @inbounds for j in 1:Ny, i in 1:Nx
        lev(i, j) == 0 && continue
        L = _law(wm.law, i, j)
        h = _height(height, i, j, h0)
        if form === :local
            a[i, j] = wind === nothing ? wall_drag(L, Ue[i, j], h; ν = ν) :
                                         _input_drag(L, Ue[i, j], Uc[i, j], h, ν)
            Uu = Ue[i, j]
        elseif form === :schumann
            a[i, j] = wind === nothing ? wall_drag(L, Ubar, h; ν = ν) :
                                         _input_drag(L, Ubar, Ures, h, ν)
            Uu = Ubar
        else                                    # :moeng
            τm = wall_stress(L, Ubar, h; ν = ν)
            a[i, j] = τm / M
            c = τm * (Ue[i, j] - Ubar) / (Ubar * M)
            bx[i, j] = c * ubar; by[i, j] = c * vbar
            Uu = Ubar
        end
        ustar === nothing || (ustar[i, j] = friction_velocity(L, Uu, h; ν = ν))
        wind_out === nothing || (wind_out[i, j] = Uu)
    end

    # 4. apply, face by face, with the mean of the two columns' coefficients
    return _apply_coefficients!(sim, a, bx, by, Δt, kw, mask)
end

# Apply the per-column stress τ_i = a u_i + b_i (kinematic, removed from
# the flow) implicitly to the first velocity level of every walled
# column, face by face, with the mean of the two columns' coefficients
# (a column contributes only at its own wall level). Shared by every
# AbstractWallModel. Returns the volume-integrated force (Fx, Fy).
function _apply_coefficients!(sim, a, bx, by, Δt, kw, mask)
    mesh = sim.mesh
    Nx, Ny, Nz = mesh.Nx, mesh.Ny, mesh.Nz
    Hx, Hy, Hz, dz = mesh.Hx, mesh.Hy, mesh.Hz, mesh.dz
    T = typeof(float(dz))
    u, v = sim.u.data, sim.v.data
    xper = sim.bcs.west isa Periodic
    yper = sim.bcs.south isa Periodic
    lev(i, j) = _level(mask, i, j, kw, Nz)
    west(i) = i > 1 ? i - 1 : (xper ? Nx : 0)
    south(j) = j > 1 ? j - 1 : (yper ? Ny : 0)
    levw(i, j) = (iw = west(i); iw == 0 ? 0 : lev(iw, j))
    levs(i, j) = (js = south(j); js == 0 ? 0 : lev(i, js))
    Fx = Fy = zero(T)
    @inbounds for j in 1:Ny, i in 1:Nx
        iw, js = west(i), south(j)
        for (k, _) in _face_levels(lev(i, j), levw(i, j))
            k == 0 && continue
            A = zero(T); B = zero(T)
            lev(i, j) == k && (A += a[i, j]; B += bx[i, j])
            iw != 0 && levw(i, j) == k && (A += a[iw, j]; B += bx[iw, j])
            idx = CartesianIndex(Hx + i, Hy + j, Hz + k)
            u0 = u[idx]
            u[idx] = (u0 - Δt * (B / 2) / dz) / (1 + Δt * (A / 2) / dz)
            Fx += u[idx] - u0
        end
        for (k, _) in _face_levels(lev(i, j), levs(i, j))
            k == 0 && continue
            A = zero(T); B = zero(T)
            lev(i, j) == k && (A += a[i, j]; B += by[i, j])
            js != 0 && levs(i, j) == k && (A += a[i, js]; B += by[i, js])
            idx = CartesianIndex(Hx + i, Hy + j, Hz + k)
            v0 = v[idx]
            v[idx] = (v0 - Δt * (B / 2) / dz) / (1 + Δt * (A / 2) / dz)
            Fy += v[idx] - v0
        end
    end
    dV = mesh.dx * mesh.dy * dz
    return (Fx * dV / Δt, Fy * dV / Δt)
end

# The wall levels of a face between a column at level `k1` and its
# neighbour at `k2` (0 = no wall), each with the share of the face it
# carries (1/2 per column; 1 when both are at the same level). Always two
# entries; an unused one has level 0 (skipped by the callers).
@inline function _face_levels(k1::Int, k2::Int)
    k1 == k2 && return ((k1, 1.0), (0, 0.0))
    return ((k1, k1 == 0 ? 0.0 : 0.5), (k2, k2 == 0 ? 0.0 : 0.5))
end

# ---------------------------------------------------------------------------
# the wall stress as a post term of the fused sweep (#533 step 6)
# ---------------------------------------------------------------------------

"""
    WallStressTerm

[`apply_wall_stress!`](@ref) with an equilibrium [`WallModel`](@ref) on the
bottom walls, `formulation = :local`, as a post term of the fused sweep
(`Operators.FiniteVolume.PostTerm`, #533 step 6), built by
[`wall_stress_term`](@ref). Phase 1 of a row computes the coefficient
``a`` of every column whose wall level is the row's (the resolved speed
reads the column's four faces, footprint 1 in x and y), with `ustar` and
`wind_out`; phase 2 updates the u and v faces of the row from the mean of
the two columns' coefficients and keeps each face's change, which
`finish!` folds into the force, in the order of the whole-field operator
(`F`, the `(Fx, Fy)` it returns). The non-local formulations need plane
means of the predicted velocity, so they have no row form.
"""
struct WallStressTerm{W<:WallModel,M,WI,H,US,WO,T<:AbstractFloat} <: FiniteVolume.PostTerm
    wm::W
    mask::M
    wind::WI
    height::H
    ustar::US
    wind_out::WO
    a::Matrix{T}
    dqu::Array{T,3}
    dqv::Array{T,3}
    Δt::T
    dz::T
    dV::T
    Hx::Int
    Hy::Int
    Hz::Int
    Nx::Int
    Ny::Int
    Nz::Int
    kmax::Int
    xper::Bool
    yper::Bool
    F::Base.RefValue{NTuple{2,T}}
end

"""
    wall_stress_term(sim, wm::WallModel, Δt; side = :bottom, mask = nothing,
                     formulation = :local, wind = nothing, height = nothing,
                     ustar = nothing, wind_out = nothing) -> Union{WallStressTerm,Nothing}

The post term of the fused sweep that does what
`apply_wall_stress!(sim, wm, Δt; …)` does with the same keywords, or
`nothing` for what it has no row form of: a top wall (`side ≠ :bottom`)
or a non-local formulation (`:schumann`, `:moeng`: plane means of the
predicted velocity).
"""
function wall_stress_term(sim, wm::WallModel, Δt::Real; side::Symbol = :bottom, mask = nothing,
                          formulation::Symbol = :local, wind = nothing, height = nothing,
                          ustar = nothing, wind_out = nothing)
    (side === :bottom && formulation === :local) || return nothing
    m = sim.mesh
    T = typeof(float(m.dz))
    Nx, Ny, Nz = m.Nx, m.Ny, m.Nz
    kmax = mask === nothing ? 1 : maximum(((i, j),) -> _level(mask, i, j, 1, Nz), Iterators.product(1:Nx, 1:Ny); init = 0)
    return WallStressTerm(wm, mask, wind, height, ustar, wind_out, zeros(T, Nx, Ny), zeros(T, Nx, Ny, 2),
                          zeros(T, Nx, Ny, 2), T(Δt), T(m.dz), T(m.dx * m.dy * m.dz), m.Hx, m.Hy, m.Hz, Nx, Ny, Nz,
                          kmax, sim.bcs.west isa Periodic, sim.bcs.south isa Periodic, Ref((zero(T), zero(T))))
end
FiniteVolume.footprint(::WallStressTerm) = (1, 1, 0)

@inline _lev(t::WallStressTerm, i, j) = _level(t.mask, i, j, 1, t.Nz)
@inline _west(t::WallStressTerm, i) = i > 1 ? i - 1 : (t.xper ? t.Nx : 0)
@inline _south(t::WallStressTerm, j) = j > 1 ? j - 1 : (t.yper ? t.Ny : 0)
@inline _wout!(::Nothing, i, j, x) = nothing
@inline _wout!(o::AbstractMatrix, i, j, x) = (@inbounds o[i, j] = x; nothing)

# phase 1: the coefficient of every column of row j whose wall is at level k
function FiniteVolume.post_row!(t::WallStressTerm{W,M,WI,H,US,WO,T}, ::Val{1}, J, K, I1, I2, st) where {W,M,WI,H,US,WO,T}
    k, j = K - t.Hz, J - t.Hy
    1 <= k <= t.kmax || return nothing
    u, v, ν, h0 = st.u, st.v, t.wm.ν, t.dz / 2
    @inbounds for I in I1:I2
        i = I - t.Hx
        _lev(t, i, j) == k || continue
        uc = (u[I, J, K] + u[I+1, J, K]) / 2
        vc = (v[I, J, K] + v[I, J+1, K]) / 2
        Uc = sqrt(uc^2 + vc^2)
        Ue = t.wind === nothing ? Uc : T(t.wind[i, j])
        L = _law(t.wm.law, i, j)
        h = _height(t.height, i, j, h0)
        t.a[i, j] = t.wind === nothing ? wall_drag(L, Ue, h; ν = ν) : _input_drag(L, Ue, Uc, h, ν)
        _wout!(t.ustar, i, j, friction_velocity(L, Ue, h; ν = ν))
        _wout!(t.wind_out, i, j, Ue)
    end
    return nothing
end

# phase 2: the u and v faces of row j at level k, from the mean of the two
# columns' coefficients (a column contributes only at its own wall level)
function FiniteVolume.post_row!(t::WallStressTerm{W,M,WI,H,US,WO,T}, ::Val{2}, J, K, I1, I2, st) where {W,M,WI,H,US,WO,T}
    k, j = K - t.Hz, J - t.Hy
    1 <= k <= t.kmax || return nothing
    u, v, a, Δt, dz = st.u, st.v, t.a, t.Δt, t.dz
    js = _south(t, j)
    @inbounds for I in I1:I2
        i = I - t.Hx
        iw = _west(t, i)
        l = _lev(t, i, j)
        lw = iw == 0 ? 0 : _lev(t, iw, j)
        ls = js == 0 ? 0 : _lev(t, i, js)
        fl = _face_levels(l, lw)
        for s in 1:2
            fl[s][1] == k || continue
            A = zero(T)
            l == k && (A += a[i, j])
            iw != 0 && lw == k && (A += a[iw, j])
            u0 = u[I, J, K]
            u[I, J, K] = (u0 - Δt * (zero(T) / 2) / dz) / (1 + Δt * (A / 2) / dz)
            t.dqu[i, j, s] = u[I, J, K] - u0
        end
        fl = _face_levels(l, ls)
        for s in 1:2
            fl[s][1] == k || continue
            A = zero(T)
            l == k && (A += a[i, j])
            js != 0 && ls == k && (A += a[i, js])
            v0 = v[I, J, K]
            v[I, J, K] = (v0 - Δt * (zero(T) / 2) / dz) / (1 + Δt * (A / 2) / dz)
            t.dqv[i, j, s] = v[I, J, K] - v0
        end
    end
    return nothing
end

# the force, folded over the faces in the order of `_apply_coefficients!`
function FiniteVolume.finish!(t::WallStressTerm{W,M,WI,H,US,WO,T}, sim, Δt) where {W,M,WI,H,US,WO,T}
    Fx = Fy = zero(T)
    @inbounds for j in 1:t.Ny, i in 1:t.Nx
        iw, js = _west(t, i), _south(t, j)
        l = _lev(t, i, j)
        fl = _face_levels(l, iw == 0 ? 0 : _lev(t, iw, j))
        for s in 1:2
            fl[s][1] == 0 || (Fx += t.dqu[i, j, s])
        end
        fl = _face_levels(l, js == 0 ? 0 : _lev(t, i, js))
        for s in 1:2
            fl[s][1] == 0 || (Fy += t.dqv[i, j, s])
        end
    end
    t.F[] = (zero(T) + Fx * t.dV / t.Δt, zero(T) + Fy * t.dV / t.Δt)
    return nothing
end
