# The wall model on the walls of an immersed-boundary (IBM) solid mask:
# every fluid velocity face with a solid tangential neighbour gets the
# modelled stress of a wall law, and the stencil's own flux through that
# wall is removed, so the two are never counted twice.

export ImmersedWalls, immersed_walls, remove_stencil_wall_flux!,
       apply_immersed_wall_stress!, immersed_wall_count

"""
    ImmersedWallFace

One (face, wall) pair of [`ImmersedWalls`](@ref): the velocity component
`c` (1 = u, 2 = v, 3 = w) at the data index `(I, J, K)`, the wall-normal
axis `d` (1 = x, 2 = y), the side `s` (−1, +1) of the solid, and the wall
law of that wall. A face between two walls (a corner, or a gap one cell
wide) has two entries.
"""
struct ImmersedWallFace{L<:AbstractWallLaw}
    c::Int8
    I::Int32
    J::Int32
    K::Int32
    d::Int8
    s::Int8
    law::L
end

"""
    ImmersedWalls

The walls of an immersed solid mask as a wall model sees them; built by
[`immersed_walls`](@ref). Fields: `faces` (a vector of
[`ImmersedWallFace`](@ref)), `ν` (the kinematic viscosity the laws are
evaluated with, m² s⁻¹), the grid spacings, and work arrays. The last
call's diagnostics are kept in `stats`:

  * `F` — the volume-integrated kinematic force of the modelled stress
    (m⁴ s⁻², per component), [`apply_immersed_wall_stress!`](@ref);
  * `τA` — ``\\sum τ_w\\,ΔA`` over the walls (m⁴ s⁻²), `area` — the wall
    area ``\\sum ΔA`` (m²), `τ_min` — the smallest modelled ``τ_w``;
  * `F_stencil` — the force the stencil exerted through the same walls,
    removed by [`remove_stencil_wall_flux!`](@ref).
"""
struct ImmersedWalls{L<:AbstractWallLaw, T<:AbstractFloat}
    faces::Vector{ImmersedWallFace{L}}
    ν::T
    Δ::NTuple{3,T}
    work::NTuple{3,Array{T,3}}
    stats::Dict{Symbol,Any}
end

"""
    immersed_wall_count(iw) -> (u, v, w)

The number of (face, wall) pairs of each velocity component.
"""
immersed_wall_count(iw::ImmersedWalls) =
    ntuple(c -> count(f -> f.c == c, iw.faces), 3)

Base.show(io::IO, iw::ImmersedWalls) =
    print(io, "ImmersedWalls(", length(iw.faces), " face-walls ",
          immersed_wall_count(iw), ", ν = ", iw.ν, ")")

"""
    immersed_walls(sim, sm, law; ν = 1.5e-5) -> ImmersedWalls

The tangential walls of the solid mask `sm` (an `IBM.SolidMask` or its
`mask` array, interior + halos) on the staggered velocity of `sim`:

  * a **fluid face** is one with no solid cell on either side (the
    direct-forcing IBM zeroes every face of a solid cell, see
    `IBM.apply_solid_mask!`);
  * its tangential neighbour in direction ``±e_d`` is a **wall** when
    that neighbour face touches a solid cell, for
      - u faces across ±y,
      - v faces across ±x,
      - w faces across ±x and ±y.

The ±z walls of u and v (the ground and the roofs) are the z-walls of
[`apply_wall_stress!`](@ref), not repeated here. Across a periodic
boundary of `sim.bcs` the mask is read periodically; a non-periodic
domain boundary is not an immersed wall.

`law` is one [`AbstractWallLaw`](@ref) for every wall, or a function
`(i, j, k) -> AbstractWallLaw` of the interior index of the solid cell
behind the wall (a roughness per building, per facade material). `ν`
is the viscosity the laws are evaluated with (smooth-wall laws only).

**Wall distance.** The solid cell's face is the wall, as for the z-walls
of [`apply_wall_stress!`](@ref) (roofs included): the face centre of the
fluid face is ``h = Δ_d/2`` from it. The stencil the model replaces
treats the zeroed neighbour face, ``Δ_d`` away, as a no-slip wall
(its stress is ``ν_e u/Δ_d``); [`remove_stencil_wall_flux!`](@ref)
takes that flux out.
"""
function immersed_walls(sim, sm, law; ν::Real = 1.5e-5)
    mesh = sim.mesh
    mask = sm isa AbstractArray ? sm : sm.mask
    Nx, Ny, Nz = mesh.Nx, mesh.Ny, mesh.Nz
    Hx, Hy, Hz = mesh.Hx, mesh.Hy, mesh.Hz
    T = typeof(float(mesh.dx))
    per = (sim.bcs.west isa Periodic, sim.bcs.south isa Periodic, sim.bcs.bottom isa Periodic)
    N = (Nx, Ny, Nz); H = (Hx, Hy, Hz)
    # interior index of data index D along axis a, or 0 outside the domain
    # (the one BC-aware neighbour helper, N-open-d #262)
    inner(D, a) = neighbor_index(mesh, sim.bcs, D - H[a], a; outside = :none)
    function solid(I, J, K)
        i, j, k = inner(I, 1), inner(J, 2), inner(K, 3)
        (i == 0 || j == 0 || k == 0) && return (false, (0, 0, 0))
        return (mask[Hx+i, Hy+j, Hz+k], (i, j, k))
    end
    e = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
    # the face of component c at data index F touches cells F − e_c and F;
    # returns (touches a solid, interior index of a solid cell it touches)
    function face_solid(c, F)
        G = F .- e[c]
        s1, i1 = solid(G...)
        s1 && return (true, i1)
        s2, i2 = solid(F...)
        return (s2, i2)
    end
    lawof = law isa AbstractWallLaw ? (i, j, k) -> law : law
    entries = Tuple{Int,NTuple{3,Int},Int,Int,Any}[]
    dirs = ((2,), (1,), (1, 2))                     # wall-normal axes of u, v, w
    for c in 1:3
        for K in Hz+1:Hz+Nz, J in Hy+1:Hy+Ny, I in Hx+1:Hx+Nx
            F = (I, J, K)
            # the face must lie in the domain: a w face at K = Hz+1 (the
            # floor) or a u/v face on a non-periodic west/south boundary is
            # a boundary face, not a fluid face of the interior
            (per[c] || F[c] > H[c] + 1) || continue
            first(face_solid(c, F)) && continue
            for d in dirs[c], s in (-1, 1)
                Nb = F .+ s .* e[d]
                inner(Nb[d], d) == 0 && continue
                ws, ijk = face_solid(c, Nb)
                ws || continue
                push!(entries, (c, F, d, s, lawof(ijk...)))
            end
        end
    end
    laws = [x[5] for x in entries]
    L = isempty(laws) ? (law isa AbstractWallLaw ? typeof(law) : AbstractWallLaw) :
        mapreduce(typeof, typejoin, laws)
    faces = ImmersedWallFace{L}[ImmersedWallFace{L}(Int8(c), Int32(F[1]), Int32(F[2]), Int32(F[3]),
                                                    Int8(d), Int8(s), l)
                                for (c, F, d, s, l) in entries]
    dims = size(sim.u.data)
    work = (zeros(T, dims), zeros(T, size(sim.v.data)), zeros(T, size(sim.w.data)))
    stats = Dict{Symbol,Any}(:F => (zero(T), zero(T), zero(T)), :τA => zero(T),
                             :area => zero(T), :τ_min => zero(T),
                             :F_stencil => (zero(T), zero(T), zero(T)))
    return ImmersedWalls{L,T}(faces, T(ν), (T(mesh.dx), T(mesh.dy), T(mesh.dz)), work, stats)
end

# The speed parallel to the wall of face `f`: the face's own component and
# the other tangential component interpolated to the face (the four
# nearest faces of that component).
@inline function _tangential_speed(f::ImmersedWallFace, u, v, w)
    I, J, K = Int(f.I), Int(f.J), Int(f.K)
    if f.c == 1                                   # u, wall normal y: (u, w̄)
        q = u[I, J, K]
        o = (w[I-1, J, K] + w[I, J, K] + w[I-1, J, K+1] + w[I, J, K+1]) / 4
    elseif f.c == 2                               # v, wall normal x: (v, w̄)
        q = v[I, J, K]
        o = (w[I, J-1, K] + w[I, J, K] + w[I, J-1, K+1] + w[I, J, K+1]) / 4
    elseif f.d == 1                               # w, wall normal x: (w, v̄)
        q = w[I, J, K]
        o = (v[I, J, K-1] + v[I, J+1, K-1] + v[I, J, K] + v[I, J+1, K]) / 4
    else                                          # w, wall normal y: (w, ū)
        q = w[I, J, K]
        o = (u[I, J, K-1] + u[I+1, J, K-1] + u[I, J, K] + u[I+1, J, K]) / 4
    end
    return sqrt(q^2 + o^2)
end

"""
    apply_immersed_wall_stress!(sim, iw::ImmersedWalls, Δt) -> (Fx, Fy, Fz)

Apply the modelled stress of every wall of `iw` to the velocity of
`sim`, implicitly, over the step `Δt`. A `post_predictor!` hook, run
**after** the direct forcing of the IBM (so the solid faces are already
zero) and before the projection:

```julia
iw = immersed_walls(sim, sm, RoughLogLaw(0.05))
step!(sim, dt;
      body_force!     = (s, Δt) -> remove_stencil_wall_flux!(s, iw),
      post_predictor! = (s, Δt) -> (implicit_ibm_correction!(s, sm; dt = Δt);
                                    apply_immersed_wall_stress!(s, iw, Δt)))
```

At each (face, wall) pair the law is evaluated at the speed parallel to
that wall, ``U = |(q, \\bar q_o)|`` (the face's component `q` and the
other tangential component interpolated to the face), at
``h = Δ_d/2``, giving the drag coefficient ``a = τ_w(U)/U``
([`wall_drag`](@ref)). The stress on the face's component is
``τ = a\\,q`` (along the wall, against the flow; Piomelli & Balaras
2002), applied implicitly through the face's control volume,
``Δq = -Δt\\,τ/Δ_d``:

```math
q ← \\frac{q}{1 + Δt \\sum_{walls} a/Δ_d},
```

so a face between two walls (a corner, a one-cell gap) gets the sum of
both, the stress is evaluated with the end-of-step velocity, the face
velocity never changes sign and the update is stable for any `Δt`. The
momentum removed is exactly ``\\sum τ_w ΔA Δt`` (``ΔA = ΔV/Δ_d``).

Returns the volume-integrated kinematic force ``\\sum Δq\\,ΔV/Δt`` per
component (m⁴ s⁻²); `iw.stats` also holds ``\\sum τ_w ΔA``, the wall
area and the smallest ``τ_w`` of the step.

This is the IBM wall-stress model of Tessicini et al. (2002) and Chester,
Meneveau & Parlange (2007) in its simplest, equilibrium form: the stress
of a wall law at the first fluid point replaces the stencil's unresolved
no-slip flux (which [`remove_stencil_wall_flux!`](@ref) removes).

# References

  * Tessicini, F., Iaccarino, G., Fatica, M., Wang, M. & Verzicco, R.
    (2002). Wall modeling for large-eddy simulation using an immersed
    boundary method. *Annual Research Briefs 2002*, Center for
    Turbulence Research, Stanford, 181-187.
  * Chester, S., Meneveau, C. & Parlange, M. B. (2007). Modeling
    turbulent flow over fractal trees with renormalized numerical
    simulation. *J. Comput. Phys.*, 225, 427-448.
  * Piomelli, U. & Balaras, E. (2002). *Annu. Rev. Fluid Mech.*, 34,
    349-374.
"""
function apply_immersed_wall_stress!(sim, iw::ImmersedWalls{L,T}, Δt::Real) where {L,T}
    fill_halos!(sim.u, sim.bcs); fill_halos!(sim.v, sim.bcs); fill_halos!(sim.w, sim.bcs)
    u, v, w = sim.u.data, sim.v.data, sim.w.data
    q = (u, v, w)
    A = iw.work
    Δ = iw.Δ
    ν = iw.ν
    dV = Δ[1] * Δ[2] * Δ[3]
    # 1. the drag coefficient of every (face, wall) pair, summed per face
    a = Vector{T}(undef, length(iw.faces))
    @inbounds for (n, f) in enumerate(iw.faces)
        U = _tangential_speed(f, u, v, w)
        a[n] = T(wall_drag(f.law, U, Δ[f.d] / 2; ν = ν)) / Δ[f.d]
        A[f.c][f.I, f.J, f.K] += a[n]
    end
    # 2. the implicit update, once per face (the first of its entries)
    F = zeros(T, 3)
    dt = T(Δt)
    @inbounds for f in iw.faces
        Af = A[f.c][f.I, f.J, f.K]
        Af == 0 && continue
        q0 = q[f.c][f.I, f.J, f.K]
        q1 = q0 / (1 + dt * Af)
        q[f.c][f.I, f.J, f.K] = q1
        F[f.c] += q1 - q0
        A[f.c][f.I, f.J, f.K] = 0
    end
    # 3. diagnostics: τ_w ΔA at the end-of-step speed along each wall
    τA = zero(T); area = zero(T); τmin = T(Inf)
    @inbounds for (n, f) in enumerate(iw.faces)
        dA = dV / Δ[f.d]
        τ = a[n] * Δ[f.d] * abs(q[f.c][f.I, f.J, f.K])
        τA += τ * dA; area += dA
        τmin = min(τmin, a[n])
    end
    Fo = (F[1] * dV / dt, F[2] * dV / dt, F[3] * dV / dt)
    iw.stats[:F] = Fo
    iw.stats[:τA] = τA
    iw.stats[:area] = area
    iw.stats[:τ_min] = isempty(iw.faces) ? zero(T) : τmin
    return Fo
end

"""
    remove_stencil_wall_flux!(sim, iw::ImmersedWalls; ν = sim.ν, νt = sim.νt,
                              sgs = sim.has_sgs) -> (Fx, Fy, Fz)

Remove from the momentum tendencies of `sim` the flux the viscous and
eddy-viscosity stencils put through the walls of `iw`, so that
[`apply_immersed_wall_stress!`](@ref) is the only momentum flux there.
A `body_force!` hook: it must run inside the step, after the tendencies
are formed (it reads the start-of-step velocity and the step's
``ν_t``) and before the predictor.

For a face `F` with a wall across ``s\\,e_d`` (neighbour face ``N``,
zeroed by the IBM), the stencils of `Simulations.step!` (the molecular
Laplacian and `Operators.add_eddy_viscosity_*!`) added

```math
\\frac{(ν + ν_{t,e})\\,(q_N - q_F)}{Δ_d^2},
```

``ν_{t,e}`` the mean of the four cell-centre ``ν_t`` around the edge
between `F` and `N`; this is subtracted. The correction of a step is
therefore *modelled stress minus stencil stress*: no double counting.
With `sgs = false` (no eddy viscosity in the step) only the molecular
part is removed; pass `sgs = true` when `step!` gets an
`eddy_viscosity!` closure (it fills `sim.νt` without `sim.has_sgs`).

Returns the volume-integrated kinematic force the stencil had exerted
through the walls (m⁴ s⁻²), also kept in `iw.stats[:F_stencil]`.

With the IMEX stepper the diffusion is implicit and its wall flux is
not this explicit expression; the removal is then first-order accurate
in time (exact for `step!`, the explicit projection step).
"""
function remove_stencil_wall_flux!(sim, iw::ImmersedWalls{L,T};
                                   ν::Real = (hasproperty(sim, :has_viscosity) && sim.has_viscosity) ? sim.ν : 0,
                                   νt = sim.νt,
                                   sgs::Bool = hasproperty(sim, :has_sgs) && sim.has_sgs) where {L,T}
    q = (sim.u.data, sim.v.data, sim.w.data)
    t = (sim.tu.data, sim.tv.data, sim.tw.data)
    nt = νt isa AbstractArray ? νt : νt.data
    Δ = iw.Δ
    dV = Δ[1] * Δ[2] * Δ[3]
    νT = T(ν)
    F = zeros(T, 3)
    @inbounds for f in iw.faces
        c, d, s = Int(f.c), Int(f.d), Int(f.s)
        I, J, K = Int(f.I), Int(f.J), Int(f.K)
        Ni, Nj, Nk = I + s * (d == 1), J + s * (d == 2), K
        ci, cj, ck = (c == 1), (c == 2), (c == 3)
        νe = νT
        if sgs
            νe += (nt[I, J, K] + nt[I-ci, J-cj, K-ck] +
                   nt[Ni, Nj, Nk] + nt[Ni-ci, Nj-cj, Nk-ck]) / 4
        end
        flux = νe * (q[c][Ni, Nj, Nk] - q[c][I, J, K]) / Δ[d]^2
        t[c][I, J, K] -= flux
        F[c] += flux
    end
    Fo = (F[1] * dV, F[2] * dV, F[3] * dV)
    iw.stats[:F_stencil] = Fo
    return Fo
end
