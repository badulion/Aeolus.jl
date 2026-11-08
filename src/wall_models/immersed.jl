# The wall model on the walls of an immersed-boundary (IBM) solid mask:
# every fluid velocity face with a solid tangential neighbour gets the
# modelled stress of a wall law, and the stencil's own flux through that
# wall is removed, so the two are never counted twice.

export ImmersedWalls, immersed_walls, remove_stencil_wall_flux!,
       apply_immersed_wall_stress!, immersed_wall_count, immersed_wall_drag!

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
[`ImmersedWallFace`](@ref), one per (face, wall) pair), `ν` (the
kinematic viscosity the laws are evaluated with, m² s⁻¹), the grid
spacings `Δ`, the sizes `dims` of the `u`, `v`, `w` data arrays, and the
**face index** (KA-k, #307): `face_ptr` and `face_entry` group the pairs
by velocity face (compressed sparse rows: the pairs of face `p` are
`face_entry[face_ptr[p]:face_ptr[p+1]-1]`, the faces in the order of
their first pair, the pairs of a face in their order in `faces`), with
the component of every pair and face (`pair_c`, `face_c`) and the wall
area of every pair (`pair_dA` = ``ΔV/Δ_d``) as compact vectors for the
ordered folds of the diagnostics, and the faces split into those with
one pair (`single_face`, with that pair in `single_pair`) and those with
several (`multi_face`), which the face update launches separately (a
flat loop over the single-pair faces, the majority); `ordered` is true
when the face index lists the pairs in their own order (as
[`immersed_walls`](@ref) builds them), and `pair_seg`/`face_seg` are the
index ranges of the u, v, w pairs/faces when they are sorted by
component. A face
between two walls (a corner, a gap one cell wide) has two pairs; the
operators sum them per face (owner computes), so no two work items
write the same face. `a`, `τdA`, `flux` (per pair) and `Af`, `dq` (per
face) are the work vectors of the last call. The last call's
diagnostics are kept in `stats`:

  * `F` — the volume-integrated kinematic force of the modelled stress
    (m⁴ s⁻², per component), [`apply_immersed_wall_stress!`](@ref);
  * `τA` — ``\\sum τ_w\\,ΔA`` over the walls (m⁴ s⁻²), `area` — the wall
    area ``\\sum ΔA`` (m²), `τ_min` — the smallest modelled ``τ_w``;
  * `F_stencil` — the force the stencil exerted through the same walls,
    removed by [`remove_stencil_wall_flux!`](@ref).

    ImmersedWalls(faces, ν, Δ, dims)

builds the face index and the work vectors for any vector of pairs.
"""
struct ImmersedWalls{L<:AbstractWallLaw, T<:AbstractFloat}
    faces::Vector{ImmersedWallFace{L}}
    ν::T
    Δ::NTuple{3,T}
    dims::NTuple{3,NTuple{3,Int}}
    face_ptr::Vector{Int32}
    face_entry::Vector{Int32}
    pair_c::Vector{Int8}
    pair_dA::Vector{T}
    face_c::Vector{Int8}
    single_face::Vector{Int32}
    single_pair::Vector{Int32}
    multi_face::Vector{Int32}
    pair_seg::Union{Nothing,NTuple{3,UnitRange{Int}}}
    face_seg::Union{Nothing,NTuple{3,UnitRange{Int}}}
    ordered::Bool
    a::Vector{T}
    τdA::Vector{T}
    flux::Vector{T}
    Af::Vector{T}
    dq::Vector{T}
    stats::Dict{Symbol,Any}
end

function ImmersedWalls(faces::Vector{ImmersedWallFace{L}}, ν::T, Δ::NTuple{3,T},
                       dims::NTuple{3,NTuple{3,Int}}) where {L, T<:AbstractFloat}
    ptr, ent = _face_index(faces)
    n, nf = length(faces), length(ptr) - 1
    dV = Δ[1] * Δ[2] * Δ[3]
    pair_c = Int8[f.c for f in faces]
    pair_dA = T[dV / Δ[f.d] for f in faces]
    face_c = Int8[faces[ent[ptr[p]]].c for p in 1:nf]
    single = [p for p in 1:nf if ptr[p+1] - ptr[p] == 1]
    multi = Int32[p for p in 1:nf if ptr[p+1] - ptr[p] > 1]
    stats = Dict{Symbol,Any}(:F => (zero(T), zero(T), zero(T)), :τA => zero(T),
                             :area => zero(T), :τ_min => zero(T),
                             :F_stencil => (zero(T), zero(T), zero(T)))
    return ImmersedWalls{L,T}(faces, ν, Δ, dims, ptr, ent, pair_c, pair_dA, face_c,
                              Int32.(single), Int32[ent[ptr[p]] for p in single], multi,
                              _segments(pair_c), _segments(face_c), _identity_entries(ent),
                              zeros(T, n), zeros(T, n), zeros(T, n),
                              zeros(T, nf), zeros(T, nf), stats)
end

# The index ranges of components 1, 2, 3 when `c` is sorted (the order
# immersed_walls builds), else nothing.
function _segments(c::Vector{Int8})
    issorted(c) || return nothing
    return ntuple(k -> searchsortedfirst(c, k):searchsortedlast(c, k), 3)
end

# Σ x[n] per component c[n], each a strict left fold in index order (the
# order of the former loops). With the component segments known the three
# folds are interleaved (three independent chains of additions; the same
# bits), else one loop picks the accumulator per element.
@inline function _fold3(x::AbstractVector{T}, c, seg, skip = nothing) where {T}
    F1 = F2 = F3 = zero(T)
    if seg === nothing
        @inbounds for n in eachindex(x)
            skip === nothing || skip[n] == 0 && continue
            cn = c[n]; xn = x[n]
            cn == 1 ? (F1 += xn) : cn == 2 ? (F2 += xn) : (F3 += xn)
        end
    else
        r1, r2, r3 = seg
        @inbounds for m in 0:max(length(r1), length(r2), length(r3))-1
            if m < length(r1)
                n = first(r1) + m
                (skip === nothing || skip[n] != 0) && (F1 += x[n])
            end
            if m < length(r2)
                n = first(r2) + m
                (skip === nothing || skip[n] != 0) && (F2 += x[n])
            end
            if m < length(r3)
                n = first(r3) + m
                (skip === nothing || skip[n] != 0) && (F3 += x[n])
            end
        end
    end
    return (F1, F2, F3)
end

# The pairs grouped by velocity face (c, I, J, K), built once: the faces in
# the order of their first pair, the pairs of a face in their order in
# `faces`. Summing a face's pairs in this order adds them in the order the
# former scatter `A[c][I,J,K] += a[n]` (a loop over `faces`) did, so the
# gather gives the same bits.
function _face_index(faces::AbstractVector{<:ImmersedWallFace})
    id = Dict{NTuple{4,Int32},Int32}()
    fid = Vector{Int32}(undef, length(faces))
    for (n, f) in enumerate(faces)
        fid[n] = get!(id, (Int32(f.c), f.I, f.J, f.K), Int32(length(id) + 1))
    end
    nf = length(id)
    ptr = ones(Int32, nf + 1)
    for p in fid
        ptr[p+1] += 1
    end
    for p in 1:nf
        ptr[p+1] += ptr[p] - 1
    end
    next = ptr[1:nf]
    ent = Vector{Int32}(undef, length(faces))
    for (n, p) in enumerate(fid)
        ent[next[p]] = n
        next[p] += 1
    end
    return ptr, ent
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
    dims = (size(sim.u.data), size(sim.v.data), size(sim.w.data))
    return ImmersedWalls(faces, T(ν), (T(mesh.dx), T(mesh.dy), T(mesh.dz)), dims)
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
    immersed_wall_drag!(a, iw::ImmersedWalls, u, v, w) -> a

The drag rate ``a_n = τ_w(U)/(U\\,Δ_d)`` (s⁻¹) of every (face, wall) pair
`n` of `iw`, for the velocity data arrays `u, v, w` (halos filled): the
law of the pair evaluated at the speed parallel to that wall,
``U = |(q, \\bar q_o)|``, at ``h = Δ_d/2`` ([`wall_drag`](@ref)), divided
by the face's width ``Δ_d`` across the wall. The modelled stress on the
face's component `q` is ``τ = a_n Δ_d\\, q`` and its kinematic force per
unit volume ``a_n q``, so ``a_n q^2`` is the rate per unit volume at which
the wall stress takes resolved kinetic energy out of that face.

The one evaluation of the immersed wall law:
[`apply_immersed_wall_stress!`](@ref) applies it, and the facade
production of the prognostic TKE closure (`PrognosticSGS.TKEClosure`
with `walls`, #232) books ``a_n q^2`` as shear production.
"""
function immersed_wall_drag!(a::AbstractVector, iw::ImmersedWalls{L,T}, u, v, w) where {L,T}
    length(a) == length(iw.faces) ||
        throw(DimensionMismatch("immersed_wall_drag!: need one entry per face-wall ($(length(iw.faces))), got $(length(a))"))
    launch!(backend(u), _iw_drag_body!, (1:length(iw.faces),), a, iw.faces, u, v, w, iw.Δ, iw.ν)
    return a
end

# the drag rate of pair n (pointwise over the pairs; reads u, v, w only)
@inline function _iw_drag_body!(n, a::AbstractVector{T}, faces, u, v, w, Δ, ν) where {T}
    @inbounds begin
        f = faces[n]
        U = _tangential_speed(f, u, v, w)
        a[n] = T(wall_drag(f.law, U, Δ[f.d] / 2; ν = ν)) / Δ[f.d]
    end
    return nothing
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

**Kernels** (KA-k, #307). Two launches over sparse lists: the drag of
every pair ([`immersed_wall_drag!`](@ref)), then one work item per
*face* (the face index of [`ImmersedWalls`](@ref)) that sums its pairs'
drag in pair order, updates its own velocity and records ``τ_w ΔA`` of
its pairs. No two work items write the same face, so the result is the
same at every thread count, and the same bits as the former serial
scatter. The diagnostics are folded on the calling task in pair order.

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
    Δ = iw.Δ
    dV = Δ[1] * Δ[2] * Δ[3]
    dt = T(Δt)
    faces, ptr, ent = iw.faces, iw.face_ptr, iw.face_entry
    # 1. the drag coefficient of every (face, wall) pair, from the velocity
    #    before any face is updated
    a = immersed_wall_drag!(iw.a, iw, u, v, w)
    # 2. per face (owner computes): the sum of its pairs' drag, the implicit
    #    update, and τ_w ΔA of each of its walls at the end-of-step velocity
    b = backend(u)
    sf, sp, mf = iw.single_face, iw.single_pair, iw.multi_face
    launch!(b, _iw_face1_body!, (1:length(sf),), sf, sp, u, v, w, faces, a, iw.Af, iw.dq, iw.τdA, dt, Δ, dV)
    launch!(b, _iw_faceN_body!, (1:length(mf),), mf, u, v, w, faces, ptr, ent, a, iw.Af, iw.dq, iw.τdA,
            dt, Δ, dV)
    # 3. the diagnostics, folded in a fixed order: the force over the faces
    #    (in the order of their first pair), τ_w ΔA, the area and the
    #    smallest drag over the pairs (in their order) — the order of the
    #    former loops over `faces`, so the same bits
    F1, F2, F3 = _fold3(iw.dq, iw.face_c, iw.face_seg, iw.Af)    # (a face with Σ a = 0 is skipped)
    τA = zero(T); area = zero(T); τmin = T(Inf)
    τdA, dA = iw.τdA, iw.pair_dA
    @inbounds for n in eachindex(τdA)
        τA += τdA[n]; area += dA[n]
        τmin = min(τmin, a[n])
    end
    Fo = (F1 * dV / dt, F2 * dV / dt, F3 * dV / dt)
    iw.stats[:F] = Fo
    iw.stats[:τA] = τA
    iw.stats[:area] = area
    iw.stats[:τ_min] = isempty(faces) ? zero(T) : τmin
    return Fo
end

# Face p of an ImmersedWalls: q ← q / (1 + Δt Σ a) over its pairs (summed in
# their order), Af[p] = Σ a, dq[p] = the change (0 when Σ a = 0, the face
# then untouched), τdA[n] = τ_w ΔA of each of its pairs. Writes only face
# p's velocity and its own entries; reads only face p's velocity, so the
# faces are independent (the drag of every pair was computed before).
# `_iw_face1_body!` is a face with one pair (the m-th of `single_face`),
# `_iw_faceN_body!` one with several (the m-th of `multi_face`).
@inline function _iw_face_update!(p, f, A, u, v, w, Af, dq, dt)
    @inbounds begin
        c, I, J, K = Int(f.c), Int(f.I), Int(f.J), Int(f.K)
        q = c == 1 ? u : c == 2 ? v : w
        Af[p] = A
        if A == 0
            dq[p] = zero(A)
        else
            q0 = q[I, J, K]
            q1 = q0 / (1 + dt * A)
            q[I, J, K] = q1
            dq[p] = q1 - q0
        end
        return abs(q[I, J, K])
    end
end

@inline function _iw_face1_body!(m, sf, sp, u, v, w, faces, a, Af, dq, τdA, dt, Δ, dV)
    @inbounds begin
        p, n = Int(sf[m]), Int(sp[m])
        f = faces[n]
        an = a[n]
        qe = _iw_face_update!(p, f, zero(an) + an, u, v, w, Af, dq, dt)
        Δd = Δ[f.d]
        τdA[n] = (an * Δd * qe) * (dV / Δd)
    end
    return nothing
end

@inline function _iw_faceN_body!(m, mf, u, v, w, faces, ptr, ent, a, Af, dq, τdA, dt, Δ, dV)
    @inbounds begin
        p = Int(mf[m])
        e1, e2 = ptr[p], ptr[p+1] - 1
        A = zero(eltype(a))
        for e in e1:e2
            A += a[ent[e]]
        end
        qe = _iw_face_update!(p, faces[ent[e1]], A, u, v, w, Af, dq, dt)
        for e in e1:e2
            n = ent[e]
            Δd = Δ[faces[n].d]
            τdA[n] = (a[n] * Δd * qe) * (dV / Δd)
        end
    end
    return nothing
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

One work item per face (KA-k, #307) subtracts its pairs' fluxes from
its own tendency in pair order (the former scatter's order: the same
bits at every thread count); the force is folded in pair order.

With the IMEX stepper the diffusion is implicit and its wall flux is
not this explicit expression; the removal is then first-order accurate
in time (exact for `step!`, the explicit projection step).
"""
function remove_stencil_wall_flux!(sim, iw::ImmersedWalls{L,T};
                                   ν::Real = (hasproperty(sim, :has_viscosity) && sim.has_viscosity) ? sim.ν : 0,
                                   νt = sim.νt,
                                   sgs::Bool = hasproperty(sim, :has_sgs) && sim.has_sgs) where {L,T}
    u, v, w = sim.u.data, sim.v.data, sim.w.data
    tu, tv, tw = sim.tu.data, sim.tv.data, sim.tw.data
    nt = νt isa AbstractArray ? νt : νt.data
    Δ = iw.Δ
    dV = Δ[1] * Δ[2] * Δ[3]
    νT = T(ν)
    faces, ptr, ent = iw.faces, iw.face_ptr, iw.face_entry
    # per face (owner computes): each pair's flux, taken out of the face's
    # tendency in the order of its pairs
    nf = length(ptr) - 1
    # (`nothing`: the pairs are already in face order, as immersed_walls builds them)
    entx = iw.ordered ? nothing : ent
    launch_weighted!(backend(u), _IW_BLOCK, _iw_flux_block!, (1:cld(nf, _IW_BLOCK),),
                     nf, tu, tv, tw, u, v, w, nt, faces, ptr, entx, iw.flux, νT, sgs, Δ)
    # the force, folded over the pairs in their order (the former loop's)
    F1, F2, F3 = _fold3(iw.flux, iw.pair_c, iw.pair_seg)
    Fo = (F1 * dV, F2 * dV, F3 * dV)
    iw.stats[:F_stencil] = Fo
    return Fo
end

# Pair n: flux[n] = (ν + ν_t,e)(q_N − q_F)/Δ_d², subtracted from its face's
# tendency; the pairs of a face are taken in pair order by one work item.
@inline function _iw_pair_flux!(n, tu, tv, tw, u, v, w, nt, faces, flux::AbstractVector{T},
                                νT, sgs, Δ) where {T}
    @inbounds begin
        f = faces[n]
        c = Int(f.c)
        t = c == 1 ? tu : c == 2 ? tv : tw
        x = _iw_pair_x(f, u, v, w, nt, νT, sgs, Δ)
        t[Int(f.I), Int(f.J), Int(f.K)] -= x
        flux[n] = x
    end
    return nothing
end
# the stencil's flux of pair `f` through its wall: the one expression of the
# removal, in the separate pass above and in the fused sweep (#533)
@inline function _iw_pair_x(f, u, v, w, nt, νT, sgs, Δ)
    @inbounds begin
        c, d, s = Int(f.c), Int(f.d), Int(f.s)
        I, J, K = Int(f.I), Int(f.J), Int(f.K)
        q = c == 1 ? u : c == 2 ? v : w
        Ni, Nj, Nk = I + s * (d == 1), J + s * (d == 2), K
        ci, cj, ck = (c == 1), (c == 2), (c == 3)
        νe = νT
        if sgs
            νe += (nt[I, J, K] + nt[I-ci, J-cj, K-ck] +
                   nt[Ni, Nj, Nk] + nt[Ni-ci, Nj-cj, Nk-ck]) / 4
        end
        return νe * (q[Ni, Nj, Nk] - q[I, J, K]) / Δ[d]^2
    end
end

# One work item = a block of `_IW_BLOCK` consecutive faces: their pairs in
# face-index order, one flat loop (a face's pairs are consecutive there, so
# a face is never split between work items, and the block size does not
# depend on the thread count). The flat loop runs as fast as the former loop
# over the pairs; a loop per face did not (1.35x at 1 thread, ka_port.md).
const _IW_BLOCK = 64

@inline function _iw_flux_block!(b, nf, tu, tv, tw, u, v, w, nt, faces, ptr, ent, flux, νT, sgs, Δ)
    @inbounds begin
        e1 = Int(ptr[(b-1)*_IW_BLOCK+1])
        e2 = Int(ptr[min(b * _IW_BLOCK, nf)+1]) - 1
        for e in e1:e2
            n = ent === nothing ? e : Int(ent[e])
            _iw_pair_flux!(n, tu, tv, tw, u, v, w, nt, faces, flux, νT, sgs, Δ)
        end
    end
    return nothing
end

# whether the face index lists the pairs in their own order (1, 2, …, n)
_identity_entries(ent) = (@inbounds for e in eachindex(ent); ent[e] == e || return false; end; true)

# ---------------------------------------------------------------------------
# the removal inside the fused sweep (#533 step 5)
# ---------------------------------------------------------------------------
# The fluxes depend only on the stage-start velocity and ν_t, which the
# sweep does not change before a row's predictor: they are computed for
# every pair before the sweep (`prepare!`, one work item per pair, and the
# force folded as above), then each row of the sweep subtracts the fluxes
# of its pairs from its tendency buffer, a face's pairs in pair order: the
# operations of `remove_stencil_wall_flux!` in the same order.

"""
    stencil_wall_fluxes!(sim, iw::ImmersedWalls; ν, νt, sgs) -> (Fx, Fy, Fz)

The fluxes [`remove_stencil_wall_flux!`](@ref) removes, per pair, into
`iw.flux`, and their force into `iw.stats[:F_stencil]`, without touching
the tendencies (the fused sweep subtracts them, [`StencilWallFluxTerm`](@ref)).
The keywords are those of `remove_stencil_wall_flux!`.
"""
function stencil_wall_fluxes!(sim, iw::ImmersedWalls{L,T};
                              ν::Real = (hasproperty(sim, :has_viscosity) && sim.has_viscosity) ? sim.ν : 0,
                              νt = sim.νt,
                              sgs::Bool = hasproperty(sim, :has_sgs) && sim.has_sgs) where {L,T}
    u, v, w = sim.u.data, sim.v.data, sim.w.data
    nt = νt isa AbstractArray ? νt : νt.data
    Δ = iw.Δ
    dV = Δ[1] * Δ[2] * Δ[3]
    n = length(iw.faces)
    n > 0 && launch!(backend(u), _iw_flux_only!, (1:n,), u, v, w, nt, iw.faces, iw.flux, T(ν), sgs, Δ)
    F1, F2, F3 = _fold3(iw.flux, iw.pair_c, iw.pair_seg)
    Fo = (F1 * dV, F2 * dV, F3 * dV)
    iw.stats[:F_stencil] = Fo
    return Fo
end
@inline function _iw_flux_only!(n, u, v, w, nt, faces, flux, νT, sgs, Δ)
    @inbounds flux[n] = _iw_pair_x(faces[n], u, v, w, nt, νT, sgs, Δ)
    return nothing
end

"""
    StencilWallFluxTerm

The fused-sweep term ([`FiniteVolume.SweepTerm`](@ref Aeolus.Operators.FiniteVolume.SweepTerm))
of [`remove_stencil_wall_flux!`](@ref): its `prepare!` computes every
pair's flux from the stage-start state ([`stencil_wall_fluxes!`](@ref)),
its row kernel subtracts the fluxes of the row's faces from their
tendencies (`rowptr`/`rowent`: the pairs of storage row `(J, K)`, row
`J + (K − 1) nJ`, in face order). Built by
[`stencil_wall_flux_term`](@ref); the wall faces are inputs, so the term
reads no state in its kernel (footprint 0).
"""
struct StencilWallFluxTerm{W<:ImmersedWalls,F,V,N} <: FiniteVolume.SweepTerm
    iw::W
    faces::F
    flux::V
    rowptr::Vector{Int32}
    rowent::Vector{Int32}
    nJ::Int
    i1::Int
    i2::Int
    ν::Float64
    νt::N
    sgs::Bool
end
FiniteVolume.footprint(::StencilWallFluxTerm) = (0, 0, 0)
FiniteVolume.sweep_inputs(t::StencilWallFluxTerm) = (t.flux,)
FiniteVolume.prepare!(t::StencilWallFluxTerm, sim, Δt) =
    (stencil_wall_fluxes!(sim, t.iw; ν = t.ν, νt = t.νt, sgs = t.sgs); nothing)
@noinline function FiniteVolume.add_row!(t::StencilWallFluxTerm, buf, a, b, s, j, k, ii, st)
    r = j + (k - 1) * t.nJ
    @inbounds e1, e2 = Int(t.rowptr[r]), Int(t.rowptr[r+1]) - 1
    e1 > e2 && return nothing
    _iw_row!(buf, a, b, s, e1, e2, t.rowent, t.faces, t.flux, t.i1, t.i2)
end
@noinline function _iw_row!(buf, a, b, s, e1, e2, rowent, faces, flux, i1, i2)
    @inbounds for e in e1:e2
        n = Int(rowent[e])
        f = faces[n]
        I = Int(f.I)
        i1 <= I <= i2 && (buf[I, a, b, Int(f.c), s] -= flux[n])
    end
    return nothing
end

"""
    wall_rows(iw::ImmersedWalls) -> (rowptr, rowent, nJ)

The pairs of `iw` grouped by storage row `(J, K)` (row `J + (K − 1) nJ`,
`nJ` the largest y-size of the u, v, w arrays): compressed sparse rows,
the pairs of a row in face order and a face's pairs in pair order (the
order of `remove_stencil_wall_flux!`'s subtraction).
"""
function wall_rows(iw::ImmersedWalls)
    nJ = maximum(d -> d[2], iw.dims); nK = maximum(d -> d[3], iw.dims)
    nr = nJ * nK
    order = iw.ordered ? collect(Int32, eachindex(iw.faces)) : Int32.(iw.face_entry)
    cnt = zeros(Int32, nr + 1)
    for n in order
        f = iw.faces[n]
        cnt[Int(f.J) + (Int(f.K) - 1) * nJ + 1] += 1
    end
    rowptr = Int32.(cumsum(cnt) .+ 1)        # (cumsum widens Int32)
    fillp = rowptr[1:nr]
    rowent = Vector{Int32}(undef, length(order))
    for n in order                         # stable: face order kept within a row
        f = iw.faces[n]
        r = Int(f.J) + (Int(f.K) - 1) * nJ
        rowent[fillp[r]] = n
        fillp[r] += 1
    end
    return rowptr, rowent, nJ
end

"""
    stencil_wall_flux_term(sim, iw::ImmersedWalls; rows = wall_rows(iw), ν, νt, sgs)
        -> StencilWallFluxTerm

The fused-sweep term of `remove_stencil_wall_flux!(sim, iw; ν, νt, sgs)`
(the same keywords and defaults). `rows` (from [`wall_rows`](@ref)) is
kept by the caller, so building the term each substage allocates nothing
per wall.
"""
function stencil_wall_flux_term(sim, iw::ImmersedWalls; rows = wall_rows(iw),
                                ν::Real = (hasproperty(sim, :has_viscosity) && sim.has_viscosity) ? sim.ν : 0,
                                νt = sim.νt,
                                sgs::Bool = hasproperty(sim, :has_sgs) && sim.has_sgs)
    ii = first(interior(sim.mesh))
    return StencilWallFluxTerm(iw, iw.faces, iw.flux, rows[1], rows[2], rows[3], first(ii), last(ii),
                               Float64(ν), νt, sgs)
end
