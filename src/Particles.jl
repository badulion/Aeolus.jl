module Particles

# Phase 20 (component) — Lagrangian particle tracker.
#
# A minimal implementation: each particle carries a position (x, y, z)
# and is advected by the Eulerian velocity field using linear
# interpolation. Extensions: inertial particles with drag, particle
# emissions from a source, back-reaction on the fluid.

using ..Grids: RegularGrid, interior
using ..Fields: Field
using ..Simulations: BoussinesqSimulation

export ParticleSet, add_particles!, advance_particles!, positions

"""
    ParticleSet(mesh)

A dynamic set of Lagrangian particle positions on `mesh`. Positions are
stored as `Vector{NTuple{3, Float64}}`.
"""
mutable struct ParticleSet{T<:AbstractFloat}
    mesh::RegularGrid{T}
    pos::Vector{NTuple{3, T}}
end

ParticleSet(mesh::RegularGrid{T}) where {T} = ParticleSet{T}(mesh, NTuple{3,T}[])

"""
    add_particles!(ps, positions)

Append `positions` (any iterable of `(x, y, z)` tuples) to the set.
"""
function add_particles!(ps::ParticleSet{T}, new_pos) where {T}
    for p in new_pos
        push!(ps.pos, (T(p[1]), T(p[2]), T(p[3])))
    end
    return ps
end

"Read-only view of the current particle positions."
positions(ps::ParticleSet) = ps.pos

# Linear interpolation of a face-centered velocity to an arbitrary
# location. Handles periodic wrap simply — assumes velocity halos are
# already filled.
function _interp_faceX(u::Field, x, y, z, m::RegularGrid)
    ix = (x - m.x0) / m.dx + 0.5           # face at (i-1)dx = value at index i
    jy = (y - m.y0) / m.dy + 0.5
    kz = (z - m.z0) / m.dz + 0.5
    return _trilinear(u.data, ix, jy, kz, m)
end

_interp_faceY(v::Field, x, y, z, m::RegularGrid) =
    _trilinear(v.data,
               (x - m.x0) / m.dx + 0.5,
               (y - m.y0) / m.dy + 0.5,
               (z - m.z0) / m.dz + 0.5,  m)

_interp_faceZ(w::Field, x, y, z, m::RegularGrid) =
    _trilinear(w.data,
               (x - m.x0) / m.dx + 0.5,
               (y - m.y0) / m.dy + 0.5,
               (z - m.z0) / m.dz + 0.5,  m)

# Trilinear interpolation in the (Hx+1..Hx+Nx) interior window.
function _trilinear(arr::AbstractArray{T,3}, xf, yf, zf, m::RegularGrid) where {T}
    Hx, Hy, Hz = m.Hx, m.Hy, m.Hz
    Nx, Ny, Nz = m.Nx, m.Ny, m.Nz
    # clamp/wrap into interior (periodic)
    xi = mod(xf - 1, Nx) + 1
    yj = mod(yf - 1, Ny) + 1
    zk = mod(zf - 1, Nz) + 1
    i0 = Int(floor(xi)); i1 = i0 + 1
    j0 = Int(floor(yj)); j1 = j0 + 1
    k0 = Int(floor(zk)); k1 = k0 + 1
    a  = xi - i0
    b  = yj - j0
    c  = zk - k0
    # wrap indices
    wrap(i, N) = mod(i - 1, N) + 1
    i0 = wrap(i0, Nx); i1 = wrap(i1, Nx)
    j0 = wrap(j0, Ny); j1 = wrap(j1, Ny)
    k0 = wrap(k0, Nz); k1 = wrap(k1, Nz)
    # gather values with halo offset
    val(i, j, k) = arr[Hx + i, Hy + j, Hz + k]
    v000 = val(i0, j0, k0); v100 = val(i1, j0, k0)
    v010 = val(i0, j1, k0); v110 = val(i1, j1, k0)
    v001 = val(i0, j0, k1); v101 = val(i1, j0, k1)
    v011 = val(i0, j1, k1); v111 = val(i1, j1, k1)
    v00 = v000*(1-a) + v100*a
    v10 = v010*(1-a) + v110*a
    v01 = v001*(1-a) + v101*a
    v11 = v011*(1-a) + v111*a
    v0  = v00*(1-b) + v10*b
    v1  = v01*(1-b) + v11*b
    return v0*(1-c) + v1*c
end

"""
    advance_particles!(ps, sim, dt)

Advance every particle by `dt` under `sim`'s Eulerian velocity field
using explicit Euler and trilinear interpolation. Wraps positions
around the domain periodically (works cleanly for triply-periodic runs;
walls / IBM will trim escaped particles in a future refinement).
"""
function advance_particles!(ps::ParticleSet{T}, sim::BoussinesqSimulation, dt::Real) where {T}
    m = ps.mesh
    Δt = T(dt)
    @inbounds for i in eachindex(ps.pos)
        x, y, z = ps.pos[i]
        u = _interp_faceX(sim.u, x, y, z, m)
        v = _interp_faceY(sim.v, x, y, z, m)
        w = _interp_faceZ(sim.w, x, y, z, m)
        x += Δt * u; y += Δt * v; z += Δt * w
        # periodic wrap into [x0, x0+L)
        x = x - m.Lx * floor((x - m.x0) / m.Lx)
        y = y - m.Ly * floor((y - m.y0) / m.Ly)
        z = z - m.Lz * floor((z - m.z0) / m.Lz)
        ps.pos[i] = (x, y, z)
    end
    return ps
end

end # module
