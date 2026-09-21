module IO

# Phase 9 — I/O infrastructure.
#
# We ship two lightweight facilities using only the Julia standard
# library:
#
#   * Snapshot / restart via `Serialization` — binary Julia format.
#     Portable across machines running the same Julia version.
#   * Configuration via `TOML` — human-editable namelists.
#
# A NetCDF-backed writer (interoperable with Python / VAPOR / Paraview)
# is planned as an optional binding once the HDF5 stack stabilizes on
# the development machine.

using Serialization
using TOML
using ..Grids: interior, RegularGrid
using ..Fields: Field, interior_view, location
using ..Simulations: BoussinesqSimulation

export write_snapshot, read_snapshot, checkpoint!, restart!, load_config

# -------------------------------------------------------------------------
# Snapshots — a dictionary of interior arrays + metadata, serialized.
# -------------------------------------------------------------------------

"""
    write_snapshot(path, sim; extra = Dict())

Serialize the interior of `sim.u, sim.v, sim.w, sim.p, sim.θ` plus the
scalar time / clock to `path`. The `extra` dict is copied into the
snapshot for user-defined metadata (dt, step number, tags, …).
Round-trips through [`read_snapshot`](@ref).
"""
function write_snapshot(path::AbstractString, sim::BoussinesqSimulation;
                        extra::AbstractDict = Dict{Symbol,Any}())
    payload = Dict{Symbol,Any}(
        :t   => sim.t,
        :Nx  => sim.mesh.Nx, :Ny => sim.mesh.Ny, :Nz => sim.mesh.Nz,
        :Lx  => sim.mesh.Lx, :Ly => sim.mesh.Ly, :Lz => sim.mesh.Lz,
        :dx  => sim.mesh.dx, :dy => sim.mesh.dy, :dz => sim.mesh.dz,
        :u   => Array(interior_view(sim.u)),
        :v   => Array(interior_view(sim.v)),
        :w   => Array(interior_view(sim.w)),
        :p   => Array(interior_view(sim.p)),
        :θ   => Array(interior_view(sim.θ)),
        :qv  => Array(interior_view(sim.qv)),
        :has_moisture => sim.has_moisture,
    )
    for (k, v) in extra
        payload[Symbol(k)] = v
    end
    open(path, "w") do io
        serialize(io, payload)
    end
    return path
end

"""
    read_snapshot(path) -> Dict{Symbol,Any}

Read a snapshot back from disk. Returns the payload dictionary written
by [`write_snapshot`](@ref).
"""
function read_snapshot(path::AbstractString)
    open(path, "r") do io
        deserialize(io)
    end
end

"""
    checkpoint!(sim, path)

Alias for `write_snapshot(path, sim)` — recording the current state so
`restart!` can resume from it.
"""
checkpoint!(sim::BoussinesqSimulation, path::AbstractString) = write_snapshot(path, sim)

"""
    restart!(sim, path)

Load a snapshot from `path` and copy `u, v, w, p, θ, t` into `sim`.
Mesh dimensions must match. Returns `sim`.
"""
function restart!(sim::BoussinesqSimulation, path::AbstractString)
    d = read_snapshot(path)
    m = sim.mesh
    (d[:Nx], d[:Ny], d[:Nz]) == (m.Nx, m.Ny, m.Nz) ||
        error("restart!: mesh size mismatch: file $(d[:Nx])×$(d[:Ny])×$(d[:Nz]) vs. sim $(m.Nx)×$(m.Ny)×$(m.Nz)")
    interior_view(sim.u) .= d[:u]
    interior_view(sim.v) .= d[:v]
    interior_view(sim.w) .= d[:w]
    interior_view(sim.p) .= d[:p]
    interior_view(sim.θ) .= d[:θ]
    if haskey(d, :qv)
        interior_view(sim.qv) .= d[:qv]
    end
    sim.t = d[:t]
    return sim
end

# -------------------------------------------------------------------------
# Configuration — TOML namelists.
# -------------------------------------------------------------------------

"""
    load_config(path) -> Dict{String,Any}

Parse a TOML configuration file into a nested dictionary. Users
translate the entries into `BoussinesqSimulation` constructor kwargs.
Example TOML:

```toml
[mesh]
size   = [64, 64, 32]
extent = [3200.0, 3200.0, 1600.0]

[physics]
θ₀            = 300.0
g_accel       = 9.81
C_s           = 0.17
θ_flux_bottom = 0.24
```
"""
load_config(path::AbstractString) = TOML.parsefile(path)

end # module
