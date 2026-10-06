module IO

# Phase 9 — I/O infrastructure.
#
# We ship lightweight facilities using only the Julia standard library:
#
#   * Field snapshots via `Serialization` (`write_snapshot`) — binary Julia
#     format, portable across machines running the same Julia version.
#   * Full-state, versioned checkpoints with bitwise restart (#245,
#     io/checkpoint.jl): fields with halos, SGS closure memory, wall-model
#     and other component state, stepper and RNG state, FFTW provenance;
#     periodic checkpoints with automatic resume (`CheckpointSchedule`).
#   * Configuration via `TOML` — human-editable namelists.
#
# A NetCDF-backed writer (interoperable with Python / VAPOR / Paraview)
# is planned as an optional binding once the HDF5 stack stabilizes on
# the development machine.

using Serialization
using TOML
using Logging: @logmsg, Info, Debug
using ..RunLog: current_domain   # structured events (LOG-a, #333)
using ..Grids: interior, RegularGrid
using ..Fields: Field, interior_view, location
using ..Simulations: BoussinesqSimulation

export write_snapshot, read_snapshot, checkpoint!, restart!, load_config
# #245: full-state checkpoints (io/checkpoint.jl)
export CHECKPOINT_VERSION, read_checkpoint, checkpoint_state, restore_checkpoint_state!,
       fieldwise_checkpoint_state, restore_fieldwise!, checkpoint_fields,
       StepperState, record_step!, previous_dt,
       CheckpointSchedule, checkpoint_schedule, resume!, maybe_checkpoint!, checkpoint_due,
       latest_checkpoint, checkpoint_path, wisdom_path, apply_fftw_config!,
       driver_checkpoints, finish_checkpoints!

# -------------------------------------------------------------------------
# Snapshots — a dictionary of interior arrays + metadata, serialized.
# -------------------------------------------------------------------------

"""
    write_snapshot(path, sim; extra = Dict())

Serialize the interior of `sim.u, sim.v, sim.w, sim.p, sim.θ, sim.qv` plus
the scalar time / clock to `path`. The `extra` dict is copied into the
snapshot for user-defined metadata (dt, step number, tags, …).
Round-trips through [`read_snapshot`](@ref).

A snapshot is **output, not a restart file**: it has no closure memory,
halos or stepper state, and [`restart!`](@ref) refuses it. Use
[`checkpoint!`](@ref) to continue a run.
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

# Full-state, versioned checkpoint / restart (#245): checkpoint!, restart!,
# the checkpoint_state component generic, StepperState, CheckpointSchedule.
include("io/checkpoint.jl")

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
