module Parallel

# Phase 15 — Parallelization scaffolding.
#
# This module defines the abstractions Aeolus uses to reason about
# distributed execution without pulling in MPI as a hard dependency.
# The FieldBoundaryConditions machinery already handles per-face halo
# fill for periodic and wall BCs; when the mesh is split across MPI
# ranks, an additional halo-exchange step is inserted along the split
# axes.
#
# Shipped in Phase 15:
#   * `AbstractPartition` type hierarchy.
#   * `SerialPartition`  — single-rank, no communication (current default).
#   * `PencilPartition`  — 2D pencil layout over (x, y) [scaffold; the
#     actual MPI wiring lives in a future weak-dep extension so Aeolus
#     stays MPI-independent when running serially].
#
# Deferred:
#   * MPI weak-dep extension (`ext/AeolusMPI.jl`) that overrides
#     `exchange_halos_distributed!` when `MPI.jl` is loaded.
#   * `PencilFFTs.jl` integration for distributed FFT.

export AbstractPartition, SerialPartition, PencilPartition,
       nranks, is_serial, exchange_halos_distributed!

"""
    AbstractPartition

Base type for describing how a mesh is split across MPI ranks.
Concrete subtypes:

  * [`SerialPartition`](@ref) — no distribution, single rank.
  * [`PencilPartition`](@ref) — 2D pencil split over `x` and `y`.

Serial execution should always work: everything in the operator layer
is written such that a `SerialPartition` is a no-op in
`exchange_halos_distributed!`.
"""
abstract type AbstractPartition end

"Serial (single-rank) partition. The default."
struct SerialPartition <: AbstractPartition end

"""
    PencilPartition(nx, ny)

Two-dimensional MPI pencil layout with `nx` ranks along `x` and `ny`
ranks along `y`. `z` is local to every rank.

Constructing a `PencilPartition` does *not* by itself require MPI —
the actual `MPI.Comm` and neighbour lookup are populated by the
future `AeolusMPI` extension when `MPI.jl` is loaded. Without MPI,
using a `PencilPartition` in a running simulation will trip a helpful
error rather than segfault.
"""
struct PencilPartition <: AbstractPartition
    nx::Int
    ny::Int
end

nranks(::SerialPartition) = 1
nranks(p::PencilPartition) = p.nx * p.ny

is_serial(::SerialPartition) = true
is_serial(::PencilPartition) = false

"""
    exchange_halos_distributed!(field, partition)

Exchange the halo layers of `field` between MPI neighbours according to
`partition`. On `SerialPartition` this is a no-op — halo fill via the
per-field BCs already handles the periodic/wall wrap.

On `PencilPartition` (MPI extension), the interior slabs on the split
axes are packed into send buffers and mirrored into the receiver's
halo cells via non-blocking `Isend`/`Irecv`.
"""
exchange_halos_distributed!(field, ::SerialPartition) = field
exchange_halos_distributed!(field, ::PencilPartition) = error(
    "PencilPartition halo exchange requires the AeolusMPI extension. " *
    "Load MPI.jl (`using MPI`) after `using Aeolus` to enable it."
)

end # module
