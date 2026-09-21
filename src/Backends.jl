module Backends

# Phase 17 — Compute-backend scaffolding.
#
# Aeolus fields currently wrap plain `Array{T,3}` allocated on the host.
# For GPU execution we want to be able to swap in `CuArray`, `ROCArray`,
# or `MtlArray` transparently. This module introduces the abstraction —
# the actual GPU kernels live in a future `KernelAbstractions`-based
# port, which will provide extensions like `AeolusCUDA`.
#
# Shipped in Phase 17:
#   * `AbstractBackend` type + `CPUBackend` (default) + `GPUBackend` stub.
#   * `allocate(backend, T, dims)` — routes to `Array` or (future)
#     GPU array constructor.
#   * `to_host(x)` — no-op on CPU; copies device → host on GPU.
#
# Deferred:
#   * `KernelAbstractions.jl` port of every FV kernel.
#   * FFT plans on the device (cuFFT / rocFFT / vkFFT).
#   * GPU-native CG solver.

export AbstractBackend, CPUBackend, GPUBackend, allocate, to_host, backend_name

"""
    AbstractBackend

Base type for the compute-backend selector. Subtypes:

  * [`CPUBackend`](@ref) — default; uses `Base.Array` and standard CPU
    kernels.
  * [`GPUBackend`](@ref) — scaffold. Requires a future
    `KernelAbstractions.jl` extension.
"""
abstract type AbstractBackend end

"CPU backend using `Base.Array`. The current default."
struct CPUBackend <: AbstractBackend end

"""
    GPUBackend(kind = :cuda)

GPU backend stub. `kind` is one of `:cuda`, `:rocm`, `:oneapi`, `:metal`.
Using this backend without the corresponding extension (loading
`CUDA.jl`, `AMDGPU.jl`, …) trips a helpful error rather than silently
falling back to the CPU.
"""
struct GPUBackend <: AbstractBackend
    kind::Symbol
end
GPUBackend() = GPUBackend(:cuda)

backend_name(::CPUBackend) = :cpu
backend_name(b::GPUBackend) = b.kind

"""
    allocate(backend, T, dims) -> Array

Allocate an array of the appropriate type for the given backend.
"""
allocate(::CPUBackend, ::Type{T}, dims::NTuple{N,Int}) where {T, N} =
    zeros(T, dims...)

allocate(b::GPUBackend, ::Type{T}, dims::NTuple{N,Int}) where {T, N} =
    error("allocate: GPUBackend($(b.kind)) requires an extension package " *
          "(load CUDA.jl / AMDGPU.jl / oneAPI.jl / Metal.jl before use).")

"""
    to_host(x) -> Array

Copy `x` to the host. No-op if it already lives on the host.
"""
to_host(x::AbstractArray) = Array(x)

end # module
