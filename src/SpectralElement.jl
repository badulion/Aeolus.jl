module SpectralElement

# Phase 18 — Spectral element mesh backend (scaffold).
#
# Introduces `SpectralElementMesh <: AbstractMesh` — the second concrete
# mesh type in Aeolus. Storage layout is (N+1) × (N+1) × (N+1) × Nel
# where N is the polynomial order and Nel the number of elements.
# Values sit at Gauss–Lobatto–Legendre (GLL) nodes inside each element.
#
# What this scaffold ships:
#
#   * `SpectralElementMesh` type + `AbstractMesh` interface impl:
#     `totalsize`, `interior`, `xnodes/ynodes/znodes`.
#   * `gll_nodes(N)` and `gll_weights(N)` — reference-element quadrature
#     via the classical Newton iteration on Legendre-polynomial roots.
#
# What's deferred (a whole follow-up phase in itself):
#
#   * Tensor-product differentiation operators (`_∂x!` etc.) for
#     `SpectralElementMesh` — these plug in via multiple dispatch on
#     the mesh type in the existing `Operators` façade.
#   * Element-local mass matrix and matrix-free CG pressure solve.
#   * `BoussinesqSimulation` awareness of SEM storage layout.
#
# The point of shipping the scaffold now is to prove the two-backend
# extensibility promise: any code written against `AbstractMesh` will
# accept `SpectralElementMesh` once the operator methods land.

using ..Grids: AbstractMesh, Location, CellCenter

export SpectralElementMesh, gll_nodes, gll_weights

"""
    SpectralElementMesh{T}(; order, elements_per_dim, extent, origin = (0,0,0))

Uniform-brick spectral-element mesh: `Nex × Ney × Nez` elements, each a
polynomial of order `N = order` in every direction. Storage per field
is `(N+1) × (N+1) × (N+1) × (Nex × Ney × Nez)` values at GLL nodes.

Coordinate access returns the flattened GLL-node positions per axis
(with duplicates at element interfaces — the "continuous Galerkin"
gather comes later). Halos are zero (SEM handles inter-element
communication through element-local weak boundary terms, not halo
copies), so `interior` returns the full index range.
"""
struct SpectralElementMesh{T<:AbstractFloat} <: AbstractMesh{T}
    order::Int
    Nex::Int; Ney::Int; Nez::Int
    Lx::T;   Ly::T;   Lz::T
    x0::T;   y0::T;   z0::T
    ξ::Vector{T}      # reference-element GLL nodes on [-1, 1]
    w::Vector{T}      # reference-element GLL weights
end

function SpectralElementMesh(::Type{T}=Float64;
                             order::Int,
                             elements_per_dim::NTuple{3,Int},
                             extent::NTuple{3,Real},
                             origin::NTuple{3,Real}=(0,0,0)) where {T<:AbstractFloat}
    Nex, Ney, Nez = elements_per_dim
    Lx, Ly, Lz = T.(extent)
    x0, y0, z0 = T.(origin)
    ξ, w = _gll_nodes_and_weights(T, order)
    return SpectralElementMesh{T}(order, Nex, Ney, Nez, Lx, Ly, Lz, x0, y0, z0, ξ, w)
end

# --- AbstractMesh interface impl -----------------------------------------

# Storage per scalar field: (N+1) values along each axis × (Nex × Ney × Nez)
# elements — flattened here into a 4-tuple.
Base.eltype(::SpectralElementMesh{T}) where {T} = T

"""
    totalsize(sem_mesh)

Returns the 4-tuple `(N+1, N+1, N+1, Nex·Ney·Nez)` — polynomial-node
count in each intra-element direction times the total element count.
"""
function totalsize(m::SpectralElementMesh)
    Np = m.order + 1
    return (Np, Np, Np, m.Nex * m.Ney * m.Nez)
end

# No halo layer in SEM storage — inter-element coupling is done in weak form.
function interior(m::SpectralElementMesh)
    Np = m.order + 1
    return (1:Np, 1:Np, 1:Np, 1:m.Nex * m.Ney * m.Nez)
end

# 1-D coordinate access. Returns a vector of node positions along the
# axis, listing every element's Np nodes in order. Duplicate values at
# element interfaces are intentional (values are single-valued per
# element in the discontinuous representation used at storage level).
function _axis_nodes(order::Int, ξ::Vector{T}, Nel::Int, L::T, x0::T) where {T}
    Np    = order + 1
    Δe    = L / Nel                        # element extent
    nodes = zeros(T, Np * Nel)
    for e in 1:Nel
        e_x0 = x0 + (e - 1) * Δe
        for p in 1:Np
            nodes[(e - 1) * Np + p] = e_x0 + 0.5 * Δe * (ξ[p] + 1)
        end
    end
    return nodes
end

# SEM has no staggering — every field lives at GLL nodes. We ignore the
# `Location` argument for now.
xnodes(m::SpectralElementMesh, ::Location) = _axis_nodes(m.order, m.ξ, m.Nex, m.Lx, m.x0)
ynodes(m::SpectralElementMesh, ::Location) = _axis_nodes(m.order, m.ξ, m.Ney, m.Ly, m.y0)
znodes(m::SpectralElementMesh, ::Location) = _axis_nodes(m.order, m.ξ, m.Nez, m.Lz, m.z0)
xnodes(m::SpectralElementMesh) = xnodes(m, CellCenter())
ynodes(m::SpectralElementMesh) = ynodes(m, CellCenter())
znodes(m::SpectralElementMesh) = znodes(m, CellCenter())

function Base.show(io::IO, m::SpectralElementMesh{T}) where {T}
    print(io, "SpectralElementMesh{$T}(order=$(m.order), elements=($(m.Nex),$(m.Ney),$(m.Nez)), ")
    print(io, "extent=($(m.Lx),$(m.Ly),$(m.Lz)))")
end

# =========================================================================
# GLL nodes and weights on [-1, 1] — Newton iteration on Lobatto polys.
# =========================================================================

"""
    gll_nodes(N) -> Vector

Gauss–Lobatto–Legendre nodes of order `N` on the reference interval
`[-1, 1]`. Returns `N+1` values; the two endpoints are exactly ∓1.
"""
gll_nodes(N::Int) = _gll_nodes_and_weights(Float64, N)[1]

"""
    gll_weights(N) -> Vector

Corresponding GLL quadrature weights; summing them yields `2` (the
length of the reference interval).
"""
gll_weights(N::Int) = _gll_nodes_and_weights(Float64, N)[2]

function _gll_nodes_and_weights(::Type{T}, N::Int) where {T}
    N >= 1 || error("GLL: order N must be ≥ 1")
    Np = N + 1
    if Np == 2
        return T[-1, 1], T[1, 1]
    end
    # Chebyshev initial guess for interior nodes; endpoints exact
    x = zeros(T, Np)
    x[1] = -1; x[end] = 1
    for k in 2:(Np - 1)
        x[k] = -cos(π * (k - 1) / (Np - 1))
    end
    # Legendre polynomial matrix P[i,j] = P_{j-1}(x_i), evaluated via
    # the three-term recurrence.
    P = zeros(T, Np, Np)
    xold = fill(T(2), Np)
    iter = 0
    while maximum(abs.(x .- xold)) > 4eps(T) && iter < 100
        xold .= x
        P[:, 1] .= 1
        P[:, 2] .= x
        for k in 2:N
            P[:, k + 1] .= ((2k - 1) .* x .* P[:, k] .- (k - 1) .* P[:, k - 1]) ./ k
        end
        # Newton update: only interior nodes move
        for i in 2:(Np - 1)
            x[i] = xold[i] -
                   (x[i] * P[i, Np] - P[i, Np - 1]) /
                   (Np * P[i, Np])
        end
        iter += 1
    end
    w = 2 ./ (N * Np .* P[:, Np] .^ 2)
    return x, w
end

end # module
