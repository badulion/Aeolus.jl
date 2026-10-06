# P59-d (#185) — the free-stream (transpiration) top boundary of a
# spatially developing boundary layer.
#
# Included inside `module BoundaryConditions`, after open_boundary.jl: the
# free-stream top is a special case of the open-face kernels there (N-open-b,
# #255) — a fixed normal-velocity profile (no boundary data, no mass-flux
# correction: its net transpiration is zero by construction), the halo
# beyond the face antisymmetric (`:odd`), and zero-gradient (`:mirror`)
# tangential and scalar halos.

"""
    FreeStreamBoundary(w_top::AbstractVector)

Free-stream top boundary for a spatially developing boundary layer in an
`x`-periodic domain: zero shear stress for the tangential velocities
(`∂u/∂z = ∂v/∂z = 0`, mirror halo — as on a free-slip wall) and a
**prescribed normal velocity** `w = w_top(x)` on the top face. Cell-centred
fields (`p, θ, νₜ`) get the mirror (Neumann) halo. This is the far-field
condition of the spatial DNS of Spalart (1988) and Brandt, Schlatter &
Henningson (2004): the boundary layer displaces the free stream upward at
``W_∞ = U_∞ dδ^*/dx``, and a closed (`w = 0`) lid would instead
accelerate the free stream by ``Δδ^*/H`` and impose a spurious favourable
pressure gradient (at ``H = 40δ^*_0`` the local-similarity estimate of
the Falkner–Skan wall stress is ≈ 18 % above Blasius at the outlet).

`w_top` is indexed by the **storage** `x` index of a `FaceZ` field
(length `Nx + 2Hx`, i.e. the cell-centre `x` positions including halos)
and is used for every `y`. Because the domain is periodic in `x` and `y`
and the bottom is impermeable, solvability of the pressure problem needs
**zero net transpiration**, `∑ w_top[interior] = 0`; the constructor does
not know the grid, so the caller guarantees it (the Blasius helper
`Transition.blasius_freestream` returns a zero-mean profile, the outflow
of the plate being returned through the fringe region).

The pressure problem is unchanged (Neumann at the top face, the
`CosineFFTPoissonSolver` is exact): the prescribed face value enters the
divergence of the top cell, and the projection leaves it untouched.

It is the special case of an [`AbstractOpenBoundary`](@ref) top face with
a fixed normal velocity, an antisymmetric halo beyond the face and mirror
halos for everything else: it is filled by the same face kernels as
[`OpenBoundary`](@ref) (bit for bit the P59-d fill). Unlike an
`OpenBoundary` it carries no boundary data and is not mass corrected.

Valid only as the `top` face, paired with a wall (`:wall`, `:noslip`,
`NoSlipWall`) at the bottom:
`FieldBoundaryConditions(:periodic, :periodic; bottom = :noslip, top = FreeStreamBoundary(w))`.

References: Spalart, P. R. (1988). Direct simulation of a turbulent
boundary layer up to R_θ = 1410. *J. Fluid Mech.*, 187, 61-98.
Brandt, L., Schlatter, P. & Henningson, D. S. (2004). Transition in
boundary layers subject to free-stream turbulence. *J. Fluid Mech.*,
517, 167-198.
"""
struct FreeStreamBoundary{V<:AbstractVector{<:Real}} <: AbstractOpenBoundary
    w_top::V
end

# Allowed z pairs: an impermeable wall below an open top (free stream or
# OpenBoundary); an open face is never a bottom.
_z_pair_supported(b, t::AbstractOpenBoundary) = is_wall(b)
_z_pair_supported(b::AbstractOpenBoundary, t) = false
_z_pair_supported(b::AbstractOpenBoundary, t::AbstractOpenBoundary) = false

function _fill_z!(d, bot::_ZWall, top::FreeStreamBoundary, Hz, Nz, loc::Location)
    _fill_wall_bottom!(d, bot, Hz, loc)
    _fill_freestream_top!(d, top, Hz, Nz, loc)
    return d
end

# u, v and cell-centred fields: mirror (zero shear / Neumann).
_fill_freestream_top!(d, ::FreeStreamBoundary, Hz, Nz, ::Union{CellCenter,FaceX,FaceY}) =
    _open_mirror!(d, 3, true, Hz, Nz)

# w: the prescribed face value, halo antisymmetric about it.
function _fill_freestream_top!(d::AbstractArray{T,3}, bc::FreeStreamBoundary, Hz, Nz,
                               ::FaceZ) where {T}
    wb = bc.w_top
    length(wb) == size(d, 1) ||
        throw(DimensionMismatch("FreeStreamBoundary: w_top has $(length(wb)) entries, " *
                                "the field has $(size(d, 1)) storage x points"))
    kw = _face_index(true, Hz, Nz)
    if kw <= size(d, 3)
        @inbounds for j in axes(d, 2), i in axes(d, 1)
            d[i, j, kw] = T(wb[i])
        end
    end
    return _open_odd!(d, 3, true, Hz, Nz)
end

# Homogeneous part (implicit solvers): zero transpiration.
_inhomogeneous(bc::FreeStreamBoundary) = any(!iszero, bc.w_top)
_homogeneous(bc::FreeStreamBoundary) = FreeStreamBoundary(zero(bc.w_top))
