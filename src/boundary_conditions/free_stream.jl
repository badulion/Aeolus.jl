# P59-d (#185) — the free-stream (transpiration) top boundary of a
# spatially developing boundary layer.
#
# Included inside `module BoundaryConditions`, after noslip_wall.jl.

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

Valid only as the `top` face, paired with a wall (`:wall`, `:noslip`,
`NoSlipWall`) at the bottom:
`FieldBoundaryConditions(:periodic, :periodic; bottom = :noslip, top = FreeStreamBoundary(w))`.

References: Spalart, P. R. (1988). Direct simulation of a turbulent
boundary layer up to R_θ = 1410. *J. Fluid Mech.*, 187, 61-98.
Brandt, L., Schlatter, P. & Henningson, D. S. (2004). Transition in
boundary layers subject to free-stream turbulence. *J. Fluid Mech.*,
517, 167-198.
"""
struct FreeStreamBoundary{V<:AbstractVector{<:Real}} <: AbstractBC
    w_top::V
end

# Allowed pair: an impermeable wall below, the free stream above.
_z_pair_supported(b, t::FreeStreamBoundary) = is_wall(b)
_z_pair_supported(b::FreeStreamBoundary, t) = false
_z_pair_supported(b::FreeStreamBoundary, t::FreeStreamBoundary) = false

function _fill_z!(d, bot::_ZWall, top::FreeStreamBoundary, Hz, Nz, loc::Location)
    _fill_wall_bottom!(d, bot, Hz, loc)
    _fill_freestream_top!(d, top, Hz, Nz, loc)
    return d
end

# u, v and cell-centred fields: mirror (zero shear / Neumann).
function _fill_freestream_top!(d, ::FreeStreamBoundary, Hz, Nz,
                               ::Union{CellCenter,FaceX,FaceY})
    @inbounds for h in 1:Hz
        @views d[:, :, Hz + Nz + h] .= d[:, :, Hz + Nz + 1 - h]
    end
    return d
end

# w: the prescribed face value, halo antisymmetric about it.
function _fill_freestream_top!(d::AbstractArray{T,3}, bc::FreeStreamBoundary, Hz, Nz,
                               ::FaceZ) where {T}
    wb = bc.w_top
    length(wb) == size(d, 1) ||
        throw(DimensionMismatch("FreeStreamBoundary: w_top has $(length(wb)) entries, " *
                                "the field has $(size(d, 1)) storage x points"))
    kw = Hz + Nz + 1
    @inbounds for k_ in 0:Hz
        dst = kw + k_
        dst <= size(d, 3) || continue
        for j in axes(d, 2), i in axes(d, 1)
            d[i, j, dst] = k_ == 0 ? T(wb[i]) : 2 * T(wb[i]) - d[i, j, kw - k_]
        end
    end
    return d
end
