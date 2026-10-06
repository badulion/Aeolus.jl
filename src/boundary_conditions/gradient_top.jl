# N-parent-b (#254) — the fixed-gradient top of a scalar: FluxBC(Γ) as the
# top face, above a wall or a no-flux bottom.
#
# Included inside `module BoundaryConditions`, after free_stream.jl. Only
# the z-top scalar fill lives here; the open boundary of the nesting epic
# (N-open-b) has its own file.

# An open bottom (FreeStreamBoundary, OpenBoundary) is never valid
# (resolves the ambiguity with `_z_pair_supported(b, t::FluxBC)`).
_z_pair_supported(b::AbstractOpenBoundary, t::FluxBC) = false

# Distance between the centres of storage levels `a` and `b` (b − a): the
# centre heights of a StretchedGrid (`zc`, halos included), `dz` steps on a
# RegularGrid.
_center_gap(g, a::Int, b::Int) = hasproperty(g, :zc) ? g.zc[b] - g.zc[a] : (b - a) * g.dz

function _fill_z!(d::AbstractArray{T,3}, bot, top::FluxBC, Hz, Nz, loc::CellCenter, g, key) where {T}
    top.f isa Real ||
        throw(ArgumentError("FluxBC top: the gradient must be a number, got $(typeof(top.f))"))
    # the bottom face (and, provisionally, the top) with the bottom's own pair method
    _fill_z!(d, bot, bot, Hz, Nz, loc)
    Γ = T(top.f)
    kN = Hz + Nz
    @inbounds for h in 1:Hz
        src, dst = kN + 1 - h, kN + h
        δ = Γ * T(_center_gap(g, src, dst))
        @views d[:, :, dst] .= d[:, :, src] .+ δ
    end
    return d
end

_fill_z!(d, bot, top::FluxBC, Hz, Nz, loc::Location, g, key) =
    throw(ArgumentError("FluxBC top is a scalar (CellCenter) condition; a $(nameof(typeof(loc))) field cannot take it"))
