# The face values of AdvectionSchemes as they were before #557: upwinded
# with a branch (`if vel >= 0`, the left- or the right-biased stencil and
# its own reconstruction). `Branching(r)` is a reconstruction descriptor
# whose face bodies take that branch around `r`; everything else (stencils,
# halo, name) is `r`'s. Every operator that takes a descriptor runs with
# it unchanged, so a test (test/advection_branchfree.jl, its -t 4
# subprocess) or a benchmark compares the branch-free bodies against the
# branching ones on the same operator. Not part of the package.

using Aeolus
const _AS = Aeolus.AdvectionSchemes

struct Branching{R<:_AS.AbstractReconstruction} <: _AS.AbstractReconstruction
    r::R
end
_AS.stencil_left(b::Branching) = _AS.stencil_left(b.r)
_AS.stencil_right(b::Branching) = _AS.stencil_right(b.r)
_AS.required_halo(b::Branching) = _AS.required_halo(b.r)
_AS.scheme_name(b::Branching) = _AS.scheme_name(b.r)
_AS._check_stretched_recon(b::Branching, g::Aeolus.StretchedGrid) = _AS._check_stretched_recon(b.r, g)
_AS._bsf_threads(b::Branching) = _AS._bsf_threads(b.r)

@inline function _AS._face_x(b::Branching, d, c::Int, j::Int, k::Int, vel)
    r = b.r
    if vel >= 0
        lo, n = _AS.stencil_left(r);  return _AS.reconstruct_left(r, _AS._gather_x(d, c, j, k, lo, n))
    else
        lo, n = _AS.stencil_right(r); return _AS.reconstruct_right(r, _AS._gather_x(d, c, j, k, lo, n))
    end
end
@inline function _AS._face_y(b::Branching, d, i::Int, c::Int, k::Int, vel)
    r = b.r
    if vel >= 0
        lo, n = _AS.stencil_left(r);  return _AS.reconstruct_left(r, _AS._gather_y(d, i, c, k, lo, n))
    else
        lo, n = _AS.stencil_right(r); return _AS.reconstruct_right(r, _AS._gather_y(d, i, c, k, lo, n))
    end
end
@inline function _AS._face_z(b::Branching, d, i::Int, j::Int, c::Int, vel)
    r = b.r
    if vel >= 0
        lo, n = _AS.stencil_left(r);  return _AS.reconstruct_left(r, _AS._gather_z(d, i, j, c, lo, n))
    else
        lo, n = _AS.stencil_right(r); return _AS.reconstruct_right(r, _AS._gather_z(d, i, j, c, lo, n))
    end
end
