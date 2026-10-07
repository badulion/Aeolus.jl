# Nesting: the 1-D geometry of a parent–child pair along one axis (#328).
#
# A `NestAxis` describes, along x, y or z, where the child's faces lie
# relative to the parent's: aligned with an integer ratio (the N-nest-a
# case, every operator keeps its integer fast path), with a rational ratio
# (5 child cells in 2 parent cells), with an origin off the parent faces,
# or (z) on a stretched child grid. Everything the general operators need is
# read from it:
#
#   * the parent and child face coordinates (child faces within 1e-9 of a
#     child spacing of a parent face are snapped to that parent face, bit
#     for bit, so the Lagrange bases are exactly 0/1 there);
#   * the *common faces* (a parent face that is also a child face) and the
#     *conservation box*: the parent cells between the first and the last
#     common face inside the child. Two-way transfers (anterpolation,
#     refluxing, flux correction, composite integrals) live on the box,
#     whose planes are faces of both grids;
#   * the overlap of every box parent cell with the child cells (CSR):
#     the weights of the conservative restriction;
#   * for each box parent face, the coincident child face, or the two child
#     faces that bracket it (face restriction off the common faces).

"Child meshes a [`NestedGrid`](@ref) accepts: uniform, or stretched in z (#328)."
const NestChildMesh{T} = Union{RegularGrid{T}, StretchedGrid{T}}

"""
    NestAxis

The geometry of a [`NestedGrid`](@ref) along one axis (#328): parent and
child face coordinates (storage-indexed; child faces that lie on a parent
face carry that face's bits), the conservation box (`pbox` parent
interior cells, `cbox` child interior cells, bounded by common faces), the
overlap lengths of each box parent cell with the child cells (`ptr`,
`cidx`, `len`: compressed rows), the child face of each box parent face
(`cface`; 0 off the common faces, where `fbr`/`ftau` give the bracketing
child face and the linear weight) and, for an axis aligned with an integer
ratio and a uniform child, `integer = (r, offset)`.
"""
struct NestAxis
    pf::Vector{Float64}
    cf::Vector{Float64}
    Hp::Int; Np::Int
    Hc::Int; Nc::Int
    pbox::UnitRange{Int}
    cbox::UnitRange{Int}
    ptr::Vector{Int}
    cidx::Vector{Int}
    len::Vector{Float64}
    cface::Vector{Int}
    fbr::Vector{Int}
    ftau::Vector{Float64}
    integer::Union{Nothing,NTuple{2,Int}}
end

Base.show(io::IO, a::NestAxis) =
    print(io, "NestAxis(", a.integer === nothing ? "overlap" : "r=$(a.integer[1])",
          ", box parent cells ", a.pbox, ", child cells ", a.cbox, ")")

"Number of box parent cells along the axis."
_nbox(a::NestAxis) = length(a.pbox)

# Snap the child faces onto coincident parent faces (relative tolerance
# `rtol` of the local child spacing), in place.
function _snap_faces!(cf::Vector{Float64}, pf::Vector{Float64}; rtol = 1e-9)
    for f in eachindex(cf)
        h = f < length(cf) ? cf[f+1] - cf[f] : cf[f] - cf[f-1]
        P = searchsortedlast(pf, cf[f])
        for Q in (P, P + 1)
            1 <= Q <= length(pf) || continue
            abs(cf[f] - pf[Q]) <= rtol * abs(h) && (cf[f] = pf[Q]; break)
        end
    end
    return cf
end

"""
    NestAxis(pf, Hp, Np, cf, Hc, Nc; uniform_child = true, integer = true)

Build the axis geometry from the storage-indexed parent faces `pf`
(length `Np + 2Hp + 1`) and child faces `cf` (length `Nc + 2Hc + 1`;
snapped onto coincident parent faces here). `integer = false` refuses the
integer fast path even for an aligned axis (the overlap path, used to
prove that the integer operators are its special case).
"""
function NestAxis(pf::Vector{Float64}, Hp::Int, Np::Int, cf0::AbstractVector, Hc::Int, Nc::Int;
                  uniform_child::Bool = true, integer::Bool = true)
    length(pf) == Np + 2Hp + 1 || error("NestAxis: parent faces have the wrong length")
    length(cf0) == Nc + 2Hc + 1 || error("NestAxis: child faces have the wrong length")
    cf = _snap_faces!(Float64.(collect(cf0)), pf)
    issorted(cf; lt = <=) || error("NestAxis: child faces must be strictly increasing")
    # common faces: parent interior faces that are child interior faces
    pint = Hp+1:Hp+Np+1
    common = Tuple{Int,Int}[]
    for f in Hc+1:Hc+Nc+1
        P = searchsortedfirst(pf, cf[f])
        P in pint && pf[P] == cf[f] && push!(common, (P, f))
    end
    if length(common) >= 2
        (P0, f0) = first(common); (P1, f1) = last(common)
        pbox = (P0 - Hp):(P1 - Hp - 1)
        cbox = (f0 - Hc):(f1 - Hc - 1)
    else
        P0 = f0 = 0
        pbox = 1:0; cbox = 1:0
    end
    # overlaps of the box parent cells with the child cells
    ptr = Int[1]; cidx = Int[]; len = Float64[]
    for P in pbox
        Ps = Hp + P
        a = pf[Ps]; b = pf[Ps+1]
        for c in (Hc + first(cbox)):(Hc + last(cbox))
            l = min(b, cf[c+1]) - max(a, cf[c])
            l > 0 && (push!(cidx, c); push!(len, l))
        end
        push!(ptr, length(cidx) + 1)
    end
    # the child face of every box parent face, or its bracketing pair
    nb = length(pbox)
    cface = zeros(Int, nb + 1); fbr = zeros(Int, nb + 1); ftau = zeros(nb + 1)
    for q in 0:nb
        nb == 0 && break
        x = pf[P0 + q]
        f = searchsortedlast(cf, x)
        if cf[f] == x
            cface[q+1] = f; fbr[q+1] = f
        else
            fbr[q+1] = f; ftau[q+1] = (x - cf[f]) / (cf[f+1] - cf[f])
        end
    end
    # the integer fast path: box = whole child, r child cells per parent cell,
    # every r-th child face common, uniform child spacing
    int = nothing
    if integer && uniform_child && nb >= 1 && first(cbox) == 1 && last(cbox) == Nc && Nc % nb == 0
        r = Nc ÷ nb
        if all(n -> cface[n+1] == Hc + 1 + n * r, 0:nb)
            int = (r, first(pbox))
        end
    end
    return NestAxis(pf, cf, Hp, Np, Hc, Nc, pbox, cbox, ptr, cidx, len, cface, fbr, ftau, int)
end

# Affine storage faces of a uniform axis: x0 + (i - H - 1) Δ.
_affine_faces(x0, Δ, H, N) = Float64[x0 + (i - H - 1) * Δ for i in 1:N+2H+1]

# The faces of each axis of a mesh (z: zfaces of a stretched mesh).
_mesh_faces(m, d::Int) =
    d == 1 ? _affine_faces(m.x0, m.dx, m.Hx, m.Nx) :
    d == 2 ? _affine_faces(m.y0, m.dy, m.Hy, m.Ny) : _zfaces_storage(m)

_mesh_HN(m, d::Int) = d == 1 ? (m.Hx, m.Nx) : d == 2 ? (m.Hy, m.Ny) : (m.Hz, m.Nz)

# Overlap lengths of the box parent cells with the child cells in another
# coordinate (the reference mass μ of a sound-proof pair): the same rows
# as `a.len`, in the coordinates `pc` (parent faces) and `cc` (child faces).
function _overlap_lengths(a::NestAxis, pc::Vector{Float64}, cc::Vector{Float64})
    out = similar(a.len)
    for (n, P) in enumerate(a.pbox)
        Ps = a.Hp + P
        for q in a.ptr[n]:a.ptr[n+1]-1
            c = a.cidx[q]
            out[q] = min(pc[Ps+1], cc[c+1]) - max(pc[Ps], cc[c])
        end
    end
    return out
end
