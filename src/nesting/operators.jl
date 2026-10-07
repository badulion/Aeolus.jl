# Nesting: 1-D weights and plans, prolongation, restriction, flux mismatch
# and refluxing (moved verbatim from AdaptiveMesh.jl, N-nest-a #250).

# =========================================================================
# 1-D interpolation weights
# =========================================================================

"Lagrange basis values at `x` for the `n` integer nodes `0, 1, …, n-1`."
function _lagrange_basis(n::Int, x::Float64)
    l = Vector{Float64}(undef, n)
    @inbounds for q in 0:n-1
        p = 1.0
        for s in 0:n-1
            s == q && continue
            p *= (x - s) / (q - s)
        end
        l[q+1] = p
    end
    return l
end

"""
    _cons_weights(ns, r, m, L) -> Vector{Float64}

Weights of the *conservative* prolongation: given `ns` consecutive parent
cell averages (stencil-local cells `1…ns`, cell `q` spanning `[q-1, q]`
in units of the parent spacing), return the weights that produce the
average over sub-cell `m` of `r` inside stencil-local parent cell `L`.

Built from the primitive function: `F` is known exactly at the `ns+1`
parent faces, a degree-`ns` polynomial interpolates it, and the sub-cell
average is `r·ΔP`. Because `P` matches `F` *at the nodes*, summing the
`r` sub-cell weights recovers `δ(a, L)` exactly — machine-precision
conservation.
"""
function _cons_weights(ns::Int, r::Int, m::Int, L::Int)
    a = (L - 1) + (m - 1) / r
    b = (L - 1) + m / r
    la = _lagrange_basis(ns + 1, a)
    lb = _lagrange_basis(ns + 1, b)
    w = Vector{Float64}(undef, ns)
    acc = 0.0
    @inbounds for q in ns:-1:1
        acc += lb[q+1] - la[q+1]
        w[q] = r * acc
    end
    return w
end

"Nodal (point-value) Lagrange weights at `ξ` over `ns` unit-spaced nodes."
_nodal_weights(ns::Int, ξ::Float64) = _lagrange_basis(ns, ξ)

# --- per-dimension plan ---------------------------------------------------
# For every child array index along one dimension, record the parent array
# index where the stencil starts and the stencil weights.

function _build_plan(kind::Symbol, order::Int, r::Int, off::Int,
                     Hp::Int, Np::Int, Hc::Int, Nc::Int)
    ns  = max(order, 1)
    lo  = 1 - Hp                       # lowest representable parent cell/face number
    hi  = max(lo, Np + Hp - ns + 1)    # highest legal stencil start
    n   = Nc + 2Hc
    start = Vector{Int}(undef, n); W = Matrix{Float64}(undef, ns, n); cell = zeros(Int, n)
    @inbounds for ic in 1:n
        nn = ic - Hc                   # child cell / face number (may be <=0)
        if kind === :avg
            q = nn - 1
            I = off + fld(q, r)
            m = mod(q, r) + 1
            s0 = clamp(I - ((ns - 1) ÷ 2), lo, hi)
            start[ic] = Hp + s0; cell[ic] = clamp(Hp + I, 1, Np + 2Hp)
            W[:, ic] .= _cons_weights(ns, r, m, I - s0 + 1)
        else # :nodal — child face position in parent-face-number units
            qf = off + (nn - 1) / r
            s0 = clamp(Int(floor(qf)) - ((ns - 1) ÷ 2), lo, hi)
            start[ic] = Hp + s0
            W[:, ic] .= _nodal_weights(ns, qf - s0)
        end
    end
    return _Plan1D(start, W, cell)
end

function _plan(nest::NestedGrid, kind::Symbol, d::Int, order::Int)
    key = (kind, d, order)
    p = get(nest.plans, key, nothing)
    p === nothing || return p
    pm = nest.parent_mesh; cm = nest.child_mesh
    Hp, Np = d == 1 ? (pm.Hx, pm.Nx) : d == 2 ? (pm.Hy, pm.Ny) : (pm.Hz, pm.Nz)
    Hc, Nc = d == 1 ? (cm.Hx, cm.Nx) : d == 2 ? (cm.Hy, cm.Ny) : (cm.Hz, cm.Nz)
    r      = d == 1 ? nest.rx : d == 2 ? nest.ry : nest.rz
    off    = d == 1 ? nest.ix : d == 2 ? nest.iy : nest.iz
    p = _build_plan(kind, order, r, off, Hp, Np, Hc, Nc)
    nest.plans[key] = p
    return p
end

# --- coordinate plans (N-nest-b, #256) ------------------------------------
# The same two rules on arbitrary node positions: the vertical of a
# z-stretched parent (nodes = actual parent face heights) and the mass
# coordinate μ = ∫W dz of the sound-proof systems. Coordinates are shifted
# to the first stencil node and scaled by the mean stencil spacing before
# the Lagrange products are formed, so a child face that *is* a parent face
# (bitwise, see `_child_zfaces`) gives a basis of exactly 0 and 1.

"Lagrange basis values at `x` for the nodes `t` (any distinct positions)."
function _lagrange_basis_nodes(t::AbstractVector{Float64}, x::Float64)
    n = length(t)
    l = Vector{Float64}(undef, n)
    @inbounds for q in 1:n
        p = 1.0
        for s in 1:n
            s == q && continue
            p *= (x - t[s]) / (t[q] - t[s])
        end
        l[q] = p
    end
    return l
end

"""
    _cons_weights_nodes(x, a, b) -> Vector{Float64}

Conservative-prolongation weights on non-uniform parent faces `x`
(`ns + 1` positions bounding `ns` stencil cells): the average over
`[a, b]` of the derivative of the degree-`ns` polynomial that
interpolates the primitive `F(x_j) = Σ_{q≤j} φ_q (x_q − x_{q−1})`.
The weight of cell `q` is `h_q Σ_{j≥q} (L_j(b) − L_j(a)) / (b − a)`;
with unit spacing this is [`_cons_weights`](@ref) exactly.
"""
function _cons_weights_nodes(x::AbstractVector{Float64}, a::Float64, b::Float64)
    ns = length(x) - 1
    x0 = x[1]; H = (x[end] - x0) / ns
    t  = [(xi - x0) / H for xi in x]
    ta = (a - x0) / H; tb = (b - x0) / H
    la = _lagrange_basis_nodes(t, ta)
    lb = _lagrange_basis_nodes(t, tb)
    w = Vector{Float64}(undef, ns)
    acc = 0.0
    @inbounds for q in ns:-1:1
        acc += lb[q+1] - la[q+1]
        w[q] = (t[q+1] - t[q]) * acc / (tb - ta)
    end
    return w
end

"Nodal Lagrange weights at `z` over the non-uniform nodes `x`."
function _nodal_weights_nodes(x::AbstractVector{Float64}, z::Float64)
    n = length(x)
    x0 = x[1]; H = n > 1 ? (x[end] - x0) / (n - 1) : 1.0
    return _lagrange_basis_nodes([(xi - x0) / H for xi in x], (z - x0) / H)
end

# For every child storage cell (:avg) or face (:nodal) along z, the parent
# storage index where the stencil starts and the weights. `pc`, `cc` are
# the parent / child face coordinates (storage-indexed, length N + 2H + 1).
function _build_coord_plan(kind::Symbol, order::Int, pc::Vector{Float64},
                           cc::Vector{Float64})
    ns = max(order, 1)
    npc = length(pc) - 1                 # parent storage cells
    ncc = length(cc) - 1                 # child storage cells (= faces in a Face array)
    start = Vector{Int}(undef, ncc); W = Matrix{Float64}(undef, ns, ncc); cell = zeros(Int, ncc)
    @inbounds for ic in 1:ncc
        if kind === :avg
            a = cc[ic]; b = cc[ic+1]
            I  = clamp(searchsortedlast(pc, (a + b) / 2), 1, npc)
            s0 = clamp(I - ((ns - 1) ÷ 2), 1, max(1, npc - ns + 1))
            start[ic] = s0; cell[ic] = I
            W[:, ic] .= _cons_weights_nodes(view(pc, s0:s0+ns), a, b)
        else
            z  = cc[ic]
            F  = clamp(searchsortedlast(pc, z), 1, npc)
            s0 = clamp(F - ((ns - 1) ÷ 2), 1, max(1, npc - ns + 1))
            start[ic] = s0
            W[:, ic] .= _nodal_weights_nodes(view(pc, s0:s0+ns-1), z)
        end
    end
    return _Plan1D(start, W, cell)
end

"""
    _zplan(nest, kind, order, measure)

The vertical 1-D plan. On a uniform parent in the volume measure it is
the N-nest-a integer plan (`_plan`); otherwise the coordinate
plan of the measure: `:avg` in the measure coordinate (height, or the
reference mass `μ`), `:nodal` (the point value `w`) in height.
"""
function _zplan(nest::NestedGrid, kind::Symbol, order::Int, measure)
    _uses_integer_z(nest, measure) && return _plan(nest, kind, 3, order)
    m = _resolve(nest, measure)
    key = (kind, order)
    p = get(m.plans, key, nothing)
    p === nothing || return p
    p = kind === :avg ? _build_coord_plan(:avg, order, m.pc, m.cc) :
                        _build_coord_plan(:nodal, order, m.ph, m.ch)
    m.plans[key] = p
    return p
end

# Which 1-D rule applies in each direction, per staggered location:
# `:avg` = the field is a cell/face-area average there (conservative rule);
# `:nodal` = the field is a point value there (Lagrange rule).
_dim_kinds(::CellCenter) = (:avg,   :avg,   :avg)
_dim_kinds(::FaceX)      = (:nodal, :avg,   :avg)
_dim_kinds(::FaceY)      = (:avg,   :nodal, :avg)
_dim_kinds(::FaceZ)      = (:avg,   :avg,   :nodal)

# =========================================================================
# Prolongation: parent -> child
# =========================================================================

# One box of child storage points `ir × jr × kr`, written to `dst` at the
# local indices `(i, j, k) .- off` (`off = (0, 0, 0)` for a whole child
# array; a slab of a halo-only `_ShellArray` otherwise). Owner computes
# (KA-f, #297): one launch, each child point interpolates from its parent
# stencil. A function barrier with concrete argument types: no allocation,
# no dynamic dispatch.
function _interp_box!(dst::Array{T,3}, off::NTuple{3,Int}, pd::AbstractArray{<:Real,3},
                      px::_Plan1D, py::_Plan1D, pz::_Plan1D,
                      ir::UnitRange{Int}, jr::UnitRange{Int}, kr::UnitRange{Int}) where {T}
    launch!(Backends.backend(dst), _interp_body!, (ir, jr, kr), dst, off, pd, px, py, pz)
    return dst
end

@inline function _interp_body!(i, j, k, dst::AbstractArray{T}, off, pd, px::_Plan1D,
                               py::_Plan1D, pz::_Plan1D) where {T}
    nx = _stencil(px); ny = _stencil(py); nz = _stencil(pz)
    Wx = px.w; Wy = py.w; Wz = pz.w
    @inbounds begin
        bk = pz.start[k]; bj = py.start[j]; bi = px.start[i]
        s = 0.0
        for c in 1:nz
            wc = Wz[c, k]; kp = bk + c - 1
            for b in 1:ny
                wbc = wc * Wy[b, j]; jp = bj + b - 1
                for a in 1:nx
                    s += wbc * Wx[a, i] * pd[bi + a - 1, jp, kp]
                end
            end
        end
        dst[i - off[1], j - off[2], k - off[3]] = T(s)
    end
    return nothing
end

# The storage boxes of a region of a child field at `loc`: the whole array
# (`:all`), or the six non-overlapping halo slabs, `H` deep, one deeper on
# the low side of the field's own staggered axis for `:halo_interface` (the
# boundary-normal interior face plane of the west/south/bottom faces).
function _region_boxes(cm::RegularGrid, loc::Location, region::Symbol)
    n = (cm.Nx + 2cm.Hx, cm.Ny + 2cm.Hy, cm.Nz + 2cm.Hz)
    region === :all && return ((1:n[1], 1:n[2], 1:n[3]),)
    ax = _stagger_axis(loc)
    lo = ntuple(d -> (cm.Hx, cm.Hy, cm.Hz)[d] + (region === :halo_interface && d == ax), 3)
    return _shell_boxes(n, lo, (cm.Hx, cm.Hy, cm.Hz))
end

_stagger_axis(::CellCenter) = 0
_stagger_axis(::FaceX) = 1
_stagger_axis(::FaceY) = 2
_stagger_axis(::FaceZ) = 3

function _interp_child!(cf::Field, pf::Field, nest::NestedGrid;
                        order::Int=3, region::Symbol=:halo, measure=nothing,
                        limiter::Symbol=:none)
    cm = nest.child_mesh
    size(cf.data) == totalsize(cm) ||
        error("_interp_child!: child field is not on the nest's child mesh")
    size(pf.data) == totalsize(nest.parent_mesh) ||
        error("_interp_child!: parent field is not on the nest's parent mesh")
    region in (:all, :halo, :halo_interface) ||
        throw(ArgumentError("_interp_child!: region must be :all, :halo or :halo_interface, got :$region"))
    loc = location(cf)
    kinds = _dim_kinds(loc)
    px = _plan(nest, kinds[1], 1, order)
    py = _plan(nest, kinds[2], 2, order)
    pz = _zplan(nest, kinds[3], order, measure)
    _interp_into!(cf.data, pf.data, px, py, pz, cm, loc, region)
    limiter in _LIMITERS ||
        throw(ArgumentError("prolongation limiter must be one of $(_LIMITERS), got :$limiter"))
    if limiter !== :none && order > 1
        boxes = _dst_boxes(cf.data, cm, loc, region)
        if loc isa CellCenter
            _limit_cells!(boxes, pf.data, px, py, pz, nest, limiter === :monotone)
        else
            limiter === :monotone ||
                throw(ArgumentError("a velocity takes limiter = :none or :monotone, not :$limiter"))
            _clip_faces!(boxes, pf.data, px, py, pz)
        end
    end
    return cf
end

function _interp_into!(cd::Array, pd, px, py, pz, cm, loc, region)
    for (ir, jr, kr) in _region_boxes(cm, loc, region)
        _interp_box!(cd, (0, 0, 0), pd, px, py, pz, ir, jr, kr)
    end
    return cd
end

# (array, offset, storage box) of every box a prolongation wrote.
_dst_boxes(cd::Array, cm, loc, region) =
    map(b -> (cd, (0, 0, 0), b), _region_boxes(cm, loc, region))
_dst_boxes(cd::_ShellArray, cm, loc, region) = _shell_slabs(cd)

# -------------------------------------------------------------------------
# The prolongation limiter (#329, item 9)
# -------------------------------------------------------------------------
#
# Scalars: a conservative scaling of the children of each parent cell P
# about the parent value φ̄_P,
#
#     φ_c ← φ̄_P + θ_P (φ_c − φ̄_P),   0 ≤ θ_P ≤ 1,
#
# with the same θ_P for every child of P. The map is affine, so the
# (volume- or reference-mass-weighted) mean of the children, which the
# primitive-function reconstruction makes equal to φ̄_P, is unchanged:
# conservation is kept wherever all children of P are prolongated (the
# child interior of region = :all). Two bounds:
#
#   * :positive (Zhang & Shu 2010, JCP 229, 3091, §2.3): θ_P makes the
#     smallest child ≥ 0 where φ̄_P ≥ 0. Inactive away from zero, so a smooth
#     field keeps the full order; the default for scalars (qv, tracers).
#   * :monotone (Barth & Jespersen 1989, AIAA 89-0366): θ_P keeps every
#     child within the range [m_P, M_P] of the parent values over P and its
#     26 neighbours. No new extrema at a front; it also clips smooth extrema
#     to second order, hence an option.
#
# θ_P < 1 only where a child violates the bound; everywhere else the values
# are left bit for bit.

const _LIMITERS = (:none, :positive, :monotone)

function _lim_scratch(nest::NestedGrid, dims::NTuple{3,Int})
    return get!(nest.limscratch, dims) do
        (Array{Float64,3}(undef, dims), Array{Float64,3}(undef, dims),
         Array{Float64,3}(undef, dims))
    end
end

function _limit_cells!(boxes, pd::AbstractArray{<:Real,3}, px::_Plan1D, py::_Plan1D,
                       pz::_Plan1D, nest::NestedGrid, monotone::Bool)
    c0 = (first(px.cell), first(py.cell), first(pz.cell)) .- 1
    dims = (last(px.cell), last(py.cell), last(pz.cell)) .- c0
    cmax, cmin, θ = _lim_scratch(nest, dims)
    fill!(cmax, -Inf); fill!(cmin, Inf)
    b = Backends.backend(cmax)
    # Owner computes (KA-f, #297): per box, each parent cell the box reaches
    # gathers the extrema of its children in the box (formerly a scatter of
    # every child into its parent); the boxes run one after another.
    for (d, off, (ir, jr, kr)) in boxes
        (isempty(ir) || isempty(jr) || isempty(kr)) && continue
        pr = (px.cell[first(ir)] - c0[1] : px.cell[last(ir)] - c0[1],
              py.cell[first(jr)] - c0[2] : py.cell[last(jr)] - c0[2],
              pz.cell[first(kr)] - c0[3] : pz.cell[last(kr)] - c0[3])
        launch!(b, _extrema_body!, pr, cmax, cmin, d, off, ir, jr, kr, px, py, pz, c0)
    end
    launch!(b, _factor_body!, (1:dims[1], 1:dims[2], 1:dims[3]), θ, cmax, cmin, pd, c0, monotone)
    for (d, off, (ir, jr, kr)) in boxes
        launch!(Backends.backend(d), _scale_body!, (ir, jr, kr), d, off, θ, pd, px.cell,
                py.cell, pz.cell, c0)
    end
    return nothing
end

# The child indices in `r` whose parent cell is `P`.
@inline function _children(p::_Plan1D, P::Int, r::UnitRange{Int})
    q = P - p.c1 + 1
    @inbounds return 1 <= q <= length(p.clo) ? intersect(r, p.clo[q]:p.chi[q]) : (1:0)
end

# Fold the children of parent cell (I, J, K) in one box into its running
# max/min. max and min return one of their operands, so the order of the
# children does not change the result.
@inline function _extrema_body!(I, J, K, cmax, cmin, d, off, ir, jr, kr, px, py, pz, c0)
    ks = _children(pz, K + c0[3], kr); js = _children(py, J + c0[2], jr)
    is = _children(px, I + c0[1], ir)
    @inbounds begin
        mx = cmax[I, J, K]; mn = cmin[I, J, K]
        for k in ks, j in js, i in is
            v = d[i - off[1], j - off[2], k - off[3]]
            mx = max(mx, v); mn = min(mn, v)
        end
        cmax[I, J, K] = mx; cmin[I, J, K] = mn
    end
    return nothing
end

@inline function _factor_body!(I, J, K, θ, cmax, cmin, pd, c0, monotone::Bool)
    n1, n2, n3 = size(pd)
    @inbounds begin
        θ[I, J, K] = 1.0
        cmax[I, J, K] == -Inf && return nothing      # no child of this cell written
        P = (I + c0[1], J + c0[2], K + c0[3])
        φ̄ = Float64(pd[P[1], P[2], P[3]])
        t = 1.0
        if monotone
            m = φ̄; M = φ̄
            for c in max(P[3] - 1, 1):min(P[3] + 1, n3), b in max(P[2] - 1, 1):min(P[2] + 1, n2),
                a in max(P[1] - 1, 1):min(P[1] + 1, n1)
                v = Float64(pd[a, b, c]); m = min(m, v); M = max(M, v)
            end
            cmax[I, J, K] > M && (t = min(t, (M - φ̄) / (cmax[I, J, K] - φ̄)))
            cmin[I, J, K] < m && (t = min(t, (m - φ̄) / (cmin[I, J, K] - φ̄)))
        elseif φ̄ >= 0 && cmin[I, J, K] < 0
            t = φ̄ / (φ̄ - cmin[I, J, K])
        end
        θ[I, J, K] = max(t, 0.0)
    end
    return nothing
end

@inline function _scale_body!(i, j, k, d::AbstractArray{T}, off, θ, pd, cx, cy, cz, c0) where {T}
    @inbounds begin
        Ip = cx[i]; Jp = cy[j]; Kp = cz[k]
        t = θ[Ip - c0[1], Jp - c0[2], Kp - c0[3]]
        t < 1.0 || return nothing
        φ̄ = Float64(pd[Ip, Jp, Kp])
        q = (i - off[1], j - off[2], k - off[3])
        d[q[1], q[2], q[3]] = T(φ̄ + t * (d[q[1], q[2], q[3]] - φ̄))
    end
    return nothing
end

# Velocities (optional): clip each prolongated value to the range of the
# parent values its stencil reads. Monotone, but neither conservative nor
# flux-preserving, hence off by default.
function _clip_faces!(boxes, pd::AbstractArray{<:Real,3}, px::_Plan1D, py::_Plan1D,
                      pz::_Plan1D)
    for (d, off, (ir, jr, kr)) in boxes
        launch!(Backends.backend(d), _clip_body!, (ir, jr, kr), d, off, pd, px, py, pz)
    end
    return nothing
end

@inline function _clip_body!(i, j, k, d::AbstractArray{T}, off, pd, px::_Plan1D, py::_Plan1D,
                             pz::_Plan1D) where {T}
    nx = _stencil(px); ny = _stencil(py); nz = _stencil(pz)
    @inbounds begin
        bi = px.start[i]; bj = py.start[j]; bk = pz.start[k]
        lo = Inf; hi = -Inf
        for c in 0:nz-1, b in 0:ny-1, a in 0:nx-1
            v = Float64(pd[bi + a, bj + b, bk + c]); lo = min(lo, v); hi = max(hi, v)
        end
        q = (i - off[1], j - off[2], k - off[3])
        d[q[1], q[2], q[3]] = T(clamp(Float64(d[q[1], q[2], q[3]]), lo, hi))
    end
    return nothing
end

# A halo-only array stores exactly the halo slabs (and the interface plane):
# prolongate each slab in place. The interior is not stored.
function _interp_into!(cd::_ShellArray, pd, px, py, pz, cm, loc, region)
    region === :all &&
        throw(ArgumentError("a halo-only (storage = :shell) nest state holds no child " *
                            "interior; prolongate region = :all into a storage = :full state"))
    for (slab, off, (ir, jr, kr)) in _shell_slabs(cd)
        _interp_box!(slab, off, pd, px, py, pz, ir, jr, kr)
    end
    return cd
end

"""
    interpolate_to_nest_boundary!(child_field, parent_field, nest; order=3,
                                  impose_interface=false)

Fill *every* halo cell of `child_field` (faces, edges and corners) by
conservative order-`order` interpolation of `parent_field`. The rule used
in each direction follows the field's [`Location`](@ref):

  * cell-centred scalars — conservative in all three directions, so the
    child sub-cell averages sum back to the parent average exactly;
  * `FaceX`/`FaceY`/`FaceZ` velocities — *nodal* (point-value) in their
    own direction and conservative in the two tangential directions. That
    is what preserves the **flux**: the summed fine-face fluxes through a
    parent face equal the parent flux, and a child face coincident with a
    parent face reproduces the parent value exactly.

`parent_field`'s halos must already be filled. With
`impose_interface=true` the child's boundary-normal interior face plane
(e.g. `i = Hx+1` for `FaceX`) is also overwritten with the parent value,
i.e. a Dirichlet inflow face.

On a z-stretched parent the vertical stencils use the parent's actual
face heights. `measure` (a [`NestMeasure`](@ref)) selects the vertical
measure; with the mass measure of a sound-proof pair
([`nest_measure`](@ref)) the conservative directions conserve `∫ W φ dV`
(`W = ρ̄` or `ρ̄θ̄`) instead of `∫ φ dV`. The default is the volume measure.

`limiter` (#329): the 3rd- and 5th-order reconstructions overshoot at a
sharp front — a negative `qv` or tracer next to a dry or clean parent
cell. For a cell-centred field the children of every parent cell are
scaled about the parent value, which keeps the prolongation conservative
(the children's mean is the parent value):

  * `:positive` (Zhang & Shu 2010, *J. Comput. Phys.* 229, 3091) — no
    child below zero where the parent cell is non-negative; inactive away
    from zero, so a smooth positive field keeps the full order. For
    quantities that cannot be negative (`qv`, tracer concentrations): a
    signed field crossing zero inside a parent cell would be clipped;
  * `:monotone` (Barth & Jespersen 1989, AIAA paper 89-0366) — no child
    outside the range of its parent cell and the 26 neighbours: no new
    extrema at a front, but smooth extrema are clipped to second order;
  * `:none` (default of this low-level operator; the field set of a
    [`NestedRun`](@ref) limits `qv` and the tracers with `:positive`, see
    [`interpolate_parent_state!`](@ref)).

A velocity takes `:none` or `:monotone`, a clip to the range of the parent
values its stencil reads (monotone, but not flux-preserving).
Where no child overshoots, the values are unchanged bit for bit.
"""
interpolate_to_nest_boundary!(cf::Field, pf::Field, nest::NestedGrid;
                              order::Int=3, impose_interface::Bool=false,
                              measure=nothing, limiter::Symbol=:none) =
    _interp_child!(cf, pf, nest; order, measure, limiter,
                   region = impose_interface ? :halo_interface : :halo)

"""
    interpolate_parent_to_child!(child_field, parent_field, nest; order=3)

As [`interpolate_to_nest_boundary!`](@ref) but fills the child field
*everywhere* (interior included). This is the AMR prolongation operator —
used to initialize newly refined blocks and to build the reference state
for [`apply_nest_sponge!`](@ref). Accepts the same `measure` and
`limiter` keywords.
"""
interpolate_parent_to_child!(cf::Field, pf::Field, nest::NestedGrid; order::Int=3,
                             measure=nothing, limiter::Symbol=:none) =
    _interp_child!(cf, pf, nest; order, measure, limiter, region=:all)

# =========================================================================
# Restriction: child -> parent
# =========================================================================

"""
    restrict_to_parent!(parent_field, child_field, nest; measure=nothing, solid=nothing)
    restrict_to_parent!(parent_face_field, child_face_field, nest;
                        include_boundary=false, measure=nothing, solid=nothing)

Conservative restriction.

  * `CellCenter` — volume-weighted average of the `rx·ry·rz` child cells
    inside each covered parent cell. Since `V_parent = rx·ry·rz · V_child`
    in the (uniform) overlap, `∫` over the overlap is preserved to machine
    precision.
  * `FaceX`/`FaceY`/`FaceZ` — area-weighted average of the `r·r` child
    faces covering each aligned parent face, i.e. flux-preserving. By
    default only faces *strictly inside* the nest are overwritten; the two
    interface planes are owned by the flux correction
    ([`apply_flux_correction!`](@ref)). Pass `include_boundary=true` to
    overwrite them as well.

`measure` — with the mass measure of a sound-proof pair
([`nest_measure`](@ref)) the averages are weighted by the child cell
mass `W Δz` (`W = ρ̄` anelastic, `ρ̄θ̄` pseudo-incompressible): a scalar
becomes `Σ W φ ΔV / Σ W ΔV`, so `∫ W φ dV` is preserved, and a
horizontal-normal velocity (`u`, `v`) becomes `Σ W u A / Σ W A`, so the
parent face carries the child's summed mass flux. `w` faces are
horizontal planes at one height, where `W` is common to all `r·r`
children and cancels: the plain area mean is already the mass-flux mean.

`solid` — the child's IBM [`SolidMask`](@ref) (or a storage-sized
Boolean array, `true` = solid, such as the Berlin `solid` field) makes
the restriction **solid-aware** (Hellsten et al. 2021, GMD 14, 3185,
§3.5): only fluid child cells, and only fluid child faces (both
neighbouring cells fluid), enter the average, so no parent value is ever
averaged from the zero velocity inside a building. A parent cell or face
with no fluid child keeps its value. The fluid-weighted integral
`Σ_fluid W φ ΔV` is what is preserved.
"""
function restrict_to_parent!(pf::Field{CellCenter}, cf::Field{CellCenter},
                             nest::NestedGrid; measure=nothing, solid=nothing)
    χ = _fluid(nest, solid)
    ωz = _zweights(measure)
    (χ === nothing && ωz === nothing) && return _restrict_cc_volume!(pf, cf, nest)
    return _restrict_cc_weighted!(pf, cf, nest, ωz === nothing ? _ones_z(nest) : ωz, χ)
end

restrict_to_parent!(pf::Field{L}, cf::Field{L}, nest::NestedGrid;
                    include_boundary::Bool=false, measure=nothing,
                    solid=nothing) where {L<:Union{FaceX,FaceY,FaceZ}} =
    _restrict_face_any!(pf, cf, nest, _axis_of(L), include_boundary,
                        _zweights(measure), _fluid(nest, solid))

# z weights of a restriction: `nothing` for the volume measure (the child
# layers of a parent layer are equally thick), the child cell masses W Δz
# for the mass measure.
_zweights(::Nothing) = nothing
_zweights(m::NestMeasure) = m.kind === :volume ? nothing : m.cw
_ones_z(nest::NestedGrid) = ones(Float64, nest.child_mesh.Nz + 2nest.child_mesh.Hz)

# The restrictions are owner-computes launches (KA-f, #297): each covered
# parent cell (or face) gathers its children, in the order the loops used.

# The covered parent cells, as parent storage ranges.
_covered_cells(nest::NestedGrid) =
    (nest.parent_mesh.Hx + nest.ix : nest.parent_mesh.Hx + nest.ix + nest.nx - 1,
     nest.parent_mesh.Hy + nest.iy : nest.parent_mesh.Hy + nest.iy + nest.ny - 1,
     nest.parent_mesh.Hz + nest.iz : nest.parent_mesh.Hz + nest.iz + nest.nz - 1)

function _restrict_cc_volume!(pf::Field, cf::Field, nest::NestedGrid)
    pm = nest.parent_mesh; cm = nest.child_mesh
    r = (nest.rx, nest.ry, nest.rz)
    o = (pm.Hx + nest.ix - 1, pm.Hy + nest.iy - 1, pm.Hz + nest.iz - 1)
    launch!(Backends.backend(pf.data), _restrict_cc_body!, _covered_cells(nest), pf.data, cf.data,
            r, o, (cm.Hx, cm.Hy, cm.Hz), 1 / (r[1] * r[2] * r[3]))
    return pf
end

@inline function _restrict_cc_body!(ip, jp, kp, pd, cd, r, o, Hc, inv)
    I = ip - o[1]; J = jp - o[2]; K = kp - o[3]
    s = 0.0
    @inbounds for kc in 1:r[3], jc in 1:r[2], ic in 1:r[1]
        s += cd[Hc[1] + (I-1)*r[1] + ic, Hc[2] + (J-1)*r[2] + jc, Hc[3] + (K-1)*r[3] + kc]
    end
    @inbounds pd[ip, jp, kp] = s * inv
    return nothing
end

function _restrict_cc_weighted!(pf::Field, cf::Field, nest::NestedGrid,
                                ωz::Vector{Float64}, χ)
    pm = nest.parent_mesh; cm = nest.child_mesh
    r = (nest.rx, nest.ry, nest.rz)
    o = (pm.Hx + nest.ix - 1, pm.Hy + nest.iy - 1, pm.Hz + nest.iz - 1)
    launch!(Backends.backend(pf.data), _restrict_ccw_body!, _covered_cells(nest), pf.data, cf.data,
            r, o, (cm.Hx, cm.Hy, cm.Hz), ωz, χ)
    return pf
end

@inline function _restrict_ccw_body!(ip, jp, kp, pd, cd, r, o, Hc, ωz, χ)
    I = ip - o[1]; J = jp - o[2]; K = kp - o[3]
    num = 0.0; den = 0.0
    @inbounds for kc in 1:r[3]
        k = Hc[3] + (K-1)*r[3] + kc
        wz = ωz[k]
        for jc in 1:r[2], ic in 1:r[1]
            i = Hc[1] + (I-1)*r[1] + ic; j = Hc[2] + (J-1)*r[2] + jc
            (χ === nothing || χ[i, j, k]) || continue
            num += wz * cd[i, j, k]; den += wz
        end
    end
    den > 0 || return nothing           # no fluid child: keep the parent value
    @inbounds pd[ip, jp, kp] = num / den
    return nothing
end

function _restrict_face_any!(pf::Field, cf::Field, nest::NestedGrid, ax::Int,
                             include_boundary::Bool, ωz, χ)
    # w faces are horizontal: W is common to the children and cancels
    ωe = ax == 3 ? nothing : ωz
    (ωe === nothing && χ === nothing) &&
        return _restrict_face!(pf, cf, nest, ax, include_boundary)
    return _restrict_face_weighted!(pf, cf, nest, ax, include_boundary,
                                    ωe === nothing ? _ones_z(nest) : ωe, χ)
end

# The parent faces normal to `ax` that a face restriction writes (interface
# planes q = 0 and n included or not), as parent storage ranges.
function _restricted_faces(nest::NestedGrid, ax::Int, include_boundary::Bool)
    cr = _covered_cells(nest)
    nn = (nest.nx, nest.ny, nest.nz)
    q0 = include_boundary ? 0 : 1
    q1 = include_boundary ? nn[ax] : nn[ax] - 1
    p0 = first(cr[ax])                   # the parent index of face q = 0
    return ntuple(d -> d == ax ? (p0 + q0 : p0 + q1) : cr[d], 3)
end

# (q, A, B) of the parent face (ip, jp, kp) normal to `ax`: the face number
# along `ax` and the nest-local tangential indices.
@inline function _face_qab(ax::Int, ip, jp, kp, o)
    q = ax == 1 ? ip - o[1] - 1 : ax == 2 ? jp - o[2] - 1 : kp - o[3] - 1
    A = ax == 1 ? jp - o[2] : ip - o[1]
    B = ax == 3 ? jp - o[2] : kp - o[3]
    return q, A, B
end

function _restrict_face_weighted!(pf::Field, cf::Field, nest::NestedGrid, ax::Int,
                                  include_boundary::Bool, ωz::Vector{Float64}, χ)
    pm = nest.parent_mesh; cm = nest.child_mesh
    r  = (nest.rx, nest.ry, nest.rz)
    o = (pm.Hx + nest.ix - 1, pm.Hy + nest.iy - 1, pm.Hz + nest.iz - 1)
    launch!(Backends.backend(pf.data), _restrict_facew_body!,
            _restricted_faces(nest, ax, include_boundary), pf.data, cf.data, ax, r, o,
            (cm.Hx, cm.Hy, cm.Hz), ωz, χ)
    return pf
end

@inline function _restrict_facew_body!(ip, jp, kp, pd, cd, ax, r, o, Hc, ωz, χ)
    q, A, B = _face_qab(ax, ip, jp, kp, o)
    t1, t2 = _tang(ax)
    cq = Hc[ax] + q * r[ax] + 1
    num = 0.0; den = 0.0
    @inbounds for b in 1:r[t2], a in 1:r[t1]
        cidx = _pos(ax, t1, cq, Hc[t1] + (A-1)*r[t1] + a, Hc[t2] + (B-1)*r[t2] + b)
        if χ !== nothing
            bidx = Base.setindex(cidx, cidx[ax] - 1, ax)
            (χ[cidx[1], cidx[2], cidx[3]] && χ[bidx[1], bidx[2], bidx[3]]) || continue
        end
        w = ωz[cidx[3]]
        num += w * cd[cidx[1], cidx[2], cidx[3]]; den += w
    end
    den > 0 || return nothing           # no fluid child face: keep the parent value
    @inbounds pd[ip, jp, kp] = num / den
    return nothing
end

# The two tangential axes of a plane normal to `ax`, in increasing order, and
# the storage index with `q` on `ax` and `a`, `b` on them — a tuple, not a
# heap-allocated `[0, 0, 0]` (#329).
@inline _tang(ax::Int) = ax == 1 ? (2, 3) : ax == 2 ? (1, 3) : (1, 2)
@inline _pos(ax::Int, t1::Int, q::Int, a::Int, b::Int) =
    ax == 1 ? (q, a, b) : ax == 2 ? (a, q, b) : (a, b, q)

_axis_of(::Type{FaceX}) = 1
_axis_of(::Type{FaceY}) = 2
_axis_of(::Type{FaceZ}) = 3

function _restrict_face!(pf::Field, cf::Field, nest::NestedGrid, ax::Int,
                         include_boundary::Bool)
    pm = nest.parent_mesh; cm = nest.child_mesh
    r  = (nest.rx, nest.ry, nest.rz)
    t1, t2 = _tang(ax)
    o = (pm.Hx + nest.ix - 1, pm.Hy + nest.iy - 1, pm.Hz + nest.iz - 1)
    launch!(Backends.backend(pf.data), _restrict_face_body!,
            _restricted_faces(nest, ax, include_boundary), pf.data, cf.data, ax, r, o,
            (cm.Hx, cm.Hy, cm.Hz), 1 / (r[t1] * r[t2]))
    return pf
end

@inline function _restrict_face_body!(ip, jp, kp, pd, cd, ax, r, o, Hc, inv)
    q, A, B = _face_qab(ax, ip, jp, kp, o)
    t1, t2 = _tang(ax)
    cq = Hc[ax] + q * r[ax] + 1
    s = 0.0
    @inbounds for b in 1:r[t2], a in 1:r[t1]
        cidx = _pos(ax, t1, cq, Hc[t1] + (A-1)*r[t1] + a, Hc[t2] + (B-1)*r[t2] + b)
        s += cd[cidx[1], cidx[2], cidx[3]]
    end
    @inbounds pd[ip, jp, kp] = s * inv
    return nothing
end

# =========================================================================
# Flux mismatch + correction (refluxing)
# =========================================================================

"""
    NestFluxMismatch

Per-face-patch difference between the parent's coarse flux and the summed
fine child fluxes on the six nest-boundary planes, in *flux* units
(field value × area). Populated pairs depend on the field's
[`Location`](@ref): `FaceX` fills `west`/`east` (size `ny × nz`),
`FaceY` fills `south`/`north` (`nx × nz`), `FaceZ` fills `bottom`/`top`
(`nx × ny`). Unused planes are `0 × 0`.
"""
struct NestFluxMismatch{T}
    west::Matrix{T};   east::Matrix{T}
    south::Matrix{T};  north::Matrix{T}
    bottom::Matrix{T}; top::Matrix{T}
end

function NestFluxMismatch{T}() where {T}
    z = zeros(T, 0, 0)
    NestFluxMismatch{T}(copy(z), copy(z), copy(z), copy(z), copy(z), copy(z))
end

Base.:+(a::NestFluxMismatch{T}, b::NestFluxMismatch{T}) where {T} =
    NestFluxMismatch{T}(_addm(a.west, b.west), _addm(a.east, b.east),
                        _addm(a.south, b.south), _addm(a.north, b.north),
                        _addm(a.bottom, b.bottom), _addm(a.top, b.top))
_addm(a, b) = isempty(a) ? copy(b) : isempty(b) ? copy(a) : a .+ b

"Sum of `|mismatch|` over all six planes (flux units)."
mismatch_l1(m::NestFluxMismatch) =
    sum(x -> isempty(x) ? 0.0 : sum(abs, x),
        (m.west, m.east, m.south, m.north, m.bottom, m.top))

# Parent layer "thickness" in a measure: the height Δz (volume; the actual
# layer thickness of a stretched parent) or the reference mass Δμ = W Δz.
_pthick(nest::NestedGrid, ::Nothing, K::Integer) = _dzp(nest.parent_mesh, K)
function _pthick(nest::NestedGrid, m::NestMeasure, K::Integer)
    m.kind === :volume && return _dzp(nest.parent_mesh, K)
    P = nest.parent_mesh.Hz + K
    return m.pc[P+1] - m.pc[P]
end

# Mass (or thickness) of an *uncovered* parent cell at interior level K: the
# parent's own W_p Δz_p, which is what the parent simulation conserves (see
# NestMeasure). Used by the composite integral and the refluxing correction.
_pcell(nest::NestedGrid, ::Nothing, K::Integer) = _dzp(nest.parent_mesh, K)
_pcell(nest::NestedGrid, m::NestMeasure, K::Integer) =
    m.kind === :volume ? _dzp(nest.parent_mesh, K) : m.pw[nest.parent_mesh.Hz + K]

# Area of the parent interface face (A, B) of a plane normal to `ax`, in the
# measure (A, B index the two tangential directions in increasing order).
function _face_area(nest::NestedGrid, m, ax::Int, B::Integer)
    pm = nest.parent_mesh
    ax == 3 && return pm.dx * pm.dy
    return (ax == 1 ? pm.dy : pm.dx) * _pthick(nest, m, nest.iz + B - 1)
end

"Total area of the six nest-interface planes."
function mismatch_area(nest::NestedGrid)
    pm = nest.parent_mesh
    hz = sum(_dzp(pm, nest.iz + K - 1) for K in 1:nest.nz)
    return 2 * (nest.ny * hz * pm.dy + nest.nx * hz * pm.dx +
                nest.nx * nest.ny * pm.dx * pm.dy)
end

"""
    compute_nest_flux_mismatch(parent_field, child_field, nest; measure=nothing)

`coarse flux − Σ fine fluxes` across each nest-boundary face patch, for a
face-located field (a velocity, or a time-accumulated flux register
`Φ = ∫F dt`). Returns a [`NestFluxMismatch`](@ref).

A nonzero value is exactly the amount by which the two grids disagree
about the transport through the interface — i.e. the leak that
[`apply_flux_correction!`](@ref) removes. Face areas use the parent's
actual layer thickness (stretched parents). With the mass `measure` of a
sound-proof pair, a velocity's fine flux is the `W`-area-weighted one
and the mismatch is in mass-flux units (pass the same `measure` to
[`apply_flux_correction!`](@ref)); leave it `nothing` for flux registers
that already carry `W` (refluxing).
"""
compute_nest_flux_mismatch(pf::Field{L}, cf::Field{L}, nest::NestedGrid;
                           measure=nothing) where {L<:Union{FaceX,FaceY,FaceZ}} =
    _face_mismatch(pf, cf, nest, _axis_of(L), measure)

function _face_mismatch(pf::Field, cf::Field, nest::NestedGrid, ax::Int, measure=nothing)
    T  = eltype(pf.data)
    nn = (nest.nx, nest.ny, nest.nz)
    t1, t2 = _tang(ax)
    lowm  = zeros(T, nn[t1], nn[t2])
    highm = zeros(T, nn[t1], nn[t2])
    _plane_mismatch!(lowm, pf, cf, nest, ax, 0, measure)
    _plane_mismatch!(highm, pf, cf, nest, ax, nn[ax], measure)
    e = zeros(T, 0, 0)
    ax == 1 && return NestFluxMismatch{T}(lowm, highm, copy(e), copy(e), copy(e), copy(e))
    ax == 2 && return NestFluxMismatch{T}(copy(e), copy(e), lowm, highm, copy(e), copy(e))
    return NestFluxMismatch{T}(copy(e), copy(e), copy(e), copy(e), lowm, highm)
end

# coarse − Σ fine (flux units) on the interface plane `q` (0: low, n: high)
# normal to `ax`, written into `M` (a launch over the plane, KA-f #297);
# with `M === nothing` the plane's Σ |coarse − fine| is returned instead
# and nothing is stored (a serial sum: a diagnostic). One kernel
# (`_mismatch_at`) for the mismatch, the interface imbalance and the face
# flux correction.
function _plane_mismatch!(M, pf::Field, cf::Field, nest::NestedGrid, ax::Int, q::Int,
                          measure=nothing)
    ωz = ax == 3 ? nothing : _zweights(measure)
    # one branch per weight type, so `ωz === nothing` is known in the kernel
    return ωz === nothing ?
        _plane_mismatch!(M, pf.data, cf.data, _plane_geom(nest, ax, q, measure, nothing)) :
        _plane_mismatch!(M, pf.data, cf.data, _plane_geom(nest, ax, q, measure, ωz))
end

function _plane_mismatch!(M, pd, cd, g)
    if M !== nothing
        launch!(Backends.backend(M), _mismatch_body!, (1:g.n1, 1:g.n2), M, pd, cd, g)
        return 0.0
    end
    l1 = 0.0
    for B in 1:g.n2, A in 1:g.n1
        l1 += abs(_mismatch_at(pd, cd, g, A, B)[1])
    end
    return l1
end

@inline function _mismatch_body!(A, B, M, pd, cd, g)
    @inbounds M[A, B] = _mismatch_at(pd, cd, g, A, B)[1]
    return nothing
end

# The geometry of the interface plane q normal to `ax`, computed once per
# plane; `ωz` the z weights (`nothing`: the plain area mean).
function _plane_geom(nest::NestedGrid, ax::Int, q::Int, measure, ωz)
    pm = nest.parent_mesh; cm = nest.child_mesh
    r  = (nest.rx, nest.ry, nest.rz)
    nn = (nest.nx, nest.ny, nest.nz)
    oo = (nest.ix, nest.iy, nest.iz)
    Hp = (pm.Hx, pm.Hy, pm.Hz); Hc = (cm.Hx, cm.Hy, cm.Hz)
    t1, t2 = _tang(ax)
    return (; nest, measure, ax, t1, r1 = r[t1], r2 = r[t2], n1 = nn[t1], n2 = nn[t2],
            ωz, inv = 1 / (r[t1] * r[t2]),
            pq = Hp[ax] + oo[ax] + q, cq = Hc[ax] + q * r[ax] + 1,
            p1 = Hp[t1] + oo[t1] - 1, p2 = Hp[t2] + oo[t2] - 1, c1 = Hc[t1], c2 = Hc[t2])
end

# (coarse − Σ fine)·A_p at the interface face (A, B) of the plane `g`, and
# the parent storage index of that face.
@inline function _mismatch_at(pd, cd, g, A::Int, B::Int)
    ax = g.ax; t1 = g.t1; ωz = g.ωz
    pidx = _pos(ax, t1, g.pq, g.p1 + A, g.p2 + B)
    Ap = _face_area(g.nest, g.measure, ax, B)
    s = 0.0; den = 0.0
    @inbounds for b in 1:g.r2, a in 1:g.r1
        cidx = _pos(ax, t1, g.cq, g.c1 + (A-1)*g.r1 + a, g.c2 + (B-1)*g.r2 + b)
        if ωz === nothing
            s += cd[cidx[1], cidx[2], cidx[3]]
        else
            w = ωz[cidx[3]]
            s += w * cd[cidx[1], cidx[2], cidx[3]]; den += w
        end
    end
    fine = ωz === nothing ? s * g.inv : s / den
    @inbounds return (pd[pidx[1], pidx[2], pidx[3]] - fine) * Ap, pidx
end

"""
    apply_flux_correction!(parent_field, mismatch, nest; measure=nothing)          # face field
    apply_flux_correction!(parent_field, mismatch, nest; periodic, measure=nothing) # cell-centred

Remove the coarse-fine flux inconsistency recorded in `mismatch`.

  * **face-located `parent_field`** — replace the parent's interface face
    value with the summed fine flux (`u_p -= δΦ / A_p`). The parent's
    transport through the interface then matches the child's exactly.
  * **cell-centred `parent_field`** (the Berger-Colella *refluxing* step,
    with `mismatch` built from time-accumulated flux registers) — adjust
    the parent cells immediately *outside* the nest:
    `θ ± δΦ·A / V_p`. This is what makes the composite integral of a
    conserved scalar exactly invariant across a nested step. The parent's
    boundary conditions decide where the cell outside a child face that
    lies on the parent's own boundary is: along a periodic axis of the
    parent the index wraps; along a non-periodic one there is no parent
    cell beyond the face (the flux there is the parent's boundary
    condition's) and the correction of that plane is dropped (#329: it
    used to wrap on every axis, so a child touching the parent's top put
    its top-plane correction into the parent's lowest layer).
    `periodic = (px, py, pz)`, the parent's periodic axes, is
    **required** (#349: it defaulted to `(true, true, true)`, which is
    wrong for any parent with a wall or an open face and was silently
    taken by a caller that forgot it). Pass
    `periodic = nest_periodic_axes(parent_sim)` to read them from the
    parent's boundary conditions, as the two-way feedback,
    [`reflux_scalars!`](@ref) and the AMR refluxing do.

`A_p` and `V_p` use the parent's actual layer thickness (stretched
parents). With the mass `measure` of a sound-proof pair they are the
reference *mass* of the face and cell (`W Δz`; for the corrected cells,
which lie outside the nest, the parent's own `W_p Δz_p`, see
[`NestMeasure`](@ref)): a face correction then
matches the parent's mass flux to the child's, and refluxing with
mass-flux registers (`Φ = ∫W F dt`) preserves the composite `∫ W θ dV`.
Use the `measure` the mismatch was computed with (face fields).

Without this correction two-way nesting leaks: the parent kept the coarse
flux while the child used the fine ones.
"""
function apply_flux_correction!(pf::Field{L}, m::NestFluxMismatch,
                                nest::NestedGrid; measure=nothing) where {L<:Union{FaceX,FaceY,FaceZ}}
    ax = _axis_of(L)
    t1, t2 = _tang(ax)
    lowm, highm = ax == 1 ? (m.west, m.east) : ax == 2 ? (m.south, m.north) :
                                               (m.bottom, m.top)
    (isempty(lowm) || isempty(highm)) && return pf
    nn = (nest.nx, nest.ny, nest.nz)
    pm = nest.parent_mesh
    o = (pm.Hx + nest.ix - 1, pm.Hy + nest.iy - 1, pm.Hz + nest.iz - 1)
    for (q, M) in ((0, lowm), (nn[ax], highm))
        launch!(Backends.backend(pf.data), _face_correct_body!, (1:nn[t1], 1:nn[t2]), pf.data, M,
                nest, ax, o[ax] + 1 + q, o, measure)
    end
    return pf
end

# u_p -= δΦ / A_p at the parent face (A, B) of the plane at parent index pq.
@inline function _face_correct_body!(A, B, pd, M, nest, ax, pq, o, measure)
    t1, t2 = _tang(ax)
    pidx = _pos(ax, t1, pq, o[t1] + A, o[t2] + B)
    @inbounds pd[pidx[1], pidx[2], pidx[3]] -= M[A, B] / _face_area(nest, measure, ax, B)
    return nothing
end

function apply_flux_correction!(pf::Field{CellCenter}, m::NestFluxMismatch,
                                nest::NestedGrid; measure=nothing,
                                periodic::NTuple{3,Bool})
    pm = nest.parent_mesh
    Hp = (pm.Hx, pm.Hy, pm.Hz)
    nn = (nest.nx, nest.ny, nest.nz); oo = (nest.ix, nest.iy, nest.iz)
    planes = ((1, m.west, m.east), (2, m.south, m.north), (3, m.bottom, m.top))
    for (ax, lowm, highm) in planes
        (isempty(lowm) || isempty(highm)) && continue
        t1, t2 = _tang(ax)
        # cell just outside the low face:  Δθ = +δΦ·A / V
        # cell just outside the high face: Δθ = -δΦ·A / V
        for (cell, M, sgn) in ((oo[ax] - 1, lowm, +1.0),
                               (oo[ax] + nn[ax], highm, -1.0))
            # the parent cell beyond the face: wrapped on a periodic axis of
            # the parent, none (0) where the face is the parent's own boundary
            c = _reflux_cell(pm, periodic[ax], cell, ax)
            c == 0 && continue
            launch!(Backends.backend(pf.data), _reflux_body!, (1:nn[t1], 1:nn[t2]), pf.data, M,
                    nest, ax, Hp[ax] + c, (Hp[t1] + oo[t1] - 1, Hp[t2] + oo[t2] - 1), sgn, measure)
        end
    end
    return pf
end

# The one BC-aware neighbour helper (#262) on the parent's axis `ax`,
# periodic or not.
_reflux_cell(pm, periodic::Bool, cell::Int, ax::Int) =
    periodic ? neighbor_index(pm, _AXIS_BCS[1], cell, ax; outside = :none) :
               neighbor_index(pm, _AXIS_BCS[2], cell, ax; outside = :none)
const _AXIS_BCS = (FieldBoundaryConditions(Periodic(), Periodic(), Periodic(), Periodic(),
                                           Periodic(), Periodic()),
                   FieldBoundaryConditions(NoFlux(), NoFlux(), NoFlux(), NoFlux(),
                                           NoFlux(), NoFlux()))

@inline function _reflux_body!(A, B, pd, M, nest, ax, pc, ot, sgn, measure)
    pm = nest.parent_mesh
    t1, _ = _tang(ax)
    idx = _pos(ax, t1, pc, ot[1] + A, ot[2] + B)
    K = idx[3] - pm.Hz
    Vp = pm.dx * pm.dy * _pcell(nest, measure, K)
    @inbounds pd[idx[1], idx[2], idx[3]] += sgn * M[A, B] / Vp
    return nothing
end

"""
    nest_periodic_axes(parent_sim_or_bcs) -> NTuple{3,Bool}

The periodic axes `(x, y, z)` of a nest parent, from its boundary
conditions (`bcs`): what [`apply_flux_correction!`](@ref) needs to place
the refluxing correction of a child face lying on the parent's boundary.
"""
nest_periodic_axes(b::FieldBoundaryConditions) =
    (b.west isa Periodic && b.east isa Periodic, b.south isa Periodic && b.north isa Periodic,
     b.bottom isa Periodic && b.top isa Periodic)
nest_periodic_axes(sim) = nest_periodic_axes(sim.bcs)
