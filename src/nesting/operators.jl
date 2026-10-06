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
    plan = Vector{Tuple{Int,Vector{Float64}}}(undef, Nc + 2Hc)
    @inbounds for ic in 1:(Nc + 2Hc)
        n = ic - Hc                    # child cell / face number (may be <=0)
        if kind === :avg
            q = n - 1
            I = off + fld(q, r)
            m = mod(q, r) + 1
            s0 = clamp(I - ((ns - 1) ÷ 2), lo, hi)
            plan[ic] = (Hp + s0, _cons_weights(ns, r, m, I - s0 + 1))
        else # :nodal — child face position in parent-face-number units
            qf = off + (n - 1) / r
            s0 = clamp(Int(floor(qf)) - ((ns - 1) ÷ 2), lo, hi)
            plan[ic] = (Hp + s0, _nodal_weights(ns, qf - s0))
        end
    end
    return plan
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

# Which 1-D rule applies in each direction, per staggered location:
# `:avg` = the field is a cell/face-area average there (conservative rule);
# `:nodal` = the field is a point value there (Lagrange rule).
_dim_kinds(::CellCenter) = (:avg,   :avg,   :avg)
_dim_kinds(::FaceX)      = (:nodal, :avg,   :avg)
_dim_kinds(::FaceY)      = (:avg,   :nodal, :avg)
_dim_kinds(::FaceZ)      = (:avg,   :avg,   :nodal)

_interface_plane(::CellCenter, cm, i, j, k) = false
_interface_plane(::FaceX, cm, i, j, k) = (i == cm.Hx + 1)
_interface_plane(::FaceY, cm, i, j, k) = (j == cm.Hy + 1)
_interface_plane(::FaceZ, cm, i, j, k) = (k == cm.Hz + 1)

# =========================================================================
# Prolongation: parent -> child
# =========================================================================

function _interp_child!(cf::Field, pf::Field, nest::NestedGrid;
                        order::Int=3, region::Symbol=:halo)
    cm = nest.child_mesh
    cf.mesh === cm || size(cf.data) == totalsize(cm) ||
        error("_interp_child!: child field is not on the nest's child mesh")
    size(pf.data) == totalsize(nest.parent_mesh) ||
        error("_interp_child!: parent field is not on the nest's parent mesh")
    loc = location(cf)
    kinds = _dim_kinds(loc)
    px = _plan(nest, kinds[1], 1, order)
    py = _plan(nest, kinds[2], 2, order)
    pz = _plan(nest, kinds[3], 3, order)
    cd = cf.data; pd = pf.data
    ii, jj, kk = interior(cm)
    T = eltype(cd)
    all_region = region === :all
    iface = region === :halo_interface
    @inbounds for k in axes(cd, 3)
        bk, wk = pz[k]
        kin = k in kk
        for j in axes(cd, 2)
            bj, wj = py[j]
            jin = kin && (j in jj)
            for i in axes(cd, 1)
                if !all_region && jin && (i in ii)
                    (iface && _interface_plane(loc, cm, i, j, k)) || continue
                end
                bi, wi = px[i]
                s = 0.0
                for c in eachindex(wk)
                    wc = wk[c]; kp = bk + c - 1
                    for b in eachindex(wj)
                        wbc = wc * wj[b]; jp = bj + b - 1
                        for a in eachindex(wi)
                            s += wbc * wi[a] * pd[bi + a - 1, jp, kp]
                        end
                    end
                end
                cd[i, j, k] = T(s)
            end
        end
    end
    return cf
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
"""
interpolate_to_nest_boundary!(cf::Field, pf::Field, nest::NestedGrid;
                              order::Int=3, impose_interface::Bool=false) =
    _interp_child!(cf, pf, nest; order,
                   region = impose_interface ? :halo_interface : :halo)

"""
    interpolate_parent_to_child!(child_field, parent_field, nest; order=3)

As [`interpolate_to_nest_boundary!`](@ref) but fills the child field
*everywhere* (interior included). This is the AMR prolongation operator —
used to initialize newly refined blocks and to build the reference state
for [`apply_nest_sponge!`](@ref).
"""
interpolate_parent_to_child!(cf::Field, pf::Field, nest::NestedGrid; order::Int=3) =
    _interp_child!(cf, pf, nest; order, region=:all)

# =========================================================================
# Restriction: child -> parent
# =========================================================================

"""
    restrict_to_parent!(parent_field, child_field, nest)

Conservative restriction.

  * `CellCenter` — volume-weighted average of the `rx·ry·rz` child cells
    inside each covered parent cell. Since `V_parent = rx·ry·rz · V_child`
    on a uniform mesh, `∫` over the overlap is preserved to machine
    precision.
  * `FaceX`/`FaceY`/`FaceZ` — area-weighted average of the `r·r` child
    faces covering each aligned parent face, i.e. flux-preserving. By
    default only faces *strictly inside* the nest are overwritten; the two
    interface planes are owned by the flux correction
    ([`apply_flux_correction!`](@ref)). Pass `include_boundary=true` to
    overwrite them as well.
"""
function restrict_to_parent!(pf::Field{CellCenter}, cf::Field{CellCenter},
                             nest::NestedGrid)
    pm = nest.parent_mesh; cm = nest.child_mesh
    rx, ry, rz = nest.rx, nest.ry, nest.rz
    inv = 1 / (rx * ry * rz)
    pd = pf.data; cd = cf.data
    @inbounds for K in 1:nest.nz, J in 1:nest.ny, I in 1:nest.nx
        s = 0.0
        for kc in 1:rz, jc in 1:ry, ic in 1:rx
            s += cd[cm.Hx + (I-1)*rx + ic, cm.Hy + (J-1)*ry + jc, cm.Hz + (K-1)*rz + kc]
        end
        pd[pm.Hx + nest.ix + I - 1, pm.Hy + nest.iy + J - 1, pm.Hz + nest.iz + K - 1] =
            s * inv
    end
    return pf
end

restrict_to_parent!(pf::Field{L}, cf::Field{L}, nest::NestedGrid;
                    include_boundary::Bool=false) where {L<:Union{FaceX,FaceY,FaceZ}} =
    _restrict_face!(pf, cf, nest, _axis_of(L), include_boundary)

_axis_of(::Type{FaceX}) = 1
_axis_of(::Type{FaceY}) = 2
_axis_of(::Type{FaceZ}) = 3

function _restrict_face!(pf::Field, cf::Field, nest::NestedGrid, ax::Int,
                         include_boundary::Bool)
    pm = nest.parent_mesh; cm = nest.child_mesh
    r  = (nest.rx, nest.ry, nest.rz)
    nn = (nest.nx, nest.ny, nest.nz)
    oo = (nest.ix, nest.iy, nest.iz)
    Hp = (pm.Hx, pm.Hy, pm.Hz); Hc = (cm.Hx, cm.Hy, cm.Hz)
    ta = ax == 1 ? (2, 3) : ax == 2 ? (1, 3) : (1, 2)
    t1, t2 = ta
    inv = 1 / (r[t1] * r[t2])
    q0 = include_boundary ? 0 : 1
    q1 = include_boundary ? nn[ax] : nn[ax] - 1
    pd = pf.data; cd = cf.data
    pidx = [0, 0, 0]; cidx = [0, 0, 0]
    @inbounds for q in q0:q1
        pidx[ax] = Hp[ax] + oo[ax] + q
        cidx[ax] = Hc[ax] + q * r[ax] + 1
        for B in 1:nn[t2], A in 1:nn[t1]
            pidx[t1] = Hp[t1] + oo[t1] + A - 1
            pidx[t2] = Hp[t2] + oo[t2] + B - 1
            s = 0.0
            for b in 1:r[t2], a in 1:r[t1]
                cidx[t1] = Hc[t1] + (A-1)*r[t1] + a
                cidx[t2] = Hc[t2] + (B-1)*r[t2] + b
                s += cd[cidx[1], cidx[2], cidx[3]]
            end
            pd[pidx[1], pidx[2], pidx[3]] = s * inv
        end
    end
    return pf
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

"Total area of the six nest-interface planes."
function mismatch_area(nest::NestedGrid)
    pm = nest.parent_mesh
    return 2 * (nest.ny * nest.nz * pm.dy * pm.dz +
                nest.nx * nest.nz * pm.dx * pm.dz +
                nest.nx * nest.ny * pm.dx * pm.dy)
end

"""
    compute_nest_flux_mismatch(parent_field, child_field, nest)

`coarse flux − Σ fine fluxes` across each nest-boundary face patch, for a
face-located field (a velocity, or a time-accumulated flux register
`Φ = ∫F dt`). Returns a [`NestFluxMismatch`](@ref).

A nonzero value is exactly the amount by which the two grids disagree
about the transport through the interface — i.e. the leak that
[`apply_flux_correction!`](@ref) removes.
"""
compute_nest_flux_mismatch(pf::Field{L}, cf::Field{L},
                           nest::NestedGrid) where {L<:Union{FaceX,FaceY,FaceZ}} =
    _face_mismatch(pf, cf, nest, _axis_of(L))

function _face_mismatch(pf::Field, cf::Field, nest::NestedGrid, ax::Int)
    T  = eltype(pf.data)
    pm = nest.parent_mesh; cm = nest.child_mesh
    r  = (nest.rx, nest.ry, nest.rz)
    nn = (nest.nx, nest.ny, nest.nz)
    oo = (nest.ix, nest.iy, nest.iz)
    Hp = (pm.Hx, pm.Hy, pm.Hz); Hc = (cm.Hx, cm.Hy, cm.Hz)
    dp = (pm.dx, pm.dy, pm.dz)
    t1, t2 = ax == 1 ? (2, 3) : ax == 2 ? (1, 3) : (1, 2)
    Ap  = dp[t1] * dp[t2]
    inv = 1 / (r[t1] * r[t2])
    pd = pf.data; cd = cf.data
    lowm  = zeros(T, nn[t1], nn[t2])
    highm = zeros(T, nn[t1], nn[t2])
    pidx = [0, 0, 0]; cidx = [0, 0, 0]
    for (q, M) in ((0, lowm), (nn[ax], highm))
        pidx[ax] = Hp[ax] + oo[ax] + q
        cidx[ax] = Hc[ax] + q * r[ax] + 1
        @inbounds for B in 1:nn[t2], A in 1:nn[t1]
            pidx[t1] = Hp[t1] + oo[t1] + A - 1
            pidx[t2] = Hp[t2] + oo[t2] + B - 1
            s = 0.0
            for b in 1:r[t2], a in 1:r[t1]
                cidx[t1] = Hc[t1] + (A-1)*r[t1] + a
                cidx[t2] = Hc[t2] + (B-1)*r[t2] + b
                s += cd[cidx[1], cidx[2], cidx[3]]
            end
            M[A, B] = (pd[pidx[1], pidx[2], pidx[3]] - s * inv) * Ap
        end
    end
    e = zeros(T, 0, 0)
    ax == 1 && return NestFluxMismatch{T}(lowm, highm, copy(e), copy(e), copy(e), copy(e))
    ax == 2 && return NestFluxMismatch{T}(copy(e), copy(e), lowm, highm, copy(e), copy(e))
    return NestFluxMismatch{T}(copy(e), copy(e), copy(e), copy(e), lowm, highm)
end

"""
    apply_flux_correction!(parent_field, mismatch, nest)

Remove the coarse-fine flux inconsistency recorded in `mismatch`.

  * **face-located `parent_field`** — replace the parent's interface face
    value with the summed fine flux (`u_p -= δΦ / A_p`). The parent's
    transport through the interface then matches the child's exactly.
  * **cell-centred `parent_field`** (the Berger-Colella *refluxing* step,
    with `mismatch` built from time-accumulated flux registers) — adjust
    the parent cells immediately *outside* the nest:
    `θ ± δΦ·A / V_p`. This is what makes the composite integral of a
    conserved scalar exactly invariant across a nested step. Indices wrap
    periodically, so a nest touching the parent boundary still works.

Without this correction two-way nesting leaks: the parent kept the coarse
flux while the child used the fine ones.
"""
function apply_flux_correction!(pf::Field{L}, m::NestFluxMismatch,
                                nest::NestedGrid) where {L<:Union{FaceX,FaceY,FaceZ}}
    ax = _axis_of(L)
    pm = nest.parent_mesh
    dp = (pm.dx, pm.dy, pm.dz)
    t1, t2 = ax == 1 ? (2, 3) : ax == 2 ? (1, 3) : (1, 2)
    Ap = dp[t1] * dp[t2]
    lowm, highm = ax == 1 ? (m.west, m.east) : ax == 2 ? (m.south, m.north) :
                                               (m.bottom, m.top)
    (isempty(lowm) || isempty(highm)) && return pf
    nn = (nest.nx, nest.ny, nest.nz); oo = (nest.ix, nest.iy, nest.iz)
    Hp = (pm.Hx, pm.Hy, pm.Hz)
    pd = pf.data; pidx = [0, 0, 0]
    for (q, M) in ((0, lowm), (nn[ax], highm))
        pidx[ax] = Hp[ax] + oo[ax] + q
        @inbounds for B in 1:nn[t2], A in 1:nn[t1]
            pidx[t1] = Hp[t1] + oo[t1] + A - 1
            pidx[t2] = Hp[t2] + oo[t2] + B - 1
            pd[pidx[1], pidx[2], pidx[3]] -= M[A, B] / Ap
        end
    end
    return pf
end

function apply_flux_correction!(pf::Field{CellCenter}, m::NestFluxMismatch,
                                nest::NestedGrid)
    pm = nest.parent_mesh
    Vp = pm.dx * pm.dy * pm.dz
    Np = (pm.Nx, pm.Ny, pm.Nz); Hp = (pm.Hx, pm.Hy, pm.Hz)
    nn = (nest.nx, nest.ny, nest.nz); oo = (nest.ix, nest.iy, nest.iz)
    pd = pf.data
    planes = ((1, m.west, m.east), (2, m.south, m.north), (3, m.bottom, m.top))
    for (ax, lowm, highm) in planes
        (isempty(lowm) || isempty(highm)) && continue
        t1, t2 = ax == 1 ? (2, 3) : ax == 2 ? (1, 3) : (1, 2)
        # cell just outside the low face:  Δθ = +δΦ·A / V
        # cell just outside the high face: Δθ = -δΦ·A / V
        for (cell, M, sgn) in ((oo[ax] - 1, lowm, +1.0),
                               (oo[ax] + nn[ax], highm, -1.0))
            c = mod1(cell, Np[ax])
            idx = [0, 0, 0]; idx[ax] = Hp[ax] + c
            @inbounds for B in 1:nn[t2], A in 1:nn[t1]
                idx[t1] = Hp[t1] + oo[t1] + A - 1
                idx[t2] = Hp[t2] + oo[t2] + B - 1
                pd[idx[1], idx[2], idx[3]] += sgn * M[A, B] / Vp
            end
        end
    end
    return pf
end
