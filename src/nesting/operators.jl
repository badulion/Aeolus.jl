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
    plan = Vector{Tuple{Int,Vector{Float64}}}(undef, ncc)
    @inbounds for ic in 1:ncc
        if kind === :avg
            a = cc[ic]; b = cc[ic+1]
            I  = clamp(searchsortedlast(pc, (a + b) / 2), 1, npc)
            s0 = clamp(I - ((ns - 1) ÷ 2), 1, max(1, npc - ns + 1))
            plan[ic] = (s0, _cons_weights_nodes(view(pc, s0:s0+ns), a, b))
        else
            z  = cc[ic]
            F  = clamp(searchsortedlast(pc, z), 1, npc)
            s0 = clamp(F - ((ns - 1) ÷ 2), 1, max(1, npc - ns + 1))
            plan[ic] = (s0, _nodal_weights_nodes(view(pc, s0:s0+ns-1), z))
        end
    end
    return plan
end

"""
    _zplan(nest, kind, order, measure)

The vertical 1-D plan. On a uniform parent in the volume measure it is
the N-nest-a integer plan ([`_plan`](@ref)); otherwise the coordinate
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

_interface_plane(::CellCenter, cm, i, j, k) = false
_interface_plane(::FaceX, cm, i, j, k) = (i == cm.Hx + 1)
_interface_plane(::FaceY, cm, i, j, k) = (j == cm.Hy + 1)
_interface_plane(::FaceZ, cm, i, j, k) = (k == cm.Hz + 1)

# =========================================================================
# Prolongation: parent -> child
# =========================================================================

function _interp_child!(cf::Field, pf::Field, nest::NestedGrid;
                        order::Int=3, region::Symbol=:halo, measure=nothing)
    cm = nest.child_mesh
    cf.mesh === cm || size(cf.data) == totalsize(cm) ||
        error("_interp_child!: child field is not on the nest's child mesh")
    size(pf.data) == totalsize(nest.parent_mesh) ||
        error("_interp_child!: parent field is not on the nest's parent mesh")
    loc = location(cf)
    kinds = _dim_kinds(loc)
    px = _plan(nest, kinds[1], 1, order)
    py = _plan(nest, kinds[2], 2, order)
    pz = _zplan(nest, kinds[3], order, measure)
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

On a z-stretched parent the vertical stencils use the parent's actual
face heights. `measure` (a [`NestMeasure`](@ref)) selects the vertical
measure; with the mass measure of a sound-proof pair
([`nest_measure`](@ref)) the conservative directions conserve `∫ W φ dV`
(`W = ρ̄` or `ρ̄θ̄`) instead of `∫ φ dV`. The default is the volume measure.
"""
interpolate_to_nest_boundary!(cf::Field, pf::Field, nest::NestedGrid;
                              order::Int=3, impose_interface::Bool=false,
                              measure=nothing) =
    _interp_child!(cf, pf, nest; order, measure,
                   region = impose_interface ? :halo_interface : :halo)

"""
    interpolate_parent_to_child!(child_field, parent_field, nest; order=3)

As [`interpolate_to_nest_boundary!`](@ref) but fills the child field
*everywhere* (interior included). This is the AMR prolongation operator —
used to initialize newly refined blocks and to build the reference state
for [`apply_nest_sponge!`](@ref). Accepts the same `measure` keyword.
"""
interpolate_parent_to_child!(cf::Field, pf::Field, nest::NestedGrid; order::Int=3,
                             measure=nothing) =
    _interp_child!(cf, pf, nest; order, measure, region=:all)

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

function _restrict_cc_volume!(pf::Field, cf::Field, nest::NestedGrid)
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

function _restrict_cc_weighted!(pf::Field, cf::Field, nest::NestedGrid,
                                ωz::Vector{Float64}, χ)
    pm = nest.parent_mesh; cm = nest.child_mesh
    rx, ry, rz = nest.rx, nest.ry, nest.rz
    pd = pf.data; cd = cf.data
    @inbounds for K in 1:nest.nz, J in 1:nest.ny, I in 1:nest.nx
        num = 0.0; den = 0.0
        for kc in 1:rz
            k = cm.Hz + (K-1)*rz + kc
            wz = ωz[k]
            for jc in 1:ry, ic in 1:rx
                i = cm.Hx + (I-1)*rx + ic; j = cm.Hy + (J-1)*ry + jc
                (χ === nothing || χ[i, j, k]) || continue
                num += wz * cd[i, j, k]; den += wz
            end
        end
        den > 0 || continue                 # no fluid child: keep the parent value
        pd[pm.Hx + nest.ix + I - 1, pm.Hy + nest.iy + J - 1, pm.Hz + nest.iz + K - 1] =
            num / den
    end
    return pf
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

function _restrict_face_weighted!(pf::Field, cf::Field, nest::NestedGrid, ax::Int,
                                  include_boundary::Bool, ωz::Vector{Float64}, χ)
    pm = nest.parent_mesh; cm = nest.child_mesh
    r  = (nest.rx, nest.ry, nest.rz)
    nn = (nest.nx, nest.ny, nest.nz)
    oo = (nest.ix, nest.iy, nest.iz)
    Hp = (pm.Hx, pm.Hy, pm.Hz); Hc = (cm.Hx, cm.Hy, cm.Hz)
    t1, t2 = ax == 1 ? (2, 3) : ax == 2 ? (1, 3) : (1, 2)
    q0 = include_boundary ? 0 : 1
    q1 = include_boundary ? nn[ax] : nn[ax] - 1
    pd = pf.data; cd = cf.data
    pidx = [0, 0, 0]; cidx = [0, 0, 0]; bidx = [0, 0, 0]
    @inbounds for q in q0:q1
        pidx[ax] = Hp[ax] + oo[ax] + q
        cidx[ax] = Hc[ax] + q * r[ax] + 1
        for B in 1:nn[t2], A in 1:nn[t1]
            pidx[t1] = Hp[t1] + oo[t1] + A - 1
            pidx[t2] = Hp[t2] + oo[t2] + B - 1
            num = 0.0; den = 0.0
            for b in 1:r[t2], a in 1:r[t1]
                cidx[t1] = Hc[t1] + (A-1)*r[t1] + a
                cidx[t2] = Hc[t2] + (B-1)*r[t2] + b
                if χ !== nothing
                    bidx[1] = cidx[1]; bidx[2] = cidx[2]; bidx[3] = cidx[3]
                    bidx[ax] -= 1
                    (χ[cidx[1], cidx[2], cidx[3]] && χ[bidx[1], bidx[2], bidx[3]]) ||
                        continue
                end
                w = ωz[cidx[3]]
                num += w * cd[cidx[1], cidx[2], cidx[3]]; den += w
            end
            den > 0 || continue             # no fluid child face: keep the parent value
            pd[pidx[1], pidx[2], pidx[3]] = num / den
        end
    end
    return pf
end

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
    pm = nest.parent_mesh; cm = nest.child_mesh
    r  = (nest.rx, nest.ry, nest.rz)
    nn = (nest.nx, nest.ny, nest.nz)
    oo = (nest.ix, nest.iy, nest.iz)
    Hp = (pm.Hx, pm.Hy, pm.Hz); Hc = (cm.Hx, cm.Hy, cm.Hz)
    t1, t2 = ax == 1 ? (2, 3) : ax == 2 ? (1, 3) : (1, 2)
    ωz = ax == 3 ? nothing : _zweights(measure)
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
            Ap = _face_area(nest, measure, ax, B)
            s = 0.0; den = 0.0
            for b in 1:r[t2], a in 1:r[t1]
                cidx[t1] = Hc[t1] + (A-1)*r[t1] + a
                cidx[t2] = Hc[t2] + (B-1)*r[t2] + b
                if ωz === nothing
                    s += cd[cidx[1], cidx[2], cidx[3]]
                else
                    w = ωz[cidx[3]]
                    s += w * cd[cidx[1], cidx[2], cidx[3]]; den += w
                end
            end
            fine = ωz === nothing ? s * inv : s / den
            M[A, B] = (pd[pidx[1], pidx[2], pidx[3]] - fine) * Ap
        end
    end
    e = zeros(T, 0, 0)
    ax == 1 && return NestFluxMismatch{T}(lowm, highm, copy(e), copy(e), copy(e), copy(e))
    ax == 2 && return NestFluxMismatch{T}(copy(e), copy(e), lowm, highm, copy(e), copy(e))
    return NestFluxMismatch{T}(copy(e), copy(e), copy(e), copy(e), lowm, highm)
end

"""
    apply_flux_correction!(parent_field, mismatch, nest; measure=nothing)

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
    `periodic = (px, py, pz)` gives the parent's periodic axes; the
    default `(true, true, true)` is the default (triply periodic)
    `BoussinesqSimulation` and the former behaviour. Pass
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
    pm = nest.parent_mesh
    t1, t2 = ax == 1 ? (2, 3) : ax == 2 ? (1, 3) : (1, 2)
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
            pd[pidx[1], pidx[2], pidx[3]] -= M[A, B] / _face_area(nest, measure, ax, B)
        end
    end
    return pf
end

function apply_flux_correction!(pf::Field{CellCenter}, m::NestFluxMismatch,
                                nest::NestedGrid; measure=nothing,
                                periodic::NTuple{3,Bool} = (true, true, true))
    pm = nest.parent_mesh
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
            if 1 <= cell <= Np[ax]
                c = cell
            elseif periodic[ax]
                c = mod1(cell, Np[ax])
            else
                continue                # the face is the parent's own boundary
            end
            idx = [0, 0, 0]; idx[ax] = Hp[ax] + c
            @inbounds for B in 1:nn[t2], A in 1:nn[t1]
                idx[t1] = Hp[t1] + oo[t1] + A - 1
                idx[t2] = Hp[t2] + oo[t2] + B - 1
                K = idx[3] - Hp[3]
                Vp = pm.dx * pm.dy * _pcell(nest, measure, K)
                pd[idx[1], idx[2], idx[3]] += sgn * M[A, B] / Vp
            end
        end
    end
    return pf
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
