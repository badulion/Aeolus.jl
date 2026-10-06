module AdaptiveMesh

# =========================================================================
# Phase 31 — block-structured AMR.
# =========================================================================
#
# The parent ↔ child operators this module is built on (Phase 30:
# `NestedGrid`, the conservative staggered prolongation, the conservative
# restriction and Berger-Colella refluxing) live in `src/Nesting.jl` — the
# single source of truth for every parent-child operator (N-nest-a, #250).
# Each AMR block carries a `NestedGrid` linking it to its parent block and
# delegates prolongation, restriction and refluxing to `Nesting`.

using ..Grids
using ..Grids: AbstractMesh, RegularGrid, CellCenter, FaceX, FaceY, FaceZ,
               Location, interior, totalsize
using ..Fields
using ..Fields: Field, location, interior_view, set!
using ..BoundaryConditions
using ..BoundaryConditions: AbstractBC, FieldBoundaryConditions, fill_halos!,
                            Periodic, NoFlux, ImpermeableWall, NoSlipWall
using ..Simulations
using ..Simulations: BoussinesqSimulation
using ..Operators: divergence
using ..TimeSteppers: step!
using ..Nesting
using ..Nesting: NestedGrid, _triple, apply_nest_boundary!
using ..BoundaryConditions: _fill_x!, _fill_y!, _fill_z!

export RefinementBlock, AMRHierarchy, amr_hierarchy,
       refinement_criterion_gradient, refinement_criterion_vorticity,
       refinement_criterion_richardson, combine_criteria,
       cluster_flagged_cells, box_efficiency,
       regrid!, check_proper_nesting, step_amr!,
       amr_total_mass, amr_error, total_cells, level_cells, block_bounds

# =========================================================================
# Phase 31 — block-structured AMR
# =========================================================================

"""
    RefinementBlock

A rectangular patch of refined mesh.

  * `level` — 1 for the base (root) block, 2 for the first refined level, …
  * `bounds` — index range `(ix, iy, iz)` in **this level's** global index
    space (`base.N .* r.^(level-1)` cells). Always aligned to the parent
    level's cell boundaries.
  * `mesh` — the block's own [`RegularGrid`](@ref) (with halos).
  * `state` — `Dict{Symbol,Field}`: the advected scalars, the velocity
    `:u, :v, :w`, and the time-accumulated flux registers
    `:Fx_θ, :Fy_θ, :Fz_θ` used for refluxing.
  * `parent` / `children` — the level links.
  * `nest` — the [`NestedGrid`](@ref) relating this block to its parent
    (prolongation / restriction / refluxing operators).
"""
mutable struct RefinementBlock{T}
    level::Int
    bounds::NTuple{3,UnitRange{Int}}
    mesh::RegularGrid{T}
    state::Dict{Symbol,Field}
    parent::Union{Nothing,RefinementBlock{T}}
    children::Vector{RefinementBlock{T}}
    nest::Union{Nothing,NestedGrid}
end

block_bounds(b::RefinementBlock) = b.bounds
Base.show(io::IO, b::RefinementBlock) =
    print(io, "RefinementBlock(level=", b.level, ", bounds=", b.bounds, ", cells=",
          b.mesh.Nx * b.mesh.Ny * b.mesh.Nz, ")")

"""
    AMRHierarchy

The level structure: `levels[ℓ]` is the vector of [`RefinementBlock`](@ref)s
at level `ℓ` (`levels[1]` always holds the single root block covering the
base grid). Blocks at one level never overlap; levels deeper than 1 are
built by [`regrid!`](@ref), which enforces proper nesting with a buffer.

Fields of interest: `base`, `refinement` (`(rx,ry,rz)`), `max_level`,
`vars` (advected scalars), `velocity` (a prescribed
`(x,y,z,t) -> (u,v,w)`), `advance!` (per-block update), `order`
(prolongation order), `buffer`, `t`, `nstep`.
"""
mutable struct AMRHierarchy{T}
    base::RegularGrid{T}
    refinement::NTuple{3,Int}
    max_level::Int
    levels::Vector{Vector{RefinementBlock{T}}}
    vars::Vector{Symbol}
    velocity::Any
    advance!::Any
    order::Int
    bcs::FieldBoundaryConditions
    buffer::Int
    t::T
    nstep::Int
end

_reg(v::Symbol, ax::Symbol) = Symbol("F", ax, "_", v)

_level_spacing(h::AMRHierarchy, ℓ::Int) =
    (h.base.dx / h.refinement[1]^(ℓ-1),
     h.base.dy / h.refinement[2]^(ℓ-1),
     h.base.dz / h.refinement[3]^(ℓ-1))

_level_size(h::AMRHierarchy, ℓ::Int) =
    (h.base.Nx * h.refinement[1]^(ℓ-1),
     h.base.Ny * h.refinement[2]^(ℓ-1),
     h.base.Nz * h.refinement[3]^(ℓ-1))

# physical coordinates of an array index (halos included)
_xf(m::RegularGrid, i) = m.x0 + (i - m.Hx - 1) * m.dx
_yf(m::RegularGrid, j) = m.y0 + (j - m.Hy - 1) * m.dy
_zf(m::RegularGrid, k) = m.z0 + (k - m.Hz - 1) * m.dz
_xc(m::RegularGrid, i) = m.x0 + (i - m.Hx - 0.5) * m.dx
_yc(m::RegularGrid, j) = m.y0 + (j - m.Hy - 0.5) * m.dy
_zc(m::RegularGrid, k) = m.z0 + (k - m.Hz - 0.5) * m.dz

function _alloc_state!(h::AMRHierarchy, b::RefinementBlock)
    m = b.mesh
    b.state[:u] = Field(FaceX(), m)
    b.state[:v] = Field(FaceY(), m)
    b.state[:w] = Field(FaceZ(), m)
    b.state[:fx] = Field(FaceX(), m)   # instantaneous flux scratch
    b.state[:fy] = Field(FaceY(), m)
    b.state[:fz] = Field(FaceZ(), m)
    for v in h.vars
        b.state[v] = Field(CellCenter(), m)
        b.state[_reg(v, :x)] = Field(FaceX(), m)
        b.state[_reg(v, :y)] = Field(FaceY(), m)
        b.state[_reg(v, :z)] = Field(FaceZ(), m)
    end
    return b
end

_coarse_bounds(h::AMRHierarchy, b::RefinementBlock) =
    ntuple(d -> ((first(b.bounds[d]) - 1) ÷ h.refinement[d] + 1) :
                (last(b.bounds[d]) ÷ h.refinement[d]), 3)

function _make_block(h::AMRHierarchy{T}, ℓ::Int, bounds::NTuple{3,UnitRange{Int}},
                     parent::Union{Nothing,RefinementBlock{T}}) where {T}
    d = _level_spacing(h, ℓ)
    n = (length(bounds[1]), length(bounds[2]), length(bounds[3]))
    mesh = RegularGrid(T; size=n,
                       extent=(n[1]*d[1], n[2]*d[2], n[3]*d[3]),
                       halo=(h.base.Hx, h.base.Hy, h.base.Hz),
                       origin=(h.base.x0 + (first(bounds[1]) - 1) * d[1],
                               h.base.y0 + (first(bounds[2]) - 1) * d[2],
                               h.base.z0 + (first(bounds[3]) - 1) * d[3]))
    b = RefinementBlock{T}(ℓ, bounds, mesh, Dict{Symbol,Field}(), parent,
                           RefinementBlock{T}[], nothing)
    _alloc_state!(h, b)
    if parent !== nothing
        cb = _coarse_bounds(h, b)
        off = ntuple(dd -> first(cb[dd]) - first(parent.bounds[dd]) + 1, 3)
        b.nest = NestedGrid(parent.mesh, mesh; refinement=h.refinement, offset=off)
    end
    _set_block_velocity!(h, b, h.t)
    return b
end

function _set_block_velocity!(h::AMRHierarchy, b::RefinementBlock, t)
    h.velocity === nothing && return b
    m = b.mesh
    ud = b.state[:u].data; vd = b.state[:v].data; wd = b.state[:w].data
    @inbounds for k in axes(ud, 3), j in axes(ud, 2), i in axes(ud, 1)
        ud[i,j,k] = h.velocity(_xf(m,i), _yc(m,j), _zc(m,k), t)[1]
        vd[i,j,k] = h.velocity(_xc(m,i), _yf(m,j), _zc(m,k), t)[2]
        wd[i,j,k] = h.velocity(_xc(m,i), _yc(m,j), _zf(m,k), t)[3]
    end
    return b
end

function _zero_registers!(h::AMRHierarchy, b::RefinementBlock)
    for v in h.vars, ax in (:x, :y, :z)
        fill!(b.state[_reg(v, ax)].data, 0)
    end
    return b
end

"""
    _amr_advect!(hierarchy, block, dt, t)

Default per-block update: flux-form first-order-upwind advection of every
scalar in `hierarchy.vars` by the block's velocity. The face fluxes are
accumulated (×`dt`) into the block's flux registers, which is what makes
Berger-Colella refluxing — and therefore exact composite conservation —
possible. Replace via the `advance!` keyword of [`amr_hierarchy`](@ref).
"""
function _amr_advect!(h::AMRHierarchy, b::RefinementBlock, dt::Real, t)
    m = b.mesh
    Hx, Hy, Hz = m.Hx, m.Hy, m.Hz
    Nx, Ny, Nz = m.Nx, m.Ny, m.Nz
    idx, idy, idz = 1/m.dx, 1/m.dy, 1/m.dz
    ud = b.state[:u].data; vd = b.state[:v].data; wd = b.state[:w].data
    fxd = b.state[:fx].data; fyd = b.state[:fy].data; fzd = b.state[:fz].data
    for v in h.vars
        θd = b.state[v].data
        @inbounds for k in Hz+1:Hz+Nz, j in Hy+1:Hy+Ny, i in Hx+1:Hx+Nx+1
            uu = ud[i,j,k]
            fxd[i,j,k] = uu * (uu >= 0 ? θd[i-1,j,k] : θd[i,j,k])
        end
        @inbounds for k in Hz+1:Hz+Nz, j in Hy+1:Hy+Ny+1, i in Hx+1:Hx+Nx
            vv = vd[i,j,k]
            fyd[i,j,k] = vv * (vv >= 0 ? θd[i,j-1,k] : θd[i,j,k])
        end
        @inbounds for k in Hz+1:Hz+Nz+1, j in Hy+1:Hy+Ny, i in Hx+1:Hx+Nx
            ww = wd[i,j,k]
            fzd[i,j,k] = ww * (ww >= 0 ? θd[i,j,k-1] : θd[i,j,k])
        end
        @inbounds for k in Hz+1:Hz+Nz, j in Hy+1:Hy+Ny, i in Hx+1:Hx+Nx
            θd[i,j,k] -= dt * ((fxd[i+1,j,k] - fxd[i,j,k]) * idx +
                               (fyd[i,j+1,k] - fyd[i,j,k]) * idy +
                               (fzd[i,j,k+1] - fzd[i,j,k]) * idz)
        end
        Fx = b.state[_reg(v,:x)].data
        Fy = b.state[_reg(v,:y)].data
        Fz = b.state[_reg(v,:z)].data
        @inbounds for k in Hz+1:Hz+Nz, j in Hy+1:Hy+Ny, i in Hx+1:Hx+Nx+1
            Fx[i,j,k] += dt * fxd[i,j,k]
        end
        @inbounds for k in Hz+1:Hz+Nz, j in Hy+1:Hy+Ny+1, i in Hx+1:Hx+Nx
            Fy[i,j,k] += dt * fyd[i,j,k]
        end
        @inbounds for k in Hz+1:Hz+Nz+1, j in Hy+1:Hy+Ny, i in Hx+1:Hx+Nx
            Fz[i,j,k] += dt * fzd[i,j,k]
        end
    end
    return b
end

"""
    amr_hierarchy(base_mesh; refinement=(2,2,2), max_level=2, vars=[:θ],
                  velocity=nothing, advance! = _amr_advect!, order=3,
                  bcs=FieldBoundaryConditions(:periodic), buffer=2)

Build a one-level hierarchy whose root block covers `base_mesh`. Call
[`regrid!`](@ref) (or [`step_amr!`](@ref) with a criterion) to create the
refined levels.
"""
function amr_hierarchy(base::RegularGrid{T};
                       refinement=(2,2,2), max_level::Int=2,
                       vars::Vector{Symbol}=[:θ],
                       velocity=nothing,
                       advance! = _amr_advect!,
                       order::Int=3,
                       bcs::FieldBoundaryConditions=FieldBoundaryConditions(:periodic),
                       buffer::Int=2) where {T}
    r = _triple(refinement)
    max_level >= 1 || error("amr_hierarchy: max_level must be >= 1")
    h = AMRHierarchy{T}(base, r, max_level, Vector{Vector{RefinementBlock{T}}}(),
                        copy(vars), velocity, advance!, order, bcs, buffer,
                        zero(T), 0)
    root = _make_block(h, 1, (1:base.Nx, 1:base.Ny, 1:base.Nz), nothing)
    push!(h.levels, RefinementBlock{T}[root])
    return h
end

total_cells(h::AMRHierarchy) =
    sum(b -> b.mesh.Nx * b.mesh.Ny * b.mesh.Nz, Iterators.flatten(h.levels); init=0)

level_cells(h::AMRHierarchy, ℓ::Int) =
    ℓ > length(h.levels) ? 0 :
    sum(b -> b.mesh.Nx * b.mesh.Ny * b.mesh.Nz, h.levels[ℓ]; init=0)

# =========================================================================
# Refinement criteria
# =========================================================================

_field_of(x::BoussinesqSimulation, name::Symbol) = getproperty(x, name)
_field_of(b::RefinementBlock, name::Symbol) = b.state[name]
_field_of(f::Field, ::Symbol) = f

function _lohner(fm::Float64, f0::Float64, fp::Float64, ε::Float64)
    num = abs(fp - 2 * f0 + fm)
    den = abs(fp - f0) + abs(f0 - fm) + ε * (abs(fp) + 2abs(f0) + abs(fm))
    den <= 0 && return 0.0
    return num / den
end

"""
    refinement_criterion_gradient(x; field=:θ, threshold=0.1, ε=0.01) -> BitArray{3}

Löhner-style normalized second-derivative indicator on `field`
(`x` may be a `BoussinesqSimulation`, a [`RefinementBlock`](@ref) or a
`Field`). For each direction

    e = |fᵢ₊₁ - 2fᵢ + fᵢ₋₁| / (|fᵢ₊₁-fᵢ| + |fᵢ-fᵢ₋₁| + ε(|fᵢ₊₁| + 2|fᵢ| + |fᵢ₋₁|))

and the mask flags cells where `max(eₓ, e_y, e_z) > threshold`. The `ε`
term is what keeps smooth *and* flat regions unflagged — without it the
indicator saturates on round-off. The indicator is dimensionless, so
`threshold` is scale-free. Domain-boundary cells contribute `0` in the
affected direction (no reliance on halo contents).
"""
function refinement_criterion_gradient(x; field::Symbol=:θ, threshold::Real=0.1,
                                       ε::Real=0.01)
    f = _field_of(x, field); m = f.mesh
    d = f.data; εf = Float64(ε)
    mask = falses(m.Nx, m.Ny, m.Nz)
    @inbounds for k in 1:m.Nz, j in 1:m.Ny, i in 1:m.Nx
        ia = m.Hx + i; ja = m.Hy + j; ka = m.Hz + k
        f0 = Float64(d[ia, ja, ka])
        e = 0.0
        if 1 < i < m.Nx
            e = max(e, _lohner(Float64(d[ia-1,ja,ka]), f0, Float64(d[ia+1,ja,ka]), εf))
        end
        if 1 < j < m.Ny
            e = max(e, _lohner(Float64(d[ia,ja-1,ka]), f0, Float64(d[ia,ja+1,ka]), εf))
        end
        if 1 < k < m.Nz
            e = max(e, _lohner(Float64(d[ia,ja,ka-1]), f0, Float64(d[ia,ja,ka+1]), εf))
        end
        mask[i,j,k] = e > threshold
    end
    return mask
end

"""
    refinement_criterion_vorticity(x; threshold=0.2, relative=true) -> BitArray{3}

Flag cells by vorticity magnitude `|ω|` — the natural LES criterion, since
`|ω|` marks the shear layers and coherent structures that carry the SGS
burden. `u, v, w` are averaged to cell centres and differentiated with
centred (one-sided at the domain edge) differences. With `relative=true`
(default) `threshold` is a fraction of `max|ω|`; otherwise it is absolute.
"""
function refinement_criterion_vorticity(x; threshold::Real=0.2, relative::Bool=true)
    u = _field_of(x, :u); v = _field_of(x, :v); w = _field_of(x, :w)
    m = u.mesh
    Nx, Ny, Nz = m.Nx, m.Ny, m.Nz
    uc = zeros(Float64, Nx, Ny, Nz); vc = similar(uc); wc = similar(uc)
    ud = u.data; vd = v.data; wd = w.data
    @inbounds for k in 1:Nz, j in 1:Ny, i in 1:Nx
        ia = m.Hx + i; ja = m.Hy + j; ka = m.Hz + k
        uc[i,j,k] = 0.5 * (ud[ia,ja,ka] + ud[ia+1,ja,ka])
        vc[i,j,k] = 0.5 * (vd[ia,ja,ka] + vd[ia,ja+1,ka])
        wc[i,j,k] = 0.5 * (wd[ia,ja,ka] + wd[ia,ja,ka+1])
    end
    ω = zeros(Float64, Nx, Ny, Nz)
    @inbounds for k in 1:Nz, j in 1:Ny, i in 1:Nx
        ip = min(i+1, Nx); im = max(i-1, 1); hx = (ip - im) * m.dx
        jp = min(j+1, Ny); jm = max(j-1, 1); hy = (jp - jm) * m.dy
        kp = min(k+1, Nz); km = max(k-1, 1); hz = (kp - km) * m.dz
        dwdy = hy > 0 ? (wc[i,jp,k] - wc[i,jm,k]) / hy : 0.0
        dvdz = hz > 0 ? (vc[i,j,kp] - vc[i,j,km]) / hz : 0.0
        dudz = hz > 0 ? (uc[i,j,kp] - uc[i,j,km]) / hz : 0.0
        dwdx = hx > 0 ? (wc[ip,j,k] - wc[im,j,k]) / hx : 0.0
        dvdx = hx > 0 ? (vc[ip,j,k] - vc[im,j,k]) / hx : 0.0
        dudy = hy > 0 ? (uc[i,jp,k] - uc[i,jm,k]) / hy : 0.0
        ω[i,j,k] = sqrt((dwdy - dvdz)^2 + (dudz - dwdx)^2 + (dvdx - dudy)^2)
    end
    ωmax = maximum(ω)
    thr = relative ? Float64(threshold) * ωmax : Float64(threshold)
    (relative && ωmax <= 0) && return falses(Nx, Ny, Nz)
    return ω .> thr
end

"""
    refinement_criterion_richardson(x; field=:θ, threshold=0.05, p=2) -> BitArray{3}

Richardson-extrapolation error estimate — the most defensible criterion
because it estimates the *discretization error itself* rather than a proxy
for it. The field is restricted to a 2× coarser mesh (in every direction
whose extent is even), prolonged back trilinearly, and the difference
divided by `2^p - 1` (with `p` the scheme's formal order) gives the local
error estimate. Normalized by the field's range; cells exceeding
`threshold` are flagged.
"""
function refinement_criterion_richardson(x; field::Symbol=:θ, threshold::Real=0.05,
                                         p::Int=2)
    f = _field_of(x, field)
    a = Float64.(Array(interior_view(f)))
    Nx, Ny, Nz = size(a)
    fx = iseven(Nx) ? 2 : 1; fy = iseven(Ny) ? 2 : 1; fz = iseven(Nz) ? 2 : 1
    Cx, Cy, Cz = Nx ÷ fx, Ny ÷ fy, Nz ÷ fz
    c = zeros(Float64, Cx, Cy, Cz)
    @inbounds for k in 1:Cz, j in 1:Cy, i in 1:Cx
        s = 0.0
        for kk in 1:fz, jj in 1:fy, ii in 1:fx
            s += a[(i-1)*fx+ii, (j-1)*fy+jj, (k-1)*fz+kk]
        end
        c[i,j,k] = s / (fx*fy*fz)
    end
    # trilinear prolongation back to the fine mesh
    b = zeros(Float64, Nx, Ny, Nz)
    @inbounds for k in 1:Nz, j in 1:Ny, i in 1:Nx
        i0, ti = _prolong_1d(i, fx, Cx)
        j0, tj = _prolong_1d(j, fy, Cy)
        k0, tk = _prolong_1d(k, fz, Cz)
        i1 = min(i0+1, Cx); j1 = min(j0+1, Cy); k1 = min(k0+1, Cz)
        b[i,j,k] =
            (1-ti)*(1-tj)*(1-tk)*c[i0,j0,k0] + ti*(1-tj)*(1-tk)*c[i1,j0,k0] +
            (1-ti)*tj*(1-tk)*c[i0,j1,k0]     + ti*tj*(1-tk)*c[i1,j1,k0] +
            (1-ti)*(1-tj)*tk*c[i0,j0,k1]     + ti*(1-tj)*tk*c[i1,j0,k1] +
            (1-ti)*tj*tk*c[i0,j1,k1]         + ti*tj*tk*c[i1,j1,k1]
    end
    err = abs.(a .- b) ./ (2^p - 1)
    rng = maximum(a) - minimum(a)
    scale = rng > 0 ? rng : max(maximum(abs, a), eps())
    return (err ./ scale) .> threshold
end

function _prolong_1d(i::Int, f::Int, Nc::Int)
    f == 1 && return (i, 0.0)
    ci = (i - 0.5) / f + 0.5            # coarse-index coordinate of the fine centre
    i0 = clamp(floor(Int, ci), 1, max(1, Nc - 1))
    return (i0, ci - i0)                # may leave [0,1] → linear extrapolation at
end                                     #   the edges, so a linear field is exact

"""
    combine_criteria(masks...; mode=:union) -> BitArray{3}

Combine refinement masks. `mode`: `:union` (flag if any), `:intersection`
(flag only if all), `:majority` (flag if at least half).
"""
function combine_criteria(masks...; mode::Symbol=:union)
    isempty(masks) && error("combine_criteria: need at least one mask")
    sz = size(first(masks))
    all(m -> size(m) == sz, masks) ||
        error("combine_criteria: masks have mismatched sizes")
    if mode === :union
        out = falses(sz); for m in masks; out .|= m; end; return out
    elseif mode === :intersection
        out = trues(sz); for m in masks; out .&= m; end; return out
    elseif mode === :majority
        cnt = zeros(Int, sz); for m in masks; cnt .+= m; end
        return cnt .>= ceil(Int, length(masks) / 2)
    end
    error("combine_criteria: unknown mode $mode (:union, :intersection, :majority)")
end

# =========================================================================
# Berger-Rigoutsos clustering
# =========================================================================

const Box3 = NTuple{3,UnitRange{Int}}

function _bbox(mask::AbstractArray{Bool,3}, box::Box3)
    lo = [typemax(Int), typemax(Int), typemax(Int)]; hi = [0, 0, 0]
    found = false
    @inbounds for k in box[3], j in box[2], i in box[1]
        if mask[i,j,k]
            found = true
            lo[1] = min(lo[1], i); hi[1] = max(hi[1], i)
            lo[2] = min(lo[2], j); hi[2] = max(hi[2], j)
            lo[3] = min(lo[3], k); hi[3] = max(hi[3], k)
        end
    end
    found || return nothing
    return (lo[1]:hi[1], lo[2]:hi[2], lo[3]:hi[3])
end

function _count_flagged(mask::AbstractArray{Bool,3}, box::Box3)
    n = 0
    @inbounds for k in box[3], j in box[2], i in box[1]
        mask[i,j,k] && (n += 1)
    end
    return n
end

_box_volume(box::Box3) = length(box[1]) * length(box[2]) * length(box[3])

function _signature(mask::AbstractArray{Bool,3}, box::Box3, d::Int)
    n = length(box[d])
    sig = zeros(Int, n)
    @inbounds for k in box[3], j in box[2], i in box[1]
        mask[i,j,k] || continue
        t = d == 1 ? i - first(box[1]) + 1 :
            d == 2 ? j - first(box[2]) + 1 : k - first(box[3]) + 1
        sig[t] += 1
    end
    return sig
end

function _split_box(box::Box3, d::Int, s::Int)
    lo = first(box[d]); hi = last(box[d])
    b1 = ntuple(e -> e == d ? (lo : lo + s - 1) : box[e], 3)
    b2 = ntuple(e -> e == d ? (lo + s : hi) : box[e], 3)
    return (b1, b2)
end

"""
    _try_split(mask, box, min_size) -> ((box1, box2) | nothing)

Berger-Rigoutsos split selection. Longest dimension first: prefer a *hole*
in the signature (a slice with no flagged cells), otherwise the strongest
*inflection* of the signature's second difference (a sign change in
`Σᵢ₊₁ - 2Σᵢ + Σᵢ₋₁`). Falls back to a midpoint bisection so progress —
and therefore termination — is guaranteed.
"""
function _try_split(mask::AbstractArray{Bool,3}, box::Box3, min_size::Int)
    lens = (length(box[1]), length(box[2]), length(box[3]))
    dorder = sortperm(collect(lens); rev=true)
    # 1. holes in the signature
    for d in dorder
        n = lens[d]
        n >= 2 * min_size + 1 || continue
        sig = _signature(mask, box, d)
        cand = [z for z in min_size:(n - min_size) if sig[z] == 0]
        isempty(cand) && continue
        mid = (n + 1) / 2
        s = cand[argmin(abs.(cand .- mid))]
        return _split_box(box, d, s)
    end
    # 2. inflection point of the signature Laplacian
    bestd = 0; bests = 0; bestval = -1.0; signchange = false
    for d in dorder
        n = lens[d]
        n >= 2 * min_size + 2 || continue
        sig = _signature(mask, box, d)
        Δ = [sig[i+1] - 2sig[i] + sig[i-1] for i in 2:n-1]
        for t in 1:length(Δ)-1
            s = t + 1                       # split after slice s
            (min_size <= s <= n - min_size) || continue
            sc = Δ[t] * Δ[t+1] < 0
            val = abs(Δ[t] - Δ[t+1])
            if (sc && !signchange) || (sc == signchange && val > bestval)
                bestd = d; bests = s; bestval = val; signchange = sc
            end
        end
        signchange && break                 # longest dim with a real inflection wins
    end
    bestd != 0 && return _split_box(box, bestd, bests)
    # 3. midpoint bisection of the longest splittable dimension
    for d in dorder
        lens[d] >= 2 * min_size && return _split_box(box, d, lens[d] ÷ 2)
    end
    return nothing
end

"""
    cluster_flagged_cells(mask; min_efficiency=0.7, max_blocks=64, min_size=1)
        -> Vector{NTuple{3,UnitRange{Int}}}

Berger-Rigoutsos signature-based clustering: turn a scattered Bool flag
mask into a small set of efficient, non-overlapping rectangular boxes.

A box is accepted when its *efficiency* (flagged cells / box cells) reaches
`min_efficiency`, when it can no longer be split, or when the `max_blocks`
budget is exhausted; otherwise it is split at a signature hole or
inflection (see `_try_split`) and each half is re-tightened to its bounding
box. Report the achieved efficiency with [`box_efficiency`](@ref).
"""
function cluster_flagged_cells(mask::AbstractArray{Bool,3};
                               min_efficiency::Real=0.7, max_blocks::Int=64,
                               min_size::Int=1)
    root = _bbox(mask, (1:size(mask,1), 1:size(mask,2), 1:size(mask,3)))
    root === nothing && return Box3[]
    accepted = Box3[]
    queue = Box3[root]
    while !isempty(queue)
        box = popfirst!(queue)
        nf = _count_flagged(mask, box)
        if nf == 0
            continue
        end
        eff = nf / _box_volume(box)
        if eff >= min_efficiency || (length(accepted) + length(queue) + 1) >= max_blocks
            push!(accepted, box); continue
        end
        sp = _try_split(mask, box, min_size)
        if sp === nothing
            push!(accepted, box); continue
        end
        for hb in sp
            t = _bbox(mask, hb)
            t === nothing || push!(queue, t)
        end
    end
    return accepted
end

"""
    box_efficiency(mask, boxes) -> Float64

Achieved clustering efficiency: flagged cells inside `boxes` divided by the
total cell count of `boxes`.
"""
function box_efficiency(mask::AbstractArray{Bool,3}, boxes::AbstractVector)
    isempty(boxes) && return 1.0
    nf = sum(b -> _count_flagged(mask, b), boxes)
    nv = sum(_box_volume, boxes)
    return nv == 0 ? 1.0 : nf / nv
end

# =========================================================================
# Regridding
# =========================================================================

function _dilate(mask::AbstractArray{Bool,3}, b::Int)
    b <= 0 && return BitArray(mask)
    nx, ny, nz = size(mask)
    out = falses(nx, ny, nz)
    @inbounds for k in 1:nz, j in 1:ny, i in 1:nx
        mask[i,j,k] || continue
        for kk in max(1,k-b):min(nz,k+b), jj in max(1,j-b):min(ny,j+b),
            ii in max(1,i-b):min(nx,i+b)
            out[ii,jj,kk] = true
        end
    end
    return out
end

"Zero the mask within `buffer` cells of `pb`'s sides that are not domain edges."
function _restrict_legal!(mask::AbstractArray{Bool,3}, h::AMRHierarchy,
                          pb::RefinementBlock, ℓ::Int, buffer::Int)
    buffer <= 0 && return mask
    N = _level_size(h, ℓ)
    n = size(mask)
    for d in 1:3
        w = min(buffer, n[d])
        if first(pb.bounds[d]) != 1
            d == 1 ? (mask[1:w, :, :] .= false) :
            d == 2 ? (mask[:, 1:w, :] .= false) : (mask[:, :, 1:w] .= false)
        end
        if last(pb.bounds[d]) != N[d]
            lo = n[d] - w + 1
            d == 1 ? (mask[lo:n[1], :, :] .= false) :
            d == 2 ? (mask[:, lo:n[2], :] .= false) : (mask[:, :, lo:n[3]] .= false)
        end
    end
    return mask
end

_eval_criterion(c, h::AMRHierarchy, pb::RefinementBlock) = c(pb)
function _eval_criterion(c::AbstractArray{Bool,3}, h::AMRHierarchy,
                         pb::RefinementBlock)
    pb.level == 1 || return falses(pb.mesh.Nx, pb.mesh.Ny, pb.mesh.Nz)
    size(c) == (pb.mesh.Nx, pb.mesh.Ny, pb.mesh.Nz) ||
        error("regrid!: mask size $(size(c)) does not match the base grid")
    return c
end

function _copy_overlap!(dst::Field, db::Box3, src::Field, sb::Box3, ov::Box3)
    dm = dst.mesh; sm = src.mesh
    dd = dst.data; sd = src.data
    @inbounds for k in ov[3], j in ov[2], i in ov[1]
        dd[dm.Hx + i - first(db[1]) + 1,
           dm.Hy + j - first(db[2]) + 1,
           dm.Hz + k - first(db[3]) + 1] =
        sd[sm.Hx + i - first(sb[1]) + 1,
           sm.Hy + j - first(sb[2]) + 1,
           sm.Hz + k - first(sb[3]) + 1]
    end
    return dst
end

function _init_block_state!(h::AMRHierarchy, nb::RefinementBlock,
                            old::Vector{<:RefinementBlock})
    # (a) conservative prolongation from the parent everywhere: the r³
    #     sub-cell averages sum back to the parent average, so the composite
    #     integral is untouched by creating new refinement.
    for v in h.vars
        interpolate_parent_to_child!(nb.state[v], nb.parent.state[v], nb.nest;
                                     order=h.order)
    end
    # (b) where an old block of the same level overlaps, keep its (more
    #     accurate) data. Both blocks are parent-cell aligned, so the overlap
    #     carries the same mass either way — still conservative.
    for ob in old
        ob.level == nb.level || continue
        ov = ntuple(d -> intersect(nb.bounds[d], ob.bounds[d]), 3)
        any(isempty, ov) && continue
        for v in h.vars
            _copy_overlap!(nb.state[v], nb.bounds, ob.state[v], ob.bounds, ov)
        end
    end
    _set_block_velocity!(h, nb, h.t)
    _zero_registers!(h, nb)
    return nb
end

"""
    regrid!(hierarchy, criteria; buffer=hierarchy.buffer, min_efficiency=0.7,
            max_blocks=64, min_size=1)

Rebuild the refined levels. `criteria` is either a function
`block -> Bool mask` (evaluated on every block of the level being refined)
or a Bool array over the base grid.

Per level: evaluate the criterion, dilate the flags by `buffer` cells,
clip to the legally nestable region (`buffer` cells inside the parent
block, except at domain boundaries), cluster with
[`cluster_flagged_cells`](@ref), then transfer state — conservative
prolongation where refinement is new, direct copy where an old block
overlapped, and the parent already holds the restricted averages where
refinement disappeared. Proper nesting is guaranteed by construction; see
[`check_proper_nesting`](@ref).
"""
function regrid!(h::AMRHierarchy{T}, criteria;
                 buffer::Int=h.buffer, min_efficiency::Real=0.7,
                 max_blocks::Int=64, min_size::Int=1) where {T}
    # parent halos must be valid before prolongation
    for ℓ in 1:length(h.levels), b in h.levels[ℓ]
        _fill_block_halos!(h, b)
    end
    newlevels = Vector{Vector{RefinementBlock{T}}}()
    push!(newlevels, h.levels[1])
    for ℓ in 1:h.max_level-1
        parents = newlevels[ℓ]
        isempty(parents) && break
        fresh = RefinementBlock{T}[]
        for pb in parents
            mask = _eval_criterion(criteria, h, pb)
            m2 = _dilate(mask, buffer)
            _restrict_legal!(m2, h, pb, ℓ, buffer)
            any(m2) || continue
            for bx in cluster_flagged_cells(m2; min_efficiency, max_blocks, min_size)
                gcb = ntuple(d -> (first(pb.bounds[d]) + first(bx[d]) - 1) :
                                  (first(pb.bounds[d]) + last(bx[d]) - 1), 3)
                fb = ntuple(d -> ((first(gcb[d]) - 1) * h.refinement[d] + 1) :
                                 (last(gcb[d]) * h.refinement[d]), 3)
                push!(fresh, _make_block(h, ℓ+1, fb, pb))
            end
        end
        old = length(h.levels) >= ℓ+1 ? h.levels[ℓ+1] : RefinementBlock{T}[]
        for nb in fresh
            _init_block_state!(h, nb, old)
        end
        isempty(fresh) && break
        push!(newlevels, fresh)
    end
    h.levels = newlevels
    for lv in h.levels, b in lv
        empty!(b.children)
    end
    for ℓ in 2:length(h.levels), b in h.levels[ℓ]
        push!(b.parent.children, b)
    end
    # bring the coarse levels back in sync with the (possibly new) fine data
    for ℓ in length(h.levels):-1:2, b in h.levels[ℓ]
        _restrict_block!(h, b)
    end
    return h
end

"""
    check_proper_nesting(hierarchy; buffer=hierarchy.buffer) -> Bool

`true` when every level-`n+1` block, coarsened to level-`n` index space,
lies inside a level-`n` block with at least `buffer` level-`n` cells of
margin on every side — except on sides where the containing level-`n`
block itself abuts the domain boundary, where the margin is unnecessary
(and impossible).
"""
function check_proper_nesting(h::AMRHierarchy; buffer::Int=h.buffer)
    for ℓ in 2:length(h.levels)
        N = _level_size(h, ℓ-1)
        for b in h.levels[ℓ]
            pb = b.parent
            pb === nothing && return false
            pb.level == ℓ-1 || return false
            cb = _coarse_bounds(h, b)
            for d in 1:3
                # coarsening must be exact (block aligned to parent cells)
                (first(b.bounds[d]) - 1) % h.refinement[d] == 0 || return false
                last(b.bounds[d]) % h.refinement[d] == 0 || return false
                first(cb[d]) >= first(pb.bounds[d]) || return false
                last(cb[d])  <= last(pb.bounds[d])  || return false
                if first(pb.bounds[d]) != 1
                    first(cb[d]) - first(pb.bounds[d]) >= buffer || return false
                end
                if last(pb.bounds[d]) != N[d]
                    last(pb.bounds[d]) - last(cb[d]) >= buffer || return false
                end
            end
        end
    end
    return true
end

# =========================================================================
# Subcycled AMR time stepping
# =========================================================================

"""
`true` when `b` spans the whole domain along dimension `d` at its own level —
then there is no coarse-fine interface in `d`, only a periodic seam.
"""
_full_span(h::AMRHierarchy, b::RefinementBlock, d::Int) =
    length(b.bounds[d]) == _level_size(h, b.level)[d]

_periodic_dim(h::AMRHierarchy, d::Int) =
    d == 1 ? (h.bcs.west isa Periodic) :
    d == 2 ? (h.bcs.south isa Periodic) : (h.bcs.bottom isa Periodic)

# In a direction the block spans completely, the "boundary" is the domain's own
# periodic seam: wrap within the block instead of prolonging the coarse data, so
# the fine fluxes on the two sides of the seam agree exactly.
function _wrap_full_span!(h::AMRHierarchy, b::RefinementBlock)
    (b.level == 1 || b.parent === nothing) && return b
    m = b.mesh
    for v in h.vars
        f = b.state[v]; loc = location(f)
        if _full_span(h, b, 1) && _periodic_dim(h, 1)
            _fill_x!(f.data, Periodic(), Periodic(), m.Hx, m.Nx, loc)
        end
        if _full_span(h, b, 2) && _periodic_dim(h, 2)
            _fill_y!(f.data, Periodic(), Periodic(), m.Hy, m.Ny, loc)
        end
        if _full_span(h, b, 3) && _periodic_dim(h, 3)
            _fill_z!(f.data, Periodic(), Periodic(), m.Hz, m.Nz, loc)
        end
    end
    return b
end

function _fill_block_halos!(h::AMRHierarchy, b::RefinementBlock)
    h.velocity === nothing || _set_block_velocity!(h, b, h.t)
    if b.level == 1 || b.parent === nothing
        for v in h.vars
            fill_halos!(b.state[v], h.bcs)
        end
    else
        for v in h.vars
            interpolate_to_nest_boundary!(b.state[v], b.parent.state[v], b.nest;
                                         order=h.order)
        end
        _wrap_full_span!(h, b)
    end
    return b
end

_restrict_block!(h::AMRHierarchy, b::RefinementBlock) = begin
    b.parent === nothing && return b
    for v in h.vars
        restrict_to_parent!(b.parent.state[v], b.state[v], b.nest)
    end
    b
end

"""
Berger-Colella refluxing of one block into its parent: the coarse cells just
outside the block absorb `coarse flux − Σ fine fluxes` on every coarse-fine
interface plane. Directions in which the block spans the whole (periodic)
domain carry no interface and are skipped.
"""
function _reflux_block!(h::AMRHierarchy, b::RefinementBlock)
    b.parent === nothing && return b
    T = eltype(b.mesh)
    for v in h.vars
        mm = NestFluxMismatch{T}()
        for (ax, sym) in ((1, :x), (2, :y), (3, :z))
            (_full_span(h, b, ax) && _periodic_dim(h, ax)) && continue
            mm = mm + compute_nest_flux_mismatch(b.parent.state[_reg(v, sym)],
                                                 b.state[_reg(v, sym)], b.nest)
        end
        apply_flux_correction!(b.parent.state[v], mm, b.nest;
                               periodic = nest_periodic_axes(h.bcs))
    end
    return b
end

function _advance_level!(h::AMRHierarchy, ℓ::Int, dt::Real)
    for b in h.levels[ℓ]
        _fill_block_halos!(h, b)
        h.advance!(h, b, dt, h.t)
    end
    if ℓ < length(h.levels) && !isempty(h.levels[ℓ+1])
        r = maximum(h.refinement)
        for b in h.levels[ℓ+1]
            _zero_registers!(h, b)
        end
        for _ in 1:r
            _advance_level!(h, ℓ+1, dt / r)
        end
        for b in h.levels[ℓ+1]
            _restrict_block!(h, b)
            _reflux_block!(h, b)
        end
    end
    return h
end

"""
    step_amr!(hierarchy, dt; regrid_interval=10, criterion=nothing)

One coarse step of the recursive, subcycled AMR advance:

  * advance every level-`ℓ` block by `dt`, accumulating face fluxes into
    the flux registers;
  * take `r = max(refinement)` substeps of level `ℓ+1` with `dt/r`, filling
    the fine halos from the coarse level by conservative prolongation;
  * **restrict** the fine interior onto the coarse cells and **reflux** the
    coarse-fine flux mismatch into the coarse cells just outside each fine
    block.

That last pair is what makes the composite integral exactly invariant:
the coarse level was advanced with coarse fluxes across the interface
while the fine level used the fine ones, and refluxing removes the
difference.

Regrids every `regrid_interval` coarse steps when `criterion !== nothing`
(pass `regrid_interval=0` to disable).
"""
function step_amr!(h::AMRHierarchy, dt::Real; regrid_interval::Integer=10,
                   criterion=nothing)
    if criterion !== nothing && regrid_interval > 0 && h.nstep % regrid_interval == 0
        regrid!(h, criterion)
    end
    for b in h.levels[1]
        _zero_registers!(h, b)
    end
    _advance_level!(h, 1, dt)
    h.t += dt
    h.nstep += 1
    return h
end

# =========================================================================
# AMR diagnostics
# =========================================================================

function _block_covered(h::AMRHierarchy, b::RefinementBlock)
    m = b.mesh
    cov = falses(m.Nx, m.Ny, m.Nz)
    for c in b.children
        cb = _coarse_bounds(h, c)
        lr = ntuple(d -> (first(cb[d]) - first(b.bounds[d]) + 1) :
                         (last(cb[d]) - first(b.bounds[d]) + 1), 3)
        cov[lr[1], lr[2], lr[3]] .= true
    end
    return cov
end

"""
    amr_total_mass(hierarchy; var=:θ) -> Float64

Composite volume integral of `var`: every cell of every block that is not
covered by a finer block, weighted by that level's cell volume. This is the
quantity [`step_amr!`](@ref) conserves exactly.
"""
function amr_total_mass(h::AMRHierarchy; var::Symbol=:θ)
    tot = 0.0
    for ℓ in 1:length(h.levels)
        d = _level_spacing(h, ℓ); V = d[1] * d[2] * d[3]
        for b in h.levels[ℓ]
            cov = _block_covered(h, b)
            m = b.mesh; θd = b.state[var].data
            @inbounds for k in 1:m.Nz, j in 1:m.Ny, i in 1:m.Nx
                cov[i,j,k] && continue
                tot += θd[m.Hx+i, m.Hy+j, m.Hz+k] * V
            end
        end
    end
    return tot
end

"""
    amr_error(hierarchy, exact; var=:θ) -> Float64

Volume-weighted composite L1 error of `var` against
`exact(x, y, z)`, evaluated on each block's uncovered cells at that
level's resolution.
"""
function amr_error(h::AMRHierarchy, exact; var::Symbol=:θ)
    num = 0.0; vol = 0.0
    for ℓ in 1:length(h.levels)
        d = _level_spacing(h, ℓ); V = d[1] * d[2] * d[3]
        for b in h.levels[ℓ]
            cov = _block_covered(h, b)
            m = b.mesh; θd = b.state[var].data
            @inbounds for k in 1:m.Nz, j in 1:m.Ny, i in 1:m.Nx
                cov[i,j,k] && continue
                e = θd[m.Hx+i, m.Hy+j, m.Hz+k] - exact(_xc(m,m.Hx+i), _yc(m,m.Hy+j),
                                                       _zc(m,m.Hz+k))
                num += abs(e) * V; vol += V
            end
        end
    end
    return vol > 0 ? num / vol : 0.0
end

end # module
