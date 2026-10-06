# N-open-b (#255) — open (Dirichlet-from-boundary-data) lateral and top
# faces, the pluggable boundary-data contract and the single global
# boundary mass-flux correction.
#
# Included inside `module BoundaryConditions`, after noslip_wall.jl (it
# uses `_ZWall` and the wall halo fills) and before free_stream.jl (the
# free-stream top is a special case of the kernels defined here).

# =========================================================================
# Face geometry
# =========================================================================

"""
    AbstractOpenBoundary <: AbstractBC

A face on which the **normal velocity is prescribed** from outside the
domain and the projection must leave it unchanged (`axis_poisson_bc` is
`:neumann`). Concrete types: [`OpenBoundary`](@ref) (data-driven
Dirichlet faces, lateral or top) and [`FreeStreamBoundary`](@ref) (the
transpiration top of a spatial boundary layer, a special case with a
fixed normal-velocity profile and zero-gradient tangential halos). Both
are filled by the same face kernels.
"""
abstract type AbstractOpenBoundary <: AbstractBC end

# side → (axis, high?)
_side_axis(s::Symbol) = s === :west  ? (1, false) : s === :east  ? (1, true) :
                        s === :south ? (2, false) : s === :north ? (2, true) :
                        s === :bottom ? (3, false) : s === :top  ? (3, true) :
                        throw(ArgumentError("unknown face side :$s (use :west, :east, " *
                                            ":south, :north, :top)"))

# The location whose component is normal to an axis.
_normal_loc(::FaceX) = 1
_normal_loc(::FaceY) = 2
_normal_loc(::FaceZ) = 3
_normal_loc(::Location) = 0

# A view of the storage slab `r` along axis `ax` (all of the other two).
@inline _slab(d, ax::Int, r) = ax == 1 ? view(d, r, :, :) :
                               ax == 2 ? view(d, :, r, :) : view(d, :, :, r)

# Storage-index range of the slab an open face writes: the H halo layers
# and, for the normal component on a low (west/south) face, the boundary
# face itself, which is interior index H+1 (on a high face the boundary
# face, index H+N+1, is the first halo layer).
_open_range(high::Bool, normal::Bool, H::Int, N::Int) =
    high ? (H + N + 1 : N + 2H) : (normal ? (1 : H + 1) : (1 : H))

# Storage index of the boundary face of the normal component.
_face_index(high::Bool, H::Int, N::Int) = high ? H + N + 1 : H + 1

# --- the three face kernels (shared by OpenBoundary and FreeStreamBoundary) ---

# Zero-gradient (mirror) halo — the Neumann condition. Used for φ and every
# field without boundary data (Kim & Moin 1985), and for the tangential
# fields of the free-stream top.
function _open_mirror!(d, ax::Int, high::Bool, H::Int, N::Int)
    @inbounds for h in 1:H
        dst, src = high ? (H + N + h, H + N + 1 - h) : (H + 1 - h, H + h)
        _slab(d, ax, dst) .= _slab(d, ax, src)
    end
    return d
end

# Halo beyond the normal face, antisymmetric about the face value
# (linear extrapolation, d[f±h] = 2 d[f] − d[f∓h]). The face must be set.
function _open_odd!(d, ax::Int, high::Bool, H::Int, N::Int)
    f = _face_index(high, H, N)
    f <= size(d, ax) || return d
    @inbounds for h in 1:H
        dst, src = high ? (f + h, f - h) : (f - h, f + h)
        1 <= dst <= size(d, ax) || continue
        _slab(d, ax, dst) .= 2 .* _slab(d, ax, f) .- _slab(d, ax, src)
    end
    return d
end

# =========================================================================
# Plane buffers
# =========================================================================

"""
    OpenPlane

The boundary data of one field on one open face: the storage slab the
face writes (`data`, shaped like `field.data[range, :, :]` for an
x face, etc.), its storage index ranges `irange, jrange, krange` and the
physical coordinates `x, y, z` of those storage points, so an
[`AbstractBoundaryData`](@ref) can evaluate itself on the slab.

`mode` is how the halo is filled:

* tangential velocity and scalars: `:value` (Dirichlet: copied from
  `data`) or `:mirror` (zero gradient; `data` unused);
* the normal velocity: the boundary face is always copied from `data`;
  the layers beyond it are `:value` (copied) or `:odd` (antisymmetric
  about the face value).

`fluid[a, b]` (over the interior tangential indices of the face) marks
the faces that carry flow; the mass-flux correction only adjusts and
counts fluid faces. Default: all fluid.
"""
struct OpenPlane{T}
    name::Symbol
    side::Symbol
    axis::Int
    high::Bool
    normal::Bool
    mode::Symbol
    range::UnitRange{Int}
    irange::UnitRange{Int}
    jrange::UnitRange{Int}
    krange::UnitRange{Int}
    x::Vector{T}
    y::Vector{T}
    z::Vector{T}
    data::Array{T,3}
    fluid::BitMatrix
end

# Coordinate of storage index `i` along an axis, for a field that is (or is
# not) staggered along that axis.
_coord(x0, Δ, H, i, staggered::Bool) = x0 + (i - H - (staggered ? 1 : 0.5)) * Δ

function OpenPlane(name::Symbol, loc::Location, side::Symbol, m::RegularGrid{T};
                   mode::Symbol = :value) where {T}
    ax, high = _side_axis(side)
    normal = _normal_loc(loc) == ax
    if normal
        mode in (:value, :odd) ||
            throw(ArgumentError("normal-velocity halo mode must be :value or :odd, got :$mode"))
    else
        mode in (:value, :mirror) ||
            throw(ArgumentError("halo mode must be :value or :mirror, got :$mode"))
    end
    Ns = (m.Nx, m.Ny, m.Nz); Hs = (m.Hx, m.Hy, m.Hz)
    full = (1:m.Nx + 2m.Hx, 1:m.Ny + 2m.Hy, 1:m.Nz + 2m.Hz)
    r = _open_range(high, normal, Hs[ax], Ns[ax])
    rr = ntuple(a -> a == ax ? r : full[a], 3)
    st = ntuple(a -> _normal_loc(loc) == a, 3)
    x = T[_coord(m.x0, m.dx, m.Hx, i, st[1]) for i in rr[1]]
    y = T[_coord(m.y0, m.dy, m.Hy, j, st[2]) for j in rr[2]]
    z = T[_coord(m.z0, m.dz, m.Hz, k, st[3]) for k in rr[3]]
    data = zeros(T, length.(rr)...)
    tang = ntuple(a -> Ns[a], 3)
    fl = ax == 1 ? trues(tang[2], tang[3]) : ax == 2 ? trues(tang[1], tang[3]) :
                                                       trues(tang[1], tang[2])
    return OpenPlane{T}(name, side, ax, high, normal, mode, r, rr..., x, y, z, data, fl)
end

# Local index of the boundary face inside a normal plane's slab.
_face_local(p::OpenPlane) = p.high ? 1 : length(p.range)

# =========================================================================
# The boundary-data contract
# =========================================================================

"""
    AbstractBoundaryData

Where an open boundary's values come from. A subtype implements

    boundary_values!(planes, data, t)

which writes, for every [`OpenPlane`](@ref) `p` in `planes` (a vector
over all fields and faces of one open boundary set) whose `p.mode !==
:mirror`, the value of field `p.name` at time `t` into `p.data`, at the
storage points with coordinates `(p.x[a], p.y[b], p.z[c])`. The normal
velocity's boundary-face values are then made mass consistent by
[`correct_boundary_mass_flux!`](@ref); `boundary_values!` does not need to
balance them.

Shipped: [`ProfileBoundaryData`](@ref) (horizontally uniform profiles)
and [`FunctionBoundaryData`](@ref) (analytic fields). The nest boundary
data of the Berlin child (N-nest-d, #261), mesoscale forcing (#58) and
ERA5 (#57) plug in here.
"""
abstract type AbstractBoundaryData end

"""
    boundary_values!(planes, data::AbstractBoundaryData, t) -> planes

Fill the open-boundary buffers `planes` from `data` at time `t`; see
[`AbstractBoundaryData`](@ref).
"""
function boundary_values! end

"""
    FunctionBoundaryData(; u = f, v = f, w = f, θ = f, …)

Analytic boundary data: each keyword is a function `(x, y, z, t) ->
value` (or a number) for the field of that name. A field with an open
plane but no function throws, except `w`, which defaults to zero.
"""
struct FunctionBoundaryData{F<:NamedTuple} <: AbstractBoundaryData
    fields::F
end
FunctionBoundaryData(; kwargs...) = FunctionBoundaryData((; kwargs...))

"""
    ProfileBoundaryData(; u = U, v = V, w = 0, θ = θ(z), …)

Horizontally uniform, steady boundary data: each keyword is a number, a
function of height `z -> value`, or a vector indexed by the **storage**
`z` index (length `Nz + 2Hz`, e.g. the halo-padded reference profile
`ref.θ̄c_pad` of a sound-proof core). A field with an open plane but no
profile throws, except `w`, which defaults to zero.
"""
struct ProfileBoundaryData{F<:NamedTuple} <: AbstractBoundaryData
    fields::F
end
ProfileBoundaryData(; kwargs...) = ProfileBoundaryData((; kwargs...))

function _data_for(fields::NamedTuple, name::Symbol, kind)
    haskey(fields, name) && return getfield(fields, name)
    name === :w && return 0
    throw(ArgumentError("$kind has no data for the open-boundary field :$name " *
                        "(it provides $(keys(fields)))"))
end

function boundary_values!(planes::AbstractVector{<:OpenPlane}, d::FunctionBoundaryData, t)
    for p in planes
        p.mode === :mirror && continue
        _fill_function!(p, _data_for(d.fields, p.name, "FunctionBoundaryData"), t)
    end
    return planes
end

function _fill_function!(p::OpenPlane{T}, f, t) where {T}
    if f isa Real
        fill!(p.data, T(f))
    else
        @inbounds for c in axes(p.data, 3), b in axes(p.data, 2), a in axes(p.data, 1)
            p.data[a, b, c] = T(f(p.x[a], p.y[b], p.z[c], t))
        end
    end
    return p
end

function boundary_values!(planes::AbstractVector{<:OpenPlane}, d::ProfileBoundaryData, t)
    for p in planes
        p.mode === :mirror && continue
        _fill_profile!(p, _data_for(d.fields, p.name, "ProfileBoundaryData"))
    end
    return planes
end

_profile_at(f::Real, z, k) = f
_profile_at(f::AbstractVector, z, k) = f[k]
_profile_at(f, z, k) = f(z)

function _fill_profile!(p::OpenPlane{T}, f) where {T}
    f isa AbstractVector && length(f) < last(p.krange) &&
        throw(DimensionMismatch("profile for :$(p.name) has $(length(f)) entries; a storage-" *
                                "indexed profile needs at least $(last(p.krange))"))
    @inbounds for (c, k) in enumerate(p.krange)
        v = T(_profile_at(f, p.z[c], k))
        p.data[:, :, c] .= v
    end
    return p
end

# =========================================================================
# The shared state of one open-boundary set, and the face type
# =========================================================================

"""
    RadiationPlane

The outflow state of one [`OpenPlane`](@ref) on a face whose outflow mode
is `:convective` or `:orlanski` (N-open-c, #260; see
[`open_boundary_conditions`](@ref)):

* `mode` — `:convective` or `:orlanski`;
* `normal` — index, in the state's `planes`, of the face's normal-velocity
  plane (whose boundary-face values decide inflow or outflow);
* `prev` — the buffer before the refresh of the current (sub)step;
* `c` — the phase speed (m/s, ≥ 0) at every tangential point of the
  plane, fixed at the start of each step by
  [`prepare_open_boundaries!`](@ref); `NaN` marks an inflow point, which
  keeps the Dirichlet data;
* `hist`, `t_hist`, `has_hist` — the interior layer next to the face at
  the previous step start, from which the Orlanski phase speed is
  estimated;
* `start1`, `start2`, `marked` — the two interior layers next to the face
  at the start of the current (sub)stage, recorded by
  [`mark_open_boundary_stage!`](@ref) (the update uses them when
  `marked`, the current interior otherwise).
"""
struct RadiationPlane{T}
    mode::Symbol
    normal::Int
    prev::Array{T,3}
    c::Matrix{T}
    hist::Matrix{T}
    t_hist::Base.RefValue{T}
    has_hist::Base.RefValue{Bool}
    start1::Matrix{T}
    start2::Matrix{T}
    marked::Base.RefValue{Bool}
end

"""
    OpenBoundaryState

What the faces of one open boundary share: the boundary data, every
[`OpenPlane`](@ref) of every face (`planes`, the buffers handed to
[`boundary_values!`](@ref)), the mass-flux correction mode, the binding
of cell-centred fields to their planes, and the ledger of the last
correction: `Q` (net outward volume, or mass, flux before the
correction), `residual` (after), `δ` (the uniform normal-velocity
adjustment) and, with `record = true`, `log`, one `(t, Q, residual)` per
update.

`outflow` maps each open side to its outflow mode (`:dirichlet`,
`:convective`, `:orlanski`); `radiation[n]` is the
[`RadiationPlane`](@ref) of `planes[n]` on a radiating face, `nothing`
otherwise (N-open-c, #260).
"""
mutable struct OpenBoundaryState{T,M<:RegularGrid{T}}
    mesh::M
    data::Any
    correction::Symbol
    planes::Vector{OpenPlane{T}}
    bound::IdDict{Any,Symbol}
    initialized::Bool
    t::T
    Q::T
    residual::T
    δ::T
    record::Bool
    log::Vector{NTuple{3,T}}
    outflow::Dict{Symbol,Symbol}
    radiation::Vector{Union{Nothing,RadiationPlane{T}}}
end

# The N-open-b positional form (every face Dirichlet).
OpenBoundaryState{T,M}(mesh, data, correction, planes, bound, initialized, t, Q, residual, δ,
                       record, log) where {T,M} =
    OpenBoundaryState{T,M}(mesh, data, correction, planes, bound, initialized, t, Q,
                           residual, δ, record, log, Dict{Symbol,Symbol}(),
                           Union{Nothing,RadiationPlane{T}}[])

"""
    OpenBoundary

An open face (`:west`, `:east`, `:south`, `:north` or `:top`) whose
values come from boundary data (PALM's Dirichlet child boundary;
Hellsten et al. 2021). On every halo fill:

* the **normal velocity** is written to the boundary-face slot — interior
  index `H+1` on a west/south face, halo index `H+N+1` on an east, north
  or top face, the slots `ImpermeableWall` uses for `w = 0` — and the
  layers beyond it are copied from the buffer (or antisymmetric about the
  face, `normal_halo = :odd`);
* **tangential velocities and scalars** with data get Dirichlet halos
  copied from the buffer (the boundary data at the halo-cell positions);
* every other field — the pressure correction `φ`, `p`, `νₜ`, a scalar
  without data — gets the mirror (zero-gradient) halo, i.e. the Neumann
  condition `∂φ/∂n = 0` that keeps a prescribed normal velocity unchanged
  by the projection (Kim & Moin 1985).

`fill_halos!` only copies, so it is idempotent. The buffers are refreshed
once per projected (sub)step by [`update_open_boundaries!`](@ref), which
also applies the global mass-flux correction.

Build a whole box with [`open_boundary_conditions`](@ref). Cell-centred
fields are matched to their planes by identity (`sim.θ`, `sim.qv`, bound
by `update_open_boundaries!`); a velocity component is matched by its
location.

References: Kim, J. & Moin, P. (1985), Application of a fractional-step
method to incompressible Navier–Stokes equations, *J. Comput. Phys.* 59,
308–323. Hellsten, A. et al. (2021), A nested multi-scale system
implemented in the large-eddy simulation model PALM model system 6.0,
*Geosci. Model Dev.* 14, 3185–3214. Kadasch, E. et al. (2021), Mesoscale
nesting interface of the PALM model system 6.0, *Geosci. Model Dev.* 14,
5435–5465.
"""
struct OpenBoundary{T,S<:OpenBoundaryState{T}} <: AbstractOpenBoundary
    side::Symbol
    planes::Dict{Symbol,OpenPlane{T}}
    state::S
    homogeneous::Bool
end

_loc_of(name::Symbol) = name === :u ? FaceX() : name === :v ? FaceY() :
                        name === :w ? FaceZ() : CellCenter()

"""
    open_boundary_conditions(mesh, data; x = :open, y = :open,
                             bottom = :wall, top = :open,
                             fields = (:u, :v, :w, :θ),
                             correction = :uniform, normal_halo = :value,
                             scalar_halo = :value, record = false,
                             outflow = :dirichlet, halo = 1)
        -> FieldBoundaryConditions

A box with open faces fed by `data` (an [`AbstractBoundaryData`](@ref)).
`x` and `y` are `:open` or `:periodic` (both faces of an axis alike);
`top` is `:open` or a wall (`:wall`, `:noslip`, `NoSlipWall`); `bottom`
is a wall, or — for a nest child that does not reach the ground
(N-nest-d, #261) — `:open` with `top = :open`, or `:periodic` with
`top = :periodic` (a child spanning its parent's periodic z extent). `fields` names the fields with boundary data: the three
velocity components and any cell-centred scalar the simulation holds
under that name (`:θ`, `:qv`). `correction` is the mode of
[`correct_boundary_mass_flux!`](@ref) (`:uniform`, `:outflow_only` or
`:none`). `scalar_halo = :mirror` gives the scalars zero-gradient halos
instead of Dirichlet ones.

All open faces share one [`OpenBoundaryState`](@ref); pass the result as
`bcs` (and, by default, `θ_bcs`) of a simulation.

`outflow` (N-open-c, #260) is the outflow condition of the open faces:
one of `:dirichlet` (default: every point takes the boundary data, PALM's
nested child), `:convective` or `:orlanski`, or a `NamedTuple` per side,
e.g. `(east = :convective,)` (unnamed sides `:dirichlet`). On a radiating
face, a point whose outward normal velocity is positive at the start of
the step is an **outflow point**: every field with a plane there (u, v,
w and each scalar) is advanced by the radiation equation
`∂φ/∂t + c ∂φ/∂n = 0` in the halo, with `c = max(u·n, 0)` (convective) or
the Orlanski (1976) phase speed clipped to `[0, Δn/Δt]` (Miller & Thorpe
1981); **inflow points** keep the Dirichlet data. See
[`prepare_open_boundaries!`](@ref) and [`update_open_boundaries!`](@ref).
In `:orlanski` mode the normal velocity is advanced with the convective
speed (see [`prepare_open_boundaries!`](@ref) and the theory page,
"Radiation outflow").

`halo` is the halo depth the advection scheme needs at an open face
(`AdvectionSchemes.required_halo(scheme)`: 3 for WENO-Z5 and TENO5, 4 for
WENO-Z7); a mesh whose halo along an open axis is shallower throws an
`ArgumentError`, since the scheme would then read halo layers that hold
no boundary data. Inflow scalars get Dirichlet data in all `H` layers.
"""
function open_boundary_conditions(m::RegularGrid{T}, data;
                                  x = :open, y = :open, bottom = :wall, top = :open,
                                  fields = (:u, :v, :w, :θ),
                                  correction::Symbol = :uniform,
                                  normal_halo::Symbol = :value,
                                  scalar_halo::Symbol = :value,
                                  record::Bool = false,
                                  outflow = :dirichlet,
                                  halo::Integer = 1) where {T}
    correction in (:uniform, :outflow_only, :none) ||
        throw(ArgumentError("correction must be :uniform, :outflow_only or :none, got :$correction"))
    (:u in fields && :v in fields && :w in fields) ||
        throw(ArgumentError("an open boundary needs data for :u, :v and :w"))
    x in (:open, :periodic) || throw(ArgumentError("x must be :open or :periodic"))
    y in (:open, :periodic) || throw(ArgumentError("y must be :open or :periodic"))
    st = OpenBoundaryState{T,typeof(m)}(m, data, correction, OpenPlane{T}[],
                                        IdDict{Any,Symbol}(), false,
                                        zero(T), zero(T), zero(T), zero(T),
                                        record, NTuple{3,T}[])
    function face(side)
        ax, _ = _side_axis(side)
        Hs = (m.Hx, m.Hy, m.Hz)
        Hs[ax] >= halo ||
            throw(ArgumentError("open face :$side: the mesh halo along axis $ax is " *
                                "$(Hs[ax]), shallower than the $halo layers the advection " *
                                "scheme needs (required_halo); build the mesh with " *
                                "halo ≥ $halo"))
        st.outflow[side] = _outflow_mode(outflow, side)
        d = Dict{Symbol,OpenPlane{T}}()
        for name in fields
            loc = _loc_of(name)
            mode = _normal_loc(loc) == ax ? normal_halo :
                   loc isa CellCenter ? scalar_halo : :value
            p = OpenPlane(name, loc, side, m; mode = mode)
            d[name] = p
            push!(st.planes, p)
        end
        return OpenBoundary{T,typeof(st)}(side, d, st, false)
    end
    w_ = x === :open ? face(:west)  : Periodic(); e_ = x === :open ? face(:east)  : Periodic()
    s_ = y === :open ? face(:south) : Periodic(); n_ = y === :open ? face(:north) : Periodic()
    if bottom === :open || bottom === :periodic
        top === bottom || throw(ArgumentError("bottom = :$bottom needs top = :$bottom"))
    end
    if bottom === :periodic
        return FieldBoundaryConditions(w_, e_, s_, n_, Periodic(), Periodic())
    end
    b_ = bottom === :open ? face(:bottom) : _bc(bottom)
    (b_ isa OpenBoundary || is_wall(b_)) ||
        throw(ArgumentError("the bottom of an open box must be a wall, :open or :periodic"))
    t_ = top === :open ? face(:top) : _bc(top)
    (t_ isa OpenBoundary || is_wall(t_)) ||
        throw(ArgumentError("top must be :open or a wall"))
    _check_outflow_sides(outflow, keys(st.outflow))
    _init_radiation!(st)
    return FieldBoundaryConditions(w_, e_, s_, n_, b_, t_)
end

"""
    set_boundary_data!(bcs, data) -> bcs

Replace the boundary data of the open boundary set in `bcs` (e.g. once
the parent of a nest exists). The buffers are refreshed at the next
[`update_open_boundaries!`](@ref).
"""
function set_boundary_data!(bcs::FieldBoundaryConditions, data)
    for st in open_boundary_states(bcs)
        st.data = data
        st.initialized = false
    end
    return bcs
end

# --- queries ---------------------------------------------------------------

_faces(b::FieldBoundaryConditions) = (b.west, b.east, b.south, b.north, b.bottom, b.top)

"""
    has_open_boundaries(bcs::FieldBoundaryConditions) -> Bool
    has_open_boundaries(sim) -> Bool

Whether any face is an [`OpenBoundary`](@ref) (in `sim.bcs` or
`sim.θ_bcs`). False for every closed configuration, which is what keeps
the open-boundary hooks of the dynamical cores no-ops there.
"""
has_open_boundaries(b::FieldBoundaryConditions) = any(f -> f isa OpenBoundary, _faces(b))
has_open_boundaries(sim) = has_open_boundaries(sim.bcs) || has_open_boundaries(sim.θ_bcs)

"""
    open_boundary_states(bcs_or_sim) -> Vector{OpenBoundaryState}

The distinct open-boundary sets of a boundary container (or of a
simulation's `bcs` and `θ_bcs`).
"""
function open_boundary_states(b::FieldBoundaryConditions)
    out = OpenBoundaryState[]
    for f in _faces(b)
        f isa OpenBoundary && !any(s -> s === f.state, out) && push!(out, f.state)
    end
    return out
end
function open_boundary_states(sim)
    out = open_boundary_states(sim.bcs)
    for s in open_boundary_states(sim.θ_bcs)
        any(x -> x === s, out) || push!(out, s)
    end
    return out
end

# =========================================================================
# Halo fills
# =========================================================================

# The plane of the field being filled, or nothing (→ mirror). Velocity
# components by location; cell-centred fields by identity of their array.
_plane_for(bc::OpenBoundary, ::FaceX, key) = get(bc.planes, :u, nothing)
_plane_for(bc::OpenBoundary, ::FaceY, key) = get(bc.planes, :v, nothing)
_plane_for(bc::OpenBoundary, ::FaceZ, key) = get(bc.planes, :w, nothing)
function _plane_for(bc::OpenBoundary, ::CellCenter, key)
    name = get(bc.state.bound, key, nothing)
    name === nothing && return nothing
    return get(bc.planes, name, nothing)
end

function _fill_open_face!(d::AbstractArray{T,3}, bc::OpenBoundary, H::Int, N::Int,
                          loc::Location, key) where {T}
    ax, high = _side_axis(bc.side)
    p = _plane_for(bc, loc, key)
    if p === nothing || p.mode === :mirror
        # φ, p, νₜ and fields without data: Neumann. A normal component
        # without data cannot happen (open_boundary_conditions requires u, v, w).
        return _open_mirror!(d, ax, high, H, N)
    end
    s = _slab(d, ax, p.range)
    if p.normal && p.mode === :odd
        f = _face_index(high, H, N)
        fl = _face_local(p)
        if bc.homogeneous
            _slab(d, ax, f) .= zero(T)
        else
            _slab(d, ax, f) .= _slab(p.data, ax, fl)
        end
        return _open_odd!(d, ax, high, H, N)
    end
    if bc.homogeneous
        fill!(s, zero(T))
    else
        s .= p.data
    end
    return d
end

# Pairs on x and y: two open faces (an inflow/outflow axis).
function _fill_x!(d, w::OpenBoundary, e::OpenBoundary, Hx, Nx, loc::Location, key)
    _fill_open_face!(d, w, Hx, Nx, loc, key)
    _fill_open_face!(d, e, Hx, Nx, loc, key)
    return d
end
function _fill_y!(d, s::OpenBoundary, n::OpenBoundary, Hy, Ny, loc::Location, key)
    _fill_open_face!(d, s, Hy, Ny, loc, key)
    _fill_open_face!(d, n, Hy, Ny, loc, key)
    return d
end
# z: a wall below an open top.
function _fill_z!(d, bot::_ZWall, top::OpenBoundary, Hz, Nz, loc::Location, key)
    _fill_wall_bottom!(d, bot, Hz, loc)
    _fill_open_face!(d, top, Hz, Nz, loc, key)
    return d
end
# z: open below and above — a nest child that sits inside its parent
# (N-nest-d, #261), not on the ground.
function _fill_z!(d, bot::OpenBoundary, top::OpenBoundary, Hz, Nz, loc::Location, key)
    _fill_open_face!(d, bot, Hz, Nz, loc, key)
    _fill_open_face!(d, top, Hz, Nz, loc, key)
    return d
end

# =========================================================================
# Solid obstacles at an open face (N-open-d, #262)
# =========================================================================

"""
    mark_open_boundary_solids!(bcs, mask) -> bcs

Tell the open faces of `bcs` where a solid obstacle (an immersed-boundary
building, `mask` a storage-sized `Bool` array, `true` = solid, e.g.
`IBM.SolidMask(...).mask`) touches them. For every plane of every open
face, a boundary point is **solid** when an interior cell it borders is
solid — the cell next to the face for the normal velocity and the
scalars, and either of the two cells a staggered tangential component
sits between — exactly the faces the direct-forcing IBM zeroes
(`IBM.apply_solid_mask!`). Solid points:

* take the **solid value** of the velocity, zero, in every halo layer,
  written by [`update_open_boundaries!`](@ref) after the boundary data —
  the data of a building cut by the box edge is never imposed on its
  inside, and the projection sees an impermeable face there;
* are excluded from the global mass-flux correction
  ([`correct_boundary_mass_flux!`](@ref), N-open-b): neither counted in
  `Q`'s area nor adjusted by `δ`, so the correction cannot push flow
  through a wall.

Scalars keep their data at solid points: with zero normal velocity no
advective flux crosses there, and the diffusive exchange with the solid
cell is the same as anywhere between air and a building. Call once after
the geometry is built; the buffers are refreshed at the next
[`update_open_boundaries!`](@ref).
"""
function mark_open_boundary_solids!(bcs::FieldBoundaryConditions, mask::AbstractArray{Bool,3})
    for st in open_boundary_states(bcs)
        m = st.mesh
        size(mask) == (m.Nx + 2m.Hx, m.Ny + 2m.Hy, m.Nz + 2m.Hz) ||
            throw(DimensionMismatch("mark_open_boundary_solids!: the mask must have the storage " *
                                    "size of the mesh, $((m.Nx + 2m.Hx, m.Ny + 2m.Hy, m.Nz + 2m.Hz))"))
        for p in st.planes
            _mark_solid!(p, m, mask)
        end
        st.initialized = false
    end
    return bcs
end

function _mark_solid!(p::OpenPlane, m, mask)
    Hs = (m.Hx, m.Hy, m.Hz); Ns = (m.Nx, m.Ny, m.Nz)
    ax = p.axis
    t1, t2 = ax == 1 ? (2, 3) : ax == 2 ? (1, 3) : (1, 2)
    loc = _loc_of(p.name)
    nl = _normal_loc(loc)
    cell = p.high ? Hs[ax] + Ns[ax] : Hs[ax] + 1      # the interior cell next to the face
    @inbounds for b in 1:Ns[t2], a in 1:Ns[t1]
        solid = false
        for da in (nl == t1 ? (0, -1) : (0,)), db in (nl == t2 ? (0, -1) : (0,))
            I = [0, 0, 0]
            I[ax] = cell; I[t1] = Hs[t1] + a + da; I[t2] = Hs[t2] + b + db
            solid |= mask[I[1], I[2], I[3]]
        end
        p.fluid[a, b] = !solid
    end
    return p
end

# Zero every velocity buffer at the solid points of its plane.
function _apply_solid_values!(st::OpenBoundaryState{T}) where {T}
    m = st.mesh
    Hs = (m.Hx, m.Hy, m.Hz)
    for p in st.planes
        _loc_of(p.name) isa CellCenter && continue
        all(p.fluid) && continue
        t1, t2 = p.axis == 1 ? (2, 3) : p.axis == 2 ? (1, 3) : (1, 2)
        @inbounds for b in axes(p.fluid, 2), a in axes(p.fluid, 1)
            p.fluid[a, b] && continue
            A, B = Hs[t1] + a, Hs[t2] + b
            for c in 1:length(p.range)
                if p.axis == 1
                    p.data[c, A, B] = zero(T)
                elseif p.axis == 2
                    p.data[A, c, B] = zero(T)
                else
                    p.data[A, B, c] = zero(T)
                end
            end
        end
    end
    return st
end

# =========================================================================
# Homogeneous part (implicit solvers) and constraint rows
# =========================================================================

_inhomogeneous(bc::OpenBoundary) = !bc.homogeneous
_homogeneous(bc::OpenBoundary{T,S}) where {T,S} =
    OpenBoundary{T,S}(bc.side, bc.planes, bc.state, true)

"""
    pinned_face_indices(loc, mesh, bcs) -> (ic, jc, kc)

Storage indices of the planes of a field at `loc` whose values a
boundary condition fixes although they lie inside the interior index
range: the boundary face of the normal velocity on a west/south open
face (`ic`/`jc = H+1`) and on a bottom wall (`kc = Hz+1`). Zero means
none. Implicit solvers remove these rows from the unknowns.
"""
pinned_face_indices(::Location, m::RegularGrid, b::FieldBoundaryConditions) = (0, 0, 0)
pinned_face_indices(::FaceX, m::RegularGrid, b::FieldBoundaryConditions) =
    (b.west isa AbstractOpenBoundary ? m.Hx + 1 : 0, 0, 0)
pinned_face_indices(::FaceY, m::RegularGrid, b::FieldBoundaryConditions) =
    (0, b.south isa AbstractOpenBoundary ? m.Hy + 1 : 0, 0)
pinned_face_indices(::FaceZ, m::RegularGrid, b::FieldBoundaryConditions) =
    (0, 0, (is_wall(b.bottom) || b.bottom isa AbstractOpenBoundary) ? m.Hz + 1 : 0)

# =========================================================================
# The global boundary mass-flux correction
# =========================================================================

"""
    boundary_flux_weights(sim) -> nothing | (Wc, Wf)

The weights of the constraint `∇·(W u) = 0` a dynamical core projects
onto, as halo-padded cell and face profiles: `nothing` (`W = 1`) for the
Boussinesq core; the sound-proof cores extend it with their `ρ̄` /
`ρ̄θ̄`. The mass-flux correction weights lateral face fluxes with
`Wc[k]` and the top flux with `Wf[Hz+Nz+1]`, exactly as the projection's
divergence does.
"""
boundary_flux_weights(sim) = nothing

@inline _wlat(::Nothing, k) = 1
@inline _wlat(W::Tuple, k) = W[1][k]
@inline _wtop(::Nothing, k) = 1
@inline _wtop(W::Tuple, k) = W[2][k]

# Visit every fluid interior face of a normal plane:
# g(value_ref_indices..., outward sign, weight·area).
function _foreach_face(g, p::OpenPlane{T}, m::RegularGrid{T}, W) where {T}
    fl = _face_local(p)
    s = p.high ? one(T) : -one(T)
    if p.axis == 1
        A = m.dy * m.dz
        for (b, k) in enumerate(m.Hz+1:m.Hz+m.Nz), (a, j) in enumerate(m.Hy+1:m.Hy+m.Ny)
            p.fluid[a, b] || continue
            g(fl, j, k, s, T(_wlat(W, k)) * A)
        end
    elseif p.axis == 2
        A = m.dx * m.dz
        for (b, k) in enumerate(m.Hz+1:m.Hz+m.Nz), (a, i) in enumerate(m.Hx+1:m.Hx+m.Nx)
            p.fluid[a, b] || continue
            g(i, fl, k, s, T(_wlat(W, k)) * A)
        end
    else
        A = m.dx * m.dy
        kt = p.high ? m.Hz + m.Nz + 1 : m.Hz + 1      # top face, or an open bottom (nest)
        wt = T(_wtop(W, kt))
        for (b, j) in enumerate(m.Hy+1:m.Hy+m.Ny), (a, i) in enumerate(m.Hx+1:m.Hx+m.Nx)
            p.fluid[a, b] || continue
            g(i, j, fl, s, wt * A)
        end
    end
    return nothing
end

"""
    open_boundary_flux(state, weights = nothing) -> Q

Net **outward** volume flux `Q = Σ u·n A` (mass flux `Σ W u·n A` with
`weights = (Wc, Wf)`) through the fluid faces of the open boundary set,
from the boundary-face values in its buffers.
"""
function open_boundary_flux(st::OpenBoundaryState{T}, W = nothing) where {T}
    Q = zero(T)
    for p in st.planes
        p.normal || continue
        _foreach_face(p, st.mesh, W) do i, j, k, s, wA
            Q += s * p.data[i, j, k] * wA
        end
    end
    return Q
end

"""
    correct_boundary_mass_flux!(state, weights = nothing; t = state.t) -> δ

Make the prescribed normal velocities of an open boundary set compatible
with the pressure problem. With every face's normal velocity prescribed,
`∇²φ = ∇·u*` with `∂φ/∂n = 0` is solvable only if the net boundary flux
vanishes; interpolated or analytic data never satisfy that to round-off,
and the solver would project out the mean and leave a uniform divergence
`Q/V` in every cell. This is the only boundary mass correction in the
code base:

* `Q = Σ W u·n A` over the open fluid faces (`W = 1` for the Boussinesq
  core; `ρ̄` or `ρ̄θ̄` for the sound-proof cores);
* `:uniform` (default): subtract `δ = Q / Σ W A` from the outward normal
  velocity of every open face (Hellsten et al. 2021, §3.3, Eq. 2);
* `:outflow_only`: subtract `δ = Q / Σ_out W A` from the outflow faces
  (`u·n > 0`) only, leaving the inflow untouched (all faces if there is
  no outflow);
* `:none`: no correction (the failure mode, kept for tests).

Records `Q`, `δ` and the post-correction `residual` in the state (and in
`state.log` with `record = true`). Returns `δ`.
"""
function correct_boundary_mass_flux!(st::OpenBoundaryState{T}, W = nothing;
                                     t::Real = st.t) where {T}
    Q = open_boundary_flux(st, W)
    δ = zero(T)
    if st.correction !== :none && Q != 0
        outflow = st.correction === :outflow_only
        A = zero(T)
        for p in st.planes
            p.normal || continue
            _foreach_face(p, st.mesh, W) do i, j, k, s, wA
                (!outflow || s * p.data[i, j, k] > 0) && (A += wA)
            end
        end
        if A == 0          # no outflow face: fall back to all faces
            outflow = false
            for p in st.planes
                p.normal || continue
                _foreach_face(p, st.mesh, W) do i, j, k, s, wA
                    A += wA
                end
            end
        end
        δ = Q / A
        for p in st.planes
            p.normal || continue
            _foreach_face(p, st.mesh, W) do i, j, k, s, wA
                if !outflow || s * p.data[i, j, k] > 0
                    p.data[i, j, k] -= s * δ
                end
            end
        end
    end
    st.Q = Q
    st.δ = δ
    st.residual = open_boundary_flux(st, W)
    st.t = T(t)
    st.record && push!(st.log, (T(t), Q, st.residual))
    return δ
end

# =========================================================================
# The one helper every core calls
# =========================================================================

# Bind the cell-centred fields of `sim` that have planes (θ, qv, …) to
# their names, so `fill_halos!` finds them by identity.
function _bind!(st::OpenBoundaryState, sim)
    for p in st.planes
        p.normal && continue
        _loc_of(p.name) isa CellCenter || continue
        hasproperty(sim, p.name) || continue
        f = getproperty(sim, p.name)
        st.bound[f.data] = p.name
    end
    return st
end

"""
    update_open_boundaries!(sim, t, dt) -> sim

Refresh the open boundaries of `sim` for a projected (sub)step that ends
at time `t` (length `dt`): fill every buffer from the boundary data at
`t` ([`boundary_values!`](@ref)); on a face with a `:convective` or
`:orlanski` outflow, advance the outflow points of every plane instead by
the radiation equation over `dt` (N-open-c, #260; see
[`open_boundary_conditions`](@ref)); then apply
[`correct_boundary_mass_flux!`](@ref) with the core's constraint weights
([`boundary_flux_weights`](@ref)). Points a solid obstacle touches
([`mark_open_boundary_solids!`](@ref), N-open-d #262) get zero velocity
after the data is written, and the correction leaves them alone.

The radiation update is the explicit second-order upstream form of
`∂φ/∂t + c ∂φ/∂n = 0`, applied to every halo layer `b` outward from the
face,

    φ_b ← φ_b⁰ − r (3φ_b⁰ − 4φ_{b−1}⁰ + φ_{b−2}⁰) / 2,    r = min(c Δt / Δn, 1),

where the superscript `⁰` marks values at the start of the (sub)stage
(the buffer, and the interior layers recorded by
[`mark_open_boundary_stage!`](@ref)) and `c` is the phase speed fixed by
[`prepare_open_boundaries!`](@ref). Being explicit in stage-start values,
it is integrated by a multi-stage driver with the driver's own weights
(SSP-RK3 for `step_rk3!`); `r ≤ 1` (the Miller & Thorpe 1981 bound
`c ≤ Δn/Δt`) keeps the update stable. The second-order upstream
difference matters with the centred interior schemes: the first-order
one reflects a scalar pulse with `∫θ'²` residual 1.9e-4 against 7.9e-6
(N-open-c test, σ = 4Δx). The next
`fill_halos!` writes the corrected face values into the predicted
velocity, so the projection that follows sees a compatible problem.

Called once per projected (sub)step, after the predictor and before the
projection, by every core: `step!` and `step_rk3!`
(`Simulations._euler_projection!`, hence also `step_adaptive!`),
`step_imex!` (`ARK2`, `Rosenbrock2`) and the sound-proof
`_step_mass_weighted!`. A no-op when `sim` has no open face, so every
closed run is bit for bit unchanged.
"""
function update_open_boundaries!(sim, t::Real, dt::Real)
    has_open_boundaries(sim) || return sim
    W = boundary_flux_weights(sim)
    for st in open_boundary_states(sim)
        _bind!(st, sim)
        radiate = st.initialized && _has_radiation(st)
        radiate && _save_previous!(st)          # the buffers of the step start
        st.data === nothing ||
            boundary_values!(st.planes, st.data, t)
        _apply_solid_values!(st)                # solid points: zero velocity (N-open-d)
        radiate && _radiate!(st, sim, dt)       # outflow points (N-open-c)
        correct_boundary_mass_flux!(st, W; t = t)
        st.initialized = true
    end
    return sim
end

"""
    prepare_open_boundaries!(sim) -> sim

Before a step: fill the buffers at the current time `sim.t` if they have
never been filled (a fresh simulation, or new data from
[`set_boundary_data!`](@ref)), so the first tendency evaluation sees the
boundary data. A no-op otherwise, and for closed boundaries.

On a face with a radiating outflow (N-open-c, #260) it also fixes, once
per step, which points are outflow points (outward normal velocity of
the boundary face `> 0`) and their phase speed `c`:

* `:convective` — `c = max(u·n, 0)`, the local outward normal velocity
  interpolated to the point;
* `:orlanski` — for the tangential velocities and the scalars,
  `c = −(∂φ/∂t)/(∂φ/∂n)` from the interior layer next to the face at this
  and the previous step start (Orlanski 1976), clipped to `[0, Δn/Δt]`
  (Miller & Thorpe 1981); the first step, without a history, uses the
  convective speed. The **normal velocity** always uses the convective
  speed: in a projected core continuity ties it to the interior, so it is
  not a freely radiating wave, and an Orlanski speed diagnosed from it and
  fed back through the pressure drifts away from the advection speed (in
  the outgoing-vortex test the diagnosed speeds of u and w settled at
  0.79 U and 1.23 U, R = 0.15; with the convective normal velocity
  R = 4.9e-3).

Every multi-stage driver calls it once at the start of the step, so the
phase speeds stay fixed over the substages; a step that is retried at the
same time keeps them.

References: Orlanski, I. (1976), A simple boundary condition for
unbounded hyperbolic flows, *J. Comput. Phys.* 21, 251–269. Miller, M. J.
& Thorpe, A. J. (1981), Radiation conditions for the lateral boundaries
of limited-area numerical models, *Q. J. R. Meteorol. Soc.* 107, 615–628.
"""
function prepare_open_boundaries!(sim)
    has_open_boundaries(sim) || return sim
    if any(st -> !st.initialized, open_boundary_states(sim))
        update_open_boundaries!(sim, sim.t, zero(sim.t))
    end
    for st in open_boundary_states(sim)
        _has_radiation(st) && _outflow_phase_speeds!(st, sim)
    end
    return sim
end

# --- buffers as part of the state (multi-stage drivers) --------------------

"""
    open_boundary_snapshot(sim) -> nothing | Vector{Array}

A copy of every open-boundary buffer of `sim` (`nothing` when closed).
The buffers are part of the state: the start of a (sub)step refills the
halos from them. `step_rk3!` combines them with the same weights as the
fields ([`combine_open_boundaries!`](@ref)); `step_adaptive!` restores
them with the fields when it rejects a step
([`restore_open_boundaries!`](@ref)).
"""
function open_boundary_snapshot(sim)
    has_open_boundaries(sim) || return nothing
    return [copy(p.data) for st in open_boundary_states(sim) for p in st.planes]
end

"""
    restore_open_boundaries!(sim, snapshot) -> sim

Copy an [`open_boundary_snapshot`](@ref) back into the buffers.
"""
restore_open_boundaries!(sim, ::Nothing) = sim
function restore_open_boundaries!(sim, snap::AbstractVector)
    n = 0
    for st in open_boundary_states(sim), p in st.planes
        n += 1
        copyto!(p.data, snap[n])
    end
    return sim
end

"""
    combine_open_boundaries!(sim, a, snapshot, b) -> sim

`buffer .= a .* snapshot .+ b .* buffer` for every open-boundary buffer:
the Shu–Osher stage combination applied to the boundary state, so the
combined stage's halos and boundary faces stay exactly those of the
combined fields.
"""
combine_open_boundaries!(sim, a, ::Nothing, b) = sim
function combine_open_boundaries!(sim, a, snap::AbstractVector, b)
    n = 0
    for st in open_boundary_states(sim), p in st.planes
        n += 1
        p.data .= a .* snap[n] .+ b .* p.data
    end
    return sim
end

"""
    assert_open_face_correction(bcs, dφdx, dφdy) -> nothing

Check, rather than assume, that the pressure correction leaves the
prescribed normal velocity of the west/south open faces (interior slots
`H+1`, which the correction loop visits) unchanged: `∂φ/∂n` must be
exactly zero there (the mirror halo of `φ`); with `dφdz` also an open
bottom (a nest child inside its parent, N-nest-d #261). East, north and
top faces lie in the halo and are never corrected. Throws an `ErrorException`
otherwise.
"""
function assert_open_face_correction(b::FieldBoundaryConditions, dφdx, dφdy,
                                     dφdz = nothing)
    has_open_boundaries(b) || return nothing
    m = dφdx.mesh
    jj = m.Hy+1:m.Hy+m.Ny; kk = m.Hz+1:m.Hz+m.Nz; ii = m.Hx+1:m.Hx+m.Nx
    if b.west isa OpenBoundary
        all(iszero, view(dφdx.data, m.Hx + 1, jj, kk)) ||
            error("open boundary: the pressure correction at the west face is not zero " *
                  "(φ halo is not Neumann); the prescribed inflow would be altered")
    end
    if b.south isa OpenBoundary
        all(iszero, view(dφdy.data, ii, m.Hy + 1, kk)) ||
            error("open boundary: the pressure correction at the south face is not zero " *
                  "(φ halo is not Neumann); the prescribed inflow would be altered")
    end
    if dφdz !== nothing && b.bottom isa OpenBoundary
        all(iszero, view(dφdz.data, ii, jj, m.Hz + 1)) ||
            error("open boundary: the pressure correction at the bottom face is not zero " *
                  "(φ halo is not Neumann); the prescribed inflow would be altered")
    end
    return nothing
end

"""
    open_boundary_ledger(sim) -> NamedTuple

`(Q, residual, δ, t)` of the last mass-flux correction of `sim`'s (first)
open boundary set — `Q` the net outward flux of the data, `residual` the
flux after the correction.
"""
function open_boundary_ledger(sim)
    sts = open_boundary_states(sim)
    isempty(sts) && return (Q = 0.0, residual = 0.0, δ = 0.0, t = 0.0)
    st = first(sts)
    return (Q = st.Q, residual = st.residual, δ = st.δ, t = st.t)
end

"""
    open_intermediate_velocity!(sim, τ) -> sim

Kim & Moin's (1985) boundary condition for the **intermediate** velocity
of a pressure-correction step whose viscous term is implicit. The
projection `u = u* − ∇φ` with `φ ≈ τ p` changes the tangential velocity
next to an open face by `τ ∂p/∂s`, so an implicit diffusion solve for
`u*` that imposes the end-of-step data `g` on the tangential halos leaves
a viscous boundary layer of amplitude `τ ∂p/∂s` after the projection
(first-order splitting error, order reduction in time). The consistent
condition is `u*_s = g_s + τ ∂p/∂s`: this adds `τ ∂p/∂s` (from the
latest pressure `sim.p`, extrapolated from the first interior layer) to
the tangential-velocity buffers of every open face. The caller restores
the buffers ([`restore_open_boundaries!`](@ref)) before the projection,
which must see the data `g` itself. Normal velocities and scalars are not
touched (`∂φ/∂n = 0`). `step_imex!` (`ARK2`) applies it around each
implicit stage.

Reference: Kim, J. & Moin, P. (1985), *J. Comput. Phys.* 59, 308–323,
§3 (boundary condition for the intermediate velocity).
"""
function open_intermediate_velocity!(sim, τ::Real)
    has_open_boundaries(sim.bcs) || return sim
    pf = sim.p
    fill_halos!(pf, sim.bcs)
    pd = pf.data
    m = pf.mesh
    T = eltype(pd)
    τT = T(τ)
    Hs = (m.Hx, m.Hy, m.Hz); Ns = (m.Nx, m.Ny, m.Nz); Δs = (m.dx, m.dy, m.dz)
    for st in open_boundary_states(sim.bcs), p in st.planes
        p.normal && continue
        dir = p.name === :u ? 1 : p.name === :v ? 2 : p.name === :w ? 3 : 0
        dir == 0 && continue
        # the interior layer the gradient is taken from, along the face normal
        lay = p.high ? Hs[p.axis] + Ns[p.axis] : Hs[p.axis] + 1
        inv = 1 / Δs[dir]
        rngs = (p.irange, p.jrange, p.krange)
        @inbounds for (c, k) in enumerate(rngs[3]), (b, j) in enumerate(rngs[2]),
                      (a, i) in enumerate(rngs[1])
            I = (i, j, k)
            # clamp: normal index to the interior layer; the others to the
            # range where the face gradient has two interior neighbours
            q = ntuple(3) do d
                d == p.axis ? lay :
                d == dir ? clamp(I[d], Hs[d] + 2, Hs[d] + Ns[d]) :
                           clamp(I[d], Hs[d] + 1, Hs[d] + Ns[d])
            end
            qm = ntuple(d -> d == dir ? q[d] - 1 : q[d], 3)
            p.data[a, b, c] += τT * (pd[q...] - pd[qm...]) * inv
        end
    end
    return sim
end
