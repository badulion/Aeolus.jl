# N-open-c (#260) — radiation outflow conditions on open faces: the
# per-face outflow mode `:dirichlet | :convective | :orlanski` of an
# `OpenBoundary` set.
#
# Included inside `module BoundaryConditions`, after open_boundary.jl.
#
# On a radiating face every plane (u, v, w and each scalar) of every
# outflow point is advanced in the halo by the upstream radiation equation
#     ∂φ/∂t + c ∂φ/∂n = 0,
# with the phase speed c fixed per step by `prepare_open_boundaries!`
# (convective: the outward normal velocity; Orlanski 1976 with the
# clipping of Miller & Thorpe 1981), and the inflow points keep the
# Dirichlet data. The mass-flux correction of N-open-b then balances the
# result exactly as it balances Dirichlet data.

"""
    OUTFLOW_MODES

The outflow conditions of an open face (`outflow` keyword of
[`open_boundary_conditions`](@ref)): `:dirichlet` (the boundary data
everywhere), `:convective` (radiation with the outward normal velocity)
and `:orlanski` (radiation with the Orlanski 1976 phase speed, clipped
after Miller & Thorpe 1981).
"""
const OUTFLOW_MODES = (:dirichlet, :convective, :orlanski)
const _OPEN_SIDES = (:west, :east, :south, :north, :top)

function _check_mode(mode)
    mode isa Symbol && mode in OUTFLOW_MODES ||
        throw(ArgumentError("outflow mode must be one of $(OUTFLOW_MODES), got $(repr(mode))"))
    return mode
end

_outflow_mode(o::Symbol, side::Symbol) = _check_mode(o)
_outflow_mode(o::NamedTuple, side::Symbol) =
    _check_mode(haskey(o, side) ? getfield(o, side) : :dirichlet)
_outflow_mode(o, side::Symbol) =
    throw(ArgumentError("outflow must be a Symbol or a NamedTuple of sides, got $(typeof(o))"))

_check_outflow_sides(::Symbol, open) = nothing
function _check_outflow_sides(o::NamedTuple, open)
    for k in keys(o)
        k in _OPEN_SIDES ||
            throw(ArgumentError("outflow: unknown side :$k (use $(_OPEN_SIDES))"))
        k in open ||
            throw(ArgumentError("outflow: side :$k is not an open face of this box"))
    end
    return nothing
end

# The two tangential dimensions of a plane normal to `ax`.
_tang_dims(ax::Int) = ax == 1 ? (2, 3) : ax == 2 ? (1, 3) : (1, 2)

function _init_radiation!(st::OpenBoundaryState{T}) where {T}
    st.radiation = Union{Nothing,RadiationPlane{T}}[nothing for _ in st.planes]
    for (n, p) in enumerate(st.planes)
        mode = get(st.outflow, p.side, :dirichlet)
        mode === :dirichlet && continue
        nidx = findfirst(q -> q.side === p.side && q.normal, st.planes)
        nidx === nothing && continue          # cannot happen: u, v, w are required
        d1, d2 = _tang_dims(p.axis)
        sz = (size(p.data, d1), size(p.data, d2))
        st.radiation[n] = RadiationPlane{T}(mode, nidx, similar(p.data),
                                            fill(T(NaN), sz), zeros(T, sz),
                                            Ref(zero(T)), Ref(false),
                                            zeros(T, sz), zeros(T, sz), Ref(false))
    end
    return st
end

_has_radiation(st::OpenBoundaryState) = any(r -> r !== nothing, st.radiation)

"""
    has_radiating_outflow(bcs_or_sim) -> Bool

Whether any open face has a `:convective` or `:orlanski` outflow
(N-open-c, #260).
"""
has_radiating_outflow(x) = any(_has_radiation, open_boundary_states(x))

# The prognostic array of the field a plane belongs to (or nothing).
function _plane_field(sim, p::OpenPlane)
    hasproperty(sim, p.name) || return nothing
    f = getproperty(sim, p.name)
    return hasproperty(f, :data) ? f.data : nothing
end

# Element of a 3-D array at layer `l` along `ax` and tangential (a, b).
@inline _at(d, ax::Int, l::Int, a::Int, b::Int) =
    ax == 1 ? d[l, a, b] : ax == 2 ? d[a, l, b] : d[a, b, l]
@inline function _set!(d, ax::Int, l::Int, a::Int, b::Int, v)
    if ax == 1
        d[l, a, b] = v
    elseif ax == 2
        d[a, l, b] = v
    else
        d[a, b, l] = v
    end
    return v
end

# Storage index of the plane's boundary layer (the face of the normal
# velocity, the first halo cell of every other field) and the outward step.
_boundary_storage(p::OpenPlane) = p.high ? first(p.range) : last(p.range)
_boundary_local(p::OpenPlane) = p.high ? 1 : length(p.range)
_outward(p::OpenPlane) = p.high ? 1 : -1

_spacing(m::RegularGrid, ax::Int) = ax == 1 ? m.dx : ax == 2 ? m.dy : m.dz
# a z-stretched mesh (a stretched nest child, #328): lateral faces only
_spacing(m::AbstractMesh, ax::Int) = ax == 1 ? m.dx : ax == 2 ? m.dy :
    throw(ArgumentError("a radiation outflow (:convective / :orlanski) on the bottom or top " *
                        "of a z-stretched mesh is not supported; use :dirichlet there"))

# Outward normal velocity at tangential point (a, b) of plane `p`, from the
# boundary-face values of the normal plane `pn` (the state of the step
# start). A velocity component staggered along a tangential direction
# averages the two normal faces it sits between.
function _outward_un(p::OpenPlane{T}, pn::OpenPlane{T}, a::Int, b::Int) where {T}
    ax = p.axis
    fl = _face_local(pn)
    s = p.high ? one(T) : -one(T)
    d1, d2 = _tang_dims(ax)
    st = _normal_loc(_loc_of(p.name))       # staggered direction of p (0: none)
    if p.normal || st == 0
        return s * _at(pn.data, ax, fl, a, b)
    elseif st == d1
        am = max(a - 1, 1)
        return s * (_at(pn.data, ax, fl, am, b) + _at(pn.data, ax, fl, a, b)) / 2
    else
        bm = max(b - 1, 1)
        return s * (_at(pn.data, ax, fl, a, bm) + _at(pn.data, ax, fl, a, b)) / 2
    end
end

# Phase speeds of every radiating plane at the start of a step.
function _outflow_phase_speeds!(st::OpenBoundaryState{T}, sim) where {T}
    m = st.mesh
    t = T(sim.t)
    for (n, rp) in enumerate(st.radiation)
        rp === nothing && continue
        p = st.planes[n]; pn = st.planes[rp.normal]
        f = _plane_field(sim, p)
        if f === nothing                       # no prognostic field: Dirichlet
            fill!(rp.c, T(NaN)); continue
        end
        Δτ = t - rp.t_hist[]
        rp.has_hist[] && Δτ == 0 && continue   # a retried step keeps its speeds
        ax = p.axis
        Δn = T(_spacing(m, ax))
        sb = _boundary_storage(p); o = _outward(p)
        I1 = sb - o; I2 = sb - 2o
        # The normal velocity is not a radiating wave variable in a projected
        # (incompressible / sound-proof) core: continuity ties it to the
        # interior, and a diagnosed phase speed fed back through the pressure
        # drifts (outgoing-vortex test: R = 0.15 with Orlanski on u·n, 4.9e-3
        # with the convective speed). It always uses the convective speed.
        orl = rp.mode === :orlanski && !p.normal && rp.has_hist[] && Δτ > 0
        cmax = orl ? Δn / Δτ : zero(T)
        # pointwise over the face (KA-f, #297)
        launch!(Backends.backend(rp.c), _phase_speed_body!, (1:size(rp.c, 1), 1:size(rp.c, 2)),
                rp.c, rp.hist, p, pn, f, ax, I1, I2, orl, Δτ, Δn, cmax)
        rp.t_hist[] = t
        rp.has_hist[] = true
    end
    return st
end

@inline function _phase_speed_body!(a, b, C::AbstractArray{T}, hist, p, pn, f, ax, I1, I2, orl,
                                    Δτ, Δn, cmax) where {T}
    un = _outward_un(p, pn, a, b)
    @inbounds begin
        φ1 = _at(f, ax, I1, a, b)
        if !(un > 0)
            C[a, b] = T(NaN)                   # inflow (or tangential): Dirichlet data
        elseif !orl
            C[a, b] = un                       # convective (and Orlanski's first step)
        else
            # Orlanski (1976): c = −(∂φ/∂t)/(∂φ/∂n) at the interior layer
            # next to the face, clipped to [0, Δn/Δt] (Miller & Thorpe 1981)
            num = -(φ1 - hist[a, b]) / Δτ
            den = (φ1 - _at(f, ax, I2, a, b)) / Δn
            c = num / den
            C[a, b] = isnan(c) ? zero(T) : clamp(c, zero(T), cmax)
        end
        hist[a, b] = φ1
    end
    return nothing
end

function _save_previous!(st::OpenBoundaryState)
    for (n, rp) in enumerate(st.radiation)
        rp === nothing || copyto!(rp.prev, st.planes[n].data)
    end
    return st
end

"""
    mark_open_boundary_stage!(sim) -> sim

Record, at the start of a projected (sub)stage, the two interior layers
next to every radiating outflow face (N-open-c, #260). The radiation
update of that stage ([`update_open_boundaries!`](@ref)) is explicit in
these stage-start values, so a multi-stage driver integrates the
boundary's semi-discrete radiation equation with its own Runge–Kutta
weights. The velocity at the update is already the predicted `u*`, which
differs from the projected velocity by `Δt ∇φ`; using it instead would
put a pressure-gradient error into the outflow. A no-op without a
radiating face. Called by `Simulations._euler_projection!` and
`Anelastic._step_mass_weighted!`; `step_imex!` updates the boundaries
before it changes the fields and needs no mark.
"""
function mark_open_boundary_stage!(sim)
    has_open_boundaries(sim) || return sim
    for st in open_boundary_states(sim)
        _has_radiation(st) || continue
        for (n, rp) in enumerate(st.radiation)
            rp === nothing && continue
            p = st.planes[n]
            f = _plane_field(sim, p)
            f === nothing && continue
            ax = p.axis
            sb = _boundary_storage(p); o = _outward(p)
            launch!(Backends.backend(rp.c), _mark_stage_body!, (1:size(rp.c, 1), 1:size(rp.c, 2)),
                    rp.start1, rp.start2, f, ax, sb - o, sb - 2o)
            rp.marked[] = true
        end
    end
    return sim
end

@inline function _mark_stage_body!(a, b, S1, S2, f, ax, l1, l2)
    @inbounds S1[a, b] = _at(f, ax, l1, a, b)
    @inbounds S2[a, b] = _at(f, ax, l2, a, b)
    return nothing
end

# Advance the outflow points of every radiating plane over `dt`: the
# explicit upstream radiation update, layer by layer outward from the
# face, from the stage-start values.
function _radiate!(st::OpenBoundaryState{T}, sim, dt::Real) where {T}
    m = st.mesh
    for (n, rp) in enumerate(st.radiation)
        rp === nothing && continue
        p = st.planes[n]
        f = _plane_field(sim, p)
        f === nothing && continue
        ax = p.axis
        Δn = T(_spacing(m, ax))
        sb = _boundary_storage(p); o = _outward(p)
        lb = _boundary_local(p); L = length(p.range)
        # one work item per face point: its buffer pencil, outward (KA-f, #297)
        launch!(Backends.backend(p.data), _radiate_body!, (1:size(rp.c, 1), 1:size(rp.c, 2)),
                p.data, rp.prev, rp.c, rp.start1, rp.start2, rp.marked[], f, ax, sb, o, lb, L,
                T(dt), Δn)
        rp.marked[] = false
    end
    return st
end

@inline function _radiate_body!(a, b, P, P0, C::AbstractArray{T}, S1, S2, marked::Bool, f, ax,
                                sb, o, lb, L, dt, Δn) where {T}
    @inbounds begin
        c = C[a, b]
        isnan(c) && return nothing
        r = min(c * dt / Δn, one(T))
        i1 = marked ? S1[a, b] : _at(f, ax, sb - o, a, b)
        i2 = marked ? S2[a, b] : _at(f, ax, sb - 2o, a, b)
        l = lb
        while 1 <= l <= L
            φ0 = _at(P0, ax, l, a, b)
            # second-order upstream: ∂φ/∂n ≈ (3φ_b − 4φ_{b−1} + φ_{b−2}) / 2Δn
            _set!(P, ax, l, a, b, φ0 - r * (3φ0 - 4i1 + i2) / 2)
            i2 = i1; i1 = φ0
            l += o
        end
    end
    return nothing
end

"""
    rewind_open_boundaries!(sim, snapshot) -> sim

Restore the open-boundary buffers to `snapshot` (from
[`open_boundary_snapshot`](@ref) at the step start) when a face has a
radiating outflow; a no-op otherwise. The radiation update advances a
buffer from its value at the start of the stage, so a driver whose stage
restarts from the step-start state (`step_imex!`'s ARK2 and Rosenbrock2
second stages) rewinds the buffers first. Dirichlet buffers are
overwritten by the data and need no rewinding.
"""
rewind_open_boundaries!(sim, ::Nothing) = sim
function rewind_open_boundaries!(sim, snap::AbstractVector)
    has_radiating_outflow(sim) || return sim
    return restore_open_boundaries!(sim, snap)
end
