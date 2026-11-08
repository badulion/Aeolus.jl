# =========================================================================
# The SGS closure as a term of the fused sweep (#533 plan step 4).
# =========================================================================
#
# Included into `module Operators`, after the closure contract and every
# built-in closure. What the separate loops of the Boussinesq step do,
#
#   f = compute_sgs!(closure, sim)
#   add_sgs_momentum!(tu, tv, tw, closure, f, sim; stress)
#   add_sgs_scalar!(tθ, θ, closure, f, sim)
#   _add_sgs_tracer!(tqv, qv, closure, f, sim, scratch)       (moist)
#
# becomes one `SGSTerm`: its `prepare!` is the `compute_sgs!` (and qv's
# own diffusivity), before the sweep; its row kernel adds the three
# tendencies from the precomputed coefficients. Which kernel each
# tendency is comes from the closure contract, not from a second copy of
# it: the momentum part from [`sgs_momentum_sweep`](@ref), the scalar
# part from [`sgs_scalar_face_diffusivity`](@ref) or
# [`sgs_scalar_tensor`](@ref) (the generics the nest's flux registers and
# the IMEX cores already read), qv's from
# [`sgs_tracer_own_diffusivity`](@ref).

const _FV = FiniteVolume

"""
    sgs_momentum_sweep(c::AbstractSGSClosure, f::SGSFields, sim, stress::Symbol)

The momentum part of the closure's sweep term (#533): the kernel its
[`add_sgs_momentum!`](@ref) runs, as a part the fused sweep evaluates per
row (`FiniteVolume.EddyMomentum`, `FiniteVolume.StressMomentum`), or
`nothing` when it has none. Default: the eddy viscosity of the default
`add_sgs_momentum!` (the Laplacian form with `f.νh` for `stress =
:laplacian`, the stress form with `f.νh`, `f.νv` for `:symmetric`, a
`RegularGrid` only, as the operator), `nothing` for a closure with its
own stress ([`sgs_explicit_stress`](@ref)). A closure that adds an
`add_sgs_momentum!` method adds the matching method here (checked by
`test/fused_sweep_sgs.jl`).
"""
function sgs_momentum_sweep(c::AbstractSGSClosure, f::SGSFields, sim, stress::Symbol)
    sgs_explicit_stress(c) && return nothing
    return _eddy_momentum_part(f, mesh(sim.u), stress)
end
function _eddy_momentum_part(f::SGSFields, g::AbstractMesh, stress::Symbol)
    T = eltype(f.νh.data)
    if stress === :laplacian
        return _FV.EddyMomentum(f.νh.data, f.νv.data, false, zero(T), zero(T), zero(T))
    elseif stress === :symmetric
        g isa RegularGrid || return nothing        # the operator refuses it: the separate loops say why
        return _FV.EddyMomentum(f.νh.data, f.νv.data, true, T(1 / g.dx), T(1 / g.dy), T(1 / g.dz))
    end
    _check_sgs_stress(stress)                     # throws: neither form
    return nothing
end

"""
    sgs_tracer_own_diffusivity(c::AbstractSGSClosure) -> Bool

Whether [`sgs_tracer_diffusivity!`](@ref) of `c` computes a scalar other
than θ its own diffusivity (`true`: the AMD scalar model) or returns
`f.κh` (`false`, the default: the scalar is diffused like θ). The fused
sweep needs it before the coefficients are computed; a closure that adds
an `sgs_tracer_diffusivity!` method adds the matching method here, and
the sweep term's `prepare!` checks the two agree.
"""
sgs_tracer_own_diffusivity(::AbstractSGSClosure) = false
sgs_tracer_own_diffusivity(::AMDClosure) = true

# the scalar part of θ: the face form, else the tensor, else none
function _scalar_part(c, f, sim)
    fd = sgs_scalar_face_diffusivity(c, f, sim)
    fd === nothing || return _FV.EddyScalar(fd[1].data, fd[2].data, fd[3])
    K = sgs_scalar_tensor(c, f, sim)
    K === nothing && return nothing
    g = mesh(sim.θ)
    g isa RegularGrid || return nothing
    T = eltype(K.K11.data)
    return _FV.TensorScalar(K.K11.data, K.K22.data, K.K33.data, K.K12.data, K.K13.data, K.K23.data,
                            T(1 / g.dx), T(1 / g.dy), T(1 / g.dz),
                            bounded_directions(_scalar_bcs(sim)), interior(g))
end

"""
    SGSTerm

The SGS closure of a Boussinesq substage as a term of the fused sweep
(#533 plan step 4): `closure` and its coefficient storage `f`; the parts
`mom` (momentum), `θ` and `qv` (evaluated when `moist`); `scratch`, the
field `_add_sgs_tracer!` computes qv's own diffusivity into, and `own`,
whether it does ([`sgs_tracer_own_diffusivity`](@ref)). Built by
[`sgs_sweep_term`](@ref). Its [`prepare!`](@ref FiniteVolume.prepare!)
computes the coefficients (`compute_sgs!`) from the stage-start state, as
the separate loops do; inside the sweep they are inputs.
"""
struct SGSTerm{C, F<:SGSFields, M, S1, S2, K} <: _FV.SweepTerm
    closure::C
    f::F
    mom::M
    θ::S1
    qv::S2
    scratch::K
    own::Bool
    moist::Bool
end

"""
    sgs_sweep_term(c::AbstractSGSClosure, sim, stress::Symbol, scratch) -> Union{SGSTerm,Nothing}

The sweep term of the SGS closure `c` on the Boussinesq simulation `sim`
with the momentum form `stress` (the `sgs_stress` keyword) and `scratch`
the field qv's own diffusivity goes into (the one the separate loops
use); `nothing` when its momentum or scalar tendency has no part the
sweep can evaluate (the substage then runs one loop per term, with the
closure's type as the reason).
"""
@inline function sgs_sweep_term(c::AbstractSGSClosure, sim, stress::Symbol, scratch)
    sgs_sweep_form_matches(c, sim, scratch) || return nothing
    f = sgs_fields(c, sim)
    mom = sgs_momentum_sweep(c, f, sim, stress)
    mom === nothing && return nothing
    θp = _scalar_part(c, f, sim)
    θp === nothing && return nothing
    moist = hasproperty(sim, :has_moisture) && sim.has_moisture
    # qv's part is a type of the closure, not of the run (dry: not evaluated)
    own = sgs_tracer_own_diffusivity(c)
    qp = own ? _FV.EddyScalar(scratch.data, scratch.data, _unit_scale(θp, eltype(scratch.data))) : θp
    return SGSTerm(c, f, mom, θp, qp, scratch, own, moist)
end
# ---- the sweep form matches the closure's own tendencies ------------------
#
# The sweep evaluates the kernels the contract generics name, not the
# closure's `add_sgs_momentum!` / `add_sgs_scalar!` / `sgs_tracer_diffusivity!`.
# A closure (from any module) that overrides one of those without the
# matching sweep form would get the default kernel inside the sweep: other
# physics than with one loop per term. So the method of each sweep generic
# that dispatch selects for the closure must be at least as specific as
# the method of the tendency it stands for; otherwise the closure stays
# out of the sweep and is reported by its type.

# the closure argument's type in the method `m` (argument `n` of the signature)
function _closure_param(m::Method, n::Int)
    p = Base.unwrap_unionall(m.sig).parameters[n]
    p isa TypeVar && (p = p.ub)
    return Base.rewrap_unionall(p, m.sig)
end
_closure_type(fn, argtypes::Tuple, n::Int) = _closure_param(which(fn, Tuple{argtypes...}), n + 1)

# the wrapped closure whose sweep forms a wrapper forwards to, or nothing
sgs_sweep_wrapped(::AbstractSGSClosure) = nothing
sgs_sweep_wrapped(c::IsentropicDiffusion) = c.inner

function _sweep_form_matches_uncached(c, sim, scratch)
    C, F, S = typeof(c), typeof(sgs_fields(c, sim)), typeof(sim)
    Tu, Tθ, K = typeof(sim.tu), typeof(sim.θ), typeof(scratch)
    tend = _closure_type(add_sgs_momentum!, (Tu, Tu, Tu, C, F, S), 4)
    _closure_type(sgs_momentum_sweep, (C, F, S, Symbol), 1) <: tend || return false
    tend = _closure_type(add_sgs_scalar!, (Tθ, Tθ, C, F, S), 3)
    _closure_type(sgs_scalar_face_diffusivity, (C, F, S), 1) <: tend || return false
    Tk = _closure_type(sgs_scalar_tensor, (C, F, S), 1)
    (Tk <: tend || Tk === AbstractSGSClosure) || return false     # (the default: no tensor)
    tend = _closure_type(sgs_tracer_diffusivity!, (K, C, F, S, Tθ), 2)
    _closure_type(sgs_tracer_own_diffusivity, (C,), 1) <: tend || return false
    inner = sgs_sweep_wrapped(c)
    return inner === nothing || sgs_sweep_form_matches(inner, sim, scratch)
end

# per closure and simulation type, recomputed when methods were added
# (an entry by reference: looking it up allocates nothing)
mutable struct _SweepFormEntry
    C::Any
    S::Any
    world::UInt
    ok::Bool
end
const _SWEEP_FORM_CACHE = Dict{UInt, _SweepFormEntry}()
const _SWEEP_FORM_LOCK = ReentrantLock()

"""
    sgs_sweep_form_matches(c::AbstractSGSClosure, sim, scratch) -> Bool

Whether the fused sweep evaluates what the closure's own tendencies do
(#533): for each of `add_sgs_momentum!`, `add_sgs_scalar!` and
`sgs_tracer_diffusivity!`, the method dispatch selects for `c` of the
generic the sweep reads instead ([`sgs_momentum_sweep`](@ref),
[`sgs_scalar_face_diffusivity`](@ref) and [`sgs_scalar_tensor`](@ref),
[`sgs_tracer_own_diffusivity`](@ref)) is at least as specific, and so for
the closure a wrapper forwards to (`sgs_sweep_wrapped`). A closure that
overrides a tendency without its sweep form returns `false` and runs one
loop per term. Cached per closure and simulation type.
"""
@noinline function sgs_sweep_form_matches(c::AbstractSGSClosure, sim, scratch)
    C, S = typeof(c), typeof(sim)
    key = hash(S, hash(C))
    w = Base.get_world_counter()
    lock(_SWEEP_FORM_LOCK)
    try
        e = get(_SWEEP_FORM_CACHE, key, nothing)
        (e !== nothing && e.C === C && e.S === S && e.world == w) && return e.ok
    finally
        unlock(_SWEEP_FORM_LOCK)
    end
    ok = _sweep_form_matches_uncached(c, sim, scratch)::Bool
    lock(_SWEEP_FORM_LOCK)
    try
        _SWEEP_FORM_CACHE[key] = _SweepFormEntry(C, S, w, ok)
    finally
        unlock(_SWEEP_FORM_LOCK)
    end
    return ok
end

# the factor of qv's own diffusion (the 1 of `add_eddy_diffusion_scalar!(tϕ, ϕ, κ)`)
# in the type of θ's, so the term's type does not depend on the closure's flags
_unit_scale(p::_FV.EddyScalar, ::Type{T}) where {T} = oftype(p.s, 1)
_unit_scale(p, ::Type{T}) where {T} = one(T)

_FV.footprint(t::SGSTerm) = map(max, _FV.part_footprint(t.mom), _FV.part_footprint(t.θ), _FV.part_footprint(t.qv))
_FV.sweep_inputs(t::SGSTerm) = (_FV.part_inputs(t.mom)..., _FV.part_inputs(t.θ)..., _FV.part_inputs(t.qv)...)

# The coefficients from the stage-start state (velocity, θ, qv halos
# filled by the core), then qv's own diffusivity: what the separate loops
# compute before their tendencies.
function _FV.prepare!(t::SGSTerm, sim, Δt)
    f = compute_sgs!(t.closure, sim)
    (f.νh === t.f.νh && f.νv === t.f.νv && f.κh === t.f.κh && f.κv === t.f.κv) ||
        throw(ArgumentError("SGSTerm: compute_sgs! of $(nameof(typeof(t.closure))) wrote other fields " *
                            "than its sgs_fields"))
    if t.moist
        κ = sgs_tracer_diffusivity!(t.scratch, t.closure, f, sim, sim.qv)
        (κ !== f.κh) == t.own ||
            throw(ArgumentError("SGSTerm: sgs_tracer_diffusivity! and sgs_tracer_own_diffusivity of " *
                                "$(nameof(typeof(t.closure))) disagree"))
    end
    return nothing
end

# momentum, then θ, then qv: the order of the separate loops
@inline function _FV.add_row!(t::SGSTerm, buf, a, b, s, j, k, ii, st)
    _FV.add_part_row!(t.mom, buf, a, b, s, j, k, ii, st)
    _FV.add_scalar_row!(t.θ, _FV.SWEEP_Θ, buf, a, b, s, j, k, ii, st.θ, st.m)
    _FV.add_scalar_row!(t.qv, _FV.SWEEP_QV, buf, a, b, s, j, k, ii, st.qv, st.m)
    return nothing
end

# the wrappers of this module forward to the closure they wrap, as their
# add_sgs_momentum! does (their scalar part is their own face form or tensor)
sgs_momentum_sweep(c::IsentropicDiffusion, f::SGSFields, sim, stress::Symbol) =
    sgs_momentum_sweep(c.inner, f, sim, stress)
