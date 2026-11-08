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
    _check_sgs_stress(stress)
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
`mom` (momentum), `θ` and `qv` (`nothing` when dry); `scratch`, the field
`_add_sgs_tracer!` computes qv's own diffusivity into, and `own`, whether
it does ([`sgs_tracer_own_diffusivity`](@ref)). Built by
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
function sgs_sweep_term(c::AbstractSGSClosure, sim, stress::Symbol, scratch)
    f = sgs_fields(c, sim)
    mom = sgs_momentum_sweep(c, f, sim, stress)
    mom === nothing && return nothing
    θp = _scalar_part(c, f, sim)
    θp === nothing && return nothing
    moist = hasproperty(sim, :has_moisture) && sim.has_moisture
    own = moist && sgs_tracer_own_diffusivity(c)
    qp = !moist ? nothing :
         own ? _FV.EddyScalar(scratch.data, scratch.data, one(eltype(scratch.data))) : θp
    return SGSTerm(c, f, mom, θp, qp, scratch, own)
end

_FV.footprint(t::SGSTerm) = map(max, _FV.part_footprint(t.mom), _FV.part_footprint(t.θ),
                                t.qv === nothing ? (0, 0, 0) : _FV.part_footprint(t.qv))
_FV.sweep_inputs(t::SGSTerm) = (_FV.part_inputs(t.mom)..., _FV.part_inputs(t.θ)...,
                                (t.qv === nothing ? () : _FV.part_inputs(t.qv))...)

# The coefficients from the stage-start state (velocity, θ, qv halos
# filled by the core), then qv's own diffusivity: what the separate loops
# compute before their tendencies.
function _FV.prepare!(t::SGSTerm, sim, Δt)
    f = compute_sgs!(t.closure, sim)
    (f.νh === t.f.νh && f.νv === t.f.νv && f.κh === t.f.κh && f.κv === t.f.κv) ||
        throw(ArgumentError("SGSTerm: compute_sgs! of $(nameof(typeof(t.closure))) wrote other fields " *
                            "than its sgs_fields"))
    if t.qv !== nothing
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
