# =========================================================================
# Era VI, Phase 54 (P54-a, #167) — the SGS closure contract.
# =========================================================================
#
# Included into `module Operators`. Every dynamical core (the Boussinesq
# `step!`, `step_rk3!`, `step_imex!`, `step_adaptive!`, and the anelastic /
# pseudo-incompressible `_step_mass_weighted!`) takes one `sgs` keyword
# whose value is an `AbstractSGSClosure`, and drives it through the five
# generics below. Nothing here is new physics: the built-in closures are
# thin wrappers of the existing kernels (`compute_smagorinsky_νt!`,
# `compute_dynamic_smagorinsky_νt!`, the `add_eddy_*` flux divergences).
#
#   sgs_coefficients!(f, c, sim)   pure: ν, κ from the current state
#   advance_sgs!(c, sim, dt)       commit memory, once per accepted step
#   sgs_state(c) / restore_sgs_state!(c, s)   roll back a rejected step
#   add_sgs_momentum!(tu, tv, tw, c, f, sim)  momentum tendency
#   add_sgs_scalar!(tϕ, ϕ, c, f, sim)         scalar tendency
#
# The simulation `sim` is duck-typed: the generics read `sim.u, sim.v,
# sim.w` (and may read `sim.θ`, `sim.qv`, `sim.mesh`, `sim.bcs`, `sim.Pr_t`,
# and, on the Boussinesq core, `sim.θ₀`, `sim.g_accel`), and the default
# storage is `sim.νt` and `sim.κt`.

"""
    AbstractSGSClosure

Supertype of the subgrid-scale closures every dynamical core accepts through
its `sgs` keyword (`step!`, `step_rk3!`, `step_imex!`, `step_adaptive!`, and
the anelastic / pseudo-incompressible `step!`). A closure implements

| generic | when the core calls it | default |
| :------ | :--------------------- | :------ |
| [`sgs_coefficients!`](@ref)`(f, c, sim)` | each time the tendencies are formed (every RK substage, every IMEX step, the CFL estimate) | **required** |
| [`advance_sgs!`](@ref)`(c, sim, dt)` | once per **accepted** step, after the projection | no-op |
| [`sgs_state`](@ref)`(c)`, [`restore_sgs_state!`](@ref)`(c, s)` | around a step [`step_adaptive!`](@ref Aeolus.TimeIntegration.step_adaptive!) may reject | no-op |
| [`add_sgs_momentum!`](@ref)`(tu, tv, tw, c, f, sim; stress)` | momentum tendency | Laplacian eddy-viscosity form with `f.νh` (`stress = :laplacian`), or the stress form with `f.νh`, `f.νv` (`:symmetric`) |
| [`add_sgs_scalar!`](@ref)`(tϕ, ϕ, c, f, sim)` | θ tendency | diagonal eddy diffusion with `f.κh` (x, y faces), `f.κv` (z faces) |
| [`sgs_fields`](@ref)`(c, sim)` | the storage `f` the coefficients go into | `SGSFields(sim.νt, sim.κt)` |

`sgs_coefficients!` must be **pure** in the closure's memory: it may read the
state and the memory last committed by `advance_sgs!`, never change it. That
is what lets a core call it several times per step (three RK3 substages, the
adaptive stepper's CFL estimate, a retried step) while a stateful closure
(Lagrangian dynamic, prognostic TKE, Reynolds-stress transport) still
advances exactly once per accepted step.

Built-in closures: [`StaticSmagorinsky`](@ref), [`DynamicSmagorinsky`](@ref),
`ML.MLSGS`, and [`FunctionSGS`](@ref) (the adapter that keeps the legacy
`eddy_viscosity! = f(νt, u, v, w)` keyword working). The design is in the
theory page *Smagorinsky-Lilly SGS*, section "Closure contract".
"""
abstract type AbstractSGSClosure end

"""
    SGSFields(νh, νv, κh, κv)
    SGSFields(ν, κ)

The cell-centred eddy coefficients a closure writes and the tendency
generics read: horizontal and vertical eddy viscosity `νh`, `νv` and
eddy diffusivity `κh`, `κv` (m² s⁻¹, `Field{CellCenter}`). An isotropic
closure aliases them, `νv === νh` and `κv === κh` (the two-argument form);
the halos are filled by the core (with the velocity boundary conditions)
after [`sgs_coefficients!`](@ref) returns, once per distinct field.
"""
struct SGSFields{F}
    νh::F
    νv::F
    κh::F
    κv::F
end
SGSFields(ν, κ) = SGSFields(ν, ν, κ, κ)

"""
    sgs_fields(c::AbstractSGSClosure, sim) -> SGSFields

The storage the closure's coefficients go into. Default: the simulation's
own `SGSFields(sim.νt, sim.κt)` (isotropic). An anisotropic closure that
needs separate vertical coefficients overrides it with its own fields.
"""
sgs_fields(::AbstractSGSClosure, sim) = SGSFields(sim.νt, sim.κt)

"""
    sgs_coefficients!(f::SGSFields, c::AbstractSGSClosure, sim) -> f

Fill the interior of the eddy coefficients `f` from the current state of
`sim` (`u, v, w` with filled halos; `θ`, `qv`, `mesh`, the buoyancy
reference) and the memory `c` last committed. **Pure**: it must not change
the closure's memory, because the cores call it more than once per step
(see [`AbstractSGSClosure`](@ref)). Every closure implements it.
"""
function sgs_coefficients! end

"""
    advance_sgs!(c::AbstractSGSClosure, sim, dt) -> c

Commit the closure's memory over an **accepted** step of length `dt`: called
exactly once per step by every core, after the pressure projection (so `sim`
holds the new state), and never on an RK substage. The default is a no-op
(a stateless closure). When [`step_adaptive!`](@ref
Aeolus.TimeIntegration.step_adaptive!) rejects a step, the commit made inside
it is undone through [`sgs_state`](@ref) / [`restore_sgs_state!`](@ref).
"""
advance_sgs!(c::AbstractSGSClosure, sim, dt) = c

"""
    sgs_state(c::AbstractSGSClosure) -> snapshot

A copy of the closure's memory, taken by [`step_adaptive!`](@ref
Aeolus.TimeIntegration.step_adaptive!) before it tries a step. Default
`nothing` (stateless). A stateful closure returns a deep copy (or copies into
a buffer it owns) so that [`restore_sgs_state!`](@ref) can roll a rejected
step back bit for bit.

**Serialisable** (#245): the snapshot must be *self-contained* — plain
arrays, numbers and tuples of them, and `restore_sgs_state!(c, s)` must read
nothing but `s` — because a checkpoint stores a deep copy of it and restores
it into a freshly constructed closure in another process
([`checkpoint!`](@ref Aeolus.IO.checkpoint!)). Every memory the closure's
next step reads (not only what `advance_sgs!` writes) belongs in it.
"""
sgs_state(::AbstractSGSClosure) = nothing

"""
    restore_sgs_state!(c::AbstractSGSClosure, snapshot) -> c

Restore the memory saved by [`sgs_state`](@ref) (no-op by default).
"""
restore_sgs_state!(c::AbstractSGSClosure, s) = c

"""
    add_sgs_momentum!(tu, tv, tw, c::AbstractSGSClosure, f::SGSFields, sim;
                      stress = :laplacian)

Add the SGS stress divergence to the momentum tendencies. `stress` is the
`sgs_stress` keyword of the cores:

* `:laplacian` (default): the Laplacian eddy-viscosity flux form
  `∂/∂xⱼ(ν ∂uᵢ/∂xⱼ)` of [`add_eddy_viscosity_u!`](@ref) (and `_v!`, `_w!`)
  with `f.νh` — the pre-Era-VI operator, bit for bit;
* `:symmetric`: the stress form `∂/∂xⱼ(2νᵢⱼSᵢⱼ)` of
  [`add_sgs_stress_divergence!`](@ref) with `νh = f.νh`, `νv = f.νv`
  (P55-b, #175).

On a sound-proof core ([`flux_weights`](@ref)`(sim)` not `nothing`) both
forms are the weighted flux divergence `(1/W) ∂ⱼ(W τᵢⱼ)`.

A closure with its own stress (an explicit Reynolds stress) adds a method
and may call [`add_stress_divergence!`](@ref). Halos of `sim.u, sim.v,
sim.w` and of `f` are filled.
"""
function add_sgs_momentum!(tu, tv, tw, c::AbstractSGSClosure, f::SGSFields, sim;
                           stress::Symbol = :laplacian)
    W = flux_weights(sim)
    if stress === :laplacian
        add_eddy_viscosity_u!(tu, sim.u, f.νh; weights = W)
        add_eddy_viscosity_v!(tv, sim.v, f.νh; weights = W)
        add_eddy_viscosity_w!(tw, sim.w, f.νh; weights = W)
    elseif stress === :symmetric
        add_sgs_stress_divergence!(tu, tv, tw, sim.u, sim.v, sim.w, f.νh, f.νv; weights = W)
    else
        _check_sgs_stress(stress)
    end
    return nothing
end

"""
    sgs_explicit_stress(c::AbstractSGSClosure) -> Bool

`true` for a closure whose momentum tendency is its own stress (an
[`add_sgs_momentum!`](@ref) method that is not an eddy viscosity, e.g. the
prognostic Reynolds stress of P56-d, #188). `step_imex!` treats the SGS
momentum implicitly as the eddy viscosity `f.νh` and has no explicit hook
for such a stress, so it refuses these closures instead of silently
replacing the stress with `ν_t`. Default `false`.
"""
sgs_explicit_stress(::AbstractSGSClosure) = false

"""
    add_sgs_scalar!(tϕ, ϕ, c::AbstractSGSClosure, f::SGSFields, sim)

Add the SGS scalar flux divergence `∂/∂xⱼ(κⱼₖ ∂ϕ/∂xₖ)` to `tϕ`. Default: the
diagonal operator [`add_eddy_diffusion_scalar!`](@ref)`(tϕ, ϕ, f.κh, f.κv)`
(P55-c, #181), `κh` on the x and y faces and `κv` on the z faces; for an
isotropic closure (`f.κv === f.κh`) that is the three-argument form
`(tϕ, ϕ, f.κh)`, bit for bit the pre-P55-c operator. A closure with a full
diffusivity tensor adds a method that calls
[`add_tensor_diffusion_scalar!`](@ref) (see [`IsentropicDiffusion`](@ref)).
Halos of `ϕ` and of `f` are filled.
"""
function add_sgs_scalar!(tϕ, ϕ, c::AbstractSGSClosure, f::SGSFields, sim)
    W = flux_weights(sim)
    if f.κv === f.κh
        add_eddy_diffusion_scalar!(tϕ, ϕ, f.κh; weights = W)
    else
        add_eddy_diffusion_scalar!(tϕ, ϕ, f.κh, f.κv; weights = W)
    end
    return nothing
end

"""
    flux_weights(sim) -> Union{Nothing, Tuple}

The vertical weights of the SGS flux divergence on the core `sim`:
`nothing` on the Boussinesq core (the unweighted operators), and the
constraint weights `(Wc, Wf)` of a sound-proof core (`W = ρ̄` anelastic,
`ρ̄θ̄` pseudo-incompressible; `Anelastic.constraint_weights`), on which the
SGS tendencies are the flux form `(1/W) ∇·(W F)` of the weighted advection
(ONE-j, #460): the SGS terms then conserve `Σ W uᵢ ΔV` and `Σ W θ ΔV`, so
the core's θ budget closes with the prescribed surface fluxes alone (whose
Boussinesq form `F/Δz` is already the weighted one: the wall faces of a
`ReferenceState` carry the weight of the adjacent cell). The default
[`add_sgs_momentum!`](@ref) and [`add_sgs_scalar!`](@ref) methods (and those
of the eddy-viscosity and stability-corrected closures) read it; a closure
with a full-tensor scalar flux or its own stress has no weighted form, and
a sound-proof core refuses it (`sgs_weighted_form`).
"""
flux_weights(sim) = nothing

"""
    sgs_weighted_form(c::AbstractSGSClosure, f::SGSFields, sim) -> Bool

Whether the closure `c`'s tendencies take the weights of
[`flux_weights`](@ref): a closure whose scalar flux has the diagonal face
form ([`sgs_scalar_face_diffusivity`](@ref) not `nothing`) and whose
momentum tendency is the eddy viscosity (not [`sgs_explicit_stress`](@ref)).
"""
sgs_weighted_form(c::AbstractSGSClosure, f::SGSFields, sim) =
    !sgs_explicit_stress(c) && sgs_scalar_face_diffusivity(c, f, sim) !== nothing

"""
    sgs_scalar_face_diffusivity(c::AbstractSGSClosure, f::SGSFields, sim) -> (Kh, Kv, s) or nothing

The face form of [`add_sgs_scalar!`](@ref) for the closures whose scalar
flux is diagonal: the SGS flux through a face normal to x or y is
`−s · ½(Kh_L + Kh_R) (ϕ_R − ϕ_L)/Δ`, and through a z face the same with
`Kv` (the cell-centred coefficient fields, halos filled, averaged
linearly to the face — the kernel of
[`add_eddy_diffusion_scalar!`](@ref)). Default `(f.κh, f.κv, 1)`; an
eddy-viscosity closure gives `(νt, νt, 1/Prₜ)` (its tendency is formed
with the factor after the flux divergence, so a flux built from this
differs from the tendency's by round-off only). `nothing` for a closure
whose scalar flux is a full tensor ([`IsentropicDiffusion`](@ref), the
Reynolds-stress closure): it has no two-point face form.

Every closure that adds an `add_sgs_scalar!` method must add the
matching method here. Used by the flux registers of a two-way nest
(`Nesting`, N-nest-e #263), which must book exactly the flux the
tendency used.
"""
sgs_scalar_face_diffusivity(::AbstractSGSClosure, f::SGSFields, sim) = (f.κh, f.κv, 1.0)

"""
    sgs_tracer_diffusivity!(κ, c::AbstractSGSClosure, f::SGSFields, sim, ϕ) -> Field

The cell-centred SGS diffusivity of a scalar `ϕ` other than θ (on the
Boussinesq core: `qv`, in the explicit SGS flux of `step!` / `step_rk3!`
and the implicit diffusion of
[`step_imex!`](@ref Aeolus.TimeIntegration.step_imex!)), after
[`compute_sgs!`](@ref) has filled `f`. Default: `f.κh`, returned as is and
`κ` untouched, which is right for every closure whose `κ_t` is a property of
the flow alone (`κ_t = ν_t/Prₜ`). A closure whose `κ_t` is built from θ's own
gradient (the AMD scalar model, [`AMDClosure`](@ref)) computes `ϕ`'s into the
scratch field `κ` (halos filled with `sim.bcs`) and returns `κ`.
"""
sgs_tracer_diffusivity!(κ, c::AbstractSGSClosure, f::SGSFields, sim, ϕ) = f.κh

# -------------------------------------------------------------------------
# What the cores call
# -------------------------------------------------------------------------

"""
    compute_sgs!(c::AbstractSGSClosure, sim) -> SGSFields

[`sgs_coefficients!`](@ref) into [`sgs_fields`](@ref)`(c, sim)`, then the
halos of each distinct coefficient field (with `sim.bcs`). The velocity
halos must already be filled. This is the one call every core and the CFL
estimate make; like `sgs_coefficients!`, it changes no closure memory.
"""
function compute_sgs!(c::AbstractSGSClosure, sim)
    f = sgs_fields(c, sim)
    sgs_coefficients!(f, c, sim)
    _fill_sgs_halos!(f, c, sim)
    return f
end

# Halos of each distinct coefficient field, with the velocity BCs.
function _fill_sgs_halos!(f::SGSFields, c::AbstractSGSClosure, sim)
    bcs = sim.bcs
    fill_halos!(f.νh, bcs)
    f.νv === f.νh || fill_halos!(f.νv, bcs)
    (f.κh === f.νh || f.κh === f.νv) || fill_halos!(f.κh, bcs)
    (f.κv === f.κh || f.κv === f.νh || f.κv === f.νv) || fill_halos!(f.κv, bcs)
    return f
end

"""
    sgs_closure(x) -> Union{Nothing, AbstractSGSClosure}

Normalise an `sgs` keyword: a closure is returned as is, `nothing` stays
`nothing`, and a plain function `f(νt, u, v, w)` (the legacy
`eddy_viscosity!` hook) is wrapped in a stateless [`FunctionSGS`](@ref).
"""
sgs_closure(::Nothing) = nothing
sgs_closure(c::AbstractSGSClosure) = c
sgs_closure(f::Function) = FunctionSGS(f)
sgs_closure(x) = throw(ArgumentError("sgs: expected an AbstractSGSClosure, a function " *
                                     "f(νt, u, v, w) or nothing, got $(typeof(x))"))

# The closure a core runs: the `sgs` keyword (or its legacy alias
# `eddy_viscosity!`) when given, else the built-in Smagorinsky when the
# simulation was built with `C_s ≠ 0`, else none.
function _active_sgs(sim, sgs, eddy_viscosity!)
    sgs !== nothing && eddy_viscosity! !== nothing &&
        throw(ArgumentError("pass the SGS closure as `sgs` or as the legacy " *
                            "`eddy_viscosity!`, not both"))
    s = sgs === nothing ? eddy_viscosity! : sgs
    s === nothing || return sgs_closure(s)
    return sim.has_sgs ? StaticSmagorinsky(sim.C_s) : nothing
end

# -------------------------------------------------------------------------
# Eddy-viscosity closures with a turbulent Prandtl number
# -------------------------------------------------------------------------

"""
    AbstractEddyViscosityClosure <: AbstractSGSClosure

An isotropic closure that computes an eddy viscosity `νt` with
[`sgs_viscosity!`](@ref)`(νt, c, sim)` and takes the scalar diffusivity
from a turbulent Prandtl number, `κt = νt / Prₜ` (`Prₜ =`
[`turbulent_prandtl`](@ref)`(c, sim)`: the closure's own `Pr_t`, or the
simulation's when that is `nothing`).

`κt` is stored in `f.κh` (so the CFL limit and the implicit IMEX operator
see it), but the scalar tendency is formed as `(1/Prₜ) ∂/∂xⱼ(νt ∂ϕ/∂xⱼ)` by
the `(tϕ, ϕ, νt, Pr_t)` method of [`add_eddy_diffusion_scalar!`](@ref): the
same kernel, with the factor applied after the flux divergence, which is the
order of operations of the pre-contract code. Diffusing with the stored
`κt` instead would differ from it in the last bit (face averages of `νt/Prₜ`
are not bitwise `1/Prₜ` times face averages of `νt`).
"""
abstract type AbstractEddyViscosityClosure <: AbstractSGSClosure end

"""
    sgs_viscosity!(νt, c::AbstractEddyViscosityClosure, sim) -> νt

Fill the interior of the cell-centred eddy viscosity `νt` (the only method an
[`AbstractEddyViscosityClosure`](@ref) has to provide).
"""
function sgs_viscosity! end

"""
    turbulent_prandtl(c::AbstractEddyViscosityClosure, sim) -> Prₜ

The closure's `Pr_t` field when it has one that is not `nothing`, else
`sim.Pr_t`.
"""
turbulent_prandtl(c::AbstractEddyViscosityClosure, sim) = sim.Pr_t
_prandtl(::Nothing, sim) = sim.Pr_t
_prandtl(Pr_t::Real, sim) = Pr_t

function sgs_coefficients!(f::SGSFields, c::AbstractEddyViscosityClosure, sim)
    sgs_viscosity!(f.νh, c, sim)
    # κt = νt/Prₜ by division (not by multiplication with 1/Prₜ): the frozen
    # scalar viscosity of step_imex! has always been formed this way.
    _divide_storage!(f.κh.data, f.νh.data, turbulent_prandtl(c, sim))
    return f
end

# out = a ./ d over the whole storage (halos included), as a pointwise launch
# (KA-l, #304): `x / d` per point, the bits of the broadcast it replaces. A
# named body, not range_map! with `Base.Fix2(/, d)`: that form allocated
# 272 bytes per call (the step_rk3! allocation test of KA-c).
_divide_storage!(out::AbstractArray{<:Any,3}, a::AbstractArray{<:Any,3}, d) =
    launch!(Backends.backend(out), _divide_body!, _storage_ranges(out), out, a, d)

@inline function _divide_body!(i, j, k, out, a, d)
    @inbounds out[i, j, k] = a[i, j, k] / d
    return nothing
end

# κt = νt/Prₜ on the whole array once the halos of νt are filled: one halo
# fill instead of two (κt's halos are then νt's divided by Prₜ, which is what
# fill_halos! would give for the periodic / no-flux / wall copies anyway).
function _fill_sgs_halos!(f::SGSFields, c::AbstractEddyViscosityClosure, sim)
    fill_halos!(f.νh, sim.bcs)
    _divide_storage!(f.κh.data, f.νh.data, turbulent_prandtl(c, sim))
    return f
end

# On a sound-proof core the weighted kernel takes the stored κt = νt/Prₜ
# (its halos are νt's divided by Prₜ, _fill_sgs_halos!).
function add_sgs_scalar!(tϕ, ϕ, c::AbstractEddyViscosityClosure, f::SGSFields, sim)
    W = flux_weights(sim)
    W === nothing ? add_eddy_diffusion_scalar!(tϕ, ϕ, f.νh, turbulent_prandtl(c, sim)) :
                    add_eddy_diffusion_scalar!(tϕ, ϕ, f.κh; weights = W)
    return nothing
end
sgs_scalar_face_diffusivity(c::AbstractEddyViscosityClosure, f::SGSFields, sim) =
    (f.νh, f.νh, 1 / turbulent_prandtl(c, sim))

"""
    StaticSmagorinsky(C_s = 0.17; Pr_t = nothing)

The Smagorinsky–Lilly closure `νt = (C_s Δ)² |S|`
([`compute_smagorinsky_νt!`](@ref)), `κt = νt/Prₜ`; `Pr_t = nothing` takes
the simulation's `Pr_t`. `step!(sim, dt; sgs = StaticSmagorinsky(sim.C_s))`
is, bit for bit, the built-in closure a simulation built with `C_s ≠ 0`
runs. Stateless.

References: Smagorinsky (1963), *Mon. Wea. Rev.* 91, 99; Lilly (1967),
IBM Sci. Comp. Symp.
"""
struct StaticSmagorinsky{T<:Real, P} <: AbstractEddyViscosityClosure
    C_s::T
    Pr_t::P
end
StaticSmagorinsky(C_s::Real = 0.17; Pr_t = nothing) = StaticSmagorinsky(C_s, Pr_t)

sgs_viscosity!(νt, c::StaticSmagorinsky, sim) =
    compute_smagorinsky_νt!(νt, sim.u, sim.v, sim.w, c.C_s)
turbulent_prandtl(c::StaticSmagorinsky, sim) = _prandtl(c.Pr_t, sim)

"""
    DynamicSmagorinsky(workspace::DynamicSmagorinskyWorkspace; averaging = :plane,
                       filter_ratio = 3, cs_max = 0.5, Pr_t = nothing, mask = nothing)
    DynamicSmagorinsky(mesh::RegularGrid; kwargs...)

The Germano–Lilly dynamic Smagorinsky closure,
[`compute_dynamic_smagorinsky_νt!`](@ref) with these keywords and the
workspace (whose `cs2` holds the coefficient of the last call), and
`κt = νt/Prₜ`. Stateless in the sense of the contract: the coefficient is
recomputed from the resolved field at every call, so
[`advance_sgs!`](@ref) is a no-op. `step!(sim, dt; sgs = DynamicSmagorinsky(ws))`
is bit for bit the legacy `eddy_viscosity! = (νt, u, v, w) ->
compute_dynamic_smagorinsky_νt!(νt, u, v, w; workspace = ws)`.

`averaging` is `:plane` or `:local`; the Lagrangian average along
pathlines is the stateful [`LagrangianDynamicSmagorinsky`](@ref).
`mask` (optional, storage-sized cell mask, `true` = solid, e.g.
`IBM.SolidMask.mask`) makes the procedure obstacle aware (see
[`compute_dynamic_smagorinsky_νt!`](@ref)); its halos are filled from the
simulation's boundary conditions on first use (wrapped where periodic).

References: Germano, Piomelli, Moin & Cabot (1991), *Phys. Fluids A* 3,
1760; Lilly (1992), *Phys. Fluids A* 4, 633.
"""
struct DynamicSmagorinsky{W<:DynamicSmagorinskyWorkspace, R<:Real, C<:Real, P, M} <: AbstractEddyViscosityClosure
    workspace::W
    averaging::Symbol
    filter_ratio::R
    cs_max::C
    Pr_t::P
    mask::M
end
function DynamicSmagorinsky(ws::DynamicSmagorinskyWorkspace{T}; averaging::Symbol = :plane,
                            filter_ratio::Real = 3, cs_max::Real = 0.5, Pr_t = nothing,
                            mask = nothing) where {T}
    averaging in (:plane, :local) ||
        throw(ArgumentError("averaging must be :plane or :local (the Lagrangian average is " *
                            "LagrangianDynamicSmagorinsky), got $(repr(averaging))"))
    return DynamicSmagorinsky(ws, averaging, filter_ratio, cs_max, Pr_t, _LazyMask(mask, T))
end
DynamicSmagorinsky(g::RegularGrid; kwargs...) =
    DynamicSmagorinsky(DynamicSmagorinskyWorkspace(g); kwargs...)

sgs_viscosity!(νt, c::DynamicSmagorinsky, sim) =
    compute_dynamic_smagorinsky_νt!(νt, sim.u, sim.v, sim.w; workspace = c.workspace,
                                    averaging = c.averaging, filter_ratio = c.filter_ratio,
                                    cs_max = c.cs_max, mask = _sgs_mask(c.mask, sim))
turbulent_prandtl(c::DynamicSmagorinsky, sim) = _prandtl(c.Pr_t, sim)

# A solid mask whose halos are filled from the simulation's boundary
# conditions the first time a closure sees the simulation (a cache, not
# closure memory: it never changes once built).
struct _LazyMask{A, T}
    src::A
    cache::Base.RefValue{Union{Nothing, _SGSMask{Array{Bool,3}, T}}}
end
_LazyMask(src, ::Type{T}) where {T} =
    _LazyMask(src, Base.RefValue{Union{Nothing, _SGSMask{Array{Bool,3}, T}}}(nothing))
_sgs_mask(::_LazyMask{Nothing}, sim) = nothing
function _sgs_mask(m::_LazyMask, sim)
    c = m.cache[]
    c === nothing || return c
    b = sim.bcs
    per = (b.west isa Periodic, b.south isa Periodic, b.bottom isa Periodic)
    return m.cache[] = _SGSMask(m.src, mesh(sim.u), per)
end

"""
    FunctionSGS(f)

Stateless adapter for a plain function `f(νt, u, v, w)` that fills the eddy
viscosity (the pre-contract `eddy_viscosity!` keyword), with
`κt = νt / sim.Pr_t`. A function passed as `sgs` (or as the legacy
`eddy_viscosity!`) is wrapped in one automatically, so existing user code
keeps working unchanged.
"""
struct FunctionSGS{F} <: AbstractEddyViscosityClosure
    f::F
end
sgs_viscosity!(νt, c::FunctionSGS, sim) = (c.f(νt, sim.u, sim.v, sim.w); νt)

"""
    laminar_consistent(closure) -> Bool

Whether the eddy viscosity of `closure` vanishes identically in laminar
flow — pure shear, solid-body rotation and any two-dimensional flow —
so that a relaminarised region carries no SGS mixing (P59-e, #195). The
σ-model is (Nicoud et al. 2011); Smagorinsky is not (its ``ν_t ∝ |S|``
stays on in laminar shear). Wrappers (`StabilityCorrected`,
`IsentropicDiffusion`) are laminar-consistent when their base closure
is. The Berlin `turbulence_diag` diagnostic asks this to decide which
closures must give ``ν_t = 0`` in a laminar flow on its mesh. A
closure added later (AMD, P55-d) extends it.
"""
laminar_consistent(::AbstractSGSClosure) = false
laminar_consistent(::Nothing) = false

# -------------------------------------------------------------------------
# N-inflow-b (#258): the subgrid TKE of any closure
# -------------------------------------------------------------------------

"""
    sgs_tke!(e::Field{CellCenter}, c, sim; c_k = 0.1) -> e

Fill the interior of `e` with the closure's **subgrid turbulent kinetic
energy** ``e = \\tfrac12\\overline{u_i'u_i'}^{\\,\\rm SGS}`` (m² s⁻²) for the
current state of `sim`. Every closure answers through this one generic, so
a consumer (the inflow statistics of `Transition`, N-inflow-b #258) never
needs to know which closure the parent runs:

* a prognostic closure returns its memory: `PrognosticSGS.TKEClosure` its
  ``e``, `PrognosticSGS.ReynoldsStressClosure` ``\\tfrac12τ_{ii}``;
* every other closure (the default method, and `c = nothing` for a
  simulation's built-in Smagorinsky) returns **its own estimate**: the
  Deardorff (1980) eddy viscosity ``ν_t = c_k\\,l\\,\\sqrt e`` with the
  neutral mixing length ``l = Δ = (ΔxΔyΔz)^{1/3}`` (on a `StretchedGrid`
  with the thickness of each level) inverted,
  ``e = (ν_t/(c_kΔ))^2``, ``c_k = 0.1`` (Moeng 1984). ``ν_t`` is the
  horizontal viscosity the closure last wrote (`sgs_fields(c, sim).νh`,
  or `sim.νt`), so call it after a step (or after `sgs_coefficients!`).
  For Smagorinsky, ``ν_t = (C_sΔ)^2|S|`` and the estimate is the local
  equilibrium value ``e = (C_s^2Δ|S|/c_k)^2``.

References: Deardorff, J. W. (1980). Stratocumulus-capped mixed layers
derived from a three-dimensional model. *Boundary-Layer Meteorol.*, 18,
495-527. Moeng, C.-H. (1984). A large-eddy-simulation model for the study
of planetary boundary-layer turbulence. *J. Atmos. Sci.*, 41, 2052-2062.
"""
function sgs_tke!(e::Field{CellCenter}, c, sim; c_k::Real = 0.1)
    ν = c === nothing ? sim.νt : sgs_fields(c, sim).νh
    return _tke_from_viscosity!(e, ν, c_k)
end

function _tke_from_viscosity!(e::Field, ν::Field, c_k::Real)
    g = mesh(e)
    if hasproperty(g, :dzc)              # a StretchedGrid: Δ of each level (#268)
        sk = [1 / (c_k * (g.dx * g.dy * dz)^(1 / 3)) for dz in g.dzc]
        launch!(Backends.backend(e.data), _tke_from_ν_levels_body!, interior(g), e.data, ν.data, sk)
        return e
    end
    Δ = (g.dx * g.dy * g.dz)^(1 / 3)
    s = 1 / (c_k * Δ)
    launch!(Backends.backend(e.data), _tke_from_ν_body!, interior(g), e.data, ν.data, s)
    return e
end

@inline function _tke_from_ν_body!(i, j, k, ed, νd, s)
    @inbounds ed[i, j, k] = (max(νd[i, j, k], 0) * s)^2
    return nothing
end

@inline function _tke_from_ν_levels_body!(i, j, k, ed, νd, sk)
    @inbounds ed[i, j, k] = (max(νd[i, j, k], 0) * sk[k])^2
    return nothing
end
