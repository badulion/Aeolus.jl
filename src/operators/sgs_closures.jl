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
| [`add_sgs_momentum!`](@ref)`(tu, tv, tw, c, f, sim)` | momentum tendency | Laplacian eddy-viscosity form with `f.νh` |
| [`add_sgs_scalar!`](@ref)`(tϕ, ϕ, c, f, sim)` | θ tendency | eddy diffusion with `f.κh` |
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
"""
sgs_state(::AbstractSGSClosure) = nothing

"""
    restore_sgs_state!(c::AbstractSGSClosure, snapshot) -> c

Restore the memory saved by [`sgs_state`](@ref) (no-op by default).
"""
restore_sgs_state!(c::AbstractSGSClosure, s) = c

"""
    add_sgs_momentum!(tu, tv, tw, c::AbstractSGSClosure, f::SGSFields, sim)

Add the SGS stress divergence to the momentum tendencies. Default: the
Laplacian eddy-viscosity flux form `∂/∂xⱼ(ν ∂uᵢ/∂xⱼ)` of
[`add_eddy_viscosity_u!`](@ref) (and `_v!`, `_w!`) with `f.νh`. A closure
with a different stress (full stress form, `νh ≠ νv`, an explicit Reynolds
stress) adds a method. Halos of `sim.u, sim.v, sim.w` and of `f` are filled.
"""
function add_sgs_momentum!(tu, tv, tw, c::AbstractSGSClosure, f::SGSFields, sim)
    add_eddy_viscosity_u!(tu, sim.u, f.νh)
    add_eddy_viscosity_v!(tv, sim.v, f.νh)
    add_eddy_viscosity_w!(tw, sim.w, f.νh)
    return nothing
end

"""
    add_sgs_scalar!(tϕ, ϕ, c::AbstractSGSClosure, f::SGSFields, sim)

Add the SGS scalar flux divergence `∂/∂xⱼ(κ ∂ϕ/∂xⱼ)` to `tϕ`. Default: the
three-argument [`add_eddy_diffusion_scalar!`](@ref)`(tϕ, ϕ, f.κh)`. Halos of
`ϕ` and of `f` are filled.
"""
add_sgs_scalar!(tϕ, ϕ, c::AbstractSGSClosure, f::SGSFields, sim) =
    (add_eddy_diffusion_scalar!(tϕ, ϕ, f.κh); nothing)

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
    f.κh.data .= f.νh.data ./ turbulent_prandtl(c, sim)
    return f
end

# κt = νt/Prₜ on the whole array once the halos of νt are filled: one halo
# fill instead of two (κt's halos are then νt's divided by Prₜ, which is what
# fill_halos! would give for the periodic / no-flux / wall copies anyway).
function _fill_sgs_halos!(f::SGSFields, c::AbstractEddyViscosityClosure, sim)
    fill_halos!(f.νh, sim.bcs)
    f.κh.data .= f.νh.data ./ turbulent_prandtl(c, sim)
    return f
end

add_sgs_scalar!(tϕ, ϕ, c::AbstractEddyViscosityClosure, f::SGSFields, sim) =
    (add_eddy_diffusion_scalar!(tϕ, ϕ, f.νh, turbulent_prandtl(c, sim)); nothing)

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
    cache::Base.RefValue{Union{Nothing, _SGSMask{BitArray{3}, T}}}
end
_LazyMask(src, ::Type{T}) where {T} =
    _LazyMask(src, Base.RefValue{Union{Nothing, _SGSMask{BitArray{3}, T}}}(nothing))
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
