# Nesting: per-domain advance closures (N-nest-c, #257; moved out of
# nested_run.jl by N-nest-e #263 so that the two-way feedback, which hooks
# into them, can be defined before NestedRun).

# =========================================================================
# Per-domain advance closures
# =========================================================================

"""
    DomainAdvance

A per-domain `advance!(sim, dt)` closure built by [`domain_advance`](@ref):
a dynamical core plus the hooks and keywords that domain's steps must
carry (`body_force!`, `post_predictor!`, `momentum_advection`, `sgs`,
`sgs_stress`, …). Calling it, `adv(sim, dt)`, runs
`core(sim, dt, args...; kwargs...)`. A caller may pass an extra
`post_predictor!` hook, `adv(sim, dt; post_predictor! = f)`, which is
chained *after* the domain's own, and an extra `body_force!` hook,
`adv(sim, dt; body_force! = f)`, chained *before* the domain's own (the
two-way nest books its flux registers there, N-nest-e #263).
"""
struct DomainAdvance{F, A<:Tuple, K<:NamedTuple}
    core::F
    args::A
    kwargs::K
end

"""
    domain_advance(core = step!, args...; kwargs...) -> DomainAdvance

Bundle a dynamical core with the hooks and keywords of one domain. The
core is called through dispatch, so no step logic is copied:

| `core`, `args`                  | step taken                                             |
| :------------------------------ | :----------------------------------------------------- |
| `step!` (default)               | Boussinesq projection step, or the anelastic / pseudo-incompressible step of a `MassWeightedSimulation` |
| `step_rk3!`                     | SSP-RK3 Boussinesq step                                |
| `step_imex!, ARK2()`            | IMEX ARS(2,2,2) step (or `Rosenbrock2()`)              |
| any `f(sim, dt, args...; kw...)`| a user core |

```julia
child_adv = domain_advance(step_rk3!; body_force! = ibm_force!,
                           post_predictor! = wall_stress!, sgs = AMDClosure())
```

The keywords are forwarded verbatim. `sgs` (or `eddy_viscosity!`) and
`sgs_stress` are also what the run's [`adaptive_dt`](@ref) pick for this
domain uses, and a prognostic SGS closure's memory is rolled back with
the state when a step is rejected.
"""
domain_advance(core = step!, args...; kwargs...) =
    DomainAdvance(core, args, NamedTuple(kwargs))

function (a::DomainAdvance)(sim, dt; post_predictor! = nothing, body_force! = nothing)
    kw = a.kwargs
    if post_predictor! !== nothing
        user = get(kw, :post_predictor!, nothing)
        pp = user === nothing ? post_predictor! :
             (s, h) -> (user(s, h); post_predictor!(s, h))
        kw = merge(kw, NamedTuple{(:post_predictor!,)}((pp,)))
    end
    if body_force! !== nothing
        user = get(kw, :body_force!, nothing)
        bf = user === nothing ? body_force! :
             (s, h) -> (body_force!(s, h); user(s, h))
        kw = merge(kw, NamedTuple{(:body_force!,)}((bf,)))
    end
    a.core(sim, dt, a.args...; kw...)
    return sim
end

_imex_scheme(a::DomainAdvance{typeof(step_imex!)}) = a.args[1]
_imex_scheme(::DomainAdvance) = nothing

_domain_sgs(a::DomainAdvance) = get(a.kwargs, :sgs, get(a.kwargs, :eddy_viscosity!, nothing))

"""
    stage_times(adv::DomainAdvance) -> NTuple

The times, as fractions `cᵢ` of the step `Δt`, at which `adv`'s core
evaluates its tendencies, i.e. the times at which it reads the boundary
data in the halos: `(0,)` for the projection step (`step!`, Boussinesq or
sound-proof) and for a user core, `(0, 1, 1/2)` for SSP-RK3
(Shu & Osher 1988), `(0, γ)` for ARK2 and `(0, a₂₁)` for Rosenbrock2.
The cores' own open-boundary refresh (`update_open_boundaries!` at each
projected stage's end time; for SSP-RK3 the boundary buffers are combined
with the fields' weights) puts the time-blended parent data
`B(t₀ + cᵢΔt)` in the child's halos at exactly these times.
"""
stage_times(a::DomainAdvance) = _stage_times(a.core, a.args)
_stage_times(core, args) = (0.0,)
_stage_times(::typeof(step_rk3!), args) = (0.0, 1.0, 0.5)
_stage_times(::typeof(step_imex!), args) = _imex_stage_times(args[1])
_imex_stage_times(s::ARK2) = (0.0, Float64(s.γ))
_imex_stage_times(s::Rosenbrock2) = (0.0, Float64(s.a21))

