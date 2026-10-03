# Background geostrophic forcing, from Aeolus.LargeScaleForcing (#171),
# in the body-force hook.
#
#   mode = :pressure_gradient  (default) the large-scale pressure gradient
#                              in balance with (U_g, V_g): tu += −f V_g,
#                              tv += +f U_g. With the core's Coriolis term
#                              (f = cfg.f_coriolis = 1e-4) the mean wind
#                              can perform inertial oscillations, so the
#                              nocturnal low-level jet can form.
#   mode = :nudging            Rayleigh relaxation of the horizontal-mean
#                              wind toward (U_g, V_g) on τ (s): the case's
#                              behaviour before #171, now
#                              LargeScaleForcing.GeostrophicNudging.

"""
    GeostrophicCoupling

Additive coupling: the geostrophic forcing of `Aeolus.LargeScaleForcing`
([`apply_forcing!`](@ref) of a `GeostrophicPressureGradient` or a
`GeostrophicNudging`) in `:body_force`. Options `U_g`, `V_g` (m/s),
`mode` (`:pressure_gradient`, default, or `:nudging`) and `τ` (s, the
nudging time scale, default 1800).

It records the domain-mean tendency it added to `u` and `v` in the last
call (`dtu`, `dtv`, m s⁻²) and the domain-mean wind it acted on (`ū`,
`v̄`), for the smoke checks.
"""
mutable struct GeostrophicCoupling{F<:AbstractLargeScaleForcing} <: AbstractCoupling
    forcing::F
    mode::Symbol
    U_g::Float64
    V_g::Float64
    dtu::Float64
    dtv::Float64
    ū::Float64
    v̄::Float64
end

coupling_name(::Type{<:GeostrophicCoupling}) = :geostrophic
modules_used(::Type{<:GeostrophicCoupling}) = (:LargeScaleForcing,)
default_options(::Type{<:GeostrophicCoupling}) =
    (U_g = 1.0, V_g = 0.0, mode = :pressure_gradient, τ = 1800.0)
stages(::Type{<:GeostrophicCoupling}) = ((:body_force, 10, EVERY_STEP),)

function build(::Type{<:GeostrophicCoupling}, case, o)
    F = o.mode === :pressure_gradient ? GeostrophicPressureGradient(o.U_g, o.V_g) :
        o.mode === :nudging           ? GeostrophicNudging(o.U_g, o.V_g, o.τ) :
        error("geostrophic: mode must be :pressure_gradient or :nudging, got :$(o.mode)")
    return GeostrophicCoupling(F, o.mode, Float64(o.U_g), Float64(o.V_g), 0.0, 0.0, 0.0, 0.0)
end

function run_stage!(c::GeostrophicCoupling, case, ::Val{:body_force}, dt)
    sim = case.sim
    tu0, tv0 = mean(interior_view(sim.tu)), mean(interior_view(sim.tv))
    apply_forcing!(sim, c.forcing, sim.t, dt)
    c.dtu = mean(interior_view(sim.tu)) - tu0
    c.dtv = mean(interior_view(sim.tv)) - tv0
    c.ū, c.v̄ = mean(interior_view(sim.u)), mean(interior_view(sim.v))
    return nothing
end

report(c::GeostrophicCoupling, case) =
    @sprintf("geostrophic (%s): mean wind (%.2f, %.2f) m/s, forcing (%.2e, %.2e) m/s²",
             c.mode, c.ū, c.v̄, c.dtu, c.dtv)

smoke_variants(::Type{<:GeostrophicCoupling}) =
    ((label = "nudging", options = Dict(:geostrophic => (mode = :nudging,))),)

function smoke_check(c::GeostrophicCoupling, case)
    f = case.sim.f_coriolis
    tol = 1e-10 * (abs(c.U_g) + abs(c.V_g) + 1)
    if c.mode === :pressure_gradient
        # the pressure gradient in balance with (U_g, V_g): exactly
        # (−f V_g, +f U_g) on every u and v point
        return ["pressure gradient = (−f V_g, f U_g)" =>
                    abs(c.dtu + f * c.V_g) <= tol * abs(f) && abs(c.dtv - f * c.U_g) <= tol * abs(f)]
    else
        # every level is relaxed toward (U_g, V_g) at rate 1/τ, so the
        # domain-mean tendency is −(⟨u⟩ − U_g)/τ: toward the target, never away
        τ = c.forcing.τ
        return ["nudging tendency = −(⟨u⟩ − U_g)/τ" =>
                    abs(c.dtu + (c.ū - c.U_g) / τ) <= tol / τ && abs(c.dtv + (c.v̄ - c.V_g) / τ) <= tol / τ]
    end
end

register!(GeostrophicCoupling)
