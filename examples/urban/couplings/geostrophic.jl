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
#
# The core's Rayleigh sponge (top 25 %, α_max = 0.05 s⁻¹) relaxes u, v
# toward zero. Against a pressure gradient that is a momentum sink of
# α·⟨u⟩ which the forcing f U_g = 1e-4 m s⁻² cannot balance: in a 24 h
# run the domain-mean wind collapsed to ≈ 0.12 m/s with the top level at
# 0.003 m/s. In the pressure-gradient mode the sponge is therefore
# retargeted at the free-atmosphere wind (U_g, V_g) (`sponge_to_geostrophic`,
# default true), as LES codes with a pressure-gradient forcing do: the
# term +α(z)·U_g is added with the core's own sponge profile
# (`add_sponge!` on a zero field with `f_ref = U_g`), so that the sum is
# −α(z)(u − U_g). The nudging mode is left exactly as it was.

"""
    GeostrophicCoupling

Additive coupling: the geostrophic forcing of `Aeolus.LargeScaleForcing`
([`apply_forcing!`](@ref) of a `GeostrophicPressureGradient` or a
`GeostrophicNudging`) in `:body_force`. Options `U_g`, `V_g` (m/s),
`mode` (`:pressure_gradient`, default, or `:nudging`), `τ` (s, the
nudging time scale, default 1800) and `sponge_to_geostrophic` (default
`true`: in the pressure-gradient mode the core's Rayleigh sponge relaxes
toward `(U_g, V_g)` instead of zero).

It records the domain-mean tendency the forcing added to `u` and `v` in
the last call (`dtu`, `dtv`, m s⁻²), the part the sponge retargeting
added (`dtu_sp`, `dtv_sp`) and the domain-mean wind it acted on (`ū`,
`v̄`), for the smoke checks.
"""
mutable struct GeostrophicCoupling{F<:AbstractLargeScaleForcing,ZU,ZV} <: AbstractCoupling
    forcing::F
    mode::Symbol
    U_g::Float64
    V_g::Float64
    sponge_ref::Bool
    zero_u::ZU
    zero_v::ZV
    dtu::Float64
    dtv::Float64
    dtu_sp::Float64
    dtv_sp::Float64
    ū::Float64
    v̄::Float64
end

coupling_name(::Type{<:GeostrophicCoupling}) = :geostrophic
modules_used(::Type{<:GeostrophicCoupling}) = (:LargeScaleForcing,)
default_options(::Type{<:GeostrophicCoupling}) =
    (U_g = 1.0, V_g = 0.0, mode = :pressure_gradient, τ = 1800.0, sponge_to_geostrophic = true)
stages(::Type{<:GeostrophicCoupling}) = ((:body_force, 10, EVERY_STEP),)

function build(::Type{<:GeostrophicCoupling}, case, o)
    F = o.mode === :pressure_gradient ? GeostrophicPressureGradient(o.U_g, o.V_g) :
        o.mode === :nudging           ? GeostrophicNudging(o.U_g, o.V_g, o.τ) :
        error("geostrophic: mode must be :pressure_gradient or :nudging, got :$(o.mode)")
    sponge_ref = o.mode === :pressure_gradient && o.sponge_to_geostrophic && case.sim.has_sponge
    zu = sponge_ref ? Field(FaceX(), case.sim.mesh) : nothing
    zv = sponge_ref ? Field(FaceY(), case.sim.mesh) : nothing
    return GeostrophicCoupling(F, o.mode, Float64(o.U_g), Float64(o.V_g), sponge_ref, zu, zv,
                               0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
end

function run_stage!(c::GeostrophicCoupling, case, ::Val{:body_force}, dt)
    sim = case.sim
    tu0, tv0 = mean(interior_view(sim.tu)), mean(interior_view(sim.tv))
    apply_forcing!(sim, c.forcing, sim.t, dt)
    tu1, tv1 = mean(interior_view(sim.tu)), mean(interior_view(sim.tv))
    c.dtu, c.dtv = tu1 - tu0, tv1 - tv0
    if c.sponge_ref
        # −α(z)(0 − U_g) = +α(z) U_g: with the core's −α(z) u, the sponge
        # relaxes toward the geostrophic wind
        z_top = sim.mesh.z0 + sim.mesh.Lz
        add_sponge!(sim.tu, c.zero_u, sim.sponge_α_max, sim.sponge_z_start, z_top; f_ref = c.U_g)
        add_sponge!(sim.tv, c.zero_v, sim.sponge_α_max, sim.sponge_z_start, z_top; f_ref = c.V_g)
        c.dtu_sp = mean(interior_view(sim.tu)) - tu1
        c.dtv_sp = mean(interior_view(sim.tv)) - tv1
    end
    c.ū, c.v̄ = mean(interior_view(sim.u)), mean(interior_view(sim.v))
    return nothing
end

report(c::GeostrophicCoupling, case) =
    @sprintf("geostrophic (%s%s): mean wind (%.2f, %.2f) m/s, forcing (%.2e, %.2e) m/s²",
             c.mode, c.sponge_ref ? ", sponge → U_g" : "", c.ū, c.v̄, c.dtu, c.dtv)

smoke_variants(::Type{<:GeostrophicCoupling}) =
    ((label = "nudging", options = Dict(:geostrophic => (mode = :nudging,))),)

function smoke_check(c::GeostrophicCoupling, case)
    sim = case.sim
    f = sim.f_coriolis
    tol = 1e-10 * (abs(c.U_g) + abs(c.V_g) + 1)
    if c.mode === :pressure_gradient
        # the pressure gradient in balance with (U_g, V_g): exactly
        # (−f V_g, +f U_g) on every u and v point
        checks = ["pressure gradient = (−f V_g, f U_g)" =>
                      abs(c.dtu + f * c.V_g) <= tol * abs(f) && abs(c.dtv - f * c.U_g) <= tol * abs(f)]
        if c.sponge_ref
            # the retargeting adds U_g·α(z) on the sponge levels: its
            # domain mean is U_g times the level mean of
            # α(z) = α_max sin²(π/2 (z − z_s)/(z_top − z_s)), z at the u, v levels
            m = sim.mesh
            z_top = m.z0 + m.Lz
            ᾱ = sum(1:m.Nz) do k
                z = m.z0 + (k - 0.5) * m.dz
                z >= sim.sponge_z_start ?
                    sim.sponge_α_max * sin(π / 2 * (z - sim.sponge_z_start) / (z_top - sim.sponge_z_start))^2 : 0.0
            end / m.Nz
            push!(checks, "sponge retargeted at (U_g, V_g): mean +ᾱ (U_g, V_g)" =>
                      ᾱ > 0 && abs(c.dtu_sp - ᾱ * c.U_g) <= tol * ᾱ && abs(c.dtv_sp - ᾱ * c.V_g) <= tol * ᾱ)
        end
        return checks
    else
        # every level is relaxed toward (U_g, V_g) at rate 1/τ, so the
        # domain-mean tendency is −(⟨u⟩ − U_g)/τ: toward the target, never away
        τ = c.forcing.τ
        return ["nudging tendency = −(⟨u⟩ − U_g)/τ" =>
                    abs(c.dtu + (c.ū - c.U_g) / τ) <= tol / τ && abs(c.dtv + (c.v̄ - c.V_g) / τ) <= tol / τ]
    end
end

register!(GeostrophicCoupling)
