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
#                              LargeScaleForcing.GeostrophicNudging, which
#                              since #248 is a constructor of the generic
#                              plane-mean ProfileNudging (one nudging
#                              implementation; bit for bit the former code).
#
# θ/qv nudging and large-scale subsidence (ProfileNudging, Subsidence,
# #248) belong to the deep parent run (N-parent-e, #259), not to this
# coupling: the nested child gets its top and lateral θ/q from the parent,
# so nudging it as well would apply the large scale twice.
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
#
# With `cfg.lateral = :open` (N-open-d, #262) the top is open and the core's
# own sponge already relaxes toward the boundary data, whose wind is
# (U_g, V_g) (`sponge_target`, core/open_boundary.jl): the retargeting here
# is then not applied (it would double the target). The periodic case keeps
# it, bit for bit; the open/nested default of #265 retires it.
#
# Nested (the default, #265): in the pressure-gradient mode the child is
# driven by the parent's geostrophic wind — the same functions U_g(z, t),
# V_g(z, t) the parent's GeostrophicPressureGradient takes
# (`nest_geostrophic_wind`: the sounding wind above z_g, or this coupling's
# (U_g, V_g) without a sounding) — so the large-scale pressure gradient is
# one field across the nest. The core's sponge then relaxes toward the
# parent's mean wind over the child (couplings/nesting.jl).

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
    U_g::Float64                  # the (level-mean) geostrophic wind of the last call
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
    # nested: the parent's geostrophic wind, U_g(z, t) and V_g(z, t) (#265)
    Ug, Vg = o.mode === :pressure_gradient && case_nested(case.cfg) ?
             nest_geostrophic_wind(case.cfg) : (o.U_g, o.V_g)
    F = o.mode === :pressure_gradient ? GeostrophicPressureGradient(Ug, Vg) :
        o.mode === :nudging           ? GeostrophicNudging(o.U_g, o.V_g, o.τ) :
        error("geostrophic: mode must be :pressure_gradient or :nudging, got :$(o.mode)")
    # an open top: the core's sponge already targets the boundary data (#262)
    core_targets = has_open_boundaries(case.sim)
    sponge_ref = o.mode === :pressure_gradient && o.sponge_to_geostrophic && case.sim.has_sponge &&
                 !core_targets
    zu = sponge_ref ? Field(FaceX(), case.sim.mesh) : nothing
    zv = sponge_ref ? Field(FaceY(), case.sim.mesh) : nothing
    return GeostrophicCoupling(F, o.mode, Float64(o.U_g), Float64(o.V_g), sponge_ref, zu, zv,
                               0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
end

# the level mean of a geostrophic wind component over the step [t, t + dt]
# (a number, or a function (z, t): its step mean at the u, v levels, as
# apply_forcing! applies it)
_level_mean(U::Real, sim, t, dt) = Float64(U)
function _level_mean(U, sim, t, dt)
    m = sim.mesh
    return sum(forcing_mean(U, m.z0 + (k - 0.5) * m.dz, t, t + dt) for k in 1:m.Nz) / m.Nz
end

# the interior mean of a field, in the canonical, thread-invariant order (KA-b, #296)
_imean(f) = interior_sum(f) / length(interior_view(f))

function run_stage!(c::GeostrophicCoupling, case, ::Val{:body_force}, dt)
    sim = case.sim
    if c.forcing isa GeostrophicPressureGradient
        c.U_g = _level_mean(c.forcing.U_g, sim, sim.t, dt)
        c.V_g = _level_mean(c.forcing.V_g, sim, sim.t, dt)
    end
    tu0, tv0 = _imean(sim.tu), _imean(sim.tv)
    apply_forcing!(sim, c.forcing, sim.t, dt)
    tu1, tv1 = _imean(sim.tu), _imean(sim.tv)
    c.dtu, c.dtv = tu1 - tu0, tv1 - tv0
    if c.sponge_ref
        # −α(z)(0 − U_g) = +α(z) U_g: with the core's −α(z) u, the sponge
        # relaxes toward the geostrophic wind
        z_top = sim.mesh.z0 + sim.mesh.Lz
        add_sponge!(sim.tu, c.zero_u, sim.sponge_α_max, sim.sponge_z_start, z_top; f_ref = c.U_g)
        add_sponge!(sim.tv, c.zero_v, sim.sponge_α_max, sim.sponge_z_start, z_top; f_ref = c.V_g)
        c.dtu_sp = _imean(sim.tu) - tu1
        c.dtv_sp = _imean(sim.tv) - tv1
    end
    c.ū, c.v̄ = _imean(sim.u), _imean(sim.v)
    return nothing
end

"""
    diagnostics(c::GeostrophicCoupling, case)

The interior mean wind `u_mean`, `v_mean` (m s⁻¹) and the mean forcing
the coupling applied `forcing_u`, `forcing_v` (m s⁻²), every step.
"""
diagnostics(c::GeostrophicCoupling, case) = Diagnostic[
    Diagnostic("u_mean", "m s-1", () -> c.ū; cadence = :step),
    Diagnostic("v_mean", "m s-1", () -> c.v̄; cadence = :step),
    Diagnostic("forcing_u", "m s-2", () -> c.dtu; cadence = :step,
               description = "mean forcing ($(c.mode)$(c.sponge_ref ? ", sponge to U_g" : ""))"),
    Diagnostic("forcing_v", "m s-2", () -> c.dtv; cadence = :step)]

# The open-box variants of the case (#262) live here: the open faces are fed
# with this coupling's (U_g, V_g). Every coupling is on in both.
# (`lateral` is the parent-free alternative: the nest is switched off.)
smoke_variants(::Type{<:GeostrophicCoupling}) =
    ((label = "nudging", options = Dict(:geostrophic => (mode = :nudging,))),
     (label = "lateral = :open, Dirichlet outflow", lateral = :open, physics = Dict(:nesting => false)),
     (label = "lateral = :open, convective outflow", lateral = :open, outflow = :convective,
      physics = Dict(:nesting => false)))

# Whether the case's scalar diffusion is a tensor (the Reynolds-stress
# closure, or the Smagorinsky provider's isentropic scalar tensor): its
# cross terms at the faces are not in the booked diffusive flux (#262).
function _tensor_diffusion(case)
    find_coupling(case, :sgs_reynolds_stress) === nothing || return true
    s = find_coupling(case, :sgs_smagorinsky)
    return s !== nothing && s.closure isa IsentropicDiffusion
end

# The column transport residual an open box leaves (see smoke_check). The
# forward-Euler Boussinesq steps book exactly what the faces carried; the
# known approximations of #262 leave a residual: the IMEX steppers book the
# last stage's rate × dt, the sound-proof cores keep the unweighted-diffusion
# residual of #254, and a tensor diffusion's (`sgs = :reynolds_stress`, the
# isentropic scalar tensor) cross terms at the faces are not in the booked
# diffusive flux. Measured on the analytic open window of #262 ≈ 2e-2 W/m².
# Through the nest faces (#265) the booked heat is larger and so is the
# residual, which scales with it: in the fog-seeded smoke window 0.34 W/m²
# of 8.9 W/m² boundary heat (IMEX, 3.8 %), 0.40 of 8.4 (adaptive + IMEX,
# 4.7 %), 0.44 of 3.3 (Reynolds stress without the wall cap, 13 %). The
# tolerance of these configurations is 20 % of the booked boundary heat plus
# the 0.05 W/m² of the closed top; the exact ones keep 1e-6 W/m².
function _open_transport_tol(case)
    ts = timestepper(case)
    imex = ts isa IMEXStep || (ts isa AdaptiveStep && ts.scheme !== nothing)
    exact = case.sim isa BoussinesqSimulation && !imex && !_tensor_diffusion(case)
    exact && return 1e-6
    hb = heat_budget(case.budgets)
    hb.elapsed > 0 || return 0.05
    booked = get(hb.terms, (:dynamics, :boundary_heat), 0.0) + get(hb.terms, (:dynamics, :mass_correction), 0.0)
    return 0.05 + 0.2 * abs(booked) / hb.elapsed
end

function smoke_check(c::GeostrophicCoupling, case)
    sim = case.sim
    f = sim.f_coriolis
    tol = 1e-10 * (abs(c.U_g) + abs(c.V_g) + 1)
    nested = case_nested(case.cfg)
    open_checks = !has_open_boundaries(sim) ? Pair{String,Bool}[] :
        [nested ?
         ("nested: the core's sponge relaxes toward the parent's mean wind (level vectors, #265)" =>
              sim.sponge_target.u isa AbstractVector && sim.sponge_target.v isa AbstractVector &&
              sim.sponge_target.w == 0) :
         ("open top: the core's sponge relaxes toward (U_g, V_g), not 0 (#262)" =>
              sim.sponge_target.u == c.U_g && sim.sponge_target.v == c.V_g && sim.sponge_target.w == 0),
         "open top: the coupling does not retarget the sponge a second time" => !c.sponge_ref,
         # the open faces' heat is booked (boundary_heat, mass_correction,
         # relaxation): what stays in the column transport is the numerics
         # residual — measured ≤ 1e-9 W/m² in the smoke window for the
         # forward-Euler Boussinesq steps; the IMEX steppers (last-stage
         # rate, CG tolerance) and the sound-proof cores (unweighted
         # diffusion) leave ≈ 2e-2 W/m², as with the closed top (#254)
         (tol = _open_transport_tol(case);
          "open box: the column (:dynamics, :transport) is a residual (< $(round(tol, sigdigits = 3)) W/m²)" =>
              (hb = heat_budget(case.budgets);
               hb.elapsed == 0 || abs(get(hb.terms, (:dynamics, :transport), 0.0)) / hb.elapsed < tol))]
    if c.mode === :pressure_gradient
        # the pressure gradient in balance with (U_g, V_g): exactly
        # (−f V_g, +f U_g) on every u and v point (nested: the level mean of
        # the parent's U_g(z, t), V_g(z, t) at the time of the call)
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
        return vcat(checks, open_checks)
    else
        # every level is relaxed toward (U_g, V_g) at rate 1/τ, so the
        # domain-mean tendency is −(⟨u⟩ − U_g)/τ: toward the target, never away
        τ = c.forcing.τ
        # the tendency integrals ProfileNudging returns, over the domain
        # volume, are the domain-mean tendency measured above
        I = tendency_integrals(c.forcing)
        V = sim.mesh.Lx * sim.mesh.Ly * sim.mesh.Lz
        return vcat(["nudging tendency = −(⟨u⟩ − U_g)/τ" =>
                         abs(c.dtu + (c.ū - c.U_g) / τ) <= tol / τ && abs(c.dtv + (c.v̄ - c.V_g) / τ) <= tol / τ,
                     "one nudging implementation (ProfileNudging), integrals = mean tendency × volume" =>
                         c.forcing isa ProfileNudging && keys(I) == (:u, :v) &&
                         isapprox(I.u / V, c.dtu; rtol = 1e-10, atol = 1e-14) &&
                         isapprox(I.v / V, c.dtv; rtol = 1e-10, atol = 1e-14)],
                    open_checks)
    end
end

register!(GeostrophicCoupling)
