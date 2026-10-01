# The dynamical core: slot `:core`.
#
#   :boussinesq             Simulations.BoussinesqSimulation (default) —
#                           ∇·u = 0, buoyancy g(θ − θ₀)/θ₀ about a constant θ₀.
#   :anelastic              Anelastic.AnelasticSimulation (Lipps–Hemler) —
#                           ∇·(ρ̄u) = 0, buoyancy g(θ − θ̄)/θ̄, ρ̄-weighted
#                           flux-form advection, density-weighted projection.
#   :pseudo_incompressible  Anelastic.PseudoIncompressibleSimulation (Durran
#                           1989) — the same with the weight ρ̄θ̄.
#
# The sound-proof cores take the case's reference state as their base
# state: `case.ref.state` (core/reference_state.jl, #131), the initial θ
# profile (`initial_θ`) integrated hydrostatically from `cfg.p_surface`
# with the MoistThermodynamics constants. It is the same ρ̄(z) the
# couplings convert their fluxes with and the ledger integrates water
# with, so there is one reference state per case. The initial state has
# no buoyancy and the cores start from the same flow.
#
# The dynamics hooks of the case (IBM, canopy drag, geostrophic forcing,
# moist buoyancy, surface drag) are duck-typed on `sim.u/v/w/tu/tv/θ`,
# so they run unchanged on either core. The time-integration providers
# `:adaptive` and `:imex` are typed on BoussinesqSimulation in src and
# declare themselves incompatible with the two sound-proof cores (see
# couplings/timestepper.jl).
#
# The numerics slots (#132) and the sound-proof cores. A sound-proof core
# is its own projection and its own momentum operator: the projection
# solves ∇·(W∇φ) = ∇·(W u*) with a density-weighted CG built from the
# core's base state and weighting, momentum is advected in the W-weighted
# centred flux form, and the SGS closure is the built-in Smagorinsky
# (`_step_mass_weighted!` takes no operator keywords). Hence:
#
#   :pressure_solver     only `:density_weighted_cg` (its `tol`, `maxit`
#                        reach the core's own solver); the Boussinesq
#                        solvers (cosine FFT, multigrid, MG-CG) solve the
#                        constant-coefficient equation and are incompatible;
#   :momentum_advection  `:centered` (the core's weighted centred form; the
#                        provider passes no keyword to a sound-proof core);
#                        `:weno_z` is an unweighted operator: incompatible;
#   :sgs                 `:smagorinsky`; `:dynamic` and `:mlsgs` act through
#                        `eddy_viscosity!`, which the sound-proof step does
#                        not take: incompatible.
#
# `build_core` also refuses any `sim_options` keyword other than `tol` /
# `maxit`, so a future solver provider that forgets to declare itself
# incompatible fails loudly instead of being ignored.
#
# Tracers: under a sound-proof core ∇·u = −w ∂ln W/∂z ≠ 0, so the plain
# flux form would not keep a uniform tracer uniform and would conserve
# Σ q rather than Σ ρ̄ q. The `:scalar_advection` slot therefore advects
# with the mass flux W u (∂(W q)/∂t = −∇·(W u q)) whenever the core is
# sound-proof; see couplings/scalar_advection.jl.

"""
    BoussinesqCore

Slot `:core`, provider `:boussinesq` (default): the dry Boussinesq core
of `Aeolus.Simulations` built by [`boussinesq_core`](@ref) (buoyancy about
the constant `cfg.θ₀`, advanced by `step!`).
"""
struct BoussinesqCore <: AbstractCoupling end

coupling_name(::Type{BoussinesqCore}) = :core_boussinesq
slot(::Type{BoussinesqCore}) = :core
provider(::Type{BoussinesqCore}) = :boussinesq
modules_used(::Type{BoussinesqCore}) = (:Simulations,)
build_core(::Type{BoussinesqCore}, cfg, mesh, ref, simkw) = boussinesq_core(cfg, mesh; simkw...)

function smoke_check(c::BoussinesqCore, case)
    sim = case.sim
    return ["the core is a BoussinesqSimulation" => sim isa BoussinesqSimulation,
            "∇·u at round-off after the projection (< 1e-10 s⁻¹)" => max_divergence(sim) < 1e-10]
end

"""
    MassWeightedCore{P}

Slot `:core`, provider `P ∈ (:anelastic, :pseudo_incompressible)`: the
sound-proof cores of `Aeolus.Anelastic`, `AnelasticSimulation`
(`∇·(ρ̄u) = 0`) and `PseudoIncompressibleSimulation` (`∇·(ρ̄θ̄u) = 0`). The
base state is the case's reference state `case.ref.state`
([`build_reference`](@ref): the initial θ(z) from `cfg.p_surface`).
The projection solver is the core's own density-weighted CG; its `tol`
and `maxit` are options of the `:pressure_solver` provider
`:density_weighted_cg` (the only one these cores accept).

The constraint the projection enforces is reported with every output
record (`report`) and checked by the smoke test.
"""
struct MassWeightedCore{P} <: AbstractCoupling end

const AnelasticCore = MassWeightedCore{:anelastic}
const PseudoIncompressibleCore = MassWeightedCore{:pseudo_incompressible}

coupling_name(::Type{AnelasticCore}) = :core_anelastic
coupling_name(::Type{PseudoIncompressibleCore}) = :core_pseudo_incompressible
slot(::Type{<:MassWeightedCore}) = :core
provider(::Type{MassWeightedCore{P}}) where {P} = P
is_default(::Type{<:MassWeightedCore}) = false
modules_used(::Type{<:MassWeightedCore}) = (:Anelastic,)
incompatible(::Type{<:MassWeightedCore}) =
    (:pressure_solver => :cosine_fft, :pressure_solver => :multigrid, :pressure_solver => :mg_cg,
     :momentum_advection => :weno_z, :sgs => :dynamic, :sgs => :mlsgs)
incompatible_reason(::Type{<:MassWeightedCore}) =
    "a sound-proof core projects with its own density-weighted CG (select " *
    "pressure_solver = :density_weighted_cg), advects momentum in its weighted centred " *
    "flux form (momentum_advection = :centered) and takes no eddy_viscosity! closure " *
    "(sgs = :smagorinsky)"

"The `sim_options` keywords a sound-proof core takes (those of its default projection solver)."
const MASS_WEIGHTED_SIM_KEYWORDS = (:tol, :maxit)

_mass_weighted_type(::Type{AnelasticCore}) = Aeolus.Anelastic.AnelasticSimulation
_mass_weighted_type(::Type{PseudoIncompressibleCore}) = Aeolus.Anelastic.PseudoIncompressibleSimulation

function build_core(C::Type{<:MassWeightedCore}, cfg, mesh, ref, simkw)
    bad = setdiff(keys(simkw), MASS_WEIGHTED_SIM_KEYWORDS)
    isempty(bad) || error("coupling :$(coupling_name(C)): the sim_options keyword(s) " *
                          "$(join(bad, ", ")) are not keywords of the sound-proof core " *
                          "(it takes $(join(MASS_WEIGHTED_SIM_KEYWORDS, ", ")))")
    return _mass_weighted_type(C)(mesh; core_parameters(cfg)..., ref = ref.state,
                                  buoyancy = true, simkw...)
end

function report(c::MassWeightedCore{P}, case) where {P}
    sim = case.sim
    return @sprintf("  %s core: max|∇·(W u)| = %.2e s⁻¹ (plain max|∇·u| = %.2e), CG iterations %d",
                    P, Aeolus.Anelastic.constraint_residual(sim), max_divergence(sim),
                    Aeolus.Anelastic.solver_iterations(sim.solver))
end

function smoke_check(c::MassWeightedCore{P}, case) where {P}
    sim = case.sim
    ref = Aeolus.Anelastic.reference_state(sim)
    θ0 = initial_θ(case.cfg)
    zc = [case.mesh.z0 + (k - 0.5) * case.mesh.dz for k in 1:case.mesh.Nz]
    res = Aeolus.Anelastic.constraint_residual(sim)
    plain = max_divergence(sim)
    return ["the core is the $P MassWeightedSimulation" => sim isa _mass_weighted_type(typeof(c)),
            "the base state is the case's reference state (case.ref.state)" => ref === case.ref.state,
            "θ̄(z) is the initial sounding" => maximum(abs.(ref.θ̄ .- θ0.(0.0, 0.0, zc))) < 1e-10,
            "the base state is in discrete hydrostatic balance (< 1e-8 Pa/m)" =>
                Aeolus.Anelastic.max_hydrostatic_residual(ref) < 1e-8,
            "ρ̄ decreases with height" => all(<(0), diff(ref.ρ̄)),
            # the projection enforces the weighted constraint, not ∇·u = 0
            "∇·(W u) at the solver tolerance (< 1e-6 s⁻¹)" => res < 1e-6,
            "∇·(W u) ≪ plain ∇·u (the constraint is the weighted one)" => res < 1e-2 * plain]
end

register!(BoussinesqCore)
register!(AnelasticCore)
register!(PseudoIncompressibleCore)
