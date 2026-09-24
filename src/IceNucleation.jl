module IceNucleation

# Era IV, Phase 43 — heterogeneous ice nucleation.
#
# Ice in a mixed-phase cloud between 0 °C and about -38 °C does not form
# on its own: supercooled water is metastable there, and the first ice
# needs a solid surface — an ice-nucleating particle (INP) — to template
# the lattice. INP are rare (typically 1 in 10⁵–10⁶ ambient aerosol
# particles at -20 °C) and, crucially, they are *used up*: a particle that
# has nucleated a crystal is inside that crystal, and falls out with it.
# A scheme that diagnoses ice number from temperature alone therefore
# re-nucleates the same particles every time step, and the Arctic stratus
# it produces glaciates and collapses when the real one persists for days
# (Harrington & Olsson 2001; Morrison et al. 2012; Solomon et al. 2015).
#
# The module is split along that line:
#
#   inp.jl       (issue #5) — INP as prognostic tracers: size-resolved
#                 modal populations of dust and biological particles,
#                 their sources, their sinks (scavenging, sedimentation)
#                 and the available / consumed budget that makes
#                 "cannot nucleate twice" an exact identity.
#   freezing.jl  (issue #6) — immersion and deposition freezing
#                 parameterisations that turn an INP population, T and
#                 supersaturation into ice-number and ice-mass SOURCE
#                 TERMS, drawing their INP sink from inp.jl.
#
# Coupling contract (for the P3 ice category, issue #7): this module
# never touches a hydrometeor state. It returns rates and mutates only the
# INP tracers; the host adds the ice number/mass tendencies to its own
# ice category and removes the corresponding liquid/vapour. That keeps
# nucleation testable in isolation and independent of any one bulk scheme.
#
# One source of truth:
#   * size distributions are the Phase 37/42 log-normal modes
#     (`AerosolOptics.LogNormalMode`, `AerosolCloud.ARGMode`) and their
#     closed-form cut-off integrals (`number_above_radius`,
#     `mass_above_radius`) are reused, not re-derived;
#   * species (density, κ) come from the `AerosolOptics` species library —
#     `:mineral_dust` and `:biological`;
#   * dry-particle settling uses `Superdroplet.stokes_settling_velocity`,
#     the same drag law and air properties as cloud droplets;
#   * thermodynamics (saturation over water and ice, T_freeze, latent
#     heats) comes from `MoistThermodynamics`.
#
# References
#   Kanji et al. (2017), Meteorol. Monogr. 58, 1.1 — review of INP and
#       heterogeneous nucleation modes.
#   Hoose & Möhler (2012), ACP 12, 9817 — laboratory INP review; the
#       surface-site-density (INAS) framework.
#   DeMott et al. (2010), PNAS 107, 11217 — INP number from N(D>0.5 μm).
#   DeMott et al. (2015), ACP 15, 393 — the dust-specific calibration.
#   Niemand et al. (2012), JAS 69, 3077 — dust INAS density.
#   Ullrich et al. (2017), JAS 74, 699 — dust/soot INAS for immersion and
#       deposition.
#   Tobo et al. (2013), JGR 118, 10100 — biological/organic INP.
#   Meyers et al. (1992), JAM 31, 708 — deposition/condensation INP.

using ..MoistThermodynamics
using ..AerosolOptics: AerosolMode, AerosolSpecies, LogNormalMode,
                       aerosol_species
using ..AerosolCloud: ARGMode, arg_mode, mode_dry_mass,
                      number_above_radius, mass_above_radius
import ..AerosolCloud
using ..Superdroplet: MicrophysicalConstants, DEFAULT_MICRO,
                      stokes_settling_velocity
using ..GasOptics: erfc_series

export # --- log-normal surface-area diagnostics (inp.jl) ------------------
       mode_surface_area, surface_area_above_radius,
       DEMOTT_CUTOFF_RADIUS,
       # --- prognostic INP tracers (inp.jl) --------------------------------
       INPPopulation, inp_population, inp_nmodes, inp_species,
       inp_modes, inp_median_radius,
       inp_number, inp_mass, inp_surface_area,
       inp_number_above_radius, inp_surface_area_above_radius,
       inp_emit!, inp_entrain!,
       inp_consume!, inp_nucleate_singular!, inp_frozen_fraction,
       inp_release!, inp_budget, inp_budget_residual

include("icenuc/inp.jl")          # issue #5 — prognostic INP species
# include("icenuc/freezing.jl")   # issue #6 — immersion/deposition freezing

end # module
