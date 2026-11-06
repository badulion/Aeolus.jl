module Superdroplet

# Phase 41 — Superdroplet (Lagrangian probabilistic) microphysics.
#
# THE IDEA, AND WHY IT IS THE REFERENCE
#
# A bulk scheme carries a couple of moments of the droplet size
# distribution and must *assume* a shape (gamma, lognormal, exponential)
# to close the system. Every process rate it computes — autoconversion,
# accretion, sedimentation — is then an integral over an assumed shape,
# and the assumption is doing at least as much work as the physics.
#
# The superdroplet method (Shima et al. 2009) removes the assumption.
# The droplet population is represented by `N_sd` *computational*
# particles; particle `i` carries a **multiplicity** ξ_i — the number of
# real droplets it stands for — plus the attributes those real droplets
# share (wet radius, dissolved solute mass, position). Collisions are a
# Monte Carlo process on the computational particles, condensation is the
# diffusional growth ODE solved per particle, and the size distribution is
# never assumed: it is *measured* from the ensemble at the end. That is
# exactly what makes this the probabilistic reference against which a bulk
# scheme (Phase 40's P3, say) is judged.
#
# THE SUBTLE PART: ALL-OR-NOTHING COLLISIONS
#
# The naive Monte Carlo over all N(N-1)/2 pairs is O(N²) and unusable.
# Shima's insight is that one may instead draw ⌊N/2⌋ **non-overlapping**
# random pairs — O(N) — and multiply every collision probability by
#
#     scale = [N(N-1)/2] / ⌊N/2⌋
#
# to keep the expected number of collisions right. Because the individual
# probabilities are then large, one cannot simply flip a coin: the
# *multiple-coalescence* rule is used instead, where γ = ⌊p⌋ (plus one
# more with probability p - ⌊p⌋) coalescences happen at once, and the
# multiplicity bookkeeping absorbs them exactly. Getting that bookkeeping
# right is the whole ballgame — it is what makes water mass conserve to
# round-off while the droplet number decays at the analytically correct
# rate. The Golovin validation in `validation/superdroplet.jl` is a hard external
# check on precisely this.
#
# WHAT IS *NOT* HERE, STATED PLAINLY
#
#  * The Hall (1980) collision-efficiency table is not embedded. It is a
#    15×21 array of measured numbers; reproducing it from memory would be
#    worse than useless, because a silently wrong efficiency is very hard
#    to detect downstream. Long's (1974) analytic fit to the same data is
#    used instead, and is labelled as such.
#  * Beard's (1976) third terminal-velocity regime (drops above ~1.07 mm)
#    is replaced by the Atlas et al. (1973) exponential fit, blended
#    continuously onto regime 2. Same reason.
#  * No ventilation correction on diffusional growth, no turbulent
#    collision enhancement, no breakup. All three are real and all three
#    are additive refinements on top of this core.
#
# REFERENCES
#
#  * Shima, S., Kusano, K., Kawano, A., Sugiyama, T. & Kawahara, S.
#    (2009). The super-droplet method for the numerical simulation of
#    clouds and precipitation. *Q. J. R. Meteorol. Soc.*, 135, 1307-1320.
#  * Golovin, A. M. (1963). The solution of the coagulation equation for
#    raindrops, taking condensation into account. *Sov. Phys. Dokl.*, 8,
#    191-193.
#  * Scott, W. T. (1968). Analytic studies of cloud droplet coalescence.
#    *J. Atmos. Sci.*, 25, 54-65.
#  * Long, A. B. (1974). Solutions to the droplet collection equation for
#    polynomial kernels. *J. Atmos. Sci.*, 31, 1040-1052.
#  * Beard, K. V. (1976). Terminal velocity and shape of cloud and
#    precipitation drops aloft. *J. Atmos. Sci.*, 33, 851-864.
#  * Atlas, D., Srivastava, R. C. & Sekhon, R. S. (1973). Doppler radar
#    characteristics of precipitation at vertical incidence.
#    *Rev. Geophys.*, 11, 1-35.
#  * Petters, M. D. & Kreidenweis, S. M. (2007). A single parameter
#    representation of hygroscopic growth and cloud condensation nucleus
#    activity. *Atmos. Chem. Phys.*, 7, 1961-1971.
#  * Rogers, R. R. & Yau, M. K. (1989). *A Short Course in Cloud Physics*,
#    3rd ed. Pergamon. Ch. 7 (diffusional growth), Ch. 8 (collection).
#  * Pruppacher, H. R. & Klett, J. D. (1997). *Microphysics of Clouds and
#    Precipitation*, 2nd ed. Kluwer.

using Random: AbstractRNG, MersenneTwister, shuffle!, rand
using ..Backends: AbstractBackend, default_backend, launch!, launch_weighted!, counter_rng,
                  counter_xoshiro

# MoistThermodynamics is the shared core (Phase 39): saturation vapour
# pressure, latent heats and the thermodynamic constants all come from
# there rather than being redefined here, so this module and the bulk
# schemes cannot disagree about what saturation means.
using ..MoistThermodynamics
import ..MoistThermodynamics: effective_radius

export
    # constants & aerosol
    MicrophysicalConstants, DEFAULT_MICRO,
    AerosolProperties, NACL, AMMONIUM_SULFATE, SEASALT_KAPPA,
    aerosol_from_vant_hoff, dry_radius, dry_mass,
    surface_tension, vapor_diffusivity, thermal_conductivity,
    kelvin_a, raoult_b, kappa_kohler_supersaturation,
    droplet_kohler_supersaturation,
    droplet_critical_radius, droplet_critical_supersaturation,
    droplet_equilibrium_radius,
    # state
    SuperdropletSet, nsuperdroplets, add_superdroplet!, compact!,
    droplet_volume, droplet_mass, total_water_mass, total_number,
    init_exponential_superdroplets, init_monodisperse_superdroplets,
    assign_cells!, cell_populations, advance_positions!,
    # kernels
    CollisionKernel, GolovinKernel, HydrodynamicKernel, GravitationalKernel,
    LongKernel, collision_kernel,
    terminal_velocity, stokes_settling_velocity, collision_efficiency, LongEfficiency, UnitEfficiency,
    # collision-coalescence
    coalescence_step!, collision_coalescence!, coalescence_probability,
    # condensation
    diffusional_growth_rate, fk_thermal, fd_vapor,
    grow_droplet, condensation_step!, condensation_growth!,
    # Eulerian coupling
    EulerianCell, cell_supersaturation, cell_pressure, cell_total_water,
    cell_air_mass,
    liquid_water_mixing_ratio, apply_condensate_tendency!,
    deposit_liquid_water!, cell_from_state,
    # diagnostics
    number_concentration, liquid_water_content, mean_radius,
    relative_dispersion, radius_moment,
    size_distribution, mass_distribution, radius_percentile,
    # analytic reference
    bessel_i1, bessel_i1_scaled, golovin_number_density,
    golovin_mass_density, golovin_T

# =========================================================================
# Constants
# =========================================================================

"""
    MicrophysicalConstants{T}

Material properties of water, air and the vapour/heat diffusion pair that
the droplet-scale physics needs and that `MoistThermodynamics` — which is
a *bulk* thermodynamics contract — does not carry.

Nothing thermodynamic is duplicated here: `L_v`, `e_s`, `R_v` and the moist
heat capacity all come from `MoistThermodynamics`. What lives here is the
stuff that only appears once you resolve individual drops: surface tension
(Kelvin term), liquid density, dynamic viscosity and mean free path of air
(terminal velocity), and the diffusivities of vapour and heat.
"""
Base.@kwdef struct MicrophysicalConstants{T<:AbstractFloat}
    ρ_water::T   = 1000.0      # kg m⁻³
    ρ_air::T     = 1.225       # kg m⁻³, reference for terminal velocity
    μ_air::T     = 1.818e-5    # Pa s, dynamic viscosity at 293 K
    λ_air::T     = 6.62e-8     # m, mean free path at 1013 hPa, 293 K
    σ_water0::T  = 0.0761      # N m⁻¹, surface tension at 0 °C
    dσ_dT::T     = -1.55e-4    # N m⁻¹ K⁻¹
    D_v0::T      = 2.11e-5     # m² s⁻¹, vapour diffusivity at 273.15 K, 1 atm
    D_v_exp::T   = 1.94        # temperature exponent
    p_ref_diff::T = 101325.0   # Pa, reference pressure for D_v
    grav::T      = 9.80665     # m s⁻²
end

const DEFAULT_MICRO = MicrophysicalConstants{Float64}()

"""
    surface_tension(T, m = DEFAULT_MICRO) -> N m⁻¹

Surface tension of the water/air interface, `σ = σ₀ + (dσ/dT)(T - T₀)`
with `T₀ = 273.15 K`. Gives 0.0728 N m⁻¹ at 20 °C, the textbook value.

The temperature dependence is not cosmetic: σ enters the Kelvin term
linearly, and the Kelvin term is what sets the critical supersaturation,
so a 10 K error here moves activation thresholds by ~1 %.
"""
surface_tension(T::Real, m::MicrophysicalConstants = DEFAULT_MICRO) =
    m.σ_water0 + m.dσ_dT * (float(T) - 273.15)

"""
    vapor_diffusivity(T, p, m = DEFAULT_MICRO) -> m² s⁻¹

Diffusivity of water vapour in air, `D_v = D₀ (T/T₀)^1.94 (p₀/p)`
(Pruppacher & Klett 1997, eq. 13-3). ≈ 2.2e-5 m² s⁻¹ near the surface,
rising to ~4e-5 at 500 hPa — which is why droplets grow faster aloft at
equal supersaturation.
"""
vapor_diffusivity(T::Real, p::Real, m::MicrophysicalConstants = DEFAULT_MICRO) =
    m.D_v0 * (float(T) / 273.15)^m.D_v_exp * (m.p_ref_diff / float(p))

"""
    thermal_conductivity(T, m = DEFAULT_MICRO) -> W m⁻¹ K⁻¹

Thermal conductivity of air, `K_a = 4.1868e-3 (5.69 + 0.017 T_C)`
(Rogers & Yau 1989, Table 7.1). 0.0238 at 0 °C, 0.0243 at 20 °C.
"""
thermal_conductivity(T::Real, ::MicrophysicalConstants = DEFAULT_MICRO) =
    4.1868e-3 * (5.69 + 0.017 * (float(T) - 273.15))

# =========================================================================
# Aerosol and Köhler theory
# =========================================================================

"""
    AerosolProperties{T}

A soluble aerosol species, described by the Petters & Kreidenweis (2007)
hygroscopicity parameter `κ` and its dry density `ρ_s`.

`κ` collapses molecular weight, density, van 't Hoff factor and osmotic
coefficient into one number calibrated against measured growth factors:
κ ≈ 1.28 for NaCl, 0.61 for ammonium sulfate, ~0.1 for aged organics,
0 for an insoluble (wettable) particle. It is used here instead of the
raw van 't Hoff bundle because it is what measurements actually constrain.

Construct one from van 't Hoff quantities with
[`aerosol_from_vant_hoff`](@ref) if that is the form you have.
"""
struct AerosolProperties{T<:AbstractFloat}
    κ::T          # hygroscopicity
    ρ_s::T        # dry-particle density, kg m⁻³
    name::String
end

AerosolProperties(κ::Real, ρ_s::Real, name::AbstractString = "aerosol") =
    AerosolProperties{typeof(float(κ))}(float(κ), float(ρ_s), String(name))

const NACL              = AerosolProperties(1.28, 2160.0, "NaCl")
const AMMONIUM_SULFATE  = AerosolProperties(0.61, 1770.0, "(NH4)2SO4")
const SEASALT_KAPPA     = AerosolProperties(1.10, 2170.0, "sea salt")

Base.show(io::IO, a::AerosolProperties) =
    print(io, "AerosolProperties(", a.name, ", κ=", a.κ, ", ρ_s=", a.ρ_s, ")")

"""
    aerosol_from_vant_hoff(i, M_s, ρ_s, name; M_w = 0.018015268)

Convert the classical van 't Hoff description of a solute (dissociation
number `i`, molar mass `M_s`, density `ρ_s`) into a κ value, using
`κ = i ρ_s M_w / (ρ_w M_s)` — the dilute-limit identification of the two
formulations (Petters & Kreidenweis 2007, eq. 8).

For NaCl (`i=2`, `M_s=0.05844`, `ρ_s=2160`) this gives κ = 1.33, close to
the measured 1.28; the 4 % gap is the non-ideality the measured κ absorbs.
"""
function aerosol_from_vant_hoff(i::Real, M_s::Real, ρ_s::Real,
                                name::AbstractString = "aerosol";
                                M_w::Real = 0.018015268, ρ_w::Real = 1000.0)
    κ = float(i) * float(ρ_s) * float(M_w) / (float(ρ_w) * float(M_s))
    return AerosolProperties(κ, ρ_s, name)
end

"Dry-particle radius of a solute mass `m_s` (m)."
dry_radius(m_s::Real, a::AerosolProperties) =
    cbrt(3 * float(m_s) / (4π * a.ρ_s))

"Dry solute mass of a particle of dry radius `r_d` (kg)."
dry_mass(r_d::Real, a::AerosolProperties) =
    (4π / 3) * a.ρ_s * float(r_d)^3

"""
    kelvin_a(T, m = DEFAULT_MICRO, c = DEFAULT_CONSTANTS) -> m

The Kelvin (curvature) coefficient `a = 2σ / (ρ_w R_v T)`, so that the
equilibrium supersaturation over a pure-water drop of radius `r` is
`a/r`. About 1.2e-9 m at 273 K — i.e. a 1 μm pure drop needs 0.12 %
supersaturation just to stay alive, and a 10 nm one needs 12 %.

That divergence as `r → 0` is exactly why cloud droplets cannot nucleate
homogeneously in the atmosphere and need aerosol.
"""
function kelvin_a(T::Real, m::MicrophysicalConstants = DEFAULT_MICRO,
                  c::ThermodynamicConstants = DEFAULT_CONSTANTS)
    Rv = c.gas_constant / c.molmass_water
    return 2 * surface_tension(T, m) / (m.ρ_water * Rv * float(T))
end

"""
    raoult_b(m_s, a::AerosolProperties) -> m³

The solute (Raoult) coefficient `b = κ r_d³`, so that the solute lowering
of equilibrium supersaturation is `-b/r³`.

`b = κ r_d³` is the dilute limit of the κ-Köhler expression; see
[`kappa_kohler_supersaturation`](@ref) for the exact form and the test
suite for the demonstration that they agree to better than 1 % once
`r ≳ 3 r_d`, which covers everything from haze upward.
"""
raoult_b(m_s::Real, a::AerosolProperties) = a.κ * dry_radius(m_s, a)^3

"""
    droplet_kohler_supersaturation(r, m_s, T, a, mc, c) -> S_eq

Equilibrium supersaturation over a solution droplet, in the classical
two-term Köhler form

    S_eq(r) = a/r - b/r³

with `a` the Kelvin coefficient and `b = κ r_d³` the Raoult coefficient.

The curve rises from strongly negative at haze sizes (the solute holds
water at sub-saturation), peaks at the **critical radius**
`r_c = √(3b/a)` with value the **critical supersaturation**
`S_c = √(4a³/27b)`, and decays as `a/r` beyond. A droplet held at ambient
`S > S_c` is past the barrier and grows without bound — it is *activated*;
one at `S < S_c` settles onto the stable haze branch. Nothing imposes
this: it is a consequence of the shape of the curve, and it is why this
module can let activation *emerge* rather than parameterising it.
"""
function droplet_kohler_supersaturation(r::Real, m_s::Real, T::Real,
                                a::AerosolProperties,
                                m::MicrophysicalConstants = DEFAULT_MICRO,
                                c::ThermodynamicConstants = DEFAULT_CONSTANTS)
    A = kelvin_a(T, m, c)
    B = raoult_b(m_s, a)
    rr = float(r)
    return A / rr - B / rr^3
end

"""
    kappa_kohler_supersaturation(r, m_s, T, a, mc, c) -> S_eq

The **exact** κ-Köhler equilibrium supersaturation (Petters &
Kreidenweis 2007, eq. 6):

    S_eq(r) + 1 = (r³ - r_d³) / (r³ - r_d³(1-κ)) · exp(a/r)

valid down to `r → r_d`, where it correctly returns `-1` (a completely
dry particle has zero water activity) instead of the classical form's
divergence. Used as the reference the two-term form is checked against.
"""
function kappa_kohler_supersaturation(r::Real, m_s::Real, T::Real,
                                      a::AerosolProperties,
                                      m::MicrophysicalConstants = DEFAULT_MICRO,
                                      c::ThermodynamicConstants = DEFAULT_CONSTANTS)
    return kohler_supersaturation(r, dry_radius(m_s, a), a.κ, T, c)
end

"""
    droplet_critical_radius(m_s, T, a, mc, c) -> m

Köhler critical radius `r_c = √(3b/a)`, the top of the activation barrier.
For a 1e-16 kg NaCl particle at 283 K it is about 1.0 μm — roughly 50
times the dry radius.
"""
function droplet_critical_radius(m_s::Real, T::Real, a::AerosolProperties,
                         m::MicrophysicalConstants = DEFAULT_MICRO,
                         c::ThermodynamicConstants = DEFAULT_CONSTANTS)
    # Delegates to the MoistThermodynamics shared core. This module is parameterised by
    # solute MASS (what a superdroplet carries) and the contract by dry
    # RADIUS and kappa (what is measured); `dry_radius` converts, and
    # `raoult_b = kappa r_d^3` is the same relation the contract uses. One
    # implementation, two entry points — see `MoistThermodynamics`.
    rd = dry_radius(m_s, a)
    return first(kohler_critical_point(rd, a.κ, T, c))
end

"""
    droplet_critical_supersaturation(m_s, T, a, mc, c) -> S_c

Köhler critical supersaturation `S_c = √(4a³/27b)`, the height of the
activation barrier. Scales as `m_s^(-1/2)`: a ten-fold heavier nucleus
activates at a ~3.2 times lower supersaturation, which is why the largest
aerosol always activate first and why the activated fraction is so
sensitive to the updraught that sets peak `S`.
"""
function droplet_critical_supersaturation(m_s::Real, T::Real, a::AerosolProperties,
                                  m::MicrophysicalConstants = DEFAULT_MICRO,
                                  c::ThermodynamicConstants = DEFAULT_CONSTANTS)
    rd = dry_radius(m_s, a)
    return last(kohler_critical_point(rd, a.κ, T, c))
end

"""
    droplet_equilibrium_radius(m_s, S, T, a, mc, c) -> m

The stable (haze) equilibrium radius of an unactivated particle at ambient
supersaturation `S < S_c`: the smaller of the two roots of
`S_eq(r) = S`, found by bisection on `[r_d, r_c]` where `S_eq` is
monotonically increasing.

Returns `NaN` if `S ≥ S_c`, because then there is no equilibrium — the
particle activates.
"""
function droplet_equilibrium_radius(m_s::Real, S::Real, T::Real,
                                    a::AerosolProperties,
                                    m::MicrophysicalConstants = DEFAULT_MICRO,
                                    c::ThermodynamicConstants = DEFAULT_CONSTANTS)
    rd = dry_radius(m_s, a)
    return kohler_equilibrium_radius(float(S), rd, a.κ, T, c)
end

# =========================================================================
# Superdroplet state — struct of arrays
# =========================================================================

"""
    SuperdropletSet{T}

The computational-particle ensemble, stored **struct-of-arrays**: one
`Vector` per attribute rather than one `Vector` of structs.

That layout is not a micro-optimisation here, it is the natural one. The
two hot loops touch disjoint attribute sets — collision-coalescence reads
`ξ` and `radius` only, condensation reads `radius`, `solute_mass` and
writes `radius` — so an AoS layout would drag the position vectors
through cache on every collision sweep for nothing. It also makes the
`compact!` operation (removing multiplicity-zero particles) a simple
per-array pass.

# Fields

  * `ξ::Vector{Int}` — multiplicity: the number of *real* droplets this
    computational particle represents. Integer on purpose; the whole
    exactness of the coalescence bookkeeping rests on it.
  * `radius::Vector{T}` — wet radius (m).
  * `solute_mass::Vector{T}` — dissolved solute mass (kg).
  * `x, y, z::Vector{T}` — position (m).
  * `u, v, w::Vector{T}` — velocity (m s⁻¹); `w` is the *air* velocity,
    sedimentation is added separately at advection time.
  * `cell::Vector{Int}` — host Eulerian cell index (1-based; 0 = unassigned).
  * `aerosol::AerosolProperties{T}` — the solute species (one per set).
"""
struct SuperdropletSet{T<:AbstractFloat}
    ξ::Vector{Int}
    radius::Vector{T}
    solute_mass::Vector{T}
    x::Vector{T}
    y::Vector{T}
    z::Vector{T}
    u::Vector{T}
    v::Vector{T}
    w::Vector{T}
    cell::Vector{Int}
    aerosol::AerosolProperties{T}
end

"""
    SuperdropletSet(T = Float64; aerosol = NACL, n = 0)

An empty set (or one with `n` placeholder particles) for solute species
`aerosol`.
"""
function SuperdropletSet(::Type{T} = Float64; aerosol::AerosolProperties = NACL,
                         n::Integer = 0) where {T<:AbstractFloat}
    aT = AerosolProperties{T}(T(aerosol.κ), T(aerosol.ρ_s), aerosol.name)
    return SuperdropletSet{T}(zeros(Int, n), zeros(T, n), zeros(T, n),
                              zeros(T, n), zeros(T, n), zeros(T, n),
                              zeros(T, n), zeros(T, n), zeros(T, n),
                              zeros(Int, n), aT)
end

"Number of *computational* particles (not real droplets)."
nsuperdroplets(sd::SuperdropletSet) = length(sd.ξ)
Base.length(sd::SuperdropletSet) = nsuperdroplets(sd)

"""
    add_superdroplet!(sd; ξ, radius, solute_mass = 0, x = 0, y = 0, z = 0,
                      u = 0, v = 0, w = 0, cell = 0)

Append one computational particle.
"""
function add_superdroplet!(sd::SuperdropletSet{T}; ξ::Integer,
                           radius::Real, solute_mass::Real = 0,
                           x::Real = 0, y::Real = 0, z::Real = 0,
                           u::Real = 0, v::Real = 0, w::Real = 0,
                           cell::Integer = 0) where {T}
    ξ < 0 && throw(ArgumentError("add_superdroplet!: multiplicity must be ≥ 0"))
    radius < 0 && throw(ArgumentError("add_superdroplet!: radius must be ≥ 0"))
    push!(sd.ξ, Int(ξ));           push!(sd.radius, T(radius))
    push!(sd.solute_mass, T(solute_mass))
    push!(sd.x, T(x)); push!(sd.y, T(y)); push!(sd.z, T(z))
    push!(sd.u, T(u)); push!(sd.v, T(v)); push!(sd.w, T(w))
    push!(sd.cell, Int(cell))
    return sd
end

"""
    compact!(sd) -> sd

Remove every particle whose multiplicity has fallen to zero.

Zero-multiplicity particles are produced by the coalescence rule when a
pair with equal multiplicities merges and one half of the split rounds to
zero. They are harmless (they represent no real water) but they cost work
and they dilute the effective `N_sd`, so they are swept periodically
rather than every step.
"""
function compact!(sd::SuperdropletSet)
    keep = findall(>(0), sd.ξ)
    length(keep) == length(sd.ξ) && return sd
    for f in (sd.ξ, sd.radius, sd.solute_mass, sd.x, sd.y, sd.z,
              sd.u, sd.v, sd.w, sd.cell)
        permute_keep!(f, keep)
    end
    return sd
end

function permute_keep!(v::Vector, keep::Vector{Int})
    @inbounds for (i, k) in enumerate(keep)
        v[i] = v[k]
    end
    resize!(v, length(keep))
    return v
end

"Volume of droplet `i` (m³)."
droplet_volume(sd::SuperdropletSet, i::Integer) = (4π / 3) * sd.radius[i]^3

"Water mass of droplet `i` (kg), solute excluded."
droplet_mass(sd::SuperdropletSet, i::Integer,
             m::MicrophysicalConstants = DEFAULT_MICRO) =
    m.ρ_water * droplet_volume(sd, i)

"""
    total_water_mass(sd, m = DEFAULT_MICRO; cell = nothing) -> kg

Total *real* liquid water mass represented by the ensemble,
`Σ ξ_i ρ_w (4π/3) r_i³`. The invariant that collision-coalescence must
preserve to round-off.
"""
function total_water_mass(sd::SuperdropletSet{T},
                          m::MicrophysicalConstants = DEFAULT_MICRO;
                          cell::Union{Nothing,Integer} = nothing) where {T}
    s = zero(float(T))
    @inbounds for i in eachindex(sd.ξ)
        (cell === nothing || sd.cell[i] == cell) || continue
        s += sd.ξ[i] * sd.radius[i]^3
    end
    return m.ρ_water * (4π / 3) * s
end

"""
    total_number(sd; cell = nothing) -> Int

Number of **real** droplets, `Σ ξ_i`. Distinct from
[`nsuperdroplets`](@ref), which counts computational particles.
Coalescence reduces this monotonically; it leaves the water mass alone.
"""
function total_number(sd::SuperdropletSet; cell::Union{Nothing,Integer} = nothing)
    s = 0
    @inbounds for i in eachindex(sd.ξ)
        (cell === nothing || sd.cell[i] == cell) || continue
        s += sd.ξ[i]
    end
    return s
end

# ------------------------------------------------------------------------
# Initialisation
# ------------------------------------------------------------------------

"""
    init_exponential_superdroplets(rng, N_sd, n0, r_mean, volume;
                                   aerosol = NACL, solute_mass = 0,
                                   stratified = false, T = Float64)

Sample `N_sd` superdroplets from the **exponential mass distribution**

    n(x) dx = (n₀/x̄) exp(-x/x̄) dx,    x̄ = (4π/3) ρ_w r_mean³

at number concentration `n₀` (m⁻³) in a cell of `volume` (m³), using
*constant multiplicity* `ξ = round(n₀ · volume / N_sd)`.

This is Shima et al.'s (2009) initialisation and it is the initial
condition for which Golovin's analytic coalescence solution is written,
so it is what the headline validation needs.

Constant multiplicity rather than constant mass-per-particle: sampling
the *number* distribution and giving every computational particle the same
weight is unbiased for number-weighted quantities and is what makes the
Monte Carlo error scale as `N_sd^(-1/2)`.

With `stratified = true` the uniform deviates are drawn one per equal-
probability stratum instead of independently. This removes the leading
sampling noise (it is a quasi-Monte-Carlo variance reduction, not a change
of distribution) and is useful when one wants the deterministic part of
the coalescence error rather than the sampling part; the convergence study
uses `stratified = false` so that the `N^(-1/2)` law is actually visible.

**Random streams** (KA-n, #305). One `UInt64` is drawn from `rng`; droplet
`i` then takes its deviate from the counter-based stream `(key, 0, i)`
([`Aeolus.Backends.counter_rng`](@ref)), in a launch on `backend`, so the
population does not depend on the thread count.
"""
function init_exponential_superdroplets(rng::AbstractRNG, N_sd::Integer,
                                        n0::Real, r_mean::Real, volume::Real;
                                        aerosol::AerosolProperties = NACL,
                                        solute_mass::Real = 0.0,
                                        stratified::Bool = false,
                                        ρ_water::Real = DEFAULT_MICRO.ρ_water,
                                        T::Type{<:AbstractFloat} = Float64,
                                        backend::AbstractBackend = default_backend())
    N_sd > 0 || throw(ArgumentError("init_exponential_superdroplets: N_sd must be > 0"))
    sd = SuperdropletSet(T; aerosol = aerosol)
    x̄ = (4π / 3) * float(ρ_water) * float(r_mean)^3     # mean droplet mass, kg
    ξ = round(Int, float(n0) * float(volume) / N_sd)
    ξ >= 1 || throw(ArgumentError("init_exponential_superdroplets: n0·volume/N_sd < 1; " *
                                  "increase the volume or reduce N_sd"))
    # droplet i draws from the counter-based stream (key, 0, i) (KA-n, #305)
    key = rand(rng, UInt64)
    radii = Vector{Float64}(undef, N_sd)
    launch!(backend, _exponential_radius!, (1:Int(N_sd),), radii, key, Int(N_sd), x̄,
            float(ρ_water), stratified)
    for i in 1:N_sd
        add_superdroplet!(sd; ξ = ξ, radius = radii[i], solute_mass = solute_mass, cell = 1)
    end
    return sd
end

@inline function _exponential_radius!(i, radii, key, N_sd, x̄, ρw, stratified)
    r = counter_rng(key, 0, i)
    u = stratified ? (i - rand(r)) / N_sd : rand(r)
    u = min(max(u, 1e-16), 1 - 1e-16)
    x = -x̄ * log(u)                                  # inverse CDF
    @inbounds radii[i] = cbrt(3 * x / (4π * ρw))
    return nothing
end

"""
    init_monodisperse_superdroplets(N_sd, n0, radius, volume; kwargs...)

`N_sd` identical superdroplets of the given wet `radius`, with constant
multiplicity summing to `n₀ · volume`. Useful for condensation tests where
the initial spectrum should be clean.
"""
function init_monodisperse_superdroplets(N_sd::Integer, n0::Real, radius::Real,
                                         volume::Real;
                                         aerosol::AerosolProperties = NACL,
                                         solute_mass::Real = 0.0,
                                         T::Type{<:AbstractFloat} = Float64)
    sd = SuperdropletSet(T; aerosol = aerosol)
    ξ = round(Int, float(n0) * float(volume) / N_sd)
    for _ in 1:N_sd
        add_superdroplet!(sd; ξ = max(ξ, 1), radius = radius,
                          solute_mass = solute_mass, cell = 1)
    end
    return sd
end

"""
    assign_cells!(sd, origin, spacing, dims) -> sd

Map every superdroplet position onto a 1-based linear index of a regular
Cartesian cell grid with lower corner `origin = (x0,y0,z0)`, cell size
`spacing = (dx,dy,dz)` and `dims = (nx,ny,nz)` cells, using column-major
ordering (`i + nx(j-1) + nx·ny(k-1)`).

This is the Lagrangian→Eulerian half of the coupling: it is what decides
which droplets are allowed to collide with which, and which host cell
receives their latent heat. Positions outside the box are clamped into the
nearest cell rather than dropped, so no water is silently lost.
"""
function assign_cells!(sd::SuperdropletSet, origin::NTuple{3,<:Real},
                       spacing::NTuple{3,<:Real}, dims::NTuple{3,<:Integer})
    nx, ny, nz = dims
    @inbounds for i in eachindex(sd.cell)
        ii = clamp(floor(Int, (sd.x[i] - origin[1]) / spacing[1]) + 1, 1, nx)
        jj = clamp(floor(Int, (sd.y[i] - origin[2]) / spacing[2]) + 1, 1, ny)
        kk = clamp(floor(Int, (sd.z[i] - origin[3]) / spacing[3]) + 1, 1, nz)
        sd.cell[i] = ii + nx * (jj - 1) + nx * ny * (kk - 1)
    end
    return sd
end

"""
    cell_populations(sd, ncells) -> Vector{Vector{Int}}

Bucket superdroplet indices by host cell. `ncells` buckets are always
returned, empty ones included, so the result can be indexed directly by
cell number.
"""
function cell_populations(sd::SuperdropletSet, ncells::Integer)
    buckets = [Int[] for _ in 1:ncells]
    @inbounds for i in eachindex(sd.cell)
        ci = sd.cell[i]
        1 <= ci <= ncells && push!(buckets[ci], i)
    end
    return buckets
end

"""
    advance_positions!(sd, dt; velocity = nothing, sedimentation = true,
                       domain = nothing, m = DEFAULT_MICRO)

Advect the superdroplets for `dt` with forward Euler.

`velocity(x,y,z) -> (u,v,w)` supplies the air velocity (the stored
`u,v,w` are used if it is `nothing`); with `sedimentation = true` each
droplet additionally falls at its own [`terminal_velocity`](@ref), which
is the *only* place droplet size feeds back on transport and hence the
mechanism by which rain separates from cloud.

`domain = ((x0,x1),(y0,y1),(z0,z1))` wraps periodically in x and y and
clamps in z.
"""
function advance_positions!(sd::SuperdropletSet{T}, dt::Real;
                            velocity = nothing, sedimentation::Bool = true,
                            domain = nothing,
                            m::MicrophysicalConstants = DEFAULT_MICRO) where {T}
    Δt = T(dt)
    @inbounds for i in eachindex(sd.x)
        if velocity !== nothing
            uu, vv, ww = velocity(sd.x[i], sd.y[i], sd.z[i])
            sd.u[i] = T(uu); sd.v[i] = T(vv); sd.w[i] = T(ww)
        end
        wfall = sedimentation ? T(terminal_velocity(sd.radius[i], m)) : zero(T)
        sd.x[i] += Δt * sd.u[i]
        sd.y[i] += Δt * sd.v[i]
        sd.z[i] += Δt * (sd.w[i] - wfall)
    end
    if domain !== nothing
        (x0, x1), (y0, y1), (z0, z1) = domain
        Lx = x1 - x0; Ly = y1 - y0
        @inbounds for i in eachindex(sd.x)
            sd.x[i] = sd.x[i] - Lx * floor((sd.x[i] - x0) / Lx)
            sd.y[i] = sd.y[i] - Ly * floor((sd.y[i] - y0) / Ly)
            sd.z[i] = clamp(sd.z[i], T(z0), T(z1))
        end
    end
    return sd
end

# =========================================================================
# Terminal velocity
# =========================================================================

# Beard (1976) regime-2 polynomial, in ln(Davies number).
const _BEARD2 = (-3.18657, 0.992696, -1.53193e-3, -9.87059e-4,
                 -5.78878e-4, 8.55176e-5, -3.27815e-6)

@inline function _poly(x::Real, co::NTuple{N,Float64}) where {N}
    y = co[N]
    @inbounds for i in (N-1):-1:1
        y = y * x + co[i]
    end
    return y
end

"""
    terminal_velocity(r, m = DEFAULT_MICRO) -> m s⁻¹

Still-air terminal fall speed of a water drop of radius `r`.

Three regimes, as the physics demands — the drag law genuinely changes
character twice between a 1 μm droplet and a 5 mm raindrop:

 1. **`d < 19.5 μm`** — Stokes flow with the Cunningham slip correction
    `1 + 2.51 λ/d`. `v ∝ r²`; the slip term matters below ~2 μm where the
    drop is no longer large compared with the mean free path of air.

 2. **`19.5 μm ≤ d ≤ 1.07 mm`** — Beard's (1976) regime-2 fit: a 6th-order
    polynomial in `ln N_Da` where `N_Da = 4 ρ_a Δρ g d³ / (3 μ²)` is the
    Davies (Best) number, giving the Reynolds number and hence `v`. This
    covers the whole cloud-to-drizzle range and reproduces the measured
    4.0 m s⁻¹ for a 1 mm drop.

 3. **`d > 1.07 mm`** — the Atlas et al. (1973) fit
    `v = 9.65 - 10.3 exp(-600 d)` (m, m s⁻¹), which captures the flattening
    toward the ~9.2 m s⁻¹ ceiling that drop deformation imposes. Beard's
    own regime 3 needs a physical-property number and a second polynomial;
    the Atlas fit is used here instead because it is short enough to state
    exactly. The two fits differ by 6 % at the junction, so they are
    blended linearly over `1.07–1.4 mm` to keep `v(r)` continuous —
    a discontinuity here would show up directly as a kink in the
    gravitational collision kernel.

The drop is assumed spherical, so the fit is not extended past 7 mm, where
real drops break up anyway.
"""
function terminal_velocity(r::Real, m::MicrophysicalConstants = DEFAULT_MICRO)
    rr = float(r)
    rr <= 0 && return zero(rr)
    return _vterm_d(Float64(2 * rr), m)
end

# Diameters (m) where the drop fall-speed law changes regime: Stokes → Beard
# regime 2, start and end of the blend into the Atlas et al. fit. Bulk
# integrals over V(D) (P3 sedimentation) split their quadrature here.
const _VTERM_REGIME_D = (19.5e-6, 1.07e-3, 1.4e-3)

"""
    stokes_settling_velocity(r, ρ_p, m = DEFAULT_MICRO) -> m s⁻¹

Stokes settling speed of a rigid sphere of radius `r` (m) and density
`ρ_p` (kg m⁻³) with the Cunningham slip correction `1 + 2.51 λ/d`:

```math
v = \\left(1 + 2.51\\frac{λ}{d}\\right)\\frac{(ρ_p - ρ_a)\\, g\\, d^2}{18 μ}
```

This is regime 1 of [`terminal_velocity`](@ref), which calls it with
`ρ_p = ρ_water`; it is exposed separately so that dry aerosol (dust,
biological ice-nucleating particles in `IceNucleation`) settles with the
*same* drag law and air properties as cloud droplets. Valid for particle
Reynolds number ≪ 1, i.e. `d ≲ 20 μm` for water and `≲ 15 μm` for
mineral dust (Seinfeld & Pandis 2016, ch. 9).
"""
function stokes_settling_velocity(r::Real, ρ_p::Real,
                                  m::MicrophysicalConstants = DEFAULT_MICRO)
    rr = float(r)
    rr <= 0 && return zero(rr)
    d = 2 * rr
    Csc = 1 + 2.51 * m.λ_air / d
    return Csc * (ρ_p - m.ρ_air) * m.grav * d^2 / (18 * m.μ_air)
end

function _vterm_d(d::Float64, m::MicrophysicalConstants)
    Δρ = m.ρ_water - m.ρ_air
    d_st, d_b2, d_at = _VTERM_REGIME_D
    if d < d_st
        return stokes_settling_velocity(d / 2, m.ρ_water, m)
    elseif d <= d_b2
        return _vterm_beard2(d, m, Δρ)
    elseif d < d_at
        # Continuous blend between Beard regime 2 and the Atlas fit.
        v2 = _vterm_beard2(d, m, Δρ)
        va = _vterm_atlas(d)
        θ  = (d - d_b2) / (d_at - d_b2)
        return (1 - θ) * v2 + θ * va
    else
        return _vterm_atlas(d)
    end
end

function _vterm_beard2(d::Float64, m::MicrophysicalConstants, Δρ::Float64)
    N_Da = 4 * m.ρ_air * Δρ * m.grav * d^3 / (3 * m.μ_air^2)
    X = log(N_Da)
    Y = _poly(X, _BEARD2)
    Csc = 1 + 2.51 * m.λ_air / d
    N_Re = Csc * exp(Y)
    return m.μ_air * N_Re / (m.ρ_air * d)
end

# Atlas, Srivastava & Sekhon (1973): v = 9.65 - 10.3 exp(-600 d), d in m.
_vterm_atlas(d::Float64) = 9.65 - 10.3 * exp(-600.0 * d)

# =========================================================================
# Collision efficiency
# =========================================================================

abstract type CollisionEfficiency end

"""
    UnitEfficiency()

`E ≡ 1`: every drop swept out geometrically is collected. The idealisation
behind the bare "gravitational" kernel. It overestimates collection badly
for small droplets, where the collected drop is carried around the
collector by the flow.
"""
struct UnitEfficiency <: CollisionEfficiency end

"""
    LongEfficiency()

Long's (1974) analytic fit to the Hall/Davis-Sartor collision efficiencies:

    E(R, r) = min(1, 4.5e8 R² (1 - 3e-8/r))     for R < 50 μm
    E(R, r) = 1                                 for R ≥ 50 μm

with `R` the larger and `r` the smaller radius, both in metres (the
coefficients are Long's cgs `4.5e4 cm⁻²` and `3e-6 cm` converted).

Two things this captures and a constant efficiency does not. First the
steep `R²` growth: collection is hopeless for 10 μm droplets (E ~ 4e-3)
and easy by 50 μm (E → 1), which is precisely the "condensation-coalescence
bottleneck" that makes warm-rain initiation hard. Second the `1 - 3e-8/r`
factor, which cuts off collection of very small droplets — they follow the
streamlines around the collector.

Substituting this into `π(R+r)² E |Δv|` with Stokes fall speeds reproduces
Long's polynomial collection kernel, which is why [`LongKernel`](@ref)
and `HydrodynamicKernel(LongEfficiency())` agree in the small-drop limit.

The tabulated Hall (1980) efficiencies are more accurate, particularly in
the 20-40 μm "efficiency gap"; they are a 15×21 measured table and are
deliberately not reproduced from memory here.
"""
struct LongEfficiency <: CollisionEfficiency end

collision_efficiency(::UnitEfficiency, r1::Real, r2::Real) = 1.0

function collision_efficiency(::LongEfficiency, r1::Real, r2::Real)
    R = max(float(r1), float(r2))
    r = min(float(r1), float(r2))
    r <= 0 && return 0.0
    R >= 50e-6 && return 1.0
    E = 4.5e8 * R^2 * (1 - 3e-8 / r)
    return clamp(E, 0.0, 1.0)
end

# =========================================================================
# Collision kernels
# =========================================================================

"""
    CollisionKernel

Abstract supertype. A kernel is called as `collision_kernel(K, r1, r2)`
and returns the coalescence rate coefficient in **m³ s⁻¹**: the volume of
air swept per unit time per unit number density, so that the expected
number of collisions between two droplet classes of concentrations `n₁`
and `n₂` in volume `ΔV` per unit time is `K n₁ n₂ ΔV`.
"""
abstract type CollisionKernel end

"""
    GolovinKernel(b)

The **additive** (Golovin 1963) kernel `K = b (V₁ + V₂)` with `V` the
droplet volume and `b` in s⁻¹.

It is not a physical kernel. It is the validation kernel, because the
Smoluchowski coagulation equation with an additive kernel and an
exponential initial spectrum has a closed-form solution in a modified
Bessel function `I₁` — see [`golovin_mass_density`](@ref). Any Monte
Carlo coalescence solver that cannot reproduce it is wrong, and the
failure modes (dropped multiplicity, missing `N(N-1)/2` scaling, wrong
γ rule) all show up as visible distortions of the mass spectrum rather
than as subtle biases.

Literature runs quote `b` in the mass form `K = b_m (m₁ + m₂)` with
`b_m = 1500 cm³ g⁻¹ s⁻¹ = 1.5 m³ kg⁻¹ s⁻¹`; the volume-form coefficient is
`b = b_m ρ_w = 1500 s⁻¹`. Use `GolovinKernel(b_mass = 1.5)` to pass it in
the mass form directly.
"""
struct GolovinKernel{T<:AbstractFloat} <: CollisionKernel
    b::T     # s⁻¹, volume form
end

GolovinKernel(; b_mass::Real = 1.5, ρ_water::Real = DEFAULT_MICRO.ρ_water) =
    GolovinKernel(float(b_mass) * float(ρ_water))

collision_kernel(K::GolovinKernel, r1::Real, r2::Real) =
    K.b * (4π / 3) * (float(r1)^3 + float(r2)^3)

"""
    HydrodynamicKernel(efficiency = LongEfficiency(), m = DEFAULT_MICRO)

The **hydrodynamic** (gravitational-sweepout) kernel

    K(r₁, r₂) = π (r₁ + r₂)² E(r₁, r₂) |v(r₁) - v(r₂)|

— the geometric cross-section of the pair, times the differential fall
speed, times the probability `E` that a droplet on a collision course is
actually collected rather than swept around the collector.

Note the structural consequence of the `|Δv|` factor: **equal-sized
droplets never collide** under this kernel. Real clouds get around that
through turbulence and through the size spread that condensation leaves
behind, and the fact that a purely gravitational kernel gives exactly zero
for a monodisperse population is the cleanest statement of why warm rain
needs something to break the symmetry.
"""
struct HydrodynamicKernel{E<:CollisionEfficiency,T<:AbstractFloat} <: CollisionKernel
    efficiency::E
    consts::MicrophysicalConstants{T}
end

HydrodynamicKernel() = HydrodynamicKernel(LongEfficiency(), DEFAULT_MICRO)
HydrodynamicKernel(eff::CollisionEfficiency) = HydrodynamicKernel(eff, DEFAULT_MICRO)

function collision_kernel(K::HydrodynamicKernel, r1::Real, r2::Real)
    a = float(r1); b = float(r2)
    (a <= 0 || b <= 0) && return 0.0
    E = collision_efficiency(K.efficiency, a, b)
    E <= 0 && return 0.0
    Δv = abs(terminal_velocity(a, K.consts) - terminal_velocity(b, K.consts))
    return π * (a + b)^2 * E * Δv
end

"""
    GravitationalKernel(m = DEFAULT_MICRO)

The bare gravitational kernel, `π (r₁+r₂)² |Δv|` with unit collision
efficiency — [`HydrodynamicKernel`](@ref) with [`UnitEfficiency`](@ref).
Kept as its own type because it is the standard upper bound quoted for
collection rates, and comparing it against the Long-efficiency kernel is
the clearest way to see how much of the collection bottleneck is
hydrodynamic rather than kinematic.
"""
struct GravitationalKernel{T<:AbstractFloat} <: CollisionKernel
    consts::MicrophysicalConstants{T}
end

GravitationalKernel() = GravitationalKernel(DEFAULT_MICRO)

collision_kernel(K::GravitationalKernel, r1::Real, r2::Real) =
    collision_kernel(HydrodynamicKernel(UnitEfficiency(), K.consts), r1, r2)

"""
    LongKernel()

Long's (1974) polynomial collection kernel, in the form used by the
box-model literature (Bott 1998; Shima et al. 2009):

    K = 9.44e9 (V₁² + V₂²)      if max(r₁,r₂) ≤ 50 μm     [m⁻³ s⁻¹ · m⁶]
    K = 5.78e3 (V₁ + V₂)        otherwise                  [s⁻¹ · m³]

with `V` in m³. The first branch is the small-drop asymptotic of
`π(R+r)² E |Δv|` with Long's efficiency and Stokes fall speeds — the `R²`
of the efficiency, the `R²` of the cross-section and the `R²` of the
Stokes velocity difference multiply up to `V²`. The second branch is the
large-collector limit, where `E → 1` and the kernel becomes additive
again.

Cheap, smooth, and much closer to reality than Golovin; used here as the
"physically plausible" kernel in tests that do not need an analytic answer.
"""
struct LongKernel <: CollisionKernel end

function collision_kernel(::LongKernel, r1::Real, r2::Real)
    a = float(r1); b = float(r2)
    V1 = (4π / 3) * a^3
    V2 = (4π / 3) * b^3
    return max(a, b) <= 50e-6 ? 9.44e9 * (V1^2 + V2^2) : 5.78e3 * (V1 + V2)
end

# =========================================================================
# Stochastic collision-coalescence — Shima et al. (2009)
# =========================================================================

"""
    coalescence_probability(K, r_j, r_k, ξ_max, dt, ΔV, scale) -> p

The scaled pair coalescence probability of the superdroplet method,

    p = scale · max(ξ_j, ξ_k) · K(r_j, r_k) · Δt / ΔV.

Three factors, each with a job:

  * `max(ξ_j, ξ_k)` — the pair `(j,k)` stands for `ξ_j ξ_k` real droplet
    pairs, but a single "all-or-nothing" event is allowed to consume at
    most `min(ξ_j, ξ_k)` of them, so the per-event probability carries
    the *larger* multiplicity and the *number* of events per attempt
    carries the smaller. This is the substitution that makes the method
    O(N) instead of O(N²) without biasing the mean.
  * `K Δt / ΔV` — the ordinary Smoluchowski probability that two given
    droplets in a well-mixed volume `ΔV` collide in `Δt`.
  * `scale = [N(N-1)/2] / ⌊N/2⌋` — the compensation for sampling only
    `⌊N/2⌋` non-overlapping pairs out of the `N(N-1)/2` that exist.

`p` is routinely larger than 1 in the rain regime, which is exactly why
the caller must use the γ (multiple-coalescence) rule and not a coin flip.
"""
@inline coalescence_probability(K::CollisionKernel, r_j::Real, r_k::Real,
                                ξ_max::Integer, dt::Real, ΔV::Real,
                                scale::Real) =
    scale * ξ_max * collision_kernel(K, r_j, r_k) * float(dt) / float(ΔV)

"""
    coalescence_step!(rng, sd, kernel, dt, volume;
                      indices = eachindex(sd.ξ), m = DEFAULT_MICRO)
        -> (ncollisions, nevents)

One Shima all-or-nothing collision-coalescence sweep over the
superdroplets listed in `indices`, assumed to share a well-mixed air
`volume`.

# Algorithm

 1. Randomly permute the `N` candidates and pair them up as
    `(1,2), (3,4), …` — `L = ⌊N/2⌋` **non-overlapping** pairs. Every
    superdroplet takes part in at most one candidate collision per step,
    which is what keeps the bookkeeping consistent when multiplicities
    change mid-sweep.
 2. Scale each probability by `N(N-1)/2 / L` (see
    [`coalescence_probability`](@ref)).
 3. Draw the number of coalescences for the pair:
    `γ = ⌊p⌋ + (ϕ < p - ⌊p⌋ ? 1 : 0)` with `ϕ = rand(rng)` uniform on
    `[0, 1)` from the cell's own stream, then cap it at
    `⌊ξ_j/ξ_k⌋` with `ξ_j ≥ ξ_k` — one event can consume at most as many
    droplets as the smaller-multiplicity particle has partners for.
 4. Apply the multiplicity update:

    * `ξ_j - γ ξ_k > 0` (the generic case): `ξ_k` of the `j`-droplets are
      consumed `γ` at a time by the `k`-droplets. So `ξ_j ← ξ_j - γ ξ_k`,
      `ξ_k` unchanged, and the `k` *attributes* grow:
      `r_k ← (r_k³ + γ r_j³)^{1/3}`, `m_{s,k} ← m_{s,k} + γ m_{s,j}`.
    * `ξ_j - γ ξ_k = 0` (exhaustion): all `j`-droplets are consumed. The
      pair is then split into two identical grown droplets sharing `ξ_k`:
      `ξ_j ← ⌊ξ_k/2⌋`, `ξ_k ← ξ_k - ⌊ξ_k/2⌋`, both taking the grown
      attributes. Splitting rather than deleting is what keeps `N_sd` from
      collapsing over a long run.

# Conservation

Water mass is conserved *identically* in both branches, not to within a
tolerance:
`(ξ_j - γξ_k) r_j³ + ξ_k (r_k³ + γ r_j³) = ξ_j r_j³ + ξ_k r_k³`, and in the
exhaustion branch `(⌊ξ_k/2⌋ + ⌈ξ_k/2⌉)(r_k³ + γ r_j³) = ξ_j r_j³ + ξ_k r_k³`
because `ξ_j = γ ξ_k` there. Only the cube-root/cube round-trip on the
radius costs anything, and that is one ulp.

The real droplet number drops by exactly `γ ξ_k` per event in both
branches.

Returns `(ncollisions, nevents)`: the number of *real* coalescences
`Σ γ ξ_k` and the number of pairs that coalesced at all.
"""
function coalescence_step!(rng::AbstractRNG, sd::SuperdropletSet{T},
                           kernel::CollisionKernel, dt::Real, volume::Real;
                           indices::AbstractVector{Int} = collect(eachindex(sd.ξ)),
                           m::MicrophysicalConstants = DEFAULT_MICRO) where {T}
    N = length(indices)
    N < 2 && return (0, 0)
    L = N ÷ 2
    scale = (N * (N - 1) / 2) / L

    perm = collect(indices)
    shuffle!(rng, perm)

    ncoll = 0
    nevent = 0
    @inbounds for l in 1:L
        i1 = perm[2l - 1]
        i2 = perm[2l]
        ξ1 = sd.ξ[i1]; ξ2 = sd.ξ[i2]
        (ξ1 <= 0 || ξ2 <= 0) && continue

        # Order so that j has the larger multiplicity.
        j, k = ξ1 >= ξ2 ? (i1, i2) : (i2, i1)
        ξj = sd.ξ[j]; ξk = sd.ξ[k]

        p = coalescence_probability(kernel, sd.radius[j], sd.radius[k],
                                    ξj, dt, volume, scale)
        p <= 0 && continue

        γ = floor(p)
        if rand(rng) < (p - γ)
            γ += 1
        end
        γi = γ >= typemax(Int) ? typemax(Int) : Int(γ)
        γi < 1 && continue
        γmax = ξj ÷ ξk
        γi = min(γi, γmax)
        γi < 1 && continue

        # --- attributes of the surviving (k) droplet -----------------
        rj = sd.radius[j]
        rk = sd.radius[k]
        r_new = T(cbrt(rk^3 + γi * rj^3))
        ms_new = sd.solute_mass[k] + γi * sd.solute_mass[j]

        if ξj - γi * ξk > 0
            sd.ξ[j] = ξj - γi * ξk
            sd.radius[k] = r_new
            sd.solute_mass[k] = ms_new
        else
            # Exhaustion: ξj == γi * ξk exactly (γi was capped at ⌊ξj/ξk⌋,
            # and this branch is only reached when that cap was tight).
            half = ξk ÷ 2
            sd.ξ[j] = half
            sd.ξ[k] = ξk - half
            sd.radius[j] = r_new
            sd.radius[k] = r_new
            sd.solute_mass[j] = ms_new
            sd.solute_mass[k] = ms_new
            # Position/velocity of the annihilated particle are inherited
            # so the split pair does not teleport out of its host cell.
            sd.x[j] = sd.x[k]; sd.y[j] = sd.y[k]; sd.z[j] = sd.z[k]
            sd.u[j] = sd.u[k]; sd.v[j] = sd.v[k]; sd.w[j] = sd.w[k]
            sd.cell[j] = sd.cell[k]
        end

        ncoll += γi * ξk
        nevent += 1
    end
    return (ncoll, nevent)
end

"""
    collision_coalescence!(rng, sd, kernel, dt, cell_volumes;
                           sweep = true, backend = default_backend())
        -> (ncollisions, nevents)

Multi-cell driver: bucket the superdroplets by `sd.cell` and run one
[`coalescence_step!`](@ref) per cell with that cell's air volume.

Collisions are strictly intra-cell. That is the standard superdroplet
assumption and it is a real approximation: it says the cell is well mixed
on the timescale of `dt`, so a droplet is equally likely to meet any other
droplet in the cell and none outside it. It converges as the cell shrinks
only if the superdroplet count per cell stays adequate, which is the usual
practical constraint (~30-100 per cell).

With `sweep = true` zero-multiplicity particles are removed afterwards.

**Random streams** (KA-n, #305). One `UInt64` key is drawn from `rng` per
call (so a seeded `rng` gives a reproducible run, one key per step); cell
`ci` runs its [`coalescence_step!`](@ref) on its own generator,
[`Aeolus.Backends.counter_xoshiro`](@ref)`(key, 0, ci)` (a Xoshiro seeded
from the counter-based stream `(key, 0, ci)`: a cell draws a few hundred
numbers, where Philox would cost 1.3× the old loop), as one work item of a
launch over the cells on `backend`.
The cells share no generator, and the collisions are identical at every
thread count.
"""
function collision_coalescence!(rng::AbstractRNG, sd::SuperdropletSet,
                                kernel::CollisionKernel, dt::Real,
                                cell_volumes::AbstractVector{<:Real};
                                sweep::Bool = true,
                                m::MicrophysicalConstants = DEFAULT_MICRO,
                                backend::AbstractBackend = default_backend())
    ncells = length(cell_volumes)
    buckets = cell_populations(sd, ncells)
    # One key per call; cell ci shuffles and draws from its own generator,
    # seeded from the counter stream (key, 0, ci)
    # in its own work item (KA-n, #305). A cell's superdroplets are touched
    # by its item only, so the result does not depend on the thread count.
    key = rand(rng, UInt64)
    nc = zeros(Int, ncells); ne = zeros(Int, ncells)
    per_cell = max(1, length(sd.ξ) ÷ max(ncells, 1))
    launch_weighted!(backend, per_cell, _coalesce_cell!, (1:ncells,), nc, ne, buckets, key,
                     sd, kernel, dt, cell_volumes, m)
    sweep && compact!(sd)
    return (sum(nc), sum(ne))
end

function _coalesce_cell!(ci, nc, ne, buckets, key, sd, kernel, dt, cell_volumes, m)
    idx = buckets[ci]
    length(idx) < 2 && return nothing
    c, e = coalescence_step!(counter_xoshiro(key, 0, ci), sd, kernel, dt, cell_volumes[ci];
                             indices = idx, m = m)
    @inbounds nc[ci] = c
    @inbounds ne[ci] = e
    return nothing
end

# =========================================================================
# Condensation / evaporation
# =========================================================================

"""
    fk_thermal(T, m = DEFAULT_MICRO, c = DEFAULT_CONSTANTS) -> s m⁻²

The thermodynamic (latent-heat) resistance to diffusional growth,

    F_k = (L_v/(R_v T) - 1) · L_v ρ_w / (K_a T).

Growing a droplet releases latent heat that must be conducted away; until
it is, the droplet surface is warmer than the environment and its own
saturation vapour pressure is higher, which throttles the growth. `F_k`
and `F_d` are comparable in magnitude near 273 K — the thermal term is
*not* a correction, it halves the growth rate.
"""
function fk_thermal(T::Real, m::MicrophysicalConstants = DEFAULT_MICRO,
                    c::ThermodynamicConstants = DEFAULT_CONSTANTS)
    Tk = float(T)
    Rv = c.gas_constant / c.molmass_water
    Lv = latent_heat_vapor(Tk, c)
    Ka = thermal_conductivity(Tk, m)
    return (Lv / (Rv * Tk) - 1) * Lv * m.ρ_water / (Ka * Tk)
end

"""
    fd_vapor(T, p, m = DEFAULT_MICRO, c = DEFAULT_CONSTANTS) -> s m⁻²

The vapour-diffusion resistance,

    F_d = ρ_w R_v T / (D_v e_s(T)),

using `e_s` from [`MoistThermodynamics.saturation_vapor_pressure`](@ref) so
that the growth equation and the host model's saturation agree exactly.
"""
function fd_vapor(T::Real, p::Real, m::MicrophysicalConstants = DEFAULT_MICRO,
                  c::ThermodynamicConstants = DEFAULT_CONSTANTS)
    Tk = float(T)
    Rv = c.gas_constant / c.molmass_water
    es = saturation_vapor_pressure(Tk, c)
    Dv = vapor_diffusivity(Tk, p, m)
    return m.ρ_water * Rv * Tk / (Dv * es)
end

"""
    diffusional_growth_rate(r, m_s, S, T, p, aerosol, mc, c) -> dr/dt, m s⁻¹

The droplet diffusional growth equation with curvature and solute terms
(Rogers & Yau 1989, eq. 7.17; Pruppacher & Klett 1997 §13.1.2):

    r dr/dt = (S - a/r + b/r³) / (F_k + F_d)

`S` is the ambient supersaturation (`e/e_s - 1`), `a/r` the Kelvin term and
`b/r³` the Raoult term; `F_k` and `F_d` are the heat- and
vapour-diffusion resistances.

The `a`, `b` terms are not decoration. Without them there is no
activation barrier, every particle grows monotonically from its dry size
and the cloud droplet number becomes whatever the aerosol number is. With
them, a particle below its critical supersaturation settles at a haze
equilibrium and one above it runs away — and which particles do which
emerges from the ambient `S(t)` history rather than from a
parameterisation.

The `1/r` prefactor is also why the equation is **stiff**: a 0.1 μm haze
particle relaxes to equilibrium in microseconds while a 10 μm droplet
takes minutes, so an explicit step sized for the cloud droplets will send
the haze to negative radius. See [`grow_droplet`](@ref).
"""
function diffusional_growth_rate(r::Real, m_s::Real, S::Real, T::Real, p::Real,
                                 aerosol::AerosolProperties,
                                 mc::MicrophysicalConstants = DEFAULT_MICRO,
                                 c::ThermodynamicConstants = DEFAULT_CONSTANTS)
    rr = float(r)
    rr <= 0 && return 0.0
    A = kelvin_a(T, mc, c)
    B = raoult_b(m_s, aerosol)
    F = fk_thermal(T, mc, c) + fd_vapor(T, p, mc, c)
    return (float(S) - A / rr + B / rr^3) / (rr * F)
end

# g(r) = r dr/dt, the right-hand side in the r² variable.
@inline _growth_g(r::Float64, S::Float64, A::Float64, B::Float64, invF::Float64) =
    (S - A / r + B / (r * r * r)) * invF

"""
    grow_droplet(r0, m_s, S, T, p, dt, aerosol, mc, c;
                 substeps = 0, rtol = 1e-8) -> r

Integrate the diffusional growth equation for one droplet over `dt` at
**fixed** ambient `S`, `T`, `p`.

# Why implicit, and in `r²`

Two things make the naive `r ← r + dt·(dr/dt)` explode. First, the
equation is written in `r dr/dt`, so it is natural in `w = r²`:
`dw/dt = 2 g(√w)` removes the `1/r` singularity from the *discretisation*
even though it is still there in `g`. Second, the haze branch is a stable
equilibrium approached on a microsecond timescale: an explicit step of a
second overshoots it by orders of magnitude and alternates sign. Backward
Euler on `w`,

    w^{n+1} - w^n - 2 Δt g(√(w^{n+1})) = 0,

is unconditionally stable on that branch and lands on the equilibrium
instead of ringing about it.

The residual is solved with **safeguarded Newton**: a bracket
`[r_dry, r_big]` is maintained (the residual is negative at `r_dry`,
where the solute term dominates and forces growth, and positive at large
`r`, where the `S`-bounded growth cannot keep up with `w` itself), a
Newton step is taken when it stays inside, and the bracket is bisected
when it does not. That cannot diverge, which matters because this is
called once per droplet per substep.

`substeps = 0` (the default) picks the substep count adaptively from the
droplet's own relaxation time — activated droplets get one step, haze
particles get as many as they need.
"""
function grow_droplet(r0::Real, m_s::Real, S::Real, T::Real, p::Real, dt::Real,
                      aerosol::AerosolProperties,
                      mc::MicrophysicalConstants = DEFAULT_MICRO,
                      c::ThermodynamicConstants = DEFAULT_CONSTANTS;
                      substeps::Integer = 0, rtol::Real = 1e-10,
                      max_substeps::Integer = 128)
    r = float(r0)
    Δt = float(dt)
    (r <= 0 || Δt <= 0) && return r
    A = kelvin_a(T, mc, c)
    B = raoult_b(m_s, aerosol)
    invF = 1.0 / (fk_thermal(T, mc, c) + fd_vapor(T, p, mc, c))
    Sf = float(S)
    r_dry = m_s > 0 ? dry_radius(m_s, aerosol) : 1e-10

    ns = substeps > 0 ? Int(substeps) :
         _adaptive_substeps(r, Sf, A, B, invF, Δt, r_dry, max_substeps)
    h = Δt / ns
    for _ in 1:ns
        r = _implicit_r_step(r, h, Sf, A, B, invF, r_dry, rtol)
    end
    return r
end

# Substep count, from two competing timescales of the w = r² equation:
#
#   * the Jacobian rate |d(dw/dt)/dw| = |dg/dr| / r, which is enormous on
#     the haze branch (the solute term varies as r⁻⁴) — this is the
#     stiffness backward Euler exists to absorb, so it only sets accuracy;
#   * the relative growth rate |dw/dt| / w = 2|g|/w, which is what limits
#     accuracy for an ordinary, non-stiff cloud droplet.
#
# An activated 10 μm droplet at 0.5 % supersaturation gets ns = 1; a 50 nm
# haze particle gets the cap. That asymmetry is the point: the cost goes
# where the stiffness is instead of being paid uniformly.
function _adaptive_substeps(r::Float64, S::Float64, A::Float64, B::Float64,
                            invF::Float64, Δt::Float64, r_dry::Float64,
                            max_substeps::Integer)
    g = _growth_g(r, S, A, B, invF)
    dgdr = (A / (r * r) - 3 * B / (r^4)) * invF
    rate = max(abs(dgdr / r), 2 * abs(g) / (r * r))
    rate <= 0 && return 1
    ns = ceil(Int, Δt * rate)
    return clamp(ns, 1, Int(max_substeps))
end

# Backward Euler on w = r², solved by safeguarded Newton in r.
function _implicit_r_step(r_n::Float64, h::Float64, S::Float64, A::Float64,
                          B::Float64, invF::Float64, r_dry::Float64,
                          rtol::Real)
    w_n = r_n * r_n
    # Residual f(r) = r² - w_n - 2h g(r); f' = 2r - 2h g'(r).
    f(r) = r * r - w_n - 2h * _growth_g(r, S, A, B, invF)

    # f is NOT monotonic: as r → 0 the Kelvin term -A/r drives g → -∞ so
    # f → +∞, and as r → ∞ the r² term drives f → +∞ as well. It is
    # U-shaped, with the physical root on whichever side of the minimum the
    # droplet is currently growing or shrinking toward.
    #
    # The side is known exactly and needs no search: f(r_n) = -2h g(r_n),
    # so the sign of the growth rate at the current radius picks the
    # bracket. Choosing it by testing f at a tiny lower bound instead is
    # wrong — at r = 1e-10 the Kelvin term is huge and f is positive
    # whatever the droplet is actually doing, which silently collapses a
    # GROWING droplet onto the floor. That bit only when there was no
    # solute (the default `solute_mass = 0`), because a Raoult term at r_dry
    # otherwise masks it: a 5 μm pure-water droplet at 13 % supersaturation
    # was being evaporated to 1e-10 m in a single step.
    gn = _growth_g(r_n, S, A, B, invF)
    floor_r = B > 0 ? max(r_dry * (1 + 1e-12), 1e-12) : 1e-12

    local lo::Float64, hi::Float64
    if gn >= 0
        # growing: the root lies above r_n
        lo = r_n
        hi = max(2 * r_n, 1e-6)
        nexpand = 0
        while f(hi) < 0 && nexpand < 200
            hi *= 2; nexpand += 1
        end
    else
        # evaporating: the root lies below r_n, bounded below by the dry
        # radius when there is solute, and by nothing much when there isn't
        hi = r_n
        lo = floor_r
        if f(lo) > 0
            # f > 0 at BOTH ends of [floor, r_n]. That does not mean the
            # droplet evaporates completely: without solute the Kelvin
            # term makes f → +∞ at the floor however slight the
            # subsaturation, so testing the floor alone sent every
            # evaporating pure-water droplet to 1e-12 m in one step (a
            # 10 μm droplet at S = -0.1 % over 0.5 s; found by the
            # Phase 45 stochastic-growth tests). The physical root, if
            # any, lies between the minimum of f and r_n: locate the
            # minimum and bracket from there. With B = 0 it is closed-form,
            # f'(r) = 2r − 2hA/(F r²) = 0 at r³ = hA/F; otherwise a ternary
            # search in ln r (exact for any unimodal f). The closed form
            # matters: evaporating pure-water droplets are the common
            # case in stochastic growth, and the search cost ~150 extra
            # residual evaluations each (20x the cost of a growth step).
            r_min = if B == 0
                clamp(cbrt(h * A * invF), lo, r_n)
            else
                a, b = log(lo), log(r_n)
                for _ in 1:100
                    m1 = a + (b - a) / 3; m2 = b - (b - a) / 3
                    f(exp(m1)) < f(exp(m2)) ? (b = m2) : (a = m1)
                    b - a < 1e-12 && break
                end
                exp(0.5 * (a + b))
            end
            f(r_min) > 0 && return lo   # no root: it really evaporates away
            lo = r_min
        end
    end
    r = clamp(r_n, lo, hi)
    for _ in 1:100
        fr = f(r)
        if abs(fr) <= rtol * max(w_n, r * r)
            return r
        end
        fr < 0 ? (lo = r) : (hi = r)
        # Newton step
        dfdr = 2r - 2h * (A / (r * r) - 3 * B / (r^4)) * invF
        rn = dfdr != 0 ? r - fr / dfdr : r
        if !(lo < rn < hi) || !isfinite(rn)
            rn = 0.5 * (lo + hi)
        end
        (abs(rn - r) <= rtol * r) && return rn
        r = rn
    end
    return r
end

"""
    condensation_growth!(sd, S, T, p, dt; substeps = 0, mc, c) -> Δm

Grow every superdroplet at a prescribed ambient state and return the total
**real** water mass condensed (kg; negative for net evaporation). Does not
touch the host cell — see [`condensation_step!`](@ref) for the coupled
version.

`S` is either one supersaturation for every droplet or a vector indexed
like `sd.radius`, giving each superdroplet the supersaturation of the
sub-grid environment it sits in (Phase 45 uses this to let droplets
sample a sub-grid supersaturation distribution). A vector filled with a
single value gives bit-for-bit the scalar result.
"""
function condensation_growth!(sd::SuperdropletSet{T},
                              S::Union{Real,AbstractVector{<:Real}}, Temp::Real,
                              p::Real, dt::Real;
                              indices::Union{Nothing,AbstractVector{Int}} = nothing,
                              substeps::Integer = 0,
                              mc::MicrophysicalConstants = DEFAULT_MICRO,
                              c::ThermodynamicConstants = DEFAULT_CONSTANTS) where {T}
    idx = indices === nothing ? eachindex(sd.radius) : indices
    Δv = 0.0
    @inbounds for i in idx
        r_old = sd.radius[i]
        r_new = grow_droplet(r_old, sd.solute_mass[i], _droplet_S(S, i), Temp, p, dt,
                             sd.aerosol, mc, c; substeps = substeps)
        sd.radius[i] = T(r_new)
        Δv += sd.ξ[i] * (r_new^3 - float(r_old)^3)
    end
    return mc.ρ_water * (4π / 3) * Δv
end

_droplet_S(S::Real, i) = S
_droplet_S(S::AbstractVector, i) = S[i]

# =========================================================================
# Eulerian coupling
# =========================================================================

"""
    EulerianCell{T}

The host grid cell a superdroplet population exchanges water and heat
with. A **fixed-mass, fixed-volume** parcel: `density` and `volume` are
constant, so the air mass `ρ ΔV` is an exact invariant and specific
humidities are unambiguous.

That choice is what makes the coupling testable. With a fixed air mass,
`q_vap + q_liq` is conserved to round-off by construction — condensation
moves mass between two numbers with the same denominator. If the cell
were held at fixed *pressure* instead, `ρ` would change with `T` and the
"conservation" law would acquire a correction term that is easy to get
wrong and impossible to distinguish from a real bug.

Pressure is therefore diagnostic: [`cell_pressure`](@ref) returns
`ρ R_m T`.

# Fields
  * `temperature` (K), `q_vap` (kg kg⁻¹ of moist air),
  * `density` (kg m⁻³, moist air, fixed), `volume` (m³, fixed).
"""
mutable struct EulerianCell{T<:AbstractFloat}
    temperature::T
    q_vap::T
    density::T
    volume::T
end

EulerianCell(; temperature::Real, q_vap::Real, density::Real, volume::Real) =
    EulerianCell{Float64}(float(temperature), float(q_vap),
                          float(density), float(volume))

"""
    cell_from_state(T, p, q_vap, volume, q_liq = 0, c = DEFAULT_CONSTANTS)

Build an [`EulerianCell`](@ref) from the Eulerian prognostics a host LES
actually carries — temperature, pressure and specific humidity — by
computing the moist density from [`MoistThermodynamics.air_density`](@ref).
This is the Eulerian→Lagrangian half of the mapping.
"""
function cell_from_state(Temp::Real, p::Real, q_vap::Real, volume::Real,
                         q_liq::Real = 0.0,
                         c::ThermodynamicConstants = DEFAULT_CONSTANTS)
    q = PhasePartition(float(q_vap) + float(q_liq), float(q_liq), 0.0)
    ρ = air_density(Temp, p, q, c)
    return EulerianCell{Float64}(float(Temp), float(q_vap), ρ, float(volume))
end

"Mass of moist air in the cell (kg) — the invariant denominator."
cell_air_mass(cell::EulerianCell) = cell.density * cell.volume

"""
    cell_pressure(cell, q_liq = 0, c = DEFAULT_CONSTANTS) -> Pa

Diagnostic pressure `p = ρ R_m T`, with the moist gas constant taken from
[`MoistThermodynamics.gas_constant_moist`](@ref).
"""
function cell_pressure(cell::EulerianCell, q_liq::Real = 0.0,
                       c::ThermodynamicConstants = DEFAULT_CONSTANTS)
    q = PhasePartition(cell.q_vap + float(q_liq), float(q_liq), 0.0)
    return cell.density * gas_constant_moist(q, c) * cell.temperature
end

"""
    cell_supersaturation(cell, c = DEFAULT_CONSTANTS) -> S

Ambient supersaturation `q_v/q_sat - 1` using the **density** form of the
saturation specific humidity, `q_sat = e_s/(ρ R_v T)`.

The density form rather than the pressure form on purpose: at fixed cell
volume and mass, `ρ` is exactly known and `p` is not an independent
variable, so this is the form with no round-trip error in it.
"""
function cell_supersaturation(cell::EulerianCell,
                              c::ThermodynamicConstants = DEFAULT_CONSTANTS)
    qs = saturation_specific_humidity(cell.temperature, cell.density, c;
                                      use_density = true)
    return cell.q_vap / qs - 1
end

"""
    liquid_water_mixing_ratio(sd, cell; index = nothing, mc) -> kg kg⁻¹

Specific liquid water content of the cell implied by the superdroplets
assigned to it (`Σ ξ m_drop / m_air`).
"""
function liquid_water_mixing_ratio(sd::SuperdropletSet, cell::EulerianCell;
                                   index::Union{Nothing,Integer} = nothing,
                                   mc::MicrophysicalConstants = DEFAULT_MICRO)
    return total_water_mass(sd, mc; cell = index) / cell_air_mass(cell)
end

"Total water (vapour + resolved liquid) of a cell, kg kg⁻¹. Conserved."
function cell_total_water(sd::SuperdropletSet, cell::EulerianCell;
                          index::Union{Nothing,Integer} = nothing,
                          mc::MicrophysicalConstants = DEFAULT_MICRO)
    return cell.q_vap + liquid_water_mixing_ratio(sd, cell; index = index, mc = mc)
end

"""
    apply_condensate_tendency!(cell, Δm, q_liq = 0, c = DEFAULT_CONSTANTS)
        -> (Δq_vap, ΔT)

Push a condensed water mass `Δm` (kg, positive = condensation) back onto
the host cell:

    Δq_vap = -Δm / m_air,
    ΔT     = +L_v(T) Δm / (m_air c_pm).

Both halves of the coupling in one place, deliberately: vapour removal and
latent heating must use the *same* `Δm` and the *same* `L_v(T)`, or the
cell's moist static energy drifts. `L_v` and `c_pm` both come from
`MoistThermodynamics`, so they are consistent with the `e_s` used to
compute the supersaturation that drove the growth in the first place.
"""
function apply_condensate_tendency!(cell::EulerianCell, Δm::Real,
                                    q_liq::Real = 0.0,
                                    c::ThermodynamicConstants = DEFAULT_CONSTANTS)
    m_air = cell_air_mass(cell)
    Δq = -float(Δm) / m_air
    q = PhasePartition(cell.q_vap + float(q_liq), float(q_liq), 0.0)
    cpm = heat_capacity_moist(q, c)
    Lv = latent_heat_vapor(cell.temperature, c)
    ΔT = -Lv * Δq / cpm
    cell.q_vap += Δq
    cell.temperature += ΔT
    return (Δq, ΔT)
end

"""
    condensation_step!(sd, cell, dt; index = nothing, nsub = 0,
                       substeps = 0, mc, c) -> (Δm, Δq_vap, ΔT)

**Two-way coupled** condensation: grow the droplets in the ambient
supersaturation, remove the condensed vapour from the cell and deposit the
latent heat there, both consistently.

The outer loop is sub-stepped because the coupling itself is stiff: the
*phase relaxation time*

    τ_phase ≈ [4π D_v Σ(ξ r) / ΔV]⁻¹

is a few seconds in a typical cloud and a fraction of a second in a
polluted one. A single Euler step longer than `τ_phase` drives `S` past
zero and back, and the droplets alternately grow and evaporate. With
`nsub = 0` the count is chosen from `τ_phase` itself; the inner
[`grow_droplet`](@ref) then handles the per-droplet stiffness on top.

Returns the condensed mass (kg) and the cell tendencies actually applied.
"""
function condensation_step!(sd::SuperdropletSet, cell::EulerianCell, dt::Real;
                            index::Union{Nothing,Integer} = nothing,
                            nsub::Integer = 0, substeps::Integer = 0,
                            mc::MicrophysicalConstants = DEFAULT_MICRO,
                            c::ThermodynamicConstants = DEFAULT_CONSTANTS)
    idx = index === nothing ? collect(eachindex(sd.radius)) :
                              findall(==(Int(index)), sd.cell)
    isempty(idx) && return (0.0, 0.0, 0.0)
    Δt = float(dt)

    ns = nsub > 0 ? Int(nsub) : _phase_substeps(sd, cell, Δt, idx, mc, c)
    h = Δt / ns

    Δm_tot = 0.0; Δq_tot = 0.0; ΔT_tot = 0.0
    for _ in 1:ns
        p = cell_pressure(cell, 0.0, c)
        S = cell_supersaturation(cell, c)
        Δm = condensation_growth!(sd, S, cell.temperature, p, h;
                                  indices = idx, substeps = substeps,
                                  mc = mc, c = c)
        q_liq = total_water_mass(sd, mc; cell = index) / cell_air_mass(cell)
        Δq, ΔT = apply_condensate_tendency!(cell, Δm, q_liq, c)
        Δm_tot += Δm; Δq_tot += Δq; ΔT_tot += ΔT
    end
    return (Δm_tot, Δq_tot, ΔT_tot)
end

function _phase_substeps(sd::SuperdropletSet, cell::EulerianCell, Δt::Float64,
                         idx::AbstractVector{Int},
                         mc::MicrophysicalConstants,
                         c::ThermodynamicConstants)
    Σξr = 0.0
    @inbounds for i in idx
        Σξr += sd.ξ[i] * sd.radius[i]
    end
    return _phase_substeps(Σξr, cell, Δt, mc, c)
end

# The same rule from the droplet-radius sum Σξr alone, so that a scheme
# carrying moments instead of droplets (Phase 45 bulk broadening) uses the
# identical τ_phase and sub-step criterion.
function _phase_substeps(Σξr::Real, cell::EulerianCell, Δt::Float64,
                         mc::MicrophysicalConstants,
                         c::ThermodynamicConstants)
    Σξr <= 0 && return 1
    p = cell_pressure(cell, 0.0, c)
    Dv = vapor_diffusivity(cell.temperature, p, mc)
    rate = 4π * Dv * Σξr / cell.volume        # 1/τ_phase, s⁻¹
    rate <= 0 && return 1
    return clamp(ceil(Int, Δt * rate / 0.2), 1, 100_000)
end

"""
    deposit_liquid_water!(qc, sd, cells; mc = DEFAULT_MICRO) -> qc

Deposit the superdroplet liquid water onto an Eulerian array `qc`
(specific liquid water content, kg kg⁻¹), one entry per cell, given the
matching vector of [`EulerianCell`](@ref)s. `qc` is zeroed first.

Nearest-cell ("cloud in cell") deposition rather than a smoothing kernel:
it conserves mass exactly and introduces no length scale that is not
already in the grid. A trilinear deposition would be smoother but would
leak mass across the boundary of the domain unless carefully closed.
"""
function deposit_liquid_water!(qc::AbstractVector{<:Real}, sd::SuperdropletSet,
                               cells::AbstractVector{<:EulerianCell};
                               mc::MicrophysicalConstants = DEFAULT_MICRO)
    fill!(qc, 0)
    fac = mc.ρ_water * (4π / 3)
    @inbounds for i in eachindex(sd.ξ)
        ci = sd.cell[i]
        (1 <= ci <= length(qc)) || continue
        qc[ci] += fac * sd.ξ[i] * sd.radius[i]^3
    end
    @inbounds for ci in eachindex(qc)
        qc[ci] /= cell_air_mass(cells[ci])
    end
    return qc
end

# =========================================================================
# Diagnostics
# =========================================================================

"""
    radius_moment(sd, k; volume = 1, cell = nothing) -> Σ ξ r^k / volume

The `k`-th raw moment of the droplet size distribution per unit air
volume. `k = 0` gives the number concentration, `k = 2` the surface-area
moment that optical depth needs, `k = 3` the mass moment.
"""
function radius_moment(sd::SuperdropletSet, k::Integer; volume::Real = 1.0,
                       cell::Union{Nothing,Integer} = nothing)
    s = 0.0
    @inbounds for i in eachindex(sd.ξ)
        (cell === nothing || sd.cell[i] == cell) || continue
        s += sd.ξ[i] * float(sd.radius[i])^k
    end
    return s / float(volume)
end

"Droplet number concentration, m⁻³."
number_concentration(sd::SuperdropletSet, volume::Real;
                     cell::Union{Nothing,Integer} = nothing) =
    radius_moment(sd, 0; volume = volume, cell = cell)

"Liquid water content, kg m⁻³."
liquid_water_content(sd::SuperdropletSet, volume::Real;
                     cell::Union{Nothing,Integer} = nothing,
                     mc::MicrophysicalConstants = DEFAULT_MICRO) =
    mc.ρ_water * (4π / 3) * radius_moment(sd, 3; volume = volume, cell = cell)

"Number-weighted mean radius, m."
function mean_radius(sd::SuperdropletSet; cell::Union{Nothing,Integer} = nothing)
    n = radius_moment(sd, 0; cell = cell)
    n <= 0 && return 0.0
    return radius_moment(sd, 1; cell = cell) / n
end

"""
    effective_radius(sd; cell = nothing) -> m

`r_eff = Σξr³ / Σξr²` — the ratio of the volume to the area moment.

This, not the mean radius, is what radiative transfer sees: the extinction
of a droplet population depends on its total cross-section and its water
content, and `r_eff` is exactly the size of the monodisperse population
with the same two. For a typical cloud spectrum it exceeds the mean radius
by 10-25 %, and using the mean radius instead biases cloud albedo the
wrong way by several per cent.
"""
function effective_radius(sd::SuperdropletSet; cell::Union{Nothing,Integer} = nothing)
    m2 = radius_moment(sd, 2; cell = cell)
    m2 <= 0 && return 0.0
    return radius_moment(sd, 3; cell = cell) / m2
end

"""
    relative_dispersion(sd; cell = nothing) -> (σ_r, ε)

Standard deviation of the number-weighted radius distribution and the
**relative dispersion** `ε = σ_r / r̄`.

`ε` is the single number that summarises spectral width, and it is the
parameter bulk schemes must assume. Observed warm clouds sit at
`ε ≈ 0.2-0.5`; pure condensational growth drives it *down* (see the
narrowing test), and it is collision-coalescence and entrainment mixing
that push it back up. A scheme that gets `ε` wrong gets autoconversion
wrong by a large factor, because autoconversion depends on the tail.
"""
function relative_dispersion(sd::SuperdropletSet;
                             cell::Union{Nothing,Integer} = nothing)
    n  = radius_moment(sd, 0; cell = cell)
    n <= 0 && return (0.0, 0.0)
    r̄  = radius_moment(sd, 1; cell = cell) / n
    r2 = radius_moment(sd, 2; cell = cell) / n
    var = max(r2 - r̄^2, 0.0)
    σ = sqrt(var)
    return (σ, r̄ > 0 ? σ / r̄ : 0.0)
end

"""
    size_distribution(sd, edges, volume; cell = nothing, log = true)
        -> counts

Binned droplet **number** distribution from the superdroplet ensemble.

With `log = true` (the default) `edges` are radii and the result is
`dN/d(ln r)` in m⁻³ per unit `ln r`; with `log = false` it is `dN/dr` in
m⁻³ m⁻¹. Either way it is a *measurement* of the ensemble, not a fitted
shape — which is the whole point of the method.
"""
function size_distribution(sd::SuperdropletSet, edges::AbstractVector{<:Real},
                           volume::Real; cell::Union{Nothing,Integer} = nothing,
                           log::Bool = true)
    nb = length(edges) - 1
    out = zeros(Float64, nb)
    @inbounds for i in eachindex(sd.ξ)
        (cell === nothing || sd.cell[i] == cell) || continue
        r = float(sd.radius[i])
        b = _bin_index(edges, r)
        b == 0 && continue
        out[b] += sd.ξ[i]
    end
    @inbounds for b in 1:nb
        width = log ? (Base.log(edges[b+1]) - Base.log(edges[b])) :
                      (edges[b+1] - edges[b])
        out[b] /= (width * float(volume))
    end
    return out
end

"""
    mass_distribution(sd, edges, volume; cell = nothing, mc) -> g(ln r)

Binned **mass** distribution `g(ln r) = dM/d(ln r)` in kg m⁻³ per unit
`ln r`, the standard presentation of a coalescence calculation
(Berry & Reinhardt 1974; Shima et al. 2009 Fig. 2).

Mass rather than number because coalescence is a *mass* cascade: the
number density falls monotonically and hides everything, while `g(ln r)`
shows the cloud mode draining into a drizzle mode as a second hump —
which is the feature the analytic Golovin solution predicts and the
feature a Monte Carlo solver with broken multiplicity bookkeeping gets
visibly wrong.
"""
function mass_distribution(sd::SuperdropletSet, edges::AbstractVector{<:Real},
                           volume::Real; cell::Union{Nothing,Integer} = nothing,
                           mc::MicrophysicalConstants = DEFAULT_MICRO)
    nb = length(edges) - 1
    out = zeros(Float64, nb)
    fac = mc.ρ_water * (4π / 3)
    @inbounds for i in eachindex(sd.ξ)
        (cell === nothing || sd.cell[i] == cell) || continue
        r = float(sd.radius[i])
        b = _bin_index(edges, r)
        b == 0 && continue
        out[b] += sd.ξ[i] * fac * r^3
    end
    @inbounds for b in 1:nb
        out[b] /= ((Base.log(edges[b+1]) - Base.log(edges[b])) * float(volume))
    end
    return out
end

# Binary search for the bin containing x; 0 if outside.
function _bin_index(edges::AbstractVector{<:Real}, x::Real)
    (x < edges[1] || x >= edges[end]) && return 0
    lo = 1; hi = length(edges)
    while hi - lo > 1
        mid = (lo + hi) ÷ 2
        edges[mid] <= x ? (lo = mid) : (hi = mid)
    end
    return lo
end

"""
    radius_percentile(sd, q; cell = nothing) -> m

The `q`-quantile (0 ≤ q ≤ 1) of the **multiplicity-weighted** radius
distribution — i.e. the radius below which a fraction `q` of the *real*
droplets lie. Weighting by ξ is essential; the unweighted quantile over
computational particles is a different (and meaningless) number whenever
the multiplicities are not all equal.
"""
function radius_percentile(sd::SuperdropletSet, q::Real;
                           cell::Union{Nothing,Integer} = nothing)
    idx = [i for i in eachindex(sd.ξ)
           if (cell === nothing || sd.cell[i] == cell) && sd.ξ[i] > 0]
    isempty(idx) && return 0.0
    sort!(idx; by = i -> sd.radius[i])
    total = sum(sd.ξ[i] for i in idx)
    target = float(q) * total
    acc = 0
    for i in idx
        acc += sd.ξ[i]
        acc >= target && return float(sd.radius[i])
    end
    return float(sd.radius[idx[end]])
end

# =========================================================================
# Analytic reference: modified Bessel I₁ and the Golovin solution
# =========================================================================

"""
    bessel_i1(z) -> I₁(z)

Modified Bessel function of the first kind, order 1, written out here
because `SpecialFunctions` is a binary dependency that does not build in
this environment — and because this is the function the headline
validation rests on, so it had better be independently verifiable.

Two representations:

  * **Series**, `I₁(z) = (z/2) Σ (z²/4)^k / (k!(k+1)!)`, for `|z| ≤ 50`.
    All terms are positive, so there is no cancellation and the relative
    accuracy is at the level of the accumulated round-off, ~1e-15.
  * **Asymptotic**, `I₁(z) ~ e^z/√(2πz) [1 - 3/(8z) - 15/(128z²) - …]`,
    beyond. At `z = 50` the two agree to 5e-11, which is the crossover
    accuracy.

Verified in the test suite against tabulated `I₁(0.5), I₁(1), I₁(2),
I₁(5), I₁(10)`.
"""
bessel_i1(z::Real) = exp(float(z)) * bessel_i1_scaled(z)

"""
    bessel_i1_scaled(z) -> e^{-z} I₁(z)

The exponentially scaled `I₁`. This is the form the Golovin solution
actually needs: at late times its argument reaches several hundred, where
`I₁` itself overflows `Float64` while the product with the accompanying
`exp(-(1+T)x̃)` is a perfectly ordinary number. Combining the exponents
analytically — the Golovin exponent becomes `-x̃(1-√T)²`, which is never
positive — is the difference between a solution that evaluates and one
that returns `NaN` exactly where the interesting drizzle tail lives.
"""
function bessel_i1_scaled(z::Real)
    x = float(z)
    x < 0 && return -bessel_i1_scaled(-x)
    x == 0 && return 0.0
    if x <= 50.0
        h = x * x / 4
        term = x / 2
        s = term
        k = 0
        while k < 500
            k += 1
            term *= h / (k * (k + 1))
            s += term
            term < 1e-18 * s && break
        end
        return s * exp(-x)
    else
        # I₁(z) ~ e^z/√(2πz) Σ (-1)^k a_k(1)/z^k, μ = 4
        inv = 1 / x
        s = 1.0
        s -= 0.375 * inv
        s -= 0.1171875 * inv^2
        s -= 0.1025390625 * inv^3
        s -= 0.1441955566406250 * inv^4
        s -= 0.2775764465332031 * inv^5
        return s / sqrt(2π * x)
    end
end

"""
    golovin_T(b, L, t) -> T

The dimensionless time of the Golovin solution,
`T = 1 - exp(-b L t)`, where `b` is the **mass-form** additive kernel
coefficient (m³ kg⁻¹ s⁻¹) and `L` the liquid water content (kg m⁻³).

`T` runs from 0 to 1 over the whole evolution: it is the fraction of the
initial droplet population that has been consumed by coalescence, since
`N(t) = N₀(1-T)` exactly for this kernel. `T = 0.5` is "half the droplets
gone", `T = 0.99` a fully developed drizzle spectrum.
"""
golovin_T(b_mass::Real, L::Real, t::Real) =
    1 - exp(-float(b_mass) * float(L) * float(t))

"""
    golovin_number_density(x, t, n0, x̄, b) -> n(x,t), m⁻³ kg⁻¹

**The analytic solution of the Smoluchowski coagulation equation** for the
additive kernel `K(x,y) = b(x+y)` with the exponential initial condition
`n(x,0) = (n₀/x̄) e^{-x/x̄}` (Golovin 1963; Scott 1968).

With `x̃ = x/x̄` and `T = 1 - e^{-b n₀ x̄ t}`,

    n(x,t) = (n₀/x̄) · (1-T)/(x̃ √T) · exp[-(1+T) x̃] · I₁(2 x̃ √T)

Evaluated here in the numerically safe grouping

    n = (n₀/x̄) · (1-T)/(x̃ √T) · exp[-x̃(1-√T)²] · [e^{-2x̃√T} I₁(2x̃√T)]

whose exponent is never positive, so it is finite everywhere.

# Why this is the right validation

It is a genuine external check. The solution was derived from the
continuous coagulation equation with no reference to any numerical method,
it is nontrivial (a Bessel function, not an exponential), and every way of
getting the superdroplet bookkeeping wrong distorts it in a visible and
*different* way — dropping the `N(N-1)/2` scaling slows the whole
evolution, mishandling γ for `p > 1` truncates the large-drop tail,
splitting multiplicities wrongly breaks mass conservation. Two exact
properties make it self-checking as well: `∫n dx = n₀(1-T)` and
`∫x n dx = n₀x̄` for every `T`, both of which the test suite verifies by
quadrature before comparing anything to the Monte Carlo.

The `T → 0` limit reduces to the initial exponential, since
`I₁(z) → z/2`.
"""
function golovin_number_density(x::Real, t::Real, n0::Real, x̄::Real,
                                b_mass::Real)
    xx = float(x) / float(x̄)
    L = float(n0) * float(x̄)
    T = golovin_T(b_mass, L, t)
    (xx <= 0) && return 0.0
    if T <= 0
        return float(n0) / float(x̄) * exp(-xx)
    end
    sT = sqrt(T)
    z = 2 * xx * sT
    pref = float(n0) / float(x̄) * (1 - T) / (xx * sT)
    return pref * exp(-xx * (1 - sT)^2) * bessel_i1_scaled(z)
end

"""
    golovin_mass_density(r, t, n0, r̄, b; ρ_water) -> g(ln r), kg m⁻³

The Golovin solution expressed the way coalescence results are actually
plotted: mass per unit volume of air per unit `ln r`.

`dM = x n(x) dx` and `x ∝ r³ ⟹ dx = 3x d(ln r)`, hence

    g(ln r) = 3 x² n(x, t)

with `x = (4π/3) ρ_w r³`. This is the curve
[`mass_distribution`](@ref) is compared against.
"""
function golovin_mass_density(r::Real, t::Real, n0::Real, r̄::Real,
                              b_mass::Real;
                              ρ_water::Real = DEFAULT_MICRO.ρ_water)
    fac = (4π / 3) * float(ρ_water)
    x  = fac * float(r)^3
    x̄  = fac * float(r̄)^3
    n = golovin_number_density(x, t, n0, x̄, b_mass)
    return 3 * x^2 * n
end

end # module
