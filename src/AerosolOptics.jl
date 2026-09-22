module AerosolOptics

# Phase 37 — Aerosol and gas radiative feedback.
#
# Aerosol is the last large free parameter in a radiation stack. Gas
# optics (Phase 32) is essentially solved physics; cloud optics (Phase 33)
# is parameterised but well constrained; aerosol spans four orders of
# magnitude in concentration, two in single-scattering albedo, and its
# optical properties depend on a *wet* size distribution that is itself a
# function of the ambient humidity. That is why the aerosol direct effect
# remains the largest single uncertainty in the anthropogenic forcing
# budget (IPCC AR6 Ch. 7).
#
# This module therefore does not tabulate bulk optical properties. It
# computes them, from first principles, along the chain that the physics
# actually follows:
#
#   refractive index m(λ)  ─┐
#   dry radius r_d          ├─→ κ-Köhler wet radius r_w ──┐
#   relative humidity RH   ─┘                             │
#                                                          ├─→ Mie(x, m_w)
#   size parameter x = 2πr_w/λ  ←─────────────────────────┘
#                    │
#                    └─→ ∫ over log-normal dN/dln r ─→ β_ext, ω₀, g per band
#                                                            │
#              vertical shape × AOD(550 nm) normalisation ───┴─→ OpticalProperties
#
# Three things in that chain are easy to get wrong and are handled
# explicitly here:
#
#   1. **Mie recursion stability.** The logarithmic derivative D_n(mx) of
#      the Riccati-Bessel function ψ_n is *unstable upward* — for
#      absorbing media the upward recursion loses all significance within
#      a few tens of terms. Wiscombe (1980) showed the downward
#      recursion, seeded with D_{Nmx} = 0 well above the truncation
#      order, is stable to machine precision. That is what is used
#      below, and the classic benchmark (x = 10, m = 1.5 ⇒ Q_ext =
#      2.8820, g = 0.742913) is reproduced to seven digits in the test
#      suite. Below x = 10⁻³ the Riccati-Bessel upward recursion for ψ_n
#      itself suffers catastrophic cancellation, so the analytic Rayleigh
#      limit is used there instead.
#
#   2. **The r⁶ tail.** For a log-normal with σ_g ≈ 2 the scattering
#      integrand ∝ r²Q_sca is weighted towards radii several σ above the
#      number-median radius. Integrating over ±3σ silently loses a large
#      fraction of the extinction. The quadrature range below is
#      *asymmetric*, extended on the coarse side by 3 ln²σ_g (the shift
#      of the r³-weighted mode), and truncated at a physical 20 μm as
#      OPAC does.
#
#   3. **Mixing rules.** Optical depths add; single-scattering albedo is
#      a τ-weighted mean; asymmetry parameter is a *scattering*-optical-
#      depth-weighted mean. `RadiationCore.combine_optics!` does this
#      correctly and this module's `compute_optics!` uses exactly the
#      same algebra so that aerosol, gas and cloud merge consistently.
#
# References
#   Bohren & Huffman (1983), *Absorption and Scattering of Light by
#       Small Particles*, §4.4 and the BHMIE listing — the a_n/b_n
#       recursion used here.
#   Wiscombe (1980), *Improved Mie scattering algorithms*, Appl. Opt. 19,
#       1505 — downward D_n recursion, truncation criterion, benchmarks.
#   Hess, Koepke & Schult (1998), *Optical Properties of Aerosols and
#       Clouds (OPAC)*, BAMS 79, 831 — component definitions, size
#       distributions and the standard mixtures.
#   Petters & Kreidenweis (2007), ACP 7, 1961 — the κ hygroscopicity
#       parameterisation.
#   Hale & Querry (1973), Appl. Opt. 12, 555 — liquid water refractive
#       index, 0.2–200 μm.
#   Ångström (1929), Geogr. Ann. 11, 156 — the wavelength exponent.
#   Chylek & Coakley (1974), Science 183, 75 — critical surface albedo
#       for the sign of the aerosol TOA forcing.
#   Haywood & Shine (1995), GRL 22, 603 — the same, with the standard
#       two-stream formulation used for the diagnostic here.

using ..RadiationCore
using ..RadiationCore: SpectralGrid, OpticalProperties, RadiativeFluxes,
                       AtmosphericColumn, AbstractAerosolOptics,
                       nbands, ngpoints, gpoint_range, band_limits,
                       allocate_optics, allocate_fluxes, combine_optics!,
                       shortwave_bands, longwave_bands, SOLAR_CONSTANT
using ..GasOptics: rayleigh_optics!
import ..RadiationCore: compute_optics!

export # --- Mie core -------------------------------------------------------
       MieResult, mie_sphere, mie_efficiencies,
       rayleigh_efficiencies, size_parameter,
       # --- species / refractive index -------------------------------------
       AerosolSpecies, aerosol_species, aerosol_species_names,
       refractive_index, water_refractive_index,
       # --- size distributions ---------------------------------------------
       AerosolMode, LogNormalMode, lognormal_mode,
       number_pdf, mode_number, mode_mass, mode_optics,
       bulk_optics, mass_extinction_coefficient,
       mass_absorption_coefficient, effective_radius, clear_optics_cache!,
       # --- hygroscopic growth ----------------------------------------------
       kohler_saturation, wet_radius, growth_factor, wet_refractive_index,
       KELVIN_A_298,
       # --- band optics -------------------------------------------------------
       AerosolBandOptics, aerosol_band_optics, band_wavelength,
       HygroscopicTable, hygroscopic_table, interpolate_rh,
       angstrom_exponent,
       # --- standard mixtures --------------------------------------------------
       aerosol_mixture, AEROSOL_MIXTURE_NAMES,
       continental_clean, continental_polluted, urban, desert,
       maritime_clean, maritime_polluted, antarctic,
       # --- column profile -----------------------------------------------------
       AerosolProfile, aerosol_profile, rebuild_bands,
       aerosol_optical_depth, aerosol_optical_depth_550,
       aerosol_extinction_profile, relative_humidity_profile,
       # --- radiative diagnostics ----------------------------------------------
       aerosol_radiative_forcing, critical_surface_albedo,
       aerosol_heating_rate

# =========================================================================
# 1. Mie scattering from a homogeneous sphere
# =========================================================================

"""
    MieResult{T}

Efficiency factors and the first phase-function moment for a single
homogeneous sphere:

  * `Q_ext`  — extinction efficiency, `C_ext / πr²`
  * `Q_sca`  — scattering efficiency
  * `Q_abs`  — absorption efficiency, `Q_ext - Q_sca` (≥ 0)
  * `Q_back` — backscattering efficiency (lidar-relevant)
  * `g`      — asymmetry parameter `⟨cos Θ⟩ ∈ [-1, 1]`
  * `nterms` — number of terms retained in the Mie series

The efficiencies are per *geometric* cross-section, so `Q_ext → 2` as
`x → ∞` (the extinction paradox: a large sphere removes twice the light
its shadow accounts for, half by absorption/reflection and half by
near-forward diffraction).
"""
struct MieResult{T<:AbstractFloat}
    Q_ext::T
    Q_sca::T
    Q_abs::T
    Q_back::T
    g::T
    nterms::Int
end

Base.show(io::IO, r::MieResult) =
    print(io, "MieResult(Qext=", round(r.Q_ext; sigdigits = 6),
              ", Qsca=", round(r.Q_sca; sigdigits = 6),
              ", Qabs=", round(r.Q_abs; sigdigits = 6),
              ", g=", round(r.g; sigdigits = 6),
              ", n=", r.nterms, ")")

"Size parameter `x = 2πr/λ` for radius and wavelength in the same units."
@inline size_parameter(r::Real, λ::Real) = 2 * π * r / λ

# Scratch buffer for the logarithmic-derivative recursion. The recursion
# needs O(x + |m|x + 15) complex numbers; reallocating that for every one
# of the ~10⁵ Mie calls a mixture costs is pure garbage-collector churn,
# so one buffer is grown and reused. Single-threaded by construction —
# `mie_sphere` is called from the (serial) optics-table build.
const _D_BUFFER = ComplexF64[]

"""
    rayleigh_efficiencies(x, m) -> MieResult

Analytic small-particle (Rayleigh) limit of the Mie solution, with
`μ = (m² - 1)/(m² + 2)` the Clausius-Mossotti factor:

```math
Q_{sca} = \\tfrac{8}{3}x^4|μ|^2, \\qquad
Q_{abs} = 4x\\,\\mathrm{Im}(μ), \\qquad
g = O(x^2) \\approx 0
```

Used by [`mie_sphere`](@ref) for `x < 10⁻³`, where the upward
Riccati-Bessel recursion `ψ_n = (2n-1)ψ_{n-1}/x - ψ_{n-2}` loses all
significant digits to cancellation (ψ₁ ≈ x²/3 is formed as a difference
of two numbers of order 1). The relative error of the truncation is
`O(x²) ≲ 10⁻⁶` at the switch-over, verified in the test suite by
continuity across `x = 10⁻³`.
"""
function rayleigh_efficiencies(x::Real, m::Complex)
    xx = Float64(x); mm = ComplexF64(m)
    m2 = mm * mm
    μ = (m2 - 1) / (m2 + 2)
    Qsca = 8 / 3 * xx^4 * abs2(μ)
    Qabs = 4 * xx * imag(μ)
    Qabs = max(Qabs, 0.0)
    # Backscatter for a dipole: |S₁(180°)|² relation gives 4x⁴|μ|²
    Qback = 4 * xx^4 * abs2(μ)
    return MieResult(Qsca + Qabs, Qsca, Qabs, Qback, 0.0, 1)
end

"""
    mie_sphere(x, m; nmax = 0) -> MieResult

Mie solution for a homogeneous sphere of size parameter `x = 2πr/λ` and
complex refractive index `m = n + ik`, **k ≥ 0 for an absorbing
material** (the Bohren-Huffman `exp(-iωt)` convention).

The series coefficients are

```math
a_n = \\frac{[D_n(mx)/m + n/x]\\,ψ_n(x) - ψ_{n-1}(x)}
            {[D_n(mx)/m + n/x]\\,ξ_n(x) - ξ_{n-1}(x)},\\qquad
b_n = \\frac{[m D_n(mx) + n/x]\\,ψ_n(x) - ψ_{n-1}(x)}
            {[m D_n(mx) + n/x]\\,ξ_n(x) - ξ_{n-1}(x)}
```

with `D_n(z) = ψ'_n(z)/ψ_n(z)` the logarithmic derivative, `ψ_n = z j_n`,
`ξ_n = ψ_n - iχ_n`, `χ_n = -z y_n` the Riccati-Bessel functions. From
these

```math
Q_{ext} = \\frac{2}{x^2}\\sum_n (2n+1)\\,\\mathrm{Re}(a_n + b_n),\\qquad
Q_{sca} = \\frac{2}{x^2}\\sum_n (2n+1)(|a_n|^2 + |b_n|^2)
```

```math
g\\,Q_{sca} = \\frac{4}{x^2}\\left[
  \\sum_n \\frac{n(n+2)}{n+1}\\mathrm{Re}(a_n a_{n+1}^* + b_n b_{n+1}^*)
+ \\sum_n \\frac{2n+1}{n(n+1)}\\mathrm{Re}(a_n b_n^*)\\right]
```

`D_n` is obtained by **downward** recursion
`D_{n-1} = n/z - 1/(D_n + n/z)` seeded at order `N_mx = max(N_stop,
|mx|) + 15` with `D_{N_mx} = 0`. The upward form of the same recursion is
numerically unstable — for `|mx| ≳ 10` it loses every significant digit
within a few tens of terms, which produces plausible-looking but
completely wrong absorption. This is the single most important numerical
point in a Mie code (Wiscombe 1980, §2).

Truncation order `N_stop = ⌈x + 4x^{1/3} + 2⌉` (Wiscombe 1980 Eq. 20);
pass `nmax` to override.

Benchmarks reproduced by the test suite (Wiscombe 1980, MIEV0 test set):

| x     | m     | Q_ext      | g         |
|:------|:------|:-----------|:----------|
| 0.101 | 0.75  | 8.03354e-6 | 1.5e-3    |
| 10    | 1.5   | 2.881999   | 0.742913  |
| 100   | 1.5   | 2.094387   | 0.818243  |
| 1000  | 1.5   | 2.013079   | 0.828011  |
"""
function mie_sphere(x::Real, m::Complex; nmax::Int = 0)
    xx = Float64(x)
    mm = ComplexF64(m)
    xx > 0 || return MieResult(0.0, 0.0, 0.0, 0.0, 0.0, 0)
    imag(mm) >= 0 ||
        error("mie_sphere: refractive index must have Im(m) ≥ 0 " *
              "(Bohren-Huffman convention m = n + ik); got $mm")
    real(mm) > 0 || error("mie_sphere: Re(m) must be positive, got $mm")

    # Small-particle branch: the Riccati-Bessel upward recursion is
    # hopeless here, and the Rayleigh limit is exact to O(x²).
    xx < 1e-3 && return rayleigh_efficiencies(xx, mm)

    y = mm * xx
    nstop = nmax > 0 ? nmax : ceil(Int, xx + 4 * cbrt(xx) + 2)
    nstop = max(nstop, 1)
    nmx = max(nstop, ceil(Int, abs(y))) + 15

    D = _D_BUFFER
    length(D) < nmx && resize!(D, nmx)
    @inbounds D[nmx] = zero(ComplexF64)
    @inbounds for n in (nmx - 1):-1:1
        en = n + 1
        t = en / y
        D[n] = t - 1 / (D[n + 1] + t)
    end

    # Riccati-Bessel functions by upward recursion. `psi0`/`chi0` hold
    # order n-2, `psi1`/`chi1` order n-1 at the top of each iteration.
    psi0 = cos(xx); psi1 = sin(xx)          # ψ₋₁, ψ₀
    chi0 = -sin(xx); chi1 = cos(xx)         # χ₋₁, χ₀
    xi1 = complex(psi1, -chi1)              # ξ₀

    sum_ext = 0.0
    sum_sca = 0.0
    sum_g   = 0.0
    back    = zero(ComplexF64)
    an_prev = zero(ComplexF64)
    bn_prev = zero(ComplexF64)

    @inbounds for n in 1:nstop
        en = Float64(n)
        fn = (2en + 1) / (en * (en + 1))
        psi = (2en - 1) / xx * psi1 - psi0
        chi = (2en - 1) / xx * chi1 - chi0
        xi  = complex(psi, -chi)

        Dn = D[n]
        ta = Dn / mm + en / xx
        tb = Dn * mm + en / xx
        an = (ta * psi - psi1) / (ta * xi - xi1)
        bn = (tb * psi - psi1) / (tb * xi - xi1)

        sum_ext += (2en + 1) * real(an + bn)
        sum_sca += (2en + 1) * (abs2(an) + abs2(bn))
        sum_g   += fn * real(an * conj(bn))
        if n > 1
            sum_g += ((en - 1) * (en + 1) / en) *
                     real(an_prev * conj(an) + bn_prev * conj(bn))
        end
        back += (2en + 1) * (isodd(n) ? -1 : 1) * (an - bn)

        psi0 = psi1; psi1 = psi
        chi0 = chi1; chi1 = chi
        xi1  = complex(psi1, -chi1)
        an_prev = an; bn_prev = bn
    end

    inv_x2 = 1 / (xx * xx)
    Qext = 2 * inv_x2 * sum_ext
    Qsca = 2 * inv_x2 * sum_sca
    # Round-off can push Qsca a hair above Qext for a non-absorbing
    # sphere; absorption is physically non-negative so clamp it.
    Qabs = max(Qext - Qsca, 0.0)
    g    = Qsca > 0 ? 4 * inv_x2 * sum_g / Qsca : 0.0
    Qback = inv_x2 * abs2(back)
    return MieResult(Qext, Qsca, Qabs, Qback, g, nstop)
end

"""
    mie_efficiencies(r, λ, m) -> MieResult

Convenience wrapper: Mie solution for a sphere of radius `r` at
wavelength `λ` (same length units) and refractive index `m`.
"""
mie_efficiencies(r::Real, λ::Real, m::Complex) =
    mie_sphere(size_parameter(r, λ), m)

# =========================================================================
# 2. Aerosol species and their refractive indices
# =========================================================================

"""
    AerosolSpecies

A chemical aerosol component: a tabulated complex refractive index
`m(λ) = n(λ) + i k(λ)`, a material density and a κ hygroscopicity.

  * `name`  — identifying symbol
  * `λ`     — wavelength table, μm, strictly increasing
  * `n`, `k`— real and imaginary parts at those wavelengths (`k ≥ 0`)
  * `ρ`     — material density, g cm⁻³
  * `κ`     — Petters-Kreidenweis hygroscopicity parameter
  * `source`— literature provenance of the optical constants

The imaginary part spans ~8 decades across the species set (10⁻⁸ for
sulphate in the visible, 0.44 for soot), so `k` is interpolated
logarithmically; `n` varies by tens of percent and is interpolated
linearly. Both are interpolated against `log λ` and *clamped* outside
the table rather than extrapolated — extrapolating a resonance band is a
good way to invent absorption that does not exist.
"""
struct AerosolSpecies
    name::Symbol
    λ::Vector{Float64}
    n::Vector{Float64}
    k::Vector{Float64}
    ρ::Float64
    κ::Float64
    source::String
end

function AerosolSpecies(name::Symbol, λ, n, k, ρ, κ, source)
    length(λ) == length(n) == length(k) ||
        error("AerosolSpecies($name): λ, n, k must have equal length")
    issorted(λ) || error("AerosolSpecies($name): λ table must be increasing")
    all(>(0), n) || error("AerosolSpecies($name): n must be positive")
    all(>=(0), k) || error("AerosolSpecies($name): k must be non-negative")
    ρ > 0 || error("AerosolSpecies($name): density must be positive")
    κ >= 0 || error("AerosolSpecies($name): κ must be non-negative")
    return AerosolSpecies(name, collect(Float64, λ), collect(Float64, n),
                          collect(Float64, k), Float64(ρ), Float64(κ),
                          String(source))
end

Base.show(io::IO, s::AerosolSpecies) =
    print(io, "AerosolSpecies(:", s.name, ", ρ=", s.ρ, " g/cm³, κ=", s.κ,
              ", λ ", s.λ[1], "–", s.λ[end], " μm)")

# Lower bound on k. A strictly zero imaginary part is unphysical and
# makes log-interpolation singular; 1e-9 is far below any radiative
# significance (it gives ω₀ > 0.99999 for accumulation-mode particles).
const _K_FLOOR = 1e-9

"""
    refractive_index(sp::AerosolSpecies, λ) -> ComplexF64

Complex refractive index of `sp` at wavelength `λ` (μm), `n + ik` with
`k ≥ 0`. Linear-in-`log λ` for `n`, linear-in-`log λ` for `log k`,
clamped at the table ends.
"""
function refractive_index(sp::AerosolSpecies, λ::Real)
    λt = sp.λ
    x = Float64(λ)
    if x <= λt[1]
        return complex(sp.n[1], max(sp.k[1], _K_FLOOR))
    elseif x >= λt[end]
        return complex(sp.n[end], max(sp.k[end], _K_FLOOR))
    end
    j = searchsortedlast(λt, x)
    j = clamp(j, 1, length(λt) - 1)
    t = (log(x) - log(λt[j])) / (log(λt[j + 1]) - log(λt[j]))
    nv = sp.n[j] + t * (sp.n[j + 1] - sp.n[j])
    k0 = max(sp.k[j], _K_FLOOR); k1 = max(sp.k[j + 1], _K_FLOOR)
    kv = exp(log(k0) + t * (log(k1) - log(k0)))
    return complex(nv, kv)
end

# --- the species library -------------------------------------------------
#
# Optical constants are abridged from the standard compilations. The
# tables are deliberately coarse in the windows where m(λ) is flat and
# dense across the resonance features (the 3 μm OH stretch of water, the
# 9 μm Si-O band of silicate dust, the 8-9 μm sulphate band, the 7.2 μm
# NO₃⁻ band), because that is where band-mean optics is actually
# sensitive to the sampling.

# Liquid water — Hale & Querry (1973), abridged.
const _WATER = AerosolSpecies(:water,
    [0.20,   0.30,   0.40,    0.55,    0.70,    1.00,    1.50,   2.00,
     2.50,   2.85,   3.00,    3.20,    3.50,    4.00,    5.00,   6.00,
     6.10,   7.00,   8.00,    9.00,    10.00,   11.00,   12.00,  14.00,
     16.00,  20.00,  25.00,   40.00,   100.00],
    [1.396,  1.349,  1.339,   1.333,   1.331,   1.327,   1.322,  1.306,
     1.261,  1.142,  1.371,   1.478,   1.401,   1.351,   1.325,  1.265,
     1.303,  1.317,  1.291,   1.262,   1.218,   1.153,   1.111,  1.196,
     1.336,  1.480,  1.616,   1.822,   2.070],
    [1.1e-7, 1.6e-8, 1.86e-9, 1.96e-9, 2.93e-8, 2.89e-6, 1.0e-4, 1.1e-3,
     1.75e-3,0.2610, 0.2720,  0.0920,  0.0090,  0.0046,  0.0124, 0.1076,
     0.1280, 0.0392, 0.0339,  0.0392,  0.0508,  0.0662,  0.0968, 0.1420,
     0.2000, 0.3500, 0.3900,  0.5100,  0.6600],
    1.0, 0.0, "Hale & Querry (1973)")

# Sulphate — a 75 % H₂SO₄ solution droplet, OPAC component `suso`.
# Essentially non-absorbing in the solar spectrum (k ~ 10⁻⁸), strongly
# absorbing in the 8-10 μm SO₄²⁻ bands.
const _SULPHATE = AerosolSpecies(:sulphate,
    [0.25,   0.30,   0.40,   0.55,   0.70,    1.00,    1.50,    2.00,
     2.50,   3.00,   3.50,   4.00,   5.00,    6.00,    7.00,    8.00,
     8.50,   9.00,   10.00,  11.00,  12.00,   14.00,   16.00,   20.00,
     25.00,  40.00,  100.00],
    [1.484,  1.469,  1.452,  1.440,  1.432,   1.422,   1.410,   1.393,
     1.372,  1.398,  1.359,  1.344,  1.322,   1.302,   1.376,   1.655,
     1.810,  1.899,  1.635,  1.710,  1.830,   1.800,   1.750,   1.800,
     1.900,  1.950,  1.980],
    [1.0e-8, 1.0e-8, 1.0e-8, 1.0e-8, 1.47e-8, 1.55e-6, 1.85e-4, 1.14e-3,
     1.61e-3,0.0362, 0.0111, 0.0102, 0.0135,  0.0705,  0.1417,  0.4784,
     0.5380, 0.4915, 0.4525, 0.4200, 0.4000,  0.3500,  0.3000,  0.2500,
     0.2000, 0.1800, 0.1600],
    1.7, 0.60, "OPAC `suso`; Palmer & Williams (1975)")

# Black carbon / soot — OPAC `soot`. Large and nearly flat k ≈ 0.44-0.46
# through the solar spectrum: this is what makes BC the only common
# aerosol that heats the atmosphere.
const _BLACK_CARBON = AerosolSpecies(:black_carbon,
    [0.25,  0.30,  0.40,  0.55,  0.70,  1.00,  1.50,  2.00,  2.50,
     3.00,  3.50,  4.00,  5.00,  6.00,  7.00,  8.00,  9.00,  10.00,
     12.00, 14.00, 16.00, 20.00, 25.00, 40.00, 100.00],
    [1.62,  1.74,  1.75,  1.75,  1.75,  1.76,  1.79,  1.81,  1.82,
     1.84,  1.86,  1.88,  1.92,  1.94,  1.97,  2.00,  2.02,  2.06,
     2.13,  2.20,  2.26,  2.37,  2.52,  2.85,  3.30],
    [0.45,  0.47,  0.46,  0.44,  0.43,  0.43,  0.44,  0.45,  0.45,
     0.46,  0.48,  0.49,  0.51,  0.54,  0.56,  0.57,  0.59,  0.60,
     0.63,  0.66,  0.68,  0.72,  0.78,  0.89,  1.05],
    1.0, 0.0, "OPAC `soot`; d'Almeida et al. (1991)")

# Organic carbon — weakly absorbing in the visible, with the
# characteristic "brown carbon" rise into the near-UV.
const _ORGANIC_CARBON = AerosolSpecies(:organic_carbon,
    [0.25,  0.30,  0.35,  0.40,  0.45,  0.55,  0.70,  1.00,  1.50,
     2.00,  2.50,  3.00,  3.50,  4.00,  5.00,  6.00,  7.00,  8.00,
     9.00,  10.00, 12.00, 14.00, 16.00, 20.00, 25.00, 40.00, 100.00],
    [1.530, 1.530, 1.530, 1.530, 1.530, 1.530, 1.530, 1.520, 1.510,
     1.460, 1.420, 1.420, 1.450, 1.460, 1.450, 1.420, 1.400, 1.420,
     1.430, 1.400, 1.420, 1.450, 1.470, 1.500, 1.530, 1.600, 1.650],
    [0.0700,0.0350,0.0190,0.0100,0.0060,0.0050,0.0050,0.0050,0.0080,
     0.0120,0.0140,0.0820,0.0280,0.0160,0.0140,0.0320,0.0510,0.0450,
     0.0480,0.0600,0.0800,0.1000,0.1200,0.1600,0.2000,0.2800,0.3600],
    1.5, 0.10, "OPAC water-soluble organic; Kirchstetter et al. (2004)")

# Mineral dust — silicate with a few percent hematite. Strongly
# absorbing in the near-UV/blue (which is why dust plumes look brown),
# almost transparent in the red and near-IR, and with the huge 9-10 μm
# Si-O band that gives dust its longwave warming effect.
const _MINERAL_DUST = AerosolSpecies(:mineral_dust,
    [0.25,   0.30,   0.35,   0.40,   0.45,   0.50,   0.55,   0.60,
     0.70,   1.00,   1.50,   2.00,   2.50,   3.00,   3.50,   4.00,
     5.00,   6.00,   7.00,   8.00,   8.50,   9.00,   9.50,   10.00,
     11.00,  12.00,  14.00,  16.00,  20.00,  25.00,  40.00,  100.00],
    [1.530,  1.530,  1.530,  1.530,  1.530,  1.530,  1.530,  1.530,
     1.530,  1.530,  1.530,  1.520,  1.510,  1.470,  1.420,  1.420,
     1.420,  1.450,  1.470,  1.520,  1.680,  1.830,  1.950,  1.860,
     1.650,  1.720,  1.860,  2.020,  2.200,  2.100,  2.200,  2.300],
    [0.0400, 0.0250, 0.0170, 0.0080, 0.0055, 0.0042, 0.0037, 0.0030,
     0.0025, 0.0015, 0.0011, 0.0012, 0.0014, 0.0080, 0.0055, 0.0040,
     0.0038, 0.0050, 0.0080, 0.0450, 0.1400, 0.3200, 0.4600, 0.4200,
     0.2300, 0.1600, 0.2100, 0.3400, 0.5000, 0.4500, 0.5500, 0.6500],
    2.6, 0.03, "OPAC mineral components; Volz (1973), Sokolik & Toon (1999)")

# Sea salt — a deliquesced NaCl-dominated droplet. Non-absorbing in the
# solar spectrum and the most hygroscopic common aerosol.
const _SEA_SALT = AerosolSpecies(:sea_salt,
    [0.25,   0.30,   0.40,   0.55,   0.70,    1.00,    1.50,    2.00,
     2.50,   3.00,   3.50,   4.00,   5.00,    6.00,    7.00,    8.00,
     9.00,   10.00,  11.00,  12.00,  14.00,   16.00,   20.00,   25.00,
     40.00,  100.00],
    [1.510,  1.510,  1.500,  1.500,  1.490,   1.480,   1.470,   1.470,
     1.460,  1.450,  1.450,  1.450,  1.430,   1.610,   1.490,   1.480,
     1.480,  1.470,  1.480,  1.490,  1.500,   1.510,   1.520,   1.550,
     1.600,  1.650],
    [1.0e-8, 1.0e-8, 1.0e-8, 1.0e-8, 1.60e-8, 1.50e-6, 1.00e-4, 1.50e-4,
     3.0e-4, 2.6e-3, 1.2e-3, 8.0e-4, 1.5e-3,  1.4e-2,  4.0e-3,  3.0e-3,
     3.5e-3, 5.0e-3, 8.0e-3, 1.2e-2, 2.5e-2,  5.0e-2,  1.0e-1,  1.5e-1,
     2.5e-1, 3.5e-1],
    2.2, 1.10, "OPAC `ssam`; Shettle & Fenn (1979)")

# Ammonium nitrate — the dominant secondary inorganic aerosol in
# agricultural/urban Europe. Transparent in the solar, with a very
# strong NO₃⁻ ν₃ band at 7.2 μm.
const _AMMONIUM_NITRATE = AerosolSpecies(:ammonium_nitrate,
    [0.25,   0.30,   0.40,   0.55,   0.70,    1.00,    1.50,    2.00,
     2.50,   3.00,   3.50,   4.00,   5.00,    6.00,    7.00,    7.30,
     7.80,   8.00,   9.00,   10.00,  11.00,   12.00,   14.00,   16.00,
     20.00,  25.00,  40.00,  100.00],
    [1.600,  1.580,  1.560,  1.554,  1.550,   1.540,   1.530,   1.520,
     1.510,  1.480,  1.440,  1.470,  1.480,   1.490,   1.900,   1.300,
     1.420,  1.500,  1.520,  1.540,  1.550,   1.570,   1.600,   1.620,
     1.650,  1.700,  1.750,  1.800],
    [1.0e-8, 1.0e-8, 1.0e-8, 1.0e-8, 2.00e-8, 3.00e-7, 2.00e-5, 3.00e-4,
     1.0e-3, 3.5e-2, 2.0e-2, 8.0e-3, 5.0e-3,  1.0e-2,  5.5e-1,  6.0e-1,
     1.2e-1, 3.0e-2, 2.0e-2, 2.5e-2, 3.0e-2,  4.0e-2,  6.0e-2,  8.0e-2,
     1.2e-1, 1.5e-1, 2.0e-1, 2.6e-1],
    1.725, 0.67, "Jarzembski et al. (2003); Toon et al. (1976)")

const _SPECIES_REGISTRY = Dict{Symbol, AerosolSpecies}(
    :water            => _WATER,
    :sulphate         => _SULPHATE,
    :sulfate          => _SULPHATE,
    :black_carbon     => _BLACK_CARBON,
    :soot             => _BLACK_CARBON,
    :organic_carbon   => _ORGANIC_CARBON,
    :mineral_dust     => _MINERAL_DUST,
    :dust             => _MINERAL_DUST,
    :sea_salt         => _SEA_SALT,
    :ammonium_nitrate => _AMMONIUM_NITRATE,
)

"""
    aerosol_species(name::Symbol) -> AerosolSpecies

Look up a species from the built-in library. Available:
`:sulphate` (alias `:sulfate`), `:black_carbon` (alias `:soot`),
`:organic_carbon`, `:mineral_dust` (alias `:dust`), `:sea_salt`,
`:ammonium_nitrate`, and `:water` (the condensed phase used for
hygroscopic mixing).

The returned objects are library singletons; the internal optics cache
relies on that, so do not mutate the returned tables.
"""
function aerosol_species(name::Symbol)
    haskey(_SPECIES_REGISTRY, name) ||
        error("aerosol_species: unknown species :$name. Known: " *
              string(sort(collect(keys(_SPECIES_REGISTRY)))))
    return _SPECIES_REGISTRY[name]
end

"Canonical (non-alias) species names in the library."
aerosol_species_names() = [:sulphate, :black_carbon, :organic_carbon,
                           :mineral_dust, :sea_salt, :ammonium_nitrate]

"Complex refractive index of liquid water at `λ` μm (Hale & Querry 1973)."
water_refractive_index(λ::Real) = refractive_index(_WATER, λ)

# =========================================================================
# 3. Hygroscopic growth — κ-Köhler
# =========================================================================

# Kelvin coefficient A = 2σ_w M_w / (R T ρ_w), in μm, at 298.15 K with
# σ_w = 0.0728 N m⁻¹. ≈ 1.06 nm.
const _SIGMA_WATER = 0.0728      # N m⁻¹
const _M_WATER     = 0.0180153   # kg mol⁻¹
const _R_GAS       = 8.314462618 # J mol⁻¹ K⁻¹
const _RHO_WATER   = 1000.0      # kg m⁻³

"Kelvin coefficient `A = 2σM/(RTρ)` in μm at 298.15 K (≈ 1.06 × 10⁻³ μm)."
const KELVIN_A_298 = 2 * _SIGMA_WATER * _M_WATER /
                     (_R_GAS * 298.15 * _RHO_WATER) * 1e6

kelvin_coefficient(T_K::Real) =
    2 * _SIGMA_WATER * _M_WATER / (_R_GAS * T_K * _RHO_WATER) * 1e6

"""
    kohler_saturation(r_wet, r_dry, κ; T_K = 298.15) -> S

Equilibrium water-vapour saturation ratio over a solution droplet of wet
radius `r_wet` grown from a dry radius `r_dry` (both μm), in the
single-parameter (κ-Köhler) representation of Petters & Kreidenweis
(2007):

```math
S(r_w) = \\frac{r_w^3 - r_d^3}{r_w^3 - r_d^3(1-κ)}\\;
         \\exp\\!\\left(\\frac{A}{r_w}\\right)
```

The first factor is the solute (Raoult) term — `κ` is the volume of
water taken up per unit volume of dry solute at unit water activity —
and the exponential is the Kelvin curvature term.
"""
function kohler_saturation(r_wet::Real, r_dry::Real, κ::Real;
                           T_K::Real = 298.15)
    rw = Float64(r_wet); rd = Float64(r_dry)
    rw > rd || return 0.0
    A = kelvin_coefficient(T_K)
    rw3 = rw^3; rd3 = rd^3
    solute = (rw3 - rd3) / (rw3 - rd3 * (1 - Float64(κ)))
    return solute * exp(A / rw)
end

"""
    wet_radius(r_dry, rh, κ; T_K = 298.15) -> r_wet

Equilibrium wet radius (μm) of a dry particle of radius `r_dry` (μm) at
relative humidity `rh ∈ [0,1)`, by inverting [`kohler_saturation`](@ref).

For sub-saturated air the Köhler curve `S(r_w)` rises monotonically from
0 at `r_w = r_d` to the critical supersaturation `S_c > 1` and then
decays back towards 1 from *above*, so `S(r_w) = RH < 1` has exactly one
root and it lies on the stable (lower) branch. Bisection is therefore
globally convergent and needs no branch selection.

Hydrophobic species (`κ ≈ 0`) return `r_dry` unchanged: with no solute
the only term left is the Kelvin factor, which is > 1 for every radius,
so no sub-saturated equilibrium exists — physically, a soot particle
does not take up water.

Deliquescence/efflorescence hysteresis is *not* represented: κ-Köhler
gives the metastable, continuously-deliquesced branch. That is the
appropriate choice for aged ambient aerosol (which has usually been
above its deliquescence point at some time in its history), but it will
overestimate uptake for freshly crystalline salt between the
efflorescence and deliquescence humidities.
"""
function wet_radius(r_dry::Real, rh::Real, κ::Real; T_K::Real = 298.15)
    rd = Float64(r_dry)
    rd > 0 || return 0.0
    kap = Float64(κ)
    S = Float64(rh)
    (kap <= 1e-8 || S <= 0) && return rd
    S = min(S, 0.9999)                    # cap: no supersaturated branch
    lo = rd * (1 + 1e-12)
    hi = rd * 1e4                         # S(hi) > 1 > S always
    kohler_saturation(hi, rd, kap; T_K = T_K) > S ||
        return hi                         # defensive; never taken in practice
    for _ in 1:200
        mid = 0.5 * (lo + hi)
        (mid <= lo || mid >= hi) && break
        if kohler_saturation(mid, rd, kap; T_K = T_K) < S
            lo = mid
        else
            hi = mid
        end
        (hi - lo) <= 1e-14 * hi && break
    end
    return 0.5 * (lo + hi)
end

"""
    growth_factor(r_dry, rh, κ; T_K = 298.15) -> r_wet / r_dry

Hygroscopic growth factor. In the large-particle limit the Kelvin term
is negligible and the κ-Köhler relation inverts analytically to

```math
GF = \\left(1 + κ\\frac{RH}{1-RH}\\right)^{1/3}
```

so `GF` diverges as `(1-RH)^{-1/3}` at saturation. The `(1-RH)^{-1}`
inside the cube root is the reason aerosol extinction explodes in the
last few percent of humidity and is the dominant control on visibility
in haze.

Measured values from this implementation at `r_dry = 0.1 μm`: sea salt
(κ = 1.10) grows by ×2.19 at RH = 90 % and ×4.47 at RH = 99 %; sulphate
(κ = 0.60) by ×1.83 and ×3.61; ammonium nitrate (κ = 0.67) by ×1.89 and
×3.75; organic carbon (κ = 0.10) by ×1.22 and ×1.95; mineral dust
(κ = 0.03) by ×1.08 and ×1.39; black carbon (κ = 0) not at all.
"""
growth_factor(r_dry::Real, rh::Real, κ::Real; T_K::Real = 298.15) =
    wet_radius(r_dry, rh, κ; T_K = T_K) / r_dry

"""
    wet_refractive_index(m_dry, r_dry, r_wet) -> ComplexF64

Refractive index of a grown particle by **volume mixing** of the dry
material and the condensed water:

```math
m_{wet} = f\\,m_{dry} + (1-f)\\,m_{water},\\qquad f = (r_d/r_w)^3
```

Volume mixing is the rule used by OPAC and by most GCM aerosol schemes.
It is exact in neither limit — a Maxwell-Garnett or Bruggeman effective
medium is formally better for an inclusion-in-matrix morphology — but
for a genuinely dissolved solute (which is what a deliquesced sulphate
or sea-salt droplet is) volume mixing of the *solution* refractive index
is the physically correct construction, and the differences are a few
percent in `ω₀`.
"""
function wet_refractive_index(m_dry::Complex, r_dry::Real, r_wet::Real,
                              m_water::Complex)
    r_wet <= r_dry && return ComplexF64(m_dry)
    f = (Float64(r_dry) / Float64(r_wet))^3
    return f * ComplexF64(m_dry) + (1 - f) * ComplexF64(m_water)
end

wet_refractive_index(m_dry::Complex, r_dry::Real, r_wet::Real, λ::Real) =
    wet_refractive_index(m_dry, r_dry, r_wet, water_refractive_index(λ))

# =========================================================================
# 4. Size distributions
# =========================================================================

"""
    AerosolMode

Abstract supertype for one mode of an aerosol size distribution. A
complete aerosol population is a `Vector{<:AerosolMode}` — the standard
representation, because ambient aerosol is essentially never unimodal
(nucleation, accumulation and coarse modes have different sources, sinks
and chemistry).
"""
abstract type AerosolMode end

"""
    LogNormalMode{T} <: AerosolMode

One log-normal mode:

```math
\\frac{dN}{d\\ln r} = \\frac{N}{\\sqrt{2π}\\,\\ln σ_g}
  \\exp\\!\\left[-\\frac{\\ln^2(r/r_g)}{2\\ln^2σ_g}\\right]
```

  * `species` — the chemical component
  * `N`       — total number concentration, cm⁻³
  * `r_g`     — geometric median **dry** radius, μm
  * `σ_g`     — geometric standard deviation (dimensionless, > 1)

The log-normal is not a convenience: coagulation and condensation drive
a size distribution towards self-preserving log-normal shape, which is
why measured aerosol spectra are described by two or three of them.

Moments are analytic, `⟨r^p⟩ = r_g^p \\exp(p^2\\ln^2σ_g/2)`, and grow
explosively with `p`: for `σ_g = 2` the sixth moment (which controls
Rayleigh scattering) is 5700 `r_g^6`. This is why the quadrature range
in [`mode_optics`](@ref) is extended on the coarse side.
"""
struct LogNormalMode{T<:AbstractFloat} <: AerosolMode
    species::AerosolSpecies
    N::T
    r_g::T
    σ_g::T
end

"""
    lognormal_mode(species, N, r_g, σ_g; T = Float64) -> LogNormalMode

Construct a log-normal mode. `species` may be an `AerosolSpecies` or a
symbol resolved through [`aerosol_species`](@ref).
"""
function lognormal_mode(species, N::Real, r_g::Real, σ_g::Real;
                        T::Type{<:AbstractFloat} = Float64)
    sp = species isa AerosolSpecies ? species : aerosol_species(Symbol(species))
    N >= 0 || error("lognormal_mode: N must be non-negative")
    r_g > 0 || error("lognormal_mode: r_g must be positive")
    σ_g > 1 || error("lognormal_mode: σ_g must exceed 1")
    return LogNormalMode{T}(sp, T(N), T(r_g), T(σ_g))
end

Base.show(io::IO, m::LogNormalMode) =
    print(io, "LogNormalMode(:", m.species.name, ", N=", m.N, " cm⁻³, r_g=",
              m.r_g, " μm, σ_g=", m.σ_g, ")")

"""
    number_pdf(mode, r) -> dN/dln r  [cm⁻³]

Log-normal number distribution per unit `ln r`, evaluated at radius `r`
(μm).
"""
function number_pdf(m::LogNormalMode, r::Real)
    lnσ = log(m.σ_g)
    z = (log(Float64(r)) - log(m.r_g)) / lnσ
    return m.N / (sqrt(2π) * lnσ) * exp(-0.5 * z * z)
end

"Total number concentration of a mode or a mode list, cm⁻³."
mode_number(m::LogNormalMode) = Float64(m.N)
mode_number(ms::AbstractVector{<:AerosolMode}) = sum(mode_number, ms; init = 0.0)

"""
    mode_mass(mode) -> μg m⁻³

Dry mass concentration of a log-normal mode,
`M = N ρ (4π/3) r_g³ exp(4.5 ln²σ_g)`.
"""
function mode_mass(m::LogNormalMode)
    lnσ2 = log(m.σ_g)^2
    r3 = Float64(m.r_g)^3 * exp(4.5 * lnσ2)          # μm³
    # N [cm⁻³] · r³ [μm³] · ρ [g cm⁻³]:
    # 1 μm³ = 1e-12 cm³; per cm⁻³ → g cm⁻³ → ×1e12 μg m⁻³
    return Float64(m.N) * (4π / 3) * r3 * m.species.ρ * 1e-12 * 1e12
end
mode_mass(ms::AbstractVector{<:AerosolMode}) = sum(mode_mass, ms; init = 0.0)

"""
    effective_radius(mode) -> μm

Area-weighted (effective) radius `⟨r³⟩/⟨r²⟩ = r_g exp(2.5 ln²σ_g)` —
the radius a radiatively equivalent monodisperse population would have.
"""
effective_radius(m::LogNormalMode) = Float64(m.r_g) * exp(2.5 * log(m.σ_g)^2)

function effective_radius(ms::AbstractVector{<:AerosolMode})
    num = 0.0; den = 0.0
    for m in ms
        lnσ2 = log(m.σ_g)^2
        num += m.N * m.r_g^3 * exp(4.5 * lnσ2)
        den += m.N * m.r_g^2 * exp(2.0 * lnσ2)
    end
    return den > 0 ? num / den : 0.0
end

# --- integration controls -------------------------------------------------
# Physical truncation of the distribution, following OPAC: no ambient
# aerosol particle survives sedimentation beyond ~20 μm dry radius, and
# below 1 nm one is describing molecular clusters, not particles.
const R_MIN_UM = 1.0e-3
const R_MAX_UM = 20.0
const NQUAD_DEFAULT = 120
const NQUAD_CEILING_FACTOR = 6      # hard cap = NQUAD_CEILING_FACTOR·nquad
const POINTS_PER_RIPPLE = 5

# Cache of per-particle bulk optics. Key is
# (species name, r_g, σ_g, λ, RH, nquad). The species registry returns
# singletons so the name uniquely identifies the optical constants;
# a caller supplying a *custom* `AerosolSpecies` whose name collides with
# a library one would alias, which is why `mode_optics` bypasses the
# cache for species that are not the registry singletons.
const _BULK_CACHE = Dict{NTuple{6, Any}, NTuple{4, Float64}}()

"Empty the memoised single-mode optics cache."
clear_optics_cache!() = (empty!(_BULK_CACHE); nothing)

"""
    _quadrature_points(r_lo, r_hi, λ, nquad) -> Int

Number of Simpson intervals for one mode. `nquad` is a *floor*: for
coarse modes the interference ("ripple") structure of `Q_ext(x)` has a
period of roughly `Δx = π`, so resolving it needs about
`POINTS_PER_RIPPLE·(x_hi - x_lo)/π` points. A fixed 120 intervals is
ample for an accumulation mode (`x < 40`) but under-resolves a 20 μm
sea-salt or dust particle at 0.4 μm (`x > 300`), where it leaves a ~0.3 %
ripple-aliasing error in `β_ext`. Capped at `NQUAD_CEILING_FACTOR·nquad`
so the cost stays bounded in the ultraviolet bands.
"""
function _quadrature_points(r_lo::Real, r_hi::Real, λ::Real, nquad::Int)
    xspan = size_parameter(r_hi, λ) - size_parameter(r_lo, λ)
    want = ceil(Int, POINTS_PER_RIPPLE * xspan / π)
    nq = clamp(want, nquad, NQUAD_CEILING_FACTOR * nquad)
    return iseven(nq) ? nq : nq + 1
end

_is_library_species(sp::AerosolSpecies) =
    haskey(_SPECIES_REGISTRY, sp.name) && _SPECIES_REGISTRY[sp.name] === sp

"""
    mode_optics(mode, λ; rh = 0.0, T_K = 298.15, nquad = 120)
        -> (β_ext, β_sca, β_abs, g)

Bulk optical coefficients of a single log-normal mode at wavelength `λ`
(μm) and relative humidity `rh`, **per unit number concentration**
(i.e. for `N = 1 cm⁻³`), in m⁻¹:

```math
β_{ext} = \\int_0^∞ π r_w^2\\,Q_{ext}(x_w, m_w)\\,\\frac{dN}{d\\ln r_d}\\,d\\ln r_d
```

with, at each dry radius `r_d`, the wet radius `r_w` from
[`wet_radius`](@ref), the volume-mixed refractive index `m_w` from
[`wet_refractive_index`](@ref) and `x_w = 2πr_w/λ`. Note the integral is
over the **dry** distribution: growth conserves particle number, so each
dry particle maps to exactly one wet particle.

The asymmetry parameter returned is the *scattering-weighted* mean,
`g = ∫ πr²Q_sca g dN / ∫ πr²Q_sca dN`, which is the only weighting that
makes `combine_optics!` associative.

# Quadrature
Composite Simpson in `ln r` over
`[r_g e^{-4\\lnσ_g},\\; r_g e^{+4\\lnσ_g + 3\\ln^2σ_g}]`, clipped to
`[1 nm, 20 μm]`. The asymmetric upper limit tracks the mode of the
`r³`-weighted integrand (`\\ln r_g + 3\\ln^2σ_g`), which for `σ_g = 2.5`
sits **2.7 σ above** the number median — a symmetric range would
truncate the very particles that dominate extinction.

`nquad` is a floor on the number of Simpson intervals; the actual count
is raised for coarse modes by [`_quadrature_points`](@ref) so that the
Mie ripple is resolved. Measured convergence of `β_ext(550 nm)` against a
7680-interval reference: better than 0.15 % for every library mixture at
the default `nquad = 120` (6 × 10⁻⁵ for the fine-mode-dominated ones;
the residual is ripple aliasing in the 20 μm sea-salt and dust tails).
"""
function mode_optics(m::LogNormalMode, λ::Real; rh::Real = 0.0,
                     T_K::Real = 298.15, nquad::Int = NQUAD_DEFAULT)
    sp = m.species
    key = (sp.name, Float64(m.r_g), Float64(m.σ_g), Float64(λ),
           Float64(rh), nquad)
    cacheable = _is_library_species(sp) && T_K == 298.15
    if cacheable
        hit = get(_BULK_CACHE, key, nothing)
        hit === nothing || return (hit[1] * m.N, hit[2] * m.N,
                                   hit[3] * m.N, hit[4])
    end

    lnσ = log(Float64(m.σ_g))
    rg = Float64(m.r_g)
    lo = max(R_MIN_UM, rg * exp(-4 * lnσ))
    hi = min(R_MAX_UM, rg * exp(4 * lnσ + 3 * lnσ * lnσ))
    hi > lo || (hi = lo * 1.0001)

    nq = _quadrature_points(lo, hi, λ, nquad)
    a = log(lo); b = log(hi)
    h = (b - a) / nq

    m_dry_cache = refractive_index(sp, λ)
    m_wat = water_refractive_index(λ)
    κ = sp.κ
    λf = Float64(λ)

    # unit-N distribution: divide the pdf by N
    invN = m.N > 0 ? 1 / Float64(m.N) : 1.0
    Se = 0.0; Ss = 0.0; Sg = 0.0
    @inbounds for i in 0:nq
        w = (i == 0 || i == nq) ? 1.0 : (isodd(i) ? 4.0 : 2.0)
        r_d = exp(a + i * h)
        pdf = number_pdf(m, r_d) * invN
        pdf <= 0 && continue
        r_w = κ > 1e-8 && rh > 0 ? wet_radius(r_d, rh, κ; T_K = T_K) : r_d
        mw = r_w > r_d ? wet_refractive_index(m_dry_cache, r_d, r_w, m_wat) :
                         m_dry_cache
        x = size_parameter(r_w, λf)
        q = mie_sphere(x, mw)
        area = π * r_w * r_w                       # μm²
        Se += w * pdf * area * q.Q_ext
        Ss += w * pdf * area * q.Q_sca
        Sg += w * pdf * area * q.Q_sca * q.g
    end
    f = h / 3
    # μm² cm⁻³ → m⁻¹ :  1e-12 m² × 1e6 m⁻³ = 1e-6 m⁻¹
    β_ext = Se * f * 1e-6
    β_sca = Ss * f * 1e-6
    β_abs = max(β_ext - β_sca, 0.0)
    gbar  = Ss > 0 ? Sg / Ss : 0.0

    cacheable && (_BULK_CACHE[key] = (β_ext, β_sca, β_abs, gbar))
    return (β_ext * m.N, β_sca * m.N, β_abs * m.N, gbar)
end

"""
    bulk_optics(modes, λ; rh = 0.0, T_K = 298.15, nquad = 120)
        -> (β_ext, β_sca, β_abs, ssa, g)

External mixture of the modes: extinction and scattering coefficients
(m⁻¹) add, the single-scattering albedo is `β_sca/β_ext`, and the
asymmetry parameter is weighted by *scattering* coefficient. `rh` may be
a scalar (applied to every mode) — each mode grows according to its own
species' κ.

"External mixture" means every mode is its own particle population; the
alternative (internal mixing, where soot is a core inside a sulphate
shell) raises BC absorption by a factor 1.5-2 and is a well-known
structural uncertainty in aerosol forcing. This implementation is
external, which is the OPAC convention.
"""
function bulk_optics(modes::AbstractVector{<:AerosolMode}, λ::Real;
                     rh::Real = 0.0, T_K::Real = 298.15,
                     nquad::Int = NQUAD_DEFAULT)
    βe = 0.0; βs = 0.0; sg = 0.0
    for m in modes
        e, s, _, g = mode_optics(m, λ; rh = rh, T_K = T_K, nquad = nquad)
        βe += e; βs += s; sg += s * g
    end
    ssa = βe > 0 ? βs / βe : 0.0
    gbar = βs > 0 ? sg / βs : 0.0
    return (βe, βs, max(βe - βs, 0.0), clamp(ssa, 0.0, 1.0), gbar)
end

"""
    mass_extinction_coefficient(mode_or_modes, λ; rh = 0.0, kwargs...)
        -> m² g⁻¹

Mass extinction cross-section: `β_ext` divided by the **dry** mass
concentration. The corresponding absorption quantity (the mass
absorption cross-section, MAC) is the standard observational metric for
black carbon; for fresh soot the accepted value at 550 nm is
7.5 ± 1.2 m² g⁻¹ (Bond & Bergstrom 2006).
"""
function mass_extinction_coefficient(modes::AbstractVector{<:AerosolMode},
                                     λ::Real; kwargs...)
    βe, _, _, _, _ = bulk_optics(modes, λ; kwargs...)
    M = mode_mass(modes)                     # μg m⁻³
    M > 0 || return 0.0
    # β [m⁻¹] / M [μg m⁻³] = m² μg⁻¹ → ×1e6 → m² g⁻¹
    return βe / M * 1e6
end

mass_extinction_coefficient(m::LogNormalMode, λ::Real; kwargs...) =
    mass_extinction_coefficient([m], λ; kwargs...)

"""
    mass_absorption_coefficient(modes, λ; kwargs...) -> m² g⁻¹

As [`mass_extinction_coefficient`](@ref) but for absorption only.
"""
function mass_absorption_coefficient(modes::AbstractVector{<:AerosolMode},
                                     λ::Real; kwargs...)
    _, _, βa, _, _ = bulk_optics(modes, λ; kwargs...)
    M = mode_mass(modes)
    return M > 0 ? βa / M * 1e6 : 0.0
end

# =========================================================================
# 5. Band-mean optics and the Ångström exponent
# =========================================================================

"""
    band_wavelength(spec, b; sub = 1, nsub = 1) -> μm

Representative wavelength of sub-interval `sub` of `nsub` within band
`b`. The band limits are in wavenumber, and the sub-intervals are equal
in wavenumber (the natural measure for a band-integrated flux), so

```math
λ_{sub} = 10^4 / \\left[ν_{lo} + (sub - ½)\\,(ν_{hi}-ν_{lo})/n_{sub}\\right]
```

With `nsub = 1` this is the band-centre wavelength, which is what the
rest of the Era III stack uses.
"""
function band_wavelength(spec::SpectralGrid, b::Int; sub::Int = 1,
                         nsub::Int = 1)
    lo, hi = band_limits(spec, b)
    ν = lo + (sub - 0.5) * (hi - lo) / nsub
    return 1e4 / ν
end

"""
    AerosolBandOptics{T}

Band-mean bulk aerosol optical properties for one population at one
humidity.

  * `λ`     — representative wavelength per band, μm
  * `β_ext` — extinction coefficient per band, m⁻¹
  * `ssa`   — single-scattering albedo per band
  * `g`     — asymmetry parameter per band
  * `β_550` — monochromatic extinction at 550 nm, m⁻¹ (the AOD reference)
  * `rh`    — the relative humidity these were computed at
"""
struct AerosolBandOptics{T<:AbstractFloat}
    spec::SpectralGrid{T}
    λ::Vector{T}
    β_ext::Vector{T}
    ssa::Vector{T}
    g::Vector{T}
    β_550::T
    rh::T
end

Base.show(io::IO, o::AerosolBandOptics) =
    print(io, "AerosolBandOptics(", o.spec.kind, ", ", length(o.λ),
              " bands, RH=", round(o.rh; digits = 3),
              ", β₅₅₀=", round(o.β_550; sigdigits = 4), " m⁻¹)")

"""
    aerosol_band_optics(modes, spec; rh = 0.0, nsub = 2, nquad = 120)
        -> AerosolBandOptics

Integrate the mixture's Mie optics onto the bands of `spec`. Each band
is sampled at `nsub` wavelengths equally spaced in wavenumber and the
results averaged — `nsub = 1` (band centre) is noticeably biased for the
widest shortwave bands, where `β_ext` varies by tens of percent across
the band because of the `λ^{-α}` Ångström dependence.

Averaging is done on the *coefficients*, i.e. `β_ext` and `β_sca` are
averaged and `ω₀`, `g` recomputed from them, rather than averaging
`ω₀` and `g` directly. This is the same weighting `combine_optics!`
uses and keeps the band mean energetically consistent.
"""
function aerosol_band_optics(modes::AbstractVector{<:AerosolMode},
                             spec::SpectralGrid{T};
                             rh::Real = 0.0, T_K::Real = 298.15,
                             nsub::Int = 2,
                             nquad::Int = NQUAD_DEFAULT) where {T}
    nsub >= 1 || error("aerosol_band_optics: nsub must be ≥ 1")
    nb = nbands(spec)
    λv = zeros(T, nb); βe = zeros(T, nb); ssa = zeros(T, nb); gv = zeros(T, nb)
    for b in 1:nb
        e = 0.0; s = 0.0; sg = 0.0; λsum = 0.0
        for j in 1:nsub
            λ = band_wavelength(spec, b; sub = j, nsub = nsub)
            λsum += λ
            ej, sj, _, _, gj = bulk_optics(modes, λ; rh = rh, T_K = T_K,
                                           nquad = nquad)
            e += ej; s += sj; sg += sj * gj
        end
        λv[b]  = T(λsum / nsub)
        βe[b]  = T(e / nsub)
        ssa[b] = T(e > 0 ? clamp(s / e, 0.0, 1.0) : 0.0)
        gv[b]  = T(s > 0 ? sg / s : 0.0)
    end
    β550, _, _, _, _ = bulk_optics(modes, 0.55; rh = rh, T_K = T_K,
                                   nquad = nquad)
    return AerosolBandOptics{T}(spec, λv, βe, ssa, gv, T(β550), T(rh))
end

"""
    angstrom_exponent(optics::AerosolBandOptics, b1, b2) -> α

Ångström wavelength exponent between two bands,

```math
α = -\\frac{\\ln(β_{ext}(λ_1)/β_{ext}(λ_2))}{\\ln(λ_1/λ_2)}
```

so that `β_ext ∝ λ^{-α}` (Ångström 1929).

`α` is the classical remote-sensing discriminator of particle size,
because it measures how far the particles are from the two limits:

  * `α → 4` — Rayleigh (molecules, `x ≪ 1`)
  * `α ≈ 1.5–2.5` — fine/accumulation mode: urban pollution, biomass
    smoke, secondary sulphate
  * `α ≈ 0.5–1.2` — mixed continental
  * `α ≈ 0–0.5` — coarse mode: desert dust, sea salt; `α → 0` is the
    geometric-optics limit where `Q_ext → 2` independent of wavelength

A negative `α` is possible for a very coarse population and is not an
error.
"""
function angstrom_exponent(o::AerosolBandOptics, b1::Int, b2::Int)
    (o.β_ext[b1] > 0 && o.β_ext[b2] > 0) ||
        error("angstrom_exponent: zero extinction in band $b1 or $b2")
    o.λ[b1] != o.λ[b2] ||
        error("angstrom_exponent: bands $b1 and $b2 have the same wavelength")
    return -log(o.β_ext[b1] / o.β_ext[b2]) / log(o.λ[b1] / o.λ[b2])
end

"""
    angstrom_exponent(modes, λ1, λ2; rh = 0.0, kwargs...) -> α

Monochromatic form: the Ångström exponent of a mixture between two
wavelengths in μm (the AERONET reference pair is 440/870 nm).
"""
function angstrom_exponent(modes::AbstractVector{<:AerosolMode},
                           λ1::Real, λ2::Real; kwargs...)
    e1, _, _, _, _ = bulk_optics(modes, λ1; kwargs...)
    e2, _, _, _, _ = bulk_optics(modes, λ2; kwargs...)
    (e1 > 0 && e2 > 0) || error("angstrom_exponent: zero extinction")
    return -log(e1 / e2) / log(Float64(λ1) / Float64(λ2))
end

# --- humidity table -------------------------------------------------------

# Nodes are equally spaced in u = -ln(1-RH) up to RH = 0.995. In that
# variable the growth factor is nearly linear — GF ≈ (1 + κ RH/(1-RH))^⅓
# ⇒ ln GF ≈ (u - ln κ⁻¹)/3 at high RH — so linear interpolation of
# ln β_ext against u is accurate to well under a percent even across the
# steep 95-99 % range where β_ext changes by a factor of three.
const _RH_NODES = [0.0, 0.30, 0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90,
                   0.925, 0.95, 0.96, 0.97, 0.98, 0.985, 0.99, 0.995]

"""
    HygroscopicTable{T}

Band optics of one aerosol mixture tabulated on a humidity grid, so that
a column with a different RH in every layer costs one interpolation per
layer instead of one Mie integration per layer.

Fields `rh` (nodes), `u = -ln(1-rh)`, `β_ext`, `ssa`, `g`
(`n_rh × n_band` matrices), `β_550` (per node), and `λ` (per band).

If every mode of the mixture is hydrophobic (`κ < 10⁻⁶` — pure soot or
pure dust) the table collapses to a single node, because there is
nothing to interpolate.
"""
struct HygroscopicTable{T<:AbstractFloat}
    spec::SpectralGrid{T}
    rh::Vector{T}
    u::Vector{T}
    λ::Vector{T}
    β_ext::Matrix{T}
    ssa::Matrix{T}
    g::Matrix{T}
    β_550::Vector{T}
end

"""
    hygroscopic_table(modes, spec; rh_nodes = ..., nsub = 2, nquad = 120)
        -> HygroscopicTable

Build the humidity-dependent band-optics table for a mixture.
"""
function hygroscopic_table(modes::AbstractVector{<:AerosolMode},
                           spec::SpectralGrid{T};
                           rh_nodes = nothing, T_K::Real = 298.15,
                           nsub::Int = 2,
                           nquad::Int = NQUAD_DEFAULT) where {T}
    hydrophobic = all(m -> m.species.κ < 1e-6, modes)
    nodes = rh_nodes !== nothing ? collect(Float64, rh_nodes) :
            (hydrophobic ? [0.0] : copy(_RH_NODES))
    nr = length(nodes); nb = nbands(spec)
    βe = zeros(T, nr, nb); ss = zeros(T, nr, nb); gg = zeros(T, nr, nb)
    β550 = zeros(T, nr); λv = zeros(T, nb)
    for (i, rh) in enumerate(nodes)
        o = aerosol_band_optics(modes, spec; rh = rh, T_K = T_K,
                                nsub = nsub, nquad = nquad)
        i == 1 && (λv .= o.λ)
        @views βe[i, :] .= o.β_ext
        @views ss[i, :] .= o.ssa
        @views gg[i, :] .= o.g
        β550[i] = o.β_550
    end
    u = T[-log(max(1 - r, 1e-4)) for r in nodes]
    return HygroscopicTable{T}(spec, T.(nodes), u, λv, βe, ss, gg, β550)
end

"""
    interpolate_rh(tab, rh) -> (β_ext::Vector, ssa::Vector, g::Vector, β_550)

Band optics at an arbitrary relative humidity, by linear interpolation
in `u = -ln(1-RH)` (logarithmic for `β_ext`, linear for `ssa` and `g`).
Humidities above the top node are clamped, not extrapolated: beyond
RH ≈ 0.995 the growth is limited by kinetics and by the finite water
supply, not by equilibrium thermodynamics, so extrapolating the
equilibrium curve would be actively wrong.
"""
function interpolate_rh(tab::HygroscopicTable{T}, rh::Real) where {T}
    nr = length(tab.rh)
    nb = size(tab.β_ext, 2)
    if nr == 1
        return (copy(@view tab.β_ext[1, :]), copy(@view tab.ssa[1, :]),
                copy(@view tab.g[1, :]), tab.β_550[1])
    end
    r = clamp(Float64(rh), Float64(tab.rh[1]), Float64(tab.rh[end]))
    u = -log(max(1 - r, 1e-4))
    j = searchsortedlast(tab.u, u)
    j = clamp(j, 1, nr - 1)
    du = tab.u[j + 1] - tab.u[j]
    t = du > 0 ? T((u - tab.u[j]) / du) : zero(T)
    βe = Vector{T}(undef, nb); ss = Vector{T}(undef, nb); gv = Vector{T}(undef, nb)
    @inbounds for b in 1:nb
        b0 = tab.β_ext[j, b]; b1 = tab.β_ext[j + 1, b]
        βe[b] = (b0 > 0 && b1 > 0) ? exp((1 - t) * log(b0) + t * log(b1)) :
                                     (1 - t) * b0 + t * b1
        ss[b] = clamp((1 - t) * tab.ssa[j, b] + t * tab.ssa[j + 1, b],
                      zero(T), one(T))
        gv[b] = (1 - t) * tab.g[j, b] + t * tab.g[j + 1, b]
    end
    c0 = tab.β_550[j]; c1 = tab.β_550[j + 1]
    β550 = (c0 > 0 && c1 > 0) ? exp((1 - t) * log(c0) + t * log(c1)) :
                                (1 - t) * c0 + t * c1
    return (βe, ss, gv, T(β550))
end

# =========================================================================
# 6. Standard aerosol mixtures
# =========================================================================

# OPAC-style mixtures (Hess, Koepke & Schult 1998, Table 1). OPAC's
# "water-soluble" component is a lumped sulphate + organic + nitrate
# population; here it is resolved into those three species explicitly
# (with OPAC's `waso` size distribution r_g = 0.0212 μm, σ_g = 2.24),
# which is what lets the module answer composition questions — e.g. how
# much of an urban aerosol's absorption is soot rather than brown carbon.
#
# Number concentrations are in cm⁻³. The soot mode uses r_g = 0.0118 μm,
# σ_g = 2.0 (OPAC `soot`); the sizes chosen reproduce a bulk ω₀ ≈ 0.22
# and MAC ≈ 9 m² g⁻¹ at 550 nm, consistent with the observational
# consensus for fresh, externally-mixed black carbon (Bond & Bergstrom
# 2006: ω₀ ≈ 0.2-0.3, MAC = 7.5 ± 1.2 m² g⁻¹).

const _WASO_RG = 0.0212
const _WASO_SG = 2.24
const _SOOT_RG = 0.0118
const _SOOT_SG = 2.00

function _waso(n_sulph, n_oc, n_nit)
    ms = LogNormalMode{Float64}[]
    n_sulph > 0 && push!(ms, lognormal_mode(:sulphate, n_sulph, _WASO_RG, _WASO_SG))
    n_oc    > 0 && push!(ms, lognormal_mode(:organic_carbon, n_oc, _WASO_RG, _WASO_SG))
    n_nit   > 0 && push!(ms, lognormal_mode(:ammonium_nitrate, n_nit, _WASO_RG, _WASO_SG))
    return ms
end

"""
    continental_clean() -> Vector{LogNormalMode}

Remote continental background: water-soluble fine mode plus a trace of
insoluble mineral dust, no soot. OPAC `continental clean`.
"""
continental_clean() = vcat(
    _waso(1400.0, 1000.0, 200.0),
    [lognormal_mode(:mineral_dust, 0.15, 0.471, 2.51)])

"""
    continental_polluted() -> Vector{LogNormalMode}

Polluted continental boundary layer: heavy water-soluble loading and
3.4 × 10⁴ cm⁻³ of soot. OPAC `continental polluted`.
"""
continental_polluted() = vcat(
    _waso(7000.0, 6000.0, 2700.0),
    [lognormal_mode(:black_carbon, 34300.0, _SOOT_RG, _SOOT_SG),
     lognormal_mode(:mineral_dust, 0.6, 0.471, 2.51)])

"""
    urban() -> Vector{LogNormalMode}

Urban aerosol: 1.3 × 10⁵ cm⁻³ of soot on top of a large secondary
inorganic/organic fine mode. OPAC `urban`. Strongly absorbing
(`ω₀ ≈ 0.65` at 550 nm) and strongly fine-mode (`α ≈ 1.5`).
"""
urban() = vcat(
    _waso(12000.0, 11000.0, 5000.0),
    [lognormal_mode(:black_carbon, 130000.0, _SOOT_RG, _SOOT_SG),
     lognormal_mode(:mineral_dust, 1.5, 0.471, 2.51)])

"""
    desert() -> Vector{LogNormalMode}

Desert dust: the three OPAC mineral modes (nucleation 0.07 μm,
accumulation 0.39 μm, coarse 1.9 μm) with a modest water-soluble
background. Coarse-mode dominated, so `α ≈ 0.1-0.3`, and the only
mixture here with a first-order longwave effect.
"""
desert() = vcat(
    _waso(1200.0, 600.0, 200.0),
    [lognormal_mode(:mineral_dust, 269.5, 0.07, 1.95),
     lognormal_mode(:mineral_dust, 30.5, 0.39, 2.00),
     lognormal_mode(:mineral_dust, 0.142, 1.90, 2.15)])

"""
    maritime_clean() -> Vector{LogNormalMode}

Clean marine boundary layer: accumulation and coarse sea-salt modes plus
a small non-sea-salt sulphate background. Nearly conservative scattering
(`ω₀ > 0.98`) and very strongly hygroscopic.
"""
maritime_clean() = vcat(
    _waso(1000.0, 400.0, 100.0),
    [lognormal_mode(:sea_salt, 20.0, 0.209, 2.03),
     lognormal_mode(:sea_salt, 3.2e-3, 1.75, 2.03)])

"""
    maritime_polluted() -> Vector{LogNormalMode}

Marine boundary layer under a continental outflow plume: the clean
marine population plus 5.2 × 10³ cm⁻³ of soot. OPAC
`maritime polluted`.
"""
maritime_polluted() = vcat(
    _waso(2200.0, 1200.0, 400.0),
    [lognormal_mode(:sea_salt, 20.0, 0.209, 2.03),
     lognormal_mode(:sea_salt, 3.2e-3, 1.75, 2.03),
     lognormal_mode(:black_carbon, 5180.0, _SOOT_RG, _SOOT_SG)])

"""
    antarctic() -> Vector{LogNormalMode}

Antarctic background: a sulphate droplet mode, a trace of sea salt and a
trace of transported mineral dust. The cleanest tropospheric aerosol on
the planet — total number ~43 cm⁻³, three to four orders of magnitude
below urban.
"""
antarctic() = [lognormal_mode(:sulphate, 42.9, 0.0695, 2.03),
               lognormal_mode(:sea_salt, 0.47, 0.209, 2.03),
               lognormal_mode(:mineral_dust, 0.0053, 0.50, 2.20)]

const AEROSOL_MIXTURE_NAMES = (:continental_clean, :continental_polluted,
                               :urban, :desert, :maritime_clean,
                               :maritime_polluted, :antarctic)

"""
    aerosol_mixture(name::Symbol; scale = 1.0) -> Vector{LogNormalMode}

Look up one of the standard mixtures by name (see
[`AEROSOL_MIXTURE_NAMES`](@ref)) and optionally multiply every number
concentration by `scale`. Because Mie optics are linear in number
concentration, `scale` rescales `β_ext` exactly and leaves `ω₀` and `g`
untouched.
"""
function aerosol_mixture(name::Symbol; scale::Real = 1.0)
    modes = name === :continental_clean     ? continental_clean()     :
            name === :continental_polluted  ? continental_polluted()  :
            name === :urban                 ? urban()                 :
            name === :desert                ? desert()                :
            name === :maritime_clean        ? maritime_clean()        :
            name === :maritime_polluted     ? maritime_polluted()     :
            name === :antarctic             ? antarctic()             :
            error("aerosol_mixture: unknown mixture :$name. Known: " *
                  string(AEROSOL_MIXTURE_NAMES))
    scale == 1 && return modes
    return [lognormal_mode(m.species, m.N * scale, m.r_g, m.σ_g)
            for m in modes]
end

# =========================================================================
# 7. Column aerosol profile
# =========================================================================

"""
    AerosolProfile{T} <: AbstractAerosolOptics

A vertically resolved aerosol population on a radiation column.

  * `modes`  — the reference mixture (number concentrations as given)
  * `scale`  — per-layer multiplier on those number concentrations
  * `rh`     — per-layer relative humidity, [0,1]
  * `dz`     — layer thickness, m
  * `z`      — layer-centre height, m
  * `table`  — the humidity-interpolated band-optics table
  * `τ`, `ssa`, `g` — per-layer, per-band optical properties, already
    multiplied by `scale·dz` so `τ[k,b]` is the layer optical depth

The profile is a `AbstractAerosolOptics`, so it plugs into the Era III
`compute_optics!` contract and merges with gas and cloud optics.

Vertical shape: an exponentially decaying boundary-layer term plus an
optional elevated layer (a Saharan air layer, a biomass-burning plume or
a volcanic/stratospheric sheet), the whole normalised to a prescribed
550 nm aerosol optical depth. Prescribing AOD rather than mass is the
operational convention because AOD is what is actually measured — by
sun photometers on the ground and by satellite from space.
"""
struct AerosolProfile{T<:AbstractFloat} <: AbstractAerosolOptics
    modes::Vector{LogNormalMode{T}}
    scale::Vector{T}
    rh::Vector{T}
    dz::Vector{T}
    z::Vector{T}
    spec::SpectralGrid{T}
    table::HygroscopicTable{T}
    τ::Matrix{T}      # (nz, nband)
    ssa::Matrix{T}
    g::Matrix{T}
    β_550::Vector{T}  # per-layer 550 nm extinction, m⁻¹ (with `scale`)
end

nlayers(p::AerosolProfile) = length(p.scale)

Base.show(io::IO, p::AerosolProfile{T}) where {T} =
    print(io, "AerosolProfile{$T}(", length(p.modes), " modes, ",
              nlayers(p), " layers, ", p.spec.kind,
              ", AOD₅₅₀=", round(aerosol_optical_depth_550(p); digits = 4), ")")

"""
    relative_humidity_profile(col) -> Vector

Relative humidity with respect to liquid water, diagnosed from an
`AtmosphericColumn`'s water-vapour volume mixing ratio, temperature and
pressure. Saturation vapour pressure follows Bolton (1980):

```math
e_s(T) = 611.2\\,\\exp\\!\\left[\\frac{17.67\\,(T-273.15)}{T-29.65}\\right]\\ \\mathrm{Pa}
```

accurate to 0.1 % over -35…35 °C. Returns zeros if the column carries no
`:h2o`.
"""
function relative_humidity_profile(col::AtmosphericColumn{T}) where {T}
    nz = length(col.p_lay)
    out = zeros(T, nz)
    haskey(col.vmr, :h2o) || return out
    q = col.vmr[:h2o]
    @inbounds for k in 1:nz
        Tk = col.T_lay[k]
        Tk > 0 || continue
        es = 611.2 * exp(17.67 * (Tk - 273.15) / (Tk - 29.65))
        e = q[k] / (1 + q[k]) * col.p_lay[k]     # vmr → partial pressure
        out[k] = T(clamp(e / es, 0.0, 0.999))
    end
    return out
end

"""
    aerosol_profile(mixture, nz, z; aod_550 = 0.2, scale_height = 2000.0,
                    rh = 0.0, spec = shortwave_bands(),
                    elevated_layer = nothing, elevated_fraction = 0.0,
                    dz = nothing, nsub = 2, nquad = 120) -> AerosolProfile

Build a column aerosol profile.

  * `mixture` — a `Vector{<:AerosolMode}` (e.g. `urban()`) or a symbol
    accepted by [`aerosol_mixture`](@ref)
  * `nz`      — number of layers
  * `z`       — layer-centre heights (m), a vector of length `nz`, or a
    scalar domain top for a uniform grid
  * `aod_550` — target column optical depth at 550 nm; the vertical
    shape is rescaled so the integral matches this *exactly*
  * `scale_height` — e-folding depth `H` of the boundary-layer term,
    `f(z) = exp(-z/H)`; 1-2 km is typical for a well-mixed continental
    boundary layer, 0.5-1 km for marine
  * `elevated_layer` — `(z_bottom, z_top)` in m for an additional
    uniform slab, carrying `elevated_fraction` of the total 550 nm AOD
  * `rh` — scalar or per-layer relative humidity; pass
    `relative_humidity_profile(col)` to drive it from the thermodynamic
    state

The exponential shape is the observed one: aerosol is emitted at the
surface and removed by wet deposition, and turbulent mixing against that
gives a near-exponential decay with a scale height close to the
boundary-layer depth. The elevated-layer option exists because the two
optically most important aerosol features on Earth — the Saharan air
layer and stratospheric volcanic sulphate — are *not* surface-attached,
and their radiative effect (especially over cloud) depends on being
placed correctly in the vertical.
"""
function aerosol_profile(mixture, nz::Int, z;
                         aod_550::Real = 0.2,
                         scale_height::Real = 2000.0,
                         rh = 0.0,
                         spec::SpectralGrid{T} = shortwave_bands(Float64),
                         elevated_layer = nothing,
                         elevated_fraction::Real = 0.0,
                         dz = nothing,
                         T_K::Real = 298.15,
                         nsub::Int = 2,
                         nquad::Int = NQUAD_DEFAULT) where {T}
    nz >= 1 || error("aerosol_profile: need at least one layer")
    aod_550 >= 0 || error("aerosol_profile: aod_550 must be non-negative")
    scale_height > 0 || error("aerosol_profile: scale_height must be positive")

    modes0 = mixture isa Symbol ? aerosol_mixture(mixture) :
             collect(LogNormalMode{Float64}, mixture)
    modes = LogNormalMode{T}[lognormal_mode(m.species, m.N, m.r_g, m.σ_g; T = T)
                             for m in modes0]

    # --- vertical grid
    zc = if z isa Real
        dzu = Float64(z) / nz
        T[T((k - 0.5) * dzu) for k in 1:nz]
    else
        length(z) == nz ||
            error("aerosol_profile: z has $(length(z)) entries, expected nz=$nz")
        collect(T, z)
    end
    dzv = if dz !== nothing
        dz isa Real ? fill(T(dz), nz) : collect(T, dz)
    elseif z isa Real
        fill(T(Float64(z) / nz), nz)
    else
        d = Vector{T}(undef, nz)
        if nz == 1
            d[1] = T(2 * zc[1])
        else
            d[1] = T(zc[2] - zc[1])
            for k in 2:nz-1
                d[k] = T((zc[k + 1] - zc[k - 1]) / 2)
            end
            d[nz] = T(zc[nz] - zc[nz - 1])
        end
        d
    end
    all(>(0), dzv) || error("aerosol_profile: layer thicknesses must be positive")

    rhv = rh isa Real ? fill(T(clamp(rh, 0, 0.999)), nz) :
          T[T(clamp(r, 0, 0.999)) for r in rh]
    length(rhv) == nz ||
        error("aerosol_profile: rh has $(length(rhv)) entries, expected $nz")

    tab = hygroscopic_table(modes, spec; T_K = T_K, nsub = nsub, nquad = nquad)

    # --- per-layer band optics at the layer's own humidity
    nb = nbands(spec)
    βe = Matrix{T}(undef, nz, nb)
    ssa = Matrix{T}(undef, nz, nb)
    gv = Matrix{T}(undef, nz, nb)
    β550 = Vector{T}(undef, nz)
    for k in 1:nz
        e, s, g, b5 = interpolate_rh(tab, rhv[k])
        @views βe[k, :] .= e
        @views ssa[k, :] .= s
        @views gv[k, :] .= g
        β550[k] = b5
    end

    # --- vertical shape, normalised to the requested 550 nm AOD
    H = T(scale_height)
    w_bl = T[exp(-zc[k] / H) for k in 1:nz]
    fe = T(clamp(elevated_fraction, 0, 1))
    w_el = zeros(T, nz)
    if elevated_layer !== nothing && fe > 0
        zb, zt = Float64(elevated_layer[1]), Float64(elevated_layer[2])
        zt > zb || error("aerosol_profile: elevated_layer must have z_top > z_bottom")
        for k in 1:nz
            zlo = zc[k] - dzv[k] / 2; zhi = zc[k] + dzv[k] / 2
            ov = min(zhi, zt) - max(zlo, zb)
            w_el[k] = ov > 0 ? T(ov / dzv[k]) : zero(T)
        end
        sum(w_el) > 0 ||
            error("aerosol_profile: elevated_layer does not intersect the grid")
    else
        fe = zero(T)
    end

    τ_bl = sum(β550[k] * w_bl[k] * dzv[k] for k in 1:nz)
    τ_el = fe > 0 ? sum(β550[k] * w_el[k] * dzv[k] for k in 1:nz) : zero(T)
    scale = zeros(T, nz)
    if aod_550 > 0
        τ_bl > 0 || error("aerosol_profile: boundary-layer shape has zero optical depth")
        a_bl = T((1 - fe) * aod_550) / τ_bl
        @inbounds for k in 1:nz
            scale[k] = a_bl * w_bl[k]
        end
        if fe > 0
            a_el = T(fe * aod_550) / τ_el
            @inbounds for k in 1:nz
                scale[k] += a_el * w_el[k]
            end
        end
    end

    τ = Matrix{T}(undef, nz, nb)
    @inbounds for b in 1:nb, k in 1:nz
        τ[k, b] = βe[k, b] * scale[k] * dzv[k]
    end
    @inbounds for k in 1:nz
        β550[k] *= scale[k]
    end
    return AerosolProfile{T}(modes, scale, rhv, dzv, zc, spec, tab,
                             τ, ssa, gv, β550)
end

"""
    rebuild_bands(profile, spec) -> AerosolProfile

Re-evaluate a profile's optics on a different spectral grid (typically
to go from the shortwave to the longwave grid) keeping the vertical
distribution — and hence the 550 nm AOD — exactly as it is.
"""
function rebuild_bands(p::AerosolProfile{T}, spec::SpectralGrid{T};
                       nsub::Int = 2, nquad::Int = NQUAD_DEFAULT) where {T}
    nz = nlayers(p)
    tab = hygroscopic_table(p.modes, spec; nsub = nsub, nquad = nquad)
    nb = nbands(spec)
    τ = Matrix{T}(undef, nz, nb)
    ssa = Matrix{T}(undef, nz, nb)
    gv = Matrix{T}(undef, nz, nb)
    for k in 1:nz
        e, s, g, _ = interpolate_rh(tab, p.rh[k])
        @inbounds for b in 1:nb
            τ[k, b] = e[b] * p.scale[k] * p.dz[k]
            ssa[k, b] = s[b]
            gv[k, b] = g[b]
        end
    end
    return AerosolProfile{T}(p.modes, copy(p.scale), copy(p.rh), copy(p.dz),
                             copy(p.z), spec, tab, τ, ssa, gv, copy(p.β_550))
end

"""
    aerosol_optical_depth(profile, band) -> τ

Column aerosol optical depth in a spectral band.
"""
aerosol_optical_depth(p::AerosolProfile, band::Int) = sum(@view p.τ[:, band])

"""
    aerosol_optical_depth(profile) -> Vector

Column AOD in every band of the profile's spectral grid.
"""
aerosol_optical_depth(p::AerosolProfile{T}) where {T} =
    T[sum(@view p.τ[:, b]) for b in 1:size(p.τ, 2)]

"""
    aerosol_optical_depth_550(profile) -> τ₅₅₀

Column aerosol optical depth at 550 nm — the quantity a sun photometer
measures and the quantity [`aerosol_profile`](@ref) normalises to.
"""
aerosol_optical_depth_550(p::AerosolProfile) =
    sum(p.β_550[k] * p.dz[k] for k in 1:nlayers(p))

"""
    aerosol_extinction_profile(profile; band = nothing) -> Vector

Per-layer extinction coefficient in m⁻¹, at 550 nm by default or in the
given band.
"""
function aerosol_extinction_profile(p::AerosolProfile{T};
                                    band::Union{Nothing, Int} = nothing) where {T}
    band === nothing && return copy(p.β_550)
    return T[p.τ[k, band] / p.dz[k] for k in 1:nlayers(p)]
end

"""
    angstrom_exponent(profile, band1, band2) -> α

Ångström exponent of a column profile, from the *column* optical depths
in the two bands. Identical to the coefficient form when the humidity is
uniform; when it is not, this is the correctly AOD-weighted column
value, which is what a sun photometer retrieves.
"""
function angstrom_exponent(p::AerosolProfile, b1::Int, b2::Int)
    τ1 = aerosol_optical_depth(p, b1)
    τ2 = aerosol_optical_depth(p, b2)
    (τ1 > 0 && τ2 > 0) ||
        error("angstrom_exponent: zero optical depth in band $b1 or $b2")
    λ1 = p.table.λ[b1]; λ2 = p.table.λ[b2]
    return -log(τ1 / τ2) / log(λ1 / λ2)
end

# =========================================================================
# 8. The `compute_optics!` contract
# =========================================================================

"""
    compute_optics!(optics, aerosol::AerosolProfile, col, spec; add = true)

Era III optics contract for aerosol. Writes the profile's layer optical
depth, single-scattering albedo and asymmetry parameter into `optics`
for every g-point.

With `add = true` (the default) the aerosol is **merged into** whatever
`optics` already holds, using exactly the algebra of
`RadiationCore.combine_optics!`:

```math
τ = τ_g + τ_a,\\qquad
ω = \\frac{τ_gω_g + τ_aω_a}{τ},\\qquad
g = \\frac{τ_gω_g g_g + τ_aω_a g_a}{τ_gω_g + τ_aω_a}
```

so the usual call sequence is

```julia
compute_optics!(o, gas_optics, col)          # fills
compute_optics!(o, cloud_optics_buffer, col) # fills a second buffer
combine_optics!(o, cloud_buffer)
compute_optics!(o, aerosol_profile, col)     # merges in place
```

Pass `add = false` to overwrite instead (the behaviour of the gas and
cloud methods), which is what you want when filling a dedicated aerosol
buffer for a later `combine_optics!`.

Within a band all g-points get the same aerosol properties: aerosol
extinction varies smoothly across a band, so it carries no correlated-k
structure of its own. That is the standard treatment (RRTMGP does the
same) and it is why the aerosol table is per *band* rather than per
g-point.
"""
function compute_optics!(optics::OpticalProperties{T},
                         a::AerosolProfile{T},
                         col::AtmosphericColumn{T},
                         spec::SpectralGrid{T};
                         add::Bool = true) where {T}
    _check_spec(a, spec)
    return _apply_aerosol!(optics, a, spec, add)
end

function compute_optics!(optics::OpticalProperties{T},
                         a::AerosolProfile{T},
                         col::AtmosphericColumn{T};
                         add::Bool = true) where {T}
    return _apply_aerosol!(optics, a, a.spec, add)
end

"""
    compute_optics!(optics, aerosol::AerosolProfile; add = true)

Column-free form, for when there is no thermodynamic state to consult —
the profile already carries its own humidity and geometry.
"""
compute_optics!(optics::OpticalProperties{T}, a::AerosolProfile{T};
                add::Bool = true) where {T} =
    _apply_aerosol!(optics, a, a.spec, add)

function _check_spec(a::AerosolProfile, spec::SpectralGrid)
    spec === a.spec && return nothing
    (spec.kind === a.spec.kind && nbands(spec) == nbands(a.spec) &&
     spec.band_lo == a.spec.band_lo && spec.band_hi == a.spec.band_hi) ||
        error("compute_optics!: the aerosol profile was built on a " *
              "$(a.spec.kind) grid with $(nbands(a.spec)) bands and cannot " *
              "be evaluated on this $(spec.kind) grid with $(nbands(spec)) " *
              "bands. Use `rebuild_bands(profile, spec)`.")
    return nothing
end

function _apply_aerosol!(optics::OpticalProperties{T}, a::AerosolProfile{T},
                         spec::SpectralGrid{T}, add::Bool) where {T}
    nz = nlayers(a)
    size(optics.τ, 1) == nz ||
        error("compute_optics!: optics has $(size(optics.τ, 1)) layers, " *
              "aerosol profile has $nz")
    ngpoints(spec) == size(optics.τ, ndims(optics.τ)) ||
        error("compute_optics!: g-point count mismatch")
    add || (fill!(optics.τ, 0); fill!(optics.ssa, 0); fill!(optics.g, 0))
    @inbounds for b in 1:nbands(spec)
        for gp in gpoint_range(spec, b)
            for k in 1:nz
                τa = a.τ[k, b]
                τa > 0 || continue
                ωa = a.ssa[k, b]; ga = a.g[k, b]
                if !add
                    optics.τ[k, gp] = τa
                    optics.ssa[k, gp] = ωa
                    optics.g[k, gp] = ga
                    continue
                end
                τ0 = optics.τ[k, gp]; ω0 = optics.ssa[k, gp]; g0 = optics.g[k, gp]
                τn = τ0 + τa
                s0 = τ0 * ω0; sa = τa * ωa
                sn = s0 + sa
                optics.τ[k, gp] = τn
                optics.ssa[k, gp] = τn > 0 ? sn / τn : zero(T)
                optics.g[k, gp] = sn > 0 ? (s0 * g0 + sa * ga) / sn : zero(T)
            end
        end
    end
    return optics
end

# =========================================================================
# 9. Radiative-effect diagnostics
# =========================================================================

# The two-stream solver lives in `TwoStream`, which is included *after*
# this module (it depends on the same optics contract but is loaded
# later). Resolving it through the parent module at call time keeps the
# include order free of a cycle; the cost is one dynamic `getfield` per
# forcing calculation, which is irrelevant next to the RT solve itself.
function _twostream_module()
    parent = parentmodule(@__MODULE__)
    isdefined(parent, :TwoStream) ||
        error("aerosol_radiative_forcing: the TwoStream module is not " *
              "available; pass `solver = ...` explicitly.")
    return getfield(parent, :TwoStream)
end

_default_solver(spec::SpectralGrid) = _twostream_module().two_stream_solver(spec)

"""
    aerosol_radiative_forcing(profile, col; μ₀ = 0.5, sfc_albedo = 0.05,
                              gas = nothing, rayleigh = true,
                              solver = nothing)
        -> (toa, surface, atmosphere, ...)

Shortwave **direct aerosol radiative effect**: the difference in net
flux between an atmosphere with the aerosol and the same atmosphere
without it, solved twice with the Phase 33 two-stream solver.

Returns a `NamedTuple`

  * `toa`        — `ΔF_net` at the model top, W m⁻². Negative = the
    planet loses energy = cooling.
  * `surface`    — `ΔF_net` at the surface, W m⁻². Negative = dimming.
  * `atmosphere` — `toa - surface`, the change in energy *absorbed
    within the column*. Positive = the aerosol layer heats.
  * `net_toa_clean`, `net_toa_aerosol`, `net_sfc_clean`,
    `net_sfc_aerosol` — the four raw fluxes, for diagnosis
  * `aod_550`    — the column 550 nm optical depth, for normalisation
  * `forcing_efficiency` — `toa / aod_550`, W m⁻² per unit AOD, the
    quantity usually quoted for intercomparison

The identity `ΔF_atm = ΔF_toa - ΔF_sfc` is exact by construction and is
the cleanest way to separate the two aerosol effects that a single TOA
number conflates:

  * a **scattering** aerosol (sulphate, sea salt) over a dark surface
    reflects sunlight back to space: `ΔF_toa < 0`, `ΔF_sfc < 0`,
    `ΔF_atm ≈ 0`. It cools the planet and the surface and does nothing
    to the atmosphere.
  * an **absorbing** aerosol (black carbon) also dims the surface
    strongly, but the energy it removes from the beam stays in the
    column: `ΔF_atm > 0`. The TOA sign then depends on the surface
    albedo — see [`critical_surface_albedo`](@ref).

`gas` may be a concrete `AbstractGasOptics` (Phase 32); if it is
`nothing` the clean atmosphere is molecular-scattering only
(`rayleigh = true`) or completely transparent (`rayleigh = false`). The
comparison is differential, so the choice mostly affects the magnitude,
not the sign — but Rayleigh scattering matters for the critical albedo
because it provides a floor of atmospheric backscatter.
"""
function aerosol_radiative_forcing(p::AerosolProfile{T},
                                   col::AtmosphericColumn{T};
                                   μ₀::Real = 0.5,
                                   sfc_albedo = 0.05,
                                   gas = nothing,
                                   rayleigh::Bool = true,
                                   solver = nothing) where {T}
    spec = p.spec
    spec.kind === :shortwave ||
        error("aerosol_radiative_forcing: requires a shortwave profile " *
              "(got $(spec.kind)); use `rebuild_bands` to move grids.")
    nz = nlayers(p)
    length(col.p_lay) == nz ||
        error("aerosol_radiative_forcing: column has $(length(col.p_lay)) " *
              "layers, aerosol profile has $nz")
    slv = solver === nothing ? _default_solver(spec) : solver

    clean = allocate_optics(T, (nz,), spec)
    if gas !== nothing
        compute_optics!(clean, gas, col)
    elseif rayleigh
        rayleigh_optics!(clean, spec, col)
    end

    dirty = allocate_optics(T, (nz,), spec)
    copyto!(dirty.τ, clean.τ)
    copyto!(dirty.ssa, clean.ssa)
    copyto!(dirty.g, clean.g)
    _apply_aerosol!(dirty, p, spec, true)

    nlev = nz + 1
    f_clean = allocate_fluxes(T, (), nlev)
    f_dirty = allocate_fluxes(T, (), nlev)
    RadiationCore.solve_rt!(f_clean, slv, clean, col; μ₀ = μ₀,
                            sfc_albedo_direct = sfc_albedo)
    RadiationCore.solve_rt!(f_dirty, slv, dirty, col; μ₀ = μ₀,
                            sfc_albedo_direct = sfc_albedo)

    toa_c = f_clean.flux_net[nlev]; toa_a = f_dirty.flux_net[nlev]
    sfc_c = f_clean.flux_net[1];    sfc_a = f_dirty.flux_net[1]
    Δtoa = toa_a - toa_c
    Δsfc = sfc_a - sfc_c
    aod = aerosol_optical_depth_550(p)
    return (toa = Δtoa, surface = Δsfc, atmosphere = Δtoa - Δsfc,
            net_toa_clean = toa_c, net_toa_aerosol = toa_a,
            net_sfc_clean = sfc_c, net_sfc_aerosol = sfc_a,
            aod_550 = aod,
            forcing_efficiency = aod > 0 ? Δtoa / aod : zero(T))
end

"""
    critical_surface_albedo(profile, col; μ₀ = 0.5, tol = 1e-4, kwargs...)
        -> α_c

Surface albedo at which the TOA shortwave aerosol forcing changes sign.

For an aerosol with single-scattering albedo `ω₀` and upscatter fraction
`β`, the single-scattering TOA forcing is (Chylek & Coakley 1974;
Haywood & Shine 1995)

```math
ΔF_{TOA} \\propto (1-α_s)^2\\,2β\\,ω_0 - 4α_s(1-ω_0)
```

which vanishes at a critical albedo `α_c`. Below `α_c` the surface is
dark, the aerosol's backscatter dominates and the aerosol cools; above
`α_c` the surface already returns most of the sunlight, the aerosol's
absorption dominates, and the same aerosol *warms*. This is why black
carbon deposited over snow, ice or bright cloud is so much more potent
than the same soot over ocean.

This function finds `α_c` numerically from the full multiple-scattering
solution rather than from the single-scattering formula, by bisecting
[`aerosol_radiative_forcing`](@ref)`.toa` on `α ∈ [0, 0.99]`. Returns
`NaN` if the forcing has the same sign at both ends — which is the
correct answer for a purely scattering aerosol, whose TOA forcing is
negative at every albedo below 1.
"""
function critical_surface_albedo(p::AerosolProfile{T},
                                 col::AtmosphericColumn{T};
                                 μ₀::Real = 0.5, tol::Real = 1e-4,
                                 α_min::Real = 0.0, α_max::Real = 0.99,
                                 kwargs...) where {T}
    f(α) = aerosol_radiative_forcing(p, col; μ₀ = μ₀, sfc_albedo = α,
                                     kwargs...).toa
    lo = Float64(α_min); hi = Float64(α_max)
    flo = f(lo); fhi = f(hi)
    flo == 0 && return T(lo)
    fhi == 0 && return T(hi)
    sign(flo) == sign(fhi) && return T(NaN)
    for _ in 1:200
        mid = 0.5 * (lo + hi)
        fm = f(mid)
        if sign(fm) == sign(flo)
            lo = mid; flo = fm
        else
            hi = mid; fhi = fm
        end
        (hi - lo) < tol && break
    end
    return T(0.5 * (lo + hi))
end

"""
    aerosol_heating_rate(profile, col; μ₀ = 0.5, sfc_albedo = 0.05,
                         kwargs...) -> Vector

Per-layer shortwave heating rate *difference* caused by the aerosol,
K day⁻¹. A black-carbon plume produces a sharply peaked positive
anomaly at its own altitude — this is the direct-effect analogue of the
"semi-direct" cloud response, and the vertical structure is what
determines whether the plume stabilises or destabilises the layer below.
"""
function aerosol_heating_rate(p::AerosolProfile{T}, col::AtmosphericColumn{T};
                              μ₀::Real = 0.5, sfc_albedo = 0.05,
                              gas = nothing, rayleigh::Bool = true,
                              solver = nothing) where {T}
    spec = p.spec
    spec.kind === :shortwave ||
        error("aerosol_heating_rate: requires a shortwave profile")
    nz = nlayers(p); nlev = nz + 1
    slv = solver === nothing ? _default_solver(spec) : solver
    clean = allocate_optics(T, (nz,), spec)
    if gas !== nothing
        compute_optics!(clean, gas, col)
    elseif rayleigh
        rayleigh_optics!(clean, spec, col)
    end
    dirty = allocate_optics(T, (nz,), spec)
    copyto!(dirty.τ, clean.τ); copyto!(dirty.ssa, clean.ssa)
    copyto!(dirty.g, clean.g)
    _apply_aerosol!(dirty, p, spec, true)

    f_clean = allocate_fluxes(T, (), nlev)
    f_dirty = allocate_fluxes(T, (), nlev)
    RadiationCore.solve_rt!(f_clean, slv, clean, col; μ₀ = μ₀,
                            sfc_albedo_direct = sfc_albedo)
    RadiationCore.solve_rt!(f_dirty, slv, dirty, col; μ₀ = μ₀,
                            sfc_albedo_direct = sfc_albedo)
    hc = zeros(T, nz); hd = zeros(T, nz)
    RadiationCore.heating_rate!(hc, f_clean, col.dz, col.ρ)
    RadiationCore.heating_rate!(hd, f_dirty, col.dz, col.ρ)
    return T[(hd[k] - hc[k]) * 86400 for k in 1:nz]
end

end # module
