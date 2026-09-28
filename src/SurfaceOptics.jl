module SurfaceOptics

# Phase 36 — Spectral surface albedo and BRDF.
#
# The lower boundary condition is where a radiation scheme most often goes
# quietly wrong. A grey albedo of 0.2 gets the broadband number roughly
# right for a mid-latitude city and is badly wrong everywhere it matters:
# it puts half the reflected energy in the wrong part of the spectrum, it
# cannot reproduce the near-infrared plateau that makes a vegetated
# surface reflect four times more NIR than visible light, and it has no
# angular structure at all, so a low sun over water reflects the same 20%
# that a noon sun does when the true value swings from 3% to 35%.
#
# This module therefore builds the surface from three independent pieces,
# each of which can be validated on its own:
#
#   1. **Spectral reflectance curves.** Every material is defined by a
#      piecewise-linear reflectance spectrum ρ(λ) anchored on literature
#      values (ASTER/USGS spectral libraries; Oke, *Boundary Layer
#      Climates* Table 1.1; Bonan, *Ecological Climatology* §12). Band
#      values are obtained by integrating that curve against the solar
#      spectrum, not by copying a number into a table. Change the spectral
#      grid and the band albedos follow automatically.
#
#   2. **BRDF kernels.** Four bidirectional reflectance models —
#      Lambertian, Rahman-Pinty-Verstraete, Ross-Li (the MODIS operational
#      kernel pair) and Cook-Torrance — each returning a bidirectional
#      reflectance *factor*, and each reciprocal by construction.
#
#   3. **Integrated albedos computed from those kernels by quadrature.**
#      `black_sky_albedo`, `white_sky_albedo` and `blue_sky_albedo` are
#      genuine numerical integrals of the BRDF over the view (and source)
#      hemisphere. Nothing here is a fitted formula, which is what makes
#      the numbers trustworthy: the Lambertian case must reproduce its own
#      albedo to machine-ish precision, and that single test validates the
#      quadrature for every other kernel. As an independent check, the
#      white-sky integrals of the two MODIS kernels come out as 0.189186
#      and -1.377660 against the published operational constants 0.189184
#      and -1.377622 — five significant figures, obtained from nothing but
#      the kernel formulae and Gauss-Legendre.
#
# **A limitation that is stated rather than hidden.** The semi-empirical
# kernels are not energy-constrained, and two of them are formally
# *unbounded* as the incidence angle approaches the horizon: the RPV
# Minnaert factor carries `(cos θ_i)^{k-1}` which diverges for `k < 1`,
# and the Li-sparse kernel carries `sec θ_i`, which diverges outright.
# Every preset shipped here has been swept and conserves energy —
# `α_bs ≤ 1` — for all incidence angles out to 85°, with the largest
# value being 0.946 for polished metal; beyond about 87° several of them
# exceed unity (RPV forest reaches 1.22 and Ross-Li urban 2.00 at 89.5°).
# This is a property of the published models, not of this implementation,
# and it is why MODIS restricts its retrievals to solar zenith angles
# below 70°. Clamping the integral would conceal it, so the integral is
# reported exactly; the surface-optics wrappers (`BRDFSurface` and
# friends), which are what a solver actually sees, do clamp to [0,1].
#
# On top of those sit the physically special surfaces — Fresnel water with
# a Cox-Munk wave-slope distribution, and snow with grain-growth, ageing
# and soot darkening — plus `TiledSurface`, the area-weighted mosaic that
# an urban grid cell actually is.
#
# Conventions, fixed here and used throughout:
#
#   * Angles are in **radians**. θ is measured from the local zenith, φ is
#     azimuth; only the relative azimuth φ_v − φ_i matters for the kernels
#     shipped here (they are all azimuthally isotropic in the sense that
#     they depend on the relative azimuth alone).
#   * `brdf(model, θi, φi, θv, φv)` returns the **bidirectional
#     reflectance factor** BRF = π·f_r, where f_r is the BRDF proper in
#     sr⁻¹. A Lambertian surface of albedo a has BRF = a in every
#     direction, which makes the numbers directly readable.
#   * Directions are "pointing away from the surface": the incident
#     direction is the direction *towards the sun*, the view direction is
#     the direction *towards the sensor*. Exact backscatter (the hot spot)
#     is therefore θ_v = θ_i, φ_v = φ_i.
#   * Albedos are dimensionless in [0, 1]. Band indices refer to the
#     shortwave `SpectralGrid` the object was built on (longwave grid for
#     emissivity).
#
# Fidelity ceiling, stated plainly: the spectral library is a curated set
# of representative curves, not a measurement of any particular city. Two
# asphalt surfaces differ by ±0.03 in broadband albedo depending on age
# and aggregate. What this module guarantees is the *shape* — the red
# edge, the snow NIR fall-off, the Fresnel rise at grazing incidence, the
# emissivity outlier of bare metal — because those shapes are what change
# the answer by tens of W m⁻².

using ..RadiationCore
using ..RadiationCore: SpectralGrid, AbstractSurfaceOptics,
                       nbands, ngpoints, band_limits, gpoint_range,
                       shortwave_bands, longwave_bands,
                       PLANCK_H, LIGHT_C, BOLTZMANN_K
import ..RadiationCore: surface_albedo, surface_emissivity,
                        material_albedo, material_emissivity

export # spectral library
       material_names, spectral_reflectance, band_average_reflectance,
       material_emissivity, spectral_emissivity,
       SpectralAlbedo, material_albedo,
       band_center_wavelength,
       # BRDF models
       AbstractBRDF, LambertianBRDF, RPVBRDF, RossLiBRDF, CookTorranceBRDF,
       brdf, rpv_preset, rossli_preset, cook_torrance_preset,
       ross_thick_kernel, li_sparse_kernel,
       # integrated albedos
       black_sky_albedo, white_sky_albedo, blue_sky_albedo,
       hemisphere_quadrature_albedo,
       BRDFSurface,
       # water
       FresnelWater, fresnel_reflectance, fresnel_polarized_reflectance,
       cox_munk_slope_variance,
       whitecap_fraction, water_albedo_direct, water_albedo_diffuse,
       # snow
       SnowSurface, snow_grain_radius, snow_albedo, snow_albedo_spectrum,
       SNOW_REFERENCE_RADIUS, SNOW_GROWTH_RATE, SNOW_SOOT_CONSTANT,
       # tiles and integration helpers
       TiledSurface, tiled_surface, tile_fractions, tile_components,
       direct_albedo_vector, diffuse_albedo_vector, emissivity_vector,
       broadband_albedo, broadband_emissivity

# =========================================================================
# Quadrature
# =========================================================================
#
# Gauss-Legendre nodes by Newton iteration on the Legendre polynomial.
# Duplicated from `RadiationCore` deliberately: this module's numerical
# results are its whole reason for existing, so it owns its quadrature
# rather than depending on a private symbol in another module.

const _GL_CACHE = Dict{Int, Tuple{Vector{Float64}, Vector{Float64}}}()

"""
    _gauss_legendre(n) -> (x, w)

`n`-point Gauss-Legendre nodes and weights on `[-1, 1]`. Cached.
"""
function _gauss_legendre(n::Int)
    n >= 1 || error("_gauss_legendre: n must be ≥ 1")
    haskey(_GL_CACHE, n) && return _GL_CACHE[n]
    x = zeros(Float64, n); w = zeros(Float64, n)
    for i in 1:n
        z = cos(π * (i - 0.25) / (n + 0.5))      # Chebyshev-like start
        for _ in 1:100
            p0 = 1.0; p1 = z
            for k in 2:n
                p0, p1 = p1, ((2k - 1) * z * p1 - (k - 1) * p0) / k
            end
            dp = n * (z * p1 - p0) / (z^2 - 1)
            dz = p1 / dp
            z -= dz
            abs(dz) < 1e-15 && break
        end
        p0 = 1.0; p1 = z
        for k in 2:n
            p0, p1 = p1, ((2k - 1) * z * p1 - (k - 1) * p0) / k
        end
        dp = n * (z * p1 - p0) / (z^2 - 1)
        x[i] = z
        w[i] = 2 / ((1 - z^2) * dp^2)
    end
    _GL_CACHE[n] = (x, w)
    return (x, w)
end

"Gauss-Legendre nodes and weights mapped to `[0, 1]`."
function _gl01(n::Int)
    x, w = _gauss_legendre(n)
    return ((x .+ 1) ./ 2, w ./ 2)
end

# =========================================================================
# Spectral weighting
# =========================================================================

# Planck spectral radiance per unit wavenumber, ν in cm⁻¹. Used purely as
# a *weighting function*: for the shortwave it stands in for the solar
# spectrum (a 5778 K blackbody reproduces the shape of the extraterrestrial
# spectrum to a few percent across the bands that carry the energy), and
# for the longwave it is the terrestrial Planck function at the surface
# temperature. Only ratios enter, so the normalisation is irrelevant.
@inline function _planck_nu(T_K::Real, ν_cm::Real)
    ν = ν_cm * 100.0
    c1 = 2 * PLANCK_H * LIGHT_C^2
    c2 = PLANCK_H * LIGHT_C / BOLTZMANN_K
    x = c2 * ν / T_K
    x > 700 && return 0.0
    return c1 * ν^3 / expm1(x)
end

const SOLAR_BRIGHTNESS_TEMPERATURE = 5778.0   # K
const TERRESTRIAL_TEMPERATURE      = 288.0    # K

"Linear interpolation on a λ-sorted anchor list, clamped outside the range."
function _interp_anchors(anchors::Vector{NTuple{2,Float64}}, λ::Real)
    n = length(anchors)
    λ <= anchors[1][1]  && return anchors[1][2]
    λ >= anchors[n][1]  && return anchors[n][2]
    lo = 1; hi = n
    while hi - lo > 1
        mid = (lo + hi) >>> 1
        if anchors[mid][1] <= λ
            lo = mid
        else
            hi = mid
        end
    end
    λ0, r0 = anchors[lo]; λ1, r1 = anchors[hi]
    t = (λ - λ0) / (λ1 - λ0)
    return r0 + t * (r1 - r0)
end

"""
    _band_mean(anchors, ν_lo, ν_hi, T_weight; ngl = 6) -> Real

Planck-weighted band mean of a piecewise-linear spectrum:

```math
\\bar{ρ} = \\frac{\\int_{ν_{lo}}^{ν_{hi}} ρ(λ(ν))\\,B_ν(T_w)\\,dν}
                 {\\int_{ν_{lo}}^{ν_{hi}} B_ν(T_w)\\,dν}
```

The integral is split at every anchor wavelength that falls inside the
band, so each sub-interval has a *smooth* integrand (linear × Planck) and
a 6-point Gauss-Legendre rule is essentially exact. Integrating straight
through the kinks — which is what a single quadrature over the whole band
would do — costs two to three digits on materials with sharp features
such as the vegetation red edge.
"""
function _band_mean(anchors::Vector{NTuple{2,Float64}},
                    ν_lo::Real, ν_hi::Real, T_weight::Real; ngl::Int = 6)
    pts = Float64[Float64(ν_lo), Float64(ν_hi)]
    for (λ, _) in anchors
        ν = 1e4 / λ
        (ν_lo < ν < ν_hi) && push!(pts, ν)
    end
    sort!(pts); unique!(pts)
    x, w = _gauss_legendre(ngl)
    num = 0.0; den = 0.0
    for i in 1:(length(pts) - 1)
        a, b = pts[i], pts[i + 1]
        h = (b - a) / 2; m = (b + a) / 2
        for j in eachindex(x)
            ν = m + h * x[j]
            ww = w[j] * h * _planck_nu(T_weight, ν)
            num += ww * _interp_anchors(anchors, 1e4 / ν)
            den += ww
        end
    end
    den <= 0 && return _interp_anchors(anchors, 2e4 / (ν_lo + ν_hi))
    return num / den
end

"""
    band_center_wavelength(spec, b; T_weight = 5778) -> μm

Source-weighted mean wavelength of band `b`. This is the wavelength at
which a spectrally varying process (soot absorption, for instance) should
be evaluated when it is applied band by band; the arithmetic mean of the
band edges is noticeably biased for the wide near-infrared bands.
"""
function band_center_wavelength(spec::SpectralGrid, b::Integer;
                                T_weight::Real = SOLAR_BRIGHTNESS_TEMPERATURE)
    lo, hi = band_limits(spec, Int(b))
    # mean of λ itself, weighted by the source
    anchors = NTuple{2,Float64}[(1e4 / hi, 1e4 / hi), (1e4 / lo, 1e4 / lo)]
    return _band_mean(anchors, lo, hi, T_weight)
end

# =========================================================================
# Spectral reflectance library
# =========================================================================
#
# Each entry is a list of (λ [μm], ρ) anchors, sorted by wavelength, and
# is interpreted as the hemispherical reflectance of a horizontal sample.
# The anchors are chosen to reproduce the features that matter:
#
#   * the iron-oxide rise from blue to red in brick and clay tile,
#   * the chlorophyll green bump at 0.55 μm, the chlorophyll absorption
#     well at 0.68 μm and the red edge climbing to the NIR plateau
#     between 0.70 and 0.78 μm in all vegetation,
#   * the liquid-water absorption bands at 1.45 and 1.95 μm in vegetation
#     and in wet soil,
#   * the collapse of snow reflectance beyond 1.4 μm, where ice becomes a
#     strong absorber, which is why snow looks white to the eye and dark
#     to a near-infrared sensor.

const _REFLECTANCE_CURVES = Dict{Symbol, Vector{NTuple{2,Float64}}}(

    # --- urban, built ---------------------------------------------------
    :asphalt => [(0.30, 0.050), (0.40, 0.060), (0.50, 0.070), (0.70, 0.090),
                 (1.00, 0.110), (1.50, 0.130), (2.00, 0.140), (2.50, 0.140),
                 (4.00, 0.120), (12.0, 0.100)],

    :concrete => [(0.30, 0.150), (0.40, 0.220), (0.50, 0.280), (0.70, 0.330),
                  (1.00, 0.360), (1.50, 0.380), (2.00, 0.360), (2.50, 0.320),
                  (4.00, 0.280), (12.0, 0.250)],

    :brick => [(0.30, 0.080), (0.40, 0.090), (0.50, 0.120), (0.60, 0.220),
               (0.70, 0.300), (0.90, 0.380), (1.20, 0.420), (1.60, 0.450),
               (2.00, 0.420), (2.50, 0.350), (4.00, 0.300), (12.0, 0.280)],

    :roof_tile_red => [(0.30, 0.070), (0.40, 0.080), (0.50, 0.100),
                       (0.60, 0.180), (0.70, 0.260), (0.90, 0.330),
                       (1.20, 0.360), (1.60, 0.380), (2.00, 0.350),
                       (2.50, 0.300), (4.00, 0.260), (12.0, 0.240)],

    :roof_tile_grey => [(0.30, 0.100), (0.50, 0.140), (0.70, 0.170),
                        (1.00, 0.200), (1.50, 0.220), (2.00, 0.210),
                        (2.50, 0.190), (4.00, 0.170), (12.0, 0.160)],

    # Float glass: two air-glass interfaces at n = 1.52 give ~8% at normal
    # incidence, very nearly grey. The directional behaviour — which is
    # what actually matters for a glazed façade — belongs to
    # `CookTorranceBRDF`, not to this curve.
    :glass => [(0.30, 0.090), (0.40, 0.080), (0.70, 0.075), (1.00, 0.075),
               (1.50, 0.080), (2.00, 0.090), (2.50, 0.100), (4.00, 0.120),
               (12.0, 0.150)],

    # Weathered galvanised steel / aluminium sheet: bright and rising into
    # the infrared. Paired with an emissivity of 0.25, this is the surface
    # that runs hottest in a city despite reflecting the most sunlight.
    :metal_roof => [(0.30, 0.450), (0.40, 0.500), (0.50, 0.550),
                    (0.70, 0.580), (1.00, 0.620), (1.50, 0.660),
                    (2.00, 0.700), (2.50, 0.720), (4.00, 0.750),
                    (12.0, 0.800)],

    # --- bare ground ----------------------------------------------------
    :soil_dry => [(0.30, 0.050), (0.40, 0.090), (0.50, 0.130), (0.60, 0.180),
                  (0.70, 0.220), (0.90, 0.280), (1.10, 0.310), (1.40, 0.320),
                  (1.60, 0.350), (1.90, 0.300), (2.20, 0.330), (2.50, 0.300),
                  (4.00, 0.250), (12.0, 0.200)],

    # Wetting roughly halves the reflectance (the water film fills the
    # inter-grain voids and suppresses first-surface scattering) and
    # deepens the 1.45 and 1.95 μm water bands.
    :soil_wet => [(0.30, 0.020), (0.40, 0.040), (0.50, 0.060), (0.60, 0.080),
                  (0.70, 0.100), (0.90, 0.130), (1.10, 0.150), (1.40, 0.110),
                  (1.60, 0.160), (1.90, 0.070), (2.20, 0.130), (2.50, 0.100),
                  (4.00, 0.100), (12.0, 0.080)],

    :sand => [(0.30, 0.100), (0.40, 0.180), (0.50, 0.260), (0.60, 0.340),
              (0.70, 0.400), (0.90, 0.460), (1.10, 0.490), (1.40, 0.500),
              (1.60, 0.520), (1.90, 0.480), (2.20, 0.500), (2.50, 0.450),
              (4.00, 0.350), (12.0, 0.250)],

    # --- vegetation -----------------------------------------------------
    :grass => [(0.30, 0.030), (0.40, 0.040), (0.45, 0.040), (0.50, 0.060),
               (0.55, 0.120), (0.60, 0.070), (0.65, 0.040), (0.68, 0.035),
               (0.70, 0.100), (0.72, 0.280), (0.75, 0.450), (0.80, 0.480),
               (0.90, 0.500), (1.00, 0.480), (1.10, 0.470), (1.30, 0.400),
               (1.45, 0.180), (1.60, 0.320), (1.80, 0.250), (1.95, 0.080),
               (2.20, 0.200), (2.50, 0.100), (4.00, 0.050), (12.0, 0.030)],

    :tree_deciduous => [(0.30, 0.030), (0.40, 0.035), (0.45, 0.035),
                        (0.50, 0.050), (0.55, 0.090), (0.60, 0.055),
                        (0.65, 0.035), (0.68, 0.030), (0.70, 0.080),
                        (0.72, 0.220), (0.75, 0.380), (0.80, 0.430),
                        (0.90, 0.450), (1.00, 0.440), (1.10, 0.420),
                        (1.30, 0.360), (1.45, 0.150), (1.60, 0.280),
                        (1.80, 0.210), (1.95, 0.060), (2.20, 0.160),
                        (2.50, 0.080), (4.00, 0.040), (12.0, 0.030)],

    # Needleleaf canopies are markedly darker than broadleaf: the shoot
    # geometry traps photons, so both the visible and the NIR plateau drop
    # by roughly a third relative to a deciduous canopy.
    :tree_coniferous => [(0.30, 0.020), (0.40, 0.025), (0.45, 0.025),
                         (0.50, 0.035), (0.55, 0.060), (0.60, 0.040),
                         (0.65, 0.028), (0.68, 0.025), (0.70, 0.050),
                         (0.72, 0.130), (0.75, 0.240), (0.80, 0.280),
                         (0.90, 0.300), (1.00, 0.290), (1.10, 0.280),
                         (1.30, 0.240), (1.45, 0.100), (1.60, 0.190),
                         (1.80, 0.140), (1.95, 0.040), (2.20, 0.110),
                         (2.50, 0.050), (4.00, 0.030), (12.0, 0.020)],

    # --- water ----------------------------------------------------------
    # Nadir Fresnel reflection (2.1% at n = 1.34) plus the water-leaving
    # radiance of clear open water. This preset is the *diffuse* value;
    # for the solar-zenith dependence use `FresnelWater`, which is an
    # order of magnitude larger at grazing incidence and cannot be
    # represented by a fixed spectrum.
    :water => [(0.30, 0.041), (0.40, 0.046), (0.45, 0.051), (0.50, 0.046),
               (0.55, 0.036), (0.60, 0.029), (0.65, 0.026), (0.70, 0.024),
               (0.80, 0.022), (1.00, 0.0215), (1.50, 0.021), (2.50, 0.021),
               (4.00, 0.021), (12.0, 0.021)],

    # --- snow -----------------------------------------------------------
    # Fine-grained *pure* fresh snow, effective grain radius 50 μm, the
    # reference spectrum for `SnowSurface`. Nearly perfectly white through
    # the visible, falling through the NIR as ice absorption switches on:
    # ~0.60 at 1.5 μm, ~0.20 at 2.0 μm and essentially black beyond
    # 2.5 μm (Wiscombe & Warren 1980, Fig. 2).
    #
    # The visible values are those of snow with *no* impurities, ~0.99
    # rather than the 0.97 typically measured in the field, because the
    # soot loading is an explicit parameter of `SnowSurface` rather than
    # something baked into the reference curve. Folding a trace impurity
    # into the reference would double-count it and would also compress the
    # visible co-albedo, which is precisely the quantity that sets how
    # much a given soot loading darkens the snow.
    :snow_fresh => [(0.30, 0.980), (0.40, 0.990), (0.50, 0.992),
                    (0.60, 0.990), (0.70, 0.982), (0.80, 0.968),
                    (0.90, 0.945), (1.00, 0.915), (1.10, 0.875),
                    (1.20, 0.825), (1.30, 0.720), (1.40, 0.560),
                    (1.50, 0.610), (1.60, 0.555), (1.80, 0.300),
                    (2.00, 0.200), (2.20, 0.150), (2.50, 0.050),
                    (4.00, 0.020), (12.0, 0.010)],

    # Coarse-grained melt-metamorphosed snow with a little accumulated
    # dust: the visible drops to ~0.8 and the NIR collapses.
    :snow_old => [(0.30, 0.750), (0.40, 0.800), (0.50, 0.820),
                  (0.60, 0.810), (0.70, 0.790), (0.80, 0.750),
                  (0.90, 0.700), (1.00, 0.640), (1.10, 0.560),
                  (1.20, 0.480), (1.30, 0.360), (1.40, 0.220),
                  (1.50, 0.250), (1.60, 0.200), (1.80, 0.100),
                  (2.00, 0.060), (2.20, 0.050), (2.50, 0.020),
                  (4.00, 0.010), (12.0, 0.010)],

    # Subsurface (water-leaving) reflectance of clear open ocean, used by
    # `FresnelWater` on top of the specular term. Blue-peaked and
    # effectively zero beyond 0.8 μm, where liquid water absorbs within
    # millimetres.
    :water_leaving => [(0.30, 0.020), (0.40, 0.025), (0.45, 0.030),
                       (0.50, 0.025), (0.55, 0.015), (0.60, 0.008),
                       (0.65, 0.005), (0.70, 0.003), (0.80, 0.001),
                       (1.00, 0.0005), (1.50, 0.0), (2.50, 0.0),
                       (12.0, 0.0)],
)

# Materials offered as surface presets (`:water_leaving` is an internal
# component of `FresnelWater`, not a surface in its own right).
const _MATERIALS = Symbol[:asphalt, :concrete, :brick, :roof_tile_red,
                          :roof_tile_grey, :glass, :metal_roof,
                          :soil_dry, :soil_wet, :sand,
                          :grass, :tree_deciduous, :tree_coniferous,
                          :water, :snow_fresh, :snow_old]

"""
    material_names() -> Vector{Symbol}

The materials available to [`material_albedo`](@ref), in table order:
urban surfaces first, then bare ground, vegetation, water and snow.
"""
material_names() = copy(_MATERIALS)

"""
    spectral_reflectance(name, λ) -> ρ

Hemispherical reflectance of material `name` at wavelength `λ` in μm, by
linear interpolation between the tabulated anchors and clamped to the end
values outside the tabulated range.
"""
function spectral_reflectance(name::Symbol, λ::Real)
    haskey(_REFLECTANCE_CURVES, name) ||
        error("spectral_reflectance: unknown material $name; " *
              "available: $(material_names())")
    return _interp_anchors(_REFLECTANCE_CURVES[name], λ)
end

"""
    band_average_reflectance(name, ν_lo, ν_hi; T_weight = 5778) -> ρ̄

Solar-weighted mean reflectance of material `name` over the wavenumber
band `[ν_lo, ν_hi]` (cm⁻¹). The weighting is a 5778 K blackbody, which
reproduces the shape of the extraterrestrial solar spectrum well enough
that the residual error is far below the spread between two samples of
the same material.
"""
function band_average_reflectance(name::Symbol, ν_lo::Real, ν_hi::Real;
                                  T_weight::Real = SOLAR_BRIGHTNESS_TEMPERATURE)
    haskey(_REFLECTANCE_CURVES, name) ||
        error("band_average_reflectance: unknown material $name")
    return _band_mean(_REFLECTANCE_CURVES[name], ν_lo, ν_hi, T_weight)
end

# =========================================================================
# Longwave emissivity library
# =========================================================================
#
# Broadband (8-14 μm window and beyond) emissivities. Almost every natural
# and built surface sits between 0.90 and 0.99 — and then there is bare
# metal at 0.25, which is the single most consequential number in this
# table. A metal roof reflects 60% of the sunlight that hits it and still
# runs hotter than asphalt at night, because it cannot radiate: its
# equilibrium temperature is set by εσT⁴ with ε four times smaller than
# everything around it. Painting the same sheet white raises ε to ~0.9 and
# drops the daytime surface temperature by 10-20 K.

const _EMISSIVITY_BROADBAND = Dict{Symbol, Float64}(
    :asphalt         => 0.95,
    :concrete        => 0.91,
    :brick           => 0.93,
    :roof_tile_red   => 0.90,
    :roof_tile_grey  => 0.90,
    :glass           => 0.94,
    :metal_roof      => 0.25,
    :soil_dry        => 0.92,
    :soil_wet        => 0.95,
    :sand            => 0.90,
    :grass           => 0.98,
    :tree_deciduous  => 0.98,
    :tree_coniferous => 0.98,
    :water           => 0.96,
    :snow_fresh      => 0.99,
    :snow_old        => 0.98,
)

# Spectral emissivity where it genuinely departs from grey. Quartz sand
# has the silicate reststrahlen band: ε drops to ~0.70 between 8 and
# 9.5 μm, which is why thermal-infrared remote sensing can identify sand
# from space and why a grey ε for a desert surface biases the retrieved
# skin temperature by several kelvin.
const _EMISSIVITY_CURVES = Dict{Symbol, Vector{NTuple{2,Float64}}}(
    :sand => [(4.0, 0.93), (7.5, 0.93), (8.0, 0.88), (8.5, 0.72),
              (9.0, 0.70), (9.5, 0.80), (10.0, 0.90), (11.0, 0.93),
              (14.0, 0.95), (50.0, 0.95)],
)

"""
    material_emissivity(name) -> ε

Broadband longwave emissivity of material `name`. For materials with a
spectral emissivity curve this is the Planck-weighted (288 K) mean of
that curve, so the scalar and the band values are mutually consistent.
"""
function material_emissivity(name::Symbol)
    if haskey(_EMISSIVITY_CURVES, name)
        # 10-50000 cm⁻¹ covers the whole terrestrial Planck function
        return _band_mean(_EMISSIVITY_CURVES[name], 10.0, 3000.0,
                          TERRESTRIAL_TEMPERATURE; ngl = 8)
    end
    haskey(_EMISSIVITY_BROADBAND, name) ||
        error("material_emissivity: unknown material $name")
    return _EMISSIVITY_BROADBAND[name]
end

"""
    spectral_emissivity(name, λ) -> ε

Longwave emissivity of material `name` at wavelength `λ` (μm). Grey for
most materials; for quartz sand this resolves the 8-9.5 μm reststrahlen
band.
"""
function spectral_emissivity(name::Symbol, λ::Real)
    haskey(_EMISSIVITY_CURVES, name) &&
        return _interp_anchors(_EMISSIVITY_CURVES[name], λ)
    haskey(_EMISSIVITY_BROADBAND, name) ||
        error("spectral_emissivity: unknown material $name")
    return _EMISSIVITY_BROADBAND[name]
end

"Per-band emissivity vector of a material on a longwave grid."
function _emissivity_bands(::Type{T}, name::Symbol,
                           lw::SpectralGrid) where {T}
    nb = nbands(lw)
    ε = Vector{T}(undef, nb)
    if haskey(_EMISSIVITY_CURVES, name)
        curve = _EMISSIVITY_CURVES[name]
        for b in 1:nb
            lo, hi = band_limits(lw, b)
            ε[b] = T(_band_mean(curve, lo, hi, TERRESTRIAL_TEMPERATURE))
        end
    else
        fill!(ε, T(material_emissivity(name)))
    end
    return ε
end

# =========================================================================
# Solar-zenith dependence of the direct-beam albedo
# =========================================================================
#
# Briegleb et al. (1986, J. Clim. Appl. Meteor. 25, 214) fitted the
# direct-beam albedo of natural surfaces to
#
#     α_dir(μ₀) ∝ (1 + d) / (1 + 2 d μ₀)
#
# with d a surface-dependent steepness: ~0.4 for vegetation canopies,
# ~0.1 for soils, near zero for snow. The functional form captures the
# right physics — at low sun the beam penetrates less deeply, is scattered
# nearer the surface, and escapes more readily — and it is the form used
# operationally in BATS and CLM.
#
# One constraint that is *not* part of the original fit is imposed here:
# the white-sky albedo is by definition the cosine-weighted hemispherical
# average of the black-sky albedo,
#
#     α_diffuse = 2 ∫₀¹ α_dir(μ₀) μ₀ dμ₀,
#
# so the shape function is normalised to satisfy it exactly. Without that
# normalisation the direct and diffuse albedos of the same surface are
# mutually inconsistent by 5-10%, and a model that switches between clear
# and overcast conditions then shows a spurious albedo jump.

const _ZENITH_D = Dict{Symbol, Float64}(
    :asphalt         => 0.10,
    :concrete        => 0.10,
    :brick           => 0.12,
    :roof_tile_red   => 0.12,
    :roof_tile_grey  => 0.12,
    :glass           => 0.80,
    :metal_roof      => 0.50,
    :soil_dry        => 0.15,
    :soil_wet        => 0.20,
    :sand            => 0.15,
    :grass           => 0.40,
    :tree_deciduous  => 0.40,
    :tree_coniferous => 0.40,
    :water           => 1.50,
    :snow_fresh      => 0.03,
    :snow_old        => 0.05,
)

"Normalisation `N(d) = 2∫₀¹ (1+d)/(1+2dμ) μ dμ`, by quadrature."
function _zenith_norm(d::Real)
    x, w = _gl01(32)
    acc = 0.0
    for i in eachindex(x)
        acc += w[i] * 2 * x[i] * (1 + d) / (1 + 2 * d * x[i])
    end
    return acc
end

"""
    _zenith_shape(d, μ₀)

Briegleb direct-beam shape function, normalised so that its
cosine-weighted hemispherical mean is exactly 1. Multiplying the
white-sky albedo by this gives the black-sky albedo at `μ₀`.
"""
_zenith_shape(d::Real, μ₀::Real) =
    (1 + d) / (1 + 2 * d * clamp(μ₀, 0.0, 1.0)) / _zenith_norm(d)

# =========================================================================
# SpectralAlbedo
# =========================================================================

"""
    SpectralAlbedo{T} <: AbstractSurfaceOptics

Per-band direct and diffuse shortwave albedo plus per-band longwave
emissivity for a single homogeneous material.

Fields:

  * `name`       — material symbol
  * `spec`       — the shortwave `SpectralGrid` the albedos live on
  * `lw_spec`    — the longwave `SpectralGrid` the emissivities live on
  * `direct`     — black-sky (directional-hemispherical) albedo per
                   shortwave band **at overhead sun**, `μ₀ = 1`
  * `diffuse`    — white-sky (bihemispherical) albedo per shortwave band
  * `emissivity` — emissivity per longwave band
  * `zenith_d`   — Briegleb steepness parameter for the solar-zenith
                   dependence of the direct-beam albedo

`direct` and `diffuse` are genuinely different quantities, not two copies
of the same table: `diffuse` is the integral of the material's measured
reflectance curve over the band, and `direct` is that value evaluated at
overhead sun through the normalised Briegleb shape, where it is typically
15-20% lower for a vegetated surface. Query the albedo at an arbitrary
sun angle with `surface_albedo(s, band, μ₀)`.

One caveat for very bright bands. The Briegleb shape multiplies the
diffuse albedo by up to `(1+d)/N(d)` at grazing incidence, which for a
band albedo already near unity — pure snow in the visible is 0.99 — would
push the result above 1. `surface_albedo` clamps, and the clamp costs a
few parts per thousand of the exact hemispheric consistency between the
direct and diffuse albedos. For snow use [`SnowSurface`](@ref) instead:
its escape-function form `α_dir = α_dif^u(μ₀)` is bounded in `[0,1]` by
construction and needs no clamp.
"""
struct SpectralAlbedo{T<:AbstractFloat} <: AbstractSurfaceOptics
    name::Symbol
    spec::SpectralGrid{T}
    lw_spec::SpectralGrid{T}
    direct::Vector{T}
    diffuse::Vector{T}
    emissivity::Vector{T}
    zenith_d::T
end

"""
    material_albedo(name, spec = shortwave_bands(); lw_spec = longwave_bands())

Build a [`SpectralAlbedo`](@ref) for material `name` on the shortwave grid
`spec`. Band albedos are obtained by integrating the material's
reflectance curve against the solar spectrum over each band — change the
band structure and the numbers follow — and the emissivities by
integrating the emissivity curve against the terrestrial Planck function
over each longwave band.

```julia
sw = shortwave_bands()
α  = material_albedo(:grass, sw)
surface_albedo(α, 10, 0.5)   # visible band, 60° sun
surface_albedo(α,  8, 0.5)   # near-infrared: ~6× larger — the red edge
```

Available materials: $(join(string.(_MATERIALS), ", ")).
"""
function material_albedo(name::Symbol,
                         spec::SpectralGrid{T} = shortwave_bands();
                         lw_spec::SpectralGrid{T} = longwave_bands(T)) where {T}
    name in _MATERIALS ||
        error("material_albedo: unknown material $name; " *
              "available: $(material_names())")
    spec.kind === :shortwave ||
        error("material_albedo: `spec` must be a shortwave grid")
    lw_spec.kind === :longwave ||
        error("material_albedo: `lw_spec` must be a longwave grid")
    nb = nbands(spec)
    d  = T(_ZENITH_D[name])
    diffuse = Vector{T}(undef, nb)
    direct  = Vector{T}(undef, nb)
    shape1  = _zenith_shape(d, 1.0)
    for b in 1:nb
        lo, hi = band_limits(spec, b)
        ρ = T(band_average_reflectance(name, lo, hi))
        diffuse[b] = clamp(ρ, zero(T), one(T))
        direct[b]  = clamp(ρ * T(shape1), zero(T), one(T))
    end
    ε = _emissivity_bands(T, name, lw_spec)
    return SpectralAlbedo{T}(name, spec, lw_spec, direct, diffuse, ε, d)
end

function Base.show(io::IO, s::SpectralAlbedo{T}) where {T}
    print(io, "SpectralAlbedo{$T}(:", s.name, ", ", nbands(s.spec),
          " SW bands, white-sky ",
          string(round(broadband_albedo(s); digits = 3)),
          ", ε ", string(round(material_emissivity(s.name); digits = 3)), ")")
end

"""
    surface_albedo(s::SpectralAlbedo, band, μ₀) -> α

Black-sky (direct-beam) albedo in shortwave `band` at cosine solar zenith
angle `μ₀`, from the stored overhead-sun value and the normalised
Briegleb shape function. Clamped to `[0, 1]`.

    surface_albedo(s::SpectralAlbedo, band) -> α

White-sky (diffuse) albedo in `band`.
"""
function surface_albedo(s::SpectralAlbedo{T}, band::Integer, μ₀::Real) where {T}
    1 <= band <= nbands(s.spec) ||
        error("surface_albedo: band $band outside 1:$(nbands(s.spec))")
    return clamp(s.diffuse[band] * T(_zenith_shape(s.zenith_d, μ₀)),
                 zero(T), one(T))
end

surface_albedo(s::SpectralAlbedo, band::Integer) = s.diffuse[band]

"""
    surface_emissivity(s::SpectralAlbedo, band) -> ε

Longwave emissivity in longwave `band`.
"""
function surface_emissivity(s::SpectralAlbedo, band::Integer)
    1 <= band <= length(s.emissivity) ||
        error("surface_emissivity: band $band outside 1:$(length(s.emissivity))")
    return s.emissivity[band]
end


# =========================================================================
# BRDF models
# =========================================================================

"""
    AbstractBRDF

A bidirectional reflectance distribution model. Concrete subtypes
implement

    brdf(model, θ_i, φ_i, θ_v, φ_v) -> BRF

returning the bidirectional reflectance **factor** `BRF = π f_r`, with
all angles in radians, θ measured from the zenith, and both directions
pointing away from the surface (towards the sun and towards the sensor
respectively).

Every model shipped here is **reciprocal**:
`brdf(m, θi, φi, θv, φv) == brdf(m, θv, φv, θi, φi)`. Reciprocity is a
consequence of time-reversal symmetry of the microscopic optics and is
not optional — a non-reciprocal BRDF makes the surface a source or a sink
of radiation depending on which way the light happens to be travelling,
and the test suite checks it for every model and preset.

Integrated quantities ([`black_sky_albedo`](@ref),
[`white_sky_albedo`](@ref), [`blue_sky_albedo`](@ref)) are obtained by
numerical integration of `brdf` and therefore work for any subtype
without further code.
"""
abstract type AbstractBRDF end

# Guard against the coordinate singularity at the horizon: the kernels
# contain sec θ and cos θ ^ (k-1) factors that are infinite at θ = π/2.
# The quadratures never place a node there, but a user might.
@inline _safe_cos(θ::Real) = max(cos(θ), 1e-9)

# --- Lambertian ----------------------------------------------------------

"""
    LambertianBRDF{T}(ρ)

Isotropic reflector: `f_r = ρ/π` in every direction, so `BRF = ρ`.

This is the reference case, and it exists mainly to be a test oracle. Its
black-sky albedo is `ρ` at every solar zenith angle and its white-sky
albedo is `ρ`; any quadrature that fails to reproduce those to ~1e-10 is
broken, and no result obtained with it can be trusted for the other
kernels.

Real surfaces are never Lambertian. Even fresh snow, the closest natural
approximation, shows a 10-20% forward-scattering enhancement at low sun.
"""
struct LambertianBRDF{T<:AbstractFloat} <: AbstractBRDF
    ρ::T
    # Inner constructors, here and in the other models, are not stylistic:
    # a parametric struct with no inner constructor gets a default
    # `Model(a::T, b::T, ...)` outer constructor which is *more specific*
    # than a validating `Model(a::Real, b::Real, ...)` method and silently
    # takes precedence for the common all-Float64 call. Validation has to
    # live in the inner constructor or it does not run at all.
    function LambertianBRDF{T}(ρ::Real) where {T<:AbstractFloat}
        0 <= ρ <= 1 || error("LambertianBRDF: ρ must lie in [0,1], got $ρ")
        return new{T}(T(ρ))
    end
end

LambertianBRDF(ρ::Real) = LambertianBRDF{typeof(float(ρ))}(ρ)

brdf(m::LambertianBRDF, θ_i::Real, φ_i::Real, θ_v::Real, φ_v::Real) = m.ρ

Base.show(io::IO, m::LambertianBRDF) = print(io, "LambertianBRDF(ρ=", m.ρ, ")")

# --- shared geometry -----------------------------------------------------

"""
    _phase_cosine(θi, θv, Δφ) -> cos g

Cosine of the **phase angle** `g`, the angle between the view direction
and the incident direction, both pointing away from the surface:

```math
\\cos g = \\cos θ_i \\cos θ_v + \\sin θ_i \\sin θ_v \\cos Δφ
```

`g = 0` is exact backscatter (the hot spot); `g = π` would be exact
forward transmission. Note that the *scattering* angle ξ, measured
between the photon's incoming and outgoing propagation directions,
satisfies `cos ξ = -cos g`: backscatter is ξ = 180°. Both the RPV phase
function and the Ross-thick kernel are conventionally written in terms of
`g`, and that is the convention used here.
"""
@inline function _phase_cosine(θi::Real, θv::Real, Δφ::Real)
    return cos(θi) * cos(θv) + sin(θi) * sin(θv) * cos(Δφ)
end

# --- Rahman-Pinty-Verstraete --------------------------------------------

"""
    RPVBRDF{T}(ρ0, k, Θ)

Rahman-Pinty-Verstraete (1993) semi-empirical BRDF:

```math
BRF(θ_i, θ_v, Δφ) = ρ_0 \\,
   \\bigl[\\cos θ_i \\cos θ_v (\\cos θ_i + \\cos θ_v)\\bigr]^{k-1} \\,
   F(g) \\, \\bigl[1 + R(G)\\bigr]
```

with

```math
F(g) = \\frac{1 - Θ^2}{\\bigl(1 + Θ^2 + 2Θ\\cos g\\bigr)^{3/2}},
\\qquad
R(G) = \\frac{1 - ρ_0}{1 + G},
\\qquad
G = \\sqrt{\\tan^2 θ_i + \\tan^2 θ_v - 2\\tanθ_i\\tanθ_v\\cos Δφ}
```

The three parameters separate the three things a reflectance field
actually does:

  * `ρ0` sets the overall brightness (roughly the nadir reflectance at
    nadir illumination);
  * `k ∈ (0, 1]` controls the **bowl shape** — the Minnaert exponent. For
    `k < 1` reflectance rises towards the limb, which is what a rough,
    porous surface does; `k = 1` removes the anisotropy of the
    Minnaert term entirely;
  * `Θ ∈ (-1, 1)` is the Henyey-Greenstein asymmetry. `Θ < 0` is
    backward-scattering (the normal case for vegetation and soil:
    the scatterers hide their own shadows), `Θ > 0` forward-scattering
    (snow, wet surfaces).

`R(G)` is the **hot-spot** factor. `G` is the distance between the
incident and view directions projected onto the surface plane, so `G = 0`
exactly at backscatter, where the factor reaches its maximum `2 - ρ0`,
and decays to 1 as the directions separate. Physically this is the
shadow-hiding opposition effect: looking down the illumination direction
you see no shadows, because every shadow is hidden behind the object that
casts it. It is a genuinely sharp feature — the width of `1/(1+G)` is set
by nothing but geometry — and it is the single largest source of error in
BRDF models that omit it.

RPV is *not* intrinsically energy-conserving, and for `k < 1` the
Minnaert factor diverges as `θ_i → π/2`, so the black-sky albedo grows
without bound at the horizon. The presets shipped in
[`rpv_preset`](@ref) all satisfy `α_bs ≤ 1` out to 85° incidence, which
is checked by the test suite; do not rely on the model beyond that.

Reference: Rahman, Pinty & Verstraete (1993), *J. Geophys. Res.* **98**,
20791-20801.
"""
struct RPVBRDF{T<:AbstractFloat} <: AbstractBRDF
    ρ0::T
    k::T
    Θ::T
    function RPVBRDF{T}(ρ0::Real, k::Real, Θ::Real) where {T<:AbstractFloat}
        0 <= ρ0 <= 1 || error("RPVBRDF: ρ0 must lie in [0,1], got $ρ0")
        k > 0        || error("RPVBRDF: k must be positive, got $k")
        -1 < Θ < 1   || error("RPVBRDF: Θ must lie in (-1,1), got $Θ")
        return new{T}(T(ρ0), T(k), T(Θ))
    end
end

function RPVBRDF(ρ0::Real, k::Real, Θ::Real)
    T = promote_type(typeof(float(ρ0)), typeof(float(k)), typeof(float(Θ)))
    return RPVBRDF{T}(ρ0, k, Θ)
end

function brdf(m::RPVBRDF, θ_i::Real, φ_i::Real, θ_v::Real, φ_v::Real)
    μi = _safe_cos(θ_i); μv = _safe_cos(θ_v)
    Δφ = φ_v - φ_i
    # Minnaert-like bowl term
    M = (μi * μv * (μi + μv))^(m.k - 1)
    # Henyey-Greenstein phase function at phase angle g
    cg = _phase_cosine(θ_i, θ_v, Δφ)
    denom = 1 + m.Θ^2 + 2 * m.Θ * cg
    F = (1 - m.Θ^2) / max(denom, 1e-12)^1.5
    # hot spot
    ti = sqrt(max(1 - μi^2, 0.0)) / μi
    tv = sqrt(max(1 - μv^2, 0.0)) / μv
    G = sqrt(max(ti^2 + tv^2 - 2 * ti * tv * cos(Δφ), 0.0))
    H = 1 + (1 - m.ρ0) / (1 + G)
    return m.ρ0 * M * F * H
end

Base.show(io::IO, m::RPVBRDF) =
    print(io, "RPVBRDF(ρ0=", m.ρ0, ", k=", m.k, ", Θ=", m.Θ, ")")

# Literature-range parameter sets. Broadband values; RPV parameters are
# in principle wavelength-dependent (ρ0 most strongly, k and Θ weakly),
# and a spectrally resolved application should scale ρ0 by the band
# albedo ratio, which is what `BRDFSurface` does.
const _RPV_PRESETS = Dict{Symbol, NTuple{3, Float64}}(
    :grass  => (0.12, 0.65, -0.15),
    :forest => (0.08, 0.55, -0.20),
    :soil   => (0.25, 0.75, -0.10),
    :sand   => (0.35, 0.85, -0.08),
    :snow   => (0.80, 0.96,  0.10),
    :urban  => (0.15, 0.80, -0.05),
)

"""
    rpv_preset(name) -> RPVBRDF

Representative RPV parameters. `name` is one of `:grass`, `:forest`,
`:soil`, `:sand`, `:snow`, `:urban`.

The trend across the presets is the physical story: dark, strongly
structured canopies have small `ρ0`, small `k` (pronounced bowl shape)
and strongly negative `Θ` (shadow hiding); bright, smooth, weakly
structured surfaces move towards `k → 1` and `Θ → 0`, and snow crosses
into mild forward scattering.
"""
function rpv_preset(name::Symbol)
    haskey(_RPV_PRESETS, name) ||
        error("rpv_preset: unknown preset $name; available: " *
              "$(sort(collect(keys(_RPV_PRESETS))))")
    ρ0, k, Θ = _RPV_PRESETS[name]
    return RPVBRDF(ρ0, k, Θ)
end

# --- Ross-Li (MODIS operational kernels) ---------------------------------

"""
    ross_thick_kernel(θi, θv, Δφ) -> K_vol

Ross-thick volume-scattering kernel (Roujean et al. 1992; Wanner et al.
1995):

```math
K_{vol} = \\frac{(π/2 - g)\\cos g + \\sin g}{\\cos θ_i + \\cos θ_v}
          - \\frac{π}{4}
```

where `g` is the phase angle. It is the single-scattering radiative
transfer solution for a *dense, optically thick* layer of randomly
oriented, equally transmitting and reflecting facets — a leaf canopy with
a large leaf-area index. It is normalised so that it vanishes for nadir
view and nadir illumination, which makes the three-kernel model's `f_iso`
directly interpretable as the nadir-nadir reflectance.

`K_vol` peaks at backscatter, so it carries part of the hot spot, but its
peak is broad: the sharp part of a real hot spot comes from the geometric
kernel.
"""
function ross_thick_kernel(θi::Real, θv::Real, Δφ::Real)
    μi = _safe_cos(θi); μv = _safe_cos(θv)
    cg = clamp(_phase_cosine(θi, θv, Δφ), -1.0, 1.0)
    g  = acos(cg)
    return ((π / 2 - g) * cg + sin(g)) / (μi + μv) - π / 4
end

"""
    li_sparse_kernel(θi, θv, Δφ; br = 1, hb = 2) -> K_geo

Li-sparse-**reciprocal** geometric-optical kernel (Li & Strahler 1992;
Wanner et al. 1995; Lucht, Schaaf & Strahler 2000):

```math
K_{geo} = O(θ_i', θ_v', Δφ) - \\sec θ_i' - \\sec θ_v'
          + \\tfrac{1}{2}\\bigl(1 + \\cos ξ'\\bigr)
            \\sec θ_i' \\sec θ_v'
```

with the mutual-shadowing overlap area

```math
O = \\frac{1}{π}\\,(t - \\sin t \\cos t)\\,
    (\\sec θ_i' + \\sec θ_v'),
\\qquad
\\cos t = \\frac{h}{b}\\,
  \\frac{\\sqrt{D^2 + (\\tanθ_i'\\tanθ_v'\\sin Δφ)^2}}
        {\\sec θ_i' + \\sec θ_v'},
```

```math
D = \\sqrt{\\tan^2θ_i' + \\tan^2θ_v' - 2\\tanθ_i'\\tanθ_v'\\cosΔφ}
```

and `tan θ' = (b/r) tan θ`. The model idealises the scene as spheroids of
vertical radius `b` and horizontal radius `r` whose centres sit at height
`h`, casting shadows on a Lambertian background; `O` is the area in which
the illuminated and viewed shadow ellipses overlap, which is exactly the
shadow-hiding that produces the hot spot. MODIS fixes the crown shape at
`b/r = 1` (spherical crowns) and `h/b = 2`, so the kernel has no free
parameters and the whole angular model is linear in the three weights.

Two details that are routinely got wrong and are handled here:

  * `cos t` must be **clamped to [-1, 1]**. Outside that range the
    ellipses do not overlap at all and the correct value is `O = 0`; a
    naive `acos` returns `NaN` and silently poisons the retrieval.
  * The last term carries `sec θ_i' sec θ_v'` (both directions), not
    `sec θ_v'` alone. That product is what makes this the *reciprocal*
    variant: swapping illumination and view leaves `K_geo` unchanged, as
    it must. The non-reciprocal variant is still occasionally published
    and is wrong for this purpose.
"""
function li_sparse_kernel(θi::Real, θv::Real, Δφ::Real;
                          br::Real = 1.0, hb::Real = 2.0)
    # crown-shape-corrected angles: tan θ' = (b/r) tan θ
    ti = br * tan(min(θi, π / 2 - 1e-9))
    tv = br * tan(min(θv, π / 2 - 1e-9))
    θi′ = atan(ti); θv′ = atan(tv)
    seci = 1 / _safe_cos(θi′); secv = 1 / _safe_cos(θv′)
    cΔφ = cos(Δφ); sΔφ = sin(Δφ)
    D2 = ti^2 + tv^2 - 2 * ti * tv * cΔφ
    D  = sqrt(max(D2, 0.0))
    cost = hb * sqrt(D^2 + (ti * tv * sΔφ)^2) / (seci + secv)
    cost = clamp(cost, -1.0, 1.0)
    t = acos(cost)
    O = (t - sin(t) * cost) * (seci + secv) / π
    cξ′ = cos(θi′) * cos(θv′) + sin(θi′) * sin(θv′) * cΔφ
    return O - seci - secv + 0.5 * (1 + cξ′) * seci * secv
end

"""
    RossLiBRDF{T}(f_iso, f_vol, f_geo; br = 1, hb = 2, clamp_negative = true)

The MODIS operational kernel-driven BRDF — a linear combination of an
isotropic term, the Ross-thick volume kernel and the Li-sparse-reciprocal
geometric kernel:

```math
BRF = f_{iso} + f_{vol}\\,K_{vol}(θ_i, θ_v, Δφ)
             + f_{geo}\\,K_{geo}(θ_i, θ_v, Δφ)
```

This is the model behind MCD43, the global albedo product, and its
appeal is that it is **linear in the unknowns**: three cloud-free
observations at different geometries determine `(f_iso, f_vol, f_geo)` by
least squares, and the integrated albedos then follow from the kernel
integrals, which are pure numbers independent of the scene.

Both kernels carry `sec θ`, so the model diverges at the horizon and its
black-sky albedo is not bounded by unity for `θ_i` beyond about 87°; the
shipped presets are verified to conserve energy out to 85°. MODIS itself
limits retrievals to solar zenith angles below 70° for this reason.

Because the kernels are not individually non-negative (`K_geo` in
particular is strongly negative in the forward-scattering direction), the
linear combination can extrapolate below zero at extreme geometries. With
`clamp_negative = true` (the default) the returned factor is floored at
zero, which is physical — a surface cannot reflect negatively — and
preserves reciprocity, since `max(0, ·)` of a symmetric function is
symmetric. Set it to `false` to recover the raw linear model, which is
what the published kernel-integral constants refer to.

References: Wanner, Li & Strahler (1995), *J. Geophys. Res.* **100**,
21077; Lucht, Schaaf & Strahler (2000), *IEEE TGRS* **38**, 977.
"""
struct RossLiBRDF{T<:AbstractFloat} <: AbstractBRDF
    f_iso::T
    f_vol::T
    f_geo::T
    br::T
    hb::T
    clamp_negative::Bool
    function RossLiBRDF{T}(f_iso::Real, f_vol::Real, f_geo::Real,
                           br::Real, hb::Real,
                           clamp_negative::Bool) where {T<:AbstractFloat}
        br > 0 || error("RossLiBRDF: b/r must be positive, got $br")
        hb > 0 || error("RossLiBRDF: h/b must be positive, got $hb")
        f_iso >= 0 ||
            error("RossLiBRDF: f_iso must be non-negative, got $f_iso")
        return new{T}(T(f_iso), T(f_vol), T(f_geo), T(br), T(hb),
                      clamp_negative)
    end
end

function RossLiBRDF(f_iso::Real, f_vol::Real, f_geo::Real;
                    br::Real = 1.0, hb::Real = 2.0,
                    clamp_negative::Bool = true)
    T = promote_type(typeof(float(f_iso)), typeof(float(f_vol)),
                     typeof(float(f_geo)), typeof(float(br)),
                     typeof(float(hb)))
    return RossLiBRDF{T}(f_iso, f_vol, f_geo, br, hb, clamp_negative)
end

function brdf(m::RossLiBRDF{T}, θ_i::Real, φ_i::Real,
              θ_v::Real, φ_v::Real) where {T}
    Δφ = φ_v - φ_i
    Kv = ross_thick_kernel(θ_i, θ_v, Δφ)
    Kg = li_sparse_kernel(θ_i, θ_v, Δφ; br = m.br, hb = m.hb)
    r = m.f_iso + m.f_vol * Kv + m.f_geo * Kg
    return m.clamp_negative ? max(r, zero(r)) : r
end

Base.show(io::IO, m::RossLiBRDF) =
    print(io, "RossLiBRDF(f_iso=", m.f_iso, ", f_vol=", m.f_vol,
          ", f_geo=", m.f_geo, ")")

# Broadband kernel weights of the same order as MCD43A1 retrievals.
const _ROSSLI_PRESETS = Dict{Symbol, NTuple{3, Float64}}(
    :grass  => (0.15, 0.08, 0.02),
    :forest => (0.10, 0.06, 0.02),
    :soil   => (0.25, 0.05, 0.03),
    :sand   => (0.38, 0.04, 0.02),
    :snow   => (0.85, 0.02, 0.01),
    :urban  => (0.16, 0.04, 0.05),
)

"""
    rossli_preset(name; kwargs...) -> RossLiBRDF

Representative broadband Ross-Li kernel weights for `:grass`, `:forest`,
`:soil`, `:sand`, `:snow`, `:urban`. Note that `f_geo` is largest for the
urban preset: a city is the archetypal *sparse* scene of tall objects
casting long shadows on a bright background, which is exactly what the
Li-sparse kernel describes.
"""
function rossli_preset(name::Symbol; kwargs...)
    haskey(_ROSSLI_PRESETS, name) ||
        error("rossli_preset: unknown preset $name; available: " *
              "$(sort(collect(keys(_ROSSLI_PRESETS))))")
    fi, fv, fg = _ROSSLI_PRESETS[name]
    return RossLiBRDF(fi, fv, fg; kwargs...)
end

# --- Cook-Torrance microfacet --------------------------------------------

"""
    CookTorranceBRDF{T}(ρd, F0, α)

Microfacet (Cook-Torrance 1982) BRDF for specular surfaces — glazing,
sheet metal, wet asphalt — in the modern GGX formulation:

```math
f_r = \\frac{ρ_d}{π}
    + \\frac{D(θ_h)\\,G(θ_i, θ_v)\\,F(c_d)}{4\\cosθ_i\\cosθ_v}
```

with

```math
D(θ_h) = \\frac{α^2}{π\\bigl[\\cos^2θ_h(α^2-1)+1\\bigr]^2},
\\qquad
G = G_1(θ_i)\\,G_1(θ_v),
\\qquad
G_1(θ) = \\frac{2\\cosθ}{\\cosθ + \\sqrt{α^2 + (1-α^2)\\cos^2θ}},
```

```math
F(c_d) = F_0 + (1 - F_0)(1 - c_d)^5
```

where `h` is the half vector between the incident and view directions,
`θ_h` its zenith angle, and `c_d = s·h = v·h`.

The three ingredients each fix a specific failure of a naive specular
model:

  * **D**, the Trowbridge-Reitz/GGX normal distribution, gives the
    microfacet slopes. Its long tails — decaying as `θ_h⁻⁴` rather than
    the Gaussian `exp(-θ_h²/α²)` of the original Beckmann form — are why
    real glossy surfaces have a bright halo around the specular
    highlight, and why GGX replaced Beckmann in every renderer.
  * **G**, the Smith shadowing-masking term, removes the energy that
    would be reflected by microfacets which are hidden behind their
    neighbours. Without it the model gains energy at grazing incidence.
  * **F**, Schlick's approximation to the Fresnel coefficient, takes the
    reflectance from its normal-incidence value `F0` to unity at grazing.
    This is what makes a window opaque and mirror-like when the sun is
    low, and it is the reason a glazed façade can throw more radiation at
    the street than the sky does.

`α` is the RMS microfacet slope: `α ≈ 0.02` for float glass, `0.1-0.2`
for painted or weathered metal, `0.2-0.3` for a wet road. `F0` follows
from the refractive index, `F0 = ((n-1)/(n+1))²` — 0.043 for glass
(n = 1.52), 0.021 for a water film (n = 1.34) — or is measured directly
for metals, where it is large and coloured.

Reciprocity holds exactly: `D`, `G` and the prefactor are symmetric under
`i ↔ v`, and `s·h = v·h` by construction.

Reference: Cook & Torrance (1982), *ACM Trans. Graph.* **1**, 7-24;
Walter et al. (2007) for the GGX distribution and its Smith term.
"""
struct CookTorranceBRDF{T<:AbstractFloat} <: AbstractBRDF
    ρd::T
    F0::T
    α::T
    function CookTorranceBRDF{T}(ρd::Real, F0::Real,
                                 α::Real) where {T<:AbstractFloat}
        0 <= ρd <= 1 || error("CookTorranceBRDF: ρd must lie in [0,1], got $ρd")
        0 <= F0 <= 1 || error("CookTorranceBRDF: F0 must lie in [0,1], got $F0")
        α > 0        || error("CookTorranceBRDF: α must be positive, got $α")
        return new{T}(T(ρd), T(F0), T(α))
    end
end

function CookTorranceBRDF(ρd::Real, F0::Real, α::Real)
    T = promote_type(typeof(float(ρd)), typeof(float(F0)), typeof(float(α)))
    return CookTorranceBRDF{T}(ρd, F0, α)
end

"Smith GGX masking-shadowing function for one direction."
@inline function _smith_g1(μ::Real, α::Real)
    μ <= 0 && return 0.0
    return 2 * μ / (μ + sqrt(α^2 + (1 - α^2) * μ^2))
end

"GGX / Trowbridge-Reitz microfacet normal distribution."
@inline function _ggx_d(cosθh::Real, α::Real)
    cosθh <= 0 && return 0.0
    den = cosθh^2 * (α^2 - 1) + 1
    return α^2 / (π * den^2)
end

"Schlick's approximation to the Fresnel reflectance."
@inline _schlick(F0::Real, c::Real) = F0 + (1 - F0) * (1 - clamp(c, 0.0, 1.0))^5

@inline function _direction(θ::Real, φ::Real)
    s = sin(θ)
    return (s * cos(φ), s * sin(φ), cos(θ))
end

function brdf(m::CookTorranceBRDF, θ_i::Real, φ_i::Real,
              θ_v::Real, φ_v::Real)
    μi = _safe_cos(θ_i); μv = _safe_cos(θ_v)
    s = _direction(θ_i, φ_i)
    v = _direction(θ_v, φ_v)
    hx, hy, hz = s[1] + v[1], s[2] + v[2], s[3] + v[3]
    hn = sqrt(hx^2 + hy^2 + hz^2)
    hn < 1e-12 && return m.ρd          # exactly opposed directions
    hx /= hn; hy /= hn; hz /= hn
    c_d = s[1] * hx + s[2] * hy + s[3] * hz
    D = _ggx_d(hz, m.α)
    G = _smith_g1(μi, m.α) * _smith_g1(μv, m.α)
    F = _schlick(m.F0, c_d)
    spec = D * G * F / (4 * μi * μv)
    return m.ρd + π * spec             # BRF = π f_r
end

Base.show(io::IO, m::CookTorranceBRDF) =
    print(io, "CookTorranceBRDF(ρd=", m.ρd, ", F0=", m.F0, ", α=", m.α, ")")

const _CT_PRESETS = Dict{Symbol, NTuple{3, Float64}}(
    #                      ρd     F0     α
    :glass           => (0.020, 0.043, 0.020),
    :metal_roof      => (0.200, 0.600, 0.200),
    :polished_metal  => (0.050, 0.900, 0.060),
    :wet_asphalt     => (0.040, 0.021, 0.150),
)

"""
    cook_torrance_preset(name) -> CookTorranceBRDF

Microfacet parameters for `:glass`, `:metal_roof`, `:polished_metal`,
`:wet_asphalt`. `F0` for the dielectrics is computed from the refractive
index; the metals carry a measured normal-incidence reflectance.
"""
function cook_torrance_preset(name::Symbol)
    haskey(_CT_PRESETS, name) ||
        error("cook_torrance_preset: unknown preset $name; available: " *
              "$(sort(collect(keys(_CT_PRESETS))))")
    ρd, F0, α = _CT_PRESETS[name]
    return CookTorranceBRDF(ρd, F0, α)
end

# =========================================================================
# Integrated albedos
# =========================================================================
#
# These are the quantities a radiative-transfer solver actually consumes,
# and they are computed here by integrating the BRDF — not by evaluating a
# fitted polynomial in the solar zenith angle, which is how most models
# obtain them. The difference matters: a fit is only valid for the
# parameter range it was fitted over, whereas the quadrature is correct
# for any kernel and any parameters, and it is verifiable — the Lambertian
# case must return its own albedo exactly.
#
# Definitions (Schaepman-Strub et al. 2006 use the strict nomenclature;
# the MODIS community names are given alongside):
#
#   black-sky albedo  = directional-hemispherical reflectance
#       α_bs(θ_i) = ∫_{2π} f_r(θ_i, θ_v, Δφ) cos θ_v dΩ_v
#                 = (1/π) ∫₀^{2π} ∫₀¹ BRF(θ_i, θ_v, Δφ) μ_v dμ_v dΔφ
#
#   white-sky albedo  = bihemispherical reflectance under isotropic light
#       α_ws = 2 ∫₀¹ α_bs(μ_i) μ_i dμ_i
#
#   blue-sky albedo   = the operational combination actually seen by the
#       surface, (1 - f) α_bs + f α_ws with f the diffuse fraction of the
#       incoming shortwave.
#
# The μ substitution (μ = cos θ) turns `cos θ sin θ dθ` into `μ dμ`, which
# is a polynomial weight that Gauss-Legendre handles exactly; the azimuth
# integrand is periodic, so a uniform rule converges spectrally there.

"Default (nμ, nφ) quadrature orders for a model."
_quad_hint(::AbstractBRDF)     = (48, 96)
_quad_hint(::LambertianBRDF)   = (8, 8)
_quad_hint(::RPVBRDF)          = (64, 128)
_quad_hint(::RossLiBRDF)       = (64, 128)
# The GGX lobe has angular width ~α, so the generic hemisphere rule needs
# a node spacing finer than that. `black_sky_albedo` has a specialised
# method for this model which does not; the generic path is kept for
# cross-validation.
_quad_hint(m::CookTorranceBRDF) =
    (clamp(ceil(Int, 12 / m.α), 64, 320), clamp(ceil(Int, 24 / m.α), 128, 640))

"""
    hemisphere_quadrature_albedo(model, θ_i; nμ, nφ) -> α

Directional-hemispherical reflectance of `model` at incidence `θ_i`, by
explicit `nμ × nφ` quadrature over the view hemisphere. This is the
generic path, valid for any `AbstractBRDF`; [`black_sky_albedo`](@ref)
calls it unless the model provides a better-conditioned specialisation.

Kept public so that a specialised method can be cross-checked against the
brute-force integral, which is how the Cook-Torrance implementation below
is validated.
"""
function hemisphere_quadrature_albedo(m::AbstractBRDF, θ_i::Real;
                                      nμ::Int = _quad_hint(m)[1],
                                      nφ::Int = _quad_hint(m)[2])
    xμ, wμ = _gl01(nμ)
    dφ = 2π / nφ
    acc = 0.0
    @inbounds for a in 1:nμ
        μv = xμ[a]
        θv = acos(clamp(μv, 0.0, 1.0))
        wa = wμ[a] * μv
        inner = 0.0
        for b in 0:(nφ - 1)
            φv = (b + 0.5) * dφ
            inner += brdf(m, θ_i, 0.0, θv, φv)
        end
        acc += wa * inner * dφ
    end
    return acc / π
end

"""
    black_sky_albedo(model, θ_i; kwargs...) -> α

Black-sky (directional-hemispherical) albedo: the fraction of a
collimated beam arriving at zenith angle `θ_i` that the surface reflects
back into the upward hemisphere.

Computed by numerical integration of the model's own BRDF, so it is
consistent with `brdf` by construction and not by assertion. For a
[`LambertianBRDF`](@ref) of albedo `ρ` the result is `ρ` at every angle,
to quadrature precision — this is the calibration test for everything
else in this section.
"""
black_sky_albedo(m::AbstractBRDF, θ_i::Real; kwargs...) =
    hemisphere_quadrature_albedo(m, θ_i; kwargs...)

# Exact for the Lambertian case; skips the quadrature entirely.
black_sky_albedo(m::LambertianBRDF, θ_i::Real; kwargs...) = m.ρ

"""
    black_sky_albedo(m::CookTorranceBRDF, θ_i; nψ = 64, nφ = 128) -> α

Specialised quadrature for the microfacet model, obtained by changing the
integration variable from the outgoing direction `v` to the half vector
`h` (Jacobian `dΩ_v = 4 (v·h) dΩ_h`) and then stretching the half-vector
polar angle by the roughness, `u = tan θ_h / α`, `u = tan ψ`:

```math
α_{bs}(θ_i) = ρ_d + \\frac{1}{π\\cosθ_i}
   \\int_0^{2π}\\!\\!\\int_0^{π/2}
   \\sinψ\\cosψ\\sqrt{1 + α^2\\tan^2ψ}\\;
   G\\,F\\,(s\\!\\cdot\\!h)\\;
   \\mathbb{1}[\\cosθ_v > 0]\\;dψ\\,dφ_h
```

This is the *same integral*, exactly transformed — no approximation is
introduced. What it buys is conditioning: in the original variables the
integrand is a spike of angular width `α` sitting at a location that
moves with `θ_i`, so a uniform hemisphere rule needs `O(1/α)` nodes per
dimension and silently *under*-estimates the albedo when it does not have
them. After the transformation the peak is pinned at `ψ` of order unity
and 64 × 128 nodes suffice for `α` down to 0.01. The test suite checks
the two paths against each other at a roughness where the brute-force
rule is still trustworthy.

The sanity check the transformation must pass: for a perfect mirror
(`α → 0`, `F0 = 1`) the integrand reduces to `sin ψ cos ψ / π` and the
double integral gives exactly 1.
"""
function black_sky_albedo(m::CookTorranceBRDF, θ_i::Real;
                          nψ::Int = 64, nφ::Int = 128)
    μi = clamp(cos(θ_i), 1e-9, 1.0)
    si = sqrt(max(1 - μi^2, 0.0))
    sx, sy, sz = si, 0.0, μi
    xψ, wψ = _gl01(nψ)
    dφ = 2π / nφ
    g1i = _smith_g1(μi, m.α)
    acc = 0.0
    @inbounds for a in 1:nψ
        ψ = (π / 2) * xψ[a]
        wa = wψ[a] * (π / 2)
        sψ, cψ = sin(ψ), cos(ψ)
        tψ = cψ > 1e-12 ? sψ / cψ : 1e12
        jac = sψ * cψ * sqrt(1 + (m.α * tψ)^2) / π
        jac <= 0 && continue
        θh = atan(m.α * tψ)
        sh, ch = sin(θh), cos(θh)
        inner = 0.0
        for b in 0:(nφ - 1)
            φh = (b + 0.5) * dφ
            hx = sh * cos(φh); hy = sh * sin(φh); hz = ch
            c_d = sx * hx + sy * hy + sz * hz
            c_d <= 0 && continue
            # v = 2 (s·h) h - s
            vz = 2 * c_d * hz - sz
            vz <= 0 && continue
            G = g1i * _smith_g1(vz, m.α)
            G <= 0 && continue
            inner += G * _schlick(m.F0, c_d) * c_d
        end
        acc += wa * jac * inner * dφ
    end
    return m.ρd + acc / μi
end

"""
    white_sky_albedo(model; nμi = 40, kwargs...) -> α

White-sky (bihemispherical) albedo: the reflectance of the surface under
perfectly isotropic diffuse illumination,

```math
α_{ws} = 2\\int_0^1 α_{bs}(μ_i)\\,μ_i\\,dμ_i
```

This is a genuine double integral over both hemispheres — the outer
quadrature here, the inner one inside [`black_sky_albedo`](@ref). It is
the albedo that applies under an overcast sky, and it is the quantity
that MODIS reports as the sky-independent half of MCD43.
"""
function white_sky_albedo(m::AbstractBRDF; nμi::Int = 40, kwargs...)
    xμ, wμ = _gl01(nμi)
    acc = 0.0
    @inbounds for i in eachindex(xμ)
        μi = xμ[i]
        acc += wμ[i] * 2 * μi * black_sky_albedo(m, acos(clamp(μi, 0.0, 1.0));
                                                 kwargs...)
    end
    return acc
end

white_sky_albedo(m::LambertianBRDF; kwargs...) = m.ρ

"""
    blue_sky_albedo(model, θ_i, f_diffuse; kwargs...) -> α

The albedo the surface actually presents to the atmosphere:

```math
α_{blue}(θ_i, f) = (1 - f)\\,α_{bs}(θ_i) + f\\,α_{ws}
```

`f_diffuse` is the diffuse fraction of the incoming shortwave — near 0.1
for a clear sky with a high sun, 1.0 under thick cloud. The linear
combination is exact to the extent that the diffuse sky is isotropic;
under a clear sky it is not (it is strongly peaked around the sun and
towards the horizon), which is the leading error in this formulation and
is worth a percent or two of the albedo at low sun.
"""
function blue_sky_albedo(m::AbstractBRDF, θ_i::Real, f_diffuse::Real;
                         kwargs...)
    f = clamp(f_diffuse, 0.0, 1.0)
    bs = f < 1 ? black_sky_albedo(m, θ_i; kwargs...) : 0.0
    ws = f > 0 ? white_sky_albedo(m; kwargs...) : 0.0
    return (1 - f) * bs + f * ws
end

# =========================================================================
# BRDFSurface — a BRDF as a spectrally resolved surface
# =========================================================================

"""
    BRDFSurface{T,B} <: AbstractSurfaceOptics

Adapter that turns an [`AbstractBRDF`](@ref) — which describes *angular*
structure and carries a single broadband brightness — into a full surface
optics object with per-band albedos.

The construction separates the two: the BRDF supplies the shape (and thus
the ratio of black-sky to white-sky albedo at every sun angle), while a
material's spectral curve supplies the per-band brightness. Band `b` gets

```math
α_{dir}(b, μ_0) = σ_b \\, α_{bs}(μ_0), \\qquad
α_{dif}(b)      = σ_b \\, α_{ws},
```

with `σ_b` the ratio of the material's band albedo to its solar-weighted
broadband albedo, so that the broadband albedo of the composite is
exactly the BRDF's own. Setting `material = nothing` gives a spectrally
flat surface.
"""
struct BRDFSurface{T<:AbstractFloat, B<:AbstractBRDF} <: AbstractSurfaceOptics
    model::B
    spec::SpectralGrid{T}
    lw_spec::SpectralGrid{T}
    shape::Vector{T}        # per-band spectral shape, solar-weighted mean 1
    emissivity::Vector{T}
    wsa::T                  # cached white-sky albedo of the BRDF
end

"Solar flux fraction in each shortwave band, normalised to sum to 1."
function _band_solar_weights(::Type{T}, spec::SpectralGrid) where {T}
    nb = nbands(spec)
    w = zeros(T, nb)
    for b in 1:nb, g in gpoint_range(spec, b)
        w[b] += T(spec.solar_src[g])
    end
    s = sum(w)
    s > 0 ? (w ./= s) : fill!(w, one(T) / nb)
    return w
end

"""
    BRDFSurface(model, spec = shortwave_bands(); material = nothing,
                emissivity = 0.95, lw_spec = longwave_bands())

Wrap `model` as surface optics on shortwave grid `spec`. If `material` is
a symbol from [`material_names`](@ref), its spectral curve sets the
band-to-band variation; `emissivity` may be a scalar, a material symbol
or a per-longwave-band vector.
"""
function BRDFSurface(model::AbstractBRDF,
                     spec::SpectralGrid{T} = shortwave_bands();
                     material::Union{Nothing, Symbol} = nothing,
                     emissivity = 0.95,
                     lw_spec::SpectralGrid{T} = longwave_bands(T)) where {T}
    spec.kind === :shortwave ||
        error("BRDFSurface: `spec` must be a shortwave grid")
    nb = nbands(spec)
    shape = ones(T, nb)
    if material !== nothing
        sw = _band_solar_weights(T, spec)
        ρb = Vector{T}(undef, nb)
        for b in 1:nb
            lo, hi = band_limits(spec, b)
            ρb[b] = T(band_average_reflectance(material, lo, hi))
        end
        ρbar = sum(sw .* ρb)
        ρbar > 0 || error("BRDFSurface: material $material has zero albedo")
        shape .= ρb ./ ρbar
    end
    ε = emissivity isa Symbol ? _emissivity_bands(T, emissivity, lw_spec) :
        emissivity isa Real   ? fill(T(emissivity), nbands(lw_spec)) :
                                collect(T, emissivity)
    length(ε) == nbands(lw_spec) ||
        error("BRDFSurface: emissivity vector must have $(nbands(lw_spec)) entries")
    wsa = T(white_sky_albedo(model))
    return BRDFSurface{T, typeof(model)}(model, spec, lw_spec, shape, ε, wsa)
end

function surface_albedo(s::BRDFSurface{T}, band::Integer, μ₀::Real) where {T}
    1 <= band <= nbands(s.spec) ||
        error("surface_albedo: band $band outside 1:$(nbands(s.spec))")
    θ = acos(clamp(μ₀, 0.0, 1.0))
    return clamp(s.shape[band] * T(black_sky_albedo(s.model, θ)),
                 zero(T), one(T))
end

surface_albedo(s::BRDFSurface{T}, band::Integer) where {T} =
    clamp(s.shape[band] * s.wsa, zero(T), one(T))

surface_emissivity(s::BRDFSurface, band::Integer) = s.emissivity[band]

Base.show(io::IO, s::BRDFSurface) =
    print(io, "BRDFSurface(", s.model, ", white-sky ",
          string(round(s.wsa; digits = 4)), ")")

# =========================================================================
# Fresnel water surface with Cox-Munk wave slopes
# =========================================================================
#
# Water is the surface where a fixed albedo is most obviously wrong. A
# flat water surface reflects 2.1% of an overhead beam and 35% of a beam
# arriving at 80° from the zenith — a factor of seventeen, entirely
# determined by the Fresnel equations and the refractive index. Every
# sunrise and sunset over a lake or a canal runs through that curve.
#
# Roughening the surface *reduces* the grazing-angle reflection, which is
# the counter-intuitive but decisive effect. Wind tilts the facets; the
# facets that face the sun intercept most of the beam, and for them the
# local incidence angle is smaller than the nominal solar zenith angle,
# so they sit lower on the steeply rising Fresnel curve. At 80° solar
# zenith a 15 m/s wind cuts the specular reflectance by a third. Models
# that use a flat-water Fresnel curve therefore overestimate the reflected
# shortwave over rough seas at high latitudes, precisely where the
# sun is always low.

"""
    fresnel_polarized_reflectance(θ, n) -> (R_s, R_p, cosθ_t)

Fresnel power reflectances of the two polarisation components at a
smooth interface from air (index 1) into a non-absorbing dielectric of
real refractive index `n > 1`, at incidence angle `θ` (radians):

```math
r_s = \\frac{\\cosθ - n\\cosθ_t}{\\cosθ + n\\cosθ_t},
\\qquad
r_p = \\frac{n\\cosθ - \\cosθ_t}{n\\cosθ + \\cosθ_t},
\\qquad
R_s = r_s^2,\\; R_p = r_p^2,
```

with `sin θ_t = sin θ / n` from Snell's law; `cosθ_t` (the cosine of the
refraction angle inside the medium) is returned as well because the path
length through an absorbing layer depends on it. By reciprocity the same
`R_s`, `R_p` apply to the ray leaving the medium at `θ_t`. This is the
single implementation of the Fresnel equations in the package:
[`fresnel_reflectance`](@ref) (water) and the glazing optics of
`UrbanSurface` both call it.

Limits: `R_s = R_p = ((n-1)/(n+1))²` at normal incidence, `R_p = 0` at
the Brewster angle `atan(n)`, and both tend to 1 at grazing incidence.
"""
function fresnel_polarized_reflectance(θ::Real, n::Real)
    n > 1 || throw(ArgumentError("fresnel_polarized_reflectance: n must exceed 1"))
    c = cos(clamp(θ, 0.0, π / 2))
    s = sqrt(max(1 - c^2, 0.0))
    st = s / n
    ct = sqrt(max(1 - st^2, 0.0))
    rs = (c - n * ct) / (c + n * ct)
    rp = (n * c - ct) / (n * c + ct)
    return (clamp(rs^2, 0.0, 1.0), clamp(rp^2, 0.0, 1.0), ct)
end

"""
    fresnel_reflectance(θ, n = 1.34) -> R

Unpolarised Fresnel reflectance of a smooth dielectric interface at
incidence angle `θ` (radians) for real refractive index `n`, as the mean
of the two polarisation components of
[`fresnel_polarized_reflectance`](@ref):

```math
R = \\tfrac{1}{2}\\,(R_s + R_p)
```

`n = 1.34` is the
appropriate broadband value for sea water across the solar spectrum (it
runs from 1.35 in the blue to 1.32 at 2.5 μm; the variation changes the
normal-incidence reflectance by less than 0.002 and is neglected here).

Limits: `R(0) = ((n-1)/(n+1))² = 0.0211`, and `R → 1` as `θ → π/2`. The
`r_p` component passes through zero at the Brewster angle
`θ_B = atan(n) = 53.3°`, which is why reflected skylight off water is
strongly polarised and why a polarising filter darkens it.
"""
function fresnel_reflectance(θ::Real, n::Real = 1.34)
    Rs, Rp, _ = fresnel_polarized_reflectance(θ, n)
    return clamp(0.5 * (Rs + Rp), 0.0, 1.0)
end

"""
    cox_munk_slope_variance(U10) -> σ²

Total mean-square wave slope as a function of the 10 m wind speed
(m s⁻¹), from the sun-glitter photographs of Cox & Munk (1954):

```math
σ^2 = σ_c^2 + σ_u^2 = 0.003 + 0.00512\\,U_{10}
```

`σ_c²` (crosswind) and `σ_u²` (upwind) differ by about 30% in the
original fit; the isotropic total is used here, split equally between the
two horizontal components, because the anisotropy only matters for the
azimuthal structure of the glitter pattern and not for the
hemispherically integrated albedo.

The non-zero intercept at `U10 = 0` is real: even a nominally calm
surface carries capillary ripples from swell and surface currents.
"""
cox_munk_slope_variance(U10::Real) = 0.003 + 0.00512 * max(U10, 0.0)

"""
    whitecap_fraction(U10) -> W

Fraction of the sea surface covered by breaking-wave foam (Monahan &
O'Muircheartaigh 1980):

```math
W = 2.95\\times10^{-6}\\,U_{10}^{3.52}
```

Negligible below 6 m/s (< 0.2%) and a few percent in a gale. Whitecaps
are bright — a broadband reflectance near 0.22 — so they matter for the
ocean albedo only when the wind is strong, but there they partly offset
the roughness-driven reduction of the specular term.
"""
whitecap_fraction(U10::Real) = clamp(2.95e-6 * max(U10, 0.0)^3.52, 0.0, 1.0)

"""
    FresnelWater{T} <: AbstractSurfaceOptics

Wind-roughened water surface: Cox-Munk wave-slope statistics, unpolarised
Fresnel reflection off each facet, plus whitecap and water-leaving
contributions.

Fields: `wind_speed` (m s⁻¹ at 10 m), `n_water` (real refractive index),
`whitecap_reflectance`, the per-band `water_leaving` reflectance and the
longwave `emissivity`.

The albedo of the direct beam is obtained by integrating over the facet
slope distribution rather than over the view hemisphere:

```math
α_F(θ_0) = \\frac{\\displaystyle\\int\\!\\!\\int p(z_x, z_y)\\,
                  R_F(θ_{loc})\\,\\frac{\\max(0,\\, \\hat{s}\\cdot\\hat{n})}
                                       {\\cos θ_n}\\,dz_x\\,dz_y}
                {\\displaystyle\\int\\!\\!\\int p(z_x, z_y)\\,
                  \\frac{\\max(0,\\, \\hat{s}\\cdot\\hat{n})}{\\cos θ_n}\\,
                  dz_x\\,dz_y}
```

where `p` is the Gaussian slope density, `n̂ = (-z_x, -z_y, 1)/√(1+z_x²+z_y²)`
is the facet normal, `cos θ_loc = ŝ·n̂` the local incidence angle, and the
`1/cos θ_n` converts horizontal area to true facet area. The weighting by
the projected area `ŝ·n̂` is what makes wind reduce the grazing-angle
albedo: the facets that collect the beam are the ones tilted towards the
sun, and those have the smallest local incidence angle.

Writing it as a normalised average (rather than dividing by `cos θ_0`)
guarantees `α_F ≤ max R_F ≤ 1` for any wind speed, and reduces exactly to
`R_F(θ_0)` in the flat-water limit. Facet shadowing at extreme grazing
incidence is neglected, which is the main approximation here; it becomes
significant only beyond ~85°, where the direct beam carries almost no
energy anyway.
"""
struct FresnelWater{T<:AbstractFloat} <: AbstractSurfaceOptics
    spec::SpectralGrid{T}
    lw_spec::SpectralGrid{T}
    wind_speed::T
    n_water::T
    whitecap_reflectance::T
    water_leaving::Vector{T}
    emissivity::Vector{T}
end

"""
    FresnelWater(spec = shortwave_bands(); wind_speed = 5.0, n_water = 1.34,
                 whitecap_reflectance = 0.22, emissivity = 0.96,
                 lw_spec = longwave_bands())

Construct a wind-roughened water surface on shortwave grid `spec`.
"""
function FresnelWater(spec::SpectralGrid{T} = shortwave_bands();
                      wind_speed::Real = 5.0,
                      n_water::Real = 1.34,
                      whitecap_reflectance::Real = 0.22,
                      emissivity = 0.96,
                      lw_spec::SpectralGrid{T} = longwave_bands(T)) where {T}
    spec.kind === :shortwave ||
        error("FresnelWater: `spec` must be a shortwave grid")
    wind_speed >= 0 || error("FresnelWater: wind_speed must be ≥ 0")
    n_water > 1     || error("FresnelWater: n_water must exceed 1")
    nb = nbands(spec)
    wl = Vector{T}(undef, nb)
    for b in 1:nb
        lo, hi = band_limits(spec, b)
        wl[b] = T(_band_mean(_REFLECTANCE_CURVES[:water_leaving], lo, hi,
                             SOLAR_BRIGHTNESS_TEMPERATURE))
    end
    ε = emissivity isa Real ? fill(T(emissivity), nbands(lw_spec)) :
                              collect(T, emissivity)
    return FresnelWater{T}(spec, lw_spec, T(wind_speed), T(n_water),
                           T(whitecap_reflectance), wl, ε)
end

"""
    water_albedo_direct(w::FresnelWater, θ_0; nslope = 40) -> α

Specular (Fresnel) albedo of the wind-roughened surface for a direct beam
at solar zenith angle `θ_0`, by Gauss-Legendre integration over the
two-dimensional slope distribution out to ±6σ per component. Excludes
whitecaps and the water-leaving term, which `surface_albedo` adds.
"""
function water_albedo_direct(w::FresnelWater, θ_0::Real; nslope::Int = 40)
    σ2 = cox_munk_slope_variance(w.wind_speed)
    s  = sqrt(σ2 / 2)                        # per-component RMS slope
    μ0 = cos(clamp(θ_0, 0.0, π / 2 - 1e-6))
    s0 = sqrt(max(1 - μ0^2, 0.0))
    x, wq = _gauss_legendre(nslope)
    L = 6 * s
    num = 0.0; den = 0.0
    @inbounds for a in eachindex(x)
        zx = L * x[a]
        wa = wq[a] * L
        for b in eachindex(x)
            zy = L * x[b]
            wb = wa * wq[b] * L
            p = exp(-(zx^2 + zy^2) / (2 * σ2 / 2)) / (2π * (σ2 / 2))
            secn = sqrt(1 + zx^2 + zy^2)      # 1 / cos θ_n
            cloc = (-zx * s0 + μ0) / secn     # ŝ · n̂
            cloc <= 0 && continue
            wt = wb * p * cloc * secn
            num += wt * fresnel_reflectance(acos(clamp(cloc, 0.0, 1.0)),
                                            w.n_water)
            den += wt
        end
    end
    den <= 0 && return fresnel_reflectance(θ_0, w.n_water)
    return num / den
end

"""
    water_albedo_diffuse(w::FresnelWater; nμ = 40) -> α

Bihemispherical (white-sky) specular albedo of the water surface,
`2∫₀¹ α_F(μ₀) μ₀ dμ₀`. Close to 0.066 for flat water — three times the
overhead-sun value, because an isotropic sky puts a lot of its energy at
large zenith angles where the Fresnel reflectance is high. This is the
number to use for the diffuse component of the incoming shortwave, and
using the overhead-sun value instead is a standard and easily avoided
error.
"""
function water_albedo_diffuse(w::FresnelWater; nμ::Int = 40, kwargs...)
    x, wq = _gl01(nμ)
    acc = 0.0
    @inbounds for i in eachindex(x)
        μ = x[i]
        acc += wq[i] * 2 * μ * water_albedo_direct(w, acos(μ); kwargs...)
    end
    return acc
end

"""
    surface_albedo(w::FresnelWater, band, μ₀) -> α

Total direct-beam albedo of the water surface in shortwave `band`:

```math
α = (1 - W)\\,\\bigl[α_F(θ_0) + ρ_{wl}(b)\\bigr] + W\\,ρ_{wc}
```

combining the Cox-Munk Fresnel term, the band's water-leaving (subsurface
backscatter) reflectance and the whitecap fraction `W`. At overhead sun
in the visible this gives ≈ 0.04, of which about half is water-leaving —
which is why clear ocean looks blue rather than mirror-grey from above,
and why the near-infrared bands, where the water-leaving term vanishes,
are the ones used to detect the sun glint.
"""
function surface_albedo(w::FresnelWater{T}, band::Integer, μ₀::Real) where {T}
    1 <= band <= nbands(w.spec) ||
        error("surface_albedo: band $band outside 1:$(nbands(w.spec))")
    W = T(whitecap_fraction(w.wind_speed))
    αF = T(water_albedo_direct(w, acos(clamp(μ₀, 0.0, 1.0))))
    return clamp((1 - W) * (αF + w.water_leaving[band]) +
                 W * w.whitecap_reflectance, zero(T), one(T))
end

function surface_albedo(w::FresnelWater{T}, band::Integer) where {T}
    W = T(whitecap_fraction(w.wind_speed))
    αF = T(water_albedo_diffuse(w))
    return clamp((1 - W) * (αF + w.water_leaving[band]) +
                 W * w.whitecap_reflectance, zero(T), one(T))
end

surface_emissivity(w::FresnelWater, band::Integer) = w.emissivity[band]

Base.show(io::IO, w::FresnelWater{T}) where {T} =
    print(io, "FresnelWater{$T}(U10=", w.wind_speed, " m/s, n=", w.n_water,
          ", σ²=", string(round(cox_munk_slope_variance(w.wind_speed);
                                digits = 4)), ")")

# =========================================================================
# Snow: grain size, ageing and impurities
# =========================================================================
#
# Snow albedo is not a number, it is a three-parameter family, and the
# three parameters do quite different things:
#
#   * **Grain size** controls the near-infrared. Ice is a weak absorber in
#     the visible and a strong one beyond 1.4 μm, so growing the grains —
#     which lengthens the mean photon path inside each grain — barely
#     touches the visible albedo but collapses the NIR. Fresh snow
#     (r ≈ 50 μm) and old melting snow (r ≈ 1000 μm) differ by 0.05 in the
#     visible and by 0.4 at 1.5 μm.
#
#   * **Age** acts through grain growth: vapour diffusion in the
#     temperature gradient coarsens the grains roughly as
#     `r² = r₀² + Gt` (parabolic growth), so a snowpack darkens steadily
#     for weeks after it falls even without melting.
#
#   * **Impurities** — soot, dust — control the visible. A part per
#     million by mass of black carbon, an amount invisible to the eye,
#     drops the visible albedo from 0.98 to 0.83, because it adds
#     absorption exactly where the ice has none. This is why Arctic and
#     Himalayan snow darkening is a first-order climate forcing while the
#     same soot loading in the NIR does nothing measurable.
#
# The parameterisation used here is the asymptotic radiative-transfer
# result for a semi-infinite, weakly absorbing, strongly scattering medium
# (Wiscombe & Warren 1980; Kokhanovsky & Zege 2004):
#
#       α = exp(-A √(γ_abs · r))
#
# with `γ_abs` the bulk absorption coefficient and `r` the effective grain
# radius. Two consequences are used directly, and neither is a fit:
#
#   1. Rescaling the grain size at fixed composition gives
#      `α(r) = α(r₀)^√(r/r₀)`, which needs no new constants at all — the
#      reference spectrum carries all the spectroscopy.
#   2. Adding an absorber adds to `γ_abs` *inside the square root*:
#      `α = exp(-√(ln²α_clean + s))`, with `s` proportional to the
#      impurity mixing ratio, to the grain size, and (for black carbon,
#      whose absorption cross-section scales as 1/λ) to the inverse
#      wavelength.
#
# The soot constant is calibrated against Wiscombe & Warren's Figure 8:
# at r = 50 μm, 0.1 ppmw of soot takes the visible albedo from 0.98 to
# 0.94, 1 ppmw to 0.83, and 10 ppmw to 0.56.

"Reference grain radius of the tabulated fresh-snow spectrum, μm."
const SNOW_REFERENCE_RADIUS = 50.0

"Parabolic grain-growth rate, μm² per day."
const SNOW_GROWTH_RATE = 1.0e4

"Soot absorption constant at 0.5 μm, per ppmw, at the reference radius."
const SNOW_SOOT_CONSTANT = 0.0342

"""
    snow_grain_radius(r0, age_days; growth_rate = SNOW_GROWTH_RATE) -> μm

Effective grain radius after `age_days` of metamorphism, from parabolic
growth `r² = r₀² + G t`. With the default rate, 50 μm fresh snow reaches
112 μm after a day, 230 μm after five days and 550 μm after a month.

The growth rate is strongly temperature- and gradient-dependent in
reality (it roughly doubles per 10 K near the melting point, and melt
metamorphism is far faster still); `growth_rate` is exposed so a land
model that tracks snow temperature can supply its own.
"""
function snow_grain_radius(r0::Real, age_days::Real;
                           growth_rate::Real = SNOW_GROWTH_RATE)
    r0 > 0 || error("snow_grain_radius: r0 must be positive")
    age_days >= 0 || error("snow_grain_radius: age_days must be ≥ 0")
    return sqrt(r0^2 + growth_rate * age_days)
end

"""
    SnowSurface{T} <: AbstractSurfaceOptics

Snow with grain-size, ageing and impurity dependence.

Fields: `grain_radius` (the *fresh* grain radius, μm), `age_days`,
`soot_ppmw` (black carbon mass mixing ratio in parts per million by
weight), the derived `effective_radius`, the per-band `diffuse` albedo
and the longwave `emissivity`.

Snow is also the surface with the strongest solar-zenith dependence that
still behaves itself: the direct-beam albedo follows the asymptotic
escape function,

```math
α_{dir}(μ_0) = α_{dif}^{\\,u(μ_0)}, \\qquad
u(μ_0) = \\tfrac{3}{7}\\,(1 + 2μ_0)
```

which has two properties worth stating. It is bounded in `[0,1]` for any
`μ_0`, unlike the linear-in-`1/μ₀` parameterisations that overshoot unity
for bright snow at low sun; and `2∫₀¹ u(μ)μ dμ = 1`, so the
hemispherically averaged exponent is exactly 1 and the direct albedo
integrates back to the diffuse albedo, as the definition of the white-sky
albedo requires.
"""
struct SnowSurface{T<:AbstractFloat} <: AbstractSurfaceOptics
    spec::SpectralGrid{T}
    lw_spec::SpectralGrid{T}
    grain_radius::T
    age_days::T
    soot_ppmw::T
    effective_radius::T
    diffuse::Vector{T}
    band_λ::Vector{T}
    emissivity::Vector{T}
end

"""
    SnowSurface(spec = shortwave_bands(); grain_radius = 50, age_days = 0,
                soot_ppmw = 0, growth_rate = SNOW_GROWTH_RATE,
                emissivity = 0.99, lw_spec = longwave_bands())

Build a snow surface. The reference spectrum is the tabulated
`:snow_fresh` curve at 50 μm; grain growth, ageing and soot are applied
on top of it band by band.
"""
function SnowSurface(spec::SpectralGrid{T} = shortwave_bands();
                     grain_radius::Real = SNOW_REFERENCE_RADIUS,
                     age_days::Real = 0.0,
                     soot_ppmw::Real = 0.0,
                     growth_rate::Real = SNOW_GROWTH_RATE,
                     emissivity = 0.99,
                     lw_spec::SpectralGrid{T} = longwave_bands(T)) where {T}
    spec.kind === :shortwave ||
        error("SnowSurface: `spec` must be a shortwave grid")
    soot_ppmw >= 0 || error("SnowSurface: soot_ppmw must be ≥ 0")
    r_eff = snow_grain_radius(grain_radius, age_days; growth_rate = growth_rate)
    scale = sqrt(r_eff / SNOW_REFERENCE_RADIUS)
    nb = nbands(spec)
    diffuse = Vector{T}(undef, nb)
    band_λ  = Vector{T}(undef, nb)
    for b in 1:nb
        lo, hi = band_limits(spec, b)
        α0 = clamp(band_average_reflectance(:snow_fresh, lo, hi), 1e-6, 1 - 1e-9)
        λ  = band_center_wavelength(spec, b)
        band_λ[b] = T(λ)
        # grain-size rescaling of the clean-snow albedo
        lnα = scale * log(α0)
        # soot: adds to the absorption coefficient inside the square root
        s = SNOW_SOOT_CONSTANT * (0.5 / λ) * soot_ppmw * (r_eff / SNOW_REFERENCE_RADIUS)
        diffuse[b] = T(clamp(exp(-sqrt(lnα^2 + s)), 0.0, 1.0))
    end
    ε = emissivity isa Real ? fill(T(emissivity), nbands(lw_spec)) :
                              collect(T, emissivity)
    return SnowSurface{T}(spec, lw_spec, T(grain_radius), T(age_days),
                          T(soot_ppmw), T(r_eff), diffuse, band_λ, ε)
end

"Kokhanovsky escape function; `2∫₀¹ u(μ)μ dμ = 1` exactly."
@inline _snow_escape(μ₀::Real) = (3 / 7) * (1 + 2 * clamp(μ₀, 0.0, 1.0))

"""
    snow_albedo(s::SnowSurface, band, μ₀) -> α

Direct-beam albedo of the snowpack in shortwave `band` at cosine solar
zenith angle `μ₀`.
"""
function snow_albedo(s::SnowSurface{T}, band::Integer, μ₀::Real) where {T}
    1 <= band <= nbands(s.spec) ||
        error("snow_albedo: band $band outside 1:$(nbands(s.spec))")
    return clamp(s.diffuse[band]^T(_snow_escape(μ₀)), zero(T), one(T))
end

"""
    snow_albedo_spectrum(s::SnowSurface; μ₀ = nothing) -> Vector

Per-band albedo of the snowpack — diffuse if `μ₀` is `nothing`, direct at
`μ₀` otherwise.
"""
snow_albedo_spectrum(s::SnowSurface; μ₀ = nothing) =
    μ₀ === nothing ? copy(s.diffuse) :
                     [snow_albedo(s, b, μ₀) for b in 1:nbands(s.spec)]

surface_albedo(s::SnowSurface, band::Integer, μ₀::Real) =
    snow_albedo(s, band, μ₀)
surface_albedo(s::SnowSurface, band::Integer) = s.diffuse[band]
surface_emissivity(s::SnowSurface, band::Integer) = s.emissivity[band]

Base.show(io::IO, s::SnowSurface{T}) where {T} =
    print(io, "SnowSurface{$T}(r=", string(round(s.grain_radius; digits = 1)),
          " μm, age=", string(round(s.age_days; digits = 2)), " d, r_eff=",
          string(round(s.effective_radius; digits = 1)), " μm, soot=",
          s.soot_ppmw, " ppmw, broadband ",
          string(round(broadband_albedo(s); digits = 3)), ")")

# =========================================================================
# Tiled (mosaic) surfaces
# =========================================================================

"""
    TiledSurface{T} <: AbstractSurfaceOptics

An area-weighted mosaic of surfaces — the thing an urban grid cell
actually is. A 100 m cell in a European city is perhaps 35% roof, 25%
road and pavement, 25% garden and 15% tree canopy, and each of those
reflects and emits differently.

```math
α(b, μ_0) = \\sum_i f_i\\,α_i(b, μ_0), \\qquad
ε(b)      = \\sum_i f_i\\,ε_i(b), \\qquad
\\sum_i f_i = 1
```

Linear area weighting is exact for the *reflected flux* provided the
tiles do not exchange radiation with each other. In a street canyon they
emphatically do — that is what Phase 35's radiosity solver is for — so
`TiledSurface` is the right model for a flat mosaic (a field with patches
of bare soil and grass, a flat roofscape) and the correct *input* to the
radiosity solver for a three-dimensional one, not a replacement for it.

Components may be any `AbstractSurfaceOptics`, including another
`TiledSurface`, so mosaics nest.
"""
struct TiledSurface{T<:AbstractFloat} <: AbstractSurfaceOptics
    components::Vector{AbstractSurfaceOptics}
    fractions::Vector{T}
    function TiledSurface{T}(components::AbstractVector,
                             fractions::AbstractVector) where {T<:AbstractFloat}
        length(components) == length(fractions) ||
            error("TiledSurface: $(length(components)) components but " *
                  "$(length(fractions)) fractions")
        isempty(components) &&
            error("TiledSurface: needs at least one component")
        all(f -> f >= 0, fractions) ||
            error("TiledSurface: fractions must be non-negative")
        s = sum(fractions)
        isapprox(s, 1.0; atol = 1e-8) ||
            error("TiledSurface: fractions sum to $s, expected 1")
        return new{T}(AbstractSurfaceOptics[c for c in components],
                      collect(T, fractions))
    end
end

"""
    TiledSurface(components, fractions)

Build a mosaic from a vector of surface-optics objects and their
fractional areas. The fractions must be non-negative and sum to 1 to
within `1e-8`; silently renormalising them would hide the most common
cause of a wrong urban energy balance, which is a land-cover table that
does not close.
"""
function TiledSurface(components::AbstractVector,
                      fractions::AbstractVector{<:Real})
    isempty(fractions) && error("TiledSurface: needs at least one component")
    return TiledSurface{typeof(float(first(fractions)))}(components, fractions)
end

"""
    tiled_surface(spec, pairs...; lw_spec = longwave_bands())

Convenience constructor from `material => fraction` pairs, e.g.

```julia
sw = shortwave_bands()
cell = tiled_surface(sw, :asphalt => 0.35, :concrete => 0.25,
                          :grass => 0.25, :tree_deciduous => 0.15)
broadband_albedo(cell, 0.8)
```

A `Symbol` is expanded through [`material_albedo`](@ref); an
`AbstractSurfaceOptics` is used as given, so `FresnelWater` and
`SnowSurface` components can be mixed in.
"""
function tiled_surface(spec::SpectralGrid{T}, pairs::Pair...;
                       lw_spec::SpectralGrid{T} = longwave_bands(T)) where {T}
    comps = AbstractSurfaceOptics[]
    fracs = T[]
    for (k, f) in pairs
        c = k isa Symbol ? material_albedo(k, spec; lw_spec = lw_spec) :
                           k::AbstractSurfaceOptics
        push!(comps, c); push!(fracs, T(f))
    end
    return TiledSurface(comps, fracs)
end

"Fractional areas of the tiles."
tile_fractions(t::TiledSurface) = copy(t.fractions)

"The component surfaces."
tile_components(t::TiledSurface) = t.components

function surface_albedo(t::TiledSurface{T}, band::Integer, μ₀::Real) where {T}
    acc = zero(T)
    @inbounds for i in eachindex(t.components)
        acc += t.fractions[i] * T(surface_albedo(t.components[i], band, μ₀))
    end
    return acc
end

function surface_albedo(t::TiledSurface{T}, band::Integer) where {T}
    acc = zero(T)
    @inbounds for i in eachindex(t.components)
        acc += t.fractions[i] * T(surface_albedo(t.components[i], band))
    end
    return acc
end

function surface_emissivity(t::TiledSurface{T}, band::Integer) where {T}
    acc = zero(T)
    @inbounds for i in eachindex(t.components)
        acc += t.fractions[i] * T(surface_emissivity(t.components[i], band))
    end
    return acc
end

function Base.show(io::IO, t::TiledSurface{T}) where {T}
    print(io, "TiledSurface{$T}(", length(t.components), " tiles, f=",
          string(round.(t.fractions; digits = 3)), ")")
end

# =========================================================================
# Integration with the RadiationCore contract
# =========================================================================
#
# `TwoStream.solve_sw!` takes `sfc_albedo_direct` / `sfc_albedo_diffuse`
# as per-band vectors and `solve_lw!` takes `sfc_emissivity` the same way,
# indexed by band. These three helpers produce exactly those vectors, so
# any surface defined in this module drops straight into the solver:
#
#     sfc = tiled_surface(sw, :asphalt => 0.6, :grass => 0.4)
#     solve_sw!(fluxes, solver, optics; μ₀ = 0.8,
#               sfc_albedo_direct  = direct_albedo_vector(sfc, 0.8),
#               sfc_albedo_diffuse = diffuse_albedo_vector(sfc))

"Shortwave grid a surface was built on (recursing into mosaics)."
_sw_grid(s::SpectralAlbedo) = s.spec
_sw_grid(s::BRDFSurface)    = s.spec
_sw_grid(s::FresnelWater)   = s.spec
_sw_grid(s::SnowSurface)    = s.spec
_sw_grid(t::TiledSurface)   = _sw_grid(first(t.components))

"Longwave grid a surface was built on."
_lw_grid(s::SpectralAlbedo) = s.lw_spec
_lw_grid(s::BRDFSurface)    = s.lw_spec
_lw_grid(s::FresnelWater)   = s.lw_spec
_lw_grid(s::SnowSurface)    = s.lw_spec
_lw_grid(t::TiledSurface)   = _lw_grid(first(t.components))

"""
    direct_albedo_vector(s, μ₀) -> Vector

Per-shortwave-band black-sky albedo at cosine solar zenith angle `μ₀`,
ready to pass to `solve_sw!` as `sfc_albedo_direct`.
"""
function direct_albedo_vector(s::AbstractSurfaceOptics, μ₀::Real)
    spec = _sw_grid(s)
    T = eltype(spec)
    return T[T(surface_albedo(s, b, μ₀)) for b in 1:nbands(spec)]
end

"""
    diffuse_albedo_vector(s) -> Vector

Per-shortwave-band white-sky albedo, ready to pass to `solve_sw!` as
`sfc_albedo_diffuse`.
"""
function diffuse_albedo_vector(s::AbstractSurfaceOptics)
    spec = _sw_grid(s)
    T = eltype(spec)
    return T[T(surface_albedo(s, b)) for b in 1:nbands(spec)]
end

"""
    emissivity_vector(s) -> Vector

Per-longwave-band emissivity, ready to pass to `solve_lw!` as
`sfc_emissivity`.
"""
function emissivity_vector(s::AbstractSurfaceOptics)
    lw = _lw_grid(s)
    T = eltype(lw)
    return T[T(surface_emissivity(s, b)) for b in 1:nbands(lw)]
end

"""
    broadband_albedo(s, μ₀ = nothing) -> α

Solar-weighted broadband albedo — direct-beam at `μ₀`, or white-sky if
`μ₀` is `nothing`:

```math
\\bar{α} = \\frac{\\sum_b S_b\\,α_b}{\\sum_b S_b}
```

with `S_b` the top-of-atmosphere solar irradiance in band `b` taken from
the spectral grid. This is the single number that is usually quoted for a
surface, and it is derived here rather than tabulated, so it is
guaranteed consistent with the spectral albedos the solver sees.
"""
function broadband_albedo(s::AbstractSurfaceOptics, μ₀ = nothing)
    spec = _sw_grid(s)
    T = eltype(spec)
    w = _band_solar_weights(T, spec)
    acc = zero(T)
    for b in 1:nbands(spec)
        α = μ₀ === nothing ? surface_albedo(s, b) : surface_albedo(s, b, μ₀)
        acc += w[b] * T(α)
    end
    return acc
end

"""
    broadband_emissivity(s, lw_spec = longwave_bands(); T_ref = 288) -> ε

Planck-weighted broadband longwave emissivity of a surface — the
counterpart of [`broadband_albedo`](@ref), and derived the same way
rather than tabulated, so the scalar and the per-band values cannot
disagree.

Weighted by the Planck function at `T_ref`, because emissivity is a
property of the surface but the *broadband* number depends on which
part of the spectrum the surface is actually radiating into. 288 K is
the usual terrestrial reference.

This matters most for metal roofs. Their emissivity is ~0.25 against
≥0.90 for every other urban material, and a grey treatment that
averages it away loses the reason metal roofs reach extreme daytime
temperatures: they absorb like everything else but cannot radiate.
"""
function broadband_emissivity(s::AbstractSurfaceOptics,
                              lw_spec::SpectralGrid = longwave_bands();
                              T_ref::Real = TERRESTRIAL_TEMPERATURE)
    lw_spec.kind === :longwave ||
        error("broadband_emissivity: `lw_spec` must be a longwave grid")
    num = 0.0; den = 0.0
    for b in 1:nbands(lw_spec)
        lo, hi = band_limits(lw_spec, b)
        w = planck_band(T_ref, lo, hi)
        num += w * Float64(surface_emissivity(s, b))
        den += w
    end
    return den > 0 ? num / den : 1.0
end

end # module
