# Phase 43, issue #5 — prognostic ice-nucleating-particle species.
#
# Units are SI throughout, matching `AerosolCloud.ARGMode`: number in m⁻³,
# mass in kg m⁻³, radius in m, surface area in m² m⁻³.

# =========================================================================
# 1. Log-normal surface-area moments
# =========================================================================
#
# `AerosolCloud` already provides the 0th and 3rd moments above a cut-off
# (`number_above_radius`, `mass_above_radius`). Surface-site-density
# (INAS) freezing needs the 2nd. The algebra is the same: the r²-weighted
# distribution of a log-normal is log-normal with the same σ_g and a
# median shifted to r_g exp(2 ln²σ_g).

"""
    DEMOTT_CUTOFF_RADIUS = 0.25e-6

Dry radius (m) of the DeMott et al. (2010, 2015) aerosol cut-off: their
INP parameterisations are written in terms of `N_{>0.5 μm}`, the number
of particles with *diameter* above 0.5 μm, i.e. radius above 0.25 μm.
"""
const DEMOTT_CUTOFF_RADIUS = 0.25e-6

"""
    mode_surface_area(m::ARGMode) -> m² m⁻³

Total particle surface area of a log-normal mode,

```math
A = 4π N r_g^2 \\exp(2\\ln^2σ_g).
```

This is the quantity immersion-freezing INAS parameterisations
(Niemand et al. 2012; Ullrich et al. 2017) multiply by the ice-nucleation
active site density `n_s(T)`.
"""
mode_surface_area(m::ARGMode) = 4π * m.N * m.r_g^2 * exp(2 * log(m.σ_g)^2)
mode_surface_area(m::LogNormalMode) = mode_surface_area(arg_mode(m))

"""
    surface_area_above_radius(m::ARGMode, r_cut) -> m² m⁻³

Surface area carried by particles of dry radius above `r_cut`,

```math
A(>r) = \\frac{A}{2}\\,\\mathrm{erfc}\\!\\left(
        \\frac{\\ln(r/r_g)}{\\sqrt2 \\ln σ_g} - \\frac{2\\lnσ_g}{\\sqrt2}\\right),
```

the second-moment companion of `AerosolCloud.number_above_radius`
(shift 0) and `AerosolCloud.mass_above_radius` (shift `3lnσ_g/√2`). It is
exact for the log-normal.
"""
function surface_area_above_radius(m::ARGMode, r_cut::Real)
    A = mode_surface_area(m)
    r_cut <= 0 && return A
    lnσ = log(m.σ_g)
    u = log(float(r_cut) / m.r_g) / (sqrt(2.0) * lnσ) - 2 * lnσ / sqrt(2.0)
    return A * 0.5 * erfc_series(u)
end
surface_area_above_radius(m::LogNormalMode, r_cut::Real) =
    surface_area_above_radius(arg_mode(m), r_cut)

# =========================================================================
# 2. Quadrature over a log-normal for size-dependent per-particle rates
# =========================================================================
#
# Most sinks act particle by particle with a size-dependent probability
# p(r): settling out of a layer (1 - exp(-v(r)Δt/Δz)), scavenging, and
# stochastic freezing (1 - exp(-J(T) 4πr² Δt)). The removed number and
# mass are
#
#     ΔN = N ∫ p(r_g e^{lnσ z})              φ(z) dz
#     ΔM = M ∫ p(r_g e^{lnσ (z + 3 lnσ)})    φ(z) dz
#
# because the mass distribution is log-normal with median r_g e^{3ln²σ}.
# Writing each moment in its own standard-normal variable makes p ≡ c
# integrate to exactly c for every moment, so a size-independent sink
# conserves r_g to rounding. The trapezoid rule on a Gaussian is
# spectrally accurate for smooth p; p may be steep (freezing
# probabilities switch on over a few kelvin, i.e. a factor of a few in
# r), so the node spacing is fine (h = 0.025σ) and the range ±10σ.

const _QZ_H = 0.025
const _QZ = collect(-10.0:_QZ_H:10.0)
const _QW = let w = [_QZ_H * exp(-0.5 * z^2) / sqrt(2π) for z in _QZ]
    w ./ sum(w)          # normalise so ∫φ = 1 to rounding
end

# ∫ p(r) weighted by the k-th moment of the log-normal (k = 0 number,
# 2 surface, 3 mass), as a fraction of that moment.
function _moment_fraction(p, r_g::Real, σ_g::Real, k::Integer)
    lnσ = log(σ_g)
    shift = k * lnσ
    s = 0.0
    @inbounds for j in eachindex(_QZ)
        pj = Float64(p(r_g * exp(lnσ * (_QZ[j] + shift))))
        s += _QW[j] * clamp(pj, 0.0, 1.0)
    end
    return s
end
