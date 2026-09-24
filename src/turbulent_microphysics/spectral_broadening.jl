# Phase 45, issue #14 — spectral broadening: a bulk moment closure of the
# sub-grid supersaturation broadening, validated against the Phase 41
# superdroplet reference and published DNS.
#
# Part-file of `TurbulentMicrophysics`; included from
# src/TurbulentMicrophysics.jl.
#
# Issue #13 lets every superdroplet see its own supersaturation
# S̄ + σ_S ξ_i(t). A bulk scheme has no droplets to give an S to; what it
# can carry is the moments of the distribution those droplets would have.
# Growth is linear in w = r² (r dr/dt = G S  ⇔  dw/dt = 2G S), so the
# moments of w close on the joint (w, ξ) statistics with no assumption at
# all except for the Kelvin term and for converting w-moments back to
# radius moments. Those two need a shape, and the shape used here is a
# gamma distribution in w — the one assumption of the scheme, stated and
# tested.

export SpectralMoments, spectral_moments, gamma_moment_ratio,
       closure_radius_moment, r2_standard_deviation

# =========================================================================
# Gamma ratio Γ(k+p) / (Γ(k) k^p) — the one special function the closure needs
# =========================================================================

# ln Γ(K+p) − ln Γ(K) for K ≥ 16, from Stirling's series written as a
# difference so that no O(K ln K) terms cancel:
#   (K+p−½) ln(K+p) − (K−½) ln K − p + [φ(K+p) − φ(K)],
# φ(z) = 1/(12z) − 1/(360z³) + 1/(1260z⁵) − 1/(1680z⁷), truncation error
# below 1/(1188 z⁹) < 1e-14 at z ≥ 16.
@inline _stirling_tail(z) = (z2 = z * z;
    (1 / 12 - (1 / 360 - (1 / 1260 - 1 / (1680 * z2)) / z2) / z2) / z)

"""
    gamma_moment_ratio(k, p) -> Γ(k+p) / (Γ(k) kᵖ)

Moment factor of the gamma distribution: if `w ~ Gamma(shape k, mean μ)`
then `⟨wᵖ⟩ = μᵖ · gamma_moment_ratio(k, p)`. Tends to 1 as `k → ∞`
(a monodisperse population) and requires `k + p > 0`.

Evaluated as a logarithm: the recurrence `Γ(z+1) = zΓ(z)` shifts `k` up
to at least 16, where Stirling's series is summed as a difference of
the two log-gammas, so neither large `k` (narrow spectra, `k ~ 10⁶`) nor
small `k` loses precision. Relative accuracy ≈ 1e-14 (tested against
`Γ(3/2) = √π/2`, `Γ(1/2) = √π` and the recurrence).
"""
function gamma_moment_ratio(k::Real, p::Real)
    kk = float(k); pp = float(p)
    kk > 0 || throw(DomainError(k, "gamma_moment_ratio: shape k must be > 0"))
    kk + pp > 0 || throw(DomainError(k, "gamma_moment_ratio: needs k + p > 0"))
    isinf(kk) && return 1.0
    return exp(_log_gamma_moment_ratio(kk, pp))
end

function _log_gamma_moment_ratio(k::Float64, p::Float64)
    # Γ(k+p)/Γ(k) = Γ(K+p)/Γ(K) · Π_{j<n} (k+j)/(k+j+p),  K = k + n ≥ 16
    L = 0.0
    K = k
    while K < 16
        L += log(K) - log(K + p)
        K += 1
    end
    # subtract p ln k: p ln(K+p) − p ln k = p ln((K+p)/k)
    return L + (K - 0.5) * log1p(p / K) + p * log((K + p) / k) - p +
           (_stirling_tail(K + p) - _stirling_tail(K))
end

# =========================================================================
# The bulk state: moments of w = r² and its covariance with the fluctuation
# =========================================================================

"""
    SpectralMoments

Bulk (moment) description of a cloud-droplet population subject to
sub-grid supersaturation fluctuations — the state a two-moment-plus-width
bulk scheme carries in place of individual droplets.

  * `N` — number of real droplets in the host cell (conserved: no
          activation or collection here)
  * `μ` — mean squared radius `⟨r²⟩` (m²)
  * `V` — variance of the squared radius `Var(r²)` (m⁴)
  * `C` — covariance `Cov(r², ξ)` of the squared radius with the
          unit-variance sub-grid fluctuation (m²)

`C` is what makes the scheme non-Markovian in the right way: droplets
that have spent time in moist sub-volumes are larger *and* still sit in
moist sub-volumes for about a correlation time, so their growth advantage
persists. For an Ornstein-Uhlenbeck `ξ` the three moment equations are
closed and exact (see [`bulk_condensation_step!`](@ref)); only the Kelvin
term and radius diagnostics need the shape closure
([`closure_radius_moment`](@ref)).
"""
mutable struct SpectralMoments
    N::Float64
    μ::Float64
    V::Float64
    C::Float64
end

"""
    spectral_moments(sd; fl = nothing, indices = nothing) -> SpectralMoments

Multiplicity-weighted moments of `w = r²` of a superdroplet population
(all droplets, or `indices`). With a [`SupersaturationFluctuations`](@ref)
`fl`, `C` is the sample covariance with `fl.ξ`; otherwise `C = 0`, which
is exact for fluctuations drawn independently of the radii.
"""
function spectral_moments(sd::SuperdropletSet;
                          fl::Union{Nothing,SupersaturationFluctuations} = nothing,
                          indices::Union{Nothing,AbstractVector{Int}} = nothing)
    idx = indices === nothing ? eachindex(sd.radius) : indices
    N = 0.0; s1 = 0.0
    @inbounds for i in idx
        N += sd.ξ[i]; s1 += sd.ξ[i] * float(sd.radius[i])^2
    end
    N > 0 || throw(ArgumentError("spectral_moments: empty population"))
    μ = s1 / N
    V = 0.0; C = 0.0; ξ̄ = 0.0
    if fl !== nothing
        @inbounds for i in idx
            ξ̄ += sd.ξ[i] * fl.ξ[i]
        end
        ξ̄ /= N
    end
    @inbounds for i in idx
        d = float(sd.radius[i])^2 - μ
        V += sd.ξ[i] * d^2
        fl === nothing || (C += sd.ξ[i] * d * (fl.ξ[i] - ξ̄))
    end
    return SpectralMoments(N, μ, V / N, C / N)
end

"Gamma shape parameter `k = μ²/V` of the closure (`Inf` if monodisperse)."
_shape(m::SpectralMoments) = m.V > 0 ? m.μ^2 / m.V : Inf

"""
    closure_radius_moment(m::SpectralMoments, p) -> ⟨rᵖ⟩

Radius moment implied by the bulk state under the **gamma closure**:
`w = r²` is taken to be gamma-distributed with mean `μ` and variance `V`,
so `⟨rᵖ⟩ = ⟨w^{p/2}⟩ = μ^{p/2} Γ(k+p/2)/(Γ(k) k^{p/2})` with `k = μ²/V`
([`gamma_moment_ratio`](@ref)). `p = 2` returns `μ` exactly, whatever the
shape, because growth is linear in `w`; every other power is an
assumption. For a spectrum of relative dispersion `ε` the difference
from a log-normal of the same `μ, V` is `O(ε⁴)` in `⟨r⟩`.
"""
function closure_radius_moment(m::SpectralMoments, p::Real)
    p == 2 && return m.μ
    return m.μ^(p / 2) * gamma_moment_ratio(_shape(m), p / 2)
end

"Number-weighted mean radius of the bulk state (gamma closure), m."
mean_radius(m::SpectralMoments) = closure_radius_moment(m, 1)

"""
    relative_dispersion(m::SpectralMoments) -> (σ_r, ε)

Radius standard deviation and relative dispersion `ε = σ_r/r̄` of the bulk
state — the same diagnostic `Superdroplet.relative_dispersion` measures
on the ensemble. Uses `⟨r²⟩ = μ` exactly, so
`σ_r² = μ (1 − R²)` with `R = Γ(k+½)/(Γ(k)√k)`, evaluated with `expm1`
to keep narrow spectra (`k ≫ 1`, `1 − R² ≈ 1/(4k)`) accurate.
"""
function relative_dispersion(m::SpectralMoments)
    k = _shape(m)
    isinf(k) && return (0.0, 0.0)
    L = _log_gamma_moment_ratio(k, 0.5)
    var = m.μ * max(-expm1(2L), 0.0)
    r̄ = sqrt(m.μ) * exp(L)
    σ = sqrt(var)
    return (σ, σ / r̄)
end

"Effective radius `⟨r³⟩/⟨r²⟩` of the bulk state (gamma closure), m."
effective_radius(m::SpectralMoments) = closure_radius_moment(m, 3) / m.μ

"""
    r2_standard_deviation(m::SpectralMoments) -> σ_{r²}
    r2_standard_deviation(sd::SuperdropletSet; indices = nothing) -> σ_{r²}

Standard deviation of the squared radius, `σ_{r²} = √Var(r²)` (m²) — the
spectral-width measure DNS reports (Lanotte et al. 2009; Sardina et al.
2015), because it is the one that turbulent condensation drives directly:
`dr²/dt = 2G S` makes it a random walk in `r²`. Not a closure quantity:
exact for both the bulk state and the ensemble.
"""
r2_standard_deviation(m::SpectralMoments) = sqrt(max(m.V, 0.0))
r2_standard_deviation(sd::SuperdropletSet; indices = nothing) =
    r2_standard_deviation(spectral_moments(sd; indices = indices))
