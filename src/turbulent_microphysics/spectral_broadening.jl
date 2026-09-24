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

# Why a ratio routine and not lgamma(k+p) − lgamma(k): for narrow spectra
# k ~ 1e6, lgamma ~ 1.3e7, and a difference of two log-gammas that are
# each good to 1e-15 relative leaves ~1e-9 absolute error in a quantity
# (1 − R², R = Γ(k+½)/(Γ(k)√k)) of size 1/(4k) ~ 2.5e-7 — a 1 % error in
# ε². Summing the Stirling series as a difference avoids that. (P3's
# gamma functions, issue #1, are not on the base branch; once both are
# merged this ratio belongs next to them.)
#
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

# =========================================================================
# Bulk condensation with sub-grid fluctuations: the moment equations
# =========================================================================
#
# Droplet i: dw_i/dt = 2G (S̄ + σ_S ξ_i − a w_i^{-1/2})   (Raoult dropped)
#            dξ_i    = −ξ_i dt/τ + √(2/τ) dW_i           (unit OU)
#
# Taking ensemble moments, with ⟨ξ⟩ = 0 and ⟨ξ²⟩ = 1 (stationary):
#
#   dμ/dt = 2G (S̄ − a⟨w^{-1/2}⟩)
#   dV/dt = 4G σ_S C − 4G a Cov(w, w^{-1/2})
#   dC/dt = 2G σ_S − C/τ − 2G a Cov(w^{-1/2}, ξ)
#
# w carries no white noise itself (ξ is continuous), so there is no Itô
# term. Without the Kelvin term these are exact and closed, and for an
# initially uncorrelated population give V − V₀ = (2G)² σ_S² · 2τ[t −
# τ(1 − e^{−t/τ})] — `supersaturation_integral_variance`, the #13 result.
# The Kelvin terms need the shape: ⟨w^{-1/2}⟩ and Cov(w, w^{-1/2}) come
# from the gamma closure, and Cov(w^{-1/2}, ξ) ≈ −½ μ^{-3/2} C is its
# linearisation about μ. At S̄ = 0.2 % and r = 5 μm, a/r ≈ 0.024 % —
# an eighth of the drive, so the terms matter for r̄ and are small
# corrections to the width.
#
# The Raoult term b/r³ is dropped: for an activated droplet with 1e-18 kg
# NaCl it is 5.2e-6 at 3 μm and 1.1e-6 at 5 μm, two to three orders below S̄. This scheme is
# for activated cloud droplets; activation itself is `sgs_activation`.

export bulk_condensation_step!, parcel_ascent!

# ⟨w^{-1/2}⟩ and Cov(w, w^{-1/2}) under the gamma closure. For k ≤ ½
# (ε ≳ 0.35 in r, broader than any condensational spectrum) ⟨w^{-1/2}⟩
# diverges for the gamma shape; fall back to the monodisperse value.
function _kelvin_moments(m::SpectralMoments)
    k = _shape(m)
    sμ = sqrt(m.μ)
    (isinf(k) || k <= 0.5) && return (1 / sμ, 0.0)
    Rm = gamma_moment_ratio(k, -0.5)
    Rp = gamma_moment_ratio(k, 0.5)
    return (Rm / sμ, sμ * (Rp - Rm))
end

# (h − (1 − e^{−λh})/λ)/λ = ∫₀ʰ (1 − e^{−λs})/λ ds, stable for λh → 0
function _ou_double_integral(λ::Float64, h::Float64)
    x = λ * h
    abs(x) < 1e-4 && return h^2 * (0.5 - x / 6 + x^2 / 24)
    return (x + expm1(-x)) / λ^2
end

# Real droplets' ⟨r³⟩ × N — the liquid water of the bulk state, per 4πρ_w/3
_volume_moment(m::SpectralMoments) = m.N * closure_radius_moment(m, 3)

"""
    bulk_condensation_step!(m, cell, dt; σ_S = 0, τ = Inf, nsub = 0,
                            mc, c) -> (Δm, Δq_vap, ΔT)

Bulk counterpart of [`sgs_condensation_step!`](@ref): advance the moment
state `m` ([`SpectralMoments`](@ref)) of the droplets in `cell` over `dt`
under sub-grid supersaturation fluctuations of standard deviation `σ_S`
and Lagrangian correlation time `τ` (the `σ_S`, `τ` of a
[`SupersaturationFluctuations`](@ref); `σ_S` is typically
[`SubgridCloud`](@ref)`.σ_S` of issue #12), two-way coupled to the cell.

The moment equations for `w = r²` under the Ornstein-Uhlenbeck
fluctuation (Paoli & Shariff 2009; Sardina et al. 2015),

```math
\\dot μ = 2G(\\bar S - a⟨w^{-1/2}⟩),\\quad
\\dot V = 4Gσ_S C - 4Ga\\,\\mathrm{Cov}(w, w^{-1/2}),\\quad
\\dot C = 2Gσ_S - C/τ + Ga\\,μ^{-3/2} C,
```

are integrated with `C` exact over each sub-step (it is linear) and `S̄`,
`T`, `p` frozen — the same operator split as the superdroplet step. `G`
is the Phase 41 growth factor `1/(F_k + F_d)` and `a` the Phase 41
Kelvin coefficient, so the bulk and the superdroplet scheme differ only
in representing the spectrum by moments. Without the Kelvin term the
equations are exact and give `supersaturation_integral_variance`.

The condensed mass is `(4π/3)ρ_w N Δ⟨r³⟩` with `⟨r³⟩` from the gamma
closure, and goes to the cell through Phase 41's
`apply_condensate_tendency!`, so the cell's total water is conserved to
round-off. Sub-steps: at most `0.2 τ_phase` (as `condensation_step!`)
and, when `σ_S > 0`, `τ/10`.

Limits: activated droplets only (no Raoult term, no activation or
complete evaporation — `μ` must stay positive); the fluctuation is
prescribed, not depleted by the droplets it drives (see the in-cloud
caveat on the theory page).
"""
function bulk_condensation_step!(m::SpectralMoments, cell::EulerianCell, dt::Real;
                                 σ_S::Real = 0.0, τ::Real = Inf, nsub::Integer = 0,
                                 mc::MicrophysicalConstants = DEFAULT_MICRO,
                                 c::ThermodynamicConstants = DEFAULT_CONSTANTS)
    σ_S >= 0 || throw(ArgumentError("bulk_condensation_step!: σ_S must be ≥ 0"))
    τ > 0 || throw(ArgumentError("bulk_condensation_step!: τ must be > 0"))
    Δt = float(dt); σ = float(σ_S); τf = float(τ)
    m.N > 0 || return (0.0, 0.0, 0.0)
    ns = nsub > 0 ? Int(nsub) : _bulk_substeps(m, cell, Δt, σ, τf, mc, c)
    h = Δt / ns
    fac = mc.ρ_water * (4π / 3)
    m_air = cell_air_mass(cell)

    Δm_tot = 0.0; Δq_tot = 0.0; ΔT_tot = 0.0
    M3 = _volume_moment(m)
    for _ in 1:ns
        Tk = cell.temperature
        p = cell_pressure(cell, 0.0, c)
        S̄ = cell_supersaturation(cell, c)
        G = 1 / (fk_thermal(Tk, mc, c) + fd_vapor(Tk, p, mc, c))
        a = kelvin_a(Tk, mc, c)
        invsq, cov_kel = _kelvin_moments(m)

        # C: linear, dC/dt = b − λC, integrated exactly; ∫C dt feeds V
        b = 2G * σ
        λ = (isfinite(τf) ? 1 / τf : 0.0) - G * a * m.μ^(-1.5)
        E1 = abs(λ * h) < 1e-12 ? h : -expm1(-λ * h) / λ     # ∫₀ʰ e^{−λs} ds
        intC = m.C * E1 + b * _ou_double_integral(λ, h)
        m.V = max(m.V + 4G * σ * intC - 4G * a * cov_kel * h, 0.0)
        m.C = m.C * exp(-λ * h) + b * E1
        m.μ += 2G * h * (S̄ - a * invsq)
        m.μ > 0 || throw(DomainError(m.μ,
            "bulk_condensation_step!: droplets evaporated completely; " *
            "the moment scheme covers activated droplets only"))

        M3n = _volume_moment(m)
        Δm = fac * (M3n - M3)
        M3 = M3n
        Δq, ΔT = apply_condensate_tendency!(cell, Δm, fac * M3 / m_air, c)
        Δm_tot += Δm; Δq_tot += Δq; ΔT_tot += ΔT
    end
    return (Δm_tot, Δq_tot, ΔT_tot)
end

# Sub-step count: Phase 41's τ_phase rule (h ≤ 0.2 τ_phase, from the
# droplet-radius sum N⟨r⟩) and, when fluctuations are on, issue #13's
# h ≤ τ/10 — the same two helpers the superdroplet steps call, so the
# bulk and the reference cannot drift apart in their sub-stepping.
_bulk_substeps(m::SpectralMoments, cell::EulerianCell, Δt::Float64,
               σ::Float64, τ::Float64,
               mc::MicrophysicalConstants, c::ThermodynamicConstants) =
    max(_phase_substeps(m.N * mean_radius(m), cell, Δt, mc, c),
        _ou_substeps(σ, τ, Δt))

"""
    parcel_ascent!(cell, w, dt; q_liq = 0, c = DEFAULT_CONSTANTS) -> cell

Lift a fixed-mass [`EulerianCell`](@ref) adiabatically by `w dt` (m):
dry-adiabatic cooling `ΔT = −(g/c_pm) w dt`
(`MoistThermodynamics.dry_adiabatic_lapse_rate`), hydrostatic pressure
fall `Δp = −ρ g w dt`, and the density that follows from the gas law at
the new `(T, p)`. The volume expands by the inverse density ratio, so the
air mass `ρ ΔV` — the invariant denominator of the Phase 41 coupling — is
unchanged, and so are the specific humidities.

Latent heating is not here: it is added by the condensation step that
follows (`apply_condensate_tendency!`), and the pair together is a
saturated adiabat. Without droplets the supersaturation rises at
`AerosolCloud.supersaturation_source(T, p) · w` (tested).
"""
function parcel_ascent!(cell::EulerianCell, w::Real, dt::Real;
                        q_liq::Real = 0.0,
                        c::ThermodynamicConstants = DEFAULT_CONSTANTS)
    dz = float(w) * float(dt)
    q = PhasePartition(cell.q_vap + float(q_liq), float(q_liq), 0.0)
    p = cell_pressure(cell, q_liq, c)
    T_new = cell.temperature - dry_adiabatic_lapse_rate(q, c) * dz
    p_new = p - cell.density * c.grav * dz
    ρ_new = p_new / (gas_constant_moist(q, c) * T_new)
    cell.volume *= cell.density / ρ_new
    cell.density = ρ_new
    cell.temperature = T_new
    return cell
end

# =========================================================================
# The validation case: an adiabatic parcel with sub-grid fluctuations
# =========================================================================

export BroadeningCase, BroadeningHistory, run_broadening_case,
       broadening_initial_population, fit_power_law

"""
    BroadeningCase(; kwargs...)

Adiabatic-parcel spectral-broadening case: a fixed-mass cloudy parcel
ascending at `w`, its activated droplets growing in the parcel's own
supersaturation plus sub-grid fluctuations (`σ_S`, `τ`). Run it with
[`run_broadening_case`](@ref) as the superdroplet reference or as the
bulk moment scheme; both start from the same droplets and see the same
parcel dynamics, so they differ only in how the spectrum is represented.

Keyword fields (SI units):

  * `T0 = 283.0`, `p0 = 9e4`, `S0 = 0.0` — initial temperature,
    pressure and supersaturation (the parcel starts at cloud base)
  * `N = 1e8` — droplet number concentration (m⁻³; 100 cm⁻³)
  * `r0 = 5e-6`, `ε0 = 0.1` — median radius and log-normal width of the
    initial activated spectrum (`ε0 = 0` is monodisperse)
  * `solute_mass = 1e-18` — NaCl per droplet (kg)
  * `n_sd = 2000` — superdroplets (reference only)
  * `w = 0.5` — updraught (m s⁻¹)
  * `σ_S = 1e-3`, `τ = 5.0` — sub-grid supersaturation spread and its
    Lagrangian correlation time (s)
  * `t_end = 300.0`, `dt = 1.0` — duration and output/coupling step (s)
  * `seed = 1` — initial radii and fluctuation noise
"""
Base.@kwdef struct BroadeningCase
    T0::Float64 = 283.0
    p0::Float64 = 9e4
    S0::Float64 = 0.0
    N::Float64 = 1e8
    r0::Float64 = 5e-6
    ε0::Float64 = 0.1
    solute_mass::Float64 = 1e-18
    n_sd::Int = 2000
    w::Float64 = 0.5
    σ_S::Float64 = 1e-3
    τ::Float64 = 5.0
    t_end::Float64 = 300.0
    dt::Float64 = 1.0
    seed::Int = 1
end

"""
    BroadeningHistory

Time series from [`run_broadening_case`](@ref), one entry per `dt`
(including `t = 0`): `t` (s), `S` (mean supersaturation), `T` (K),
`mean_radius` (m), `ε` (relative dispersion), `σ_r2` (`√Var r²`, m²),
`total_water` (kg kg⁻¹, vapour + liquid; conserved).
"""
struct BroadeningHistory
    t::Vector{Float64}
    S::Vector{Float64}
    T::Vector{Float64}
    mean_radius::Vector{Float64}
    ε::Vector{Float64}
    σ_r2::Vector{Float64}
    total_water::Vector{Float64}
end

"""
    broadening_initial_population(case) -> (sd, cell)

The initial superdroplets and parcel of `case`: `n_sd` equal-multiplicity
superdroplets with radii `r0 exp(ε0 z)`, `z ~ N(0,1)` (seeded), in a 1 m³
cell at `(T0, p0)` with vapour at `1 + S0` times saturation. The bulk
run takes its initial moments from this very population.
"""
function broadening_initial_population(case::BroadeningCase;
                                       c::ThermodynamicConstants = DEFAULT_CONSTANTS)
    rng = MersenneTwister(case.seed)
    sd = init_monodisperse_superdroplets(case.n_sd, case.N, case.r0, 1.0;
                                         solute_mass = case.solute_mass)
    if case.ε0 > 0
        for i in eachindex(sd.radius)
            sd.radius[i] = case.r0 * exp(case.ε0 * randn(rng))
        end
    end
    # vapour at 1 + S0 times saturation; iterate once so that the density
    # (which depends on q_vap) and the density-form q_sat agree
    cell = cell_from_state(case.T0, case.p0, 0.0, 1.0, 0.0, c)
    for _ in 1:3
        qs = saturation_specific_humidity(case.T0, cell.density, c; use_density = true)
        cell = cell_from_state(case.T0, case.p0, (1 + case.S0) * qs, 1.0, 0.0, c)
    end
    return sd, cell
end

"""
    run_broadening_case(case; model = :superdroplet, mc, c) -> BroadeningHistory

Integrate the [`BroadeningCase`](@ref) with

  * `model = :superdroplet` — the Phase 41 reference: every superdroplet
    grows in its own `S̄ + σ_S ξ_i(t)` (issue #13
    `sgs_condensation_step!`, which delegates to `condensation_growth!`);
  * `model = :bulk` — the moment scheme [`bulk_condensation_step!`](@ref),
    started from [`spectral_moments`](@ref) of the same initial droplets.

Each step lifts the parcel ([`parcel_ascent!`](@ref)) and then condenses,
two-way coupled. The superdroplet run's `σ_S = 0` path is Phase 41's
`condensation_step!` bit for bit.
"""
function run_broadening_case(case::BroadeningCase; model::Symbol = :superdroplet,
                             mc::MicrophysicalConstants = DEFAULT_MICRO,
                             c::ThermodynamicConstants = DEFAULT_CONSTANTS)
    model in (:superdroplet, :bulk) || throw(ArgumentError(
        "run_broadening_case: model must be :superdroplet or :bulk"))
    sd, cell = broadening_initial_population(case; c = c)
    nsteps = round(Int, case.t_end / case.dt)
    fac = mc.ρ_water * (4π / 3)
    H = BroadeningHistory(Float64[], Float64[], Float64[], Float64[],
                          Float64[], Float64[], Float64[])
    if model === :superdroplet
        fl = supersaturation_fluctuations(sd; σ_S = case.σ_S, τ = case.τ,
                                          rng = MersenneTwister(case.seed + 1))
        qliq() = total_water_mass(sd, mc) / cell_air_mass(cell)
        record_sd!(t) = _record!(H, t, cell, mean_radius(sd),
                                 relative_dispersion(sd)[2],
                                 r2_standard_deviation(sd), qliq(), c)
        record_sd!(0.0)
        for n in 1:nsteps
            parcel_ascent!(cell, case.w, case.dt; q_liq = qliq(), c = c)
            sgs_condensation_step!(sd, cell, case.dt, fl; mc = mc, c = c)
            record_sd!(n * case.dt)
        end
    else
        m = spectral_moments(sd)
        qliq_b() = fac * _volume_moment(m) / cell_air_mass(cell)
        record_b!(t) = _record!(H, t, cell, mean_radius(m),
                                relative_dispersion(m)[2],
                                r2_standard_deviation(m), qliq_b(), c)
        record_b!(0.0)
        for n in 1:nsteps
            parcel_ascent!(cell, case.w, case.dt; q_liq = qliq_b(), c = c)
            bulk_condensation_step!(m, cell, case.dt; σ_S = case.σ_S,
                                    τ = case.τ, mc = mc, c = c)
            record_b!(n * case.dt)
        end
    end
    return H
end

function _record!(H::BroadeningHistory, t, cell::EulerianCell, r̄, ε, σr2, q_liq, c)
    push!(H.t, t); push!(H.S, cell_supersaturation(cell, c))
    push!(H.T, cell.temperature); push!(H.mean_radius, r̄)
    push!(H.ε, ε); push!(H.σ_r2, σr2); push!(H.total_water, cell.q_vap + q_liq)
    return H
end

"""
    fit_power_law(t, y) -> (exponent, prefactor)

Least-squares fit of `y = prefactor · t^exponent` in log-log space (all
`t, y > 0`). Used to measure the growth exponent of `σ_{r²}(t)` against
the `t^{1/2}` law of turbulent condensational broadening (Sardina et al.
2015).
"""
function fit_power_law(t::AbstractVector{<:Real}, y::AbstractVector{<:Real})
    length(t) == length(y) || throw(DimensionMismatch("fit_power_law"))
    length(t) >= 2 || throw(ArgumentError("fit_power_law: need ≥ 2 points"))
    all(>(0), t) && all(>(0), y) || throw(DomainError(0, "fit_power_law: t, y must be > 0"))
    x = log.(float.(t)); z = log.(float.(y))
    x̄ = sum(x) / length(x); z̄ = sum(z) / length(z)
    b = sum((x .- x̄) .* (z .- z̄)) / sum((x .- x̄) .^ 2)
    return (b, exp(z̄ - b * x̄))
end
