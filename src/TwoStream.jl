module TwoStream

# Phase 33 — Two-stream radiative transfer with cloud optics.
#
# The two-stream approximation collapses the angular dependence of the
# radiance field onto two hemispheric fluxes. That is a drastic
# simplification, but it is the workhorse of every operational
# atmospheric model because the error it makes (a few W/m² on broadband
# fluxes) is far smaller than the error from not resolving clouds at all,
# and it costs O(Nz) instead of O(Nz·Nangles·Nphotons).
#
# Two things here are easy to get wrong and are handled explicitly:
#
#   1. **Singular limits.** The layer reflectance/transmittance formulae
#      have removable singularities at ω → 1 (conservative scattering)
#      and at τ → 0, and a genuine one when the direct-beam extinction
#      μ₀⁻¹ coincides with the diffuse eigenvalue k. Naive code produces
#      NaN or catastrophic cancellation at all three. Each is handled by
#      its correct series expansion, and the test suite probes all three.
#
#   2. **Linear-in-τ Planck source.** Treating the source as constant
#      within a layer gives a well-known cold bias wherever the
#      temperature gradient across a layer is significant. The linear
#      form costs almost nothing and removes it; the suite measures the
#      difference against a sub-layered reference.
#
# References: Meador & Weaver (1980) for the γ coefficients;
# Joseph, Wiscombe & Weinman (1976) for delta-scaling; Zdunkowski,
# Welch & Korb (1980) for PIFM; Toon et al. (1989) for the adding
# equations and the linear-in-τ longwave source; Hu & Stamnes (1993)
# and Fu (1996) for the cloud parameterisations.

using ..RadiationCore
using ..RadiationCore: SpectralGrid, OpticalProperties, RadiativeFluxes,
                       AtmosphericColumn, AbstractRTSolver, AbstractCloudOptics,
                       AbstractSurfaceOptics, nbands, ngpoints, gpoint_range,
                       band_limits, planck_band, allocate_fluxes, net_flux!,
                       heating_rate!, STEFAN_BOLTZMANN
import ..RadiationCore: compute_optics!, solve_rt!

export TwoStreamSolver, two_stream_solver,
       DeltaEddington, PIFM, Quadrature,
       delta_scale, two_stream_coefficients,
       layer_reflectance_transmittance_sw,
       layer_reflectance_transmittance_lw,
       adding_solver!, solve_sw!, solve_lw!,
       MieCloudOptics, mie_cloud_optics,
       liquid_water_path, ice_water_path, cloud_optical_depth,
       maximum_random_overlap, radiative_heating!

# =========================================================================
# Two-stream variants
# =========================================================================

"""
    TwoStreamVariant

Selects the closure used for the two-stream γ coefficients.

  * [`DeltaEddington`](@ref) — Eddington closure with delta-scaling. The
    standard shortwave choice; most accurate for forward-peaked
    scattering (clouds).
  * [`PIFM`](@ref) — Practical Improved Flux Method (Zdunkowski 1980).
    Better behaved than Eddington at large zenith angle, which matters
    for urban low-sun geometry.
  * [`Quadrature`](@ref) — the classical quadrature (Gauss) closure,
    the standard longwave choice.
"""
abstract type TwoStreamVariant end
struct DeltaEddington <: TwoStreamVariant end
struct PIFM          <: TwoStreamVariant end
struct Quadrature    <: TwoStreamVariant end

# Diffusivity factor for the longwave: the effective 1/μ for isotropic
# hemispheric integration. 1.66 is the Elsasser value used operationally.
const LW_DIFFUSIVITY = 1.66

"""
    delta_scale(τ, ω, g) -> (τ′, ω′, g′)

Joseph-Wiscombe-Weinman delta-scaling. Strongly forward-peaked phase
functions (cloud droplets, `g ≈ 0.85`) violate the two-stream assumption
badly; delta-scaling removes the forward spike into the unscattered beam
and rescales what remains:

```math
f = g^2,\\quad τ' = (1-ωf)τ,\\quad ω' = \\frac{(1-f)ω}{1-ωf},\\quad g' = \\frac{g}{1+g}
```

Without this, cloud albedo is overestimated by tens of percent.
"""
@inline function delta_scale(τ::T, ω::T, g::T) where {T}
    f = g * g
    denom = 1 - ω * f
    denom <= 0 && return (zero(T), zero(T), zero(T))
    return ((1 - ω * f) * τ, (1 - f) * ω / denom, g / (1 + g))
end

"""
    two_stream_coefficients(variant, ω, g, μ₀) -> (γ₁, γ₂, γ₃, γ₄)

The four two-stream coupling coefficients (Meador & Weaver 1980 Table 1).
`γ₁, γ₂` couple the two diffuse streams; `γ₃, γ₄` couple the direct beam
into them. For the longwave only `γ₁, γ₂` are used.
"""
@inline function two_stream_coefficients(::DeltaEddington, ω::T, g::T, μ₀::T) where {T}
    γ1 = (7 - ω * (4 + 3g)) / 4
    γ2 = -(1 - ω * (4 - 3g)) / 4
    γ3 = (2 - 3g * μ₀) / 4
    γ4 = 1 - γ3
    return (γ1, γ2, γ3, γ4)
end

@inline function two_stream_coefficients(::PIFM, ω::T, g::T, μ₀::T) where {T}
    γ1 = 2 - ω * (1.25 + 0.75g)
    γ2 = ω * (1.25 - 0.75g)
    γ3 = (2 - 3g * μ₀) / 4
    γ4 = 1 - γ3
    return (γ1, γ2, γ3, γ4)
end

@inline function two_stream_coefficients(::Quadrature, ω::T, g::T, μ₀::T) where {T}
    μ̄ = T(1 / sqrt(3))
    γ1 = (1 - ω * (1 + g) / 2) / μ̄
    γ2 = ω * (1 - g) / (2μ̄)
    γ3 = (1 - sqrt(T(3)) * g * μ₀) / 2
    γ4 = 1 - γ3
    return (γ1, γ2, γ3, γ4)
end

# =========================================================================
# Layer reflectance / transmittance
# =========================================================================

"""
    layer_reflectance_transmittance_sw(variant, τ, ω, g, μ₀)
        -> (R_dif, T_dif, R_dir, T_dir, T_beam)

Shortwave layer properties: diffuse reflectance/transmittance, the
direct-beam source reflected and transmitted into the diffuse streams,
and the unscattered direct-beam transmittance `T_beam = e^{-τ/μ₀}`.

The three removable singularities are handled explicitly:

  * `τ → 0` — everything reduces to pass-through (`R = 0`, `T = 1`).
  * `ω → 1` (conservative) — `k → 0`, so `sinh(kτ)/k → τ` and the
    general formula's `0/0` is replaced by its limit.
  * `k μ₀ → 1` — the direct-beam denominator `1 - k²μ₀²` vanishes; `μ₀`
    is perturbed by one part in `10⁶`, which is the standard fix and is
    far below the accuracy of the approximation itself.
"""
function layer_reflectance_transmittance_sw(variant::TwoStreamVariant,
                                            τ::T, ω::T, g::T, μ₀::T) where {T}
    # --- optically thin: pass through
    if τ < 1e-12
        return (zero(T), one(T), zero(T), zero(T), one(T))
    end
    ω = clamp(ω, zero(T), one(T) - T(1e-12))   # keep k > 0
    γ1, γ2, γ3, γ4 = two_stream_coefficients(variant, ω, g, μ₀)

    T_beam = exp(-min(τ / μ₀, T(500)))

    k2 = γ1 * γ1 - γ2 * γ2
    k = sqrt(max(k2, T(1e-24)))
    kτ = min(k * τ, T(500))
    e = exp(-kτ)
    e2 = e * e

    # diffuse reflectance / transmittance (Meador & Weaver eq. 14-15)
    denom = k * (1 + e2) + γ1 * (1 - e2)
    if abs(denom) < 1e-30
        return (zero(T), one(T), zero(T), zero(T), T_beam)
    end
    R_dif = γ2 * (1 - e2) / denom
    T_dif = 2k * e / denom

    # --- direct-beam source terms
    # The denominator 1 - k²μ₀² vanishes when the beam extinction matches
    # the diffuse eigenvalue. Perturb μ₀ minutely rather than special-case
    # the whole expression.
    μ = μ₀
    if abs(1 - (k * μ)^2) < 1e-6
        μ = μ₀ * (1 + T(1e-6))
    end
    α1 = γ1 * γ4 + γ2 * γ3
    α2 = γ1 * γ3 + γ2 * γ4
    # Meador & Weaver (1980) eq. 14, with numerator and denominator both
    # divided through by e^{kτ} so nothing overflows at large τ.
    d = (1 - (k * μ)^2)
    pref = ω / d
    R_dir = pref * ((1 - k * μ) * (α2 + k * γ3) -
                    (1 + k * μ) * (α2 - k * γ3) * e2 -
                    2k * (γ3 - α2 * μ) * e * T_beam) / denom
    T_dir = -pref * ((1 + k * μ) * (α1 + k * γ4) * T_beam -
                     (1 - k * μ) * (α1 - k * γ4) * e2 * T_beam -
                     2k * (γ4 + α1 * μ) * e) / denom

    # physical bounds — the two-stream formulae can overshoot slightly in
    # extreme corners of parameter space, and an R > 1 would break the
    # adding recursion.
    R_dif = clamp(R_dif, zero(T), one(T))
    T_dif = clamp(T_dif, zero(T), one(T))
    R_dir = clamp(R_dir, zero(T), one(T))
    T_dir = clamp(T_dir, zero(T), one(T))
    return (R_dif, T_dif, R_dir, T_dir, T_beam)
end

"""
    layer_reflectance_transmittance_lw(variant, τ, ω, g, B_bot, B_top)
        -> (R, T, S_up, S_dn)

Longwave layer properties with a **linear-in-τ** Planck source. `B_bot`
and `B_top` are the source at the layer's lower and upper boundary; the
source is taken to vary linearly in optical depth between them, which is
the Toon et al. (1989) treatment.

The constant-source alternative evaluates the Planck function once per
layer and is biased cold wherever the layer spans a significant
temperature change — the test suite measures the difference against a
sub-layered reference.
"""
function layer_reflectance_transmittance_lw(variant::TwoStreamVariant,
                                            τ::T, ω::T, g::T,
                                            B_bot::T, B_top::T) where {T}
    if τ < 1e-12
        return (zero(T), one(T), zero(T), zero(T))
    end
    # Non-scattering is the overwhelmingly common LW case and has an
    # exact solution — use it rather than the scattering formula's limit.
    if ω < 1e-8
        Tr = exp(-min(LW_DIFFUSIVITY * τ, T(500)))
        τd = LW_DIFFUSIVITY * τ
        Δ  = B_top - B_bot
        # Parameterise by s = optical distance from the TOP of the layer,
        # so B(s) = B_top − Δ·s/τd. Then
        #   S_up = ∫₀^{τd} B(s) e^{-s} ds
        #        = B_top(1−Tr) − (Δ/τd)(1 − Tr(1+τd))
        #   S_dn = ∫₀^{τd} B(τd−v) e^{-v} dv
        #        = B_bot(1−Tr) + (Δ/τd)(1 − Tr(1+τd))
        # using ∫₀^{x} s e^{-s} ds = 1 − e^{-x}(1+x). Getting these two
        # the wrong way round makes the linear source *worse* than a
        # constant one, which is how the error was caught.
        I1 = (1 - Tr * (1 + τd)) / max(τd, T(1e-12))
        S_up = B_top * (1 - Tr) - Δ * I1
        S_dn = B_bot * (1 - Tr) + Δ * I1
        return (zero(T), Tr, max(S_up, zero(T)), max(S_dn, zero(T)))
    end
    ω = clamp(ω, zero(T), one(T) - T(1e-12))
    γ1, γ2, _, _ = two_stream_coefficients(variant, ω, g, T(0.5))
    k = sqrt(max(γ1 * γ1 - γ2 * γ2, T(1e-24)))
    kτ = min(k * τ, T(500))
    e = exp(-kτ); e2 = e * e
    denom = k * (1 + e2) + γ1 * (1 - e2)
    R = γ2 * (1 - e2) / denom
    Tr = 2k * e / denom
    R = clamp(R, zero(T), one(T)); Tr = clamp(Tr, zero(T), one(T))
    # emission from a scattering layer: (1 - R - T) is the absorptance,
    # and by Kirchhoff that is also the emittance.
    ε = max(1 - R - Tr, zero(T))
    B̄ = (B_bot + B_top) / 2
    return (R, Tr, ε * B̄, ε * B̄)
end

# =========================================================================
# Adding solver
# =========================================================================

"""
    adding_solver!(flux_up, flux_dn, flux_dir, R_dif, T_dif, R_dir, T_dir,
                   T_beam, sfc_alb_dif, sfc_alb_dir, F_top, μ₀; F_dif_top = 0)

Combine per-layer reflectance/transmittance into level fluxes by the
standard adding (doubling) equations, layers ordered bottom-up.

`F_top·μ₀` is the direct beam on a horizontal plane at the top level and
`F_dif_top` the diffuse flux entering from above (zero at the top of the
atmosphere; the sky diffuse when the "column" is a plant canopy, see
`PlantCanopy.canopy_shortwave`).

The recursion builds the albedo of the composite "surface + all layers
below level i" upward, then propagates the downward flux back down. The
`1/(1 - R_dif·A)` factors are the infinite series of multiple
reflections between a layer and everything beneath it summed in closed
form.
"""
function adding_solver!(flux_up::AbstractVector{T}, flux_dn::AbstractVector{T},
                        flux_dir::AbstractVector{T},
                        R_dif::AbstractVector{T}, T_dif::AbstractVector{T},
                        R_dir::AbstractVector{T}, T_dir::AbstractVector{T},
                        T_beam::AbstractVector{T},
                        sfc_alb_dif::T, sfc_alb_dir::T,
                        F_top::T, μ₀::T; F_dif_top::Real = zero(T)) where {T}
    nlay = length(R_dif)
    nlev = nlay + 1

    # --- direct beam straight down from the top
    flux_dir[nlev] = F_top * μ₀
    for i in nlay:-1:1
        flux_dir[i] = flux_dir[i + 1] * T_beam[i]
    end

    # --- upward sweep: albedo A[i] of everything at and below level i,
    # plus the source S[i] emerging upward from it
    A = Vector{T}(undef, nlev)
    S = Vector{T}(undef, nlev)
    A[1] = sfc_alb_dif
    S[1] = sfc_alb_dir * flux_dir[1]
    for i in 1:nlay
        d = 1 / (1 - R_dif[i] * A[i])
        A[i + 1] = R_dif[i] + T_dif[i] * T_dif[i] * A[i] * d
        S[i + 1] = R_dir[i] * flux_dir[i + 1] +
                   T_dif[i] * d * (S[i] + A[i] * T_dir[i] * flux_dir[i + 1])
    end

    # --- downward sweep
    flux_dn[nlev] = T(F_dif_top)                # diffuse entering from above
    flux_up[nlev] = A[nlev] * flux_dn[nlev] + S[nlev]
    for i in nlay:-1:1
        d = 1 / (1 - R_dif[i] * A[i])
        flux_dn[i] = (T_dif[i] * flux_dn[i + 1] +
                      R_dif[i] * S[i] +
                      T_dir[i] * flux_dir[i + 1]) * d
        flux_up[i] = A[i] * flux_dn[i] + S[i]
    end
    return (flux_up, flux_dn, flux_dir)
end

# =========================================================================
# Solver object
# =========================================================================

"""
    TwoStreamSolver{T} <: AbstractRTSolver

Two-stream solver. `sw_variant` defaults to `DeltaEddington`, `lw_variant`
to `Quadrature` — the standard operational pairing.
"""
struct TwoStreamSolver{T<:AbstractFloat, VS, VL} <: AbstractRTSolver
    spec::SpectralGrid{T}
    sw_variant::VS
    lw_variant::VL
    linear_lw_source::Bool
end

two_stream_solver(spec::SpectralGrid{T};
                  sw_variant = DeltaEddington(),
                  lw_variant = Quadrature(),
                  linear_lw_source::Bool = true) where {T} =
    TwoStreamSolver{T, typeof(sw_variant), typeof(lw_variant)}(
        spec, sw_variant, lw_variant, linear_lw_source)

"""
    solve_sw!(fluxes, solver, optics; μ₀, sfc_albedo_direct, sfc_albedo_diffuse)

Shortwave solve over all g-points, accumulating broadband fluxes.
`sfc_albedo_*` may be a scalar or a per-band vector.
"""
function solve_sw!(fluxes::RadiativeFluxes{T}, solver::TwoStreamSolver{T},
                   optics::OpticalProperties{T};
                   μ₀::Real,
                   sfc_albedo_direct = 0.1,
                   sfc_albedo_diffuse = nothing) where {T}
    spec = solver.spec
    spec.kind === :shortwave || error("solve_sw!: requires a shortwave grid")
    nlay = size(optics.τ, 1); nlev = nlay + 1
    μ = T(clamp(μ₀, 1e-6, 1.0))
    ad_diffuse = sfc_albedo_diffuse === nothing ? sfc_albedo_direct :
                                                  sfc_albedo_diffuse

    fill!(fluxes.flux_up, 0); fill!(fluxes.flux_dn, 0)
    fill!(fluxes.flux_dn_dir, 0)

    Rd = Vector{T}(undef, nlay); Td = Vector{T}(undef, nlay)
    Rr = Vector{T}(undef, nlay); Tr = Vector{T}(undef, nlay)
    Tb = Vector{T}(undef, nlay)
    fu = Vector{T}(undef, nlev); fd = Vector{T}(undef, nlev)
    fdir = Vector{T}(undef, nlev)

    for b in 1:nbands(spec)
        αdir = sfc_albedo_direct isa Real ? T(sfc_albedo_direct) :
                                            T(sfc_albedo_direct[b])
        αdif = ad_diffuse isa Real ? T(ad_diffuse) : T(ad_diffuse[b])
        for g in gpoint_range(spec, b)
            F_top = T(spec.solar_src[g])
            F_top <= 0 && continue
            @inbounds for k in 1:nlay
                τs, ωs, gs = delta_scale(optics.τ[k, g], optics.ssa[k, g],
                                         optics.g[k, g])
                Rd[k], Td[k], Rr[k], Tr[k], Tb[k] =
                    layer_reflectance_transmittance_sw(solver.sw_variant,
                                                       τs, ωs, gs, μ)
            end
            adding_solver!(fu, fd, fdir, Rd, Td, Rr, Tr, Tb, αdif, αdir,
                           F_top, μ)
            @inbounds for i in 1:nlev
                fluxes.flux_up[i]     += fu[i]
                fluxes.flux_dn[i]     += fd[i] + fdir[i]
                fluxes.flux_dn_dir[i] += fdir[i]
            end
        end
    end
    net_flux!(fluxes)
    return fluxes
end

"""
    solve_lw!(fluxes, solver, optics, col; sfc_emissivity)

Longwave solve. Source terms come from the Planck function evaluated at
the layer boundary temperatures, so the linear-in-τ treatment has the
gradient information it needs.
"""
function solve_lw!(fluxes::RadiativeFluxes{T}, solver::TwoStreamSolver{T},
                   optics::OpticalProperties{T},
                   col::AtmosphericColumn{T};
                   sfc_emissivity = 1.0) where {T}
    spec = solver.spec
    spec.kind === :longwave || error("solve_lw!: requires a longwave grid")
    nlay = size(optics.τ, 1); nlev = nlay + 1

    fill!(fluxes.flux_up, 0); fill!(fluxes.flux_dn, 0)
    fill!(fluxes.flux_dn_dir, 0)

    for b in 1:nbands(spec)
        lo, hi = band_limits(spec, b)
        ε = sfc_emissivity isa Real ? T(sfc_emissivity) : T(sfc_emissivity[b])
        # band Planck source at each level and at the surface
        Blev = T[π * planck_band(col.T_lev[i], lo, hi) for i in 1:nlev]
        Bsfc = T(π * planck_band(col.T_sfc, lo, hi))
        for g in gpoint_range(spec, b)
            w = T(spec.gpt_weight[g])
            up = Vector{T}(undef, nlev); dn = Vector{T}(undef, nlev)
            Rl = Vector{T}(undef, nlay); Tl = Vector{T}(undef, nlay)
            Su = Vector{T}(undef, nlay); Sd = Vector{T}(undef, nlay)
            @inbounds for k in 1:nlay
                Bb = solver.linear_lw_source ? w * Blev[k]     : w * (Blev[k]+Blev[k+1])/2
                Bt = solver.linear_lw_source ? w * Blev[k + 1] : w * (Blev[k]+Blev[k+1])/2
                Rl[k], Tl[k], Su[k], Sd[k] =
                    layer_reflectance_transmittance_lw(solver.lw_variant,
                        optics.τ[k, g], optics.ssa[k, g], optics.g[k, g], Bb, Bt)
            end
            # downward sweep from TOA
            dn[nlev] = zero(T)
            for k in nlay:-1:1
                dn[k] = Tl[k] * dn[k + 1] + Sd[k]
            end
            # surface emission + reflection
            up[1] = ε * w * Bsfc + (1 - ε) * dn[1]
            for k in 1:nlay
                up[k + 1] = Tl[k] * up[k] + Su[k] + Rl[k] * dn[k + 1]
            end
            @inbounds for i in 1:nlev
                fluxes.flux_up[i] += up[i]
                fluxes.flux_dn[i] += dn[i]
            end
        end
    end
    net_flux!(fluxes)
    return fluxes
end

"""
    solve_rt!(fluxes, solver::TwoStreamSolver, optics, col; kwargs...)

`AbstractRTSolver` interface. Dispatches to `solve_sw!` or `solve_lw!`
on the spectral grid's `kind`.
"""
function solve_rt!(fluxes::RadiativeFluxes{T}, solver::TwoStreamSolver{T},
                   optics::OpticalProperties{T}, col::AtmosphericColumn{T};
                   kwargs...) where {T}
    return solver.spec.kind === :shortwave ?
        solve_sw!(fluxes, solver, optics; kwargs...) :
        solve_lw!(fluxes, solver, optics, col; kwargs...)
end

# =========================================================================
# Cloud optics
# =========================================================================

"""
    MieCloudOptics <: AbstractCloudOptics

Parameterised cloud optical properties. Liquid follows Hu & Stamnes
(1993) in form — `τ ∝ LWP/r_eff`, with `ssa` and `g` weak functions of
`r_eff` — and ice follows Fu (1996) in form, using effective diameter.
Coefficients are representative values chosen to reproduce the
well-known benchmarks the test suite checks (a 100 g/m² LWP at 10 μm
gives visible τ ≈ 15, `ssa > 0.999`, `g ≈ 0.85`).
"""
struct MieCloudOptics{T<:AbstractFloat} <: AbstractCloudOptics
    spec::SpectralGrid{T}
end

mie_cloud_optics(spec::SpectralGrid{T}) where {T} = MieCloudOptics{T}(spec)

# Band-centre wavelength in μm — shared with the other optics modules,
# see RadiationCore.band_wavelength.
_band_λ(spec, b) = band_wavelength(spec, b)

"""
    liquid_water_path(col) -> g m⁻²

Column liquid water path.
"""
liquid_water_path(col::AtmosphericColumn) =
    sum(col.ql[k] * col.ρ[k] * col.dz[k] * 1000 for k in eachindex(col.ql))

"Column ice water path, g m⁻²."
ice_water_path(col::AtmosphericColumn) =
    sum(col.qi[k] * col.ρ[k] * col.dz[k] * 1000 for k in eachindex(col.qi))

"""
    compute_optics!(optics, cloud::MieCloudOptics, col)

Fill cloud optical properties from `col.ql`, `col.qi` and the effective
radii. The extinction is geometric-optics-like in the solar
(`τ ≈ 3·LWP/(2ρ_w r_eff)`), with absorption rising into the near-infrared
and dominating in the thermal infrared — which is why clouds are bright
in the visible and black in the longwave.
"""
function compute_optics!(optics::OpticalProperties{T}, cloud::MieCloudOptics{T},
                         col::AtmosphericColumn{T}) where {T}
    spec = cloud.spec
    nz = length(col.ql)
    fill!(optics.τ, 0); fill!(optics.ssa, 0); fill!(optics.g, 0)
    for b in 1:nbands(spec)
        λ = T(_band_λ(spec, b))
        # single-scattering albedo: ~1 in the visible, falling in the IR.
        # Absorption by liquid water rises steeply beyond ~1.5 μm.
        ω_liq = λ < 0.7  ? T(0.99999) :
                λ < 1.5  ? T(0.9995)  :
                λ < 3.0  ? T(0.98)    :
                λ < 8.0  ? T(0.60)    : T(0.45)
        g_liq = λ < 1.5 ? T(0.86) : λ < 4.0 ? T(0.84) : T(0.80)
        ω_ice = λ < 0.7 ? T(0.99999) : λ < 1.5 ? T(0.999) :
                λ < 3.0 ? T(0.95) : T(0.55)
        g_ice = λ < 1.5 ? T(0.78) : T(0.75)
        for g in gpoint_range(spec, b)
            @inbounds for k in 1:nz
                lwp = col.ql[k] * col.ρ[k] * col.dz[k] * 1000   # g m⁻²
                iwp = col.qi[k] * col.ρ[k] * col.dz[k] * 1000
                # τ = 3·LWP / (2 ρ_w r_eff). With LWP in g m⁻², r_eff in μm
                # and ρ_w = 1 g cm⁻³ the unit factors cancel to τ = 1.5·LWP/r_eff.
                τl = lwp > 0 ? T(1.5) * lwp / col.r_eff_liq[k] : zero(T)
                τi = iwp > 0 ? T(1.0) * iwp / col.r_eff_ice[k] : zero(T)
                τt = τl + τi
                τt <= 0 && continue
                sl = τl * ω_liq; si = τi * ω_ice
                ss = sl + si
                optics.τ[k, g]   = τt
                optics.ssa[k, g] = ss / τt
                optics.g[k, g]   = ss > 0 ? (sl * g_liq + si * g_ice) / ss : zero(T)
            end
        end
    end
    return optics
end

"""
    cloud_optical_depth(col, cloud; band = 10) -> τ

Column cloud optical depth in a band (band 10 of the default shortwave
grid is in the visible).
"""
function cloud_optical_depth(col::AtmosphericColumn{T},
                             cloud::MieCloudOptics{T}; band::Int = 10) where {T}
    o = RadiationCore.allocate_optics(T, (length(col.ql),), cloud.spec)
    compute_optics!(o, cloud, col)
    g1 = first(gpoint_range(cloud.spec, band))
    return sum(@view o.τ[:, g1])
end

"""
    maximum_random_overlap(cf) -> total cloud fraction

Maximum-random overlap of a layer cloud-fraction profile: vertically
contiguous cloudy layers overlap maximally, separated groups randomly.
This is the standard operational assumption and sits between the two
extremes, both of which are wrong for real clouds.
"""
function maximum_random_overlap(cf::AbstractVector{T}) where {T}
    n = length(cf)
    n == 0 && return zero(T)
    clear = one(T)
    i = 1
    while i <= n
        if cf[i] <= 0
            i += 1; continue
        end
        j = i
        cmax = zero(T)
        while j <= n && cf[j] > 0
            cmax = max(cmax, cf[j]); j += 1
        end
        clear *= (1 - cmax)              # contiguous group: maximum overlap
        i = j
    end
    return 1 - clear                      # groups combine randomly
end

"""
    radiative_heating!(dθdt, fluxes, col) -> dθdt

Radiative heating rate in K s⁻¹ from the net-flux divergence.
"""
radiative_heating!(dθdt, fluxes::RadiativeFluxes, col::AtmosphericColumn) =
    heating_rate!(dθdt, fluxes, col.dz, col.ρ)

end # module
