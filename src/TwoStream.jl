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
                       heating_rate!, STEFAN_BOLTZMANN,
                       liquid_mass_extinction, ice_mass_extinction
import ..RadiationCore: compute_optics!, solve_rt!
using ..Backends: AbstractBackend, default_backend, launch_weighted!

export TwoStreamSolver, two_stream_solver,
       DeltaEddington, PIFM, Quadrature,
       delta_scale, two_stream_coefficients,
       layer_reflectance_transmittance_sw,
       layer_reflectance_transmittance_lw,
       adding_solver!, solve_sw!, solve_lw!, TwoStreamWorkspace,
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
  * `k μ₀ → 1` — the direct-beam denominator `1 - k²μ₀²` vanishes; the
    beam source terms are evaluated at `μ₀(1 ± 10⁻⁴)` (each with its own
    beam transmittance) and averaged, which is accurate to ~1e-8.
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
    # the diffuse eigenvalue (a removable singularity). There the formula
    # is evaluated at μ₀(1 ± h) and averaged, which is second-order
    # accurate (error O(h²) ≈ 1e-8) while keeping the 0/0 cancellation at
    # |1 - k²μ²| ≥ h. The beam transmittance inside the formula must be
    # that of the shifted μ: mixing e^{-τ/μ₀} with a shifted μ leaves an
    # O(1) error in the numerator that the small denominator amplifies.
    α1 = γ1 * γ4 + γ2 * γ3
    α2 = γ1 * γ3 + γ2 * γ4
    # Meador & Weaver (1980) eq. 14, with numerator and denominator both
    # divided through by e^{kτ} so nothing overflows at large τ.
    function beam_source(μ)
        Tb = exp(-min(τ / μ, T(500)))
        d = (1 - (k * μ)^2)
        pref = ω / d
        Rd = pref * ((1 - k * μ) * (α2 + k * γ3) -
                     (1 + k * μ) * (α2 - k * γ3) * e2 -
                     2k * (γ3 - α2 * μ) * e * Tb) / denom
        Td = -pref * ((1 + k * μ) * (α1 + k * γ4) * Tb -
                      (1 - k * μ) * (α1 - k * γ4) * e2 * Tb -
                      2k * (γ4 + α1 * μ) * e) / denom
        return Rd, Td
    end
    if abs(1 - (k * μ₀)^2) < T(1e-4)       # shifts land at |d| ≥ 1e-4
        h = T(1e-4)
        Rp, Tp = beam_source(μ₀ * (1 + h))
        Rm, Tm = beam_source(μ₀ * (1 - h))
        R_dir = (Rp + Rm) / 2
        T_dir = (Tp + Tm) / 2
    else
        R_dir, T_dir = beam_source(μ₀)
    end

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
    nlev = length(R_dif) + 1
    A = Vector{T}(undef, nlev)
    S = Vector{T}(undef, nlev)
    return _adding!(flux_up, flux_dn, flux_dir, R_dif, T_dif, R_dir, T_dir, T_beam,
                    sfc_alb_dif, sfc_alb_dir, F_top, μ₀, T(F_dif_top), A, S)
end

# The adding equations with caller-provided scratch `A`, `S` (length nlev):
# the multi-column solve runs one per (g-point, column) without allocating.
function _adding!(flux_up::AbstractVector{T}, flux_dn::AbstractVector{T},
                  flux_dir::AbstractVector{T},
                  R_dif::AbstractVector{T}, T_dif::AbstractVector{T},
                  R_dir::AbstractVector{T}, T_dir::AbstractVector{T},
                  T_beam::AbstractVector{T},
                  sfc_alb_dif::T, sfc_alb_dir::T,
                  F_top::T, μ₀::T, F_dif_top::T,
                  A::AbstractVector{T}, S::AbstractVector{T}) where {T}
    nlay = length(R_dif)
    nlev = nlay + 1

    # --- direct beam straight down from the top
    @inbounds flux_dir[nlev] = F_top * μ₀
    @inbounds for i in nlay:-1:1
        flux_dir[i] = flux_dir[i + 1] * T_beam[i]
    end

    # --- upward sweep: albedo A[i] of everything at and below level i,
    # plus the source S[i] emerging upward from it
    @inbounds A[1] = sfc_alb_dif
    @inbounds S[1] = sfc_alb_dir * flux_dir[1]
    @inbounds for i in 1:nlay
        d = 1 / (1 - R_dif[i] * A[i])
        A[i + 1] = R_dif[i] + T_dif[i] * T_dif[i] * A[i] * d
        S[i + 1] = R_dir[i] * flux_dir[i + 1] +
                   T_dif[i] * d * (S[i] + A[i] * T_dir[i] * flux_dir[i + 1])
    end

    # --- downward sweep
    @inbounds flux_dn[nlev] = F_dif_top                # diffuse entering from above
    @inbounds flux_up[nlev] = A[nlev] * flux_dn[nlev] + S[nlev]
    @inbounds for i in nlay:-1:1
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
    TwoStreamWorkspace{T}()

Scratch of the multi-column two-stream solves (KA-n, #305): per g-point and
column, the layer reflectances / transmittances / sources and the level
fluxes of one `(g, column)` work item, for up to `TWOSTREAM_CHUNK` columns at
a time. Pass the same workspace to every call (`workspace = ws`) and a solve
allocates nothing; it grows when a call needs more layers, g-points or
columns than it holds.
"""
mutable struct TwoStreamWorkspace{T<:AbstractFloat}
    lay::Array{T,4}      # (nlay, 5, ngpt, chunk): R_dif, T_dif, R_dir, T_dir, T_beam (LW: R, T, S_up, S_dn)
    lev::Array{T,4}      # (nlev, 5, ngpt, chunk): up, dn, dir, A, S
    planck::Array{T,3}   # (nlev + 1, nbands, chunk): π B_band at the levels, then at the surface
end
TwoStreamWorkspace{T}() where {T} =
    TwoStreamWorkspace{T}(zeros(T, 0, 5, 0, 0), zeros(T, 0, 5, 0, 0), zeros(T, 0, 0, 0))

"Columns solved per launch by the multi-column two-stream solves (bounds the workspace)."
const TWOSTREAM_CHUNK = 64

function _ensure!(ws::TwoStreamWorkspace{T}, nlay, ngpt, nb, nc) where {T}
    if size(ws.lay, 1) < nlay || size(ws.lay, 3) < ngpt || size(ws.lay, 4) < nc
        ws.lay = zeros(T, nlay, 5, ngpt, nc)
        ws.lev = zeros(T, nlay + 1, 5, ngpt, nc)
    end
    if size(ws.planck, 1) < nlay + 2 || size(ws.planck, 2) < nb || size(ws.planck, 3) < nc
        ws.planck = zeros(T, nlay + 2, nb, nc)
    end
    return ws
end

# A single column's arrays as one-column multi-column arrays (no copy).
_one_column(o::OpticalProperties{T,2}) where {T} =
    OpticalProperties{T,3}(reshape(o.τ, size(o.τ)..., 1), reshape(o.ssa, size(o.ssa)..., 1),
                           reshape(o.g, size(o.g)..., 1), o.spec)
_one_column(f::RadiativeFluxes{T,1}) where {T} =
    RadiativeFluxes{T,2}(reshape(f.flux_up, 1, :), reshape(f.flux_dn, 1, :),
                         reshape(f.flux_dn_dir, 1, :), reshape(f.flux_net, 1, :))

# Per-band, per-column surface property for the multi-column solves: a Real
# for every band and column, a vector per column, a (band, column) matrix.
@inline _sfc(a::Real, b, c, ::Type{T}) where {T} = T(a)
@inline _sfc(a::AbstractVector, b, c, ::Type{T}) where {T} = T(a[c])
@inline _sfc(a::AbstractMatrix, b, c, ::Type{T}) where {T} = T(a[b, c])
# the single-column convention: a Real, or one value per band
_band_column(a::Real) = a
_band_column(a::AbstractVector) = reshape(a, :, 1)

"""
    solve_sw!(fluxes, solver, optics; μ₀, sfc_albedo_direct, sfc_albedo_diffuse,
              workspace = TwoStreamWorkspace{T}(), backend = default_backend())

Shortwave solve over all g-points, accumulating broadband fluxes.
`sfc_albedo_*` may be a scalar or a per-band vector.

**Multi-column form** (KA-n, #305): `optics` of shape `(nlay, ngpt, ncol)`
(`τ[k, g, c]`) and `fluxes` of shape `(ncol, nlev)`
(`allocate_fluxes(T, (ncol,), nlev)`). `μ₀` is a scalar or one value per
column; `sfc_albedo_*` a scalar, one value per column, or an `(nband, ncol)`
matrix. Every `(g-point, column)` is one work item of a
[`launch_weighted!`](@ref Aeolus.Backends.launch_weighted!) on `backend`:
the adding recurrence runs sequentially over the layers inside the item, on
the item's slice of `workspace`; a second launch sums each column's
g-points in ascending order. Column `c` gets the bits of the single-column
solve, and the single-column form is this solve on one column, so the fluxes
are identical at every thread count.
"""
function solve_sw!(fluxes::RadiativeFluxes{T,1}, solver::TwoStreamSolver{T},
                   optics::OpticalProperties{T,2};
                   μ₀::Real,
                   sfc_albedo_direct = 0.1,
                   sfc_albedo_diffuse = nothing,
                   workspace::TwoStreamWorkspace{T} = TwoStreamWorkspace{T}(),
                   backend::AbstractBackend = default_backend()) where {T}
    ad_diffuse = sfc_albedo_diffuse === nothing ? sfc_albedo_direct : sfc_albedo_diffuse
    _solve_sw_columns!(_one_column(fluxes), solver, _one_column(optics), μ₀,
                       _band_column(sfc_albedo_direct), _band_column(ad_diffuse),
                       workspace, backend)
    return fluxes
end

function solve_sw!(fluxes::RadiativeFluxes{T,2}, solver::TwoStreamSolver{T},
                   optics::OpticalProperties{T,3};
                   μ₀,
                   sfc_albedo_direct = 0.1,
                   sfc_albedo_diffuse = nothing,
                   workspace::TwoStreamWorkspace{T} = TwoStreamWorkspace{T}(),
                   backend::AbstractBackend = default_backend()) where {T}
    ad_diffuse = sfc_albedo_diffuse === nothing ? sfc_albedo_direct : sfc_albedo_diffuse
    _solve_sw_columns!(fluxes, solver, optics, μ₀, sfc_albedo_direct, ad_diffuse,
                       workspace, backend)
    return fluxes
end

function _check_columns(fluxes, optics, name)
    nlay, _, ncol = size(optics.τ)
    size(fluxes.flux_up) == (ncol, nlay + 1) ||
        error("$name: fluxes are $(size(fluxes.flux_up)), expected ($ncol, $(nlay + 1)) " *
              "for optics of $nlay layers × $ncol columns")
    return nlay, ncol
end

function _solve_sw_columns!(fluxes, solver::TwoStreamSolver{T}, optics, μ₀, αdir, αdif,
                            ws, backend) where {T}
    spec = solver.spec
    spec.kind === :shortwave || error("solve_sw!: requires a shortwave grid")
    nlay, ncol = _check_columns(fluxes, optics, "solve_sw!")
    ngpt = ngpoints(spec)
    _ensure!(ws, nlay, ngpt, nbands(spec), min(ncol, TWOSTREAM_CHUNK))
    for c0 in 0:TWOSTREAM_CHUNK:ncol-1
        nc = min(TWOSTREAM_CHUNK, ncol - c0)
        launch_weighted!(backend, nlay, _sw_gpoint!, (1:ngpt * nc,), ws.lay, ws.lev,
                         solver, optics.τ, optics.ssa, optics.g, μ₀, αdir, αdif, c0, ngpt)
        launch_weighted!(backend, ngpt, _sw_gsum!, (1:nc, 1:nlay + 1), fluxes.flux_up,
                         fluxes.flux_dn, fluxes.flux_dn_dir, ws.lev, spec, c0)
    end
    net_flux!(fluxes)
    return fluxes
end

@inline _μ(μ₀::Real, c, ::Type{T}) where {T} = T(clamp(μ₀, 1e-6, 1.0))
@inline _μ(μ₀::AbstractVector, c, ::Type{T}) where {T} = T(clamp(μ₀[c], 1e-6, 1.0))

# One (g-point, column) of the shortwave: layer properties, then the adding
# recurrence over the layers, into the item's slice of the workspace.
function _sw_gpoint!(n, lay::AbstractArray{T,4}, lev, solver, τ3, ω3, g3, μ₀, αdir_in, αdif_in,
                     c0, ngpt) where {T}
    spec = solver.spec
    g = (n - 1) % ngpt + 1
    cc = (n - 1) ÷ ngpt + 1
    c = c0 + cc
    F_top = T(spec.solar_src[g])
    F_top <= 0 && return nothing
    b = band_of_gpoint(spec, g)
    μ = _μ(μ₀, c, T)
    αdir = _sfc(αdir_in, b, c, T)
    αdif = _sfc(αdif_in, b, c, T)
    nlay = size(τ3, 1)
    @inbounds for k in 1:nlay
        τs, ωs, gs = delta_scale(τ3[k, g, c], ω3[k, g, c], g3[k, g, c])
        lay[k, 1, g, cc], lay[k, 2, g, cc], lay[k, 3, g, cc], lay[k, 4, g, cc], lay[k, 5, g, cc] =
            layer_reflectance_transmittance_sw(solver.sw_variant, τs, ωs, gs, μ)
    end
    r = 1:nlay; l = 1:nlay + 1
    _adding!(view(lev, l, 1, g, cc), view(lev, l, 2, g, cc), view(lev, l, 3, g, cc),
             view(lay, r, 1, g, cc), view(lay, r, 2, g, cc), view(lay, r, 3, g, cc),
             view(lay, r, 4, g, cc), view(lay, r, 5, g, cc),
             αdif, αdir, F_top, μ, zero(T), view(lev, l, 4, g, cc), view(lev, l, 5, g, cc))
    return nothing
end

# The broadband sums of level i of column c0 + cc: the g-points in ascending
# order (band by band), skipping those without solar source — the
# accumulation order of the single-column solve.
function _sw_gsum!(cc, i, up::AbstractArray{T}, dn, dir, lev, spec, c0) where {T}
    su = zero(T); sd = zero(T); sr = zero(T)
    @inbounds for b in 1:nbands(spec), g in gpoint_range(spec, b)
        T(spec.solar_src[g]) <= 0 && continue
        fd = lev[i, 3, g, cc]
        su += lev[i, 1, g, cc]
        sd += lev[i, 2, g, cc] + fd
        sr += fd
    end
    c = c0 + cc
    @inbounds up[c, i] = su
    @inbounds dn[c, i] = sd
    @inbounds dir[c, i] = sr
    return nothing
end

"""
    solve_lw!(fluxes, solver, optics, col; sfc_emissivity,
              workspace = TwoStreamWorkspace{T}(), backend = default_backend())

Longwave solve. Source terms come from the Planck function evaluated at
the layer boundary temperatures, so the linear-in-τ treatment has the
gradient information it needs.

**Multi-column form** (KA-n, #305): `optics` of shape `(nlay, ngpt, ncol)`,
`cols` a vector of `ncol` columns, `fluxes` of shape `(ncol, nlev)`;
`sfc_emissivity` a scalar, one value per column or an `(nband, ncol)` matrix.
The band Planck sources are one launch over `(level, column)`, the
`(g-point, column)` sweeps a second, the g-point sums a third, as in
[`solve_sw!`](@ref). Column `c` gets the bits of the single-column solve on
`cols[c]`, at every thread count.
"""
function solve_lw!(fluxes::RadiativeFluxes{T,1}, solver::TwoStreamSolver{T},
                   optics::OpticalProperties{T,2},
                   col::AtmosphericColumn{T};
                   sfc_emissivity = 1.0,
                   workspace::TwoStreamWorkspace{T} = TwoStreamWorkspace{T}(),
                   backend::AbstractBackend = default_backend()) where {T}
    _solve_lw_columns!(_one_column(fluxes), solver, _one_column(optics), (col,),
                       _band_column(sfc_emissivity), workspace, backend)
    return fluxes
end

function solve_lw!(fluxes::RadiativeFluxes{T,2}, solver::TwoStreamSolver{T},
                   optics::OpticalProperties{T,3},
                   cols::AbstractVector{<:AtmosphericColumn{T}};
                   sfc_emissivity = 1.0,
                   workspace::TwoStreamWorkspace{T} = TwoStreamWorkspace{T}(),
                   backend::AbstractBackend = default_backend()) where {T}
    _solve_lw_columns!(fluxes, solver, optics, cols, sfc_emissivity, workspace, backend)
    return fluxes
end

function _solve_lw_columns!(fluxes, solver::TwoStreamSolver{T}, optics, cols, ε_in,
                            ws, backend) where {T}
    spec = solver.spec
    spec.kind === :longwave || error("solve_lw!: requires a longwave grid")
    nlay, ncol = _check_columns(fluxes, optics, "solve_lw!")
    length(cols) == ncol ||
        error("solve_lw!: optics have $ncol columns, got $(length(cols)) columns")
    ngpt = ngpoints(spec)
    nb = nbands(spec)
    _ensure!(ws, nlay, ngpt, nb, min(ncol, TWOSTREAM_CHUNK))
    fill!(fluxes.flux_dn_dir, 0)
    for c0 in 0:TWOSTREAM_CHUNK:ncol-1
        nc = min(TWOSTREAM_CHUNK, ncol - c0)
        launch_weighted!(backend, nb, _lw_planck!, (1:nc, 1:nlay + 2), ws.planck, spec, cols, c0)
        launch_weighted!(backend, nlay, _lw_gpoint!, (1:ngpt * nc,), ws.lay, ws.lev, ws.planck,
                         solver, optics.τ, optics.ssa, optics.g, ε_in, c0, ngpt)
        launch_weighted!(backend, ngpt, _lw_gsum!, (1:nc, 1:nlay + 1), fluxes.flux_up,
                         fluxes.flux_dn, ws.lev, spec, c0)
    end
    net_flux!(fluxes)
    return fluxes
end

# π B_band at level i of column c0 + cc for every band; i = nlev + 1 is the
# surface (the single-column solve's `Blev`, `Bsfc`).
function _lw_planck!(cc, i, P::AbstractArray{T,3}, spec, cols, c0) where {T}
    col = cols[c0 + cc]
    nlev = length(col.T_lev)
    Tk = i <= nlev ? col.T_lev[i] : col.T_sfc
    for b in 1:nbands(spec)
        lo, hi = band_limits(spec, b)
        @inbounds P[i, b, cc] = T(π * planck_band(Tk, lo, hi))
    end
    return nothing
end

function _lw_gpoint!(n, lay::AbstractArray{T,4}, lev, P, solver, τ3, ω3, g3, ε_in,
                     c0, ngpt) where {T}
    spec = solver.spec
    g = (n - 1) % ngpt + 1
    cc = (n - 1) ÷ ngpt + 1
    c = c0 + cc
    b = band_of_gpoint(spec, g)
    ε = _sfc(ε_in, b, c, T)
    w = T(spec.gpt_weight[g])
    nlay = size(τ3, 1); nlev = nlay + 1
    @inbounds for k in 1:nlay
        Bb = solver.linear_lw_source ? w * P[k, b, cc]     : w * (P[k, b, cc]+P[k + 1, b, cc])/2
        Bt = solver.linear_lw_source ? w * P[k + 1, b, cc] : w * (P[k, b, cc]+P[k + 1, b, cc])/2
        lay[k, 1, g, cc], lay[k, 2, g, cc], lay[k, 3, g, cc], lay[k, 4, g, cc] =
            layer_reflectance_transmittance_lw(solver.lw_variant,
                τ3[k, g, c], ω3[k, g, c], g3[k, g, c], Bb, Bt)
    end
    @inbounds begin
        # downward sweep from TOA
        lev[nlev, 2, g, cc] = zero(T)
        for k in nlay:-1:1
            lev[k, 2, g, cc] = lay[k, 2, g, cc] * lev[k + 1, 2, g, cc] + lay[k, 4, g, cc]
        end
        # surface emission + reflection
        lev[1, 1, g, cc] = ε * w * P[nlev + 1, b, cc] + (1 - ε) * lev[1, 2, g, cc]
        for k in 1:nlay
            lev[k + 1, 1, g, cc] = lay[k, 2, g, cc] * lev[k, 1, g, cc] + lay[k, 3, g, cc] +
                                   lay[k, 1, g, cc] * lev[k + 1, 2, g, cc]
        end
    end
    return nothing
end

function _lw_gsum!(cc, i, up::AbstractArray{T}, dn, lev, spec, c0) where {T}
    su = zero(T); sd = zero(T)
    @inbounds for b in 1:nbands(spec), g in gpoint_range(spec, b)
        su += lev[i, 1, g, cc]
        sd += lev[i, 2, g, cc]
    end
    c = c0 + cc
    @inbounds up[c, i] = su
    @inbounds dn[c, i] = sd
    return nothing
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
function compute_optics!(optics::OpticalProperties{T,2}, cloud::MieCloudOptics{T},
                         col::AtmosphericColumn{T};
                         backend::AbstractBackend = default_backend()) where {T}
    return _cloud_optics!(optics, _one_column(optics), cloud, (col,), backend)
end

"""
    compute_optics!(optics, cloud::MieCloudOptics, cols::AbstractVector{<:AtmosphericColumn};
                    backend = default_backend())

Multi-column form (KA-n, #305): `optics` of shape `(nlay, ngpt, ncol)`, one
work item per `(layer, column)`; column `c` gets the bits of the
single-column `compute_optics!` on `cols[c]`.
"""
function compute_optics!(optics::OpticalProperties{T,3}, cloud::MieCloudOptics{T},
                         cols::AbstractVector{<:AtmosphericColumn{T}};
                         backend::AbstractBackend = default_backend()) where {T}
    size(optics.τ, 3) == length(cols) ||
        error("compute_optics!: optics has $(size(optics.τ, 3)) columns, got $(length(cols))")
    return _cloud_optics!(optics, optics, cloud, cols, backend)
end

function _cloud_optics!(optics, o3::OpticalProperties{T,3}, cloud, cols, backend) where {T}
    fill!(o3.τ, 0); fill!(o3.ssa, 0); fill!(o3.g, 0)
    nlay, ngpt, ncol = size(o3.τ)
    launch_weighted!(backend, ngpt, _cloud_optics_layer!, (1:ncol, 1:nlay),
                     o3.τ, o3.ssa, o3.g, cloud.spec, cols)
    return optics
end

# Layer k of column c, every band and g-point (each element written once).
function _cloud_optics_layer!(c, k, τ3::AbstractArray{T,3}, ω3, g3, spec, cols) where {T}
    col = cols[c]
    lwp = col.ql[k] * col.ρ[k] * col.dz[k] * 1000   # g m⁻²
    iwp = col.qi[k] * col.ρ[k] * col.dz[k] * 1000
    (lwp > 0 || iwp > 0) || return nothing
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
        # τ = 3·LWP / (2 ρ_w r_eff) = 1.5·LWP/r_eff (LWP in g m⁻², r_eff in
        # μm): RadiationCore.liquid_mass_extinction, shared with the 3-D
        # cloud shadow of SolarGeometry.
        τl = lwp > 0 ? T(liquid_mass_extinction(col.r_eff_liq[k]) * lwp) : zero(T)
        τi = iwp > 0 ? T(ice_mass_extinction(col.r_eff_ice[k]) * iwp) : zero(T)
        τt = τl + τi
        τt <= 0 && continue
        sl = τl * ω_liq; si = τi * ω_ice
        ss = sl + si
        @inbounds for g in gpoint_range(spec, b)
            τ3[k, g, c] = τt
            ω3[k, g, c] = ss / τt
            g3[k, g, c] = ss > 0 ? (sl * g_liq + si * g_ice) / ss : zero(T)
        end
    end
    return nothing
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
