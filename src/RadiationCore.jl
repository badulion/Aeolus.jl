module RadiationCore

# Era III — shared radiation interface.
#
# This module owns the *contract* that every radiation component in Era III
# speaks. It deliberately contains no physics: it defines the spectral
# discretisation, the optical-property container, the flux container, and
# the abstract solver/optics interfaces. Gas optics (Phase 32), the
# two-stream solver (Phase 33), Monte-Carlo RT (Phase 34), urban radiosity
# (Phase 35), surface optics (Phase 36) and aerosol optics (Phase 37) all
# plug into these types.
#
# Why a shared core: a radiation stack is a pipeline —
#
#   gas optics ─┐
#   cloud optics├─→ OpticalProperties ─→ RT solver ─→ RadiativeFluxes ─→ heating rate
#   aerosol    ─┘                            ↑
#                                    SurfaceOptics (albedo / emissivity / BRDF)
#
# If each stage invents its own array layout the pipeline cannot be
# reassembled. Fixing the contract up front is what lets the stages be
# developed and validated independently.
#
# Conventions, fixed once here:
#   * Spectral: shortwave and longwave are separate `SpectralGrid`s. A grid
#     carries `nband` bands, each subdivided into g-points (correlated-k
#     quadrature points). `ngpt` is the total. Band limits are in cm⁻¹
#     (wavenumber), the convention used by RRTM-family gas optics.
#   * Vertical: optical properties are *layer* quantities (`Nz` of them);
#     fluxes are *level* (interface) quantities (`Nz+1` of them), ordered
#     bottom-up so index 1 is the surface and index `Nz+1` the model top.
#   * Sign: `flux_dn` and `flux_up` are both positive magnitudes.
#     `flux_net = flux_dn - flux_up` is positive downward (i.e. positive
#     net flux warms the column below).
#   * Heating rate is returned in K s⁻¹.

using ..Grids
using ..Grids: RegularGrid, CellCenter, interior
using ..Fields
using ..Fields: Field, interior_view

export SpectralGrid, shortwave_bands, longwave_bands,
       nbands, ngpoints, band_of_gpoint, band_limits, gpoint_range,
       OpticalProperties, allocate_optics, zero_optics!,
       combine_optics!, scale_optics!, total_optical_depth,
       RadiativeFluxes, allocate_fluxes, zero_fluxes!,
       net_flux!, flux_divergence!, heating_rate!,
       AbstractGasOptics, AbstractCloudOptics, AbstractAerosolOptics,
       AbstractRTSolver, AbstractSurfaceOptics,
       compute_optics!, solve_rt!, surface_albedo, surface_emissivity,
       material_albedo, material_emissivity,
       band_wavelength, band_wavelength_limits,
       AtmosphericColumn, column_from_simulation, column_from_profiles,
       stitch_columns,
       liquid_mass_extinction, ice_mass_extinction,
       STEFAN_BOLTZMANN, SOLAR_CONSTANT, PLANCK_H, LIGHT_C, BOLTZMANN_K,
       planck_band, planck_integrated

# =========================================================================
# Physical constants
# =========================================================================

const STEFAN_BOLTZMANN = 5.670374419e-8   # W m⁻² K⁻⁴ (CODATA 2018)
const SOLAR_CONSTANT   = 1361.0           # W m⁻² total solar irradiance
const PLANCK_H         = 6.62607015e-34   # J s
const LIGHT_C          = 2.99792458e8     # m s⁻¹
const BOLTZMANN_K      = 1.380649e-23     # J K⁻¹
const CP_DRY_AIR       = 1004.6           # J kg⁻¹ K⁻¹
const GRAVITY          = 9.80665          # m s⁻²

# =========================================================================
# Spectral discretisation
# =========================================================================

"""
    SpectralGrid{T}

Correlated-k spectral discretisation. `nband` contiguous wavenumber bands,
each subdivided into g-points. Fields:

  * `band_lo`, `band_hi` — band limits in cm⁻¹, length `nband`
  * `gpt_per_band`       — g-points in each band, length `nband`
  * `gpt_band`           — band index of each g-point, length `ngpt`
  * `gpt_weight`         — quadrature weight of each g-point; within a band
                           the weights sum to 1, so a band-integrated
                           quantity is `Σ_g w_g · x_g` over that band's
                           g-points
  * `solar_src`          — top-of-atmosphere solar irradiance per g-point
                           (W m⁻², sums to `SOLAR_CONSTANT` for a shortwave
                           grid; all zeros for a longwave grid)
  * `kind`               — `:shortwave` or `:longwave`

The g-point weights are what make a correlated-k scheme a *quadrature*
rather than a band average: within a band the absorption coefficient is
reordered monotonically and integrated in cumulative-probability space, so
a handful of g-points reproduces the line-by-line band transmission.
"""
struct SpectralGrid{T<:AbstractFloat}
    kind::Symbol
    band_lo::Vector{T}
    band_hi::Vector{T}
    gpt_per_band::Vector{Int}
    gpt_band::Vector{Int}
    gpt_weight::Vector{T}
    solar_src::Vector{T}
end

function SpectralGrid(kind::Symbol, band_lo::AbstractVector{T},
                      band_hi::AbstractVector{T},
                      gpt_per_band::AbstractVector{<:Integer},
                      gpt_weight::AbstractVector{T},
                      solar_src::AbstractVector{T}) where {T}
    kind === :shortwave || kind === :longwave ||
        error("SpectralGrid: kind must be :shortwave or :longwave, got $kind")
    nb = length(band_lo)
    length(band_hi) == nb == length(gpt_per_band) ||
        error("SpectralGrid: band arrays must agree in length")
    ngpt = sum(gpt_per_band)
    length(gpt_weight) == ngpt ||
        error("SpectralGrid: gpt_weight length $(length(gpt_weight)) ≠ ngpt $ngpt")
    length(solar_src) == ngpt ||
        error("SpectralGrid: solar_src length $(length(solar_src)) ≠ ngpt $ngpt")
    all(band_hi .> band_lo) ||
        error("SpectralGrid: every band must have band_hi > band_lo")
    gpt_band = Vector{Int}(undef, ngpt)
    g = 1
    for b in 1:nb, _ in 1:gpt_per_band[b]
        gpt_band[g] = b; g += 1
    end
    # weights within each band must sum to 1 — this is the property that
    # makes band-integration a plain weighted sum.
    for b in 1:nb
        r = gpoint_range_impl(gpt_per_band, b)
        s = sum(@view gpt_weight[r])
        isapprox(s, one(T); atol = 1e-10) ||
            error("SpectralGrid: g-point weights in band $b sum to $s, expected 1")
    end
    return SpectralGrid{T}(kind, collect(band_lo), collect(band_hi),
                           collect(Int, gpt_per_band), gpt_band,
                           collect(gpt_weight), collect(solar_src))
end

"""
    band_wavelength_limits(spec, b) -> (λ_lo, λ_hi)

Wavelength limits of band `b` in μm. Bands are stored in wavenumber
(cm⁻¹), which is the natural coordinate for the transfer equation
because it is linear in energy; wavelength is what optical-property
tables are almost always keyed on. Note the order reverses — the
*high*-wavenumber edge is the *short*-wavelength one.
"""
function band_wavelength_limits(spec::SpectralGrid, b::Integer)
    lo, hi = band_limits(spec, Int(b))
    return (1e4 / hi, 1e4 / lo)
end

"""
    band_wavelength(spec, b; sub = 1, nsub = 1) -> λ in μm

Wavelength of a band, or of one of `nsub` equal-wavenumber sub-intervals
within it. With the defaults this is the wavelength of the band's
wavenumber midpoint.

The sub-interval form exists for optical-property models that need to
integrate a strongly wavelength-dependent quantity across a band — Mie
efficiencies ripple on a scale far finer than an RRTMG band, so
evaluating them once at the band centre is not good enough.

This lives in `RadiationCore` rather than in any one optics module
because four separate copies of it had appeared by the end of Era III
(in `SurfaceOptics`, `AerosolOptics`, `TwoStream` and, in spirit,
`SolarGeometry`). It is a property of the spectral grid, so it belongs
with the spectral grid.

See also `SurfaceOptics.band_center_wavelength`, which is a *different*
quantity: the Planck-weighted mean wavelength over the band. That one
needs a source temperature and stays where the weighting machinery is.
"""
function band_wavelength(spec::SpectralGrid, b::Integer;
                         sub::Integer = 1, nsub::Integer = 1)
    lo, hi = band_limits(spec, Int(b))
    ν = lo + (sub - 0.5) * (hi - lo) / nsub
    return 1e4 / ν
end

function gpoint_range_impl(gpt_per_band, b::Int)
    lo = 1
    for i in 1:b-1
        lo += gpt_per_band[i]
    end
    return lo:(lo + gpt_per_band[b] - 1)
end

nbands(s::SpectralGrid)   = length(s.band_lo)
ngpoints(s::SpectralGrid) = length(s.gpt_band)
band_of_gpoint(s::SpectralGrid, g::Int) = s.gpt_band[g]
band_limits(s::SpectralGrid, b::Int) = (s.band_lo[b], s.band_hi[b])

"Index range of the g-points belonging to band `b`."
gpoint_range(s::SpectralGrid, b::Int) = gpoint_range_impl(s.gpt_per_band, b)

Base.eltype(::SpectralGrid{T}) where {T} = T

function Base.show(io::IO, s::SpectralGrid{T}) where {T}
    print(io, "SpectralGrid{$T}($(s.kind), $(nbands(s)) bands, ",
              "$(ngpoints(s)) g-points, ",
              "$(round(minimum(s.band_lo); digits=1))–",
              "$(round(maximum(s.band_hi); digits=1)) cm⁻¹)")
end

# --- default band structures --------------------------------------------
# These mirror the RRTMG band layout, which is the de-facto standard
# discretisation for atmospheric radiation. A concrete `AbstractGasOptics`
# is free to supply its own grid; these exist so the RT solvers and the
# surface-optics layer have a well-defined default to be validated against.

const _RRTMG_SW_BAND_LO = Float64[2600, 3250, 4000, 4650, 5150, 6150, 7700,
                                 8050, 12850, 16000, 22650, 29000, 38000, 820]
const _RRTMG_SW_BAND_HI = Float64[3250, 4000, 4650, 5150, 6150, 7700, 8050,
                                  12850, 16000, 22650, 29000, 38000, 50000, 2600]
# Band-integrated TOA solar irradiance (W m⁻²), normalised below to sum to
# SOLAR_CONSTANT. Relative distribution follows the RRTMG SW source.
const _RRTMG_SW_BAND_SOLAR = Float64[12.11, 20.36, 23.73, 22.43, 55.63,
                                     102.93, 24.29, 345.74, 218.19, 347.20,
                                     129.49, 50.15, 3.08, 12.89]

"""
    shortwave_bands(T = Float64; ngpt_per_band = 2)

Default shortwave `SpectralGrid` on the 14 RRTMG SW bands, with
`ngpt_per_band` equal-weight g-points per band (so `ngpt =
14·ngpt_per_band`). TOA solar irradiance is distributed across bands
following the RRTMG source function and renormalised so the total is
exactly `SOLAR_CONSTANT`.

Equal g-point weights are the correct default for a *reduced* grid: they
make the quadrature a plain band average, which is what a gas-optics
implementation should reduce to when it has not been given a real
k-distribution. Phase 32 supplies proper non-uniform weights.
"""
function shortwave_bands(::Type{T} = Float64; ngpt_per_band::Int = 2) where {T}
    ngpt_per_band >= 1 || error("shortwave_bands: ngpt_per_band must be ≥ 1")
    nb = length(_RRTMG_SW_BAND_LO)
    gpb = fill(ngpt_per_band, nb)
    w   = T[one(T) / ngpt_per_band for _ in 1:(nb * ngpt_per_band)]
    scale = SOLAR_CONSTANT / sum(_RRTMG_SW_BAND_SOLAR)
    src = Vector{T}(undef, nb * ngpt_per_band)
    g = 1
    for b in 1:nb, _ in 1:ngpt_per_band
        src[g] = T(_RRTMG_SW_BAND_SOLAR[b] * scale / ngpt_per_band)
        g += 1
    end
    return SpectralGrid(:shortwave, T.(_RRTMG_SW_BAND_LO),
                        T.(_RRTMG_SW_BAND_HI), gpb, w, src)
end

const _RRTMG_LW_BAND_LO = Float64[10, 350, 500, 630, 700, 820, 980, 1080,
                                  1180, 1390, 1480, 1800, 2080, 2250, 2380, 2600]
const _RRTMG_LW_BAND_HI = Float64[350, 500, 630, 700, 820, 980, 1080, 1180,
                                  1390, 1480, 1800, 2080, 2250, 2380, 2600, 3250]

"""
    longwave_bands(T = Float64; ngpt_per_band = 2)

Default longwave `SpectralGrid` on the 16 RRTMG LW bands. `solar_src` is
all zeros (there is no solar source in the thermal infrared); the source
for a longwave problem is the Planck function, available via
[`planck_band`](@ref).
"""
function longwave_bands(::Type{T} = Float64; ngpt_per_band::Int = 2) where {T}
    ngpt_per_band >= 1 || error("longwave_bands: ngpt_per_band must be ≥ 1")
    nb = length(_RRTMG_LW_BAND_LO)
    gpb = fill(ngpt_per_band, nb)
    w   = T[one(T) / ngpt_per_band for _ in 1:(nb * ngpt_per_band)]
    src = zeros(T, nb * ngpt_per_band)
    return SpectralGrid(:longwave, T.(_RRTMG_LW_BAND_LO),
                        T.(_RRTMG_LW_BAND_HI), gpb, w, src)
end

# =========================================================================
# Planck function
# =========================================================================

"""
    planck_band(T_K, ν_lo, ν_hi; nquad = 8) -> W m⁻² sr⁻¹

Planck radiance integrated over a wavenumber band, `ν` in cm⁻¹:

```math
B_Δν(T) = \\int_{ν_{lo}}^{ν_{hi}} \\frac{2hc^2ν^3}{e^{hcν/k_BT} - 1}\\,dν
```

Evaluated by `nquad`-point Gauss-Legendre quadrature, which is accurate to
better than 1e-6 relative for atmospheric bands and temperatures. Units
come out in W m⁻² sr⁻¹ (per band, not per wavenumber).
"""
function planck_band(T_K::Real, ν_lo::Real, ν_hi::Real; nquad::Int = 8)
    T_K > 0 || return 0.0
    # Gauss-Legendre nodes/weights on [-1,1] for a few small orders.
    x, w = _gauss_legendre(nquad)
    half = (ν_hi - ν_lo) / 2
    mid  = (ν_hi + ν_lo) / 2
    acc = 0.0
    @inbounds for i in eachindex(x)
        ν_cm = mid + half * x[i]
        acc += w[i] * _planck_spectral(T_K, ν_cm)
    end
    return acc * half
end

# Spectral Planck radiance at wavenumber ν [cm⁻¹], per cm⁻¹, W m⁻² sr⁻¹ / cm⁻¹
@inline function _planck_spectral(T_K::Real, ν_cm::Real)
    ν = ν_cm * 100.0                      # cm⁻¹ → m⁻¹
    c1 = 2 * PLANCK_H * LIGHT_C^2         # W m² sr⁻¹
    c2 = PLANCK_H * LIGHT_C / BOLTZMANN_K # m K
    x  = c2 * ν / T_K
    x > 700 && return 0.0                 # avoid overflow; contribution is nil
    # B_ν(ν) dν with ν in m⁻¹; convert the differential to cm⁻¹ (×100)
    return c1 * ν^3 / expm1(x) * 100.0
end

"""
    planck_integrated(T_K) -> W m⁻²

`σT⁴` — the Stefan-Boltzmann integral of the Planck function over all
wavenumbers and the hemisphere. Used to check that a band decomposition
is complete: `π·Σ_b planck_band(T, νlo_b, νhi_b)` should approach this.
"""
planck_integrated(T_K::Real) = STEFAN_BOLTZMANN * T_K^4

# Gauss-Legendre nodes and weights, computed by Newton iteration on the
# Legendre polynomial. The one source of these nodes in Aeolus
# (SurfaceOptics and Radiosity delegate here; KA-m, #301, merged their
# copies, which were the same arithmetic and gave the same bits).
function _gauss_legendre_nodes(n::Int)
    x = zeros(Float64, n); w = zeros(Float64, n)
    for i in 1:n
        # initial guess (Chebyshev-like)
        z = cos(π * (i - 0.25) / (n + 0.5))
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
    return (x, w)
end

# Orders 1-64 are computed when the module is loaded (precompiled): a
# column kernel that asks for them only reads an immutable table, so a
# first use under many threads cannot race (KA-m, #301). Higher orders
# are memoised under a lock.
const _GL_PREFILLED = 64
const _GL_TABLE = Tuple{Vector{Float64}, Vector{Float64}}[_gauss_legendre_nodes(n) for n in 1:_GL_PREFILLED]
const _GL_CACHE = Dict{Int, Tuple{Vector{Float64}, Vector{Float64}}}()
const _GL_LOCK = ReentrantLock()

"""
    _gauss_legendre(n) -> (x, w)

`n`-point Gauss-Legendre nodes and weights on `[-1, 1]` (Newton iteration
on the Legendre polynomial ``P_n``). Orders up to 64 come from a table
filled when the module loads; higher orders are memoised under a lock, so
the function is safe to call from any thread. The returned vectors are
shared: do not modify them.
"""
function _gauss_legendre(n::Int)
    n >= 1 || error("_gauss_legendre: n must be ≥ 1, got $n")
    n <= _GL_PREFILLED && return @inbounds _GL_TABLE[n]
    return lock(_GL_LOCK) do
        get!(() -> _gauss_legendre_nodes(n), _GL_CACHE, n)
    end
end

# =========================================================================
# Optical properties
# =========================================================================

"""
    OpticalProperties{T}

Layer optical properties on a column or a 3-D field, per g-point.

  * `τ`   — optical depth (dimensionless, ≥ 0)
  * `ssa` — single-scattering albedo `ω₀ ∈ [0,1]`
  * `g`   — asymmetry parameter `∈ [-1,1]`

Array shape is `(dims..., ngpt)` where `dims` is `(Nz,)` for a single
column or `(Nx, Ny, Nz)` for a 3-D field. Layer index runs bottom-up.

A purely absorbing medium has `ssa = 0`; conservative scattering has
`ssa = 1`. For the two-stream solver `g` is the first moment of the phase
function (0 isotropic, ~0.85 typical for cloud droplets).
"""
struct OpticalProperties{T<:AbstractFloat, N}
    τ::Array{T, N}
    ssa::Array{T, N}
    g::Array{T, N}
    spec::SpectralGrid{T}
end

"""
    allocate_optics(T, dims, spec) -> OpticalProperties

Allocate zeroed optical properties for layer shape `dims` (a tuple) and
spectral grid `spec`. The trailing dimension is `ngpoints(spec)`.
"""
function allocate_optics(::Type{T}, dims::Tuple, spec::SpectralGrid{T}) where {T}
    shp = (dims..., ngpoints(spec))
    return OpticalProperties{T, length(shp)}(zeros(T, shp), zeros(T, shp),
                                             zeros(T, shp), spec)
end

allocate_optics(dims::Tuple, spec::SpectralGrid{T}) where {T} =
    allocate_optics(T, dims, spec)

function zero_optics!(o::OpticalProperties)
    fill!(o.τ, 0); fill!(o.ssa, 0); fill!(o.g, 0)
    return o
end

Base.size(o::OpticalProperties) = size(o.τ)
Base.eltype(::OpticalProperties{T}) where {T} = T
nlayers(o::OpticalProperties) = size(o.τ)[end - 1]

"""
    combine_optics!(dst, src)

Add the optical properties of `src` into `dst`, correctly combining the
scattering moments. Optical depths add, but `ssa` and `g` are *weighted*
averages — `ssa` by optical depth, `g` by scattering optical depth:

```math
τ = τ_1 + τ_2, \\qquad
ω = \\frac{τ_1ω_1 + τ_2ω_2}{τ}, \\qquad
g = \\frac{τ_1ω_1g_1 + τ_2ω_2g_2}{τ_1ω_1 + τ_2ω_2}
```

Getting this wrong (e.g. averaging `g` by optical depth) is a classic and
silent source of error when combining gas, cloud and aerosol optics.
"""
function combine_optics!(dst::OpticalProperties{T}, src::OpticalProperties{T}) where {T}
    size(dst.τ) == size(src.τ) ||
        error("combine_optics!: shape mismatch $(size(dst.τ)) vs $(size(src.τ))")
    @inbounds for i in eachindex(dst.τ)
        τ1 = dst.τ[i];  ω1 = dst.ssa[i]; g1 = dst.g[i]
        τ2 = src.τ[i];  ω2 = src.ssa[i]; g2 = src.g[i]
        τs = τ1 + τ2
        if τs <= 0
            dst.τ[i] = zero(T); dst.ssa[i] = zero(T); dst.g[i] = zero(T)
            continue
        end
        s1 = τ1 * ω1; s2 = τ2 * ω2       # scattering optical depths
        ss = s1 + s2
        dst.τ[i]   = τs
        dst.ssa[i] = ss / τs
        dst.g[i]   = ss > 0 ? (s1 * g1 + s2 * g2) / ss : zero(T)
    end
    return dst
end

"Multiply all optical depths by `a` (ssa and g unchanged)."
function scale_optics!(o::OpticalProperties, a::Real)
    o.τ .*= a
    return o
end

"""
    total_optical_depth(o, idx...) -> Vector

Band-integrated column optical depth per band at the given horizontal
index (omit `idx` for a column-shaped `OpticalProperties`). Uses the
g-point weights, so this is the correct spectral integral rather than a
naive mean.
"""
function total_optical_depth(o::OpticalProperties{T}, idx::Int...) where {T}
    spec = o.spec
    nb = nbands(spec)
    out = zeros(T, nb)
    for b in 1:nb
        acc = zero(T)
        for g in gpoint_range(spec, b)
            s = zero(T)
            if isempty(idx)
                @inbounds for k in axes(o.τ, 1)
                    s += o.τ[k, g]
                end
            else
                @inbounds for k in axes(o.τ, 3)
                    s += o.τ[idx[1], idx[2], k, g]
                end
            end
            acc += spec.gpt_weight[g] * s
        end
        out[b] = acc
    end
    return out
end

# =========================================================================
# Radiative fluxes
# =========================================================================

"""
    RadiativeFluxes{T}

Broadband fluxes at layer *interfaces* (levels), shape `(dims..., Nz+1)`
with level 1 the surface and level `Nz+1` the model top, all positive
magnitudes:

  * `flux_up`     — upward flux
  * `flux_dn`     — total downward flux (direct + diffuse)
  * `flux_dn_dir` — direct-beam downward flux (shortwave only; zero in LW)
  * `flux_net`    — `flux_dn - flux_up`, positive downward

Fluxes are broadband (already summed over g-points with their weights)
because that is what the thermodynamics needs. A solver that wants
per-band output should keep its own spectral array and reduce into this.
"""
struct RadiativeFluxes{T<:AbstractFloat, N}
    flux_up::Array{T, N}
    flux_dn::Array{T, N}
    flux_dn_dir::Array{T, N}
    flux_net::Array{T, N}
end

"""
    allocate_fluxes(T, dims, nlev) -> RadiativeFluxes

Allocate zeroed fluxes for horizontal shape `dims` (a possibly-empty
tuple) and `nlev` levels.
"""
function allocate_fluxes(::Type{T}, dims::Tuple, nlev::Int) where {T}
    shp = (dims..., nlev)
    z() = zeros(T, shp)
    return RadiativeFluxes{T, length(shp)}(z(), z(), z(), z())
end

function zero_fluxes!(f::RadiativeFluxes)
    fill!(f.flux_up, 0); fill!(f.flux_dn, 0)
    fill!(f.flux_dn_dir, 0); fill!(f.flux_net, 0)
    return f
end

"Fill `flux_net = flux_dn - flux_up`."
function net_flux!(f::RadiativeFluxes)
    @inbounds @. f.flux_net = f.flux_dn - f.flux_up
    return f
end

nlevels(f::RadiativeFluxes) = size(f.flux_up)[end]

"""
    flux_divergence!(div, f, dz) -> div

Layer flux divergence `∂F_net/∂z` from level fluxes, in W m⁻³. `div` has
the layer shape (one fewer element in the last dimension than `f`). `dz`
may be a scalar or a per-layer vector.

Sign: a *negative* divergence of the downward-positive net flux means the
layer is absorbing energy, hence warming. `heating_rate!` applies that
sign convention.
"""
function flux_divergence!(div::AbstractArray{T}, f::RadiativeFluxes{T}, dz) where {T}
    nlev = nlevels(f)
    nlay = nlev - 1
    size(div)[end] == nlay ||
        error("flux_divergence!: div has $(size(div)[end]) layers, expected $nlay")
    fn = f.flux_net
    if ndims(fn) == 1
        @inbounds for k in 1:nlay
            h = dz isa Real ? dz : dz[k]
            div[k] = (fn[k + 1] - fn[k]) / h
        end
    else
        nx, ny = size(fn, 1), size(fn, 2)
        @inbounds for k in 1:nlay, j in 1:ny, i in 1:nx
            h = dz isa Real ? dz : dz[k]
            div[i, j, k] = (fn[i, j, k + 1] - fn[i, j, k]) / h
        end
    end
    return div
end

"""
    heating_rate!(dθdt, f, dz, ρ; cp = 1004.6) -> dθdt

Radiative heating rate in K s⁻¹ from the net-flux divergence:

```math
\\left(\\frac{∂θ}{∂t}\\right)_{rad} = \\frac{1}{ρ c_p}\\frac{∂F_{net}}{∂z}
```

with `F_net` positive downward, so a downward-decreasing net flux (energy
deposited in the layer) gives positive heating. `ρ` may be a scalar or a
per-layer vector.
"""
function heating_rate!(dθdt::AbstractArray{T}, f::RadiativeFluxes{T}, dz, ρ;
                       cp::Real = CP_DRY_AIR) where {T}
    flux_divergence!(dθdt, f, dz)
    if ndims(dθdt) == 1
        @inbounds for k in eachindex(dθdt)
            r = ρ isa Real ? ρ : ρ[k]
            dθdt[k] /= (r * cp)
        end
    else
        nx, ny, nz = size(dθdt)
        @inbounds for k in 1:nz, j in 1:ny, i in 1:nx
            r = ρ isa Real ? ρ : ρ[k]
            dθdt[i, j, k] /= (r * cp)
        end
    end
    return dθdt
end

# =========================================================================
# Abstract interfaces
# =========================================================================

"""
    AbstractGasOptics

Gas optical properties from the thermodynamic and composition state.
Concrete types implement

    compute_optics!(optics, gasopt, column)

filling `optics.τ`, `.ssa`, `.g` for every layer and g-point. Phase 32
supplies the correlated-k implementation.
"""
abstract type AbstractGasOptics end

"""
    AbstractCloudOptics

Cloud optical properties from the condensate state (`ql`, `qi`, effective
radii). Implements the same `compute_optics!` contract. Phase 33.
"""
abstract type AbstractCloudOptics end

"""
    AbstractAerosolOptics

Aerosol optical properties from the aerosol state. Same contract.
Phase 37.
"""
abstract type AbstractAerosolOptics end

"""
    AbstractRTSolver

Radiative-transfer solver. Concrete types implement

    solve_rt!(fluxes, solver, optics, column, surface; μ₀, ...)

Phase 33 supplies two-stream; Phase 34 supplies 3-D Monte Carlo; Phase 35
supplies urban radiosity.
"""
abstract type AbstractRTSolver end

"""
    AbstractSurfaceOptics

Surface reflection and emission. Concrete types implement
`surface_albedo(s, band, μ₀, ...)` and `surface_emissivity(s, band)`.
Phase 36 supplies the spectral/BRDF implementation.
"""
abstract type AbstractSurfaceOptics end

# Interface stubs: these exist so the generic functions are defined and
# concrete implementations in later phases add methods rather than
# accidentally shadowing.
"""
    material_albedo(material, args...)

Shortwave reflectance of a named surface material.

Declared here, with no method, so that the modules which know about
materials can each add their own without importing one another —
the same pattern used for `surface_albedo`, `surface_emissivity`,
`compute_optics!` and `solve_rt!` below.

Implemented by:

  * `SurfaceOptics.material_albedo(name::Symbol, spec)` → `SpectralAlbedo`,
    the spectral authority: per-band direct and diffuse reflectance
    derived from a reflectance curve.
  * `Radiosity.material_albedo(m::UrbanMaterials, class)` → scalar, the
    grey per-facet-class value the radiosity solve needs. Those greys
    are themselves *derived* from the spectral curves above, so the two
    cannot drift apart.
"""
function material_albedo end

"""
    material_emissivity(material, args...)

Longwave emissivity of a named surface material. See
`material_albedo` for why this generic is declared here.
"""
function material_emissivity end

function compute_optics! end
function solve_rt! end
function surface_albedo end
function surface_emissivity end

# =========================================================================
# Cloud mass extinction (shared by TwoStream and SolarGeometry)
# =========================================================================

"""
    liquid_mass_extinction(r_eff) -> m² g⁻¹

Shortwave mass extinction coefficient of liquid cloud with effective
radius `r_eff` (μm), in the geometric-optics limit (extinction
efficiency 2):

```math
k = \\frac{3}{2\\,ρ_w\\,r_{eff}} = \\frac{1.5}{r_{eff}[\\mathrm{μm}]}\\ \\mathrm{m^2\\,g^{-1}},
```

so a layer of liquid water path `LWP` (g m⁻²) has `τ = k·LWP` (Stephens
1978; Hu & Stamnes 1993 in form). 10 μm gives 0.15 m² g⁻¹. It is the one
definition used by `TwoStream.MieCloudOptics` and by
`SolarGeometry.cloud_optical_depth_column`, so the cloud a column
radiation scheme sees and the cloud that casts a 3-D shadow have the
same optical depth.
"""
liquid_mass_extinction(r_eff::Real) = 1.5 / r_eff

"""
    ice_mass_extinction(r_eff) -> m² g⁻¹

Shortwave mass extinction coefficient of ice cloud with effective
radius `r_eff` (μm), `k = 1/r_eff` m² g⁻¹ (the form of Fu 1996 with an
effective radius). 30 μm gives 0.033 m² g⁻¹. Shared like
[`liquid_mass_extinction`](@ref).
"""
ice_mass_extinction(r_eff::Real) = 1.0 / r_eff

# =========================================================================
# Atmospheric column
# =========================================================================

"""
    AtmosphericColumn{T}

The thermodynamic and composition state a radiation scheme needs, on a
single column, all bottom-up.

Layer quantities (`Nz`): `p_lay` (Pa), `T_lay` (K), `dz` (m), `ρ` (kg m⁻³),
`vmr` (a `Dict{Symbol,Vector{T}}` of volume mixing ratios by gas),
`ql`, `qi` (kg kg⁻¹ condensate), `r_eff_liq`, `r_eff_ice` (μm).

Level quantities (`Nz+1`): `p_lev` (Pa), `T_lev` (K).

Surface: `T_sfc` (K).

`vmr` keys are gas symbols — `:h2o`, `:co2`, `:o3`, `:ch4`, `:n2o`, `:o2`,
`:n2`. A gas-optics implementation documents which it requires.
"""
struct AtmosphericColumn{T<:AbstractFloat}
    p_lay::Vector{T}
    T_lay::Vector{T}
    p_lev::Vector{T}
    T_lev::Vector{T}
    dz::Vector{T}
    ρ::Vector{T}
    vmr::Dict{Symbol, Vector{T}}
    ql::Vector{T}
    qi::Vector{T}
    r_eff_liq::Vector{T}
    r_eff_ice::Vector{T}
    T_sfc::T
end

function AtmosphericColumn(::Type{T}, nz::Int;
                           p_lay = nothing, T_lay = nothing,
                           p_lev = nothing, T_lev = nothing,
                           dz = nothing, ρ = nothing,
                           vmr = Dict{Symbol, Vector{T}}(),
                           ql = nothing, qi = nothing,
                           r_eff_liq = nothing, r_eff_ice = nothing,
                           T_sfc = T(288)) where {T}
    z(n) = zeros(T, n)
    return AtmosphericColumn{T}(
        p_lay === nothing ? z(nz) : collect(T, p_lay),
        T_lay === nothing ? z(nz) : collect(T, T_lay),
        p_lev === nothing ? z(nz+1) : collect(T, p_lev),
        T_lev === nothing ? z(nz+1) : collect(T, T_lev),
        dz    === nothing ? z(nz) : collect(T, dz),
        ρ     === nothing ? z(nz) : collect(T, ρ),
        Dict{Symbol, Vector{T}}(k => collect(T, v) for (k, v) in vmr),
        ql        === nothing ? z(nz) : collect(T, ql),
        qi        === nothing ? z(nz) : collect(T, qi),
        r_eff_liq === nothing ? fill(T(10), nz) : collect(T, r_eff_liq),
        r_eff_ice === nothing ? fill(T(30), nz) : collect(T, r_eff_ice),
        T(T_sfc))
end

nlayers(c::AtmosphericColumn) = length(c.p_lay)
Base.eltype(::AtmosphericColumn{T}) where {T} = T

function Base.show(io::IO, c::AtmosphericColumn{T}) where {T}
    print(io, "AtmosphericColumn{$T}($(nlayers(c)) layers, ",
              "p $(round(c.p_lev[1]/100; digits=1))–",
              "$(round(c.p_lev[end]/100; digits=1)) hPa, ",
              "T_sfc $(round(c.T_sfc; digits=1)) K, ",
              "gases $(sort(collect(keys(c.vmr))))")
end

"""
    column_from_simulation(sim, i, j; kwargs...) -> AtmosphericColumn

Extract a radiation column from a running `BoussinesqSimulation` at
horizontal index `(i, j)`. Because the Boussinesq system carries no
pressure or density profile, a hydrostatic reference must be supplied:
pass `p_surf` and use the simulation's θ to build `T`. Gas mixing ratios
default to a standard mid-latitude composition; override via `vmr`.

This is a convenience for coupling the radiation stack to the LES; for
the anelastic core the `ReferenceState` already carries `p̄` and `ρ̄` and
should be used instead.
"""
function column_from_simulation(sim, i::Int, j::Int;
                                p_surf::Real = 1.01325e5,
                                vmr = nothing,
                                T_sfc = nothing)
    T = eltype(sim.θ)
    m = sim.mesh
    nz = m.Nz
    Hx, Hy, Hz = m.Hx, m.Hy, m.Hz
    θ  = T[sim.θ.data[Hx + i, Hy + j, Hz + k] for k in 1:nz]
    qv = sim.has_moisture ? T[sim.qv.data[Hx + i, Hy + j, Hz + k] for k in 1:nz] : nothing
    return column_from_profiles(θ, fill(T(m.dz), nz); p_surf = p_surf, qv = qv,
                                vmr = vmr, T_sfc = T_sfc)
end

"""
    column_from_profiles(θ, dz; p_surf = 101325, qv = nothing, ql = nothing,
                         qi = nothing, vmr = nothing, T_sfc = nothing,
                         r_eff_liq = 10, r_eff_ice = 30) -> AtmosphericColumn

A radiation column from layer profiles of potential temperature `θ` (K,
reference pressure 1000 hPa) on layers of thickness `dz` (m), bottom-up.
Pressure is integrated hydrostatically from `p_surf` with the dry-air
density of each layer, `T = θ·(p/p₀)^{R_d/c_p}` at the layer mid-point;
levels get the mean of the adjacent layer temperatures (the bottom and
top level the temperature of their layer).

`qv` (specific humidity, kg/kg) becomes the `:h2o` volume mixing ratio
`qv·M_air/M_h2o`; the other gases come from `vmr` or a standard
mid-latitude composition. `ql`, `qi` (kg/kg) are the condensate the cloud
optics reads, with effective radii `r_eff_liq`, `r_eff_ice` (μm, scalars
or per-layer). `T_sfc` defaults to the lowest layer's temperature.

This is the hydrostatic core of [`column_from_simulation`](@ref), for a
caller that has already reduced the LES to a profile (a domain mean, a
masked mean over the fluid cells).
"""
function column_from_profiles(θ::AbstractVector, dz;
                              p_surf::Real = 1.01325e5,
                              qv = nothing, ql = nothing, qi = nothing,
                              vmr = nothing, T_sfc = nothing,
                              r_eff_liq = 10, r_eff_ice = 30)
    T = float(eltype(θ))
    nz = length(θ)
    dzv = dz isa Real ? fill(T(dz), nz) : collect(T, dz)
    length(dzv) == nz || error("column_from_profiles: $(length(dzv)) layer thicknesses for $nz layers")
    Rd = T(287.05); cp = T(CP_DRY_AIR); g = T(GRAVITY); p0 = T(1e5)
    p_lev = zeros(T, nz + 1); p_lev[1] = T(p_surf)
    p_lay = zeros(T, nz); T_lay = zeros(T, nz); ρ = zeros(T, nz)
    for k in 1:nz
        # iterate once: use the level-1 pressure to get T, then integrate
        π_ex = (p_lev[k] / p0)^(Rd / cp)
        Tk   = θ[k] * π_ex
        ρk   = p_lev[k] / (Rd * Tk)
        p_lev[k + 1] = p_lev[k] - ρk * g * dzv[k]
        p_lay[k] = (p_lev[k] + p_lev[k + 1]) / 2
        π_l  = (p_lay[k] / p0)^(Rd / cp)
        T_lay[k] = θ[k] * π_l
        ρ[k] = p_lay[k] / (Rd * T_lay[k])
    end
    T_lev = zeros(T, nz + 1)
    T_lev[1] = T_lay[1]
    for k in 2:nz
        T_lev[k] = (T_lay[k-1] + T_lay[k]) / 2
    end
    T_lev[nz + 1] = T_lay[nz]

    vmr_use = vmr === nothing ? _standard_vmr(T, nz) :
              Dict{Symbol, Vector{T}}(k => collect(T, v) for (k, v) in vmr)
    if qv !== nothing
        # specific humidity → volume mixing ratio
        vmr_use[:h2o] = T[q * T(28.9647 / 18.01528) for q in qv]
    end
    prof(x, d) = x === nothing ? fill(T(d), nz) : x isa Real ? fill(T(x), nz) : collect(T, x)
    return AtmosphericColumn{T}(p_lay, T_lay, p_lev, T_lev, dzv, ρ, vmr_use,
                               prof(ql, 0), prof(qi, 0),
                               prof(r_eff_liq, 10), prof(r_eff_ice, 30),
                               T_sfc === nothing ? T_lay[1] : T(T_sfc))
end

"""
    stitch_columns(lower, upper) -> AtmosphericColumn

Put the column `lower`, which spans the heights `0 … z_L` (`z_L =
Σ lower.dz`), under the part of `upper` that lies above `z_L`, with both
columns measured from the same ground. The typical use is an LES domain
a few hundred metres deep under a standard atmosphere
(`GasOptics.us_standard_column`) up to the stratosphere, so that the
downwelling longwave and the shortwave at the LES top see the whole
atmosphere.

* The layer of `upper` that straddles `z_L` is cut at `z_L`; its part
  above becomes the first upper layer (dropped if thinner than 1 mm).
* The pressure is continuous: every upper pressure (levels, layers) and
  density is scaled by `p_L / p_upper(z_L)`, where `p_L` is the top
  pressure of `lower` and `p_upper(z_L)` the upper column's own pressure
  there (log-linear in the straddling layer). This keeps the upper
  column's hydrostatic shape, and its layer masses Δp/g, up to that
  factor.
* The temperature of the joining level is `lower`'s top-level
  temperature; the upper levels and layers keep their own.
* Gases present in either column are carried over; a gas missing from
  one column has zero mixing ratio there. Condensate and effective radii
  are taken from each column. The surface temperature is `lower`'s.
"""
function stitch_columns(lower::AtmosphericColumn{T}, upper::AtmosphericColumn{T}) where {T}
    nl = length(lower.dz); nu = length(upper.dz)
    zL = sum(lower.dz)
    zu = zeros(T, nu + 1)
    for k in 1:nu
        zu[k + 1] = zu[k] + upper.dz[k]
    end
    zu[end] > zL || error("stitch_columns: the upper column (top $(zu[end]) m) does not reach above the lower one ($zL m)")
    m = findfirst(k -> zu[k + 1] > zL, 1:nu)             # straddling layer
    # upper pressure at z_L, log-linear within layer m
    f = (zL - zu[m]) / upper.dz[m]
    pz = exp((1 - f) * log(upper.p_lev[m]) + f * log(upper.p_lev[m + 1]))
    s = lower.p_lev[end] / pz
    keep_partial = zu[m + 1] - zL > T(1e-3)
    ups = keep_partial ? (m:nu) : (m + 1:nu)
    nup = length(ups)
    n = nl + nup
    p_lay = zeros(T, n); T_lay = zeros(T, n); dz = zeros(T, n); ρ = zeros(T, n)
    p_lev = zeros(T, n + 1); T_lev = zeros(T, n + 1)
    ql = zeros(T, n); qi = zeros(T, n); rl = zeros(T, n); ri = zeros(T, n)
    p_lev[1:nl + 1] .= lower.p_lev
    T_lev[1:nl + 1] .= lower.T_lev
    for k in 1:nl
        p_lay[k] = lower.p_lay[k]; T_lay[k] = lower.T_lay[k]; dz[k] = lower.dz[k]
        ρ[k] = lower.ρ[k]; ql[k] = lower.ql[k]; qi[k] = lower.qi[k]
        rl[k] = lower.r_eff_liq[k]; ri[k] = lower.r_eff_ice[k]
    end
    for (a, ku) in enumerate(ups)
        k = nl + a
        p_lev[k + 1] = s * upper.p_lev[ku + 1]
        T_lev[k + 1] = upper.T_lev[ku + 1]
        dz[k] = zu[ku + 1] - max(zu[ku], zL)
        p_lay[k] = (p_lev[k] + p_lev[k + 1]) / 2
        T_lay[k] = upper.T_lay[ku]
        ρ[k] = s * upper.ρ[ku]
        ql[k] = upper.ql[ku]; qi[k] = upper.qi[ku]
        rl[k] = upper.r_eff_liq[ku]; ri[k] = upper.r_eff_ice[ku]
    end
    vmr = Dict{Symbol, Vector{T}}()
    for gas in union(keys(lower.vmr), keys(upper.vmr))
        v = zeros(T, n)
        haskey(lower.vmr, gas) && (v[1:nl] .= lower.vmr[gas])
        if haskey(upper.vmr, gas)
            for (a, ku) in enumerate(ups)
                v[nl + a] = upper.vmr[gas][ku]
            end
        end
        vmr[gas] = v
    end
    return AtmosphericColumn{T}(p_lay, T_lay, p_lev, T_lev, dz, ρ, vmr,
                               ql, qi, rl, ri, lower.T_sfc)
end

# Standard mid-latitude-summer-ish composition; well-mixed gases constant.
function _standard_vmr(::Type{T}, nz::Int) where {T}
    return Dict{Symbol, Vector{T}}(
        :h2o => fill(T(5e-3), nz),
        :co2 => fill(T(420e-6), nz),
        :o3  => fill(T(5e-8), nz),
        :ch4 => fill(T(1.9e-6), nz),
        :n2o => fill(T(330e-9), nz),
        :o2  => fill(T(0.2095), nz),
        :n2  => fill(T(0.7808), nz),
    )
end

end # module
