module MonteCarloRT

# Phase 34 — 3-D Monte-Carlo radiative transfer.
#
# Two-stream (Phase 33) collapses the radiation field onto two hemispheric
# fluxes in a stack of horizontally infinite layers. That is the right
# engineering trade for a global model, but it cannot represent anything
# an LES actually resolves: a cloud that shadows the surface *next to* it
# rather than beneath it, photons that leak out of the side of a cumulus,
# a street canyon that traps shortwave between two walls. Those are the
# "3-D radiative effects", and the only reference method that gets them
# right without further approximation is Monte-Carlo photon transport.
#
# The scheme here is a forward (sun-to-sensor) Monte-Carlo solver on a
# Cartesian voxel grid with periodic lateral boundaries, i.e. exactly the
# geometry of the LES domain. What makes such a solver usable rather than
# a curiosity is variance reduction; four techniques are used, and each is
# noted where it appears:
#
#   1. **Exact voxel traversal** (Amanatides & Woo 1987, "A fast voxel
#      traversal algorithm for ray tracing"). Optical depth is accumulated
#      segment-by-segment along the exact chord through each voxel, so a
#      heterogeneous extinction field is integrated *exactly*, not
#      approximated by fixed-length steps. Fixed steps are the single most
#      common source of bias in home-grown MC-RT codes: they systematically
#      miss thin high-extinction features and they make the answer depend
#      on the step size.
#
#   2. **Implicit capture** (survival weighting). A photon is never killed
#      by absorption; instead `(1-ω₀)w` is deposited as absorbed energy and
#      the weight becomes `ω₀w`. Every photon therefore contributes to the
#      absorption field on every collision rather than one photon in
#      `1/(1-ω₀)` contributing everything.
#
#   3. **Local (next-event) estimation**, in two forms:
#
#      * *Boundary/level crossing estimator.* Along every traced ray, at
#        every level crossing and at the ray's terminus we score
#        `w·e^{-τ}` — the energy that *would* have travelled that far
#        without colliding — instead of waiting to see whether the photon
#        actually got there. This replaces a Bernoulli counter of variance
#        `p(1-p)` with its deterministic mean, and it makes the unscattered
#        direct beam *exact*: the first segment of every photon scores
#        `e^{-τ_slant}`, i.e. Beer-Lambert with zero variance.
#      * *Radiance estimator.* At every scattering and every surface
#        reflection, the contribution to the top-of-domain radiance in each
#        requested viewing direction is accumulated as
#        `w·P(Θ)/(4π)·e^{-τ_esc}/μ_v` (phase function × escape
#        transmittance), which is the standard local estimate used by
#        MYSTIC (Mayer 2009) and SHDOM-family reference codes.
#
#   4. **Russian roulette** on low-weight photons, with the unbiased weight
#      boost `w ← w/p_survive`. Without it the implicit-capture weight
#      decays geometrically and photons never terminate; with a naive
#      cutoff instead of roulette the result is biased low.
#
# Uncertainty. Every reported quantity carries a standard error. Photons
# are run in `nbatch` equal batches; the batch sums are i.i.d., so the
# sample variance across batches gives an honest standard error for
# scalars *and* for every element of the 3-D absorption field and the
# surface irradiance maps. The error must fall as `N^{-1/2}`; the test
# suite checks that it does.
#
# Energy. Weight is conserved exactly along a photon path: every unit of
# launched weight ends up in exactly one of {atmospheric absorption,
# surface absorption, building absorption, escape through the top,
# Russian-roulette imbalance, geometric cutoff}. `mc_energy_budget`
# reports all six and the residual is zero to machine precision — a much
# stronger statement than "conserves on average", and it is what catches
# a mis-deposited weight immediately.
#
# Urban geometry. Voxels may be flagged *solid* (buildings). A solid voxel
# does not transmit: a ray terminates on its face, `(1-α)w` is absorbed by
# the building and `αw` is re-emitted as a Lambertian (cosine) reflection
# about the face normal. The mask convention is Phase 13's `SolidMask`
# (equivalently `CutCell.solid_mask(geom)`), so the same buildings that the
# flow solver sees are the ones the photons see.
#
# References:
#   Marchuk et al. (1980), *The Monte Carlo Methods in Atmospheric Optics*
#   Cashwell & Everett (1959) — rotation into the laboratory frame
#   Henyey & Greenstein (1941) — the phase function and its inverse CDF
#   Amanatides & Woo (1987) — voxel traversal
#   Marshak & Davis (2005), *3D Radiative Transfer in Cloudy Atmospheres*
#   Mayer (2009), "Radiative transfer in the cloudy atmosphere" (MYSTIC)
#   Cahalan et al. (2005) — the I3RC Monte-Carlo intercomparison cases

using Random
using Random: AbstractRNG

using ..Grids: RegularGrid
using ..IBM: SolidMask
using ..RadiationCore
using ..RadiationCore: SpectralGrid, OpticalProperties, RadiativeFluxes,
                       AtmosphericColumn, AbstractRTSolver,
                       nbands, ngpoints, gpoint_range, allocate_fluxes,
                       net_flux!, CP_DRY_AIR
import ..RadiationCore: solve_rt!

export MCDomain, mc_domain, domain_extent, domain_top,
       MCMedium, mc_medium, homogeneous_medium, medium_from_optics,
       set_solid!, solid_from_mask!, add_box_building!, optical_depth_column,
       MonteCarloSolver, monte_carlo_solver,
       MCResult, mc_solve, mc_energy_budget, mc_closure_residual,
       mc_heating_rate, mc_surface_irradiance, mc_sun_direction,
       sample_free_path, sample_henyey_greenstein, sample_rayleigh,
       henyey_greenstein_phase, rayleigh_phase,
       rotate_scatter, scatter_direction, sample_lambertian

# =========================================================================
# Domain
# =========================================================================

"""
    MCDomain{T}

Uniform Cartesian voxel grid the photons travel through: `nx × ny × nz`
voxels of size `dx × dy × dz` with lower corner `(x0, y0, z0)`.

Boundary conventions, fixed here and assumed everywhere below:

  * **x, y — periodic.** A photon leaving the east face re-enters at the
    west face. This matches the LES lateral boundary conditions and makes
    the domain a statistically infinite, horizontally heterogeneous slab.
  * **z = z0 — the ground.** A reflecting Lambertian surface with a
    per-cell albedo; what is not reflected is absorbed there.
  * **z = z0 + nz·dz — the top.** Photons that cross it have escaped to
    space and are scored as top-of-domain upwelling flux.

Voxel `(i,j,k)` spans `[x0+(i-1)dx, x0+i·dx] × … × [z0+(k-1)dz, z0+k·dz]`,
so `k = 1` is the layer touching the ground — the same bottom-up ordering
as `OpticalProperties` and `RadiativeFluxes`.
"""
struct MCDomain{T<:AbstractFloat}
    nx::Int
    ny::Int
    nz::Int
    dx::T
    dy::T
    dz::T
    x0::T
    y0::T
    z0::T
end

"""
    mc_domain(T = Float64; size, spacing, origin = (0,0,0)) -> MCDomain
    mc_domain(mesh::RegularGrid)                            -> MCDomain

Build a voxel grid. `size` is `(nx, ny, nz)` and `spacing` is
`(dx, dy, dz)` in metres. The `RegularGrid` form takes the interior of an
Aeolus mesh, so the radiation grid and the flow grid coincide cell for
cell.
"""
function mc_domain(::Type{T} = Float64; size::NTuple{3,Int},
                   spacing::NTuple{3,Real},
                   origin::NTuple{3,Real} = (0, 0, 0)) where {T<:AbstractFloat}
    nx, ny, nz = size
    (nx > 0 && ny > 0 && nz > 0) || error("mc_domain: every dimension must be ≥ 1")
    all(>(0), spacing) || error("mc_domain: spacings must be positive")
    return MCDomain{T}(nx, ny, nz, T(spacing[1]), T(spacing[2]), T(spacing[3]),
                       T(origin[1]), T(origin[2]), T(origin[3]))
end

mc_domain(mesh::RegularGrid{T}) where {T} =
    MCDomain{T}(mesh.Nx, mesh.Ny, mesh.Nz, mesh.dx, mesh.dy, mesh.dz,
                mesh.x0, mesh.y0, mesh.z0)

Base.size(d::MCDomain) = (d.nx, d.ny, d.nz)
Base.eltype(::MCDomain{T}) where {T} = T

"Lateral and vertical extent `(Lx, Ly, Lz)` of the domain, in metres."
domain_extent(d::MCDomain) = (d.nx * d.dx, d.ny * d.dy, d.nz * d.dz)

"Height of the model top, `z0 + nz·dz`."
domain_top(d::MCDomain) = d.z0 + d.nz * d.dz

"Horizontal area of the domain, `Lx·Ly` (m²)."
domain_area(d::MCDomain) = (d.nx * d.dx) * (d.ny * d.dy)

function Base.show(io::IO, d::MCDomain{T}) where {T}
    Lx, Ly, Lz = domain_extent(d)
    print(io, "MCDomain{$T}($(d.nx)×$(d.ny)×$(d.nz) voxels, ",
              "$(round(Lx; digits=1))×$(round(Ly; digits=1))×",
              "$(round(Lz; digits=1)) m)")
end

# =========================================================================
# Medium
# =========================================================================

"""
    MCMedium{T}

The optical state of the domain at one wavelength (one g-point):

  * `kext`  — extinction coefficient per voxel, m⁻¹. The optical depth of
    a chord of length `s` through the voxel is `kext·s`.
  * `ssa`   — single-scattering albedo per voxel, `ω₀ ∈ [0,1]`.
  * `g`     — Henyey-Greenstein asymmetry parameter per voxel.
  * `phase` — `:henyey_greenstein` or `:rayleigh`. Rayleigh ignores `g`.
  * `solid` — `true` marks a building voxel: it does not transmit.
  * `solid_albedo` — Lambertian albedo of each solid voxel's surface.
  * `sfc_albedo`   — Lambertian albedo of the ground, per `(i,j)` cell.

`kext` is stored rather than a per-voxel optical depth because the chord
through a voxel is not `dz` for a slanted ray; storing an extinction
*coefficient* is what lets the traversal integrate the slant path exactly.
"""
struct MCMedium{T<:AbstractFloat}
    domain::MCDomain{T}
    kext::Array{T,3}
    ssa::Array{T,3}
    g::Array{T,3}
    phase::Symbol
    solid::Array{Bool,3}
    solid_albedo::Array{T,3}
    sfc_albedo::Array{T,2}
end

_as3(::Type{T}, d::MCDomain, v::Real) where {T} = fill(T(v), d.nx, d.ny, d.nz)
function _as3(::Type{T}, d::MCDomain, a::AbstractArray{<:Real,3}) where {T}
    size(a) == (d.nx, d.ny, d.nz) ||
        error("MCMedium: field has size $(size(a)), expected $((d.nx, d.ny, d.nz))")
    return Array{T,3}(a)
end

_as2(::Type{T}, d::MCDomain, v::Real) where {T} = fill(T(v), d.nx, d.ny)
function _as2(::Type{T}, d::MCDomain, a::AbstractArray{<:Real,2}) where {T}
    size(a) == (d.nx, d.ny) ||
        error("MCMedium: surface map has size $(size(a)), expected $((d.nx, d.ny))")
    return Array{T,2}(a)
end

_asbool3(d::MCDomain, ::Nothing) = falses(d.nx, d.ny, d.nz) |> Array
function _asbool3(d::MCDomain, a::AbstractArray{Bool,3})
    size(a) == (d.nx, d.ny, d.nz) ||
        error("MCMedium: solid mask has size $(size(a)), expected $((d.nx, d.ny, d.nz))")
    return Array{Bool,3}(a)
end

"""
    mc_medium(domain; kext = 0, ssa = 0, g = 0, phase = :henyey_greenstein,
              solid = nothing, solid_albedo = 0.3, sfc_albedo = 0.0)

Build an [`MCMedium`](@ref). Every optical field accepts either a scalar
(uniform) or a full `(nx, ny, nz)` array; `sfc_albedo` accepts a scalar or
an `(nx, ny)` map.
"""
function mc_medium(domain::MCDomain{T};
                   kext = zero(T), ssa = zero(T), g = zero(T),
                   phase::Symbol = :henyey_greenstein,
                   solid = nothing, solid_albedo = T(0.3),
                   sfc_albedo = zero(T)) where {T}
    phase === :henyey_greenstein || phase === :rayleigh ||
        error("mc_medium: phase must be :henyey_greenstein or :rayleigh, got $phase")
    k = _as3(T, domain, kext)
    w = _as3(T, domain, ssa)
    a = _as3(T, domain, g)
    any(<(0), k) && error("mc_medium: extinction coefficient must be ≥ 0")
    all(x -> 0 <= x <= 1, w) || error("mc_medium: ssa must lie in [0,1]")
    all(x -> -1 < x < 1, a) || error("mc_medium: asymmetry parameter must lie in (-1,1)")
    sm = _asbool3(domain, solid)
    sa = _as3(T, domain, solid_albedo)
    fa = _as2(T, domain, sfc_albedo)
    all(x -> 0 <= x <= 1, sa) || error("mc_medium: solid albedo must lie in [0,1]")
    all(x -> 0 <= x <= 1, fa) || error("mc_medium: surface albedo must lie in [0,1]")
    return MCMedium{T}(domain, k, w, a, phase, sm, sa, fa)
end

"""
    homogeneous_medium(domain; τ, ssa, g = 0, kwargs...) -> MCMedium

Convenience constructor for the plane-parallel validation problems: a
uniform medium whose *column* optical depth is `τ`, i.e.
`kext = τ / (nz·dz)`. Remaining keywords are forwarded to
[`mc_medium`](@ref).
"""
function homogeneous_medium(domain::MCDomain{T}; τ::Real, ssa::Real,
                            g::Real = 0, kwargs...) where {T}
    _, _, Lz = domain_extent(domain)
    return mc_medium(domain; kext = T(τ) / T(Lz), ssa = T(ssa), g = T(g),
                     kwargs...)
end

"""
    medium_from_optics(domain, optics, gpt; kwargs...) -> MCMedium

Build a medium from the shared [`OpticalProperties`](@ref) contract at
g-point `gpt`. `optics` may be column-shaped `(nz, ngpt)` — the profile is
then replicated across every column — or 3-D `(nx, ny, nz, ngpt)`.

`OpticalProperties` stores a layer optical depth `τ`, which is defined for
a *vertical* path; the conversion is `kext = τ / dz`.
"""
function medium_from_optics(domain::MCDomain{T}, optics::OpticalProperties{T},
                            gpt::Int; kwargs...) where {T}
    nd = ndims(optics.τ)
    k = Array{T,3}(undef, domain.nx, domain.ny, domain.nz)
    w = similar(k); a = similar(k)
    if nd == 2
        size(optics.τ, 1) == domain.nz ||
            error("medium_from_optics: optics has $(size(optics.τ,1)) layers, ",
                  "domain has $(domain.nz)")
        @inbounds for kk in 1:domain.nz, jj in 1:domain.ny, ii in 1:domain.nx
            k[ii, jj, kk] = optics.τ[kk, gpt] / domain.dz
            w[ii, jj, kk] = optics.ssa[kk, gpt]
            a[ii, jj, kk] = optics.g[kk, gpt]
        end
    elseif nd == 4
        size(optics.τ)[1:3] == (domain.nx, domain.ny, domain.nz) ||
            error("medium_from_optics: optics shape $(size(optics.τ)[1:3]) ≠ ",
                  "domain $((domain.nx, domain.ny, domain.nz))")
        @inbounds for kk in 1:domain.nz, jj in 1:domain.ny, ii in 1:domain.nx
            k[ii, jj, kk] = optics.τ[ii, jj, kk, gpt] / domain.dz
            w[ii, jj, kk] = optics.ssa[ii, jj, kk, gpt]
            a[ii, jj, kk] = optics.g[ii, jj, kk, gpt]
        end
    else
        error("medium_from_optics: optics must be 2-D (column) or 4-D (3-D field)")
    end
    return mc_medium(domain; kext = k, ssa = w, g = a, kwargs...)
end

"""
    set_solid!(medium, predicate) -> medium

Flag every voxel whose *centre* `(x, y, z)` satisfies `predicate(x, y, z)`
as a solid building voxel.
"""
function set_solid!(med::MCMedium, predicate)
    d = med.domain
    @inbounds for k in 1:d.nz, j in 1:d.ny, i in 1:d.nx
        x = d.x0 + (i - 0.5) * d.dx
        y = d.y0 + (j - 0.5) * d.dy
        z = d.z0 + (k - 0.5) * d.dz
        med.solid[i, j, k] = predicate(x, y, z)
    end
    return med
end

"""
    solid_from_mask!(medium, sm::SolidMask) -> medium

Copy a Phase-13 [`SolidMask`](@ref) (halo-padded storage) into the
medium's solid flags. `CutCell.solid_mask(geom)` returns exactly such a
mask, so cut-cell urban geometry drives the photon transport directly.
"""
function solid_from_mask!(med::MCMedium, sm::SolidMask)
    d = med.domain
    m = sm.mesh
    (m.Nx, m.Ny, m.Nz) == (d.nx, d.ny, d.nz) ||
        error("solid_from_mask!: mask grid $((m.Nx, m.Ny, m.Nz)) ≠ ",
              "domain $((d.nx, d.ny, d.nz))")
    @inbounds for k in 1:d.nz, j in 1:d.ny, i in 1:d.nx
        med.solid[i, j, k] = sm.mask[m.Hx + i, m.Hy + j, m.Hz + k]
    end
    return med
end

"""
    add_box_building!(medium, irange, jrange, nlevels; albedo = nothing)

Mark a rectangular block of voxels solid — the simplest urban element.
`nlevels` counts voxels upward from the ground (`k = 1:nlevels`).
"""
function add_box_building!(med::MCMedium{T}, irange, jrange, nlevels::Int;
                           albedo = nothing) where {T}
    d = med.domain
    for k in 1:min(nlevels, d.nz), j in jrange, i in irange
        (1 <= i <= d.nx && 1 <= j <= d.ny) ||
            error("add_box_building!: index ($i,$j) outside the domain")
        med.solid[i, j, k] = true
        albedo === nothing || (med.solid_albedo[i, j, k] = T(albedo))
    end
    return med
end

"""
    optical_depth_column(medium, i, j) -> T

Vertical optical depth of column `(i,j)`, `Σ_k kext·dz`. Solid voxels are
skipped — they are opaque, not optically thick.
"""
function optical_depth_column(med::MCMedium{T}, i::Int, j::Int) where {T}
    d = med.domain
    s = zero(T)
    @inbounds for k in 1:d.nz
        med.solid[i, j, k] || (s += med.kext[i, j, k] * d.dz)
    end
    return s
end

# =========================================================================
# Sampling routines — every one takes an `rng`
# =========================================================================

"""
    sample_free_path(rng, T = Float64) -> τ

Optical depth to the next collision, `τ = -ln ξ` with `ξ` uniform on
`(0,1]`. This is the inverse CDF of the Beer-Lambert survival probability
`e^{-τ}`, so the sample is exact for *any* extinction field: the
heterogeneity is handled later, by marching until the accumulated optical
depth reaches `τ`, not by re-sampling per voxel.

`-log1p(-ξ)` is used rather than `-log(ξ)` because Julia's `rand` returns
`ξ ∈ [0,1)`; `1-ξ ∈ (0,1]` can never be zero, so the result can never be
`Inf`.
"""
@inline sample_free_path(rng::AbstractRNG, ::Type{T} = Float64) where {T} =
    -log1p(-rand(rng, T))

"""
    henyey_greenstein_phase(g, μ) -> P

The Henyey-Greenstein phase function at `μ = cos Θ`, normalised so that
`∫ P dΩ / 4π = 1` (isotropic scattering gives `P ≡ 1`):

```math
P(μ) = \\frac{1-g^2}{(1+g^2-2gμ)^{3/2}}
```
"""
@inline function henyey_greenstein_phase(g::T, μ::Real) where {T<:AbstractFloat}
    m = T(μ)
    den = 1 + g * g - 2 * g * m
    den = max(den, T(1e-12))
    return (1 - g * g) / (den * sqrt(den))
end

henyey_greenstein_phase(g::Real, μ::Real) =
    henyey_greenstein_phase(float(g), float(μ))

"""
    rayleigh_phase(μ) -> P

Rayleigh (molecular) phase function, `P(μ) = ¾(1+μ²)`, with the same
`∫ P dΩ / 4π = 1` normalisation. Its first moment is zero (forward and
backward scattering are equally likely) and its second moment is `2/5`.
"""
@inline rayleigh_phase(μ::Real) = 0.75 * (1 + float(μ)^2)

"""
    sample_henyey_greenstein(rng, g) -> μ

Sample `cos Θ` from the Henyey-Greenstein phase function by its
closed-form inverse CDF (Witt 1977):

```math
μ = \\frac{1}{2g}\\left[1+g^2-\\left(\\frac{1-g^2}{1-g+2gξ}\\right)^2\\right]
```

The expression is `0/0` at `g = 0`; the `|g| < 1e-6` branch uses the
isotropic limit `μ = 2ξ-1` instead. Forgetting that branch is the classic
way to get `NaN` directions out of an otherwise correct solver.
"""
@inline function sample_henyey_greenstein(rng::AbstractRNG, g::T) where {T<:AbstractFloat}
    ξ = rand(rng, T)
    if abs(g) < T(1e-6)
        return 2ξ - one(T)
    end
    gc = clamp(g, -one(T) + T(1e-6), one(T) - T(1e-6))
    s = (1 - gc * gc) / (1 - gc + 2 * gc * ξ)
    μ = (1 + gc * gc - s * s) / (2 * gc)
    return clamp(μ, -one(T), one(T))
end

sample_henyey_greenstein(rng::AbstractRNG, g::Real) =
    sample_henyey_greenstein(rng, float(g))

"""
    sample_rayleigh(rng, T = Float64) -> μ

Sample `cos Θ` from the Rayleigh phase function. The CDF is a cubic,

```math
F(μ) = \\tfrac{1}{2} + \\tfrac{3}{8}\\left(μ + \\tfrac{μ^3}{3}\\right),
```

so inverting `F(μ) = ξ` means solving `μ³ + 3μ - (8ξ-4) = 0`. The
discriminant is strictly positive, so Cardano's formula gives the single
real root directly — no rejection sampling and no iteration:

```math
a = 4ξ-2,\\quad D=\\sqrt{a^2+1},\\quad μ = \\sqrt[3]{a+D}+\\sqrt[3]{a-D}
```
"""
@inline function sample_rayleigh(rng::AbstractRNG, ::Type{T} = Float64) where {T}
    ξ = rand(rng, T)
    a = 4ξ - 2
    D = sqrt(a * a + one(T))
    return clamp(cbrt(a + D) + cbrt(a - D), -one(T), one(T))
end

"""
    rotate_scatter(ux, uy, uz, μ, φ) -> (vx, vy, vz)

Rotate a unit direction by a scattering angle `Θ` (`μ = cos Θ`) and an
azimuth `φ` about the incident direction, into the laboratory frame
(Cashwell & Everett 1959):

```math
\\begin{aligned}
v_x &= \\frac{\\sinΘ}{\\sqrt{1-u_z^2}}(u_xu_z\\cosφ - u_y\\sinφ) + u_xμ\\\\
v_y &= \\frac{\\sinΘ}{\\sqrt{1-u_z^2}}(u_yu_z\\cosφ + u_x\\sinφ) + u_yμ\\\\
v_z &= -\\sinΘ\\cosφ\\sqrt{1-u_z^2} + u_zμ
\\end{aligned}
```

**The `|u_z| → 1` special case matters.** The formula divides by
`√(1-u_z²)`, which vanishes for a vertically travelling photon — precisely
the direction every solar photon starts in when the sun is overhead, and
the direction half the photons in a thick cloud end up near. The limit is
simply `v = (sinΘ cosφ, sinΘ sinφ, sign(u_z)·μ)`, applied below whenever
`1-u_z² < 10⁻¹²`. Omitting it produces `NaN` or, worse, silently biased
directions when the division is merely ill-conditioned.
"""
@inline function rotate_scatter(ux::T, uy::T, uz::T, μ::T, φ::T) where {T<:AbstractFloat}
    m = clamp(μ, -one(T), one(T))
    sθ = sqrt(max(one(T) - m * m, zero(T)))
    cφ = cos(φ); sφ = sin(φ)
    s2 = one(T) - uz * uz
    local vx::T, vy::T, vz::T
    if s2 > T(1e-12)
        sq = sqrt(s2)
        vx = sθ * (ux * uz * cφ - uy * sφ) / sq + ux * m
        vy = sθ * (uy * uz * cφ + ux * sφ) / sq + uy * m
        vz = -sθ * cφ * sq + uz * m
    else
        sgn = uz >= 0 ? one(T) : -one(T)
        vx = sθ * cφ
        vy = sθ * sφ
        vz = sgn * m
    end
    n = sqrt(vx * vx + vy * vy + vz * vz)
    n <= 0 && return (zero(T), zero(T), one(T))
    return (vx / n, vy / n, vz / n)
end

rotate_scatter(u::NTuple{3,<:Real}, μ::Real, φ::Real) =
    rotate_scatter(float(u[1]), float(u[2]), float(u[3]), float(μ), float(φ))

"""
    scatter_direction(rng, u, g, phase) -> (vx, vy, vz)

Sample a new propagation direction: draw `cos Θ` from the phase function
(`:henyey_greenstein` or `:rayleigh`), draw the azimuth uniformly on
`[0, 2π)` — both phase functions are azimuthally symmetric about the
incident direction — and rotate into the laboratory frame.
"""
@inline function scatter_direction(rng::AbstractRNG, ux::T, uy::T, uz::T,
                                   g::T, phase::Symbol) where {T<:AbstractFloat}
    μ = phase === :rayleigh ? sample_rayleigh(rng, T) :
                              sample_henyey_greenstein(rng, g)
    φ = 2 * T(π) * rand(rng, T)
    return rotate_scatter(ux, uy, uz, μ, φ)
end

scatter_direction(rng::AbstractRNG, u::NTuple{3,<:Real}, g::Real, phase::Symbol) =
    scatter_direction(rng, float(u[1]), float(u[2]), float(u[3]),
                      float(g), phase)

"""
    sample_lambertian(rng, nx, ny, nz) -> (vx, vy, vz)

Sample a direction from the cosine (Lambertian) distribution about the
unit normal `n`. The polar angle satisfies `cos θ = √ξ`, which is the
inverse CDF of `p(θ) ∝ cos θ sin θ` — the distribution whose radiance is
isotropic, i.e. a perfectly diffuse reflector. Sampling `cos θ = ξ`
instead (a common slip) makes the surface too bright at grazing angles.
"""
@inline function sample_lambertian(rng::AbstractRNG, nx::T, ny::T, nz::T) where {T<:AbstractFloat}
    μ = sqrt(rand(rng, T))
    φ = 2 * T(π) * rand(rng, T)
    return rotate_scatter(nx, ny, nz, μ, φ)
end

sample_lambertian(rng::AbstractRNG, n::NTuple{3,<:Real}) =
    sample_lambertian(rng, float(n[1]), float(n[2]), float(n[3]))

"""
    mc_sun_direction(x; azimuth = 0.0) -> (east, north, up)

Unit vector pointing **toward** the sun in the local east-north-up frame —
the same convention as `SolarGeometry.sun_direction_enu`. Accepts

  * a 3-tuple, which is simply normalised;
  * a `SolarPosition` (or any object carrying `elevation` and `azimuth`
    fields in degrees), from which the ENU vector is built;
  * a cosine of the solar zenith angle `μ₀ ∈ (0,1]`, with the azimuth
    given separately in degrees — the form the 1-D validation cases use.

Accepting `SolarPosition` structurally rather than by type avoids a
module-load-order dependency on Phase 38 while remaining fully compatible
with it.
"""
function mc_sun_direction(v::NTuple{3,<:Real}; azimuth = nothing)
    n = sqrt(float(v[1])^2 + float(v[2])^2 + float(v[3])^2)
    n > 0 || error("mc_sun_direction: zero-length direction")
    return (float(v[1]) / n, float(v[2]) / n, float(v[3]) / n)
end

function mc_sun_direction(μ₀::Real; azimuth::Real = 0.0)
    0 < μ₀ <= 1 || error("mc_sun_direction: μ₀ must lie in (0,1]")
    ch = sqrt(max(1 - float(μ₀)^2, 0.0))
    ar = deg2rad(float(azimuth))
    return (ch * sin(ar), ch * cos(ar), float(μ₀))
end

function mc_sun_direction(p; azimuth = nothing)
    (hasproperty(p, :elevation) && hasproperty(p, :azimuth)) ||
        error("mc_sun_direction: expected a 3-tuple, a μ₀ value, or an object ",
              "with `elevation` and `azimuth` fields; got $(typeof(p))")
    hr = deg2rad(float(getproperty(p, :elevation)))
    ar = deg2rad(float(getproperty(p, :azimuth)))
    ch = cos(hr)
    return (ch * sin(ar), ch * cos(ar), sin(hr))
end

# =========================================================================
# Solver
# =========================================================================

"""
    MonteCarloSolver{T} <: AbstractRTSolver

3-D Monte-Carlo shortwave solver.

  * `nphoton` — photons per solve. Rounded *up* to a multiple of `nbatch`
    so that all batches are the same size and the batch variance is an
    unbiased variance estimate.
  * `nbatch` — number of independent batches used for the standard-error
    estimate. Eight is the practical minimum for a stable variance; the
    default of 16 costs nothing.
  * `weight_min`, `rr_survival` — Russian-roulette threshold and survival
    probability. A photon whose weight falls below `weight_min` survives
    with probability `rr_survival` and has its weight divided by that
    probability, which leaves the expected weight unchanged.
  * `max_scatter` — hard cap on scattering events per photon.
  * `local_estimation` — enable the next-event estimators. Turning it off
    leaves a pure analogue solver; the difference in standard error at
    fixed cost is the measure of what the variance reduction buys.
  * `stratify` — launch photons on a jittered stratified grid over the
    domain top instead of independent uniform points. Removes the
    clustering noise from the surface irradiance maps at no cost.
  * `tau_cutoff` — abandon a local-estimation ray once `e^{-τ}` is beyond
    negligible (default `τ = 30`, i.e. 10⁻¹³).
  * `max_steps` — voxel-traversal step cap, a guard against a pathological
    near-horizontal ray in an optically empty domain.
  * `view_dirs` — viewing directions for the top-of-domain radiance
    estimator. Empty by default because each direction costs one extra ray
    per scattering event.
"""
struct MonteCarloSolver{T<:AbstractFloat, S} <: AbstractRTSolver
    spec::S
    nphoton::Int
    nbatch::Int
    weight_min::T
    rr_survival::T
    max_scatter::Int
    local_estimation::Bool
    stratify::Bool
    tau_cutoff::T
    max_steps::Int
    view_dirs::Vector{NTuple{3,T}}
end

"""
    monte_carlo_solver([spec]; nphoton = 100_000, kwargs...) -> MonteCarloSolver

Construct a solver. Pass a shortwave [`SpectralGrid`](@ref) to use the
solver through the `solve_rt!` contract; omit it for monochromatic
(single-wavelength) work driven directly by [`mc_solve`](@ref).
See [`MonteCarloSolver`](@ref) for the keyword meanings.
"""
function monte_carlo_solver(::Type{T} = Float64;
                            nphoton::Integer = 100_000,
                            nbatch::Integer = 16,
                            weight_min::Real = 1e-3,
                            rr_survival::Real = 0.2,
                            max_scatter::Integer = 10_000,
                            local_estimation::Bool = true,
                            stratify::Bool = true,
                            tau_cutoff::Real = 30.0,
                            max_steps::Integer = 50_000,
                            view_dirs = NTuple{3,T}[]) where {T<:AbstractFloat}
    nbatch >= 2 || error("monte_carlo_solver: nbatch must be ≥ 2 for a variance estimate")
    nphoton >= nbatch || error("monte_carlo_solver: nphoton must be ≥ nbatch")
    0 < rr_survival <= 1 || error("monte_carlo_solver: rr_survival must lie in (0,1]")
    weight_min >= 0 || error("monte_carlo_solver: weight_min must be ≥ 0")
    per = cld(Int(nphoton), Int(nbatch))
    vd = NTuple{3,T}[mc_sun_direction(v) for v in view_dirs]
    return MonteCarloSolver{T, Nothing}(nothing, per * Int(nbatch), Int(nbatch),
                                        T(weight_min), T(rr_survival),
                                        Int(max_scatter), local_estimation,
                                        stratify, T(tau_cutoff), Int(max_steps), vd)
end

function monte_carlo_solver(spec::SpectralGrid{T}; kwargs...) where {T}
    s = monte_carlo_solver(T; kwargs...)
    return MonteCarloSolver{T, SpectralGrid{T}}(spec, s.nphoton, s.nbatch,
                                                s.weight_min, s.rr_survival,
                                                s.max_scatter, s.local_estimation,
                                                s.stratify, s.tau_cutoff,
                                                s.max_steps, s.view_dirs)
end

Base.eltype(::MonteCarloSolver{T}) where {T} = T

function Base.show(io::IO, s::MonteCarloSolver{T}) where {T}
    print(io, "MonteCarloSolver{$T}($(s.nphoton) photons, $(s.nbatch) batches",
              s.local_estimation ? ", local estimation" : ", analogue", ")")
end

# =========================================================================
# Tallies with batch-based variance
# =========================================================================

# A quantity is accumulated into `batch`; at the end of each batch the
# batch total is folded into `total` and `totalsq`. Batch totals are
# i.i.d. across batches, so their sample variance is an honest estimator
# of the Monte-Carlo variance — including for every element of a 3-D
# field, at the cost of two extra arrays rather than `nbatch` of them.
mutable struct BatchAccum{T<:AbstractFloat, N}
    batch::Array{T,N}
    total::Array{T,N}
    totalsq::Array{T,N}
end

BatchAccum(::Type{T}, dims::Tuple) where {T} =
    BatchAccum{T,length(dims)}(zeros(T, dims), zeros(T, dims), zeros(T, dims))

function flush_batch!(a::BatchAccum{T}) where {T}
    @inbounds for i in eachindex(a.batch)
        b = a.batch[i]
        a.total[i] += b
        a.totalsq[i] += b * b
        a.batch[i] = zero(T)
    end
    return a
end

# mean over photons (× scale) and its standard error, from the batch sums.
function batch_mean_se(a::BatchAccum{T,N}, nb::Int, per::Int, scale::Real) where {T,N}
    μ = Array{T,N}(undef, size(a.total))
    σ = Array{T,N}(undef, size(a.total))
    ntot = nb * per
    sc = T(scale)
    @inbounds for i in eachindex(a.total)
        s = a.total[i]
        v = (a.totalsq[i] - s * s / nb) / max(nb - 1, 1)
        μ[i] = s / ntot * sc
        σ[i] = sqrt(max(v, zero(T)) / nb) / per * sc
    end
    return μ, σ
end

# Scalar tally slots.
const S_TOA_LE      = 1   # top escape, local estimate
const S_TOA_ANA     = 2   # top escape, analogue
const S_SFC_DIR     = 3   # surface downward, unscattered, local estimate
const S_SFC_DIF     = 4   # surface downward, scattered, local estimate
const S_SFC_ANA     = 5   # surface downward, analogue
const S_ABS_ATM     = 6   # absorbed in the atmosphere
const S_ABS_SFC     = 7   # absorbed by the ground
const S_ABS_SOLID   = 8   # absorbed by buildings
const S_RR_LOSS     = 9   # weight killed by Russian roulette
const S_RR_GAIN     = 10  # weight created by the roulette boost
const S_LOST        = 11  # weight abandoned by a geometric cutoff
const S_NSCAT       = 12  # scattering events (diagnostic)
const S_SFC_TOT     = 13  # surface downward, direct + diffuse, local estimate
const N_FIXED_SCAL  = 13

struct MCTally{T<:AbstractFloat}
    absorbed::BatchAccum{T,3}
    sfc_dir::BatchAccum{T,2}
    sfc_dif::BatchAccum{T,2}
    sfc_abs::BatchAccum{T,2}
    scal::BatchAccum{T,1}
    solid_inc::Array{T,3}
    solid_abs::Array{T,3}
    fup::Array{T,3}
    fdn::Array{T,3}
    fdir::Array{T,3}
end

function MCTally(::Type{T}, d::MCDomain, nview::Int) where {T}
    return MCTally{T}(BatchAccum(T, (d.nx, d.ny, d.nz)),
                      BatchAccum(T, (d.nx, d.ny)),
                      BatchAccum(T, (d.nx, d.ny)),
                      BatchAccum(T, (d.nx, d.ny)),
                      BatchAccum(T, (N_FIXED_SCAL + nview,)),
                      zeros(T, d.nx, d.ny, d.nz),
                      zeros(T, d.nx, d.ny, d.nz),
                      zeros(T, d.nx, d.ny, d.nz + 1),
                      zeros(T, d.nx, d.ny, d.nz + 1),
                      zeros(T, d.nx, d.ny, d.nz + 1))
end

function flush_tally!(t::MCTally)
    flush_batch!(t.absorbed); flush_batch!(t.sfc_dir); flush_batch!(t.sfc_dif)
    flush_batch!(t.sfc_abs);  flush_batch!(t.scal)
    return t
end

# =========================================================================
# Voxel traversal (3-D DDA, Amanatides & Woo)
# =========================================================================

const EXIT_TOP    = 1
const EXIT_GROUND = 2
const EXIT_SOLID  = 3
const EXIT_CUTOFF = 4   # optical depth exhausted — the ray no longer matters
const EXIT_STEPS  = 5   # step guard tripped: the remaining weight is lost

@inline function _wrap(v::T, lo::T, L::T) where {T}
    t = v - lo
    t -= L * floor(t / L)
    (t < 0 || t >= L) && (t = zero(T))
    return lo + t
end

"""
    _dda!(tal, med, px, py, pz, ux, uy, uz, τ_target, τ_stop, w, scattered,
          score, max_steps)

March a ray through the voxel grid, exactly.

The ray starts at `(px,py,pz)` — wrapped into the domain laterally — and
travels along the unit vector `u`. At each step the parametric distance to
the next `x`, `y` and `z` voxel face is compared and the smallest taken,
so every segment lies inside exactly one voxel and the optical-depth
integral `∫ k_ext ds` is exact.

Two things happen along the way:

  * **Collision.** The first point at which the accumulated optical depth
    reaches `τ_target` is recorded. The march does *not* stop there: it
    continues so that the optical depth all the way to the domain boundary
    is known, which is what the local estimators need.
  * **Level-crossing local estimate.** Whenever the ray crosses a
    horizontal level, `w·e^{-τ}` is scored into the up- or downward flux at
    that level — the energy that reaches the level without colliding. This
    is an expected-value estimator of the analogue "count the photons that
    cross" tally; it has the same mean and far less variance, and it makes
    the unscattered direct beam exact.

Returns a `NamedTuple` describing the collision (if any) and the ray's
terminus. Lateral indices wrap, so a ray can circulate many times before
exiting through the top or the ground.
"""
function _dda!(tal::MCTally{T}, med::MCMedium{T},
               px::T, py::T, pz::T, ux::T, uy::T, uz::T,
               τ_target::T, τ_stop::T, w::T, scattered::Bool,
               score::Bool, max_steps::Int) where {T}
    d = med.domain
    Lx = d.nx * d.dx; Ly = d.ny * d.dy
    x = _wrap(px, d.x0, Lx)
    y = _wrap(py, d.y0, Ly)
    z = pz

    i = min(max(floor(Int, (x - d.x0) / d.dx) + 1, 1), d.nx)
    j = min(max(floor(Int, (y - d.y0) / d.dy) + 1, 1), d.ny)
    k = min(max(floor(Int, (z - d.z0) / d.dz) + 1, 1), d.nz)

    stepi = ux > 0 ? 1 : (ux < 0 ? -1 : 0)
    stepj = uy > 0 ? 1 : (uy < 0 ? -1 : 0)
    stepk = uz > 0 ? 1 : (uz < 0 ? -1 : 0)

    tdx = ux == 0 ? T(Inf) : d.dx / abs(ux)
    tdy = uy == 0 ? T(Inf) : d.dy / abs(uy)
    tdz = uz == 0 ? T(Inf) : d.dz / abs(uz)

    tmx = ux > 0 ? (d.x0 + i * d.dx - x) / ux :
          (ux < 0 ? (d.x0 + (i - 1) * d.dx - x) / ux : T(Inf))
    tmy = uy > 0 ? (d.y0 + j * d.dy - y) / uy :
          (uy < 0 ? (d.y0 + (j - 1) * d.dy - y) / uy : T(Inf))
    tmz = uz > 0 ? (d.z0 + k * d.dz - z) / uz :
          (uz < 0 ? (d.z0 + (k - 1) * d.dz - z) / uz : T(Inf))
    tmx < 0 && (tmx = zero(T)); tmy < 0 && (tmy = zero(T)); tmz < 0 && (tmz = zero(T))

    s = zero(T); τ = zero(T)
    collided = false
    ci = i; cj = j; ck = k
    cx = x; cy = y; cz = z
    exitcode = EXIT_STEPS
    ei = i; ej = j; ek = k
    ex = x; ey = y; ez = z
    axis = 0
    doscore = score

    # A photon should never start inside a building; if it does, treat the
    # step as an immediate surface hit rather than tunnelling through.
    @inbounds if med.solid[i, j, k]
        return (collided = false, ci = i, cj = j, ck = k, cx = x, cy = y, cz = z,
                exitcode = EXIT_SOLID, taub = zero(T),
                ei = i, ej = j, ek = k, ex = x, ey = y, ez = z,
                axis = 3, stepi = stepi, stepj = stepj, stepk = stepk)
    end

    @inbounds for _ in 1:max_steps
        kext = med.kext[i, j, k]

        if tmx <= tmy && tmx <= tmz
            axis = 1; tnext = tmx
        elseif tmy <= tmz
            axis = 2; tnext = tmy
        else
            axis = 3; tnext = tmz
        end
        ds = tnext - s
        dτ = kext * ds

        if !collided && kext > 0 && τ + dτ >= τ_target
            sc = s + (τ_target - τ) / kext
            cx = x + ux * sc; cy = y + uy * sc; cz = z + uz * sc
            ci = i; cj = j; ck = k
            collided = true
        end

        τ += dτ
        s = tnext

        if τ >= τ_stop
            exitcode = EXIT_CUTOFF
            ex = x + ux * s; ey = y + uy * s; ez = z + uz * s
            ei = i; ej = j; ek = k
            break
        end

        if axis == 1
            i += stepi
            i > d.nx && (i = 1)
            i < 1 && (i = d.nx)
            tmx += tdx
        elseif axis == 2
            j += stepj
            j > d.ny && (j = 1)
            j < 1 && (j = d.ny)
            tmy += tdy
        else
            kl = stepk > 0 ? k + 1 : k
            if doscore
                ww = w * exp(-τ)
                if stepk > 0
                    tal.fup[i, j, kl] += ww
                else
                    tal.fdn[i, j, kl] += ww
                    scattered || (tal.fdir[i, j, kl] += ww)
                end
            end
            k += stepk
            tmz += tdz
            if k > d.nz
                exitcode = EXIT_TOP
                ex = x + ux * s; ey = y + uy * s; ez = domain_top(d)
                ei = i; ej = j; ek = d.nz
                break
            elseif k < 1
                exitcode = EXIT_GROUND
                ex = x + ux * s; ey = y + uy * s; ez = d.z0
                ei = i; ej = j; ek = 1
                break
            end
        end

        if med.solid[i, j, k]
            exitcode = EXIT_SOLID
            ex = x + ux * s; ey = y + uy * s; ez = z + uz * s
            ei = i; ej = j; ek = k
            break
        end
    end

    return (collided = collided, ci = ci, cj = cj, ck = ck,
            cx = cx, cy = cy, cz = cz,
            exitcode = exitcode, taub = τ,
            ei = ei, ej = ej, ek = ek, ex = ex, ey = ey, ez = ez,
            axis = axis, stepi = stepi, stepj = stepj, stepk = stepk)
end

# Escape transmittance from a point in a given direction, no scoring.
# Used by the radiance local estimator.
@inline function _escape_transmittance(tal::MCTally{T}, med::MCMedium{T},
                                       x::T, y::T, z::T, vx::T, vy::T, vz::T,
                                       τ_cut::T, max_steps::Int) where {T}
    r = _dda!(tal, med, x, y, z, vx, vy, vz, T(Inf), τ_cut, zero(T), true,
              false, max_steps)
    return r.exitcode == EXIT_TOP ? exp(-r.taub) : zero(T)
end

# =========================================================================
# Photon transport
# =========================================================================

# Radiance local estimate from an emission event of (post-capture) weight
# `w` at `(x,y,z)`. `kind = :scatter` uses the phase function about the
# incident direction `u`; `kind = :lambert` uses the cosine BRDF about the
# surface normal `u`. Both are normalised so that the emitted intensity
# integrates to `w`.
@inline function _score_radiance!(tal::MCTally{T}, med::MCMedium{T},
                                  solver::MonteCarloSolver{T},
                                  x::T, y::T, z::T,
                                  ux::T, uy::T, uz::T, w::T,
                                  gv::T, lambert::Bool) where {T}
    nv = length(solver.view_dirs)
    nv == 0 && return nothing
    @inbounds for v in 1:nv
        vx, vy, vz = solver.view_dirs[v]
        vz <= 0 && continue                      # not an upward-looking ray
        cosΘ = ux * vx + uy * vy + uz * vz
        if lambert
            cosΘ <= 0 && continue                # below the reflecting surface
            # cosine BRDF: intensity = w·cosθ/π, radiance ⇒ divide by μ_v
            # (the two cosines cancel for an upward-facing horizontal face
            # but not for a wall, so both are kept explicitly)
            pf = cosΘ / T(π)
        else
            pf = (med.phase === :rayleigh ? T(rayleigh_phase(cosΘ)) :
                                            henyey_greenstein_phase(gv, cosΘ)) /
                 (4 * T(π))
        end
        Tesc = _escape_transmittance(tal, med, x, y, z, vx, vy, vz,
                                     solver.tau_cutoff, solver.max_steps)
        Tesc <= 0 && continue
        tal.scal.batch[N_FIXED_SCAL + v] += w * pf * Tesc / vz
    end
    return nothing
end

# Trace one photon from the top of the domain to its end. Returns nothing;
# everything is scored into `tal`.
function _trace_photon!(tal::MCTally{T}, med::MCMedium{T},
                        solver::MonteCarloSolver{T},
                        x::T, y::T, z::T, ux::T, uy::T, uz::T,
                        rng::AbstractRNG) where {T}
    d = med.domain
    nudge = T(1e-9) * min(d.dx, min(d.dy, d.dz))
    w = one(T)
    scattered = false
    nsc = 0
    le = solver.local_estimation
    τcut = solver.tau_cutoff

    while true
        τ_target = sample_free_path(rng, T)
        τ_stop = max(τcut, τ_target)
        r = _dda!(tal, med, x, y, z, ux, uy, uz, τ_target, τ_stop, w,
                  scattered, true, solver.max_steps)

        # --- local estimation of the energy that reaches the ray's terminus
        if le
            ec = r.exitcode
            if ec == EXIT_TOP
                tal.scal.batch[S_TOA_LE] += w * exp(-r.taub)
            elseif ec == EXIT_GROUND
                contrib = w * exp(-r.taub)
                tal.scal.batch[S_SFC_TOT] += contrib
                if scattered
                    tal.sfc_dif.batch[r.ei, r.ej] += contrib
                    tal.scal.batch[S_SFC_DIF] += contrib
                else
                    tal.sfc_dir.batch[r.ei, r.ej] += contrib
                    tal.scal.batch[S_SFC_DIR] += contrib
                end
            elseif ec == EXIT_SOLID
                tal.solid_inc[r.ei, r.ej, r.ek] += w * exp(-r.taub)
            end
        end

        if r.collided
            # --- implicit capture: absorb a fraction, never kill the photon
            ω = med.ssa[r.ci, r.cj, r.ck]
            wabs = (one(T) - ω) * w
            tal.absorbed.batch[r.ci, r.cj, r.ck] += wabs
            tal.scal.batch[S_ABS_ATM] += wabs
            w *= ω
            if w <= 0
                return nothing
            end
            x = r.cx; y = r.cy; z = r.cz
            gv = med.g[r.ci, r.cj, r.ck]
            _score_radiance!(tal, med, solver, x, y, z, ux, uy, uz, w, gv, false)
            ux, uy, uz = scatter_direction(rng, ux, uy, uz, gv, med.phase)
            scattered = true
            nsc += 1
            tal.scal.batch[S_NSCAT] += one(T)
            if nsc >= solver.max_scatter
                tal.scal.batch[S_LOST] += w
                return nothing
            end
        else
            ec = r.exitcode
            if ec == EXIT_TOP
                tal.scal.batch[S_TOA_ANA] += w
                return nothing
            elseif ec == EXIT_GROUND
                tal.scal.batch[S_SFC_ANA] += w
                α = med.sfc_albedo[r.ei, r.ej]
                wabs = (one(T) - α) * w
                tal.sfc_abs.batch[r.ei, r.ej] += wabs
                tal.scal.batch[S_ABS_SFC] += wabs
                w *= α
                w <= 0 && return nothing
                x = r.ex; y = r.ey; z = d.z0 + nudge
                # the reflected beam leaves through level 1 immediately
                tal.fup[r.ei, r.ej, 1] += w
                _score_radiance!(tal, med, solver, x, y, z,
                                 zero(T), zero(T), one(T), w, zero(T), true)
                ux, uy, uz = sample_lambertian(rng, zero(T), zero(T), one(T))
                scattered = true
            elseif ec == EXIT_SOLID
                α = med.solid_albedo[r.ei, r.ej, r.ek]
                wabs = (one(T) - α) * w
                tal.solid_abs[r.ei, r.ej, r.ek] += wabs
                tal.scal.batch[S_ABS_SOLID] += wabs
                w *= α
                w <= 0 && return nothing
                # outward normal of the face the ray struck
                nxv = zero(T); nyv = zero(T); nzv = zero(T)
                if r.axis == 1
                    nxv = -T(r.stepi)
                elseif r.axis == 2
                    nyv = -T(r.stepj)
                else
                    nzv = -T(r.stepk)
                    # the ray crossed a level to reach a roof/soffit; the
                    # reflected beam crosses it straight back
                    kl = r.stepk > 0 ? r.ek : r.ek + 1
                    if 1 <= kl <= d.nz + 1
                        if r.stepk > 0
                            tal.fdn[r.ei, r.ej, kl] += w
                        else
                            tal.fup[r.ei, r.ej, kl] += w
                        end
                    end
                end
                x = r.ex + nxv * nudge
                y = r.ey + nyv * nudge
                z = r.ez + nzv * nudge
                _score_radiance!(tal, med, solver, x, y, z, nxv, nyv, nzv,
                                 w, zero(T), true)
                ux, uy, uz = sample_lambertian(rng, nxv, nyv, nzv)
                scattered = true
            else
                # EXIT_CUTOFF without a collision, or the step guard: the
                # ray has nowhere left to go. Book the weight as lost so the
                # energy budget still closes exactly.
                tal.scal.batch[S_LOST] += w
                return nothing
            end
        end

        # --- Russian roulette
        if w < solver.weight_min
            if rand(rng, T) < solver.rr_survival
                tal.scal.batch[S_RR_GAIN] += w * (one(T) / solver.rr_survival - one(T))
                w /= solver.rr_survival
            else
                tal.scal.batch[S_RR_LOSS] += w
                return nothing
            end
        end
    end
end

# =========================================================================
# Results
# =========================================================================

"""
    MCResult{T}

Everything a Monte-Carlo solve produces, with a standard error on every
statistical quantity.

Normalisation: all fluxes are absolute (W m⁻²), and `incident` is the
horizontal irradiance at the top of the domain, `F₀μ₀`. Dimensionless
fractions (`albedo`, `transmittance`, `absorptance_*`) are relative to
`incident`, so `albedo + absorptance_atm + absorptance_sfc +
absorptance_solid` equals one up to the Russian-roulette imbalance.

Fields:

  * `absorbed`, `absorbed_err` — volumetric absorption rate, W m⁻³, per
    voxel (zero inside buildings).
  * `sfc_direct`, `sfc_diffuse` (+ `_err`) — downward surface irradiance
    maps, W m⁻², split into the unscattered beam and everything else.
  * `sfc_absorbed` — what the ground keeps, W m⁻².
  * `solid_incident`, `solid_absorbed` — power on and into each building
    voxel, W.
  * `flux_up`, `flux_dn`, `flux_dn_dir` — level fluxes, `(nx, ny, nz+1)`,
    W m⁻², bottom-up, matching the `RadiativeFluxes` convention.
  * `albedo`, `transmittance`, `absorptance_atm`, `absorptance_sfc`,
    `absorptance_solid` (+ `_err`) — the broadband budget.
  * `albedo_analogue`, `transmittance_analogue` (+ `_err`) — the same two
    quantities from the pure analogue estimator. Comparing them against
    the local estimates is a direct measurement of the variance reduction
    and a cross-check that the local estimator is unbiased.
  * `radiance_up`, `radiance_up_err` — top-of-domain diffuse radiance
    (W m⁻² sr⁻¹) in each of `view_dirs`. The unscattered solar beam is
    *not* included; it is a delta function in angle.
  * `rr_loss`, `rr_gain`, `lost` — weight bookkeeping, as fractions of the
    incident flux.
"""
struct MCResult{T<:AbstractFloat}
    domain::MCDomain{T}
    nphoton::Int
    nbatch::Int
    incident::T
    solar_flux::T
    μ₀::T
    sun_dir::NTuple{3,T}

    absorbed::Array{T,3}
    absorbed_err::Array{T,3}
    sfc_direct::Array{T,2}
    sfc_direct_err::Array{T,2}
    sfc_diffuse::Array{T,2}
    sfc_diffuse_err::Array{T,2}
    sfc_absorbed::Array{T,2}
    sfc_absorbed_err::Array{T,2}
    solid_incident::Array{T,3}
    solid_absorbed::Array{T,3}
    flux_up::Array{T,3}
    flux_dn::Array{T,3}
    flux_dn_dir::Array{T,3}

    albedo::T;               albedo_err::T
    transmittance::T;        transmittance_err::T
    direct_transmittance::T; direct_transmittance_err::T
    absorptance_atm::T;      absorptance_atm_err::T
    absorptance_sfc::T;      absorptance_sfc_err::T
    absorptance_solid::T;    absorptance_solid_err::T
    albedo_analogue::T;      albedo_analogue_err::T
    transmittance_analogue::T; transmittance_analogue_err::T

    rr_loss::T
    rr_gain::T
    lost::T
    mean_scatter::T
    radiance_up::Vector{T}
    radiance_up_err::Vector{T}
    view_dirs::Vector{NTuple{3,T}}
end

function Base.show(io::IO, r::MCResult{T}) where {T}
    print(io, "MCResult{$T}(", r.nphoton, " photons, albedo ",
          round(r.albedo; digits = 4), " ± ", round(r.albedo_err; digits = 4),
          ", transmittance ", round(r.transmittance; digits = 4),
          " ± ", round(r.transmittance_err; digits = 4), ")")
end

"""
    mc_energy_budget(res) -> NamedTuple

The complete weight ledger as fractions of the incident flux:
`(albedo, atmosphere, surface, buildings, rr_loss, rr_gain, lost, total)`.
`total` is the sum of every sink minus the roulette gain and must equal
one to machine precision — see [`mc_closure_residual`](@ref).
"""
function mc_energy_budget(r::MCResult{T}) where {T}
    total = r.albedo_analogue + r.absorptance_atm + r.absorptance_sfc +
            r.absorptance_solid + r.rr_loss - r.rr_gain + r.lost
    return (albedo = r.albedo_analogue, atmosphere = r.absorptance_atm,
            surface = r.absorptance_sfc, buildings = r.absorptance_solid,
            rr_loss = r.rr_loss, rr_gain = r.rr_gain, lost = r.lost,
            total = total)
end

"""
    mc_closure_residual(res) -> T

`1 - total` from [`mc_energy_budget`](@ref). Every unit of launched weight
is deposited into exactly one accumulator, so this is zero to floating-point
round-off *for every realisation* — not merely in expectation. A non-zero
value means weight is being created or destroyed somewhere in the transport
loop, which is the first thing to check when a Monte-Carlo solver
misbehaves.
"""
mc_closure_residual(r::MCResult) = one(eltype(r.albedo)) - mc_energy_budget(r).total

"""
    mc_surface_irradiance(res) -> Array{T,2}

Total downward shortwave irradiance at the ground, `direct + diffuse`
(W m⁻²).
"""
mc_surface_irradiance(r::MCResult) = r.sfc_direct .+ r.sfc_diffuse

"""
    mc_heating_rate(res, ρ; cp = 1004.6) -> Array{T,3}

Radiative heating rate in K s⁻¹ from the per-voxel absorption,
`∂θ/∂t = Q_abs / (ρ c_p)`. `ρ` may be a scalar or a length-`nz` profile.
"""
function mc_heating_rate(r::MCResult{T}, ρ; cp::Real = CP_DRY_AIR) where {T}
    d = r.domain
    out = similar(r.absorbed)
    @inbounds for k in 1:d.nz, j in 1:d.ny, i in 1:d.nx
        ρk = ρ isa Real ? T(ρ) : T(ρ[k])
        out[i, j, k] = r.absorbed[i, j, k] / (ρk * T(cp))
    end
    return out
end

# =========================================================================
# Driver
# =========================================================================

"""
    mc_solve(medium, solver; rng, sun_dir, solar_flux = 1.0) -> MCResult

Run the Monte-Carlo solve. `sun_dir` points **toward** the sun in the
east-north-up frame (anything [`mc_sun_direction`](@ref) accepts, including
a `SolarPosition` or a bare `μ₀`); `solar_flux` is the beam-normal
irradiance `F₀`, so the horizontal irradiance at the top of the domain is
`F₀μ₀`.

`rng` is required to be explicit so that a seeded `MersenneTwister` makes
the whole solve bit-for-bit reproducible.

Photons are launched from the top face, one per stratum of a jittered grid
when `solver.stratify` is set. Each carries unit weight; the physical
energy a unit of weight represents is `F₀μ₀·A/N`, which is how the tallies
become absolute fluxes at the end.
"""
function mc_solve(med::MCMedium{T}, solver::MonteCarloSolver{T};
                  rng::AbstractRNG,
                  sun_dir,
                  solar_flux::Real = one(T)) where {T}
    d = med.domain
    sd = mc_sun_direction(sun_dir)
    sx, sy, sz = T(sd[1]), T(sd[2]), T(sd[3])
    sz > 0 || error("mc_solve: the sun must be above the horizon (up-component > 0)")
    μ₀ = sz
    # photons travel *away* from the sun
    ux0 = -sx; uy0 = -sy; uz0 = -sz

    nb = solver.nbatch
    per = solver.nphoton ÷ nb
    nview = length(solver.view_dirs)
    tal = MCTally(T, d, nview)

    Lx = d.nx * d.dx; Ly = d.ny * d.dy
    ztop = domain_top(d)
    ncell = d.nx * d.ny
    slant = (ztop - d.z0) / μ₀      # horizontal run of the beam over the depth
    pidx = 0

    for _ in 1:nb
        for _ in 1:per
            if solver.stratify
                # Stratify where the unattenuated beam would *land*, not
                # where it enters. The map from the entry point to the
                # ground intercept is a rigid translation modulo the
                # periodic box, hence measure preserving: entry points stay
                # uniform over the top face, but the direct-beam surface
                # irradiance map gets exactly one sample per cell per pass
                # instead of a Poisson-distributed number of them.
                q = pidx % ncell
                i0 = q % d.nx
                j0 = q ÷ d.nx
                xg = d.x0 + (i0 + rand(rng, T)) * d.dx
                yg = d.y0 + (j0 + rand(rng, T)) * d.dy
                x = _wrap(xg + sx * slant, d.x0, Lx)
                y = _wrap(yg + sy * slant, d.y0, Ly)
            else
                x = d.x0 + rand(rng, T) * Lx
                y = d.y0 + rand(rng, T) * Ly
            end
            pidx += 1
            # incoming beam crosses the model top
            ii = min(max(floor(Int, (x - d.x0) / d.dx) + 1, 1), d.nx)
            jj = min(max(floor(Int, (y - d.y0) / d.dy) + 1, 1), d.ny)
            tal.fdn[ii, jj, d.nz + 1] += one(T)
            tal.fdir[ii, jj, d.nz + 1] += one(T)
            _trace_photon!(tal, med, solver, x, y, ztop, ux0, uy0, uz0, rng)
        end
        flush_tally!(tal)
    end

    return _finalise(tal, med, solver, T(solar_flux), μ₀, (sx, sy, sz), nb, per)
end

function _finalise(tal::MCTally{T}, med::MCMedium{T}, solver::MonteCarloSolver{T},
                   F0::T, μ₀::T, sd::NTuple{3,T}, nb::Int, per::Int) where {T}
    d = med.domain
    N = nb * per
    incident = F0 * μ₀                     # W m⁻², horizontal, domain mean
    A = domain_area(d)
    ncell = d.nx * d.ny
    E_ph = incident * A / N                # joules-per-second per unit weight
    cell_scale = incident * ncell          # weight-mean → per-cell irradiance
    vol_scale = incident * ncell / d.dz    # weight-mean → W m⁻³

    absorbed, absorbed_err = batch_mean_se(tal.absorbed, nb, per, vol_scale)
    @inbounds for idx in eachindex(absorbed)
        if med.solid[idx]
            absorbed[idx] = zero(T); absorbed_err[idx] = zero(T)
        end
    end
    sfc_dir, sfc_dir_err = batch_mean_se(tal.sfc_dir, nb, per, cell_scale)
    sfc_dif, sfc_dif_err = batch_mean_se(tal.sfc_dif, nb, per, cell_scale)
    sfc_abs, sfc_abs_err = batch_mean_se(tal.sfc_abs, nb, per, cell_scale)

    sm, ss = batch_mean_se(tal.scal, nb, per, one(T))

    solid_inc = tal.solid_inc .* (E_ph)
    solid_abs = tal.solid_abs .* (E_ph)
    fscale = incident * ncell
    fup = tal.fup .* (fscale / N)
    fdn = tal.fdn .* (fscale / N)
    fdir = tal.fdir .* (fscale / N)

    alb   = sm[S_TOA_LE];  alb_e  = ss[S_TOA_LE]
    tdir  = sm[S_SFC_DIR]; tdir_e = ss[S_SFC_DIR]
    # The direct and diffuse arrivals are strongly anti-correlated, so the
    # total is tallied in its own slot rather than reconstructed by adding
    # two variances — that would overstate the error by a large factor.
    tr    = sm[S_SFC_TOT]; tr_e = ss[S_SFC_TOT]

    nview = length(solver.view_dirs)
    rad = T[sm[N_FIXED_SCAL + v] * incident for v in 1:nview]
    rad_e = T[ss[N_FIXED_SCAL + v] * incident for v in 1:nview]

    return MCResult{T}(d, N, nb, incident, F0, μ₀, sd,
                       absorbed, absorbed_err,
                       sfc_dir, sfc_dir_err, sfc_dif, sfc_dif_err,
                       sfc_abs, sfc_abs_err,
                       solid_inc, solid_abs, fup, fdn, fdir,
                       alb, alb_e, tr, tr_e, tdir, tdir_e,
                       sm[S_ABS_ATM], ss[S_ABS_ATM],
                       sm[S_ABS_SFC], ss[S_ABS_SFC],
                       sm[S_ABS_SOLID], ss[S_ABS_SOLID],
                       sm[S_TOA_ANA], ss[S_TOA_ANA],
                       sm[S_SFC_ANA], ss[S_SFC_ANA],
                       sm[S_RR_LOSS], sm[S_RR_GAIN], sm[S_LOST],
                       sm[S_NSCAT],
                       rad, rad_e, copy(solver.view_dirs))
end

# =========================================================================
# `solve_rt!` contract
# =========================================================================

"""
    solve_rt!(fluxes, solver::MonteCarloSolver, optics, col::AtmosphericColumn;
              rng, μ₀ = 1.0, sun_dir = nothing, sfc_albedo = 0.0, kwargs...)

`AbstractRTSolver` interface, column form. The column's layer thicknesses
`col.dz` must be uniform — the photon grid is a Cartesian voxel mesh — and
the resulting domain is a single column with periodic lateral boundaries,
which is exactly the plane-parallel problem the two-stream solver treats.

The solve is run once per g-point with a non-zero solar source and the
broadband level fluxes are accumulated into `fluxes`, so the output is
directly comparable with `TwoStream.solve_sw!` on the same optics.

    solve_rt!(fluxes, solver, optics, domain::MCDomain; rng, kwargs...)

3-D form: `optics` is `(nx, ny, nz, ngpt)` (or a column profile replicated
across the domain) and `fluxes` has shape `(nx, ny, nz+1)`.
"""
function solve_rt!(fluxes::RadiativeFluxes{T}, solver::MonteCarloSolver{T},
                   optics::OpticalProperties{T}, col::AtmosphericColumn{T};
                   kwargs...) where {T}
    nz = length(col.dz)
    dz = col.dz[1]
    all(x -> isapprox(x, dz; rtol = 1e-9), col.dz) ||
        error("solve_rt!(MonteCarloSolver): the Monte-Carlo voxel grid needs a ",
              "uniform layer thickness; got dz ∈ [$(minimum(col.dz)), ",
              "$(maximum(col.dz))]")
    dom = mc_domain(T; size = (1, 1, nz), spacing = (dz, dz, dz))
    return solve_rt!(fluxes, solver, optics, dom; kwargs...)
end

function solve_rt!(fluxes::RadiativeFluxes{T}, solver::MonteCarloSolver{T},
                   optics::OpticalProperties{T}, dom::MCDomain{T};
                   rng::AbstractRNG = Random.default_rng(),
                   μ₀::Real = 1.0,
                   sun_dir = nothing,
                   sfc_albedo = zero(T),
                   solid = nothing,
                   solid_albedo = T(0.3),
                   phase::Symbol = :henyey_greenstein) where {T}
    spec = solver.spec
    spec === nothing &&
        error("solve_rt!(MonteCarloSolver): the solver was built without a ",
              "SpectralGrid; use `monte_carlo_solver(spec)` or call `mc_solve` ",
              "directly")
    spec.kind === :shortwave ||
        error("solve_rt!(MonteCarloSolver): only shortwave grids are supported; ",
              "the longwave problem needs thermal emission sources, which is ",
              "the two-stream solver's job")

    sd = sun_dir === nothing ? mc_sun_direction(μ₀) : mc_sun_direction(sun_dir)

    fill!(fluxes.flux_up, 0); fill!(fluxes.flux_dn, 0)
    fill!(fluxes.flux_dn_dir, 0)
    is3d = ndims(fluxes.flux_up) == 3

    for b in 1:nbands(spec)
        α = sfc_albedo isa Real ? T(sfc_albedo) :
            (sfc_albedo isa AbstractVector ? T(sfc_albedo[b]) : sfc_albedo)
        for g in gpoint_range(spec, b)
            F0 = T(spec.solar_src[g])
            F0 <= 0 && continue
            med = medium_from_optics(dom, optics, g; sfc_albedo = α,
                                     solid = solid, solid_albedo = solid_albedo,
                                     phase = phase)
            res = mc_solve(med, solver; rng = rng, sun_dir = sd, solar_flux = F0)
            if is3d
                @inbounds for kl in 1:dom.nz+1, jj in 1:dom.ny, ii in 1:dom.nx
                    fluxes.flux_up[ii, jj, kl]     += res.flux_up[ii, jj, kl]
                    fluxes.flux_dn[ii, jj, kl]     += res.flux_dn[ii, jj, kl]
                    fluxes.flux_dn_dir[ii, jj, kl] += res.flux_dn_dir[ii, jj, kl]
                end
            else
                @inbounds for kl in 1:dom.nz+1
                    s_up = zero(T); s_dn = zero(T); s_dd = zero(T)
                    for jj in 1:dom.ny, ii in 1:dom.nx
                        s_up += res.flux_up[ii, jj, kl]
                        s_dn += res.flux_dn[ii, jj, kl]
                        s_dd += res.flux_dn_dir[ii, jj, kl]
                    end
                    n = dom.nx * dom.ny
                    fluxes.flux_up[kl]     += s_up / n
                    fluxes.flux_dn[kl]     += s_dn / n
                    fluxes.flux_dn_dir[kl] += s_dd / n
                end
            end
        end
    end
    net_flux!(fluxes)
    return fluxes
end

end # module
