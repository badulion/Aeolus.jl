"""
    SurfaceTiles

Sub-grid surface heterogeneity by *tiling* (the "mosaic" approach of
Avissar & Pielke 1989): a grid cell holds ``N`` surfaces — any
[`AbstractSurface`](@ref Aeolus.SurfaceExchange.AbstractSurface)
implementers — with area fractions ``f_i ≥ 0``, ``\\sum f_i = 1``. Each
tile keeps its own prognostic state and solves its own energy balance
against the grid-cell forcing through the `SurfaceExchange` contract;
the cell sees the **area-weighted average of the tile fluxes**

```math
\\overline{F} = \\sum_i f_i\\, F_i(\\text{tile } i),
```

not the flux of one surface with averaged parameters,
``F(\\bar p)``. The two differ whenever the flux is a nonlinear function
of the heterogeneous parameter (stability, roughness, saturation
humidity), which is the whole reason to tile.

A [`SurfaceMosaic`](@ref) is itself an `AbstractSurface`, so it can be
stepped, nested, or handed to anything that speaks the contract. Its
radiative properties are the exact aggregates: albedo and emissivity are
area-weighted, and [`effective_radiative_temperature`](@ref) is the
temperature that reproduces the tile-summed emitted longwave, so that the
one-surface formula ``R_n = (1-\\bar α) SW↓ + \\bar ε LW↓ - \\bar ε σ T_{rad}^4``
equals ``\\sum f_i R_{n,i}`` identically.

This is the *energy-balance* mosaic. Its optical sibling is
`SurfaceOptics.TiledSurface`, which area-weights spectral albedos of
`AbstractSurfaceOptics` components; a [`SurfaceMosaic`](@ref) instead
asks each of its tiles for its own albedo (which a tile may well obtain
from `SurfaceOptics`) and weights the result the same way.

The five-class urban surface map of `examples/urban` (one class per LES
cell) is the special case of one tile per cell; [`class_fractions`](@ref)
and [`tiles_from_class_map`](@ref) coarse-grain such a map into tiled
cells.

A plain `SurfaceMosaic` forces every tile with the grid-cell forcing
(or a forcing per tile passed to `step_surface!`). The **blending-height**
formulation (Wieringa 1986; Mason 1988; Claussen 1991) is the second half
of this module: [`effective_roughness`](@ref) at a blending height ``l_b``,
closures for ``l_b`` ([`MasonBlendingHeight`](@ref),
[`ClaussenBlendingHeight`](@ref), [`BouZeidBlendingHeight`](@ref),
[`IBLBlendingHeight`](@ref)), internal-boundary-layer growth laws
([`ibl_height`](@ref)), and [`BlendedMosaic`](@ref), whose tiles are forced
with the wind, temperature and humidity *at the blending height* obtained
by matching the aggregate flux to the profile above it.

# References

  * Avissar, R. & Pielke, R. A. (1989). A parameterization of
    heterogeneous land surfaces for atmospheric numerical models and its
    impact on regional meteorology. *Mon. Wea. Rev.*, 117, 2113-2136.
  * Koster, R. D. & Suarez, M. J. (1992). Modeling the land surface
    boundary in climate models as a composite of independent vegetation
    stands. *J. Geophys. Res.*, 97(D3), 2697-2715.
  * Mahrt, L. (1987). Grid-averaged surface fluxes. *Mon. Wea. Rev.*,
    115, 1550-1560.
  * Mason, P. J. (1988). The formation of areally-averaged roughness
    lengths. *Q. J. R. Meteorol. Soc.*, 114, 399-420.
  * Giorgi, F. & Avissar, R. (1997). Representation of heterogeneity
    effects in Earth system modeling: experience from land surface
    modeling. *Rev. Geophys.*, 35, 413-437.
  * Wieringa, J. (1986). Roughness-dependent geographical interpolation
    of surface wind speed averages. *Q. J. R. Meteorol. Soc.*, 112,
    867-889.
  * Claussen, M. (1991). Estimation of areally-averaged surface fluxes.
    *Boundary-Layer Meteorol.*, 54, 387-410.
  * Bou-Zeid, E., Meneveau, C. & Parlange, M. B. (2004). Large-eddy
    simulation of neutral atmospheric boundary layer flow over
    heterogeneous surfaces: blending height and effective surface
    roughness. *Water Resour. Res.*, 40, W02505.
  * Bou-Zeid, E., Parlange, M. B. & Meneveau, C. (2007). On the
    parameterization of surface roughness at regional scales.
    *J. Atmos. Sci.*, 64, 216-227.
  * Savelyev, S. A. & Taylor, P. A. (2005). Internal boundary layers:
    I. Height formulae for neutral and diabatic flows.
    *Boundary-Layer Meteorol.*, 115, 1-25.
  * Taylor, P. A. (1987). Comments and further analysis on effective
    roughness lengths for use in numerical 3-dimensional models.
    *Boundary-Layer Meteorol.*, 39, 403-418.
  * Kramm, G. & Dlugi, R. (2009). On the accuracy with which the lower
    boundary conditions can be determined in numerical models of the
    atmosphere. arXiv:0911.2286 (Section 5.1 quotes the Mason and
    Claussen blending-height formulae).
"""
module SurfaceTiles

using ..SurfaceExchange: AbstractSurface, SurfaceForcing, SurfaceFluxes,
                         SurfaceConstants, DEFAULT_SURFACE_CONSTANTS,
                         AbstractStabilityFunctions, BusingerDyerBeljaarsHoltslag,
                         psi_m, psi_h, obukhov_length, _virtual_coefficient
using ..MoistThermodynamics: PhasePartition, air_density, heat_capacity_moist,
                             dry_adiabatic_lapse_rate
import ..SurfaceExchange: surface_temperature, roughness_lengths,
                          surface_albedo, surface_emissivity,
                          moisture_availability, net_radiation,
                          turbulent_fluxes, step_surface!

export SurfaceMosaic, ntiles, mosaic_fractions, set_mosaic_fractions!,
       aggregate_fluxes, tile_fluxes, last_tile_fluxes,
       effective_radiative_temperature, tile_mean_temperature,
       ParameterAggregatedSurface, parameter_aggregate,
       class_fractions, tiles_from_class_map,
       # blending height (#42)
       effective_roughness, effective_scalar_roughness,
       AbstractBlendingHeight, FixedBlendingHeight, MasonBlendingHeight,
       ClaussenBlendingHeight, BouZeidBlendingHeight, IBLBlendingHeight,
       blending_height,
       AbstractIBLModel, ElliottIBL, PowerLawIBL, DiffusionIBL, ibl_height

# =========================================================================
# Fractions
# =========================================================================

# Validate and normalise area fractions. Normalising (rather than only
# checking) removes the user's O(atol) error, so Σ f = 1 holds to
# rounding (a few ulp; dividing by the sum does not make the
# floating-point sum exactly 1 in general).
function _checked_fractions(f::AbstractVector{<:Real}, n::Integer; atol::Real = 1e-6)
    length(f) == n ||
        throw(DimensionMismatch("SurfaceTiles: $(length(f)) fractions for $n tiles"))
    n > 0 || throw(ArgumentError("SurfaceTiles: need at least one tile"))
    all(x -> isfinite(x) && x >= 0, f) ||
        throw(ArgumentError("SurfaceTiles: fractions must be finite and non-negative, got $f"))
    s = sum(f)
    abs(s - 1) <= atol ||
        throw(ArgumentError("SurfaceTiles: fractions must sum to 1 (got $s)"))
    return Float64[x / s for x in f]
end

# =========================================================================
# SurfaceMosaic
# =========================================================================

"""
    SurfaceMosaic(tiles, fractions; atol = 1e-6)

A grid cell made of the surfaces `tiles` (a `Tuple` or `AbstractVector`
of `AbstractSurface`s, which may be of different types — including
other `SurfaceMosaic`s) covering area fractions `fractions`.

Fractions must be non-negative and sum to one within `atol`; they are
then renormalised by their sum, so they sum to one to rounding (a few
ulp). A zero-fraction tile is allowed
(it is still stepped, but carries no weight), which is how a tile that
appears seasonally — snow, a flooded patch — is kept warm.

The tiles are the *state*: `SurfaceMosaic` stores no temperature of its
own. After each [`step_surface!`](@ref) the per-tile fluxes that were
applied are available from [`last_tile_fluxes`](@ref).

Interface methods (all `SurfaceExchange` generics):

| method                         | aggregate                                         |
|:------------------------------ |:------------------------------------------------- |
| `step_surface!(ts, f, dt)`     | steps every tile, returns ``\\sum f_i F_i``        |
| `turbulent_fluxes(ts, f)`      | ``\\sum f_i F_i`` without stepping (flux aggregation) |
| `net_radiation(ts, f)`         | ``\\sum f_i R_{n,i}``                               |
| `surface_albedo(ts)`           | ``\\sum f_i α_i`` (exact for reflected SW)          |
| `surface_emissivity(ts)`       | ``\\sum f_i ε_i`` (exact for absorbed LW)           |
| `surface_temperature(ts)`      | [`effective_radiative_temperature`](@ref)          |
| `roughness_lengths(ts)`        | logarithmic average — a *parameter* aggregate      |
| `moisture_availability(ts)`    | ``\\sum f_i β_i`` — a *parameter* aggregate         |

The last two are there so a `SurfaceMosaic` satisfies the whole contract
(and so [`parameter_aggregate`](@ref) can be built); the fluxes of a
tiled cell never use them.
"""
mutable struct SurfaceMosaic{TS} <: AbstractSurface
    tiles::TS
    fractions::Vector{Float64}
    fluxes::Vector{SurfaceFluxes{Float64}}
end

function SurfaceMosaic(tiles::Union{Tuple,AbstractVector}, fractions::AbstractVector{<:Real};
                      atol::Real = 1e-6)
    all(t -> t isa AbstractSurface, tiles) ||
        throw(ArgumentError("SurfaceMosaic: every tile must be an AbstractSurface"))
    f = _checked_fractions(fractions, length(tiles); atol = atol)
    return SurfaceMosaic{typeof(tiles)}(tiles, f, SurfaceFluxes{Float64}[])
end
SurfaceMosaic(tiles::Union{Tuple,AbstractVector}, fractions::Tuple; kw...) =
    SurfaceMosaic(tiles, collect(Float64, fractions); kw...)

"Number of tiles in a [`SurfaceMosaic`](@ref)."
ntiles(ts::SurfaceMosaic) = length(ts.tiles)

"Area fractions of the tiles of a [`SurfaceMosaic`](@ref) (renormalised; sum 1 to rounding)."
mosaic_fractions(ts::SurfaceMosaic) = ts.fractions

"""
    set_mosaic_fractions!(ts, fractions; atol = 1e-6) -> ts

Replace the tile fractions (e.g. a snow-cover or flooding fraction that
evolves). The tile states are left untouched: redistributing the heat
and water of a shrinking tile into its neighbours is a physical choice
the caller makes, not something the framework does behind its back.
"""
function set_mosaic_fractions!(ts::SurfaceMosaic, f::AbstractVector{<:Real}; atol::Real = 1e-6)
    ts.fractions = _checked_fractions(f, ntiles(ts); atol = atol)
    return ts
end

"""
    last_tile_fluxes(ts) -> Vector{SurfaceFluxes}

The per-tile fluxes applied by the most recent [`step_surface!`](@ref)
of `ts` (empty before the first step). Their fraction-weighted sum is
what that call returned.
"""
last_tile_fluxes(ts::SurfaceMosaic) = ts.fluxes

# =========================================================================
# Flux aggregation
# =========================================================================

_f64(fl::SurfaceFluxes) = SurfaceFluxes{Float64}(fl.Rn, fl.H, fl.LE, fl.G, fl.τ, fl.E)

"""
    aggregate_fluxes(fluxes, fractions) -> SurfaceFluxes

Area-weighted average ``\\sum_i f_i F_i`` of each component of the
per-tile `fluxes` (`Rn`, `H`, `LE`, `G`, `τ`, `E`). Because every
component is averaged with the same weights, a per-tile energy balance
``R_{n,i} = H_i + LE_i + G_i`` carries over to the aggregate exactly.

The momentum flux is averaged as a magnitude: all tiles of one cell see
the same wind direction, so the stress vectors are parallel.
"""
function aggregate_fluxes(fluxes, fractions::AbstractVector{<:Real})
    length(fluxes) == length(fractions) ||
        throw(DimensionMismatch("aggregate_fluxes: $(length(fluxes)) fluxes, $(length(fractions)) fractions"))
    isempty(fluxes) && throw(ArgumentError("aggregate_fluxes: no fluxes"))
    # Start from the first tile rather than from 0.0 so that a single
    # tile of fraction 1 is reproduced bit for bit (signed zeros included).
    w, fl = float(fractions[1]), first(fluxes)
    Rn, H, LE, G, τ, E = w * fl.Rn, w * fl.H, w * fl.LE, w * fl.G, w * fl.τ, w * fl.E
    for (fl, w) in Iterators.drop(zip(fluxes, fractions), 1)
        Rn += w * fl.Rn; H += w * fl.H; LE += w * fl.LE
        G  += w * fl.G;  τ += w * fl.τ; E  += w * fl.E
    end
    return SurfaceFluxes{Float64}(Rn, H, LE, G, τ, E)
end

# Call a generic with keyword arguments only when there are some, so
# tiles whose methods take no keywords still work.
_step(t, f, dt, kw) = isempty(kw) ? step_surface!(t, f, dt) : step_surface!(t, f, dt; kw...)
_turb(t, f, kw) = isempty(kw) ? turbulent_fluxes(t, f) : turbulent_fluxes(t, f; kw...)

"""
    tile_fluxes(ts, forcing; kwargs...) -> Vector{SurfaceFluxes}

The fluxes each tile of `ts` would exchange with `forcing` in its
current state (`turbulent_fluxes` per tile; nothing is stepped).
`forcing` is one `SurfaceForcing` for all tiles or a vector with one per
tile. `kwargs` go to each tile's `turbulent_fluxes`.
"""
tile_fluxes(ts::SurfaceMosaic, f::SurfaceForcing; kwargs...) =
    [_f64(_turb(t, f, kwargs)) for t in ts.tiles]
function tile_fluxes(ts::SurfaceMosaic, fs::AbstractVector{<:SurfaceForcing}; kwargs...)
    _check_forcings(ts, fs)
    return [_f64(_turb(t, f, kwargs)) for (t, f) in zip(ts.tiles, fs)]
end

_check_forcings(ts, fs) = length(fs) == ntiles(ts) ||
    throw(DimensionMismatch("SurfaceTiles: $(length(fs)) forcings for $(ntiles(ts)) tiles"))

"""
    turbulent_fluxes(ts::SurfaceMosaic, forcing; kwargs...) -> SurfaceFluxes

**Flux aggregation**: the fraction-weighted mean of the tiles' own
`turbulent_fluxes`, ``\\sum f_i F_i``. Each tile uses its own skin
temperature, humidity, roughness and stability; nothing is averaged
before the flux is computed. Compare
`turbulent_fluxes(parameter_aggregate(ts), forcing)`.
"""
turbulent_fluxes(ts::SurfaceMosaic, f::SurfaceForcing; kwargs...) =
    aggregate_fluxes(tile_fluxes(ts, f; kwargs...), ts.fractions)
turbulent_fluxes(ts::SurfaceMosaic, fs::AbstractVector{<:SurfaceForcing}; kwargs...) =
    aggregate_fluxes(tile_fluxes(ts, fs; kwargs...), ts.fractions)

"""
    step_surface!(ts::SurfaceMosaic, forcing, dt; kwargs...) -> SurfaceFluxes

Advance every tile by `dt` with its own `step_surface!` — each solves its
own energy balance — and return the fraction-weighted mean of the fluxes
the tiles actually applied. `forcing` is one `SurfaceForcing` for the
whole cell or a vector with one per tile (e.g. tile-specific wind at a
blending height). `kwargs` are passed on to every tile's
`step_surface!`. The per-tile fluxes are kept in
[`last_tile_fluxes`](@ref).

Conservation: if each tile stores ``G_i\\,Δt`` per unit area, the cell
stores ``\\sum f_i G_i Δt = \\bar G Δt`` — exactly the aggregate `G`
returned here.
"""
function step_surface!(ts::SurfaceMosaic, f::SurfaceForcing, dt::Real; kwargs...)
    ts.fluxes = [_f64(_step(t, f, dt, kwargs)) for t in ts.tiles]
    return aggregate_fluxes(ts.fluxes, ts.fractions)
end
function step_surface!(ts::SurfaceMosaic, fs::AbstractVector{<:SurfaceForcing}, dt::Real; kwargs...)
    _check_forcings(ts, fs)
    ts.fluxes = [_f64(_step(t, f, dt, kwargs)) for (t, f) in zip(ts.tiles, fs)]
    return aggregate_fluxes(ts.fluxes, ts.fractions)
end

"""
    net_radiation(ts::SurfaceMosaic, forcing; c = DEFAULT_SURFACE_CONSTANTS)

``\\sum f_i R_{n,i}``, each tile with its own albedo, emissivity and skin
temperature. Identical (to rounding) to the one-surface formula with
`surface_albedo(ts)`, `surface_emissivity(ts)` and
`surface_temperature(ts)`.
"""
net_radiation(ts::SurfaceMosaic, f::SurfaceForcing; c::SurfaceConstants = DEFAULT_SURFACE_CONSTANTS) =
    sum(w * net_radiation(t, f; c = c) for (t, w) in zip(ts.tiles, ts.fractions))

# =========================================================================
# Radiative aggregates
# =========================================================================

_wsum(g, ts::SurfaceMosaic) = sum(w * g(t) for (t, w) in zip(ts.tiles, ts.fractions))

# Broadband: area-weighted, which is exact for the reflected shortwave
# Σ fᵢ αᵢ SW↓ and the absorbed longwave Σ fᵢ εᵢ LW↓ under a uniform
# downwelling field.
surface_albedo(ts::SurfaceMosaic) = _wsum(surface_albedo, ts)
surface_emissivity(ts::SurfaceMosaic) = _wsum(surface_emissivity, ts)
# Spectral / angular RadiationCore forms, when every tile implements them.
surface_albedo(ts::SurfaceMosaic, band, args...) =
    _wsum(t -> surface_albedo(t, band, args...), ts)
surface_emissivity(ts::SurfaceMosaic, band, args...) =
    _wsum(t -> surface_emissivity(t, band, args...), ts)

"""
    effective_radiative_temperature(ts) -> K

The temperature ``T_{rad}`` that makes a single surface of emissivity
``\\bar ε = \\sum f_i ε_i`` emit what the tiles emit together:

```math
\\bar ε\\, σ T_{rad}^4 = \\sum_i f_i ε_i σ T_i^4 .
```

Because ``T^4`` is convex, ``T_{rad} ≥ \\sum f_i T_i`` (equality only
for uniform temperatures). If every tile has zero emissivity the
fraction-mean temperature is returned.
"""
function effective_radiative_temperature(ts::SurfaceMosaic)
    εbar = surface_emissivity(ts)
    εbar > 0 || return tile_mean_temperature(ts)
    return (_wsum(t -> surface_emissivity(t) * surface_temperature(t)^4, ts) / εbar)^(1 / 4)
end

"""
    tile_mean_temperature(ts) -> K

Area-weighted mean skin temperature ``\\sum f_i T_i`` — the temperature
a parameter-averaging scheme would carry. It is *not* the radiative
temperature; see [`effective_radiative_temperature`](@ref).
"""
tile_mean_temperature(ts::SurfaceMosaic) = _wsum(surface_temperature, ts)

"The skin temperature of a tiled cell is its [`effective_radiative_temperature`](@ref)."
surface_temperature(ts::SurfaceMosaic) = effective_radiative_temperature(ts)

"""
    roughness_lengths(ts::SurfaceMosaic)

Parameter aggregate of the tile roughness lengths by logarithmic
averaging, ``\\ln z_{0,eff} = \\sum f_i \\ln z_{0,i}`` (for each of
`z0m`, `z0h`, `z0q`). Log-averaging is the natural choice because the
neutral drag depends on ``\\ln z_0``; it is still only an approximation
of the tile-aggregated stress: it is the ``l_b → ∞`` limit of the
blending-height [`effective_roughness`](@ref) (Mason 1988), which a
[`BlendedMosaic`](@ref) reports instead. A tiled cell's own fluxes never
use it.
"""
function roughness_lengths(ts::SurfaceMosaic)
    lg(k) = exp(_wsum(t -> log(getfield(roughness_lengths(t), k)), ts))
    return (z0m = lg(:z0m), z0h = lg(:z0h), z0q = lg(:z0q))
end

"Parameter aggregate ``\\sum f_i β_i`` of the tiles' moisture availabilities."
moisture_availability(ts::SurfaceMosaic) = _wsum(moisture_availability, ts)

# =========================================================================
# Parameter aggregation, for comparison
# =========================================================================

"""
    ParameterAggregatedSurface(; T, albedo, emissivity, z0m, z0h, z0q, β)

A single homogeneous surface described by effective parameters — the
*parameter-aggregation* alternative to tiling. It is a diagnostic
snapshot (no prognostic state, no `step_surface!`): the default
`SurfaceExchange` hooks turn it into fluxes, so
`turbulent_fluxes(p, forcing)` is ``F(\\bar p)`` to set against a tiled
cell's ``\\sum f_i F(p_i)``. Build one from a tiled cell with
[`parameter_aggregate`](@ref).
"""
Base.@kwdef struct ParameterAggregatedSurface <: AbstractSurface
    T::Float64
    albedo::Float64
    emissivity::Float64
    z0m::Float64
    z0h::Float64
    z0q::Float64 = z0h
    β::Float64 = 0.0
end

surface_temperature(p::ParameterAggregatedSurface) = p.T
roughness_lengths(p::ParameterAggregatedSurface) = (z0m = p.z0m, z0h = p.z0h, z0q = p.z0q)
surface_albedo(p::ParameterAggregatedSurface) = p.albedo
surface_emissivity(p::ParameterAggregatedSurface) = p.emissivity
moisture_availability(p::ParameterAggregatedSurface) = p.β

"""
    parameter_aggregate(ts; temperature = :mean) -> ParameterAggregatedSurface

The one-surface stand-in for a tiled cell with averaged parameters:
area-weighted albedo, emissivity and moisture availability,
log-averaged roughness lengths, and as skin temperature either the
area-weighted mean (`temperature = :mean`, what a single-tile scheme
with averaged heat capacity would carry) or the effective radiative
temperature (`:radiative`, which gets the emitted longwave right).

Where the fluxes are linear in the parameters that differ between tiles,
`turbulent_fluxes(parameter_aggregate(ts), f)` equals the flux-aggregated
`turbulent_fluxes(ts, f)`; where they are not — stability, roughness,
saturation humidity — the difference is the aggregation error that
tiling removes (Avissar & Pielke 1989; Mahrt 1987).
"""
function parameter_aggregate(ts::SurfaceMosaic; temperature::Symbol = :mean)
    T = temperature === :mean      ? tile_mean_temperature(ts) :
        temperature === :radiative ? effective_radiative_temperature(ts) :
        throw(ArgumentError("parameter_aggregate: temperature must be :mean or :radiative"))
    z0 = roughness_lengths(ts)
    return ParameterAggregatedSurface(T = T, albedo = surface_albedo(ts),
                                      emissivity = surface_emissivity(ts),
                                      z0m = z0.z0m, z0h = z0.z0h, z0q = z0.z0q,
                                      β = moisture_availability(ts))
end

# =========================================================================
# Class maps (generalising the urban five-class surface map)
# =========================================================================

"""
    class_fractions(classes, nclass) -> Vector{Float64}

Area fractions of the integer classes `1:nclass` in the array `classes`
(each element one equal-area cell), e.g. the block of a fine surface
class map that falls into one coarse grid cell. Throws if a class lies
outside `1:nclass`.
"""
function class_fractions(classes::AbstractArray{<:Integer}, nclass::Integer)
    counts = zeros(Int, nclass)
    for c in classes
        1 <= c <= nclass ||
            throw(ArgumentError("class_fractions: class $c outside 1:$nclass"))
        counts[c] += 1
    end
    return counts ./ length(classes)
end

"""
    tiles_from_class_map(class_map, block, make_tile; nclass = maximum(class_map))
        -> Matrix{SurfaceMosaic}

Coarse-grain a 2-D surface class map (such as the urban
street/roof/park/water/wall map of `examples/urban`) into tiled cells.
`block = (bx, by)` fine cells make one coarse cell; the map size must be
divisible by it. In each coarse cell there is one tile per class that
occurs, built by `make_tile(class)` (a fresh `AbstractSurface` each
call), with the class's area fraction.

The per-cell class map of the urban example is the limit `block = (1, 1)`,
one pure tile per cell.
"""
function tiles_from_class_map(class_map::AbstractMatrix{<:Integer}, block::NTuple{2,Integer},
                              make_tile; nclass::Integer = maximum(class_map))
    nx, ny = size(class_map)
    bx, by = block
    (bx > 0 && by > 0 && nx % bx == 0 && ny % by == 0) ||
        throw(ArgumentError("tiles_from_class_map: map $(size(class_map)) not divisible by block $block"))
    cells = Matrix{SurfaceMosaic}(undef, nx ÷ bx, ny ÷ by)
    for J in axes(cells, 2), I in axes(cells, 1)
        blk = view(class_map, (I - 1) * bx + 1:I * bx, (J - 1) * by + 1:J * by)
        f = class_fractions(blk, nclass)
        present = findall(>(0), f)
        cells[I, J] = SurfaceMosaic(AbstractSurface[make_tile(k) for k in present], f[present])
    end
    return cells
end

# =========================================================================
# Blending height: effective roughness (#42)
# =========================================================================
#
# Below the blending height l_b the flow is in equilibrium with each tile;
# above it the tiles' internal boundary layers have merged and one
# horizontally uniform profile exists (Wieringa 1986; Mason 1988). Wind,
# temperature and humidity are therefore common to all tiles *at l_b*,
# and each tile exchanges with that common state through its own surface
# layer of depth l_b.

# Tiles that carry weight; a zero-fraction tile never constrains l_b.
_active(z0s, f) = [(z, w) for (z, w) in zip(z0s, f) if w > 0]

function _check_blending(l_b, z0s, f)
    length(z0s) == length(f) ||
        throw(DimensionMismatch("blending height: $(length(z0s)) roughness lengths, $(length(f)) fractions"))
    act = _active(z0s, f)
    isempty(act) && throw(ArgumentError("blending height: no tile with a positive fraction"))
    all(((z, _),) -> 0 < z < l_b, act) ||
        throw(ArgumentError("blending height l_b = $l_b m must exceed every roughness length (got $(maximum(first, act)) m)"))
    return act
end

"""
    effective_roughness(z0s, fractions, l_b) -> z0_eff

The effective momentum roughness length of a patchwork of tiles with
roughness lengths `z0s` and area fractions `fractions`, for a blending
height `l_b` (m):

```math
\\frac{1}{\\ln^2(l_b/z_{0,eff})} = \\sum_i \\frac{f_i}{\\ln^2(l_b/z_{0,i})} .
```

This is the neutral drag law applied tile by tile with the wind common
to all tiles at ``l_b`` (Wieringa 1986; Mason 1988): the tile stresses
``ρ[κU(l_b)/\\ln(l_b/z_{0,i})]^2`` average to the stress of one surface of
roughness ``z_{0,eff}``. Because ``1/\\ln^2`` is convex, ``z_{0,eff}``
lies above the logarithmic mean ``\\exp\\sum f_i\\ln z_{0,i}`` and below the
largest ``z_{0,i}``; it tends to the logarithmic mean as ``l_b → ∞``
(the log-average model of Taylor 1987) as
``\\ln z_{0,eff} = \\overline{\\ln z_0} + \\tfrac32\\,\\mathrm{Var}(\\ln z_0)/\\ln(l_b/z_0) + O(\\ln^{-2})``,
and grows as the blending height falls, i.e. as the patches become
smaller. Form drag of obstacles at patch edges is not included: this is
the aggregate of the tiles' own log-law drag.

Zero-fraction tiles are ignored; ``l_b`` must exceed every remaining
``z_{0,i}``.

    effective_roughness(m::SurfaceMosaic, l_b) -> (z0m, z0h, z0q)
    effective_roughness(m::SurfaceMosaic, bh::AbstractBlendingHeight) -> (z0m, z0h, z0q)

All three effective roughness lengths of a mosaic's tiles, with the
scalar ones from [`effective_scalar_roughness`](@ref); with a
blending-height model `bh` the height is [`blending_height`](@ref)`(bh, m)`.
"""
function effective_roughness(z0s::AbstractVector{<:Real}, f::AbstractVector{<:Real}, l_b::Real)
    act = _check_blending(l_b, z0s, f)
    S = sum(w / log(l_b / z)^2 for (z, w) in act) / sum(last, act)
    return l_b * exp(-1 / sqrt(S))
end
effective_roughness(z0s::Tuple, f, l_b::Real) = effective_roughness(collect(Float64, z0s), collect(Float64, f), l_b)

"""
    effective_scalar_roughness(z0ms, z0hs, fractions, l_b; z0m_eff = effective_roughness(z0ms, fractions, l_b))

Effective roughness length for heat (or moisture) at blending height
`l_b`: the neutral bulk transfer coefficient ``κ^2/[\\ln(l_b/z_{0m})
\\ln(l_b/z_{0h})]`` is averaged over the tiles,

```math
\\frac{1}{\\ln(l_b/z_{0m,eff})\\,\\ln(l_b/z_{0h,eff})} =
\\sum_i \\frac{f_i}{\\ln(l_b/z_{0m,i})\\,\\ln(l_b/z_{0h,i})} ,
```

which is exact for the cell heat flux when every tile has the same
surface-to-``l_b`` temperature difference under neutral stratification
— the same construction as [`effective_roughness`](@ref), applied to
the scalar exchange coefficient.
"""
function effective_scalar_roughness(z0ms::AbstractVector{<:Real}, z0hs::AbstractVector{<:Real},
                                    f::AbstractVector{<:Real}, l_b::Real;
                                    z0m_eff::Real = effective_roughness(z0ms, f, l_b))
    length(z0hs) == length(z0ms) ||
        throw(DimensionMismatch("effective_scalar_roughness: $(length(z0ms)) z0m, $(length(z0hs)) z0h"))
    _check_blending(l_b, z0ms, f); _check_blending(l_b, z0hs, f)
    S = sum(w / (log(l_b / zm) * log(l_b / zh)) for (zm, zh, w) in zip(z0ms, z0hs, f) if w > 0) /
        sum(w for w in f if w > 0)
    return l_b * exp(-1 / (S * log(l_b / z0m_eff)))
end

_tile_roughness(m::SurfaceMosaic, k::Symbol) = Float64[getfield(roughness_lengths(t), k) for t in m.tiles]

function effective_roughness(m::SurfaceMosaic, l_b::Real)
    zm, zh, zq = _tile_roughness(m, :z0m), _tile_roughness(m, :z0h), _tile_roughness(m, :z0q)
    f = m.fractions
    z0m = effective_roughness(zm, f, l_b)
    return (z0m = z0m,
            z0h = effective_scalar_roughness(zm, zh, f, l_b; z0m_eff = z0m),
            z0q = effective_scalar_roughness(zm, zq, f, l_b; z0m_eff = z0m))
end

# =========================================================================
# Blending-height closures
# =========================================================================

"""
    AbstractBlendingHeight

A rule for the blending height ``l_b`` of a patchwork of tiles.
Implementers extend `blending_height(bh, z0s, fractions; c)`. Provided:
[`FixedBlendingHeight`](@ref), [`MasonBlendingHeight`](@ref),
[`BouZeidBlendingHeight`](@ref).
"""
abstract type AbstractBlendingHeight end

"""
    FixedBlendingHeight(l_b)

A prescribed blending height ``l_b`` (m), e.g. taken from where the
resolved flow of an LES over the patches becomes horizontally uniform
(Bou-Zeid et al. 2004 determine it from where the velocity quartile
profiles merge).
"""
struct FixedBlendingHeight <: AbstractBlendingHeight
    l_b::Float64
    function FixedBlendingHeight(l_b::Real)
        l_b > 0 || throw(ArgumentError("FixedBlendingHeight: l_b must be positive"))
        new(Float64(l_b))
    end
end

"""
    MasonBlendingHeight(L; C = 2)

Mason's (1988) heuristic blending height for patches of horizontal
length scale `L` (m): the height at which ``(u_*/U(l_b))^2 ≈ l_b/(C L)``,

```math
\\frac{l_b}{L}\\,\\ln^2\\!\\frac{l_b}{z_{0,eff}} = C κ^2, \\qquad C ≈ 2,
```

(form as quoted by Kramm & Dlugi 2009, Eq. 5.4), solved jointly with the
[`effective_roughness`](@ref) ``z_{0,eff}(l_b)``. For ``L`` of a few
kilometres and ``z_0 ∼ 0.01``–``0.1`` m it gives ``l_b`` of tens of
metres, ``O(L/100)``.
"""
struct MasonBlendingHeight <: AbstractBlendingHeight
    L::Float64
    C::Float64
    function MasonBlendingHeight(L::Real; C::Real = 2.0)
        (L > 0 && C > 0) || throw(ArgumentError("MasonBlendingHeight: L and C must be positive"))
        new(Float64(L), Float64(C))
    end
end

"""
    ClaussenBlendingHeight(L; C = 1.75)

Claussen's (1991) blending height for patches of horizontal scale `L`
(m),

```math
\\frac{l_b}{L}\\,\\ln\\frac{l_b}{z_{0,eff}} = C κ, \\qquad C ≈ 1.75,
```

(form and constant as quoted by Kramm & Dlugi 2009, Eq. 5.5), solved
jointly with [`effective_roughness`](@ref). It is
[`BouZeidBlendingHeight`](@ref) without the ``-1``; Bou-Zeid et al.
(2007, Sect. 6) note that the two give very similar ``z_{0,eff}``, the
``-1`` mattering only for short patches.
"""
struct ClaussenBlendingHeight <: AbstractBlendingHeight
    L::Float64
    C::Float64
    function ClaussenBlendingHeight(L::Real; C::Real = 1.75)
        (L > 0 && C > 0) || throw(ArgumentError("ClaussenBlendingHeight: L and C must be positive"))
        new(Float64(L), Float64(C))
    end
end

"""
    BouZeidBlendingHeight(L; C = 1.7)

The blending height of Bou-Zeid, Meneveau & Parlange (2004), tested
against LES of complex patchworks in Bou-Zeid et al. (2007): blending is
complete where internal boundary layers growing by the diffusion analogy
over about twice the patch length `L` (m) meet,
``l_b[\\ln(l_b/z_{0,eff}) - 1] = C κ L``. Combined with
[`effective_roughness`](@ref) this is Eq. (2) of Bou-Zeid et al. (2007),

```math
\\left(\\frac{l_b}{C κ L + l_b}\\right)^2 = \\sum_i \\frac{f_i}{\\ln^2(l_b/z_{0,i})},
\\qquad z_{0,eff} = l_b \\exp\\!\\left(-\\frac{C κ L}{l_b} - 1\\right),
```

with ``C = 1.7`` and ``κ = 0.4``. For regular stripes `L` is the mean
patch length; for patches that are not regular stripes Bou-Zeid et al.
(2007, Eq. 6) take the integral length scale of the roughness structure
function.
"""
struct BouZeidBlendingHeight <: AbstractBlendingHeight
    L::Float64
    C::Float64
    function BouZeidBlendingHeight(L::Real; C::Real = 1.7)
        (L > 0 && C > 0) || throw(ArgumentError("BouZeidBlendingHeight: L and C must be positive"))
        new(Float64(L), Float64(C))
    end
end

"""
    blending_height(bh::AbstractBlendingHeight, z0s, fractions; c = DEFAULT_SURFACE_CONSTANTS) -> l_b
    blending_height(bh, m::SurfaceMosaic; c = DEFAULT_SURFACE_CONSTANTS) -> l_b

Blending height (m) of tiles with momentum roughness lengths `z0s` and
area `fractions` (or of the tiles of a mosaic) under the closure `bh`.
The von Kármán constant is `c.κ`.
"""
function blending_height end

blending_height(bh::FixedBlendingHeight, z0s, f; c::SurfaceConstants = DEFAULT_SURFACE_CONSTANTS) = bh.l_b

_lnlz(z0s, f, l) = sqrt(sum(w for w in f if w > 0) / sum(w / log(l / z)^2 for (z, w) in zip(z0s, f) if w > 0))

function blending_height(bh::MasonBlendingHeight, z0s, f; c::SurfaceConstants = DEFAULT_SURFACE_CONSTANTS)
    κ = c.κ
    return _increasing_root(l -> (l / bh.L) * _lnlz(z0s, f, l)^2 - bh.C * κ^2, z0s, f)
end

function blending_height(bh::ClaussenBlendingHeight, z0s, f; c::SurfaceConstants = DEFAULT_SURFACE_CONSTANTS)
    κ = c.κ
    return _increasing_root(l -> _lnlz(z0s, f, l) - bh.C * κ * bh.L / l, z0s, f)
end

function blending_height(bh::BouZeidBlendingHeight, z0s, f; c::SurfaceConstants = DEFAULT_SURFACE_CONSTANTS)
    κ = c.κ
    return _increasing_root(l -> _lnlz(z0s, f, l) - bh.C * κ * bh.L / l - 1, z0s, f)
end

blending_height(bh::AbstractBlendingHeight, m::SurfaceMosaic; c::SurfaceConstants = DEFAULT_SURFACE_CONSTANTS) =
    blending_height(bh, _tile_roughness(m, :z0m), m.fractions; c = c)

effective_roughness(m::SurfaceMosaic, bh::AbstractBlendingHeight; c::SurfaceConstants = DEFAULT_SURFACE_CONSTANTS) =
    effective_roughness(m, blending_height(bh, m; c = c))

# Root of g(l), increasing in l, above the largest active roughness length
# (where both closures are negative). Bisection in ln l.
function _increasing_root(g, z0s, f)
    zmax = maximum(z for (z, w) in zip(z0s, f) if w > 0)
    lo = zmax * (1 + 1e-9)
    g(lo) < 0 || return lo
    hi = 2zmax
    while g(hi) <= 0
        hi *= 2
        hi < 1e9 || throw(ArgumentError("blending height: no root below 1e9 m"))
    end
    for _ in 1:200
        mid = sqrt(lo * hi)
        g(mid) <= 0 ? (lo = mid) : (hi = mid)
        hi / lo - 1 <= 4eps() && break
    end
    return sqrt(lo * hi)
end

# =========================================================================
# Internal-boundary-layer growth
# =========================================================================

"""
    AbstractIBLModel

A formula for the depth ``δ(x)`` of the internal boundary layer at fetch
``x`` downstream of a step change of roughness from `z0_up` to
`z0_down`, under neutral stratification. See [`ibl_height`](@ref) and the
compendium of Savelyev & Taylor (2005, Table I).
"""
abstract type AbstractIBLModel end

"""
    ElliottIBL()

Elliott (1958):
``δ/z_{0D} = (0.75 - 0.03 \\ln(z_{0D}/z_{0U}))\\,(x/z_{0D})^{0.8}``
(Savelyev & Taylor 2005, Eq. 1).
"""
struct ElliottIBL <: AbstractIBLModel end

"""
    PowerLawIBL(; a = 0.28, b = 0.8)

``δ/z_{0r} = a\\,(x/z_{0r})^{b}`` with ``z_{0r} = \\max(z_{0U}, z_{0D})``.
The defaults are Wood (1982), who argued that only the rougher surface
matters (Savelyev & Taylor 2005, Eq. 5).
"""
struct PowerLawIBL <: AbstractIBLModel
    a::Float64
    b::Float64
end
PowerLawIBL(; a::Real = 0.28, b::Real = 0.8) = PowerLawIBL(Float64(a), Float64(b))

"""
    DiffusionIBL(; C = 1.25)

Miyake's diffusion analogy: the interface rises at the vertical velocity
scale, ``U(δ)\\,dδ/dx = C u_*`` with ``C = σ_w/u_*``, and a logarithmic
downstream profile ``U = (u_*/κ)\\ln(z/z_{0D})``, integrated from
``δ(0) = z_{0D}``:

```math
C κ \\frac{x}{z_{0D}} = \\frac{δ}{z_{0D}}\\left(\\ln\\frac{δ}{z_{0D}} - 1\\right) + 1 .
```

``C = 1.25`` is Panofsky & Dutton (1984) (Savelyev & Taylor 2005,
Table I); Miyake's (1965) original form has ``1.73`` and starts from a
finite ``δ_0``. Differentiating gives ``dδ/dx = Cκ/\\ln(δ/z_{0D})``.
"""
struct DiffusionIBL <: AbstractIBLModel
    C::Float64
end
DiffusionIBL(; C::Real = 1.25) = DiffusionIBL(Float64(C))

"""
    ibl_height(model::AbstractIBLModel, x; z0_up, z0_down, c = DEFAULT_SURFACE_CONSTANTS) -> δ

Internal-boundary-layer depth (m) at fetch `x` (m) past a step change of
roughness `z0_up` → `z0_down` (m). The formulas are for the surface
layer, roughly ``δ ≲ 0.1``–``0.2 z_i`` (Wood 1982). The IBL depth at a
fetch of the order of the patch length is the natural scale of the
blending height (Bou-Zeid et al. 2004, 2007); see
[`IBLBlendingHeight`](@ref).
"""
function ibl_height end

function _check_ibl(x, z0_up, z0_down)
    x >= 0 || throw(ArgumentError("ibl_height: fetch must be ≥ 0"))
    (z0_up > 0 && z0_down > 0) || throw(ArgumentError("ibl_height: roughness lengths must be positive"))
end

function ibl_height(::ElliottIBL, x::Real; z0_up::Real, z0_down::Real,
                    c::SurfaceConstants = DEFAULT_SURFACE_CONSTANTS)
    _check_ibl(x, z0_up, z0_down)
    return z0_down * (0.75 - 0.03 * log(z0_down / z0_up)) * (x / z0_down)^0.8
end

function ibl_height(m::PowerLawIBL, x::Real; z0_up::Real, z0_down::Real,
                    c::SurfaceConstants = DEFAULT_SURFACE_CONSTANTS)
    _check_ibl(x, z0_up, z0_down)
    z0r = max(z0_up, z0_down)
    return z0r * m.a * (x / z0r)^m.b
end

function ibl_height(m::DiffusionIBL, x::Real; z0_up::Real, z0_down::Real,
                    c::SurfaceConstants = DEFAULT_SURFACE_CONSTANTS)
    _check_ibl(x, z0_up, z0_down)
    rhs = m.C * c.κ * x / z0_down - 1          # = y (ln y - 1), y = δ/z0D ≥ 1
    rhs <= -1 && return float(z0_down)
    # Newton on s = ln y for h(s) = e^s (s - 1) - rhs, h' = s e^s (convex, increasing for s > 0).
    s = max(log(max(rhs, 1.0)), 1.0)
    for _ in 1:100
        es = exp(s)
        ds = (es * (s - 1) - rhs) / (s * es)
        s -= ds
        s = max(s, 1e-12)
        abs(ds) <= 4eps() * max(abs(s), 1.0) && break
    end
    return z0_down * exp(s)
end

"""
    IBLBlendingHeight(model::AbstractIBLModel, L; fetch_factor = 2)

Blending height from internal-boundary-layer growth: blending is complete
where the IBLs that start at the patch edges have grown over
`fetch_factor × L` (the premise of Bou-Zeid et al. 2004, who found that
blending needs about twice the patch length `L`). ``l_b`` is the deepest
IBL [`ibl_height`](@ref)`(model, fetch_factor * L)` over every ordered
pair of distinct active tiles (upstream ``z_{0,j}``, downstream
``z_{0,i}``); with a single active tile it is the depth over that
surface. With [`DiffusionIBL`](@ref) this is, up to its constant, the
IBL half of [`BouZeidBlendingHeight`](@ref) evaluated with the patch
roughness instead of ``z_{0,eff}``.
"""
struct IBLBlendingHeight{M<:AbstractIBLModel} <: AbstractBlendingHeight
    model::M
    L::Float64
    fetch_factor::Float64
    function IBLBlendingHeight(model::M, L::Real; fetch_factor::Real = 2.0) where {M<:AbstractIBLModel}
        (L > 0 && fetch_factor > 0) ||
            throw(ArgumentError("IBLBlendingHeight: L and fetch_factor must be positive"))
        new{M}(model, Float64(L), Float64(fetch_factor))
    end
end

function blending_height(bh::IBLBlendingHeight, z0s, f; c::SurfaceConstants = DEFAULT_SURFACE_CONSTANTS)
    act = [z for (z, w) in zip(z0s, f) if w > 0]
    isempty(act) && throw(ArgumentError("blending height: no tile with a positive fraction"))
    x = bh.fetch_factor * bh.L
    length(act) == 1 && return ibl_height(bh.model, x; z0_up = act[1], z0_down = act[1], c = c)
    return maximum(ibl_height(bh.model, x; z0_up = zu, z0_down = zd, c = c)
                   for (j, zu) in enumerate(act), (i, zd) in enumerate(act) if i != j)
end


end # module
