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

The blending-height formulation (tile-specific forcing heights) is a
separate phase; tiles here all see the grid-cell forcing, or a forcing
per tile passed explicitly to `step_surface!`.

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
"""
module SurfaceTiles

using ..SurfaceExchange: AbstractSurface, SurfaceForcing, SurfaceFluxes,
                         SurfaceConstants, DEFAULT_SURFACE_CONSTANTS
import ..SurfaceExchange: surface_temperature, roughness_lengths,
                          surface_albedo, surface_emissivity,
                          moisture_availability, net_radiation,
                          turbulent_fluxes, step_surface!

export SurfaceMosaic, ntiles, mosaic_fractions, set_mosaic_fractions!,
       aggregate_fluxes, tile_fluxes, last_tile_fluxes,
       effective_radiative_temperature, tile_mean_temperature,
       ParameterAggregatedSurface, parameter_aggregate,
       class_fractions, tiles_from_class_map

# =========================================================================
# Fractions
# =========================================================================

# Validate and normalise area fractions. Normalising (rather than only
# checking) makes Σ f = 1 hold to the last bit that floating point
# allows, so the aggregate conserves what the tiles conserve.
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
then renormalised so the sum is exact. A zero-fraction tile is allowed
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

"Area fractions of the tiles of a [`SurfaceMosaic`](@ref) (sum exactly 1)."
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
of the tile-aggregated stress (it ignores the blending height, Mason
1988). A tiled cell's own fluxes never use it.
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

end # module
