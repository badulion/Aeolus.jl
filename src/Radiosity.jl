module Radiosity

# Phase 35 — Full urban radiosity.
#
# What this replaces. The Era-I urban example (`examples/urban/urban_physics.jl`)
# treats urban radiation with two scalar diagnostics per ground cell: a binary
# ray-marched shadow mask and a hemispherically ray-cast sky-view factor. Both
# are geometrically sensible, but the *reflections* are not modelled at all —
# the downward longwave inside a canyon is blended as `320·svf + 380·(1-svf)`
# with hard-coded temperatures, and shortwave that bounces off a wall onto the
# street is simply lost. In a dense canyon that is not a small correction: for
# a height-to-width ratio of 2 and a facade albedo of 0.2, multiple bounces
# roughly halve the radiation that actually escapes to the sky.
#
# Why that matters physically. An urban canyon is a *cavity*. Radiation that
# enters it gets many chances to be absorbed, because each reflection off a
# facade is aimed mostly at another facade rather than at the sky. Two
# first-order consequences drive the urban heat island:
#
#   * Shortwave: the canyon's **effective albedo** is strictly smaller than the
#     albedo of the material it is built from, and shrinks as the canyon
#     deepens. A street of asphalt (α = 0.1) and brick (α = 0.2) behaves like a
#     flat surface of albedo ≈ 0.1 or less once H/W ≳ 2.
#   * Longwave: a facet deep in a canyon exchanges most of its thermal emission
#     with neighbouring facets at a similar temperature instead of with the
#     cold sky, so its *net* radiative cooling collapses. This is why city
#     centres cool so slowly after sunset.
#
# Both effects are consequences of the same object: the facet-to-facet
# view-factor matrix. This module computes it properly and solves the two
# radiosity systems that ride on it.
#
# Method.
#
#   1. `build_facets` turns a building heightmap (the `height[i,j]` =
#      "solid top of column (i,j)" convention shared with
#      `CutCell.cut_cell_geometry_from_heightmap`) into explicit rectangular
#      facets: one ground patch per open column, one roof patch per built
#      column, and — for each of the four sides of each built column — a wall
#      rectangle spanning the height *step* against that neighbour, subdivided
#      vertically. Grid-aligned rectangles are not a limitation here: they are
#      exactly what a heightmap is.
#
#   2. `view_factors` builds `F[i,j]` = fraction of the radiation leaving facet
#      `i` that lands on facet `j`. Two exactness properties are engineered in
#      rather than hoped for:
#
#        * **Reciprocity** `A_i F_ij = A_j F_ji` holds to machine precision,
#          because the *exchange area* `A_i F_ij` is computed once per pair by
#          a formula that is manifestly symmetric in `i` and `j`, then divided
#          by the two areas.
#        * **Closure** `Σ_j F_ij + F_sky_i = 1` holds by construction, because
#          the sky is not given a formula of its own: it is the leftover,
#          `F_sky_i = 1 - Σ_j F_ij`. That makes the sky-view factor a *test* of
#          the facet-to-facet matrix rather than an independent approximation,
#          and `sky_view_factor_raycast` provides the external cross-check.
#
#      The exchange area itself uses the **double contour integral** (Stokes'
#      reduction of the double area integral; Modest, *Radiative Heat Transfer*
#      3rd ed. §4.5, eq. 4.42)
#
#          A_1 F_12 = (1/2π) ∮_{C1} ∮_{C2} ln r  (dx₁dx₂ + dy₁dy₂ + dz₁dz₂)
#
#      whenever the two facets are fully mutually visible. For planar polygons
#      this is *exact*, and it is dramatically better conditioned than the
#      naive area quadrature for the touching-facet pairs that dominate an
#      urban scene (a street patch and the wall rising out of its edge). Edge
#      pairs that are perpendicular contribute nothing (`dr₁·dr₂ = 0`); edge
#      pairs that are collinear are singular and are integrated in closed form;
#      everything else is Gauss-Legendre. When a pair is *partially* occluded
#      by an intervening building, the exact exchange area is scaled by the
#      occluded/unoccluded ratio of a subdivided area quadrature carrying a
#      per-sample-pair ray test. That occlusion test is what makes this an
#      urban model rather than a two-facet canyon idealisation.
#
#   3. `solve_shortwave` solves `B = E + diag(α) F B` — outgoing radiosity given
#      the primary (once-reflected) source `E_i = α_i Q^ext_i`. Dense LU and
#      Gauss-Seidel / progressive-refinement solvers are both provided and are
#      checked against each other.
#
#   4. `solve_longwave` solves `B = ε σ T⁴ + (1-ε)(F B + F_sky L_sky)` and
#      returns the net longwave loss per facet.
#
# Accuracy statement, stated plainly. The contour integral is exact for the
# unoccluded pair; its only error is the Gauss-Legendre truncation on
# non-singular edge pairs, which for the grid-aligned geometry here is at the
# 1e-6 level with the default 8 nodes. The *occluded* pairs carry the error of
# the occlusion ratio, which is a subdivided-quadrature estimate and is
# first-order accurate in the subdivision — typically a few times 1e-3 of the
# pair's exchange area. Scenes are assumed opaque and grey within each of the
# shortwave and longwave bands, reflections diffuse (Lambertian), and the sky
# isotropic. Specular glazing needs a BRDF and is out of scope here; see
# `examples/urban/glass_material.jl`.
#
# Dependency note. `SolarGeometry` (Phase 38) is included *after* this module
# in `src/Aeolus.jl`, so it cannot be `using`-ed here. `solar_unit_vector`
# therefore duck-types its argument: it accepts a raw east-north-up tuple, or
# any object exposing `.elevation` and `.azimuth` in degrees — which is exactly
# `SolarGeometry.SolarPosition`, and reproduces `sun_direction_enu` on it bit
# for bit.

using LinearAlgebra
using ..Grids
using ..Grids: RegularGrid
using ..RadiationCore
using ..RadiationCore: STEFAN_BOLTZMANN, SOLAR_CONSTANT, AbstractSurfaceOptics,
                       SpectralGrid, shortwave_bands, longwave_bands
import ..RadiationCore: material_albedo, material_emissivity
using ..SurfaceOptics: TiledSurface, tiled_surface, FresnelWater,
                       broadband_albedo, broadband_emissivity

export Facet, facet_corners, facet_sample_points, facet_edges,
       UrbanMaterials, default_urban_materials, uniform_materials,
       material_albedo, material_emissivity,
       URBAN_STREET, URBAN_ROOF, URBAN_PARK, URBAN_WATER, URBAN_WALL,
       RadiosityScene, build_facets, build_scene, heightmap_from_solid_mask,
       nfacets, facet_areas, facet_albedos, facet_emissivities,
       facet_temperatures, set_temperatures!, set_temperature!,
       ViewFactors, view_factors, exchange_area, view_factor_contour,
       sky_view_factor_raycast, sky_view_factors_raycast,
       reciprocity_error, closure_error, total_sky_exchange_area,
       view_factor_parallel_rectangles, view_factor_perpendicular_rectangles,
       canyon_sky_view_factor, canyon_centreline_sky_view_factor,
       canyon_wallside_sky_view_factor,
       solar_unit_vector, facet_sunlit_fraction, sunlit_fractions,
       direct_beam_irradiance, diffuse_sky_irradiance,
       solve_radiosity_direct, solve_radiosity_iterative,
       ShortwaveSolution, solve_shortwave,
       LongwaveSolution, solve_longwave,
       LongwaveResponse, longwave_response,
       effective_albedo, longwave_trapping_factor,
       UrbanRadiationForcing, aggregate_to_grid,
       ray_blocked, segment_blocked

# =========================================================================
# Small vector helpers
# =========================================================================
# NTuple{3} arithmetic. StaticArrays does not build in this environment and
# is not needed: tuples of three floats are already stack-allocated and
# fully unrolled.

const Vec3{T} = NTuple{3, T}

@inline _add(a::Vec3, b::Vec3) = (a[1] + b[1], a[2] + b[2], a[3] + b[3])
@inline _sub(a::Vec3, b::Vec3) = (a[1] - b[1], a[2] - b[2], a[3] - b[3])
@inline _mul(a::Vec3, s::Real) = (a[1] * s, a[2] * s, a[3] * s)
@inline _dot(a::Vec3, b::Vec3) = a[1] * b[1] + a[2] * b[2] + a[3] * b[3]
@inline _cross(a::Vec3, b::Vec3) = (a[2] * b[3] - a[3] * b[2],
                                    a[3] * b[1] - a[1] * b[3],
                                    a[1] * b[2] - a[2] * b[1])
@inline _norm(a::Vec3) = sqrt(_dot(a, a))
@inline function _unit(a::Vec3)
    n = _norm(a)
    n == 0 && return a
    return _mul(a, inv(n))
end

# -------------------------------------------------------------------------
# Gauss-Legendre nodes (local copy)
# -------------------------------------------------------------------------
# `RadiationCore` has an equivalent private helper. It is duplicated here
# deliberately: this module must not depend on another module's non-exported
# internals, and the routine is twenty lines of Newton iteration on the
# Legendre polynomial.

const _GL_NODES = Dict{Int, Tuple{Vector{Float64}, Vector{Float64}}}()

"""
    _gauss_nodes(n) -> (x, w)

`n`-point Gauss-Legendre nodes and weights on `[-1, 1]`, by Newton iteration
on the Legendre polynomial `P_n`. Cached, because view-factor assembly calls
it once per edge pair.
"""
function _gauss_nodes(n::Int)
    haskey(_GL_NODES, n) && return _GL_NODES[n]
    n >= 1 || error("_gauss_nodes: need n ≥ 1, got $n")
    x = zeros(Float64, n); w = zeros(Float64, n)
    for i in 1:n
        z = cos(π * (i - 0.25) / (n + 0.5))      # Chebyshev-like initial guess
        p1 = 0.0; dp = 1.0
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
    _GL_NODES[n] = (x, w)
    return (x, w)
end

# =========================================================================
# Materials
# =========================================================================

"""
Surface-class indices. These deliberately coincide with the `SURFACE_*`
constants of `examples/urban/urban_geometry.jl`, so a `Facet`'s material index
can be used directly as an index into that example's `SurfaceParams` arrays and
a `RadiosityScene` can be driven from an existing `surface_class_map`.
"""
const URBAN_STREET = 1
const URBAN_ROOF   = 2
const URBAN_PARK   = 3
const URBAN_WATER  = 4
const URBAN_WALL   = 5

"""
    UrbanMaterials{T}

Grey broadband optical properties per surface class:

  * `albedo`     — shortwave hemispherical reflectance `α ∈ [0, 1)`
  * `emissivity` — longwave emissivity `ε ∈ (0, 1]`; the longwave reflectance
                   is `1 - ε` by Kirchhoff's law
  * `name`       — human-readable class label

Reflection is assumed **Lambertian** in both bands, which is what makes the
radiosity formulation valid: a Lambertian surface's outgoing radiance is
direction-independent, so a single scalar `B_i` per facet is a complete
description of its emission. Specular facades (glazing) violate this and need
a directional method.

A subtype of `RadiationCore.AbstractSurfaceOptics`; `material_albedo` and
`material_emissivity` are the accessors — methods of the generics declared
in `RadiationCore`, which `SurfaceOptics` also extends with its spectral
versions. Same name, dispatched on argument type: ask for the albedo of a
`Symbol` and you get a spectral object, ask for the albedo of an
`UrbanMaterials` class and you get the grey scalar the radiosity solve
needs. The greys are derived from the spectral curves, so they agree by
construction.
"""
struct UrbanMaterials{T<:AbstractFloat} <: AbstractSurfaceOptics
    albedo::Vector{T}
    emissivity::Vector{T}
    name::Vector{String}
end

function UrbanMaterials(albedo::AbstractVector, emissivity::AbstractVector;
                        name::AbstractVector{<:AbstractString} =
                            ["class $i" for i in eachindex(albedo)])
    T = promote_type(eltype(float.(albedo)), eltype(float.(emissivity)))
    length(albedo) == length(emissivity) ||
        error("UrbanMaterials: albedo and emissivity must have equal length")
    all(0 .<= albedo .< 1) ||
        error("UrbanMaterials: albedo must lie in [0, 1) — a perfectly " *
              "reflecting scene has no radiosity solution")
    all(0 .< emissivity .<= 1) ||
        error("UrbanMaterials: emissivity must lie in (0, 1]")
    return UrbanMaterials{T}(collect(T, albedo), collect(T, emissivity),
                             collect(String, name))
end

# =========================================================================
# Urban surface classes, derived from the Phase 36 spectral library
# =========================================================================
#
# Each urban class is a MOSAIC of real materials, not a hand-entered
# number. A city street is mostly asphalt with concrete pavements; a
# European city-centre roofscape is clay tile plus some bitumen flat roof
# and a little sheet metal; a facade is render plus brick plus glazing.
#
# Deriving the class albedo from those fractions, rather than tabulating
# it, is what keeps this module and `SurfaceOptics` from drifting apart.
# Before this, `Radiosity` carried its own five grey numbers and
# `SurfaceOptics` carried sixteen spectral reflectance curves, and the
# albedo of concrete was therefore defined twice in the same package.
# They happened to agree; nothing made them agree.
#
# Two things fall out of the derivation that the old table could not
# represent:
#
#   * WATER becomes solar-zenith dependent. A flat water surface reflects
#     ~0.03 overhead and ~0.29 at 80° — a tenfold swing that a single
#     grey 0.06 cannot express, and which dominates the energy balance of
#     an urban river or lake in the morning and evening.
#   * METAL ROOFS drag the roof-class emissivity down to ~0.85. That is
#     physically right and is precisely why metal roofs run so hot: they
#     absorb shortwave like everything else but cannot radiate it away.
#
# The mixture fractions below are for a warm-summer European city centre.
# They are editable — that is the point — via `urban_class_mixtures`.

"""
    URBAN_CLASS_MIXTURES

Material composition of each urban surface class, as
`class => (material => area fraction, ...)`. Fractions within a class
sum to one. Materials are names from
[`SurfaceOptics.material_names`](@ref).

The default is a warm-summer European city centre. Pass a modified
mapping to [`default_urban_materials`](@ref) to describe a different
building stock — a North American downtown with flat bitumen roofs and
curtain-wall glazing has a very different roof and wall composition, and
that difference is worth several degrees of surface temperature.
"""
const URBAN_CLASS_MIXTURES = (
    street = (:asphalt => 0.85, :concrete => 0.15),
    roof   = (:roof_tile_red => 0.40, :roof_tile_grey => 0.25,
              :asphalt => 0.25, :metal_roof => 0.10),
    park   = (:grass => 0.65, :tree_deciduous => 0.30, :soil_dry => 0.05),
    water  = (:water => 1.00,),
    wall   = (:concrete => 0.35, :brick => 0.35, :glass => 0.30),
)

"""
    urban_class_mixtures() -> NamedTuple

The default class composition, [`URBAN_CLASS_MIXTURES`](@ref).
"""
urban_class_mixtures() = URBAN_CLASS_MIXTURES

"""
    default_urban_materials(T = Float64; μ₀ = 0.7, mixtures, spec, lw_spec)

Urban material properties for a warm-summer European city centre,
**derived** from the Phase 36 spectral library by band-averaging a
mosaic of real materials — see [`URBAN_CLASS_MIXTURES`](@ref).

`μ₀` is the cosine of the solar zenith angle. Every class brightens
toward grazing incidence — the Phase 36 library carries a Briegleb
direct-beam shape function, and rough surfaces really do get brighter at
low sun — but water does so by a different mechanism and by an order of
magnitude more:

| class  | α(μ₀=1) | α(μ₀=0.2) | ratio |
|:-------|:--------|:----------|:------|
| street | 0.119   | 0.138     | 1.15  |
| wall   | 0.212   | 0.262     | 1.24  |
| roof   | 0.198   | 0.257     | 1.30  |
| park   | 0.178   | 0.273     | 1.53  |
| water  | 0.030   | 0.258     | 8.60  |

Water is a smooth Fresnel interface, not a rough Lambertian one. For a
diurnal run, rebuild the materials as the sun moves rather than fixing
them once — otherwise a morning or evening water surface is wrong by a
factor of eight, and vegetation by half.

Derived values at `μ₀ = 0.7` (the previous hand-entered table in
brackets, for the record):

| class  | α            | ε            |
|:-------|:-------------|:-------------|
| street | 0.125 (0.10) | 0.944 (0.95) |
| roof   | 0.216 (0.20) | 0.848 (0.90) |
| park   | 0.205 (0.22) | 0.977 (0.98) |
| water  | 0.040 (0.06) | 0.960 (0.98) |
| wall   | 0.227 (0.18) | 0.926 (0.94) |

The street and wall classes come out brighter than the old table. That
is the derivation disagreeing with a guess, and the derivation has the
better provenance: measured facade albedos are 0.20-0.40 for light
render and 0.25-0.35 for brick, so 0.18 was on the dark side. The roof
emissivity drop from 0.90 to 0.85 is the metal fraction asserting
itself, as it should.
"""
function default_urban_materials(::Type{T} = Float64;
                                 μ₀::Real = 0.7,
                                 mixtures = URBAN_CLASS_MIXTURES,
                                 spec::SpectralGrid = shortwave_bands(T),
                                 lw_spec::SpectralGrid = longwave_bands(T)
                                 ) where {T<:AbstractFloat}
    classes = keys(mixtures)
    α = Vector{T}(undef, length(classes))
    ε = Vector{T}(undef, length(classes))
    for (i, c) in enumerate(classes)
        surf = _class_surface(spec, lw_spec, mixtures[c])
        α[i] = T(broadband_albedo(surf, float(μ₀)))
        ε[i] = T(broadband_emissivity(surf, lw_spec))
    end
    return UrbanMaterials(α, ε; name = [String(c) for c in classes])
end

# Build the `AbstractSurfaceOptics` for one class. `:water` is special:
# it is not a Lambertian reflectance curve but a Fresnel interface, so it
# uses `FresnelWater` to get the zenith dependence right.
function _class_surface(spec::SpectralGrid, lw_spec::SpectralGrid, mix)
    pairs = Pair[]
    for (m, f) in mix
        comp = m === :water ? FresnelWater() : m
        push!(pairs, comp => f)
    end
    return tiled_surface(spec, pairs...; lw_spec = lw_spec)
end

"""
    uniform_materials(α, ε; nclass = 5) -> UrbanMaterials

All classes given the same `α` and `ε`. This is the configuration in which the
effective-albedo signature of radiative trapping is unambiguous: any departure
of the scene's effective albedo from `α` is geometry, not material contrast.
"""
function uniform_materials(α::Real, ε::Real; nclass::Int = 5)
    T = promote_type(typeof(float(α)), typeof(float(ε)))
    return UrbanMaterials(fill(T(α), nclass), fill(T(ε), nclass);
                          name = ["uniform $i" for i in 1:nclass])
end

material_albedo(m::UrbanMaterials, i::Integer) = m.albedo[i]
material_emissivity(m::UrbanMaterials, i::Integer) = m.emissivity[i]

# Plug into the shared Era-III surface-optics contract. The methods are
# defined on the concrete `UrbanMaterials` (never on `AbstractSurfaceOptics`)
# so that Phase 36's spectral implementation cannot be shadowed by them.
RadiationCore.surface_albedo(m::UrbanMaterials, i::Integer) = m.albedo[i]
RadiationCore.surface_emissivity(m::UrbanMaterials, i::Integer) = m.emissivity[i]

Base.length(m::UrbanMaterials) = length(m.albedo)

function Base.show(io::IO, m::UrbanMaterials{T}) where {T}
    print(io, "UrbanMaterials{$T}(", length(m), " classes: ",
          join(m.name, ", "), ")")
end

# =========================================================================
# Facets
# =========================================================================

"""
    Facet{T}

One planar rectangular radiating patch.

  * `centroid`    — centre of the rectangle, metres, in the scene frame
                    `(x east, y north, z up)`
  * `normal`      — **outward** unit normal, i.e. pointing into the air
  * `u`, `v`      — orthonormal in-plane axes with `u × v = normal`
  * `lu`, `lv`    — extents along `u` and `v`, metres
  * `area`        — `lu · lv`
  * `kind`        — `:ground`, `:roof` or `:wall`
  * `material`    — index into a `UrbanMaterials`
  * `temperature` — surface temperature, K (the longwave source)
  * `col`         — the horizontal grid column `(i, j)` this facet's energy is
                    attributed to when aggregating back onto the LES grid

Walls are attributed to the **open** column they face, not to the building
column they belong to. That matches `tag_building_adjacent!` in
`examples/urban/urban_geometry.jl`, which tags the street cell next to a
building as `SURFACE_WALL` precisely so a 2-D surface scheme can carry the
wall's heat load.

The type is immutable; `set_temperature!` rebuilds a facet in place inside a
scene, which is the only field that changes during a simulation.
"""
struct Facet{T<:AbstractFloat}
    centroid::Vec3{T}
    normal::Vec3{T}
    u::Vec3{T}
    v::Vec3{T}
    lu::T
    lv::T
    area::T
    kind::Symbol
    material::Int
    temperature::T
    col::Tuple{Int, Int}
end

function Facet(centroid::NTuple{3, <:Real}, normal::NTuple{3, <:Real},
               u::NTuple{3, <:Real}, v::NTuple{3, <:Real},
               lu::Real, lv::Real, kind::Symbol, material::Integer,
               temperature::Real, col::NTuple{2, Integer})
    T = float(promote_type(eltype(centroid), eltype(normal), eltype(u),
                           eltype(v), typeof(lu), typeof(lv),
                           typeof(temperature)))
    kind in (:ground, :roof, :wall) ||
        error("Facet: kind must be :ground, :roof or :wall, got $kind")
    lu > 0 && lv > 0 || error("Facet: extents must be positive")
    c3(x) = (T(x[1]), T(x[2]), T(x[3]))
    n = c3(normal); uu = c3(u); vv = c3(v)
    # u × v = n is what orients the boundary contour; getting it wrong flips
    # the sign of every exchange area this facet takes part in.
    isapprox(_norm(_sub(_cross(uu, vv), n)), 0; atol = 1e-8) ||
        error("Facet: (u, v, normal) must be a right-handed orthonormal " *
              "frame with u × v = normal")
    return Facet{T}(c3(centroid), n, uu, vv, T(lu), T(lv), T(lu) * T(lv),
                    kind, Int(material), T(temperature),
                    (Int(col[1]), Int(col[2])))
end

"""
    facet_corners(f) -> NTuple{4, NTuple{3, T}}

The four corners, traversed **counter-clockwise when viewed from the front**
(from the `+normal` side). That orientation is required by the double contour
integral: reversing it flips the sign of the exchange area.
"""
@inline function facet_corners(f::Facet{T}) where {T}
    a = f.lu / 2; b = f.lv / 2
    au = _mul(f.u, a); bv = _mul(f.v, b)
    c = f.centroid
    return (_sub(_sub(c, au), bv),
            _sub(_add(c, au), bv),
            _add(_add(c, au), bv),
            _add(_sub(c, au), bv))
end

"""
    facet_edges(f) -> NTuple{4, Tuple{Vec3, Vec3, T}}

The four boundary edges as `(start point, unit direction, length)`, in the
counter-clockwise order of [`facet_corners`](@ref).
"""
@inline function facet_edges(f::Facet{T}) where {T}
    c = facet_corners(f)
    e1 = (c[1], f.u, f.lu)
    e2 = (c[2], f.v, f.lv)
    e3 = (c[3], _mul(f.u, -one(T)), f.lu)
    e4 = (c[4], _mul(f.v, -one(T)), f.lv)
    return (e1, e2, e3, e4)
end

"""
    facet_sample_points(f, nu, nv) -> Vector{NTuple{3,T}}, T

Midpoints of an `nu × nv` regular subdivision of the facet, plus the
sub-patch area. Used for shadow sampling and for the occlusion ratio of
partially blocked facet pairs.
"""
function facet_sample_points(f::Facet{T}, nu::Int, nv::Int) where {T}
    pts = Vector{Vec3{T}}(undef, nu * nv)
    n = 0
    for iu in 1:nu, iv in 1:nv
        su = ((iu - T(0.5)) / nu - T(0.5)) * f.lu
        sv = ((iv - T(0.5)) / nv - T(0.5)) * f.lv
        n += 1
        pts[n] = _add(f.centroid, _add(_mul(f.u, su), _mul(f.v, sv)))
    end
    return pts, f.area / (nu * nv)
end

function Base.show(io::IO, f::Facet{T}) where {T}
    print(io, "Facet{$T}(", f.kind, " @ (",
          round(f.centroid[1]; digits = 2), ", ",
          round(f.centroid[2]; digits = 2), ", ",
          round(f.centroid[3]; digits = 2), "), n = (",
          round(f.normal[1]; digits = 2), ", ",
          round(f.normal[2]; digits = 2), ", ",
          round(f.normal[3]; digits = 2), "), A = ",
          round(f.area; digits = 3), " m², mat ", f.material,
          ", T = ", round(f.temperature; digits = 2), " K)")
end

# =========================================================================
# Scene
# =========================================================================

"""
    RadiosityScene{T}

A facet list plus the heightmap it came from. The heightmap is retained
because it is the *occluder*: every ray test in this module — facet-pair
occlusion, direct-beam shadowing, hemispherical sky sampling — marches
against it with a 2-D digital-differential-analyser traversal, which is exact
for piecewise-constant columns and far cheaper than voxel marching.

  * `facets`      — the radiating patches
  * `heightmap`   — `Nx × Ny`, `heightmap[i,j]` is the solid top of column
                    `(i,j)` in metres above `z = 0`; `0` means open ground
  * `dx`, `dy`    — horizontal cell size, metres
  * `x0`, `y0`    — coordinate of the column-`(1,1)` lower corner
  * `hmax`        — `maximum(heightmap)`; a ray above this can no longer be
                    blocked, which terminates the sky march
  * `materials`   — the optical-property table the facets index into

Lateral boundaries are **open**: the world outside the heightmap is empty, so
a ray leaving the domain reaches the sky. That is the right convention for an
isolated block of city, and it is what makes the energy budget close exactly
(everything that leaves is counted as leaving). For a periodic LES domain,
pad the heightmap with a copy of the opposite edge before building the scene.
"""
struct RadiosityScene{T<:AbstractFloat}
    facets::Vector{Facet{T}}
    heightmap::Matrix{T}
    dx::T
    dy::T
    x0::T
    y0::T
    hmax::T
    materials::UrbanMaterials{T}
end

nfacets(s::RadiosityScene) = length(s.facets)
Base.eltype(::RadiosityScene{T}) where {T} = T
Base.length(s::RadiosityScene) = length(s.facets)

facet_albedos(s::RadiosityScene{T}) where {T} =
    T[s.materials.albedo[f.material] for f in s.facets]
facet_emissivities(s::RadiosityScene{T}) where {T} =
    T[s.materials.emissivity[f.material] for f in s.facets]
facet_temperatures(s::RadiosityScene{T}) where {T} =
    T[f.temperature for f in s.facets]
facet_areas(s::RadiosityScene{T}) where {T} = T[f.area for f in s.facets]

"""
    set_temperature!(scene, i, T_new)

Replace facet `i`'s temperature. `Facet` is immutable, so this rebuilds the
struct; that is cheap (it is a bitstype) and keeps the facet array densely
packed, which matters for the view-factor assembly loop.
"""
function set_temperature!(s::RadiosityScene{T}, i::Integer, T_new::Real) where {T}
    f = s.facets[i]
    s.facets[i] = Facet{T}(f.centroid, f.normal, f.u, f.v, f.lu, f.lv, f.area,
                           f.kind, f.material, T(T_new), f.col)
    return s
end

"""
    set_temperatures!(scene, Ts)

Set every facet temperature from a vector of length `nfacets(scene)`, or from
a scalar. Call this each surface-scheme step before `solve_longwave`.
"""
function set_temperatures!(s::RadiosityScene, Ts::AbstractVector)
    length(Ts) == nfacets(s) ||
        error("set_temperatures!: got $(length(Ts)) values for " *
              "$(nfacets(s)) facets")
    for i in eachindex(Ts)
        set_temperature!(s, i, Ts[i])
    end
    return s
end

function set_temperatures!(s::RadiosityScene, T_new::Real)
    for i in 1:nfacets(s)
        set_temperature!(s, i, T_new)
    end
    return s
end

function Base.show(io::IO, s::RadiosityScene{T}) where {T}
    ng = count(f -> f.kind === :ground, s.facets)
    nr = count(f -> f.kind === :roof, s.facets)
    nw = count(f -> f.kind === :wall, s.facets)
    print(io, "RadiosityScene{$T}(", nfacets(s), " facets: ", ng, " ground, ",
          nr, " roof, ", nw, " wall; heightmap ", size(s.heightmap, 1), "×",
          size(s.heightmap, 2), " @ ", s.dx, "×", s.dy, " m, ",
          "h_max = ", round(s.hmax; digits = 2), " m)")
end

# -------------------------------------------------------------------------
# Facet decomposition
# -------------------------------------------------------------------------

"""
    build_facets(heightmap, dx, dy; kwargs...) -> Vector{Facet}

Decompose a building heightmap into explicit radiating facets.

`heightmap[i,j]` is the solid top of grid column `(i,j)` in metres, the same
piecewise-constant-column convention used by
`CutCell.cut_cell_geometry_from_heightmap`. A column with `height ≤ 0` is open
ground.

Produces

  * one **ground** facet per open column, at `z = ground_level`, normal `+ẑ`,
    area `dx·dy`;
  * one **roof** facet per built column, at `z = heightmap[i,j]`, normal `+ẑ`;
  * for each built column and each of its four sides, a **wall** rectangle
    spanning the height *step* against that neighbour — from the neighbour's
    height up to this column's height — with the outward normal pointing at
    the neighbour, subdivided into `wall_subdivisions` equal vertical strips.

Using the height *step* rather than the full height is what makes the
decomposition watertight and non-overlapping on stepped geometry: where a tall
building abuts a short one, only the exposed part of the tall building's wall
is a facet, and the short building's roof closes the gap. Outside the domain
the height is taken as `ground_level`, so perimeter walls are fully exposed.

# Keyword arguments
  * `wall_subdivisions = 4` — vertical strips per wall rectangle. Walls carry
    strong vertical gradients in both sky-view factor and direct-beam
    shadowing, so 1 is rarely enough; 4 resolves a canyon of H/W ≈ 2 well.
  * `x0`, `y0`, `ground_level = 0` — scene-frame origin.
  * `ground_material`, `roof_material`, `wall_material` — class indices,
    defaulting to `URBAN_STREET`, `URBAN_ROOF`, `URBAN_WALL`.
  * `class_map` — optional `Nx × Ny` integer array overriding the ground class
    per column (to mark parks and water), used only for open columns.
  * `temperature = 288.15` — initial facet temperature, K.
  * `min_height = 1e-9` — a column is "built" when its height exceeds this.
"""
function build_facets(heightmap::AbstractMatrix{<:Real}, dx::Real, dy::Real;
                      wall_subdivisions::Integer = 4,
                      x0::Real = 0, y0::Real = 0, ground_level::Real = 0,
                      ground_material::Integer = URBAN_STREET,
                      roof_material::Integer = URBAN_ROOF,
                      wall_material::Integer = URBAN_WALL,
                      class_map::Union{Nothing, AbstractMatrix{<:Integer}} = nothing,
                      temperature::Real = 288.15,
                      min_height::Real = 1e-9)
    T = promote_type(Float64, eltype(float.(heightmap)))
    nsub = Int(wall_subdivisions)
    nsub >= 1 || error("build_facets: wall_subdivisions must be ≥ 1")
    dx > 0 && dy > 0 || error("build_facets: dx and dy must be positive")
    nx, ny = size(heightmap)
    class_map === nothing || size(class_map) == (nx, ny) ||
        error("build_facets: class_map must match the heightmap size")

    Tdx = T(dx); Tdy = T(dy); Tx0 = T(x0); Ty0 = T(y0); Tg = T(ground_level)
    Ttemp = T(temperature); hmin = T(min_height)

    ex = (one(T), zero(T), zero(T))
    ey = (zero(T), one(T), zero(T))
    ez = (zero(T), zero(T), one(T))
    mex = _mul(ex, -one(T)); mey = _mul(ey, -one(T))

    @inline hgt(i, j) = (1 <= i <= nx && 1 <= j <= ny) ?
                        max(T(heightmap[i, j]), Tg) : Tg

    facets = Facet{T}[]
    for j in 1:ny, i in 1:nx
        xc = Tx0 + (i - T(0.5)) * Tdx
        yc = Ty0 + (j - T(0.5)) * Tdy
        h  = hgt(i, j)
        built = h > Tg + hmin

        if built
            # Roof: horizontal, normal +z, at the column top.
            push!(facets, Facet((xc, yc, h), ez, ex, ey, Tdx, Tdy,
                                :roof, roof_material, Ttemp, (i, j)))
            # Four wall sides; only the exposed height step is a facet.
            # (neighbour offset, face centre offset, outward normal, u, v,
            #  lu (horizontal extent), attributed column)
            sides = ((( 1,  0), (Tdx / 2, zero(T)),  ex, ey, ez, Tdy, (i + 1, j)),
                     ((-1,  0), (-Tdx / 2, zero(T)), mex, ez, ey, Tdy, (i - 1, j)),
                     (( 0,  1), (zero(T), Tdy / 2),  ey, ez, ex, Tdx, (i, j + 1)),
                     (( 0, -1), (zero(T), -Tdy / 2), mey, ex, ez, Tdx, (i, j - 1)))
            for (off, fc, nrm, uu, vv, lhor, ncol) in sides
                hn = hgt(i + off[1], j + off[2])
                Δ = h - hn
                Δ > hmin || continue
                dz = Δ / nsub
                xf = xc + fc[1]; yf = yc + fc[2]
                # Attribute to the open neighbour column when it exists.
                acol = (1 <= ncol[1] <= nx && 1 <= ncol[2] <= ny) ? ncol : (i, j)
                for k in 1:nsub
                    zc = hn + (k - T(0.5)) * dz
                    # u is the horizontal in-plane axis for +x/-y walls and the
                    # vertical one for -x/+y walls; keep u × v = n by construction.
                    lu, lv = (uu === ez) ? (dz, lhor) : (lhor, dz)
                    push!(facets, Facet((xf, yf, zc), nrm, uu, vv, lu, lv,
                                        :wall, wall_material, Ttemp, acol))
                end
            end
        else
            cls = class_map === nothing ? Int(ground_material) :
                                          Int(class_map[i, j])
            push!(facets, Facet((xc, yc, Tg), ez, ex, ey, Tdx, Tdy,
                                :ground, cls, Ttemp, (i, j)))
        end
    end
    return facets
end

"""
    build_scene(heightmap, dx, dy; materials = default_urban_materials(), kwargs...)

Build a [`RadiosityScene`](@ref): [`build_facets`](@ref) plus the heightmap and
grid metadata the ray tests need. All `build_facets` keywords are accepted.
"""
function build_scene(heightmap::AbstractMatrix{<:Real}, dx::Real, dy::Real;
                     materials::UrbanMaterials = default_urban_materials(),
                     x0::Real = 0, y0::Real = 0, ground_level::Real = 0,
                     kwargs...)
    facets = build_facets(heightmap, dx, dy; x0 = x0, y0 = y0,
                          ground_level = ground_level, kwargs...)
    T = eltype(facets[1].lu)
    hm = Matrix{T}(undef, size(heightmap)...)
    @inbounds for I in eachindex(hm)
        hm[I] = max(T(heightmap[I]), T(ground_level))
    end
    mats = UrbanMaterials{T}(collect(T, materials.albedo),
                             collect(T, materials.emissivity),
                             copy(materials.name))
    maxmat = maximum(f.material for f in facets)
    maxmat <= length(mats) ||
        error("build_scene: a facet references material $maxmat but the " *
              "material table has only $(length(mats)) classes")
    return RadiosityScene{T}(facets, hm, T(dx), T(dy), T(x0), T(y0),
                             maximum(hm), mats)
end

"""
    build_scene(mesh::RegularGrid, heightmap; kwargs...)

Convenience overload taking the horizontal spacing and origin from an LES
mesh. `heightmap` must be `Nx × Ny` on the mesh interior.
"""
function build_scene(mesh::RegularGrid, heightmap::AbstractMatrix{<:Real};
                     kwargs...)
    size(heightmap) == (mesh.Nx, mesh.Ny) ||
        error("build_scene: heightmap is $(size(heightmap)), mesh interior " *
              "is $((mesh.Nx, mesh.Ny))")
    return build_scene(heightmap, mesh.dx, mesh.dy;
                       x0 = mesh.x0, y0 = mesh.y0, ground_level = mesh.z0,
                       kwargs...)
end

"""
    heightmap_from_solid_mask(sm) -> Matrix

Collapse a 3-D `IBM.SolidMask` to the heightmap convention: the top of the
highest solid cell in each interior column. Ducks on `sm.mesh` and `sm.mask`
so it also accepts any object with those fields.

Note the discretisation: a mask resolves heights only to `dz`, so the returned
heightmap is the mask's staircase. Where the true building height is known
(e.g. from `urban_geometry.load_city_geojson`) rasterise it directly instead —
a wall facet built from the real height is exact, whereas one built from the
mask inherits a half-cell bias.
"""
function heightmap_from_solid_mask(sm)
    mesh = sm.mesh
    Nx, Ny, Nz = mesh.Nx, mesh.Ny, mesh.Nz
    Hx, Hy, Hz = mesh.Hx, mesh.Hy, mesh.Hz
    T = typeof(mesh.dz)
    h = zeros(T, Nx, Ny)
    @inbounds for j in 1:Ny, i in 1:Nx
        ktop = 0
        for k in Nz:-1:1
            if sm.mask[Hx + i, Hy + j, Hz + k]
                ktop = k
                break
            end
        end
        h[i, j] = mesh.z0 + ktop * mesh.dz
    end
    return h
end

# =========================================================================
# Ray marching against the heightmap
# =========================================================================
# A 2-D Amanatides-Woo DDA over grid columns. Inside a column the obstruction
# is the plane z = h[i,j], and the ray's z is linear in the march parameter,
# so the column is hit exactly when min(z_enter, z_exit) < h. No stepping
# heuristic, no missed thin obstacles — this is exact for a heightmap, which
# is precisely why the heightmap representation is worth keeping.

"""
    ray_blocked(scene, origin, dir, tmax; ztol = 0) -> Bool

March from `origin` along `dir` (not necessarily unit length) for parameter
`t ∈ (0, tmax]` and report whether the ray passes below the top of any grid
column, i.e. whether it is blocked by a building or by the ground plane.

Returns `false` as soon as the ray leaves the heightmap footprint (the lateral
boundary is open) or rises above `scene.hmax` while travelling upward.

`ztol` guards against a facet blocking itself: the ray must dip `ztol` *below*
a column top to count as blocked.
"""
function ray_blocked(s::RadiosityScene{T}, origin::Vec3{T}, dir::Vec3{T},
                     tmax::Real; ztol::Real = 0) where {T}
    hm = s.heightmap
    nx, ny = size(hm)
    invdx = inv(s.dx); invdy = inv(s.dy)
    # Column coordinates: column (i,j) spans grid coordinate [i-1, i).
    gx = (origin[1] - s.x0) * invdx
    gy = (origin[2] - s.y0) * invdy
    dgx = dir[1] * invdx
    dgy = dir[2] * invdy
    dz  = dir[3]
    z0  = origin[3]
    tz  = T(ztol)
    tmx = T(tmax)

    i = floor(Int, gx) + 1
    j = floor(Int, gy) + 1
    (1 <= i <= nx && 1 <= j <= ny) || return false

    stepi = dgx > 0 ? 1 : (dgx < 0 ? -1 : 0)
    stepj = dgy > 0 ? 1 : (dgy < 0 ? -1 : 0)

    tMaxX = stepi == 0 ? T(Inf) :
            (stepi > 0 ? (T(i) - gx) / dgx : (T(i - 1) - gx) / dgx)
    tMaxY = stepj == 0 ? T(Inf) :
            (stepj > 0 ? (T(j) - gy) / dgy : (T(j - 1) - gy) / dgy)
    tDeltaX = stepi == 0 ? T(Inf) : abs(inv(dgx))
    tDeltaY = stepj == 0 ? T(Inf) : abs(inv(dgy))

    t_enter = zero(T)
    # Safety bound: the DDA cannot visit more cells than the footprint has,
    # twice over (one x- and one y-crossing per cell at worst).
    maxsteps = 2 * (nx + ny) + 4
    @inbounds for _ in 1:maxsteps
        t_exit = min(tMaxX, tMaxY, tmx)
        h = hm[i, j]
        za = z0 + t_enter * dz
        zb = z0 + t_exit * dz
        if min(za, zb) < h - tz
            return true
        end
        t_exit >= tmx && return false
        # An upward ray above the tallest column can never be blocked again.
        dz > 0 && zb > s.hmax && return false
        t_enter = t_exit
        if tMaxX < tMaxY
            i += stepi
            tMaxX += tDeltaX
        else
            j += stepj
            tMaxY += tDeltaY
        end
        (1 <= i <= nx && 1 <= j <= ny) || return false
    end
    return false
end

"""
    segment_blocked(scene, p, q; ztol = 0) -> Bool

Whether the open segment from `p` to `q` is obstructed. The endpoints are
expected to have already been nudged off their facets along the facet normals
(see `_nudge`), so no endpoint exclusion logic is needed.
"""
@inline function segment_blocked(s::RadiosityScene{T}, p::Vec3{T},
                                 q::Vec3{T}; ztol::Real = 0) where {T}
    return ray_blocked(s, p, _sub(q, p), one(T); ztol = ztol)
end

"Offset a point off its facet along the outward normal, to avoid self-hits."
@inline _nudge(p::Vec3{T}, n::Vec3{T}, δ::T) where {T} = _add(p, _mul(n, δ))

"Geometric nudge length for a scene: small compared with any facet."
@inline _nudge_length(s::RadiosityScene{T}) where {T} =
    T(1e-7) * min(s.dx, s.dy)

# =========================================================================
# View factors
# =========================================================================

"""
    ViewFactors{T}

  * `F`      — `F[i,j]`, the fraction of radiation leaving facet `i` (diffusely)
               that arrives at facet `j`. `F[i,i] = 0` for planar facets.
  * `F_sky`  — `F_sky[i] = 1 - Σ_j F[i,j]`, the sky-view factor. It is the
               *closure residual*, deliberately, so that it inherits the
               accuracy of `F` rather than being an independent approximation.
  * `area`   — facet areas, cached because every energy identity needs them.

Two identities hold and are the correctness backbone of the whole phase:

```math
A_i F_{ij} = A_j F_{ji} \\qquad\\text{(reciprocity)}
```
```math
\\sum_j F_{ij} + F_{sky,i} = 1 \\qquad\\text{(closure)}
```

Reciprocity holds to machine precision because the symmetric exchange area is
what is actually computed; closure holds exactly by construction. Together
they make the shortwave and longwave energy budgets close identically — see
[`solve_shortwave`](@ref).
"""
struct ViewFactors{T<:AbstractFloat}
    F::Matrix{T}
    F_sky::Vector{T}
    area::Vector{T}
end

Base.size(vf::ViewFactors) = size(vf.F)
Base.eltype(::ViewFactors{T}) where {T} = T

function Base.show(io::IO, vf::ViewFactors{T}) where {T}
    n = length(vf.F_sky)
    print(io, "ViewFactors{$T}(", n, " facets, mean F_sky = ",
          round(sum(vf.F_sky .* vf.area) / sum(vf.area); digits = 4),
          ", reciprocity err ", reciprocity_error(vf),
          ", closure err ", closure_error(vf), ")")
end

"""
    reciprocity_error(vf) -> T

`max_{ij} |A_i F_ij - A_j F_ji| / max(A_i F_ij)` — the relative violation of
reciprocity. Should be at machine-epsilon level.
"""
function reciprocity_error(vf::ViewFactors{T}) where {T}
    n = length(vf.F_sky)
    worst = zero(T); scale = zero(T)
    @inbounds for j in 1:n, i in 1:n
        a = vf.area[i] * vf.F[i, j]
        b = vf.area[j] * vf.F[j, i]
        worst = max(worst, abs(a - b))
        scale = max(scale, abs(a))
    end
    return scale > 0 ? worst / scale : worst
end

"""
    closure_error(vf) -> T

`max_i |Σ_j F_ij + F_sky_i - 1|`.
"""
function closure_error(vf::ViewFactors{T}) where {T}
    n = length(vf.F_sky)
    worst = zero(T)
    @inbounds for i in 1:n
        s = vf.F_sky[i]
        for j in 1:n
            s += vf.F[i, j]
        end
        worst = max(worst, abs(s - 1))
    end
    return worst
end

"""
    total_sky_exchange_area(vf) -> T

`Σ_i A_i F_sky_i` — the total exchange area between the scene and the sky.

For a scene that is closed from above (every vertical line through the
footprint ends on a facet, which is true of any heightmap scene with ground at
`z = 0`), reciprocity with the sky "facet" forces this to equal the **plan
area** of the footprint, `Nx·Ny·dx·dy`, independent of how much building
surface the scene contains. That is a strong, cheap, global test of the whole
view-factor matrix — it fails immediately if any pair is double-counted or
dropped. Open lateral boundaries add a small positive bias, because facets
near the domain edge can see "sky" through directions that a larger domain
would have blocked.
"""
total_sky_exchange_area(vf::ViewFactors) = sum(vf.F_sky .* vf.area)

# -------------------------------------------------------------------------
# Exchange area of one pair
# -------------------------------------------------------------------------

"""
    _parallel_edge_antiderivative(x, d) -> T

`G(x)` with `G''(x) = ln(x² + d²)`, namely

```math
G(x) = \\tfrac12 (x^2 - d^2)\\ln(x^2+d^2) - \\tfrac32 x^2 + 2 d x \\tan^{-1}(x/d),
```

which degenerates correctly to `2f(x) = x^2\\ln|x| - \\tfrac32 x^2` as `d → 0`.
"""
@inline function _parallel_edge_antiderivative(x::T, d::T) where {T}
    if d <= 0
        x == 0 && return zero(T)
        return x * x * log(abs(x)) - T(1.5) * x * x
    end
    r2 = x * x + d * d
    return T(0.5) * (x * x - d * d) * log(r2) - T(1.5) * x * x +
           2 * d * x * atan(x / d)
end

"""
    _edge_log_integral(P1, e1, L1, P2, e2, L2, ngauss) -> T

```math
\\int_0^{L_1}\\!\\!\\int_0^{L_2} \\ln\\lVert (P_1 + t_1 e_1) - (P_2 + t_2 e_2)\\rVert \\, dt_2\\, dt_1
```

the inner kernel of the double contour integral. Two cases:

  * **Parallel edges** — evaluated in *closed form*, for any perpendicular
    offset `d` including `d = 0`. Two parallel segments have
    `r² = (s₁-s₂)² + d²` with `d` constant, so

    ```math
    \\int_{a_1}^{b_1}\\!\\!\\int_{a_2}^{b_2}\\ln\\sqrt{(s_1-s_2)^2+d^2}\\;ds_2\\,ds_1
      = \\tfrac12\\big[G(b_1-a_2) - G(b_1-b_2) - G(a_1-a_2) + G(a_1-b_2)\\big]
    ```

    with `G` from [`_parallel_edge_antiderivative`](@ref). This case matters
    enormously: it is *collinear* (`d = 0`, logarithmically singular) for a
    street patch and the wall rising out of its edge, and *nearly* collinear
    for closely spaced facades — and Gauss-Legendre is poor on both. In
    grid-aligned heightmap geometry every edge pair that contributes at all is
    parallel (perpendicular pairs have `dr₁·dr₂ = 0` and are discarded by the
    caller), so taking this branch makes the whole contour integral exact to
    round-off.

  * **Everything else** — Gauss-Legendre in both parameters, for the
    non-axis-aligned facets a future extension might introduce.
"""
function _edge_log_integral(P1::Vec3{T}, e1::Vec3{T}, L1::T,
                            P2::Vec3{T}, e2::Vec3{T}, L2::T,
                            ngauss::Int) where {T}
    c = _dot(e1, e2)
    w = _sub(P2, P1)
    if abs(abs(c) - 1) < 1e-10
        s0 = _dot(w, e1)                      # P2's position along the line
        dv = _sub(w, _mul(e1, s0))            # perpendicular offset vector
        d = _norm(dv)
        d <= 1e-12 * max(L1, L2) && (d = zero(T))
        a2, b2 = c > 0 ? (s0, s0 + L2) : (s0 - L2, s0)
        G(x) = _parallel_edge_antiderivative(x, d)
        return T(0.5) * (G(L1 - a2) - G(L1 - b2) - G(-a2) + G(-b2))
    end
    x, wt = _gauss_nodes(ngauss)
    h1 = L1 / 2; h2 = L2 / 2
    acc = zero(T)
    @inbounds for a in 1:ngauss
        t1 = h1 * (1 + x[a])
        p1 = _add(P1, _mul(e1, t1))
        inner = zero(T)
        for b in 1:ngauss
            t2 = h2 * (1 + x[b])
            r = _norm(_sub(_add(P2, _mul(e2, t2)), p1))
            inner += wt[b] * log(r)
        end
        acc += wt[a] * inner * h2
    end
    return acc * h1
end

"""
    view_factor_contour(fi, fj; ngauss = 8) -> T

The exchange area `A_i F_ij` of two **fully mutually visible** planar
rectangles, by the double contour integral (Modest, *Radiative Heat Transfer*
3rd ed. eq. 4.42; originally Mitalas & Stephenson 1966):

```math
A_1 F_{12} = \\frac{1}{2\\pi}\\oint_{C_1}\\oint_{C_2} \\ln r \\; d\\vec{r}_1\\cdot d\\vec{r}_2
```

Exact up to the Gauss truncation on the non-singular edge pairs. The result is
**symmetric in the two arguments by inspection**, which is where machine-
precision reciprocity comes from: the caller computes it once and divides by
`A_i` and `A_j` respectively.

Note that it returns the *exchange area* (m²), not the view factor; divide by
`fi.area` for `F_ij`.
"""
function view_factor_contour(fi::Facet{T}, fj::Facet{T};
                             ngauss::Int = 8) where {T}
    ei = facet_edges(fi)
    ej = facet_edges(fj)
    acc = zero(T)
    @inbounds for a in 1:4
        P1, d1, L1 = ei[a]
        for b in 1:4
            P2, d2, L2 = ej[b]
            c = _dot(d1, d2)
            abs(c) < 1e-12 && continue     # perpendicular edges contribute 0
            acc += c * _edge_log_integral(P1, d1, L1, P2, d2, L2, ngauss)
        end
    end
    return acc / T(2π)
end

# Corner-based back-face cull. Returns false when one facet lies entirely in
# the other's back half-space, in which case the exchange area is exactly zero.
#
# For heightmap geometry this test is *exact*, not conservative: all facets are
# axis-aligned and lie on grid planes, so no facet ever straddles another's
# plane. For general geometry a straddling pair would need splitting; the
# module documents that restriction rather than silently mis-handling it.
@inline function _front_facing(fi::Facet{T}, fj::Facet{T}, tol::T) where {T}
    cj = facet_corners(fj)
    maxd = T(-Inf)
    @inbounds for p in cj
        maxd = max(maxd, _dot(_sub(p, fi.centroid), fi.normal))
    end
    maxd > tol || return false
    ci = facet_corners(fi)
    maxd = T(-Inf)
    @inbounds for p in ci
        maxd = max(maxd, _dot(_sub(p, fj.centroid), fj.normal))
    end
    return maxd > tol
end

# Subdivided area quadrature of the exchange integral, accumulating the
# unoccluded and the occluded sums in one pass. Symmetric in (i, j) by
# construction, so it preserves reciprocity exactly.
function _area_exchange_pair(s::RadiosityScene{T}, fi::Facet{T}, fj::Facet{T},
                             niu::Int, niv::Int, nju::Int, njv::Int,
                             δ::T) where {T}
    pi_, dAi = facet_sample_points(fi, niu, niv)
    pj_, dAj = facet_sample_points(fj, nju, njv)
    ni = fi.normal; nj = fj.normal
    invπ = inv(T(π))
    un = zero(T); oc = zero(T)
    @inbounds for a in eachindex(pi_)
        pa = pi_[a]
        pan = _nudge(pa, ni, δ)
        for b in eachindex(pj_)
            pb = pj_[b]
            d = _sub(pb, pa)
            r2 = _dot(d, d)
            r2 <= 0 && continue
            r = sqrt(r2)
            ci = _dot(d, ni) / r
            ci <= 0 && continue
            cj = -_dot(d, nj) / r
            cj <= 0 && continue
            k = ci * cj * invπ / r2 * dAi * dAj
            un += k
            segment_blocked(s, pan, _nudge(pb, nj, δ)) || (oc += k)
        end
    end
    return un, oc
end

"""
    exchange_area(scene, i, j; kwargs...) -> T

The exchange area `A_i F_ij = A_j F_ji` between two facets of a scene,
including occlusion by intervening buildings.

Algorithm:

  1. **Back-face cull.** If either facet lies entirely behind the other's
     plane, the exchange is exactly zero. This alone removes most pairs in an
     urban scene (all ground-ground, all roof-roof, all ground-roof pairs —
     every co-facing parallel pair) and it removes them *exactly*, not
     approximately.
  2. **Occlusion ratio.** A subdivided area quadrature is accumulated twice in
     one pass: once ignoring obstructions, once with a ray test between every
     sub-patch pair. The subdivision is adapted to the size-to-separation
     ratio, so touching facets get the finest sampling and distant ones a
     single sample.
  3. **Exact unoccluded value.** The fully-visible exchange area comes from the
     contour integral, which is exact; it is then multiplied by the
     occluded/unoccluded ratio from step 2. When nothing is blocked the ratio
     is 1 and the answer is exact; when everything is blocked it is 0. The
     approximation is confined to *partially* shadowed pairs, where it is the
     quadrature's estimate of the blocked fraction weighted by the transfer
     kernel — which is the right weighting, and much better than scaling by a
     bare unweighted visible-sample count.

# Keyword arguments
  * `ngauss = 8` — Gauss nodes per contour edge.
  * `nsub_max = 3` — cap on the per-axis subdivision used for the occlusion
    ratio (so at most `nsub_max^4` rays per pair).
  * `subdiv_density = 1.5` — target sub-patch size relative to the facet
    separation; larger means finer sampling.
"""
function exchange_area(s::RadiosityScene{T}, i::Integer, j::Integer;
                       ngauss::Int = 8, nsub_max::Int = 3,
                       subdiv_density::Real = 1.5) where {T}
    i == j && return zero(T)
    return _exchange_area(s, s.facets[i], s.facets[j], ngauss, nsub_max,
                          T(subdiv_density), _nudge_length(s),
                          T(1e-9) * max(s.dx, s.dy))
end

function _exchange_area(s::RadiosityScene{T}, fi::Facet{T}, fj::Facet{T},
                        ngauss::Int, nsub_max::Int, density::T,
                        δ::T, tol::T) where {T}
    _front_facing(fi, fj, tol) || return zero(T)

    d = _norm(_sub(fj.centroid, fi.centroid))
    d <= 0 && return zero(T)
    nsub(l) = clamp(ceil(Int, density * l / d), 1, nsub_max)
    niu = nsub(fi.lu); niv = nsub(fi.lv)
    nju = nsub(fj.lu); njv = nsub(fj.lv)

    un, oc = _area_exchange_pair(s, fi, fj, niu, niv, nju, njv, δ)
    oc <= 0 && return zero(T)
    X = view_factor_contour(fi, fj; ngauss = ngauss)
    X <= 0 && return zero(T)
    # Fully visible (bitwise-equal sums ⇒ no sample pair was blocked).
    oc == un && return X
    return X * (oc / un)
end

"""
    view_factors(scene; kwargs...) -> ViewFactors

Assemble the full facet-to-facet view-factor matrix and the sky-view factors.

Each unordered pair is visited once; the symmetric exchange area is computed by
[`exchange_area`](@ref) and written into both `F[i,j] = X/A_i` and
`F[j,i] = X/A_j`. Reciprocity is therefore exact to rounding, and the sky-view
factor is the closure residual `1 - Σ_j F[i,j]`.

Cost is `O(n²)` pair tests, but the back-face cull disposes of the majority in
a few dozen flops. A scene of ~300 facets assembles in well under a second.

# Keyword arguments
  * `ngauss`, `nsub_max`, `subdiv_density` — passed to [`exchange_area`](@ref).
  * `clamp_sky = true` — clamp `F_sky` into `[0, 1]`. Quadrature error on a
    facet that sees almost nothing but other facets (the bottom strip of a deep
    canyon wall) can push the residual a few times `1e-4` negative; a negative
    sky-view factor is unphysical and would inject energy. The un-clamped
    residual magnitude is a useful diagnostic — see `closure_error` and the
    `raw_sky` return of [`sky_view_factors_raycast`](@ref) for the independent
    check.
"""
function view_factors(s::RadiosityScene{T};
                      ngauss::Int = 8, nsub_max::Int = 3,
                      subdiv_density::Real = 1.5,
                      clamp_sky::Bool = true) where {T}
    n = nfacets(s)
    F = zeros(T, n, n)
    A = facet_areas(s)
    δ = _nudge_length(s)
    tol = T(1e-9) * max(s.dx, s.dy)
    dens = T(subdiv_density)
    fs = s.facets
    @inbounds for i in 1:n-1
        fi = fs[i]
        for j in i+1:n
            X = _exchange_area(s, fi, fs[j], ngauss, nsub_max, dens, δ, tol)
            X == 0 && continue
            F[i, j] = X / A[i]
            F[j, i] = X / A[j]
        end
    end
    F_sky = Vector{T}(undef, n)
    @inbounds for i in 1:n
        acc = zero(T)
        for j in 1:n
            acc += F[i, j]
        end
        v = one(T) - acc
        F_sky[i] = clamp_sky ? clamp(v, zero(T), one(T)) : v
    end
    return ViewFactors{T}(F, F_sky, A)
end

# -------------------------------------------------------------------------
# Independent sky-view factor: hemispherical ray cast
# -------------------------------------------------------------------------

"""
    sky_view_factor_raycast(scene, i; nθ = 16, nφ = 32) -> T

The sky-view factor of facet `i` by **cosine-weighted hemispherical ray
casting**: `nθ × nφ` stratified directions are drawn about the facet normal
with density `cos θ / π`, each is marched against the heightmap, and the plain
(unweighted) mean of the unobstructed fraction is returned.

Cosine weighting is not cosmetic. The sky-view factor is

```math
F_{sky} = \\frac{1}{\\pi}\\int_{2\\pi} V(\\hat{\\omega})\\cos\\theta \\, d\\omega,
```

so sampling *with* density `cos θ/π` turns it into an unweighted average and
removes the `cos θ` from the estimator — the same variance-reduction trick a
Monte-Carlo renderer uses. Sampling uniformly in `(θ, φ)` and weighting
afterwards, as the Era-I `compute_sky_view_factor` does, converges far more
slowly near the horizon, which is exactly where urban obstruction lives.

This is a genuinely **independent** estimate: it never touches the view-factor
matrix. Comparing it against `ViewFactors.F_sky` (the closure residual) is the
external validation of the whole pair-integration machinery.
"""
function sky_view_factor_raycast(s::RadiosityScene{T}, i::Integer;
                                 nθ::Int = 16, nφ::Int = 32) where {T}
    f = s.facets[i]
    n = f.normal; u = f.u; v = f.v
    δ = _nudge_length(s)
    p = _nudge(f.centroid, n, δ)
    tmax = 4 * (T(size(s.heightmap, 1)) * s.dx + T(size(s.heightmap, 2)) * s.dy
                + s.hmax) + one(T)
    vis = 0
    tot = nθ * nφ
    @inbounds for a in 1:nθ, b in 1:nφ
        ξ1 = (a - T(0.5)) / nθ
        ξ2 = (b - T(0.5)) / nφ
        r = sqrt(ξ1)                  # cosine-weighted (Malley's method)
        φ = 2 * T(π) * ξ2
        w = sqrt(max(zero(T), one(T) - ξ1))
        d = _add(_add(_mul(u, r * cos(φ)), _mul(v, r * sin(φ))), _mul(n, w))
        ray_blocked(s, p, d, tmax) || (vis += 1)
    end
    return T(vis) / T(tot)
end

"""
    sky_view_factors_raycast(scene; nθ = 16, nφ = 32) -> Vector

[`sky_view_factor_raycast`](@ref) for every facet.
"""
function sky_view_factors_raycast(s::RadiosityScene{T}; kwargs...) where {T}
    return T[sky_view_factor_raycast(s, i; kwargs...) for i in 1:nfacets(s)]
end

# =========================================================================
# Analytic view-factor benchmarks
# =========================================================================

"""
    view_factor_parallel_rectangles(a, b, c) -> F

Closed-form view factor between **two identical parallel rectangles**, `a × b`,
directly opposed (coaxial), separated by `c`. Howell, *A Catalog of Radiation
Heat Transfer Configuration Factors* C-11; Modest eq. (4.41):

With `X = a/c`, `Y = b/c`,

```math
F_{12} = \\frac{2}{\\pi X Y}\\Bigg\\{
  \\ln\\sqrt{\\frac{(1+X^2)(1+Y^2)}{1+X^2+Y^2}}
  + X\\sqrt{1+Y^2}\\,\\tan^{-1}\\!\\frac{X}{\\sqrt{1+Y^2}}
  + Y\\sqrt{1+X^2}\\,\\tan^{-1}\\!\\frac{Y}{\\sqrt{1+X^2}}
  - X\\tan^{-1}X - Y\\tan^{-1}Y \\Bigg\\}
```

Sanity value: two unit squares one unit apart give `F = 0.199825`.
"""
function view_factor_parallel_rectangles(a::Real, b::Real, c::Real)
    a > 0 && b > 0 && c > 0 ||
        error("view_factor_parallel_rectangles: all dimensions must be > 0")
    X = a / c; Y = b / c
    x2 = X^2; y2 = Y^2
    t = log(sqrt((1 + x2) * (1 + y2) / (1 + x2 + y2))) +
        X * sqrt(1 + y2) * atan(X / sqrt(1 + y2)) +
        Y * sqrt(1 + x2) * atan(Y / sqrt(1 + x2)) -
        X * atan(X) - Y * atan(Y)
    return 2 * t / (π * X * Y)
end

"""
    view_factor_perpendicular_rectangles(l, w, h) -> F

Closed-form view factor between **two perpendicular rectangles sharing a common
edge**. Surface 1 lies in the `z = 0` plane, spans the common edge of length
`l` and extends a width `w` away from it; surface 2 stands on the same edge and
rises to height `h`. `F` is `F_{1→2}`. Howell C-14; Modest eq. (4.40):

With `W = w/l`, `H = h/l`,

```math
\\begin{aligned}
\\pi W F_{12} =\\;& W\\tan^{-1}\\frac{1}{W} + H\\tan^{-1}\\frac{1}{H}
   - \\sqrt{H^2+W^2}\\,\\tan^{-1}\\frac{1}{\\sqrt{H^2+W^2}} \\\\
 &+ \\tfrac14\\ln\\Bigg[\\frac{(1+W^2)(1+H^2)}{1+W^2+H^2}
   \\left(\\frac{W^2(1+W^2+H^2)}{(1+W^2)(W^2+H^2)}\\right)^{W^2}
   \\left(\\frac{H^2(1+H^2+W^2)}{(1+H^2)(H^2+W^2)}\\right)^{H^2}\\Bigg]
\\end{aligned}
```

Sanity value: two unit squares sharing an edge (two faces of a cube) give
`F = 0.200044`.
"""
function view_factor_perpendicular_rectangles(l::Real, w::Real, h::Real)
    l > 0 && w > 0 && h > 0 ||
        error("view_factor_perpendicular_rectangles: all dimensions must be > 0")
    W = w / l; H = h / l
    W2 = W^2; H2 = H^2
    s = sqrt(H2 + W2)
    t = W * atan(1 / W) + H * atan(1 / H) - s * atan(1 / s)
    lg = log((1 + W2) * (1 + H2) / (1 + W2 + H2)) +
         W2 * log(W2 * (1 + W2 + H2) / ((1 + W2) * (W2 + H2))) +
         H2 * log(H2 * (1 + H2 + W2) / ((1 + H2) * (H2 + W2)))
    return (t + lg / 4) / (π * W)
end

"""
    canyon_sky_view_factor(H_over_W) -> ψ

Floor-**averaged** sky-view factor of an infinitely long street canyon of
height-to-width ratio `H/W`:

```math
\\psi = \\sqrt{(H/W)^2 + 1} - H/W
```

This is the standard urban-canopy result (Oke, *Boundary Layer Climates* 2nd
ed. §8; Johnson & Watson 1984). It follows from the 2-D differential view
factor: a point on the floor at distance `a` from one wall and `b` from the
other sees sky fraction `½(a/\\sqrt{a^2+H^2} + b/\\sqrt{b^2+H^2})`, and
averaging that over `a ∈ [0, W]` gives the expression above.
"""
canyon_sky_view_factor(hw::Real) = sqrt(hw^2 + 1) - hw

"""
    canyon_sky_view_factor(H, W, a1, a2) -> ψ

Sky-view factor of an infinitely long canyon floor averaged over the **strip**
`a ∈ [a1, a2]`, measured from one wall across a canyon of width `W` between
walls of height `H`. With the 2-D differential sky fraction

```math
\\psi(a) = \\tfrac12\\left(\\frac{a}{\\sqrt{a^2+H^2}} + \\frac{W-a}{\\sqrt{(W-a)^2+H^2}}\\right)
```

the strip average integrates in closed form,

```math
\\bar\\psi = \\frac{1}{2(a_2-a_1)}\\Big[\\sqrt{a_2^2+H^2}-\\sqrt{a_1^2+H^2}
 + \\sqrt{(W-a_1)^2+H^2}-\\sqrt{(W-a_2)^2+H^2}\\Big].
```

Setting `(a1, a2) = (0, W)` recovers `sqrt((H/W)²+1) - H/W`. This per-strip form
is the sharp test: it must be matched by **every individual floor facet** of a
long modelled canyon, not merely by their mean, so it catches a facet
decomposition that is right on average but mis-registered in `x`.
"""
function canyon_sky_view_factor(H::Real, W::Real, a1::Real, a2::Real)
    a2 > a1 || error("canyon_sky_view_factor: need a2 > a1")
    s(a) = sqrt(a^2 + H^2)
    return (s(a2) - s(a1) + s(W - a1) - s(W - a2)) / (2 * (a2 - a1))
end

"""
    canyon_centreline_sky_view_factor(H_over_W) -> ψ

Sky-view factor at the **centreline** of an infinitely long canyon floor,

```math
\\psi_c = \\frac{1}{\\sqrt{1 + 4(H/W)^2}},
```

the `a = b = W/2` case of the 2-D differential view factor above.

This is the *maximum* over the canyon floor, and therefore always **larger**
than the floor average `sqrt((H/W)^2+1) - H/W`: the centreline is the point
furthest from both walls, so it is the point that sees the most sky. The
minimum sits hard against a wall, at `1/(2\\sqrt{1+(H/W)^2})` — half the
centreline value in the shallow-canyon limit, because one entire side of the
hemisphere is masonry. That cross-street profile is exactly the structure a
single canopy-average sky-view factor throws away, and having the centreline,
the wall-side value and the average as three independent closed forms is a much
sharper test than the average alone: a facet decomposition that is subtly
mis-registered in `x` reproduces the mean but not the profile.
"""
canyon_centreline_sky_view_factor(hw::Real) = 1 / sqrt(1 + 4 * hw^2)

"""
    canyon_wallside_sky_view_factor(H_over_W) -> ψ

Sky-view factor of the canyon floor immediately against a wall,

```math
\\psi_w = \\frac{1}{2\\sqrt{1+(H/W)^2}},
```

the `a = 0` case. The *minimum* over the floor; see
[`canyon_centreline_sky_view_factor`](@ref).
"""
canyon_wallside_sky_view_factor(hw::Real) = 1 / (2 * sqrt(1 + hw^2))

# =========================================================================
# Solar geometry interface
# =========================================================================

"""
    solar_unit_vector(sun) -> (east, north, up)

Unit vector pointing **toward** the sun in the scene's east-north-up frame.
Accepts

  * an `NTuple{3,<:Real}` or length-3 vector — used as-is after normalisation;
  * any object with `.elevation` and `.azimuth` fields in degrees, azimuth
    measured clockwise from north. That is exactly
    `SolarGeometry.SolarPosition`, and the formula reproduces
    `SolarGeometry.sun_direction_enu` identically.

Duck typing rather than a hard dependency, because `SolarGeometry` (Phase 38)
is included *after* this module in `src/Aeolus.jl` and so cannot be imported
here.
"""
function solar_unit_vector(sun::NTuple{3, <:Real})
    v = (Float64(sun[1]), Float64(sun[2]), Float64(sun[3]))
    n = _norm(v)
    n == 0 && error("solar_unit_vector: zero-length solar direction")
    return _mul(v, inv(n))
end

function solar_unit_vector(sun::AbstractVector{<:Real})
    length(sun) == 3 ||
        error("solar_unit_vector: expected a 3-component direction")
    return solar_unit_vector((sun[1], sun[2], sun[3]))
end

function solar_unit_vector(sun)
    (hasproperty(sun, :elevation) && hasproperty(sun, :azimuth)) ||
        error("solar_unit_vector: argument is neither a 3-vector nor an " *
              "object with .elevation / .azimuth (degrees)")
    hr = deg2rad(Float64(getproperty(sun, :elevation)))
    ar = deg2rad(Float64(getproperty(sun, :azimuth)))
    ch = cos(hr)
    return (ch * sin(ar), ch * cos(ar), sin(hr))
end

"""
    facet_sunlit_fraction(scene, i, sun_dir; samples = 2) -> T

The fraction of facet `i` that sees the solar disc, by ray-casting from a
`samples × samples` grid of sub-patch midpoints toward the sun.

Returns `0` when the sun is at or below the horizon, and `0` when the facet
faces away from the sun (`n̂·ŝ ≤ 0`) — a self-shadowed facet is shadowed
whatever the surrounding buildings do, and that back-face test is what makes
the east and west facades of the same building behave oppositely through the
day.
"""
function facet_sunlit_fraction(s::RadiosityScene{T}, i::Integer,
                               sun_dir::Vec3{T}; samples::Int = 2) where {T}
    sun_dir[3] <= 0 && return zero(T)
    f = s.facets[i]
    _dot(f.normal, sun_dir) <= 0 && return zero(T)
    δ = _nudge_length(s)
    tmax = 4 * (T(size(s.heightmap, 1)) * s.dx + T(size(s.heightmap, 2)) * s.dy
                + s.hmax) + one(T)
    pts, _ = facet_sample_points(f, samples, samples)
    lit = 0
    @inbounds for p in pts
        ray_blocked(s, _nudge(p, f.normal, δ), sun_dir, tmax) || (lit += 1)
    end
    return T(lit) / T(length(pts))
end

"""
    sunlit_fractions(scene, sun; samples = 2) -> Vector

[`facet_sunlit_fraction`](@ref) for every facet. `sun` is anything
[`solar_unit_vector`](@ref) accepts.
"""
function sunlit_fractions(s::RadiosityScene{T}, sun; samples::Int = 2) where {T}
    d0 = solar_unit_vector(sun)
    d = (T(d0[1]), T(d0[2]), T(d0[3]))
    return T[facet_sunlit_fraction(s, i, d; samples = samples)
             for i in 1:nfacets(s)]
end

"""
    direct_beam_irradiance(scene, sun; dni, samples = 2) -> Vector

Direct-beam irradiance on each facet, W m⁻² of *facet* area:

```math
E^{dir}_i = I_{DNI}\\,\\max(0, \\hat{n}_i\\cdot\\hat{s})\\,\\sigma_i
```

with `I_DNI` the **direct normal irradiance** (measured perpendicular to the
beam, e.g. `SolarGeometry.clear_sky_irradiance(...).dni`) and `σ_i` the sunlit
fraction from the ray-cast shadow test.

Summing `A_i E^dir_i` over an opaque closed scene recovers
`I_DNI · μ₀ · (plan area)` exactly, because the facets tile the horizon as seen
from the sun — a useful check that the shadow test is neither double-counting
nor leaking.
"""
function direct_beam_irradiance(s::RadiosityScene{T}, sun; dni::Real,
                                samples::Int = 2) where {T}
    d0 = solar_unit_vector(sun)
    d = (T(d0[1]), T(d0[2]), T(d0[3]))
    E = zeros(T, nfacets(s))
    d[3] <= 0 && return E
    I = T(dni)
    @inbounds for i in 1:nfacets(s)
        f = s.facets[i]
        ci = _dot(f.normal, d)
        ci <= 0 && continue
        σ = facet_sunlit_fraction(s, i, d; samples = samples)
        σ <= 0 && continue
        E[i] = I * ci * σ
    end
    return E
end

"""
    diffuse_sky_irradiance(vf, dhi) -> Vector

Diffuse irradiance from an **isotropic** sky on each facet, `E^dif_i = DHI ·
F_sky,i`, where `DHI` is the diffuse horizontal irradiance.

The identity is exact for an isotropic sky: an unobstructed horizontal surface
under sky radiance `L` receives `πL = DHI`, and a facet with sky-view factor
`F_sky` receives `πL F_sky` by the definition of the view factor. Note this is
where the *closure* property earns its keep — an independently computed
sky-view factor would break the energy budget at this step.
"""
diffuse_sky_irradiance(vf::ViewFactors{T}, dhi::Real) where {T} =
    T(dhi) .* vf.F_sky

# =========================================================================
# Radiosity solvers
# =========================================================================

"""
    solve_radiosity_direct(F, ρ, E) -> B

Dense LU solve of the radiosity system

```math
B_i = E_i + \\rho_i \\sum_j F_{ij} B_j
\\quad\\Longleftrightarrow\\quad
(I - \\mathrm{diag}(\\rho) F)\\,B = E .
```

`ρ` is the facet reflectance (shortwave albedo `α`, or longwave `1 - ε`) and
`E` the primary source (see [`solve_shortwave`](@ref) and
[`solve_longwave`](@ref) for what goes into it).

The system is always non-singular for `ρ < 1`: `diag(ρ)F` is non-negative with
row sums `ρ_i(1 - F_{sky,i}) ≤ max ρ < 1`, so its spectral radius is below one
and the Neumann series `Σ_k (diag(ρ)F)^k E` — physically, the sum over bounce
orders — converges geometrically. That is the mathematical statement of
"radiation eventually gets absorbed".
"""
function solve_radiosity_direct(F::AbstractMatrix{T}, ρ::AbstractVector{T},
                                E::AbstractVector{T}) where {T}
    n = length(E)
    size(F) == (n, n) ||
        error("solve_radiosity_direct: F is $(size(F)), expected ($n, $n)")
    length(ρ) == n ||
        error("solve_radiosity_direct: ρ has $(length(ρ)) entries, expected $n")
    M = Matrix{T}(undef, n, n)
    @inbounds for j in 1:n, i in 1:n
        M[i, j] = (i == j ? one(T) : zero(T)) - ρ[i] * F[i, j]
    end
    return M \ collect(E)
end

"""
    solve_radiosity_iterative(F, ρ, E; method = :gauss_seidel, tol = 1e-12,
                              maxiter = 10_000) -> (B, iters, residual)

Iterative solve of the same system, for scenes too large for a dense `O(n³)`
factorisation.

  * `:gauss_seidel` — sweep `B_i ← E_i + ρ_i Σ_j F_ij B_j` in place, using
    already-updated values within the sweep. Because the iteration matrix is
    non-negative with spectral radius `≤ max ρ`, Gauss-Seidel converges
    monotonically from `B = E` and every iterate is a strict under-estimate:
    after `k` sweeps the solution is bounded below by the first `k` bounce
    orders. Convergence is geometric at rate `≈ max ρ`, so a facade albedo of
    0.2 needs ~18 sweeps for `1e-12`, while a snow-covered `α = 0.8` scene
    needs ~125.

  * `:progressive` — **progressive refinement** (Cohen, Chen, Wallace &
    Greenberg 1988): repeatedly pick the facet holding the largest unshot
    energy `A_i ΔB_i` and distribute it to all others through the column of the
    exchange-area matrix. Each shot costs `O(n)` rather than `O(n²)`, and
    because the largest contributor is always chosen first, the *visually and
    energetically* important transport happens in the first few shots. This is
    the method of choice when `n` is large and a few significant digits suffice.

Returns the radiosity, the iteration/shot count, and the final residual
`‖B - E - diag(ρ)FB‖_∞`.
"""
function solve_radiosity_iterative(F::AbstractMatrix{T}, ρ::AbstractVector{T},
                                   E::AbstractVector{T};
                                   method::Symbol = :gauss_seidel,
                                   tol::Real = 1e-12,
                                   maxiter::Int = 10_000) where {T}
    n = length(E)
    size(F) == (n, n) ||
        error("solve_radiosity_iterative: F is $(size(F)), expected ($n, $n)")
    if method === :gauss_seidel
        B = collect(E)
        scale = max(maximum(abs, E), eps(T))
        iters = 0
        for it in 1:maxiter
            iters = it
            Δ = zero(T)
            @inbounds for i in 1:n
                acc = zero(T)
                for j in 1:n
                    acc += F[i, j] * B[j]
                end
                new = E[i] + ρ[i] * acc
                Δ = max(Δ, abs(new - B[i]))
                B[i] = new
            end
            Δ <= tol * scale && break
        end
        return B, iters, _radiosity_residual(F, ρ, E, B)
    elseif method === :progressive
        return _progressive_refinement(F, ρ, E; tol = tol, maxiter = maxiter)
    else
        error("solve_radiosity_iterative: unknown method $method " *
              "(:gauss_seidel or :progressive)")
    end
end

function _radiosity_residual(F::AbstractMatrix{T}, ρ::AbstractVector{T},
                             E::AbstractVector{T}, B::AbstractVector{T}) where {T}
    n = length(E)
    worst = zero(T)
    @inbounds for i in 1:n
        acc = zero(T)
        for j in 1:n
            acc += F[i, j] * B[j]
        end
        worst = max(worst, abs(B[i] - E[i] - ρ[i] * acc))
    end
    return worst
end

# Progressive refinement. `unshot` is the radiosity a facet has received but
# not yet redistributed; shooting facet j adds ρ_i F_ij ΔB_j to every facet i.
function _progressive_refinement(F::AbstractMatrix{T}, ρ::AbstractVector{T},
                                 E::AbstractVector{T};
                                 tol::Real, maxiter::Int) where {T}
    n = length(E)
    B = collect(E)
    unshot = collect(E)
    # Areas are not available here, so energy is ranked by ΔB alone; with
    # equal-area facets these coincide, and the ordering only affects the
    # convergence path, never the fixed point.
    total = max(sum(abs, E), eps(T))
    shots = 0
    for it in 1:maxiter
        shots = it
        k = 1; best = abs(unshot[1])
        @inbounds for i in 2:n
            a = abs(unshot[i])
            a > best && (best = a; k = i)
        end
        best <= tol * total && break
        Δ = unshot[k]
        unshot[k] = zero(T)
        @inbounds for i in 1:n
            F[i, k] == 0 && continue
            inc = ρ[i] * F[i, k] * Δ
            B[i] += inc
            unshot[i] += inc
        end
    end
    return B, shots, _radiosity_residual(F, ρ, E, B)
end

# -------------------------------------------------------------------------
# Shortwave
# -------------------------------------------------------------------------

"""
    ShortwaveSolution{T}

Result of [`solve_shortwave`](@ref), all per facet and in W m⁻² of facet area
unless noted:

  * `radiosity`     — `B_i`, the diffusely reflected flux leaving facet `i`
  * `incident`      — total shortwave arriving: direct + sky diffuse + all
                      inter-reflections
  * `absorbed`      — `(1-α_i)·incident_i`, the surface-energy-balance term
  * `direct`        — the direct-beam component of `incident`
  * `diffuse_sky`   — the sky-diffuse component of `incident`
  * `interreflected`— `Σ_j F_ij B_j`, the part that came off other facets;
                      this is the term the Era-I scheme omitted entirely
  * `total_incident`, `total_absorbed`, `total_escaping` — scene totals in W
  * `effective_albedo` — `total_escaping / total_incident`
"""
struct ShortwaveSolution{T<:AbstractFloat}
    radiosity::Vector{T}
    incident::Vector{T}
    absorbed::Vector{T}
    direct::Vector{T}
    diffuse_sky::Vector{T}
    interreflected::Vector{T}
    total_incident::T
    total_absorbed::T
    total_escaping::T
    effective_albedo::T
end

function Base.show(io::IO, r::ShortwaveSolution{T}) where {T}
    print(io, "ShortwaveSolution{$T}(", length(r.radiosity), " facets, ",
          "incident ", round(r.total_incident; digits = 2), " W, ",
          "absorbed ", round(r.total_absorbed; digits = 2), " W, ",
          "escaping ", round(r.total_escaping; digits = 2), " W, ",
          "α_eff = ", round(r.effective_albedo; digits = 5), ")")
end

"""
    solve_shortwave(scene, vf, sun; dni, dhi, solver = :direct, samples = 2,
                    tol = 1e-12) -> ShortwaveSolution

Solve the shortwave radiosity system for a scene.

The external (uncollided) irradiance on facet `i` is

```math
Q^{ext}_i = I_{DNI}\\max(0,\\hat n_i\\cdot\\hat s)\\,\\sigma_i + \\mathrm{DHI}\\,F_{sky,i},
```

the primary source is `E_i = α_i Q^ext_i` — the *first* reflection — and the
system solved is `B = E + diag(α) F B`. Total incident is then
`Q_i = Q^ext_i + Σ_j F_ij B_j` and the absorbed flux is `(1-α_i) Q_i`.

**Energy conservation is an identity here, not an approximation.** With
`B_i = α_i Q_i`,

```math
\\sum_i A_i Q^{ext}_i \\;=\\; \\sum_i A_i (1-\\alpha_i) Q_i \\;+\\; \\sum_i A_i B_i F_{sky,i},
```

which follows from reciprocity (`Σ_i A_i F_ij = A_j(1 - F_sky,j)`) and closure
alone. So the shortwave budget closing to round-off is a *test of the view
factors*, and its failure would localise the bug immediately.

`solver` is `:direct` (dense LU), `:gauss_seidel`, or `:progressive`.
"""
function solve_shortwave(s::RadiosityScene{T}, vf::ViewFactors{T}, sun;
                         dni::Real, dhi::Real,
                         solver::Symbol = :direct,
                         samples::Int = 2,
                         tol::Real = 1e-12,
                         maxiter::Int = 10_000) where {T}
    n = nfacets(s)
    length(vf.F_sky) == n ||
        error("solve_shortwave: view factors have $(length(vf.F_sky)) " *
              "facets, scene has $n")
    α = facet_albedos(s)
    A = vf.area

    Edir = direct_beam_irradiance(s, sun; dni = dni, samples = samples)
    Edif = diffuse_sky_irradiance(vf, dhi)
    Qext = Edir .+ Edif
    E = α .* Qext

    B = if solver === :direct
        solve_radiosity_direct(vf.F, α, E)
    else
        first(solve_radiosity_iterative(vf.F, α, E; method = solver,
                                        tol = tol, maxiter = maxiter))
    end

    inter = vf.F * B
    Q = Qext .+ inter
    absorbed = (one(T) .- α) .* Q

    tot_in  = sum(A .* Qext)
    tot_abs = sum(A .* absorbed)
    tot_esc = sum(A .* B .* vf.F_sky)
    αeff = tot_in > 0 ? tot_esc / tot_in : zero(T)

    return ShortwaveSolution{T}(B, Q, absorbed, Edir, Edif, inter,
                                tot_in, tot_abs, tot_esc, αeff)
end

"""
    effective_albedo(scene, vf, sun; dni = 900, dhi = 100, kwargs...) -> T

The **effective albedo** of the scene: the fraction of the shortwave energy
entering it that eventually escapes back to the sky, after every bounce.

```math
\\alpha_{eff} = \\frac{\\sum_i A_i B_i F_{sky,i}}{\\sum_i A_i Q^{ext}_i}
```

For a flat surface this is exactly the material albedo `α`. For any scene with
relief it is **strictly smaller**, and it decreases monotonically as the canyon
deepens: each reflection off a facade is aimed mostly at another facade, so the
photon gets more chances to be absorbed before it finds the sky. The
approximate canopy-scale scaling is `α_eff ≈ α·ψ_sky/(1 - α(1-ψ_sky))`, the
geometric series of the bounces; the full solve here resolves the facet-level
structure that the single-ψ approximation smears out.

This ratio is the quantitative signature of urban radiative trapping and the
headline validation of this phase. Reported by `ShortwaveSolution` too.

**Caveat on scene edges.** The denominator counts what actually entered the
scene, so the budget closes exactly whatever the geometry. But with open
lateral boundaries a scene's *perimeter* walls face empty space: they see
nothing but sky, reflect straight back to it, and so behave like a flat
surface of albedo `α`, diluting the canyon signal. Their area grows with
building height, which can make `α_eff` non-monotone in `H/W` for a small test
domain under an oblique sun. Two ways to avoid the artefact: illuminate from
the zenith with no diffuse component, which leaves vertical perimeter walls
receiving exactly nothing; or pad the heightmap so the region of interest is
interior. The zenith configuration is the controlled experiment used in the
test suite.

The two-argument convenience form `effective_albedo(scene, sun; ...)` assembles
the view factors itself — do not use it in a loop, since assembly is the
expensive part.
"""
function effective_albedo(s::RadiosityScene{T}, vf::ViewFactors{T}, sun;
                          dni::Real = 900, dhi::Real = 100, kwargs...) where {T}
    return solve_shortwave(s, vf, sun; dni = dni, dhi = dhi,
                           kwargs...).effective_albedo
end

function effective_albedo(s::RadiosityScene, sun; vf_kwargs = NamedTuple(),
                          kwargs...)
    return effective_albedo(s, view_factors(s; vf_kwargs...), sun; kwargs...)
end

# -------------------------------------------------------------------------
# Longwave
# -------------------------------------------------------------------------

"""
    LongwaveSolution{T}

Result of [`solve_longwave`](@ref), per facet, W m⁻² of facet area:

  * `radiosity`  — `B_i`, emitted plus reflected longwave leaving the facet
  * `incident`   — `Q_i`, longwave arriving from other facets and from the sky
  * `emitted`    — `ε_i σ T_i⁴`
  * `absorbed`   — `ε_i Q_i`
  * `net_loss`   — `ε_i(σT_i⁴ - Q_i) = B_i - Q_i`, positive when the facet is
                   cooling radiatively. This is the term the surface energy
                   balance needs.
  * `from_sky`   — `F_sky,i · L_sky`, the sky's share of `incident`
  * totals in W, and `escaping_to_sky` = `Σ A_i B_i F_sky,i`
"""
struct LongwaveSolution{T<:AbstractFloat}
    radiosity::Vector{T}
    incident::Vector{T}
    emitted::Vector{T}
    absorbed::Vector{T}
    net_loss::Vector{T}
    from_sky::Vector{T}
    total_emitted::T
    total_absorbed::T
    total_from_sky::T
    total_escaping::T
end

function Base.show(io::IO, r::LongwaveSolution{T}) where {T}
    print(io, "LongwaveSolution{$T}(", length(r.radiosity), " facets, ",
          "emitted ", round(r.total_emitted; digits = 2), " W, ",
          "absorbed ", round(r.total_absorbed; digits = 2), " W, ",
          "sky in ", round(r.total_from_sky; digits = 2), " W, ",
          "sky out ", round(r.total_escaping; digits = 2), " W)")
end

"""
    solve_longwave(scene, vf; L_sky, solver = :direct, tol = 1e-12) -> LongwaveSolution

Solve the longwave radiosity system

```math
B_i = \\varepsilon_i\\sigma T_i^4 + (1-\\varepsilon_i)\\Big(\\sum_j F_{ij}B_j + F_{sky,i} L_{sky}\\Big),
```

where `L_sky` is the downwelling sky longwave irradiance on a horizontal
unobstructed surface (W m⁻²) and `σ` is `RadiationCore.STEFAN_BOLTZMANN`. The
facet temperatures come from `scene.facets[i].temperature`; set them with
[`set_temperatures!`](@ref) before each call.

The primary source passed to the solver is
`E_i = ε_i σ T_i⁴ + (1-ε_i) F_sky,i L_sky` and the reflectance is `1 - ε_i`
(Kirchhoff).

**Canyon trapping.** The physically important output is `net_loss`. A facet on
an exposed roof loses `ε(σT⁴ - L_sky)`, typically 80-120 W m⁻² on a clear
night. The same material at the same temperature on a canyon floor exchanges
most of its emission with walls that are almost as warm, so its net loss
collapses — by a factor of 3 or more for `H/W = 2`. See
[`longwave_trapping_factor`](@ref) for the diagnostic.

Energy closes as an identity, again from reciprocity and closure:
`absorbed = emitted + sky_in - sky_out`.
"""
function solve_longwave(s::RadiosityScene{T}, vf::ViewFactors{T};
                        L_sky::Real, solver::Symbol = :direct,
                        tol::Real = 1e-12, maxiter::Int = 10_000) where {T}
    n = nfacets(s)
    length(vf.F_sky) == n ||
        error("solve_longwave: view factors have $(length(vf.F_sky)) " *
              "facets, scene has $n")
    ε = facet_emissivities(s)
    ρ = one(T) .- ε
    A = vf.area
    σ = T(STEFAN_BOLTZMANN)
    Ls = T(L_sky)

    emitted = T[ε[i] * σ * s.facets[i].temperature^4 for i in 1:n]
    from_sky = vf.F_sky .* Ls
    E = emitted .+ ρ .* from_sky

    B = if solver === :direct
        solve_radiosity_direct(vf.F, ρ, E)
    else
        first(solve_radiosity_iterative(vf.F, ρ, E; method = solver,
                                        tol = tol, maxiter = maxiter))
    end

    Q = vf.F * B .+ from_sky
    absorbed = ε .* Q
    net = B .- Q

    return LongwaveSolution{T}(B, Q, emitted, absorbed, net, from_sky,
                               sum(A .* emitted), sum(A .* absorbed),
                               sum(A .* from_sky), sum(A .* B .* vf.F_sky))
end

"""
    longwave_trapping_factor(scene, vf, lw; kinds = (:ground,)) -> T

The ratio

```math
f = \\frac{\\text{net longwave loss from the selected facets}}
          {\\text{loss the same facets would have with } F_{sky}=1}
```

i.e. actual cooling divided by the cooling of an exposed flat surface of the
same material at the same temperature under the same sky. `f = 1` for an open
plain; `f → 0` for a facet buried in an isothermal cavity.

This is the cleanest single number for canyon longwave trapping: it isolates
geometry, because the material, the temperature and the sky are held fixed
between numerator and denominator.
"""
function longwave_trapping_factor(s::RadiosityScene{T}, vf::ViewFactors{T},
                                  lw::LongwaveSolution{T};
                                  L_sky::Real,
                                  kinds = (:ground,)) where {T}
    σ = T(STEFAN_BOLTZMANN)
    ε = facet_emissivities(s)
    num = zero(T); den = zero(T)
    @inbounds for i in 1:nfacets(s)
        f = s.facets[i]
        f.kind in kinds || continue
        num += vf.area[i] * lw.net_loss[i]
        den += vf.area[i] * ε[i] * (σ * f.temperature^4 - T(L_sky))
    end
    return den == 0 ? T(NaN) : num / den
end

"""
    LongwaveResponse{T}

The longwave radiosity system written as a **linear map** from the facet
emission to the net longwave each facet absorbs. Returned by
[`longwave_response`](@ref); per unit facet area,

```math
N_i \\;=\\; \\varepsilon_i Q_i - \\varepsilon_i\\sigma T_i^4
     \\;=\\; \\sum_j J_{ij}\\,E_j + j^{sky}_i\\,L_{sky},
\\qquad E_j = \\varepsilon_j \\sigma T_j^4 .
```

  * `dN_dE`      — `J`, `n × n`: the change in facet `i`'s net longwave gain
                   per W m⁻² of extra emission by facet `j`
  * `dN_dLsky`   — `j^{sky}`, length `n`: the same for the sky irradiance
  * `emissivity` — the facet emissivities it was built with; `J` depends on
                   geometry and emissivity only, **not on temperature**, so it
                   is valid until the emissivities change

`N = -LongwaveSolution.net_loss` for any temperatures and sky.
"""
struct LongwaveResponse{T<:AbstractFloat}
    dN_dE::Matrix{T}
    dN_dLsky::Vector{T}
    emissivity::Vector{T}
end

"""
    longwave_response(scene, vf) -> LongwaveResponse

The Jacobian of the longwave radiosity solution with respect to the facet
emission and the sky irradiance. Eliminating the radiosity
`B = E + (1-ε)Q` from `Q = F B + F_sky L_sky` gives

```math
Q = (I - F\\,\\mathrm{diag}(1-\\varepsilon))^{-1}\\,(F E + F_{sky} L_{sky}),
```

so the net absorbed longwave `N = diag(ε) Q - E` is **exactly linear** in the
emission `E`; the only nonlinearity of longwave exchange is `E = εσT⁴`. Hence

```math
J = \\mathrm{diag}(\\varepsilon)\\,(I - F\\,\\mathrm{diag}(1-\\varepsilon))^{-1} F - I,
\\qquad
j^{sky} = \\mathrm{diag}(\\varepsilon)\\,(I - F\\,\\mathrm{diag}(1-\\varepsilon))^{-1} F_{sky},
```

one LU factorisation of the same operator [`solve_longwave`](@ref) solves.

This is what an implicit surface scheme needs: with ``∂E_j/∂T_j = 4ε_jσT_j^3``,
``∂N_i/∂T_j = J_{ij}\\,4ε_jσT_j^3`` couples every facet's energy balance to
every other facet's temperature within one time step. Two properties make it
conservative (both are asserted in the test suite):

  * **Reciprocity of total exchange areas** (Hottel): for `i ≠ j`,
    `A_i J_ij ε_j = A_j J_ji ε_i`.
  * **Closure**: `Σ_i A_i J_ij = -A_j × (fraction of j's emission that
    escapes to the sky)`; in an enclosure with no sky the area-weighted
    column sums vanish, so extra emission by one facet is absorbed, in full,
    by the others.
"""
function longwave_response(s::RadiosityScene{T}, vf::ViewFactors{T}) where {T}
    n = nfacets(s)
    length(vf.F_sky) == n ||
        error("longwave_response: view factors have $(length(vf.F_sky)) " *
              "facets, scene has $n")
    ε = facet_emissivities(s)
    M = Matrix{T}(undef, n, n)
    @inbounds for j in 1:n, i in 1:n
        M[i, j] = (i == j ? one(T) : zero(T)) - vf.F[i, j] * (one(T) - ε[j])
    end
    X = lu!(M) \ hcat(vf.F, vf.F_sky)
    J = ε .* X[:, 1:n]
    @inbounds for i in 1:n
        J[i, i] -= one(T)
    end
    return LongwaveResponse{T}(J, ε .* X[:, n+1], ε)
end

# =========================================================================
# Coupling back to the LES surface scheme
# =========================================================================

"""
    UrbanRadiationForcing{T}

Facet-level radiation aggregated onto the `(Nx, Ny)` LES horizontal grid, in
the form `examples/urban/urban_physics.jl`'s `step_surfaces!` consumes.

All flux arrays are **per unit plan area** — facet fluxes are multiplied by
facet area and divided by `dx·dy`. That is the correct normalisation for a
column-integrated urban surface energy balance: a cell containing a tall wall
legitimately absorbs more than `1000 W m⁻²` of plan area, and hiding that by
normalising per facet area would silently discard the wall's heat load, which
is the very thing this phase exists to capture.

  * `sw_absorbed`  — absorbed shortwave, W per m² of plan area
  * `lw_net`       — net longwave **loss**, W per m² of plan area (positive =
                     cooling)
  * `area_density` — `Σ A_facet / (dx·dy)` per column; `1` for flat ground,
                     larger where walls stand. The "urban rugosity" factor.
  * `svf`          — facet-area-weighted sky-view factor per column
  * `sw_effective` — the value to pass as `sw_in_arr[i,j]` to `step_surfaces!`
                     so that its `(1-α)·sw_in` reproduces `sw_absorbed`
                     exactly, given the column's class albedo
  * `lw_down_effective` — the value to pass as `lw_in[i,j]` so that
                     `ε·lw_in - ε σ T⁴` reproduces `-lw_net`
  * `class` — the material class used for each column when forming the two
              effective fields

The two `*_effective` fields exist so this module can be dropped into the
existing surface scheme without rewriting it: they invert that scheme's own
algebra so the *energy* is right even though the scheme still thinks it is
doing a flat-plate balance.
"""
struct UrbanRadiationForcing{T<:AbstractFloat}
    sw_absorbed::Matrix{T}
    lw_net::Matrix{T}
    area_density::Matrix{T}
    svf::Matrix{T}
    sw_effective::Matrix{T}
    lw_down_effective::Matrix{T}
    class::Matrix{Int}
end

function Base.show(io::IO, f::UrbanRadiationForcing{T}) where {T}
    print(io, "UrbanRadiationForcing{$T}(", size(f.sw_absorbed, 1), "×",
          size(f.sw_absorbed, 2), " columns, ",
          "SW_abs ", round(sum(f.sw_absorbed) / length(f.sw_absorbed); digits = 2),
          " W/m² mean, LW_net ",
          round(sum(f.lw_net) / length(f.lw_net); digits = 2), " W/m² mean)")
end

"""
    aggregate_to_grid(scene, vf; sw = nothing, lw = nothing,
                      T_surface = nothing) -> UrbanRadiationForcing

Project the facet-level solution back onto the LES horizontal grid.

Each facet's flux is weighted by its area and deposited in the column given by
`Facet.col` — the open column a wall faces, the column itself for ground and
roof. The result is divided by the cell plan area `dx·dy`.

Either solution may be omitted (`nothing`), in which case the corresponding
arrays are zero. `T_surface` is the temperature used when inverting the
longwave algebra for `lw_down_effective`; it defaults to the area-weighted mean
facet temperature in each column, which is the consistent choice when the
caller's surface scheme carries one temperature per column.
"""
function aggregate_to_grid(s::RadiosityScene{T}, vf::ViewFactors{T};
                           sw::Union{Nothing, ShortwaveSolution{T}} = nothing,
                           lw::Union{Nothing, LongwaveSolution{T}} = nothing,
                           T_surface::Union{Nothing, AbstractMatrix} = nothing) where {T}
    nx, ny = size(s.heightmap)
    cell = s.dx * s.dy
    swa = zeros(T, nx, ny)
    lwn = zeros(T, nx, ny)
    aden = zeros(T, nx, ny)
    svf = zeros(T, nx, ny)
    Tw = zeros(T, nx, ny)
    cls = fill(URBAN_STREET, nx, ny)
    # Per column, the dominant class is the one holding the most facet area —
    # for a built column that is the roof, for a canyon column the ground or
    # the wall, whichever carries more surface.
    areaper = Dict{Tuple{Int, Int, Int}, T}()

    @inbounds for i in 1:nfacets(s)
        f = s.facets[i]
        ci, cj = f.col
        (1 <= ci <= nx && 1 <= cj <= ny) || continue
        a = f.area
        sw === nothing || (swa[ci, cj] += a * sw.absorbed[i])
        lw === nothing || (lwn[ci, cj] += a * lw.net_loss[i])
        aden[ci, cj] += a
        svf[ci, cj] += a * vf.F_sky[i]
        Tw[ci, cj] += a * f.temperature
        k = (ci, cj, f.material)
        areaper[k] = get(areaper, k, zero(T)) + a
    end

    @inbounds for j in 1:ny, i in 1:nx
        if aden[i, j] > 0
            svf[i, j] /= aden[i, j]
            Tw[i, j] /= aden[i, j]
        else
            svf[i, j] = one(T)
            Tw[i, j] = T(288.15)
        end
        swa[i, j] /= cell
        lwn[i, j] /= cell
        aden[i, j] /= cell
    end
    # Dominant class per column.
    best = Dict{Tuple{Int, Int}, T}()
    for ((i, j, m), a) in areaper
        if a > get(best, (i, j), T(-Inf))
            best[(i, j)] = a
            cls[i, j] = m
        end
    end

    swe = zeros(T, nx, ny)
    lwe = zeros(T, nx, ny)
    σ = T(STEFAN_BOLTZMANN)
    @inbounds for j in 1:ny, i in 1:nx
        m = cls[i, j]
        α = s.materials.albedo[m]
        ε = s.materials.emissivity[m]
        swe[i, j] = swa[i, j] / max(one(T) - α, eps(T))
        Tsfc = T_surface === nothing ? Tw[i, j] : T(T_surface[i, j])
        # step_surfaces! forms ε·lw_in - ε σ T⁴; we want that to equal -lw_net.
        lwe[i, j] = σ * Tsfc^4 - lwn[i, j] / max(ε, eps(T))
    end

    return UrbanRadiationForcing{T}(swa, lwn, aden, svf, swe, lwe, cls)
end

end # module
