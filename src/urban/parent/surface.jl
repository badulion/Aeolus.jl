# The Berlin parent's surface (N-parent-d, #249): a tiled urban mosaic per
# parent column.
#
# Parent cells are 31.25 m wide, so the buildings (mean height ~20 m) are
# not resolved: the parent sees Berlin as a heterogeneous surface. Each
# parent column is an `Aeolus.SurfaceTiles.SurfaceMosaic` (or a
# `BlendedMosaic`) of one tile per surface class present, with the class's
# area fraction:
#
#   * fractions — from `data/berlin_landuse_6km.csv` (the OSM extract of
#     AeolusPrep.jl's `aeolusprep fetch-landuse`: the child's classifier and
#     rasteriser on the child's grid, coarse-grained; cropped and
#     coarsened to the parent grid by `aeolusprep static`, which
#     writes them to the parent's static driver, CFG-f #361), and, under the
#     child's footprint, from coarse-graining the child's own class map,
#     so parent and child agree exactly there;
#   * tiles — built by the case's one class-to-surface factory,
#     `class_surface` (couplings/surface_exchange.jl): soil columns for
#     street, wall and park, a SlabLand for roof and water today, and
#     whatever the park (#140) and water (#141) couplings register later;
#   * unresolved buildings — the built tiles (street, roof, wall and any
#     class above 5, e.g. glass) take the Macdonald et al. (1998)
#     momentum roughness `z0` of the column's building morphometry
#     (`SurfaceTiles.macdonald_roughness`; never less than the class's
#     own z0m; the scalar roughness stays the class's), and exchange
#     with the air at the first level above the mean roof height, at a
#     height measured from the displacement `d`; park and water tiles
#     keep their own roughness and d = 0 and exchange with the first
#     level.
#
# Nothing here is coupled to a 3-D run yet: the parent case (N-parent-e,
# #259) builds it and the nest (N-berlin-nest, #265) makes it default.

using ..Aeolus.SurfaceTiles: SurfaceMosaic, BlendedMosaic, AbstractBlendingHeight,
                           class_fractions, macdonald_roughness, mosaic_fractions,
                           last_tile_fluxes

"Classes of the built fabric: their tiles take the unresolved-building roughness and displacement."
is_built_class(c::Integer) = c == SURFACE_STREET || c == SURFACE_ROOF || c == SURFACE_WALL || c > SURFACE_WALL

"""
    ParentLanduse

The parent land-use table, as read by [`read_parent_landuse`](@ref):

* `nx`, `ny`, `dx` — the parent grid (m); `x0`, `y0` — its lower-left
  corner relative to the OSM anchor (m); `r` — child cells per parent
  cell per axis;
* `counts[I, J, c]` — child cells of class `c` (1 street, 2 roof, 3 park,
  4 water, 5 wall) in parent cell `(I, J)`, out of `r²`;
* `λf`, `hmean`, `hmax` — frontal area density, roof-area-weighted mean
  and maximum building height (m) per parent cell;
* `header` — the provenance lines of the file.
"""
struct ParentLanduse
    nx::Int
    ny::Int
    dx::Float64
    x0::Float64
    y0::Float64
    r::Int
    counts::Array{Int,3}
    λf::Matrix{Float64}
    hmean::Matrix{Float64}
    hmax::Matrix{Float64}
    header::Vector{String}
end

read_parent_landuse(::Nothing) =
    error("[geometry] landuse is \"none\": the tiled parent surface needs a land-use table " *
          "(AeolusPrep.jl: `aeolusprep fetch-landuse`) or [geometry] parent_static")

"""
    read_parent_landuse(path) -> ParentLanduse

Read the parent land-use CSV (`I,J,n1..n5,lambda_p,lambda_f,h_mean,h_max`
with `#` provenance lines; the grid is taken from the `# parent grid:`
line). Checks that every cell appears once, that the counts of every
cell add up to `r²` and that `lambda_p = n2/r²`.
"""
function read_parent_landuse(path::AbstractString)
    lines = readlines(path)
    header = [l for l in lines if startswith(l, "#")]
    g = nothing
    for l in header
        m = match(r"nx = (\d+), ny = (\d+), dx = ([\d.]+) m, x0 = y0 = (-?[\d.]+) m.*refinement r = (\d+)", l)
        m === nothing || (g = m)
    end
    g === nothing && error("read_parent_landuse: no '# parent grid:' line in $path")
    nx, ny = parse(Int, g[1]), parse(Int, g[2])
    dx, x0, r = parse(Float64, g[3]), parse(Float64, g[4]), parse(Int, g[5])
    counts = fill(-1, nx, ny, 5)
    λf, hm, hx = zeros(nx, ny), zeros(nx, ny), zeros(nx, ny)
    data = [l for l in lines if !startswith(l, "#") && !isempty(strip(l))]
    startswith(data[1], "I,J,") || error("read_parent_landuse: missing column header")
    for l in data[2:end]
        v = split(l, ',')
        I, J = parse(Int, v[1]), parse(Int, v[2])
        counts[I, J, 1] == -1 || error("read_parent_landuse: cell ($I, $J) twice")
        n = parse.(Int, v[3:7])
        sum(n) == r^2 || error("read_parent_landuse: cell ($I, $J) counts $(sum(n)) ≠ $(r^2)")
        parse(Float64, v[8]) == n[2] / r^2 || error("read_parent_landuse: cell ($I, $J) λ_p ≠ n2/r²")
        counts[I, J, :] .= n
        λf[I, J], hm[I, J], hx[I, J] = parse(Float64, v[9]), parse(Float64, v[10]), parse(Float64, v[11])
    end
    any(==(-1), counts) && error("read_parent_landuse: cells missing in $path")
    return ParentLanduse(nx, ny, dx, x0, x0, r, counts, λf, hm, hx, header)
end

"""
    child_footprint(pl, cfg) -> (Is, Js)

The parent cells covered by the child domain of `cfg` (centred
`cfg.offset` m from the anchor). Errors unless the child grid nests:
``Δx_{parent} = r Δx_{child}``, the child's edges on parent cell faces.
"""
function child_footprint(pl::ParentLanduse, cfg::CaseConfig)
    dxc = cfg.extent[1] / cfg.size[1]
    (cfg.extent[1] / cfg.size[1] == cfg.extent[2] / cfg.size[2] && pl.dx == pl.r * dxc) ||
        error("child_footprint: parent Δx $(pl.dx) m ≠ r·Δx_child = $(pl.r) × $dxc m")
    nI, nJ = cfg.size[1] ÷ pl.r, cfg.size[2] ÷ pl.r
    (nI * pl.r == cfg.size[1] && nJ * pl.r == cfg.size[2]) ||
        error("child_footprint: child size $(cfg.size[1:2]) not divisible by r = $(pl.r)")
    fi = (cfg.offset[1] - cfg.extent[1] / 2 - pl.x0) / pl.dx
    fj = (cfg.offset[2] - cfg.extent[2] / 2 - pl.y0) / pl.dx
    (isinteger(fi) && isinteger(fj)) ||
        error("child_footprint: the child's edges are not on parent cell faces ($fi, $fj)")
    I0, J0 = Int(fi) + 1, Int(fj) + 1
    (I0 >= 1 && J0 >= 1 && I0 + nI - 1 <= pl.nx && J0 + nJ - 1 <= pl.ny) ||
        error("child_footprint: the child domain leaves the parent box")
    return I0:I0+nI-1, J0:J0+nJ-1
end

"""
    parent_fractions(pl; child_sfc = nothing, footprint = nothing, nclass = 5)
        -> Array{Float64,3}

Area fractions `f[I, J, c]` of the surface classes `1:nclass` in every
parent cell: `counts / r²` from the land-use table and, inside
`footprint = (Is, Js)` (see [`child_footprint`](@ref)), the class
fractions of the `r × r` block of the child's own class map `child_sfc`
(`SurfaceTiles.class_fractions`), so that the parent sees exactly the
child's surface there (including classes the child adds during its
build, e.g. glass, 6). `nclass` is raised to the largest class of
`child_sfc`.
"""
function parent_fractions(pl::ParentLanduse; child_sfc = nothing, footprint = nothing,
                          nclass::Integer = 5)
    child_sfc === nothing || (nclass = max(nclass, Int(maximum(child_sfc))))
    f = zeros(pl.nx, pl.ny, nclass)
    f[:, :, 1:5] .= pl.counts ./ pl.r^2
    if child_sfc !== nothing
        footprint === nothing && error("parent_fractions: child_sfc needs its footprint")
        Is, Js = footprint
        size(child_sfc) == (length(Is) * pl.r, length(Js) * pl.r) ||
            error("parent_fractions: child map $(size(child_sfc)) does not fill the footprint")
        r = pl.r
        for (b, J) in enumerate(Js), (a, I) in enumerate(Is)
            blk = view(child_sfc, (a - 1) * r + 1:a * r, (b - 1) * r + 1:b * r)
            f[I, J, :] .= class_fractions(blk, nclass)
        end
    end
    return f
end

"""
    parent_morphometry(pl, frac; window = 8) -> (λp, λf, h)

The building morphometry each parent column's roughness is computed
from: plan area density `λp` (roof fraction of `frac`), frontal area
density `λf` and roof-area-weighted mean height `h` (m), averaged over a
`window × window` block of parent cells centred on the column (truncated
at the edge of the box). Macdonald et al. (1998) describe arrays of many
obstacles; one 31.25 m cell holds at most a few buildings (its own λp is
0, 1/4, …, 1), so the morphometry is taken over a neighbourhood of
several street blocks — 250 m by default, the source-area scale of the
morphometric methods reviewed by Grimmond & Oke (1999).
"""
function parent_morphometry(pl::ParentLanduse, frac::AbstractArray{<:Real,3}; window::Integer = 8)
    window >= 1 || error("parent_morphometry: window must be ≥ 1")
    λp1 = frac[:, :, SURFACE_ROOF]
    nx, ny = pl.nx, pl.ny
    lo, hi = (window - 1) ÷ 2, window ÷ 2
    λp, λf, h = zeros(nx, ny), zeros(nx, ny), zeros(nx, ny)
    for J in 1:ny, I in 1:nx
        Is, Js = max(1, I - lo):min(nx, I + hi), max(1, J - lo):min(ny, J + hi)
        n = length(Is) * length(Js)
        sp = sum(view(λp1, Is, Js))
        λp[I, J] = sp / n
        λf[I, J] = sum(view(pl.λf, Is, Js)) / n
        h[I, J] = sp > 0 ? sum(λp1[i, j] * pl.hmean[i, j] for i in Is, j in Js) / sp : 0.0
    end
    return λp, λf, h
end

"""
    parent_forcing_level(z_centres, h) -> k

The parent level whose air state forces a column's surface: the lowest
level whose centre lies at or above the mean building height `h`. Below
the roofs the air is inside the unresolved canopy, where the
Monin–Obukhov profile ``\\ln((z - d)/z_0)`` of the built tiles does not
hold (and ``z - d`` can be negative: with ``Δz = 25`` m the first centre,
12.5 m, lies below a 20 m roof level and its displacement). Above `h`,
``z - d ≥ h - d > z_0`` for every λ (Macdonald et al. 1998), so the
logarithm is positive. `k = 1` where there are no buildings.
"""
function parent_forcing_level(z_centres::AbstractVector{<:Real}, h::Real)
    k = findfirst(>=(h), z_centres)
    k === nothing && error("parent_forcing_level: buildings of $h m reach above the top level")
    return k
end

"""
    ParentColumn

One parent column's surface: `surface` (a `SurfaceMosaic`, or a
`BlendedMosaic` of one), the class of each tile (`classes`), the
displacement height `d` (m) under each tile (Macdonald's `d` for built
tiles, 0 for park and water), the level `ktile` whose air state forces
each tile, and the column's forcing level `k = maximum(ktile)` (the one
level of a `BlendedMosaic`).

Built tiles are forced at the first level above the mean roof height
([`parent_forcing_level`](@ref)); park and water tiles, which have no
canopy of their own, at the first level (`k = 1`). Forcing a smooth
tile above the neighbouring roofs instead decoupled it: a pure water
column next to 58 m buildings, forced at 62.5 m with its own
``z_0 = 2·10^{-4}`` m, had a bulk Richardson number of about 1.8 at
13:00 (skin 8 K below the air) and returned LE = 0.7 W m⁻² (offline run
of #249).
"""
struct ParentColumn{S}
    surface::S
    classes::Vector{UInt8}
    d::Vector{Float64}
    ktile::Vector{Int}
    k::Int
end

"""
    ParentSurface

The tiled surface of the parent (or of the subset of columns built):
`landuse`, the class `fractions[I, J, c]`, the morphometry `λp`, `λf`,
`h`, the unresolved-building roughness `z0` and displacement `d` (m) per
column, the parent level centres `z_centres` (m), the `footprint` of the
child (`nothing` if none was given) and `columns[I, J]` (a
[`ParentColumn`](@ref), or `nothing` where none was built).
"""
struct ParentSurface
    landuse::ParentLanduse
    fractions::Array{Float64,3}
    λp::Matrix{Float64}
    λf::Matrix{Float64}
    h::Matrix{Float64}
    z0::Matrix{Float64}
    d::Matrix{Float64}
    z_centres::Vector{Float64}
    footprint::Any
    columns::Matrix{Union{Nothing,ParentColumn}}
end

"""
    build_parent_surface(case, pl = read_parent_landuse(case.cfg.landuse); child = case,
                         columns = :all, z_centres = 12.5:25:2987.5,
                         window = 8, blending = nothing,
                         macdonald = (;)) -> ParentSurface

Build the parent's tiled surface.

* `case` — a built Berlin case: its registry provides the one
  class-to-surface factory ([`class_surface`](@ref)), the per-class
  parameters, roughness table and initial temperatures (`cfg.Ts0`,
  water `+ cfg.Ts_water_offset`).
* `child` — the child case whose class map fills the parent under its
  footprint ([`parent_fractions`](@ref)), or `nothing` for the OSM table
  everywhere.
* `columns` — `:all`, or the `CartesianIndex`es / a `CartesianIndices`
  range of the columns to build (the whole parent is 36 864 columns of
  up to six tiles; building only a subset is how the tests stay cheap).
* `z_centres` — the parent's level centres (m above ground; default the
  uniform Δz = 25 m of the maintainer's grid; only the lowest levels are
  looked at).
* `window` — the morphometry window ([`parent_morphometry`](@ref)).
* `blending` — `nothing` for a plain `SurfaceMosaic` per column (each
  tile forced at its own level, measured from its own displacement; see
  [`ParentColumn`](@ref)), or an `AbstractBlendingHeight` for a
  `BlendedMosaic`, whose tiles are forced at the blending height (one
  forcing per column, at the column level above the area-mean
  displacement).
* `macdonald` — keyword overrides for
  `SurfaceTiles.macdonald_roughness` (`A`, `β`, `C_D`, `κ`).
* `unresolved = false` switches the unresolved-building treatment off
  (every tile with its class's own roughness, `d = 0`, forced at the
  first level): the parent is then the class-weighted mean of the
  child's surfaces, which the tests use as a control.

Each tile is `class_surface(case, SurfaceSite(c; T_skin, z0m, z0h))`.
A built tile takes `z0m = max(z0_Macdonald, class z0m)` but keeps its
class's scalar roughness `z0h`: the form drag of the buildings raises the
momentum exchange, and heat has no pressure-drag analogue — the
transfer of heat from bluff obstacles stays governed by their facets,
which is why urban arrays have large ``kB^{-1} = \\ln(z_{0m}/z_{0h})``
(Kanda et al. 2007). Inheriting ``z_{0h} = 0.1 z_{0m}`` from the child's
ratio instead puts the parent's footprint-mean sensible heat flux
+8.8 % above the child's at 13:00 and +9.8 % over 06-18 h (+2.8 % and
+3.8 % with the class z0h; offline run of test 4 of #249). Park and water tiles keep both
of their own roughness lengths.

Reference: Kanda, M., Kanega, M., Kawai, T., Moriwaki, R. & Sugawara, H.
(2007). Roughness lengths for momentum and heat derived from outdoor
urban scale models. *J. Appl. Meteorol. Climatol.*, 46, 1067-1079.
"""
function build_parent_surface(case, pl::ParentLanduse = read_parent_landuse(case.cfg.landuse);
                              child = case, columns = :all,
                              z_centres = 12.5:25.0:2987.5, window::Integer = 8,
                              blending = nothing, macdonald::NamedTuple = (;),
                              unresolved::Bool = true)
    fp = child === nothing ? nothing : child_footprint(pl, child.cfg)
    frac = parent_fractions(pl; child_sfc = child === nothing ? nothing : child.geom.sfc,
                            footprint = fp)
    λp, λf, h = parent_morphometry(pl, frac; window = window)
    rough = [macdonald_roughness(λp[I], λf[I], h[I]; macdonald...) for I in CartesianIndices(λp)]
    z0 = [unresolved ? x.z0 : 0.0 for x in rough]
    d = [unresolved ? x.d : 0.0 for x in rough]
    zc = collect(Float64, z_centres)
    cols = Matrix{Union{Nothing,ParentColumn}}(nothing, pl.nx, pl.ny)
    idx = columns === :all ? CartesianIndices((pl.nx, pl.ny)) : columns
    cfg = case.cfg
    for I in idx
        present = [c for c in 1:size(frac, 3) if frac[I, c] > 0]
        tiles = AbstractSurface[]
        dd = Float64[]
        for c in present
            Ts = cfg.Ts0 + (c == SURFACE_WATER ? cfg.Ts_water_offset : 0.0)
            built = is_built_class(c)
            z0c, z0hc = class_roughness(case, SurfaceSite(c; T_skin = Ts))
            site = built ? SurfaceSite(c; T_skin = Ts, z0m = max(z0[I], z0c), z0h = z0hc) :
                           SurfaceSite(c; T_skin = Ts)
            push!(tiles, class_surface(case, site))
            push!(dd, built ? d[I] : 0.0)
        end
        m = SurfaceMosaic(tiles, frac[I, present])
        s = blending === nothing ? m : BlendedMosaic(m, blending)
        kb = unresolved ? parent_forcing_level(zc, h[I]) : 1
        kt = [is_built_class(c) ? kb : 1 for c in present]
        cols[I] = ParentColumn(s, UInt8.(present), dd, kt, maximum(kt))
    end
    return ParentSurface(pl, frac, λp, λf, h, z0, d, zc, fp, cols)
end

_mosaic(col::ParentColumn{<:SurfaceMosaic}) = col.surface
_mosaic(col::ParentColumn{<:BlendedMosaic}) = col.surface.mosaic

"""
    parent_tile_forcings(ps, col, air) -> Vector{SurfaceForcing}

The forcing of each tile of `col`. `air` is the air state the column
sees: one `SurfaceForcing` (the same `T_air`, `q_air`, `U`, `p` and
radiation at every level, as in the offline tests) or a vector of them
indexed by parent level. Tile `i` takes the state of its level
`col.ktile[i]`, with its reference height measured from its own
displacement, ``z_{ref,i} = z_{k_i} - d_i`` (`z_ref` of `air` is ignored).
"""
function parent_tile_forcings(ps::ParentSurface, col::ParentColumn, air)
    return [_at_height(_air_at(air, k), ps.z_centres[k] - di) for (k, di) in zip(col.ktile, col.d)]
end

_air_at(f::SurfaceForcing, k) = f
_air_at(fs::AbstractVector{<:SurfaceForcing}, k) = fs[k]
_at_height(f::SurfaceForcing, z) =
    SurfaceForcing(; sw_down = f.sw_down, lw_down = f.lw_down, T_air = f.T_air, q_air = f.q_air,
                   U = f.U, p = f.p, z_ref = z, rain = f.rain, snow = f.snow)

"""
    step_parent_column!(ps, col, air, dt; kwargs...) -> SurfaceFluxes

Advance every tile of `col` by `dt` under the air state `air` (see
[`parent_tile_forcings`](@ref)) and return the fraction-weighted fluxes
``\\sum_i f_i F_i`` (`SurfaceTiles`). A plain mosaic forces each tile at
its own level and ``z_{k_i} - d_i``; a blended one forces the cell at the
column level `col.k`, ``z_k - \\bar d`` (``\\bar d`` the
fraction-weighted displacement), and its tiles at the blending height.
`kwargs` (`U_min`, `Q_extra`, …) go to every tile's `step_surface!`.
"""
function step_parent_column!(ps::ParentSurface, col::ParentColumn{<:SurfaceMosaic},
                             air, dt::Real; kwargs...)
    return step_surface!(col.surface, parent_tile_forcings(ps, col, air), dt; kwargs...)
end
function step_parent_column!(ps::ParentSurface, col::ParentColumn{<:BlendedMosaic},
                             air, dt::Real; kwargs...)
    dbar = sum(mosaic_fractions(col.surface) .* col.d)
    fc = _at_height(_air_at(air, col.k), ps.z_centres[col.k] - dbar)
    return step_surface!(col.surface, fc, dt; kwargs...)
end

"""
    parent_offline_forcing(cfg, hour; U = 3.0, q_air = 0.008, T_mean = 295.0, T_amp = 6.0)
        -> SurfaceForcing

A **synthetic** clear-sky forcing for offline checks of the parent
surface (the tests of #249 and the PR diagnostic), at local `hour` of
`cfg.date` at the anchor: the Bird clear-sky global horizontal
irradiance (`SolarGeometry.clear_sky_irradiance`, default aerosol) of
the case's solar position, an air temperature
``T_{mean} + T_{amp}\\sin(2π(t - 9)/24)`` (maximum at 15:00), constant
`q_air` and wind `U`, the case's ground pressure
([`case_surface_pressure`](@ref)), and the clear-sky downwelling
longwave ``ε_a σ T^4`` with Brutsaert's (1975) ``ε_a = 1.24
(e/T)^{1/7}`` (``e`` in hPa). `z_ref` is left at 10 m: the parent sets
its own per tile.

Reference: Brutsaert, W. (1975). On a derivable formula for long-wave
radiation from clear skies. *Water Resour. Res.*, 11, 742-744.
"""
function parent_offline_forcing(cfg::CaseConfig, hour::Real; U::Real = 3.0, q_air::Real = 0.008,
                                T_mean::Real = 295.0, T_amp::Real = 6.0)
    sp = solar_position_at(cfg, mod(hour, 24))
    sw = clear_sky_irradiance(sp, BirdClearSky()).ghi
    ps = case_surface_pressure(cfg)
    T = T_mean + T_amp * sin(2π * (hour - 9) / 24)
    e = q_air * ps / (0.622 + 0.378 * q_air) / 100      # hPa
    lw = 1.24 * (e / T)^(1 / 7) * STEFAN * T^4
    return SurfaceForcing(; sw_down = sw, lw_down = lw, T_air = T, q_air = q_air, U = U,
                          p = ps, z_ref = 10.0)
end
