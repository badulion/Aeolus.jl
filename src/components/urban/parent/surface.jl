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
#     `class_surface` (components/urban/surface_exchange.jl): soil columns for
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

"""
    coarsen_landuse(pl, b) -> ParentLanduse

The parent land-use table on a grid `b` times coarser per axis (for the
reduced-resolution parents of the tests): class counts summed over each
`b × b` block (`r` becomes `b r`), frontal area density averaged, the
mean height roof-area weighted, the maximum height the block maximum.
"""
function coarsen_landuse(pl::ParentLanduse, b::Integer)
    b == 1 && return pl
    (pl.nx % b == 0 && pl.ny % b == 0) || error("coarsen_landuse: $(pl.nx) × $(pl.ny) not divisible by $b")
    nx, ny = pl.nx ÷ b, pl.ny ÷ b
    counts = zeros(Int, nx, ny, size(pl.counts, 3))
    λf, hm, hx = zeros(nx, ny), zeros(nx, ny), zeros(nx, ny)
    for J in 1:ny, I in 1:nx
        is, js = (I-1)*b+1:I*b, (J-1)*b+1:J*b
        counts[I, J, :] .= vec(sum(view(pl.counts, is, js, :); dims = (1, 2)))
        λf[I, J] = sum(view(pl.λf, is, js)) / b^2
        w = view(pl.counts, is, js, SURFACE_ROOF)
        hm[I, J] = sum(w) > 0 ? sum(w .* view(pl.hmean, is, js)) / sum(w) : 0.0
        hx[I, J] = maximum(view(pl.hmax, is, js))
    end
    return ParentLanduse(nx, ny, pl.dx * b, pl.x0, pl.y0, pl.r * b, counts, λf, hm, hx,
                         vcat(pl.header, "# coarsened by $b × $b (coarsen_landuse)"))
end

"""
    crop_landuse(pl, Is, Js) -> ParentLanduse

The cells `Is × Js` of the land-use table `pl`, with `x0`, `y0` moved to
the corner of the first one.
"""
function crop_landuse(pl::ParentLanduse, Is::AbstractUnitRange, Js::AbstractUnitRange)
    (Is == 1:pl.nx && Js == 1:pl.ny) && return pl
    return ParentLanduse(length(Is), length(Js), pl.dx, pl.x0 + (first(Is) - 1) * pl.dx,
                         pl.y0 + (first(Js) - 1) * pl.dx, pl.r,
                         pl.counts[Is, Js, :], pl.λf[Is, Js], pl.hmean[Is, Js], pl.hmax[Is, Js],
                         vcat(pl.header, "# cropped to cells $Is × $Js (crop_landuse)"))
end

"""
    landuse_static_tiles(pl::ParentLanduse, mesh; origin_lat, origin_lon, origin_x, origin_y) -> StaticTiles

The tile form of the static driver (`STATIC_TILES_SCHEMA`) of a domain on
`mesh` from the land-use table `pl` (its cells relative to the anchor at
`origin_lat`, `origin_lon`, which sits at `(origin_x, origin_y)` in the
mesh's coordinates): `pl` cropped to the mesh's box
([`crop_landuse`](@ref)) and coarsened to its cells
([`coarsen_landuse`](@ref)). With `child = (sfc, x0, y0)` (a nested urban
site's class map and its corner in the mesh's coordinates) the columns
under the site's footprint hold the site's own classes, as the parent's
surface of #249 had them. What AeolusPrep.jl's `aeolusprep static`
writes for a `[[domain]]` with `[surface] model = "tiles"`, and what that
domain's [`TiledUrbanSurface`](@ref) reads back. An error when the box is
not on the table's cell faces, leaves it, or is not a whole number of
table cells per mesh cell.
"""
function landuse_static_tiles(pl::ParentLanduse, mesh; origin_lat::Real, origin_lon::Real,
                              origin_x::Real, origin_y::Real, child = nothing)
    fi = (mesh.x0 - origin_x - pl.x0) / pl.dx
    fj = (mesh.y0 - origin_y - pl.y0) / pl.dx
    ni, nj = mesh.Lx / pl.dx, mesh.Ly / pl.dx
    all(x -> abs(x - round(x)) < 1e-9, (fi, fj, ni, nj)) ||
        error("landuse_static_tiles: the domain's box is not on the land-use cell faces ($fi, $fj, $ni, $nj cells)")
    I0, J0 = round(Int, fi) + 1, round(Int, fj) + 1
    Is, Js = I0:I0+round(Int, ni)-1, J0:J0+round(Int, nj)-1
    (first(Is) >= 1 && first(Js) >= 1 && last(Is) <= pl.nx && last(Js) <= pl.ny) ||
        error("landuse_static_tiles: the domain's box ($Is, $Js) leaves the land-use table (1:$(pl.nx), 1:$(pl.ny))")
    c = crop_landuse(pl, Is, Js)
    b = c.nx ÷ mesh.Nx
    (b * mesh.Nx == c.nx && b * mesh.Ny == c.ny && c.dx * b ≈ mesh.dx) ||
        error("landuse_static_tiles: the land use ($(c.nx) cells of $(c.dx) m) does not coarsen to $(mesh.Nx) cells of $(mesh.dx) m")
    q = coarsen_landuse(c, b)
    counts, notes = child === nothing ? (Int32.(q.counts), q.header) : _footprint_counts(q, mesh, child)
    return StaticTiles(collect(Float64, Aeolus.Grids.xnodes(mesh, CellCenter())),
                       collect(Float64, Aeolus.Grids.ynodes(mesh, CellCenter())),
                       q.dx, q.dx, Float64(origin_lat), Float64(origin_lon), Float64(origin_x), Float64(origin_y),
                       q.x0, q.y0, q.r, counts, q.λf, q.hmean, q.hmax, notes, Dict{String,Any}())
end

# The class counts of the coarsened table `q` on `mesh` with the columns
# under a nested urban site's footprint taken from the site's own class map
# (`child = (sfc, x0, y0)`: its built surface classes, its corner in the
# mesh's coordinates): each covered column holds the classes of its
# r × r site cells (the site's cells are the table's subcells), so the
# parent sees the site's surface there, glass (6) included (#249; NEST-p3
# #483: data of the parent's static driver).
function _footprint_counts(q::ParentLanduse, mesh, child)
    sfc = child.sfc
    r = q.r
    nI, nJ = size(sfc, 1) ÷ r, size(sfc, 2) ÷ r
    (nI * r == size(sfc, 1) && nJ * r == size(sfc, 2)) ||
        error("landuse_static_tiles: the site's $(size(sfc)) cells are not whole parent columns of $r × $r cells")
    fi, fj = (child.x0 - mesh.x0) / q.dx, (child.y0 - mesh.y0) / q.dx
    all(x -> abs(x - round(x)) < 1e-9, (fi, fj)) ||
        error("landuse_static_tiles: the site's corner is not on a parent cell face ($fi, $fj)")
    Is, Js = round(Int, fi) .+ (1:nI), round(Int, fj) .+ (1:nJ)
    (first(Is) >= 1 && first(Js) >= 1 && last(Is) <= q.nx && last(Js) <= q.ny) ||
        error("landuse_static_tiles: the site's footprint ($Is, $Js) leaves the domain")
    nclass = max(size(q.counts, 3), Int(maximum(sfc)))
    counts = zeros(Int32, q.nx, q.ny, nclass)
    counts[:, :, 1:size(q.counts, 3)] .= q.counts
    for (b, J) in enumerate(Js), (a, I) in enumerate(Is)
        blk = view(sfc, (a - 1) * r + 1:a * r, (b - 1) * r + 1:b * r)
        for c in 1:nclass
            counts[I, J, c] = count(==(c), blk)
        end
    end
    return counts, vcat(q.header, "# the site's surface classes under its footprint, columns $Is × $Js")
end

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
function child_footprint(pl::ParentLanduse, cfg::NamedTuple)
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
    nclass = max(nclass, size(pl.counts, 3))          # (a static driver's counts may carry glass, 6)
    f = zeros(pl.nx, pl.ny, nclass)
    f[:, :, 1:size(pl.counts, 3)] .= pl.counts ./ pl.r^2
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
    parent_forcing_level(z_centres, h; d = 0, z0 = 0) -> k

The parent level whose air state forces a built tile: the lowest level
whose centre lies at or above the mean building height `h` **and** at
least twice the tile's roughness length `z0` above its displacement
height `d`, ``z_k - d ≥ 2 z_0``. Below the roofs the air is inside the
unresolved canopy, where the Monin–Obukhov profile ``\\ln((z - d)/z_0)``
of the built tiles does not hold (and ``z - d`` can be negative: with
``Δz = 25`` m the first centre, 12.5 m, lies below a 20 m roof level and
its displacement). With `d = z0 = 0` this is the first centre at or
above `h` (`k = 1` where there are no buildings).

The first centre above `h` alone does not guarantee a valid surface
layer, although ``h - d > z_0`` holds for Macdonald et al.'s (1998)
roughness at every ``λ_p < 1``: (i) as ``λ_p → 1``, ``d → h`` and
``z_k - d → z_k - h``, which is arbitrarily small when a level centre
lies just above the roofs; (ii) a built tile's roughness is
``\\max(z_{0,\\mathrm{Macdonald}}, z_{0,\\mathrm{class}})``, and the
class floor is not covered by Macdonald's inequality. On the Berlin 20 m
grid (#568) one roof tile, under a fully roofed 160 m morphometry window
(``λ_p = 1``, ``h = 9.96`` m, ``d = 9.961`` m), had its first centre at
10 m, so ``z_k - d = 0.039`` m below its class ``z_0 = 0.05`` m, and the
surface-layer solve stopped the run. The margin ``z_0 < z/2`` is PALM's
check of the first grid level against the roughness length
(`land_surface_model_mod.f90`, which stops the run or, with
`allow_roughness_limitation`, lowers ``z_0`` to ``z/2``); here the
forcing level is raised instead and the roughness kept.
"""
function parent_forcing_level(z_centres::AbstractVector{<:Real}, h::Real; d::Real = 0.0, z0::Real = 0.0)
    k = findfirst(z -> z >= h && z - d >= 2z0, z_centres)
    k === nothing && error("parent_forcing_level: no level centre at or above the buildings ($h m) " *
                           "and $(2z0) m above their displacement ($d m)")
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
that lies at least ``2 z_0`` above their displacement height
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
child (`nothing` if none was given), `columns[I, J]` (a
[`ParentColumn`](@ref), or `nothing` where none was built) and `raised`,
one record per built tile whose forcing level was raised above the first
level over the roofs (see `raise_forcing_level` of
[`build_parent_surface`](@ref)): `(I, J, class, λp, λf, h, d, z0m, z0h,
k_first, k)`.
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
    raised::Vector{NamedTuple}
end
ParentSurface(pl, frac, λp, λf, h, z0, d, zc, fp, cols) =
    ParentSurface(pl, frac, λp, λf, h, z0, d, zc, fp, cols, NamedTuple[])

"""
    build_parent_surface(case, pl::ParentLanduse; child = case,
                         columns = :all, z_centres = 12.5:25:2987.5,
                         window = 8, blending = nothing,
                         macdonald = (;)) -> ParentSurface

Build the parent's tiled surface.

* `case` — the class-to-surface factory: a built urban site (its
  registry's [`class_surface`](@ref), per-class parameters, roughness
  table and initial temperatures, `cfg.Ts0`, water `+ cfg.Ts_water_offset`)
  or the [`TileClasses`](@ref) of a tiled surface's own options.
* `child` — the urban site whose class map fills the parent under its
  footprint ([`parent_fractions`](@ref)): `case` when it is a site (the
  tests of #249), else `nothing`, the land use everywhere (a domain's
  static driver holds the child's classes under its footprint).
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
* `raise_forcing_level = false` — what to do with a built tile whose
  first level at or above the roofs lies less than ``2 z_0`` above its
  displacement height ([`parent_forcing_level`](@ref); #568): `false`
  stops the build with an `ArgumentError` naming the tiles (as PALM stops
  a run whose first level is below ``2 z_0``); `true` forces those tiles
  at the lowest level that is a valid Monin–Obukhov height (PALM's
  `allow_roughness_limitation` lowers ``z_0`` instead) and lists them in
  the surface's `raised`. Such tiles sit under a morphometry window that
  is (almost) fully roofed, i.e. buildings about as large as the window,
  which a grid that fine may resolve rather than parametrise.
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
function build_parent_surface(case, pl::ParentLanduse;
                              child = case isa UrbanSite ? case : nothing, columns = :all,
                              z_centres = 12.5:25.0:2987.5, window::Integer = 8,
                              blending = nothing, macdonald::NamedTuple = (;),
                              unresolved::Bool = true, raise_forcing_level::Bool = false)
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
    # one column per iteration, each writing only its own cols[I] and
    # raisedcol[n]: on the threads (the soil columns of every tile, about
    # 4 µs each; #567)
    cidx = collect(idx)
    raisedcol = [NamedTuple[] for _ in cidx]
    try
        Threads.@threads :dynamic for n in eachindex(cidx)
            I = cidx[n]
            present = [c for c in 1:size(frac, 3) if frac[I, c] > 0]
            tiles = AbstractSurface[]
            dd = Float64[]
            kt = Int[]
            for c in present
                Ts = tile_skin(case, c)
                built = is_built_class(c)
                z0c, z0hc = class_roughness(case, SurfaceSite(c; T_skin = Ts))
                z0mt = max(z0[I], z0c)
                site = built ? SurfaceSite(c; T_skin = Ts, z0m = z0mt, z0h = z0hc) :
                               SurfaceSite(c; T_skin = Ts)
                push!(tiles, class_surface(case, site))
                push!(dd, built ? d[I] : 0.0)
                # a built tile: the first level above the roofs, or (#568) the
                # lowest one that is a valid Monin–Obukhov height over its d and z0
                k = 1
                if built && unresolved
                    k1 = parent_forcing_level(zc, h[I])
                    k = parent_forcing_level(zc, h[I]; d = d[I], z0 = max(z0mt, z0hc))
                    k == k1 || push!(raisedcol[n], (I = I[1], J = I[2], class = c, λp = λp[I], λf = λf[I], h = h[I],
                                                    d = d[I], z0m = z0mt, z0h = z0hc, k_first = k1, k = k))
                end
                push!(kt, k)
            end
            m = SurfaceMosaic(tiles, frac[I, present])
            s = blending === nothing ? m : BlendedMosaic(m, blending)
            cols[I] = ParentColumn(s, UInt8.(present), dd, kt, maximum(kt))
        end
    catch e                       # the error of a column, not the threads' wrapper
        e isa CompositeException && !isempty(e.exceptions) || rethrow()
        x = first(e.exceptions)
        throw(x isa TaskFailedException ? x.task.exception : x)
    end
    raised = reduce(vcat, raisedcol; init = NamedTuple[])     # in column order, as a serial loop
    isempty(raised) || raise_forcing_level || throw(ArgumentError(_raised_message(raised, zc)))
    return ParentSurface(pl, frac, λp, λf, h, z0, d, zc, fp, cols, raised)
end

# The build's refusal (#568): how many built tiles, the worst of them.
function _raised_message(raised, zc)
    w = argmin(r -> (zc[r.k_first] - r.d) / max(r.z0m, r.z0h), raised)
    zr = zc[w.k_first] - w.d
    ncol = length(unique((r.I, r.J) for r in raised))
    f(x) = round(x; sigdigits = 4)
    return "build_parent_surface: $(length(raised)) built tile(s) in $ncol column(s) would be forced less than " *
           "2 z0 above their displacement height, where the Monin–Obukhov surface layer does not hold (#568). " *
           "Worst: column ($(w.I), $(w.J)), class $(w.class): λp = $(f(w.λp)), λf = $(f(w.λf)), h = $(f(w.h)) m, " *
           "d = $(f(w.d)) m, z0m = $(f(w.z0m)) m, z0h = $(f(w.z0h)) m; the first level above the roofs, " *
           "$(f(zc[w.k_first])) m, is $(f(zr)) m above d. Such columns lie under an (almost) fully roofed " *
           "morphometry window: buildings about the size of the window, which a grid this fine may resolve " *
           "rather than parametrise. To force these tiles at the lowest level with z − d ≥ 2 z0 instead, set " *
           "[surface] raise_forcing_level = true (the run then lists them in results.toml)."
end

"The initial skin temperature (K) of a tile of class `c`: the site's `Ts0`, water `+ Ts_water_offset`."
tile_skin(case::UrbanSite, c::Integer) = case.cfg.Ts0 + (c == SURFACE_WATER ? case.cfg.Ts_water_offset : 0.0)

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
function parent_offline_forcing(cfg::NamedTuple, hour::Real; U::Real = 3.0, q_air::Real = 0.008,
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
