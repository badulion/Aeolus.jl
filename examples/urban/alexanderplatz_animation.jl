# Renderer for the Alexanderplatz run — NO PHYSICS.
#
# Reads the NetCDF file written by the flagship run
# (examples/urban/configs/berlin_summer.toml) and writes one
# self-contained, compressed APNG of the whole run
# (examples/urban/output/alexanderplatz.nc). Run the simulation first:
#
#   julia --project=. -m Aeolus run examples/urban/configs/berlin_summer.toml --set output.cadence=600
#   julia --project=. examples/urban/alexanderplatz_animation.jl
#
# (a record every 10 min gives a smooth 144-frame day; the default 30 min
# gives 48 frames). This script only reads the file: the physics lives in
# one place, the Berlin case, and re-colouring a panel never re-runs it.
#
# Panels (3 × 3, every map north up, east right):
#   row 1  surface temperature Ts · shortwave at the surface · sensible heat flux H
#   row 2  θ anomaly · vertical wind w · wind speed       (horizontal slice at z_slice)
#   row 3  NO2 · O3 in the first air cell · stack plume (column maximum of the tracer)
# Buildings taller than the slice are drawn dark grey on the slice panels;
# the two columns solid to the lid have no surface and are grey everywhere.
#
# Colour limits are fixed for the whole run (the 1st–99th percentile over
# every record, symmetric for anomalies) and printed under each panel:
# an autoscaled animation would make a calm night look as dramatic as
# solar noon.
#
# Environment:
#   AEOLUS_NC       input file   (default examples/urban/output/alexanderplatz.nc)
#   AEOLUS_SLICE_Z  slice height in m (default 10)
#   AEOLUS_FRAME_MS frame delay in ms (default 120)

using Printf
using Aeolus
using Statistics: mean, quantile

include(joinpath(@__DIR__, "..", "viz", "apng.jl"))

const OUT    = joinpath(@__DIR__, "output")
const NCPATH = get(ENV, "AEOLUS_NC", joinpath(OUT, "alexanderplatz.nc"))

isfile(NCPATH) || error("""
    No output file at $NCPATH.
    Run the simulation first:
        julia --project=. examples/urban/alexanderplatz_diurnal.jl
    """)

ncf   = NCDataset(NCPATH)
nrec  = ncf.dim["time"]
nrec > 0 || error("$NCPATH contains no time levels — did the run finish?")
hours = Array(ncf["hour"])
elevs = Array(ncf["solar_elevation"])
zc    = Array(ncf["z"])
hb    = Array(ncf["building_height"])
k_slice = argmin(abs.(zc .- parse(Float64, get(ENV, "AEOLUS_SLICE_Z", "10"))))
z_slice = zc[k_slice]
solid   = hb .> z_slice                       # building cells on the slice
# columns solid to the lid have no surface (H_sfc is NaN there) and keep
# their initial Ts: grey them out rather than let them pin the colour scale
nosfc   = isnan.(ncf["H_sfc"][:, :, 1])
@printf("Rendering %d time levels from %s (%.1f MB); slice at z = %.1f m (%d of %d cells solid)\n",
        nrec, NCPATH, filesize(NCPATH) / 1e6, z_slice, count(solid), length(solid))

"Field `Z[x, y]` as an image matrix: rows north (top) to south, columns west to east."
north_up(Z) = reverse(permutedims(Z), dims = 1)

# ---------------------------------------------------------------------------
# The panels: name, units, colormap, symmetric?, and how to get the field of
# record r as a 2-D (x, y) matrix with NaN where nothing is defined.
# ---------------------------------------------------------------------------
slice(name, r) = ncf[name][:, :, k_slice, r]          # reads one chunk
masked(A) = (B = Float64.(A); B[solid] .= NaN; B)
sfc(name, r) = (B = Float64.(ncf[name][:, :, r]); B[nosfc] .= NaN; B)

const PANELS = [
    (title = "Ts",              units = "K",     cmap = :viridis, sym = false,
     get = r -> sfc("Ts", r)),
    (title = "SW in",           units = "W/m2",  cmap = :viridis, sym = false,
     get = r -> sfc("sw_in", r)),
    (title = "H",               units = "W/m2",  cmap = :rdbu,    sym = true,
     get = r -> sfc("H_sfc", r)),
    (title = @sprintf("theta' %.0fm", z_slice), units = "K", cmap = :rdbu, sym = true,
     get = r -> (θ = masked(slice("theta", r)); θ .- mean(filter(isfinite, θ)))),
    (title = @sprintf("w %.0fm", z_slice), units = "m/s", cmap = :rdbu, sym = true,
     get = r -> masked(slice("w", r))),
    (title = @sprintf("wind %.0fm", z_slice), units = "m/s", cmap = :viridis, sym = false,
     get = r -> masked(hypot.(slice("u", r), slice("v", r)))),
    (title = "NO2 sfc", units = "ppb", cmap = :viridis, sym = false,
     get = r -> sfc("NO2_sfc", r)),
    (title = "O3 sfc",  units = "ppb", cmap = :viridis, sym = false,
     get = r -> sfc("O3_sfc", r)),
    (title = "plume max", units = "1", cmap = :viridis, sym = false,
     get = r -> Float64.(dropdims(maximum(ncf["pollutant"][:, :, :, r]; dims = 3); dims = 3))),
]

# Fixed colour limits over the whole run.
t0 = time()
fields = [[p.get(r) for r in 1:nrec] for p in PANELS]
limits = map(zip(PANELS, fields)) do (p, F)
    vals = filter(isfinite, reduce(vcat, vec.(F)))
    isempty(vals) && return (0.0, 1.0)
    if p.sym
        a = quantile(abs.(vals), 0.99)
        a = a > 0 ? a : 1.0
        (-a, a)
    else
        lo, hi = quantile(vals, (0.01, 0.99))
        hi > lo ? (lo, hi) : (lo - 0.5, lo + 0.5)
    end
end
@printf("Read %d fields x %d records in %.1f s\n", length(PANELS), nrec, time() - t0)
for (p, l) in zip(PANELS, limits)
    @printf("  %-26s %10.3g .. %-10.3g %s\n", p.title, l[1], l[2], p.units)
end

# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
const UP    = 4                               # pixels per cell
const NX, NY = size(hb)
const PW, PH = UP * NX, UP * NY               # map size
const PAD   = 8
const TITLE_H, BAR_H, BARLAB_H = 24, 10, 26
const CELL_W, CELL_H = PW + 2PAD, TITLE_H + PH + 4 + BAR_H + BARLAB_H
const HEAD_H = 62
const BG = UInt8(24)

fmt(v) = abs(v) >= 100 ? @sprintf("%.0f", v) : abs(v) >= 10 ? @sprintf("%.1f", v) :
         abs(v) >= 0.1 ? @sprintf("%.2f", v) : @sprintf("%.2g", v)

function blit!(dst, src, row, col)
    h, w = size(src, 1), size(src, 2)
    dst[row:row+h-1, col:col+w-1, :] .= src
    return dst
end

function panel_rgb(i, r)
    p = PANELS[i]; (lo, hi) = limits[i]
    cell = fill(BG, CELL_H, CELL_W, 3)
    add_label!(cell, p.title * " (" * p.units * ")"; row = 5, col = PAD, scale = 2)
    Z = north_up(fields[i][r])
    m = scalar_to_rgb(Z, lo, hi; upscale = UP, cmap = p.cmap)
    # solid / undefined cells: dark grey, not the colormap's mid-gray
    for y in 1:size(Z, 1), x in 1:size(Z, 2)
        isfinite(Z[y, x]) && continue
        m[UP*(y-1)+1:UP*y, UP*(x-1)+1:UP*x, :] .= UInt8(55)
    end
    blit!(cell, m, TITLE_H + 1, PAD + 1)
    by = TITLE_H + PH + 5
    blit!(cell, colorbar_rgb(PW, BAR_H; cmap = p.cmap), by, PAD + 1)
    ly = by + BAR_H + 4
    add_label!(cell, fmt(lo); row = ly, col = PAD + 1, scale = 2)
    hs = fmt(hi)
    add_label!(cell, hs; row = ly, col = PAD + PW - label_width(hs; scale = 2), scale = 2)
    if p.sym
        add_label!(cell, "0"; row = ly, col = PAD + PW ÷ 2 - 4, scale = 2)
    end
    return cell
end

function render(r)
    ncols, nrows = 3, 3
    frame = fill(BG, HEAD_H + nrows * CELL_H, ncols * CELL_W, 3)
    h = hours[r]
    hh = floor(Int, mod(h, 24)); mm = round(Int, 60 * (mod(h, 24) - hh)); mm == 60 && (hh += 1; mm = 0)
    add_label!(frame, @sprintf("%02d:%02d local   sun %+.0f deg", hh, mm, elevs[r]);
               row = 8, col = PAD + 1, scale = 3)
    add_label!(frame, "Berlin Alexanderplatz - Aeolus.jl flagship, 1 km x 1 km, north up";
               row = 38, col = PAD + 1, scale = 2, color = (170, 170, 170))
    for i in eachindex(PANELS)
        row, col = divrem(i - 1, ncols)
        blit!(frame, panel_rgb(i, r), HEAD_H + row * CELL_H + 1, col * CELL_W + 1)
    end
    return frame
end

t0 = time()
frames = [render(r) for r in 1:nrec]
@printf("Rendered %d frames in %.1f s\n", nrec, time() - t0)

mkpath(OUT)
out_path = joinpath(OUT, "alexanderplatz_animation.png")
t0 = time()
write_apng(out_path, frames; delay_ms = parse(Int, get(ENV, "AEOLUS_FRAME_MS", "120")))
@printf("APNG written: %s  (%.1f MB, %d frames, %dx%d) in %.1f s\n",
        out_path, filesize(out_path) / 1e6, length(frames),
        size(frames[1], 2), size(frames[1], 1), time() - t0)
