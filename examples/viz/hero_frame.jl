# Render a single still from an Aeolus NetCDF output file, for the README.
# Output analysis only: it reads the file the urban run writes and runs no
# simulation code.
#
# A 48-frame APNG of a full diurnal cycle is ~75 MB — far too heavy for a
# repository front page — so the hero image is one well-chosen time level
# showing the four fields that carry the urban-heat-island story.
#
#   julia --project=. examples/viz/hero_frame.jl [path.nc] [hour]
#
# With no arguments it uses the Alexanderplatz output and picks the time
# level with the warmest surface, which is where the heat island is
# clearest.

using Printf
using Aeolus
using Statistics: mean

include(joinpath(@__DIR__, "apng.jl"))

const NCPATH = length(ARGS) >= 1 ? ARGS[1] :
    joinpath(@__DIR__, "..", "urban", "output", "alexanderplatz.nc")
isfile(NCPATH) || error("""
    No output at $NCPATH — run the simulation first:
        julia --project -m Aeolus run examples/urban/berlin_summer.toml
    """)

ncf   = NCDataset(NCPATH)
nrec  = ncf.dim["time"]
hours = Array(ncf["hour"])
elevs = Array(ncf["solar_elevation"])
z     = Array(ncf["z"])

# Pick the requested hour, or the warmest time level.
rec = if length(ARGS) >= 2
    h = parse(Float64, ARGS[2])
    argmin(abs.(hours .- h))
else
    argmax([maximum(ncf["Ts"][:, :, r]) for r in 1:nrec])
end
@printf("Rendering record %d of %d  (t = %.1f h, sun %.1f°)\n",
        rec, nrec, hours[rec], elevs[rec])

k = max(1, min(length(z), searchsortedfirst(z, 30.0)))
Ts  = ncf["Ts"][:, :, rec]
sw  = ncf["sw_in"][:, :, rec]
u   = ncf["u"][:, :, k, rec]
v   = ncf["v"][:, :, k, rec]
w3  = ncf["w"][:, :, k, rec]
uh  = sqrt.(u .^ 2 .+ v .^ 2)
hgt = Array(ncf["building_height"])

up = parse(Int, get(ENV, "AEOLUS_HERO_UPSCALE", "6"))

# Percentile clipping, not min/max. A single 200 m tower (the Fernsehturm)
# otherwise sets the height scale and squashes every ordinary 10-40 m
# building into the bottom of the colour map; the same happens to surface
# temperature, where a few hot roofs flatten the street-level contrast that
# is the whole point of the figure.
function pclip(A, lo, hi)
    v = sort(vec(collect(Float64, A)))
    n = length(v)
    return (v[clamp(ceil(Int, lo*n), 1, n)], v[clamp(ceil(Int, hi*n), 1, n)])
end
Ts_lo, Ts_hi   = pclip(Ts, 0.02, 0.98)
hgt_lo, hgt_hi = 0.0, pclip(filter(>(0), hgt), 0.0, 0.95)[2]

panels = [scalar_to_rgb(Ts,  Ts_lo, Ts_hi; upscale = up),
          scalar_to_rgb(sw,  0.0, max(1.0, maximum(sw)); upscale = up),
          scalar_to_rgb(hgt, hgt_lo, max(hgt_hi, 1.0); upscale = up),
          scalar_to_rgb(w3, -1.5, 1.5; upscale = up)]
# Label colour follows the panel: white text on a bright viridis panel
# (the sunlit shortwave field is almost entirely yellow) is unreadable, so
# pick whichever of black or white contrasts with what is actually there.
function label_colour(panel, row, nrow, ncol)
    h, w = size(panel, 1), size(panel, 2)
    r0, r1 = clamp(row, 1, h), clamp(row + nrow, 1, h)
    c0, c1 = 1, clamp(ncol, 1, w)
    lum = 0.0; n = 0
    for r in r0:r1, c in c0:c1
        lum += 0.299panel[r, c, 1] + 0.587panel[r, c, 2] + 0.114panel[r, c, 3]
        n += 1
    end
    return (lum / max(n, 1)) > 140 ? (0, 0, 0) : (255, 255, 255)
end

for (p, t) in zip(panels, (@sprintf("surface temperature %.0f-%.0f K", Ts_lo, Ts_hi),
                           @sprintf("shortwave at surface 0-%.0f W/m2", maximum(sw)),
                           @sprintf("building height 0-%.0f m", hgt_hi),
                           "vertical velocity -1.5 to 1.5 m/s at 30 m"))
    add_label!(p, t; row = 6, col = 6,
               color = label_colour(p, 6, 8, 6 * length(t) + 6))
end
mosaic = tile_frames(reshape(panels, (2, 2)))
# No caption baked into the image: it would land on the bright shortwave
# panel, where white text is unreadable. The surrounding document carries
# the time and location instead.

out = get(ENV, "AEOLUS_HERO_OUT",
          joinpath(@__DIR__, "..", "..", "assets", "hero.png"))
write_png(out, mosaic)
@printf("%s  (%.2f MB, %dx%d)  Ts %.1f-%.1f K\n",
        out, filesize(out)/1e6, size(mosaic, 2), size(mosaic, 1),
        minimum(Ts), maximum(Ts))
