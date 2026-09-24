# Renderer for the Alexanderplatz run — NO PHYSICS.
#
# Reads the frame files written by `alexanderplatz_diurnal.jl` into
# `output/frames/` and stitches them into a single self-contained APNG.
# Run the simulation first:
#
#   julia --project=. examples/urban/alexanderplatz_diurnal.jl
#   julia --project=. examples/urban/alexanderplatz_animation.jl
#
# This script used to run its OWN 24-hour simulation in order to capture
# frames from live state. That meant the physics existed in two places,
# and the two copies drifted: the other script had soil conduction,
# latent heat, chemistry and aerosol; this one had none of them, while
# the documentation claimed both ran "the same physics". The drift is
# what allowed an energy-conservation bug in `step_soil!` to survive —
# the script that exercised the soil model was not the script whose
# diagnostics anyone was reading. It also meant every animation cost a
# second full integration, and re-colouring a panel meant re-running the
# physics.
#
# Panels (2 rows × 4 columns):
#   1. Surface temperature Ts        5. Horizontal wind |uh|
#   2. Shortwave at the surface      6. Pressure anomaly p'
#   3. Potential-temp anomaly θ'     7. Pollutant tracer c
#   4. Vertical velocity w           8. Sky-view factor (static)

using Printf
using Aeolus
using Statistics: mean

include(joinpath(@__DIR__, "..", "viz", "plot_utils.jl"))
include(joinpath(@__DIR__, "..", "viz", "apng.jl"))

const OUT    = joinpath(@__DIR__, "output")
const NCPATH = joinpath(OUT, "alexanderplatz.nc")

isfile(NCPATH) || error("""
    No output file at $NCPATH.
    Run the simulation first:
        julia --project=. examples/urban/alexanderplatz_diurnal.jl
    """)

# Read the model output back through the same dependency-free NetCDF
# layer that wrote it. Records are read one at a time rather than all at
# once: the full 3-D state for a day is a few hundred MB and the renderer
# only ever needs one horizontal slice.
ncf   = nc_open(NCPATH)
nrec  = nc_nrecords(ncf)
nrec > 0 || error("$NCPATH contains no time levels — did the run finish?")
svf   = nc_read(ncf, "sky_view_factor")
hours = nc_read(ncf, "hour")
elevs = nc_read(ncf, "solar_elevation")
nz    = length(nc_read(ncf, "z"))
k_slice = max(1, min(nz, round(Int, 30.0 / (nc_read(ncf, "z")[1] * 2)) ))
@printf("Rendering %d time levels from %s (%.1f MB)\n",
        nrec, NCPATH, filesize(NCPATH) / 1e6)
@printf("Horizontal panels at z = %.1f m\n", nc_read(ncf, "z")[k_slice])

# Fixed colour limits so amplitudes stay comparable across the whole day.
# Deliberately NOT per-frame autoscaled: an autoscaled animation makes a
# calm night look as dramatic as solar noon.
const upscale = 4
const lim_Ts  = (280.0, 335.0)
const lim_θ   = (-4.0, 4.0)
const lim_w   = (-2.0, 2.0)
const lim_uh  = (0.0, 4.0)
const lim_p   = (-3.0, 3.0)
const lim_c   = (0.0, 5.0)
const lim_sw  = (0.0, 1200.0)

# The sky-view factor is geometry, so it is static — render it once.
svf_panel = scalar_to_rgb(Array(svf), 0.0, 1.0; upscale = upscale)
add_label!(svf_panel, "SVF"; row = 4, col = 4)

"Render one time level. 3-D fields are sliced at `k_slice` on read."
function render(rec::Int)
    sl(name) = @view nc_read_record(ncf, name, rec)[:, :, k_slice]
    u  = sl("u"); v = sl("v"); w = sl("w")
    θ  = sl("theta"); pr = sl("p"); cc = sl("pollutant")
    Ts = nc_read_record(ncf, "Ts", rec)
    sw = nc_read_record(ncf, "sw_in", rec)

    uh = sqrt.(u .^ 2 .+ v .^ 2)
    θa = θ  .- mean(θ)
    pa = pr .- mean(pr)

    panels = [scalar_to_rgb(Ts, lim_Ts...; upscale = upscale),
              scalar_to_rgb(sw, lim_sw...; upscale = upscale),
              scalar_to_rgb(θa, lim_θ...;  upscale = upscale),
              scalar_to_rgb(w,  lim_w...;  upscale = upscale),
              scalar_to_rgb(uh, lim_uh...; upscale = upscale),
              scalar_to_rgb(pa, lim_p...;  upscale = upscale),
              scalar_to_rgb(cc, lim_c...;  upscale = upscale),
              svf_panel]
    for (panel, title) in zip(panels, ("Ts K", "SW W/m2", "theta anom K",
                                       "w m/s", "uh m/s", "p anom",
                                       "pollutant", "SVF"))
        title == "SVF" && continue        # already labelled, and shared
        add_label!(panel, title; row = 4, col = 4)
    end
    mosaic = tile_frames(reshape(panels, (2, 4)))
    add_label!(mosaic, @sprintf("t %04.1fh  sun %+05.1f deg",
                                hours[rec], elevs[rec]);
               row = size(mosaic, 1) - 12, col = 4)
    return mosaic
end

t0 = time()
frames = [render(r) for r in 1:nrec]
@printf("Rendered in %.1f s\n", time() - t0)

out_path = joinpath(OUT, "alexanderplatz_animation.png")
write_apng(out_path, frames; delay_ms = 250)
@printf("APNG written: %s  (%.1f MB, %d frames, %dx%d)\n",
        out_path, filesize(out_path)/1e6,
        length(frames), size(frames[1], 2), size(frames[1], 1))
