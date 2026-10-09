# =========================================================================
# The nocturnal boundary layer of the Berlin (Alexanderplatz) 24 h run,
# next to GABLS3 — P58-d (#194), validation only
#
# Reads the NetCDF of examples/urban/configs/berlin_summer.toml (it does not
# integrate anything) and, for every record of the night (solar elevation
# below 0), forms fluid-cell plane means per level (cells above the
# building height of their column) and reports
#   * the low-level jet: the maximum of the plane-mean wind speed below the
#     sponge (z < 0.75 Lz) and its height, reported as "none" when the speed
#     grows to the sponge (no interior maximum);
#   * the stratification: Δ⟨θ⟩ between the first level above the mean
#     building height and 100 m;
#   * the depth of the turbulent layer: the lowest level above the mean
#     building height where the turbulent fraction γ(z) (Transition, the
#     `turbulence_diag` addon) drops below 0.5;
#   * the surface-inversion depth: the lowest level above the mean building
#     height where ∂⟨θ⟩/∂z falls below a quarter of its maximum over the
#     levels below;
#   * the plane mean u* (the surface-exchange output).
#
#   julia --project=. examples/sbl/berlin_night.jl [path/to/alexanderplatz.nc]
# =========================================================================

using Aeolus: NCDataset
using Printf
using Statistics: mean

const NCPATH = length(ARGS) >= 1 ? ARGS[1] :
               joinpath(@__DIR__, "..", "urban", "output", "alexanderplatz.nc")

# a run still in progress has every completed record on disk (the writer
# syncs after each one); reading it while the writer holds the file open
# needs HDF5_USE_FILE_LOCKING=FALSE in the environment
f = NCDataset(NCPATH)
nrec = f.dim["time"]
z = Float64.(Array(f["z"])); dz = z[2] - z[1]; Lz = z[end] + dz / 2
bh = Float64.(Array(f["building_height"]))
hour = Float64.(Array(f["hour"])); elev = Float64.(Array(f["solar_elevation"]))
time = f["time"].var[:]                 # seconds since the case start
hmean = mean(bh)
println("Berlin run $(NCPATH): $(nrec) records, Lz = $Lz m, dz = $dz m, mean building height $(round(hmean; digits = 1)) m, sponge from $(0.75Lz) m")
@printf("%6s %6s | %10s %8s | %8s %8s | %6s\n", "hour", "elev", "jet (m/s)", "z (m)", "γ<0.5", "θ-inv", "u*")
function night_rows(f, z, dz, Lz, bh, hour, elev, hmean)
    rows = NamedTuple[]
    for r in 1:f.dim["time"]
        elev[r] < 0 || continue
        u = f["u"][:, :, :, r]; v = f["v"][:, :, :, r]; θ = f["theta"][:, :, :, r]
        γ = Float64.(f["gamma_turb"][:, r]); us = f["u_star"][:, :, r]
        nx, ny, nz = size(θ)
        S = zeros(nz); T = zeros(nz)
        for k in 1:nz
            s = 0.0; t = 0.0; n = 0
            for j in 1:ny, i in 1:nx
                z[k] > bh[i, j] || continue
                uc = 0.5 * (u[i, j, k] + u[mod1(i + 1, nx), j, k])
                vc = 0.5 * (v[i, j, k] + v[i, mod1(j + 1, ny), k])
                s += hypot(uc, vc); t += θ[i, j, k]; n += 1
            end
            S[k] = n > 0 ? s / n : NaN; T[k] = n > 0 ? t / n : NaN
        end
        ks = [k for k in 1:nz if z[k] < 0.75Lz && isfinite(S[k])]
        kj = ks[argmax(S[ks])]
        above = [k for k in 1:nz if z[k] > hmean]
        kγ = findfirst(k -> γ[k] < 0.5, above); zγ = kγ === nothing ? NaN : z[above[kγ]]
        g = [(T[k+1] - T[k]) / dz for k in 1:nz-1]
        kinv = NaN
        for (ii, k) in enumerate(above[1:end-1])
            gmax = maximum(g[above[1]:k])
            if g[k] < 0.25 * gmax && gmax > 0
                kinv = z[k] + dz / 2; break
            end
        end
        interior = kj < ks[end]
        k100 = argmin(abs.(z .- 100.0))
        dθ = T[k100] - T[above[1]]
        push!(rows, (hour = hour[r], S = S[kj], zj = interior ? z[kj] : NaN, zγ = zγ, zinv = kinv,
                     us = mean(filter(isfinite, us)), dθ = dθ, Stop = S[ks[end]]))
        @printf("%6.1f %6.1f | %10s %8s | %8.1f %8.1f | %6.3f | Δθ(%.0f-100 m) %+.2f K, speed at %.0f m %.2f m/s\n",
                hour[r], elev[r], interior ? @sprintf("%.2f", S[kj]) : "none", interior ? @sprintf("%.1f", z[kj]) : "-",
                zγ, kinv, mean(filter(isfinite, us)), z[above[1]], dθ, z[ks[end]], S[ks[end]])
    end

    return rows
end
rows = night_rows(f, z, dz, Lz, bh, hour, elev, hmean)
if !isempty(rows)
    nj = count(r -> isfinite(r.zj), rows)
    zi = filter(isfinite, [r.zinv for r in rows]); zg = filter(isfinite, [r.zγ for r in rows])
    @printf("night (%d records): interior jet in %d; inversion depth %s m; turbulent depth (γ < 0.5) %s m; Δθ(→100 m) %.2f K; u* %.3f m/s\n",
            length(rows), nj, isempty(zi) ? "-" : @sprintf("%.1f", mean(zi)), isempty(zg) ? "- (γ ≥ 0.5 at every level)" : @sprintf("%.1f", mean(zg)),
            mean(r.dθ for r in rows), mean(r.us for r in rows))
end
