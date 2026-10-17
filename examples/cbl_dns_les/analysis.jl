# =========================================================================
# Convective boundary layer: LES against DNS (#503). Output analysis only:
# this script reads finished run folders and runs no simulation code.
#
# The runs are the configurations of examples/cbl_dns_les/: the reference
# DNS (dns.toml, run name "cbl_dns") and the LES of four closures
# (les_<model>.toml). For every run and output record (every 10 min) it
# computes, from output/output.nc and the run's own config (run.toml):
#
#   * the total upward heat flux: the core's resolved + SGS profile plus the
#     molecular part −ν ∂θ/∂z (Prandtl number 1; the core's profile omits it);
#   * the boundary-layer height z_i, the height of the minimum of the total
#     heat flux (the entrainment zone), and the entrainment ratio
#     A = −(the minimum) / Q;
#   * the convective scales w* = (g/θ₀ Q z_i)^(1/3) and θ* = Q / w*;
#   * the w variance and the resolved TKE in mixed-layer scaling (σ_w²/w*²,
#     e/w*² against z/z_i), their maxima and the height of the σ_w² maximum;
#   * the horizontal spectrum of w at z = 0.5 z_i (radially binned E_w(k)).
#
# Against the DNS, each LES is scored over the averaging window (the second
# hour, t ≥ 3600 s: past the spin-up; single records are noisy, A varies by
# a factor 2 between records): the RMS difference of the window-averaged
# σ_w²/w*² and heat flux / Q over 0.1 ≤ z/z_i ≤ 1.2 (each record
# interpolated to the same z/z_i before averaging), and the relative
# differences of the window means of z_i and A. Records of different runs
# are matched by time to the nearest one (the output lands on the first
# step past each 10-min mark). It prints the tables and writes, into
# examples/cbl_dns_les/output/:
#
#   cbl_series.csv     run, t, z_i, w*, θ*, A, max σ_w²/w*², z(max)/z_i
#   cbl_profiles.csv   run, t, z/z_i, θ, total heat flux/Q, σ_w²/w*², e/w*²
#   cbl_spectra.csv    run, t, k, E_w(k) at 0.5 z_i
#   cbl_scores.csv     run, t, the scores against the DNS
#
# plots.jl reads these files (and the run folders for the animation).
#
#   julia --project -t 12 -m Aeolus run examples/cbl_dns_les/dns.toml
#   for m in smagorinsky dynamic_smagorinsky amd tke; do
#       julia --project -t 12 -m Aeolus run examples/cbl_dns_les/les_$m.toml; done
#   julia --project examples/cbl_dns_les/analysis.jl runs/*_cbl_*
#
# The run named "cbl_dns" is the reference.
# =========================================================================

using TOML
using Printf
using Aeolus.NCIO: NCDataset
import Aeolus.Operators.FFTW as FFTW

const OUT = joinpath(@__DIR__, "output")
const SCORE_RANGE = (0.1, 1.2)          # z/z_i range of the profile scores
const WINDOW = 3600.0                    # the averaging window: t ≥ WINDOW (the second hour)
const ZETA = collect(range(SCORE_RANGE...; length = 50))   # the common z/z_i heights

"A run folder: its name, its parameters (from run.toml) and its output (profiles, the 3-D w)."
function read_run(folder)
    cfg = TOML.parsefile(joinpath(folder, "run.toml"))["config"]
    dom = haskey(cfg, "domain") ? only(cfg["domain"]) : cfg       # the one domain's sections
    sec(name) = get(dom, name, get(cfg, name, Dict{String,Any}()))
    par = (ν = Float64(sec("dynamics")["nu"]), θ₀ = Float64(sec("dynamics")["theta0"]),
           g = Float64(sec("dynamics")["g"]), Q = Float64(sec("surface_theta_flux")["bottom"]),
           sgs = String(sec("sgs")["model"]))
    NCDataset(joinpath(folder, "output", "output.nc")) do ds
        has(k) = haskey(ds, k)
        (name = cfg["run"]["name"], folder = folder, par = par,
         x = Float64.(ds["x"][:]), y = Float64.(ds["y"][:]), z = Float64.(ds["z"][:]), t = Float64.(ds["time"][:]),
         θ = Float64.(ds["theta_profile"][:, :]), H = Float64.(ds["heat_flux_profile"][:, :]),
         w2 = Float64.(ds["w_variance_profile"][:, :]), e = Float64.(ds["resolved_tke_profile"][:, :]),
         esgs = has("sgs_tke_profile") ? Float64.(ds["sgs_tke_profile"][:, :]) : nothing,
         w = Float64.(ds["w"][:, :, :, :]))
    end
end

"∂f/∂z at the levels `z` (centred inside, one-sided at the ends)."
function ddz(f, z)
    n = length(z); d = similar(f)
    d[1] = (f[2] - f[1]) / (z[2] - z[1]); d[n] = (f[n] - f[n-1]) / (z[n] - z[n-1])
    for k in 2:n-1
        d[k] = (f[k+1] - f[k-1]) / (z[k+1] - z[k-1])
    end
    return d
end

"The total upward heat flux of record `r`: the core's (resolved + SGS) plus the molecular −ν ∂θ/∂z."
total_flux(run, r) = run.H[:, r] .- run.par.ν .* ddz(run.θ[:, r], run.z)

"Linear interpolation of (zs, fs) at z (clamped to the ends)."
function interp(zs, fs, z)
    z <= zs[1] && return fs[1]
    z >= zs[end] && return fs[end]
    k = searchsortedlast(zs, z)
    return fs[k] + (fs[k+1] - fs[k]) * (z - zs[k]) / (zs[k+1] - zs[k])
end

"The scales of record `r`: z_i (the minimum of the total heat flux, above the lowest 5 % of the domain), w*, θ*, A."
function scales(run, r)
    F = total_flux(run, r); z = run.z
    above = findall(>(0.05 * z[end]), z)
    k = above[argmin(F[above])]
    zi = z[k]; p = run.par
    wstar = cbrt(p.g / p.θ₀ * p.Q * zi)
    return (zi = zi, wstar = wstar, θstar = p.Q / wstar, A = -F[k] / p.Q, F = F)
end

"The radially binned horizontal spectrum E_w(k) of w at the level nearest to `zlev` of record `r`."
function w_spectrum(run, r, zlev)
    kz = argmin(abs.(run.z .- zlev))
    w = run.w[:, :, kz, r]; w = w .- sum(w) / length(w)
    nx, ny = size(w); Lx = (run.x[2] - run.x[1]) * nx; Ly = (run.y[2] - run.y[1]) * ny
    W = FFTW.fft(w) ./ (nx * ny)
    dk = 2π / max(Lx, Ly)
    kx(i) = 2π / Lx * (i - 1 <= nx ÷ 2 ? i - 1 : i - 1 - nx)
    ky(j) = 2π / Ly * (j - 1 <= ny ÷ 2 ? j - 1 : j - 1 - ny)
    nb = min(nx, ny) ÷ 2
    E = zeros(nb)
    for j in 1:ny, i in 1:nx
        b = round(Int, hypot(kx(i), ky(j)) / dk)
        1 <= b <= nb && (E[b] += abs2(W[i, j]) / dk)     # Σ_k E dk = ½⟨w'²⟩·2 = ⟨w'²⟩ over the bins
    end
    return (k = dk .* (1:nb), E = E)
end

"RMS difference of two profiles on the common heights ZETA."
profile_rms(a, b) = sqrt(sum(abs2, a .- b) / length(a))

"The window means of a run: σ_w²/w*² and the heat flux / Q on ZETA, z_i and A, over its records with t ≥ WINDOW."
function window_means(norm, name)
    recs = [v for ((n, t), v) in norm if n == name && t >= WINDOW - 30]
    isempty(recs) && return nothing
    avg(f) = sum(f, recs) / length(recs)
    return (n = length(recs), w2 = avg(r -> [interp(r.ζ, r.w2, s) for s in ZETA]),
            F = avg(r -> [interp(r.ζ, r.F, s) for s in ZETA]), zi = avg(r -> r.zi), A = avg(r -> r.A),
            sdA = sqrt(avg(r -> r.A^2) - avg(r -> r.A)^2))
end

function analyse(folders)
    isempty(folders) && error("usage: julia --project examples/cbl_dns_les/analysis.jl <run folders>")
    runs = [read_run(f) for f in folders]
    iref = findfirst(r -> r.name == "cbl_dns", runs)
    iref === nothing && @warn "no run named \"cbl_dns\": no scores against the DNS"
    mkpath(OUT)
    series = ["run,t,z_i,w_star,theta_star,A,max_w2_norm,z_w2max_over_zi"]
    profs = ["run,t,z_over_zi,theta,heat_flux_over_Q,w2_norm,tke_norm"]
    specs = ["run,t,k,E_w"]
    norm = Dict{Tuple{String,Float64},Any}()
    @printf("%-28s %8s %8s %7s %7s %7s %10s %8s\n", "run", "t (s)", "z_i (m)", "w*", "θ* (K)", "A", "maxσw²/w*²", "z/z_i")
    for run in runs, r in eachindex(run.t)
        run.t[r] > 0 || continue          # no flux yet at t = 0: z_i is undefined
        s = scales(run, r)
        ζ = run.z ./ s.zi
        w2n = run.w2[:, r] ./ s.wstar^2
        en = (run.e[:, r] .+ (run.esgs === nothing ? 0.0 : run.esgs[:, r])) ./ s.wstar^2
        km = argmax(w2n)
        @printf("%-28s %8.0f %8.1f %7.3f %7.4f %7.3f %10.3f %8.2f\n", run.name, run.t[r], s.zi, s.wstar, s.θstar, s.A, w2n[km], ζ[km])
        push!(series, @sprintf("%s,%.1f,%.3f,%.5f,%.6f,%.5f,%.5f,%.4f", run.name, run.t[r], s.zi, s.wstar, s.θstar, s.A, w2n[km], ζ[km]))
        for k in eachindex(ζ)
            push!(profs, @sprintf("%s,%.1f,%.5f,%.5f,%.6f,%.6f,%.6f", run.name, run.t[r], ζ[k], run.θ[k, r], s.F[k] / run.par.Q, w2n[k], en[k]))
        end
        sp = w_spectrum(run, r, 0.5 * s.zi)
        for (k, E) in zip(sp.k, sp.E)
            push!(specs, @sprintf("%s,%.1f,%.6e,%.6e", run.name, run.t[r], k, E))
        end
        norm[(run.name, run.t[r])] = (ζ = ζ, w2 = w2n, F = s.F ./ run.par.Q, zi = s.zi, A = s.A)
    end
    scores = ["run,records,rms_w2_norm,rms_heat_flux_over_Q,rel_z_i,rel_A,A,sd_A"]
    if iref !== nothing
        ref = runs[iref]; mr = window_means(norm, ref.name)
        println("\nagainst the DNS, window means over t ≥ ", WINDOW / 60, " min (", SCORE_RANGE[1], " ≤ z/z_i ≤ ", SCORE_RANGE[2], ")")
        @printf("%-28s %7s %12s %12s %9s %9s %14s\n", "run", "records", "RMS σw²/w*²", "RMS flux/Q", "Δz_i/z_i", "ΔA/A", "A ± sd")
        for run in runs
            m = window_means(norm, run.name)
            m === nothing && continue
            rw = profile_rms(m.w2, mr.w2); rf = profile_rms(m.F, mr.F)
            dz = (m.zi - mr.zi) / mr.zi; dA = (m.A - mr.A) / mr.A
            @printf("%-28s %7d %12.4f %12.4f %9.3f %9.3f %8.3f ± %.3f\n", run.name, m.n, rw, rf, dz, dA, m.A, m.sdA)
            push!(scores, @sprintf("%s,%d,%.6f,%.6f,%.5f,%.5f,%.5f,%.5f", run.name, m.n, rw, rf, dz, dA, m.A, m.sdA))
        end
    end
    for (name, lines) in (("cbl_series.csv", series), ("cbl_profiles.csv", profs),
                          ("cbl_spectra.csv", specs), ("cbl_scores.csv", scores))
        write(joinpath(OUT, name), join(lines, "\n") * "\n")
    end
    println("\nwrote ", OUT, "/cbl_{series,profiles,spectra,scores}.csv")
end

analyse(ARGS)
