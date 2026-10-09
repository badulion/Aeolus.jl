# =========================================================================
# Spectral broadening by sub-grid supersaturation fluctuations
#
# Era IV, Phase 45 (issue #14). Validation of the bulk moment closure
# (TurbulentMicrophysics.bulk_condensation_step!) against the Phase 41
# superdroplet reference with per-droplet stochastic supersaturation
# (issue #13), and of both against the DNS scaling σ_{r²} ∝ t^{1/2}.
#
#   Sardina, G., Picano, F., Brandt, L. & Caballero, R. (2015).
#     Continuous growth of droplet size variance due to condensation in
#     turbulent clouds. Phys. Rev. Lett., 115, 184501.
#   Lanotte, A. S., Seminara, A. & Toschi, F. (2009). Cloud droplet growth
#     by condensation in homogeneous isotropic turbulence. J. Atmos. Sci.,
#     66, 1685-1697.
#   Grabowski, W. W. & Wang, L.-P. (2013). Growth of cloud droplets in a
#     turbulent environment. Annu. Rev. Fluid Mech., 45, 293-324.
#
# WHAT IT DOES
#
# 1. Ascending parcel (w = 0.5 m/s, 300 s, 100 cm⁻³ of 5 μm droplets,
#    ε₀ = 0.1) for σ_S = 0, 0.1, 0.2 %: relative dispersion ε(t) in the
#    superdroplet reference and in the bulk scheme.
# 2. Stationary fluctuations (w = 0, monodisperse 10 μm, τ = 5 s, 600 s):
#    σ_{r²}(t) on log-log axes against the Ornstein-Uhlenbeck curve and the
#    t^{1/2} law, with the fitted exponents.
#
# The runs (the parcels of 1. and 2., each with the superdroplet reference
# and the bulk scheme) are the sweep examples/validation/configs/
# spectral_broadening.toml (the `column` run type, case "droplet_parcel");
# each records its history and the Ornstein–Uhlenbeck σ_{r²}(t). This
# script reads their output/results.toml.
#
#   julia --project -m Aeolus run examples/validation/configs/spectral_broadening.toml
#   julia --project examples/validation/spectral_broadening.jl [runs-root]
# Writes two SVGs to examples/validation/output/ (gitignored).
# =========================================================================

using Printf
include(joinpath(@__DIR__, "read_runs.jl"))
include(joinpath(@__DIR__, "..", "viz", "plot_utils.jl"))

outdir = joinpath(@__DIR__, "output")
mkpath(outdir)

runs = member_runs(runs_root(), validation_config("spectral_broadening"))
H = Dict(l => (; t = Float64.(r.res["t"]), ε = Float64.(r.res["epsilon"]), σ_r2 = Float64.(r.res["sigma_r2"]),
               ou = Float64.(r.res["sigma_r2_ou"])) for (l, r) in runs)
cfg_of = Dict(l => r.cfg for (l, r) in runs)

"Least-squares exponent of y ∝ t^p in log-log space."
function power_law_exponent(t, y)
    x = log.(t); z = log.(y); x̄ = sum(x) / length(x); z̄ = sum(z) / length(z)
    return sum((x .- x̄) .* (z .- z̄)) / sum((x .- x̄) .^ 2)
end

# --- 1. ascending parcel ---------------------------------------------------
println("Ascending parcel, w = 0.5 m/s, τ = 5 s, n_sd = 8000")
println("  σ_S     ε(300 s) SD   ε bulk   Δε      σ_r² SD   σ_r² bulk  (μm²)")
series = Vector{Vector{Float64}}(); labels = String[]
t_axis = Float64[]
for (σ, lab) in ((0.0, "0.0"), (1e-3, "0.1"), (2e-3, "0.2"))
    Hs = H["parcel_$(lab)_superdroplet"]
    Hb = H["parcel_$(lab)_bulk"]
    global t_axis = Hs.t
    @printf("  %.1f %%   %.4f      %.4f   %+.1f %%  %.2f     %.2f\n",
            100σ, Hs.ε[end], Hb.ε[end], 100(Hb.ε[end] / Hs.ε[end] - 1),
            1e12Hs.σ_r2[end], 1e12Hb.σ_r2[end])
    push!(series, Hs.ε); push!(labels, @sprintf("SD %.1f%%", 100σ))
    push!(series, Hb.ε); push!(labels, @sprintf("bulk %.1f%%", 100σ))
end
save_multiline(joinpath(outdir, "broadening_parcel_eps.svg"), t_axis, series, labels;
               xlabel = "time (s)", ylabel = "relative dispersion",
               title = "Ascending parcel: dispersion vs sigma_S", width = 720, height = 440)

# --- 2. stationary fluctuations: the t^{1/2} law ---------------------------
pc = cfg_of["stationary_superdroplet"]["parcel"]
τ, σ = pc["tau"], pc["sigma_s"]
Hs = H["stationary_superdroplet"]
Hb = H["stationary_bulk"]
t = Hs.t[2:end]
ou = Hb.ou[2:end]                                   # the OU curve at the bulk run's final temperature
half = ou[end] .* sqrt.(t ./ t[end])                 # pure t^{1/2}, matched at 600 s
w = t .>= 10τ
p_sd = power_law_exponent(t[w], Hs.σ_r2[2:end][w])
p_b  = power_law_exponent(t[w], Hb.σ_r2[2:end][w])
p_ou = power_law_exponent(t[w], ou[w])
@printf("\nStationary S' (σ_S = %.1f %%, τ = %g s): σ_r² ∝ t^p for t ≥ %g s\n", 100σ, τ, 10τ)
@printf("  p = %.4f (superdroplet), %.4f (bulk), %.4f (OU analytic), 0.5 (DNS law)\n",
        p_sd, p_b, p_ou)
@printf("  σ_r²(600 s) = %.2f μm² (SD), %.2f μm² (bulk), %.2f μm² (OU)\n",
        1e12Hs.σ_r2[end], 1e12Hb.σ_r2[end], 1e12ou[end])
save_multiline(joinpath(outdir, "broadening_sigma_r2.svg"), t,
               [1e12 .* Hs.σ_r2[2:end], 1e12 .* Hb.σ_r2[2:end], 1e12 .* ou, 1e12 .* half],
               ["superdroplet", "bulk", "OU analytic", "t^(1/2)"];
               xlabel = "time (s)", ylabel = "sigma_r2 (um^2)",
               title = "Stationary fluctuations: width of r^2", logx = true, logy = true,
               width = 720, height = 440)
println("\nFigures written to ", outdir)
