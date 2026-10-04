# Kelvin–Helmholtz billow — Era VI, Phase 59 (P59-c, #180).
#
# A weakly stratified tanh shear layer (u = U tanh(z/δ), θ = θ₀ + Δθ
# tanh(z/δ), minimum Ri_g = J = 0.05 at the centre, Re = Uδ/ν = 1000) in
# 2-D (Ny = 1) with free-slip walls at z = ±14δ, seeded at the most
# unstable wavelength αδ = 0.4446 with w′ = 1e-4 U. Unlike the linear test
# (test/era6_transition_kh.jl) the base state is free to diffuse and the
# billow runs into its nonlinear stage.
#
# Diagnostics, all from Aeolus.Transition:
#   * the amplitude A(t) = rms w and its linear growth rate (growth_rate);
#   * the turbulent fraction γ(z, t) of the vorticity-fluctuation
#     indicator, and its mean over the layer |z| < 4δ;
#   * the flux Richardson number Ri_f = −B/P of the layer, the mixing
#     efficiency proxy (Ivey & Imberger 1991), instantaneous (printed) and
#     cumulative (plotted).
#
# Figures (docs/src/assets/transition/): kh_amplitude.svg, kh_gamma.svg,
# kh_rif.svg and kh_vorticity.png (spanwise vorticity at three times).
#
#   julia --project=. examples/transition/kelvin_helmholtz.jl

using Aeolus
using Aeolus.Fields: set!, interior_view
using Aeolus.Operators: _vorticity_at
using Printf

include(joinpath(@__DIR__, "..", "viz", "plot_utils.jl"))
include(joinpath(@__DIR__, "..", "viz", "apng.jl"))

const OUT = joinpath(@__DIR__, "..", "..", "docs", "src", "assets", "transition")
mkpath(OUT)

δ, U, Re, J, αδ = 1.0, 1.0, 1000.0, 0.05, 0.4446
Nx, Nz = 128, 256
Lx = 2π * δ / αδ; dx = Lx / Nx; Lz = Nz * dx
mesh = RegularGrid(; size = (Nx, 1, Nz), extent = (Lx, dx, Lz),
                   origin = (0.0, 0.0, -Lz / 2), halo = (2, 2, 2))
θ₀, g = 300.0, 9.81
bcs = FieldBoundaryConditions(:periodic, :periodic, :wall)
sim = BoussinesqSimulation(mesh; ν = U * δ / Re, θ₀ = θ₀, g_accel = g, bcs = bcs)
Δθ = J * U^2 * θ₀ / (g * δ)
set!(sim.u, (x, y, z) -> U * tanh(z / δ))
set!(sim.θ, (x, y, z) -> θ₀ + Δθ * tanh(z / δ))
set!(sim.w, (x, y, z) -> 1e-4 * U * sin(αδ * x / δ) * exp(-(z / δ)^2))

zc = [mesh.z0 + (k - 0.5) * mesh.dz for k in 1:Nz]
layer = findall(z -> abs(z) < 4δ, zc)

"spanwise vorticity ω_y at the cell centres (x, z)"
function omega_y(sim)
    m = sim.mesh
    fill_halos!(sim.u, sim.bcs); fill_halos!(sim.v, sim.bcs); fill_halos!(sim.w, sim.bcs)
    ω = zeros(m.Nz, m.Nx)
    for k in 1:m.Nz, i in 1:m.Nx
        ω[k, i] = _vorticity_at(sim.u.data, sim.v.data, sim.w.data, m.Hx + i, m.Hy + 1,
                                m.Hz + k, 1 / m.dx, 1 / m.dy, 1 / m.dz)[2]
    end
    return ω
end

tend = 160.0
dt = 0.6 * dx / U; nsteps = ceil(Int, tend / dt); dt = tend / nsteps
every = max(1, round(Int, 1.0 / dt))                   # diagnostics every ~1 δ/U
ts, As, γm, Rif, Bs, Ps = Float64[], Float64[], Float64[], Float64[], Float64[], Float64[]
γ_snap = Dict{Float64,Vector{Float64}}()
snaps_t = (60.0, 90.0, 140.0); snaps = Dict{Float64,Matrix{Float64}}()
t0 = time()
for n in 1:nsteps
    step_rk3!(sim, dt)
    if n % every == 0 || n == nsteps
        wi = interior_view(sim.w)
        push!(ts, sim.t); push!(As, sqrt(sum(abs2, wi) / length(wi)))
        γ = turbulent_fraction(sim)
        push!(γm, sum(γ[layer]) / length(layer))
        r = flux_richardson(sim)
        B = sum(r.B[layer]); P = sum(r.P[layer])
        push!(Bs, B); push!(Ps, P); push!(Rif, P == 0 ? NaN : -B / P)
        for ts_ in snaps_t
            if !haskey(snaps, ts_) && sim.t >= ts_
                snaps[ts_] = omega_y(sim); γ_snap[ts_] = γ
            end
        end
    end
end
@printf("KH billow J = %.2f, Re = %.0f, %d×1×%d: %d steps in %.1f s\n", J, Re, Nx, Nz, nsteps, time() - t0)

fit = growth_rate(ts, As)
@printf("linear growth rate σ = %.4f U/δ (R² = %.6f over t = %.0f-%.0f δ/U)\n", fit.rate, fit.r2, fit.window...)
ipk = argmax(As)
@printf("saturation: A_max = %.3f U at t = %.0f δ/U\n", As[ipk], ts[ipk])
for t_ in (40.0, 60.0, 80.0, 100.0, 140.0, 160.0)
    i = argmin(abs.(ts .- t_))
    @printf("t = %5.0f: A = %.2e, γ(|z|<4δ) = %.3f, Ri_f = %.3f\n", ts[i], As[i], γm[i], Rif[i])
end
# cumulative flux Richardson number −∫B dt / ∫P dt (the mixing-efficiency
# proxy): through the growth to saturation, and over the whole run (after
# saturation the 2-D billow exchanges energy back and forth with the
# stratification, B changes sign and P passes through zero, so the
# instantaneous Ri_f is singular there)
# (accumulated from t = 10 δ/U, after the adjustment of the seed)
i10 = findfirst(>=(10.0), ts)
Rif_run = -cumsum(Bs[i10:end]) ./ cumsum(Ps[i10:end])
@printf("cumulative Ri_f from t = 10: %.3f to saturation (t = %.0f), %.3f to t = %.0f\n",
        Rif_run[ipk-i10+1], ts[ipk], Rif_run[end], ts[end])

# ---- figures ----------------------------------------------------------------
save_multiline(joinpath(OUT, "kh_amplitude.svg"), ts, [As], ["rms w / U"];
               xlabel = "t U / δ", ylabel = "rms w / U (log10)", logy = true,
               title = @sprintf("KH J = %.2f, Re = %.0f: σ = %.3f U/δ", J, Re, fit.rate))
save_multiline(joinpath(OUT, "kh_gamma.svg"), ts, [γm], ["γ, |z| ≤ 4δ"];
               xlabel = "t U / δ", ylabel = "turbulent fraction γ",
               title = "Turbulent fraction of the shear layer")
save_multiline(joinpath(OUT, "kh_rif.svg"), ts[i10:end], [Rif_run], ["−∫B dt / ∫P dt"];
               xlabel = "t U / δ", ylabel = "cumulative flux Richardson number",
               title = "Cumulative flux Richardson number of the layer")

# spanwise vorticity, three panels stacked (|z| < 6δ), blue-white-red
function rgb_panel(ω, kk; lim)
    Z = ω[reverse(kk), :]
    up = 2
    H, W = size(Z, 1) * up, size(Z, 2) * up
    img = Array{UInt8}(undef, H, W, 3)
    for j in 1:H, i in 1:W
        v = clamp(Z[cld(j, up), cld(i, up)] / lim, -1, 1)
        r, gg, b = v >= 0 ? (1.0, 1 - v, 1 - v) : (1 + v, 1 + v, 1.0)
        img[j, i, 1] = round(UInt8, 255r); img[j, i, 2] = round(UInt8, 255gg)
        img[j, i, 3] = round(UInt8, 255b)
    end
    return img
end
kk = findall(z -> abs(z) < 6δ, zc)
panels = [rgb_panel(snaps[t_], kk; lim = 1.0) for t_ in snaps_t if haskey(snaps, t_)]
sep = fill(UInt8(0), 4, size(panels[1], 2), 3)
write_png(joinpath(OUT, "kh_vorticity.png"),
          reduce((a, b) -> vcat(a, sep, b), panels))
println("figures in ", normpath(OUT), " (vorticity panels at t = ", join(snaps_t, ", "), ")")
