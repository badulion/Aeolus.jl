# Era VI, Phase 59 (P59-c, #180) — Transition: Kelvin–Helmholtz and
# Miles–Howard validation of the growth-rate diagnostics.
#
#   1. unstratified KH (tanh shear layer, αδ = 0.4446, Re = 1000,
#      128×1×256): growth rate within 5 % of the inviscid 0.1897 U/δ
#      (Michalke 1964);
#   2. Miles–Howard / Drazin neutral curve at αδ = 0.7071 (J_c = α²(1−α²)
#      = 1/4 there): J = 0.15 grows faster than 0.03 U/δ, J = 0.30 does not
#      grow over 5 e-folding times of the J = 0 case;
#   5. the SGS effect on the KH growth rate at 64×1×128: Smagorinsky
#      C_s = 0.1 vs no SGS, quoted and not asserted; the σ-model (P59-b,
#      #184), whose ν_t vanishes in any 2-D flow, within 3 % of no SGS.
#
# The diagnostics stay tested in test/transition_kh.jl (3. turbulent
# fraction γ(z); 4. intermittency, Re_b, V_TKE, Ri_f, growth_rate against
# hand values).
#
# The KH runs hold the base state (u = U tanh(z/δ), θ = θ₀ + Δθ tanh(z/δ))
# against its own viscous diffusion with a body force −ν∇²(base), the
# frozen-parallel-flow assumption of the linear-stability references, and
# use free-slip walls at z = ±Lz/2 (≥ 8.9δ, where the eigenfunction has
# decayed to e^{−α|z|} ≲ 2e-3).

using Aeolus
using Aeolus.Fields: set!, interior_view
using Aeolus.Operators: laplacian
using Aeolus.Transition
using Random
using Printf

# ---------------------------------------------------------------------------
# Kelvin–Helmholtz helper
# ---------------------------------------------------------------------------

"""
Run the 2-D (Ny = 1) tanh shear layer with bulk Richardson number `J`
(N²(z) = J U²/δ² sech²(z/δ), so the minimum Ri_g, at z = 0, is J) and
return the times and the rms of w (the perturbation amplitude; w has no
base-state part).
"""
function kh_run(; αδ = 0.4446, J = 0.0, Re = 1000.0, Nx = 128, Nz = 256, ε = 1e-4,
                tend = 45.0, cfl = 0.6, C_s = 0.0, sgs = nothing)
    δ = 1.0; U = 1.0
    Lx = 2π * δ / αδ; dx = Lx / Nx; Lz = Nz * dx
    mesh = RegularGrid(; size = (Nx, 1, Nz), extent = (Lx, dx, Lz),
                       origin = (0.0, 0.0, -Lz / 2), halo = (2, 2, 2))
    θ₀ = 300.0; g = 9.81
    bcs = FieldBoundaryConditions(:periodic, :periodic, :wall)
    sim = BoussinesqSimulation(mesh; ν = U * δ / Re, θ₀ = J > 0 ? θ₀ : 0.0, g_accel = g,
                               bcs = bcs, C_s = C_s)
    Δθ = J * U^2 * θ₀ / (g * δ)
    set!(sim.u, (x, y, z) -> U * tanh(z / δ))
    set!(sim.θ, (x, y, z) -> θ₀ + Δθ * tanh(z / δ))
    set!(sim.w, (x, y, z) -> ε * U * sin(αδ * x / δ) * exp(-(z / δ)^2))
    # −ν∇²(base): the base state is a steady solution of the forced problem
    fill_halos!(sim.u, bcs); fill_halos!(sim.θ, bcs)
    fu = Field(FaceX(), mesh); fθ = Field(CellCenter(), mesh)
    laplacian(fu, sim.u); laplacian(fθ, sim.θ)
    fu.data .*= -sim.ν; fθ.data .*= -sim.ν
    hold_base!(s, Δt) = (s.tu.data .+= fu.data; s.tθ.data .+= fθ.data; nothing)
    dt = cfl * dx / U; n = ceil(Int, tend / dt); dt = tend / n
    ts = Float64[]; As = Float64[]
    for _ in 1:n
        step_rk3!(sim, dt; body_force! = hold_base!, sgs = sgs)
        wi = interior_view(sim.w)
        push!(ts, sim.t); push!(As, sqrt(sum(abs2, wi) / length(wi)))
    end
    return ts, As, sim
end

@testset "Transition (P59-c, #180)" begin

    # -----------------------------------------------------------------------
    @testset "1. unstratified Kelvin–Helmholtz: Michalke (1964)" begin
        t0 = time()
        ts, As, sim = kh_run(; αδ = 0.4446, J = 0.0, tend = 45.0)
        fit = growth_rate(ts, As)
        σ_ref = 0.1897                       # inviscid, Michalke (1964)
        short = (σ_ref - fit.rate) / σ_ref
        @printf("    KH αδ=0.4446 Re=1000 128×1×256: σ = %.4f U/δ (R² = %.6f, window %.1f-%.1f δ/U, %d samples), viscous+numerical shortfall %.2f %% vs 0.1897; %.1f s\n",
                fit.rate, fit.r2, fit.window..., fit.npoints, 100short, time() - t0)
        @test fit.r2 >= 0.999
        @test abs(fit.rate - σ_ref) / σ_ref < 0.05
        @test fit.rate < σ_ref               # viscosity can only slow the mode
        # the linear window spans ≥ 4 e-foldings of clean growth
        @test fit.rate * (fit.window[2] - fit.window[1]) > 4
        @test max_divergence(sim) < 1e-10
    end

    # -----------------------------------------------------------------------
    @testset "2. Miles–Howard / Drazin neutral curve at αδ = 0.7071" begin
        α = 0.7071
        @test α^2 * (1 - α^2) ≈ 0.25 atol = 1e-4     # Drazin (1958): J_c(α) = α²(1−α²)
        t0 = time()
        ts0, A0, _ = kh_run(; αδ = α, J = 0.0, tend = 36.0)
        f0 = growth_rate(ts0, A0)
        τ0 = 1 / f0.rate
        ts15, A15, _ = kh_run(; αδ = α, J = 0.15, tend = 60.0)
        # the J = 0.15 amplitude carries the gravity-wave oscillation of the
        # seed's stable modes, so R² of any window stays < 0.999: fit the
        # explicit window after the adjustment
        f15 = growth_rate(ts15, A15; window = (15.0, 60.0))
        ts30, A30, _ = kh_run(; αδ = α, J = 0.30, tend = 5τ0)
        i1 = findfirst(>=(τ0), ts30)
        f30 = growth_rate(ts30, A30; window = (τ0, 5τ0))
        @printf("    MH αδ=0.7071: σ(J=0) = %.4f (R² %.6f); σ(J=0.15) = %.4f (R² %.4f, 15-60 δ/U); J=0.30: A(5τ₀)/A(τ₀) = %.3f, fitted σ = %.4f over τ₀-5τ₀ (τ₀ = %.1f); %.1f s\n",
                f0.rate, f0.r2, f15.rate, f15.r2, A30[end] / A30[i1], f30.rate, τ0, time() - t0)
        @test f0.r2 >= 0.999
        @test f15.rate > 0.03
        @test f15.rate < f0.rate                       # stratification stabilises
        @test A30[end] < A30[i1]                       # no growth over 5 e-folding times
        @test f30.rate < 0
    end

    # -----------------------------------------------------------------------
    @testset "5. SGS effect on the KH growth rate (Smagorinsky quoted; σ-model asserted)" begin
        t0 = time()
        ts, A, _ = kh_run(; Nx = 64, Nz = 128)
        tss, As, _ = kh_run(; Nx = 64, Nz = 128, C_s = 0.1)
        tsσ, Aσ, simσ = kh_run(; Nx = 64, Nz = 128, sgs = SigmaModel())
        f = growth_rate(ts, A); fs = growth_rate(tss, As); fσ = growth_rate(tsσ, Aσ)
        @printf("    KH 64×1×128: σ no SGS = %.4f, Smagorinsky C_s=0.1 = %.4f (%.2f %%), σ-model C_σ=1.35 = %.4f (%.2e %%), max ν_t(σ) = %.1e; %.1f s\n",
                f.rate, fs.rate, 100 * (fs.rate - f.rate) / f.rate, fσ.rate,
                100 * (fσ.rate - f.rate) / f.rate, maximum(interior_view(simσ.νt)), time() - t0)
        @test isfinite(f.rate) && isfinite(fs.rate)
        # P59-b (#184): the flow is 2-D (N_y = 1, v = 0), so g has rank ≤ 2,
        # σ₃ = 0 and the σ-model adds nothing; the growth rate is the
        # molecular one (target: within 3 %)
        @test abs(fσ.rate - f.rate) <= 0.03 * f.rate
        @test maximum(interior_view(simσ.νt)) == 0
    end
end
