# Era VI, Phase 59 (P59-c, #180) — Transition: turbulent fraction γ(z),
# intermittency, Ri_f / Re_b / V_TKE, growth-rate fit; Kelvin–Helmholtz and
# Miles–Howard validation.
#
#   1. unstratified KH (tanh shear layer, αδ = 0.4446, Re = 1000,
#      128×1×256): growth rate within 5 % of the inviscid 0.1897 U/δ
#      (Michalke 1964);
#   2. Miles–Howard / Drazin neutral curve at αδ = 0.7071 (J_c = α²(1−α²)
#      = 1/4 there): J = 0.15 grows faster than 0.03 U/δ, J = 0.30 does not
#      grow over 5 e-folding times of the J = 0 case;
#   3. γ: laminar parallel shear 0 exactly, decaying HIT > 0.95, a
#      half-turbulent synthetic field 0.50 ± 0.03;
#   4. time-series diagnostics: intermittency 0.3 exactly for a 0.3 duty
#      cycle; Re_b, V_TKE and Ri_f (resolved + SGS) against hand values;
#      growth_rate on a synthetic series;
#   5. the SGS effect on the KH growth rate at 64×1×128: Smagorinsky
#      C_s = 0.1 vs no SGS, quoted and not asserted; the σ-model (P59-b,
#      #184), whose ν_t vanishes in any 2-D flow, within 3 % of no SGS.
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
    @testset "3. turbulent fraction γ(z)" begin
        N = 32
        bcs = FieldBoundaryConditions(:periodic, :periodic, :periodic)

        # laminar parallel shear: ω uniform in each plane → γ = 0 exactly
        m1 = RegularGrid(; size = (16, 16, 16), extent = (1.0, 1.0, 1.0), halo = (2, 2, 2))
        s1 = BoussinesqSimulation(m1; bcs = FieldBoundaryConditions(:periodic, :periodic, :wall))
        set!(s1.u, (x, y, z) -> 3 * tanh(8 * (z - 0.5)))
        γ1 = turbulent_fraction(s1)
        @test all(==(0.0), γ1)
        @test all(==(0.0), turbulent_fraction(s1; normalization = :volume))

        # decaying HIT: a random solenoidal field (curl of a smoothed random
        # vector potential) decaying for 20 steps at Re_λ ~ O(10)
        function random_hit!(sim, rng)
            m = sim.mesh
            A = [randn(rng, m.Nx, m.Ny, m.Nz) for _ in 1:3]
            for a in A, _ in 1:2                        # 2 periodic 1-2-1 passes: k ~ N/6 peak
                for d in 1:3
                    a .= 0.25 .* circshift(a, ntuple(i -> i == d ? 1 : 0, 3)) .+ 0.5 .* a .+
                         0.25 .* circshift(a, ntuple(i -> i == d ? -1 : 0, 3))
                end
            end
            # A_x, A_y, A_z on the edges; u = ∇×A on the faces (exactly ∇·u = 0)
            Ax, Ay, Az = A
            ui = (circshift(Az, (0, -1, 0)) .- Az) ./ m.dy .- (circshift(Ay, (0, 0, -1)) .- Ay) ./ m.dz
            vi = (circshift(Ax, (0, 0, -1)) .- Ax) ./ m.dz .- (circshift(Az, (-1, 0, 0)) .- Az) ./ m.dx
            wi = (circshift(Ay, (-1, 0, 0)) .- Ay) ./ m.dx .- (circshift(Ax, (0, -1, 0)) .- Ax) ./ m.dy
            s = 1 / sqrt(sum(abs2, ui) / length(ui))
            interior_view(sim.u) .= s .* ui; interior_view(sim.v) .= s .* vi
            interior_view(sim.w) .= s .* wi
            return sim
        end
        mh = RegularGrid(; size = (N, N, N), extent = (2π, 2π, 2π), halo = (2, 2, 2))
        sh = BoussinesqSimulation(mh; ν = 2e-2, bcs = bcs)
        random_hit!(sh, MersenneTwister(180))
        ke0 = kinetic_energy(sh)
        for _ in 1:20
            step_rk3!(sh, 0.02)
        end
        @test kinetic_energy(sh) < ke0                 # decaying
        γh = turbulent_fraction(sh)
        @printf("    γ decaying HIT 32³: min %.4f, mean %.4f\n", minimum(γh), sum(γh) / N)
        @test minimum(γh) > 0.95
        @test all(x -> 0 <= x <= 1, γh)

        # half-turbulent synthetic field: the random field in x < 4π, at rest
        # in x ≥ 4π. The cut is a vortex sheet, and the centred stencil marks
        # about 1.8 cell columns per interface turbulent (measured: γ = 0.528
        # with 64 columns); with 128 columns that is 0.014 of the plane.
        mhalf = RegularGrid(; size = (4N, N, N), extent = (8π, 2π, 2π), halo = (2, 2, 2))
        sf = BoussinesqSimulation(mhalf; bcs = bcs)
        random_hit!(sf, MersenneTwister(59))
        for f in (sf.u, sf.v, sf.w)
            interior_view(f)[2N+1:end, :, :] .= 0
        end
        γhalf = turbulent_fraction(sf)
        @printf("    γ half-turbulent: min %.4f, max %.4f\n", extrema(γhalf)...)
        @test all(x -> abs(x - 0.5) <= 0.03, γhalf)

        # laminar mask = fluid ∧ ¬turbulent; solid cells are neither
        fill_halos!(sf.u, bcs); fill_halos!(sf.v, bcs); fill_halos!(sf.w, bcs)
        fluid = trues(4N, N, N); fluid[:, :, 1:4] .= false; fluid[1:8, 1:8, :] .= false
        ind = Array{Bool,3}(undef, 4N, N, N); lam = similar(ind)
        turbulent_indicator!(ind, sf.u, sf.v, sf.w; fluid = fluid)
        laminar_mask!(lam, sf.u, sf.v, sf.w; fluid = fluid)
        @test !any(ind .& .!fluid) && !any(lam .& .!fluid)
        @test all((ind .⊻ lam) .== fluid)
        γf = turbulent_fraction(ind; fluid = fluid)
        @test all(isnan, γf[1:4])                      # levels without fluid
        @test all(k -> γf[k] ≈ count(ind[:, :, k]) / count(fluid[:, :, k]), 5:N)
        @test_throws ArgumentError turbulent_fraction(sf; indicator = :nope)
    end

    # -----------------------------------------------------------------------
    @testset "4. time-series and flux diagnostics" begin
        # intermittency: duty cycle 0.3, exactly
        sig = repeat([1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 100)
        @test intermittency_factor(sig; threshold = 0.5) === 0.3
        t = collect(0.0:1.0:1000.0); sigt = vcat(sig, 0.0)  # the last sample closes the record
        @test intermittency_factor(t, sigt; threshold = 0.5) === 0.3
        @test intermittency_factor(fill(2.0, 7); threshold = 1.0) == 1.0
        @test intermittency_factor(zeros(7); threshold = 0.0) == 0.0

        # Re_b = ε/(νN²): ε = 1e-6 W/kg, ν = 1.5e-5, N = 1e-2 s⁻¹ → 66.67
        @test buoyancy_reynolds(1e-6, 1.5e-5, 1e-4) ≈ 1e-6 / (1.5e-5 * 1e-4) rtol = 1e-15
        @test buoyancy_reynolds(1e-6, 1.5e-5, 1e-4) ≈ 2000 / 3 rtol = 1e-12
        @test buoyancy_reynolds(1e-6, 1.5e-5, 0.0) == Inf
        # V_TKE (Sun et al. 2012): σ = (0.3, 0.4, 0.5) m/s → e = 0.25, V = 0.5
        @test turbulence_velocity_scale(0.3, 0.4, 0.5) ≈ 0.5 rtol = 1e-15
        @test turbulence_velocity_scale(0.25) == 0.5
        @test_throws DomainError turbulence_velocity_scale(-1.0)
        @test grid_reynolds_number(4.0, 6.0, 2.0, 1.0) == 8.0

        # Ri_f on an analytic field: u = S z + a sin kx, w = b sin kx,
        # θ = Γ z + c sin kx, constant ν_t. Hand values: on the C grid u′ at
        # the centre is a sin(kx) cos(kΔx/2) and w′ = b sin kx, so
        # ⟨u′w′⟩ = ab cos(kΔx/2)/2, ⟨w′θ′⟩ = bc/2; the SGS fluxes are
        # −ν_t (S + ⟨∂w/∂x⟩) = −ν_t S and −(ν_t/Pr_t) Γ.
        Nx, Ny, Nz = 32, 4, 16
        m = RegularGrid(; size = (Nx, Ny, Nz), extent = (1.0, 0.125, 0.5), halo = (2, 2, 2))
        S, a, b, c, Γ, k = 0.02, 0.3, 0.2, -0.5, 0.01, 2π
        νt0, Pr, θ₀, g = 1.7e-2, 0.7, 300.0, 9.81
        u = Field(FaceX(), m); v = Field(FaceY(), m); w = Field(FaceZ(), m)
        θ = Field(CellCenter(), m); νt = Field(CellCenter(), m)
        # fill everything, halos included, from the analytic functions
        for (f, fn, off) in ((u, (x, z) -> S * z + a * sin(k * x), (0.0, 0.5)),
                             (w, (x, z) -> b * sin(k * x), (0.5, 0.0)),
                             (θ, (x, z) -> θ₀ + Γ * z + c * sin(k * x), (0.5, 0.5)))
            for K in axes(f.data, 3), J in axes(f.data, 2), I in axes(f.data, 1)
                x = (I - m.Hx - 1 + off[1]) * m.dx; z = (K - m.Hz - 1 + off[2]) * m.dz
                f.data[I, J, K] = fn(x, z)
            end
        end
        fill!(νt.data, νt0)
        r = flux_richardson(u, v, w, θ; θ₀ = θ₀, g = g, νt = νt, Pr_t = Pr)
        uw = a * b * cos(k * m.dx / 2) / 2 - νt0 * S
        wθ = b * c / 2 - νt0 / Pr * Γ
        Rif = -(g / θ₀) * wθ / (-uw * S)
        @test all(kk -> isapprox(r.uw[kk], uw; rtol = 1e-12), 1:Nz)
        @test all(kk -> isapprox(r.wθ[kk], wθ; rtol = 1e-12), 1:Nz)
        @test all(kk -> isapprox(r.Ri_f[kk], Rif; rtol = 1e-10), 1:Nz)
        @printf("    Ri_f analytic: %.10f (hand %.10f)\n", r.Ri_f[Nz ÷ 2], Rif)
        # resolved only
        r0 = flux_richardson(u, v, w, θ; θ₀ = θ₀, g = g)
        @test all(kk -> isapprox(r0.Ri_f[kk],
                                 -(g / θ₀) * (b * c / 2) / (-(a * b * cos(k * m.dx / 2) / 2) * S);
                                 rtol = 1e-10), 1:Nz)

        # growth_rate: exponential growth that saturates (tanh), with a
        # decaying transient at the start
        tt = collect(0.0:0.1:80.0)
        σ = 0.19
        A = @. 1e-4 * exp(σ * tt) / sqrt(1 + (1e-4 * exp(σ * tt) / 0.2)^2) + 3e-5 * exp(-tt)
        f = growth_rate(tt, A)
        @test f.r2 >= 0.999
        @test abs(f.rate - σ) / σ < 0.01
        @test f.window[1] > 1 && f.window[2] < 40      # excludes the transient and the saturation
        fw = growth_rate(tt, A; window = (8.0, 20.0))
        @test fw.rate ≈ σ rtol = 1e-3
        @test_throws ArgumentError growth_rate(tt, -A)

        # supercritical fraction counts the field of gradient_richardson!
        Ri = reshape(repeat([0.1, 0.3, 0.25, 1.0], 8), 4, 2, 4)
        fl = trues(4, 2, 4); fl[1, :, 2] .= false
        @test supercritical_fraction(Ri) == fill(0.5, 4)
        @test supercritical_fraction(Ri; fluid = fl)[2] == 4 / 6
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
