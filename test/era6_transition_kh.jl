# Era VI, Phase 59 (P59-c, #180) — Transition diagnostics: turbulent
# fraction γ(z), intermittency, Ri_f / Re_b / V_TKE, growth-rate fit.
#
#   3. γ: laminar parallel shear 0 exactly, decaying HIT > 0.95, a
#      half-turbulent synthetic field 0.50 ± 0.03;
#   4. time-series diagnostics: intermittency 0.3 exactly for a 0.3 duty
#      cycle; Re_b, V_TKE and Ri_f (resolved + SGS) against hand values;
#      growth_rate on a synthetic series.
#
# The Kelvin–Helmholtz and Miles–Howard validation runs (and the SGS effect
# on the KH growth rate) are in validation/era6_transition_kh.jl.

using Test
using Aeolus
using Aeolus.Fields: set!, interior_view
using Aeolus.Transition
using Random
using Printf

@testset "Transition diagnostics (P59-c, #180)" begin

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
end
