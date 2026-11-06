# Era VI, Phase 59 (P59-c, #180) — Transition diagnostics: turbulent
# fraction γ(z), intermittency, Ri_f / Re_b / V_TKE, growth-rate fit.
#
#   3. γ: laminar parallel shear 0 exactly, a half-turbulent synthetic
#      field 0.50 ± 0.04;
#   4. time-series diagnostics: intermittency 0.3 exactly for a 0.3 duty
#      cycle; Re_b, V_TKE and Ri_f (resolved + SGS) against hand values;
#      growth_rate on a synthetic series.
#
# The Kelvin–Helmholtz and Miles–Howard validation runs (and the SGS effect
# on the KH growth rate) are in validation/transition_kh.jl.

using Test
using Aeolus
using Aeolus.Fields: set!, interior_view
using Aeolus.Transition
using Random

@testset "Transition diagnostics (P59-c, #180)" begin

    # -----------------------------------------------------------------------
    @testset "3. turbulent fraction γ(z)" begin
        N = 16
        bcs = FieldBoundaryConditions(:periodic, :periodic, :periodic)
        # laminar parallel shear: ω uniform in each plane → γ = 0 exactly
        m1 = RegularGrid(; size = (8, 8, 16), extent = (1.0, 1.0, 1.0), halo = (2, 2, 2))
        s1 = BoussinesqSimulation(m1)   # any u(z) is plane-uniform, whatever the z halos
        set!(s1.u, (x, y, z) -> 3 * tanh(8 * (z - 0.5)))
        @test all(==(0.0), turbulent_fraction(s1))

        # half-turbulent synthetic field: a random solenoidal field (curl of a
        # smoothed random vector potential) in x < L/2, at rest beyond. The
        # cut is a vortex sheet, and the centred stencil marks about 1.8 cell
        # columns per interface turbulent: with 64 columns that is 0.028 of
        # the plane, so γ = 0.5 within 0.04 (0.03 with 128 columns before the
        # grid was halved for the fast suite).
        mhalf = RegularGrid(; size = (4N, N, N), extent = (8π, 2π, 2π), halo = (2, 2, 2))
        sf = BoussinesqSimulation(mhalf; bcs = bcs)
        rng = MersenneTwister(59)
        A = [randn(rng, 4N, N, N) for _ in 1:3]
        for a in A, _ in 1:2, d in 1:3
            a .= 0.25 .* circshift(a, ntuple(i -> i == d ? 1 : 0, 3)) .+ 0.5 .* a .+
                 0.25 .* circshift(a, ntuple(i -> i == d ? -1 : 0, 3))
        end
        Ax, Ay, Az = A; m = mhalf
        interior_view(sf.u) .= (circshift(Az, (0, -1, 0)) .- Az) ./ m.dy .- (circshift(Ay, (0, 0, -1)) .- Ay) ./ m.dz
        interior_view(sf.v) .= (circshift(Ax, (0, 0, -1)) .- Ax) ./ m.dz .- (circshift(Az, (-1, 0, 0)) .- Az) ./ m.dx
        interior_view(sf.w) .= (circshift(Ay, (-1, 0, 0)) .- Ay) ./ m.dx .- (circshift(Ax, (0, -1, 0)) .- Ax) ./ m.dy
        for f in (sf.u, sf.v, sf.w)
            interior_view(f)[2N+1:end, :, :] .= 0
        end
        @test all(x -> abs(x - 0.5) <= 0.04, turbulent_fraction(sf))

    end

    # -----------------------------------------------------------------------
    @testset "4. time-series and flux diagnostics" begin
        # intermittency: duty cycle 0.3, exactly
        sig = repeat([1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 100)
        @test intermittency_factor(sig; threshold = 0.5) === 0.3
        t = collect(0.0:1.0:1000.0); sigt = vcat(sig, 0.0)  # the last sample closes the record
        @test intermittency_factor(t, sigt; threshold = 0.5) === 0.3

        # Re_b = ε/(νN²): ε = 1e-6 W/kg, ν = 1.5e-5, N = 1e-2 s⁻¹ → 66.67
        @test buoyancy_reynolds(1e-6, 1.5e-5, 1e-4) ≈ 2000 / 3 rtol = 1e-12
        @test buoyancy_reynolds(1e-6, 1.5e-5, 0.0) == Inf
        # V_TKE (Sun et al. 2012): σ = (0.3, 0.4, 0.5) m/s → e = 0.25, V = 0.5
        @test turbulence_velocity_scale(0.3, 0.4, 0.5) ≈ 0.5 rtol = 1e-15
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
        @test all(kk -> isapprox(r.uw[kk], uw; rtol = 1e-12) && isapprox(r.wθ[kk], wθ; rtol = 1e-12) &&
                        isapprox(r.Ri_f[kk], Rif; rtol = 1e-10), 1:Nz)
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

        # supercritical fraction counts the field of gradient_richardson!
        Ri = reshape(repeat([0.1, 0.3, 0.25, 1.0], 8), 4, 2, 4)
        fl = trues(4, 2, 4); fl[1, :, 2] .= false
        @test supercritical_fraction(Ri) == fill(0.5, 4)
        @test supercritical_fraction(Ri; fluid = fl)[2] == 4 / 6
    end
end
