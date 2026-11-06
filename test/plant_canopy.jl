# Phase 48 — PlantCanopy (issue #22): leaf-area-density profiles
# and implicit canopy drag.

using Test
using Aeolus
using Aeolus.PlantCanopy
using Aeolus.Fields: set!

@testset "PlantCanopy: LAD and implicit drag (#22)" begin

    @testset "LAD profiles: LAI, published shape, layer conservation" begin
        h = 20.0
        # ∫₀ʰ a dz = LAI by an independent midpoint rule (2e4 cells)
        worst = 0.0
        for p in (LalicMihailovicLAD(5.0, h; z_m = 0.6h), BetaLAD(3.5, h; p = 6.5, q = 1.5))
            N = 20_000; dz = h / N
            s = sum(leaf_area_density(p, (k - 0.5) * dz) for k in 1:N) * dz
            worst = max(worst, abs(s / leaf_area_index(p) - 1))
        end
        @test worst < 1e-6
        # Lalic & Mihailovic (2004), the published formula by hand
        lm = LalicMihailovicLAD(5.0, h; z_m = 12.0)
        r(z) = (h - 12.0) / (h - z)
        @test leaf_area_density(lm, 6.0) ≈ lm.L_m * r(6.0)^6 * exp(6 * (1 - r(6.0)))
        @test leaf_area_density(lm, 16.0) ≈ lm.L_m * sqrt(r(16.0)) * exp(0.5 * (1 - r(16.0)))
        # layer LAD conserves LAI on any grid, also when a layer straddles the top
        worst = 0.0
        for p in (UniformLAD(4.0, h), lm),
            zf in (collect(0:6.25:50.0), cumsum([0.0; 0.5 .* 1.15 .^ (0:30)]))
            worst = max(worst, abs(sum(layer_leaf_area_density(p, zf) .* diff(zf)) / leaf_area_index(p) - 1))
        end
        @test worst < 1e-12
    end

    @testset "Implicit drag: exact quadratic-drag solution at any dt" begin
        C_d, a, U0 = 0.2, 1.0, 8.0
        exact(t) = U0 / (1 + C_d * a * U0 * t)
        worst = 0.0
        for dt in (0.01, 2.0, 600.0)
            nstep = max(round(Int, 60.0 / dt), 1)
            u = U0
            for n in 1:nstep
                u *= canopy_drag_factor(C_d, a, abs(u), dt)
                worst = max(worst, abs(u / exact(n * dt) - 1))
            end
        end
        @test worst < 1e-13
        # direction preserved, drag along −U
        u2 = implicit_canopy_drag(C_d, a, 3.0, -4.0, 1.0, 2.0)
        @test u2[1] / 3.0 ≈ u2[2] / -4.0 ≈ u2[3] / 1.0
    end

    @testset "Resolved drag on an LES grid: exact per layer, momentum budget" begin
        # apply_canopy_drag! as the LES calls it after the predictor (the
        # time stepper itself is not needed for the drag's own physics)
        Nx, Ny, Nz = 8, 6, 10
        mesh = RegularGrid(; size = (Nx, Ny, Nz), extent = (400.0, 300.0, 40.0),
                           halo = (2, 2, 2))
        bcs = FieldBoundaryConditions(:periodic, :periodic, :wall)
        lad = canopy_lad_field(mesh, UniformLAD(4.0, 20.0), trues(Nx, Ny))   # a = 0.2 m⁻¹
        C_d = 1.0                          # dt·Cd·a·U0 = 3.2 > 2: explicit is unstable
        U0 = 8.0; dt = 2.0; nsteps = 5
        sim = BoussinesqSimulation(mesh; bcs = bcs)
        set!(sim.u, U0)
        drag = ResolvedCanopyDrag(mesh, lad; C_d = C_d)
        uprev = copy(sim.u.data)
        sink_err = 0.0
        ii, jj, kk = 3:Nx+2, 3:Ny+2, 3:Nz+2
        for n in 1:nsteps
            F = apply_canopy_drag!(sim, drag, dt)
            ΔM = sum(sim.u.data[ii, jj, kk] .- uprev[ii, jj, kk]) * mesh.dx * mesh.dy * mesh.dz
            sink_err = max(sink_err, abs(F.Fx * dt - ΔM) / abs(ΔM))
            uprev .= sim.u.data
        end
        @test sink_err < 1e-10
        ucol = sim.u.data[3, 3, 3:Nz+2]
        uex = U0 / (1 + C_d * 0.2 * U0 * nsteps * dt)
        @test maximum(abs.(ucol[1:5] ./ uex .- 1)) < 1e-12
        @test all(ucol[6:end] .≈ U0)
    end

    @testset "1-D canopy column reproduces the Inoue / Harman-Finnigan profile" begin
        # dense uniform canopy, h = 20 m, LAI = 10, C_d = 0.25, β = u*/U_h = 0.3:
        # l = 2β³L_c and γ = (C_d a/2l²)^(1/3) = β/l
        h, LAI, C_d, β, U_h = 20.0, 10.0, 0.25, 0.3, 3.0
        a = LAI / h
        l = harman_finnigan_mixing_length(β, C_d, a)
        γ = canopy_attenuation_coefficient(C_d, a, l)
        @test γ ≈ β / l rtol = 1e-12
        U(z) = canopy_exponential_wind(z; h = h, u_h = U_h, C_d = C_d, a = a, β = β)
        errs = Float64[]; ustar = Float64[]; bal = 0.0
        for N in (200, 400)
            zf = collect(range(0, h; length = N + 1))
            lad = layer_leaf_area_density(UniformLAD(LAI, h), zf)
            sol = solve_canopy_wind_column(zf, lad; C_d = C_d, l_mix = l, u_top = U_h, ν = 1e-10)
            bal = max(bal, sol.residual, abs(sol.u_star^2 / sol.drag_integral - 1))  # stress = drag
            sel = sol.z .> h - 5 / γ
            push!(errs, maximum(abs.(sol.u[sel] ./ U.(sol.z[sel]) .- 1)))
            push!(ustar, sol.u_star / U_h)
        end
        @test bal < 1e-10
        @test errs[end] < 5e-4
        @test 3.5 < errs[1] / errs[2] < 4.5         # 2nd-order convergence
        @test ustar[end] ≈ β rtol = 1e-3
        # sparse canopy over no-slip ground: stress splits between leaves and ground
        zf = collect(range(0, h; length = 201))
        sol = solve_canopy_wind_column(zf, layer_leaf_area_density(UniformLAD(1.0, h), zf);
                                       C_d = C_d, l_mix = 1.0, u_top = U_h, ν = 1e-10,
                                       bottom = :no_slip)
        @test sol.tau_ground > 0
        @test sol.u_star^2 ≈ sol.drag_integral + sol.tau_ground rtol = 1e-10
    end
end
