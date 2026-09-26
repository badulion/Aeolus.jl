# Era V, Phase 48 — PlantCanopy (issue #22): leaf-area-density profiles
# and implicit canopy drag.

using Test, Printf
using Aeolus
using Aeolus.PlantCanopy
using Aeolus.Fields: set!

@testset "Era V — PlantCanopy: LAD and implicit drag (#22)" begin

    @testset "LAD profiles integrate to LAI and have the published shape" begin
        h = 20.0
        profs = (UniformLAD(4.0, h),
                 UniformLAD(3.0, h; z_base = 8.0),
                 LalicMihailovicLAD(5.0, h; z_m = 0.6h),
                 LalicMihailovicLAD(2.0, h; z_m = 0.8h),
                 BetaLAD(4.0, h; p = 3, q = 2),
                 BetaLAD(6.0, h; p = 1, q = 1),
                 BetaLAD(3.5, h; p = 6.5, q = 1.5))
        worst = 0.0
        for p in profs
            # ∫₀ʰ a dz = LAI — here with an independent brute-force midpoint
            # rule (1e5 cells) rather than the module's own quadrature.
            N = 100_000; dz = h / N
            s = sum(leaf_area_density(p, (k - 0.5) * dz) for k in 1:N) * dz
            worst = max(worst, abs(s / leaf_area_index(p) - 1))
            @test s ≈ leaf_area_index(p) rtol = 1e-6
            # Cumulative leaf area: 0 at top, LAI at ground, monotone.
            @test cumulative_leaf_area(p, h) == 0
            @test cumulative_leaf_area(p, 0.0) ≈ leaf_area_index(p) rtol = 1e-12
            Λ = [cumulative_leaf_area(p, z) for z in 0:0.5:h]
            @test all(diff(Λ) .<= 1e-14)
            # No leaves outside the canopy.
            @test leaf_area_density(p, h + 0.1) == 0
            @test leaf_area_density(p, -0.1) == 0
            @test canopy_height(p) == h
        end
        @info "LAD normalisation (midpoint, 1e5 cells)" worst_rel_err = worst

        # Uniform: a = LAI / crown depth.
        @test leaf_area_density(UniformLAD(4.0, h), 10.0) ≈ 0.2
        @test leaf_area_density(UniformLAD(3.0, h; z_base = 8.0), 5.0) == 0
        @test leaf_area_density(UniformLAD(3.0, h; z_base = 8.0), 10.0) ≈ 0.25

        # Lalic & Mihailovic (2004): maximum L_m at z_m, n = 6 below and
        # n = 1/2 above, vanishing at the top.
        lm = LalicMihailovicLAD(5.0, h; z_m = 12.0)
        zs = range(0, h; length = 4001)
        a = leaf_area_density.(Ref(lm), zs)
        @test zs[argmax(a)] ≈ 12.0 atol = h / 4000
        @test leaf_area_density(lm, 12.0) ≈ lm.L_m
        @test leaf_area_density(lm, h) == 0
        @test leaf_area_density(lm, h - 1e-6) < 1e-3 * lm.L_m
        # The published formula, evaluated by hand at two heights.
        r(z) = (h - 12.0) / (h - z)
        @test leaf_area_density(lm, 6.0)  ≈ lm.L_m * r(6.0)^6 * exp(6 * (1 - r(6.0)))
        @test leaf_area_density(lm, 16.0) ≈ lm.L_m * sqrt(r(16.0)) * exp(0.5 * (1 - r(16.0)))
        # n = 6 below makes the lower trunk space sparse and the crown dense:
        # the lowest z_m/2 (30 % of the depth) holds < 15 % of the leaf
        # area, the 40 % of the depth above z_m holds > 40 % of it.
        @test leaf_area_index(lm) - cumulative_leaf_area(lm, 6.0) < 0.15 * leaf_area_index(lm)
        @test cumulative_leaf_area(lm, 12.0) > 0.40 * leaf_area_index(lm)

        # Beta distribution: known moments of Beta(p, q).
        for (pp, qq) in ((3.0, 2.0), (2.0, 5.0), (6.5, 1.5))
            b = BetaLAD(4.0, h; p = pp, q = qq)
            N = 200_000; dz = h / N
            zc = [(k - 0.5) * dz for k in 1:N]
            w  = leaf_area_density.(Ref(b), zc)
            zmean = sum(w .* zc) / sum(w)
            @test zmean ≈ h * pp / (pp + qq) rtol = 1e-6
            @test zc[argmax(w)] ≈ h * (pp - 1) / (pp + qq - 2) atol = 2dz
        end
        # Beta(1,1) is the uniform canopy.
        @test leaf_area_density(BetaLAD(6.0, h; p = 1, q = 1), 7.3) ≈
              leaf_area_density(UniformLAD(6.0, h), 7.3)

        @test_throws ArgumentError UniformLAD(-1.0, h)
        @test_throws ArgumentError LalicMihailovicLAD(1.0, h; z_m = h)
        @test_throws ArgumentError BetaLAD(1.0, h; p = 0.5)
    end

    @testset "Layer LAD conserves LAI on any grid" begin
        h = 20.0
        for p in (UniformLAD(4.0, h), LalicMihailovicLAD(5.0, h; z_m = 12.0),
                  BetaLAD(4.0, h; p = 3, q = 2))
            # Coarse, canopy-top-straddling, and stretched grids.
            for zf in (collect(0:6.25:50.0),             # the urban example's dz
                       collect(range(0, 31; length = 4)),
                       cumsum([0.0; 0.5 .* 1.15 .^ (0:30)]))
                a = layer_leaf_area_density(p, zf)
                col = sum(a .* diff(zf))
                @test col ≈ leaf_area_index(p) rtol = 1e-12
            end
        end
        # A point sample at layer centres would not conserve LAI when a layer
        # straddles the top: with dz = 6.25 m and h = 20 m the top canopy
        # layer is [18.75, 25] and centre-sampling puts zero leaf there.
        u = UniformLAD(4.0, h)
        zf = collect(0:6.25:50.0)
        sampled = sum(leaf_area_density.(Ref(u), zf[1:end-1] .+ 3.125) .* 6.25)
        @test sampled ≈ 3.75            # 3 of 3.2 layers: 6 % of LAI lost
        @test sum(layer_leaf_area_density(u, zf) .* 6.25) ≈ 4.0
    end

    @testset "Implicit drag: exact quadratic-drag solution at any dt" begin
        C_d, a, U0 = 0.2, 1.0, 8.0
        α = C_d * a
        exact(t) = U0 / (1 + α * U0 * t)
        worst = 0.0
        for dt in (0.01, 0.5, 2.0, 10.0, 600.0)
            nstep = round(Int, 60.0 / dt) |> x -> max(x, 1)
            u = U0
            for n in 1:nstep
                u *= canopy_drag_factor(C_d, a, abs(u), dt)
                worst = max(worst, abs(u / exact(n * dt) - 1))
            end
        end
        # The factor is the exact solution, so the error is round-off
        # independent of dt (explicit Euler would be off by O(1) at dt = 2 s).
        @test worst < 1e-13
        @info "implicit drag vs U0/(1+CdaU0t)" max_rel_err = worst

        # dt = 2 s, the urban example's step: explicit Euler oscillates with
        # growing amplitude (dt·Cd·a·U0 = 3.2 > 2), implicit decays monotonically.
        ue = U0; ui = U0; dt = 2.0
        grew = false; monotone = true
        for n in 1:6
            ue += dt * canopy_drag_force(C_d, a, ue, 0.0, 0.0)[1]
            ui_new = implicit_canopy_drag(C_d, a, ui, 0.0, 0.0, dt)[1]
            monotone &= 0 < ui_new < ui
            ui = ui_new
            grew |= abs(ue) > 10U0
        end
        @test grew
        @test monotone
        @test ui ≈ exact(12.0) rtol = 1e-13
        @info "dt = 2 s, 6 steps" explicit = ue implicit = ui exact = exact(12.0)

        # Frozen-speed (linear, Rayleigh) limit: the factor is backward Euler
        # for dU/dt = -k U and converges to the analytic exp(-k t) at first
        # order — halving dt halves the error.
        k = α * 5.0
        errs = map((0.4, 0.2, 0.1, 0.05)) do dt
            n = round(Int, 4.0 / dt)
            abs(canopy_drag_factor(C_d, a, 5.0, dt)^n - exp(-k * 4.0))
        end
        ratios = errs[1:end-1] ./ errs[2:end]
        @test all(r -> 1.9 < r < 2.1, ratios)
        @info "frozen-speed drag vs exp(-kt), dt halving" errs ratios

        # Direction is preserved and the drag is along -U.
        u2 = implicit_canopy_drag(C_d, a, 3.0, -4.0, 1.0, 2.0)
        @test u2[1] / 3.0 ≈ u2[2] / -4.0 ≈ u2[3] / 1.0
        F = canopy_drag_force(C_d, a, 3.0, -4.0, 0.0)
        @test F[1] ≈ -C_d * a * 5.0 * 3.0 && F[2] ≈ C_d * a * 5.0 * 4.0
        # Momentum removed per step → Cd a |U| U as dt → 0.
        for dt in (1e-2, 1e-4)
            du = (implicit_canopy_drag(C_d, a, 5.0, 0.0, 0.0, dt)[1] - 5.0) / dt
            @test du ≈ -C_d * a * 25.0 rtol = 2 * dt * α * 5.0
        end
    end

    @testset "Resolved drag in the LES: stable at dt = 2 s, exact per layer" begin
        Nx, Ny, Nz = 8, 6, 10
        mesh = RegularGrid(; size = (Nx, Ny, Nz), extent = (400.0, 300.0, 40.0),
                           halo = (2, 2, 2))
        bcs = FieldBoundaryConditions(:periodic, :periodic, :wall)
        prof = UniformLAD(4.0, 20.0)                   # a = 0.2 m⁻¹, 5 layers
        mask = trues(Nx, Ny)
        lad = canopy_lad_field(mesh, prof, mask)
        @test all(sum(lad; dims = 3) .* mesh.dz .≈ 4.0)
        C_d = 1.0                          # dt·Cd·a·U0 = 2·1·0.2·8 = 3.2 > 2
        U0 = 8.0; dt = 2.0; nsteps = 30

        # (a) implicit, via post_predictor!
        sim = BoussinesqSimulation(mesh; bcs = bcs)
        set!(sim.u, U0)
        drag = ResolvedCanopyDrag(mesh, lad; C_d = C_d)
        F = Ref((Fx = 0.0, Fy = 0.0, Fz = 0.0))
        uprev = copy(sim.u.data)
        sink_err = 0.0
        for n in 1:nsteps
            step!(sim, dt; post_predictor! = (s, Δt) -> (F[] = apply_canopy_drag!(s, drag, Δt)))
            # Momentum budget: the returned force is exactly the change.
            ΔM = sum(sim.u.data[3:Nx+2, 3:Ny+2, 3:Nz+2] .- uprev[3:Nx+2, 3:Ny+2, 3:Nz+2]) *
                 mesh.dx * mesh.dy * mesh.dz
            sink_err = max(sink_err, abs(F[].Fx * dt - ΔM) / abs(ΔM))
            uprev .= sim.u.data
        end
        @test sink_err < 1e-10
        t = nsteps * dt
        ucol = sim.u.data[3, 3, 3:Nz+2]
        # Canopy layers (z < 20 m): exact U0/(1 + Cd a U0 t); above: untouched.
        uex = U0 / (1 + C_d * 0.2 * U0 * t)
        err_in = maximum(abs.(ucol[1:5] ./ uex .- 1))
        @test err_in < 1e-12
        @test all(ucol[6:end] .≈ U0)
        @test all(isfinite, sim.u.data)
        @test maximum(abs, sim.w.data) < 1e-12        # a sheared u(z) stays 2-D
        @info "LES implicit drag, dt=2 s, 30 steps" u_canopy = ucol[1] exact = uex err_in

        # (b) the same term as an explicit tendency (the retired path): unstable.
        sime = BoussinesqSimulation(mesh; bcs = bcs)
        set!(sime.u, U0)
        explicit! = (s, Δt) -> begin
            @inbounds for k in 1:Nz, j in 1:Ny, i in 1:Nx
                I, J, K = i + 2, j + 2, k + 2
                s.tu.data[I, J, K] -= C_d * drag.a_u[i, j, k] * abs(s.u.data[I, J, K]) * s.u.data[I, J, K]
            end
        end
        blew = false
        for n in 1:nsteps
            step!(sime, dt; body_force! = explicit!)
            m = maximum(abs, sime.u.data)
            if !isfinite(m) || m > 1e3 * U0
                blew = true; break
            end
        end
        @test blew

        # (c) a fully 3-D random velocity: every component shrinks, none
        # changes sign, kinetic energy drops, and the call does not allocate.
        simr = BoussinesqSimulation(mesh; bcs = bcs)
        lad2 = canopy_lad_field(mesh, LalicMihailovicLAD(3.0, 25.0; z_m = 15.0),
                                [isodd(i + j) for i in 1:Nx, j in 1:Ny])
        drag2 = ResolvedCanopyDrag(mesh, lad2; C_d = 0.25)
        for f in (simr.u, simr.v, simr.w)
            f.data .= 5 .* (rand(size(f.data)...) .- 0.5)
        end
        Aeolus.fill_halos!(simr.w, bcs)
        u0 = copy(simr.u.data); v0 = copy(simr.v.data); w0 = copy(simr.w.data)
        F2 = apply_canopy_drag!(simr, drag2, 50.0)
        ii, jj, kk = 3:Nx+2, 3:Ny+2, 3:Nz+2
        for (new, old) in ((simr.u.data, u0), (simr.v.data, v0), (simr.w.data, w0))
            n_ = new[ii, jj, kk]; o_ = old[ii, jj, kk]
            @test all(abs.(n_) .<= abs.(o_))
            @test all(sign.(n_) .== sign.(o_) .|| n_ .== 0)
        end
        @test sum(abs2, simr.u.data[ii, jj, kk]) < sum(abs2, u0[ii, jj, kk])
        @test F2.Fx * 50.0 ≈ sum(simr.u.data[ii, jj, kk] .- u0[ii, jj, kk]) * mesh.dx * mesh.dy * mesh.dz
        # Face LAD is the mean of the two cells sharing the face (periodic
        # wrap): a u face between a bare and a vegetated column gets half.
        bare = findfirst(!, [isodd(i + j) for i in 1:Nx, j in 1:Ny])
        @test drag2.a_u[bare[1], bare[2], 1] == lad2[mod1(bare[1] - 1, Nx), bare[2], 1] / 2
        # The only allocation is inside `fill_halos!` (pre-existing, ~15 kB
        # per field on this grid); the drag loops themselves allocate nothing.
        _alloc_drag(s, d) = @allocated apply_canopy_drag!(s, d, 2.0)
        _alloc_halo(s) = (@allocated(Aeolus.fill_halos!(s.u, s.bcs)) +
                          @allocated(Aeolus.fill_halos!(s.v, s.bcs)) +
                          @allocated(Aeolus.fill_halos!(s.w, s.bcs)))
        _alloc_drag(simr, drag2); _alloc_halo(simr)
        @test _alloc_drag(simr, drag2) <= _alloc_halo(simr)
    end
end
