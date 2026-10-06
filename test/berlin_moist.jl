# Berlin case: hydrostatic reference state and moist buoyancy (#131).
#
#   * the reference p̄(z) is hydrostatic (against an independent RK4
#     integration of the continuous equation) and consistent with the
#     ideal gas law and the Exner function of MoistThermodynamics; the
#     ledger integrates water with its ρ̄;
#   * the moist buoyancy on a unit column is g (0.61 q_v′ − q_l′), has no
#     net force, and moves moist air up and fog-laden air down;
#   * fog in the Berlin case conserves moist enthalpy at the reference
#     pressure to 1e-10.

using Test
using Aeolus
using Aeolus.Fields: interior_view, set!
using Printf
using Random: Xoshiro

isdefined(Main, :BerlinCase) ||
    include(joinpath(@__DIR__, "..", "examples", "urban", "BerlinCase.jl"))
const BC = Main.BerlinCase
const MT = Aeolus.MoistThermodynamics

@testset "Berlin: reference state and moist buoyancy (#131)" begin

    @testset "reference state is hydrostatic, ideal-gas and Exner-consistent" begin
        c = MT.DEFAULT_CONSTANTS
        R_d = c.gas_constant / c.molmass_dry
        for sc in (:summer, :winter, :smoke)
            cfg = BC.CaseConfig(sc; verbose = false, output_path = nothing)
            mesh = RegularGrid(; size = cfg.size, extent = cfg.extent, halo = cfg.halo)
            ref = BC.build_reference(cfg, mesh)
            err = BC.hydrostatic_check(cfg, ref, mesh)
            res = Aeolus.Anelastic.max_hydrostatic_residual(ref.state)
            old = [1.01325e5 * exp(-(k - 0.5) * mesh.dz / 8000.0) for k in 1:mesh.Nz]
            Δold = maximum(abs.(ref.p̄ .- old) ./ ref.p̄)
            @info @sprintf("  %-6s p̄ vs continuous hydrostatic: max rel %.1e; discrete residual %.1e Pa/m; ρ̄ %.4f→%.4f kg/m³; old isothermal profile off by %.2f %%",
                           sc, err, res, ref.ρ̄[1], ref.ρ̄[end], 100Δold)
            @test err < 1e-3                  # acceptance: within 0.1 % (measured ≲ 6e-8)
            @test err < 1e-6
            @test res < 1e-9                  # Pa/m: the grid balance is exact
            @test all(<(0), diff(ref.p̄)) && ref.p̄[1] < ref.p_surface
            @test ref.p_surface == BC.case_surface_pressure(cfg)
            θ0 = BC.initial_θ(cfg)
            for k in 1:mesh.Nz
                @test ref.T̄[k] ≈ ref.θ̄[k] * ref.π̄[k] rtol = 1e-12
                @test ref.π̄[k] ≈ MT.exner_function(ref.p̄[k]) rtol = 1e-12
                @test ref.ρ̄[k] ≈ MT.air_density(ref.T̄[k], ref.p̄[k]) rtol = 1e-12
                @test ref.θ̄[k] ≈ θ0(0.0, 0.0, (k - 0.5) * mesh.dz) rtol = 1e-14
                # the idealised profile of :winter and :smoke is the analytic one
                cfg.sounding === nothing &&
                    @test ref.θ̄[k] ≈ cfg.θ_surface + max(0, (k - 0.5) * mesh.dz - cfg.θ_inversion_z) * cfg.θ_lapse rtol = 1e-14
            end
        end
        # the case carries it, and the ledger integrates water with its ρ̄
        case = BC.build_case(BC.CaseConfig(:smoke; verbose = false, output_path = nothing))
        @test case.ledger.ρ == case.ref.ρ̄
        @test BC.surface_pressure(case) == 101325.0
        @test 1.19 < BC.ref_density(case, 1) < 1.21   # 294 K, 1013 hPa
    end

    @testset "initial state from the Lindenberg sounding of 29 June 2019 (#247)" begin
        cfg = BC.CaseConfig(:summer; verbose = false, output_path = nothing)
        S = BC.case_sounding(cfg)
        L = S.sounding.launches
        # 06:00 CEST = 04:00 UTC, between the 22:45 (28 June) and 04:45 UTC releases
        t0 = BC.case_start_time(cfg)
        @test t0 == 4 * 3600.0
        @test L[1].t < t0 < L[2].t
        w = (t0 - L[1].t) / (L[2].t - L[1].t)                     # 0.875
        θ0, q0 = BC.initial_θ(cfg), BC.initial_qv(cfg)
        P = S.profiles
        for z in (0.0, 3.125, 50.0, 100.0, 196.875, 400.0, 1500.0)
            θa, θb = forcing_value(P.θ, z, L[1].t), forcing_value(P.θ, z, L[2].t)
            @test θ0(0.0, 0.0, z) ≈ (1 - w) * θa + w * θb rtol = 1e-14
            qa, qb = forcing_value(P.qv, z, L[1].t), forcing_value(P.qv, z, L[2].t)
            @test q0(0.0, 0.0, z) ≈ (1 - w) * qa + w * qb rtol = 1e-12
            @test 0 < q0(0.0, 0.0, z) < 0.02
        end
        # a surface-based inversion at 06 CEST: z_i (parcel) = 0, θ rising
        # through the lowest 100 m; at 14 CEST the mixed layer is ~1.2 km deep
        @test BC.initial_inversion_height(cfg) == 0
        @test θ0(0.0, 0.0, 100.0) > θ0(0.0, 0.0, 0.0)
        zi14 = BC.initial_inversion_height(BC.CaseConfig(:summer; start_h = 14.0, verbose = false))
        @test 1000 < zi14 < 1500
        # ground pressure: the launch-site value reduced from 112 m to 34 m,
        # ≈ ρ g Δz = 1.2 · 9.81 · 78 ≈ 9.2 hPa higher
        pL = forcing_value(P.p, 0.0, t0)
        ps = BC.case_surface_pressure(cfg)
        @test 880 < ps - pL < 960
        # the case day is a Saturday (the building schedules follow it)
        @test BC.case_weekday(cfg) == 6
        @test BC.case_weekday(BC.CaseConfig(:winter)) == 1             # Monday 15 January 2024
        # the idealised scenarios keep the analytic profile
        @test BC.case_sounding(BC.CaseConfig(:winter)) === nothing
        @test BC.case_surface_pressure(BC.CaseConfig(:smoke)) == 101325.0
        # the Berlin diagnostic of #247: the observed against the former profile
        old_θ(z) = 293.0 + max(0, z - 100.0) * 3e-3
        old_q(z) = 0.010 * exp(-z / 800.0)
        @info "  initial profile at 06 CEST, sounding vs former analytic (θ K, qv g/kg): " *
              join([@sprintf("z=%.0f m: θ %.2f vs %.2f, qv %.2f vs %.2f", z, θ0(0, 0, z), old_θ(z),
                             1e3q0(0, 0, z), 1e3old_q(z)) for z in (0.0, 25.0, 50.0, 100.0, 150.0, 200.0, 400.0)], "; ") *
              @sprintf("; p_surface %.1f hPa (was 1013.25)", ps / 100)
    end

    @testset "ledger: an incompressible exchange between levels is density redistribution, not transport" begin
        case = BC.build_case(BC.CaseConfig(:smoke; verbose = false, output_path = nothing))
        Nx, Ny, Nz = case.mesh.Nx, case.mesh.Ny, case.mesh.Nz
        dz, A = case.mesh.dz, Nx * Ny
        # a conservative (Σ dq = 0) exchange: 1 g/kg moved from level 1 to 2 in one column
        dq = zeros(Nx, Ny, Nz); dq[3, 3, 1] = -1e-3; dq[3, 3, 2] = 1e-3
        flat, redis = BC.ledger_water_split(case, dq)
        ρ̄ = case.ref.ρ̄
        @test abs(flat) < 1e-18
        # moving vapour up into thinner air: the ρ̄-weighted water drops by Δρ̄·δq·Δz/A
        @test redis ≈ (ρ̄[2] - ρ̄[1]) * 1e-3 * dz / A rtol = 1e-12
        @test redis < 0
        @test flat + redis ≈ BC.ledger_water(case, dq) rtol = 1e-12
        # a net source splits into ρ̄_m·Σdq and the level-dependent rest
        dq .= 0; dq[:, :, 4] .= 1e-4
        flat, redis = BC.ledger_water_split(case, dq)
        @test flat ≈ sum(ρ̄) / Nz * 1e-4 * dz rtol = 1e-12
        @test flat + redis ≈ ρ̄[4] * 1e-4 * dz rtol = 1e-12
    end

    @testset "moist buoyancy on a unit column: g (0.61 q_v′ − q_l′)" begin
        # 4 × 4 × 6, uniform θ = θ₀ and q_v = 10 g/kg; column (1,1) has 1 g/kg
        # more vapour in levels 3–4, column (2,3) 1 g/kg of fog there.
        θ₀, g = 293.0, 9.81
        Nx, Ny, Nz = 4, 4, 6
        θ = fill(θ₀, Nx, Ny, Nz); qv = fill(0.010, Nx, Ny, Nz)
        ql = zeros(Nx, Ny, Nz); qr = zeros(Nx, Ny, Nz)
        qv[1, 1, 3:4] .+= 1e-3
        ql[2, 3, 3:4] .= 1e-3
        fluid = trues(Nx, Ny, Nz)
        Δ = zeros(Nx, Ny, Nz)
        BC.moist_buoyancy_perturbation!(Δ, θ, qv, [ql, qr], Array(fluid))
        mesh = RegularGrid(; size = (Nx, Ny, Nz), extent = (4.0, 4.0, 6.0), halo = (1, 1, 1))
        tw = Aeolus.Fields.Field(Aeolus.Grids.FaceZ(), mesh)
        BC.add_moist_buoyancy!(tw, Δ, θ₀, g)
        b(i, j, k) = tw.data[mesh.Hx+i, mesh.Hy+j, mesh.Hz+k]   # face k: bottom of cell k
        n = Nx * Ny
        # the face between levels 3 and 4 sits inside both anomalies
        # primes are deviations from the level mean, which contains both
        # anomalies: the vapour column's q_l′ is −1/n of the fog, the fog
        # column's q_v′ is −1/n of the vapour anomaly
        b_vap = g * (0.61 * 1e-3 * (1 - 1 / n) - (-1e-3 / n))  # 0.61 q_v′ − q_l′
        b_fog = g * (0.61 * (-1e-3 / n) - 1e-3 * (1 - 1 / n))  # 0.61 q_v′ − q_l′
        @info @sprintf("  unit column: b(vapour) = %.4e (0.61 q_v′ − q_l′: %.4e), b(fog) = %.4e (expected %.4e) m/s²",
                       b(1, 1, 4), b_vap, b(2, 3, 4), b_fog)
        @test b(1, 1, 4) ≈ b_vap rtol = 5e-3     # 1/ε − 1 = 0.608, not 0.61
        @test b(2, 3, 4) ≈ b_fog rtol = 5e-3
        @test b(1, 1, 4) > 0 && b(2, 3, 4) < 0
        # half the anomaly on the faces at its edges, nothing further out
        @test b(1, 1, 3) ≈ b(1, 1, 4) / 2 rtol = 1e-12
        @test b(1, 1, 5) ≈ b(1, 1, 4) / 2 rtol = 1e-12
        @test b(1, 1, 2) == 0 && b(1, 1, 6) == 0
        @test b(1, 1, 1) == 0                    # the wall face is untouched
        # no net force on any level
        for k in 1:Nz
            @test abs(sum(b(i, j, k) for i in 1:Nx, j in 1:Ny)) < 1e-15
        end
        # solid cells get no buoyancy and do not enter the level mean
        fl = trues(Nx, Ny, Nz); fl[4, 4, :] .= false
        qv[4, 4, :] .= 0.5                        # junk inside a building
        BC.moist_buoyancy_perturbation!(Δ, θ, qv, [ql, qr], Array(fl))
        @test all(Δ[4, 4, :] .== 0)
        @test Δ[1, 1, 1] == 0
        @test Δ[1, 1, 3] ≈ θ₀ * (0.6078 * 1e-3 * (1 - 1 / 15) + 1e-3 / 15) rtol = 1e-4
    end

    @testset "moist buoyancy in the dynamics: moist air rises, fog-laden air sinks" begin
        mesh = RegularGrid(; size = (8, 8, 8), extent = (80.0, 80.0, 80.0), halo = (3, 3, 3))
        function run(moist)
            sim = BoussinesqSimulation(mesh; ν = 0.0, θ₀ = 293.0, g_accel = 9.81,
                bcs = FieldBoundaryConditions(:periodic, :periodic, :wall),
                θ_bcs = FieldBoundaryConditions(:periodic, :periodic, :noflux))
            set!(sim.θ, (x, y, z) -> 293.0)
            qv = fill(0.010, 8, 8, 8); ql = zeros(8, 8, 8)
            qv[2, 2, 3:5] .+= 1e-3            # a moist column
            ql[6, 6, 3:5] .= 1e-3              # a fog column
            Δ = zeros(8, 8, 8); fluid = fill(true, 8, 8, 8)
            bf = moist ? (s, dt) -> (BC.moist_buoyancy_perturbation!(Δ, interior_view(s.θ), qv, [ql], fluid);
                                     BC.add_moist_buoyancy!(s.tw, Δ, 293.0, 9.81)) : nothing
            step!(sim, 1.0; body_force! = bf)
            return interior_view(sim.w)
        end
        w_dry, w = run(false), run(true)
        @info @sprintf("  one 1 s step: w in the moist column %.2e m/s, in the fog column %.2e m/s (dry run: %.1e)",
                       w[2, 2, 4], w[6, 6, 4], maximum(abs, w_dry))
        @test maximum(abs, w_dry) < 1e-12        # uniform θ: nothing moves without the moist term
        @test w[2, 2, 4] > 1e-4                  # vapour is light: 0.61 g q_v′ ≈ 6e-3 m/s² before the pressure response
        @test w[6, 6, 4] < -1e-4                 # condensate loading
        @test w[2, 2, 4] < 0.61 * 9.81 * 1e-3    # the projection only takes away from the local acceleration
    end

    @testset "fog in the Berlin case conserves moist enthalpy at p̄ (1e-10)" begin
        case = BC.build_case(BC.CaseConfig(:smoke; verbose = false, output_path = nothing))
        θ = interior_view(case.sim.θ)
        qv = interior_view(BC.tracer(case, :qv)); ql = interior_view(BC.tracer(case, :ql))
        rng = Xoshiro(131)
        Nx, Ny, Nz = size(θ)
        # levels 1–6: supersaturated by up to 3 %, or subsaturated with fog
        # that must evaporate
        for k in 1:6, j in 1:Ny, i in 1:Nx
            T = θ[i, j, k] * BC.ref_exner(case, k)
            qs = MT.saturation_specific_humidity(T, BC.ref_pressure(case, k))
            if isodd(i + j)
                qv[i, j, k] = qs * (1 + 0.03 * rand(rng))
            else
                qv[i, j, k] = qs * (1 - 0.03 * rand(rng)); ql[i, j, k] = 2e-4 * rand(rng)
            end
        end
        h(k, I) = MT.moist_enthalpy(θ[I, k] * BC.ref_exner(case, k),
                                    MT.PhasePartition(qv[I, k] + ql[I, k], ql[I, k]))
        # the first air cell of a dew_mask column exchanges dew with the
        # surface (#129): k = 1 on open ground, above the roof on buildings
        dew_cell(k, I) = case.bus.dew_mask[I] && case.bus.k_air[I] == k
        cells = [(k, I) for k in 2:Nz for I in CartesianIndices((Nx, Ny)) if !dew_cell(k, I)]
        h0 = [h(k, I) for (k, I) in cells]
        qt0 = [qv[I, k] + ql[I, k] for (k, I) in cells]
        ql0 = sum(ql)
        fd = BC.find_coupling(case, :fog_dew)
        BC.run_stage!(fd, case, Val(:fog_dew), case.cfg.dt)
        h1 = [h(k, I) for (k, I) in cells]
        qt1 = [qv[I, k] + ql[I, k] for (k, I) in cells]
        rel = maximum(abs.(h1 .- h0) ./ abs.(h0))
        # saturation after the adjustment, at the reference T and p̄
        rh = maximum(abs(MT.relative_humidity(θ[I, k] * BC.ref_exner(case, k), BC.ref_pressure(case, k), qv[I, k]) - 1)
                     for (k, I) in cells if ql[I, k] > 0)
        @info @sprintf("  fog at p̄: max |Δh|/h = %.1e over %d cells; condensed %.2e kg/kg (sum); max |RH − 1| in fog %.1e",
                       rel, length(cells), sum(ql) - ql0, rh)
        @test rel < 1e-10
        @test maximum(abs.(qt1 .- qt0)) < 1e-17   # total water is untouched by fog (round-off of q ≈ 1e-2)
        @test sum(ql) != ql0                      # it did work
        @test rh < 1e-6                           # fog cells are left at saturation of the reference p̄
    end
end
