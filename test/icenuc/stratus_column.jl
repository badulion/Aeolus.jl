# Phase 43, issue #7 — mixed-phase Arctic stratus persistence (ISDAC-style
# 1-D column; see src/icenuc/stratus_column.jl for what is simplified).
#
# Asserted here: the ISDAC initial profile; over a short (10 min) column run
# the water and INP budgets close and every ice particle came from a
# consumed INP; glaciation_time on a synthetic LWP series. The 2-8 h runs
# (persistence with realistic INP, LWP monotone in INP, glaciation at 100×
# INP) are the validation run validation/era4_icenucleation.jl.

@testset "Arctic stratus column (issue #7)" begin
    @testset "ISDAC initial profile" begin
        @test isdac_initial_profile(600.0) == (265.0, 1.5e-3)
        @test isdac_initial_profile(0.0)[1] ≈ 263.4
        @test isdac_initial_profile(0.0)[2] ≈ 1.8e-3
        @test isdac_initial_profile(826.0)[1] ≈ 267.0
        @test isdac_initial_profile(1000.0)[2] == 1.2e-3
    end

    @testset "Short column run: budgets close, no ice without an INP" begin
        # 20 steps of 30 s with realistic INP. (The 2-8 h runs at five INP
        # scalings are the validation run validation/era4_icenucleation.jl.)
        col = ArcticStratusColumn(inp_scale = 1.0, hours = 20 * 30.0 / 3600, output_every = 300.0)
        r = run_arctic_stratus(col)
        @test r.t[end] ≈ 20 * 30.0 / 3600
        # water: column + precipitation − large-scale (subsidence) source
        @test r.water ≈ r.water0 + r.water_subsidence - r.precip rtol = 1e-12
        # INP: every particle is available, inside ice, or reached the ground
        @test r.inp_total0 > 0
        @test r.inp_total ≈ r.inp_total0 + r.inp_subsidence - r.inp_surface rtol = 1e-12
        # no ice without an INP: column ice number ≤ consumed INP, and some
        # INP have been consumed (ice exists after 10 min)
        Ni_col = sum(r.ρ[k] * r.states[k].N_i * col.dz for k in eachindex(r.states))
        Nc_col = sum(i.N_cons[1] * col.dz for i in r.inps)
        @test Ni_col > 0
        @test Ni_col <= Nc_col * (1 + 1e-9)
    end

    @testset "glaciation_time: first time LWP falls below a fraction of its start" begin
        run = (t = [0.0, 1.0, 2.0, 3.0, 4.0], LWP = [20.0, 15.0, 5.0, 2.5, 0.5])
        @test glaciation_time(run) == 4.0                    # 0.5 < 2 (10 %); 2.5 is not
        @test glaciation_time(run; fraction = 0.5) == 2.0    # 5 < 10
        @test glaciation_time(run; fraction = 0.01) == Inf   # never below 0.2
        @test glaciation_time((t = [0.0, 1.0], LWP = [10.0, 10.0])) == Inf
    end
end
