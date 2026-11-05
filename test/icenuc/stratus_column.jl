# Phase 43, issue #7 — mixed-phase Arctic stratus column (ISDAC-style 1-D
# column; see src/icenuc/stratus_column.jl for what is simplified).
#
# Asserted here: the ISDAC initial profile; over a short column run the
# water and INP budgets close and every ice particle came from a consumed
# INP; glaciation_time on a synthetic LWP series. The 2-8 h persistence
# runs are the validation run validation/era4_icenucleation.jl.

@testset "Arctic stratus column (issue #7)" begin
    @test isdac_initial_profile(600.0) == (265.0, 1.5e-3)
    @test isdac_initial_profile(826.0)[1] ≈ 267.0

    @testset "Short column run: budgets close, no ice without an INP" begin
        nsteps = 2                    # ice exists after the first step
        col = ArcticStratusColumn(inp_scale = 1.0, hours = nsteps * 30.0 / 3600, output_every = 300.0)
        r = run_arctic_stratus(col)
        # water: column + precipitation − large-scale (subsidence) source
        @test r.water ≈ r.water0 + r.water_subsidence - r.precip rtol = 1e-12
        # INP: available, inside ice, or reached the ground
        @test r.inp_total ≈ r.inp_total0 + r.inp_subsidence - r.inp_surface rtol = 1e-12
        Ni_col = sum(r.ρ[k] * r.states[k].N_i * col.dz for k in eachindex(r.states))
        Nc_col = sum(i.N_cons[1] * col.dz for i in r.inps)
        @test 0 < Ni_col <= Nc_col * (1 + 1e-9)
    end

    run = (t = [0.0, 1.0, 2.0, 3.0, 4.0], LWP = [20.0, 15.0, 5.0, 2.5, 0.5])
    @test glaciation_time(run) == 4.0                    # first below 10 % of the start
    @test glaciation_time(run; fraction = 0.01) == Inf
end
