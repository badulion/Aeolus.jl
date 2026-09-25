# Phase 43, issue #7 — mixed-phase Arctic stratus persistence (ISDAC-style
# 1-D column; see src/icenuc/stratus_column.jl for what is simplified).
#
# Asserted: the column's water and INP budgets close; every ice particle
# came from a consumed INP; with realistic INP (0.1–1 L⁻¹ ice) the liquid
# layer persists for 8 h; ice lowers LWP monotonically with INP (as in the
# ISDAC intercomparison); raising INP 100× glaciates the deck, and the
# glaciation time shortens as INP rise.

@testset "Arctic stratus column (issue #7)" begin
    @testset "ISDAC initial profile" begin
        @test isdac_initial_profile(600.0) == (265.0, 1.5e-3)
        @test isdac_initial_profile(0.0)[1] ≈ 263.4
        @test isdac_initial_profile(0.0)[2] ≈ 1.8e-3
        @test isdac_initial_profile(826.0)[1] ≈ 267.0
        @test isdac_initial_profile(1000.0)[2] == 1.2e-3
    end

    runs = Dict{Float64,Any}()
    for (scale, hours) in ((0.0, 8.0), (1.0, 8.0), (10.0, 8.0), (100.0, 4.0), (300.0, 2.0))
        el = @elapsed runs[scale] = run_arctic_stratus(ArcticStratusColumn(inp_scale = scale, hours = hours))
        r = runs[scale]
        @printf("    INP ×%-5g %3.0f h in %5.1f s: LWP %6.2f → %6.2f g/m², IWP(end) %6.3f g/m², N_i(cloud,end) %6.3f L⁻¹, t_glac %s h\n",
                scale, hours, el, r.LWP[1], r.LWP[end], r.IWP[end], r.Ni_cloud[end],
                string(glaciation_time(r)))
    end
    base, noice, ten, hundred, threehundred =
        runs[1.0], runs[0.0], runs[10.0], runs[100.0], runs[300.0]

    @testset "Initial cloud matches the ISDAC case" begin
        # ~14 g/m² initial LWP reported for ISDAC flight 31 (Paukert &
        # Hoose 2014): checks the profiles, pressure and adjustment together
        @test 13.0 <= base.LWP[1] <= 16.0
        @test base.LWP[1] == noice.LWP[1]       # no ice at t = 0
    end

    @testset "Budgets close" begin
        for r in values(runs)
            # water: column + precipitation − large-scale (subsidence) source
            @test r.water ≈ r.water0 + r.water_subsidence - r.precip rtol = 1e-12
            # INP: every particle is available, inside ice, or reached the ground
            if r.inp_total0 > 0
                @test r.inp_total ≈ r.inp_total0 + r.inp_subsidence - r.inp_surface rtol = 1e-12
                # no ice without an INP: column ice number ≤ consumed INP
                # (Hallett–Mossop is inactive, the cloud is colder than -8 °C)
                Ni_col = sum(r.ρ[k] * r.states[k].N_i * 25.0 for k in eachindex(r.states))
                Nc_col = sum(i.N_cons[1] * 25.0 for i in r.inps)
                @test Ni_col <= Nc_col * (1 + 1e-9)
            end
        end
        @test minimum(s.T for s in base.states) < 265.15   # cloud outside the HM window
    end

    @testset "Mixed-phase persistence with realistic INP" begin
        i1 = findfirst(>=(1.0), base.t)
        # ice number in the liquid layer: 0.1–1 L⁻¹ from the first hour on
        @test all(0.1 .<= base.Ni_cloud[i1:end] .<= 1.0)
        # ... and within a band around the ISDAC numbers at the end of the
        # run (observed ~0.35 L⁻¹; the intercomparison prescribed 1 and
        # 4 L⁻¹). The band is ours, not a published range.
        @test 0.25 <= base.Ni_cloud[end] <= 4.0
        # liquid persists: LWP after 8 h is not below its initial value
        @test base.LWP[end] >= base.LWP[1]
        @test glaciation_time(base) == Inf
        @test all(base.IWP[2:end] .> 0)
        @printf("    base case after %.0f h: LWP %.2f g/m² (no ice: %.2f), IWP %.3f g/m², N_i %.3f L⁻¹\n",
                base.t[end], base.LWP[end], noice.LWP[end], base.IWP[end], base.Ni_cloud[end])
    end

    @testset "Ice reduces LWP monotonically with INP" begin
        @test noice.IWP[end] == 0
        @test noice.LWP[end] > base.LWP[end] > ten.LWP[end]
        @test base.IWP[end] < ten.IWP[end]
        # 10× INP: 2.5–5 L⁻¹, still mixed-phase after 8 h
        @test glaciation_time(ten) == Inf
    end

    @testset "Glaciation when INP are raised 100×" begin
        tg100 = glaciation_time(hundred)
        tg300 = glaciation_time(threehundred)
        @test tg100 < 4.0
        @test hundred.LWP[end] < 0.01 * hundred.LWP[1]
        @test maximum(hundred.Ni_cloud) > 10.0
        # sign of the sensitivity: more INP, faster glaciation
        @test tg300 < tg100 < glaciation_time(base)
        @printf("    glaciation time (LWP < 10 %% of initial): ×1 %s, ×10 %s, ×100 %.2f h, ×300 %.2f h\n",
                string(glaciation_time(base)), string(glaciation_time(ten)), tg100, tg300)
    end
end
