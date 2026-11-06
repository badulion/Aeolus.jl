# Phase 52 — diurnal warm-layer validation (issue #40): the Fairall and
# Zeng–Beljaars warm layers, held at the 7-hour mean forcing, against the
# OBSERVED peak afternoon warming of 42 days from R/V Moana Wave in TOGA
# COARE, Fairall et al. (1996a), Table 2 (validation/data/
# fairall1996a_table2.txt; provenance in test/data/README_coare.md). Sea
# spray, the full COARE algorithm and the warm-layer schemes are
# unit-tested in test/sea_spray_coare.jl and
# test/ocean_mixed_layer.jl.

using Test, Printf
using Aeolus
using Aeolus.OceanMixedLayer

const _MT = Aeolus.MoistThermodynamics

isdefined(@__MODULE__, :read_coare_subset) ||
    include(joinpath(@__DIR__, "..", "test", "support", "coare36_subset.jl"))

@testset "diurnal warm layer vs TOGA COARE observations (#40)" begin
    @testset "Diurnal warm layer vs TOGA COARE observations (Fairall et al. 1996a, Table 2)" begin
        t = read_coare_subset(joinpath(@__DIR__, "data", "fairall1996a_table2.txt"))
        nd = length(t[:Q])
        @test nd == 42
        # Transcription check against the printed column fw Rns - Q
        # (clipped at 0 by the paper). Rows 14, 35 and 42 are inconsistent in
        # the paper itself (0.66·680-180 = 269 vs 230, 0.88·275-160 = 82 vs
        # 0, 0.66·500-130 = 200 vs 180 printed); the other 39 agree to the
        # rounding of fw.
        net = max.(t[:fw] .* t[:Rns] .- t[:Q], 0)
        @test findall(abs.(net .- t[:net]) .>= 12) == [14, 35, 42]
        # Forcing held at its 7-hour means, as in the paper's eq. (25).
        # Assumed, since the table does not give them: warm-pool water
        # 29.5 °C, 34 psu; air 28 °C, 1008 hPa, q = 19 g/kg for ρ_a.
        eos = UNESCOWaterEOS()
        Tw, S = 302.65, 34.0
        α, ρw = water_thermal_expansion(eos, Tw, S), water_density(eos, Tw, S)
        ρa = _MT.air_density(301.15, 100800.0, _MT.PhasePartition(0.019))
        fa = zeros(nd); zb = zeros(nd); D = zeros(nd)
        for i in 1:nd
            w, z = FairallWarmLayer(), ZengBeljaarsWarmLayer()
            τ = ρa * t[:us2][i]
            for _ in 1:84                                  # 7 h, 5-min steps
                step_warm_layer!(w, t[:Rns][i], t[:Q][i], τ, 300.0; α = α, ρ_w = ρw, c_w = 4000.0, g = 9.78)
                step_warm_layer!(z, t[:Rns][i], t[:Q][i], τ, 300.0; α = α, ρ_w = ρw, c_w = 4000.0, g = 9.78)
            end
            fa[i], zb[i], D[i] = warm_layer_warming(w), warm_layer_warming(z), warm_layer_depth(w)
        end
        obs = t[:dTm]
        stats(x) = (rmse = sqrt(sum(abs2, x .- obs) / nd), bias = sum(x .- obs) / nd,
                    r = (a = x .- sum(x) / nd; b = obs .- sum(obs) / nd;
                         sum(a .* b) / sqrt(sum(abs2, a) * sum(abs2, b))))
        sP, sF, sZ = stats(t[:dTw]), stats(fa), stats(zb)
        # The Fairall scheme explains the observed day-to-day variation
        # (r ≥ 0.9) with an error comparable to the paper's own eq. (25)
        # evaluation (whose f_w, D_T and constants we do not reproduce
        # exactly).
        @test sF.r > 0.9
        @test sF.rmse < 1.5 * sP.rmse
        @test sZ.r > 0.85
        # Wind control: the calm days (U < 2 m/s) warm far more than the
        # windy ones (U ≥ 5 m/s), in the observations and in both schemes.
        calm, windy = t[:U] .< 2, t[:U] .>= 5
        mean(x) = sum(x) / length(x)
        for x in (obs, fa, zb)
            @test mean(x[calm]) > 5 * mean(x[windy])
        end
        # Where the paper's model has no warm layer (net heating 0), neither
        # has the scheme.
        @test all(fa[t[:net] .== 0] .== 0)
        @info "Peak warming vs Moana Wave observations (42 days)" paper_eq25 = sP fairall = sF zeng_beljaars = sZ calm_obs_mean = mean(obs[calm]) calm_fairall_mean = mean(fa[calm]) depth_ratio_vs_paper = extrema(D[t[:net] .> 0] ./ t[:DT][t[:net] .> 0])
    end
end
