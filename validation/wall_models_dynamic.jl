# Era VI, Phase 57 (P57-e, #190): validation of the dynamic wall model for
# partially resolved roughness (Anderson & Meneveau 2011, JFM 679) — grid
# invariance (reduced): on a neutral rough half channel over a random k⁻²
# surface, run to 6 H/u*, the plane-mean u_* the model predicts at 32² and
# 16² (a priori, on the 32² LES wind) agrees within 5 % with the dynamic α
# and disagrees by more with α fixed at its 32² value. The model stays
# tested in test/wall_models_dynamic.jl (analytic invariance, the
# sub-filter statistics, the momentum budget with the form drag).

using Test
using Printf
using Random
using Statistics
using Aeolus
# the wall-bounded benchmark harness (half channel, channel) is a test fixture (ONE-l, #475)
isdefined(@__MODULE__, :ChannelCases) || include(joinpath(@__DIR__, "..", "test", "fixtures", "channel_cases.jl"))
using .ChannelCases
using Aeolus.WallModels
# _k2_surface, _blockmean, _rough_half_channel: the random k⁻² surface and
# the rough half channel over it (shared with test/wall_models_dynamic.jl)
isdefined(@__MODULE__, :_rough_half_channel) || include(joinpath(@__DIR__, "..", "test", "support", "rough_half_channel.jl"))

@testset "Era VI — dynamic roughness: grid invariance (#190)" begin

@testset "Grid invariance (reduced): rough half channel, 32² vs 16²" begin
    # A random k⁻² surface (256² map, 1 ≤ k ≤ 128, σ_h = 0.04 H), resolved
    # through the surface-gradient drag (C_d = 4), the rest sub-filter.
    # Half channel 2πH × 2πH × H, 32 × 32 × 8, StaticSmagorinsky(0.1), with
    # the dynamic roughness, run to 6 H/u*; at t = 4, 5, 6 the first-cell
    # wind is taken as is (32²) and block-averaged 2 × 2 (16²), and the
    # plane-mean u_* = √(wall + form drag) of the model is evaluated at both
    # resolutions with the dynamic α of each, and with α fixed at its 32²
    # value.
    (; C_d, z0f, st32, st16, case, z) = _rough_half_channel()
    t0 = time()
    run_channel!(case, 1.0)
    # the five steps the test side checks the form-drag budget on
    # (wall_models_dynamic.jl): taken here too, so this run follows the
    # trajectory its tolerances were set on
    for _ in 1:5
        advance_channel!(case, channel_dt(case))
    end
    run_channel!(case, 4.0)
    dr32 = DynamicRoughness(st32; z0_floor = z0f, C_d = C_d)
    dr16 = DynamicRoughness(st16; z0_floor = z0f, C_d = C_d)
    us = Dict(k => Float64[] for k in (:d32, :d16, :f16)); αs = Dict(k => Float64[] for k in (:a32, :a16))
    for t in (4.0, 5.0, 6.0)
        run_channel!(case, t)
        uc, vc = ChannelCases._first_cell_velocity(case.sim)
        u16, v16 = _blockmean(uc, 2), _blockmean(vc, 2)
        a32, c32 = dynamic_alpha(dr32, uc, vc, z)
        a16, c16 = dynamic_alpha(dr16, u16, v16, z)
        @test c32 && c16
        push!(αs[:a32], a32); push!(αs[:a16], a16)
        push!(us[:d32], sqrt(roughness_surface_drag(dr32, uc, vc, z; α = a32).total))
        push!(us[:d16], sqrt(roughness_surface_drag(dr16, u16, v16, z; α = a16).total))
        push!(us[:f16], sqrt(roughness_surface_drag(dr16, u16, v16, z; α = a32).total))
    end
    u32, ud, uf = mean(us[:d32]), mean(us[:d16]), mean(us[:f16])
    ed, ef = ud / u32 - 1, uf / u32 - 1
    @info @sprintf("  grid invariance: α(32²) = %.3f, α(16²) = %.3f; u*(32²) = %.4f, u*(16²) dynamic = %.4f (%+.1f %%), fixed α = %.4f (%+.1f %%); %.1f s",
                   mean(αs[:a32]), mean(αs[:a16]), u32, ud, 100ed, uf, 100ef, time() - t0)
    @test abs(ed) < 0.05
    @test abs(ef) > abs(ed)
    @test mean(αs[:a16]) < mean(αs[:a32])          # α is scale dependent: the point of the dynamic model
end

end
