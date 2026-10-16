# The comparison helpers of test/support/agreement.jl (#495): results are
# compared within a tolerance of the field's size, not bitwise, and the
# helpers must still tell a change of the computation from round-off.

using Test
using TOML
using Random: Xoshiro, randn
include(joinpath(@__DIR__, "support", "agreement.jl"))

@testset "agreement: round-off agrees, a change of the computation does not (#495)" begin
    rng = Xoshiro(495)
    u = 3 .+ randn(rng, 16, 16, 8)
    # the same sum in another order: round-off, agrees
    a = [sum(u[:, :, k]) for k in 1:8]
    b = [sum(reverse(vec(u[:, :, k]))) for k in 1:8]
    @test agree(a, b)
    # a perturbation at round-off scale agrees, one of 1e-8 of the field does not
    @test agree(u .* (1 + 1e-14), u)
    @test !agree(u .+ 1e-8 * maximum(abs, u), u)
    # a single changed cell is caught
    v = copy(u); v[3, 4, 5] += 1e-6
    @test !agree(v, u)
    # NaN in the same places agrees, elsewhere it does not; sizes must match
    w = copy(u); w[1] = NaN; w2 = copy(w)
    @test agree(w, w2) && !agree(w, u) && !agree(u[:, :, 1:7], u)
    # an all-zero field: the absolute difference counts
    @test agree(zeros(4), zeros(4)) && !agree([1e-30, 0, 0, 0], zeros(4)) && agree([1e-30, 0, 0, 0], zeros(4); atol = 1e-20)
    # numbers
    @test agree(1.0 + 1e-15, 1.0) && !agree(1.0 + 1e-6, 1.0)

    # fingerprints: through TOML and back, round-off agrees, a change does not
    fp = TOML.parse(sprint(io -> TOML.print(io, Dict("f" => fingerprint(u)))))["f"]
    @test fingerprint_agree(fingerprint(u .* (1 + 1e-14)), fp)
    @test !fingerprint_agree(fingerprint(u .+ 1e-6), fp)          # the mean moves
    v = copy(u); v[1] = 100.0                                       # a sampled value and the max move
    @test !fingerprint_agree(fingerprint(v), fp)
    @test !fingerprint_agree(fingerprint(u[1:end-1]), fp)           # the length
    @test fingerprints_differing(Dict("u" => u, "v" => 2u), Dict("u" => fp, "v" => fingerprint(2u))) == String[]
    @test fingerprints_differing(Dict("u" => u .+ 1e-3), Dict("u" => fp, "w" => fp)) == ["u", "w"]

    # nested outputs (results.toml, NetCDF variables), CSV, summaries
    @test values_agree(Dict("a" => [1.0, 2.0], "b" => "x", "c" => 3), Dict("a" => [1.0, 2.0 + 1e-15], "b" => "x", "c" => 3))
    @test !values_agree(Dict("a" => [1.0, 2.0], "c" => 3), Dict("a" => [1.0, 2.0], "c" => 4))
    @test csv_agree(["t,x,name", "1.0,2.0,a"], ["t,x,name", "1.0,2.0000000000000004,a"])
    @test !csv_agree(["t,x,name", "1.0,2.0,a"], ["t,x,name", "1.0,2.1,a"])
    @test !csv_agree(["t,x,name", "1.0,2.0,a"], ["t,x,name", "1.0,2.0,b"])
    s = summary_of((u = u, n = 7, mask = trues(3, 3), dt = [0.5, 0.25]))
    @test summaries_agree(s, summary_of((u = u .* (1 + 1e-14), n = 7, mask = trues(3, 3), dt = [0.5, 0.25])))
    @test !summaries_agree(s, summary_of((u = u, n = 8, mask = trues(3, 3), dt = [0.5, 0.25])))    # integer data exact
    @test !summaries_agree(s, summary_of((u = u, n = 7, mask = falses(3, 3), dt = [0.5, 0.25])))
    @test !summaries_agree(s, summary_of((u = u .+ 1e-6, n = 7, mask = trues(3, 3), dt = [0.5, 0.25])))
end
