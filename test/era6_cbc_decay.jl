# Era VI, Phase 54 (P54-c, #168): the set-up of the Comte-Bellot & Corrsin
# (1971) decaying-grid-turbulence benchmark — the reference table and the
# initial state. The benchmark itself (every closure run to the stations
# tU0/M = 98 and 171 and compared with CBC) is a validation run,
# validation/era6_cbc_decay.jl, whose header carries the measured numbers.
#
# Harness: test/support/cbc_benchmark_driver.jl (shared with the validation
# file). Reference: CBC table 3 (test/data/cbc1971_table3.txt).

using Test
using Aeolus

isdefined(@__MODULE__, :cbc_benchmark) || include(joinpath(@__DIR__, "support", "cbc_benchmark_driver.jl"))

@testset "Era VI — Comte-Bellot–Corrsin set-up (P54-c, #168)" begin

    @testset "reference data: table 3 against table 4" begin
        k, E, urms = cbc_table()
        @test size(E) == (20, 3) && count(isnan, E) == 4
        @test k[1] ≈ 15.0 && k[end] ≈ 2000.0                       # 0.15 … 20 cm⁻¹
        for s in 1:3
            f = cbc_spectrum(s)
            ok = .!isnan.(E[:, s])
            @test f.(k[ok]) ≈ E[ok, s] rtol = 1e-12                 # passes through the table
            # isotropy: ∫E dk = (3/2) u'² with u' from table 4 (an
            # independent measurement); within 5 % (measured 4.2, 1.3, 0.3 %)
            kk = exp.(range(log(1.0), log(3000.0), length = 20001))
            I = sum((f.(kk[1:end-1]) .+ f.(kk[2:end])) ./ 2 .* diff(kk))
            @test abs(I / (1.5 * urms[s]^2) - 1) < 0.05
        end
    end

    # r: the benchmark's initial state of one realisation, not its run: one
    # short evolve-and-rescale cycle (t_spin = 0.03 s, a few RK3 steps)
    # exercises the same initialiser, spin-up and `rescale_to_spectrum!` as
    # the benchmark's two 0.284 s cycles. (In the validation file this read
    # the 3-seed mean of the :dynamic benchmark; the rescale makes every
    # realisation exact, so one seed asserts the same.)
    @testset "set-up: station-1 spectrum exact in every resolved shell" begin
        r = cbc_initial_state(:dynamic; N = 32, seed = 1, spinup = 1, t_spin = 0.03)
        @test r.E_les[1][2:17] ≈ r.E_ref[1][2:17] rtol = 1e-10
        @test r.k[2] ≈ 2π / (11 * 0.0508)
        # filtering to 32³ keeps more of the decay than the full K = 3/2 u'²
        _, _, urms = cbc_table()
        @test r.K_ratio_ref[1] > (urms[2] / urms[1])^2
        @test r.K_ratio_ref[2] > (urms[3] / urms[1])^2
    end
end
