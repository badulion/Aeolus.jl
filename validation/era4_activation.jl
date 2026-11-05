# Era IV, Phase 42 — aerosol-cloud interaction: the headline validation.
# The Abdul-Razzak & Ghan parameterisation against an EXPLICIT adiabatic
# parcel integration over w = 0.1-5 m/s and N_a = 100-1000 cm⁻³. ARG is the
# operational workhorse; the parcel model is the truth it approximates.
# The module (Köhler theory, ARG, the parcel model's single case, Twomey,
# prognostic aerosol) stays tested in test/era4_activation.jl.

using Test, Printf
using Aeolus

@testset "Era IV — Aerosol-cloud interaction (Phase 42)" begin

    @testset "ARG vs explicit parcel — the headline validation" begin
        T, p = 283.0, 9.0e4
        errs_S = Float64[]; errs_f = Float64[]
        rows = String[]
        for w in (0.1, 0.5, 2.0, 5.0), Na in (100.0, 1000.0)
            modes = [arg_mode(Na * 1e6, 0.05e-6, 2.0, 0.6)]
            a  = arg_activation(modes, w, activation_coefficients(T, p))
            pr = parcel_activation(modes, w, T, p)
            @test pr.S_max > 0
            @test 0 < pr.fraction[1] <= 1
            @test 0 < a.fraction[1] <= 1
            push!(errs_S, abs(a.S_max - pr.S_max) / pr.S_max)
            push!(errs_f, abs(a.fraction[1] - pr.fraction[1]) / pr.fraction[1])
            push!(rows, @sprintf("w=%.1f N=%4.0f: ARG S=%.4f%% f=%.3f | parcel S=%.4f%% f=%.3f",
                                 w, Na, 100a.S_max, a.fraction[1],
                                 100pr.S_max, pr.fraction[1]))
        end
        # ARG is documented as accurate to roughly 10-20% over this range.
        # It systematically UNDER-predicts S_max, most at high updraught.
        @test maximum(errs_S) < 0.35
        @test maximum(errs_f) < 0.35
        @test sum(errs_f) / length(errs_f) < 0.15
        @info "ARG vs parcel" worst_Smax=maximum(errs_S) worst_fraction=maximum(errs_f) mean_fraction=sum(errs_f)/length(errs_f)
        for r in rows; @info r; end
    end
end
