# Era V, Phase 51 — lake and sea-ice thermodynamics validation (issue #37):
# the Maykut & Untersteiner (1971) benchmark. 20-year runs against the BL99
# reference code (same configuration, numbers measured in Octave) and
# 100-year equilibria against the published Bitz & Lipscomb (1999) Tables
# 1-2 (thicknesses, amplitudes, sensitivities to α_i and F_L). No
# observations are used. Material functions, Stefan/Neumann growth and
# energy conservation are unit-tested in test/era5_ice_thermodynamics.jl.

using Test, Printf
using Aeolus
using Aeolus.IceThermodynamics

isdefined(@__MODULE__, :_run_mu71) || include(joinpath(@__DIR__, "..", "test", "support", "mu71.jl"))

@testset "Era V — sea-ice thermodynamics: MU71 benchmark (#37)" begin

    @testset "MU71 benchmark vs. the BL99 reference code (same configuration)" begin
        # Reference numbers measured by running the BL99 reference code
        # (C. M. Bitz, https://www.atmos.washington.edu/~bitz/column.tar,
        # column.m with its data.mu71) in GNU Octave 11.1: N = 10, Δt = 12 h,
        # vertically varying salinity, α_i = 0.63, i₀ = 0.3 snow-free only.
        #   original code, year 1 (cm):            mean 264.09  max 293.61  min 242.82
        #   original code, 100-yr equilibrium:     265.07
        #   code with first-order (half-cell)
        #   boundary fluxes, as used here, yr 1:   mean 263.68  max 292.93  min 242.44
        #                                  yr 20:  mean 250.01
        # (Re-measured independently in review: identical to 0.01 cm. The
        # code's own 20-yr driver switches to Δt = 4 h in years 18-20; this
        # driver keeps 12 h throughout.) c₀ = 2054 as in the code.
        p = IceParameters(ε = 1.0, constants = K_BL99_CODE)
        r = _run_mu71(20; params = p)
        y1 = _year(r.daily, 1) .* 100; y20 = _year(r.daily, 20) .* 100
        # First annual cycle: agrees with the unmodified reference code.
        @test abs(_mean(y1) - 264.09) < 0.5
        @test abs(maximum(y1) - 293.61) < 1.0
        @test abs(minimum(y1) - 242.82) < 0.5
        # Twenty years on: agrees with the reference code carrying the same
        # boundary discretisation (the remaining physics is identical).
        @test abs(_mean(y1) - 263.68) < 0.5
        @test abs(_mean(y20) - 250.01) < 1.5
        @info "MU71 vs BL99 reference code" year1_mean = _mean(y1) year1_max = maximum(y1) year1_min = minimum(y1) year20_mean = _mean(y20)
    end

    @testset "MU71 equilibrium vs. published BL99 Tables 1-2" begin
        # BL99 Table 1 configuration: Δt = 4 h, N = 10, T_f = -1.95 °C,
        # i₀ = 0.3·10/(h_s[cm] + 10) (Bettge et al. 1996), 100 years.
        base = (ε = 1.0, constants = K_BL99, i0_snow_scale = 0.1)
        eq(p; salinity = :bl99, lw = 0.0) = begin
            r = _run_mu71(100; nday = 6, params = p, salinity, Tf = -1.95, lw_pert = lw)
            y1 = _year(r.daily, 1)
            (h = 100 * _mean(_year(r.daily, 100)),
             ampl = 100 * (maximum(y1[1:200]) - minimum(y1[150:end])), res = r.maxres)
        end
        vv  = eq(IceParameters(; base...))
        vva = eq(IceParameters(; base..., α_ice = 0.62))
        vvl = eq(IceParameters(; base...); lw = 1.0)
        iso = eq(IceParameters(; base..., T_melt_ice = -0.1); salinity = 3.2)
        isa = eq(IceParameters(; base..., T_melt_ice = -0.1, α_ice = 0.60); salinity = 3.2)
        # Published (BL99 Table 1, energy-conserving, 100-yr): 281, 400, 259 cm,
        # first-cycle amplitudes 42, 34, 41 cm. BL99 Table 2 (vertically varying
        # salinity): lowering α_i by 0.01 changes the equilibrium by 27 cm
        # (thinner — the sign is lost in the scanned table, physics fixes it)
        # and raising F_L by 1 W m⁻² by -48 cm.
        # We are 6-11 % thinner. The offset is a boundary-discretisation
        # effect of the reference: with N = 10 the BL99 code fits a quadratic
        # through the snow and first ice layer across the snow/ice
        # conductivity jump; replacing that one stencil by the half-cell flux
        # used here moves the reference code onto our solution (250.0 vs
        # 250.4 cm after 20 years), and the reference itself drifts towards
        # it with resolution (265 → 259 → 255 cm at N = 10, 30, 60, year 20).
        @test abs(vv.h / 281 - 1) < 0.08
        # Direct check of that diagnosis, same Table-1 configuration (Δt = 4 h,
        # Bettge i₀, T_f = -1.95 °C), BL99 reference code run in Octave 11.1
        # during review: unmodified 281.04 cm at year 100 (reproduces Table 1),
        # with only the snow-surface stencil made half-cell 262.62 cm.
        # Isosaline α_i 0.63 (T_s,melt = -0.1 °C): 418.2 / 368.4 cm.
        @test abs(vv.h - 262.62) < 3.0
        @test abs(iso.h / 400 - 1) < 0.13
        @test abs(isa.h / 259 - 1) < 0.12
        @test iso.h > vv.h > isa.h                 # BL99's ordering
        @test abs(vv.ampl / 42 - 1) < 0.15
        @test abs(isa.ampl / 41 - 1) < 0.15
        dα = vva.h - vv.h; dL = vvl.h - vv.h
        @test abs(dα / -27 - 1) < 0.15             # Table 2 (published -27 cm)
        @test abs(dL / -48 - 1) < 0.15             # Table 2 (published -48 cm)
        @test max(vv.res, iso.res, isa.res) < 1e-8
        @info "MU71 equilibrium (cm) vs BL99 Table 1/2" varying = vv.h iso_063 = iso.h iso_060 = isa.h ampl_varying = vv.ampl ampl_iso060 = isa.ampl dh_dα_minus001 = dα dh_dFL_plus1 = dL
    end
end
