# Phase 50 — BuildingEnergy validation (#31): the resolved zone (layered
# envelope columns, internal mass) against the ISO 13790 5R1C model of the
# same construction, 150-day periodic regime under a diurnal forcing, heavy
# and light box: mean, amplitude and timing of the operative temperature.
# The zone model and the 5R1C parameter mapping are unit-tested in
# test/building_energy.jl.

using Test, Printf
using Aeolus
using Aeolus.BuildingEnergy
using Aeolus.UrbanSurface

isdefined(@__MODULE__, :test_box) || include(joinpath(@__DIR__, "..", "test", "support", "building_box.jl"))

@testset "Era V — BuildingEnergy: resolved zone vs ISO 13790 5R1C (#31)" begin
    @testset "Resolved zone vs ISO 13790 5R1C for the same construction" begin
        # Periodic regime under the same diurnal forcing: outdoor 15 ± 5 °C,
        # office-hours internal gains, transmitted solar peaking at noon.
        Tout(t) = 288.15 + 5sin(2π * (t / 86400 - 0.375))
        Qs(t) = max(0.0, 1000sin(2π * (t / 86400 - 0.25)))
        Qi(t) = 8 <= mod(t / 3600, 24) < 20 ? 600.0 : 100.0
        out = Dict{Bool, NamedTuple}()
        for heavy in (true, false)
            z = test_box(heavy = heavy); iso = iso13790_zone(z)
            da, dop, ia, iop = Float64[], Float64[], Float64[], Float64[]
            Qs_det, Qs_iso = 0.0, 0.0
            # The light box's insulated roof slab couples to the outdoors through
            # 10 cm XPS: its slowest mode has τ ≈ 18 d, so the spin-up is 150 d.
            for day in 1:150, h in 1:24
                for k in 1:6
                    t = (day - 1) * 86400 + (h - 1) * 3600 + (k - 0.5) * 600
                    step_zone!(z, ZoneForcing(T_out = Tout(t), Q_internal = Qi(t), Q_solar = Qs(t)), 600.0)
                    if day > 148
                        push!(da, z.T_air); push!(dop, zone_operative_temperature(z))
                        Qs_det += Qs(t)
                    end
                end
                t = (day - 1) * 86400 + (h - 0.5) * 3600
                θa, θs, _ = step_iso13790!(iso, Tout(t), 3600.0; Φ_int = Qi(t), Φ_sol = Qs(t))
                if day > 148
                    push!(ia, θa); push!(iop, 0.3θa + 0.7θs)      # ISO 13790 eq. (C.12)
                    Qs_iso += Qs(t)
                end
            end
            mean(x) = sum(x) / length(x)
            amp(x) = (maximum(x) - minimum(x)) / 2
            # Linearity: the periodic-regime mean equals the steady solution for
            # the period-mean forcing, in each model separately.
            @test mean(da) ≈ steady_air_temperature(z, 288.15, 350.0, Qs_det / length(da), 0.0) atol = 1e-4
            iso_ss = deepcopy(iso)
            for _ in 1:24 * 300
                step_iso13790!(iso_ss, 288.15; Φ_int = 350.0, Φ_sol = Qs_iso / length(ia))
            end
            @test mean(ia) ≈ iso_ss.θ_air atol = 1e-4
            peak(x) = mod(argmax(x) - 1, length(x) ÷ 2) # sample index within the day
            out[heavy] = (det_mean = mean(dop), iso_mean = mean(iop), det_amp = amp(dop),
                          iso_amp = amp(iop), det_peak = peak(dop) * 600 / 3600 + 1 / 12,
                          iso_peak = peak(iop) + 0.5, det_air = mean(da), iso_air = mean(ia))
        end
        for heavy in (true, false)
            o = out[heavy]
            @printf("  %s: operative mean detailed %.3f °C / 5R1C %.3f °C; amplitude %.3f / %.3f K; peak %.1f h / %.1f h\n",
                    heavy ? "heavy" : "light", o.det_mean - 273.15, o.iso_mean - 273.15,
                    o.det_amp, o.iso_amp, o.det_peak, o.iso_peak)
            # Same steady conductances ⇒ the mean excess over outdoors agrees to
            # within the structural differences of the networks (where gains and
            # glazing attach): 3 % of the mean excess.
            @test abs(o.det_mean - o.iso_mean) < 0.03 * (o.det_mean - 288.15)
            # One lumped capacitance vs resolved columns: amplitudes within 40 %,
            # peak times within 3 h (measured-discrepancy envelopes, see PR).
            @test abs(o.det_amp / o.iso_amp - 1) < 0.4
            @test abs(o.det_peak - o.iso_peak) < 3
        end
        # Both models agree the heavy building swings less.
        @test out[true].det_amp < out[false].det_amp && out[true].iso_amp < out[false].iso_amp
    end
end
