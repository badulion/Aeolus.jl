# Era V, Phase 52 — OceanMixedLayer validation (issue #38): the Fairall
# and Zeng–Beljaars diurnal warm layers, driven by the COARE 3.6 ship
# fluxes over days 19-25, against the ship's near-surface thermometers
# (sea snake T(0.05 m) minus TSG T(5.3 m)). Reported, not tuned. The
# column, cool skin and warm-layer schemes are unit-tested (and checked
# against the COARE 3.6 reference code) in test/era5_ocean_mixed_layer.jl.

using Test, Printf
using Aeolus
using Aeolus.OceanMixedLayer

isdefined(@__MODULE__, :read_coare_subset) ||
    include(joinpath(@__DIR__, "..", "test", "support", "coare36_subset.jl"))

@testset "Era V — OceanMixedLayer: warm layers vs ship observations (#38)" begin
    coare = coare36_warm_cool_subset()
    ncoare = length(coare[:jd])

    @testset "Warm layers against the ship's near-surface thermometers" begin
        # Observed T(0.05 m) - T(5.3 m) from the sea snake and the TSG, and
        # the same difference from each scheme, driven by the COARE fluxes.
        # Reported, not tuned: the observed signal is small (≤ 0.14 K)
        # because the trade winds kept mixing the layer.
        ld = floor.(Int, coare[:jd] .+ ((coare[:lon] .+ 7.5) ./ 15) ./ 24)
        obs = coare[:tsnk] .- coare[:tsg]
        night = coare[:sw_net] .== 0
        offset = sort(obs[night])[cld(count(night), 2)]    # sensor offset: night median
        function drive(wl)
            out = zeros(ncoare); jtime = 0.0; started = false
            for i in 1:ncoare
                jd, lon = coare[:jd][i], coare[:lon][i]
                chk = (lon + 7.5) / 15 + (jd - trunc(jd)) * 24
                chk > 24 && (chk -= 24)
                newtime = (chk - 24 * trunc(chk / 24)) * 3600
                if i > 1 && (newtime <= 21600 || started)
                    started = true
                    if newtime < jtime
                        wl isa FairallWarmLayer && reset_warm_layer!(wl)
                    else
                        tsg = coare[:tsg][i]
                        lw_up = 0.97 * (5.67e-8 * (tsg - coare[:dT_skin][i-1] + 273.16)^4 - coare[:lw_dn][i])
                        q_out = lw_up + coare[:hsb][i-1] + coare[:hlb][i-1] + coare[:hrain][i-1]
                        step_warm_layer!(wl, coare[:sw_net][i], q_out, coare[:tau][i-1], newtime - jtime;
                                         α = 2.1e-5 * (tsg + 3.2)^0.79, ρ_w = 1022.0, c_w = 4000.0,
                                         g = coare_grav(coare[:lat][i]))
                    end
                end
                out[i] = warm_layer_warming(wl, coare[:ztsg][i]) - warm_layer_warming(wl, 0.05)
                jtime = newtime
            end
            out
        end
        F = drive(FairallWarmLayer()); Z = drive(ZengBeljaarsWarmLayer())
        days = 19:25
        mo = [maximum(obs[ld .== k]) - offset for k in days]
        mF = [maximum(F[ld .== k]) for k in days]
        mZ = [maximum(Z[ld .== k]) for k in days]
        τday = [sum(coare[:tau][ld .== k]) / count(ld .== k) for k in days]
        # The calmest day (25) has the largest observed warming, and both
        # schemes put their largest warming on it too.
        @test argmin(τday) == argmax(mo) == argmax(mF) == argmax(mZ)
        day = .!night
        rmsF = sqrt(sum(abs2, F[day] .- (obs[day] .- offset)) / count(day))
        rmsZ = sqrt(sum(abs2, Z[day] .- (obs[day] .- offset)) / count(day))
        @info "Warm layer vs ship observations, T(0.05 m) - T(5.3 m), days 19-25" sensor_offset_K = offset daily_max_obs = round.(mo; digits = 3) daily_max_Fairall = round.(mF; digits = 3) daily_max_ZengBeljaars = round.(mZ; digits = 3) daytime_rms_Fairall = rmsF daytime_rms_ZB = rmsZ
    end
end
