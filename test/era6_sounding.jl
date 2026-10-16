# N-parent-a (#247) — radiosonde soundings → (z, t) profiles, and the one
# boundary-layer-height diagnostic.
#
# What is checked against what:
#   1. hydrostatics: the hypsometric heights against the analytic
#      isothermal atmosphere (exact) and against the geopotential heights
#      IGRA reports for the Lindenberg launches of 2019-06-29/30;
#   2. θ = T (p₀/p)^(R_d/c_pd) to 1e-12 on every level; the 12 UTC
#      profile is stable above the diagnosed z_i and well mixed below;
#   3. every `boundary_layer_height` method on analytic profiles (mixed
#      layer + inversion, linear stable profile); the LES half of this
# test (z_i of the CBL harness, :parcel within 10 %) is
#      in era6_tke.jl testset 6, which already runs that LES;
#   4. the observed z_i and surface-inversion depths, quoted, against the
#      Lindenberg mixing-height climatology of Beyrich & Leps (2012);
#   5. the ForcingTable round trip: exact at the launches' own (z, t),
#      linear in t between launches.

using Test, Printf
using Aeolus
using Aeolus.MoistThermodynamics: DEFAULT_CONSTANTS, PhasePartition, virtual_temperature

const SOUNDING_CSV = joinpath(@__DIR__, "..", "examples", "urban", "data",
                              "lindenberg_10393_20190629.csv")

@testset "N-parent-a: Lindenberg sounding and boundary_layer_height (#247)" begin

c = DEFAULT_CONSTANTS
Rd, cp, g0, p0 = c.gas_constant / c.molmass_dry, c.cp_dry, c.grav, c.p_ref
snd = read_sounding(SOUNDING_CSV)
P = sounding_profiles(snd; z_top = 3000.0)
launch(tag) = snd.launches[findfirst(l -> l.tag == tag, snd.launches)]
θv_of(θ, q) = virtual_temperature(θ, PhasePartition(q))   # T_v/T is independent of T

@testset "read: six launches, epoch, elevation, release times" begin
    @test length(snd.launches) == 6
    @test snd.epoch == (2019, 6, 29)
    @test snd.elevation == 112.0
    # release times 22:45 (previous day), 04:45, 10:45, 16:47, 22:51, 04:48 UTC
    @test [l.t for l in snd.launches] == [-4500.0, 17100.0, 38700.0, 60420.0, 82260.0, 103680.0]
    @test sounding_time(snd, 2019, 6, 29, 4.0) == 14400.0         # the Berlin case start, 06 CEST
    @test sounding_time(snd, 2019, 7, 1, 0.0) == 2 * 86400.0      # across a month end
    @test sounding_time(snd, 2020, 6, 29, 0.0) == 366 * 86400.0   # 2020 is a leap year
    @test all(l -> l.z[1] == 0 && all(>(0), diff(l.z)), snd.launches)
end

@testset "1. hydrostatic consistency" begin
    # analytic: an isothermal dry atmosphere, z = (R_d T/g₀) ln(p_s/p)
    p = [101000.0, 95000.0, 85000.0, 70000.0, 50000.0]
    l = SoundingLaunch(0.0, p, fill(280.0, 5), fill(NaN, 5))
    @test maximum(abs.(l.z .- Rd * 280 / g0 .* log.(p[1] ./ p)) ./ max.(l.z, 1)) < 1e-13
    # observed: hypsometric z + station elevation against the reported GPH
    # of the standard levels below 3 km (1000/925/850/700 hPa)
    for l in snd.launches
        sel = findall(k -> isfinite(l.gph[k]) && l.z[k] < 3000, eachindex(l.z))
        d = l.z[sel] .+ snd.elevation .- l.gph[sel]
        rms = sqrt(sum(abs2, d) / length(d))
        # bound on |Δz|: IGRA's GTS surface pressure is in whole hPa (±50 Pa
        # → ±R_d T_v/(g₀ p_s)·50 Pa ≈ ±4.3 m, a constant offset of every
        # height) plus the whole-metre GPH (±0.5 m)
        Tv1 = virtual_temperature(l.T[1], PhasePartition(l.qv[1]))
        bound = Rd * Tv1 / (g0 * l.p[1]) * 50 + 0.5
        @info @sprintf("  %s: %d standard levels < 3 km, height error RMS %.2f m, max %.2f m (bound %.2f m), %s",
                       l.tag, length(d), rms, maximum(abs.(d)), bound, string(round.(d; digits = 1)))
        @test length(d) >= 3
        @test rms < 10.0
        @test maximum(abs.(d)) < bound
        # the issue's "max < 0.5 % of z" holds above 1 km; below, the 4 m
        # surface-pressure quantisation alone is > 0.5 % (see the PR)
        hi = findall(>(1000), l.z[sel])
        @test !isempty(hi) && maximum(abs.(d[hi]) ./ (l.z[sel][hi] .+ snd.elevation)) < 0.005
    end
    # the station elevation: with IGRA's station-list 97.7 m instead of the
    # launch-site 112 m every launch misses the GPH by 11-19 m
    for l in snd.launches
        sel = findall(k -> isfinite(l.gph[k]) && l.z[k] < 3000, eachindex(l.z))
        d = l.z[sel] .+ 97.7 .- l.gph[sel]
        @test sqrt(sum(abs2, d) / length(d)) > 10.0
    end
end

# 12 UTC (10:45 release) profile on the launch's own thermodynamic levels
l12 = launch("2019-06-29T12Z")
θ12 = [T * (p0 / p)^(Rd / cp) for (T, p) in zip(l12.T, l12.p)]
θv12 = θv_of.(θ12, l12.qv)
zi_parcel = boundary_layer_height(l12.z, θv12; method = :parcel)
# bulk Richardson on the common grid, with the sounding's wind
zg = P.z
θvg = [θv_of(forcing_value(P.θ, h, l12.t), forcing_value(P.qv, h, l12.t)) for h in zg]
ug = [forcing_value(P.u, h, l12.t) for h in zg]
vg = [forcing_value(P.v, h, l12.t) for h in zg]
zi_rib = boundary_layer_height(zg, θvg; method = :bulk_richardson, u = ug, v = vg, g = g0)
zi_grad = boundary_layer_height(l12.z, θ12; method = :max_gradient)

@testset "2. θ conversion and the 12 UTC mixed layer" begin
    for l in snd.launches, k in eachindex(l.z)
        θ = forcing_value(P.θ, l.z[k], l.t)
        l.z[k] <= 3000 && @test θ ≈ l.T[k] * (p0 / l.p[k])^(Rd / cp) rtol = 1e-12
    end
    dθ = diff(θ12) ./ diff(l12.z)
    zm = (l12.z[1:end-1] .+ l12.z[2:end]) ./ 2
    above = findall(z -> zi_parcel < z < 3000, zm)
    @test !isempty(above) && all(>(0), dθ[above])
    # Well mixed in the bulk of the CBL. The issue asked |dθ/dz| < 1 K/km
    # everywhere below 0.8 z_i; that expectation was wrong in the surface
    # layer, which is superadiabatic by day (here θ falls 1.9 K in the
    # lowest 74 m, −26 K/km; Stull 1988, §11.1, Fig. 11.4): the bound is
    # applied above the surface layer, 0.1 z_i < z < 0.8 z_i.
    mixed = findall(z -> 0.1zi_parcel < z < 0.8zi_parcel, zm)
    @info @sprintf("  12 UTC: dθ/dz in 0.1-0.8 z_i: %s K/km; surface layer (0-%.0f m) %.1f K/km",
                   string(round.(1000 .* dθ[mixed]; digits = 2)), l12.z[2], 1000dθ[1])
    @test !isempty(mixed) && all(x -> abs(x) < 1e-3, dθ[mixed])
    @test dθ[1] < -1e-2                                   # superadiabatic surface layer
end

@testset "3. boundary_layer_height on analytic profiles" begin
    Δ = 25.0
    z = collect(Δ/2:Δ:2000.0)
    h = 800.0
    θml = [zz < h ? 300.0 : 303.0 + 0.006 * (zz - h) for zz in z]   # 3 K jump at h
    for m in (:parcel, :max_gradient, :bulk_richardson)
        zi = boundary_layer_height(z, θml; method = m)
        @test abs(zi - h) <= Δ
    end
    # bulk Richardson with a wind profile: 5 m/s in the mixed layer, 10 m/s above
    u = [zz < h ? 5.0 : 10.0 for zz in z]
    @test abs(boundary_layer_height(z, θml; method = :bulk_richardson, u = u, v = zero(u)) - h) <= Δ
    # the minimum of a CBL flux profile 1 − 1.2 z/h below h, relaxing to 0 above
    fl = [zz <= h ? 1 - 1.2zz / h : -0.2 * exp(-(zz - h) / 100) for zz in z]
    @test abs(boundary_layer_height(z, fl; method = :min_flux) - h) <= Δ
    # parcel with an excess: the crossing of θ(z₁) + 1 K inside the inversion
    θlin = [zz < h ? 300.0 : 300.0 + 0.01 * (zz - h) for zz in z]
    @test boundary_layer_height(z, θlin; method = :parcel, excess = 1.0) ≈ h + 100 atol = 1e-9
    # a linear stable profile: no mixed layer, z_i is the first level
    θst = 290.0 .+ 0.004 .* z
    @test boundary_layer_height(z, θst; method = :parcel) == z[1]
    @test boundary_layer_height(z, θst; method = :bulk_richardson) ≈ z[1] atol = 1e-3
    @test_throws ArgumentError boundary_layer_height(z, θst; method = :nope)
    @test_throws ArgumentError boundary_layer_height(z, θst[1:end-1])
end

@testset "4. observed numbers against the Lindenberg climatology" begin
    # surface-based inversion depth: top of the layer in which T does not
    # decrease with height from the ground (00/06 UTC launches)
    function sbi_depth(l)
        k = 1
        while k < length(l.T) && l.T[k+1] >= l.T[k]; k += 1; end
        return l.z[k]
    end
    d = Dict(l.tag => sbi_depth(l) for l in snd.launches)
    @info @sprintf("  12 UTC (10:45 UTC release) z_i: parcel %.0f m, bulk Ri (Ri_c = 0.25) %.0f m, max dθ/dz %.0f m above ground",
                   zi_parcel, zi_rib, zi_grad)
    @info @sprintf("  surface-inversion depth: 00 UTC 29 Jun %.0f m, 06 UTC 29 Jun %.0f m, 00 UTC 30 Jun %.0f m, 06 UTC 30 Jun %.0f m",
                   d["2019-06-29T00Z"], d["2019-06-29T06Z"], d["2019-06-30T00Z"], d["2019-06-30T06Z"])
    # the methods agree on one capping inversion (884-879 hPa)
    @test abs(zi_parcel - zi_rib) < 0.1zi_parcel
    @test abs(zi_parcel - zi_grad) < 0.1zi_parcel
    # Beyrich & Leps (2012), Meteorol. Z. 21, 337-348, abstract: monthly mean
    # Ri-based mixing height at 12 UTC 1200-1400 m in early summer, at 00
    # UTC 150-200 m in summer. One day is not a monthly mean: the test asks
    # for the 12 UTC z_i within ±30 % of that band.
    @test 0.7 * 1200 < zi_parcel < 1.3 * 1400
    @test 0.7 * 1200 < zi_rib < 1.3 * 1400
    # Night: every 00/06 UTC launch has a surface-based inversion, no deeper
    # than 3 × the 200 m climatological night mixing height. Beyrich & Leps'
    # night values are Ri-based mixing heights from the full-resolution
    # profiles; the GTS significant levels kept by IGRA do not resolve a
    # nocturnal boundary layer (the bulk Ri from the surface record crosses
    # 0.25 below the first significant level, 2-61 m), so the inversion
    # depth is what is compared, not a night z_i.
    for tag in ("2019-06-29T00Z", "2019-06-29T06Z", "2019-06-30T00Z", "2019-06-30T06Z")
        @test 0 < d[tag] < 3 * 200
    end
end

@testset "5. ForcingTable round trip" begin
    for (n, l) in enumerate(snd.launches)
        θl = [T * (p0 / p)^(Rd / cp) for (T, p) in zip(l.T, l.p)]
        for k in eachindex(l.z)
            l.z[k] <= 3000 || continue
            @test forcing_value(P.θ, l.z[k], l.t) == P.θ.values[findfirst(==(l.z[k]), P.z), n]
            @test forcing_value(P.θ, l.z[k], l.t) ≈ θl[k] rtol = 1e-14
            isnan(l.Td[k]) || @test forcing_value(P.qv, l.z[k], l.t) ≈ l.qv[k] rtol = 1e-14
            @test forcing_value(P.p, l.z[k], l.t) ≈ l.p[k] rtol = 1e-14
        end
        for k in eachindex(l.zw)
            l.zw[k] <= 3000 || continue
            @test forcing_value(P.u, l.zw[k], l.t) ≈ l.u[k] atol = 1e-12
            @test forcing_value(P.v, l.zw[k], l.t) ≈ l.v[k] atol = 1e-12
        end
    end
    # linear in t between launches, at every grid height
    t1, t2 = snd.launches[2].t, snd.launches[3].t
    for h in P.z, (tab, w) in ((P.θ, 0.5), (P.qv, 0.25), (P.u, 0.75))
        a, b = forcing_value(tab, h, t1), forcing_value(tab, h, t2)
        @test forcing_value(tab, h, (1 - w) * t1 + w * t2) ≈ (1 - w) * a + w * b rtol = 1e-12 atol = 1e-14
    end
    # held before the first and after the last launch
    @test forcing_value(P.θ, 100.0, -1e6) == forcing_value(P.θ, 100.0, snd.launches[1].t)
    @test forcing_value(P.θ, 100.0, 1e7) == forcing_value(P.θ, 100.0, snd.launches[end].t)
end

end
