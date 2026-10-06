# The Berlin parent case (N-parent-e, #259), validated alone at reduced
# resolution. The full-resolution numbers come from
# examples/urban/berlin_parent_diurnal.jl (docs/src/design/berlin_parent.md).
#
#   1. encroachment (Tennekes 1973; Stull 1988 Ch. 11): z_i(t) of a CBL
#      under Q(t) = 0.15 sin(πt/D) into Γ = 3 K/km;
#   2. the heat budget of the air: closure at every output, the θ damping
#      below z_i;
#   3. Wangara Day 33 (Clarke et al. 1971; Deardorff 1974) against the
#      MYNN reference z_i of Nakanishi & Niino (2009);
#   4. Lindenberg, 29/30 June 2019, 24 h (the Berlin parent itself, coarse):
#      z_i at 14 LT, θ at 500 m and 1 km, the residual layer at 03 LT, the
#      night nudging (Heinze et al. 2017);
#   5. Boussinesq against anelastic on a uniform grid, CBL to ~2 km;
#   6. robustness: 24 h without NaN, the adaptive dt history.
#
# Numbers measured on this branch are quoted next to each assertion.

using Test
using Aeolus
using Aeolus.Fields: interior_view, set!
using Printf
using Random: Xoshiro

isdefined(Main, :BerlinCase) ||
    include(joinpath(@__DIR__, "..", "examples", "urban", "BerlinCase.jl"))
const BP = Main.BerlinCase

"Run `p` to `hours` in chunks of `p.pc.output_period`, calling `f(p)` after each."
function run_chunks!(f, p, hours)
    n = round(Int, 3600hours / p.pc.output_period)
    for _ in 1:n
        BP.run_parent!(p; hours = p.pc.output_period / 3600)
        f(p)
    end
    return p
end

@testset "Berlin parent case (#259)" begin

idealised(; kw...) = BP.ParentConfig(BP.CaseConfig(; verbose = false, output_path = nothing,
                                                   sounding = nothing, kw...);
                                     extent = (4000.0, 4000.0, 3000.0), refinement = (8, 8, 4),
                                     surface = :prescribed, radiation = false, geostrophic = false,
                                     nudging_τ = Inf, divergence = 0.0, u_init = 0.0,
                                     verbose = false, output_period = 900.0)

@testset "1-2. encroachment and the heat budget" begin
    D, Γ, A = 12 * 3600.0, 3e-3, 0.2
    Q(t) = 0.15 * sin(π * t / D)
    pc = idealised()
    pc.moisture = false
    pc.Q_surface = Q
    pc.θ_init = z -> 290 + Γ * z
    p = BP.build_parent(pc)
    ratios = Float64[]; worst = 0.0
    zi = Dict{Int,Float64}()
    run_chunks!(p, 6.0) do p
        t = p.sim.t
        zf, fl = BP.parent_heat_flux(p)
        t >= 2 * 3600 && push!(ratios, minimum(fl) / Q(t))
        hb = BP.heat_budget(p.heat)
        worst = max(worst, abs(hb.closure) / maximum(abs, values(hb.terms)))
        h = round(Int, t / 3600)
        isapprox(t, 3600h; atol = 1) && (zi[h] = BP.parent_zi(p))
    end
    theory(t) = sqrt(2 * (1 + 2A) * 0.15 * D / π * (1 - cos(π * t / D)) / Γ)
    for h in (3, 6)
        err = zi[h] / theory(3600h) - 1
        @info @sprintf("encroachment: z_i(%d h) = %.0f m (min flux), theory %.0f m, %+.1f %%", h, zi[h], theory(3600h), 100err)
        @test abs(err) < 0.10       # measured +4.7 % (3 h), +1.4 % (6 h)
    end
    R = sum(ratios) / length(ratios)
    @info @sprintf("entrainment flux ratio (2-6 h mean of %d profiles): %.3f (range %.3f…%.3f)",
                   length(ratios), R, minimum(ratios), maximum(ratios))
    @test -0.25 <= R <= -0.10       # measured -0.130
    # heat budget: ∫ρ̄c_pΠ̄(θ − θ_init) dz = Σ booked terms at every output
    @test worst < 1e-10             # measured 6.8e-16 (the closure relative to the largest term)
    hb = BP.heat_budget(p.heat)
    zi6 = zi[6]
    below = BP.heat_budget(p.heat; layer = (0.0, zi6))
    sfc = hb.terms[(:surface_flux, :prescribed)]
    @test abs(get(below.terms, (:dynamics, :damping), 0.0)) < 0.01 * sfc   # measured 0 (the layer starts at 2250 m)
    @info @sprintf("heat budget over 6 h: surface %.4e J/m², storage %.4e, damping %.3e (column) %.3e (below z_i), top %.3e, closure %.1e relative",
                   sfc, hb.storage, get(hb.terms, (:dynamics, :damping), 0.0),
                   get(below.terms, (:dynamics, :damping), 0.0), get(hb.terms, (:dynamics, :top_flux), 0.0), worst)
end

@testset "3. Wangara Day 33" begin
    rows = [split(l, ',') for l in readlines(joinpath(@__DIR__, "..", "examples", "sbl", "data", "wangara_day33.csv"))
            if !startswith(l, "#") && !startswith(l, "table")]
    prof = [parse.(Float64, r[2:end]) for r in rows if r[1] == "profile"]
    ref = Dict(round(Int, parse(Float64, r[2])) => parse(Float64, r[3]) for r in rows if r[1] == "zi")
    z = [q[1] for q in prof]; P = [q[4] for q in prof]; T = [q[5] + 273.15 for q in prof]
    r = [q[6] / 1000 for q in prof]
    κ = 287.04 / 1004.67
    θ = T .* (1000 ./ P) .^ κ
    qv = r ./ (1 .+ r)
    @test θ[1] ≈ 276.85 atol = 0.01        # 5.5 °C at 1023 hPa
    lin(x, y, h) = h >= x[end] ? y[end] :
        (k = searchsortedlast(x, h); k == 0 ? y[1] : y[k] + (y[k+1] - y[k]) * (h - x[k]) / (x[k+1] - x[k]))
    Γtop = (θ[end] - θ[end-5]) / (z[end] - z[end-5])        # 5.4 K/km, 1500-2000 m
    θf(h) = h > z[end] ? θ[end] + Γtop * (h - z[end]) : lin(z, θ, h)
    # Nakanishi & Niino (2009): <wθ>_s = 0.216 cos(π(t_h − 13)/11) K m/s
    shape(t) = cos(π * (9 + t / 3600 - 13) / 11)
    pc = idealised(; start_h = 9.0)
    pc.Q_surface = t -> 0.216 * shape(t)
    pc.E_surface = t -> 2.29e-5 * shape(t)
    pc.θ_init = θf
    pc.qv_init = h -> lin(z, qv, h)
    p = BP.build_parent(pc)
    zi = Dict{Int,Float64}()
    run_chunks!(p, 7.0) do p
        h = 9 + p.sim.t / 3600
        isapprox(h, round(h); atol = 1e-6) && (zi[round(Int, h)] = BP.parent_zi(p))
    end
    for h in (12, 14, 16)
        err = zi[h] / ref[h] - 1
        @info @sprintf("Wangara 33: z_i(%d LT) = %.0f m, NN09 (MYNN) %.0f m, %+.1f %%", h, zi[h], ref[h], 100err)
        @test abs(err) < 0.15       # measured +7.8 %, +6.1 %, +7.6 % (z_i jumps by whole levels, 90-100 m here)
    end
end

@testset "4 + 6. Lindenberg 29/30 June 2019, 24 h (coarse parent)" begin
    child = BP.CaseConfig(; verbose = false, output_path = nothing)
    pc = BP.ParentConfig(child; refinement = (16, 16, 8), verbose = false, output_period = 1800.0)
    p = BP.build_parent(pc)
    @test (p.mesh.Nx, p.mesh.Ny, p.mesh.Nz) == (24, 24, 38)
    BP.run_parent!(p)
    @test all(isfinite, interior_view(p.sim.θ)) && all(isfinite, interior_view(p.sim.qv))
    @test p.retries == 0
    dts = p.dt_hist
    @info @sprintf("Lindenberg 24 h: %d steps, dt %.2f–%.2f s (mean %.2f), %d retries, %.0f s wall",
                   length(dts), minimum(dts), maximum(dts), sum(dts) / length(dts), p.retries, p.timing[:total])
    @test minimum(dts) >= 0.01 && maximum(dts) <= 10.0

    S = BP.case_sounding(child)
    P = S.profiles
    MT = Aeolus.MoistThermodynamics
    t12 = BP.sounding_time(S.sounding, 2019, 6, 29, 12.0)         # 14 LT = 12 UTC (released 10:45)
    θv_obs = [MT.virtual_temperature(BP.forcing_value(P.θ, h, t12), MT.PhasePartition(BP.forcing_value(P.qv, h, t12)))
              for h in P.z]
    zi_obs = boundary_layer_height(P.z, θv_obs; method = :parcel)
    rec(hlt) = p.diag[findfirst(r -> isapprox(r.hour, hlt; atol = 1e-6), p.diag)]
    r14 = rec(14.0)
    err = r14.zi_parcel / zi_obs - 1
    @info @sprintf("z_i(14 LT) parcel %.0f m (min flux %.0f m), 12 UTC sounding %.0f m: %+.1f %%",
                   r14.zi_parcel, r14.zi, zi_obs, 100err)
    @test abs(err) < 0.25           # measured +12.6 % (1332 m against 1183 m)
    zc = BP.level_centres(p.mesh)
    θat(r, h) = (k = searchsortedlast(zc, h); r.θ[k] + (r.θ[k+1] - r.θ[k]) * (h - zc[k]) / (zc[k+1] - zc[k]))
    for h in (500.0, 1000.0)
        d = θat(r14, h) - BP.forcing_value(P.θ, h, t12)
        @info @sprintf("⟨θ⟩(%.0f m, 14 LT) = %.2f K, sounding %.2f K: %+.2f K", h, θat(r14, h),
                       BP.forcing_value(P.θ, h, t12), d)
        @test abs(d) < 1.5          # measured +0.12 K (500 m), -0.17 K (1000 m)
    end
    # residual layer at 03 LT: |dθ/dz| < 1.5 K/km between 300 m and 0.8 z_i,max
    zimax = maximum(r.zi_parcel for r in p.diag)
    r03 = rec(3.0)
    bulk = (θat(r03, 0.8zimax) - θat(r03, 300.0)) / (0.8zimax - 300)
    @info @sprintf("residual layer 03 LT: bulk dθ/dz %.2f K/km over 300-%.0f m; z_i,max %.0f m",
                   1e3bulk, 0.8zimax, zimax)
    @test abs(bulk) < 1.5e-3        # measured 0.26 K/km (z_i,max 1529 m)
    # the night nudging against the daytime surface flux (Heinze et al. 2017)
    night = BP.heat_budget(p.heat; window = (20, 5))
    day = BP.heat_budget(p.heat; window = (10, 16))
    nn = night.terms[(:dynamics, :nudging)] / night.elapsed
    hs = day.terms[(:surface_flux, :tiles)] / day.elapsed
    @info @sprintf("nudging 20-05 LT %.2f W/m², surface 10-16 LT %.1f W/m² (ratio %.3f); 24 h: surface %.1f, radiation %.1f, subsidence %.1f, nudging %.1f, damping %.1f, top %.3f W/m²",
                   nn, hs, nn / hs,
                   (BP.heat_budget(p.heat).terms[k] / 86400 for k in ((:surface_flux, :tiles), (:column_tendency, :column_radiation),
                    (:dynamics, :subsidence), (:dynamics, :nudging), (:dynamics, :damping), (:dynamics, :top_flux)))...)
    # measured: 8.5 W/m² at night against 299 W/m² by day (ratio 0.028). The
    # observed free troposphere warmed by 3.7-6.5 K between the 04:45 UTC
    # launches of the 29th and the 30th (+6.0 K at 2 km, the warm-air
    # advection of the heat wave): over 24 h the nudging (12 W/m²) and the
    # damping (23 W/m²) carry part of that into the box, above 1.2 z_i.
    @test abs(nn) < 0.05 * hs
    hb = BP.heat_budget(p.heat)
    @test abs(hb.closure) < 1e-10 * maximum(abs, values(hb.terms))
end

@testset "5. Boussinesq against anelastic (uniform grid, CBL to ~2 km)" begin
    function cbl(core; N = 24, Nz = 40, L = 3000.0, H = 3000.0, Q = 0.25, Γ = 3e-3, hours = 5.0, dt = 5.0)
        m = RegularGrid(; size = (N, N, Nz), extent = (L, L, H), halo = (3, 3, 3))
        θf(z) = 300 + Γ * z
        bcs = FieldBoundaryConditions(:periodic, :periodic, :wall)
        θb = FieldBoundaryConditions(:periodic, :periodic; bottom = :noflux, top = FluxBC(Γ))
        kw = (θ_flux_bottom = Q, sponge_α_max = 0.01, sponge_z_start = 0.75H, bcs = bcs, θ_bcs = θb)
        sim = core === :boussinesq ? BoussinesqSimulation(m; θ₀ = 300.0, kw...) :
              Aeolus.Anelastic.AnelasticSimulation(m; ref = Aeolus.Anelastic.reference_from_profile(m, θf, 101325.0), kw...)
        set!(sim.θ, (x, y, z) -> θf(z))
        rng = Xoshiro(5)
        interior_view(sim.θ)[:, :, 1:2] .+= 0.2 .* (rand(rng, N, N, 2) .- 0.5)
        c = TKEClosure(m; transport = true)
        initialize_tke!(c, sim; e_min = 1e-4)
        for _ in 1:round(Int, hours * 3600 / dt)
            step!(sim, dt; sgs = c)
        end
        zf, fl = BP.mean_heat_flux(sim, c.κt)
        zi = boundary_layer_height(zf, fl; method = :min_flux)
        θm = vec(sum(interior_view(sim.θ); dims = (1, 2))) ./ N^2
        zc = [(k - 0.5) * m.dz for k in 1:Nz]
        ml = [θm[k] for k in 1:Nz if 0.2zi < zc[k] < 0.8zi]
        return (zi = zi, θml = sum(ml) / length(ml))
    end
    b = cbl(:boussinesq); a = cbl(:anelastic)
    dzi = (a.zi - b.zi) / b.zi; dθ = a.θml - b.θml
    @info @sprintf("core sensitivity, 5 h CBL (Q = 0.25 K m/s, Γ = 3 K/km): z_i Boussinesq %.0f m, anelastic %.0f m (Δz_i/z_i = %+.3f); θ_ML %.2f vs %.2f K (Δθ_ML = %+.2f K)",
                   b.zi, a.zi, dzi, b.θml, a.θml, dθ)
    @test 1700 < b.zi < 2500        # "to about 2 km"
    # the anelastic CBL is deeper and warmer: the same kinematic surface flux
    # heats air whose density falls with height (ρ̄ is ~20 % lower at 2 km)
    @test 0 <= dzi < 0.2            # measured +0.107
    @test 0 <= dθ < 0.5             # measured +0.23 K
end

end
