# P58-d (#194), ONE-g (#456): GABLS3 (Cabauw, 2 July 2006; Basu 2008 case
# description, Bosveld et al. 2014a,b) on the one model — the case data,
# the dynamic driver and the forcing as examples/sbl/gabls3.toml builds
# them, the surface-layer diagnostics and towers on a briefly stepped
# domain, and the provenance of the GABLS4 reference table. The reduced
# 00-04 UTC run (momentum and heat budgets, first-cell u*/θ*, the night
# jet) is in validation/era6_gabls3.jl.

using Test
using Aeolus
using Aeolus.Components: Components, LargeScale, SurfaceLayer, Sponge, step_adaptive!
using Aeolus.Fields: interior_view
using Aeolus.Checks: read_reference
using Statistics: mean

isdefined(Main, :gabls3_data) || include(joinpath(@__DIR__, "support", "sbl_data.jl"))
# gabls3_domain, _g3comp: shared with validation/era6_gabls3.jl
isdefined(@__MODULE__, :gabls3_domain) || include(joinpath(@__DIR__, "support", "sbl_domains.jl"))

@testset "GABLS3 on the one model: case, forcing, diagnostics (P58-d, ONE-g)" begin

@testset "Case data, the driver and the forcing" begin
    d = gabls3_data()
    tab = sbl_read_csv(joinpath(SBL_DATA, "gabls3_case.csv"))
    @test any(occursin("Revised: October 6, 2008", l) for l in tab.header)   # provenance
    @test size(d.wind) == (25, 3) && size(d.thermo) == (19, 4) && size(d.surface) == (10, 4)
    d0 = gabls3_domain(; N = (8, 8, 8))           # the case as gabls3.toml builds it
    sl = _g3comp(d0, SurfaceLayer); ls = _g3comp(d0, LargeScale)
    @test sl.z0m == 0.15 && sl.z0h == 0.25
    @test _g3comp(d0, Components.Coriolis).f == coriolis_parameter(51.9711)
    @test _g3comp(d0, Sponge).kind === :mean
    F = (U_g = ls.U_g, V_g = ls.V_g, adv = only(ls.extra), T_s = sl.Ts)
    # geostrophic wind: (-6.5, 4.5) at 23 UTC, (-5, 4.5) at 03 UTC, linear in t,
    # linear in z to (-2, 2) at 2000 m (case description)
    @test forcing_value(F.U_g, 0.0, -3600.0) ≈ -6.5 && forcing_value(F.U_g, 0.0, 10800.0) ≈ -5.0
    @test forcing_value(F.U_g, 0.0, 3600.0) ≈ -6.5 + 1.5 * (2 / 4)
    @test forcing_value(F.U_g, 1000.0, 10800.0) ≈ (-5.0 - 2.0) / 2
    @test forcing_value(F.V_g, 2000.0, 0.0) ≈ 2.0
    # advection: constant above 200 m, linear to zero at the surface
    @test F.adv isa AdvectiveTendencies
    @test forcing_value(F.adv.u, 500.0, 0.0) ≈ 5e-4 && forcing_value(F.adv.u, 100.0, 0.0) ≈ 2.5e-4
    @test forcing_value(F.adv.θ, 300.0, 7200.0) ≈ 7.5e-5 && forcing_value(F.adv.θ, 300.0, 0.0) ≈ -2.5e-5
    @test forcing_value(F.adv.u, 300.0, 4 * 3600.0) == 0.0
    @test surface_temperature(F.T_s, 1800.0) ≈ (291.28 + 290.34) / 2
    # the driver's tables are the case table's columns
    for r in 1:size(d.geostrophic, 1)
        t = d.geostrophic[r, 1]
        @test forcing_value(F.U_g, 0.0, t) ≈ d.geostrophic[r, 2] && forcing_value(F.V_g, 0.0, t) ≈ d.geostrophic[r, 3]
    end
    @test [surface_temperature(F.T_s, t) for t in d.surface[:, 1]] ≈ d.surface[:, 3]
    # the initial profiles of the TOML: u, v log below 10 m from z0, θ linear from the 0.25 m value
    s = d0.state; m = s.mesh; z = [(k - 0.5) * m.dz for k in 1:m.Nz]
    U0 = [mean(interior_view(s.u)[:, :, k]) for k in 1:m.Nz]
    k1 = findfirst(>=(10.0), z)
    @test all(isapprox(U0[k], d.wind[1, 2] * log(z[k] / 0.15) / log(10 / 0.15); rtol = 0.2) for k in 1:k1-1)
    # the case's MOST functions: ψ_M = ψ_H = -5 z/L in stable conditions
    @test psi_m(sl.sf, 0.3) ≈ -1.5 && psi_h(sl.sf, 0.3) ≈ -1.5
    # the envelope file carries its provenance and the digitisation uncertainty
    env = gabls3_envelope()
    @test any(occursin("Basu, Holtslag", l) for l in env.header)
    @test sbl_csv_value(env, "jet_height", "low") == 100 && sbl_csv_value(env, "jet_height", "high") == 130
    # the digitised median profile has its jet at 105-120 m with 8.37-8.42 m/s
    ref = sbl_read_csv(joinpath(SBL_DATA, "gabls3_reference.csv"))
    zr = [r[1] for r in ref.rows]; Sr = [hypot(r[4], r[9]) for r in ref.rows]
    ok = .!isnan.(Sr); kr = argmax(Sr[ok])
    @test 105 <= zr[ok][kr] <= 120 && 8.3 <= Sr[ok][kr] <= 8.5
    # ... and is what Checks.read_reference gives the model's checks (z_m header, NaN gaps)
    rp = read_reference(joinpath(SBL_DATA, "gabls3_reference.csv"))
    @test rp["u_p50"].z == zr && isequal(rp["u_p50"].values, [r[4] for r in ref.rows])
end

@testset "Surface-layer diagnostics and towers" begin
    # A GABLS3 domain after 3 steps (the stable night start, ν_t non-zero),
    # not the 4 h run: the screen-level identity and the towers are
    # identities on any state with a stable surface layer (96 % of the
    # columns have L > 0 after 3 steps, t = 5.7 s; 2 m θ is then 291.69 K
    # between θ_0.25 = 291.28 K and θ(z1) = 291.85 K).
    d = gabls3_domain(); dt_prev = nothing
    for _ in 1:3
        _, dt_prev, _ = step_adaptive!(d; dt_prev = dt_prev)
    end
    c = _g3comp(d, SurfaceLayer); m = d.state.mesh; z1 = m.dz / 2
    # screen level: at z → z1 the MOST profiles return the first-level θ and wind
    Components.run_stage!(c, d, Val(:surface_flux), 0.0)
    sc = Components.sl_screen(c, d; z_θ = z1 * (1 - 1e-12), z_U = z1 * (1 - 1e-12))
    fl = [Components._first_level(d.state, i, j) for i in 1:m.Nx, j in 1:m.Ny]
    @test sc.θ_screen ≈ mean(f[3] for f in fl) rtol = 1e-9
    @test sc.U_screen ≈ mean(hypot(f[1], f[2]) for f in fl) rtol = 1e-9
    # 2 m θ lies between θ_0.25 and θ(z1) in the stable surface layer
    sc2 = Components.sl_screen(c, d)
    @test surface_temperature(c.Ts, d.state.t) < sc2.θ_screen < sc.θ_screen
    # virtual towers on a frozen state: zero resolved variance, V_TKE = √⟨e_sgs⟩
    s = d.state
    T = tower_sampler(m, [10.0, 50.0]; window = 2.0)
    sample_towers!(T, s.u, s.v, s.w, s.νt, 1.0); sample_towers!(T, s.u, s.v, s.w, s.νt, 1.0)
    @test T.nseg == 1 && sum(T.count) == 2 * m.Nx * m.Ny
    for (l, k) in enumerate(T.ks)
        Δ = cbrt(m.dx * m.dy * m.dz)
        es = [(s.νt.data[m.Hx+i, m.Hy+j, m.Hz+k] / (0.1Δ))^2 for i in 1:m.Nx, j in 1:m.Ny]
        V = [hypot(0.5 * (s.u.data[m.Hx+i, m.Hy+j, m.Hz+k] + s.u.data[m.Hx+i+1, m.Hy+j, m.Hz+k]),
                   0.5 * (s.v.data[m.Hx+i, m.Hy+j, m.Hz+k] + s.v.data[m.Hx+i, m.Hy+j+1, m.Hz+k]))
             for i in 1:m.Nx, j in 1:m.Ny]
        @test sum(T.sumE[:, l]) ≈ sum(sqrt.(es)) rtol = 1e-10     # V_TKE = √e_sgs per tower
        @test sum(T.sumV[:, l]) ≈ sum(V) rtol = 1e-10
    end
    # hockey-stick fit: an exact two-segment line returns its break point
    V = collect(0.5:0.5:10.0); E = [v < 5.2 ? 0.2 + 0.02v : 0.2 + 0.02 * 5.2 + 0.25 * (v - 5.2) for v in V]
    h = hockey_stick_threshold(V, E)
    @test h.V_s ≈ 5.2 atol = 0.03
    @test h.slope1 ≈ 0.02 atol = 1e-3
    @test h.slope2 ≈ 0.25 atol = 1e-3
end

@testset "GABLS4 reference provenance (Couvreux et al. 2020, Tables 2-4; #237)" begin
    ref = gabls4_reference()
    col(name) = findfirst(==(name), ref.columns)
    get(r, name) = r[col(name)]
    rows(kind) = [r for r in ref.rows if r[1] == kind]
    e3 = rows("exp3"); e2 = rows("exp2")
    @test length(e3) == 7 && length(e2) == 7
    # Table 2: six experiment-3 runs at 1 m isotropic, CSIRO-LES at 3.9 m / 1.17 m
    fine = [r for r in e3 if get(r, "dx_m") == 1 && get(r, "dz_m") == 1]
    @test length(fine) == 6 && all(r[2] != "CSIRO-LES" for r in fine)
    csiro = only(r for r in e3 if r[2] == "CSIRO-LES")
    @test get(csiro, "dx_m") == 3.9 && get(csiro, "dz_m") == 1.17
    # the experiment-3 envelope is that of the six 1 m runs; CSIRO alone sets the old 25 m top
    lo = only(rows("envelope_low")); hi = only(rows("envelope_high"))
    for q in ("h_m", "L_m", "u_star", "H_Wm2", "theta_star")
        @test get(lo, q) == minimum(get(r, q) for r in fine)
        @test get(hi, q) == maximum(get(r, q) for r in fine)
    end
    @test (get(lo, "h_m"), get(hi, "h_m")) == (14.5, 19.5)
    @test mean(get(r, "h_m") for r in fine) ≈ 17.4 atol = 0.05
    @test get(csiro, "h_m") == maximum(get(r, "h_m") for r in e3) == 25
    # the checks file of gabls4_exp3.toml restates that envelope, one quantity per row
    env = read_reference(joinpath(SBL_DATA, "gabls4_exp3_envelope.csv"))
    for (q, c) in (("bl_height", "h_m"), ("obukhov_length", "L_m"), ("u_star", "u_star"), ("surface_heat_flux", "H_Wm2"))
        @test (env[q].low, env[q].high) == (get(lo, c), get(hi, c))
    end
    # experiment 2: MicroHH and DALES rows are their experiment-1 runs (z0m = 1e-2 m)
    for m in ("MicroHH", "DALES")
        r = only(x for x in e2 if x[2] == m)
        @test endswith(get(r, "run"), "E1") && get(r, "z0m") == 1e-2
    end
    e2ref = [r for r in e2 if endswith(get(r, "run"), "E2") && get(r, "dz_m") == 2 && get(r, "z0m") == 1e-3]
    @test sort([r[2] for r in e2ref]) == ["MONC", "MesoNH", "PALM", "UConn"]
    lo2 = only(rows("envelope_exp2_low")); hi2 = only(rows("envelope_exp2_high"))
    for q in ("h_m", "L_m", "u_star", "H_Wm2", "theta_star")
        @test get(lo2, q) == minimum(get(r, q) for r in e2ref)
        @test get(hi2, q) == maximum(get(r, q) for r in e2ref)
    end
    # Table 4's H is in W m-2 (its header says K m s-1): H = -ρ c_p u* θ*. Each row,
    # with u*, θ* rounded to ±0.005, bounds ρ c_p; one ρ c_p must fit all 14 rows and
    # be that of air at 651 hPa between 220 and 250 K (911-1036 J m-3 K-1). Measured:
    # the rows overlap in 991-992 J m-3 K-1 (ρ = 0.99 kg m-3, 230 K).
    ρcp(T) = 65100 / (287.05 * T) * 1005
    b = [(-get(r, "H_Wm2") / ((get(r, "u_star") + 0.005) * (get(r, "theta_star") + 0.005)),
          -get(r, "H_Wm2") / ((get(r, "u_star") - 0.005) * (get(r, "theta_star") - 0.005))) for r in vcat(e2, e3)]
    lo_ρcp, hi_ρcp = maximum(first.(b)), minimum(last.(b))
    @test lo_ρcp <= hi_ρcp
    @test ρcp(250) <= lo_ρcp && hi_ρcp <= ρcp(220)
end

end # testset
