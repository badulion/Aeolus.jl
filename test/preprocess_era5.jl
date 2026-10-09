# The ERA5 preprocessor of the dynamic driver (CFG-h, #363):
# src/preprocess/era5.jl, the `simulation` driver's `[initial] dynamic`,
# open faces and `[preprocess.era5]`, `aeolus preprocess dynamic --from-era5`.
#
# Acceptance tests of the issue covered here:
#   * the analytic fixture (an isothermal hydrostatic atmosphere, T = 280 K
#     at the site, every field linear in the local x, y; written in-test
#     with NCDatasets, test/support/era5_analytic.jl): the recovered height
#     of each pressure level matches z = (R T/g₀) ln(p₀/p) to 1e-9, θ
#     matches T (p₀/p)^κ to 1e-12, the geostrophic wind the prescribed Φ
#     gradient to 1e-10, the advective tendency of the linear θ field
#     −u·∇θ to 1e-10;
#   * GRIB and NetCDF agree: the same fields as GRIB2 (the committed
#     fixture test/data/era5/analytic_*.grib2, 64-bit IEEE packed) give the
#     same driver bit for bit;
#   * the real sample (examples/data/era5_lindenberg_20190629_*.nc, ERA5
#     at 00 UTC 29 June 2019) against the bundled Lindenberg sounding over
#     200-3000 m: θ RMSE ≤ 1.5 K, qv RMSE ≤ 1.5 g/kg (measured values
#     printed);
#   * errors: a run top above the ERA5 top, a domain outside the file's
#     lat/lon box, both reported with the numbers;
#   * round trip, bit-identical under ESTIMATE FFTW: a 16³ open-boundary
#     `simulation`-driver run from the ERA5-made driver, 20 steps, with the
#     file hashed under [inputs].
# Plus: model levels (the IFS hydrostatic integration, exact half-level
# heights for an isothermal column), the boundary planes equal the
# analytic field, the time axis, the surface series.

using Test
using Printf
using Dates
using Aeolus
using Aeolus.Config
using Aeolus.InputDrivers
using Aeolus.Grids: RegularGrid
using Aeolus.BoundaryConditions: open_boundary_conditions, open_boundary_states
using Aeolus.LargeScaleForcing: forcing_value
using Aeolus.MoistThermodynamics: DEFAULT_CONSTANTS
using Aeolus.Fields: interior_view
using Aeolus.IO: read_run_toml
import NCDatasets

include(joinpath(@__DIR__, "support", "era5_analytic.jl"))
include(joinpath(@__DIR__, "config_roundtrip_harness.jl"))

const E5_GRIB = [joinpath(@__DIR__, "data", "era5", "analytic_pl.grib2"),
                 joinpath(@__DIR__, "data", "era5", "analytic_sfc.grib2")]
const E5_DATA = joinpath(pkgdir(Aeolus), "examples", "data")
const E5_SAMPLE = [joinpath(E5_DATA, "era5_lindenberg_20190629_pl.nc"), joinpath(E5_DATA, "era5_lindenberg_20190629_sfc.nc")]
const E5_CFG = joinpath(pkgdir(Aeolus), "examples", "configs", "lindenberg_era5.toml")
const E5_KAPPA = EA_RD / DEFAULT_CONSTANTS.cp_dry
const E5_LAT, E5_LON = EA_SITE
const E5_PA = 100 .* EA_PLEV

e5_err(f) = try
    f(); ""
catch e
    sprint(showerror, e)
end
relerr(a, b) = maximum(abs.(a .- b) ./ max.(abs.(b), eps()))

@testset "meters_to_latlon inverts latlon_to_meters" begin
    for (la, lo) in ((52.6, 13.4), (51.3, 15.9), (52.21, 14.12))
        x, y = latlon_to_meters(la, lo, E5_LAT, E5_LON)
        la2, lo2 = meters_to_latlon(x, y, E5_LAT, E5_LON)
        @test abs(la2 - la) < 1e-12 && abs(lo2 - lo) < 1e-12
    end
    # 1 km north is 1000/R radians of latitude
    @test meters_to_latlon(0.0, 1000.0, E5_LAT, E5_LON)[1] - E5_LAT ≈ rad2deg(1000.0 / 6_371_000.0) rtol = 1e-9
end

@testset "ERA5: the analytic fixture (heights, θ, geostrophic wind, advection, subsidence)" begin
    e = read_era5(ea_write_netcdf(mktempdir()))
    @test e.levels === :pressure && e.lat == EA_LAT && e.lon == EA_LON && e.time == EA_TIMES
    @test e.p[1, 1, :, 1] == E5_PA                           # bottom first, Pa
    C = era5_columns(e; lat = E5_LAT, lon = E5_LON, box = 0.5)
    z_an = ea_height.(E5_PA)
    rz = maximum(abs.(C.z[:, 1] .- z_an)[2:end] ./ z_an[2:end])
    @test abs(C.z[1, 1]) < 1e-9 && rz < 1e-9                 # 1000 hPa: the ground (z = 0)
    θ_an = EA_T0 .* (EA_P0 ./ E5_PA) .^ E5_KAPPA
    rθ = relerr(C.θ[:, 1], θ_an)
    @test rθ < 1e-12
    f = coriolis_parameter(E5_LAT)
    @test f ≈ 2 * 7.292115e-5 * sind(E5_LAT)
    rug = relerr(C.u_g, fill(-EA_PHI_Y / f, size(C.u_g))); rvg = relerr(C.v_g, fill(EA_PHI_X / f, size(C.v_g)))
    @test rug < 1e-10 && rvg < 1e-10
    # −u·∇θ with the wind at the centre (u grows with height and by 1 m/s per record)
    adv = [-((EA_U + (k - 1) + EA_U_Z * ea_height(p)) * EA_T_X + EA_V * EA_T_Y) * (EA_P0 / p)^E5_KAPPA
           for p in E5_PA, k in eachindex(EA_TIMES)]
    radv = relerr(C.θ_adv, adv)
    @test radv < 1e-10
    # −u·∇q passes through zero (u = 6 m/s at 1000 hPa, record 2): relative to its largest value
    qadv = [-((EA_U + (k - 1) + EA_U_Z * ea_height(p)) * EA_Q_X + EA_V * EA_Q_Y) for p in E5_PA, k in eachindex(EA_TIMES)]
    rqadv = maximum(abs.(C.qv_adv .- qadv)) / maximum(abs.(qadv))
    @test rqadv < 1e-10
    # subsidence w = −ω/(ρ g₀), ρ = p/(R_m T), R_m = R_d (1 − q) + R_v q. w is a
    # product of T and q, which are linear in x and y, so it is not bilinear:
    # exact at a grid node, within the bilinear error between nodes
    Rv = DEFAULT_CONSTANTS.gas_constant / DEFAULT_CONSTANTS.molmass_water
    w_an(la, lo) = [-ea_omega(p, la, lo, 1) * (EA_RD * (1 - ea_q(p, la, lo)) + Rv * ea_q(p, la, lo)) *
                    ea_T(p, la, lo) / (p * EA_G0) for p in E5_PA]
    Cn = era5_columns(e; lat = 52.25, lon = 14.0, box = 0.5)
    rwn = relerr(Cn.w[:, 1], w_an(52.25, 14.0)); rw = relerr(C.w[:, 1], w_an(E5_LAT, E5_LON))
    @test rwn < 1e-12 && rw < 1e-6 && all(>(0), C.w)        # ω < 0: rising air
    @info @sprintf("ERA5 analytic: height rel. err %.2e, θ %.2e, u_g %.2e, v_g %.2e, θ_adv %.2e, qv_adv %.2e, w at a node %.2e, between nodes %.2e (u_g = %.4f m/s, v_g = %.4f m/s)",
                   rz, rθ, rug, rvg, radv, rqadv, rwn, rw, -EA_PHI_Y / f, EA_PHI_X / f)
end

@testset "ERA5: the driver's tables, profiles, times and surface series" begin
    e = read_era5(ea_write_netcdf(mktempdir()))
    zr = collect(25.0:50.0:1575.0)
    d = era5_dynamic_inputs(e; lat = E5_LAT, lon = E5_LON, z = zr)
    @test d.init_time == "2019-06-29T00:00:00Z" && d.attributes["dynamic_source"] == "era5"
    @test sort!(collect(keys(d.tables))) == sort!([:profile_theta, :profile_qv, :profile_u, :profile_v, :profile_p,
                                                   :u_g, :v_g, :w_subs, :theta_adv, :qv_adv])
    @test all(tab -> tab.t == [0.0, 3600.0, 7200.0] && tab.z == zr, values(d.tables))
    # an isothermal atmosphere: ln p is linear in z, so the ln-p interpolation is exact between levels
    p_an = [EA_P0 * exp(-EA_G0 * z / (EA_RD * EA_T0)) for z in zr, _ in 1:3]
    @test relerr(d.tables[:profile_p].values, p_an) < 1e-12
    # u is linear in height: exact on the run's z; u_g is uniform
    @test relerr(d.tables[:profile_u].values, [EA_U + (k - 1) + EA_U_Z * z for z in zr, k in 1:3]) < 1e-12
    f = coriolis_parameter(E5_LAT)
    @test relerr(d.tables[:u_g].values, fill(-EA_PHI_Y / f, length(zr), 3)) < 1e-10
    # the initial profiles: the tables at the first record
    @test d.profiles.z == zr && d.profiles.theta == d.tables[:profile_theta].values[:, 1]
    @test d.profiles.u == d.tables[:profile_u].values[:, 1] && d.p_surface == EA_P0
    # θ between levels: linear in z between the level values (θ = T (p₀/p)^κ grows with height)
    @test issorted(d.profiles.theta) && d.profiles.theta[1] > EA_T0
    # the surface series: skt as a potential temperature at sp = p_ref (Π = 1)
    @test d.surface.t == [0.0, 3600.0, 7200.0] && d.surface.temperature == [290.0, 291.0, 292.0]
    @test d.surface.pressure == fill(EA_P0, 3)
    # a later start: seconds since that time
    d2 = era5_dynamic_inputs(e; lat = E5_LAT, lon = E5_LON, z = zr, init_time = EA_TIMES[2], outputs = (:large_scale,))
    @test d2.init_time == "2019-06-29T01:00:00Z" && d2.tables[:profile_theta].t == [-3600.0, 0.0, 3600.0]
    @test d2.profiles === nothing && !haskey(d2.tables, :u_g)
    # through the file and back, as LargeScaleForcing objects
    path = joinpath(mktempdir(), "era5_dyn.nc")
    m = RegularGrid(; size = (8, 8, 32), extent = (800.0, 800.0, 1600.0))
    write_era5_driver(path, e; lat = E5_LAT, lon = E5_LON, mesh = m)
    r = read_dynamic_driver(path; planes = false)
    @test r.tables[:theta_adv].values == d.tables[:theta_adv].values
    @test r.attributes["origin_lat"] == E5_LAT && r.attributes["origin_x"] == 400.0 && r.profiles.theta == d.profiles.theta
    F = dynamic_forcing(r)
    @test F.U_g isa Aeolus.LargeScaleForcing.ForcingTable && length(F.extra) == 2   # AdvectiveTendencies, Subsidence
    @test forcing_value(F.U_g, 800.0, 1800.0) ≈ -EA_PHI_Y / f rtol = 1e-10
end

@testset "ERA5: model levels (IFS hydrostatic integration)" begin
    # a dry isothermal column on 6 hybrid levels: the half-level heights are
    # exact, z_{k+1/2} = (R_d T/g₀) ln(p_s/p_{k+1/2}), and the full level lies
    # α_k R_d T/g₀ above its lower half level
    a = [0.0, 2000.0, 5000.0, 8000.0, 6000.0, 2000.0, 0.0]          # Pa, top down
    b = [0.0, 0.0, 0.05, 0.2, 0.5, 0.8, 1.0]
    ps = 101_300.0; T0 = 280.0
    lon = [14.0, 14.25]; lat = [52.0, 52.25]; nl = 6
    function ml_file(path; coefficients = true)
        NCDatasets.NCDataset(path, "c") do ds
            NCDatasets.defVar(ds, "longitude", lon, ("longitude",))
            NCDatasets.defVar(ds, "latitude", lat, ("latitude",))
            NCDatasets.defVar(ds, "model_level", Float64.(1:nl), ("model_level",))
            NCDatasets.defVar(ds, "valid_time", [Dates.value(EA_TIMES[1] - DateTime(1970)) ÷ 1000], ("valid_time",);
                              attrib = ["units" => "seconds since 1970-01-01"])
            dims = ("longitude", "latitude", "model_level", "valid_time")
            NCDatasets.defVar(ds, "t", fill(T0, 2, 2, nl, 1), dims)
            NCDatasets.defVar(ds, "q", zeros(2, 2, nl, 1), dims)
            if coefficients
                NCDatasets.defVar(ds, "a_half", a, ("half_level",)); NCDatasets.defVar(ds, "b_half", b, ("half_level",))
            end
            # as ERA5 delivers them: lnsp and the orography z on model level 1
            sdims = ("longitude", "latitude", "model_level1", "valid_time")
            NCDatasets.defVar(ds, "lnsp", fill(log(ps), 2, 2, 1, 1), sdims)
            NCDatasets.defVar(ds, "z", fill(EA_G0 * 30.0, 2, 2, 1, 1), sdims)
        end
        return path
    end
    dir = mktempdir()
    path = ml_file(joinpath(dir, "ml.nc"))
    e = read_era5(path; levels = :model)
    @test e.levels === :model && e.sp[1, 1, 1] ≈ ps && e.zs[1, 1] ≈ 30.0
    ph = a .+ b .* e.sp[1, 1, 1]                                     # half levels, top down (p_s = exp(lnsp))
    H = EA_RD * T0 / EA_G0
    for m in 1:nl                                                    # m = 1: level 6, the bottom
        k = nl - m + 1
        lo, hi = ph[k+1], ph[k]
        α = k == 1 ? log(2.0) : 1 - hi / (lo - hi) * log(lo / hi)
        @test e.p[1, 1, m, 1] == (lo + hi) / 2
        @test e.z[1, 1, m, 1] ≈ 30.0 + H * (log(ps / lo) + α) rtol = 1e-13
        k == 1 || @test 30.0 + H * log(ps / lo) < e.z[1, 1, m, 1] < 30.0 + H * log(ps / hi)
    end
    # the geostrophic wind needs pressure levels
    @test occursin("pressure levels", e5_err(() -> era5_dynamic_inputs(e; lat = 52.1, lon = 14.1, z = [10.0])))
    d = era5_dynamic_inputs(e; lat = 52.1, lon = 14.1, z = [10.0, 100.0], outputs = (:profiles,))
    @test d.profiles.theta ≈ T0 .* (1e5 ./ [e.p[1, 1, 1, 1], e.p[1, 1, 1, 1]]) .^ E5_KAPPA rtol = 1e-12  # below the lowest level: its value
    nocoef = ml_file(joinpath(dir, "ml_nocoef.nc"); coefficients = false)
    @test occursin("hybrid coefficients", e5_err(() -> read_era5(nocoef; levels = :model)))
    # the coefficients can be passed instead: the same heights
    @test read_era5(nocoef; levels = :model, hybrid = (a, b)).z == e.z
end

@testset "ERA5: GRIB2 and NetCDF give the same driver, bit for bit" begin
    @test all(isfile, E5_GRIB) && sum(filesize, E5_GRIB) <= 200 * 1024
    en = read_era5(ea_write_netcdf(mktempdir())); eg = read_era5(E5_GRIB)
    for f in (:lon, :lat, :time, :p, :z, :T, :q, :u, :v, :omega, :sp, :skt, :t2m, :zs)
        @test isequal(getfield(en, f), getfield(eg, f))
    end
    zr = collect(25.0:50.0:1575.0)
    dn = era5_dynamic_inputs(en; lat = E5_LAT, lon = E5_LON, z = zr)
    dg = era5_dynamic_inputs(eg; lat = E5_LAT, lon = E5_LON, z = zr)
    @test keys(dn.tables) == keys(dg.tables) && all(k -> isequal(dn.tables[k].values, dg.tables[k].values), keys(dn.tables))
    @test isequal(dn.profiles, dg.profiles) && isequal(dn.surface, dg.surface)
    # the boundary planes too
    m = RegularGrid(; size = (8, 8, 16), extent = (1600.0, 1600.0, 1600.0))
    planes() = only(open_boundary_states(open_boundary_conditions(m, nothing; fields = (:u, :v, :w, :θ, :qv)))).planes
    pn = joinpath(mktempdir(), "n.nc"); pg = joinpath(mktempdir(), "g.nc")
    write_era5_driver(pn, en; lat = E5_LAT, lon = E5_LON, mesh = m, planes = planes())
    write_era5_driver(pg, eg; lat = E5_LAT, lon = E5_LON, mesh = m, planes = planes())
    bc(p) = NCDatasets.NCDataset(ds -> Dict(k => Array(ds[k].var[:]) for k in keys(ds) if startswith(k, "bc_")), p)
    bn, bg = bc(pn), bc(pg)
    @test !isempty(bn) && keys(bn) == keys(bg) && all(k -> isequal(bn[k], bg[k]), keys(bn))
    @info "ERA5: GRIB2 (64-bit IEEE packed, $(sum(filesize, E5_GRIB)) bytes) and NetCDF give the same driver bitwise: " *
          "$(length(dn.tables)) tables, $(length(bn)) boundary slabs"
end

@testset "ERA5: the boundary planes are the analytic field" begin
    e = read_era5(E5_GRIB)
    m = RegularGrid(; size = (8, 8, 16), extent = (3200.0, 3200.0, 1600.0), halo = (2, 2, 2))
    bcs = open_boundary_conditions(m, nothing; fields = (:u, :v, :w, :θ, :qv))
    path = joinpath(mktempdir(), "planes.nc")
    write_era5_driver(path, e; lat = E5_LAT, lon = E5_LON, mesh = m, planes = only(open_boundary_states(bcs)).planes)
    d = read_dynamic_driver(path; mesh = m)
    f = d.boundary
    @test f.time == [0.0, 3600.0, 7200.0] && isnan(f.step[1]) && f.step[2:end] == [3600.0, 3600.0]
    planes = only(open_boundary_states(bcs)).planes
    check_planes(f, planes)
    # u = EA_U + (k − 1) + EA_U_Z (z + z_s(x, y)) + EA_U_X x above the ground,
    # the 1000 hPa value (z = 0) in the halo below it; x, y about the centre
    worst = 0.0
    for k in 1:3
        feed_window!(f, min(k, 2)); s = k == 3 ? f.data.b : f.data.a
        for p in planes
            (p.name === :u && p.mode !== :mirror) || continue
            A = Aeolus.BoundaryConditions.snapshot_values(s, :u)
            for (c, z) in enumerate(p.z), (b, y) in enumerate(p.y), (a, x) in enumerate(p.x)
                la, lo = meters_to_latlon(x - 1600.0, y - 1600.0, E5_LAT, E5_LON)
                zs = ea_phi(EA_P0, la, lo) / EA_G0
                want = EA_U + (k - 1) + EA_U_Z * (max(z, 0.0) + zs) + EA_U_X * (x - 1600.0)
                got = A[p.irange[a], p.jrange[b], p.krange[c]]
                worst = max(worst, abs(got - want))
            end
        end
    end
    @info @sprintf("ERA5 planes: largest |u − u_analytic| over the open faces, 3 records: %.2e m/s", worst)
    @test worst < 1e-9
end

@testset "ERA5: errors name the numbers" begin
    e = read_era5(E5_GRIB)
    msg = e5_err(() -> era5_dynamic_inputs(e; lat = E5_LAT, lon = E5_LON, z = [100.0, 7000.0]))
    @test occursin("z = 7000.0 m lies above the ERA5 top", msg) && occursin("5680.9 m", msg)
    msg = e5_err(() -> era5_dynamic_inputs(e; lat = 55.0, lon = E5_LON, z = [100.0]))
    @test occursin("(55.0000 N, 14.1200 E) lies outside the file's lat/lon box [51.2500, 53.2500] N × [13.0000, 15.0000] E", msg)
    # the box of the centred differences must fit too
    msg = e5_err(() -> era5_dynamic_inputs(e; lat = 53.0, lon = E5_LON, z = [100.0], box = 0.5))
    @test occursin("(53.5000 N", msg) && occursin("outside", msg)
    @test occursin("unknown output", e5_err(() -> era5_dynamic_inputs(e; lat = E5_LAT, lon = E5_LON, z = [1.0],
                                                                      outputs = (:nope,))))
    @test occursin("no file", e5_err(() -> read_era5("/nonexistent/era5.grib2")))
    m = RegularGrid(; size = (4, 4, 8), extent = (400.0, 400.0, 8000.0))
    @test occursin("above the ERA5 top", e5_err(() -> write_era5_driver(joinpath(mktempdir(), "x.nc"), e;
                                                                        lat = E5_LAT, lon = E5_LON, mesh = m)))
end

@testset "ERA5: the Lindenberg sample against the Lindenberg sounding (00 UTC, 200-3000 m)" begin
    @test all(isfile, E5_SAMPLE) && sum(filesize, E5_SAMPLE) <= 1024^2
    for f in E5_SAMPLE                                       # provenance and licence in the file
        NCDatasets.NCDataset(f) do ds
            @test occursin("Copernicus", ds.attrib["licence"]) && occursin("doi:10.24381", ds.attrib["attribution"])
            @test occursin("\"area\":[53.21,13.12,51.21,15.12]", ds.attrib["cds_request"])
        end
    end
    e = read_era5(E5_SAMPLE)
    @test e.time == [DateTime(2019, 6, 29, h) for h in 0:12] && size(e.T, 3) == 16 && e.p[1, 1, end, 1] == 50_000.0
    lat, lon = 52.2085, 14.1180                              # the launch site (IGRA2)
    C = era5_columns(e; lat, lon)
    snd = read_sounding(joinpath(pkgdir(Aeolus), "examples", "urban", "data", "lindenberg_10393_20190629.csv"))
    P = sounding_profiles(snd; z_top = 3000.0)
    l0 = snd.launches[1]                                     # "2019-06-29T00Z", released 22:45 UTC on the 28th
    zg = collect(200.0:50.0:3000.0)                          # m above the ground (ERA5's, the launch site's)
    θe = era5_vertical(C.z[:, 1], C.θ[:, 1], zg); qe = era5_vertical(C.z[:, 1], C.qv[:, 1], zg)
    θs = [forcing_value(P.θ, z, l0.t) for z in zg]; qs = [forcing_value(P.qv, z, l0.t) for z in zg]
    rms(a) = sqrt(sum(abs2, a) / length(a))
    rθ, rq = rms(θe .- θs), 1e3 * rms(qe .- qs)
    @info @sprintf("ERA5 vs the Lindenberg sounding, 00 UTC 29 June 2019, 200-3000 m above the ground: θ RMSE %.3f K (bias %+.3f K), qv RMSE %.3f g/kg (bias %+.3f g/kg); ERA5 ground %.1f m, launch site %.1f m",
                   rθ, sum(θe .- θs) / length(zg), rq, 1e3 * sum(qe .- qs) / length(zg), C.ground, snd.elevation)
    @test rθ <= 1.5
    @test rq <= 1.5
end

@testset "ERA5: aeolus preprocess dynamic --from-era5 and the open simulation box (16³, 20 steps, round trip)" begin
    dir = mktempdir()
    out = joinpath(dir, "lindenberg_era5_dynamic.nc")
    rc = redirect_stdout(devnull) do
        Aeolus.main(["preprocess", "dynamic", "--from-era5", E5_SAMPLE..., "--config", E5_CFG, "--out", out])
    end
    @test rc == 0 && isfile(out)
    d = read_dynamic_driver(out; planes = false)
    @test d.attributes["dynamic_source"] == "era5" && d.attributes["origin_lat"] == 52.2085
    @test d.attributes["preprocessor"] == "aeolus preprocess dynamic --from-era5"
    @test occursin("era5_lindenberg_20190629_sfc.nc", d.attributes["era5_files"])
    @test d.init_time == "2019-06-29T00:00:00Z" && size(d.tables[:profile_theta].values) == (16, 13)
    # usage errors exit with 2
    rc2 = redirect_stderr(devnull) do
        Aeolus.main(["preprocess", "dynamic", "--from-era5", E5_SAMPLE...])        # no --config
    end
    @test rc2 == 2
    # the round trip, the driver hashed under [inputs]
    r = roundtrip_check(E5_CFG; nsteps = 20, overrides = ["dynamic_driver=\"$out\""])
    @test r.fields_equal && r.diagnostics_equal && r.config_equal
    rec = read_run_toml(r.runs[1].folder.path)
    @test rec["inputs"]["domain.main.dynamic_driver"]["fnv1a64"] == content_hash(out)
    run1 = r.runs[1].obj
    sim1 = Aeolus.SimulationDriver.simulation(run1)
    feed1 = only(run1.model.domains).drivers.boundary          # the domain's dynamic-driver feed (ONE-a)
    @test feed1 isa FileBoundaryFeed && sim1.t == 40.0 && feed1.data.t0 == 0.0
    # the initial state is the ERA5 profile at the domain centre
    cfg0 = load_run_config(E5_CFG; overrides = ["dynamic_driver=\"$out\""])
    b0 = Config.build(Aeolus.SimulationDriver.SimulationDriverType(), cfg0)
    θ0 = interior_view(Aeolus.SimulationDriver.simulation(b0).θ)
    @test θ0[3, 5, :] == d.profiles.theta && θ0[1, 1, :] == d.profiles.theta
    # after 20 steps the open faces hold the ERA5 planes blended at t = 40 s
    st = only(open_boundary_states(sim1.bcs))
    pu = only(p for p in st.planes if p.name === :θ && p.side === :west)
    f = read_dynamic_driver(out; mesh = sim1.mesh).boundary
    A = Aeolus.BoundaryConditions.snapshot_values(f.data.a, :θ); B = Aeolus.BoundaryConditions.snapshot_values(f.data.b, :θ)
    τ = 40.0 / 3600.0
    want = [(1 - τ) * A[i, j, k] + τ * B[i, j, k] for i in pu.irange, j in pu.jrange, k in pu.krange]
    @test maximum(abs.(pu.data .- want)) < 1e-12
    @info @sprintf("open ERA5 box over Lindenberg: 20 steps to t = %.0f s, bitwise in the round trip; θ at the west face %.2f-%.2f K, u in the box %.2f-%.2f m/s",
                   sim1.t, extrema(pu.data)..., extrema(interior_view(sim1.u))...)
end

@testset "ERA5: the model's config checks" begin
    err(ov) = try
        Config.build(Aeolus.SimulationDriver.SimulationDriverType(), load_run_config(E5_CFG; overrides = ov)); nothing
    catch e
        e isa ConfigError || rethrow(); e
    end
    e1 = err(["dynamic_driver=\"none\""])
    @test e1 !== nothing && occursin("set the domain's dynamic_driver", sprint(showerror, e1))
    e2 = err(["initial.theta={kind=\"constant\", value=300.0}"])
    @test e2 !== nothing && occursin("give one of the two", sprint(showerror, e2))
    e3 = err(["preprocess.era5={lon=14.1}"])
    @test e3 !== nothing && occursin("lat: required", sprint(showerror, e3))
end
