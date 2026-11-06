# The urban site's initial soil (#521): the `[ground] soil_init` of the
# `soil_column` provider (GroundSoil) on the synthetic city
# (test/data/urban/small_city.toml), its dynamic driver the committed
# summer sounding driver with the soil appended as AeolusPrep.jl's ERA5
# soil appends it (PALM's PIDS init_soil_t / init_soil_m over zsoil):
#
#   * "driver" with a profile: every soil column holds the profile on its
#     nodes, the base at the deepest level (PALM's rule), the water clipped
#     into the texture's (θ_r, θ_s];
#   * "driver" with a volume: each column its own profile, a column without
#     data (NaN, not land) the mean profile of the others;
#   * "annual_wave": the analytic annual wave on the nodes, its value at the
#     base interface the base;
#   * "auto" without soil in the driver: the uniform t_soil0 of before;
#     "driver" without it: an error;
#   * a run from the driver's soil closes its budgets.
#
# The interpolation and the wave are tested in soil_thermal.jl; the driver
# variables in input_dynamic.jl.

using Test
using NCDatasets
using Dates: dayofyear, Date
using Aeolus
using Aeolus.Config
using Aeolus.Grids: xnodes, ynodes, CellCenter

const USI = Aeolus.Urban
const SCI = Aeolus.SoilColumn
include(joinpath(@__DIR__, "support", "berlin_configs.jl"))

const SI_CITY = joinpath(@__DIR__, "data", "urban", "small_city.toml")
const SI_SOUNDING = joinpath(pkgdir(Aeolus), "examples", "urban", "input", "berlin_20190629_dynamic.nc")
const SI_Z = [0.035, 0.175, 0.64, 1.945]                    # ERA5's layer centres

# the sounding driver with the soil appended (AeolusPrep.jl's write_soil!)
function soil_driver(t, m; xy = nothing)
    p = joinpath(mktempdir(), "drv.nc")
    cp(SI_SOUNDING, p); chmod(p, 0o644)
    NCDataset(p, "a") do ds
        ds.dim["zsoil"] = length(SI_Z)
        defVar(ds, "zsoil", SI_Z, ("zsoil",); attrib = ["units" => "m", "positive" => "down"])
        dims = ndims(t) == 3 ? ("x", "y", "zsoil") : ("zsoil",)
        if xy !== nothing
            ds.dim["x"] = length(xy[1]); ds.dim["y"] = length(xy[2])
            defVar(ds, "x", xy[1], ("x",); attrib = ["units" => "m"])
            defVar(ds, "y", xy[2], ("y",); attrib = ["units" => "m"])
        end
        defVar(ds, "init_soil_t", t, dims; fillvalue = NaN, attrib = ["units" => "K"])
        m === nothing || defVar(ds, "init_soil_m", m, dims; fillvalue = NaN, attrib = ["units" => "m3 m-3"])
    end
    return p
end
si_case(ov...) = (Aeolus.Operators.set_fftw_policy!(; planner = :estimate);
                  USI.build_case(USI.site_options(load_run_config(SI_CITY; overrides = ["fftw.planner=\"estimate\"", ov...]))))
soil_cols(case) = [(I, case.bus.surface[I]) for I in CartesianIndices(case.bus.surface)
                   if case.bus.surface[I] isa SCI.SoilThermalColumn]

@testset "urban soil init: the driver's profile (#521)" begin
    t, m = [295.0, 292.0, 288.0, 284.0], [0.15, 0.20, 0.25, 0.60]      # 0.60 > θ_s of loam (0.43)
    case = @test_logs (:warn, r"1 of 4 values outside") match_mode = :any si_case("dynamic_driver=\"$(soil_driver(t, m))\"")
    g = USI.find_coupling(case, :ground_soil)
    @test g.init[].kind === :driver
    cols = soil_cols(case)
    @test length(cols) == length(g.cols) > 0
    h = g.hydraulics
    z = cols[1][2].water.grid.z
    # the temperature through the enthalpy round trip (round-off); the base the deepest level (copied)
    @test all(((_, s),) -> isapprox(s.water.T_soil, SCI.soil_profile_on_nodes(SI_Z, t, z); rtol = 1e-12), cols)
    @test all(((_, s),) -> s.bottom.T == t[end], cols)
    # the driver's levels are clipped into (θ_r, θ_s], then interpolated
    θw = SCI.soil_profile_on_nodes(SI_Z, min.(m, SCI.soil_porosity(h)), z)
    @test all(((_, s),) -> isapprox(s.water.θ_liq, θw; rtol = 1e-12), cols)
    @test maximum(θw) == SCI.soil_porosity(h)
    # a run from it closes its budgets
    run_site!(case)
    @test USI.run_results(case)["budgets"]["closed"] === true
end

@testset "urban soil init: a volume, a column without data (#521)" begin
    c0 = si_case()
    M = c0.mesh
    xy = (collect(Float64, xnodes(M, CellCenter())), collect(Float64, ynodes(M, CellCenter())))
    T3 = [290.0 + 0.1i - 0.05j - 2.0k for i in 1:M.Nx, j in 1:M.Ny, k in 1:4]
    I0 = first(I for (I, _) in soil_cols(c0))
    T3[I0[1], I0[2], :] .= NaN                                       # not land in the source
    case = si_case("dynamic_driver=\"$(soil_driver(T3, nothing; xy))\"")
    g = USI.find_coupling(case, :ground_soil)
    tmean = [sum(filter(!isnan, vec(T3[:, :, k]))) / count(!isnan, T3[:, :, k]) for k in 1:4]
    want(I) = I == I0 ? tmean : T3[I[1], I[2], :]
    cols = soil_cols(case)
    @test all(((I, s),) -> isapprox(s.water.T_soil, SCI.soil_profile_on_nodes(SI_Z, want(I), s.water.grid.z);
                                    rtol = 1e-12), cols)
    @test all(((I, s),) -> s.bottom.T == want(I)[end], cols)
    @test all(((I, s),) -> all(==(g.θ0[case.geom.sfc[I]]), s.water.θ_liq), cols)      # no water: θ0 (copied)
end

@testset "urban soil init: the annual wave (#521)" begin
    case = si_case("ground.soil_init=\"annual_wave\"", "ground.wave_t_mean=283.0", "ground.wave_amplitude=9.0",
                   "ground.wave_day_max=200.0")
    g = USI.find_coupling(case, :ground_soil)
    @test g.init[].kind === :annual_wave
    t = (dayofyear(Date(2019, 6, 29)) - 1) * 86400.0 + 12 * 3600.0
    P = SCI.soil_porosity(g.hydraulics)
    function wave(I)
        θ0 = g.θ0[case.geom.sfc[I]]
        d = SCI.soil_damping_depth(SCI.soil_thermal_conductivity(g.thermal, P, θ0, 0.0),
                                   SCI.soil_heat_capacity(g.thermal, P, θ0, 0.0))
        return z -> SCI.annual_soil_temperature(z, t; T_mean = 283.0, amplitude = 9.0, damping_depth = d,
                                                t_max = 199 * 86400.0)
    end
    cols = soil_cols(case)
    @test all(((I, s),) -> isapprox(s.water.T_soil, wave(I).(s.water.grid.z); rtol = 1e-12), cols)
    @test all(((I, s),) -> s.bottom.T == wave(I)(s.water.grid.zh[end]), cols)
    # end of June, 19 days before the surface maximum: warm at the top, the
    # deep soil still near the mean (lagging by z/d)
    s = soil_cols(case)[1][2]
    @test s.water.T_soil[1] > 291 && s.water.T_soil[1] > s.water.T_soil[end] > 283.0
end

@testset "urban soil init: uniform without soil data; driver without it an error (#521)" begin
    case = si_case()                                                 # no dynamic driver: auto → uniform
    g = USI.find_coupling(case, :ground_soil)
    @test g.init[].kind === :uniform
    @test all(((I, s),) -> all(==(case.cfg.T_soil0), s.water.T_soil) && s.bottom.T == case.cfg.T_soil0, soil_cols(case))
    # the sounding driver alone has no soil: auto warns, driver is an error
    c2 = @test_logs (:warn, r"starts uniform") match_mode = :any si_case("dynamic_driver=\"$SI_SOUNDING\"")
    @test USI.find_coupling(c2, :ground_soil).init[].kind === :uniform
    @test_throws ArgumentError si_case("dynamic_driver=\"$SI_SOUNDING\"", "ground.soil_init=\"driver\"")
    @test_throws Exception si_case("ground.soil_init=\"spun_up\"")
end
