# The surface spin-up of the urban site (#519; PALM's spinup_time) on the
# synthetic city (test/data/urban/small_city.toml, 16³, local noon):
#
#   * the atmosphere, the clock and the couplings' schedules are back at the
#     start afterwards (copied data: exactly), the surfaces are not;
#   * the soil water is frozen unless [spinup] soil_moisture;
#   * the solar forcing is PALM's θ = θ_m + A cos ζ(t − 1 h);
#   * the spin-up converges: the day-over-day drift of the daily-mean
#     surface temperature falls from the second to the third day;
#   * the run after it closes its budgets (the spin-up books nothing);
#   * the configuration: the step divides the radiation period, which
#     divides the length; driver forcing needs a driver; a spin-up no
#     domain can take is an error.
#
# The steady initial envelope (steady_conduction!, steady_envelope!) is
# tested in urban_surface.jl and building_energy.jl.

using Test
using Aeolus
using Aeolus.Config
using Aeolus.Fields: interior_view

const UCS = Aeolus.Urban
include(joinpath(@__DIR__, "support", "berlin_configs.jl"))

const SPIN_CITY = joinpath(@__DIR__, "data", "urban", "small_city.toml")
# a radiation period and a surface step of a spin-up (the city's 8 s and 4 s
# are for its 12 s run)
const SPIN_OVERRIDES = ["urban.radiation_period=600.0", "urban.soil_period=60.0", "fftw.planner=\"estimate\""]

spin_case() = (Aeolus.Operators.set_fftw_policy!(; planner = :estimate);
               UCS.build_case(UCS.site_options(load_run_config(SPIN_CITY; overrides = SPIN_OVERRIDES))))
spin_options(time; dt = nothing, forcing = :solar, pt_mean = nothing, pt_amplitude = 0.0,
             soil_moisture = false, buildings = true) =
    (; time = Float64(time), dt, forcing, pt_mean, pt_amplitude, soil_moisture, buildings)

soil_water(case) = [copy(s.water.θ_liq) for s in case.bus.surface if s isa Aeolus.SoilColumn.SoilThermalColumn]

@testset "urban spin-up: the atmosphere at the start, the surfaces spun up (#519)" begin
    case = spin_case()
    se = UCS.find_coupling(case, :surface_exchange)
    air0 = Dict(k => copy(v) for (k, v) in pairs(UCS.case_state(case)) if k !== :Ts)
    Ts0, w0, n0, t0, P0 = copy(case.bus.Ts), soil_water(case), case.clock.n, case.sim.t, se.P
    @test Aeolus.Components.spinup!(case, case.domain, spin_options(6 * 3600; pt_amplitude = 5.0))
    # the atmosphere, the clock and the schedule are exactly as before (copied back)
    @test all(k -> UCS.case_state(case)[k] == air0[k], keys(air0))
    @test case.sim.t == t0 && case.clock.n == n0
    @test se.P == P0 && !se.stepped && se.t_next == t0 && case.spin.stop == Inf
    # the ledgers hold nothing of the spin-up
    @test all(a -> all(iszero, a), (se.eRn, se.eH, se.eLE, se.eG, se.edS, se.pending, case.bus.sensible))
    # the surfaces moved and are finite; the soil water did not (frozen: copied, exactly)
    @test all(isfinite, case.bus.Ts)
    @test maximum(abs.(case.bus.Ts .- Ts0)) > 1.0
    @test soil_water(case) == w0
    @test all(s -> !(s isa Aeolus.SoilColumn.SoilThermalColumn) || s.hydrology, case.bus.surface)   # thawed again
    rep = case.spin.report
    @test rep["steps"] == 6 * 60 && rep["dt_s"] == 60.0 && !haskey(rep, "drift_K_per_day")  # < 2 days: no drift
    # the run after the spin-up: its budgets close (the spin-up booked nothing)
    run_site!(case)
    r = UCS.run_results(case)
    @test r["budgets"]["closed"] === true
    @test r["spinup"]["time_s"] == 6 * 3600.0
end

@testset "urban spin-up: soil water with soil_moisture = true (#519)" begin
    case = spin_case()
    w0 = soil_water(case)
    Aeolus.Components.spinup!(case, case.domain, spin_options(6 * 3600; soil_moisture = true))
    # noon to evening: the soil evaporates
    @test sum(sum, soil_water(case)) < sum(sum, w0)
end

@testset "urban spin-up: PALM's solar forcing (#519)" begin
    case = spin_case()
    cfg = case.cfg
    cells = UCS._surface_cells(case)
    f = UCS.spinup_air(case, spin_options(86400; pt_mean = 290.0, pt_amplitude = 4.0), cells, -86400.0, 0.0)
    # θ = θ_m + A cos ζ one hour earlier; the case starts at local noon (start_h = 12)
    for t in (-86400.0, -43200.0, -3600.0, 0.0)
        ζ = UCS.solar_position_at(cfg, UCS.hour_of_day(cfg, t - 3600.0)).zenith
        @test f(t) ≈ 290.0 + 4.0 * cosd(ζ) rtol = 1e-14
    end
    @test f(3600.0 - 86400.0) > 290.0 > f(-43200.0 + 3600.0)      # 13:00 above the mean, 01:00 below
    # pt_mean = none: the initial mean of the air cells above the surfaces; no amplitude: constant
    g = UCS.spinup_air(case, spin_options(3600), cells, -3600.0, 0.0)
    θ = interior_view(case.sim.θ)
    @test g(-1800.0) ≈ sum(θ[i, j, k] for (i, j, k) in cells) / length(cells) rtol = 1e-14
    # the local hour wraps before the start
    @test UCS.hour_of_day(cfg, -13 * 3600.0) ≈ 23.0
end

@testset "urban spin-up: the drift falls from the second to the third day (#519)" begin
    drift(days) = (c = spin_case();
                   Aeolus.Components.spinup!(c, c.domain, spin_options(days * 86400; dt = 300.0, pt_amplitude = 4.0));
                   c.spin.report["drift_max_K_per_day"])
    d2, d3 = drift(2), drift(3)
    @info "spin-up drift (max over classes)" d2 d3
    @test isfinite(d2) && isfinite(d3)
    @test d3 < d2
end

@testset "urban spin-up: configuration (#519)" begin
    bad(ov) = try
        load_run_config(SPIN_CITY; overrides = vcat(SPIN_OVERRIDES, ov)); nothing
    catch e
        e isa ConfigError ? e : rethrow()
    end
    @test bad(["spinup.time=7200.0"]) === nothing
    @test occursin("spinup.dt", sprint(showerror, bad(["spinup.time=7200.0", "spinup.dt=70.0"])))
    @test occursin("spinup.time", sprint(showerror, bad(["spinup.time=1000.0"])))
    @test occursin("spinup.forcing", sprint(showerror, bad(["spinup.time=7200.0", "spinup.forcing=\"driver\""])))
    # a model without an urban site has nothing to spin up: an error, not a silent no-op
    cfg = load_run_config(SPIN_CITY; overrides = vcat(SPIN_OVERRIDES, ["spinup.time=7200.0"]))
    @test_throws ConfigError Aeolus.SimulationDriver._spinup!(cfg, Any[])
    @test Aeolus.SimulationDriver._spinup!(load_run_config(SPIN_CITY; overrides = SPIN_OVERRIDES), Any[]) === false
end
