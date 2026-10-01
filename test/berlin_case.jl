# The Berlin case infrastructure (issue #126): coupling registry, stage
# schedule, cadences in seconds, tracer registry, and the conservation
# properties of the stages that move energy and mass between the surface
# bus and the air.
#
# The refactor itself is validated bit-for-bit against the pre-refactor
# script (1 h run, every NetCDF variable ==; see the PR). These tests pin
# the pieces that make that possible and that later couplings rely on.

using Test
using Aeolus
using Aeolus.Fields: interior_view
using Printf

isdefined(Main, :BerlinCase) ||
    include(joinpath(@__DIR__, "..", "examples", "urban", "BerlinCase.jl"))
const BC = Main.BerlinCase

@testset "Berlin case: registry, stages, bus, tracers (#126)" begin

    @testset "Cadence in seconds reproduces the step-count schedule" begin
        # dt = 2 s with sim.t accumulated by +dt, as step! does: the old
        # `n % 300 == 0 || n == 1` (radiation), `n % 15 == 0` (soil) and
        # `n % 900 == 0` (output) over a full day.
        rad  = BC.resolve(BC.Cadence(:radiation; first = true), BC.CaseConfig())
        soil = BC.resolve(BC.Cadence(:soil), BC.CaseConfig())
        out  = BC.resolve(BC.Cadence(:output), BC.CaseConfig())
        t = 0.0; mism = 0
        for n in 1:43_200
            tn = t + 2.0
            mism += (BC.fires(rad, n, t, tn)  != (n % 300 == 0 || n == 1))
            mism += (BC.fires(soil, n, t, tn) != (n % 15 == 0))
            mism += (BC.fires(out, n, t, tn)  != (n % 900 == 0))
            t = tn
        end
        @test mism == 0
        @test BC.fires(BC.EVERY_STEP, 17, 32.0, 34.0)
        # With a varying dt the entry fires once per period crossed: over
        # 3 h of random steps in (0.5, 7) s a 600 s cadence fires 18 times.
        rng_dt = [0.5 + 6.5 * abs(sin(1.7 * n)) for n in 1:4000]
        t = 0.0; nf = 0; n = 0
        while t < 10_800 - 1e-9
            n += 1
            dt = min(rng_dt[n], 10_800 - t)
            nf += BC.fires(BC.Cadence(600.0, false), n, t, t + dt)
            t += dt
        end
        @test nf == 18
    end

    @testset "registry: one default provider per exclusive slot, defaults select exactly one" begin
        slots = BC.slot_providers()
        @test !isempty(slots)
        for (s, provs) in slots
            @test allunique(provs)
            @test count(C -> BC.slot(C) === s && BC.is_default(C), BC.REGISTRY) == 1
        end
        sel = BC.selected_couplings(BC.CaseConfig())
        for s in keys(slots)
            @test count(C -> BC.slot(C) === s, sel) == 1
        end
        # the initial couplings of the refactor, all on by default (#129
        # replaced the slab and the soil adapter by the surface exchange
        # and the soil ground)
        @test Set(BC.coupling_name.(sel)) ==
              Set([:ibm, :geostrophic, :radiosity_sun, :buildings_citymean, :surface_exchange,
                   :ground_soil, :sw_modifiers, :glass, :aq_chem, :aq_aerosol, :transport_weno,
                   :micro_kessler, :pollutant, :fog_dew, :veg_drag, :snow_legacy,
                   :cooling_prescribed, :traffic_heat, :moist_buoyancy])
        # every coupling names only real Aeolus submodules
        for C in BC.REGISTRY, m in BC.modules_used(C)
            @test isdefined(Aeolus, m) && getfield(Aeolus, m) isa Module
        end
        # configuration errors are caught, not ignored
        @test_throws ErrorException BC.selected_couplings(BC.CaseConfig(; physics = Dict{Symbol,Any}(:microphysics => :nonexistent)))
        @test_throws ErrorException BC.selected_couplings(BC.CaseConfig(; physics = Dict{Symbol,Any}(:no_such_key => true)))
        @test_throws ErrorException BC.selected_couplings(BC.CaseConfig(; physics = Dict{Symbol,Any}(:micro_kessler => false)))
        # an additive coupling can be switched off
        @test !(BC.PollutantCoupling in BC.selected_couplings(BC.CaseConfig(; physics = Dict{Symbol,Any}(:pollutant => false))))
    end

    # One small case shared by the remaining tests: the :smoke window.
    case = BC.build_case(BC.CaseConfig(:smoke; verbose = false))
    mesh, sim, bus = case.mesh, case.sim, case.bus
    Nx, Ny, Nz = mesh.Nx, mesh.Ny, mesh.Nz

    @testset "stage schedule" begin
        order = [(S, BC.coupling_name(e.coupling)) for S in BC.ALL_STAGES for e in case.schedule[S]]
        # the order of the pre-refactor alexanderplatz_diurnal.jl loop, with
        # the slab replaced by the surface exchange (#129): its fluxes go
        # into the air in :surface_flux (the slab's latent flux at the end
        # of :anthropogenic and its Ts clamp in :sanity are gone), its
        # surface drag follows the IBM, and the surfaces (soil included)
        # are stepped in :surface_update
        @test order == [
            (:radiation, :radiosity_sun), (:radiation, :buildings_citymean),
            (:sw_modifiers, :sw_modifiers),
            (:surface_flux, :surface_exchange),
            (:anthropogenic, :buildings_citymean), (:anthropogenic, :traffic_heat),
            (:anthropogenic, :glass),
            (:chemistry, :aq_chem), (:chemistry, :aq_aerosol),
            (:transport, :transport_weno),
            (:microphysics, :micro_kessler),
            (:tracers, :pollutant),
            (:fog_dew, :fog_dew),
            (:body_force, :geostrophic), (:body_force, :moist_buoyancy),
            (:post_predictor, :ibm), (:post_predictor, :surface_exchange),
            (:post_predictor, :veg_drag),
            (:surface_update, :surface_exchange), (:surface_update, :snow_legacy),
            (:surface_update, :aq_chem),              # r_a / v_d refresh (#135)
            (:column_tendency, :cooling_prescribed)]
        # the chemistry species are advected tracers since #135
        @test [t.name for t in case.tracers] == [:qv, :ql, :qr, :NO, :NO2, :O3, :VOC, :HNO3, :pollutant]
        @test [t.name for t in case.tracers if t.advect] == [:qv, :ql, :NO, :NO2, :O3, :VOC, :HNO3]
    end

    @testset "tracer registry: ensure_tracer! is idempotent and ORs the flags" begin
        f1 = BC.ensure_tracer!(case, :qr; advect = false)
        @test f1 === BC.tracer(case, :qr)
        n = length(case.tracers)
        BC.ensure_tracer!(case, :qr; positive = true)
        @test length(case.tracers) == n
        @test case.tracers.list[case.tracers.index[:qr]].positive
        @test_throws ErrorException BC.tracer(case, :no_such_tracer)
    end

    # The surface_flux stage (surface exchange, #129) is tested in
    # berlin_surface_exchange.jl.

    @testset "anthropogenic stage: traffic heat is Q_urban over the domain" begin
        θ0 = copy(interior_view(sim.θ))
        BC.run_stage!(BC.find_coupling(case, :traffic_heat), case, Val(:anthropogenic), 2.0)
        ρcp = BC.ref_density(case, 1) * Aeolus.MoistThermodynamics.DEFAULT_CONSTANTS.cp_dry
        Q = sum(interior_view(sim.θ) .- θ0) * ρcp * mesh.dz / 2.0 / (Nx * Ny)
        @info @sprintf("  traffic heat: %.6f W/m² domain mean (expected 8)", Q)
        @test Q ≈ 8.0 rtol = 1e-9
        interior_view(sim.θ) .= θ0
    end

    @testset "transport stage: flux-form conservation, and the stale-halo leak (KNOWN)" begin
        # give the flow some structure first
        for _ in 1:4
            BC.step_case!(case, 2.0)
        end
        ql = interior_view(BC.tracer(case, :ql))
        ql .= 1e-4 .* (1 .+ sin.(reshape(1:Nx*Ny*Nz, Nx, Ny, Nz)))
        saved = Dict(t.name => copy(interior_view(t.field)) for t in case.tracers)
        tr = BC.find_coupling(case, :transport_weno)
        mass() = Dict(t.name => sum(interior_view(t.field)) for t in case.tracers)
        restore!() = foreach(t -> interior_view(t.field) .= saved[t.name], case.tracers)

        # (a) As shipped. step! corrects the interior velocity after the
        # projection but leaves the halos at the predictor values, so the
        # transport that runs at the start of the next step sees a
        # velocity that is not divergence-free across the periodic seam
        # and the flux form stops conserving. Kept bit-for-bit here; the
        # scalar-advection issue refreshes the halos.
        m0 = mass()
        BC.run_stage!(tr, case, Val(:transport), 2.0)
        m1 = mass()
        leak = Dict(n => (m1[n] - m0[n]) / m0[n] for n in (:qv, :ql))
        @info @sprintf("  transport as shipped (stale velocity halos): relative mass change per step qv %.2e, ql %.2e",
                       leak[:qv], leak[:ql])
        @test_broken abs(leak[:qv]) < 1e-12

        # (b) With the velocity halos refreshed the same stage conserves
        # every advected tracer to round-off, keeps it non-negative
        # (to round-off: the limiter guarantees ≥ 0 in exact arithmetic),
        # and leaves the non-advected tracers untouched.
        restore!()
        for f in (case.sim.u, case.sim.v, case.sim.w)
            fill_halos!(f, case.sim.bcs)
        end
        m0 = mass()
        BC.run_stage!(tr, case, Val(:transport), 2.0)
        m1 = mass()
        for t in case.tracers
            if t.advect
                @info @sprintf("  transport %s with fresh halos: relative mass change %.1e, min %.1e",
                               t.name, abs(m1[t.name] - m0[t.name]) / m0[t.name],
                               minimum(interior_view(t.field)))
                @test m1[t.name] ≈ m0[t.name] rtol = 1e-12
                @test minimum(interior_view(t.field)) >= -1e-18
            else
                @test m1[t.name] == m0[t.name]
            end
        end
    end

    @testset "a short run stays finite and writes the registered output" begin
        path = joinpath(mktempdir(), "smoke.nc")
        c2 = BC.build_case(BC.CaseConfig(:smoke; verbose = false, output_path = path))
        BC.run_case!(c2)
        @test c2.clock.n == 12 && !c2.clock.blowup
        f = nc_open(path)
        @test nc_nrecords(f) == 2
        vars = nc_varnames(f)
        for v in ("u", "v", "w", "theta", "p", "qv", "ql", "qr", "pollutant",
                  "Ts", "sw_in", "lw_in", "T_soil1", "G_soil", "theta_soil1",
                  "H_sfc", "LE_sfc", "u_star", "beta_sfc")
            @test v in vars
        end
        @test nc_read(f, "time") == [12.0, 24.0]
        @test all(isfinite, nc_read(f, "theta"))
    end
end
