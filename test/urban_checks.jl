# The urban site's code-path checks on the Berlin configurations (EX-a,
# #438), formerly part of the by-hand examples/urban/berlin_checks.jl
# (#320), before that test/berlin_smoke.jl, test/berlin_parent_smoke.jl
# and check 1 of test/berlin_parent.jl:
#
#   smoke   every coupling of the run type in a short window of the real OSM
#           scene (examples/urban/berlin_smoke.toml: 16³, 250 m ×
#           250 m × 100 m, 12 steps of 2 s), in every configuration the
#           registry knows about (test/support/smoke_matrix.jl, shared with
#           the thread-invariance harness): a finite state, every coupling's
#           own `smoke_check`s, and the surface-energy, water and air-heat
#           ledgers closed (49 runs, 6494 coupling checks, 204 s measured on
#           2026-10-07);
#   parent  the nest parent (a domain of the model's core, NEST-p3 #483:
#           berlin_parent.toml at its coarse [test] size) with every forcing
#           on for 12 steps, its energy and water budgets closed (#259),
#           every forcing a named term; the parent's class fractions over the
#           child's footprint (#249: exact identities, no run);
#   zi      the parent's boundary-layer height (`parent_zi`) and its heat-flux
#           profile (`mean_heat_flux`) on a synthetic unstepped state with a
#           known flux minimum, and the synthetic offline forcing of the
#           parent surface (`parent_offline_forcing`).
#
# The end-to-end validation runs (the flagship past step 16, encroachment,
# Wangara Day 33, the 24 h parent mosaic against the child) are in
# validation/urban_checks.jl.

using Test
using Aeolus
using Aeolus.Fields: interior_view, set!
using Printf
using Logging
using Random: Xoshiro

const BC = Aeolus.Urban
isdefined(@__MODULE__, :berlin_default) || include(joinpath(@__DIR__, "support", "berlin_checks.jl"))
# smoke_matrix(registry) and seed_fog!(case, rng), shared with the
# thread-invariance harness (KA-a, #294)
include(joinpath(@__DIR__, "support", "smoke_matrix.jl"))

allfinite(x) = all(isfinite, x)

"smoke: every run of the registry's smoke matrix, its checks and ledgers."
function berlin_smoke_checks()
    @testset "Berlin smoke matrix" begin
        t_start = time()
        runs = smoke_matrix()
        @info "Berlin smoke matrix: $(length(runs)) runs — " * join(first.(runs), "; ")
        exercised = Set{Symbol}()
        nchecks = 0
        for (irun, (label, rc)) in enumerate(runs)
            @testset "$label" begin
                t0 = time()
                # the site nested in its parent domain (NEST-p3 #483), or alone
                m = build_config_model(rc)
                case = urban_site(m)
                cfg = case.cfg
                seed_fog!(case, Xoshiro(127))
                length(m.domains) > 1 ? run_model!(m) : run_site!(case)
                union!(exercised, BC.component_name.(case.couplings))
                sim, bus = case.sim, case.bus
                # nsteps·dt of simulated time (in nsteps steps, unless the
                # time stepper chooses its own step, #133)
                @test !case.clock.blowup && isapprox(sim.t, cfg.nsteps * cfg.dt; rtol = 1e-12)
                BC.variable_dt(typeof(BC.timestepper(case))) || @test case.clock.n == cfg.nsteps
                @test all(allfinite ∘ interior_view, (sim.u, sim.v, sim.w, sim.θ))
                @test all(t -> allfinite(interior_view(t.field)), case.tracers.list)
                @test all(allfinite, (bus.Ts, bus.sw, bus.lw, bus.dew))
                # every coupling's own physics checks
                failed = String[]
                for c in case.couplings, (what, ok) in BC.smoke_check(c, case)
                    nchecks += 1
                    ok || push!(failed, "$(BC.component_name(c)): $what")
                end
                @test failed == String[]
                # the surface energy ledger closes per class
                rows = BC.surface_budget(case.budgets)
                @test !isempty(rows)
                for r in rows
                    @test abs(r.closure) < 1e-6          # W/m²; measured ≲ 3e-10
                end
                dom = rows[end]
                # the water ledger closes
                w = BC.water_budget(case.budgets)
                @test abs(w.residual) < 1e-12            # kg/m²; measured ≲ 3e-16
                # the air-heat ledger closes (#253), nothing changed θ unattributed
                hb = BC.heat_budget(case.budgets)
                @test hb.elapsed > 0
                @test abs(hb.closure / hb.elapsed) <= 1e-10     # W/m²
                @test !haskey(hb.terms, (:core, :unattributed))
                @test !haskey(hb.terms, (:output, :between_steps))
                # every diagnostic and profile of the run evaluates (#338): a
                # source that throws gives NaN and a warning
                tl = Test.TestLogger(min_level = Logging.Warn)
                with_logger(tl) do
                    foreach(evaluate_diagnostic, BC.case_diagnostics(case))
                    foreach(evaluate_profile, BC.case_profiles(case))
                end
                @test isempty(tl.logs)
                @info @sprintf("  %-26s %5.1f s  domain Rn %7.2f H %6.2f LE %5.2f G %7.2f dS %7.2f W/m², closure %.1e; water residual %.1e kg/m²; air heat closure %.1e W/m²",
                               label, time() - t0, dom.Rn, dom.H, dom.LE, dom.G, dom.dS,
                               dom.closure, w.residual, hb.closure / hb.elapsed)
                if irun == 1
                    print(BC.budget_report(case.budgets))
                    # midday sun: the surfaces gain radiative energy and warm
                    @test dom.Rn > 0 && dom.dS > 0
                    # the flux-form transport conserves water (#132)
                    @test abs(get(w.sources, :transport, 0.0)) < 1e-12
                    # and the heat of the air: the dynamics' transport term is
                    # round-off (measured -1.1e-8 J m⁻² over the 24 s), and
                    # the open faces' θ relaxation is booked once, as the
                    # dynamics' (:relaxation), not also measured as the
                    # site's :tendencies entry (which had left +119.6 J m⁻²
                    # in the transport term to compensate)
                    @test abs(get(hb.terms, (:dynamics, :transport), 0.0)) < 1e-6
                    @test !haskey(hb.terms, (:tendencies, :urban))
                end
            end
        end
        @testset "every registered coupling ran in some configuration" begin
            @test isempty(setdiff(Set(BC.component_name.(BC.URBAN_COMPONENTS)), exercised))
        end
        @info @sprintf("Berlin smoke matrix: %d runs, %d coupling checks, %.1f s", length(runs), nchecks,
                       time() - t_start)
    end
end

"parent: the parent smoke run (12 steps, every forcing)."
function berlin_parent_checks()
    @testset "Berlin parent smoke: every forcing, 12 steps, budgets close (#259)" begin
        # berlin_parent.toml at its coarse [test] size, at 10 LT (sun up,
        # convective); subsidence is off by default (divergence = 0, #312): on
        # here at D = 3e-6 s⁻¹
        toml = berlin_toml("parent")
        tt = Aeolus.Config.config_test_table(toml)
        cfg = cd(() -> load_run_config(toml; overrides = vcat(tt.overrides, ["fftw.planner=\"estimate\"",
                                                     "large_scale_forcing.divergence=3e-6", "radiation.start_hour=10.0",
                                                     "checks.energy_closure={required = false}",
                                                     "checks.water_closure={required = false}"])), dirname(toml))
        m = build_config_model(cfg)
        d = only(m.domains)
        @test map(Aeolus.Components.component_name, d.components) ==
              (:advection, :tiled_surface, :sgs, :mean_column_radiation, :coriolis, :large_scale_forcing, :sponge)
        ls = only(c for c in d.components if c isa Aeolus.Components.LargeScale)
        # the sounding's geostrophic wind, its nudging above the tracked z_i,
        # the subsidence and the θ damping
        @test ls.geostrophic !== nothing && ls.zi !== nothing && length(ls.extra) == 3
        @test Aeolus.Components.sgs_closure(d) isa TKEClosure
        dt = 2.0
        for _ in 1:12
            _, dt, _ = Aeolus.Components.step_adaptive!(d; dt_prev = dt)
        end
        s = d.state
        @test all(isfinite, interior_view(s.θ)) && all(isfinite, interior_view(s.qv))
        @test minimum(interior_view(Aeolus.Components.sgs_closure(d).e)) >= 0
        hb = domain_budget(d, :energy); wb = domain_budget(d, :water)
        scale = maximum(abs, values(hb.terms)); wscale = maximum(abs, values(wb.terms))
        @info "parent smoke: the energy budget's terms" hb.terms
        @test abs(hb.closure) <= 1e-10 * scale
        @test abs(wb.closure) <= 1e-10 * wscale
        # every forcing is a named term of the heat of the air
        for c in (:tiled_surface, :mean_column_radiation, :large_scale_forcing)
            @test any(k -> k[2] === c, keys(hb.terms))
        end
        sfc = Aeolus.Components.slot_component(d, :surface)
        rad = Aeolus.Components.slot_component(d, :radiation)
        @test rad.sw_down[] > 100 && 250 < rad.lw_down[] < 450
        @test isfinite(sfc.sfc_mean[:delivered_H]) && sfc.sfc_mean[:delivered_E] > 0
        @info @sprintf("parent smoke: 12 steps to t = %.1f s; closure %.1e (energy) %.1e (water)",
                       s.t, hb.closure / scale, wb.closure / wscale)
    end
end

"parent surface: the class fractions of the parent over the child's footprint (#249)."
function berlin_parent_surface_checks()
  @testset "Berlin parent surface (#249)" begin
    PCFG = berlin_default()
    PCASE = BC.build_case(PCFG)
    PL = BC.read_parent_landuse(BERLIN_LANDUSE)
    PFP = BC.child_footprint(PL, PCFG)

    @testset "Parent class fractions" begin
        @test (PL.nx, PL.ny, PL.dx, PL.r) == (192, 192, 31.25, 2)    # 6 km at Δx = 31.25 m, r = 2
        @test PFP == (81:112, 81:112)                                  # the 1 km child, centred
        sfc = PCASE.geom.sfc
        f = BC.parent_fractions(PL; child_sfc = sfc, footprint = PFP)
        @test size(f, 3) == max(5, Int(maximum(sfc)))
        @test maximum(abs.(sum(f; dims = 3) .- 1)) <= 1e-12
        @test all(>=(0), f)
        # under the footprint: exactly the class counts of the r × r child block
        r = PL.r
        @test all(begin
                      blk = sfc[(a - 1) * r + 1:a * r, (b - 1) * r + 1:b * r]
                      all(f[I, J, c] == count(==(c), blk) / r^2 for c in axes(f, 3))
                  end for (b, J) in enumerate(PFP[2]), (a, I) in enumerate(PFP[1]))
        # outside it: the table's counts
        out = [I for I in CartesianIndices((PL.nx, PL.ny)) if !(I[1] in PFP[1] && I[2] in PFP[2])]
        @test all(f[I, c] == PL.counts[I, c] / r^2 for I in out, c in 1:5)
        # the parent's area-mean over the footprint is the child's area-mean
        for c in axes(f, 3)
            @test sum(f[PFP..., c]) / length(f[PFP..., c]) ≈ count(==(c), sfc) / length(sfc) atol = 1e-14
        end
        # the 6 km table itself: every cell a full block, roofs = λp
        @test all(sum(PL.counts; dims = 3) .== r^2)
        # morphometry: the 8 × 8-cell (250 m) window keeps the box-mean λp
        λp, λf, h = BC.parent_morphometry(PL, f)
        @test all(0 .<= λp .<= 1) && all(λf .>= 0) && all(h .>= 0)
        @test abs(sum(λp) / length(λp) - sum(f[:, :, BC.SURFACE_ROOF]) / (PL.nx * PL.ny)) < 0.01
        @test maximum(h) <= maximum(PL.hmax)
    end

    @testset "parent_offline_forcing: the synthetic clear-sky day" begin
        # one evaluation at 13 h (sun up) and at 0 h (Berlin in June: below
        # the horizon); the air temperature T_mean + T_amp sin(2π(t − 9)/24)
        # and Brutsaert's (1975) clear-sky longwave in closed form
        f13 = BC.parent_offline_forcing(PCFG, 13.0)
        f0 = BC.parent_offline_forcing(PCFG, 0.0)
        for f in (f13, f0)
            @test all(isfinite, (f.sw_down, f.lw_down, f.T_air, f.q_air, f.U, f.p))
            @test f.U == 3.0 && f.q_air == 0.008
            @test 9.0e4 < f.p < 1.05e5
            e = f.q_air * f.p / (0.622 + 0.378 * f.q_air) / 100
            @test f.lw_down ≈ 1.24 * (e / f.T_air)^(1 / 7) * 5.670374419e-8 * f.T_air^4 rtol = 1e-6
        end
        @test f13.sw_down > 300 && f0.sw_down == 0
        @test f13.T_air ≈ 295 + 6 * sin(2π * 4 / 24) rtol = 1e-14
        @test f0.T_air ≈ 295 + 6 * sin(-2π * 9 / 24) rtol = 1e-14
        @test 250 < f0.lw_down < f13.lw_down < 450
    end
  end
end

"""
A one-domain model of the core (8 × 8 × 16, Δz = 100 m, the TKE closure),
built and not stepped: the state the `parent_zi` checks write by hand.
"""
function zi_probe_domain()
    text = """
        schema_version = 2
        [run]
        name = "zi_probe"
        [time]
        dt = 1.0
        duration = 10.0
        [grid]
        size = [8, 8, 16]
        extent = [800.0, 800.0, 1600.0]
        halo = [2, 2, 2]
        [dynamics]
        theta0 = 300.0
        [boundaries]
        z = "free_slip"
        [initial]
        theta = {kind = "constant", value = 300.0}
        u = {kind = "constant", value = 0.0}
        [sgs]
        model = "tke"
        [output]
        variables = []
        """
    m = build_config_model(load_run_config(IOBuffer(text); overrides = ["fftw.planner=\"estimate\""]))
    return only(m.domains)
end

"zi: the parent's boundary-layer height on a synthetic state with a known heat-flux minimum."
function parent_zi_checks()
    @testset "parent_zi and mean_heat_flux on a synthetic state" begin
        d = zi_probe_domain()
        s = d.state; msh = s.mesh
        Hx, Hy, H, Nx, Ny, Nz, dz = msh.Hx, msh.Hy, msh.Hz, msh.Nx, msh.Ny, msh.Nz, msh.dz
        I, J = Hx+1:Hx+Nx, Hy+1:Hy+Ny
        # θ' = a(x), w = h(z) a(x) with a = cos(2πx/L): plane mean 0, mean(a²) = 1/2,
        # so the resolved flux at face k is h(k)/2 exactly; the minimum, −0.15
        # K m/s, is at face 6 (500 m). θ: mixed (300 K) up to 700 m, 5 K/km above.
        a = [cos(2π * (i - 0.5) / Nx) for i in 1:Nx, j in 1:Ny]
        hk(k) = k == 6 ? -0.3 : 0.1 * sin(π * k / Nz)
        θbar(k) = 300.0 + max(0.0, 5e-3 * ((k - 0.5) * dz - 650.0))
        fill!(s.w.data, 0.0)
        for k in 1:Nz
            s.θ.data[I, J, H+k] .= θbar(k) .+ a
        end
        for k in 2:Nz
            s.w.data[I, J, H+k] .= hk(k) .* a
        end
        κ = Aeolus.Operators.sgs_fields(Aeolus.Components.sgs_closure(d), s).κv
        fill!(κ.data, 0.0)                     # no SGS part: the resolved flux alone
        zf, fl = BC.mean_heat_flux(s, κ)
        @test zf ≈ [(k - 1) * dz for k in 2:Nz] rtol = 1e-14
        @test maximum(abs.(fl .- [hk(k) / 2 for k in 2:Nz])) < 1e-12     # measured 3e-15
        @test BC.parent_zi(d) == 500.0                                  # the face of the minimum
        @test BC.parent_zi(d; method = :min_flux) == BC.parent_zi(d)
        # :parcel is the parcel height of the plane-mean θ: the top of the mixed layer
        @test BC.parent_zi(d; method = :parcel) == Aeolus.Components.parcel_zi(s)
        @test BC.parent_zi(d; method = :parcel) ≈ 650.0 atol = 1e-9
        @test_throws ArgumentError BC.parent_zi(d; method = :unknown)
        # the SGS part: w = 0, θ at 3 K/km, κ = 2 m²/s: −κ ∂θ/∂z = −0.006 K m/s at every face
        fill!(s.w.data, 0.0)
        for k in 1:Nz
            s.θ.data[:, :, H+k] .= 300.0 + 3e-3 * (k - 0.5) * dz
        end
        fill!(κ.data, 2.0)
        _, fl2 = BC.mean_heat_flux(s, κ)
        @test maximum(abs.(fl2 .+ 0.006)) < 1e-12                       # measured 1.1e-15
    end
end

@testset "urban run type: code-path checks on the Berlin configs" begin
    berlin_smoke_checks()
    berlin_parent_checks()
    berlin_parent_surface_checks()
    parent_zi_checks()
end
