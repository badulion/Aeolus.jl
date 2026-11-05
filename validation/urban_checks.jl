# The end-to-end checks of the urban run type (EX-a, #438) on the Berlin
# configurations: formerly the by-hand examples/urban/berlin_checks.jl
# (#320), before that test/berlin_smoke.jl, test/berlin_parent_smoke.jl
# and checks 1-3 of test/berlin_parent.jl. Full tier (test_tags.toml):
#
#   flagship  the Berlin summer case (64³, 1 km × 1 km × 400 m, nested) for
#           20 adaptive steps, past step 16 where step!(model, cfg.dt)
#           failed (#392): no retry, CFL ≤ C_advective, |u_i| < 20 m/s, θ
#           within 2 K of its initial range; the forced 2 s step is rejected;
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
#           every forcing a named term; encroachment of a CBL under
#           Q(t) = 0.15 sin(πt/D) into Γ = 3 K/km (Tennekes 1973; Stull 1988
#           Ch. 11) with the heat budget closed at every output; Wangara
#           Day 33 (Clarke et al. 1971) against the MYNN z_i of Nakanishi &
#           Niino (2009); the tiled parent surface over the child's
#           footprint against the child's surfaces, 24 h offline (#249:
#           class fractions, the no-unresolved-buildings control equal to
#           the child to round-off, the Macdonald mosaic within its
#           margins, energy closure per tile).
#
# The idealised parents are one-domain configs (TOML text below) of the
# model's core with a prescribed surface flux. The 24 h Lindenberg checks
# of the parent are an analysis of a run of
# examples/urban/berlin_parent_coarse.toml
# (examples/urban/berlin_parent_lindenberg_check.jl). Numbers measured when
# these were tests are quoted next to the assertions.

using Test
using Aeolus
using Aeolus.Fields: interior_view, set!
using Printf
using Logging
using Random: Xoshiro
using Aeolus.SurfaceTiles: last_tile_fluxes, parameter_aggregate, mosaic_fractions

const BC = Aeolus.Urban
include(joinpath(@__DIR__, "..", "test", "support", "berlin_configs.jl"))
# smoke_matrix(registry) and seed_fog!(case, rng), shared with the
# thread-invariance harness (KA-a, #294)
include(joinpath(@__DIR__, "..", "test", "support", "smoke_matrix.jl"))

"""
    berlin_default(; kw...) -> NamedTuple

The Berlin summer site's options (berlin_summer.toml's site block, the
fields `kw` set through their keys) without its nest (the `:nesting`
component dropped from the selection): the site alone, for the offline
surface checks, which need its geometry and its class factories only.
"""
function berlin_default(; kw...)
    c = site_config(berlin_toml("summer"); kw...)
    delete!(c.components, :nesting)
    return c
end

"The Berlin land-use table of the nest parent's tiles (examples/urban/data)."
const BERLIN_LANDUSE = joinpath(berlin_data_dir(), "berlin_landuse_6km.csv")

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

"""
    idealised_parent(; surface, theta, qv = nothing, start_h = 12.0) -> String

The idealised nest parent of #259 as a one-domain config (TOML text): 4 km
× 4 km × 3 km on 32 × 32 × 53 cells (Δx = 125 m, Δz = 25 m up to 500 m,
then × 1.08 per level up to 100 m), at rest, f = 1e-4 s⁻¹, the TKE closure,
the θ damping and the velocity sponge over the top 25 %, a prescribed
surface flux (`surface`: the `[surface]` lines), the initial `theta` (and
`qv`: moist) profile tables; no radiation, no geostrophic wind, no
nudging. The budgets of its heat (and water) are measured.
"""
function idealised_parent(; surface, theta, qv = nothing)
    tab(z, v) = "{kind = \"table\", z = [$(join(z, ", "))], values = [$(join(v, ", "))]}"
    return """
        schema_version = 2
        [run]
        name = "idealised_parent"
        seed = 259
        [time]
        dt = "adaptive"
        cfl = 0.9
        dt_max = 10.0
        duration = 86400.0
        [grid]
        size = [32, 32, 53]
        extent = [4000.0, 4000.0, 3000.0]
        halo = [2, 2, 2]
        stretching = "surface_refined"
        dz_surface = 25.0
        z_uniform = 500.0
        stretching_parameter = 1.08
        dz_max = 100.0
        [timestep]
        scheme = "rk3"
        [dynamics]
        nu = 0.0
        theta0 = 293.0
        moisture = $(qv !== nothing)
        [boundaries]
        z = "free_slip"
        theta_top = "initial_gradient"
        [initial]
        theta = $(tab(theta...))
        $(qv === nothing ? "" : "qv = " * tab(qv...))
        u = {kind = "constant", value = 0.0}
        perturbation_amplitude = 0.4
        perturbation_depth = 100.0
        [coriolis]
        f = 1e-4
        [sponge]
        alpha_max = 0.01
        z_start = 2250.0
        [sgs]
        model = "tke"
        e_min = 1e-4
        wall_cap = true
        wall_cap_top = false
        # a prescribed flux reports no u* to the TKE closure (#491): no surface value
        surface_value = false
        [surface]
        $surface
        [large_scale_forcing]
        damping_rate = 0.01
        [checks]
        energy_closure = {required = false}
        $(qv === nothing ? "" : "water_closure = {required = false}")
        [output]
        variables = []
        """
end

"A kinematic surface flux `f(t)` as a `{kind = \"series\"}` table, sampled every `dt` seconds to `T`."
series(f, T; dt = 300.0) = (ts = collect(0.0:dt:T);
    "{kind = \"series\", t = [$(join(ts, ", "))], value = [$(join(f.(ts), ", "))]}")

"Build a config's model from its TOML text (the estimate FFTW planner)."
text_model(text) = build_config_model(load_run_config(IOBuffer(text); overrides = ["fftw.planner=\"estimate\""]))

"""
    run_hours!(f, m, hours; every = 900.0)

Run the one-domain model `m` adaptively for `hours`, landing on every
multiple of `every` seconds and calling `f(d)` there.
"""
function run_hours!(f, m, hours; every = 900.0)
    d = only(m.domains)
    dt = 2.0
    for n in 1:round(Int, 3600hours / every)
        while d.state.t < n * every * (1 - 1e-12)
            _, dt, _ = Aeolus.Components.step_adaptive!(d; dt_prev = dt, t_end = n * every)
        end
        f(d)
    end
    return m
end

"The energy (or water) budget of a domain: its stage budget (optionally of a layer)."
domain_budget(d, q; kw...) = Aeolus.BudgetSystem.stage_budget(d.budgets.budgets.measured[q]; kw...)

"The 24 h offline run of the parent columns over the footprint and of the child's surfaces (one per class)."
function parent_footprint_run(ps, case, cfg, PFP; dt = 300.0, hours = 24.0, probe = nothing)
    sfc = case.geom.sfc
    ncls = Int(maximum(sfc))
    fch = [count(==(c), sfc) / length(sfc) for c in 1:ncls]
    zr = case.mesh.dz / 2                         # the child's own forcing height (bus.z_ref)
    Ts(c) = cfg.Ts0 + (c == BC.SURFACE_WATER ? cfg.Ts_water_offset : 0.0)
    child = Dict(c => BC.class_surface(case, BC.SurfaceSite(c; T_skin = Ts(c))) for c in 1:ncls if fch[c] > 0)
    cols = [ps.columns[I] for I in CartesianIndices(PFP)]
    tiles = [t for col in cols for t in BC._mosaic(col).tiles]
    E0 = [BC.surface_store_energy(t) for t in tiles]
    sumG = zeros(length(tiles))
    skin = 0.0                                    # max |Rn - H - LE - G| of any tile, any step
    nst = round(Int, 3600hours / dt)
    out = (t = Float64[], Hp = Float64[], LEp = Float64[], Hc = Float64[], LEc = Float64[])
    snap = nothing
    for n in 1:nst
        hour = (n - 1) * dt / 3600
        f = BC.parent_offline_forcing(cfg, hour)
        Hs = LEs = 0.0
        k = 0
        for col in cols
            fl = BC.step_parent_column!(ps, col, f, dt; U_min = 0.1)
            Hs += fl.H; LEs += fl.LE
            for tf in last_tile_fluxes(BC._mosaic(col))
                k += 1
                sumG[k] += tf.G * dt
                skin = max(skin, abs(tf.Rn - tf.H - tf.LE - tf.G))
            end
            if probe !== nothing && col === ps.columns[probe] && hour == 13.0
                snap = (f = f, fl = fl, tiles = last_tile_fluxes(BC._mosaic(col)),
                        frac = mosaic_fractions(col.surface), pa = parameter_aggregate(BC._mosaic(col)))
            end
        end
        Hc = LEc = 0.0
        fc = BC.SurfaceForcing(; sw_down = f.sw_down, lw_down = f.lw_down, T_air = f.T_air,
                                q_air = f.q_air, U = f.U, p = f.p, z_ref = zr)
        for (c, s) in child
            fl = step_surface!(s, fc, dt; U_min = 0.1)
            Hc += fch[c] * fl.H; LEc += fch[c] * fl.LE
        end
        push!(out.t, hour); push!(out.Hp, Hs / length(cols)); push!(out.LEp, LEs / length(cols))
        push!(out.Hc, Hc); push!(out.LEc, LEc)
    end
    E1 = [BC.surface_store_energy(t) for t in tiles]
    store = maximum(abs.((E1 .- E0) .- sumG)) / (3600hours)   # W m⁻², mean over the run
    return (; out..., skin, store, ntiles = length(tiles), snap)
end

"parent surface: the tiled mosaic over the child's footprint against the child (#249)."
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

    @testset "Parent mosaic vs the child under its footprint" begin
        zr = PCASE.mesh.dz / 2
        # Control: no unresolved-building treatment, forced at the child's own
        # height. Same fractions, same factory, same forcing: the parent's
        # footprint mean is the child's area mean to round-off.
        ctl = BC.build_parent_surface(PCASE, PL; columns = CartesianIndices(PFP),
                                       unresolved = false, z_centres = [zr])
        rc = parent_footprint_run(ctl, PCASE, PCFG, PFP)
        @test maximum(abs.(rc.Hp .- rc.Hc)) < 1e-9
        @test maximum(abs.(rc.LEp .- rc.LEc)) < 1e-9
        @info @sprintf("parent control vs child, 24 h: max |ΔH| %.1e, max |ΔLE| %.1e W m⁻²",
                       maximum(abs.(rc.Hp .- rc.Hc)), maximum(abs.(rc.LEp .- rc.LEc)))

        # The parent proper: Macdonald z0 and d for the built tiles, forced
        # at the first level above the roofs, z - d from each tile's d.
        ps = BC.build_parent_surface(PCASE, PL; columns = CartesianIndices(PFP))
        # a mixed column: park next to street and roofs
        probe = argmax(I -> (ps.fractions[I, BC.SURFACE_PARK] > 0) * min(ps.fractions[I, BC.SURFACE_PARK],
                                                                           ps.fractions[I, BC.SURFACE_STREET] +
                                                                           ps.fractions[I, BC.SURFACE_WALL]),
                       CartesianIndices(PFP))
        r = parent_footprint_run(ps, PCASE, PCFG, PFP; probe = probe)
        n13 = findfirst(==(13.0), r.t)
        day = 6 .<= r.t .< 18
        dH13, dLE13 = r.Hp[n13] / r.Hc[n13] - 1, r.LEp[n13] / r.LEc[n13] - 1
        dHd = sum(r.Hp[day]) / sum(r.Hc[day]) - 1
        dLEd = sum(r.LEp[day]) / sum(r.LEc[day]) - 1
        @info @sprintf("footprint mean vs child at 13:00: H %.1f vs %.1f (%+.2f %%), LE %.1f vs %.1f (%+.1f %%); 06-18 h: H %+.2f %%, LE %+.2f %%",
                       r.Hp[n13], r.Hc[n13], 100dH13, r.LEp[n13], r.LEc[n13], 100dLE13, 100dHd, 100dLEd)

        @testset "4. footprint consistency" begin
            # The issue asked for 2 %. That holds for the control above (to
            # 4e-12 W m⁻²), not for the parent proper: its built tiles
            # exchange at z_k - d (k the first level above the roofs) with
            # Macdonald's z0, the child's surfaces at Δz/2 with their own,
            # under the same air state; MOST transfer then differs with height
            # and stability. These bounds are margins on a model difference,
            # not a physical law, and they depend on the forcing's timing:
            # #249 measured H +2.8 % at 13:00, +3.8 % over 06-18 h and LE
            # +4.7 % over 06-18 h with the case sun running 2 h early (local
            # hour used as UTC). With the sun on UTC (#259) solar noon moves
            # from 11:05 to 13:05 CEST, against the synthetic air temperature
            # that peaks at 15:00, and the same comparison measures H +5.97 %
            # at 13:00, +3.08 % over 06-18 h, LE +11.55 % over 06-18 h (the
            # date, 2019-06-29 vs 2024-06-20, changes these by < 0.6 points;
            # the ground pressure of the sounding by < 0.02). The old 5 % and
            # 10 % margins were set on the wrong sun; re-measured, not a code
            # error. The control above still agrees to 4e-12 W m⁻².
            @test abs(dH13) < 0.08
            @test abs(dHd) < 0.05
            # LE over 06-18 h. At one instant around noon the child's street
            # and park soils hit their dry limit at a different time from the
            # parent's (−21 % or +893 % at 13:00, depending on the sun), so LE
            # at one instant is not asserted.
            @test abs(dLEd) < 0.15
        end

        @testset "5. energy closure per tile, 24 h" begin
            @test r.ntiles > length(CartesianIndices(PFP))     # mixed columns carry several tiles
            @test r.skin < 1e-10          # Rn - H - LE - G at every step, every tile
            @test r.store < 1e-10         # Σ G dt = change of the tile's stored energy
            @info @sprintf("tile closure, %d tiles, 24 h: max |Rn-H-LE-G| %.1e, max |ΔE/Δt - mean G| %.1e W m⁻²",
                           r.ntiles, r.skin, r.store)
        end

        @testset "3. mosaic flux vs parameter aggregation" begin
            s = r.snap
            @test s !== nothing
            H = sum(s.frac .* [t.H for t in s.tiles])
            LE = sum(s.frac .* [t.LE for t in s.tiles])
            @test s.fl.H ≈ H rtol = 1e-13
            @test s.fl.LE ≈ LE rtol = 1e-13
            # the tiles' stabilities differ: an evaporating, cooler park tile
            # under stable stratification next to hot, unstable street/roofs
            @test any(t -> t.H < 0, s.tiles) && any(t -> t.H > 0, s.tiles)
            col = ps.columns[probe]
            dbar = sum(s.frac .* col.d)
            fpa = BC.SurfaceForcing(; sw_down = s.f.sw_down, lw_down = s.f.lw_down, T_air = s.f.T_air,
                                     q_air = s.f.q_air, U = s.f.U, p = s.f.p,
                                     z_ref = ps.z_centres[col.k] - dbar)
            pa = turbulent_fluxes(s.pa, fpa; U_min = 0.1)
            @info @sprintf("mixed column %s at 13:00 (classes %s, fractions %s): mosaic H %.1f LE %.1f, parameter-aggregated H %.1f LE %.1f W m⁻²",
                           Tuple(probe), string(Int.(col.classes)), string(round.(s.frac; digits = 3)), s.fl.H, s.fl.LE, pa.H, pa.LE)
            @test abs(pa.H - s.fl.H) > 10 || abs(pa.LE - s.fl.LE) > 10
        end
    end
  end
end

"parent: the parent smoke run and the idealised validations."
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

    @testset "parent: encroachment and the heat budget (#259)" begin
        D, Γ, A = 12 * 3600.0, 3e-3, 0.2
        Q(t) = 0.15 * sin(π * t / D)
        m = text_model(idealised_parent(; surface = "model = \"prescribed\"\nq_surface = " * series(Q, 12 * 3600.0),
                                        theta = ([0.0, 3000.0], [290.0, 290 + Γ * 3000])))
        ratios = Float64[]; worst = 0.0
        zi = Dict{Int,Float64}()
        run_hours!(m, 6.0) do d
            t = d.state.t
            cl = Aeolus.Components.sgs_closure(d)
            zf, fl = BC.mean_heat_flux(d.state, Aeolus.Operators.sgs_fields(cl, d.state).κv)
            t >= 2 * 3600 && push!(ratios, minimum(fl) / Q(t))
            hb = domain_budget(d, :energy)
            worst = max(worst, abs(hb.closure) / maximum(abs, values(hb.terms)))
            h = round(Int, t / 3600)
            isapprox(t, 3600h; atol = 1) && (zi[h] = BC.parent_zi(d))
        end
        theory(t) = sqrt(2 * (1 + 2A) * 0.15 * D / π * (1 - cos(π * t / D)) / Γ)
        for h in (3, 6)
            err = zi[h] / theory(3600h) - 1
            @info @sprintf("encroachment: z_i(%d h) = %.0f m (min flux), theory %.0f m, %+.1f %%", h, zi[h], theory(3600h), 100err)
            @test abs(err) < 0.10       # measured +4.7 % (3 h), +1.4 % (6 h) on the ParentCase parent
        end
        R = sum(ratios) / length(ratios)
        @info @sprintf("entrainment flux ratio (2-6 h mean of %d profiles): %.3f", length(ratios), R)
        @test -0.25 <= R <= -0.10       # measured -0.130 on the ParentCase parent
        @test worst < 1e-10
        d = only(m.domains)
        hb = domain_budget(d, :energy)
        below = domain_budget(d, :energy; layer = (0.0, zi[6]))
        damp = sum(v for (k, v) in below.terms if k[2] === :large_scale_forcing; init = 0.0)
        sfc = sum(v for (k, v) in hb.terms if k[2] === :prescribed_surface_flux; init = 0.0)
        @test sfc > 0 && abs(damp) < 0.01 * sfc
    end

    @testset "parent: Wangara Day 33 against MYNN (#259)" begin
        rows = [split(l, ',') for l in readlines(joinpath(pkgdir(Aeolus), "examples", "sbl", "data", "wangara_day33.csv"))
                if !startswith(l, "#") && !startswith(l, "table")]
        prof = [parse.(Float64, r[2:end]) for r in rows if r[1] == "profile"]
        ref = Dict(round(Int, parse(Float64, r[2])) => parse(Float64, r[3]) for r in rows if r[1] == "zi")
        z = [q[1] for q in prof]; P = [q[4] for q in prof]; T = [q[5] + 273.15 for q in prof]
        r = [q[6] / 1000 for q in prof]
        κ = 287.04 / 1004.67
        θ = T .* (1000 ./ P) .^ κ
        qv = r ./ (1 .+ r)
        @test θ[1] ≈ 276.85 atol = 0.01        # 5.5 °C at 1023 hPa
        Γtop = (θ[end] - θ[end-5]) / (z[end] - z[end-5])        # 5.4 K/km, 1500-2000 m
        # Nakanishi & Niino (2009): <wθ>_s = 0.216 cos(π(t_h − 13)/11) K m/s, from 09 LT
        shape(t) = cos(π * (9 + t / 3600 - 13) / 11)
        surface = "model = \"prescribed\"\nq_surface = " * series(t -> 0.216 * shape(t), 8 * 3600.0) *
                  "\ne_surface = " * series(t -> 2.29e-5 * shape(t), 8 * 3600.0)
        m = text_model(idealised_parent(; surface, theta = (vcat(z, 3000.0), vcat(θ, θ[end] + Γtop * (3000 - z[end]))),
                                        qv = (z, qv)))
        zi = Dict{Int,Float64}()
        run_hours!(m, 7.0) do d
            h = 9 + d.state.t / 3600
            isapprox(h, round(h); atol = 1e-6) && (zi[round(Int, h)] = BC.parent_zi(d))
        end
        for h in (12, 14, 16)
            err = zi[h] / ref[h] - 1
            @info @sprintf("Wangara 33: z_i(%d LT) = %.0f m, NN09 (MYNN) %.0f m, %+.1f %%", h, zi[h], ref[h], 100err)
            @test abs(err) < 0.15       # measured +7.8 %, +6.1 %, +7.6 % on the ParentCase parent
        end
    end
end

"""
flagship: the Berlin summer case (64³, 1 km × 1 km × 400 m,
every default coupling, nested in its parent domain), for at least
`FLAGSHIP_STEPS` steps (whole parent windows) of the
adaptive stepper (#392). `step!(model, cfg.dt)` used to force 2 s steps
against a stable step that fell to 0.09 s: the guard passed them, |v|
grew to 49 m/s and the run failed at step 16; that dt is now rejected,
and the stepper's own steps run past step 16 without a retry.
"""
const FLAGSHIP_STEPS = 20

function berlin_flagship_checks()
    @testset "Berlin flagship: default case past step 16 (#392)" begin
        # the site nested in the 6 km parent domain (NEST-p3 #483)
        m = build_config_model(load_run_config(berlin_toml("summer")))
        case = urban_site(m)
        cfg = case.cfg
        sim = case.sim
        θ0 = extrema(interior_view(sim.θ))
        ts = BC.timestepper(case)
        # the forced fixed step of the failing run is refused, nothing changed
        @test_throws ArgumentError step!(case.model, cfg.dt)
        @test case.clock.n == 0 && sim.t == 0
        t_start = time()
        vmax = 0.0
        # the nest's windows until the site has taken FLAGSHIP_STEPS steps
        while case.clock.n < FLAGSHIP_STEPS
            Aeolus.Components.step_adaptive!(m)
            @test !case.clock.blowup
            vmax = max(vmax, maximum(abs, interior_view(sim.u)), maximum(abs, interior_view(sim.v)),
                       maximum(abs, interior_view(sim.w)))
        end
        s, k = ts.stats, ts.constraints
        θ1 = extrema(interior_view(sim.θ))
        @info @sprintf("flagship: %d steps in %.0f s, t = %.2f s, dt %.3f–%.3f s, %d retries, max CFL %.2f, max |u_i| %.1f m/s, θ %.2f–%.2f K (initial %.2f–%.2f)",
                       s.n, time() - t_start, sim.t, s.dt_min, s.dt_max, s.retries, s.cfl_max, vmax,
                       θ1..., θ0...)
        @test s.n >= FLAGSHIP_STEPS && s.retries == 0
        @test s.cfl_max <= k.C_advective + 1e-9
        @test max_divergence(sim) < 1e-10
        # measured 2026-10-07 (32 steps): max |v| 11.2 m/s, θ 288.33–292.42 K
        # from 288.69–292.41; the forced-2-s run had |v| 49 m/s, θ 229–360 K
        @test vmax < 20
        @test θ0[1] - 2 < θ1[1] && θ1[2] < θ0[2] + 2
        @test !case.clock.blowup
    end
end

@testset "urban run type: end-to-end checks on the Berlin configs" begin
    berlin_flagship_checks()
    berlin_smoke_checks()
    berlin_parent_checks()
    berlin_parent_surface_checks()
end
