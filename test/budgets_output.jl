# One budget system and one output writer (ONE-c, #452): src/BudgetSystem.jl,
# src/budgets/*.jl, src/Output.jl, src/components/{budgets,outputs}.jl.
#
#   * the level meter: the heat of the air and a mass against their
#     closed forms, layers, several fields at once;
#   * the stage budget: attribution of known changes to contexts, nested
#     and tendency contexts, `book!`, the dynamics' split, deferral and the
#     discard of a retried step; the closure is round-off;
#   * the diagnosed momentum budget and the surface/water bookkeeping;
#   * the writer: records and time series, `:mean` averaging against the
#     analytic mean, groups that appear during the run, a resumed file;
#   * the model: the budgets attribute the prescribed surface heat flux,
#     Coriolis and the sponge to their components exactly (euler, RK3 and an
#     adaptive step that retries), close to round-off, and leave the state
#     bit for bit unchanged; the catalogue and the config validation.

using Test
using Aeolus
using Aeolus.BudgetSystem
using Aeolus.Output
using Aeolus.Components
const BOC = Aeolus.Components
using Aeolus.Fields: set!, interior_view
using Aeolus.NCIO: NCDataset
using Aeolus.Anelastic: AnelasticSimulation, isentropic_reference
using Aeolus.TimeIntegration: CFLConstraints
using Aeolus.MoistThermodynamics: DEFAULT_CONSTANTS

const BO_MESH = RegularGrid(; size = (6, 5, 8), extent = (600.0, 500.0, 400.0), halo = (2, 2, 2))
bo_bcs() = (bcs = FieldBoundaryConditions(:periodic, :periodic, :wall),
            θ_bcs = FieldBoundaryConditions(:periodic, :periodic, :noflux))
bo_core(; kw...) = BoussinesqSimulation(BO_MESH; bo_bcs()..., θ₀ = 300.0, kw...)
"ρ₀ = p_ref / (R_d θ₀) and c_p of the Boussinesq budgets at θ₀ = 300 K."
const BO_RHO0 = DEFAULT_CONSTANTS.p_ref / (Aeolus.MoistThermodynamics.R_d(DEFAULT_CONSTANTS) * 300.0)
const BO_CP = DEFAULT_CONSTANTS.cp_dry

function bo_domain(sim, specs...; scheme = :euler, cfl = nothing, extra = ())
    d0 = Domain(sim; scheme = scheme, cfl = cfl, validate = false)
    return Domain(d0; components = (CentredAdvection(), BOC.build_components(specs, d0)..., extra...))
end
bo_bits(sim) = [reinterpret(UInt64, vec(copy(f.data))) for f in (sim.u, sim.v, sim.w, sim.θ, sim.p)]
"A divergence-free sheared flow (u(y, z), v(x), w = 0) over a stable θ with a wave."
function bo_state!(sim)
    set!(sim.u, (x, y, z) -> 0.3 * sin(2π * y / 500) * cos(π * z / 400))
    set!(sim.v, (x, y, z) -> 0.2 * sin(2π * x / 600))
    set!(sim.θ, (x, y, z) -> 300.0 + 0.003 * z + 0.1 * sin(2π * x / 600))
    return sim
end

# ---------------------------------------------------------------------------
# the level meter
# ---------------------------------------------------------------------------

@testset "level meter: the heat of the air and a mass in closed form" begin
    m = BO_MESH; Nz = m.Nz
    ρ̄ = [1.2 - 0.01k for k in 1:Nz]; π̄ = [1.0 - 0.002k for k in 1:Nz]
    hm = LevelMeter(m, ρ̄, π̄, 1004.0)
    θ = Field(CellCenter(), m); set!(θ, (x, y, z) -> 290.0 + 0.01z)
    lv = measure_levels!(zeros(Nz), hm, θ)
    # H_k = ρ̄_k c_p Π̄_k θ_k Δz (per unit ground area: Σ_ij θ / (Nx Ny) = θ_k)
    want = [ρ̄[k] * 1004.0 * π̄[k] * (290.0 + 0.01 * (k - 0.5) * m.dz) * m.dz for k in 1:Nz]
    @test lv ≈ want rtol = 1e-14
    @test measure(hm, θ) ≈ sum(want) rtol = 1e-14
    # a layer that cuts a level counts its overlap: 0 to 1.5 levels
    @test measure(hm, θ; layer = (0.0, 1.5m.dz)) ≈ want[1] + 0.5want[2] rtol = 1e-14
    # a mass meter (c = Π̄ = 1) of two fields at once is their sum
    mm = LevelMeter(m, ρ̄, ones(Nz), 1.0)
    q1 = Field(CellCenter(), m); set!(q1, (x, y, z) -> 0.01)
    q2 = Field(CellCenter(), m); set!(q2, (x, y, z) -> 0.002)
    @test measure(mm, (q1, q2)) ≈ sum(ρ̄) * 0.012 * m.dz rtol = 1e-14
    # solid cells are not air: a solid level of a column is left out
    solid = falses(size(θ.data)); solid[m.Hx+1, m.Hy+1, m.Hz+1] = true
    sm = LevelMeter(m, ρ̄, ones(Nz), 1.0, solid)
    @test measure_levels!(zeros(Nz), sm, q1)[1] ≈ ρ̄[1] * 0.01 * m.dz * (m.Nx * m.Ny - 1) / (m.Nx * m.Ny) rtol = 1e-14
    @test solid_levels!(zeros(Nz), sm)[1] ≈ ρ̄[1] * 0.01 * m.dz / (m.Nx * m.Ny) rtol = 1e-14
end

# ---------------------------------------------------------------------------
# the stage budget
# ---------------------------------------------------------------------------

@testset "stage budget: contexts attribute known changes, the closure is round-off" begin
    m = BO_MESH; Nz = m.Nz
    meter = LevelMeter(m, ones(Nz), ones(Nz), 1.0)
    x = Field(CellCenter(), m); set!(x, (a, b, c) -> 1.0)
    add!(k, v) = (interior_view(x)[:, :, k] .+= v; nothing)          # +v in level k: v·Δz per area
    b = StageBudget(; bins = 1, order = (:radiation, :surface_flux))
    budget_start!(b, meter)
    budget_begin_step!(b, x)
    budget_open!(b, :radiation, :sun, x); add!(Nz, 0.25); budget_close!(b, x)
    budget_open!(b, :surface_flux, :ground, x)
    add!(1, 1.0)
    book!(b, :named, 0.5 * m.dz; k = 1)                               # half of it named inside the entry
    budget_close!(b, x)
    # the dynamics: a nested entry, an exchange between levels, a deferral
    budget_open!(b, :dynamics, :core, x; split = true, defer = true)
    budget_open!(b, :body_force, :push, x); add!(2, 0.5); budget_close!(b, x)
    add!(3, 0.125); add!(4, -0.125)                                    # transport: conserved
    budget_close!(b, x)
    budget_end_step!(b, x, 10.0)
    r = stage_budget(b)
    @test r.storage ≈ (0.25 + 1.0 + 0.5) * m.dz rtol = 1e-14
    @test r.terms[(:radiation, :sun)] ≈ 0.25m.dz rtol = 1e-14
    @test r.terms[(:surface_flux, :ground)] ≈ 0.5m.dz rtol = 1e-14
    @test r.terms[(:surface_flux, :named)] ≈ 0.5m.dz rtol = 1e-14
    @test r.terms[(:body_force, :push)] ≈ 0.5m.dz rtol = 1e-14
    @test abs(r.terms[(:dynamics, :transport)]) < 1e-12
    @test abs(r.closure) < 1e-12 && r.elapsed == 10.0
    # a layer: the dynamics carried 0.125 Δz out of the lowest three levels
    lay = stage_budget(b; layer = (0.0, 3m.dz))
    @test lay.terms[(:dynamics, :transport)] ≈ 0.125m.dz rtol = 1e-12
    # a change outside every context cannot hide
    budget_begin_step!(b, x)
    add!(5, 1.0)
    budget_end_step!(b, x, 10.0)
    @test stage_budget(b).terms[(:core, :unattributed)] ≈ m.dz rtol = 1e-14
    @test abs(stage_budget(b).closure) < 1e-12
    # the report names every term
    rep = stage_report(b; layers = (:column,), title = "test budget", units = ("u", "u/s"))
    @test occursin("test budget", rep) && occursin("radiation / sun", rep) && occursin("dynamics / transport", rep)
end

@testset "stage budget: a tendency context and the discard of a retried step" begin
    m = BO_MESH; Nz = m.Nz
    meter = LevelMeter(m, ones(Nz), ones(Nz), 1.0)
    x = Field(CellCenter(), m); tx = Field(CellCenter(), m)
    b = StageBudget(; bins = 1)
    budget_start!(b, meter)
    budget_begin_step!(b, x)
    budget_open!(b, :dynamics, :core, x; split = true, defer = true)
    # attempt 1 (rejected): a tendency of 2 in level 1 over 4 s, left open by a throw
    budget_open!(b, :tendencies, :heater, tx; scale = 4.0); interior_view(tx)[:, :, 1] .+= 2.0
    budget_discard!(b)                                                 # the core retries
    @test length(b.stack) == 1
    # attempt 2: the tendency acts over 2 s; the predictor applies it
    fill!(tx.data, 0)
    budget_open!(b, :tendencies, :heater, tx; scale = 2.0); interior_view(tx)[:, :, 1] .+= 2.0
    budget_close!(b, tx)
    interior_view(x) .+= 2.0 .* interior_view(tx)
    budget_close!(b, x)
    budget_end_step!(b, x, 2.0)
    r = stage_budget(b)
    @test r.terms[(:tendencies, :heater)] ≈ 4.0m.dz rtol = 1e-14       # 2 s × 2 per s, attempt 1 dropped
    @test r.storage ≈ 4.0m.dz rtol = 1e-14
    @test abs(r.terms[(:dynamics, :transport)]) < 1e-12 && abs(r.closure) < 1e-12
end

@testset "momentum budget: terms, the surface stress and the residual" begin
    U0 = [1.0, 2.0, 3.0]; V0 = zeros(3)
    mb = MomentumBudget(U0, V0, 0.0; terms = (:turb, :cor), diagnostics = (:model,), surface_term = :turb)
    for k in 1:3
        book_momentum!(mb, :turb, k, 0.1k, 0.0); book_momentum!(mb, :cor, k, 0.0, -0.2)
        book_momentum!(mb, :model, k, 5.0, 5.0)                        # not in the closure
    end
    book_surface_momentum!(mb, -0.05, 0.01)
    r = momentum_closure(mb, U0 .+ [0.05, 0.2, 0.3], V0 .+ [-0.19, -0.2, -0.2])
    @test r.u[:turb] ≈ [0.05, 0.2, 0.3] && r.v[:cor] == fill(-0.2, 3)
    @test maximum(abs, r.res_u) < 1e-15 && maximum(abs, r.res_v) < 1e-15 && r.rel < 1e-14
    @test r.largest ≈ 0.3
    @test_throws ArgumentError MomentumBudget(U0, V0, 0.0; terms = (:a,), surface_term = :b)
end

@testset "budgets: the surface classes and the water of the air" begin
    b = Budgets()
    sfc = UInt8[1 2; 1 3]
    add_surface!(b, :Rn, [100.0 200.0; 100.0 50.0], sfc, 10.0)
    add_surface!(b, :H, [60.0 150.0; 60.0 20.0], sfc, 10.0)
    add_surface!(b, :G, [40.0 50.0; 40.0 30.0], sfc, 10.0)
    b.ncol = [2, 1, 1, 0, 0, 0]; b.elapsed = 10.0; b.nsteps = 1
    rows = surface_budget(b)
    street = only(r for r in rows if r.name == "street")
    @test (street.Rn, street.H, street.G, street.residual) == (100.0, 60.0, 40.0, 0.0)
    @test only(r for r in rows if r.name == "domain").Rn == (2 * 100.0 + 200.0 + 50.0) / 4
    add_water!(b, :evaporation, 0.3); add_water!(b, :precipitation, -0.1); b.water_air = 0.2
    @test water_budget(b).residual ≈ 0.0 atol = 1e-15
    @test occursin("Surface energy budget", budget_report(b))
end

# ---------------------------------------------------------------------------
# the writer
# ---------------------------------------------------------------------------

@testset "output writer: records, time series, the mean, groups, a resumed file" begin
    dir = mktempdir(); path = joinpath(dir, "out.nc")
    x, y, z = cell_centres(BO_MESH)
    @test x[1] == 50.0 && z[end] == 375.0
    s = Ref(0.0); g = Dict{String,Float64}()
    ctx = nothing
    field = OutputVariable("f", "1", "a field", :xyz, _ -> fill(s[], 6, 5, 8))
    prof = OutputVariable("p", "1", "a profile", :z, _ -> fill(2s[], 8))
    ser = OutputVariable("s", "1", "a scalar", :scalar, _ -> s[])
    grp = OutputVariable("g", "1", "a group", :scalar, _ -> g; group = true)
    @test_throws ArgumentError OutputVariable("bad", "1", "", :xyzt, identity)
    @test_throws ArgumentError check_catalogue([ser, ser])
    @test_throws ArgumentError select_variables([field, ser], ["f", "nope"])
    @test catalogue_names(select_variables([field, ser, prof], ["s", "f"])) == ["s", "f"]
    # :mean — s grows linearly, s(t) = t: the mean of each interval is its midpoint
    w = OutputWriter(path; x, y, z, records = [field, prof], series = [ser, grp], ctx = ctx, averaging = :mean)
    t = 0.0
    for n in 1:8
        t += 1.0; s[] = t                               # the state at the end of the step
        n == 3 && (g["a"] = 1.0)
        n == 5 && (g["b"] = 2.0)
        accumulate_output!(w, ctx, 1.0)
        n % 4 == 0 && write_record!(w, ctx, t)
        n % 2 == 0 && write_series!(w, ctx, t)
    end
    close_output!(w)
    NCDataset(path) do nc
        @test nc["time"].var[:] == [4.0, 8.0] && nc["time_series"].var[:] == [2.0, 4.0, 6.0, 8.0]
        @test nc["f"].var[1, 1, 1, :] == Float32[2.5, 6.5]                   # means of 1..4, 5..8
        @test nc["p"].var[3, :] == Float32[5.0, 13.0]
        @test nc["s"].var[:] == [1.5, 3.5, 5.5, 7.5]
        @test nc["s"].attrib["cell_methods"] == "time_series: mean"
        # the group's members appear as booked: a from step 3, b from step 5
        @test nc["g__a"].var[2:4] == [1.0, 1.0, 1.0] && nc["g__b"].var[3:4] == [2.0, 2.0]
    end
    # :instant, and a resumed file continues after the last sample at or before t
    p2 = joinpath(dir, "inst.nc")
    w = OutputWriter(p2; x, y, z, records = [prof], series = [ser], ctx = ctx)
    for t in 0.0:1.0:5.0
        s[] = t; write_record!(w, ctx, t); write_series!(w, ctx, t)
    end
    close_output!(w)
    w = OutputWriter(p2; x, y, z, records = [prof], series = [ser], ctx = ctx, resume_time = 3.0)
    s[] = 40.0; write_record!(w, ctx, 4.0); write_series!(w, ctx, 4.0)
    close_output!(w)
    NCDataset(p2) do nc
        @test nc["time"].var[:] == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]           # record 5 rewritten, 6 left
        @test nc["s"].var[5] == 40.0 && nc["p"].var[1, 5] == 80.0f0
    end
    # nothing selected: no file
    w = OutputWriter(joinpath(dir, "none.nc"); x, y, z)
    @test output_path(w) === nothing && !isfile(joinpath(dir, "none.nc"))
end

# ---------------------------------------------------------------------------
# the model: budgets attribute what the components do
# ---------------------------------------------------------------------------

const BO_FLUX = (bottom = 0.2, top = -0.05)                   # [surface_theta_flux], K m s⁻¹

@testset "model budgets: the surface heat flux is attributed exactly ($label)" for (label, scheme, cfl, nsteps, dt) in
        (("euler", :euler, nothing, 6, 4.0), ("rk3", :rk3, nothing, 6, 4.0),
         ("rk3 adaptive", :rk3, CFLConstraints(; dt_max = 5.0), 6, nothing))
    sim = bo_state!(bo_core(; ν = 0.5))
    d = with_budgets(bo_domain(sim, SurfaceThetaFlux => BO_FLUX; scheme = scheme, cfl = cfl), (:energy, :momentum_x))
    model = Model(d)
    t = 0.0
    for _ in 1:nsteps
        cfl === nothing ? step!(model, dt) : Aeolus.TimeIntegration.step_adaptive!(model)
    end
    t = sim.t
    r = stage_budget(BOC.budgets(d).measured[:energy])
    # the prescribed fluxes put ρ₀ c_p (F_bottom − F_top) t into the air
    want = BO_RHO0 * BO_CP * (BO_FLUX.bottom - BO_FLUX.top) * t
    @test r.terms[(:tendencies, :surface_theta_flux)] ≈ want rtol = 1e-12
    @test r.storage ≈ want rtol = 1e-10                 # the transport conserves Σθ (Boussinesq)
    @test abs(r.closure) <= 1e-12 * want
    @test r.elapsed ≈ t rtol = 1e-14
    # the momentum barely changes here: its closure is round-off of the content ρ₀ H |u| (u ≈ 0.3 m/s)
    rm = stage_budget(BOC.budgets(d).measured[:momentum_x])
    @test abs(rm.closure) <= 1e-14 * BO_RHO0 * BO_MESH.Lz * 0.3
end

@testset "model budgets: the state is bit for bit that of a run without them" begin
    for scheme in (:euler, :rk3)
        sims = [bo_state!(bo_core(; ν = 0.5)) for _ in 1:2]
        specs = (SurfaceThetaFlux => BO_FLUX, Coriolis => (; f = 1e-3), Sponge => (alpha_max = 0.02, z_start = 250.0))
        d1 = bo_domain(sims[1], specs...; scheme = scheme)
        d2 = with_budgets(bo_domain(sims[2], specs...; scheme = scheme), (:energy, :momentum_x, :momentum_y))
        for _ in 1:4
            step!(Model(d1), 3.0); step!(Model(d2), 3.0)
        end
        @test bo_bits(sims[1]) == bo_bits(sims[2])
        # with the budgets off nothing is measured and the fast path holds
        @test BOC.budgets(d1) === nothing
        @test BOC.core_hooks(bo_domain(bo_core(); scheme = scheme)).body_force! === nothing
        @test BOC.core_hooks(with_budgets(bo_domain(bo_core(); scheme = scheme), (:energy,))).body_force! !== nothing
    end
end

@testset "model budgets: Coriolis and the sponge in the momentum budget" begin
    # a uniform flow with Coriolis: d⟨u⟩/dt = f⟨v⟩, all of it the component's
    f = 1e-3
    sim = bo_core()
    set!(sim.u, (x, y, z) -> 5.0); set!(sim.v, (x, y, z) -> 2.0)
    d = with_budgets(bo_domain(sim, Coriolis => (; f = f)), (:momentum_x, :momentum_y))
    ρ0H = BO_RHO0 * BO_MESH.Lz
    for _ in 1:10
        step!(Model(d), 10.0)
    end
    bx = stage_budget(BOC.budgets(d).measured[:momentum_x]); by = stage_budget(BOC.budgets(d).measured[:momentum_y])
    @test bx.terms[(:tendencies, :coriolis)] ≈ bx.storage rtol = 1e-12
    @test by.terms[(:tendencies, :coriolis)] ≈ by.storage rtol = 1e-12
    # the storage is the forward-Euler inertial rotation of (5, 2): ρ₀ H Δu
    @test bx.storage ≈ ρ0H * (interior_view(sim.u)[1, 1, 1] - 5.0) rtol = 1e-12
    @test abs(get(bx.terms, (:dynamics, :transport), 0.0)) <= 1e-12 * abs(bx.storage)
    # the sponge: u decays above z_start only, and the budget names it
    sim = bo_core()
    set!(sim.u, (x, y, z) -> 3.0)
    d = with_budgets(bo_domain(sim, Sponge => (alpha_max = 0.05, z_start = 200.0)), (:momentum_x,))
    for _ in 1:5
        step!(Model(d), 2.0)
    end
    b = stage_budget(BOC.budgets(d).measured[:momentum_x])
    @test b.terms[(:body_force, :sponge)] < 0
    @test b.terms[(:body_force, :sponge)] ≈ b.storage rtol = 1e-12
    below = stage_budget(BOC.budgets(d).measured[:momentum_x]; layer = (0.0, 200.0))
    @test abs(below.terms[(:body_force, :sponge)]) <= 1e-12 * abs(b.storage)
end

"A heater that throws on the first attempt of the first step (the adaptive step retries with dt/2)."
mutable struct BOHeater <: AbstractComponent
    thrown::Bool
end
BOC.component_name(::Type{BOHeater}) = :heater
BOC.section(::Type{BOHeater}) = "heater"
BOC.stages(::BOHeater) = (Stage(:tendencies),)
function BOC.run_stage!(c::BOHeater, d, ::Val{:tendencies}, Δt)
    interior_view(d.state.tθ)[:, :, 1] .+= 1e-3
    c.thrown || (c.thrown = true; error("BOHeater: the first attempt fails"))
    return nothing
end

@testset "model budgets: a retried adaptive step books its in-core terms once" begin
    for scheme in (:euler, :rk3)
        sim = bo_core()
        d = with_budgets(bo_domain(sim; scheme = scheme, cfl = CFLConstraints(; dt_max = 4.0), extra = (BOHeater(false),)),
                         (:energy,))
        res = Aeolus.TimeIntegration.step_adaptive!(Model(d))
        @test res[3] == 1 && res[2] == 2.0                          # one retry, at half the step
        Aeolus.TimeIntegration.step_adaptive!(Model(d))
        r = stage_budget(BOC.budgets(d).measured[:energy])
        # 1e-3 K/s in the lowest level over the 2 + 4 s taken
        want = BO_RHO0 * BO_CP * 1e-3 * BO_MESH.dz * sim.t
        @test r.terms[(:tendencies, :heater)] ≈ want rtol = 1e-12
        @test r.storage ≈ want rtol = 1e-10
        @test abs(r.closure) <= 1e-12 * want
    end
end

@testset "model budgets: the anelastic core, the water budget, the checkpoint" begin
    # anelastic: the weights are the base state's, the transport conserves Σ ρ̄ θ
    ref = isentropic_reference(BO_MESH; θ_const = 300.0)
    sim = AnelasticSimulation(BO_MESH; ref = ref, bo_bcs()...)
    d = with_budgets(bo_domain(sim, SurfaceThetaFlux => BO_FLUX), (:energy,))
    for _ in 1:4
        step!(Model(d), 4.0)
    end
    r = stage_budget(BOC.budgets(d).measured[:energy])
    @test abs(r.closure) <= 1e-12 * abs(r.storage)
    @test r.terms[(:tendencies, :surface_theta_flux)] > 0
    # the water budget needs moisture; with it, qv measures
    @test_throws ArgumentError with_budgets(bo_domain(bo_core()), (:water,))
    sim = bo_core(; moisture = true)
    set!(sim.qv, (x, y, z) -> 0.01)
    d = with_budgets(bo_domain(sim), (:water,))
    step!(Model(d), 2.0)
    @test abs(stage_budget(BOC.budgets(d).measured[:water]).closure) < 1e-15
    @test_throws ArgumentError with_budgets(bo_domain(bo_core()), (:heat,))
    # the budgets are in the checkpoint components and restore
    sim = bo_core()
    d = with_budgets(bo_domain(sim, SurfaceThetaFlux => BO_FLUX), (:energy,))
    step!(Model(d), 4.0)
    cc = BOC.checkpoint_components(d)
    @test haskey(cc, :budgets)
    s = deepcopy(Aeolus.IO.checkpoint_state(cc.budgets))
    step!(Model(d), 4.0)
    Aeolus.IO.restore_checkpoint_state!(cc.budgets, s)
    @test stage_budget(BOC.budgets(d).measured[:energy]).elapsed == 4.0
end

# ---------------------------------------------------------------------------
# the catalogue and the configuration
# ---------------------------------------------------------------------------

@testset "catalogue: the core's variables, the components' and the selection" begin
    sim = bo_state!(bo_core())
    d = with_budgets(bo_domain(sim), (:energy, :momentum_x, :momentum_y))
    cat = domain_catalogue(d)
    names = catalogue_names(cat)
    @test issubset(["u", "v", "w", "p", "theta", "u_profile", "theta_profile", "u_mean", "theta_mean",
                    "budget_energy", "budget_momentum"], names)
    @test !("qv" in names) && !("budget_water" in names) && !("budget_tracer" in names)
    @test issubset(names, BOC.CORE_OUTPUTS)
    v = only(select_variables(cat, ["theta_mean"]))
    @test v.getter(d) ≈ sum(interior_view(sim.θ)) / length(interior_view(sim.θ)) rtol = 1e-14
    p = only(select_variables(cat, ["theta_profile"])).getter(d)
    @test p ≈ [sum(interior_view(sim.θ)[:, :, k]) / 30 for k in 1:8] rtol = 1e-14
    # every component's catalogue matches the names its type declares
    for c in d.components
        @test catalogue_names(BOC.outputs(c)) == collect(BOC.output_names(typeof(c), nothing))
    end
end

@testset "config: [output] selects from the catalogues; budgets turn on with their variable" begin
    load(ov) = load_run_config(IOBuffer(""); overrides = vcat(["grid.size=[8,8,8]", "fftw.planner=\"estimate\""], ov))
    @test load([]).output.variables == collect(BOC.DEFAULT_OUTPUTS)
    for (ov, msg) in (("output.variables=[\"qv\"]", "moisture"),
                      ("output.variables=[\"ql\"]", "no enabled component publishes"),
                      ("output.variables=[\"budget_tracer\"]", "carries a tracer"),
                      ("output.variables=[\"budget_water\"]", "moisture"),
                      ("output.variables=[\"nonsense\"]", "no enabled component publishes"))
        err = try load([ov]); nothing catch e; e end
        @test err isa Aeolus.Config.ConfigError && occursin(msg, sprint(showerror, err))
    end
    @test_throws Aeolus.Config.ConfigError load(["output.variables=[\"budget_energy\"]", "dynamics.theta0=0.0"])
    # a run with budgets selected writes their terms, and they close
    cfg = load(["time.duration=\"none\"", "time.nsteps=4", "surface_theta_flux.bottom=0.1",
                "output.variables=[\"theta_mean\",\"budget_energy\"]", "output.series_cadence=0.0"])
    r = run_config(cfg; root = mktempdir(), console = devnull, repo = mktempdir())
    NCDataset(joinpath(r.folder.path, "output", "output.nc")) do nc
        @test length(nc["time_series"].var[:]) == 5                     # t = 0 and every step
        st = nc["budget_energy__storage"].var[:]; cl = nc["budget_energy__closure"].var[:]
        @test st[end] > 0 && all(abs.(cl[2:end]) .<= 1e-12 .* abs.(st[2:end]))
        @test nc["budget_energy__tendencies__surface_theta_flux"].var[end] ≈ st[end] rtol = 1e-9
        @test !haskey(nc, "u")
    end
    # without a budget variable the domain measures nothing
    r = run_config(load(["time.duration=\"none\"", "time.nsteps=1"]); root = mktempdir(), console = devnull,
                   repo = mktempdir())
    @test BOC.budgets(only(r.obj.model.domains)) === nothing
end
