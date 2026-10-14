# Checkpoint / restart of a nested run (N-berlin-restart, #269):
# src/nesting/restart.jl (nest_state / restore_nest_state!) over the
# structural walk of src/io/structural_state.jl.
#
# The assertion is **bitwise**: a parent + child pair driven the way the
# model's nest engine drives an urban site (the parent window opened, the
# child stepped by itself with subcycle_dt, the window closed: the engine's
# internals of Aeolus.Nesting, NEST-p3 #483), checkpointed at a window
# boundary (where the run loop checkpoints), written to disk and restored
# into a freshly built run,
# continues across the following window boundaries identical to the
# uninterrupted run, bit for bit, in every field of both domains (with
# halos), the closures' memory, the parent time levels, the child's open
# boundary buffers, the inflow generator's planes and the clocks. Adaptive
# steps in both domains (refinement 3: about 3 child steps per parent
# step), SGS closures with memory in both domains (prognostic TKE in the
# parent, Lagrangian-dynamic in the child), the synthetic inflow generator
# on (DigitalFilterGenerator, parent-resolved statistics); one-way and
# two-way (refluxing, inflow band). The suite runs at one
# thread; the same file at 12 threads (`test/run.jl --threads 12 --only
# nest_restart.jl`) is quoted in #269's PR.
#
# A restart of the two simulations alone (IO's checkpoint! / restart!,
# without the nest state) is shown to **differ**: the test can see what the
# nest state adds.

using Test
using Printf
using Random
using Serialization
using Aeolus
using Aeolus.Nesting
using Aeolus.Fields: interior_view
using Aeolus.TimeIntegration: CFLConstraints, adaptive_dt
using Aeolus.BoundaryConditions: open_boundary_snapshot

const NR_θ = 300.0
const NR_CFL = CFLConstraints(C_advective = 0.5)
const NR_N = 20                 # child steps of the uninterrupted run
const NR_W = 3                  # the checkpoint: after this parent window

mutable struct _NRDriver
    run::Any
    it::Any
    pc::Any                      # the parent's closure
    cc::Any                      # the child's closure
    window::Any                  # the open parent window, or nothing
    n::Int                       # child steps taken
    dtc::Union{Nothing,Float64}  # the child's last step
    windows::Int                 # windows closed
end

function _nr_build(feedback::Symbol; seed = 2)
    pm = RegularGrid(Float64; size = (16, 16, 8), extent = (1600.0, 1600.0, 800.0), halo = (3, 3, 3))
    p = BoussinesqSimulation(pm; θ₀ = NR_θ, C_s = 0.17, θ_flux_bottom = 0.1,
                             bcs = FieldBoundaryConditions(:periodic, :periodic, :wall))
    rng = Xoshiro(seed)
    for k in axes(p.θ.data, 3)
        z = (k - pm.Hz - 0.5) * pm.dz
        p.θ.data[:, :, k] .= NR_θ + (z > 400 ? 0.005 * (z - 400) : 0.0)
    end
    interior_view(p.θ)[:, :, 1:2] .+= 0.2 .* (rand(rng, 16, 16, 2) .- 0.5)
    p.u.data .= 3.0; p.v.data .= 1.0
    interior_view(p.u) .+= 0.3 .* randn(rng, size(interior_view(p.u)))
    interior_view(p.v) .+= 0.3 .* randn(rng, size(interior_view(p.v)))
    step!(p, 1e-6); p.t = 0.0                       # project the noise
    nest = nested_grid(pm, ((500.0, 1100.0), (500.0, 1100.0), (0.0, 400.0)), 3)
    cm = nest_child_mesh(nest)
    c = BoussinesqSimulation(cm; θ₀ = NR_θ, C_s = 0.17, θ_flux_bottom = 0.1,
                             bcs = nest_bcs(cm, :nest, :nest; bottom = :wall, top = :nest))
    st = nest_interp_state(nest)
    interpolate_parent_state!(st, p, nest; order = 3, region = :all)
    for (cf, sf) in ((c.u, st.u), (c.v, st.v), (c.w, st.w), (c.θ, st.θ))
        cf.data .= sf.data
    end
    pc = TKEClosure(pm); initialize_tke!(pc, p)
    cc = LagrangianDynamicSmagorinsky(cm)
    run = NestedRun(p, c, nest; feedback, cfl_parent = NR_CFL,
                    advance_parent = domain_advance(step_rk3!; sgs = pc),
                    advance_child = domain_advance(step_rk3!; sgs = cc))
    step!(c, 1e-9); c.t = 0.0                       # project the prolongated start
    # the statistics are folded where the parent is at rest (window opening),
    # as the Berlin nest does: a parent step running ahead must not be read
    it = InflowTurbulence(run; statistics_kwargs = (τ_avg = 60.0,), statistics_interval = Inf)
    feedback === :two_way && exclude_inflow_feedback!(run, it)
    return _NRDriver(run, it, pc, cc, nothing, 0, nothing, 0)
end

# One child step, opening and closing parent windows as the nest engine
# does for a child that picks its own steps (the urban site).
function _nr_step!(d::_NRDriver)
    NX = Aeolus.Nesting
    run = d.run; c = run.child
    if d.window === nothing
        d.window = NX._begin_parent_step!(run)
        fold_inflow_statistics!(d.it; interval = 10.0)
    end
    w = d.window
    lim = adaptive_dt(c, NR_CFL; dt_prev = d.dtc, sgs = d.cc)
    h = subcycle_dt(nest_window_end(run) - c.t, lim)
    run.sub_t0 = Float64(c.t)
    NX._begin!(run, :child)
    run.advance_child(c, h; body_force! = NX._hook(run, :child))
    NX._commit!(run, :child)
    d.dtc = h; d.n += 1
    if w.t_child_end - c.t <= 1e-9 * w.Δt
        NX._end_parent_step!(run, w, true)
        d.window = nothing; d.windows += 1
    end
    return d
end

# a parent time level: a full array, or the halo slabs of a :shell state
_nr_data(a) = a isa Array ? copy(a) : map(copy, Aeolus.Nesting._slabs(a))

# Everything a later step reads, for the bitwise comparison.
function _nr_digest(d::_NRDriver)
    run = d.run
    p, c = run.parent, run.child
    out = Pair{String,Any}[]
    for (nm, s) in (("parent", p), ("child", c)), (k, f) in pairs(checkpoint_fields(s))
        push!(out, "$nm.$k" => copy(f.data))
    end
    push!(out, "parent.t" => p.t, "child.t" => c.t, "run.t" => run.t, "run.t0" => run.t0,
          "run.Δt" => run.Δt, "dt_prev_parent" => run.dt_prev_parent, "dtc" => d.dtc,
          "parent closure" => deepcopy(Aeolus.Operators.sgs_state(d.pc)),
          "child closure" => deepcopy(Aeolus.Operators.sgs_state(d.cc)),
          "child open buffers" => open_boundary_snapshot(c),
          "s_old" => [_nr_data(f.data) for f in (run.s_old.u, run.s_old.v, run.s_old.w, run.s_old.θ)],
          "s_new" => [_nr_data(f.data) for f in (run.s_new.u, run.s_new.v, run.s_new.w, run.s_new.θ)],
          "generator" => [deepcopy(generator_state(f.gen)) for f in d.it.faces],
          "steps" => (run.stats.parent_steps, run.stats.child_substeps, d.n, d.windows))
    return out
end

_nr_differing(a, b) = [ka for ((ka, va), (kb, vb)) in zip(a, b) if !(ka == kb && isequal(va, vb))]

@testset "N-berlin-restart: nested run restart, bitwise (#269), $(Threads.nthreads()) thread(s)" begin
    # the FFT plans of a fresh build must be the checkpointed run's: the
    # estimate planner (a resumed run imports the run folder's wisdom instead)
    fftw0 = fftw_policy().planner
    set_fftw_policy!(; planner = :estimate)
    dir = mktempdir()
    for feedback in (:one_way, :two_way)
        label = "$feedback"
        @testset "$label" begin
            # the uninterrupted run
            A = _nr_build(feedback)
            for _ in 1:NR_N
                _nr_step!(A)
            end
            # the interrupted run: checkpoint at the end of parent window NR_W
            B = _nr_build(feedback)
            while B.windows < NR_W
                _nr_step!(B)
            end
            K = B.n
            @test B.window === nothing && 0 < K < NR_N - 4
            path = joinpath(dir, "nest_$(feedback).jls")
            t_w = @elapsed serialize(path, (nest = nest_state(B.run),
                                            driver = (n = B.n, dtc = B.dtc, windows = B.windows)))
            mb = filesize(path) / 2^20
            # B goes on: the checkpoint did not disturb it
            for _ in K+1:NR_N
                _nr_step!(B)
            end
            # restarted from the file into a freshly built run
            C = _nr_build(feedback)
            ck = deserialize(path)
            t_r = @elapsed restore_nest_state!(C.run, ck.nest)
            C.n, C.dtc, C.windows = ck.driver.n, ck.driver.dtc, ck.driver.windows
            w0 = C.windows
            for _ in K+1:NR_N
                _nr_step!(C)
            end
            dA, dB, dC = _nr_digest(A), _nr_digest(B), _nr_digest(C)
            @printf("  %-28s checkpoint %.2f MiB, write %.3f s, restore %.3f s; %d windows after the restart, %d parent steps, %d child steps\n",
                    label, mb, t_w, t_r, C.windows - w0, C.run.stats.parent_steps, C.n)
            @test C.windows - w0 >= 2                          # across window boundaries
            @test isempty(_nr_differing(dA, dB))               # checkpointing is read-only
            diff = _nr_differing(dA, dC)
            isempty(diff) || @info "differing after the restart" diff
            @test isempty(diff)                                # bitwise
            @test all(isfinite, interior_view(C.run.child.u)) && maximum(abs, interior_view(C.run.child.u)) < 20
            # the generator injected something, and the parent stepped
            @test any(f -> any(!iszero, inflow_perturbation(C.it, f.side)[1]), C.it.faces)

            # the two simulations alone (IO checkpoint!, closures included),
            # without the nest state: the continuation is not the run's
            D = _nr_build(feedback)
            while D.windows < NR_W
                _nr_step!(D)
            end
            pp, cp = joinpath(dir, "p.jls"), joinpath(dir, "c.jls")
            checkpoint!(D.run.parent, pp; sgs = D.pc); checkpoint!(D.run.child, cp; sgs = D.cc)
            E = _nr_build(feedback)
            restart!(E.run.parent, pp; sgs = E.pc); restart!(E.run.child, cp; sgs = E.cc)
            E.n, E.dtc = D.n, D.dtc
            dE = try
                for _ in K+1:NR_N
                    _nr_step!(E)
                end
                _nr_differing(dA, _nr_digest(E))
            catch err                                  # (or the cold nest fails outright)
                ["failed: " * sprint(showerror, err)[1:min(end, 80)]]
            end
            @printf("  %-28s simulations alone (no nest state): %d of %d items differ (%s)\n",
                    label, length(dE), length(dA), first(dE, 1))
            @test !isempty(dE)
        end
    end

    @testset "negative: no nest state, another mode, through checkpoint!" begin
        A = _nr_build(:one_way)
        for _ in 1:4
            _nr_step!(A)
        end
        s = nest_state(A.run)
        T = _nr_build(:two_way)
        @test_throws ArgumentError restore_nest_state!(T.run, s)         # one-way state, two-way run
        F = _nr_build(:one_way)
        err = try restore_nest_state!(F.run, nothing); nothing catch e; e end
        @test err isa ArgumentError && occursin("no nest state", sprint(showerror, err))
        # a snapshot of the child alone cannot restart the nest
        cp = joinpath(dir, "child_only.jls")
        checkpoint!(A.run.child, cp; sgs = A.cc)
        @test_throws ArgumentError restart!(F.run.child, cp; sgs = F.cc, components = (nest = F.run,))
        # as a checkpoint component, at a window boundary
        while A.window !== nothing
            _nr_step!(A)
        end
        cp2 = joinpath(dir, "with_nest.jls")
        checkpoint!(A.run.child, cp2; sgs = A.cc, components = (nest = A.run,))
        G = _nr_build(:one_way)
        restart!(G.run.child, cp2; sgs = G.cc, components = (nest = G.run,))
        G.n, G.dtc, G.windows = A.n, A.dtc, A.windows
        for _ in 1:4
            _nr_step!(A); _nr_step!(G)
        end
        @test isempty(_nr_differing(_nr_digest(A), _nr_digest(G)))
    end
    set_fftw_policy!(; planner = fftw0)
end
