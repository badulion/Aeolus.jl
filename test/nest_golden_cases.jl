# Golden cases of NestedRun (#328): 20 parent steps of six configurations,
# reduced to exact checksums. `test/nest_golden_gen.jl` recorded them on the
# code before the nest hierarchy refactor (commit "NEST-g: golden snapshot
# of NestedRun"); `era2_nest_hierarchy.jl` asserts that NestedRun, now a
# two-domain NestHierarchy, still reproduces them.
#
# Included by both scripts; defines `nest_golden_cases()`, a vector of
# `name => () -> checksums::Vector{Float64}`.

using Random
using Aeolus
using Aeolus.Nesting
using Aeolus.Fields: interior_view

function _golden_box(; core = step!, feedback = :one_way, sponge = nothing, solid = false,
                     adaptive = false, moist = false)
    L = 1600.0
    pm = RegularGrid(Float64; size = (16, 16, 12), extent = (L, L, 1200.0), halo = (3, 3, 3))
    p = BoussinesqSimulation(pm; C_s = 0.17, ν = 2.0, moisture = moist,
                             bcs = FieldBoundaryConditions(:periodic))
    nest = nested_grid(pm, ((400.0, 1200.0), (400.0, 1200.0), (300.0, 900.0)), 2)
    cm = nest_child_mesh(nest)
    c = BoussinesqSimulation(cm; C_s = 0.17, ν = 2.0, moisture = moist,
                             bcs = nest_bcs(cm, :nest, :nest, :nest; moisture = moist))
    rng = Xoshiro(7)
    for k in axes(p.θ.data, 3), j in axes(p.θ.data, 2), i in axes(p.θ.data, 1)
        p.θ.data[i, j, k] = 1.0 + exp(-((i - 8)^2 + (j - 8)^2 + (k - 7)^2) / 6)
        moist && (p.qv.data[i, j, k] = 0.01 * (1 + exp(-((i - 10)^2 + (j - 8)^2) / 8)))
    end
    fill!(p.u.data, 3.0); fill!(p.v.data, 1.0)
    interior_view(p.u) .+= 0.5 .* randn(rng, size(interior_view(p.u)))
    interior_view(p.v) .+= 0.5 .* randn(rng, size(interior_view(p.v)))
    Aeolus.Simulations._project_velocity!(p)
    for (cf, pf) in ((c.θ, p.θ), (c.u, p.u), (c.v, p.v), (c.w, p.w), (c.qv, p.qv))
        interpolate_parent_to_child!(cf, pf, nest)
    end
    sm = nothing
    if solid
        sm = falses(size(c.θ.data))
        sm[10:12, 10:13, 1:9] .= true             # a block on the child's floor region
    end
    cfl = adaptive ? CFLConstraints(C_advective = 0.5) : nothing
    run = NestedRun(p, c, nest; feedback, sponge, solid = sm,
                    advance_parent = domain_advance(core), advance_child = domain_advance(core),
                    cfl_parent = cfl, cfl_child = cfl)
    return p, c, run
end

function _golden_checksums(p, c, run)
    out = Float64[]
    for s in (p, c), f in (s.u, s.v, s.w, s.θ)
        v = interior_view(f)
        push!(out, sum(v), sum(abs2, v))
    end
    st = run.stats
    append!(out, Float64[st.parent_steps, st.child_substeps, st.parent_retries,
                         st.child_retries, st.n_min, st.n_max, st.dt_parent_min,
                         st.dt_parent_max, st.dt_child_min, st.dt_child_max,
                         st.max_child_cfl, run.t])
    return out
end

function _golden_run(nsteps; dt = nothing, kw...)
    p, c, run = _golden_box(; kw...)
    for _ in 1:nsteps
        step_nested!(run; dt)
    end
    return _golden_checksums(p, c, run)
end

nest_golden_cases() = [
    "one-way step! fixed" => () -> _golden_run(20; dt = 4.0),
    "one-way step_rk3! adaptive sponge" =>
        () -> _golden_run(20; core = step_rk3!, adaptive = true, sponge = NestSponge(width = 3)),
    "two-way step! fixed moist" => () -> _golden_run(20; dt = 4.0, feedback = :two_way, moist = true),
    "two-way step_rk3! adaptive" =>
        () -> _golden_run(20; core = step_rk3!, adaptive = true, feedback = :two_way),
    "two-way step! solid" => () -> _golden_run(20; dt = 4.0, feedback = :two_way, solid = true),
    "one-way step! adaptive solid" => () -> _golden_run(20; adaptive = true, solid = true),
]
