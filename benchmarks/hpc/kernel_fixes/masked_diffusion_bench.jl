# E5 (33b3ed00 + review 811a9b8e, #559): the masked eddy diffusion vectorises
# (@propagate_inbounds on _open and _vertical_divergence). No old copy: one
# process per build, alternating bench/all and bench/all-revert-E5
# (node17.sh). Berlin 100 m does not run this kernel (no urban tracer
# transport, the TKE closure without a mask, a Boussinesq core: the changed
# methods are never called; `berlin_mask` records the closure's mask when
# the process has built Berlin), so the kernel runs on the synthetic case
# of the build's test/support/tracer_diffusion_cases.jl at Berlin's size,
# 200 x 200 x 60 with two solid blocks (the laptop's case, #559):
# sgs_diffuse! with the separate loops (never) and the blocked pass
# (if_possible), q reset before every call (copy_q times the reset alone).
#
# Without `ref` the process writes q after one call of each (`<out>.E5.jls`);
# with `ref=<a bench/all TOML>` it compares with that (the input is seeded:
# the same in every process).
#
#   julia -t N --project=<build> masked_diffusion_bench.jl <out.toml> [ref=<toml>] [key=value ...]
@isdefined(FIXES) || include(joinpath(@__DIR__, "common.jl"))
include(joinpath(SUPPORT, "tracer_diffusion_cases.jl"))

FIXES["E5"] = function ()
    g = RegularGrid(; size = (200, 200, 60), extent = (400.0, 400.0, 120.0), halo = (3, 3, 3))
    case, d, q, _ = tdc_case(g; core = :boussinesq)
    q0 = copy(q.data); κ = case.sim.κt
    run(set) = (SIM.set_loop_fusion!(case.sim, set); copyto!(q.data, q0);
                TDU.sgs_diffuse!(d, case, q, 0.8; water = true, κ = κ, κv = κ))
    run(:never); qs = copy(q.data); run(:if_possible); qb = copy(q.data)
    out = Dict{String,Any}("case" => "tdc_case 200x200x60 boussinesq, mask", "solid_cells" => count(d.mask),
                           "rel_dq_pass_separate" => relmax(qb, qs))
    STATE[] === nothing || (out["berlin_mask"] = STATE[].c.mask !== nothing)
    ref = read_ref("E5")
    if ref === nothing
        write_dump("E5", (separate = qs, pass = qb))
    else
        # agreement: the same expressions, only the bounds checks differ (0 expected)
        out["rel_dq_separate"] = relmax(qs, ref.separate); out["rel_dq_pass"] = relmax(qb, ref.pass)
        out["tol"] = 1e-12; out["agree"] = max(out["rel_dq_separate"], out["rel_dq_pass"]) <= 1e-12
        println("  rel max|Δq| against ref: separate ", out["rel_dq_separate"], "  pass ", out["rel_dq_pass"])
    end
    out["timings"] = measure(("separate" => () -> run(:never), "pass" => () -> run(:if_possible),
                              "copy_q" => () -> copyto!(q.data, q0)))
    return out
end

standalone("E5", @__FILE__)
