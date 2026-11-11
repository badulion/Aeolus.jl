# E1 (16ad00c9, the update of e: row-parallel, branch-free, vectorised) and
# E2 (cf704ba3, the dual strain rate: bulk loop without checks), old against
# new in one process on the Berlin 100 m closure after 3 steps: the old
# kernels are the copies of tke_update_strain_old.jl. The inputs of the
# update (ν, N², D, the transport, P_w) are those of one separate-loop
# advance from the stepped state; e is reset before every call (copy_e
# times the reset alone).
#
#   julia -t N --project=<build> tke_update_strain_bench.jl <out.toml> [key=value ...]
@isdefined(FIXES) || include(joinpath(@__DIR__, "common.jl"))
include(joinpath(@__DIR__, "tke_update_strain_old.jl"))

FIXES["E1E2"] = function ()
    s = berlin(); c, sim = s.c, s.sim
    g = PS.mesh(c.e); Δt = 2.0
    e0 = copy(c.e.data); st0 = copy(c.stats)
    PS._advance_separate!(c, sim, Δt)          # this step's ν, N², D, tend, Pw in the scratch
    copyto!(c.e.data, e0)
    pbs = PS._surface_buoyancy(c, sim)
    u, v, w, bcs = sim.u, sim.v, sim.w, sim.bcs
    upd(f) = (copyto!(c.e.data, e0); f(c, Δt, pbs))
    # agreement: e pointwise (the same expressions, selected instead of
    # branched: 0 expected), the budget sums to round-off (a different
    # summation order), D the same shares in the same order (0 expected)
    fill!(c.stats, 0); upd(PS._update_old!); eo = copy(c.e.data); so = copy(c.stats)
    fill!(c.stats, 0); upd(PS._update!); en = copy(c.e.data); sn = copy(c.stats)
    PS._dual_strain_old!(c.D, u, v, w, bcs, g); Do = copy(c.D.data)
    PS._dual_strain!(c.D, u, v, w, bcs, g); Dn = copy(c.D.data)
    ii, jj, kk = PS.interior(g)
    de = relmax(en, eo); ds = relmax(sn, so); dD = relmax(Dn[ii, jj, kk], Do[ii, jj, kk])
    println("  rel max|Δe| ", de, "  rel max|Δstats| ", ds, "  rel max|ΔD| ", dD)
    t = measure(("update_old" => () -> upd(PS._update_old!), "update_new" => () -> upd(PS._update!),
                 "strain_old" => () -> PS._dual_strain_old!(c.D, u, v, w, bcs, g),
                 "strain_new" => () -> PS._dual_strain!(c.D, u, v, w, bcs, g),
                 "copy_e" => () -> copyto!(c.e.data, e0)))
    copyto!(c.e.data, e0); copyto!(c.stats, st0)
    tol = 4eps()
    return Dict{String,Any}("timings" => t, "grid" => collect(size(c.e.data)), "mask" => c.mask !== nothing,
        "rel_de" => de, "rel_dstats" => ds, "rel_dD" => dD, "tol" => tol, "tol_stats" => 1e-12,
        "agree" => de <= tol && dD <= tol && ds <= 1e-12)
end

standalone("E1E2", @__FILE__)
