# Numerics swap-ins of the Berlin case (#132): the pressure-solver, SGS,
# momentum- and scalar-advection slots, in the :smoke window.
#
#   * the multigrid-preconditioned CG pressure solve gives the cosine-FFT
#     result to 1e-8 after 100 steps and keeps the flow divergence-free;
#   * MLSGS with the Smagorinsky closure reproduces the default run bit
#     for bit (the eddy_viscosity! hook of step! adds its terms where the
#     built-in closure does);
#   * the σ-model provider (`sgs = :sigma`, #184): ν_t ≥ 0, exactly 0 in a
#     laminar parallel flow on the Berlin mesh, and far below Smagorinsky
#     in the (γ-laminar) free stream above the sponge start;
#   * configuration errors (halo too small for a scheme, a step! keyword
#     claimed by two couplings) are caught.

using Test
using Aeolus
using Aeolus.Fields: interior_view
using Printf
using Statistics: mean

isdefined(Main, :BerlinCase) ||
    include(joinpath(@__DIR__, "..", "examples", "urban", "BerlinCase.jl"))
const BC = Main.BerlinCase

function _run_smoke(nsteps; physics...)
    cfg = BC.CaseConfig(:smoke; verbose = false, nsteps = nsteps,
                        physics = Dict{Symbol,Any}(physics...))
    case = BC.build_case(cfg; io = devnull)
    BC.run_case!(case)
    return case
end

@testset "Berlin numerics swap-ins (#132)" begin
    N = 100
    ref = _run_smoke(N)
    @test ref.sim.solver isa CosineFFTPoissonSolver

    @testset "MG-preconditioned CG: FFT result to 1e-8 after $N steps, divergence-free" begin
        mg = _run_smoke(N; pressure_solver = :mg_cg)
        @test mg.sim.solver isa MGPreconditionedCG
        d = Dict(f => maximum(abs, interior_view(getfield(mg.sim, f)) .- interior_view(getfield(ref.sim, f)))
                 for f in (:u, :v, :w, :θ))
        dq = maximum(abs, interior_view(BC.tracer(mg, :qv)) .- interior_view(BC.tracer(ref, :qv)))
        div = max_divergence(mg.sim)
        @info @sprintf("  MG-CG vs cosine FFT after %d steps: max|Δ| u %.1e v %.1e w %.1e m/s, θ %.1e K, qv %.1e; max|∇·u| %.1e s⁻¹ (FFT %.1e)",
                       N, d[:u], d[:v], d[:w], d[:θ], dq, div, max_divergence(ref.sim))
        @test all(<(1e-8), values(d))
        @test dq < 1e-8
        @test div < 1e-8
    end

    @testset "MLSGS(Smagorinsky) reproduces the built-in closure bit for bit" begin
        ml = _run_smoke(N; sgs = :mlsgs)
        @test BC.find_coupling(ml, :sgs_mlsgs) !== nothing
        # interiors (the halos of `ref` were refreshed by max_divergence above)
        @test all(f -> interior_view(getfield(ml.sim, f)) == interior_view(getfield(ref.sim, f)),
                  (:u, :v, :w, :θ, :p, :νt))
        @test all(((a, b),) -> interior_view(a.field) == interior_view(b.field), zip(ml.tracers, ref.tracers))
        @test ml.bus.Ts == ref.bus.Ts
    end

    @testset "σ-model provider (#184): ν_t ≥ 0, laminar free stream" begin
        sg = _run_smoke(N; sgs = :sigma)
        c = BC.find_coupling(sg, :sgs_sigma)
        @test c !== nothing && c.closure isa SigmaModel
        sim, m = sg.sim, sg.mesh
        @test minimum(interior_view(sim.νt)) >= 0
        @test all(isfinite, interior_view(sim.u))
        p = BC.sigma_νt_profile(c, sg)
        @test p.min_σ >= 0
        # the free stream above the sponge start is laminar by the
        # Transition indicator (γ = 0 there) …
        for f in (sim.u, sim.v, sim.w)
            fill_halos!(f, sim.bcs)
        end
        lam = falses(m.Nx, m.Ny, m.Nz)
        laminar_mask!(lam, sim.u, sim.v, sim.w)
        γfree = 1 - mean(lam[:, :, p.ksp:end])
        @info @sprintf("  σ-model after %d steps: mean ν_t canopy %.3e (Smagorinsky on the same flow %.3e), above the sponge start %.3e (%.3e; ratio %.2e, max %.2e) m²/s; γ there %.3f",
                       N, p.canopy_σ, p.canopy_smag, p.free_σ, p.free_smag, p.free_σ / p.free_smag, p.free_σ_max, γfree)
        @test γfree == 0
        # … but not an exactly parallel flow: the canopy's weak 3-D
        # disturbances reach it (|ω′| below the γ threshold), so σ₃ ≠ 0
        # and ν_t is small, not zero. Smagorinsky sees the mean shear.
        @test p.free_σ <= 0.05 * p.free_smag
        @test p.canopy_σ > 0
        # exactly zero for an exactly laminar parallel flow on the Berlin
        # mesh, through the provider's closure object (compute_sgs!, the
        # call every core makes): a turning, sheared profile u(z), v(z), w = 0
        zc(k) = (k - m.Hz - 0.5) * m.dz
        for K in axes(sim.u.data, 3)
            sim.u.data[:, :, K] .= 5 * log1p(max(zc(K), 0.0) / 0.5)
            sim.v.data[:, :, K] .= 0.02 * zc(K)
        end
        fill!(sim.w.data, 0.0)
        f = compute_sgs!(c.closure, sim)
        νs = Field(CellCenter(), m); compute_smagorinsky_νt!(νs, sim.u, sim.v, sim.w, sg.cfg.C_s)
        @test maximum(interior_view(f.νh)) == 0 && maximum(interior_view(f.κh)) == 0
        @test minimum(interior_view(νs)) > 0
    end

    @testset "configuration errors" begin
        # WENO-Z7 needs a halo of 4; the case default is 3
        cfg = BC.CaseConfig(:smoke; verbose = false,
                            options = Dict{Symbol,Any}(:scalar_advection => (scheme = :weno_z7,)))
        @test_throws ErrorException BC.build_case(cfg; io = devnull)
        # the positivity limiter needs a reconstruction
        cfg = BC.CaseConfig(:smoke; verbose = false,
                            options = Dict{Symbol,Any}(:scalar_advection => (scheme = :upwind,)))
        @test_throws ErrorException BC.build_case(cfg; io = devnull)
        # a keyword claimed by two couplings
        @test_throws ErrorException BC.merge_keywords("step!", (:a => (x = 1,), :b => (x = 2,)))
        @test BC.merge_keywords("step!", (:a => (x = 1,), :b => (;), :c => (y = 2,))) == (x = 1, y = 2)
    end
end
