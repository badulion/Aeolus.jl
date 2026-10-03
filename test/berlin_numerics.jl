# Numerics swap-ins of the Berlin case (#132): the pressure-solver, SGS,
# momentum- and scalar-advection slots, in the :smoke window.
#
#   * the multigrid-preconditioned CG pressure solve gives the cosine-FFT
#     result to 1e-8 after 100 steps and keeps the flow divergence-free;
#   * MLSGS with the Smagorinsky closure reproduces the default run bit
#     for bit (the eddy_viscosity! hook of step! adds its terms where the
#     built-in closure does);
#   * the SGS option scalar_tensor = :isentropic (P55-c, #181): with the
#     closure's own κv (= κh) the tensor is κh I and the run is the default
#     one to round-off; with diapycnal = 0.1 it rotates and changes θ;
#   * the SGS stability correction (P55-e, #187) is on by default; it cuts
#     νv in the capping inversion; `sgs_stability = :none` is the bare closure;
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

    function run_opts(o; kw...)
        cfg = BC.CaseConfig(:smoke; verbose = false,
                            options = Dict{Symbol,Any}(:sgs_smagorinsky => o), (; nsteps = N, kw...)...)
        case = BC.build_case(cfg; io = devnull)
        BC.run_case!(case)
        return case
    end
    # the uncorrected closure (sgs_stability = :none, P55-e): νh = νv, κh = κv
    ref0 = run_opts((;); sgs_stability = :none)

    @testset "SGS scalar_tensor = :isentropic (P55-c, #181)" begin
        # with the uncorrected closure (κv = κh, the setting #181 was
        # validated in; the default closure is stability-corrected since
        # P55-e and has κv ≠ κh)
        iso = run_opts((scalar_tensor = :isentropic,); sgs_stability = :none)
        c = BC.find_coupling(iso, :sgs_smagorinsky)
        @test c.closure isa IsentropicDiffusion
        # κv = κh: K = κh I, the triad operator is the diagonal one to round-off
        dθ = maximum(abs, interior_view(iso.sim.θ) .- interior_view(ref0.sim.θ))
        rot = run_opts((scalar_tensor = :isentropic, diapycnal = 0.1); sgs_stability = :none)
        dθr = maximum(abs, interior_view(rot.sim.θ) .- interior_view(ref0.sim.θ))
        fr = BC.isentropic_fraction(BC.find_coupling(rot, :sgs_smagorinsky).closure)
        @info @sprintf("  isentropic scalar tensor after %d steps: max|Δθ| vs default %.1e K (κv = κh), %.1e K (κv = 0.1 κh, %.1f %% of the cells rotated)",
                       N, dθ, dθr, 100fr)
        @test dθ < 1e-9
        @test dθr > 1e-6
        @test 0 < fr < 1
        @test all(isfinite, interior_view(rot.sim.θ))
        @test_throws ArgumentError run_opts((scalar_tensor = :bogus,))
        # with the default (stability-corrected) closure the isentropic tensor
        # gets the corrected κv < κh in stable air and rotates without `diapycnal`
        isc = run_opts((scalar_tensor = :isentropic,))
        cc = BC.find_coupling(isc, :sgs_smagorinsky).closure
        @test cc isa IsentropicDiffusion && cc.inner isa StabilityCorrected
        @test BC.isentropic_fraction(cc) > 0
        @test all(isfinite, interior_view(isc.sim.θ))
    end

    @testset "SGS stability correction (P55-e, #187)" begin
        c = BC.find_coupling(ref, :sgs_smagorinsky)
        @test c.closure isa StabilityCorrected                   # on by default
        @test c.closure.scheme isa BrownEtAl1994
        @test c.closure.ℓh ≈ 2.5^(2 / 3) && c.closure.ℓv ≈ 2.5^(-4 / 3)
        @test c.closure.qv === BC.tracer(ref, :qv)               # θ_v of the moist buoyancy
        @test BC.find_coupling(ref0, :sgs_smagorinsky).closure isa StaticSmagorinsky   # :none
        # the capping inversion of :smoke (θ lapse from 60 m), in the smoke
        # window (12 steps; by ~50 steps the resolved overturning has mixed
        # it away in θ_v, with or without the correction): the two levels
        # above 60 m (z = 65.6, 71.9 m) are stable and the correction cuts νv
        s12 = run_opts((;); nsteps = 12); s12_0 = run_opts((;); nsteps = 12, sgs_stability = :none)
        c12 = BC.find_coupling(s12, :sgs_smagorinsky).closure
        kinv = 11:12
        s1 = BC.stability_stats(c12, s12; levels = kinv)
        m = s12.mesh; sm = s12.geom.sm.mask
        fluid = [!sm[m.Hx+i, m.Hy+j, m.Hz+k] for i in 1:m.Nx, j in 1:m.Ny, k in kinv]
        νv0 = mean(view(interior_view(s12_0.sim.νt), :, :, kinv)[fluid])   # νv = νh uncorrected
        @info @sprintf("  stability correction, :smoke 12 steps: mean νv in the capping inversion (z = 65.6–71.9 m) %.3e m²/s corrected vs %.3e uncorrected; %.0f %% of its cells Ri ≥ Ri_c; mean κv/κh %.3f",
                       s1.νv, νv0, 100 * s1.supercritical, s1.κv_over_κh)
        @test s1.νv < 0.1 * νv0
        @test s1.supercritical > 0.3
        @test maximum(abs, interior_view(ref.sim.θ) .- interior_view(ref0.sim.θ)) > 1e-6
        @test all(isfinite, interior_view(ref.sim.θ))
        @test_throws ErrorException BC.build_case(BC.CaseConfig(:smoke; verbose = false,
                                                                sgs_stability = :bogus); io = devnull)
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
