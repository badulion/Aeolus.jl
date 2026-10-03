# Numerics swap-ins of the Berlin case (#132): the pressure-solver, SGS,
# momentum- and scalar-advection slots, in the :smoke window.
#
#   * the multigrid-preconditioned CG pressure solve gives the cosine-FFT
#     result to 1e-8 after 100 steps and keeps the flow divergence-free;
#   * MLSGS with the Smagorinsky closure reproduces the `:smagorinsky` run
#     (the default provider before #182) bit for bit (the eddy_viscosity! hook of step! adds its terms where the
#     built-in closure does);
#   * the SGS option scalar_tensor = :isentropic (P55-c, #181): with the
#     closure's own κv (= κh) the tensor is κh I and the run is the default
#     one to round-off; with diapycnal = 0.1 it rotates and changes θ;
#   * configuration errors (halo too small for a scheme, a step! keyword
#     claimed by two couplings) are caught.

using Test
using Aeolus
using Aeolus.Fields: interior_view, Field
using Printf

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
    # the Smagorinsky provider (the default before #182): the MLSGS check
    # below is its bit-for-bit equivalence, and the MG-CG numbers were
    # measured with it; the default :tke provider is run by the smoke test
    ref = _run_smoke(N; sgs = :smagorinsky)
    @test ref.sim.solver isa CosineFFTPoissonSolver

    @testset "MG-preconditioned CG: FFT result to 1e-8 after $N steps, divergence-free" begin
        mg = _run_smoke(N; pressure_solver = :mg_cg, sgs = :smagorinsky)
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

    @testset "SGS diffusion of the tracers with κt, mask-aware (#177)" begin
        # The :transport stage diffuses every advected tracer with sim.κt of the
        # last dynamics step through the src kernel, with the solid mask.
        sa = BC.find_coupling(ref, :scalar_advection)
        d = sa.diffusion
        @test d isa BC.TracerSGSDiffusion && d.weights === nothing   # Boussinesq core
        m = ref.mesh
        @test maximum(interior_view(ref.sim.κt)) > 0 && minimum(interior_view(ref.sim.κt)) >= 0
        # one diffusion step of qv from the state after N steps
        q = deepcopy(BC.tracer(ref, :qv))
        q0 = copy(interior_view(q))
        solid = view(ref.geom.sm.mask, m.Hx+1:m.Hx+m.Nx, m.Hy+1:m.Hy+m.Ny, m.Hz+1:m.Hz+m.Nz)
        fluid = .!solid
        @test count(solid) > 0
        BC.sgs_diffuse!(d, ref, q, 2.0)
        dq = interior_view(q) .- q0
        # one source of truth: the increment is dt × the src kernel with the mask
        t = Field(CellCenter(), m); qq = deepcopy(q); interior_view(qq) .= q0
        fill_halos!(qq, d.bcs)
        add_eddy_diffusion_scalar!(t, qq, ref.sim.κt; mask = d.mask)
        @test interior_view(q) == q0 .+ 2.0 .* interior_view(t)     # bit for bit
        @test maximum(abs, dq) > 0                                  # it does something
        @test all(iszero, dq[solid])                                # buildings untouched
        @test abs(sum(dq[fluid])) <= 1e-12 * sum(q0[fluid])         # Σ q V conserved
        # the fluctuation about each level's fluid mean decays
        var(x) = sum(k -> (v = x[:, :, k][fluid[:, :, k]]; isempty(v) ? 0.0 :
                          sum(abs2, v .- sum(v) / length(v))), 1:m.Nz)
        @test var(interior_view(q)) < var(q0)
        # the unmasked kernel would exchange vapour with the building interiors
        tu = Field(CellCenter(), m)
        add_eddy_diffusion_scalar!(tu, qq, ref.sim.κt)
        leak = 2.0 * sum(interior_view(tu)[fluid])
        @info @sprintf("  SGS tracer diffusion (2 s, qv): max|Δq| %.2e kg/kg, Σ_fluid Δq/Σ_fluid q %.1e masked vs %.1e unmasked (vapour from the building interiors)",
                       maximum(abs, dq), sum(dq[fluid]) / sum(q0[fluid]), leak / sum(q0[fluid]))
        @test abs(leak) > 1e3 * abs(sum(dq[fluid]))
        # off switch: the pre-#177 transport (no tracer diffusion)
        off = BC.build_case(BC.CaseConfig(:smoke; verbose = false, nsteps = 1,
                                          options = Dict{Symbol,Any}(:scalar_advection => (sgs_diffusion = false,)));
                            io = devnull)
        @test BC.find_coupling(off, :scalar_advection).diffusion === nothing
        @test BC.tracer_sgs_diffusion(off) === nothing
    end

    @testset "SGS scalar_tensor = :isentropic (P55-c, #181)" begin
        function run_opts(o)
            cfg = BC.CaseConfig(:smoke; verbose = false, nsteps = N,
                                physics = Dict{Symbol,Any}(:sgs => :smagorinsky),
                                options = Dict{Symbol,Any}(:sgs_smagorinsky => o))
            case = BC.build_case(cfg; io = devnull)
            BC.run_case!(case)
            return case
        end
        iso = run_opts((scalar_tensor = :isentropic,))
        c = BC.find_coupling(iso, :sgs_smagorinsky)
        @test c.closure isa IsentropicDiffusion
        # κv = κh: K = κh I, the triad operator is the diagonal one to round-off
        dθ = maximum(abs, interior_view(iso.sim.θ) .- interior_view(ref.sim.θ))
        rot = run_opts((scalar_tensor = :isentropic, diapycnal = 0.1))
        dθr = maximum(abs, interior_view(rot.sim.θ) .- interior_view(ref.sim.θ))
        fr = BC.isentropic_fraction(BC.find_coupling(rot, :sgs_smagorinsky).closure)
        @info @sprintf("  isentropic scalar tensor after %d steps: max|Δθ| vs default %.1e K (κv = κh), %.1e K (κv = 0.1 κh, %.1f %% of the cells rotated)",
                       N, dθ, dθr, 100fr)
        @test dθ < 1e-9
        @test dθr > 1e-6
        @test 0 < fr < 1
        @test all(isfinite, interior_view(rot.sim.θ))
        @test_throws ArgumentError run_opts((scalar_tensor = :bogus,))
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
