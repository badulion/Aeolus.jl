# Numerics swap-ins of the Berlin case (#132): the pressure-solver, SGS,
# momentum- and scalar-advection slots, in the :smoke window.
#
#   * the multigrid-preconditioned CG pressure solve gives the cosine-FFT
#     result to 1e-8 after 100 steps and keeps the flow divergence-free;
#   * MLSGS with the Smagorinsky closure reproduces the `:smagorinsky` run
#     (the default provider before #182) bit for bit (the eddy_viscosity! hook of step! adds its terms where the
#     built-in closure does);
#   * the σ-model provider (`sgs = :sigma`, #184): ν_t ≥ 0, exactly 0 on
#     the laminar version of the current flow and in a laminar parallel
#     flow on the Berlin mesh, where Smagorinsky is on; below 1e-2 ν in the
#     (volume-γ laminar) free stream above the sponge start (P59-e, #195);
#   * the SGS option scalar_tensor = :isentropic (P55-c, #181): with the
#     closure's own κv (= κh) the tensor is κh I and the run is the default
#     one to round-off; with diapycnal = 0.1 it rotates and changes θ;
#   * the SGS stability correction (P55-e, #187) is on by default; it cuts
#     νv in the capping inversion; `sgs_stability = :none` is the bare closure;
#   * configuration errors (halo too small for a scheme, a step! keyword
#     claimed by two couplings) are caught.

using Test
using Aeolus
using Aeolus.Fields: interior_view, Field
using Printf
using Statistics: mean

isdefined(Main, :BerlinCase) ||
    include(joinpath(@__DIR__, "..", "examples", "urban", "BerlinCase.jl"))
const BC = Main.BerlinCase

# The runs of this file are the periodic box the quantitative checks were
# measured in: the nest (#265) is switched off (`:nesting => false`). The
# nested default runs every numerics provider in the smoke matrix.
function _run_smoke(nsteps; physics...)
    cfg = BC.CaseConfig(:smoke; verbose = false, nsteps = nsteps,
                        physics = Dict{Symbol,Any}(:nesting => false, physics...))
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

    # runs on the :smagorinsky provider (the slot default is :tke since #182)
    function run_opts(o; kw...)
        cfg = BC.CaseConfig(:smoke; verbose = false,
                            physics = Dict{Symbol,Any}(:sgs => :smagorinsky, :nesting => false),
                            options = Dict{Symbol,Any}(:sgs_smagorinsky => o), (; nsteps = N, kw...)...)
        case = BC.build_case(cfg; io = devnull)
        BC.run_case!(case)
        return case
    end
    # the uncorrected closure (sgs_stability = :none, P55-e): νh = νv, κh = κv
    ref0 = run_opts((;); sgs_stability = :none)

    @testset "σ-model provider (#184): ν_t ≥ 0, zero in laminar flow" begin
        sg = _run_smoke(N; sgs = :sigma)
        c = BC.find_coupling(sg, :sgs_sigma)
        @test c !== nothing && c.closure isa SigmaModel
        @test get(sg.step_kwargs, :sgs_stress, nothing) === :symmetric   # like every eddy-viscosity provider (#175)
        sim, m = sg.sim, sg.mesh
        @test minimum(interior_view(sim.νt)) >= 0
        @test all(isfinite, interior_view(sim.u))
        p = BC.sigma_νt_profile(c, sg)
        @test p.min_σ >= 0
        @test p.canopy_σ > 0
        for f in (sim.u, sim.v, sim.w)
            fill_halos!(f, sim.bcs)
        end
        lam = falses(m.Nx, m.Ny, m.Nz)
        laminar_mask!(lam, sim.u, sim.v, sim.w)
        γfree = 1 - mean(lam[:, :, p.ksp:end])
        @info @sprintf("  σ-model after %d steps: mean ν_t canopy %.3e (Smagorinsky on the same flow %.3e), above the sponge start %.3e (%.3e; ratio %.2f, max %.2e) m²/s; γ there %.3f",
                       N, p.canopy_σ, p.canopy_smag, p.free_σ, p.free_smag, p.free_σ / p.free_smag, p.free_σ_max, γfree)
        # Before the merge with #171 this testset asserted γ = 0 above the
        # sponge start and σ's ν_t ≤ 5 % of Smagorinsky's there in the
        # default run (2.6 % measured). That no longer describes the default
        # flow: #171 retargets the sponge at the geostrophic wind (U_g, V_g)
        # instead of zero, so the region above the sponge start is now a
        # nearly uniform stream (top level ū = 1.00 m/s against 0.05 m/s
        # before, little mean shear) carrying weak 3-D residual
        # fluctuations. The level-normalised γ is 0.96 there and the
        # σ-model, which only discounts *laminar* gradients, gives 1.7 ×
        # Smagorinsky's (small, 2.3e-3 m²/s) ν_t. The old assertion is kept
        # on the flow it was written for (the sponge relaxing to zero,
        # `sponge_to_geostrophic = false`: a laminar sheared layer), and the
        # laminar-consistency property is asserted exactly on the default
        # flow's laminar version (level means of u, v; w = 0), the check AMD
        # and Vreman share:
        ν, cells = BC.laminar_reference_νt(sg, c.closure)
        νs, _ = BC.laminar_reference_νt(sg, sg.cfg.C_s)
        @test count(cells) > 0
        @test all(==(0), ν[cells]) && maximum(νs[cells]) > 0
        sz = BC.build_case(BC.CaseConfig(:smoke; verbose = false, nsteps = N,
                                         physics = Dict{Symbol,Any}(:sgs => :sigma, :nesting => false),
                                         options = Dict{Symbol,Any}(:geostrophic => (sponge_to_geostrophic = false,)));
                           io = devnull)
        BC.run_case!(sz)
        pz = BC.sigma_νt_profile(BC.find_coupling(sz, :sgs_sigma), sz)
        foreach(f -> fill_halos!(f, sz.sim.bcs), (sz.sim.u, sz.sim.v, sz.sim.w))
        lz = falses(m.Nx, m.Ny, m.Nz)
        laminar_mask!(lz, sz.sim.u, sz.sim.v, sz.sim.w)
        γz = 1 - mean(lz[:, :, pz.ksp:end])
        @info @sprintf("  σ-model, sponge relaxing to zero: above the sponge start %.3e (Smagorinsky %.3e; ratio %.3f) m²/s; γ there %.3f",
                       pz.free_σ, pz.free_smag, pz.free_σ / pz.free_smag, γz)
        @test γz == 0                                  # a laminar sheared layer …
        @test pz.free_σ <= 0.05 * pz.free_smag         # … where σ stays far below Smagorinsky
        # The relaminarisation view of the same default free stream (P59-e,
        # #195): it is laminar against the domain-mean vorticity
        # (volume-normalised γ, the Transition setting for a quiet free
        # stream — the plane-normalised γ above is scale-free and counts the
        # weak 3-D residual disturbances, σ_w 2–5 mm/s, as turbulent), and
        # σ's ν_t there is negligible against the background ν (the
        # relaminarisation criterion ν_t/ν < 1e-2).
        laminar_mask!(lam, sim.u, sim.v, sim.w; normalization = :volume)
        γfree_vol = 1 - mean(lam[:, :, p.ksp:end])
        @info @sprintf("  σ-model, default free stream: ν_t %.3e m²/s = %.1e ν; γ there %.3f (plane), %.3f (volume)",
                       p.free_σ, p.free_σ / sg.cfg.ν, γfree, γfree_vol)
        @test γfree_vol < 0.1
        @test p.free_σ / sg.cfg.ν < 1e-2
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
        # (on a provider that is stability-corrected: the slot default :tke
        # since #182 is not wrapped and does not read sgs_stability)
        @test_throws ErrorException BC.build_case(BC.CaseConfig(:smoke; verbose = false,
                                                                physics = Dict{Symbol,Any}(:sgs => :smagorinsky),
                                                                sgs_stability = :bogus); io = devnull)
        # the tracers (#177) get θ's diagonal diffusivity: κh = sim.κt on the
        # x, y faces and the corrected κv on the z faces
        tq = only(t for t in ref.tracers if t.name === :qv)
        @test BC.tracer_diffusivity(tq, ref) === ref.sim.κt
        @test BC.tracer_vertical_diffusivity(tq, ref) === c.closure.κv
        tq0 = only(t for t in ref0.tracers if t.name === :qv)
        @test BC.tracer_vertical_diffusivity(tq0, ref0) === ref0.sim.κt      # :none: κv = κh
        d = BC.find_coupling(ref, :scalar_advection).diffusion
        q = deepcopy(BC.tracer(ref, :qv)); q0 = copy(interior_view(q))
        BC.sgs_diffuse!(d, ref, q, 2.0; κv = c.closure.κv)
        tt = Field(CellCenter(), ref.mesh); qq = deepcopy(q); interior_view(qq) .= q0
        fill_halos!(qq, d.bcs)
        add_eddy_diffusion_scalar!(tt, qq, ref.sim.κt, c.closure.κv; mask = d.mask)
        @test interior_view(q) == q0 .+ 2.0 .* interior_view(tt)            # bit for bit
        q1 = deepcopy(BC.tracer(ref, :qv)); BC.sgs_diffuse!(d, ref, q1, 2.0)  # isotropic κh
        dv, dh = interior_view(q) .- q0, interior_view(q1) .- q0
        @info @sprintf("  tracer SGS diffusion (2 s, qv) with the corrected κv: max|Δq| %.2e vs %.2e with κv = κh",
                       maximum(abs, dv), maximum(abs, dh))
        @test dv != dh
        @test all(isfinite, dv)
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

    @testset "configuration errors" begin
        # WENO-Z7 needs a halo of 4; the case default is 3. Nested (the
        # default, #265), the nest faces refuse the mesh first (an open face
        # needs the scheme's halo too); with the nest off, the transport does
        cfg = BC.CaseConfig(:smoke; verbose = false,
                            options = Dict{Symbol,Any}(:scalar_advection => (scheme = :weno_z7,)))
        @test_throws ArgumentError BC.build_case(cfg; io = devnull)
        cfg = BC.CaseConfig(:smoke; verbose = false, physics = Dict{Symbol,Any}(:nesting => false),
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
