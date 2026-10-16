# Era VI, Phase 54 (P54-b, #173): the Lagrangian-averaged dynamic
# Smagorinsky closure of Meneveau, Lund & Cabot (1996, JFM 319, 353), and
# the obstacle-aware (masked) dynamic procedure it shares with the plane /
# local averaging.
#
#   1. geometric relaxation of the memory toward a frozen source (fixed T),
#      and the MLC96 time scale T = θΔ(I_LM I_MM)^(-1/8);
#   2. pure pathline transport (ε = 0): the domain sum is conserved, the
#      (unwrapped) centroid moves with U t, the maximum never grows;
#   3. laminar shear u = S z: L_ij M_ij = 0, so C_s² decays to 0;
#   4. isotropic turbulence (32³, t = 1 and 3): domain-mean C_s against
#      the plane value (MLC96 fig. 6); CBC decay with the closure in
#      test/era6_cbc_decay.jl;
#   5. positivity over 200 steps with the closure driving the flow;
#   6. a wall-mounted cube in a periodic channel (mask);
#   7. the contract: pure coefficients, one commit per accepted step, the
#      memory of a rejected adaptive step rolled back.

using Test
using Aeolus
using Aeolus.Fields: interior_view, set!, Field
using Random: Xoshiro, randn, rand
using Statistics: mean
using Printf

const OPS = Aeolus.Operators

# A solenoidal random field with E(k) ∝ k^(-5/3) up to k_c, u_rms = 1 (the
# initialiser of P54-c; shared with era2_sgs_numerics.jl, #317).
isdefined(@__MODULE__, :kolmogorov_field!) || include(joinpath(@__DIR__, "support", "kolmogorov.jl"))

function _lag_plane_cs(sim; mask = nothing)
    ws = DynamicSmagorinskyWorkspace(sim.mesh)
    νt = Field(CellCenter(), sim.mesh)
    for f in (sim.u, sim.v, sim.w)
        fill_halos!(f, sim.bcs)
    end
    compute_dynamic_smagorinsky_νt!(νt, sim.u, sim.v, sim.w; workspace = ws, mask)
    return mean(sqrt.(ws.cs2)), ws, νt
end

# Every stored velocity sample on a face touching a solid set to `x`.
function _garbage_solid_faces!(sim, cellmask, x)
    for (f, loc) in ((sim.u, FaceX()), (sim.v, FaceY()), (sim.w, FaceZ()))
        fm = staggered_solid_mask(cellmask, loc)
        f.data[fm] .= x
    end
    return sim
end

_lag_state(s) = (copy(s.u.data), copy(s.v.data), copy(s.w.data), copy(s.θ.data))

@testset "Era VI — Lagrangian dynamic Smagorinsky (P54-b, #173)" begin

    @testset "1. geometric relaxation toward a frozen source, MLC96 time scale" begin
        g = RegularGrid(; size = (8, 6, 5), extent = (8.0, 6.0, 5.0), halo = (2, 2, 2))
        rng = Xoshiro(1)
        sz = (8, 6, 5)
        I0LM = rand(rng, sz...); I0MM = 1 .+ rand(rng, sz...)
        SLM = rand(rng, sz...); SMM = 1 .+ rand(rng, sz...)
        U = zeros(sz...)                                   # u = 0: no transport
        dt = 0.3; Tm = 2.0
        ε = (dt / Tm) / (1 + dt / Tm)
        ILM, IMM = copy(I0LM), copy(I0MM)
        nLM, nMM = similar(ILM), similar(IMM)
        per = (true, true, true)
        worst = 0.0
        for n in 1:40
            OPS._lagrangian_relax!(nLM, nMM, ILM, IMM, SLM, SMM, U, U, U, dt, g, 1.5, Tm, nothing, per)
            ILM .= nLM; IMM .= nMM
            eLM = abs.(ILM .- SLM) .- (1 - ε)^n .* abs.(I0LM .- SLM)
            eMM = abs.(IMM .- SMM) .- (1 - ε)^n .* abs.(I0MM .- SMM)
            worst = max(worst, maximum(abs, eLM), maximum(abs, eMM))
        end
        @test worst < 1e-12
        @printf("  [relax] 40 steps, ε = %.4f: max ||Iⁿ − S| − (1−ε)ⁿ|I⁰ − S|| = %.1e\n", ε, worst)

        # MLC96 eq. 2.13: T = θΔ(I_LM I_MM)^(-1/8), ε = (Δt/T)/(1 + Δt/T)
        OPS._lagrangian_relax!(nLM, nMM, I0LM, I0MM, SLM, SMM, U, U, U, dt, g, 1.5, nothing, nothing, per)
        Δ = (g.dx * g.dy * g.dz)^(1/3)
        T = 1.5 .* Δ .* (I0LM .* I0MM) .^ (-1 / 8)
        εv = (dt ./ T) ./ (1 .+ dt ./ T)
        @test nLM ≈ εv .* SLM .+ (1 .- εv) .* I0LM rtol = 1e-14
        @test nMM ≈ εv .* SMM .+ (1 .- εv) .* I0MM rtol = 1e-14
        # I_LM is clipped at 0
        OPS._lagrangian_relax!(nLM, nMM, I0LM, I0MM, .-10 .* SMM, SMM, U, U, U, dt, g, 1.5, Tm, nothing, per)
        @test all(==(0), nLM)
    end

    @testset "2. pathline transport (ε → 0): conservative, centroid moves with U t" begin
        N = 32
        g = RegularGrid(; size = (N, N, N), extent = (32.0, 32.0, 32.0), halo = (2, 2, 2))
        dt = 1.0
        a = (0.4, 0.27, 0.13)                      # cells per step; CFL 0.4
        U = fill(a[1] * g.dx / dt, N, N, N); V = fill(a[2] * g.dy / dt, N, N, N)
        W = fill(a[3] * g.dz / dt, N, N, N)
        x(i) = (i - 0.5) * g.dx
        c0 = (9.0, 11.0, 14.0); σ = 2.0
        ILM = [exp(-((x(i) - c0[1])^2 + (x(j) - c0[2])^2 + (x(k) - c0[3])^2) / (2σ^2))
               for i in 1:N, j in 1:N, k in 1:N]
        IMM = ones(N, N, N)
        Z = zeros(N, N, N)
        nLM, nMM = similar(ILM), similar(IMM)
        S0 = sum(ILM); M0 = maximum(ILM)
        # the sampled, truncated blob's own centroid (not exactly c0)
        cen0 = ntuple(d -> sum(x(I[d]) * ILM[I] for I in CartesianIndices(ILM)) / S0, 3)
        # Unwrapped first moment: each step, the mass in the last plane that
        # the shift carries across the periodic boundary (fraction a) re-
        # enters at x − L; adding L × (that mass) keeps the moment of the
        # unwrapped distribution.
        wrapped = [0.0, 0.0, 0.0]
        nsteps = 50
        maxgrow = -Inf
        for _ in 1:nsteps
            wrapped[1] += a[1] * sum(view(ILM, N, :, :))
            wrapped[2] += a[2] * sum(view(ILM, :, N, :))
            wrapped[3] += a[3] * sum(view(ILM, :, :, N))
            OPS._lagrangian_relax!(nLM, nMM, ILM, IMM, Z, Z, U, V, W, dt, g, 1.5, Inf, nothing,
                                   (true, true, true))
            maxgrow = max(maxgrow, maximum(nLM) - maximum(ILM))
            ILM .= nLM; IMM .= nMM
        end
        S = sum(ILM)
        @test abs(S - S0) / S0 < 1e-12
        cen = ntuple(d -> (sum(x(I[d]) * ILM[I] for I in CartesianIndices(ILM)) + 32.0 * wrapped[d]) / S, 3)
        err = maximum(abs.(cen .- (cen0 .+ a .* nsteps .* (g.dx, g.dy, g.dz))))
        @test err < 1e-10
        @test maxgrow <= 0 && maximum(ILM) <= M0
        # a uniform memory stays uniform (to the round-off of the weights,
        # whose sum is 1 only to an ulp)
        @test maximum(abs, IMM .- 1) < 1e-14
        @printf("  [transport] 50 steps CFL 0.4: |ΔΣ|/Σ = %.1e, centroid error %.1e m, max %.3f → %.3f\n",
                abs(S - S0) / S0, err, M0, maximum(ILM))

        # a departure point inside a solid takes the cell's own value; solid
        # corners of the stencil are dropped and the weights renormalised
        cell = falses(N + 4, N + 4, N + 4)
        cell[2 + 10, 2 + 5, 2 + 5] = true          # interior cell (10, 5, 5)
        A = Float64[i for i in 1:N, j in 1:N, k in 1:N]   # I = i
        per = (true, true, true)
        @test OPS._upstream(A, A, 10.2, 5.0, 5.0, 11, 5, 5, g, per, cell) == (11.0, 11.0)
        lm, _ = OPS._upstream(A, A, 10.6, 5.0, 5.0, 12, 5, 5, g, per, cell)
        @test lm == 11.0                           # corner 10 dropped: all weight on 11
        @test OPS._upstream(A, A, 10.6, 5.0, 5.0, 12, 5, 5, g, per, nothing)[1] ≈ 10.6
    end

    @testset "3. laminar shear u = S z: L_ij M_ij = 0, C_s² → 0" begin
        mw = RegularGrid(; size = (8, 8, 16), extent = (1.0, 1.0, 2.0), halo = (3, 3, 3))
        sw = BoussinesqSimulation(mw; bcs = FieldBoundaryConditions(:periodic, :periodic, :wall))
        set!(sw.u, (x, y, z) -> 3.0 * z)
        lag = LagrangianDynamicSmagorinsky(mw)
        ks = 3:mw.Nz-2                             # levels whose stencils miss the wall halos
        dt = 0.01
        cs_hist = Float64[]
        for n in 1:1500
            advance_sgs!(lag, sw, dt)
            OPS.sgs_viscosity!(sw.νt, lag, sw)
            push!(cs_hist, maximum(view(lag.workspace.cs2, :, :, ks)))
        end
        @test maximum(abs, view(lag.workspace.LM, :, :, ks)) == 0     # the Germano identity
        @test cs_hist[1] ≈ 0.16^2                                     # MLC96 start, C₀ = 0.16
        @test all(diff(cs_hist) .<= 0)                                # monotone decay
        @test cs_hist[end] < 1e-6 * 0.16^2
        compute_smagorinsky_νt!(sw.κt, sw.u, sw.v, sw.w, 0.16)        # (C₀Δ)²|S|
        r = maximum(interior_view(sw.νt)[:, :, ks] ./ interior_view(sw.κt)[:, :, ks])
        @test r < 1e-6
        @printf("  [shear] u = 3z, t = 15 s (S t = 45): C_s²/C₀² = %.1e, ν_t/ν_t(C₀) = %.1e (exactly 0 only for t → ∞: the memory relaxes)\n",
                cs_hist[end] / 0.16^2, r)
    end

    mesh = RegularGrid(; size = (32, 32, 32), extent = (2π, 2π, 2π), halo = (3, 3, 3))

    @testset "4. isotropic turbulence: C_s against the plane value (MLC96 fig. 6)" begin
        # The flow of era2_sgs_numerics.jl (static Smagorinsky 0.17 LES of a
        # k^(-5/3) field, dt = 0.01); the Lagrangian memory follows it
        # passively (advance_sgs! after every step), so both averages see
        # the same resolved field.
        sim = BoussinesqSimulation(mesh; C_s = 0.17, ν = 1e-4)
        kolmogorov_field!(sim, Xoshiro(1))
        lag = LagrangianDynamicSmagorinsky(mesh)
        out = Dict{Int,NTuple{2,Float64}}()
        for n in 1:300
            step!(sim, 0.01)
            advance_sgs!(lag, sim, 0.01)
            if n in (100, 300)
                OPS.sgs_viscosity!(Field(CellCenter(), mesh), lag, sim)
                out[n] = (mean(sqrt.(lag.workspace.cs2)), first(_lag_plane_cs(sim)))
            end
        end
        for (n, t) in ((100, 1), (300, 3))
            cl, cp = out[n]
            @printf("  [isotropic] t = %d: Lagrangian mean C_s %.4f, plane %.4f (%+.1f %%)\n",
                    t, cl, cp, 100 * (cl / cp - 1))
            @test abs(cl / cp - 1) < 0.15
        end
        @test 0.10 < out[100][1] < 0.20 && 0.10 < out[300][1] < 0.20
    end

    @testset "5. positivity with the closure driving the flow (200 steps)" begin
        sim = BoussinesqSimulation(mesh; ν = 1e-4)              # C_s = 0: only the closure
        kolmogorov_field!(sim, Xoshiro(2))
        lag = LagrangianDynamicSmagorinsky(mesh)
        ok_LM = ok_MM = ok_ν = true
        clipmax = 0.0
        for n in 1:200
            step!(sim, 0.01; sgs = lag)
            ok_LM &= minimum(lag.I_LM) >= 0
            ok_MM &= minimum(lag.I_MM) > 0
            ok_ν &= minimum(interior_view(sim.νt)) >= 0
            clipmax = max(clipmax, mean(lag.workspace.cs2 .>= lag.cs_max^2))
        end
        @test ok_LM && ok_MM && ok_ν
        @test clipmax < 0.01
        @test all(isfinite, sim.u.data) && maximum(interior_view(sim.νt)) > 0
        @printf("  [positivity] 200 steps: min I_LM %.2e, min I_MM %.2e, max clip fraction %.2e, I_LM = 0 in %.2f %% of cells, mean C_s %.4f\n",
                minimum(lag.I_LM), minimum(lag.I_MM), clipmax, 100 * mean(lag.I_LM .== 0),
                mean(sqrt.(lag.workspace.cs2)))
    end

    @testset "6. wall-mounted cube in a periodic channel (mask)" begin
        g = RegularGrid(; size = (32, 32, 16), extent = (32.0, 32.0, 16.0), halo = (3, 3, 3))
        bcs = FieldBoundaryConditions(:periodic, :periodic, :wall)
        sm = SolidMask(g)
        mark_solid!(sm, (x, y, z) -> 12 < x < 20 && 12 < y < 20 && z < 6)     # 8 × 8 × 6 cells
        sim = BoussinesqSimulation(g; ν = 1e-3, bcs)
        kolmogorov_field!(sim, Xoshiro(6))
        interior_view(sim.u) .+= 2.0                            # mean flow past the cube
        apply_solid_mask!(sim, sm)
        lag = LagrangianDynamicSmagorinsky(g; mask = sm.mask)
        H = (g.Hx, g.Hy, g.Hz)
        solid = view(sm.mask, H[1]+1:H[1]+g.Nx, H[2]+1:H[2]+g.Ny, H[3]+1:H[3]+g.Nz)
        # fluid cells sharing a face with the cube
        near = falses(size(solid))
        for I in CartesianIndices(solid)
            solid[I] && continue
            for d in 1:3, s in (-1, 1)
                J = Base.setindex(Tuple(I), I[d] + s, d)
                checkbounds(Bool, solid, J...) && solid[J...] && (near[I] = true)
            end
        end
        allfinite = true; νsolid = 0.0; nearok = true
        dt = 0.05
        for n in 1:200
            step!(sim, dt; sgs = lag)
            apply_solid_mask!(sim, sm)
            allfinite &= all(isfinite, sim.u.data) && all(isfinite, lag.I_LM) &&
                         all(isfinite, lag.I_MM) && all(isfinite, sim.νt.data)
            νsolid = max(νsolid, maximum(abs, interior_view(sim.νt)[solid]))
            cs2 = lag.workspace.cs2
            nearok &= all(c -> 0 <= c <= lag.cs_max^2, cs2[near])
        end
        @test allfinite
        @test νsolid == 0
        @test all(==(0), lag.I_LM[solid]) && all(==(0), lag.I_MM[solid])
        @test nearok
        csn = sqrt.(lag.workspace.cs2[near]); csf = sqrt.(lag.workspace.cs2[.!solid])
        @test maximum(interior_view(sim.νt)[near]) > 0
        @printf("  [cube] 200 steps: mean C_s next to the cube %.4f, all fluid %.4f; ν_t in solids %.1f\n",
                mean(csn), mean(csf), νsolid)

        # whatever is stored on the solid faces never reaches the result
        for f in (sim.u, sim.v, sim.w)
            fill_halos!(f, sim.bcs)
        end
        a = deepcopy(lag); b = deepcopy(lag)
        sa = deepcopy(sim); sb = deepcopy(sim)
        cellmask = OPS._sgs_mask(lag.mask, sim).cell
        for x in (NaN, 1e30)
            _garbage_solid_faces!(sb, cellmask, x)
            fa = compute_sgs!(a, sa); fb = compute_sgs!(b, sb)
            @test agree(fa.νh.data, fb.νh.data) && agree(a.workspace.cs2, b.workspace.cs2)
            advance_sgs!(a, sa, dt); advance_sgs!(b, sb, dt)
            @test a.I_LM == b.I_LM && a.I_MM == b.I_MM && a.Up == b.Up && a.Wp == b.Wp
            @test a.workspace.LM == b.workspace.LM && a.workspace.MM == b.workspace.MM
        end

        # the plane / local averaging take the same mask: solids excluded
        for f in (sim.u, sim.v, sim.w)
            fill_halos!(f, sim.bcs)
        end
        dmask = OPS._sgs_mask(lag.mask, sim)
        for av in (:plane, :local)
            wa = DynamicSmagorinskyWorkspace(g); wb = DynamicSmagorinskyWorkspace(g)
            na = Field(CellCenter(), g); nb = Field(CellCenter(), g)
            s2 = deepcopy(sim); _garbage_solid_faces!(s2, cellmask, NaN)
            compute_dynamic_smagorinsky_νt!(na, sim.u, sim.v, sim.w; workspace = wa, averaging = av, mask = dmask)
            compute_dynamic_smagorinsky_νt!(nb, s2.u, s2.v, s2.w; workspace = wb, averaging = av, mask = dmask)
            @test agree(na.data, nb.data) && agree(wa.cs2, wb.cs2)
            @test all(==(0), interior_view(na)[solid]) && all(==(0), wa.cs2[solid])
            @test all(c -> 0 <= c <= 0.25, wa.cs2)
        end
        # plane average over the fluid of a level == ⟨LM⟩/⟨MM⟩ over its fluid cells
        wa = DynamicSmagorinskyWorkspace(g); na = Field(CellCenter(), g)
        compute_dynamic_smagorinsky_νt!(na, sim.u, sim.v, sim.w; workspace = wa, mask = dmask)
        k = 3
        fl = .!solid[:, :, k]
        @test wa.cs2[findfirst(fl), k] ≈ clamp(sum(wa.LM[:, :, k][fl]) / sum(wa.MM[:, :, k][fl]), 0, 0.25)
        @test all(==(0), wa.LM[solid]) && all(==(0), wa.MM[solid])
    end

    @testset "7. contract: pure coefficients, one commit per step, adaptive rollback" begin
        function fresh(; seed = 7, umax)
            g = RegularGrid(; size = (16, 16, 16), extent = (1.0, 1.0, 1.0), halo = (3, 3, 3))
            s = BoussinesqSimulation(g; ν = 1e-5, θ₀ = 300.0)
            interior_view(s.θ) .= 300.0                         # neutral: no buoyancy
            kolmogorov_field!(s, Xoshiro(seed); kc = 4)
            for f in (s.u, s.v, s.w)
                interior_view(f) .*= umax
                fill_halos!(f, s.bcs)
            end
            return s
        end
        # coefficients are pure; RK3 commits once with the full step
        s = fresh(umax = 1.0); c = LagrangianDynamicSmagorinsky(s.mesh)
        step!(s, 1e-3; sgs = c)                                 # initialises the memory
        m0 = sgs_state(c)
        compute_sgs!(c, s); compute_sgs!(c, s)
        @test sgs_state(c) == m0
        ref = deepcopy(c)
        step_rk3!(s, 1e-3; sgs = c)
        advance_sgs!(ref, s, 1e-3)                              # one commit, on the new state
        @test values_agree(sgs_state(c), sgs_state(ref))

        # A rejected adaptive step leaves the memory of a direct short step
        # (the forced-retry set-up of era6_sgs_closure.jl).
        a = 1.0; h = 0.02
        push_x!(sim, Δt) = (interior_view(sim.tu) .+= a; nothing)
        k = CFLConstraints(dt_min = 1e-6, dt_max = 1.0)
        s1 = fresh(umax = 5e-4); c1 = LagrangianDynamicSmagorinsky(s1.mesh)
        s2 = fresh(umax = 5e-4); c2 = LagrangianDynamicSmagorinsky(s2.mesh)
        step!(s1, 1e-3; sgs = c1); step!(s2, 1e-3; sgs = c2)   # memories initialised
        @test values_agree(sgs_state(c1), sgs_state(c2))
        m_before = sgs_state(c1)
        # a weak flow (u_rms = 5e-4): after a step dt, max|u| ≈ a·dt ± 3e-3,
        # so dt = h (0.02) fails the cap 0.75·a·h and h/2 (0.01) passes it
        umax0 = maximum(f -> maximum(abs, interior_view(f)), (s1.u, s1.v, s1.w))
        @test umax0 < 3e-3
        cap = 0.75 * a * h
        _, used, nretry = step_adaptive!(s1, k; dt = h, sgs = c1, body_force! = push_x!,
                                         max_velocity = cap)
        _, used2, nretry2 = step_adaptive!(s2, k; dt = h / 2, sgs = c2, body_force! = push_x!,
                                           max_velocity = cap)
        @test nretry == 1 && used == h / 2 && nretry2 == 0 && used2 == h / 2
        @test values_agree(sgs_state(c1), sgs_state(c2))
        @test values_agree(_lag_state(s1), _lag_state(s2))
        @test sgs_state(c1)[1] != m_before[1]                  # the memory did advance
        @printf("  [contract] forced retry: %d retry, memory == direct dt/2 run \n", nretry)
    end
end
