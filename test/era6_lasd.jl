# Era VI, Phase 54 (P54-e, #186): the scale-dependent Lagrangian dynamic
# model (LASD) of Bou-Zeid, Meneveau & Parlange (2005, Phys. Fluids 17,
# 025105) — `LagrangianDynamicSmagorinsky(mesh; scale_dependent = true)`,
# one type with the scale-invariant model of #173 (LASI).
#
#   0. kernels: the separable 5-cell second test filter against a direct
#      5³ sum; Q^d_ij N_ij = 0 in laminar shear; the power law recovers a
#      prescribed C²(Δ) and β = 2^φ exactly; the evaluation cadence;
#   2. exact reduction: β forced to 1 is LASI (30 steps), and the state
#      contract (rollback, pure coefficient calls).
# The validation runs (scale invariance in isotropic turbulence, the
# half-channel log law, 500-step robustness) are in
# validation/era6_lasd.jl.

using Test
using Aeolus
using Aeolus.Fields: interior_view, set!, Field
using Random: Xoshiro

const LOPS = Aeolus.Operators

# the k^(-5/3) initial field of the SGS tests (era2_sgs_numerics.jl, #317)
isdefined(@__MODULE__, :kolmogorov_field!) || include(joinpath(@__DIR__, "support", "kolmogorov.jl"))

@testset "Era VI — scale-dependent Lagrangian dynamic model (P54-e, #186)" begin

    @testset "0a. second test filter: separable passes == direct 5³ box sum" begin
        g = RegularGrid(; size = (9, 8, 7), extent = (9.0, 4.0, 7.0), halo = (2, 2, 2))
        rng = Xoshiro(3)
        for (bcs, per) in ((FieldBoundaryConditions(:periodic, :periodic, :wall), (true, true, false)),
                           (FieldBoundaryConditions(:periodic, :periodic, :periodic), (true, true, true)))
            sim = BoussinesqSimulation(g; bcs)
            for f in (sim.u, sim.v, sim.w)
                interior_view(f) .= randn(rng, size(interior_view(f)))
            end
            interior_view(sim.w)[:, :, 1] .= 0
            for f in (sim.u, sim.v, sim.w); fill_halos!(f, bcs); end
            for masked in (false, true)
                cell = nothing; m = nothing
                if masked
                    sm = SolidMask(g); mark_solid!(sm, (x, y, z) -> 3 < x < 6 && y < 2 && z < 3)
                    m = LOPS._SGSMask(sm.mask, g, per); cell = m.cell
                end
                ws = DynamicSmagorinskyWorkspace(g)
                LOPS._germano_identity!(ws, sim.u.data, sim.v.data, sim.w.data, g, 9.0, m)
                sf = LOPS._SecondTestFilter(g)
                LOPS._second_test_filter!(sf, ws, g, 25.0, cell, per)
                # direct: for every interior cell, every point of the 5³ box
                # (shell index = interior + 1; periodic directions wrap on
                # the interior, bounded ones drop what is outside it)
                N = (g.Nx, g.Ny, g.Nz); H = (g.Hx, g.Hy, g.Hz)
                Δ2 = (g.dx * g.dy * g.dz)^(2/3)
                worst = 0.0
                for k in 1:N[3], j in 1:N[2], i in 1:N[1]
                    if cell !== nothing && cell[H[1]+i, H[2]+j, H[3]+k]
                        @test sf.QN[i,j,k] == 0 && sf.NN[i,j,k] == 0
                        continue
                    end
                    acc = zeros(21); n = 0
                    for oc in -2:2, ob in -2:2, oa in -2:2
                        sh = map((ijk, o, Nd, p) -> p ? mod1(ijk + o, Nd) + 1 : ijk + o + 1,
                                 (i, j, k), (oa, ob, oc), N, per)
                        all(d -> 2 <= sh[d] <= N[d] + 1, 1:3) || continue
                        cell !== nothing && cell[(H .- 1 .+ sh)...] && continue
                        acc .+= collect(LOPS._germano_terms(ws, sh...)); n += 1
                    end
                    qn, nn = LOPS._germano_LM_MM(Tuple(acc), 1 / n, Δ2, 25.0)
                    worst = max(worst, abs(qn - sf.QN[i,j,k]) / (abs(qn) + 1e-300 + 1e-3 * nn),
                                abs(nn - sf.NN[i,j,k]) / nn)
                end
                @test worst < 1e-12
            end
        end
    end

    @testset "0b. laminar shear u = S z: Q^d_ij N_ij = 0 (as L^d_ij M_ij)" begin
        mw = RegularGrid(; size = (8, 8, 16), extent = (1.0, 1.0, 2.0), halo = (2, 2, 2))
        sw = BoussinesqSimulation(mw; bcs = FieldBoundaryConditions(:periodic, :periodic, :wall))
        set!(sw.u, (x, y, z) -> 3.0 * z)
        c = LagrangianDynamicSmagorinsky(mw; scale_dependent = true)
        advance_sgs!(c, sw, 0.01)
        ks = 4:mw.Nz-3                        # boxes clear of the wall halos
        @test maximum(abs, view(c.second.QN, :, :, ks)) < 1e-12 * maximum(view(c.second.NN, :, :, ks))
        @test minimum(view(c.second.NN, :, :, ks)) > 0
        @test maximum(abs, view(c.workspace.LM, :, :, ks)) < 1e-12 * maximum(view(c.workspace.MM, :, :, ks))
    end

    @testset "0c. power law: prescribed C²(aΔ) = C²_Δ a^φ is recovered exactly" begin
        g = RegularGrid(; size = (4, 4, 4), extent = (4.0, 4.0, 4.0), halo = (2, 2, 2))
        sim = BoussinesqSimulation(g)
        set!(sim.u, (x, y, z) -> sin(x) * cos(z)); set!(sim.v, (x, y, z) -> cos(y) * z)
        for (α, α2) in ((3.0, 5.0), (2.0, 4.0))
            c = LagrangianDynamicSmagorinsky(g; scale_dependent = true, filter_ratio = α,
                                             filter_ratio2 = α2)
            c.initialised[] = true
            rng = Xoshiro(9)
            CΔ = 0.01 .+ 0.02 .* rand(rng, 4, 4, 4)
            φ = -2.5 .+ 5 .* rand(rng, 4, 4, 4)                     # inside the clip (|φ| < 3)
            c.I_MM .= 1 .+ rand(rng, 4, 4, 4); c.I_NN .= 2 .+ rand(rng, 4, 4, 4)
            c.I_LM .= CΔ .* α .^ φ .* c.I_MM
            c.I_QN .= CΔ .* α2 .^ φ .* c.I_NN
            LOPS.sgs_viscosity!(sim.νt, c, sim)
            @test c.beta ≈ 2 .^ φ rtol = 1e-12
            @test c.workspace.cs2 ≈ min.(CΔ, 0.25) rtol = 1e-12
            if α == 2                                               # Bou-Zeid's form
                C2 = c.I_LM ./ c.I_MM; C4 = c.I_QN ./ c.I_NN
                @test c.beta ≈ C4 ./ C2 rtol = 1e-12
                @test c.workspace.cs2 ≈ C2 ./ (C4 ./ C2) rtol = 1e-12
            end
            # the clip: β ∈ [1/8, 8]; C²(α₂Δ) = 0 gives β = 1/8; C²(αΔ) = 0 gives C² = 0
            c.I_QN[1] = 0; c.I_LM[2] = 0; c.I_QN[3] = c.I_LM[3] / c.I_MM[3] * c.I_NN[3] * 1e-6
            LOPS.sgs_viscosity!(sim.νt, c, sim)
            @test c.beta[1] == 0.125 && c.beta[3] == 0.125
            @test c.workspace.cs2[2] == 0 && c.beta[2] == 1
            @test c.workspace.cs2[3] ≈ min(c.I_LM[3] / c.I_MM[3] * α^3, 0.25)
        end
        @test_throws ArgumentError LagrangianDynamicSmagorinsky(g; scale_dependent = true, filter_ratio2 = 3)
        @test_throws ArgumentError LagrangianDynamicSmagorinsky(g; beta = 1.0)
    end

    @testset "0d. cadence: transport between evaluations, ε over the elapsed time" begin
        g = RegularGrid(; size = (12, 10, 8), extent = (2π, 2.0, 1.0), halo = (2, 2, 2))
        sim = BoussinesqSimulation(g; bcs = FieldBoundaryConditions(:periodic, :periodic, :wall))
        kolmogorov_field!(sim, Xoshiro(5); kc = 3)
        interior_view(sim.u) .+= 1.0
        for f in (sim.u, sim.v, sim.w); fill_halos!(f, sim.bcs); end
        dt = 0.01; N = 4
        c = LagrangianDynamicSmagorinsky(g; scale_dependent = true, cadence = N)
        ref = LagrangianDynamicSmagorinsky(g; scale_dependent = true)
        advance_sgs!(c, sim, dt); advance_sgs!(ref, sim, dt)        # initialisation
        @test values_agree(sgs_state(c)[1:8], sgs_state(ref)[1:8])
        per = (true, true, false)
        z = zeros(size(c.I_LM))
        for n in 1:N-1                                           # transport only
            L, M, Q, R = copy(c.I_LM), copy(c.I_MM), copy(c.I_QN), copy(c.I_NN)
            a1, a2, b1, b2 = similar(L), similar(L), similar(L), similar(L)
            advance_sgs!(c, sim, dt)
            LOPS._lagrangian_relax!(a1, a2, L, M, z, z, c.Up, c.Vp, c.Wp, dt, g, 1.5, Inf, nothing, per)
            LOPS._lagrangian_relax!(b1, b2, Q, R, z, z, c.Up, c.Vp, c.Wp, dt, g, 1.5, Inf, nothing, per)
            @test c.I_LM == a1 && c.I_MM == a2 && c.I_QN == b1 && c.I_NN == b2
            @test c.since[] == n && c.acc_dt[] ≈ n * dt
        end
        # the evaluation commit: sources at the new state, ε from N dt
        L, M, Q, R = copy(c.I_LM), copy(c.I_MM), copy(c.I_QN), copy(c.I_NN)
        U, V, W = copy(c.Up), copy(c.Vp), copy(c.Wp)
        acc = c.acc_dt[] + dt
        advance_sgs!(c, sim, dt)
        a1, a2, b1, b2 = similar(L), similar(L), similar(L), similar(L)
        LOPS._lagrangian_relax!(a1, a2, L, M, c.workspace.LM, c.workspace.MM, U, V, W, dt, g, 1.5,
                                nothing, nothing, per, acc)
        LOPS._lagrangian_relax!(b1, b2, Q, R, c.second.QN, c.second.NN, U, V, W, dt, g, 1.5,
                                nothing, nothing, per, acc)
        @test c.I_LM == a1 && c.I_MM == a2 && c.I_QN == b1 && c.I_NN == b2
        @test c.since[] == 0 && c.acc_dt[] == 0
        # the counters are part of the state a rejected step restores
        s0 = sgs_state(c); advance_sgs!(c, sim, dt)
        @test c.since[] == 1
        restore_sgs_state!(c, s0)
        @test sgs_state(c) == s0 && c.since[] == 0
        @test_throws ArgumentError LagrangianDynamicSmagorinsky(g; cadence = 0)
    end

    mesh = RegularGrid(; size = (32, 32, 32), extent = (2π, 2π, 2π), halo = (3, 3, 3))

    @testset "2. exact reduction: β ≡ 1 is LASI" begin
        sa = BoussinesqSimulation(mesh; ν = 1e-4)
        kolmogorov_field!(sa, Xoshiro(4))
        sb = deepcopy(sa)
        a = LagrangianDynamicSmagorinsky(mesh)
        b = LagrangianDynamicSmagorinsky(mesh; scale_dependent = true, beta = 1.0)
        for n in 1:30
            step_rk3!(sa, 0.01; sgs = a); step_rk3!(sb, 0.01; sgs = b)
        end
        @test agree(sa.u.data, sb.u.data) && agree(sa.w.data, sb.w.data)
        @test agree(sa.νt.data, sb.νt.data)
        @test values_agree(a.workspace.cs2, b.workspace.cs2)
        @test values_agree(a.I_LM, b.I_LM) && values_agree(a.I_MM, b.I_MM)
        @test all(==(1), b.beta)
        @test maximum(b.I_QN) > 0                  # the second memory did run
        # contract: the second memory is part of the state a rejected step
        # restores, and coefficient calls are pure
        d = LagrangianDynamicSmagorinsky(mesh; scale_dependent = true)
        step!(sb, 0.01; sgs = d)
        m0 = sgs_state(d)
        compute_sgs!(d, sb); compute_sgs!(d, sb)
        @test sgs_state(d) == m0
        advance_sgs!(d, sb, 0.01)
        @test sgs_state(d)[7] != m0[7] && sgs_state(d)[8] != m0[8]
        restore_sgs_state!(d, m0)
        @test sgs_state(d) == m0
    end
end
