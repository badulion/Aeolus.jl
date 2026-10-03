# P59-a (#172) — NoSlipWall / moving-wall tangential velocity BC and the
# laminar wall-flow verification suite.
#
# What is checked against what (grids (4, 4, Nz), horizontally uniform
# flows, so every check is a 1-D analytic solution of the momentum
# equation run through the full 3-D fractional-step driver):
#   * the halo itself: antisymmetric about the wall value for u, v, so the
#     wall-face average is u_wall; w and scalars exactly as ImpermeableWall;
#   * Couette flow (Re = 100): linear profile and wall stress νU/h;
#   * plane Poiseuille flow: the parabola, its bulk velocity Gh²/(12ν) and
#     the measured order of convergence (Pope 2000, §7.1.1);
#   * Stokes' first problem: U erfc(z / 2√(νt)), with GasOptics.erfc_series;
#   * Stokes' second problem: decay length √(2ν/ω) (Schlichting 1979, §V.7);
#   * the plane-integrated momentum budget of a 3-D random flow;
#   * regression: `:wall` keeps its original code path bit for bit, and
#     IMEX with :noslip matches the explicit steady state.

using Test, Printf, Random
using Aeolus
using Aeolus.Fields: interior_view
using Aeolus.BoundaryConditions: _fill_z!, _fill_wall_bottom!, _fill_wall_top!,
                                 is_wall, homogeneous_bcs
using Aeolus.GasOptics: erfc_series

const _W_SRC = pathof(Aeolus)

wall_mesh(Nz; Lz = 1.0) = RegularGrid(size = (4, 4, Nz), extent = (1.0, 1.0, Lz))
wall_zc(m) = [(k - 0.5) * m.dz for k in 1:m.Nz]
plane_mean(f) = vec(sum(interior_view(f); dims = (1, 2))) ./ (f.mesh.Nx * f.mesh.Ny)
function set_profile!(f, prof)
    for k in eachindex(prof)
        interior_view(f)[:, :, k] .= prof[k]
    end
    return f
end

@testset "P59-a NoSlipWall (#172)" begin

    @testset "constructors and symbols" begin
        @test NoSlipWall() == NoSlipWall(0.0, 0.0)
        @test NoSlipWall(u_wall = 2) == NoSlipWall(2.0, 0.0)
        @test NoSlipWall(1, v_wall = 3).v_wall === 3.0
        b = FieldBoundaryConditions(:periodic, :periodic, :noslip)
        @test b.bottom isa NoSlipWall && b.top isa NoSlipWall && b.west isa Periodic
        oc = FieldBoundaryConditions(:periodic, :periodic; bottom = :noslip, top = :wall)
        @test oc.bottom isa NoSlipWall && oc.top isa ImpermeableWall
        lid = FieldBoundaryConditions(:periodic, :periodic; bottom = :noslip,
                                      top = NoSlipWall(1.5, 0))
        @test lid.bottom == NoSlipWall() && lid.top.u_wall == 1.5
        @test_throws ArgumentError FieldBoundaryConditions(:noslip)          # x, y: z only
        @test_throws ArgumentError FieldBoundaryConditions(:periodic, :periodic;
                                                           bottom = :noslip, top = :periodic)
        @test is_wall(NoSlipWall()) && is_wall(ImpermeableWall()) && !is_wall(Periodic())
        # the nest helper understands the symbol too
        @test Aeolus.AdaptiveMesh.nest_bcs(:periodic, :periodic, :noslip).bottom isa NoSlipWall
    end

    @testset "halo: antisymmetric about the wall velocity" begin
        m = wall_mesh(6)
        Ub, Ut, Vt = 0.7, -1.3, 2.1
        bcs = FieldBoundaryConditions(Periodic(), Periodic(), Periodic(), Periodic(),
                                      NoSlipWall(Ub, 0), NoSlipWall(Ut, Vt))
        rng = MersenneTwister(1)
        u = Field(FaceX(), m); v = Field(FaceY(), m)
        interior_view(u) .= randn(rng, size(interior_view(u)))
        interior_view(v) .= randn(rng, size(interior_view(v)))
        fill_halos!(u, bcs); fill_halos!(v, bcs)
        H, N = m.Hz, m.Nz
        # the wall-face average of first interior value and ghost is the wall value
        @test all(0.5 .* (u.data[:, :, H] .+ u.data[:, :, H + 1]) .≈ Ub)
        @test all(0.5 .* (u.data[:, :, H + N] .+ u.data[:, :, H + N + 1]) .≈ Ut)
        @test all(0.5 .* (v.data[:, :, H] .+ v.data[:, :, H + 1]) .≈ 0)
        @test all(0.5 .* (v.data[:, :, H + N] .+ v.data[:, :, H + N + 1]) .≈ Vt)
        # deeper halo: odd reflection about the wall value (a linear profile
        # through the wall is extended exactly)
        lin = Field(FaceX(), m)
        set_profile!(lin, Ub .+ 3.0 .* wall_zc(m))
        fill_halos!(lin, FieldBoundaryConditions(:periodic, :periodic; bottom = NoSlipWall(Ub, 0),
                                                 top = :wall))
        for h in 1:H
            @test lin.data[3, 3, H + 1 - h] ≈ Ub + 3.0 * (0.5 - h) * m.dz atol = 1e-14
        end

        # w and cell-centred fields: exactly the ImpermeableWall halo (bitwise)
        free = FieldBoundaryConditions(:periodic, :periodic, :wall)
        for L in (FaceZ(), CellCenter())
            a = Field(L, m); b = Field(L, m)
            interior_view(a) .= randn(rng, size(interior_view(a)))
            b.data .= a.data
            fill_halos!(a, bcs); fill_halos!(b, free)
            @test a.data == b.data
        end
    end

    @testset "regression: the `:wall` halo keeps its own code path" begin
        # (ImpermeableWall, ImpermeableWall) still dispatches to the original
        # methods in BoundaryConditions.jl, not to the generic wall-pair method.
        for L in (FaceX, FaceY, FaceZ, CellCenter)
            meth = which(_fill_z!, Tuple{Array{Float64,3}, ImpermeableWall, ImpermeableWall,
                                         Int, Int, L})
            @test endswith(string(meth.file), "BoundaryConditions.jl")
        end
        # and the per-side helpers used for mixed pairs reproduce it bit for bit
        m = wall_mesh(5)
        rng = MersenneTwister(2)
        for L in (FaceX(), FaceY(), FaceZ(), CellCenter())
            a = Field(L, m); interior_view(a) .= randn(rng, size(interior_view(a)))
            b = copy(a.data)
            _fill_z!(a.data, ImpermeableWall(), ImpermeableWall(), m.Hz, m.Nz, L)
            _fill_wall_bottom!(b, ImpermeableWall(), m.Hz, L)
            _fill_wall_top!(b, ImpermeableWall(), m.Hz, m.Nz, L)
            @test a.data == b
        end
        @test isempty(Test.detect_ambiguities(Aeolus.BoundaryConditions; recursive = false))
        free = FieldBoundaryConditions(:periodic, :periodic, :wall)
        @test homogeneous_bcs(free) === free
        @test homogeneous_bcs(FieldBoundaryConditions(:periodic, :periodic, :noslip)) isa
              FieldBoundaryConditions
    end

    # ---------------------------------------------------------------------
    # Laminar wall flows. Explicit step!, dt = 0.3 Δz²/ν (diffusive limit
    # 0.5); every flow is horizontally uniform, so the pressure correction
    # is identically zero and the column obeys ∂u/∂t = ν ∂²u/∂z² + G.
    # ---------------------------------------------------------------------

    @testset "Couette, Re = 100" begin
        U, h, Re = 1.0, 1.0, 100
        ν = U * h / Re
        m = wall_mesh(16; Lz = h)
        bcs = FieldBoundaryConditions(:periodic, :periodic; bottom = :noslip,
                                      top = NoSlipWall(U, 0))
        sim = BoussinesqSimulation(m; ν = ν, bcs = bcs)
        dt = 0.3 * m.dz^2 / ν
        for _ in 1:ceil(Int, 4h^2 / ν / dt)        # from rest; slowest mode e^{-π²νt/h²} ≈ 1e-17
            step!(sim, dt)
        end
        err = maximum(abs.(plane_mean(sim.u) .- U .* wall_zc(m) ./ h))
        τb = wall_shear_stress(sim; side = :bottom)
        τt = wall_shear_stress(sim; side = :top)
        @printf("    Couette Re=100: ‖u − Uz/h‖∞ = %.2e U, τ_b = %.15f (νU/h = %.15f), τ_t − τ_b = %.1e\n",
                err, τb, ν * U / h, τt - τb)
        @test err < 1e-10 * U
        @test abs(τb - ν * U / h) < 1e-10 * (ν * U / h)
        @test abs(τt - ν * U / h) < 1e-10 * (ν * U / h)   # uniform stress across the gap
        @test maximum(abs, interior_view(sim.w)) < 1e-12
        @test wall_shear_stress(sim; component = :v) == 0
        @test_throws ArgumentError wall_shear_stress(sim; side = :west)
        psim = BoussinesqSimulation(m; ν = ν)                  # triply periodic
        @test_throws ArgumentError wall_shear_stress(psim)
        # a free-slip wall carries no stress
        fsim = BoussinesqSimulation(m; ν = ν, bcs = FieldBoundaryConditions(:periodic,
                                    :periodic; bottom = NoSlipWall(U, 0), top = :wall))
        set_profile!(fsim.u, 0.3 .+ wall_zc(m))
        @test wall_shear_stress(fsim; side = :top) == 0
        @test wall_shear_stress(fsim; side = :bottom) ≈ ν * (2 * (0.3 + m.dz / 2 - U) / m.dz)
    end

    @testset "plane Poiseuille: second order, bulk velocity Gh²/(12ν)" begin
        G, ν, h = 1.0, 1.0, 1.0
        drive!(s, Δt) = (interior_view(s.tu) .+= G; nothing)
        res = map((16, 32, 64)) do Nz
            m = wall_mesh(Nz; Lz = h)
            sim = BoussinesqSimulation(m; ν = ν,
                                       bcs = FieldBoundaryConditions(:periodic, :periodic, :noslip))
            z = wall_zc(m)
            ue = G / (2ν) .* z .* (h .- z)
            set_profile!(sim.u, ue)        # start on the parabola; relax to the discrete steady state
            dt = 0.3 * m.dz^2 / ν
            for _ in 1:ceil(Int, 2.5h^2 / ν / dt)
                step!(sim, dt; body_force! = drive!)
            end
            u = plane_mean(sim.u)
            (L2 = sqrt(sum(abs2, u .- ue) / Nz), ub = sum(u) / Nz,
             τb = wall_shear_stress(sim), τt = wall_shear_stress(sim; side = :top))
        end
        p1 = log2(res[1].L2 / res[2].L2); p2 = log2(res[2].L2 / res[3].L2)
        ub_exact = G * h^2 / (12ν)
        rel_ub = abs(res[3].ub - ub_exact) / ub_exact
        @printf("    Poiseuille L2 errors %.3e %.3e %.3e → orders %.4f %.4f; bulk at Nz=64 off by %.2e\n",
                res[1].L2, res[2].L2, res[3].L2, p1, p2, rel_ub)
        @test abs(p1 - 2) < 0.1
        @test abs(p2 - 2) < 0.1
        @test rel_ub < 1e-3
        # steady force balance: the two walls carry the whole driving force G h
        for r in res
            @test r.τb - r.τt ≈ G * h rtol = 1e-10
            @test r.τb ≈ G * h / 2 rtol = 1e-10
        end
    end

    @testset "Stokes' first problem: U erfc(z / 2√(νt))" begin
        # The plate starts impulsively at t = 0. The reference is the
        # semi-infinite solution, so the domain is 2h deep (free-slip top):
        # at t = 0.1 h²/ν the top's image term is erfc(2h/√(4νt)) ≈ 8e-6.
        # (In a domain only h deep the top reflects 2.5 % U by then — a
        # property of the reference, not of the solver.)
        ν, h, U = 1.0, 1.0, 1.0
        m = wall_mesh(128; Lz = 2h)
        sim = BoussinesqSimulation(m; ν = ν, bcs = FieldBoundaryConditions(:periodic,
                                   :periodic; bottom = NoSlipWall(U, 0), top = :wall))
        tend = 0.1 * h^2 / ν
        n = ceil(Int, tend / (0.3 * m.dz^2 / ν))
        for _ in 1:n
            step!(sim, tend / n)
        end
        ue = U .* erfc_series.(wall_zc(m) ./ (2 * sqrt(ν * sim.t)))
        err = maximum(abs.(plane_mean(sim.u) .- ue)) / U
        τ_exact = -ν * U / sqrt(π * ν * sim.t)       # ν ∂u/∂z at the plate
        τ = wall_shear_stress(sim)
        @printf("    Stokes I (Nz=128, t=0.1h²/ν): L∞ error %.2e U; wall stress %.5f vs %.5f\n",
                err, τ, τ_exact)
        @test sim.t ≈ tend
        @test err < 0.01
        @test τ ≈ τ_exact rtol = 0.01
    end

    @testset "Stokes' second problem: decay length √(2ν/ω)" begin
        # Plate oscillating as U cos ωt, from rest; the amplitude of the ω
        # harmonic over the 6th period is fitted to e^{-z/δ} on 0.5δ ≤ z ≤ 3δ.
        ν, ω, U = 1.0, 2π, 1.0
        δ = sqrt(2ν / ω)
        m = wall_mesh(128; Lz = 5.0)                   # ≈ 8.9 δ, free-slip top
        wall(t) = FieldBoundaryConditions(:periodic, :periodic;
                                          bottom = NoSlipWall(U * cos(ω * t), 0), top = :wall)
        sim = BoussinesqSimulation(m; ν = ν, bcs = wall(0.0))
        T = 2π / ω
        nper = ceil(Int, T / (0.3 * m.dz^2 / ν)); dt = T / nper
        A = zeros(ComplexF64, m.Nz)
        for p in 1:6, _ in 1:nper
            sim.bcs = wall(sim.t)                     # the wall velocity at the step start
            step!(sim, dt)
            p == 6 && (A .+= plane_mean(sim.u) .* cis(-ω * sim.t) .* (2dt / T))
        end
        z = wall_zc(m)
        sel = findall(zz -> 0.5δ <= zz <= 3δ, z)
        c = [ones(length(sel)) z[sel]] \ log.(abs.(A[sel]))
        δfit = -1 / c[2]
        @printf("    Stokes II: fitted decay length %.5f vs √(2ν/ω) = %.5f (%.2e relative)\n",
                δfit, δ, abs(δfit - δ) / δ)
        @test abs(δfit - δ) / δ < 0.02
        # the phase lags by z/δ as well (the wave travels away from the plate)
        ϕ = -angle.(A[sel]); ϕ .-= ϕ[1]
        @test ϕ[end] ≈ (z[sel[end]] - z[sel[1]]) / δ rtol = 0.02
    end

    @testset "momentum budget of a 3-D flow, to 1e-12 per step" begin
        # Random 3-D velocity between a stationary floor and a moving lid,
        # with advection, a body force and the pressure projection all on.
        # Periodic x, y and w = 0 on the walls leave only the wall stresses
        # and the forcing in the plane-integrated budget:
        #   Δ∫ū dz = Δt [τ(top) − τ(bottom) + G Lz] .
        U, ν, G = 0.5, 0.02, 0.3
        m = RegularGrid(size = (8, 8, 16), extent = (1.0, 1.0, 1.0))
        bcs = FieldBoundaryConditions(:periodic, :periodic; bottom = :noslip,
                                      top = NoSlipWall(U, -0.2))
        sim = BoussinesqSimulation(m; ν = ν, bcs = bcs)
        @test sim.solver isa CosineFFTPoissonSolver        # pressure problem unchanged
        rng = MersenneTwister(7)
        for f in (sim.u, sim.v, sim.w)
            interior_view(f) .= 0.2 .* randn(rng, size(interior_view(f)))
        end
        drive!(s, Δt) = (interior_view(s.tu) .+= G; nothing)
        step!(sim, 1e-3; body_force! = drive!)          # project the random field
        @test max_divergence(sim) < 1e-12
        colint(f) = sum(plane_mean(f)) * m.dz
        dt = 2e-3
        worst_u = worst_v = 0.0
        for _ in 1:20
            Mu, Mv = colint(sim.u), colint(sim.v)
            fu = wall_shear_stress(sim; side = :top) - wall_shear_stress(sim) + G * m.Lz
            fv = wall_shear_stress(sim; side = :top, component = :v) -
                 wall_shear_stress(sim; component = :v)
            step!(sim, dt; body_force! = drive!)
            worst_u = max(worst_u, abs(colint(sim.u) - Mu - dt * fu))
            worst_v = max(worst_v, abs(colint(sim.v) - Mv - dt * fv))
        end
        @printf("    momentum budget residual per step: u %.1e, v %.1e (|Δt·forcing| ≈ %.1e)\n",
                worst_u, worst_v, dt * G)
        @test worst_u < 1e-12
        @test worst_v < 1e-12
        @test max_divergence(sim) < 1e-12
    end

    @testset "IMEX with :noslip matches explicit in the Couette steady state" begin
        # The moving lid makes the implicit diffusion operator affine; the CG
        # iterates with the stationary wall and carries the wall term on the
        # right-hand side (ARK2: the unknown is the state), or drops it
        # (Rosenbrock2: the unknown is an increment).
        U, ν = 1.0, 0.01
        m = wall_mesh(16)
        bcs = FieldBoundaryConditions(:periodic, :periodic; bottom = :noslip,
                                      top = NoSlipWall(U, 0))
        ex = BoussinesqSimulation(m; ν = ν, bcs = bcs)
        dt = 0.3 * m.dz^2 / ν
        for _ in 1:ceil(Int, 4 / ν / dt); step!(ex, dt); end
        for scheme in (ARK2(), Rosenbrock2())
            im = BoussinesqSimulation(m; ν = ν, bcs = bcs)
            for _ in 1:200                              # Δt = 2 s ≈ 13× the explicit limit
                step_imex!(im, 2.0, scheme; tol = 1e-14)
            end
            d = maximum(abs.(plane_mean(im.u) .- plane_mean(ex.u)))
            @printf("    IMEX %s vs explicit Couette: %.2e U, τ_b = %.15f\n",
                    nameof(typeof(scheme)), d, wall_shear_stress(im))
            @test d < 1e-10 * U
            @test abs(wall_shear_stress(im) - ν * U) < 1e-10 * ν * U
        end
        # the implicit solve alone: one backward-Euler step from rest gives the
        # discrete solution of (I − Δt ν ∂zz) u = 0 with the lid at U, which
        # is nonzero only because of the wall term
        f = Field(FaceX(), m); fill!(f.data, 0)
        st = solve_implicit_diffusion!(f, f, ν, 1.0, m, bcs; tol = 1e-14)
        @test st.converged
        u1 = plane_mean(f)
        @test u1[end] > u1[end-1] > 0 && u1[1] ≥ 0
        tmp = Field(FaceX(), m); tmp.data .= f.data; fill_halos!(tmp, bcs)
        lap = Field(FaceX(), m); laplacian(lap, tmp)
        @test maximum(abs.(interior_view(f) .- ν .* interior_view(lap))) < 1e-12
        # w is still pinned to zero on the wall face
        wz = Field(FaceZ(), m); interior_view(wz) .= 1.0
        solve_implicit_diffusion!(wz, wz, ν, 1.0, m, bcs; tol = 1e-14)
        @test all(iszero, wz.data[:, :, m.Hz + 1])
    end

    @testset "other dispatch sites accept NoSlipWall" begin
        U, ν = 1.0, 0.01
        m = wall_mesh(16)
        bcs = FieldBoundaryConditions(:periodic, :periodic; bottom = :noslip,
                                      top = NoSlipWall(U, 0))
        dt = 0.3 * m.dz^2 / ν
        n = ceil(Int, 4 / ν / dt)
        # cut-cell velocity BC as the post-predictor hook (all-fluid geometry):
        # apply_cutcell_velocity_bc! acts on apertures only, so the wall BC
        # comes through fill_halos! untouched
        geom = cut_cell_geometry(m, (x, y, z) -> 1.0)
        sim = BoussinesqSimulation(m; ν = ν, bcs = bcs)
        hook!(s, Δt) = apply_cutcell_velocity_bc!(s, geom)
        for _ in 1:n; step!(sim, dt; post_predictor! = hook!); end
        @test maximum(abs.(plane_mean(sim.u) .- U .* wall_zc(m))) < 1e-10
        # anelastic core: the same halo through fill_halos!
        ma = RegularGrid(size = (4, 4, 16), extent = (100.0, 100.0, 100.0))
        an = AnelasticSimulation(ma; ν = 10.0, buoyancy = false, bcs = bcs)
        dta = 0.3 * ma.dz^2 / 10.0
        for _ in 1:ceil(Int, 4 * 100^2 / 10.0 / dta); step_anelastic!(an, dta); end
        @test maximum(abs.(plane_mean(an.u) .- wall_zc(ma) ./ 100)) < 1e-10
    end

    @testset "regression: `:wall` runs take the unchanged code path" begin
        # The free-slip path does not touch the new code: halo fill
        # dispatches to the original methods (tested above), the IMEX solve
        # sees the very same BC object (no affine term), and the pinned
        # w-row of the implicit solve is the one it always was. (Bitwise
        # equality with master of a 3-D :wall run, explicit and ARK2, was
        # checked when this landed; see the PR.)
        free = FieldBoundaryConditions(:periodic, :periodic, :wall)
        @test homogeneous_bcs(free) === free
        m = wall_mesh(8)
        TI = Aeolus.TimeIntegration
        @test TI._constraint_index(FaceZ(), m, free) == m.Hz + 1
        @test TI._constraint_index(FaceZ(), m,
                  FieldBoundaryConditions(:periodic, :periodic, :noslip)) == m.Hz + 1
        @test TI._constraint_index(FaceZ(), m, FieldBoundaryConditions(:periodic)) == 0
        @test TI._constraint_index(FaceX(), m, free) == 0
    end
end
