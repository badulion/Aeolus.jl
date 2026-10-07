using Test
using Printf
using Random
using LinearAlgebra: eigvals
using Aeolus
using Aeolus.Anelastic
using Aeolus.Grids: CellCenter, FaceZ, znodes, interior
using Aeolus.Fields: set!, interior_view

# =========================================================================
# Era II / Phase 27 — anelastic and pseudo-incompressible dynamics.
#
# Every @testset that produces a physically interesting number also prints
# it, so the log doubles as the validation report.
# =========================================================================

"L2 norm over a field's interior."
_l2(a) = sqrt(sum(abs2, a))

"Relative L2 difference between the interiors of two like-located fields."
_reldiff(a, b) = _l2(interior_view(a) .- interior_view(b)) / _l2(interior_view(b))

"Drop the mean of an array (the null space of every singular Poisson operator)."
_demean(a) = a .- sum(a) / length(a)

@testset "Era II — Anelastic & pseudo-incompressible" begin

# =====================================================================
# 1. Reference-state hydrostatic balance — THE headline test.
#
# Every constructor must satisfy the *discrete* balance
#     (p̄[k] − p̄[k−1])/Δz = −g ρ̄_f[k]
# on the model's own grid, to round-off. This is the stencil the
# w-momentum equation actually evaluates, so a nonzero residual would
# mean the base state itself accelerates the flow.
# =====================================================================
@testset "Reference-state discrete hydrostatic balance" begin
    grids = (shallow = RegularGrid(; size=(8,8,16), extent=(1000.0,1000.0,1000.0), halo=(2,2,2)),
             deep    = RegularGrid(; size=(8,8,32), extent=(4000.0,4000.0,10000.0), halo=(2,2,2)))

    for (gname, g) in pairs(grids)
        refs = (isothermal   = isothermal_reference(g; T_surf=288.0, p_surf=1e5),
                isentropic   = isentropic_reference(g; θ_const=300.0, p_surf=1e5),
                constant_N   = constant_stability_reference(g; N=0.01, θ_surf=300.0, p_surf=1e5),
                from_profile = reference_from_profile(g, z -> 300.0 + 0.004z, 1e5))

        for (name, r) in pairs(refs)
            res   = max_hydrostatic_residual(r)
            scale = r.g * maximum(r.ρ̄)          # |∂p̄/∂z| itself, Pa m⁻¹
            @printf("  [hydrostatic] %-8s %-13s max|resid| = %.3e Pa/m   (rel %.2e)   ρ̄ ∈ [%.4f, %.4f]  H = %.0f m\n",
                    String(gname), String(name), res, res/scale,
                    minimum(r.ρ̄), maximum(r.ρ̄), reference_scale_height(r))

            @test res < 1e-10                   # the required absolute bound
            @test res / scale < 1e-13           # ... and it is really round-off

            # structural invariants of the assembled state
            @test length(r.ρ̄)   == g.Nz
            @test length(r.ρ̄_f) == g.Nz + 1
            @test length(r.N²)  == g.Nz
            @test length(r.ρ̄c_pad) == g.Nz + 2g.Hz
            @test all(>(0), r.ρ̄) && all(>(0), r.θ̄) && all(>(0), r.p̄) && all(>(0), r.T̄)
            @test all(>(0), r.ρ̄_f) && all(>(0), r.ρ̄c_pad) && all(>(0), r.ρ̄f_pad)
            @test r.P  ≈ r.ρ̄ .* r.θ̄
            @test r.P_f ≈ r.ρ̄_f .* r.θ̄_f
            # ideal gas + Exner consistency
            @test maximum(abs, r.p̄ .- r.R_d .* r.ρ̄ .* r.T̄) / maximum(r.p̄) < 1e-14
            @test maximum(abs, r.T̄ .- r.θ̄ .* r.π̄) / maximum(r.T̄) < 1e-14
            # face profiles lie between their neighbours (a valid mean)
            @test all(k -> min(r.ρ̄[k-1], r.ρ̄[k]) <= r.ρ̄_f[k] <= max(r.ρ̄[k-1], r.ρ̄[k]),
                      2:g.Nz)
            # θ̄_f is the arithmetic mean the buoyancy kernel assumes
            @test r.θ̄_f[2:g.Nz] ≈ 0.5 .* (r.θ̄[1:g.Nz-1] .+ r.θ̄[2:g.Nz])
            # wall faces mirror the adjacent cell (matches the halo fill)
            @test r.θ̄_f[1] == r.θ̄[1] && r.θ̄_f[end] == r.θ̄[g.Nz]
        end
    end

    # vector-valued θ profile is accepted too, and matches the function form
    g = grids.shallow
    θvec = [300.0 + 0.004z for z in znodes(g, CellCenter())]
    rv = reference_from_profile(g, θvec, 1e5)
    rf = reference_from_profile(g, z -> 300.0 + 0.004z, 1e5)
    @test rv.ρ̄ ≈ rf.ρ̄
    @test max_hydrostatic_residual(rv) < 1e-10
    @test_throws DimensionMismatch reference_from_profile(g, [300.0, 301.0], 1e5)
end

# =====================================================================
# 2. Isothermal base state must be the *exact* analytic exponential.
# =====================================================================
@testset "Isothermal reference matches ρ₀exp(−z/H)" begin
    g  = RegularGrid(; size=(8,8,24), extent=(2000.0,2000.0,6000.0), halo=(2,2,2))
    Ts, ps = 288.0, 1e5
    r  = isothermal_reference(g; T_surf=Ts, p_surf=ps)
    H  = r.R_d * Ts / r.g
    ρ₀ = ps / (r.R_d * Ts)
    ζ  = collect(znodes(g, CellCenter())) .- g.z0
    ρ_analytic = @. ρ₀ * exp(-ζ / H)
    p_analytic = @. ps  * exp(-ζ / H)

    err_ρ = maximum(abs, r.ρ̄ .- ρ_analytic)
    err_p = maximum(abs, r.p̄ .- p_analytic)
    @printf("  [isothermal]   H = %.3f m   max|ρ̄ − ρ₀e^(−z/H)| = %.3e kg/m³   max|p̄ − p_s e^(−z/H)| = %.3e Pa\n",
            H, err_ρ, err_p)
    @printf("                 ρ̄(top)/ρ̄(bottom) = %.4f   max|T̄ − T_surf| = %.3e K\n",
            r.ρ̄[end]/r.ρ̄[1], maximum(abs, r.T̄ .- Ts))

    @test err_ρ < 1e-8
    @test err_p / ps < 1e-14
    @test maximum(abs, r.T̄ .- Ts) < 1e-9
    @test reference_scale_height(r) ≈ H rtol=1e-12
    # and it still satisfies the discrete balance — the two are compatible
    # because ρ̄_f is the logarithmic mean.
    @test max_hydrostatic_residual(r) < 1e-10
    # θ̄ grows upward in an isothermal atmosphere (it is stably stratified)
    @test issorted(r.θ̄)
    @test all(>(0), r.N²)
end

# =====================================================================
# 3. Constant-N² base state: the diagnosed N² must be the requested one.
# =====================================================================
@testset "Constant-stability reference reproduces N²" begin
    g = RegularGrid(; size=(8,8,32), extent=(2000.0,2000.0,8000.0), halo=(2,2,2))
    for N in (0.005, 0.01, 0.02)
        r   = constant_stability_reference(g; N=N, θ_surf=300.0)
        err = maximum(abs, r.N² .- N^2)
        @printf("  [N²] requested N = %.4f  (N² = %.3e)   max|N²_diag − N²| = %.3e   (rel %.2e)   θ̄ ∈ [%.2f, %.2f]\n",
                N, N^2, err, err/N^2, minimum(r.θ̄), maximum(r.θ̄))
        @test err < 1e-8            # required bound
        @test err / N^2 < 1e-12     # ... and it is round-off
        @test max_hydrostatic_residual(r) < 1e-10
        # θ̄ = θ_surf exp(N²ζ/g)
        ζ = collect(znodes(g, CellCenter())) .- g.z0
        @test r.θ̄ ≈ 300.0 .* exp.(N^2 .* ζ ./ r.g)
    end

    # the neutral state has *identically* zero stability
    rn = isentropic_reference(g; θ_const=300.0)
    @printf("  [N²] isentropic base state: max|N²| = %.3e\n", maximum(abs, rn.N²))
    @test maximum(abs, rn.N²) == 0.0
    @test all(rn.θ̄ .== 300.0)
end

# =====================================================================
# 4. Density-weighted Poisson solver — manufactured solution.
#
# Apply the discrete ∇·(ρ̄∇·) operator to a chosen φ to get f, solve, and
# recover φ. Done with a strongly varying ρ̄ (factor > 3 across the
# domain) so the variable coefficient is genuinely exercised.
# =====================================================================
@testset "DensityWeightedPoissonSolver — manufactured solution" begin
    L, Lz = 1000.0, 1000.0
    g  = RegularGrid(; size=(16,16,16), extent=(L,L,Lz), halo=(2,2,2))
    # ρ̄ = exp(4 ζ / Lz): a factor e⁴ ≈ 55 top-to-bottom in the continuum,
    # > 3 between the extreme cell centres of this grid.
    ρ̄c = [exp(4 * (z - g.z0) / Lz)          for z in znodes(g, CellCenter())]
    ρ̄f = [exp(4 * (k - 1) * g.dz / Lz)      for k in 1:g.Nz+1]
    ratio = maximum(ρ̄c) / minimum(ρ̄c)

    s = DensityWeightedPoissonSolver(g, ρ̄c, ρ̄f; tol=1e-13, maxit=3000)

    φ = Field(CellCenter(), g); f = Field(CellCenter(), g); out = Field(CellCenter(), g)
    set!(φ, (x,y,z) -> cos(2π*x/L) * cos(2π*y/L) * cos(π*(z-g.z0)/Lz) +
                       0.3 * sin(4π*x/L) * cos(2π*(z-g.z0)/Lz) +
                       0.2 * sin(2π*y/L) * cos(3π*(z-g.z0)/Lz))
    apply_weighted_laplacian!(f, φ, s)
    solve!(out, f, s)

    err = maximum(abs, _demean(collect(interior_view(out))) .-
                       _demean(collect(interior_view(φ))))
    @printf("  [DW Poisson] ρ̄ cell-centre ratio = %.3f   max|φ_solved − φ_exact| = %.3e   iterations = %d   ‖r‖ = %.3e\n",
            ratio, err, solver_iterations(s), solver_residual(s))

    @test ratio > 3
    @test err < 1e-8
    @test solver_iterations(s) < 3000

    # --- self-adjointness: ⟨Lφ, ψ⟩ = ⟨φ, Lψ⟩ ------------------------
    # This is the property that face-averaging ρ̄ buys us; with a
    # cell-centred coefficient in front of a centred 2nd difference the
    # coupling between cells k and k+1 would be ρ̄[k] seen from below and
    # ρ̄[k+1] seen from above, the operator would not be symmetric, and CG
    # would not be applicable at all.
    rng = MersenneTwister(2027)
    ψ = Field(CellCenter(), g); Lφ = Field(CellCenter(), g); Lψ = Field(CellCenter(), g)
    φr = Field(CellCenter(), g)
    interior_view(φr) .= randn(rng, size(interior_view(φr)))
    interior_view(ψ)  .= randn(rng, size(interior_view(ψ)))
    apply_weighted_laplacian!(Lφ, φr, s)
    apply_weighted_laplacian!(Lψ, ψ,  s)
    a1 = sum(interior_view(Lφ) .* interior_view(ψ))
    a2 = sum(interior_view(φr) .* interior_view(Lψ))
    nrm = _l2(interior_view(Lφ)) * _l2(interior_view(ψ))
    @printf("  [DW Poisson] self-adjointness: ⟨Lφ,ψ⟩ = %.6e, ⟨φ,Lψ⟩ = %.6e, |Δ|/(‖Lφ‖‖ψ‖) = %.3e\n",
            a1, a2, abs(a1 - a2) / nrm)
    @test abs(a1 - a2) / nrm < 1e-13

    # ... and exactly symmetric as a matrix, on a grid small enough to
    # assemble it column by column.
    gs = RegularGrid(; size=(4,3,5), extent=(1.0,1.0,1.0), halo=(2,2,2))
    wcs = [exp(3(z - gs.z0)) for z in znodes(gs, CellCenter())]
    wfs = [exp(3(k - 1) * gs.dz) for k in 1:gs.Nz+1]
    ss  = DensityWeightedPoissonSolver(gs, wcs, wfs)
    n   = gs.Nx * gs.Ny * gs.Nz
    A   = zeros(n, n)
    e   = Field(CellCenter(), gs); o = Field(CellCenter(), gs)
    for c in 1:n
        fill!(e.data, 0)
        interior_view(e)[c] = 1.0
        apply_weighted_laplacian!(o, e, ss)
        A[:, c] .= vec(collect(interior_view(o)))
    end
    @printf("  [DW Poisson] assembled %d×%d operator: max|A − Aᵀ| = %.3e  (‖A‖∞ = %.3f), rank deficiency = %d\n",
            n, n, maximum(abs, A .- A'), maximum(abs, A), count(<(1e-10), abs.(eigvals(A))))
    @test maximum(abs, A .- A') == 0.0           # exactly, not approximately
    @test all(<=(1e-10), real.(eigvals(A)))      # negative semi-definite
    @test count(<(1e-10), abs.(eigvals(A))) == 1 # one-dimensional null space (constants)

    # --- the operator annihilates constants, and the solve is zero-mean
    set!(ψ, (x,y,z) -> 7.0)
    apply_weighted_laplacian!(Lψ, ψ, s)
    @test maximum(abs, interior_view(Lψ)) < 1e-12
    @test abs(sum(interior_view(out))) / length(interior_view(out)) < 1e-12

    # --- guard rails --------------------------------------------------
    @test_throws DimensionMismatch DensityWeightedPoissonSolver(g, ρ̄c[1:end-1], ρ̄f)
    @test_throws DimensionMismatch DensityWeightedPoissonSolver(g, ρ̄c, ρ̄f[1:end-1])
    @test_throws ErrorException DensityWeightedPoissonSolver(g, .-ρ̄c, ρ̄f)
end

# =====================================================================
# 5. With ρ̄ ≡ const the density-weighted solver must reproduce
#    CGPoissonSolver exactly (same discrete operator, up to the scale).
# =====================================================================
@testset "Density-weighted solver reduces to the constant-density case" begin
    L, Lz = 1.0, 1.0
    g = RegularGrid(; size=(12,12,12), extent=(L,L,Lz), halo=(2,2,2))
    c = 2.5

    dw = DensityWeightedPoissonSolver(g, c; tol=1e-13, maxit=5000)
    cg = CGPoissonSolver(g; tol=1e-13, maxit=5000, bc_top=:neumann, bc_bottom=:neumann)

    φ  = Field(CellCenter(), g)
    set!(φ, (x,y,z) -> cos(2π*x/L) * cos(4π*y/L) * cos(π*(z-g.z0)/Lz) +
                       0.4 * sin(2π*x/L) * cos(2π*(z-g.z0)/Lz))
    f  = Field(CellCenter(), g)
    apply_weighted_laplacian!(f, φ, dw)          # f = c ∇²φ

    f_scaled = Field(CellCenter(), g)             # f/c = ∇²φ  for the plain CG
    f_scaled.data .= f.data ./ c

    p_dw = Field(CellCenter(), g); p_cg = Field(CellCenter(), g)
    solve!(p_dw, f,        dw)
    solve!(p_cg, f_scaled, cg)

    a = _demean(collect(interior_view(p_dw)))
    b = _demean(collect(interior_view(p_cg)))
    e = _demean(collect(interior_view(φ)))
    diff = maximum(abs, a .- b)
    @printf("  [DW vs CG] max|φ_DW − φ_CG| = %.3e   (‖φ‖∞ = %.3f)   DW err vs exact = %.3e   CG err vs exact = %.3e\n",
            diff, maximum(abs, b), maximum(abs, a .- e), maximum(abs, b .- e))
    @test diff < 1e-10
    @test maximum(abs, a .- e) < 1e-10           # both recover the exact φ
    @test maximum(abs, b .- e) < 1e-10
end

# =====================================================================
# 5b. The direct solver (#309): FFT/DCT horizontally, Thomas in z. It
#     inverts the *same* operator as the CG, so it must recover a
#     manufactured φ to round-off, agree with the CG to the CG tolerance,
#     and leave the density-weighted flux divergence-free to round-off.
# =====================================================================
@testset "DirectWeightedPoissonSolver — exact inverse of the weighted operator (#309)" begin
    L, Lz = 2000.0, 9000.0                       # deep: ρ̄ drops by e^(9/8.8) ≈ 2.8
    g = RegularGrid(; size=(20,16,24), extent=(L,1.3L,Lz), halo=(2,2,2))
    ref = isothermal_reference(g; T_surf=300.0)
    rng = Xoshiro(309)
    for wt in (:ρ, :ρθ), bx in (:periodic, :neumann), by in (:periodic, :neumann)
        d = DirectWeightedPoissonSolver(g, ref; weighting=wt, bc_x=bx, bc_y=by)
        c = DensityWeightedPoissonSolver(g, ref; weighting=wt, bc_x=bx, bc_y=by,
                                         tol=1e-14, maxit=5000)
        @test Aeolus.PressureSolvers.poisson_axis_bcs(d) == Aeolus.PressureSolvers.poisson_axis_bcs(c)
        @test d.w_c == c.w_c && d.w_f == c.w_f   # one weighting, the reference state's
        # manufactured: random zero-mean φ, f = ∇·(w∇φ) with the CG's operator
        φ = Field(CellCenter(), g); f = Field(CellCenter(), g)
        interior_view(φ) .= _demean(randn(rng, size(interior_view(φ))))
        apply_weighted_laplacian!(f, φ, c)
        pd = Field(CellCenter(), g); pc = Field(CellCenter(), g); r = Field(CellCenter(), g)
        solve!(pd, f, d); solve!(pc, f, c)
        scale = maximum(abs, interior_view(φ))
        e_exact = maximum(abs, interior_view(pd) .- interior_view(φ)) / scale
        e_cg    = maximum(abs, interior_view(pd) .- interior_view(pc)) / scale
        apply_weighted_laplacian!(r, pd, d)
        res = maximum(abs, interior_view(r) .- interior_view(f)) / maximum(abs, interior_view(f))
        @printf("  [direct DW %-2s %-8s %-8s] max|φ−φ_exact|/‖φ‖ = %.2e  vs CG(tol 1e-14) = %.2e  residual = %.2e  mean = %.1e\n",
                wt, bx, by, e_exact, e_cg, res, abs(sum(interior_view(pd))) / length(interior_view(pd)))
        @test e_exact < 1e-11                        # exact, not to a tolerance
        @test e_cg < 1e-10                           # the CG's own error
        @test res < 1e-12
        @test abs(sum(interior_view(pd))) / length(interior_view(pd)) < 1e-12 * scale  # zero mean, as the CG
    end

    # an incompatible source: like the CG, the uniform part is projected
    # out, and the residual is exactly the (uniform) mean of the source
    d = DirectWeightedPoissonSolver(g, ref)
    f = Field(CellCenter(), g); p = Field(CellCenter(), g); r = Field(CellCenter(), g)
    interior_view(f) .= randn(rng, size(interior_view(f))) .+ 0.7
    fbar = sum(interior_view(f)) / length(interior_view(f))
    solve!(p, f, d)
    apply_weighted_laplacian!(r, p, d)
    @test maximum(abs, interior_view(r) .- (interior_view(f) .- fbar)) < 1e-12 * maximum(abs, interior_view(f))

    # constant weight c: c ∇²φ = f is the CosineFFT problem scaled by 1/c
    cst = 2.5
    for bx in (:periodic, :neumann)
        dc = DirectWeightedPoissonSolver(g, cst; bc_x=bx, bc_y=bx)
        cf = CosineFFTPoissonSolver(g; bc_x=bx, bc_y=bx, bc_z=:neumann)
        interior_view(f) .= _demean(randn(rng, size(interior_view(f))))
        fs = Field(CellCenter(), g); fs.data .= f.data ./ cst
        p1 = Field(CellCenter(), g); p2 = Field(CellCenter(), g)
        solve!(p1, f, dc); solve!(p2, fs, cf)
        @test maximum(abs, _demean(collect(interior_view(p1))) .- _demean(collect(interior_view(p2)))) <
              1e-11 * maximum(abs, interior_view(p2))
    end

    # guard rails
    ρ̄c, ρ̄f = ref.ρ̄, ref.ρ̄_f
    @test_throws DimensionMismatch DirectWeightedPoissonSolver(g, ρ̄c[1:end-1], ρ̄f)
    @test_throws DimensionMismatch DirectWeightedPoissonSolver(g, ρ̄c, ρ̄f[1:end-1])
    @test_throws ErrorException DirectWeightedPoissonSolver(g, .-ρ̄c, ρ̄f)
    @test_throws ErrorException DirectWeightedPoissonSolver(g, ref; bc_z=:periodic)
    @test_throws ErrorException DirectWeightedPoissonSolver(g, ref; weighting=:θ)
end

# The projection of the sound-proof cores is now the direct solve: the
# weighted constraint holds to round-off after a step, where the CG leaves
# it at its tolerance, and the two cores agree to that tolerance.
@testset "Sound-proof cores project with the direct solver (#309)" begin
    L, Lz = 3000.0, 6000.0
    g   = RegularGrid(; size=(16,16,24), extent=(L,L,Lz), halo=(2,2,2))
    bcs = FieldBoundaryConditions(:periodic, :periodic, :wall)
    ref = isothermal_reference(g; T_surf=300.0)
    bubble(x, y, z) = 300.0 + 2.0 * exp(-(((x-L/2)/500)^2 + ((y-L/2)/500)^2 + ((z-1500.0)/500)^2))
    for (lbl, S) in (("anelastic", AnelasticSimulation), ("pseudo-incompressible", PseudoIncompressibleSimulation))
        sd = S(g; ref=ref, bcs=bcs)
        sc = S(g; ref=ref, bcs=bcs, pressure_solver=:cg, tol=1e-12, maxit=3000)
        @test sd.solver isa DirectWeightedPoissonSolver
        @test sc.solver isa DensityWeightedPoissonSolver
        @test_throws ArgumentError S(g; ref=ref, bcs=bcs, pressure_solver=:multigrid)
        for sim in (sd, sc)
            set!(sim.θ, bubble)
            for _ in 1:5
                step!(sim, 2.0)
            end
        end
        rd, rc = Aeolus.Anelastic.constraint_residual(sd), Aeolus.Anelastic.constraint_residual(sc)
        wmax = maximum(abs, interior_view(sd.w))
        dw = maximum(abs, interior_view(sd.w) .- interior_view(sc.w)) / wmax
        @printf("  [%s] after 5 steps: weighted-constraint residual direct = %.2e s⁻¹, CG(1e-12) = %.2e s⁻¹; max|w_direct − w_CG|/|w|max = %.2e (|w|max = %.3f m/s)\n",
                lbl, rd, rc, dw, wmax)
        @test wmax > 1e-3                         # the bubble rises
        @test rd < 1e-14                          # round-off
        @test dw < 1e-8                           # same flow as the CG projection
    end
end

# =====================================================================
# 6. The anelastic constraint is what gets satisfied — and ∇·u is not.
# =====================================================================
@testset "Anelastic constraint after step_anelastic!" begin
    L, Lz = 3000.0, 3000.0
    g   = RegularGrid(; size=(16,16,16), extent=(L,L,Lz), halo=(2,2,2))
    bcs = FieldBoundaryConditions(:periodic, :periodic, :wall)
    ref = isentropic_reference(g; θ_const=300.0)
    sim = AnelasticSimulation(g; ref=ref, bcs=bcs, tol=1e-13, maxit=2000)

    # the constructor leaves the base state at rest
    @test max_w(sim) == 0.0
    @test mean_θ_profile(sim) ≈ ref.θ̄
    @test max_anelastic_divergence(sim) == 0.0

    set!(sim.θ, (x,y,z) -> 300.0 +
        2.0 * exp(-(((x-L/2)/500)^2 + ((y-L/2)/500)^2 + ((z-1000.0)/500)^2)))

    for n in 1:5
        step_anelastic!(sim, 2.0)
    end

    d_anel = max_anelastic_divergence(sim)      # (1/ρ̄)∇·(ρ̄u)
    d_mass = max_mass_flux_divergence(sim)      # ∇·(ρ̄u)
    d_plain = max_divergence(sim)               # ∇·u  — NOT zero
    H = reference_scale_height(ref)
    predicted = maximum(abs, interior_view(sim.w)) / H     # |w ∂logρ̄/∂z|

    @printf("  [anelastic constraint] t = %.1f s   max|(1/ρ̄)∇·(ρ̄u)| = %.3e s⁻¹   max|∇·(ρ̄u)| = %.3e   max|∇·u| = %.3e s⁻¹\n",
            sim.t, d_anel, d_mass, d_plain)
    @printf("                         |w|max/H = %.3e s⁻¹  →  ∇·u / (|w|max/H) = %.3f   (solver: %s)\n",
            predicted, d_plain/predicted, nameof(typeof(sim.solver)))

    @test d_anel  < 1e-10
    @test d_mass  < 1e-10
    # The whole point of anelastic: plain ∇·u is emphatically nonzero, and
    # its magnitude is set by the density scale height, ∇·u = −w ∂logρ̄/∂z.
    @test d_plain > 1e-6
    @test 0.2 < d_plain / predicted < 5.0

    @test isfinite(kinetic_energy(sim)) && kinetic_energy(sim) > 0
    @test length(mean_θ_profile(sim)) == g.Nz

    # a z-periodic domain has no well-posed anelastic constraint
    @test_throws ErrorException AnelasticSimulation(g; ref=ref,
        bcs=FieldBoundaryConditions(:periodic))

    # run! drives the same step
    ke_before = kinetic_energy(sim)
    run!(sim, 2.0, 3)
    @test sim.t ≈ 16.0
    @test kinetic_energy(sim) > ke_before
    @test max_anelastic_divergence(sim) < 1e-10
end

# =====================================================================
# 7. Boussinesq limit: as ρ̄ → const the anelastic solution must
#    converge to the Boussinesq one, linearly in the density variation.
# =====================================================================
@testset "Boussinesq limit and its scaling with Δρ/ρ" begin
    function limit_case(Lz; nsteps=4, dt=1.0)
        L = Lz
        g   = RegularGrid(; size=(10,10,10), extent=(L,L,Lz), halo=(2,2,2))
        bcs = FieldBoundaryConditions(:periodic, :periodic, :wall)
        ref = isentropic_reference(g; θ_const=300.0)
        # domain-relative initial condition so the cases are "the same run"
        pert(x,y,z) = 2.0 * exp(-(((x-L/2)/(0.2L))^2 + ((y-L/2)/(0.2L))^2 +
                                  ((z-0.4Lz)/(0.2Lz))^2))
        b = BoussinesqSimulation(g; θ₀=300.0, bcs=bcs)
        set!(b.θ, (x,y,z) -> 300.0 + pert(x,y,z))
        a = AnelasticSimulation(g; ref=ref, bcs=bcs, tol=1e-12, maxit=3000)
        set!(a.θ, (x,y,z) -> 300.0 + pert(x,y,z))
        for _ in 1:nsteps
            step!(b, dt)
            step_anelastic!(a, dt)
        end
        ρ̄mean = sum(ref.ρ̄) / length(ref.ρ̄)
        Δρ = (maximum(ref.ρ̄) - minimum(ref.ρ̄)) / ρ̄mean
        return (rel_w = _reldiff(a.w, b.w),
                rel_θ = _reldiff(a.θ, b.θ),
                Δρ = Δρ, H = reference_scale_height(ref), b = b, a = a)
    end

    shallow = limit_case(20.0)
    medium  = limit_case(200.0)
    deep    = limit_case(2000.0)

    for (nm, c) in (("Lz =   20 m", shallow), ("Lz =  200 m", medium), ("Lz = 2000 m", deep))
        @printf("  [Boussinesq limit] %s   Δρ/ρ̄ = %.5f   rel‖Δw‖ = %.4e   rel‖Δθ‖ = %.3e   ratio (rel‖Δw‖)/(Δρ/ρ̄) = %.5f\n",
                nm, c.Δρ, c.rel_w, c.rel_θ, c.rel_w / c.Δρ)
    end

    # (i) the shallow case IS the Boussinesq solution to within 1e-4
    @test shallow.rel_w < 1e-4
    @test shallow.rel_θ < 1e-6
    # (ii) the difference is not zero — the systems really are different
    @test shallow.rel_w > 0
    # (iii) and it scales linearly with the density variation: a ×10 deeper
    #       domain gives a ×10 larger discrepancy.
    r1 = medium.rel_w / shallow.rel_w
    r2 = deep.rel_w   / medium.rel_w
    @printf("  [Boussinesq limit] discrepancy growth per decade of Δρ/ρ̄: %.3f, %.3f  (linear ⇒ 10)\n", r1, r2)
    @test 9.0 < r1 < 11.0
    @test 9.0 < r2 < 11.0
    # (iv) constant ratio rel/Δρ across three decades
    ratios = (shallow.rel_w/shallow.Δρ, medium.rel_w/medium.Δρ, deep.rel_w/deep.Δρ)
    @test maximum(ratios) / minimum(ratios) < 1.1
    # (v) the anelastic runs satisfy their own constraint throughout
    for c in (shallow, medium, deep)
        @test max_anelastic_divergence(c.a) < 1e-10
        @test max_divergence(c.b) < 1e-10
    end
end

# =====================================================================
# 8. Warm-bubble benchmark in all three systems.
# =====================================================================
@testset "Warm bubble in Boussinesq / anelastic / pseudo-incompressible" begin
    L, Lz = 3000.0, 3000.0
    g   = RegularGrid(; size=(16,16,32), extent=(L,L,Lz), halo=(2,2,2))
    bcs = FieldBoundaryConditions(:periodic, :periodic, :wall)
    ref = isentropic_reference(g; θ_const=300.0)      # neutral stratification
    # +2 K Gaussian bubble centred at z = 800 m, radius 400–500 m
    bubble(x,y,z) = 2.0 * exp(-(((x-L/2)/500)^2 + ((y-L/2)/500)^2 + ((z-800.0)/400)^2))
    initθ(s) = set!(s.θ, (x,y,z) -> 300.0 + bubble(x,y,z))

    bq = BoussinesqSimulation(g; θ₀=300.0, bcs=bcs);                           initθ(bq)
    an = AnelasticSimulation(g; ref=ref, bcs=bcs, tol=1e-13, maxit=2000);      initθ(an)
    pi_ = PseudoIncompressibleSimulation(g; ref=ref, bcs=bcs, tol=1e-13, maxit=2000); initθ(pi_)

    top0 = bubble_top_height(bq)
    @printf("  [warm bubble] deep domain Lz = %.0f m, H = %.0f m, Lz/H = %.3f, ρ̄ ∈ [%.4f, %.4f]; initial bubble top = %.1f m\n",
            Lz, reference_scale_height(ref), Lz/reference_scale_height(ref),
            minimum(ref.ρ̄), maximum(ref.ρ̄), top0)

    dt, nsteps = 2.0, 130
    for _ in 1:nsteps
        step!(bq, dt)
        step_anelastic!(an, dt)
        step_pseudo_incompressible!(pi_, dt)
    end

    d = system_comparison_diagnostics((boussinesq = bq, anelastic = an, pseudo = pi_))
    for name in keys(d)
        r = d[name]
        @printf("  [warm bubble] %-11s t = %5.1f s  bubble_top = %7.1f m  max_w = %7.4f m/s  θ'var = %.5f K²  KE(unw) = %.4e  constraint = %.2e  |∇·u| = %.3e\n",
                String(name), r.t, r.bubble_top, r.max_w, r.θ_variance,
                r.kinetic_energy_unweighted, r.constraint_residual, r.plain_divergence)
    end
    @printf("  [warm bubble] relative to Boussinesq:  anelastic  ΔKE = %+.4f  Δmax_w = %+.5f  Δθ'var = %+.4f\n",
            d.anelastic.rel_ke, d.anelastic.rel_max_w, d.anelastic.rel_θ_variance)
    @printf("  [warm bubble] relative to Boussinesq:  pseudo-inc ΔKE = %+.4f  Δmax_w = %+.5f  Δθ'var = %+.4f\n",
            d.pseudo.rel_ke, d.pseudo.rel_max_w, d.pseudo.rel_θ_variance)

    # --- (a) all three are physically sensible -------------------------
    for (name, sim) in (("boussinesq", bq), ("anelastic", an), ("pseudo", pi_))
        @test all(isfinite, interior_view(sim.w))
        @test all(isfinite, interior_view(sim.θ))
        @test bubble_top_height(sim) > top0            # the bubble rose
        @test max_w(sim) > 1.0                         # and is still rising
        @test maximum(interior_view(sim.θ)) > 300.0    # warm core survives
        @test θ_variance(sim) > 0
    end
    # w > 0 at the bubble top: the leading edge is ascending
    zc = collect(znodes(g, CellCenter()))
    for sim in (bq, an, pi_)
        ktop = argmin(abs.(zc .- bubble_top_height(sim)))
        @test maximum(@view interior_view(sim.w)[:, :, ktop]) > 0.5
    end

    # --- (b) each core satisfies its own constraint --------------------
    @test max_divergence(bq)                       < 1e-10
    @test max_anelastic_divergence(an)             < 1e-10
    @test max_pseudo_incompressible_divergence(pi_) < 1e-10
    # and the sound-proof cores are NOT divergence-free
    @test max_divergence(an)  > 1e-6
    @test max_divergence(pi_) > 1e-6

    # --- (c) the sound-proof cores differ from Boussinesq, in the
    #         expected direction: with ∇·(ρ̄u) = 0 the *mass* flux is what
    #         is conserved, so the same buoyancy forcing produces a larger
    #         velocity where ρ̄ is smaller — i.e. the anelastic run carries
    #         more vertical velocity overall, and proportionally more of it
    #         aloft.
    #
    #         Note the discriminator is a norm, not a pointwise extremum:
    #         on a 16×16×32 grid `max_w` is a single cell value that flips
    #         sign of difference from step to step, whereas ‖w‖₂ and the
    #         kinetic energy separate the two systems monotonically.
    rel_w = _reldiff(an.w, bq.w)
    wi_b = interior_view(bq.w); wi_a = interior_view(an.w)
    upper = (g.Nz ÷ 2 + 1):g.Nz
    ratio_all   = _l2(wi_a) / _l2(wi_b)
    ratio_upper = _l2(@view wi_a[:, :, upper]) / _l2(@view wi_b[:, :, upper])
    ke_ratio    = unweighted_kinetic_energy(an) / unweighted_kinetic_energy(bq)
    @printf("  [warm bubble] anelastic vs Boussinesq: rel‖Δw‖ = %.4f;  ‖w‖₂ ratio = %.5f;  ‖w‖₂ ratio in the upper half = %.5f;  KE ratio = %.5f\n",
            rel_w, ratio_all, ratio_upper, ke_ratio)
    @test 0.01 < rel_w < 0.5          # genuinely different, not blown up
    @test ratio_all   > 1.0           # stronger ascent for the same forcing
    @test ratio_upper > ratio_all     # and the excess is concentrated aloft
    @test ke_ratio    > 1.0
    @test d.anelastic.rel_ke > 0      # same statement via the comparison tool
    @test d.pseudo.rel_ke    > 0

    # --- (d) on a NEUTRAL base state θ̄ = const the pseudo-density ρ̄θ̄ is
    #         ρ̄ times a constant, so the two projections are the *same*
    #         linear system: the two cores must agree to round-off.
    dw = maximum(abs, interior_view(an.w) .- interior_view(pi_.w))
    dθ = maximum(abs, interior_view(an.θ) .- interior_view(pi_.θ))
    @printf("  [warm bubble] anelastic vs pseudo-incompressible on θ̄ = const: max|Δw| = %.3e m/s, max|Δθ| = %.3e K\n", dw, dθ)
    @test dw < 1e-8
    @test dθ < 1e-8
end

# =====================================================================
# 9. Pseudo-incompressible constraint ∇·(ρ̄θ̄u) = 0.
# =====================================================================
@testset "Pseudo-incompressible constraint after a step" begin
    L, Lz = 4000.0, 8000.0
    g   = RegularGrid(; size=(16,16,16), extent=(L,L,Lz), halo=(2,2,2))
    bcs = FieldBoundaryConditions(:periodic, :periodic, :wall)
    ref = constant_stability_reference(g; N=0.012, θ_surf=300.0)
    sim = PseudoIncompressibleSimulation(g; ref=ref, bcs=bcs, tol=1e-13, maxit=3000)

    zc = collect(znodes(g, CellCenter()))
    ii, jj, kk = interior(g)
    for (kl, k) in enumerate(kk), (jl, j) in enumerate(jj), (il, i) in enumerate(ii)
        x = g.x0 + (il - 0.5) * g.dx
        y = g.y0 + (jl - 0.5) * g.dy
        sim.θ.data[i,j,k] = ref.θ̄[kl] +
            3.0 * exp(-(((x-L/2)/700)^2 + ((y-L/2)/700)^2 + ((zc[kl]-2000.0)/700)^2))
    end

    step_pseudo_incompressible!(sim, 2.0)
    r1 = max_pseudo_incompressible_divergence(sim)
    for _ in 1:9
        step_pseudo_incompressible!(sim, 2.0)
    end
    r10 = max_pseudo_incompressible_divergence(sim)

    # the raw ∇·(ρ̄θ̄u) residual (un-normalised), which is what the
    # projection literally annihilates
    raw = Field(CellCenter(), g)
    fill_halos!(sim.u, bcs); fill_halos!(sim.v, bcs); fill_halos!(sim.w, bcs)
    weighted_divergence!(raw, sim.u, sim.v, sim.w, ref.Pc_pad, ref.Pf_pad, g; normalize=false)
    raw_max = maximum(abs, interior_view(raw))

    @printf("  [pseudo-incompressible] after 1 step: max|(1/ρ̄θ̄)∇·(ρ̄θ̄u)| = %.3e s⁻¹; after 10 steps: %.3e s⁻¹; raw max|∇·(ρ̄θ̄u)| = %.3e\n",
            r1, r10, raw_max)
    @printf("                          anelastic residual of the same field = %.3e;  plain |∇·u| = %.3e;  solver = %s\n",
            max_anelastic_divergence(sim), max_divergence(sim), nameof(typeof(sim.solver)))

    @test r1  < 1e-10
    @test r10 < 1e-10
    @test raw_max < 1e-10
    # the pseudo-incompressible solution does NOT satisfy the anelastic
    # constraint on a stratified base state — the two systems are distinct
    @test max_anelastic_divergence(sim) > 1e-8
    @test max_divergence(sim) > 1e-6
end

# =====================================================================
# 10. On a stratified base state the two sound-proof systems separate,
#     and each violates the other's constraint by a measurable amount.
# =====================================================================
@testset "Anelastic vs pseudo-incompressible on a stratified base state" begin
    L, Lz = 4000.0, 8000.0
    g   = RegularGrid(; size=(16,16,16), extent=(L,L,Lz), halo=(2,2,2))
    bcs = FieldBoundaryConditions(:periodic, :periodic, :wall)
    ref = constant_stability_reference(g; N=0.012, θ_surf=300.0)
    zc  = collect(znodes(g, CellCenter()))

    function build(kind)
        s = kind === :anelastic ?
            AnelasticSimulation(g; ref=ref, bcs=bcs, tol=1e-13, maxit=3000) :
            PseudoIncompressibleSimulation(g; ref=ref, bcs=bcs, tol=1e-13, maxit=3000)
        ii, jj, kk = interior(g)
        for (kl, k) in enumerate(kk), (jl, j) in enumerate(jj), (il, i) in enumerate(ii)
            x = g.x0 + (il - 0.5) * g.dx
            y = g.y0 + (jl - 0.5) * g.dy
            s.θ.data[i,j,k] = ref.θ̄[kl] +
                3.0 * exp(-(((x-L/2)/700)^2 + ((y-L/2)/700)^2 + ((zc[kl]-2000.0)/700)^2))
        end
        return s
    end

    an  = build(:anelastic)
    pi_ = build(:pseudo)
    for _ in 1:90
        step_anelastic!(an, 2.0)
        step_pseudo_incompressible!(pi_, 2.0)
    end

    rel = _reldiff(an.w, pi_.w)
    @printf("  [stratified] θ̄ ∈ [%.2f, %.2f] K (N = 0.012 s⁻¹), ρ̄ ∈ [%.4f, %.4f]\n",
            minimum(ref.θ̄), maximum(ref.θ̄), minimum(ref.ρ̄), maximum(ref.ρ̄))
    @printf("  [stratified] rel‖Δw‖(anelastic, pseudo) = %.5f;  max_w = %.4f vs %.4f m/s\n",
            rel, max_w(an), max_w(pi_))
    @printf("  [stratified] anelastic run: own residual %.2e, foreign (ρ̄θ̄) residual %.2e\n",
            max_anelastic_divergence(an), max_pseudo_incompressible_divergence(an))
    @printf("  [stratified] pseudo run:    own residual %.2e, foreign (ρ̄)  residual %.2e\n",
            max_pseudo_incompressible_divergence(pi_), max_anelastic_divergence(pi_))

    @test max_anelastic_divergence(an)              < 1e-10
    @test max_pseudo_incompressible_divergence(pi_) < 1e-10
    # each violates the other's constraint → the systems are genuinely
    # different once θ̄ varies with height
    @test max_pseudo_incompressible_divergence(an) > 1e-8
    @test max_anelastic_divergence(pi_)            > 1e-8
    @test rel > 1e-4
    @test rel < 0.5
end

# =====================================================================
# 11. system_comparison_diagnostics is a well-formed instrument.
# =====================================================================
@testset "system_comparison_diagnostics contract" begin
    L, Lz = 1000.0, 1000.0
    g   = RegularGrid(; size=(8,8,8), extent=(L,L,Lz), halo=(2,2,2))
    bcs = FieldBoundaryConditions(:periodic, :periodic, :wall)
    ref = isentropic_reference(g; θ_const=300.0)
    bubble(x,y,z) = 2.0 * exp(-(((x-L/2)/200)^2 + ((y-L/2)/200)^2 + ((z-300.0)/200)^2))

    bq = BoussinesqSimulation(g; θ₀=300.0, bcs=bcs)
    set!(bq.θ, (x,y,z) -> 300.0 + bubble(x,y,z))
    an = AnelasticSimulation(g; ref=ref, bcs=bcs, tol=1e-13, maxit=2000)
    set!(an.θ, (x,y,z) -> 300.0 + bubble(x,y,z))
    pi_ = PseudoIncompressibleSimulation(g; ref=ref, bcs=bcs, tol=1e-13, maxit=2000)
    set!(pi_.θ, (x,y,z) -> 300.0 + bubble(x,y,z))
    for _ in 1:10
        step!(bq, 1.0); step_anelastic!(an, 1.0); step_pseudo_incompressible!(pi_, 1.0)
    end

    d = system_comparison_diagnostics((boussinesq = bq, anelastic = an, pseudo = pi_))
    @test d isa NamedTuple
    @test keys(d) == (:boussinesq, :anelastic, :pseudo)
    for k in keys(d)
        r = d[k]
        for f in (:t, :kinetic_energy, :kinetic_energy_unweighted, :max_w, :θ_variance,
                  :bubble_top, :constraint_residual, :plain_divergence,
                  :rel_ke, :rel_max_w, :rel_θ_variance)
            @test haskey(r, f)
            @test isfinite(getfield(r, f))
        end
        @test r.t ≈ 10.0
        @test r.constraint_residual < 1e-10
    end
    # the first entry is the reference, so its relative differences vanish
    @test d.boussinesq.rel_ke == 0
    @test d.boussinesq.rel_max_w == 0
    @test d.boussinesq.rel_θ_variance == 0
    # the density-weighted KE of a sound-proof core differs from the
    # unweighted one, while Boussinesq's are the same definition
    @test d.boussinesq.kinetic_energy == d.boussinesq.kinetic_energy_unweighted
    @test d.anelastic.kinetic_energy != d.anelastic.kinetic_energy_unweighted
    # types and single-entry edge case
    @test system_comparison_diagnostics(NamedTuple()) == NamedTuple()
    d1 = system_comparison_diagnostics((only = an,))
    @test d1.only.rel_ke == 0
end

@testset "constraint_weights: the W of ∇·(W u) = 0, and W-weighted tracer transport" begin
    g = RegularGrid(; size = (8, 8, 16), extent = (800.0, 800.0, 4000.0), halo = (2, 2, 2))
    ref = constant_stability_reference(g; N = 0.01, θ_surf = 300.0, p_surf = 1e5)
    an = AnelasticSimulation(g; ref = ref)
    psi = PseudoIncompressibleSimulation(g; ref = ref)
    @test constraint_weights(an) == (ref.ρ̄c_pad, ref.ρ̄f_pad)
    @test constraint_weights(psi) == (ref.Pc_pad, ref.Pf_pad)
    # after a projection ∇·(W u) vanishes with these weights, so the flux
    # form ∇·(W u q) of a uniform q does too, while ∇·(u q) does not
    for sim in (an, psi)
        set!(sim.θ, (x, y, z) -> sim.ref.θ̄[clamp(round(Int, z / g.dz + 0.5), 1, g.Nz)] +
                                 0.5 * exp(-((x - 400)^2 + (z - 1000)^2) / 300^2))
        for _ in 1:5
            step!(sim, 5.0)
        end
        Wc, Wf = constraint_weights(sim)
        d = Field(CellCenter(), g)
        weighted_divergence!(d, sim.u, sim.v, sim.w, Wc, Wf, g)
        Hz = g.Hz
        @test maximum(abs, interior_view(d)) / maximum(Wc[Hz+1:Hz+g.Nz]) < 1e-8
        @test max_divergence(sim) > 1e-6       # the plain divergence is not zero
    end
end

# =====================================================================
# Reference from a θ profile against the continuous atmosphere, with the
# MoistThermodynamics constants. Moved from test/berlin_moist.jl,
# test/berlin_dynamics.jl and test/berlin_air_heat.jl (#320), where it
# was asserted on the Berlin case's reference state (which is
# `reference_from_profile` of the case's θ(z)); here on synthetic
# profiles of the same shapes.
# =====================================================================
"""
The pressure at heights `z` (increasing, above the ground) of the continuous
hydrostatic atmosphere dp/dz = −g p/(R θ(z) (p/p₀)^κ), RK4 with `nsub`
sub-steps per interval: the independent reference the discrete balance is
measured against.
"""
function _continuous_hydrostatic(θ, z, p_surf; R, cp, p₀, g, nsub = 1000)
    κ = R / cp
    f(zz, pp) = -g * pp / (R * θ(zz) * (pp / p₀)^κ)
    out = similar(z, Float64)
    p, zz = Float64(p_surf), 0.0
    for k in eachindex(z)
        h = (z[k] - zz) / nsub
        for _ in 1:nsub
            k1 = f(zz, p); k2 = f(zz + h / 2, p + h / 2 * k1)
            k3 = f(zz + h / 2, p + h / 2 * k2); k4 = f(zz + h, p + h * k3)
            p += h / 6 * (k1 + 2k2 + 2k3 + k4); zz += h
        end
        out[k] = p
    end
    return out
end

@testset "reference_from_profile: continuous hydrostatics, ideal gas, Exner, scale height" begin
    MT = Aeolus.MoistThermodynamics
    c = MT.DEFAULT_CONSTANTS
    R, cp, p₀, gg = c.gas_constant / c.molmass_dry, c.cp_dry, c.p_ref, 9.81
    # a neutral layer under a 3 K/km inversion (the Berlin smoke window:
    # 16 levels over 100 m, inversion at 60 m), a deeper one (32 over 200 m,
    # inversion at 100 m), and a 3 km stable column at Δz = 12.5 m (the
    # discrete balance is second order: the error grows as Δz²)
    cases = [(RegularGrid(; size = (4, 4, 16), extent = (250.0, 250.0, 100.0), halo = (2, 2, 2)),
              z -> 294.0 + max(0, z - 60.0) * 3e-3, 101325.0),
             (RegularGrid(; size = (4, 4, 32), extent = (1000.0, 1000.0, 200.0), halo = (2, 2, 2)),
              z -> 293.0 + max(0, z - 100.0) * 3e-3, 100400.0),
             (RegularGrid(; size = (4, 4, 240), extent = (3000.0, 3000.0, 3000.0), halo = (2, 2, 2)),
              z -> 300.0 + 3e-3 * z, 101325.0)]
    for (m, θf, ps) in cases
        r = reference_from_profile(m, θf, ps; R_d = R, cp = cp, p₀ = p₀, g_accel = gg)
        zc = [(k - 0.5) * m.dz for k in 1:m.Nz]
        p_cont = _continuous_hydrostatic(θf, zc, ps; R = R, cp = cp, p₀ = p₀, g = gg)
        err = maximum(abs.(r.p̄ .- p_cont) ./ p_cont)
        res = max_hydrostatic_residual(r)
        # −∂ln ρ̄/∂z = (1 − κ) g/(R T̄) + ∂ln θ̄/∂z (hydrostatic ideal gas), on the faces
        κ = R / cp
        T̄f = (r.T̄[1:end-1] .+ r.T̄[2:end]) ./ 2
        expected = (1 - κ) .* gg ./ (R .* T̄f) .+ diff(log.(r.θ̄)) ./ m.dz
        sh = maximum(abs.((-diff(log.(r.ρ̄)) ./ m.dz) ./ expected .- 1))
        @printf("  [profile reference] Nz = %d, H = %.0f m: p̄ vs continuous hydrostatic max rel %.1e; discrete residual %.1e Pa/m; scale-height identity %.1e; Hρ %.0f m\n",
                m.Nz, m.Lz, err, res, sh, 1 / maximum(-diff(log.(r.ρ̄)) ./ m.dz))
        @test err < 1e-6                   # the Berlin acceptance was 0.1 %; measured ≲ 6e-8
        @test res < 1e-9                   # Pa/m: the grid balance is exact
        @test sh < 1e-6                    # measured 5e-9 on the Berlin smoke window
        @test all(<(0), diff(r.p̄)) && r.p̄[1] < ps
        for k in 1:m.Nz
            @test r.θ̄[k] ≈ θf(zc[k]) rtol = 1e-14
            @test r.T̄[k] ≈ r.θ̄[k] * r.π̄[k] rtol = 1e-12
            # one source of truth: the Exner function and the ideal gas of
            # MoistThermodynamics
            @test r.π̄[k] ≈ MT.exner_function(r.p̄[k]) rtol = 1e-12
            @test r.ρ̄[k] ≈ MT.air_density(r.T̄[k], r.p̄[k]) rtol = 1e-12
        end
    end
end

@testset "heat content ∫ρ̄ c_p Π̄ θ dz of a uniform-θ layer, analytically (O(Δz²))" begin
    # θ uniform: dry adiabatic, Π(z) = Π_s − g z/(c_p θ), and
    # ρΠ = p₀ Π^{c_p/R}/(R θ), so
    #   ∫ ρ c_p Π θ dz = c_p² θ p₀ /(R g) [Π^{c_p/R+1}/(c_p/R+1)]_{Π_top}^{Π_s}.
    # The midpoint sum over the discretely hydrostatic ρ̄, Π̄ agrees to O(Δz²).
    c = Aeolus.MoistThermodynamics.DEFAULT_CONSTANTS
    R, cp, p₀, gg = c.gas_constant / c.molmass_dry, c.cp_dry, c.p_ref, 9.81
    θ, ps, H = 294.0, 101325.0, 100.0
    errs = Float64[]
    for Nz in (16, 64)
        m = RegularGrid(; size = (4, 4, Nz), extent = (250.0, 250.0, H), halo = (2, 2, 2))
        r = reference_from_profile(m, z -> θ, ps; R_d = R, cp = cp, p₀ = p₀, g_accel = gg)
        Πs = (ps / p₀)^(R / cp)
        Πt = Πs - gg * H / (cp * θ)
        a = cp / R + 1
        H_exact = cp^2 * θ * p₀ / (R * gg) * (Πs^a - Πt^a) / a
        Hd = sum(r.ρ̄[k] * cp * r.π̄[k] * θ * m.dz for k in 1:Nz)
        @printf("  [heat content] uniform θ = %.0f K, Nz = %d: %.9e J/m², analytic %.9e (rel %.1e)\n",
                θ, Nz, Hd, H_exact, abs(Hd / H_exact - 1))
        # measured 3.1e-8 (Nz = 16) and 2.0e-9 (Nz = 64) on the Berlin meter
        @test Hd ≈ H_exact rtol = (Nz == 16 ? 1e-7 : 1e-8)
        push!(errs, abs(Hd / H_exact - 1))
    end
    @test errs[1] / errs[2] > 12          # second order: 16 for a 4× finer grid
end

end # @testset "Era II — Anelastic & pseudo-incompressible"
