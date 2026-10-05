# N-open-b (#255, epic #246): open (Dirichlet-from-boundary-data) lateral
# and top faces, the AbstractBoundaryData contract and the single global
# boundary mass-flux correction, in every dynamical core.
#
#   1. halo-fill semantics: face slots, Dirichlet halos, Neumann φ,
#      idempotence; the free-stream top is bit for bit the P59-d fill
#   2. boundary-data contract: ProfileBoundaryData, FunctionBoundaryData
#   3. the mass-flux correction: a prescribed net imbalance Q gives
#      max|∇·u| = |Q|/V without it and round-off with it, in step!,
#      step_rk3!, step_imex! (ARK2), step_adaptive!, anelastic and
#      pseudo-incompressible; boundary faces equal the corrected data
#   4. uniform flow through the box: 200 steps, every core, with buoyancy
#      and SGS on — u = U, p uniform, θ unchanged to < 1e-12
#   5. Dirichlet-everywhere Taylor–Green vortex in a window: second order
#      (step_rk3!, step_imex! ARK2), with Kim & Moin's intermediate-velocity
#      condition in the implicit stages
#   6. closed boundaries untouched (the hooks are no-ops)
#
# References: Kim & Moin (1985), JCP 59, 308 (Neumann φ for a prescribed
# normal velocity; boundary condition of the intermediate velocity);
# Hellsten et al. (2021), GMD 14, 3185, §3.3 Eq. 2 (uniform mass-flux
# correction); Taylor & Green (1937), Proc. R. Soc. A 158, 499.

using Test
using Aeolus
using Aeolus.Grids: interior, xnodes, ynodes, znodes, FaceX, FaceY, FaceZ, CellCenter
using Aeolus.Fields: Field, interior_view, set!
using Aeolus.BoundaryConditions: OpenBoundary, FreeStreamBoundary, ImpermeableWall,
                                 NoSlipWall, homogeneous_bcs, _z_pair_supported,
                                 axis_poisson_bc, OpenPlane
using Aeolus.Simulations: BoussinesqSimulation, step_rk3!, _default_solver, max_divergence
using Aeolus.TimeIntegration: step_imex!, ARK2, Rosenbrock2, step_adaptive!, CFLConstraints
using Aeolus.Anelastic: AnelasticSimulation, PseudoIncompressibleSimulation,
                        max_anelastic_divergence, max_pseudo_incompressible_divergence,
                        constraint_weights, weighted_divergence!
using Aeolus.PressureSolvers: CosineFFTPoissonSolver, poisson_axis_bcs
using Printf, Random

const _OB_LIN = FunctionBoundaryData(
    u = (x, y, z, t) -> 1.0 + 0.01x + 0.02y + 0.03z + t,
    v = (x, y, z, t) -> -0.5 + 0.02x - 0.01y + 0.01z,
    w = (x, y, z, t) -> 0.1 + 0.001x,
    θ = (x, y, z, t) -> 300.0 + 0.003z + 0.001x)

# ---------------------------------------------------------------------------
@testset "N-open-b halo fill: face slots, Dirichlet halos, Neumann φ, idempotent" begin
    m = RegularGrid(size = (8, 6, 5), extent = (80.0, 60.0, 50.0), origin = (10.0, -5.0, 0.0))
    H = 2; Nx, Ny, Nz = 8, 6, 5
    bcs = open_boundary_conditions(m, _OB_LIN)
    @test bcs.west isa OpenBoundary && bcs.top isa OpenBoundary && bcs.bottom isa ImpermeableWall
    st = only(open_boundary_states(bcs))
    st.correction = :none
    st.t = 0.0
    # a stand-in "simulation" so θ is bound like sim.θ
    θ = Field(CellCenter(), m); φ = Field(CellCenter(), m)
    u = Field(FaceX(), m); v = Field(FaceY(), m); w = Field(FaceZ(), m)
    sim = (bcs = bcs, θ_bcs = bcs, θ = θ, u = u, v = v, w = w, t = 0.0, mesh = m)
    update_open_boundaries!(sim, 0.25, 0.0)
    Random.seed!(3)
    for f in (u, v, w, θ, φ); f.data .= randn(size(f.data)); end
    for f in (u, v, w); fill_halos!(f, bcs); end
    fill_halos!(θ, bcs); fill_halos!(φ, bcs)
    U(x, y, z) = _OB_LIN.fields.u(x, y, z, 0.25)
    xc(i) = 10.0 + (i - H - 0.5) * 10; xf(i) = 10.0 + (i - H - 1) * 10
    yc(j) = -5.0 + (j - H - 0.5) * 10; yf(j) = -5.0 + (j - H - 1) * 10
    zc(k) = (k - H - 0.5) * 10;        zf(k) = (k - H - 1) * 10
    # normal velocity on the boundary faces: west interior slot H+1, east halo H+N+1
    @test all(u.data[H+1, j, k] ≈ U(10.0, yc(j), zc(k)) for j in H+1:H+Ny, k in H+1:H+Nz)
    @test all(u.data[H+Nx+1, j, k] ≈ U(90.0, yc(j), zc(k)) for j in H+1:H+Ny, k in H+1:H+Nz)
    @test all(w.data[i, j, H+Nz+1] ≈ 0.1 + 0.001xc(i) for i in H+1:H+Nx, j in H+1:H+Ny)
    # the bottom stays a wall
    @test all(iszero, w.data[:, :, H+1])
    # Dirichlet halos of tangential velocity and scalars (at the halo points)
    @test all(v.data[h, j, k] ≈ _OB_LIN.fields.v(xc(h), yf(j), zc(k), 0.25)
              for h in 1:H, j in H+1:H+Ny, k in H+1:H+Nz)
    @test all(θ.data[H+Nx+h, j, k] ≈ _OB_LIN.fields.θ(xc(H+Nx+h), yc(j), zc(k), 0.25)
              for h in 1:H, j in H+1:H+Ny, k in H+1:H+Nz)
    @test all(θ.data[i, j, H+Nz+h] ≈ _OB_LIN.fields.θ(xc(i), yc(j), zc(H+Nz+h), 0.25)
              for h in 1:H, i in H+1:H+Nx, j in H+1:H+Ny)
    # φ (no data): mirror — ∂φ/∂n = 0 on every open face (Kim & Moin 1985)
    @test φ.data[H, :, :] == φ.data[H+1, :, :] && φ.data[H+Nx+1, :, :] == φ.data[H+Nx, :, :]
    @test φ.data[:, H, :] == φ.data[:, H+1, :] && φ.data[:, :, H+Nz+1] == φ.data[:, :, H+Nz]
    # idempotent: a second fill changes nothing
    snap = [copy(f.data) for f in (u, v, w, θ, φ)]
    for f in (u, v, w, θ, φ); fill_halos!(f, bcs); end
    @test all(snap[n] == f.data for (n, f) in enumerate((u, v, w, θ, φ)))
    # the homogeneous part: zero data, same rules
    hb = homogeneous_bcs(bcs)
    @test hb !== bcs && hb.west.homogeneous && !bcs.west.homogeneous
    fill_halos!(u, hb)
    @test all(iszero, u.data[H+1, :, :]) && all(iszero, u.data[1:H, :, :])
    # face slots that are interior (west/south) are pinned rows for implicit solves
    @test pinned_face_indices(FaceX(), m, bcs) == (H + 1, 0, 0)
    @test pinned_face_indices(FaceY(), m, bcs) == (0, H + 1, 0)
    @test pinned_face_indices(FaceZ(), m, bcs) == (0, 0, H + 1)
    # pressure problem: Neumann on every axis, the all-Neumann DCT by default
    @test axis_poisson_bc(bcs.west) === :neumann
    @test poisson_axis_bcs(bcs) == (bc_x = :neumann, bc_y = :neumann, bc_z = :neumann)
    s = _default_solver(m, bcs)
    @test s isa CosineFFTPoissonSolver && (s.bc_x, s.bc_y, s.bc_z) == (:neumann, :neumann, :neumann)
    # a wall below an open top is a supported z pair; an open bottom is not
    @test _z_pair_supported(ImpermeableWall(), bcs.top)
    @test _z_pair_supported(NoSlipWall(), bcs.top)
    @test !_z_pair_supported(bcs.top, ImpermeableWall())
    @test_throws ArgumentError OpenPlane(:u, FaceX(), :bottom, m)
    # channel: open x, periodic y
    cb = open_boundary_conditions(m, _OB_LIN; y = :periodic, top = :wall)
    @test poisson_axis_bcs(cb) == (bc_x = :neumann, bc_y = :periodic, bc_z = :neumann)
end

# ---------------------------------------------------------------------------
@testset "FreeStreamBoundary is the special case, bit for bit the P59-d fill" begin
    # The P59-d (#185) top fill, verbatim, as the reference.
    function ref_fs_top!(d::AbstractArray{T,3}, wb, Hz, Nz, isw::Bool) where {T}
        if !isw
            for h in 1:Hz
                @views d[:, :, Hz + Nz + h] .= d[:, :, Hz + Nz + 1 - h]
            end
            return d
        end
        kw = Hz + Nz + 1
        for k_ in 0:Hz
            dst = kw + k_
            dst <= size(d, 3) || continue
            for j in axes(d, 2), i in axes(d, 1)
                d[i, j, dst] = k_ == 0 ? T(wb[i]) : 2 * T(wb[i]) - d[i, j, kw - k_]
            end
        end
        return d
    end
    for (H, bot) in ((2, NoSlipWall()), (3, ImpermeableWall()), (1, NoSlipWall(0.3, 0.0)))
        m = RegularGrid(size = (7, 5, 6), extent = (1.0, 1.0, 1.0), halo = (H, H, H))
        Random.seed!(11 + H)
        wb = randn(m.Nx + 2H)
        bcs = FieldBoundaryConditions(:periodic, :periodic; bottom = bot,
                                      top = FreeStreamBoundary(wb))
        @test bcs.top isa Aeolus.BoundaryConditions.AbstractOpenBoundary
        @test !has_open_boundaries(bcs)        # no data, no mass correction
        for loc in (FaceX(), FaceY(), FaceZ(), CellCenter())
            f = Field(loc, m); f.data .= randn(size(f.data))
            ref = copy(f.data)
            fill_halos!(f, bcs)
            # reference: periodic x, y; the wall bottom; the P59-d top
            Aeolus.BoundaryConditions._fill_x!(ref, Periodic(), Periodic(), H, m.Nx, loc)
            Aeolus.BoundaryConditions._fill_y!(ref, Periodic(), Periodic(), H, m.Ny, loc)
            Aeolus.BoundaryConditions._fill_wall_bottom!(ref, bot, H, loc)
            ref_fs_top!(ref, wb, H, m.Nz, loc isa FaceZ)
            @test f.data == ref                 # bit for bit
        end
        # homogeneous part: zero transpiration
        hb = homogeneous_bcs(bcs)
        @test all(iszero, hb.top.w_top)
    end
end

# ---------------------------------------------------------------------------
@testset "boundary-data contract" begin
    m = RegularGrid(size = (6, 4, 5), extent = (6.0, 4.0, 5.0))
    θpad = collect(290.0 .+ (1:m.Nz+2m.Hz))
    d = ProfileBoundaryData(u = 3.0, v = z -> 0.5z, θ = θpad)
    bcs = open_boundary_conditions(m, d)
    st = only(open_boundary_states(bcs))
    boundary_values!(st.planes, d, 0.0)
    pw = bcs.west.planes[:v]
    @test all(pw.data[a, b, c] ≈ 0.5pw.z[c] for a in axes(pw.data, 1), b in axes(pw.data, 2),
              c in axes(pw.data, 3))
    pt = bcs.top.planes[:θ]
    @test all(pt.data[:, :, c] == fill(θpad[k], size(pt.data, 1), size(pt.data, 2))
              for (c, k) in enumerate(pt.krange))
    @test all(==(0.0), bcs.north.planes[:w].data)       # w defaults to zero
    @test all(==(3.0), bcs.east.planes[:u].data)
    @test_throws ArgumentError boundary_values!(st.planes, ProfileBoundaryData(u = 1.0, v = 0.0), 0.0)
    @test_throws ArgumentError open_boundary_conditions(m, d; correction = :bogus)
    @test_throws ArgumentError open_boundary_conditions(m, d; fields = (:u, :θ))
end
