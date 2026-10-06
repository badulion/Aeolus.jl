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
                                 axis_poisson_bc, OpenPlane, Periodic
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
    # a wall below an open top is a supported z pair; an open bottom only
    # below an open top (a nest child inside its parent, N-nest-d #261 —
    # before it an open bottom was refused outright)
    @test _z_pair_supported(ImpermeableWall(), bcs.top)
    @test _z_pair_supported(NoSlipWall(), bcs.top)
    @test !_z_pair_supported(bcs.top, ImpermeableWall())
    @test _z_pair_supported(bcs.top, bcs.top)
    @test OpenPlane(:u, FaceX(), :bottom, m).range == 1:m.Hz
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

# ---------------------------------------------------------------------------
# 3. A prescribed net imbalance: u = 1 + 0.2 x/Lx on the faces gives an
# outflow Q = 0.2 Ly Lz through the east face (plus nothing else).
const _Lx, _Ly, _Lz = 800.0, 600.0, 400.0
const _Q_IMB = 0.2 * _Ly * _Lz
_imb_data() = FunctionBoundaryData(u = (x, y, z, t) -> 1.0 + 0.2 * x / _Lx,
                                   v = 0.0, w = 0.0, θ = 300.0)

# The contract's plug point: any subtype that fills the planes. Velocity from
# the analytic imbalance, θ from a storage-indexed reference profile.
struct _ImbalancedReference <: AbstractBoundaryData
    θ̄::Vector{Float64}
end
function Aeolus.BoundaryConditions.boundary_values!(planes::AbstractVector{<:OpenPlane},
                                                    d::_ImbalancedReference, t)
    boundary_values!(filter(p -> p.name !== :θ, planes), _imb_data(), t)
    boundary_values!(filter(p -> p.name === :θ, planes), ProfileBoundaryData(θ = d.θ̄), t)
    return planes
end

function _faces_equal_buffers(sim)
    m = sim.mesh; H = m.Hx
    jj = m.Hy+1:m.Hy+m.Ny; kk = m.Hz+1:m.Hz+m.Nz; ii = m.Hx+1:m.Hx+m.Nx
    b = sim.bcs
    return sim.u.data[H+1, jj, kk] == b.west.planes[:u].data[H+1, jj, kk] &&
           sim.u.data[H+m.Nx+1, jj, kk] == b.east.planes[:u].data[1, jj, kk] &&
           sim.v.data[ii, H+1, kk] == b.south.planes[:v].data[ii, H+1, kk] &&
           sim.v.data[ii, H+m.Ny+1, kk] == b.north.planes[:v].data[ii, 1, kk] &&
           sim.w.data[ii, jj, H+m.Nz+1] == b.top.planes[:w].data[ii, jj, 1]
end

@testset "mass-flux correction: Q/V failure mode and the fix, every core" begin
    m = RegularGrid(size = (16, 12, 8), extent = (_Lx, _Ly, _Lz))
    V = _Lx * _Ly * _Lz
    bous(corr) = (s = BoussinesqSimulation(m; bcs = open_boundary_conditions(m, _imb_data();
                                                    correction = corr, record = true),
                                           θ₀ = 300.0, C_s = 0.18);
                  set!(s.θ, 300.0); set!(s.u, 1.0); s)
    cores = (
        ("step!",          s -> step!(s, 1.0)),
        ("step_rk3!",      s -> step_rk3!(s, 1.0)),
        ("step_imex! ARK2", s -> step_imex!(s, 1.0, ARK2())),
        ("step_imex! ROS2", s -> step_imex!(s, 1.0, Rosenbrock2())),
    )
    # Every core starts from a projected state (one Euler step), as a run
    # does after its first step: SSP-RK3 combines the step-start state into
    # its result, so a non-solenoidal initial condition would survive one
    # step in any domain, open or closed.
    for (name, adv!) in cores
        @testset "$name" begin
            s0 = bous(:none)
            step!(s0, 1.0)
            adv!(s0)
            L = open_boundary_ledger(s0)
            @test L.Q ≈ _Q_IMB rtol = 1e-12
            @test L.residual == L.Q                         # nothing corrected
            # the failure mode exists: a uniform divergence Q/V everywhere
            @test abs(max_divergence(s0) - _Q_IMB / V) < 1e-12
            s1 = bous(:uniform)
            step!(s1, 1.0)
            adv!(s1)
            L1 = open_boundary_ledger(s1)
            @test L1.Q ≈ _Q_IMB rtol = 1e-12
            @test abs(L1.residual) < 1e-9 * _Q_IMB
            @test L1.δ ≈ _Q_IMB / (2 * (_Ly * _Lz + _Lx * _Lz) + _Lx * _Ly) rtol = 1e-12
            @test max_divergence(s1) < 1e-12
            @test _faces_equal_buffers(s1)                  # bit for bit
            @test !isempty(only(open_boundary_states(s1)).log)
        end
    end
    @testset "step_adaptive!" begin
        c = CFLConstraints(dt_max = 1.0)
        s0 = bous(:none)
        # the uncorrected run cannot pass the divergence check: it is rejected
        # every time and throws, restoring the state
        @test_throws ErrorException step_adaptive!(s0, c; max_retries = 2)
        @test s0.t == 0.0
        s1 = bous(:uniform)
        _, dt, nr = step_adaptive!(s1, c)
        @test nr == 0 && max_divergence(s1) < 1e-12
        @test _faces_equal_buffers(s1)
        @test open_boundary_ledger(s1).t ≈ dt
    end
    @testset "outflow_only leaves the inflow untouched" begin
        s = BoussinesqSimulation(m; bcs = open_boundary_conditions(m, _imb_data();
                                                                  correction = :outflow_only))
        set!(s.u, 1.0)
        step!(s, 1.0)
        H = m.Hx
        @test all(==(1.0), s.u.data[H+1, H+1:H+m.Ny, H+1:H+m.Nz])     # inflow face as given
        @test all(≈(1.0; atol = 1e-12), s.u.data[H+m.Nx+1, H+1:H+m.Ny, H+1:H+m.Nz]) # outflow cut back
        @test max_divergence(s) < 1e-12
    end
    for (name, ctor, divf) in (
            ("anelastic", AnelasticSimulation, max_anelastic_divergence),
            ("pseudo-incompressible", PseudoIncompressibleSimulation,
             max_pseudo_incompressible_divergence))
        @testset "$name" begin
            mk(corr) = begin
                s = ctor(m; bcs = open_boundary_conditions(m, _imb_data(); correction = corr))
                # a user-defined AbstractBoundaryData: the imbalanced u, θ = θ̄(z)
                set_boundary_data!(s.bcs, _ImbalancedReference(s.ref.θ̄c_pad))
                set!(s.u, 1.0)
                s
            end
            s0 = mk(:none)
            step!(s0, 1.0)
            Wc, Wf = constraint_weights(s0)
            QW = sum(0.2 * Wc[k] * _Ly * m.dz for k in m.Hz+1:m.Hz+m.Nz)
            @test open_boundary_ledger(s0).Q ≈ QW rtol = 1e-12
            # failure mode: ∇·(W u) = Q_W / V uniform
            dv = Field(CellCenter(), m)
            fill_halos!(s0.u, s0.bcs); fill_halos!(s0.v, s0.bcs); fill_halos!(s0.w, s0.bcs)
            weighted_divergence!(dv, s0.u, s0.v, s0.w, Wc, Wf, m)
            @test maximum(abs, interior_view(dv) .- QW / V) < 1e-9 * QW / V
            s1 = mk(:uniform)
            step!(s1, 1.0)
            @test divf(s1) < 1e-9
            @test abs(open_boundary_ledger(s1).residual) < 1e-9 * QW
            @test _faces_equal_buffers(s1)
        end
    end
end

# ---------------------------------------------------------------------------
@testset "uniform flow U = (5, 2, 0): 200 steps, every core, buoyancy and SGS" begin
    m = RegularGrid(size = (12, 10, 8), extent = (600.0, 500.0, 400.0))
    U, Vv = 5.0, 2.0
    data = ProfileBoundaryData(u = U, v = Vv, w = 0.0, θ = 300.0)
    function bous()
        s = BoussinesqSimulation(m; bcs = open_boundary_conditions(m, data),
                                 θ₀ = 300.0, C_s = 0.18, ν = 1e-5)
        set!(s.u, U); set!(s.v, Vv); set!(s.θ, 300.0)
        return s
    end
    check(s, θref) = (maximum(abs, interior_view(s.u) .- U) < 1e-12 &&
                      maximum(abs, interior_view(s.v) .- Vv) < 1e-12 &&
                      maximum(abs, interior_view(s.w)) < 1e-12 &&
                      maximum(abs, interior_view(s.θ) .- θref) < 1e-12 &&
                      (maximum(interior_view(s.p)) - minimum(interior_view(s.p))) < 1e-12)
    for (name, adv!) in (("step!", s -> step!(s, 2.0)),
                         ("step_rk3!", s -> step_rk3!(s, 2.0)),
                         ("step_imex! ARK2", s -> step_imex!(s, 2.0, ARK2())),
                         ("step_imex! ROS2", s -> step_imex!(s, 2.0, Rosenbrock2())),
                         ("step_adaptive!", s -> step_adaptive!(s, CFLConstraints(dt_max = 2.0))))
        s = bous()
        θ0 = copy(interior_view(s.θ))
        for _ in 1:200; adv!(s); end
        @test check(s, θ0)
        @test abs(open_boundary_ledger(s).residual) < 1e-9
    end
    for ctor in (AnelasticSimulation, PseudoIncompressibleSimulation)
        s = ctor(m; bcs = open_boundary_conditions(m, data), C_s = 0.18)
        only(open_boundary_states(s)).data = ProfileBoundaryData(u = U, v = Vv, θ = s.ref.θ̄c_pad)
        set!(s.u, U); set!(s.v, Vv)
        θ0 = copy(interior_view(s.θ))                 # θ̄(z) from the constructor
        for _ in 1:200; step!(s, 2.0); end
        @test check(s, θ0)
    end
end

# ---------------------------------------------------------------------------
# 5. Taylor–Green vortex u = sin kx cos ky F(t), v = −cos kx sin ky F(t),
# F = exp(−2νk²t), an exact Navier–Stokes solution, in a window that cuts
# it at arbitrary phase: every lateral face is open with the exact data.
const _TG_k = 2π
const _TG_ν = 0.05
_tgF(t) = exp(-2 * _TG_ν * _TG_k^2 * t)
_tgu(x, y, z, t) = sin(_TG_k * x) * cos(_TG_k * y) * _tgF(t)
_tgv(x, y, z, t) = -cos(_TG_k * x) * sin(_TG_k * y) * _tgF(t)

function _tg_error(N, adv!; c = 0.2, T = 0.1)
    m = RegularGrid(size = (N, N, 2), extent = (1.0, 1.0, 2.0 / N), origin = (0.13, 0.31, 0.0))
    bcs = open_boundary_conditions(m, FunctionBoundaryData(u = _tgu, v = _tgv, w = 0.0, θ = 0.0);
                                   top = :wall)
    s = BoussinesqSimulation(m; bcs = bcs, ν = _TG_ν)
    set!(s.u, (x, y, z) -> _tgu(x, y, z, 0.0)); set!(s.v, (x, y, z) -> _tgv(x, y, z, 0.0))
    n = ceil(Int, T / (c / N^2 / _TG_ν)); dt = T / n      # ν dt / h² = c
    for _ in 1:n; adv!(s, dt); end
    xs = xnodes(m, FaceX()); ys = ynodes(m, FaceX())
    ui = interior_view(s.u)
    return maximum(abs(ui[i, j, 1] - _tgu(xs[i], ys[j], 0, s.t)) for i in 1:N, j in 1:N),
           max_divergence(s)
end

@testset "Dirichlet-everywhere Taylor–Green: second order" begin
    for (name, adv!) in (("step_rk3!", (s, dt) -> step_rk3!(s, dt)),
                         ("step_imex! ARK2", (s, dt) -> step_imex!(s, dt, ARK2())))
        r = [_tg_error(N, adv!) for N in (16, 32, 64)]
        e = first.(r)
        p1 = log2(e[1] / e[2]); p2 = log2(e[2] / e[3])
        @printf("    TG window %-16s max|u-u_exact| = %.3e %.3e %.3e   order %.3f %.3f\n",
                name, e..., p1, p2)
        @test abs(p1 - 2) < 0.2 && abs(p2 - 2) < 0.2
        @test all(x -> x[2] < 1e-12, r)
    end
end

# ---------------------------------------------------------------------------
@testset "closed boundaries: the hooks are no-ops" begin
    m = RegularGrid(size = (8, 8, 8), extent = (1.0, 1.0, 1.0))
    for bcs in (FieldBoundaryConditions(:periodic),
                FieldBoundaryConditions(:periodic, :periodic, :wall),
                FieldBoundaryConditions(:periodic, :periodic; bottom = :noslip, top = :wall))
        @test !has_open_boundaries(bcs)
        @test homogeneous_bcs(bcs) === bcs
        @test pinned_face_indices(FaceX(), m, bcs) == (0, 0, 0)
        s = BoussinesqSimulation(m; bcs = bcs)
        @test update_open_boundaries!(s, 1.0, 1.0) === s
        @test open_boundary_snapshot(s) === nothing
        @test isempty(open_boundary_states(s))
    end
    # the face assertion fires if ∂φ/∂n ≠ 0 at an open west face
    bo = open_boundary_conditions(m, _OB_LIN)
    gx = Field(FaceX(), m); gy = Field(FaceY(), m)
    @test assert_open_face_correction(bo, gx, gy) === nothing
    gx.data[m.Hx + 1, m.Hy + 2, m.Hz + 2] = 1e-3
    @test_throws ErrorException assert_open_face_correction(bo, gx, gy)
end

# ---------------------------------------------------------------------------
# N-nest-d (#261): a nest child inside its parent has an open bottom as well
# as an open top (and a child spanning a periodic z extent keeps periodic z).
@testset "open bottom (nest child inside its parent): fill, correction, projection" begin
    m = RegularGrid(size = (8, 8, 8), extent = (80.0, 80.0, 80.0), origin = (0.0, 0.0, 40.0))
    # net vertical imbalance: w grows with z, so the bottom and top fluxes differ
    data = FunctionBoundaryData(u = (x, y, z, t) -> 2.0, v = (x, y, z, t) -> -1.0,
                                w = (x, y, z, t) -> 0.3 + 0.01z,
                                θ = (x, y, z, t) -> 300.0 + 0.003z)
    bcs = open_boundary_conditions(m, data; bottom = :open, top = :open)
    @test bcs.bottom isa OpenBoundary && bcs.top isa OpenBoundary
    @test poisson_axis_bcs(bcs) == (bc_x = :neumann, bc_y = :neumann, bc_z = :neumann)
    @test_throws ArgumentError open_boundary_conditions(m, data; bottom = :open, top = :wall)
    bp = open_boundary_conditions(m, data; bottom = :periodic, top = :periodic)
    @test bp.bottom isa Periodic && bp.west isa OpenBoundary
    # the w boundary face of an open bottom is a pinned row, as on a wall
    @test pinned_face_indices(FaceZ(), m, bcs) == (0, 0, m.Hz + 1)
    sim = BoussinesqSimulation(m; bcs = bcs)
    fill!(sim.u.data, 2.0); fill!(sim.v.data, -1.0); fill!(sim.θ.data, 300.0)
    step!(sim, 0.5)
    led = open_boundary_ledger(sim)
    # Q = (w_top − w_bottom)·A = 0.01·80 m/s · 80² m²
    @printf("  open bottom + top: Q = %.6e m³/s (expected %.6e), residual %.1e, max|∇·u| = %.2e\n",
            led.Q, 0.8 * 80.0^2, led.residual, max_divergence(sim))
    @test led.Q ≈ 0.8 * 80.0^2 rtol = 1e-12
    @test abs(led.residual) < 1e-10
    @test max_divergence(sim) < 1e-12
    # the corrected bottom face: w(z = 40) − δ·(−1) (outward normal −z)
    ii, jj, kk = interior(m)
    @test all(isapprox.(sim.w.data[ii, jj, m.Hz + 1], 0.3 + 0.4 + led.δ; atol = 1e-12))
    @test all(isapprox.(sim.w.data[ii, jj, m.Hz + m.Nz + 1], 0.3 + 1.2 - led.δ; atol = 1e-12))
    # the pressure correction leaves the bottom face alone, checked not assumed
    gz = Field(FaceZ(), m)
    @test assert_open_face_correction(bcs, Field(FaceX(), m), Field(FaceY(), m), gz) === nothing
    gz.data[m.Hx + 2, m.Hy + 2, m.Hz + 1] = 1e-3
    @test_throws ErrorException assert_open_face_correction(bcs, Field(FaceX(), m),
                                                            Field(FaceY(), m), gz)
end
