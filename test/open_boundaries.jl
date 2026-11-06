# N-open-b (#255, epic #246): open (Dirichlet-from-boundary-data) lateral
# and top faces and the single global boundary mass-flux correction.
#
#   the mass-flux correction of the boundary update every core calls
#   before its projection: a prescribed net imbalance Q is recorded and
#   removed (uniformly, or from the outflow only), a balanced uniform wind
#   needs none. The stepped runs (Q/V divergence without the correction,
#   uniform flow kept through 200 steps of every core) were cut from the
#   test suite (2026-11 reduction: an open-boundary step compiles for 8 s).
#
# References: Kim & Moin (1985), JCP 59, 308; Hellsten et al. (2021),
# GMD 14, 3185, §3.3 Eq. 2 (uniform mass-flux correction).

using Test
using Aeolus
using Aeolus.Fields: interior_view, set!
using Aeolus.Simulations: BoussinesqSimulation, step_rk3!
using Aeolus.IO: checkpoint_payload, restore_checkpoint!   # #528

# ---------------------------------------------------------------------------
# 3. A prescribed net imbalance: u = 1 + 0.2 x/Lx on the faces gives an
# outflow Q = 0.2 Ly Lz through the east face (plus nothing else).
const _Lx, _Ly, _Lz = 800.0, 600.0, 400.0
const _Q_IMB = 0.2 * _Ly * _Lz
_imb_data() = FunctionBoundaryData(u = (x, y, z, t) -> 1.0 + 0.2 * x / _Lx,
                                   v = 0.0, w = 0.0, θ = 300.0)

@testset "mass-flux correction: the data's imbalance Q removed, uniformly" begin
    m = RegularGrid(size = (16, 12, 8), extent = (_Lx, _Ly, _Lz))
    mk(corr) = (s = BoussinesqSimulation(m; bcs = open_boundary_conditions(m, _imb_data();
                                                   correction = corr, record = true),
                                          θ₀ = 300.0);
                set!(s.θ, 300.0); set!(s.u, 1.0); s)
    # the boundary update every core calls before its projection
    s0 = mk(:none)
    Aeolus.BoundaryConditions.update_open_boundaries!(s0, 0.0, 1.0)
    L = open_boundary_ledger(s0)
    @test L.Q ≈ _Q_IMB rtol = 1e-12
    @test L.residual == L.Q                         # nothing corrected
    s1 = mk(:uniform)
    Aeolus.BoundaryConditions.update_open_boundaries!(s1, 0.0, 1.0)
    L1 = open_boundary_ledger(s1)
    @test L1.Q ≈ _Q_IMB rtol = 1e-12
    @test abs(L1.residual) < 1e-9 * _Q_IMB
    # δ = Q / (total open area: west, east, south, north and top)
    @test L1.δ ≈ _Q_IMB / (2 * (_Ly * _Lz + _Lx * _Lz) + _Lx * _Ly) rtol = 1e-12
    # :outflow_only cuts back the outflow and leaves the inflow face as given
    s2 = mk(:outflow_only)
    Aeolus.BoundaryConditions.update_open_boundaries!(s2, 0.0, 1.0)
    @test abs(open_boundary_ledger(s2).residual) < 1e-9 * _Q_IMB
    H = m.Hx
    @test all(≈(1.0), s2.bcs.west.planes[:u].data[H+1, H+1:H+m.Ny, H+1:H+m.Nz])
    # a uniform wind through the box: the data balances, nothing to correct
    s3 = BoussinesqSimulation(m; bcs = open_boundary_conditions(m, ProfileBoundaryData(u = 5.0, v = 2.0, w = 0.0, θ = 300.0)))
    Aeolus.BoundaryConditions.update_open_boundaries!(s3, 0.0, 1.0)
    @test abs(open_boundary_ledger(s3).Q) < 1e-9 * 5.0 * _Ly * _Lz && abs(open_boundary_ledger(s3).δ) < 1e-12
end

# ---------------------------------------------------------------------------
# #528: a checkpoint of an open box continues the run. The open boundaries'
# buffers (radiated values at the outflow points), the Orlanski history and
# the binding of θ to its planes are state; without them a restart refilled
# the buffers from the data and the outflow face jumped.
_ob_data() = FunctionBoundaryData(u = (x, y, z, t) -> 2.0 + 0.3 * x / _Lx + 0.5 * sin(2π * t / 60),
                                  v = (x, y, z, t) -> 0.5 + 0.2 * cos(2π * t / 45), w = 0.0,
                                  θ = (x, y, z, t) -> 300.0 + 0.01 * z + 0.2 * sin(2π * t / 30))
function _ob_sim()
    m = RegularGrid(size = (16, 12, 8), extent = (_Lx, _Ly, _Lz))
    s = BoussinesqSimulation(m; bcs = open_boundary_conditions(m, _ob_data(); outflow = :orlanski), θ₀ = 300.0)
    set!(s.u, 2.0); set!(s.v, 0.5); set!(s.θ, 300.0)
    interior_view(s.θ) .+= 0.1 .* sin.((1:16) .* 0.7) .* cos.((1:12)' .* 0.5)   # a fixed perturbation
    return s
end
_ob_dif(a, b) = max(maximum(abs, interior_view(a.u) .- interior_view(b.u)),
                    maximum(abs, interior_view(a.θ) .- interior_view(b.θ)))

@testset "open box: a checkpoint continues the run (#528)" begin
    dt = 0.5
    a = _ob_sim()
    for _ in 1:6; step_rk3!(a, dt); end
    pay = checkpoint_payload(a)
    @test pay[:open_boundaries] isa Vector && pay[:open_boundaries][1].initialized
    @test any(r -> r !== nothing && r.has_hist, pay[:open_boundaries][1].radiation)   # the Orlanski history is in it
    old = copy(pay); delete!(old, :open_boundaries)                                  # a checkpoint from before #528
    for _ in 1:4; step_rk3!(a, dt); end
    # restored into a fresh build: the same steps on the same state, so the
    # same values to round-off (one thread; 1e-12 m/s, K is far below the
    # 1e-1 jump of the old restart)
    b = _ob_sim(); restore_checkpoint!(b, pay)
    for _ in 1:4; step_rk3!(b, dt); end
    @test _ob_dif(a, b) <= 1e-12
    # the old payload: the open faces restart from the data, warned, and the
    # run does not continue (measured 0.26 m/s, 0.99 K after one step)
    c = _ob_sim()
    @test_logs (:warn, r"holds no open-boundary state") match_mode = :any restore_checkpoint!(c, old)
    for _ in 1:4; step_rk3!(c, dt); end
    @test _ob_dif(a, c) > 1e-2
    # an open box built differently (no radiating faces) is refused
    m = RegularGrid(size = (16, 12, 8), extent = (_Lx, _Ly, _Lz))
    d = BoussinesqSimulation(m; bcs = open_boundary_conditions(m, _ob_data()), θ₀ = 300.0)
    @test_throws ArgumentError restore_checkpoint!(d, pay)
end
