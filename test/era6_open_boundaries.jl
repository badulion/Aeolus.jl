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
using Aeolus.Simulations: BoussinesqSimulation

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
