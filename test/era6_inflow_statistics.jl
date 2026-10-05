# N-inflow-b (#258): Reynolds stresses, lengths and times for the synthetic
# inflow of the child boundaries — the `InflowStatistics` contract, the
# parent-resolved statistics (resolved + SGS, scale-gap mode) and the
# parametrized statistics of a mesoscale parent (Rotach et al. 1996 blend).

using Test, Printf, Random
using Aeolus
using Aeolus.Fields: interior_view

@testset "N-inflow-b sgs_tke!: every closure's subgrid TKE" begin
    m = RegularGrid(; size = (8, 8, 8), extent = (80.0, 80.0, 40.0), halo = (3, 3, 3))
    sim = BoussinesqSimulation(m; C_s = 0.17,
                               bcs = FieldBoundaryConditions(:periodic, :periodic, :periodic))
    # uniform shear u = s z: |S| = s, Smagorinsky ν_t = (C_s Δ)² s
    s = 0.02
    zc = [(k - 0.5) * m.dz for k in 1:m.Nz]
    for k in 1:m.Nz
        sim.u.data[:, :, m.Hz+k] .= s * zc[k]
    end
    Aeolus.BoundaryConditions.fill_halos!(sim.u, sim.bcs)
    c = StaticSmagorinsky(0.17)
    f = sgs_fields(c, sim)
    sgs_coefficients!(f, c, sim)
    e = Aeolus.Fields.Field(Aeolus.Grids.CellCenter(), m)
    sgs_tke!(e, c, sim)
    Δ = cbrt(m.dx * m.dy * m.dz)
    # interior levels (the periodic wrap of u = s z breaks the shear at k = 1, Nz)
    ei = interior_view(e)[:, :, 2:end-1]
    @test maximum(abs, ei ./ (0.17^2 * Δ * s / 0.1)^2 .- 1) < 1e-12
    # the built-in closure (c = nothing) reads sim.νt
    interior_view(sim.νt) .= interior_view(f.νh)
    e2 = Aeolus.Fields.Field(Aeolus.Grids.CellCenter(), m)
    sgs_tke!(e2, nothing, sim)
    @test interior_view(e2) == interior_view(e)
    # prognostic closures return their memory
    tk = TKEClosure(m; e0 = 0.37)
    sgs_tke!(e, tk, sim)
    @test all(==(0.37), interior_view(e))
    rs = ReynoldsStressClosure(m; e0 = 0.37)
    sgs_tke!(e, rs, sim)
    @test maximum(abs, interior_view(e) .- 0.37) < 1e-15
end
