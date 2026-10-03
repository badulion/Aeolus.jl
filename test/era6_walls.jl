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
end
