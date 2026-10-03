# Era VI, Phase 56 (P56-b, #182): the Deardorff (1980) 1.5-order prognostic
# SGS TKE closure (src/PrognosticSGS.jl) with the Moeng (1984) coefficients,
# and the wall distance it caps its mixing length with (IBM.wall_distance).
#
# Numbers measured on this branch are quoted next to each assertion.

using Test
using Aeolus
using Aeolus.Fields: interior_view, set!
using Aeolus.IBM: SolidMask, mark_solid!, wall_distance, first_level_cells

@testset "Era VI — prognostic SGS TKE (P56-b, #182)" begin

@testset "wall_distance: vector distance transform" begin
    # floor only: d = Δz/2 + (k-1)Δz exactly
    m = RegularGrid(; size = (8, 8, 12), extent = (16.0, 16.0, 6.0))
    d = wall_distance(SolidMask(m); bottom = true)
    @test size(d) == (8, 8, 12)
    @test all(d[i, j, k] == 0.25 + (k - 1) * 0.5 for i in 1:8, j in 1:8, k in 1:12)
    # floor and lid: the nearer of the two
    d2 = wall_distance(SolidMask(m); bottom = true, top = true)
    @test all(d2[1, 1, k] ≈ min(0.25 + (k - 1) * 0.5, 0.25 + (12 - k) * 0.5) for k in 1:12)
    # no surface at all: infinite
    @test all(isinf, wall_distance(SolidMask(m); bottom = false))

    # a cube on a 2:2:1 grid, against the Euclidean distance to its box
    m = RegularGrid(; size = (24, 24, 24), extent = (48.0, 48.0, 24.0))
    sm = SolidMask(m)
    mark_solid!(sm, (x, y, z) -> 16 < x < 32 && 16 < y < 32 && z < 12)
    d = wall_distance(sm; bottom = false, periodic = (false, false, false))
    err = 0.0
    nsolid = 0
    for k in 1:24, j in 1:24, i in 1:24
        if sm.mask[m.Hx+i, m.Hy+j, m.Hz+k]
            nsolid += 1
            @test d[i, j, k] == 0
            continue
        end
        x, y, z = (i - 0.5) * 2, (j - 0.5) * 2, (k - 0.5)
        ex = sqrt(max(16 - x, 0, x - 32)^2 + max(16 - y, 0, y - 32)^2 + max(z - 12, 0)^2)
        err = max(err, abs(d[i, j, k] - ex) / ex)
    end
    @test nsolid == 8 * 8 * 12
    @test err < 1e-12                       # measured 0.0 (a plain chamfer: 0.236)

    # periodic seam: a wall at the east edge is seen from the west edge
    m = RegularGrid(; size = (16, 4, 4), extent = (16.0, 4.0, 4.0))
    sm = SolidMask(m)
    mark_solid!(sm, (x, y, z) -> x > 15)
    d = wall_distance(sm; bottom = false)
    @test d[1, 2, 2] ≈ 0.5 && d[3, 2, 2] ≈ 2.5 && d[14, 2, 2] ≈ 1.5
    dnp = wall_distance(sm; bottom = false, periodic = (false, false, false))
    @test dnp[1, 2, 2] ≈ 14.5

    # first level above a surface: level 1 and the cells above a roof
    m = RegularGrid(; size = (8, 8, 8), extent = (8.0, 8.0, 8.0))
    sm = SolidMask(m)
    mark_solid!(sm, (x, y, z) -> x < 4 && z < 3)
    f = first_level_cells(sm)
    @test all(f[i, j, 4] for i in 1:4, j in 1:8)           # above the roof (3 solid levels)
    @test all(f[i, j, 1] for i in 5:8, j in 1:8)           # on the floor
    @test count(f) == 4 * 8 + 4 * 8
    @test !any(f[i, j, 1] for i in 1:4, j in 1:8)          # solid cells are not fluid
end

end
