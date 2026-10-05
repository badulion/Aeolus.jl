using Test
using Printf
using Random
using Aeolus
using Aeolus.Nesting
using Aeolus.Grids: interior
using Aeolus.Fields: set!, interior_view

# =========================================================================
# Phase 30 — parent ↔ child nesting operators (src/Nesting.jl, #250)
# =========================================================================

@testset "Era II — Nesting operators (#250)" begin

# -------------------------------------------------------------------------
# Replaces the Phase 16 `NestSpec` scaffold test. The scaffold asserted a
# verbatim copy (child value = parent value), which holds only for
# refine = 1. Its intent — restricting a hotter child raises the covered
# parent cells to the child's value and leaves the rest alone — is
# re-asserted at r = 3, where the correct statement is "each covered parent
# cell becomes the volume mean of its 27 child cells".
@testset "restricting a hotter child (r = 3): covered cells → child volume mean" begin
    pm = RegularGrid(Float64; size=(8, 8, 8), extent=(1.0, 1.0, 1.0), halo=(2, 2, 2))
    nest = nested_grid(pm, ((0.25, 0.75), (0.25, 0.75), (0.25, 0.75)), 3)
    cm = nest_child_mesh(nest)
    @test (cm.Nx, cm.Ny, cm.Nz) == (12, 12, 12)
    parent = BoussinesqSimulation(pm)
    child  = BoussinesqSimulation(cm)
    set!(parent.θ, (x, y, z) -> 300.0)
    # a hotter child with structure inside each parent cell
    set!(child.θ, (x, y, z) -> 305.0 + 2.0 * sin(2π * x) * cos(2π * y) + z)
    restrict_to_parent!(parent.θ, child.θ, nest)
    cx, cy, cz = nest_covered_range(nest)
    @test (cx, cy, cz) == (3:6, 3:6, 3:6)
    θp = interior_view(parent.θ); θc = interior_view(child.θ)
    maxdev = 0.0
    for K in cz, J in cy, I in cx
        a = (I - first(cx)) * 3; b = (J - first(cy)) * 3; c = (K - first(cz)) * 3
        mean27 = sum(θc[a+1:a+3, b+1:b+3, c+1:c+3]) / 27
        maxdev = max(maxdev, abs(θp[I, J, K] - mean27))
    end
    @printf("  r=3 restriction: max |θ_parent − child volume mean| = %.3e K\n", maxdev)
    @test maxdev < 1e-12
    @test all(>(300.0), θp[cx, cy, cz])     # the covered cells got hotter
    # cells outside the nest are untouched, exactly
    outside = [θp[I, J, K] for K in 1:8, J in 1:8, I in 1:8
               if !((I in cx) && (J in cy) && (K in cz))]
    @test length(outside) == 8^3 - 4^3
    @test all(==(300.0), outside)
end

end # Era II — Nesting operators
