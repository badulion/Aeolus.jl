# The Berlin parent's tiled surface (N-parent-d, #249): the unresolved
# buildings' roughness (Macdonald et al. 1998), the OSM multipolygon
# ingest of the 6 km extract, the class-to-surface factory shared with the
# child, the per-column mosaic, and its consistency with the child under
# the child's footprint.

using Test
using Aeolus
using Aeolus.SurfaceTiles: macdonald_roughness

@testset "Berlin parent surface (#249)" begin

@testset "Macdonald et al. (1998) roughness of unresolved buildings" begin
    # d/h = 1 + A^(-λp)(λp - 1): at λp = 0.3, A = 4.43 → 1 - 0.7·4.43^(-0.3) = 0.5521
    r = macdonald_roughness(0.3, 0.3, 1.0)
    @test isapprox(r.d, 0.552, atol = 1e-3)
    @test r.d ≈ 1 - 0.7 * 4.43^(-0.3) rtol = 1e-14
    # z0 and d scale with h
    r20 = macdonald_roughness(0.3, 0.2, 20.0)
    r1 = macdonald_roughness(0.3, 0.2, 1.0)
    @test r20.z0 ≈ 20r1.z0 rtol = 1e-14
    @test r20.d ≈ 20r1.d rtol = 1e-14
    # the maximum of z0/h over cube arrays (λf = λp) lies at intermediate
    # density, λf ≈ 0.15-0.2 (Grimmond & Oke 1999, JAM 38, 1262)
    λs = 0.01:0.001:0.99
    z0s = [macdonald_roughness(λ, λ, 1.0).z0 for λ in λs]
    λpk = λs[argmax(z0s)]
    @test 0.15 <= λpk <= 0.2
    @test 0.1 < maximum(z0s) < 0.15          # z0/h ≈ 0.135 at the peak (measured)
    # limits: no buildings → d, z0 → 0; a closed roof → d → h, z0 → 0
    r0 = macdonald_roughness(1e-9, 1e-9, 25.0)
    @test r0.d < 1e-7 && r0.z0 < 1e-12
    @test macdonald_roughness(0.0, 0.0, 25.0) == (z0 = 0.0, d = 0.0)
    r1 = macdonald_roughness(1 - 1e-9, 0.5, 25.0)
    @test abs(r1.d - 25.0) < 1e-6 && r1.z0 < 1e-6
    @test macdonald_roughness(1.0, 0.5, 25.0).d == 25.0
    # d increases monotonically with λp (sheltering), 0 ≤ z0 < h - d
    ds = [macdonald_roughness(λ, 0.2, 1.0).d for λ in λs]
    @test all(diff(ds) .> 0)
    @test all(λ -> (local q = macdonald_roughness(λ, λ, 1.0); 0 <= q.z0 < 1 - q.d), λs)
    # square (aligned) arrays, A = 3.59, are less displaced than staggered ones
    @test macdonald_roughness(0.3, 0.3, 1.0; A = 3.59, β = 0.55).d < r.d
    @test_throws ArgumentError macdonald_roughness(1.2, 0.1, 10.0)
    @test_throws ArgumentError macdonald_roughness(0.2, -0.1, 10.0)
end

end # testset
