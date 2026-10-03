# Era VI, Phase 59 (P59-d, #185) — spatially developing flat-plate boundary
# layer: the shared damping profile (sponge + fringe), the Blasius solution,
# the fringe region, the free-stream disturbance generator, boundary-layer
# integral diagnostics and bypass transition.

using Aeolus
using Aeolus.Fields: set!, interior_view
using Aeolus.Operators
using Aeolus.Operators: add_sponge!
using Aeolus.Operators.FiniteVolume: in_damping
using Aeolus.StretchedGrids
using Random
using Printf

# ---------------------------------------------------------------------------
# 0. The damping profile shared by add_sponge! and the fringe
# ---------------------------------------------------------------------------

# The Phase-7 sponge kernel as it stood before the shared profile, kept here
# verbatim as the bit-for-bit oracle for the refactored add_sponge!.
function _phase7_sponge_reference!(tf, f, α_max, z_start, z_top, f_ref, zs)
    m = Aeolus.Fields.mesh(f)
    ii, jj, kk = Aeolus.Grids.interior(m)
    inv_width = 1 / (z_top - z_start)
    half_π = 0.5 * π
    for (kk_, kv) in enumerate(kk), j in jj, i in ii
        z = zs[kk_]
        if z >= z_start
            s = sin(half_π * (z - z_start) * inv_width)
            α = α_max * s * s
            tf.data[i, j, kv] -= α * (f.data[i, j, kv] - f_ref)
        end
    end
    return tf
end

@testset "P59-d damping profile (sponge + fringe share one implementation)" begin
    # Smooth step of Nordström et al. (1999), eq. (2.3).
    @test smooth_step(-0.3) == 0 && smooth_step(0.0) == 0
    @test smooth_step(1.0) == 1 && smooth_step(1.7) == 1
    @test smooth_step(0.5) ≈ 0.5 atol = 1e-15
    for x in (0.01, 0.2, 0.37, 0.8, 0.99)
        @test smooth_step(x) + smooth_step(1 - x) ≈ 1 atol = 1e-14   # antisymmetry
    end
    xs = range(0, 1; length = 1001)
    @test all(diff(smooth_step.(xs)) .>= 0)                          # monotone

    # Nordström fringe λ(x): zero outside, λ_max on the plateau.
    p = DampingProfile(NordstromRamp(), 2.0, 10.0, 3.0; stop = 30.0, fall = 2.0)
    @test damping_rate(p, 9.0) == 0 && damping_rate(p, 31.0) == 0
    @test damping_rate(p, 10.0) == 0 && damping_rate(p, 30.0) == 0
    @test damping_rate(p, 13.0) ≈ 2.0 atol = 1e-15
    @test damping_rate(p, 20.0) ≈ 2.0 atol = 1e-15
    @test damping_rate(p, 28.0) ≈ 2.0 atol = 1e-15
    @test damping_rate(p, 11.5) ≈ 1.0 atol = 1e-14    # mid-rise: S(1/2) = 1/2
    @test damping_rate(p, 29.0) ≈ 1.0 atol = 1e-14    # mid-fall

    # add_sponge! is bit for bit the Phase-7 kernel on every location and on
    # Float64 and Float32 grids (the rate now comes from sponge_profile).
    for T in (Float64, Float32)
        g = RegularGrid(T; size = (6, 5, 24), extent = (1, 1, 2))
        for L in (FaceX(), FaceY(), FaceZ(), CellCenter())
            rng = MersenneTwister(7)
            f = Field(L, g); tf = Field(L, g)
            f.data .= T.(randn(rng, size(f.data)))
            tf.data .= T.(randn(rng, size(tf.data)))
            tf_ref = deepcopy(tf)
            zs = collect(Aeolus.Grids.znodes(g, L))
            add_sponge!(tf, f, 0.37, T(1.3), T(2.0); f_ref = 0.2)
            _phase7_sponge_reference!(tf_ref, f, 0.37, T(1.3), T(2.0), 0.2, zs)
            @test tf.data == tf_ref.data
        end
    end
    # The sin² profile equals the Phase-7 formula pointwise.
    sp = sponge_profile(0.5, 1.0, 3.0)
    for z in (0.5, 1.0, 1.7, 2.9, 3.0)
        ref = z >= 1.0 ? 0.5 * sin(0.5π * (z - 1.0) * (1 / 2.0))^2 : 0.0
        @test damping_rate(sp, z) === ref
    end
    @test !in_damping(sp, 0.99) && in_damping(sp, 1.0)
end
