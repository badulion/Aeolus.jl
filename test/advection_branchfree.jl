# Branch-free face values of the flux-form advection (#557).
#
# `AdvectionSchemes._face_x/y/z` gather both upwind-biased stencils and
# pick the upwind one with `ifelse` (`_upwind_value`): for a
# mirror-symmetric reconstruction the inputs of one reconstruction are
# selected, otherwise both are evaluated and the result selected. Either
# way the reconstruction receives the same values in the same order as the
# branching form (`if vel >= 0`) of before, so the face values, and every
# operator built on them, are expected bit-identical. Tolerance: 4 eps of
# the largest value compared, only to allow the compiler a different
# order of an operation; measured 0 everywhere.
#
# The reference is the branching form itself (`Branching`,
# test/support/branching_faces.jl), run through the same operators.

using Test
using Aeolus
using Aeolus.Backends: CPUBackend, with_backend
using Random: Xoshiro, randn, rand

include(joinpath(@__DIR__, "support", "branchfree_cases.jl"))

# a reconstruction whose right-biased form is *not* the left one on the
# mirrored stencil: the face bodies must evaluate both and select
struct _Skewed <: _AS.AbstractReconstruction end
_AS.stencil_left(::_Skewed) = (Val(-1), Val(3))
_AS.stencil_right(::_Skewed) = (Val(0), Val(2))
_AS.reconstruct_left(::_Skewed, t::NTuple{3}) = (2t[1] + 3t[2] + 5t[3]) / 10
_AS.reconstruct_right(::_Skewed, t::NTuple{2}) = 0.75t[1] + 0.25t[2]

@testset "Branch-free face values (#557)" begin
    rng = Xoshiro(5571)
    d = randn(rng, 14, 14, 14)
    # velocities of both signs, and the two zeros (the left stencil, as with `vel >= 0`)
    vels = (1.3, -0.7, 0.0, -0.0, 1e-300, -1e-300)
    worst = 0.0; nface = 0
    for r in (BF_RECONS..., _Skewed()), vel in vels, c in 6:8, j in 6:8, k in 6:8
        for (new, old) in ((_AS._face_x(r, d, c, j, k, vel), _AS._face_x(Branching(r), d, c, j, k, vel)),
                           (_AS._face_y(r, d, j, c, k, vel), _AS._face_y(Branching(r), d, j, c, k, vel)),
                           (_AS._face_z(r, d, j, k, c, vel), _AS._face_z(Branching(r), d, j, k, c, vel)))
            worst = max(worst, abs(new - old)); nface += 1
        end
    end
    @test nface == 8 * 6 * 27 * 3
    @test worst <= 4eps() * maximum(abs, d)

    # the property the selection of the inputs rests on
    @test !_AS.mirror_symmetric(_Skewed())
    for r in BF_RECONS
        @test _AS.mirror_symmetric(r)
        lo, n = _AS.stencil_right(r)
        for _ in 1:200
            t = ntuple(_ -> randn(rng), n)
            @test abs(_AS.reconstruct_right(r, t) - _AS.reconstruct_left(r, reverse(t))) <=
                  4eps() * maximum(abs, t)
        end
    end
    # the centred face takes the left stencil without a choice: valid only
    # while its two stencils are the same cells (integer offsets, exact)
    @test _AS.stencil_left(_AS.Centered2()) === _AS.stencil_right(_AS.Centered2())
    # the skewed reconstruction really differs from its mirror (else the
    # fallback above would not be exercised)
    @test _AS._face_x(_Skewed(), d, 7, 7, 7, -1.0) != _AS.reconstruct_left(_Skewed(), (d[9, 7, 7], d[8, 7, 7], d[7, 7, 7]))

    # one face of the boundary and nest flux registers (`advective_face_flux`)
    for r in BF_RECONS, ax in 1:3, vel in (0.9, -0.9)
        veld = fill(vel, size(d))
        new = _AS.advective_face_flux(r, d, veld, ax, 7, 7, 7)
        old = _AS.advective_face_flux(Branching(r), d, veld, ax, 7, 7, 7)
        @test abs(new - old) <= 4eps() * maximum(abs, d)
    end
end

@testset "Branch-free faces: every operator against the branching form (#557)" begin
    for (bname, b) in (("serial", CPUBackend(serial = true)), ("threaded", CPUBackend(threshold = 0)))
        res = bf_compare(b)
        @test length(res) == 3 * length(BF_RECONS) + 1
        for (name, scheme, rel) in res
            rel <= 4eps() || @info "branch-free operator disagrees" bname name scheme rel
            @test rel <= 4eps()
        end
    end
end

@testset "Branch-free faces: the limiter's ratio where nothing flows out (#557)" begin
    # Pm = 0 (no antidiffusive outflow): the ratio is 1, the discarded Q/Pm
    # (0/0 or Q/0) does not reach it
    g = RegularGrid(; size = (4, 4, 4), extent = (4.0, 4.0, 4.0), halo = (3, 3, 3))
    ϕ = Field(CellCenter(), g)
    Fx = Field(FaceX(), g); Fy = Field(FaceY(), g); Fz = Field(FaceZ(), g)
    ϕ.data .= 1.0
    apply_positivity_limiter!(Fx, Fy, Fz, ϕ, 1.0, g; ratio = (R = Field(CellCenter(), g)))
    @test all(isone, Aeolus.Fields.interior_view(R))
    ϕ.data .= 0.0
    apply_positivity_limiter!(Fx, Fy, Fz, ϕ, 1.0, g; ratio = R)
    @test all(isone, Aeolus.Fields.interior_view(R))
end

# The gate runs this file at one thread, where the threaded backend runs
# its rows one after another; the operators at -t 4 run in a subprocess
# (four reconstructions: what runs concurrently does not depend on which).
const _BF_THREADED = """
using Aeolus
include($(repr(joinpath(@__DIR__, "support", "branchfree_cases.jl"))))
res = bf_compare(CPUBackend(threshold = 0); n = (40, 36, 12),
                 recons = (_AS.WENOZ(5, 2), _AS.TENO(5), _AS.Centered2(), _AS._UpwindFirst()))
print("threads ", Threads.nthreads(), " cases ", length(res), " worst ", maximum(r[3] for r in res))
"""

@testset "Branch-free faces: the operators at -t 4 (#557)" begin
    project = dirname(Base.active_project())
    cmd = `$(Base.julia_cmd()) --startup-file=no -t 4 --project=$project -e $_BF_THREADED`
    out = IOBuffer(); err = IOBuffer()
    p = run(pipeline(ignorestatus(cmd); stdout = out, stderr = err))
    msg = String(take!(out))
    success(p) || @info "advection_branchfree.jl: the -t 4 subprocess failed" String(take!(err))
    @test success(p)
    m = match(r"threads (\d+) cases (\d+) worst ([0-9.eE+-]+)", msg)
    @test m !== nothing
    if m !== nothing
        @test parse(Int, m[1]) == 4
        @test parse(Int, m[2]) == 3 * 4 + 1
        @test parse(Float64, m[3]) <= 4eps()
    end
end
