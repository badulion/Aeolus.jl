# The cases of test/advection_branchfree.jl (#557): every operator of
# AdvectionSchemes that takes a reconstruction descriptor, run with the
# branch-free face bodies and with the branching ones of before
# (`Branching`, test/support/branching_faces.jl), on random fields. Included
# by the test file and by its -t 4 subprocess.

using Aeolus
using Aeolus.Backends: CPUBackend, with_backend
using Random: Xoshiro, randn, rand
include(joinpath(@__DIR__, "branching_faces.jl"))

const BF_RECONS = (_AS.WENOZ(5, 2), _AS.WENOZ(7, 2), _AS.TENO(5), _AS.TENO(8), _AS.WENOJS(5),
                   _AS.Centered2(), _AS._UpwindFirst())

# random storage data (halos included, so no fill is needed); velocities of
# both signs with exact zeros of both signs (the upwind choice at vel = 0
# and −0.0 is the left stencil, as `vel >= 0` chose it)
function bf_fields(g; seed = 557)
    rng = Xoshiro(seed)
    u = Field(FaceX(), g); v = Field(FaceY(), g); w = Field(FaceZ(), g)
    for f in (u, v, w)
        d = f.data
        d .= 2 .* randn(rng, size(d)...)
        z = rand(rng, size(d)...)
        d[z .< 0.05] .= 0.0
        d[0.05 .<= z .< 0.1] .= -0.0
    end
    ϕ = Field(CellCenter(), g)
    ϕ.data .= rand(rng, size(ϕ.data)...) .* (rand(rng, size(ϕ.data)...) .> 0.2)   # zeros: the limiter acts
    return u, v, w, ϕ
end

_bf_rel(a, b) = maximum(abs.(a .- b)) / max(maximum(abs, b), floatmin())

# The relative max difference of each operator, branch-free against
# branching, on the backend `b`: a vector of (name, scheme, difference).
function bf_compare(b; n = (16, 12, 10), recons = BF_RECONS)
    out = Tuple{String,Symbol,Float64}[]
    g = RegularGrid(; size = n, extent = (160.0, 120.0, 100.0), halo = (5, 5, 5))
    sg = StretchedGrid(; size = n, extent = (160.0, 120.0, 300.0), halo = (2, 2, 2), stretching = (:tanh, 1.5))
    u, v, w, ϕ = bf_fields(g)
    ws = FluxLimiterWorkspace(g)
    dt = 0.5
    with_backend(b) do
        for r in recons, (name, op!) in (
                ("advect_positive!", (t, r) -> advect_positive!(t, u, v, w, ϕ, dt; scheme = r, workspace = ws)),
                ("advect (reconstructed)", (t, r) -> _AS._advect_reconstructed!(t, u, v, w, ϕ, r, g)))
            t1 = Field(CellCenter(), g); t2 = Field(CellCenter(), g)
            op!(t1, r); op!(t2, Branching(r))
            push!(out, (name, _AS.scheme_name(r), _bf_rel(t1.data, t2.data)))
        end
        for r in recons
            tn = (Field(FaceX(), g), Field(FaceY(), g), Field(FaceZ(), g))
            to = (Field(FaceX(), g), Field(FaceY(), g), Field(FaceZ(), g))
            _AS._momentum_advect_recon!(tn..., u, v, w, r, g)
            _AS._momentum_advect_recon!(to..., u, v, w, Branching(r), g)
            push!(out, ("momentum", _AS.scheme_name(r), maximum(_bf_rel(a.data, b.data) for (a, b) in zip(tn, to))))
        end
        us, vs, ws_, ϕs = bf_fields(sg)
        wss = FluxLimiterWorkspace(sg)
        t1 = Field(CellCenter(), sg); t2 = Field(CellCenter(), sg)
        r = _AS.Centered2()
        advect_positive!(t1, us, vs, ws_, ϕs, dt; scheme = r, workspace = wss)
        advect_positive!(t2, us, vs, ws_, ϕs, dt; scheme = Branching(r), workspace = wss)
        push!(out, ("advect_positive! stretched", :centered, _bf_rel(t1.data, t2.data)))
    end
    return out
end
