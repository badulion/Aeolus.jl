# ONE-g (#456): the DNS-against-LES example (examples/dns_vs_les/), at a
# tiny size: the named analytic initial field `isotropic_turbulence`
# ([initial] field, Operators.isotropic_turbulence_field!) is defined in
# spectral space, so the DNS field truncated to the LES modes is the LES
# field to round-off; and the runs write the resolved kinetic energy, the
# dissipation (molecular + SGS) and the energy spectrum the comparison
# script reads. The 256³ / 64³ runs and their numbers are the example's
# (the PR of ONE-g).

using Test
using Aeolus
using Aeolus.Config: load_run_config, build, run_config
using Aeolus.SimulationDriver: SimulationDriverType, simulation
using Aeolus.Fields: interior_view
using Aeolus.Grids: xnodes, ynodes, znodes
using FFTW: fft, ifft, fftfreq

const DVL = joinpath(pkgdir(Aeolus), "examples", "dns_vs_les")

"The sharp spectral truncation of the staggered field `f` (on `gf`) to the modes of `gc`, at the positions of `gc`."
function dvl_truncate(f, gf, gc, loc)
    F = fft(Array(interior_view(f)))
    Nf = (gf.Nx, gf.Ny, gf.Nz); Nc = (gc.Nx, gc.Ny, gc.Nz)
    xf = (first(xnodes(gf, loc)), first(ynodes(gf, loc)), first(znodes(gf, loc)))
    xc = (first(xnodes(gc, loc)), first(ynodes(gc, loc)), first(znodes(gc, loc)))
    k0 = (2π / gf.Lx, 2π / gf.Ly, 2π / gf.Lz)
    mf = map(n -> Int.(fftfreq(n, n)), Nf); mc = map(n -> Int.(fftfreq(n, n)), Nc)
    idx(m, d) = mod(m, Nf[d]) + 1
    G = zeros(ComplexF64, Nc...)
    for c in 1:Nc[3], b in 1:Nc[2], a in 1:Nc[1]
        m = (mc[1][a], mc[2][b], mc[3][c])
        any(d -> 2abs(m[d]) >= Nc[d], 1:3) && continue
        ph = sum(k0[d] * m[d] * (xc[d] - xf[d]) for d in 1:3)
        G[a, b, c] = F[idx(m[1], 1), idx(m[2], 2), idx(m[3], 3)] * cis(ph) * prod(Nc) / prod(Nf)
    end
    return real.(ifft(G))
end

dvl_sim(name, n) = simulation(build(SimulationDriverType(),
    load_run_config(joinpath(DVL, name * ".toml"); overrides = ["grid.size=[$n,$n,$n]", "time.duration=\"none\"",
                                                                  "time.nsteps=3"])))

@testset "DNS against LES: the isotropic-turbulence initial field and the outputs (ONE-g)" begin

@testset "the spectrum and the field, defined in spectral space" begin
    E = PassotPouquetSpectrum(1.0, 1.5)
    # ∫ E dk = 3/2 u_rms² (trapezoid on a fine k grid), maximum at k_peak
    ks = range(0, 20; length = 20001)
    @test sum(E.(ks)) * step(ks) ≈ 1.5 rtol = 1e-8
    @test argmax(E.(ks)) == findfirst(>=(1.5), ks)
    # the field of a 32³ grid truncated to the modes of a 16³ one is the 16³
    # field (same seed, same box) to round-off: before the projection, the
    # fields isotropic_turbulence_field! sets
    mk(n) = (g = RegularGrid(; size = (n, n, n), extent = (2π, 2π, 2π)); (g, Field(FaceX(), g), Field(FaceY(), g), Field(FaceZ(), g)))
    gf, uf, vf, wf = mk(32); gc, uc, vc, wc = mk(16)
    isotropic_turbulence_field!(uf, vf, wf, E; seed = 3)
    isotropic_turbulence_field!(uc, vc, wc, E; seed = 3)
    for (f, c, loc) in ((uf, uc, FaceX()), (vf, vc, FaceY()), (wf, wc, FaceZ()))
        @test maximum(abs, dvl_truncate(f, gf, gc, loc) .- interior_view(c)) < 1e-14 * maximum(abs, interior_view(c))
    end
    # another seed is another field; the shells hold E(k) Δk in expectation:
    # the 32³ field's energy in shells 1-15 against ∫ E over them
    isotropic_turbulence_field!(uc, vc, wc, E; seed = 4)
    @test maximum(abs, dvl_truncate(uf, gf, gc, FaceX()) .- interior_view(uc)) > 0.1
    k, Ek = isotropic_energy_spectrum(uf, vf, wf)
    @test 0.5 < sum(Ek[2:16]) / sum(E.(k[2:16])) < 2.0      # one realisation of the random phases
    # continuous-solenoidal: the spectral divergence vanishes mode by mode
    # (the staggered one is O((kΔ)²): the model projects it once)
    @test maximum(abs, interior_view(uf)) > 0.5
end

@testset "the configs: the DNS truncated to the LES grid is the LES start" begin
    dns = dvl_sim("dns", 32); les = dvl_sim("les_smagorinsky", 16)
    # after the model's one projection of each start (O((kΔ)²) of the divergent
    # part, different on the two grids) the large scales agree: the energy of
    # the shells 1-4 within 1 %, and the truncated DNS field within 2 % of the LES
    kd, Ed = isotropic_energy_spectrum(dns.u, dns.v, dns.w); kl, El = isotropic_energy_spectrum(les.u, les.v, les.w)
    @test all(isapprox(Ed[n], El[n]; rtol = 1e-2) for n in 2:5)
    t = dvl_truncate(dns.u, dns.mesh, les.mesh, FaceX())
    @test maximum(abs, t .- interior_view(les.u)) < 0.02 * maximum(abs, interior_view(les.u))
    @test les.ν == dns.ν == 2.0e-3 && !dns.has_buoyancy
end

@testset "the runs write the kinetic energy, the dissipation and the spectrum" begin
    root = mktempdir()
    out = Dict{String,Any}()
    for (name, n) in (("dns", 16), ("les_smagorinsky", 16))
        cfg = load_run_config(joinpath(DVL, name * ".toml");
                              overrides = ["grid.size=[$n,$n,$n]", "time.duration=0.3", "output.cadence=0.1",
                                           "fftw.planner=\"estimate\""])
        r = run_config(cfg; root = root, console = false, repo = mktempdir())
        out[name] = NCDataset(joinpath(r.folder.path, "output", "output.nc")) do ds
            (K = Float64.(ds["kinetic_energy"][:]), ε = Float64.(ds["dissipation"][:]),
             εs = Float64.(ds["dissipation_sgs"][:]), E = Float64.(ds["energy_spectrum"][:, :]),
             k = Float64.(ds["k"][:]), t = Float64.(ds["time_series"][:]))
        end
    end
    d, l = out["dns"], out["les_smagorinsky"]
    @test d.k == l.k && d.k[2] == 1.0                                   # unit shells of the 2π box
    @test size(d.E, 2) == 4                                             # t = 0, 0.1, 0.2, 0.3
    @test d.K[1] == l.K[1]                                              # the same start (16³ both)
    # the spectrum is the kinetic energy (Parseval) at every record
    @test all(isapprox(sum(d.E[:, j]), d.K[round(Int, 0.1 * (j - 1) / 0.05) + 1]; rtol = 1e-6) for j in 1:4)
    # decay: the kinetic energy falls, faster with the SGS model; the DNS has
    # no SGS dissipation, the LES a positive one
    @test all(diff(d.K) .< 0) && all(diff(l.K) .< 0) && l.K[end] < d.K[end]
    @test all(iszero, d.εs) && all(>(0), l.εs[2:end])
    @test all(>(0), d.ε)
    # the energy budget: -dK/dt ≈ ε (centred differences, the dissipation of
    # the second-order operators; within 15 % at 16³)
    for (o, name) in ((d, "dns"), (l, "les"))
        r = [(o.K[i-1] - o.K[i+1]) / (o.t[i+1] - o.t[i-1]) / o.ε[i] for i in 3:length(o.t)-1]
        @test all(x -> 0.85 < x < 1.15, r)
    end
end

end # testset
