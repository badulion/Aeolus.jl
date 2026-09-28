# Era II numerics swap-ins used by the Berlin case (#132):
#
#   * the dynamic Smagorinsky closure (Germano et al. 1991 / Lilly 1992),
#     moved to src/Operators.jl from the urban example;
#   * the `momentum_advection` and `eddy_viscosity!` keywords of step!,
#     which swap the momentum-advection operator and the SGS closure
#     without touching the algorithm.

using Test
using Aeolus
using Aeolus.Fields: interior_view, set!
using FFTW: ifft, fftfreq
using Random: Xoshiro, randn
using Statistics: mean
using Printf

# A solenoidal random field with E(k) ∝ k^(-5/3) up to k_c and random
# phases, sampled at the staggered face positions.
function _kolmogorov_field!(sim, rng; kc = sim.mesh.Nx / 4)
    m = sim.mesh; N = m.Nx; L = m.Lx; d = L / N
    ks = fftfreq(N, N)
    û = zeros(ComplexF64, N, N, N); v̂ = zero(û); ŵ = zero(û)
    for c in CartesianIndices(û)
        kv = (ks[c[1]], ks[c[2]], ks[c[3]])
        k = sqrt(sum(abs2, kv))
        (k == 0 || k > kc) && continue
        a = randn(rng, ComplexF64, 3)
        n = collect(kv) ./ k
        a .-= n .* sum(n .* a)                     # ⟂ k: divergence-free
        amp = sqrt(k^(-5/3) / (4π * k^2))
        û[c], v̂[c], ŵ[c] = amp .* a
    end
    shift(h, sx, sy, sz) = [h[c] * cis(2π / L * (ks[c[1]] * sx + ks[c[2]] * sy + ks[c[3]] * sz))
                            for c in CartesianIndices(h)]
    interior_view(sim.u) .= real.(ifft(shift(û, 0, d/2, d/2)))
    interior_view(sim.v) .= real.(ifft(shift(v̂, d/2, 0, d/2)))
    interior_view(sim.w) .= real.(ifft(shift(ŵ, d/2, d/2, 0)))
    s = 1 / sqrt(mean(abs2, interior_view(sim.u)))
    for f in (sim.u, sim.v, sim.w)
        interior_view(f) .*= s                     # u_rms = 1
        fill_halos!(f, sim.bcs)
    end
    return sim
end

function _dynamic_cs(sim; kw...)
    ws = DynamicSmagorinskyWorkspace(sim.mesh)
    νt = Aeolus.Fields.Field(CellCenter(), sim.mesh)
    for f in (sim.u, sim.v, sim.w)
        fill_halos!(f, sim.bcs)
    end
    compute_dynamic_smagorinsky_νt!(νt, sim.u, sim.v, sim.w; workspace = ws, kw...)
    return mean(sqrt.(ws.cs2)), ws, νt
end

@testset "Era II — SGS and momentum-advection swap-ins (#132)" begin

    @testset "dynamic Smagorinsky: vanishes in uniform flow and in laminar shear" begin
        mesh = RegularGrid(; size = (16, 16, 16), extent = (1.0, 1.0, 1.0), halo = (3, 3, 3))
        sim = BoussinesqSimulation(mesh; bcs = FieldBoundaryConditions(:periodic))
        set!(sim.u, (x, y, z) -> 2.0); set!(sim.v, (x, y, z) -> -1.0); set!(sim.w, (x, y, z) -> 0.5)
        cs, ws, νt = _dynamic_cs(sim)
        @test all(==(0), interior_view(νt)) && all(==(0), ws.cs2)

        # u = S z: the resolved field is linear, the test-filtered and the
        # grid-level stresses agree (Germano identity with L_ij M_ij = 0), so
        # the dynamic coefficient is zero where the static Smagorinsky
        # model would still give ν_t = (C_s Δ)² S. Walls in z; the levels
        # whose stencils see the wall halos are excluded.
        mw = RegularGrid(; size = (8, 8, 16), extent = (1.0, 1.0, 2.0), halo = (3, 3, 3))
        sw = BoussinesqSimulation(mw; bcs = FieldBoundaryConditions(:periodic, :periodic, :wall))
        set!(sw.u, (x, y, z) -> 3.0 * z)
        _, wsl, _ = _dynamic_cs(sw; averaging = :local)
        @test maximum(wsl.cs2[:, :, 3:end-2]) == 0
        compute_smagorinsky_νt!(sw.νt, sw.u, sw.v, sw.w, 0.17)
        @test minimum(interior_view(sw.νt)[:, :, 3:end-2]) > 0
    end

    mesh = RegularGrid(; size = (32, 32, 32), extent = (2π, 2π, 2π), halo = (3, 3, 3))

    @testset "dynamic Smagorinsky: same |S| as the static model, Galilean invariant" begin
        sim = BoussinesqSimulation(mesh)
        _kolmogorov_field!(sim, Xoshiro(3))
        _, ws, νt = _dynamic_cs(sim; averaging = :local)
        compute_smagorinsky_νt!(sim.νt, sim.u, sim.v, sim.w, 1.0)       # Δ²|S|
        @test interior_view(νt) ≈ ws.cs2 .* interior_view(sim.νt) rtol = 1e-13
        @test all(c -> 0 <= c <= 0.25, ws.cs2)                            # clip [0, 0.5²]
        # a uniform translation changes neither L_ij nor M_ij
        cs0 = copy(ws.cs2)
        interior_view(sim.u) .+= 5.0; interior_view(sim.v) .-= 3.0; interior_view(sim.w) .+= 1.0
        _, ws2, _ = _dynamic_cs(sim; averaging = :local)
        @test maximum(abs, ws2.cs2 .- cs0) < 1e-9 * maximum(cs0)
    end

    @testset "dynamic Smagorinsky: C_s ≈ Lilly's value in developed turbulence, ≈ 0 for random phases" begin
        # The coefficient measures the energy transfer to subfilter
        # scales. A random-phase field with a k^(-5/3) spectrum has the
        # right spectrum but no transfer, so C_s ≈ 0; after the same field
        # has evolved for ~1 eddy turnover under the resolved dynamics
        # (static Smagorinsky LES) the dynamic C_s approaches the
        # inertial-range value 0.17 of Lilly (1967).
        sim = BoussinesqSimulation(mesh; C_s = 0.17, ν = 1e-4)
        _kolmogorov_field!(sim, Xoshiro(1))
        cs_random, = _dynamic_cs(sim)
        for _ in 1:100                                                    # t = 1, ~1 turnover
            step!(sim, 0.01)
        end
        cs3, ws3, = _dynamic_cs(sim; filter_ratio = 3)
        cs2, = _dynamic_cs(sim; filter_ratio = 2)
        @info @sprintf("  dynamic C_s: random phases %.4f; evolved (t = 1): α = 3 → %.4f, α = 2 → %.4f",
                       cs_random, cs3, cs2)
        @test 0.10 < cs3 < 0.20
        @test cs_random < 0.25 * cs3
        @test cs2 > cs3               # a narrower assumed test filter needs a larger C_s
        # plane averaging: one coefficient per level
        @test all(k -> allequal(view(ws3.cs2, :, :, k)), 1:mesh.Nz)
    end

    @testset "dynamic Smagorinsky: argument checks" begin
        m1 = RegularGrid(; size = (8, 8, 8), extent = (1.0, 1.0, 1.0), halo = (1, 1, 1))
        s1 = BoussinesqSimulation(m1)
        @test_throws ArgumentError compute_dynamic_smagorinsky_νt!(s1.νt, s1.u, s1.v, s1.w)
        s2 = BoussinesqSimulation(mesh)
        @test_throws ArgumentError compute_dynamic_smagorinsky_νt!(s2.νt, s2.u, s2.v, s2.w; averaging = :box)
    end
end
