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
using Random: Xoshiro, randn
using Statistics: mean
using Printf

# A solenoidal random field with E(k) ∝ k^(-5/3) up to k_c (in units of
# the fundamental wavenumber) and random phases, scaled to u_rms = 1:
# `kolmogorov_field!`, shared with the other SGS tests (#317).
isdefined(@__MODULE__, :kolmogorov_field!) || include(joinpath(@__DIR__, "support", "kolmogorov.jl"))

# `_dynamic_cs`, shared with the validation run (validation/era2_sgs_numerics.jl)
isdefined(@__MODULE__, :_dynamic_cs) || include(joinpath(@__DIR__, "support", "dynamic_smagorinsky.jl"))

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
        kolmogorov_field!(sim, Xoshiro(3))
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

    @testset "dynamic Smagorinsky: plane averaging gives one coefficient per level" begin
        # (C_s in developed turbulence, after ~1 eddy turnover of LES, is
        # the validation run validation/era2_sgs_numerics.jl.)
        sim = BoussinesqSimulation(mesh; C_s = 0.17, ν = 1e-4)
        kolmogorov_field!(sim, Xoshiro(1))
        _, ws3, = _dynamic_cs(sim; filter_ratio = 3)
        @test all(k -> allequal(view(ws3.cs2, :, :, k)), 1:mesh.Nz)
    end

    @testset "dynamic Smagorinsky: argument checks" begin
        m1 = RegularGrid(; size = (8, 8, 8), extent = (1.0, 1.0, 1.0), halo = (1, 1, 1))
        s1 = BoussinesqSimulation(m1)
        @test_throws ArgumentError compute_dynamic_smagorinsky_νt!(s1.νt, s1.u, s1.v, s1.w)
        s2 = BoussinesqSimulation(mesh)
        @test_throws ArgumentError compute_dynamic_smagorinsky_νt!(s2.νt, s2.u, s2.v, s2.w; averaging = :box)
    end

    @testset "step!: eddy_viscosity! with the Smagorinsky closure is the built-in" begin
        mk() = (s = BoussinesqSimulation(RegularGrid(; size = (16, 16, 16), extent = (1.0, 1.0, 1.0), halo = (3, 3, 3));
                                         C_s = 0.17, ν = 1e-4, θ₀ = 300.0, f_coriolis = 1e-4,
                                         bcs = FieldBoundaryConditions(:periodic, :periodic, :wall));
                kolmogorov_field!(s, Xoshiro(7); kc = 4); set!(s.θ, (x, y, z) -> 300 + z); s)
        a, b = mk(), mk()
        ml = MLSGS((νt, u, v, w, m) -> compute_smagorinsky_νt!(νt, u, v, w, 0.17))
        for _ in 1:20
            step!(a, 1e-3)
            step!(b, 1e-3; eddy_viscosity! = (νt, u, v, w) -> apply_mlsgs!(b, ml))
        end
        @test all(f -> agree(getfield(a, f).data, getfield(b, f).data), (:u, :v, :w, :θ, :p, :νt))
        # C_s = 0 disables the built-in closure but not a passed one
        c = BoussinesqSimulation(RegularGrid(; size = (16, 16, 16), extent = (1.0, 1.0, 1.0), halo = (3, 3, 3)))
        kolmogorov_field!(c, Xoshiro(7); kc = 4)
        step!(c, 1e-3; eddy_viscosity! = (νt, u, v, w) -> compute_dynamic_smagorinsky_νt!(νt, u, v, w))
        @test maximum(interior_view(c.νt)) > 0
    end

    @testset "step!: momentum_advection = WENO-Z conserves momentum and dissipates energy" begin
        # Inviscid, triply periodic, no SGS: the flux-form operators
        # conserve the domain momentum exactly; forward Euler with the
        # centred operator gains kinetic energy, the upwind-biased WENO-Z
        # removes it.
        m16 = RegularGrid(; size = (16, 16, 16), extent = (2π, 2π, 2π), halo = (3, 3, 3))
        weno = (tu, tv, tw, u, v, w) -> momentum_advect_weno_z!(tu, tv, tw, u, v, w)
        res = Dict{Symbol,Any}()
        for (name, kw) in ((:centered, (;)), (:weno_z, (momentum_advection = weno,)))
            s = BoussinesqSimulation(m16)
            kolmogorov_field!(s, Xoshiro(5); kc = 4)
            P0 = sum(interior_view(s.u)); E0 = kinetic_energy(s)
            for _ in 1:50
                step!(s, 0.01; kw...)
            end
            res[name] = (dP = abs(sum(interior_view(s.u)) - P0), dE = kinetic_energy(s) / E0 - 1,
                         div = max_divergence(s))
        end
        @info @sprintf("  50 steps: ΔKE/KE centred %+.2e, WENO-Z %+.2e; |ΔΣu| WENO-Z %.1e; max|∇·u| %.1e",
                       res[:centered].dE, res[:weno_z].dE, res[:weno_z].dP, res[:weno_z].div)
        @test res[:weno_z].dP < 1e-10
        @test res[:weno_z].div < 1e-10
        @test res[:weno_z].dE < 0 < res[:centered].dE
        # the default keyword is the built-in operator
        a = BoussinesqSimulation(m16); b = BoussinesqSimulation(m16)
        kolmogorov_field!(a, Xoshiro(5); kc = 4); kolmogorov_field!(b, Xoshiro(5); kc = 4)
        step!(a, 0.01); step!(b, 0.01; momentum_advection = momentum_advect!)
        @test agree(a.u.data, b.u.data)
    end
end
