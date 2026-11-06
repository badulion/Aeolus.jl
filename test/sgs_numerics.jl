# Numerics swap-ins used by the Berlin case (#132): the dynamic
# Smagorinsky closure (Germano et al. 1991 / Lilly 1992) and the
# `momentum_advection` / `eddy_viscosity!` keywords of step!.

using Test
using Aeolus
using Aeolus.Fields: interior_view, set!
using Random: Xoshiro, randn
using Statistics: mean

# `kolmogorov_field!`, shared with the other SGS tests (#317)
isdefined(@__MODULE__, :kolmogorov_field!) || include(joinpath(@__DIR__, "support", "kolmogorov.jl"))
# `_dynamic_cs`, shared with the validation run (validation/sgs_numerics.jl)
isdefined(@__MODULE__, :_dynamic_cs) || include(joinpath(@__DIR__, "support", "dynamic_smagorinsky.jl"))

@testset "SGS and momentum-advection swap-ins (#132)" begin

    @testset "dynamic Smagorinsky: zero in laminar shear, |S| of the static model, Galilean" begin
        # u = S z: test-filtered and grid stresses agree (L_ij M_ij = 0), so the
        # dynamic coefficient vanishes where static Smagorinsky still gives (C_s Δ)² S
        mw = RegularGrid(; size = (8, 8, 16), extent = (1.0, 1.0, 2.0), halo = (3, 3, 3))
        sw = BoussinesqSimulation(mw; bcs = FieldBoundaryConditions(:periodic, :periodic, :wall))
        set!(sw.u, (x, y, z) -> 3.0 * z)
        _, wsl, _ = _dynamic_cs(sw; averaging = :local)
        @test maximum(wsl.cs2[:, :, 3:end-2]) == 0
        compute_smagorinsky_νt!(sw.νt, sw.u, sw.v, sw.w, 0.17)
        @test minimum(interior_view(sw.νt)[:, :, 3:end-2]) > 0

        mesh = RegularGrid(; size = (16, 16, 16), extent = (2π, 2π, 2π), halo = (3, 3, 3))
        sim = BoussinesqSimulation(mesh)
        kolmogorov_field!(sim, Xoshiro(3))
        _, ws, νt = _dynamic_cs(sim; averaging = :local)
        compute_smagorinsky_νt!(sim.νt, sim.u, sim.v, sim.w, 1.0)       # Δ²|S|
        @test interior_view(νt) ≈ ws.cs2 .* interior_view(sim.νt) rtol = 1e-13
        @test all(c -> 0 <= c <= 0.25, ws.cs2)                            # clip [0, 0.5²]
        cs0 = copy(ws.cs2)
        interior_view(sim.u) .+= 5.0; interior_view(sim.w) .+= 1.0     # uniform translation
        _, ws2, _ = _dynamic_cs(sim; averaging = :local)
        @test maximum(abs, ws2.cs2 .- cs0) < 1e-9 * maximum(cs0)
        _, ws3, = _dynamic_cs(sim)                                       # plane averaging
        @test all(k -> allequal(view(ws3.cs2, :, :, k)), 1:mesh.Nz)
    end

    @testset "step!: swapped closure and momentum advection" begin
        m = RegularGrid(; size = (8, 8, 8), extent = (2π, 2π, 2π), halo = (3, 3, 3))
        mk() = (s = BoussinesqSimulation(m; C_s = 0.17, ν = 1e-4,
                                         bcs = FieldBoundaryConditions(:periodic, :periodic, :wall));
                kolmogorov_field!(s, Xoshiro(7); kc = 2); s)
        a, b = mk(), mk()
        ml = MLSGS((νt, u, v, w, m) -> compute_smagorinsky_νt!(νt, u, v, w, 0.17))
        for _ in 1:3
            step!(a, 1e-3)
            step!(b, 1e-3; eddy_viscosity! = (νt, u, v, w) -> apply_mlsgs!(b, ml))
        end
        # the same closure through the keyword is the built-in step (round-off)
        @test all(f -> agree(getfield(a, f).data, getfield(b, f).data), (:u, :w, :νt))

        # inviscid periodic WENO-Z momentum advection: conserves Σu, dissipates KE
        s = BoussinesqSimulation(m)
        kolmogorov_field!(s, Xoshiro(5); kc = 2)
        P0, E0 = sum(interior_view(s.u)), kinetic_energy(s)
        for _ in 1:10
            step!(s, 0.01; momentum_advection = (tu, tv, tw, u, v, w) -> momentum_advect_weno_z!(tu, tv, tw, u, v, w))
        end
        @test abs(sum(interior_view(s.u)) - P0) < 1e-10
        @test kinetic_energy(s) < E0
    end
end
