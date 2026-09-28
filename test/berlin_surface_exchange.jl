# The Berlin case's surface exchange (#129): SurfaceExchange Monin–Obukhov
# fluxes over per-column AbstractSurface implementers (SoilThermalColumn
# ground, SlabLand roofs / water / glass), applied to the first AIR cell
# of every column, and the surface stress as an implicit drag.
#
# Physics asserted: the air gains exactly the energy and water the
# surfaces give (and nothing lands in a solid cell); the momentum the
# drag removes is the kinematic surface stress u*² along the wind; the
# extra energy other couplings put on the surfaces reaches them; and
# street β follows the soil water.

using Test
using Aeolus
using Aeolus.Fields: interior_view
using Printf

isdefined(Main, :BerlinCase) ||
    include(joinpath(@__DIR__, "..", "examples", "urban", "BerlinCase.jl"))
const BC = Main.BerlinCase

@testset "Berlin case: surface exchange and soil ground (#129)" begin
    case = BC.build_case(BC.CaseConfig(:smoke; verbose = false))
    mesh, sim, bus, g = case.mesh, case.sim, case.bus, case.geom
    Nx, Ny, Nz, dz = mesh.Nx, mesh.Ny, mesh.Nz, mesh.dz
    Hx, Hy, Hz = mesh.Hx, mesh.Hy, mesh.Hz
    se = BC.find_coupling(case, :surface_exchange)
    gs = BC.find_coupling(case, :ground_soil)
    solid(i, j, k) = g.sm.mask[Hx+i, Hy+j, Hz+k]

    @testset "surfaces and the roof-aware air level" begin
        @test se !== nothing && gs !== nothing
        @test BC.find_coupling(case, :surface_slab) === nothing       # the slab scheme is gone
        for I in CartesianIndices(bus.k_air)
            i, j = Tuple(I); k = bus.k_air[I]
            @test all(kk -> solid(i, j, kk), 1:k-1)
            @test k <= Nz && !solid(i, j, k)          # the smoke window has no column solid to the lid
            @test (k > 1) == (g.heightmap[I] > 0)    # above the roof exactly on building columns
        end
        nroof = count(>(1), bus.k_air)
        @info "  smoke window: $nroof building columns exchange above their roof (k_air 2–$(maximum(bus.k_air)))"
        @test nroof > 0
        for I in CartesianIndices(g.sfc)
            c, s = g.sfc[I], bus.surface[I]
            if c in (BC.SURFACE_STREET, BC.SURFACE_WALL, BC.SURFACE_PARK)
                @test s isa SoilThermalColumn
                @test s.albedo == bus.params.albedo[c] && s.emissivity == bus.params.emissivity[c]
            else
                @test s isa SlabLand
                @test s.C_s == bus.params.C_s[c] && s.albedo == bus.params.albedo[c]
                @test moisture_availability(s) == (c == BC.SURFACE_WATER ? 1 : 0)
            end
            @test bus.dew_mask[I] == (moisture_availability(s) == 0)
        end
    end

    @testset "surface_flux: the air gains what the surfaces give, in the first air cell only" begin
        θ0 = copy(interior_view(sim.θ))
        q0 = copy(interior_view(BC.tracer(case, :qv)))
        dt = 2.0
        BC.run_stage!(se, case, Val(:surface_flux), dt)   # step 1: first surface step, then the fluxes
        @test se.nsurface_steps == 1
        Δθ = interior_view(sim.θ) .- θ0
        Δq = interior_view(BC.tracer(case, :qv)) .- q0
        at_air = falses(Nx, Ny, Nz)
        for I in CartesianIndices(bus.k_air)
            at_air[I, bus.k_air[I]] = true
        end
        # solid cells: exactly zero; air cells above the first: exactly zero
        @test all(==(0), Δθ[.!at_air]) && all(==(0), Δq[.!at_air])
        nsolid = count(I -> g.sm.mask[Hx+I[1], Hy+I[2], Hz+I[3]], CartesianIndices(Δθ))
        @test nsolid > 0
        # energy and water: ρ c_p Δz Δθ = H Δt, ρ Δz Δq = E Δt (ρ = ledger density, 1)
        E_air = sum(se.cp[I] * dz * Δθ[I, bus.k_air[I]] for I in CartesianIndices(bus.k_air))
        W_air = sum(dz * Δq[I, bus.k_air[I]] for I in CartesianIndices(bus.k_air))
        E_sfc, W_sfc = sum(bus.H) * dt, sum(bus.E) * dt
        @info @sprintf("  surface_flux: air gain %.6e J/m², surface sensible %.6e J/m² (rel %.1e); water %.4e vs %.4e kg/m²",
                       E_air, E_sfc, abs(E_air - E_sfc) / abs(E_sfc), W_air, W_sfc)
        @test E_air ≈ E_sfc rtol = 1e-9          # round-off of θ ≈ 293 K against Δθ ~ 1e-4 K
        @test W_air ≈ W_sfc rtol = 1e-9
        interior_view(sim.θ) .= θ0
        interior_view(BC.tracer(case, :qv)) .= q0
    end

    @testset "momentum: the drag removes the surface stress u*² along the wind" begin
        u, v = interior_view(sim.u), interior_view(sim.v)
        u0, v0 = copy(u), copy(v)
        u .= 1.0; v .= 0.5
        dt = 2.0
        BC.run_stage!(se, case, Val(:post_predictor), dt)
        # kinematic momentum removed, per unit plan area: Σ Δu dz dx dy / (dx dy)
        Mx = -sum(1.0 .- u) * dz
        My = -sum(0.5 .- v) * dz
        # the stress of each column's surface step, u*² along its wind (u_c, v_c)/U,
        # evaluated with the wind this test imposed
        U = hypot(1.0, 0.5)
        τx = sum(se.drag[I] * 1.0 for I in CartesianIndices(se.drag)) * dt
        τy = sum(se.drag[I] * 0.5 for I in CartesianIndices(se.drag)) * dt
        amax = maximum(se.drag) / dz * dt
        @info @sprintf("  drag: momentum removed (%.5e, %.5e) m²/s vs Σ u*²(u,v)/U dt (%.5e, %.5e); max dt·rate %.1e",
                       -Mx, -My, τx, τy, amax)
        # implicit: 1 − 1/(1 + a dt) = a dt − (a dt)² + …, so the relative
        # difference is below the largest a dt
        @test -Mx ≈ τx rtol = amax
        @test -My ≈ τy rtol = amax
        @test all(<=(1.0), u) && all(<=(0.5), v) && all(>(0), u)        # a pure decay
        # only faces at a first air level were touched
        touched = [(i, j, k) for i in 1:Nx, j in 1:Ny, k in 1:Nz if u[i, j, k] != 1.0]
        @test all(t -> t[3] in (bus.k_air[t[1], t[2]], bus.k_air[mod1(t[1] - 1, Nx), t[2]]), touched)
        # the kinematic stress of the surface step: u*² = drag · max(U, U_min)
        @test all(I -> se.drag[I] * max(bus.U[I], se.U_min) ≈ bus.u_star[I]^2, CartesianIndices(se.drag))
        u .= u0; v .= v0
    end

    @testset "extra surface energy (bus.sensible) reaches the surface at its next step" begin
        I = findfirst(I -> bus.surface[I] isa SlabLand, CartesianIndices(g.sfc))
        s = bus.surface[I]
        P = se.P
        @test P == 4.0                           # the :smoke soil/surface cadence
        t_save = sim.t
        # 2 steps of 2 s during which another coupling takes −100 W/m² (gives 100)
        bus.sensible[I] = -100.0
        BC.run_stage!(se, case, Val(:surface_update), 2.0)
        @test se.pending[I] ≈ 200.0 && bus.sensible[I] == 0
        bus.sensible[I] = -100.0
        sim.t = se.t_next                        # the air has reached the surfaces' time
        T0 = s.T
        BC.run_stage!(se, case, Val(:surface_update), 2.0)
        @test se.nsurface_steps == 2
        @test se.Q[I] ≈ 100.0 && se.pending[I] == 0
        # the slab stored Rn − H − LE + Q over the interval
        @test (s.T - T0) * s.C_s / P ≈ bus.Rn[I] - bus.H[I] - bus.LE[I] + 100.0 rtol = 1e-10
        @test bus.G[I] ≈ bus.Rn[I] - bus.H[I] - bus.LE[I] + 100.0 rtol = 1e-12
        sim.t = t_save
    end

    @testset "street β follows the soil water" begin
        I = CartesianIndices(g.sfc)[findfirst(==(BC.SURFACE_STREET), vec(g.sfc))]
        f = SurfaceForcing(sw_down = 700.0, lw_down = 330.0, T_air = 295.0, q_air = 0.006,
                           U = 2.0, z_ref = dz / 2)
        ncols = length(gs.cols)
        mk(θ) = (gs.θ0[BC.SURFACE_STREET] = θ; BC.soil_surface(gs, case, I))
        dry, moist = mk(0.09), mk(0.12)
        gs.θ0[BC.SURFACE_STREET] = 0.10
        resize!(gs.cols, ncols)                  # the two test columns are not the case's
        βd, βm = moisture_availability(dry), moisture_availability(moist)
        LEd, LEm = step_surface!(dry, f, 30.0).LE, step_surface!(moist, f, 30.0).LE
        @info @sprintf("  street soil θ₁ 0.09 → β %.3f, LE %.1f W/m²; θ₁ 0.12 → β %.3f, LE %.1f W/m²",
                       βd, LEd, βm, LEm)
        @test βd < βm < 1 && LEd < LEm
        # A sunny afternoon under fixed air dries the top layer and β falls.
        # LE itself RISES (measured 267 → 302 W/m² hourly): the skin heats
        # and q_sat(T_s) grows faster than β falls — the first version of
        # this test expected LE to fall with β, which was wrong. What β
        # controls is the evaporation EFFICIENCY, LE over the LE of a
        # saturated (β = 1) surface at the same skin temperature and
        # roughness; it falls with β.
        β = Float64[]; eff = Float64[]; θ1 = Float64[]; LE = Float64[]
        for n in 1:720                           # 6 h at 30 s
            step_surface!(moist, f, 30.0)
            if n % 120 == 0
                wet = SlabLand(; T = moist.T_skin, albedo = moist.albedo, emissivity = moist.emissivity,
                               z0m = moist.z0m, z0h = moist.z0h, β = 1.0)
                le = turbulent_fluxes(moist, f).LE
                push!(β, moisture_availability(moist)); push!(θ1, moist.water.θ_liq[1])
                push!(LE, le); push!(eff, le / turbulent_fluxes(wet, f).LE)
            end
        end
        @info "  street soil under 6 h of sun, hourly: θ₁ = " * join((@sprintf("%.4f", x) for x in θ1), ", ") *
              "; β = " * join((@sprintf("%.3f", x) for x in β), ", ") *
              "; LE/LE(β=1) = " * join((@sprintf("%.3f", x) for x in eff), ", ") *
              "; LE = " * join((@sprintf("%.1f", x) for x in LE), ", ") * " W/m²"
        @test issorted(θ1; rev = true) && issorted(β; rev = true) && issorted(eff; rev = true)
        @test β[end] < βm
        # The efficiency is β at fixed stability. It comes out a little BELOW
        # β (measured 0.945–0.948 β): the saturated reference's larger vapour
        # flux adds buoyancy, so it is more unstable and has the larger C_E
        # (the same effect as in the SlabLand β test of era5_surface_exchange).
        @test all(i -> 0.9 * β[i] < eff[i] < β[i], eachindex(β))
    end
end
