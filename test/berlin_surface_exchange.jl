# The Berlin case's surface exchange (#129): SurfaceExchange Monin–Obukhov
# fluxes over per-column AbstractSurface implementers (SoilThermalColumn
# ground, SlabLand roofs / water / glass), applied to the first AIR cell
# of every column, and the surface stress as an implicit drag.
#
# Physics asserted: the air gains exactly the energy and water the
# surfaces give (and nothing lands in a solid cell); the momentum the
# ground's wall model (WallModels, #169) removes is the kinematic surface
# stress u*² along the wind; the extra energy other couplings put on the
# surfaces reaches them; and street β follows the soil water.

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
        # energy and water: ρ̄_k c_p Δz Δθ = H Δt, ρ̄_k Δz Δq = E Δt (reference density, #131)
        ρ̄(I) = BC.ref_density(case, bus.k_air[I])
        E_air = sum(ρ̄(I) * se.cp[I] * dz * Δθ[I, bus.k_air[I]] for I in CartesianIndices(bus.k_air))
        W_air = sum(ρ̄(I) * dz * Δq[I, bus.k_air[I]] for I in CartesianIndices(bus.k_air))
        E_sfc, W_sfc = sum(bus.H) * dt, sum(bus.E) * dt
        @info @sprintf("  surface_flux: air gain %.6e J/m², surface sensible %.6e J/m² (rel %.1e); water %.4e vs %.4e kg/m²",
                       E_air, E_sfc, abs(E_air - E_sfc) / abs(E_sfc), W_air, W_sfc)
        @test E_air ≈ E_sfc rtol = 1e-9          # round-off of θ ≈ 293 K against Δθ ~ 1e-4 K
        @test W_air ≈ W_sfc rtol = 1e-9
        interior_view(sim.θ) .= θ0
        interior_view(BC.tracer(case, :qv)) .= q0
    end

    @testset "momentum: the ground's wall model (WallModels) removes τ_w(U_now) along the wind" begin
        u, v = interior_view(sim.u), interior_view(sim.v)
        u0, v0 = copy(u), copy(v)
        ustar0 = copy(bus.u_star)
        u .= 1.0; v .= 0.5
        dt = 2.0
        h = dz / 2
        Unow = hypot(1.0, 0.5)                   # the instantaneous first-air wind
        # each column's law, at the wind of its surface step, gives back the
        # u* of that step (the L handed over by matching_obukhov_length)
        @test all(I -> abs(friction_velocity(se.laws[I], bus.U[I], h) - ustar0[I]) <=
                       1e-12 * ustar0[I], CartesianIndices(ustar0))
        BC.run_stage!(se, case, Val(:post_predictor), dt)
        # u_τ is re-evaluated at THIS step's wind (before #169: the u*²/U of
        # the surface step was held); the stress is wall_stress(MOSTWallLaw, U_now)
        @test all(==(Unow), se.U_wall)
        @test all(I -> bus.u_star[I]^2 ≈ wall_stress(se.laws[I], Unow, h) && bus.u_star[I] > 0,
                  CartesianIndices(bus.u_star))
        nchanged = count(I -> abs(bus.u_star[I] - ustar0[I]) > 1e-6 * ustar0[I], CartesianIndices(ustar0))
        @info @sprintf("  ground u*: surface step %.4f, at U_now = %.3f m/s %.4f m/s (domain mean; %d of %d columns changed)",
                       sum(ustar0) / length(ustar0), Unow, sum(bus.u_star) / length(bus.u_star),
                       nchanged, length(ustar0))
        # kinematic momentum removed, per unit plan area: Σ Δu dz dx dy / (dx dy)
        Mx = -sum(1.0 .- u) * dz
        My = -sum(0.5 .- v) * dz
        # the stress τ_w(U_now) along the wind (u, v)/U_now, per column
        drag = [wall_drag(se.laws[I], Unow, h) for I in CartesianIndices(bus.u_star)]
        τx = sum(drag[I] * 1.0 for I in CartesianIndices(drag)) * dt
        τy = sum(drag[I] * 0.5 for I in CartesianIndices(drag)) * dt
        amax = maximum(drag) / dz * dt
        @info @sprintf("  drag: momentum removed (%.5e, %.5e) m²/s vs Σ τ_w(U_now)(u,v)/U dt (%.5e, %.5e); max dt·rate %.1e",
                       -Mx, -My, τx, τy, amax)
        # implicit: 1 − 1/(1 + a dt) = a dt − (a dt)² + …, so the relative
        # difference is below the largest a dt
        @test -Mx ≈ τx rtol = amax
        @test -My ≈ τy rtol = amax
        @test all(<=(1.0), u) && all(<=(0.5), v) && all(>(0), u)        # a pure decay
        # only faces at a first air level were touched
        touched = [(i, j, k) for i in 1:Nx, j in 1:Ny, k in 1:Nz if u[i, j, k] != 1.0]
        @test all(t -> t[3] in (bus.k_air[t[1], t[2]], bus.k_air[mod1(t[1] - 1, Nx), t[2]]), touched)
        u .= u0; v .= v0
        bus.u_star .= ustar0
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

    @testset "dew: first air cell of dew_mask columns, at that cell's pressure" begin
        fd = BC.find_coupling(case, :fog_dew)
        qv = interior_view(BC.tracer(case, :qv))
        θs, q0s, ql0 = copy(interior_view(sim.θ)), copy(qv), copy(interior_view(BC.tracer(case, :ql)))
        Ts0, dew0, sens0 = copy(bus.Ts), copy(bus.dew), copy(bus.sensible)
        # a cold roof under humid air: dew forms on it
        I = findfirst(I -> bus.dew_mask[I] && bus.k_air[I] > 3, CartesianIndices(bus.k_air))
        @test I !== nothing
        k = bus.k_air[I]
        bus.Ts .= 330.0                              # no dew anywhere else
        bus.Ts[I] = 280.0
        qv[I, k] = 0.009
        dt = 2.0
        # the expected deposition, at the reference pressure and density of
        # the roof's air cell (#131)
        r = Aeolus.RadiationFog.dew_deposition(0.009, 280.0, BC.ref_pressure(case, k), dt;
                                               dz = dz, ρ = BC.ref_density(case, k), dew = 0.0)
        r1 = Aeolus.RadiationFog.dew_deposition(0.009, 280.0, BC.ref_pressure(case, 1), dt;
                                                dz = dz, ρ = BC.ref_density(case, 1), dew = 0.0)
        BC.run_stage!(fd, case, Val(:fog_dew), dt)
        @test r.mass > 0 && r.mass != r1.mass
        @test bus.dew[I] ≈ r.mass rtol = 1e-12
        @test fd.H_dew[I] ≈ r.surface_heat / dt rtol = 1e-12
        @test count(!=(0), bus.dew) == 1
        interior_view(sim.θ) .= θs; qv .= q0s; interior_view(BC.tracer(case, :ql)) .= ql0
        bus.Ts .= Ts0; bus.dew .= dew0; bus.sensible .= sens0
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
        # Two things decide the street's latent heat. The DEMAND of the
        # Monin–Obukhov β form, LE(β) at the current skin, RISES here
        # (measured 266 → 567 W/m² hourly): the skin heats and q_sat(T_s)
        # grows faster than β falls — the first version of this test
        # expected it to fall with β, which was wrong. β controls the
        # demand's EFFICIENCY, LE(β)/LE(β = 1) at the same skin, which falls
        # with β. What the soil DELIVERS (the E and LE step_surface! returns
        # since #129) is limited by the dry top layer and falls (measured
        # 267 → 5.8 W/m²).
        β = Float64[]; eff = Float64[]; θ1 = Float64[]; LE = Float64[]; LEd = Float64[]
        for n in 1:720                           # 6 h at 30 s
            fl = step_surface!(moist, f, 30.0)
            if n % 120 == 0
                wet = SlabLand(; T = moist.T_skin, albedo = moist.albedo, emissivity = moist.emissivity,
                               z0m = moist.z0m, z0h = moist.z0h, β = 1.0)
                le = turbulent_fluxes(moist, f).LE
                push!(β, moisture_availability(moist)); push!(θ1, moist.water.θ_liq[1])
                push!(LE, le); push!(eff, le / turbulent_fluxes(wet, f).LE); push!(LEd, fl.LE)
            end
        end
        @info "  street soil under 6 h of sun, hourly: θ₁ = " * join((@sprintf("%.4f", x) for x in θ1), ", ") *
              "; β = " * join((@sprintf("%.3f", x) for x in β), ", ") *
              "; LE(β)/LE(β=1) = " * join((@sprintf("%.3f", x) for x in eff), ", ") *
              "; demand LE(β) = " * join((@sprintf("%.1f", x) for x in LE), ", ") *
              " W/m²; delivered LE = " * join((@sprintf("%.1f", x) for x in LEd), ", ") * " W/m²"
        # delivered: 267 W/m² in the first hour (top layer still moist), then
        # the surface reaches the dry potential and it falls to ~6 W/m²
        @test LEd[end] < 0.1 * LEd[1] && issorted(LEd[2:end]; rev = true)
        @test issorted(θ1; rev = true) && issorted(β; rev = true) && issorted(eff; rev = true)
        @test β[end] < βm
        # The efficiency is β at fixed stability. It comes out a little BELOW
        # β (measured 0.945–0.948 β): the saturated reference's larger vapour
        # flux adds buoyancy, so it is more unstable and has the larger C_E
        # (the same effect as in the SlabLand β test of era5_surface_exchange).
        @test all(i -> 0.9 * β[i] < eff[i] < β[i], eachindex(β))
    end
end
