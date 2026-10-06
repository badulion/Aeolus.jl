# The open lateral boundaries of the Berlin case (N-open-d, #262):
# `CaseConfig(lateral = :open)` (core/open_boundary.jl).
#
#   * the default stays periodic: the same boundary conditions, no open
#     meter, the geostrophic coupling keeps its sponge retargeting;
#   * `:open`: open W, E, S, N and top fed by the geostrophic wind and the
#     initial θ, the core's sponge toward that wind, the all-Neumann
#     pressure solve, no building within `boundary_frame` cells of a face,
#     solid open-face points where a building touches a face;
#   * the open faces' heat in the air-heat ledger, through the case's own
#     body-force hook and `dynamics_step!`: zero when the inflow θ equals
#     the interior; ρ̄c_pΠ̄ (U ΔT/2 + ν ΔT/Δx) A Δt for a +1 K inflow
#     anomaly (the core advects θ with the centred scheme, whose face value
#     between the halo and the first cell is their mean; the diffusive
#     flux across the jump is ν ΔT/Δx);
#   * the measured cost of an open step against a periodic one.
#
# The 12-step ledgers of the open box with every coupling on are the
# smoke matrix's `geostrophic: lateral = :open, …` runs (berlin_smoke.jl).
#
# Run on its own with `julia --project=. test/berlin_open_boundaries.jl`.

using Test
using Aeolus
using Aeolus.Fields: interior_view, set!
using Aeolus.BoundaryConditions: OpenBoundary, Periodic
using Printf

isdefined(Main, :BerlinCase) ||
    include(joinpath(@__DIR__, "..", "examples", "urban", "BerlinCase.jl"))
const BC = Main.BerlinCase

_smoke(; kw...) = BC.CaseConfig(:smoke; verbose = false, output_path = nothing, kw...)

@testset "Berlin open lateral boundaries (N-open-d, #262)" begin
    @testset "the default stays periodic" begin
        cfg = _smoke()
        @test cfg.lateral === :periodic
        case = BC.build_case(cfg; io = devnull)
        b = case.sim.bcs
        @test all(f -> f isa Periodic, (b.west, b.east, b.south, b.north))
        @test case.open === nothing
        @test case.sim.sponge_target == (u = 0, v = 0, w = 0)
        @test BC.find_coupling(case, :geostrophic).sponge_ref
        @test BC.case_scalar_bcs(cfg).west isa Periodic
        @test_throws ErrorException BC.lateral_open(_smoke(lateral = :nested))
    end

    @testset "lateral = :open: faces, data, sponge, solver, frame" begin
        cfg = _smoke(lateral = :open)
        case = BC.build_case(cfg; io = devnull)
        sim, m = case.sim, case.mesh
        b = sim.bcs
        @test all(f -> f isa OpenBoundary, (b.west, b.east, b.south, b.north, b.top))
        @test sim.θ_bcs === b
        g = BC.coupling_options(BC.GeostrophicCoupling, cfg)
        @test sim.sponge_target == (u = g.U_g, v = g.V_g, w = 0.0)
        @test !BC.find_coupling(case, :geostrophic).sponge_ref       # the core targets the data
        @test all(==(:neumann), values(poisson_axis_bcs(sim.solver)))
        @test poisson_axis_bcs(sim.solver) == poisson_axis_bcs(sim.bcs)
        @test case.open isa BC.OpenBoundaryHeat
        @test case.open.damping !== nothing                          # θ relaxation (default)
        # no building within boundary_frame cells of an open face
        wf = cfg.boundary_frame
        mask = view(case.geom.sm.mask, m.Hx+1:m.Hx+m.Nx, m.Hy+1:m.Hy+m.Ny, m.Hz+1:m.Hz+m.Nz)
        @test any(mask)
        @test !any(mask[1:wf, :, :]) && !any(mask[end-wf+1:end, :, :])
        @test !any(mask[:, 1:wf, :]) && !any(mask[:, end-wf+1:end, :])
        # the buffers hold the data before the first stage reads them
        @test all(sim.u.data[m.Hx+1, m.Hy+1:m.Hy+m.Ny, m.Hz+1:m.Hz+m.Nz] .== g.U_g)
        # without the frame the scene's buildings reach the faces: their
        # open-face points are solid (zero velocity, no correction)
        c0 = BC.build_case(_smoke(lateral = :open, boundary_frame = 0); io = devnull)
        nsolid = sum(count(!, p.fluid) for st in open_boundary_states(c0.sim)
                     for p in st.planes if p.normal)
        mask0 = c0.geom.sm.mask
        touching = count(mask0[m.Hx+1, :, :]) + count(mask0[m.Hx+m.Nx, :, :]) +
                   count(mask0[:, m.Hy+1, :]) + count(mask0[:, m.Hy+m.Ny, :]) +
                   count(mask0[:, :, m.Hz+m.Nz])
        @info "open smoke window without the frame: $touching solid cells on the faces, $nsolid solid normal-velocity points"
        @test nsolid == touching > 0
    end

    @testset "a building at the east edge: no wall class across an open west face" begin
        # the wall tagging (the 2-D surface's facade heat) read its 4
        # neighbours with a periodic wrap: a street column at the west edge
        # next to a building at the east edge (across the seam) became a
        # wall. Through neighbor_index it still does when periodic, and not
        # when open
        m = RegularGrid(size = (8, 6, 4), extent = (80.0, 60.0, 40.0), halo = (3, 3, 3))
        sm = Aeolus.IBM.SolidMask(m)
        for k in 1:2, j in 2:4; sm.mask[m.Hx+8, m.Hy+j, m.Hz+k] = true; end
        per = fill(BC.SURFACE_STREET, 8, 6); opn = copy(per)
        BC.tag_building_adjacent!(per, sm, m)
        BC.tag_building_adjacent!(opn, sm, m; bcs = BC.case_scalar_bcs(_smoke(lateral = :open)))
        @test all(per[1, j] == BC.SURFACE_WALL for j in 2:4)
        @test all(opn[1, j] == BC.SURFACE_STREET for j in 2:4)
        @test per[7, 2:4] == opn[7, 2:4] == fill(BC.SURFACE_WALL, 3)   # the inner side is a wall in both
    end

    @testset "the open faces' heat in the ledger: equal inflow 0, +1 K anomaly analytic" begin
        # a uniform-θ window (the frame keeps the buildings off the faces,
        # the tallest roof is below the top), no SGS (C_s = 0), the fixed
        # stepper: one dynamics step through the case's own hook
        base = (lateral = :open, θ_lapse = 0.0, C_s = 0.0,
                physics = Dict{Symbol,Any}(:sgs => :smagorinsky, :timestepper => :fixed))
        function one_step(ΔT)
            case = BC.build_case(_smoke(; base...); io = devnull)
            sim, m, ref = case.sim, case.mesh, case.ref
            θ0 = case.cfg.θ_surface
            U = BC.coupling_options(BC.GeostrophicCoupling, case.cfg).U_g
            set!(sim.θ, θ0); set!(sim.u, U); set!(sim.v, 0.0); set!(sim.w, 0.0)
            set_boundary_data!(sim.bcs, FunctionBoundaryData(u = U, v = 0.0, w = 0.0,
                                                           θ = (x, y, z, t) -> x < 0 ? θ0 + ΔT : θ0))
            prepare_open_boundaries!(sim)
            BC.dynamics_step!(case, case.cfg.dt)
            heat = [ref.ρ̄[k] * ref.cp * ref.π̄[k] for k in 1:m.Nz]
            expected = sum(heat) * m.Ny * m.dy * m.dz * (U * ΔT / 2 + sim.ν * ΔT / m.dx) *
                       case.cfg.dt / (m.Lx * m.Ly)
            scale = sum(heat) * m.Ny * m.dy * m.dz * U * θ0 * case.cfg.dt / (m.Lx * m.Ly)
            return case.open, expected, scale, sim.ν
        end
        ob0, e0, s0, _ = one_step(0.0)
        @info @sprintf("equal inflow: boundary heat %.2e J/m² (|ρ̄c_pΠ̄ U θ A Δt| = %.2e)", ob0.total, s0)
        @test e0 == 0
        @test abs(ob0.total) <= 1e-14 * s0
        @test ob0.ctotal == 0 && ob0.rtotal == 0
        ob1, e1, s1, ν = one_step(1.0)
        @info @sprintf("+1 K inflow anomaly: boundary heat %.10e J/m², analytic ρ̄c_pΠ̄(UΔT/2 + νΔT/Δx)AΔt %.10e (ν = %.1f m²/s)",
                       ob1.total, e1, ν)
        @test ob1.total ≈ e1 rtol = 1e-12
        @test ob1.ctotal == 0                     # no net inflow: δ = 0
    end

    @testset "cost of an open step (s/step, measured)" begin
        # the fixed stepper (same dt in both), 12 steps after 2 to compile,
        # the same buildings (boundary_frame = 0: the frame removes buildings,
        # which makes the radiation cheaper and would flatter the open run);
        # the dynamics' share (step!, which includes the open faces' update,
        # the boundary-heat capture and the θ relaxation) and the whole step
        # Two rounds, the second measured: a cadence that first fires inside
        # the measured steps compiles there (the first round measured the
        # periodic case 2× slower than the open one for that reason alone).
        t = Dict{Symbol,NTuple{2,Float64}}()
        for _ in 1:2, lat in (:periodic, :open)
            case = BC.build_case(_smoke(lateral = lat, boundary_frame = 0,
                                        physics = Dict{Symbol,Any}(:timestepper => :fixed)); io = devnull)
            BC.run_case!(case; nsteps = 2)
            n0 = case.clock.n
            d0 = case.timing[(:dynamics, :step!)]
            tt = @elapsed BC.run_case!(case; nsteps = 12)
            n = case.clock.n - n0
            t[lat] = (tt / n, (case.timing[(:dynamics, :step!)] - d0) / n)
        end
        @info @sprintf("smoke window 16×16×16, every coupling, fixed dt: whole step periodic %.1f ms, open %.1f ms (×%.2f); dynamics periodic %.1f ms, open %.1f ms (×%.2f)",
                       1e3 * t[:periodic][1], 1e3 * t[:open][1], t[:open][1] / t[:periodic][1],
                       1e3 * t[:periodic][2], 1e3 * t[:open][2], t[:open][2] / t[:periodic][2])
        @test t[:open][1] > 0
    end
end
