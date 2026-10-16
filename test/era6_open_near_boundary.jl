# N-open-d (#262, epic #246): near-boundary physics of an open domain.
#
#   1. the core sponge below an open top relaxes toward the boundary data
#      (`sponge_target`): a uniform geostrophic stream U_g through an open
#      box with an open top and a sponge stays U_g to < 1e-12 in step!,
#      step_rk3!, step_imex! (ARK2) and the anelastic core; the zero target
#      of a closed top would decelerate it (the failure mode); a missing
#      target below an open top is an ArgumentError
#   2. `neighbor_index`, the one BC-aware neighbour helper: `mod1` on a
#      periodic axis, the mirror (or no neighbour) on an open one
#   3. a building at an open face: solid boundary points carry zero
#      velocity, the mass-flux correction excludes them, ∇·u = 0 to
#      round-off; across the open west face the building has no wall, no
#      wall stress and no wall-distance influence, all of which it has
#      across the periodic seam
#   4. `boundary_scalar_flux` for the budgets of an open box: per-level
#      flux, the diffusive part (ν + SGS κ) and the part carried by the
#      mass-flux correction; Δ∫θ dV of a whole `step!` with advection,
#      molecular and Smagorinsky diffusion through open faces equals
#      −Δt·(advective + diffusive boundary flux) to round-off; a +1 K
#      inflow anomaly gives the analytic flux for the upwind and centred
#      face values
#
# References: Hellsten et al. (2021), GMD 14, 3185 (PALM nesting: Rayleigh
# damping toward the parent state, §2.3; mass-flux correction, §3.3);
# Kim & Moin (1985), JCP 59, 308.

using Test
using Aeolus
using Aeolus.Grids: interior, xnodes, FaceX, FaceY, FaceZ, CellCenter
using Aeolus.Fields: Field, interior_view, set!
using Aeolus.BoundaryConditions: OpenBoundary, Periodic
using Aeolus.Simulations: BoussinesqSimulation, step_rk3!, max_divergence, add_velocity_sponge!
using Aeolus.TimeIntegration: step_imex!, ARK2
using Aeolus.Anelastic: AnelasticSimulation, max_anelastic_divergence
using Aeolus.IBM: SolidMask, implicit_ibm_correction!, wall_distance
using Aeolus.WallModels: immersed_walls, RoughLogLaw
using Aeolus.LargeScaleForcing: GeostrophicPressureGradient, apply_forcing!
using Aeolus.InputDrivers: tag_building_adjacent!, SURFACE_STREET, SURFACE_WALL
using Printf, Random

# ---------------------------------------------------------------------------
@testset "N-open-d: sponge below an open top relaxes toward the boundary data" begin
    m = RegularGrid(size = (12, 10, 16), extent = (600.0, 500.0, 400.0), halo = (2, 2, 2))
    Ug, Vg, f = 5.0, -1.5, 1e-4
    data = ProfileBoundaryData(u = Ug, v = Vg, w = 0.0, θ = 300.0)
    F = GeostrophicPressureGradient(Ug, Vg)
    bf(s, Δt) = apply_forcing!(s, F, s.t, Δt)
    kw = (θ₀ = 300.0, f_coriolis = f, C_s = 0.18, ν = 1e-3,
          sponge_α_max = 0.05, sponge_z_start = 300.0)
    tgt = sponge_target(data, m)
    @test tgt == (u = Ug, v = Vg, w = 0.0)
    # a missing target below an open top is the error of #262 …
    @test_throws ArgumentError BoussinesqSimulation(m; bcs = open_boundary_conditions(m, data), kw...)
    @test_throws ArgumentError AnelasticSimulation(m; bcs = open_boundary_conditions(m, data),
                                                   f_coriolis = f, sponge_α_max = 0.05,
                                                   sponge_z_start = 300.0)
    # … but a closed top keeps the Phase-7 zero target, and no sponge needs none
    @test BoussinesqSimulation(m; bcs = open_boundary_conditions(m, data; top = :wall),
                               kw...).sponge_target == (u = 0, v = 0, w = 0)
    @test BoussinesqSimulation(m; bcs = open_boundary_conditions(m, data), θ₀ = 300.0) isa
          BoussinesqSimulation
    @test_throws ArgumentError BoussinesqSimulation(m; bcs = open_boundary_conditions(m, data),
                                                    sponge_target = (u = 1.0, q = 2.0), kw...)

    function bous(target)
        s = BoussinesqSimulation(m; bcs = open_boundary_conditions(m, data), sponge_target = target, kw...)
        set!(s.u, Ug); set!(s.v, Vg); set!(s.θ, 300.0)
        return s
    end
    dev(s) = max(maximum(abs, interior_view(s.u) .- Ug), maximum(abs, interior_view(s.v) .- Vg),
                 maximum(abs, interior_view(s.w)))
    for (name, adv!) in (("step!", s -> step!(s, 2.0; body_force! = bf)),
                         ("step_rk3!", s -> step_rk3!(s, 2.0; body_force! = bf)),
                         ("step_imex! ARK2", s -> step_imex!(s, 2.0, ARK2(); body_force! = bf)))
        s = bous(tgt)
        for _ in 1:100; adv!(s); end
        d = dev(s)
        @info @sprintf("open top, sponge → data, %-16s max|u − U_g| = %.2e after 100 steps", name, d)
        @test d < 1e-12
        @test max_divergence(s) < 1e-12
    end
    # the failure mode: the Phase-7 zero target below the open top is a
    # momentum sink of α(z)·U_g in the layer (explicitly requested here)
    s0 = bous(sponge_target(u = 0, v = 0))
    for _ in 1:100; step!(s0, 2.0; body_force! = bf); end
    d0 = dev(s0)
    @info @sprintf("open top, sponge → 0 (failure mode): max|u − U_g| = %.3f m/s after 100 steps", d0)
    @test d0 > 0.1
    # the sound-proof core: the same call, with its reference θ̄ as the θ data
    sa = AnelasticSimulation(m; bcs = open_boundary_conditions(m, data), f_coriolis = f,
                             sponge_α_max = 0.05, sponge_z_start = 300.0, sponge_target = data)
    only(open_boundary_states(sa)).data = ProfileBoundaryData(u = Ug, v = Vg, θ = sa.ref.θ̄c_pad)
    set!(sa.u, Ug); set!(sa.v, Vg)
    for _ in 1:100; step!(sa, 2.0; body_force! = bf); end
    @test dev(sa) < 1e-12
    @test max_anelastic_divergence(sa) < 1e-9

    # a level-vector target (a function of z) is the profile at each
    # component's own levels; the one sponge call equals add_sponge! with it
    prof = ProfileBoundaryData(u = z -> 2.0 + 0.01z, v = 1.0, w = 0.0, θ = 300.0)
    tp = sponge_target(prof, m)
    zc = [(k - 0.5) * m.dz for k in 1:m.Nz]
    @test tp.u ≈ 2.0 .+ 0.01 .* zc && tp.v == 1.0 && tp.w == 0.0
    sp = BoussinesqSimulation(m; bcs = open_boundary_conditions(m, prof), sponge_target = tp, kw...)
    Random.seed!(7)
    sp.u.data .= randn(size(sp.u.data)); sp.v.data .= randn(size(sp.v.data))
    fill!(sp.tu.data, 0); fill!(sp.tv.data, 0); fill!(sp.tw.data, 0)
    add_velocity_sponge!(sp)
    tu = Field(FaceX(), m)
    add_sponge!(tu, sp.u, 0.05, 300.0, 400.0; f_ref = tp.u)
    @test values_agree(sp.tu.data, tu.data)                     #: one kernel
    # the closed-top default stays the zero target of add_sponge!
    sc = BoussinesqSimulation(m; kw...)
    sc.u.data .= sp.u.data
    fill!(sc.tu.data, 0)
    add_velocity_sponge!(sc)
    tu0 = Field(FaceX(), m)
    add_sponge!(tu0, sc.u, 0.05, 300.0, 400.0)
    @test values_agree(sc.tu.data, tu0.data)
end

# ---------------------------------------------------------------------------
@testset "N-open-d: neighbor_index, the one BC-aware neighbour helper" begin
    m = RegularGrid(size = (8, 6, 4), extent = (8.0, 6.0, 4.0))
    per = FieldBoundaryConditions(:periodic, :periodic, :wall)
    opn = open_boundary_conditions(m, ProfileBoundaryData(u = 1.0, v = 0.0, θ = 300.0))
    # periodic: exactly mod1, for every index a stencil can ask for
    @test all(neighbor_index(m, per, i, 1) == mod1(i, 8) for i in -10:20)
    @test all(neighbor_index(m, per, j, 2; face = true) == mod1(j, 6) for j in -10:20)
    # open: interior indices unchanged; mirror across the face (zero gradient)
    @test all(neighbor_index(m, opn, i, 1) == i for i in 1:8)
    @test neighbor_index(m, opn, 0, 1) == 1 && neighbor_index(m, opn, -1, 1) == 2
    @test neighbor_index(m, opn, 9, 1) == 8 && neighbor_index(m, opn, 10, 1) == 7
    # a staggered field: N+1 is the boundary face (the open face slot)
    @test neighbor_index(m, opn, 9, 1; face = true) == 9
    @test neighbor_index(m, opn, 0, 1; face = true) == 2
    @test neighbor_index(m, opn, 10, 1; face = true) == 8
    # no neighbour beyond a non-periodic face
    @test neighbor_index(m, opn, 0, 2; outside = :none) == 0
    @test neighbor_index(m, opn, 7, 2; outside = :none) == 0
    @test neighbor_index(m, per, 0, 3; outside = :none) == 0          # z walls
    @test neighbor_index(m, per, 0, 1; outside = :none) == 8          # periodic wraps
    @test_throws ArgumentError neighbor_index(m, per, 1, 4)
    @test_throws ArgumentError neighbor_index(m, opn, 0, 1; outside = :clamp)
end

# ---------------------------------------------------------------------------
@testset "N-open-d: a building at the open east face, west inflow" begin
    Nx, Ny, Nz = 16, 12, 8
    m = RegularGrid(size = (Nx, Ny, Nz), extent = (320.0, 240.0, 160.0), halo = (2, 2, 2))
    H = 2
    U = 2.0
    data = ProfileBoundaryData(u = U, v = 0.0, w = 0.0, θ = 300.0)
    # a cube touching the east face: i = Nx−2…Nx, j = 4…8, k = 1…4
    cube(sm) = (for k in 1:4, j in 4:8, i in Nx-2:Nx; sm.mask[H+i, H+j, H+k] = true; end; sm)
    sm = cube(SolidMask(m))
    bcs = open_boundary_conditions(m, data; top = :wall)
    mark_open_boundary_solids!(bcs, sm.mask)
    st = only(open_boundary_states(bcs))
    pe = only(b for b in (bcs.east,) if b isa OpenBoundary).planes[:u]
    @test count(!, pe.fluid) == 5 * 4                         # the cube's 5 × 4 east faces
    @test all(pe.fluid[j, k] == !(4 <= j <= 8 && 1 <= k <= 4) for j in 1:Ny, k in 1:Nz)
    @test all(bcs.west.planes[:u].fluid)                      # nothing touches the west face
    s = BoussinesqSimulation(m; bcs = bcs, θ₀ = 300.0, ν = 1e-2)
    set!(s.u, U); set!(s.θ, 300.0)
    pp(sim, Δt) = implicit_ibm_correction!(sim, sm; τ = 0.0, dt = Δt)
    divs = Float64[]
    for _ in 1:20
        step!(s, 1.0; post_predictor! = pp)
        push!(divs, max_divergence(s))
    end
    @info @sprintf("cube at the open east face: max|∇·u| over 20 steps %.2e, last δ = %.3e m/s, Q = %.3e m³/s",
                   maximum(divs), st.δ, st.Q)
    @test maximum(divs) < 1e-12
    # the solid points carry the solid value: zero on the east face slot …
    ud = s.u.data
    @test all(ud[H+Nx+1, H+j, H+k] == 0 for j in 4:8, k in 1:4)
    @test all(pe.data[:, H+j, H+k] == zeros(H) for j in 4:8, k in 1:4)
    # … and the correction adjusted only the fluid faces: the inflow face
    # carries U − (−δ) and every fluid east face the corrected value, Q after = 0
    @test abs(st.residual) < 1e-12 * U * 320 * 240
    # fluid open area: W + E (less the cube's 20 faces) + S + N (the top is a wall)
    Af = (2Ny * Nz - 20) * m.dy * m.dz + 2Nx * Nz * m.dx * m.dz
    @test st.Q ≈ -U * 20 * m.dy * m.dz rtol = 1e-12               # the blocked outflow
    @test st.δ ≈ st.Q / Af rtol = 1e-12
    @test all(ud[H+Nx+1, H+j, H+k] ≈ U - st.δ for j in 1:Ny, k in 1:Nz if !(4 <= j <= 8 && k <= 4))

    # across the periodic seam the same cube is next to the west edge: the
    # facade wall model has walls there and the wall distance feels it;
    # across the open west face it has neither
    sp_ = BoussinesqSimulation(m; bcs = FieldBoundaryConditions(:periodic, :periodic, :wall),
                               θ₀ = 300.0)
    law = RoughLogLaw(0.05)
    iw_per = immersed_walls(sp_, sm, law)
    iw_open = immersed_walls(s, sm, law)
    west(iw) = count(f -> f.I == H + 1 || (f.c == 1 && f.I == H + 2), iw.faces)
    @info "immersed facade walls at the west edge: periodic $(west(iw_per)), open $(west(iw_open))"
    @test west(iw_per) > 0
    @test west(iw_open) == 0
    d_per = wall_distance(sm; bottom = false)                 # periodic laterally (default)
    d_open = wall_distance(sm; bottom = false, bcs = bcs)
    @test d_per[1, 6, 2] ≈ m.dx / 2                           # the cube across the seam
    @test d_open[1, 6, 2] ≈ (Nx - 3) * m.dx - m.dx / 2        # the cube on the far side
    @test d_open[Nx-3, 6, 2] == d_per[Nx-3, 6, 2]             # unchanged near the cube
    # … and the surface wall tagging (InputDrivers.tag_building_adjacent!,
    # through neighbor_index; moved from test/berlin_open_boundaries.jl,
    # #320): across the periodic seam the street column at the west edge
    # next to the cube becomes a wall, across the open west face it stays
    # street; the cube's inner side is a wall in both
    per = fill(SURFACE_STREET, Nx, Ny); opn = copy(per)
    tag_building_adjacent!(per, sm, m)
    tag_building_adjacent!(opn, sm, m; bcs = bcs)
    @test all(per[1, j] == SURFACE_WALL for j in 4:8)
    @test all(opn[1, j] == SURFACE_STREET for j in 4:8)
    @test per[Nx-3, 4:8] == opn[Nx-3, 4:8] == fill(SURFACE_WALL, 5)
end

# ---------------------------------------------------------------------------
@testset "N-open-d: boundary_scalar_flux — levels, diffusion, mass correction" begin
    Nx, Ny, Nz = 12, 10, 8
    m = RegularGrid(size = (Nx, Ny, Nz), extent = (240.0, 200.0, 160.0), halo = (3, 3, 3))
    V = 240.0 * 200.0 * 160.0
    # a non-uniform stream and θ through all open faces, with Smagorinsky
    data = FunctionBoundaryData(u = (x, y, z, t) -> 2.0 + 0.002y + 0.001z,
                                v = (x, y, z, t) -> -0.5 + 0.001x,
                                w = (x, y, z, t) -> 0.05,
                                θ = (x, y, z, t) -> 300.0 + 0.004x - 0.003y + 0.01z)
    s = BoussinesqSimulation(m; bcs = open_boundary_conditions(m, data), θ₀ = 300.0,
                             C_s = 0.18, ν = 0.5)
    Random.seed!(11)
    set!(s.u, (x, y, z) -> 2.0 + 0.3 * sin(2π * x / 240) * cos(2π * z / 160))
    set!(s.v, (x, y, z) -> -0.5 + 0.2 * cos(2π * y / 200))
    set!(s.θ, (x, y, z) -> 300.0 + 0.004x - 0.003y + 0.01z + 0.2 * sin(2π * x / 120) * sin(2π * y / 100))
    step!(s, 1.0)                                    # a projected, consistent state
    # inside the next step, after the tendencies (κt of this state is in
    # sim.κt), the boundary flux the step uses; the step changes ∫θ by
    # exactly −Δt × (advective + diffusive) boundary flux
    lev = zeros(Nz); clev = zeros(Nz)
    F = Ref{Any}(nothing)
    hook(sim, Δt) = (F[] = boundary_scalar_flux(sim, sim.θ; fill = false, levels = lev,
                                                correction_levels = clev,
                                                diffusivity = (ν = sim.ν, κh = sim.κt, κv = sim.κt)))
    S0 = sum(interior_view(s.θ)) * m.dx * m.dy * m.dz
    Δt = 0.5
    step!(s, Δt; body_force! = hook)
    S1 = sum(interior_view(s.θ)) * m.dx * m.dy * m.dz
    Ft = F[].total
    rel = abs((S1 - S0) + Δt * Ft) / (Δt * sum(abs, (F[].west, F[].east, F[].south, F[].north, F[].top)))
    @info @sprintf("open box, step!: Δ∫θdV = %.6e K m³, −Δt·F = %.6e (relative residual %.1e); F by side W %.3e E %.3e S %.3e N %.3e T %.3e",
                   S1 - S0, -Δt * Ft, rel, F[].west, F[].east, F[].south, F[].north, F[].top)
    @test rel < 1e-12
    @test sum(lev) ≈ Ft rtol = 1e-13
    @test F[].bottom == 0                            # wall bottom
    # without the diffusive part the residual is the diffusive flux, not round-off
    lev2 = zeros(Nz)
    Fa = Ref{Any}(nothing)
    hook2(sim, Δt) = (Fa[] = boundary_scalar_flux(sim, sim.θ; fill = false, levels = lev2))
    S0 = sum(interior_view(s.θ)) * m.dx * m.dy * m.dz
    step!(s, Δt; body_force! = hook2)
    S1 = sum(interior_view(s.θ)) * m.dx * m.dy * m.dz
    @test abs((S1 - S0) + Δt * Fa[].total) > 1e3 * eps() * abs(S0)
    # the correction part: the data carry a net outflow, δ > 0 removes it;
    # with θ uniform the correction's flux is −δ θ₀ × (corrected area) exactly
    st = only(open_boundary_states(s))
    @test st.δ != 0
    sθ = BoussinesqSimulation(m; bcs = open_boundary_conditions(m,
                                 FunctionBoundaryData(u = (x, y, z, t) -> 1.0 + 0.001x, v = 0.0,
                                                      w = 0.0, θ = 300.0); top = :wall), θ₀ = 300.0)
    set!(sθ.θ, 300.0)
    prepare_open_boundaries!(sθ)                     # data + correction, θ still uniform
    stθ = only(open_boundary_states(sθ))
    clev3 = zeros(Nz)
    F3 = boundary_scalar_flux(sθ, sθ.θ; correction_levels = clev3)
    Aopen = 2 * Ny * Nz * m.dy * m.dz + 2 * Nx * Nz * m.dx * m.dz
    @test stθ.δ ≈ 0.001 * 240.0 * Ny * Nz * m.dy * m.dz / Aopen rtol = 1e-12
    @test sum(clev3) ≈ -stθ.δ * 300.0 * Aopen rtol = 1e-12
    @test abs(F3.total) < 1e-14 * 300.0 * 1.24 * Aopen   # the corrected box carries no net θ flux
    # the data's own flux: θ₀ Q, removed exactly by the correction
    @test sum(clev3) ≈ -300.0 * stθ.Q rtol = 1e-12
end

# ---------------------------------------------------------------------------
@testset "N-open-d: a +1 K inflow anomaly — upwind and centred face values" begin
    Nx, Ny, Nz = 10, 6, 4
    m = RegularGrid(size = (Nx, Ny, Nz), extent = (100.0, 60.0, 40.0), halo = (3, 3, 3))
    U, θ0, ΔT, ν = 2.0, 300.0, 1.0, 0.1
    data = FunctionBoundaryData(u = U, v = 0.0, w = 0.0,
                                θ = (x, y, z, t) -> x < 50.0 ? θ0 + ΔT : θ0)
    s = BoussinesqSimulation(m; bcs = open_boundary_conditions(m, data; y = :periodic, top = :wall),
                             θ₀ = θ0, ν = ν)
    set!(s.u, U); set!(s.θ, θ0)
    prepare_open_boundaries!(s)
    A = 60.0 * 40.0
    d = (ν = ν,)
    Fu = boundary_scalar_flux(s, s.θ; scheme = :upwind)
    Fc = boundary_scalar_flux(s, s.θ; scheme = :centered)
    Fw = boundary_scalar_flux(s, s.θ; scheme = :weno_z5)
    Fcd = boundary_scalar_flux(s, s.θ; scheme = :centered, diffusivity = d)
    # into the box: −total. Upwind (and WENO, whose smooth inflow stencil is
    # all θ₀ + ΔT) carry U ΔT A; the centred face value is the mean of the
    # halo and the first cell, so it carries U ΔT A / 2
    @test -Fu.total ≈ U * ΔT * A rtol = 1e-12
    @test -Fw.total ≈ U * ΔT * A rtol = 1e-12
    @test -Fc.total ≈ U * ΔT * A / 2 rtol = 1e-12
    # the diffusive flux across the jump between the halo and cell 1: ν ΔT / Δx
    @test -(Fcd.total - Fc.total) ≈ ν * ΔT / m.dx * A rtol = 1e-12
    # equal inflow: zero to round-off
    s2 = BoussinesqSimulation(m; bcs = open_boundary_conditions(m, ProfileBoundaryData(u = U, v = 0.0,
                              θ = θ0); y = :periodic, top = :wall), θ₀ = θ0, ν = ν)
    set!(s2.u, U); set!(s2.θ, θ0)
    @test abs(boundary_scalar_flux(s2, s2.θ; diffusivity = d).total) < 1e-12 * U * θ0 * A
end

# ---------------------------------------------------------------------------
# #327 gate: boundary_scalar_flux with a diffusivity is type-stable and
# allocation-free. A captured face-flux closure (M2) inferred as
# Tuple{Any,Any,Any}: 313 kB per call on 16³, 43× slower on 96×96×48. The
# Berlin heat and water ledgers call it every step.
function _bsf_bytes(s, sch, d)          # a function barrier: no dispatch in the measurement
    boundary_scalar_flux(s, s.θ; scheme = sch, diffusivity = d, fill = false)
    return @allocated boundary_scalar_flux(s, s.θ; scheme = sch, diffusivity = d, fill = false)
end
@testset "#327: boundary_scalar_flux with diffusivity — inferred, no allocation" begin
    m = RegularGrid(size = (16, 16, 16), extent = (160.0, 160.0, 160.0), halo = (3, 3, 3))
    data = FunctionBoundaryData(u = 2.0, v = -1.0, w = 0.0,
                                θ = (x, y, z, t) -> 300.0 + 0.003z + 0.01x)
    s = BoussinesqSimulation(m; bcs = open_boundary_conditions(m, data; top = :open),
                             θ₀ = 300.0, ν = 0.1)
    set!(s.u, 2.0); set!(s.v, -1.0); set!(s.θ, (x, y, z) -> 300.0 + 0.003z + 0.01x)
    prepare_open_boundaries!(s)
    κ = deepcopy(s.θ); fill!(κ.data, 0.5)               # a cell-centred Field, as the SGS κ
    for K in (κ, κ.data), sch in (:centered, :upwind, :weno_z5)
        d = (ν = 0.1, κh = K, κv = K, scale = 1.0)
        F = @inferred boundary_scalar_flux(s, s.θ; scheme = sch, diffusivity = d, fill = false)
        @test isfinite(F.total) && F.total isa Float64
        # zero bytes for the two-point schemes; the WENO face value boxes one
        # Float64 per face, as on the base commit e2c73f2 (not this regression)
        sch === :weno_z5 && continue
        @test _bsf_bytes(s, sch, d) == 0
    end
end
