# Era VI, Phase 57 (#169): WallModels — equilibrium wall laws and the
# implicit wall-stress operator.
#
# Physics asserted: Spalding's law and its inversion (round trip, the
# viscous and logarithmic limits), the delegation of the rough laws to
# SurfaceExchange (κ, ψ_m, the surface-layer solver), the
# viscous limit against the resolved cut-cell stress; the operator's
# momentum budget (the momentum removed is the wall stress at the
# end-of-step velocity times area and Δt) and its unconditional
# stability, and the three formulations.

using Test
using Printf
using Random
using Aeolus
using Aeolus.WallModels
using Aeolus.SurfaceExchange: BusingerDyerBeljaarsHoltslag, solve_surface_layer, von_karman

@testset "Era VI — WallModels (#169)" begin

@testset "Spalding: round trip over y⁺ ∈ [1e-2, 1e6]" begin
    law = SpaldingLaw()
    ys = exp10.(range(-2, 6; length = 200))
    # y⁺ → u⁺ (Newton) → y⁺ (explicit formula)
    err_y = maximum(y -> abs(spalding_yplus(law, spalding_uplus(law, y)) - y) / y, ys)
    # and through friction_velocity: U = u⁺ u_τ at h = y⁺ ν/u_τ gives back u_τ
    ν, uτ = 1.5e-5, 0.3
    err_u = maximum(ys) do y
        up = spalding_uplus(law, y)
        abs(friction_velocity(law, up * uτ, y * ν / uτ; ν = ν) - uτ) / uτ
    end
    @info @sprintf("  Spalding round trip, 200 points: max rel error y⁺ %.1e, u_τ %.1e", err_y, err_u)
    @test err_y < 1e-10
    @test err_u < 1e-10
    # the Lee & Moser (2015) instance is the same law with other constants
    lm = LEE_MOSER_2015
    @test lm.κ == 0.384 && lm.B == 4.27
    @test maximum(y -> abs(spalding_yplus(lm, spalding_uplus(lm, y)) - y) / y, ys) < 1e-10
end

@testset "Spalding: viscous and logarithmic limits; monotone in U" begin
    law = SpaldingLaw()
    κ, B = law.κ, law.B
    up = spalding_uplus(law, 0.1)
    @test abs(up - 0.1) / 0.1 < 1e-3                  # measured 1.4e-7
    # The log law. The issue asked for |u⁺ − (ln y⁺/κ + B)| < 1e-3 at
    # y⁺ = 1e4; that EXPECTATION is wrong for Spalding's formula, not the
    # code: the formula truncates the exponential after the cubic, so at
    # finite y⁺ it lies below the log law in y⁺ by
    # δy⁺ = u⁺ − e^{−κB}(1 + x + x²/2 + x³/6), x = κu⁺, i.e. above it in u⁺
    # by δu⁺ ≈ −δy⁺/(dy⁺/du⁺). At y⁺ = 1e4 that is 2.5e-3 (measured
    # 2.506e-3). Asserted instead: the deviation equals this first-order
    # estimate, and it is below 1e-3 from y⁺ = 1e5 on (measured 7.2e-4,
    # 1.4e-4 at 1e6).
    loglaw(y) = log(y) / κ + B
    for y in (1e4, 1e5, 1e6)
        u = spalding_uplus(law, y)
        x = κ * u
        δy = u - exp(-κ * B) * (1 + x + x^2 / 2 + x^3 / 6)
        δu_est = -δy / (1 + κ * exp(-κ * B) * (exp(x) - 1 - x - x^2 / 2))
        δu = u - loglaw(y)
        @info @sprintf("  Spalding − log law at y⁺ = %.0e: %.3e (first-order estimate %.3e)", y, δu, δu_est)
        @test δu > 0 && abs(δu - δu_est) < 0.02 * δu
        y >= 1e5 && @test abs(δu) < 1e-3
    end
    # friction velocity is strictly increasing in U, through every regime
    ν, h = 1.5e-5, 0.01
    Us = exp10.(range(-6, 2; length = 400))
    uτ = [friction_velocity(law, U, h; ν = ν) for U in Us]
    @test all(diff(uτ) .> 0)
    @test friction_velocity(law, -2.0, h; ν = ν) == friction_velocity(law, 2.0, h; ν = ν)
    @test friction_velocity(law, 0.0, h; ν = ν) == 0
    @test_throws ArgumentError friction_velocity(law, 1.0, h)          # ν is required
end

@testset "Delegation: rough laws use SurfaceExchange's κ and ψ_m" begin
    U, h, z0 = 2.0, 2.5, 0.05
    κ = von_karman()
    ref = κ * U / log(h / z0)
    @test friction_velocity(RoughLogLaw(z0), U, h) == ref
    @test wall_stress(RoughLogLaw(z0), U, h) == ref^2
    # neutral MOST (L = ±Inf) is the rough log law, to 1e-12
    sf = BusingerDyerBeljaarsHoltslag()
    for L in (Inf, -Inf)
        @test abs(friction_velocity(MOSTWallLaw(sf, z0, L), U, h) - ref) / ref < 1e-12
    end
    # stable / unstable: the solver's u_*, at ζ = ±0.5.
    # θ_a − θ_s is bisected until the solver's root is ζ = ±0.5 to 1e-6, and
    # the law is built from that solve's L. The law sees ζ' = h/L, which
    # need not be the solver's ζ in the last bit (h/(h/ζ) ≠ ζ for about half
    # of all ζ; L ↦ h/L skips representable values), so among the solves
    # within 1e-10 K of the bisected one the first with h/L == ζ is taken:
    # is then the claim that the law computes exactly what the
    # solver computes at the same ζ. For any L it is round-off (1e-14).
    function solve_at(ζt)
        lo, hi = ζt > 0 ? (0.0, 30.0) : (-30.0, 0.0)
        mk(Δ) = solve_surface_layer(sf; U = U, z_m = h, θ_a = 290.0 + Δ, θ_s = 290.0,
                                    z0m = z0, z0h = z0 / 10)
        for _ in 1:80
            Δ = (lo + hi) / 2
            (mk(Δ).ζ < ζt) ? (lo = Δ) : (hi = Δ)
        end
        Δ0 = (lo + hi) / 2
        for n in 0:200
            sol = mk(Δ0 + n * 5e-13)
            h / sol.L == sol.ζ && return sol, mk(Δ0)
        end
        error("no solve with h/L == ζ near ζ = $ζt")
    end
    for ζt in (0.5, -0.5)
        sol, sol0 = solve_at(ζt)
        @test abs(sol.ζ - ζt) < 1e-6 && sol.converged
        @test abs(friction_velocity(MOSTWallLaw(sf, z0, sol0.L), U, h) - sol0.u_star) <
              1e-14 * sol0.u_star
        @test h / sol.L == sol.ζ                       # the law sees the solver's ζ
        law = MOSTWallLaw(sf, z0, sol.L)
        @test friction_velocity(law, U, h) === sol.u_star
        @info @sprintf("  MOST at ζ = %+.1f: u_* = %.15f (log law %.15f)", ζt, sol.u_star, ref)
        @test ζt > 0 ? sol.u_star < ref : sol.u_star > ref
        # and the Obukhov length that reproduces a u_* is that L
        Lm = matching_obukhov_length(sf, sol.u_star, U, h, z0)
        @test abs(Lm - sol.L) / abs(sol.L) < 1e-10
    end
    # the minimum wind of the surface-layer solver applies to the law as well
    sol = solve_surface_layer(sf; U = 0.02, z_m = h, θ_a = 290.0, θ_s = 290.0, z0m = z0, z0h = z0)
    @test friction_velocity(MOSTWallLaw(sf, z0, Inf), 0.02, h) ≈ sol.u_star rtol = 1e-12
    @test wall_drag(MOSTWallLaw(sf, z0, Inf), 0.02, h) ≈ sol.u_star^2 / 0.1 rtol = 1e-12
end

@testset "Laminar limit: Spalding at h⁺ < 1 is the viscous stress νU/h, as CutCell's" begin
    law = SpaldingLaw()
    ν = 1.5e-2
    # the resolved cut-cell stress of a linear profile u = α (z − z_w) at a
    # flat cut-cell floor (as in era2_cutcell.jl): ν α at every stencil
    g = RegularGrid(; size = (4, 4, 32), extent = (1.0, 1.0, 1.0), halo = (2, 2, 2))
    zw = (32 ÷ 4 + 0.7) * g.dz
    geom = Aeolus.CutCell.cut_cell_geometry(g, (x, y, z) -> z - zw)
    st = Aeolus.CutCell.build_ghost_stencils(geom, g)
    α = 2.5
    u = Field(FaceX(), g); v = Field(FaceY(), g); w = Field(FaceZ(), g)
    for k in axes(u.data, 3), j in axes(u.data, 2), i in axes(u.data, 1)
        u.data[i, j, k] = α * ((k - g.Hz - 0.5) * g.dz - zw)
    end
    τc = wall_shear_stress(u, v, w, st, geom, ν)
    # the wall model at each stencil's probe: U = α·probe at h = probe
    worst, hp_max = 0.0, 0.0
    for (m, s) in enumerate(st)
        s.valid || continue
        hprobe = s.probe
        U = α * hprobe
        τm = wall_stress(law, U, hprobe; ν = ν)
        hp_max = max(hp_max, hprobe * sqrt(τm) / ν)
        worst = max(worst, abs(τm - ν * U / hprobe) / (ν * U / hprobe), abs(τm - τc[m]) / τc[m])
    end
    @info @sprintf("  laminar: h⁺ ≤ %.2f, max |τ_Spalding − νU/h|, |τ_Spalding − τ_CutCell| rel %.1e", hp_max, worst)
    @test hp_max < 1
    @test worst < 1e-3
    # and the drag coefficient's U → 0 limit is ν/h
    @test wall_drag(law, 0.0, 0.01; ν = ν) == ν / 0.01
    @test wall_drag(law, 1e-9, 0.01; ν = ν) ≈ ν / 0.01 rtol = 1e-9
end


# -------------------------------------------------------------------------
# The operator
# -------------------------------------------------------------------------

# A small channel: periodic in x, y, free-slip (ImpermeableWall) z-walls.
function wall_sim(; N = (8, 6, 5), extent = (8.0, 6.0, 2.5))
    g = RegularGrid(; size = N, extent = extent, halo = (2, 2, 2))
    return BoussinesqSimulation(g; bcs = FieldBoundaryConditions(:periodic, :periodic, :wall))
end

function fill_random!(sim; seed = 1)
    rng = Random.MersenneTwister(seed)
    for f in (sim.u, sim.v, sim.w)
        iv = Aeolus.Fields.interior_view(f)
        iv .= 2 .+ randn(rng, size(iv))
    end
    fill_halos!(sim.u, sim.bcs); fill_halos!(sim.v, sim.bcs); fill_halos!(sim.w, sim.bcs)
    return sim
end

# The expected stress per face, computed independently of the operator:
# column speeds from the start-of-step velocity, the law's drag per
# column, the mean of the two columns' drags on each face, times the
# END-of-step face velocity (the implicit stress).
function expected_impulse(sim0, sim1, law, Δt, k; ν = 1.5e-5)
    g = sim0.mesh
    Nx, Ny, dz = g.Nx, g.Ny, g.dz
    u0, v0 = Aeolus.Fields.interior_view(sim0.u), Aeolus.Fields.interior_view(sim0.v)
    u1, v1 = Aeolus.Fields.interior_view(sim1.u), Aeolus.Fields.interior_view(sim1.v)
    D = [wall_drag(law, hypot((u0[i, j, k] + u0[mod1(i + 1, Nx), j, k]) / 2,
                              (v0[i, j, k] + v0[i, mod1(j + 1, Ny), k]) / 2), dz / 2; ν = ν)
         for i in 1:Nx, j in 1:Ny]
    dA = g.dx * g.dy
    Ix = sum((D[i, j] + D[mod1(i - 1, Nx), j]) / 2 * u1[i, j, k] for i in 1:Nx, j in 1:Ny) * dA * Δt
    Iy = sum((D[i, j] + D[i, mod1(j - 1, Ny)]) / 2 * v1[i, j, k] for i in 1:Nx, j in 1:Ny) * dA * Δt
    return Ix, Iy
end

@testset "Operator: momentum removed = Σ τ_w ΔA Δt with the end-of-step velocity" begin
    for (law, name) in ((SpaldingLaw(), "Spalding"), (RoughLogLaw(0.01), "rough log"),
                        (MOSTWallLaw(0.01, 50.0), "MOST stable"))
        for side in (:bottom, :top, :both)
            sim = fill_random!(wall_sim())
            g = sim.mesh
            ref = deepcopy(sim)
            wm = WallModel(law; ν = 1.5e-5)
            Δt = 0.7
            F = apply_wall_stress!(sim, wm, Δt; side = side)
            dV = g.dx * g.dy * g.dz
            u0, u1 = Aeolus.Fields.interior_view(ref.u), Aeolus.Fields.interior_view(sim.u)
            v0, v1 = Aeolus.Fields.interior_view(ref.v), Aeolus.Fields.interior_view(sim.v)
            ΔMx = sum(u1 .- u0) * dV
            ΔMy = sum(v1 .- v0) * dV
            levels = side === :both ? (1, g.Nz) : side === :bottom ? (1,) : (g.Nz,)
            Ix = sum(expected_impulse(ref, sim, law, Δt, k)[1] for k in levels)
            Iy = sum(expected_impulse(ref, sim, law, Δt, k)[2] for k in levels)
            ex = abs(ΔMx + Ix) / abs(Ix); ey = abs(ΔMy + Iy) / abs(Iy)
            side === :both && @info @sprintf("  %-11s Δ momentum vs −Σ τ_w ΔA Δt: rel %.1e, %.1e", name, ex, ey)
            @test ex < 1e-12 && ey < 1e-12
            @test F.Fx * Δt ≈ ΔMx rtol = 1e-12
            @test F.Fy * Δt ≈ ΔMy rtol = 1e-12
            # w and every level without a wall are untouched
            @test agree(Aeolus.Fields.interior_view(sim.w), Aeolus.Fields.interior_view(ref.w))
            inner = setdiff(1:g.Nz, levels)
            @test values_agree(u1[:, :, inner], u0[:, :, inner]) && values_agree(v1[:, :, inner], v0[:, :, inner])
            @test all(abs.(u1[:, :, collect(levels)]) .< abs.(u0[:, :, collect(levels)]))
        end
    end
end

@testset "Operator: implicit, stable at 100× the explicit limit" begin
    for law in (RoughLogLaw(0.05), SpaldingLaw())
        sim = wall_sim()
        g = sim.mesh
        u = Aeolus.Fields.interior_view(sim.u); v = Aeolus.Fields.interior_view(sim.v)
        u .= 3.0; v .= -1.0
        U0 = hypot(3.0, 1.0)
        ν = law isa SpaldingLaw ? 1e-2 : 1.5e-5
        wm = WallModel(law; ν = ν)
        # explicit Euler u ← u (1 − Δt D/Δz) is stable for Δt D/Δz < 2 and
        # changes sign beyond Δt D/Δz = 1
        Δt_exp = g.dz / wall_drag(law, U0, g.dz / 2; ν = ν)
        Δt = 100 * Δt_exp
        @test 1 - Δt * wall_drag(law, U0, g.dz / 2; ν = ν) / g.dz < -1    # explicit: unstable
        us, vs = [3.0], [-1.0]
        for n in 1:40
            apply_wall_stress!(sim, wm, Δt)
            push!(us, u[1, 1, 1]); push!(vs, v[1, 1, 1])
            @test all(==(u[1, 1, 1]), u[:, :, 1]) && all(==(3.0), u[:, :, 2:end])
        end
        @info @sprintf("  %s, Δt = 100 Δt_explicit = %.3g s: u_1 %.3f → %.3e after 40 steps (monotone, no sign change)",
                       nameof(typeof(law)), Δt, us[1], us[end])
        @test all(diff(us) .< 0) && all(>(0), us)
        @test all(diff(vs) .> 0) && all(<(0), vs)
        # the direction of the stress is the wind's: u/v keeps its ratio
        @test all(i -> us[i] / vs[i] ≈ -3.0, eachindex(us))
    end
end

@testset "Operator: formulations, masks, held wind" begin
    law = RoughLogLaw(0.02)
    h = 2.5 / 5 / 2
    # uniform wind: the three formulations coincide
    outs = map((:local, :schumann, :moeng)) do form
        sim = wall_sim()
        Aeolus.Fields.interior_view(sim.u) .= 2.0
        Aeolus.Fields.interior_view(sim.v) .= 1.0
        us = zeros(8, 6)
        apply_wall_stress!(sim, law, 0.5; formulation = form, ustar = us)
        (copy(Aeolus.Fields.interior_view(sim.u)[:, :, 1]), us)
    end
    @test outs[2][1] ≈ outs[1][1] rtol = 1e-14
    @test outs[3][1] ≈ outs[1][1] rtol = 1e-14
    @test all(≈(friction_velocity(law, hypot(2.0, 1.0), h)), outs[1][2])
    # Schumann: one drag for the plane, from the plane-mean speed; Moeng:
    # the plane-mean stress is ⟨τ_w⟩ along the mean wind (small Δt, so the
    # implicit factor is 1 to 1e-8)
    sim = fill_random!(wall_sim(); seed = 3)
    u0 = copy(Aeolus.Fields.interior_view(sim.u)[:, :, 1])
    v0 = copy(Aeolus.Fields.interior_view(sim.v)[:, :, 1])
    Uc = [hypot((u0[i, j] + u0[mod1(i + 1, 8), j]) / 2, (v0[i, j] + v0[i, mod1(j + 1, 6)]) / 2)
          for i in 1:8, j in 1:6]
    Ubar = sum(Uc) / length(Uc)
    s2 = deepcopy(sim)
    apply_wall_stress!(s2, law, 1e-6; formulation = :schumann)
    D = wall_drag(law, Ubar, h)
    @test Aeolus.Fields.interior_view(s2.u)[:, :, 1] ≈ u0 ./ (1 + 1e-6 * D / 0.5) rtol = 1e-14
    Δt = 1e-6
    F = apply_wall_stress!(sim, law, Δt; formulation = :moeng)
    ub, vb = sum(u0) / length(u0), sum(v0) / length(v0)
    τm = wall_stress(law, Ubar, h)
    A = 8.0 * 6.0
    @test -F.Fx / A ≈ τm * ub / hypot(ub, vb) rtol = 1e-6
    @test -F.Fy / A ≈ τm * vb / hypot(ub, vb) rtol = 1e-6
    # a Bool mask: only masked columns' faces change; half the drag on a face
    # with one walled neighbour
    sim = wall_sim()
    Aeolus.Fields.interior_view(sim.u) .= 2.0; Aeolus.Fields.interior_view(sim.v) .= 0.0
    m = falses(8, 6); m[3:4, 2] .= true
    apply_wall_stress!(sim, law, 0.5; mask = m)
    u = Aeolus.Fields.interior_view(sim.u)
    a = wall_drag(law, 2.0, h)
    @test u[4, 2, 1] ≈ 2 / (1 + 0.5 * a / 0.5) rtol = 1e-14            # both sides walled
    @test u[3, 2, 1] ≈ 2 / (1 + 0.5 * (a / 2) / 0.5) rtol = 1e-14      # one side
    @test u[5, 2, 1] ≈ 2 / (1 + 0.5 * (a / 2) / 0.5) rtol = 1e-14
    @test count(!=(2.0), u) == 3
    # a level mask (an immersed floor): the wall acts at each column's level
    sim = wall_sim()
    Aeolus.Fields.interior_view(sim.u) .= 2.0; Aeolus.Fields.interior_view(sim.v) .= 0.0
    lv = ones(Int, 8, 6); lv[5:8, :] .= 3; lv[1, 1] = 99           # 99: no wall
    us, Uo = zeros(8, 6), zeros(8, 6)
    apply_wall_stress!(sim, law, 0.5; mask = lv, ustar = us, wind_out = Uo)
    u = Aeolus.Fields.interior_view(sim.u)
    @test u[6, 3, 3] < 2 && u[6, 3, 1] == 2 && u[2, 3, 1] < 2 && u[2, 3, 3] == 2
    @test u[5, 3, 1] < 2 && u[5, 3, 3] < 2                          # shared face, two levels
    @test us[1, 1] == 0 && all(us[I] > 0 for I in CartesianIndices(us) if I != CartesianIndex(1, 1))
    @test Uo[2, 2] == 2.0
    # a held wind: the stress has the MAGNITUDE of the law at that speed
    # and the direction of the resolved velocity, τ_i = τ_w(5) u_i/U with
    # U = 2, so a = τ_w(5)/2. (P57-b: the #169 expectation here was
    # a = wall_drag(law, 5) = τ_w(5)/5, a stress magnitude τ_w(5)·2/5,
    # contradicting the docstring; the code and this expectation were
    # both wrong and are fixed together.)
    sim = wall_sim()
    Aeolus.Fields.interior_view(sim.u) .= 2.0; Aeolus.Fields.interior_view(sim.v) .= 0.0
    F = apply_wall_stress!(sim, law, 0.5; wind = fill(5.0, 8, 6))
    @test Aeolus.Fields.interior_view(sim.u)[1, 1, 1] ≈ 2 / (1 + 0.5 * (wall_stress(law, 5.0, h) / 2) / 0.5) rtol = 1e-14
    # small Δt: the explicit limit of the plane-mean stress is τ_w(5) itself
    sim = wall_sim()
    Aeolus.Fields.interior_view(sim.u) .= 2.0; Aeolus.Fields.interior_view(sim.v) .= 0.0
    F = apply_wall_stress!(sim, law, 1e-8; wind = fill(5.0, 8, 6))
    @test -F.Fx / (8.0 * 6.0) ≈ wall_stress(law, 5.0, h) rtol = 1e-7
    # Schumann with an input wind: ⟨τ⟩ = τ_w(⟨U_in⟩)
    sim = wall_sim()
    Aeolus.Fields.interior_view(sim.u) .= 2.0; Aeolus.Fields.interior_view(sim.v) .= 0.0
    F = apply_wall_stress!(sim, law, 1e-8; wind = fill(5.0, 8, 6), formulation = :schumann)
    @test -F.Fx / (8.0 * 6.0) ≈ wall_stress(law, 5.0, h) rtol = 1e-7
    # a matching height: the law evaluated at `height`, applied at the
    # first level; for the log law τ_w ∝ [κU/ln(h/z0)]²
    for hm in (0.75, fill(1.25, 8, 6))
        sim = wall_sim()
        Aeolus.Fields.interior_view(sim.u) .= 2.0; Aeolus.Fields.interior_view(sim.v) .= 0.0
        us = zeros(8, 6)
        F = apply_wall_stress!(sim, law, 1e-8; wind = fill(3.0, 8, 6), height = hm, ustar = us)
        hh = hm isa Real ? hm : hm[1]
        @test us[3, 3] ≈ Aeolus.SurfaceExchange.von_karman() * 3.0 / log(hh / 0.02) rtol = 1e-12
        @test -F.Fx / (8.0 * 6.0) ≈ (Aeolus.SurfaceExchange.von_karman() * 3.0 / log(hh / 0.02))^2 rtol = 1e-7
        @test all(==(2.0), Aeolus.Fields.interior_view(sim.u)[:, :, 2:end])   # only level 1 changes
    end
    @test_throws ArgumentError apply_wall_stress!(sim, law, 0.5; formulation = :bogus)
    @test_throws ArgumentError apply_wall_stress!(sim, law, 0.5; side = :top, mask = lv)
end

end
