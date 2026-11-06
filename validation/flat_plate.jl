# Phase 59 (P59-d, #185) — spatially developing flat-plate boundary
# layer: validation runs — the laminar plate against Blasius (c_f, H, the
# fringe exit) and bypass transition sanity on a coarse 3-D plate.
# The pieces stay tested in test/flat_plate.jl (the shared damping
# profile, Blasius, FreeStreamBoundary, the fringe momentum budget, the
# integral diagnostics, FreeStreamDisturbance).

using Aeolus
using Aeolus.Fields: set!, interior_view
using Aeolus.Operators
using Aeolus.Operators: add_sponge!, smooth_step
using Aeolus.Operators.FiniteVolume: in_damping
using Aeolus.StretchedGrids
using Random
using Printf

# ---------------------------------------------------------------------------
# 4. Laminar plate (Tu = 0, 2-D, Re_x 1.35e4 → 1e5, NoSlipWall from P59-a)
# ---------------------------------------------------------------------------
#
# 256 × 1 × 64 on 512 × 16 δ*₀, dx = 2 δ*₀, dz = δ*₀/4, Re_δ*₀ = 200; RK3 at
# CFL 0.5, 1200 steps (2.3 flow-through times) from a uniform stream, so the
# boundary layer develops from the fringe inflow instead of being imposed.

@testset "P59-d laminar flat plate" begin
    c = flat_plate_setup()
    sim, g = c.sim, c.sim.mesh
    sim.u.data .= 1.0
    t0 = time()
    for n in 1:1200
        step_rk3!(sim, 1.0; body_force! = c.forcing!)
    end
    wall = time() - t0
    @test max_divergence(sim) < 1e-12
    x = collect(Aeolus.Grids.xnodes(g, FaceX()))
    Lp = g.Lx - (c.fringe.profile.stop - c.fringe.profile.start)     # plate length
    mid = findall(xi -> 0.2Lp <= xi <= 0.8Lp, x)
    Re_x = (x .- c.x_le) ./ c.ν
    cf = skin_friction(sim; U_e = 1.0)
    H = shape_factor(sim)
    ratio = cf[mid] ./ blasius_skin_friction.(Re_x[mid])
    @printf("    laminar plate (%.0f s): c_f/c_f,Blasius ∈ [%.4f, %.4f] over Re_x ∈ [%.0f, %.0f]; H ∈ [%.3f, %.3f]\n",
            wall, minimum(ratio), maximum(ratio), Re_x[mid[1]], Re_x[mid[end]],
            minimum(H[mid]), maximum(H[mid]))
    @test all(abs.(ratio .- 1) .< 0.05)
    @test all(abs.(H[mid] .- 2.59) .< 0.1)
    # The plane mean of τ_w(x) is the P59-a wall_shear_stress (one definition).
    τx = cf .* 0.5                                            # U_e = 1
    @test sum(τx) / length(τx) ≈ wall_shear_stress(sim) rtol = 1e-12
    # Fringe: the target is recovered at the fringe exit (x = 0) to 1 % of U.
    ub = streamwise_profile(sim)
    ut = interior_view(c.target.u)[:, 1, :]
    err_exit = maximum(abs, ub[1, :] .- ut[1, :])
    @printf("    fringe exit: max|u − u_target| = %.2e U; inflow δ* = %.4f δ*₀, θ = %.4f δ*₀\n",
            err_exit, displacement_thickness(ub, g.dz)[1], momentum_thickness(ub, g.dz)[1])
    @test err_exit < 0.01
    # The free stream enters the plate at U∞ (the transpiration top keeps
    # it from accelerating by Δδ*/H).
    Ue = edge_velocity(ub)
    @test maximum(abs.(Ue[mid] .- 1)) < 0.005
end

# ---------------------------------------------------------------------------
# 6. Bypass sanity (coarse 3-D): Tu = 6 % transitions; Tu = 0 (2-D) and a
#    3-D layer seeded with Tu = 0.1 % stay laminar
# ---------------------------------------------------------------------------
#
# Re_δ*₀ = 300, 400 × 36 × 24 δ*₀ on 96 × 24 × 48 (dx = 4.2, dy = 1.5,
# dz = 0.5 δ*₀), 80 δ*₀ fringe, plate Re_x = 3.0e4 → 1.26e5 at the fringe
# start; c_f is averaged over the last 200 of 700 steps (Δt = 1.5). No SGS
# model (a coarse DNS: the Smagorinsky ν_t of the laminar wall shear alone
# would be ≈ 4ν here). Free-stream turbulence (L11 = 10 δ*₀, 200 modes,
# κ_max = 2π/3Δ) enters through the fringe target, kept out of the inflow
# layer below 1.5 δ₉₉.
#
# Three runs:
#   * Tu = 6 %, 3-D: must transition (outlet c_f well above Blasius).
#   * Tu = 0, 2-D (Ny = 1): the cheap exact laminar check. Without a seed
#     the 3-D run stays exactly 2-D, so this is that run at 1/24 the cost.
#   * Tu = 0.1 %, 3-D on the same grid (maintainer decision on #226): the
#     same generator at a tiny amplitude. The layer must stay laminar
#     under small 3-D disturbances: c_f stays on the 2-D control and on
#     Blasius, and the spanwise velocity inside the layer stays O(Tu)
#     and does not grow once the disturbances have filled the domain.
#     Re_δ* reaches 589 at 0.9 of the plate, above the Tollmien–Schlichting
#     critical 520 (Re_x ≈ 9.1e4), but the unstable stretch is far too short
#     (amplification e^N with N ≪ 1) for TS waves to reach transition.

function _bypass_run(Tu; Ny, nsteps = 700, navg = 200, dt = 1.5, Lx = 400.0, every = 50)
    c = flat_plate_setup(; Re_δ0 = 300, Lx = Lx, Ly = 36, H = 24, Nx = round(Int, Lx / 4.1667),
                         Ny = Ny, Nz = 48, fringe_length = 80)
    sim, g = c.sim, c.sim.mesh
    force! = c.forcing!
    if Tu > 0
        fst = FreeStreamDisturbance(Tu, 10.0, 200; seed = 1, period_y = g.Ly,
                                    κ_max = 2π / (3 * max(g.dx, g.dy, g.dz)))
        δ99 = blasius_solution().eta99 * sqrt(c.ν * c.x_in)
        tgt = deepcopy(c.target)
        upd = disturbance_fringe_update(fst, c.fringe, c.target;
                                        damping = z -> smooth_step(z / (1.5δ99)))
        force! = fringe_forcing(c.fringe, tgt; update! = upd)
    end
    x = collect(Aeolus.Grids.xnodes(g, FaceX()))
    z = collect(Aeolus.Grids.znodes(g, CellCenter()))
    Lp = g.Lx - 80
    plate = findall(xi -> 0.05Lp <= xi <= 0.9Lp, x)
    η99 = blasius_solution().eta99
    # rms spanwise velocity over the plate, inside the Blasius δ₉₉(x) and in
    # the free stream above it (up to 4 δ*₀ below the top)
    function v_rms(sim)
        vi = interior_view(sim.v)
        sbl = 0.0; nbl = 0; sfs = 0.0; nfs = 0
        for i in plate, k in 1:g.Nz
            vv = sum(abs2, @view vi[i, :, k]) / g.Ny
            if z[k] < η99 * sqrt(c.ν * (x[i] - c.x_le))
                sbl += vv; nbl += 1
            elseif z[k] < g.Lz - 4
                sfs += vv; nfs += 1
            end
        end
        return sqrt(sbl / nbl), sqrt(sfs / nfs)
    end
    # both runs start from the laminar Blasius layer: the disturbances must trip it
    set_blasius!(sim; x_le = c.x_le, x_min = c.x_in)
    cf = zeros(g.Nx)
    vbl = Float64[]; vfs = Float64[]
    for n in 1:nsteps
        step_rk3!(sim, dt; body_force! = force!)
        n > nsteps - navg && (cf .+= skin_friction(sim; U_e = 1.0))
        if n % every == 0
            a, b = v_rms(sim); push!(vbl, a); push!(vfs, b)
        end
    end
    cf ./= navg
    Re_x = (x .- c.x_le) ./ c.ν
    out = findall(xi -> 0.75Lp <= xi <= 0.9Lp, x)
    return (ratio = sum(cf[out]) / sum(blasius_skin_friction.(Re_x[out])),
            div = max_divergence(sim), Re_out = (Re_x[out[1]], Re_x[out[end]]),
            cf = cf, x = x, Lp = Lp, Re_x = Re_x, plate = plate, v_bl = vbl, v_fs = vfs, every = every)
end

@testset "P59-d bypass transition sanity (coarse 3-D)" begin
    t0 = time(); r6 = _bypass_run(0.06; Ny = 24); t6 = time() - t0
    t0 = time(); r0 = _bypass_run(0.0; Ny = 1); t00 = time() - t0
    t0 = time(); rs = _bypass_run(0.001; Ny = 24); ts = time() - t0
    @printf("    bypass: outlet c_f / c_f,Blasius (Re_x %.0f–%.0f) = %.3f at Tu = 6 %% (%.0f s), %.4f at Tu = 0 2-D (%.0f s), %.4f at Tu = 0.1 %% 3-D (%.0f s)\n",
            r6.Re_out..., r6.ratio, t6, r0.ratio, t00, rs.ratio, ts)
    @test r6.ratio > 1.5
    @test abs(r0.ratio - 1) < 0.05
    @test r6.div < 1e-12 && r0.div < 1e-12 && rs.div < 1e-12

    # --- the seeded 3-D laminar control ---
    p = rs.plate                                   # 0.05–0.9 of the plate, Re_x 3.5e4–1.17e5
    # (a) the seed leaves the wall shear on the unseeded 2-D control,
    # point by point (measured within 0.19 %). Transition would raise it
    # by tens of % (Tu = 6 %: 3.5× at the outlet); 1 % is still 5× the
    # measured change.
    seed_effect = rs.cf[p] ./ r0.cf[p]
    # (b) against Blasius over the plate length. Both runs carry the same
    # stationary 2Δx (odd–even) wiggle in c_f on this dx = 4.2 δ*₀ grid:
    # ±0.6 % near the inflow, growing to ±17 % just upstream of the fringe.
    # It is a resolution artefact, not physics: the 2-D control at dx = 2.1
    # has it at ±0.03 %. The [1 2 1]/4 filter removes exactly that mode, so
    # the comparison with Blasius is made on the filtered c_f. Tolerance
    # 5 %, the same as the 2-D laminar plate above: Blasius is the
    # parallel-flow similarity solution and the plate's ends feel the inflow
    # (−2 %) and the elliptic fringe; transition would be +50 % or more.
    # The window stops at 0.85 of the plate, 48 δ*₀ before the fringe: at
    # 0.9 of the plate the fringe alone lifts c_f by 5 % (filtered 1.052
    # here, and 1.053 in the 2-D Tu = 0 run at dx = 2.1, so it is neither
    # the seed nor the grid). The first version of this check ran to 0.9 of
    # the plate and failed on that last point; the expectation, not the
    # code, was wrong.
    pb = filter(i -> rs.x[i] <= 0.85rs.Lp, p)  # Re_x 3.5e4–1.12e5
    filt(i) = (rs.cf[i-1] + 2rs.cf[i] + rs.cf[i+1]) / 4
    vs_blasius = filt.(pb) ./ blasius_skin_friction.(rs.Re_x[pb])
    raw = rs.cf[pb] ./ blasius_skin_friction.(rs.Re_x[pb])
    end_ratio = filt(p[end]) / blasius_skin_friction(rs.Re_x[p[end]])
    @printf("    seeded 3-D control: c_f / c_f(2-D, Tu = 0) ∈ [%.4f, %.4f] over Re_x %.0f–%.0f\n",
            extrema(seed_effect)..., rs.Re_x[p[1]], rs.Re_x[p[end]])
    @printf("    seeded 3-D control: [1 2 1]-filtered c_f / Blasius ∈ [%.4f, %.4f] (unfiltered [%.4f, %.4f]) over Re_x %.0f–%.0f; %.4f at Re_x %.0f (0.9 of the plate)\n",
            extrema(vs_blasius)..., extrema(raw)..., rs.Re_x[pb[1]], rs.Re_x[pb[end]],
            end_ratio, rs.Re_x[p[end]])
    @test all(abs.(seed_effect .- 1) .< 0.01)
    @test all(abs.(vs_blasius .- 1) .< 0.05)
    # (c) spanwise velocity in the layer. The seed must reach it (else the
    # test is vacuous): free-stream v_rms ≈ Tu (measured 0.9e-3) and the layer's v_rms well
    # above round-off. It must stay O(Tu): below 0.5 % of U at every sample
    # (a turbulent layer has v_rms ≈ u_τ ≈ 4–5 % U; the Tu = 6 % run has
    # ≈ 8 %). And it must not grow once the disturbances have filled the
    # domain (one flow-through ≈ 270 steps): the mean over steps 550–700 is
    # at most 1.2× the mean over steps 400–550.
    mid = findall(n -> 400 < n * rs.every <= 550, eachindex(rs.v_bl))
    late = findall(n -> n * rs.every > 550, eachindex(rs.v_bl))
    v_mid = sum(rs.v_bl[mid]) / length(mid); v_late = sum(rs.v_bl[late]) / length(late)
    @printf("    seeded 3-D control: v_rms in the layer %s (×1e-3, every %d steps); free stream %.2e at the end; Tu = 6 %%: layer %.2e\n",
            join((@sprintf("%.2f", 1e3v) for v in rs.v_bl), " "), rs.every, rs.v_fs[end], r6.v_bl[end])
    @printf("    seeded 3-D control: layer v_rms steps 400–550 → 550–700: %.2e → %.2e (ratio %.3f)\n",
            v_mid, v_late, v_late / v_mid)
    @test 0.5e-3 < rs.v_fs[end] < 1.5e-3
    @test minimum(rs.v_bl[mid]) > 1e-4
    @test maximum(rs.v_bl) < 5e-3
    @test v_late / v_mid < 1.2
    @test r6.v_bl[end] > 10 * maximum(rs.v_bl)
end
