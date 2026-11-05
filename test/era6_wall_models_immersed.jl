# Era VI, Phase 57-d (#179): the wall model on immersed (IBM mask) and
# cut-cell walls.
#
# Physics asserted: (1) the rotated channel at Re_τ = 1000 (IBM walls in y
# vs z-walls, Lee & Moser U_b⁺) is a validation run, in
# validation/era6_wall_models_immersed.jl; (2) no double counting: the momentum a step removes at an IBM wall
# is Σ τ_w ΔA Δt to round-off, the stencil's own flux being removed;
# (3) on a cut-cell channel inclined at 30°, the modelled plane-mean wall
# stress is δ dp/dx and the ν-linear one is far below it; (4) in the
# viscous sublayer both methods give the ν-linear stress.

using Test
using Printf
using Aeolus
using Aeolus.WallModels
using Aeolus.IBM: SolidMask, mark_solid!, implicit_ibm_correction!
using Aeolus.Fields: interior_view

const IV = interior_view

@testset "Era VI — wall model on immersed and cut-cell walls (#179)" begin

# -------------------------------------------------------------------------
# 2. No double counting
# -------------------------------------------------------------------------

# A step of `step!` on uniform tangential flow past one IBM wall, the
# momentum change of the fluid faces measured after the post-predictor
# hooks (before the projection, which cannot change the x-momentum of a
# periodic row). In uniform flow every interior stencil flux vanishes, so
# the only momentum flux is the one through the wall.
function one_wall_step(; comp = 1, model = true, law = RoughLogLaw(0.01), Δt = 0.3)
    if comp == 1          # u past the walls of solid rows j = 9, 10 (periodic: ±y)
        g = RegularGrid(; size = (8, 10, 4), extent = (8.0, 10.0, 4.0), halo = (2, 2, 2))
        bcs = FieldBoundaryConditions(:periodic, :periodic, :wall)
        pred = (x, y, z) -> y > 8
    elseif comp == 2      # v past the walls of solid columns i = 1, 2 (periodic: ±x)
        g = RegularGrid(; size = (10, 8, 4), extent = (10.0, 8.0, 4.0), halo = (2, 2, 2))
        bcs = FieldBoundaryConditions(:periodic, :periodic, :wall)
        pred = (x, y, z) -> x < 2
    else                  # w past an L of solid (walls in ±x, ±y, corners), periodic
        g = RegularGrid(; size = (8, 8, 6), extent = (8.0, 8.0, 6.0), halo = (2, 2, 2))
        bcs = FieldBoundaryConditions(:periodic)
        pred = (x, y, z) -> x > 6 || y > 6
    end
    sim = BoussinesqSimulation(g; ν = 0.05, C_s = 0.17, bcs = bcs)
    sm = SolidMask(g); mark_solid!(sm, pred)
    f = (sim.u, sim.v, sim.w)[comp]
    IV(f) .= 2.0
    implicit_ibm_correction!(sim, sm)                    # zero every solid face
    iw = immersed_walls(sim, sm, law)
    # every fluid face of this component (not only the walled ones)
    fl = trues(size(f.data))
    Hx, Hy, Hz = g.Hx, g.Hy, g.Hz
    e = ((1, 0, 0), (0, 1, 0), (0, 0, 1))[comp]
    for k in 1:g.Nz, j in 1:g.Ny, i in 1:g.Nx
        I, J, K = Hx + i, Hy + j, Hz + k
        sm.mask[I, J, K] && (fl[I, J, K] = false; fl[I+e[1], J+e[2], K+e[3]] = false)
    end
    inner = falses(size(f.data)); inner[Hx+1:Hx+g.Nx, Hy+1:Hy+g.Ny, Hz+1:Hz+g.Nz] .= true
    fl .&= inner
    q0 = copy(f.data)
    rec = Dict{Symbol,Any}()
    bf! = (s, dt) -> model && (rec[:stencil] = remove_stencil_wall_flux!(s, iw))
    pp! = function (s, dt)
        implicit_ibm_correction!(s, sm; dt = dt)
        rec[:qstar] = copy(f.data)                     # the velocity the drag is evaluated at
        model && apply_immersed_wall_stress!(s, iw, dt)
        rec[:q1] = copy(f.data)
    end
    step!(sim, Δt; body_force! = bf!, post_predictor! = pp!)
    dV = g.dx * g.dy * g.dz
    ΔM = sum((rec[:q1] .- q0)[fl]) * dV
    # Σ τ_w ΔA Δt, independently: the drag of each walled face from the
    # law at the predicted speed (the tangential partner is zero here), at
    # h = Δ/2, times the end-of-step face velocity
    Δ = (g.dx, g.dy, g.dz)
    imp = 0.0
    for wf in iw.faces
        wf.c == comp || continue
        qs = rec[:qstar][wf.I, wf.J, wf.K]
        a = wall_drag(law, abs(qs), Δ[wf.d] / 2)
        imp += a * rec[:q1][wf.I, wf.J, wf.K] * (dV / Δ[wf.d]) * Δt
    end
    return (; ΔM, imp, iw, sim, rec, nwall = count(wf -> wf.c == comp, iw.faces))
end

@testset "No double counting: momentum removed = Σ τ_w ΔA Δt at an IBM wall" begin
    for comp in 1:3
        r = one_wall_step(; comp = comp)
        rel = abs(r.ΔM + r.imp) / abs(r.imp)
        @info @sprintf("  %s: %d wall faces, Δ momentum %.6e, −Σ τ_w ΔA Δt %.6e, rel %.1e; stencil flux removed %.4e",
                       ("u", "v", "w")[comp], r.nwall, r.ΔM, -r.imp, rel, r.rec[:stencil][comp])
        @test r.nwall > 0
        @test rel < 1e-12
        # the stencil's own flux was there and is what got removed: without
        # the model, the step removes exactly that flux times Δt (the stencil
        # is then the only flux through the wall)
        r0 = one_wall_step(; comp = comp, model = false)
        @test r0.ΔM ≈ r.rec[:stencil][comp] * 0.3 rtol = 1e-12
        # here the stencil (ν = 0.05 plus the Smagorinsky ν_t of the wall
        # shear, over Δ) drags 3.8× harder than the log law (z0 = 0.01 m at
        # h = 0.5 m); a first guess of "> 10×" was an unmeasured expectation
        @info @sprintf("     wall model off: Δ momentum %.4e (%.1f× the modelled)", r0.ΔM, r0.ΔM / r.ΔM)
        @test r0.ΔM < r.ΔM < 0
        # the force and the τ_w ΔA diagnostics of the operator agree with it
        F = r.iw.stats[:F][comp]
        @test F * 0.3 ≈ r.ΔM rtol = 1e-12
        @test r.iw.stats[:τA] * 0.3 ≈ r.imp rtol = 1e-12
        @test r.iw.stats[:τ_min] >= 0
    end
    # corners: the w faces at the inner corner of the L have two walls
    r = one_wall_step(; comp = 3)
    seen = Dict{NTuple{3,Int},Int}()
    for wf in r.iw.faces
        k = (Int(wf.I), Int(wf.J), Int(wf.K)); seen[k] = get(seen, k, 0) + 1
    end
    @test any(==(2), values(seen))
end

# -------------------------------------------------------------------------
# 4. Laminar limit
# -------------------------------------------------------------------------

@testset "Laminar limit, IBM: h⁺ < 1 gives the ν-linear stress (1e-3)" begin
    law = SpaldingLaw()
    # IBM: a slow flow past a wall with ν large enough for h⁺ < 1
    ν = 0.5
    g = RegularGrid(; size = (8, 10, 4), extent = (8.0, 10.0, 4.0), halo = (2, 2, 2))
    sim = BoussinesqSimulation(g; bcs = FieldBoundaryConditions(:periodic, :periodic, :wall))
    sm = SolidMask(g); mark_solid!(sm, (x, y, z) -> y > 8)
    IV(sim.u) .= 0.05
    implicit_ibm_correction!(sim, sm)
    iw = immersed_walls(sim, sm, law; ν = ν)
    h = g.dy / 2
    worst, hp = 0.0, 0.0
    for wf in iw.faces
        U = sim.u.data[wf.I, wf.J, wf.K]
        τ = wall_stress(law, U, h; ν = ν)
        hp = max(hp, h * sqrt(τ) / ν)
        worst = max(worst, abs(τ - ν * U / h) / (ν * U / h))
    end
    Δt = 1e-3
    u0 = copy(sim.u.data)
    apply_immersed_wall_stress!(sim, iw, Δt)
    # the operator's update is the implicit ν-linear one
    wf = iw.faces[1]
    lin = u0[wf.I, wf.J, wf.K] / (1 + Δt * (ν / h) / g.dy)
    opr = abs(sim.u.data[wf.I, wf.J, wf.K] - lin) / abs(u0[wf.I, wf.J, wf.K] - lin)
    @info @sprintf("  IBM: h⁺ ≤ %.2f, max |τ_Spalding − νU/h|/(νU/h) = %.1e; operator vs ν-linear update: %.1e of the change",
                   hp, worst, opr)
    @test hp < 1
    @test worst < 1e-3
    @test opr < 1e-3
end


# -------------------------------------------------------------------------
# Cut-cell walls: wall_shear_stress(…, law; ν) at the image-point probe
# -------------------------------------------------------------------------

@testset "Laminar limit, cut cell: h⁺ < 1 gives the ν-linear stress (1e-3)" begin
    law = SpaldingLaw()
    # a linear profile over a flat cut floor, Spalding vs ν-linear
    νc = 1.5e-2
    gc = RegularGrid(; size = (4, 4, 32), extent = (1.0, 1.0, 1.0), halo = (2, 2, 2))
    zw = (32 ÷ 4 + 0.7) * gc.dz
    geom = Aeolus.CutCell.cut_cell_geometry(gc, (x, y, z) -> z - zw)
    st = Aeolus.CutCell.build_ghost_stencils(geom, gc)
    α = 2.5
    u = Field(FaceX(), gc); v = Field(FaceY(), gc); w = Field(FaceZ(), gc)
    for k in axes(u.data, 3), j in axes(u.data, 2), i in axes(u.data, 1)
        u.data[i, j, k] = α * ((k - gc.Hz - 0.5) * gc.dz - zw)
    end
    τl = wall_shear_stress(u, v, w, st, geom, νc)
    τn = wall_shear_stress(u, v, w, st, geom, nothing; ν = νc)
    τs = wall_shear_stress(u, v, w, st, geom, law; ν = νc)
    valid = [s.valid for s in st]
    hpc = maximum(s.probe * sqrt(τs[m]) / νc for (m, s) in enumerate(st) if s.valid)
    errc = maximum(abs(τs[m] - τl[m]) / τl[m] for m in eachindex(st) if valid[m])
    @info @sprintf("  cut cell: h⁺ ≤ %.2f, max |τ_Spalding − τ_ν-linear|/τ_ν-linear = %.1e", hpc, errc)
    @test τn == τl                                       # law = nothing is the ν-linear method
    @test hpc < 1
    @test errc < 1e-3
    # and the slip velocity of the reconstruction vanishes there (no-slip)
    slip = wall_slip_velocity(u, v, w, st, geom, law; ν = νc)
    ut = [τl[m] * s.probe / νc for (m, s) in enumerate(st)]
    @test maximum(abs(slip[m][1]) / ut[m] for m in eachindex(st) if valid[m]) < 1e-3
end

@testset "Cut-cell channel inclined at 30°: ⟨τ_w⟩ = δ dp/dx; ν-linear far below" begin
    # A channel of half-width δ = 1 inclined at 30° in the x-y plane, cut
    # through a 4 × 4 box (60 × 60 cells, Δ = 1/15), carrying the mean
    # profile of Re_τ = 1000 (Spalding's u⁺(y⁺) with u_τ = 1, ν = 1e-3)
    # along the channel. In a fully developed channel τ_w = δ dp/dx = u_τ²
    # = 1. The plane mean is area-weighted (geom.area) over the stencils
    # whose foot point is away from the box edges.
    law = SpaldingLaw()
    ν, δ, uτ, θ = 1e-3, 1.0, 1.0, π / 6
    n = (-sin(θ), cos(θ)); t = (cos(θ), sin(θ))
    N, L = 60, 4.0
    g = RegularGrid(; size = (N, N, 4), extent = (L, L, 4L / N), halo = (2, 2, 2))
    φ(x, y, z) = δ - abs((x - L / 2) * n[1] + (y - L / 2) * n[2])
    geom = Aeolus.CutCell.cut_cell_geometry(g, φ)
    U(x, y, z) = (d = φ(x, y, z); d <= 0 ? 0.0 : uτ * spalding_uplus(law, d * uτ / ν))
    u = Field(FaceX(), g); v = Field(FaceY(), g); w = Field(FaceZ(), g)
    for I in CartesianIndices(u.data)
        i, j, k = Tuple(I)
        zc = (k - g.Hz - 0.5) * g.dz
        u.data[I] = U((i - g.Hx - 1) * g.dx, (j - g.Hy - 0.5) * g.dy, zc) * t[1]
        v.data[I] = U((i - g.Hx - 0.5) * g.dx, (j - g.Hy - 1) * g.dy, zc) * t[2]
    end
    function plane_mean(st, τ)
        sel = [m for (m, s) in enumerate(st) if s.valid && geom.area[s.i, s.j, s.k] > 0 &&
               L / 4 < s.xw[1] < 3L / 4 && L / 4 < s.xw[2] < 3L / 4]
        A = [geom.area[st[m].i, st[m].j, st[m].k] for m in sel]
        return sum(A .* τ[sel]) / sum(A), length(sel), st[sel[1]].probe
    end
    # the image point two cells out (probe_factor = 2): there trilinear
    # interpolation of the logarithmic profile is accurate; at the default
    # one cell out (probe_factor = 1) the concave profile is interpolated
    # 5–7 % low (measured ⟨τ_w⟩ = 0.947 at Δ = 0.1, 0.931 at Δ = 1/15 and 0.05)
    st = Aeolus.CutCell.build_ghost_stencils(geom, g; probe_factor = 2)
    τm, nsel, probe = plane_mean(st, wall_shear_stress(u, v, w, st, geom, law; ν = ν))
    τl, _, _ = plane_mean(st, wall_shear_stress(u, v, w, st, geom, ν))
    st1 = Aeolus.CutCell.build_ghost_stencils(geom, g)
    τm1, _, _ = plane_mean(st1, wall_shear_stress(u, v, w, st1, geom, law; ν = ν))
    @info @sprintf("  30° cut-cell channel, %d stencils, probe⁺ = %.0f: ⟨τ_w⟩ Spalding %.4f, ν-linear %.4f (−%.0f %%); δ dp/dx = 1; probe_factor 1: %.4f",
                   nsel, probe * uτ / ν, τm, τl, 100 * (1 - τl), τm1)
    @test abs(τm - δ * uτ^2) / (δ * uτ^2) < 0.05
    @test τl < 0.5 * δ * uτ^2                       # the ν-linear underpredicts by > 50 %
    # the slip velocity carries the modelled stress through the linear
    # ghost-cell profile: ν_eff (u_t − u_s)/probe = τ_w, to round-off
    ν_eff = 0.05
    τs = wall_shear_stress(u, v, w, st, geom, law; ν = ν)
    slip = wall_slip_velocity(u, v, w, st, geom, law; ν = ν, ν_eff = ν_eff)
    uc, vc, wc = Aeolus.CutCell._center_velocity(u, v, w, geom, g)
    err = 0.0
    for (m, s) in enumerate(st)
        (s.valid && τs[m] > 0) || continue
        ui = Aeolus.CutCell._image_value(uc, s); vi = Aeolus.CutCell._image_value(vc, s)
        un = ui * s.n[1] + vi * s.n[2]
        ut = hypot(ui - un * s.n[1], vi - un * s.n[2])
        us = hypot(slip[m][1], slip[m][2]) * sign((ui - un * s.n[1]) * slip[m][1] + (vi - un * s.n[2]) * slip[m][2])
        err = max(err, abs(ν_eff * (ut - us) / s.probe - τs[m]) / τs[m])
    end
    @test err < 1e-12
    # and slip_wall_value hands one component to reconstruct_wall_velocity!
    wv = slip_wall_value(slip, st, geom, 1)
    s1 = st[findfirst(s -> s.valid, st)]
    @test wv(s1.xw...) == slip[findfirst(s -> s.valid, st)][1]
    su = Aeolus.CutCell.build_ghost_stencils(geom, g, FaceX())
    @test Aeolus.CutCell.reconstruct_wall_velocity!(deepcopy(u), su, geom; wall_value = wv) isa Field
end

end
