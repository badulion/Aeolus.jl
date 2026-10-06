# N-nest-d (#261, epic #246): one-way parent → child boundary data.
#
# The child's nest faces are N-open-b OpenBoundary faces fed by a
# NestBoundaryData: the time-blended (N-nest-c), prolongated (N-nest-a)
# parent field set on the normal-velocity face planes and the tangential
# and scalar halos, mass-corrected by the single correct_boundary_mass_flux!.
#
#   1. mass: divergence-free random parent → net child boundary flux after
#      the correction ≤ 1e-13·U·A, |δ|/U, child max|∇·u| < 1e-10
#   2. advection of a Gaussian θ anomaly through the nest (r = 2, 3): child
#      error ≤ parent error while inside; residual after it left, with and
#      without a NestSponge
#   3. one-way consistency in a convective boundary layer: overlap-mean
#      θ(z), u(z) of child and parent agree within the run-to-run spread
#   4. top boundary: the child top halo follows a parent θ trend, and the
#      child column responds only through the boundary flux
#   5. faces inside solids (a building cut by the east face): u·n = 0,
#      excluded from δ
#   6. perfect parent: refinement 1, the child driven by the truth
#
# References: Hellsten et al. (2021), GMD 14, 3185 (PALM nesting: Dirichlet
# child boundary from the interpolated parent, §3.2; mass-flux correction,
# §3.3, Eq. 2); Kadasch et al. (2021), GMD 14, 5435, §2.2–2.3 (no sponge at
# the child boundary); Deardorff (1970), JAS 27, 1211 (convective velocity
# scale w* = (g/θ₀ H zᵢ)^(1/3)).

using Test
using Printf
using Random
using Aeolus
using Aeolus.Nesting
using Aeolus.Grids: interior
using Aeolus.Fields: interior_view
using Aeolus.BoundaryConditions: FieldBoundaryConditions, OpenBoundary, open_boundary_states
using Aeolus.Operators: divergence
using Aeolus.AdvectionSchemes: advect_weno_z!
using Aeolus.IBM: SolidMask, mark_solid!, apply_solid_mask!

_nb_xc(m, i) = m.x0 + (i - m.Hx - 0.5) * m.dx
_nb_zc(m, k) = m.z0 + (k - m.Hz - 0.5) * m.dz

"Child initialised from the parent prolongated over the whole child."
function _nb_init_child!(csim, psim, nest)
    st = nest_interp_state(nest)
    interpolate_parent_state!(st, psim, nest; order = 3, region = :all)
    for (cf, sf) in ((csim.u, st.u), (csim.v, st.v), (csim.w, st.w), (csim.θ, st.θ))
        cf.data .= sf.data
    end
    return csim
end

"Total area of the open faces of the child's (first) open boundary set."
function _nb_open_area(csim)
    m = csim.mesh; o = nest_open_sides(csim)
    return (o[1] + o[2]) * m.Ly * m.Lz + (o[3] + o[4]) * m.Lx * m.Lz +
           (o[5] + o[6]) * m.Lx * m.Ly
end

@testset "Era II — one-way nest boundary data (N-nest-d, #261)" begin

# -------------------------------------------------------------------------
@testset "mass: divergence-free parent, net boundary flux and δ" begin
    Random.seed!(261)
    cases = (
        # a child inside a triply periodic parent: all six faces open
        (name = "inside, r = 2", r = 2, pbcs = FieldBoundaryConditions(:periodic),
         ext = ((0.25, 0.75), (0.25, 0.75), (0.25, 0.75)), bottom = :nest),
        # the Berlin shape: ground wall, open sides and top, r = (2, 2, 4)
        (name = "ground, r = (2,2,4)", r = (2, 2, 4),
         pbcs = FieldBoundaryConditions(:periodic, :periodic, :wall),
         ext = ((0.25, 0.625), (0.25, 0.625), (0.0, 0.5)), bottom = :wall),
        (name = "ground, r = 3", r = 3,
         pbcs = FieldBoundaryConditions(:periodic, :periodic, :wall),
         ext = ((0.25, 0.625), (0.25, 0.625), (0.0, 0.5)), bottom = :wall))
    for c in cases
        pm = RegularGrid(Float64; size = (16, 16, 16), extent = (1.0, 1.0, 1.0), halo = (3, 3, 3))
        nest = nested_grid(pm, c.ext, c.r; halo = (3, 3, 3))
        cm = nest_child_mesh(nest)
        psim = BoussinesqSimulation(pm; bcs = c.pbcs)
        for f in (psim.u, psim.v, psim.w)
            interior_view(f) .= randn(size(interior_view(f)))
        end
        step!(psim, 1e-10); psim.t = 0.0             # project: ∇·u = 0 to round-off
        csim = BoussinesqSimulation(cm; bcs = nest_bcs(cm, :nest, :nest;
                                                       bottom = c.bottom, top = :nest))
        _nb_init_child!(csim, psim, nest)
        run = NestedRun(psim, csim, nest; advance_parent = domain_advance(step_rk3!),
                        advance_child = domain_advance(step_rk3!),
                        cfl_parent = CFLConstraints(C_advective = 0.4))
        # project the prolongated child once (with its open faces): SSP-RK3
        # mixes the step-start state into its result, so a non-solenoidal
        # start would survive (decaying ×1/9 per step) in any domain
        step!(csim, 1e-12); csim.t = psim.t
        U = maximum(abs, interior_view(psim.u))
        A = _nb_open_area(csim)
        worst_Q = 0.0; worst_res = 0.0; worst_δ = 0.0; worst_div = 0.0
        st = only(open_boundary_states(csim))
        st.record = true
        for _ in 1:5
            step_nested!(run)
            for (_, Q, res) in st.log
                worst_Q = max(worst_Q, abs(Q)); worst_res = max(worst_res, abs(res))
            end
            empty!(st.log)
            worst_δ = max(worst_δ, abs(open_boundary_ledger(csim).δ))
            divergence(csim.div_buf, csim.u, csim.v, csim.w)
            worst_div = max(worst_div, maximum(abs, interior_view(csim.div_buf)))
        end
        @printf("  %-20s U = %.2f: |Q|/(U A) before %.2e, after %.2e; |δ|/U = %.2e; child max|∇·u| = %.2e s⁻¹\n",
                c.name, U, worst_Q / (U * A), worst_res / (U * A), worst_δ / U, worst_div)
        @test worst_res <= 1e-13 * U * A
        @test worst_div < 1e-10
        # the prolongation preserves face fluxes and the blend is linear in
        # time, so a divergence-free parent already gives a compatible
        # boundary: δ is round-off (Hellsten et al. 2021 report |δ| three to
        # four orders below U for their non-conservative interpolation)
        @test worst_δ < 1e-13 * U
    end
end

# -------------------------------------------------------------------------
# A Gaussian θ anomaly (amplitude 1 K, σ = 6 parent cells) carried by a
# uniform U = 5 m/s through a periodic parent (2400 m, Δx = 25 m) and a
# child over 800–1600 m at r = 2 and 3 (quasi-1-D: the child is periodic
# in y and z). Both domains SSP-RK3, Δt_p = 1 s, r child substeps. The
# fields are finite-volume cell averages, so they are initialised with, and
# compared against, the exact cell averages of the translated Gaussian
# (point values would put an O(Δx² θ''/24) inconsistency between the two
# grids that the conservative prolongation exposes).
function _nb_advect(r; sponge = nothing, weno = false, order = 3, σ = 150.0, U = 5.0,
                    L = 2400.0, x0 = 500.0, dt = 1.0, t_inside = 140.0, t_end = 350.0)
    pm = RegularGrid(Float64; size = (96, 4, 4), extent = (L, 100.0, 100.0), halo = (3, 3, 3))
    nest = nested_grid(pm, ((800.0, 1600.0), (0.0, 100.0), (0.0, 100.0)), (r, 1, 1))
    cm = nest_child_mesh(nest)
    erf = Aeolus.GasOptics.erf_series
    s2 = σ * sqrt(2)
    # exact average over storage cell i of mesh m at time t (periodic images)
    exact(m, i, t) = begin
        a = m.x0 + (i - m.Hx - 1) * m.dx; b = a + m.dx
        sum(σ * sqrt(π / 2) / m.dx * (erf((b - x0 - U * t + n * L) / s2) -
                                      erf((a - x0 - U * t + n * L) / s2)) for n in -1:1)
    end
    psim = BoussinesqSimulation(pm)
    csim = BoussinesqSimulation(cm; bcs = nest_bcs(cm, :nest, :periodic, :periodic))
    for s in (psim, csim)
        fill!(s.u.data, U)
        for i in axes(s.θ.data, 1)
            s.θ.data[i, :, :] .= exact(s.mesh, i, 0.0)
        end
    end
    # θ transport: the core's centred scheme, or WENO-Z5 (the Berlin
    # tracers' scheme) swapped in as the θ tendency of both domains
    bf = weno ? (s, Δt) -> (fill!(s.tθ.data, 0); advect_weno_z!(s.tθ, s.u, s.v, s.w, s.θ); s) :
                nothing
    adv = domain_advance(step_rk3!; body_force! = bf)
    run = NestedRun(psim, csim, nest; advance_parent = adv, advance_child = adv,
                    cfl_parent = nothing, cfl_child = nothing, substeps = r, sponge, order)
    cx, _, _ = nest_covered_range(nest)
    l2(s, idx, t) = sqrt(sum(abs2, interior_view(s.θ)[idx, 1, 1] .-
                                   [exact(s.mesh, i + s.mesh.Hx, t) for i in idx]) / length(idx))
    ratio = NaN
    while run.t < t_end - 1e-9
        step_nested!(run; dt)
        if abs(run.t - t_inside) < 1e-9            # centre at 1200 m, mid-child
            ratio = l2(csim, 1:cm.Nx, run.t) / l2(psim, cx, run.t)
        end
    end
    resid = maximum(abs, interior_view(csim.θ))  # centre at 2250 m: 4.3σ past the child
    return (; ratio, resid)
end

@testset "advection of a θ anomaly through the nest (r = 2, 3)" begin
    sp = NestSponge(width = 6, α_max = 0.2)
    for r in (2, 3)
        cen = _nb_advect(r)
        cen_sp = _nb_advect(r; sponge = sp)
        # WENO-Z5 needs a 5th-order prolongation (`order = 5`): with the
        # default order 3 the interface error O(Δx_p³ ∂³θ/∂x³) exceeds the
        # parent's own 5th-order error (child/parent ratio 3.5, see below)
        wen = _nb_advect(r; weno = true, order = 5)
        wen_sp = _nb_advect(r; weno = true, order = 5, sponge = sp)
        wen3 = _nb_advect(r; weno = true, order = 3)
        @printf("  r = %d, centred θ: child/parent L2 error inside %.3f; residual after it left %.2e (sponge: %.2e, ratio %.3f)\n",
                r, cen.ratio, cen.resid, cen_sp.resid, cen_sp.ratio)
        @printf("  r = %d, WENO-Z5 θ, order-5 prolongation: child/parent L2 error inside %.3f; residual after it left %.2e (sponge: %.2e, ratio %.3f); order-3 prolongation: ratio %.3f\n",
                r, wen.ratio, wen.resid, wen_sp.resid, wen_sp.ratio, wen3.ratio)
        @test wen3.ratio > 1.0      # documents why the order must match the scheme
        # the refined child is more accurate than the parent over the overlap
        for c in (cen, cen_sp, wen, wen_sp)
            @test c.ratio <= 1.0
        end
        # with an upwind-biased scalar scheme the anomaly leaves cleanly,
        # with and without the sponge
        @test wen.resid < 0.01
        @test wen_sp.resid < 0.01
        @test cen_sp.resid < 0.01
        # the core's 2nd-order centred θ advection has an undamped 2Δx mode
        # (group velocity −U) that the Dirichlet outflow face excites and the
        # Dirichlet inflow face reflects: without a sponge it stays trapped at
        # a few % (3.1e-2 at r = 2, 4.1e-2 at r = 3). A property of the
        # scheme rather than of the boundary data: the same nest with WENO-Z5
        # θ leaves 7e-5, and the sponge removes it (8e-4, 2e-3). The upwind
        # scalar outflow halo is N-open-c (#260).
        @test cen.resid < 0.05
        @test_broken cen.resid < 0.01
    end
end

# -------------------------------------------------------------------------
# A convective boundary layer (surface heat flux 0.1 K m/s, a 2 m/s mean
# wind, Smagorinsky) in a 1200² × 600 m parent with a 600² × 400 m child at
# r = 2 on the ground (open sides and top). After 1200 s of spin-up
# (~3 t* = zᵢ/w*, w* ≈ 1.2 m/s) the overlap-mean θ(z) and u(z), averaged
# over 1200–2400 s, of child and parent must agree within the run-to-run
# spread of the parent itself (4 seeds).
function _nb_cbl_parent(pm, seed)
    p = BoussinesqSimulation(pm; θ₀ = 300.0, C_s = 0.17, θ_flux_bottom = 0.1,
                             bcs = FieldBoundaryConditions(:periodic, :periodic, :wall))
    Random.seed!(seed)
    for k in axes(p.θ.data, 3)
        z = _nb_zc(pm, k)
        p.θ.data[:, :, k] .= 300.0 + (z > 300 ? 0.005 * (z - 300) : 0.0)
    end
    θi = interior_view(p.θ)
    θi[:, :, 1:3] .+= 0.2 .* (rand(size(θi, 1), size(θi, 2), 3) .- 0.5)
    fill!(p.u.data, 2.0)
    step!(p, 1e-3); p.t = 0.0
    return p
end

_nb_hmean(f, ci, cj, kk) = [sum(interior_view(f)[ci, cj, k]) / (length(ci) * length(cj)) for k in kk]

function _nb_cbl(seed; nested, T1 = 1200.0, T2 = 2400.0)
    pm = RegularGrid(Float64; size = (24, 24, 12), extent = (1200.0, 1200.0, 600.0), halo = (3, 3, 3))
    p = _nb_cbl_parent(pm, seed)
    nest = nested_grid(pm, ((300.0, 900.0), (300.0, 900.0), (0.0, 400.0)), 2)
    cm = nest_child_mesh(nest); cx, cy, cz = nest_covered_range(nest)
    cfl = CFLConstraints(C_advective = 0.5)
    c = nothing; run = nothing
    if nested
        c = BoussinesqSimulation(cm; θ₀ = 300.0, C_s = 0.17, θ_flux_bottom = 0.1,
                                 bcs = nest_bcs(cm, :nest, :nest; bottom = :wall, top = :nest))
        _nb_init_child!(c, p, nest)
        run = NestedRun(p, c, nest; cfl_parent = cfl)
    end
    ap = zeros(length(cz), 2); ac = zeros(cm.Nz, 2); n = 0; dtp = nothing
    while p.t < T2
        if nested
            step_nested!(run)
        else
            dt = adaptive_dt(p, cfl; dt_prev = dtp); step!(p, dt); dtp = dt
        end
        p.t >= T1 || continue
        ap[:, 1] .+= _nb_hmean(p.θ, cx, cy, cz); ap[:, 2] .+= _nb_hmean(p.u, cx, cy, cz)
        if nested
            ac[:, 1] .+= _nb_hmean(c.θ, 1:cm.Nx, 1:cm.Ny, 1:cm.Nz)
            ac[:, 2] .+= _nb_hmean(c.u, 1:cm.Nx, 1:cm.Ny, 1:cm.Nz)
        end
        n += 1
    end
    ap ./= n
    # child levels averaged onto the parent levels (r_z = 2)
    pc = nested ? [(ac[2k-1, j] + ac[2k, j]) / (2n) for k in 1:length(cz), j in 1:2] : nothing
    return ap, pc, run
end

@testset "one-way consistency: convective boundary layer, overlap means" begin
    pp, pc, run = _nb_cbl(1; nested = true)
    runs = [pp, (first(_nb_cbl(s; nested = false)) for s in 2:4)...]
    spread = [maximum(a[k, j] for a in runs) - minimum(a[k, j] for a in runs)
              for k in axes(pp, 1), j in 1:2]
    d = abs.(pc .- pp)
    st = run.stats
    @printf("  CBL: %d parent steps, %d child substeps (n %d..%d), retries %d/%d\n",
            st.parent_steps, st.child_substeps, st.n_min, st.n_max, st.parent_retries,
            st.child_retries)
    for k in axes(pp, 1)
        @printf("    z = %3.0f m: θ parent %.4f child %.4f |Δ| %.4f spread %.4f K | u parent %.3f child %.3f |Δ| %.3f spread %.3f m/s\n",
                25.0 + 50.0 * (k - 1), pp[k, 1], pc[k, 1], d[k, 1], spread[k, 1],
                pp[k, 2], pc[k, 2], d[k, 2], spread[k, 2])
    end
    @printf("  max |Δθ| / spread = %.2f, max |Δu| / spread = %.2f\n",
            maximum(d[:, 1] ./ spread[:, 1]), maximum(d[:, 2] ./ spread[:, 2]))
    @test all(d[:, 1] .<= spread[:, 1])
    @test all(d[:, 2] .<= spread[:, 2])
    @test st.child_retries == 0 && st.parent_retries == 0
end

# -------------------------------------------------------------------------
# A parent warming in time, θ = 300 + a t everywhere (set analytically, at
# rest). The child spans the parent horizontally (periodic) on the ground
# and is nested only through its top. Its top halo must follow the trend,
# and its column may change only through the diffusive flux through the
# top face: Δ∫θ = Δt κ Σ (θ_halo − θ_top)/Δz · A, and nothing at all
# with κ = 0.
@testset "top boundary: a parent θ trend reaches the child only through fluxes" begin
    a = 1e-3                                              # K/s
    for κ in (0.0, 5.0)
        pm = RegularGrid(Float64; size = (8, 8, 16), extent = (400.0, 400.0, 800.0), halo = (3, 3, 3))
        nest = nested_grid(pm, ((0.0, 400.0), (0.0, 400.0), (0.0, 400.0)), (2, 2, 4))
        cm = nest_child_mesh(nest)
        psim = BoussinesqSimulation(pm; bcs = FieldBoundaryConditions(:periodic, :periodic, :wall))
        fill!(psim.θ.data, 300.0)
        csim = BoussinesqSimulation(cm; ν = κ,
                                    bcs = nest_bcs(cm, :periodic, :periodic; bottom = :wall,
                                                   top = :nest))
        fill!(csim.θ.data, 300.0)
        core = (s, dt; kw...) -> (s.t += dt; fill!(s.θ.data, 300.0 + a * s.t); s)
        halo_err = Ref(0.0); flux = Ref(0.0); ctr = Ref(0)
        kt = cm.Hz + cm.Nz
        # body_force! runs after the tendencies, with the halos the core read
        probe! = (s, Δt) -> begin
            ht = view(s.θ.data, cm.Hx+1:cm.Hx+cm.Nx, cm.Hy+1:cm.Hy+cm.Ny, kt + 1)
            halo_err[] = max(halo_err[], maximum(abs, ht .- (300.0 + a * s.t)))
            inn = view(s.θ.data, cm.Hx+1:cm.Hx+cm.Nx, cm.Hy+1:cm.Hy+cm.Ny, kt)
            flux[] += Δt * κ * sum(ht .- inn) / cm.dz * cm.dx * cm.dy
            ctr[] += 1
            s
        end
        run = NestedRun(psim, csim, nest; advance_parent = domain_advance(core),
                        advance_child = domain_advance(; body_force! = probe!),
                        cfl_parent = nothing, cfl_child = nothing, substeps = 3)
        V = cm.dx * cm.dy * cm.dz
        # ∫(θ − 300): the anomaly integral, free of the 300 K cancellation
        I0 = sum(interior_view(csim.θ) .- 300.0) * V
        θ0 = copy(interior_view(csim.θ))
        for _ in 1:40
            step_nested!(run; dt = 5.0)
        end
        I1 = sum(interior_view(csim.θ) .- 300.0) * V
        budget = abs((I1 - I0) - flux[]) / max(abs(I1 - I0), 1e-300)
        top_inc = sum(interior_view(csim.θ)[:, :, end] .- 300.0) / (cm.Nx * cm.Ny)
        bot_inc = sum(interior_view(csim.θ)[:, :, 1] .- 300.0) / (cm.Nx * cm.Ny)
        @printf("  κ = %.0f m²/s, %d child steps to t = %.0f s (parent +%.3f K): top-halo error %.1e K; Δ∫θ = %.6e, boundary flux %.6e (rel. diff %.1e); top cell +%.4f K, bottom cell +%.2e K\n",
                κ, ctr[], run.t, a * run.t, halo_err[], I1 - I0, flux[], budget, top_inc, bot_inc)
        @test halo_err[] < 1e-12
        if κ == 0
            @test interior_view(csim.θ) == θ0                  # no flux, no change
        else
            @test budget < 1e-10
            @test top_inc > 0 && top_inc > 100 * abs(bot_inc)   # warming from the top down
        end
    end
end

# -------------------------------------------------------------------------
# A building cut by the child's east face. The open faces next to its solid
# cells carry u·n = 0 and are left out of the mass-flux correction, exactly.
@testset "faces inside solids: u·n = 0, excluded from δ" begin
    pm = RegularGrid(Float64; size = (16, 16, 8), extent = (800.0, 800.0, 400.0), halo = (3, 3, 3))
    nest = nested_grid(pm, ((200.0, 600.0), (200.0, 600.0), (0.0, 200.0)), 2)
    cm = nest_child_mesh(nest)
    psim = BoussinesqSimulation(pm; bcs = FieldBoundaryConditions(:periodic, :periodic, :wall))
    fill!(psim.u.data, 3.0)                               # an eastward wind: outflow at the east face
    Random.seed!(5)
    interior_view(psim.v) .= 0.2 .* randn(size(interior_view(psim.v)))
    step!(psim, 1e-6); psim.t = 0.0
    # the building: x ∈ [550, 600] m (cut by the east face at 600 m),
    # y ∈ [350, 450] m, height 100 m
    sm = SolidMask(cm)
    mark_solid!(sm, (x, y, z) -> x > 550 && 350 < y < 450 && z < 100)
    csim = BoussinesqSimulation(cm; bcs = nest_bcs(cm, :nest, :nest; bottom = :wall, top = :nest))
    _nb_init_child!(csim, psim, nest)
    apply_solid_mask!(csim, sm)
    run = NestedRun(psim, csim, nest; solid = sm,
                    advance_child = domain_advance(; post_predictor! = (s, h) -> apply_solid_mask!(s, sm)),
                    cfl_parent = CFLConstraints(C_advective = 0.4))
    st = only(open_boundary_states(csim))
    east = csim.bcs.east.planes[:u]
    nsolid = count(!, east.fluid)
    for _ in 1:5
        step_nested!(run)
    end
    # the east boundary face (storage index Hx+Nx+1) on the solid rows
    ie = cm.Hx + cm.Nx + 1
    solid_rows = [(cm.Hy + a, cm.Hz + b) for a in 1:cm.Ny, b in 1:cm.Nz if !east.fluid[a, b]]
    un_solid = maximum(abs(csim.u.data[ie, j, k]) for (j, k) in solid_rows)
    buf_solid = maximum(abs(east.data[1, j, k]) for (j, k) in solid_rows)
    # δ is Q over the fluid faces only (outward-normal area, solid faces excluded)
    led = open_boundary_ledger(csim)
    A_fluid = sum(count(p.fluid) * (p.axis == 1 ? cm.dy * cm.dz : p.axis == 2 ? cm.dx * cm.dz :
                                    cm.dx * cm.dy) for p in st.planes if p.normal)
    A_all = _nb_open_area(csim)
    U = 3.0
    @printf("  building at the east face: %d solid face(s) of %d; u·n there %.1e (buffer %.1e); Q = %.4e m³/s, δ = %.4e m/s = Q/A_fluid (%.4e), residual %.1e\n",
            nsolid, length(east.fluid), un_solid, buf_solid, led.Q, led.δ, led.Q / A_fluid,
            led.residual)
    @test nsolid == 4 * 4                                  # 4 cells in y (100 m) × 4 in z (100 m)
    @test un_solid == 0.0 && buf_solid == 0.0
    @test led.δ == led.Q / A_fluid                         # excluded from δ, exactly
    @test A_fluid < A_all
    @test abs(led.residual) <= 1e-13 * U * A_all
    @test abs(led.Q) > 1e-6 * U * A_all                    # the solid faces do make an imbalance
    divergence(csim.div_buf, csim.u, csim.v, csim.w)
    @test maximum(abs, interior_view(csim.div_buf)) < 1e-10
end

# -------------------------------------------------------------------------
# Refinement 1: the child is the parent's own grid over a sub-box, so the
# parent is the exact solution of the child's problem. Driven by it, the
# child must stay on it: with the projection step the predictor sees the
# same halos and the Neumann solve reproduces the parent's interior
# pressure gradient, so the match is round-off; with SSP-RK3 the
# intermediate stages see the time-blended parent (B(t+Δt), B(t+Δt/2))
# instead of the parent's own stage values, an O(Δt²) per-step difference.
@testset "perfect parent: refinement 1, the child stays on the truth" begin
    for (name, adv) in (("step!", domain_advance()), ("step_rk3!", domain_advance(step_rk3!)))
        L = 1.0
        pm = RegularGrid(Float64; size = (24, 24, 8), extent = (L, L, L / 3), halo = (3, 3, 3))
        nest = nested_grid(pm, ((0.25, 0.75), (0.25, 0.75), (0.0, L / 3)), (1, 1, 1))
        cm = nest_child_mesh(nest)
        psim = BoussinesqSimulation(pm; ν = 2e-3, θ₀ = 300.0, g_accel = 0.1)
        Random.seed!(11)
        for k in axes(psim.u.data, 3), j in axes(psim.u.data, 2), i in axes(psim.u.data, 1)
            x = pm.x0 + (i - pm.Hx - 1) * pm.dx; xc = x + 0.5pm.dx
            y = pm.y0 + (j - pm.Hy - 0.5) * pm.dy; yf = y - 0.5pm.dy
            psim.u.data[i, j, k] = 0.4 + sin(2π * x) * cos(2π * y)
            psim.v.data[i, j, k] = -cos(2π * xc) * sin(2π * yf)
            psim.θ.data[i, j, k] = 300.0 + exp(-((xc - 0.4)^2 + (y - 0.5)^2) / 0.02)
        end
        interior_view(psim.w) .+= 0.05 .* randn(size(interior_view(psim.w)))
        step!(psim, 1e-8); psim.t = 0.0
        csim = BoussinesqSimulation(cm; ν = 2e-3, θ₀ = 300.0, g_accel = 0.1,
                                    bcs = nest_bcs(cm, :nest, :nest, :periodic))
        _nb_init_child!(csim, psim, nest)
        run = NestedRun(psim, csim, nest; advance_parent = adv, advance_child = adv,
                        cfl_parent = nothing, cfl_child = nothing, substeps = 1)
        cx, cy, cz = nest_covered_range(nest)
        rel(fc, fp; base = 0.0) =
            sqrt(sum(abs2, interior_view(fc) .- interior_view(fp)[cx, cy, cz]) /
                 sum(abs2, interior_view(fp)[cx, cy, cz] .- base))
        worst = 0.0
        for _ in 1:100
            step_nested!(run; dt = 4e-3)
            # u: the child's interior faces are parent faces cx (the east
            # boundary face is the child's halo); θ, v: cells
            worst = max(worst, rel(csim.u, psim.u), rel(csim.v, psim.v),
                        rel(csim.θ, psim.θ; base = 300.0))
        end
        @printf("  r = 1, %-10s 100 steps: max relative L2 child − parent (u, v, θ) = %.2e\n",
                name, worst)
        @test worst < 1e-3
        name == "step!" && @test worst < 1e-10
    end
end

end # testset
