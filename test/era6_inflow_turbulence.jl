# N-inflow-c (#264, epic #246): synthetic turbulence applied on the inflow
# faces of an open-boundary child, with zero added mass flux.
#
#   1. no mass added: U = (3, 1, 0), 500 steps — the added flux of every
#      face is round-off relative to its mean-flow flux, every step
#   2. plane stresses: R_ij of the applied perturbations within 5 % (diag.)
#      and 7 % (u'w') of the target after the mean removal, whose variance
#      reduction is the exact discrete variance of the face mean,
#      f = n⁻² Σ_ij R(|i-j|Δ) ≤ 2 L_s/L_face (asymptotically equal)
#   3. face selection: the wind rotated through 360° in 8 steps — variance
#      only on the faces with U_n < 0, exactly zero elsewhere; the ramp
#      scales the perturbation linearly; corners owned by one face
#   4. divergence: max|∇·u| < 1e-10 after the projection on a refine = 3
#      nest with ParentResolvedStatistics (one-way; the wind veers)
#   5. restart: generator (and statistics) state saved and restored
#      continues bit for bit
#
# References: Lund, Wu & Squires (1998), JCP 140, 233; Xie & Castro (2008),
# FTaC 81, 449; Kadasch et al. (2021), GMD 14, 5435 (PALM's STG: mean
# removal, lateral faces only); Panofsky & Dutton (1984), §7.4.

using Test
using Printf
using Random
using Aeolus
using Aeolus.Nesting
using Aeolus.Fields: interior_view
using Aeolus.Simulations: BoussinesqSimulation, step_rk3!, max_divergence
using Aeolus.TimeIntegration: state_snapshot, save_state!, restore_state!
using Aeolus.BoundaryConditions: open_boundary_states, open_boundary_snapshot,
                                 restore_open_boundaries!, FieldBoundaryConditions
import Aeolus.Transition: _df_autocorrelation

# A statistics source with a uniform tensor and fixed lengths (the test's
# own target; the shipped sources are N-inflow-b's).
struct _ITConst <: InflowStatistics
    R::NTuple{6,Float64}
    L_s::Float64
    L_z::Float64
    L_n::Float64
end
Aeolus.Transition.inflow_stresses(s::_ITConst, z::AbstractVector, t::Real) = [s.R for _ in z]
Aeolus.Transition.inflow_lengths(s::_ITConst, z::AbstractVector, t::Real) =
    (L_s = fill(s.L_s, 3, length(z)), L_z = fill(s.L_z, 3, length(z)),
     T = fill(s.L_n, 3, length(z)))
Aeolus.Transition.inflow_normal_length(s::_ITConst, z::AbstractVector, t::Real) =
    fill(s.L_n, 3, length(z))

# Panofsky & Dutton (1984) neutral surface layer, u* = 0.3 m/s, frame (n, s, z)
const _IT_US = 0.3
const _IT_R = ((2.39_IT_US)^2, (1.92_IT_US)^2, (1.25_IT_US)^2, 0.0, -_IT_US^2, 0.0)

function _it_sim(m, U, V; C_s = 0.18)
    dir = Ref((Float64(U), Float64(V)))
    data = FunctionBoundaryData(u = (x, y, z, t) -> dir[][1], v = (x, y, z, t) -> dir[][2],
                                w = 0.0, θ = 300.0)
    sim = BoussinesqSimulation(m; θ₀ = 300.0, C_s = C_s, bcs = open_boundary_conditions(m, data))
    sim.u.data .= U; sim.v.data .= V; sim.θ.data .= 300.0
    return sim, dir
end

@testset "N-inflow-c: synthetic turbulence on the inflow faces (#264)" begin

# ---------------------------------------------------------------------------
@testset "no mass added: U = (3, 1, 0), 500 steps" begin
    m = RegularGrid(size = (24, 16, 12), extent = (240.0, 160.0, 120.0), halo = (3, 3, 3))
    sim, _ = _it_sim(m, 3.0, 1.0)
    it = InflowTurbulence(sim; statistics = _ITConst(_IT_R, 20.0, 20.0, 20.0))
    worst = Dict(s => 0.0 for s in (:west, :east, :south, :north))
    worst_level = 0.0; maxdiv = 0.0; δmax = 0.0; ke = 0.0
    for n in 1:500
        step_rk3!(sim, 0.5)
        for (s, a) in pairs(added_mass_flux(it))
            worst[s] = max(worst[s], a.relative)
        end
        a = added_mass_flux(it)
        worst_level = max(worst_level, a.west.Q_level / (3.0 * m.Ly * m.dz),
                          a.south.Q_level / (1.0 * m.Lx * m.dz))
        maxdiv = max(maxdiv, max_divergence(sim))
        # the data balance (Q_data = 0 for uniform U), untouched by the perturbations
        δmax = max(δmax, abs(open_boundary_ledger(sim).δ))
        ke += injected_ke_flux(it).total / 500
    end
    @printf("  added flux / mean-flow flux, worst over 500 steps: W %.2e E %.2e S %.2e N %.2e; per level %.2e\n",
            worst[:west], worst[:east], worst[:south], worst[:north], worst_level)
    @printf("  max|∇·u| = %.2e s⁻¹, max|δ| of the N-open-b correction = %.2e m/s, KE flux %.3f W/m²\n",
            maxdiv, δmax, ke)
    @test all(<(1e-12), values(worst))
    @test worst_level < 1e-12
    @test worst[:east] == 0 && worst[:north] == 0           # outflow faces: nothing added
    @test δmax < 1e-13                                       # the correction sees only the data
    @test maxdiv < 1e-10
    @test all(isfinite, interior_view(sim.u)) && maximum(abs, interior_view(sim.u)) < 10
    @test ke > 0
end

# ---------------------------------------------------------------------------
@testset "plane stresses after the mean removal; variance of the face mean" begin
    Ny, Nz = 128, 16
    m = RegularGrid(size = (4, Ny, Nz), extent = (40.0, 10.0Ny, 10.0Nz), halo = (3, 3, 3))
    sim, _ = _it_sim(m, 3.0, 0.0)
    Ls = 20.0
    it = InflowTurbulence(sim; statistics = _ITConst(_IT_R, Ls, 20.0, 15.0), faces = (:west,))
    T = 15.0 / 3.0
    S = zeros(6); n = 0; worstQ = 0.0
    sts = open_boundary_states(sim)
    for step in 1:1500
        for st in sts
            Aeolus.BoundaryConditions.boundary_values!(st.planes, st.data, 2T * step)
        end
        u, v, w = inflow_perturbation(it, :west)
        for k in 1:Nz, j in 1:Ny
            S .+= (u[j, k]^2, v[j, k]^2, w[j, k]^2, u[j, k] * v[j, k], u[j, k] * w[j, k],
                   v[j, k] * w[j, k])
        end
        n += Ny * Nz
        worstQ = max(worstQ, abs(sum(u) * m.dy * m.dz) / (3.0 * m.Ly * m.Lz))
    end
    R = S ./ n
    # exact variance of the per-level face mean: f = n⁻² Σ_ij R_|i-j|
    b, _, _ = digital_filter_kernel(Ls, m.dy)
    ρ = _df_autocorrelation(b)
    ρf(r) = r + 1 <= length(ρ) ? ρ[r+1] : 0.0
    f = sum(ρf(abs(i - j)) for i in 1:Ny, j in 1:Ny) / Ny^2
    bound = 2Ls / m.Ly
    @printf("  f = %.4f, 2L_s/L_face = %.4f, issue's L_s/L_face = %.4f\n", f, bound, Ls / m.Ly)
    @printf("  R11/(R11(1-f)) %.4f  R22/R22 %.4f  R33/R33 %.4f  u'w'/(R13(1-f)) %.4f  R12 %.1e  R23 %.1e\n",
            R[1] / (_IT_R[1] * (1 - f)), R[2] / _IT_R[2], R[3] / _IT_R[3],
            R[5] / (_IT_R[5] * (1 - f)), R[4], R[6])
    @test worstQ < 1e-12
    @test f <= bound                                   # exact upper bound (R ≥ 0)
    @test f >= 0.9bound                                # and its asymptote
    @test abs(R[2] / _IT_R[2] - 1) < 0.05
    @test abs(R[3] / _IT_R[3] - 1) < 0.05
    @test abs(R[1] / (_IT_R[1] * (1 - f)) - 1) < 0.05
    @test abs(R[5] / (_IT_R[5] * (1 - f)) - 1) < 0.07
    # the reduction is resolved: R11 is closer to R11(1-f) than to R11
    @test abs(R[1] / (_IT_R[1] * (1 - f)) - 1) < 0.015
    @test abs(R[4]) < 0.03 * _IT_R[1] && abs(R[6]) < 0.03 * _IT_R[1]
end

# ---------------------------------------------------------------------------
@testset "face selection: the wind through 360° in 8 steps; ramp; corners" begin
    m = RegularGrid(size = (16, 16, 8), extent = (160.0, 160.0, 80.0), halo = (3, 3, 3))
    sim, dir = _it_sim(m, 3.0, 0.0)
    u_min = 0.5
    it = InflowTurbulence(sim; statistics = _ITConst(_IT_R, 20.0, 20.0, 20.0), u_min = u_min)
    outward = (west = (-1.0, 0.0), east = (1.0, 0.0), south = (0.0, -1.0), north = (0.0, 1.0))
    var(P) = sum(abs2, P) / length(P)
    for q in 0:7
        θ = q * π / 4
        dir[] = (3round(cos(θ); digits = 14), 3round(sin(θ); digits = 14))   # exact 0 at 90°
        update_open_boundaries!(sim, sim.t + 1.0 * (q + 1), 1.0)
        for s in (:west, :east, :south, :north)
            Un = dir[][1] * outward[s][1] + dir[][2] * outward[s][2]
            P = inflow_perturbation(it, s)
            if Un < -1e-12
                @test var(P[1]) > 0.1 * _IT_R[1] && var(P[3]) > 0.1 * _IT_R[3]
            else
                @test all(iszero, P[1]) && all(iszero, P[2]) && all(iszero, P[3])
            end
        end
        if q == 1                                     # from the south-west: corners
            Pw = inflow_perturbation(it, :west); Ps = inflow_perturbation(it, :south)
            @test all(iszero, Ps[1][1, :]) && all(iszero, Ps[3][1, :])   # west owns (1, 1)
            @test any(!iszero, Pw[1][1, :])
            @test any(!iszero, Ps[1][end, :])         # east is outflow: south owns its corner
            @test abs(sum(Ps[1])) < 1e-12 * length(Ps[1])
        end
    end
    # the ramp: U_n = -u_min/2 gives exactly half the perturbation of U_n = -u_min
    # (same unit planes, same T = L_n/u_min)
    s0 = deepcopy(generator_state(it))
    dir[] = (u_min, 0.0); t1 = sim.t + 20.0
    update_open_boundaries!(sim, t1, 1.0)
    full = copy(inflow_perturbation(it, :west)[1]); fullw = copy(inflow_perturbation(it, :west)[3])
    restore_generator_state!(it, s0)
    dir[] = (u_min / 2, 0.0)
    update_open_boundaries!(sim, t1, 1.0)
    half = inflow_perturbation(it, :west)[1]; halfw = inflow_perturbation(it, :west)[3]
    @test maximum(abs, half .- 0.5 .* full) < 1e-14 * maximum(abs, full)
    @test maximum(abs, halfw .- 0.5 .* fullw) < 1e-14 * maximum(abs, fullw)
    @test all(==(0.5), it.faces[1].ramp)
    @test feedback_exclude_width(it) == 40.0
end

# ---------------------------------------------------------------------------
# A refine = 3 nest on the ground (open sides and top), statistics from the
# parent (ParentResolvedStatistics, scale gap), a wind that veers by 90°
# over the run. One-way (the two-way feedback with N-nest-e's
# exclude_width is not merged yet; see the PR).
@testset "divergence on a refine = 3 nest, veering wind, parent statistics" begin
    pm = RegularGrid(size = (24, 24, 12), extent = (1200.0, 1200.0, 600.0), halo = (3, 3, 3))
    p = BoussinesqSimulation(pm; θ₀ = 300.0, C_s = 0.17,
                             bcs = FieldBoundaryConditions(:periodic, :periodic, :wall))
    rng = Xoshiro(7)
    for f in (p.u, p.v, p.w)
        interior_view(f) .= 0.5 .* randn(rng, size(interior_view(f)))
    end
    p.u.data .+= 3.0; p.v.data .+= 1.0; p.θ.data .= 300.0
    step!(p, 1e-3); p.t = 0.0
    nest = nested_grid(pm, ((300.0, 900.0), (300.0, 900.0), (0.0, 300.0)), 3)
    cm = nest_child_mesh(nest)
    c = BoussinesqSimulation(cm; θ₀ = 300.0, C_s = 0.17,
                             bcs = nest_bcs(cm, :nest, :nest; bottom = :wall, top = :nest))
    st0 = nest_interp_state(nest)
    interpolate_parent_state!(st0, p, nest; order = 3, region = :all)
    for (cf, sf) in ((c.u, st0.u), (c.v, st0.v), (c.w, st0.w), (c.θ, st0.θ))
        cf.data .= sf.data
    end
    run = NestedRun(p, c, nest; advance_parent = domain_advance(step_rk3!),
                    advance_child = domain_advance(step_rk3!), feedback = :one_way)
    # the prolongated start is not discretely divergence-free in the child:
    # project it once (an Euler step of 1e-9 s with the nest data)
    step!(c, 1e-9); c.t = 0.0
    div0 = max_divergence(c)
    it = InflowTurbulence(run; statistics_kwargs = (τ_avg = 60.0,), statistics_interval = 10.0)
    @test it.faces[1].stats isa ParentResolvedStatistics
    maxdiv = 0.0; worst = 0.0; nsteps = 30
    Rw = zeros(6); nR = 0
    for n in 1:nsteps
        # veer the parent's mean wind by 3° per step (a uniform rotation of
        # the mean keeps the parent divergence-free), between steps: the
        # one-way run starts each window from the previous B(t₀ + Δt), so
        # the child's boundary stays continuous and the rotation enters
        # through the next window's end state. (Before the first step the
        # window start is prolongated afresh: a rotation there would be a
        # jump in the boundary data, not a test of the inflow.)
        φ = n == 1 ? 0.0 : deg2rad(3.0)
        Ub = sum(interior_view(p.u)) / length(interior_view(p.u))
        Vb = sum(interior_view(p.v)) / length(interior_view(p.v))
        p.u.data .+= (cos(φ) - 1) * Ub - sin(φ) * Vb
        p.v.data .+= sin(φ) * Ub + (cos(φ) - 1) * Vb
        step_nested!(run; dt = 2.0)
        maxdiv = max(maxdiv, max_divergence(c))
        worst = max(worst, maximum(a.relative for a in added_mass_flux(it)))
        u, v, w = inflow_perturbation(it, :west)
        if any(!iszero, u)
            for k in axes(u, 2), j in axes(u, 1)
                Rw .+= (u[j, k]^2, v[j, k]^2, w[j, k]^2, u[j, k] * v[j, k], u[j, k] * w[j, k], v[j, k] * w[j, k])
                nR += 1
            end
        end
    end
    Rw ./= max(nR, 1)
    ke = injected_ke_flux(it)
    @printf("  refine 3, %d parent steps: max|∇·u| child %.2e, worst added flux %.2e, KE flux %.4f W/m² (total)\n",
            nsteps, maxdiv, worst, ke.total)
    @printf("  west-face plane R_ij (m²/s²): R11 %.4f R22 %.4f R33 %.4f R12 %.4f R13 %.4f R23 %.4f\n", Rw...)
    @test div0 < 1e-12
    @test maxdiv < 1e-10
    @test worst < 1e-12
    @test all(isfinite, interior_view(c.u)) && maximum(abs, interior_view(c.u)) < 20
    @test nR > 0 && Rw[1] > 0
    @test isempty(inflow_projection_log(it.faces[1].stats))
    Rt = inflow_stresses(it.faces[1].stats, it.z, c.t)
    @printf("  west-face target (ParentResolvedStatistics, level mean) R11 %.4f R22 %.4f R33 %.4f\n",
            sum(r[1] for r in Rt) / length(Rt), sum(r[2] for r in Rt) / length(Rt),
            sum(r[3] for r in Rt) / length(Rt))

    # Two-way is not tested here: without N-nest-e's (#263) exclude_width
    # the feedback restricts the synthetic eddies next to the inflow faces
    # into the parent, whose prolongation returns them as boundary data
    # (measured: child KE ×5.5, parent ×3.2 in 200 s; ×1.0 without the
    # inflow turbulence). The default frame is feedback_exclude_width(it).
end

# ---------------------------------------------------------------------------
@testset "restart: generator state saved and restored continues bit for bit" begin
    m = RegularGrid(size = (16, 12, 8), extent = (160.0, 120.0, 80.0), halo = (3, 3, 3))
    sim, _ = _it_sim(m, 2.0, -1.0)
    stats = ParametrizedStatistics(; u_star = t -> 0.3 + 1e-3t, z_i = 800.0, length_scale = 25.0)
    it = InflowTurbulence(sim; statistics = stats, length_interval = 5.0)
    for _ in 1:10; step_rk3!(sim, 0.5); end
    gs = deepcopy(generator_state(it))
    snap = state_snapshot(sim); save_state!(snap, sim); ob = open_boundary_snapshot(sim)
    t0 = sim.t
    for _ in 1:20; step_rk3!(sim, 0.5); end
    ua = copy(sim.u.data); wa = copy(sim.w.data); Pa = copy(inflow_perturbation(it, :west)[1])
    restore_state!(sim, snap); restore_open_boundaries!(sim, ob); sim.t = t0
    restore_generator_state!(it, gs)
    for _ in 1:20; step_rk3!(sim, 0.5); end
    @test sim.u.data == ua
    @test sim.w.data == wa
    @test inflow_perturbation(it, :west)[1] == Pa
    @test any(!iszero, Pa)
end

# ---------------------------------------------------------------------------
# N-inflow-e (#268): what the Berlin nest needs of the machinery above.

# the refine-3 nest of the divergence test, on a parent with a weak
# stratification, one-way or two-way
function _it_nest(feedback; seed = 7)
    pm = RegularGrid(size = (24, 24, 12), extent = (1200.0, 1200.0, 600.0), halo = (3, 3, 3))
    p = BoussinesqSimulation(pm; θ₀ = 300.0, C_s = 0.17,
                             bcs = FieldBoundaryConditions(:periodic, :periodic, :wall))
    rng = Xoshiro(seed)
    for f in (p.u, p.v, p.w)
        interior_view(f) .= 0.5 .* randn(rng, size(interior_view(f)))
    end
    p.u.data .+= 3.0; p.v.data .+= 1.0
    for k in axes(p.θ.data, 3)
        p.θ.data[:, :, k] .= 300.0 + 0.003 * (k - pm.Hz - 0.5) * pm.dz
    end
    step!(p, 1e-3); p.t = 0.0
    nest = nested_grid(pm, ((300.0, 900.0), (300.0, 900.0), (0.0, 300.0)), 3)
    cm = nest_child_mesh(nest)
    c = BoussinesqSimulation(cm; θ₀ = 300.0, C_s = 0.17,
                             bcs = nest_bcs(cm, :nest, :nest; bottom = :wall, top = :nest))
    st0 = nest_interp_state(nest)
    interpolate_parent_state!(st0, p, nest; order = 3, region = :all)
    for (cf, sf) in ((c.u, st0.u), (c.v, st0.v), (c.w, st0.w), (c.θ, st0.θ))
        cf.data .= sf.data
    end
    run = NestedRun(p, c, nest; advance_parent = domain_advance(step_rk3!),
                    advance_child = domain_advance(step_rk3!), feedback = feedback)
    step!(c, 1e-9); c.t = 0.0
    return p, c, nest, run
end

@testset "ParentResolvedStatistics on a StretchedGrid parent (the Berlin parent)" begin
    # the same random state on a RegularGrid and on its uniform StretchedGrid
    # (bit-identical levels): every statistic agrees exactly, and the levels
    # are the stretched grid's own centres
    g = RegularGrid(size = (24, 20, 10), extent = (1200.0, 1000.0, 500.0), halo = (3, 3, 3))
    sg = StretchedGrid(g)
    mk(m) = BoussinesqSimulation(m; θ₀ = 300.0, C_s = 0.17,
                                 bcs = FieldBoundaryConditions(:periodic, :periodic, :wall))
    a, b = mk(g), mk(sg)
    rng = Xoshiro(11)
    for (fa, fb) in ((a.u, b.u), (a.v, b.v), (a.w, b.w))
        interior_view(fa) .= 2.0 .+ randn(rng, size(interior_view(fa)))
        interior_view(fb) .= interior_view(fa)
    end
    lev = 2:8
    sa = ParentResolvedStatistics(a; face = :west, refine = (2, 2), levels = lev)
    sb = ParentResolvedStatistics(b; face = :west, refine = (2, 2), levels = lev)
    for _ in 1:3
        update_inflow_statistics!(sa, 10.0); update_inflow_statistics!(sb, 10.0)
    end
    z = [25.0 * k for k in 3:15]
    @test sb.z == [sg.zc[sg.Hz+k] for k in lev] == sa.z
    @test inflow_stresses(sa, z, 0.0) == inflow_stresses(sb, z, 0.0)
    La, Lb = inflow_lengths(sa, z, 0.0), inflow_lengths(sb, z, 0.0)
    @test La.L_s == Lb.L_s && La.L_z == Lb.L_z && La.T == Lb.T
    @test inflow_normal_length(sa, z, 0.0) == inflow_normal_length(sb, z, 0.0)
    @test any(r -> r[1] > 0, inflow_stresses(sb, z, 0.0))
end

@testset "fold_inflow_statistics!: the parent statistics folded where the parent is at rest" begin
    p, c, nest, run = _it_nest(:one_way)
    it = InflowTurbulence(run; statistics_kwargs = (τ_avg = 60.0,), statistics_interval = Inf)
    src = it.faces[1].stats
    n0 = src.nsamples
    for _ in 1:4
        step_nested!(run; dt = 2.0)
    end
    # the refreshes never fold with an infinite interval …
    @test src.nsamples == n0 && it.t_stats == 0.0
    # … a call does, once the parent has advanced by its interval
    @test !fold_inflow_statistics!(it; interval = 10.0)            # 8 s elapsed
    step_nested!(run; dt = 2.0)
    @test fold_inflow_statistics!(it; interval = 10.0)              # 10 s
    @test it.t_stats == p.t == 10.0
    @test all(f -> f.stats.nsamples == n0 + 1, it.faces)
    @test !fold_inflow_statistics!(it; interval = 10.0)
    # with a finite interval the refresh folds by itself (the default)
    p2, c2, _, run2 = _it_nest(:one_way)
    it2 = InflowTurbulence(run2; statistics_kwargs = (τ_avg = 60.0,), statistics_interval = 10.0)
    for _ in 1:6
        step_nested!(run2; dt = 2.0)
    end
    @test it2.faces[1].stats.nsamples > n0
end

@testset "free_stream_generator: the FSD provider of the nest faces" begin
    m = RegularGrid(size = (24, 16, 12), extent = (240.0, 160.0, 120.0), halo = (3, 3, 3))
    sim, _ = _it_sim(m, 3.0, 1.0)
    it = InflowTurbulence(sim; statistics = _ITConst(_IT_R, 20.0, 20.0, 20.0),
                          generator = free_stream_generator(; U∞ = 3.0))
    @test all(f -> f.gen isa FreeStreamInflow, it.faces)
    w = only(f for f in it.faces if f.side === :west)
    @test w.gen.fst.U∞ == 3.0 && w.gen.fst.L ≈ 20.0
    worst = 0.0
    for _ in 1:40
        step_rk3!(sim, 0.5)
        worst = max(worst, maximum(a.relative for a in added_mass_flux(it)))
    end
    P = inflow_perturbation(it, :west)
    @test worst < 1e-12
    @test any(!iszero, P[1]) && all(iszero, inflow_perturbation(it, :east)[1])
    @test max_divergence(sim) < 1e-10
end

@testset "exclude_inflow_feedback!: the two-way feedback skips the inflow band" begin
    p, c, nest, run = _it_nest(TwoWayFeedback(; buffer = 2))
    it = InflowTurbulence(run; statistics_kwargs = (τ_avg = 60.0,))
    w = exclude_inflow_feedback!(run, it)
    @test w == ceil(Int, feedback_exclude_width(it) / 50.0 - 1e-9) && w >= 1
    fb = run.twoway
    @test fb.exclude_width == w && fb.buffer == 2 && fb.exclude_sides === :inflow && fb.reflux
    # never narrows a band that was already wider
    run.twoway = TwoWayFeedback(; buffer = 2, exclude_width = w + 3)
    @test exclude_inflow_feedback!(run, it) == w + 3
    # the mask: with the mean wind from the west-south-west, the west and
    # south bands are w wide, the east and north ones the buffer
    step_nested!(run; dt = 2.0)
    act = feedback_mask(run)
    nx, ny = nest.nx, nest.ny
    @test !any(act[1:w+3, :, :]) && any(act[w+4, :, :])
    @test !any(act[:, 1:w+3, :])
    @test any(act[nx-2, :, :]) && !any(act[nx-1:nx, :, :])
    # a one-way run is unchanged
    p1, c1, _, run1 = _it_nest(:one_way)
    @test exclude_inflow_feedback!(run1, InflowTurbulence(run1)) == 0 && run1.twoway === nothing
end

@testset "two-way: the synthetic inflow stays out of the parent's interface faces (#418)" begin
    # The interface flux correction gives the parent the child's flux minus
    # the generator's normal perturbation (synthetic_normal_velocity). With
    # the perturbation fed into the parent's interface faces, the next
    # window's boundary data carried it back and the generator added the
    # next one on top (measured on this nest, 100 s: child KE ×1.29 and
    # parent ×1.025 of one-way; on the 2 h CBL nest ×9.7, ×2.1, +0.24 K).
    function _run418(fb; T = 100.0)
        p, c, nest, run = _it_nest(fb)
        it = InflowTurbulence(run; statistics_kwargs = (τ_avg = 60.0,))
        exclude_inflow_feedback!(run, it)
        mb = 0.0; syn = 0.0
        while p.t < T - 1e-9
            step_nested!(run; dt = 2.0)
            run.twoway === nothing && continue
            mb = max(mb, nest.ledger[:mass_before_correction])
            # what the synthetic part would have put into the parent's west
            # faces: its mean over each parent face (m/s)
            P = synthetic_normal_velocity(it, :west, c.t)
            for K in 1:nest.nz, J in 1:nest.ny
                syn = max(syn, abs(sum(@view P[3J-2:3J, 3K-2:3K]) / 9))
            end
        end
        return (p = p, c = c, it = it, run = run, mb = mb, syn = syn)
    end
    one = _run418(:one_way)
    two = _run418(TwoWayFeedback(; buffer = 2))
    @printf("  two-way/one-way with the generator, 100 s: KE child ×%.3f, parent ×%.4f; relative interface imbalance before the correction %.1e (synthetic parent-face mean up to %.3f m/s)\n",
            kinetic_energy(two.c) / kinetic_energy(one.c), kinetic_energy(two.p) / kinetic_energy(one.p),
            two.mb, two.syn)
    # the contract: boundary data without a synthetic part declare none
    @test synthetic_normal_velocity(two.it.data, :west, two.c.t) === nothing
    @test synthetic_normal_velocity(two.it, :east, two.c.t) == zeros(size(inflow_perturbation(two.it, :east)[1]))
    # the parent's interface faces match the child's data part to round-off
    # (the correction has nothing left to copy), while the synthetic part
    # is not small
    @test two.syn > 0.01                                   # m/s
    @test two.mb < 1e-12                                   # 1.4e-2 before
    # no feedback loop: the child and the parent stay at their one-way energy
    @test abs(kinetic_energy(two.c) / kinetic_energy(one.c) - 1) < 0.05   # measured ×0.995 (×1.29 before)
    @test abs(kinetic_energy(two.p) / kinetic_energy(one.p) - 1) < 0.01   # measured ×0.999 (×1.025 before)
    # the feedback gives the child's faces back bit for bit
    run = two.run
    w = begin_parent_step!(run)
    Aeolus.Nesting._subcycle_child!(run, w.t_child_end)
    u0 = copy(two.c.u.data); v0 = copy(two.c.v.data)
    end_parent_step!(run, w)
    @test two.c.u.data == u0 && two.c.v.data == v0
end

end
