# N-nest-d (#261, epic #246): one-way parent → child boundary data.
#
# The child's nest faces are N-open-b OpenBoundary faces fed by a
# NestBoundaryData: the time-blended (N-nest-c), prolongated (N-nest-a)
# parent field set on the normal-velocity face planes and the tangential
# and scalar halos, mass-corrected by the single correct_boundary_mass_flux!.
#
#   1. mass: divergence-free random parent → net child boundary flux after
#      the correction ≤ 1e-13·U·A, |δ|/U, child max|∇·u| < 1e-10
#   2. one-way consistency in a convective boundary layer (overlap means
#      within the run-to-run spread): validation/era2_nest_boundary.jl
#   3. top boundary: the child top halo follows a parent θ trend, and the
#      child column responds only through the boundary flux
#   4. faces inside solids (a building cut by the east face): u·n = 0,
#      excluded from δ
#   5. perfect parent: refinement 1, the child driven by the truth
#   6. outflow condition through nest_bcs (#329): a child-only θ pulse
#      leaves through a convective outflow face, but is trapped by the
#      Dirichlet (parent) one; the advection halo is checked
#   7. a NestSponge keeps the child divergence-free (#329): the velocity is
#      relaxed before the projection
#   8. fill_nest_halos! (public since #329): new boundary data reach the
#      child's halos
#
# References: Hellsten et al. (2021), GMD 14, 3185 (PALM nesting: Dirichlet
# child boundary from the interpolated parent, §3.2; mass-flux correction,
# §3.3, Eq. 2); Kadasch et al. (2021), GMD 14, 5435, §2.2–2.3 (no sponge at
# the child boundary); Deardorff (1970), JAS 27, 1211 (convective velocity
# scale w* = (g/θ₀ H zᵢ)^(1/3)).

using Test
using Random
using Aeolus
using Aeolus.Nesting
using Aeolus.Grids: interior
using Aeolus.Fields: interior_view
using Aeolus.BoundaryConditions: FieldBoundaryConditions, OpenBoundary, open_boundary_states
using Aeolus.Operators: divergence
using Aeolus.IBM: SolidMask, mark_solid!, apply_solid_mask!

_nb_xc(m, i) = m.x0 + (i - m.Hx - 0.5) * m.dx

# `_nb_init_child!`, shared with the CBL validation run (validation/era2_nest_boundary.jl)
isdefined(@__MODULE__, :_nb_init_child!) || include(joinpath(@__DIR__, "support", "nest_boundary.jl"))

"Total area of the open faces of the child's (first) open boundary set."
function _nb_open_area(csim)
    m = csim.mesh; o = nest_open_sides(csim)
    return (o[1] + o[2]) * m.Ly * m.Lz + (o[3] + o[4]) * m.Lx * m.Lz +
           (o[5] + o[6]) * m.Lx * m.Ly
end

@testset "Era II — one-way nest boundary data (N-nest-d, #261)" begin

@testset "mass: divergence-free parent, net boundary flux and δ" begin
    rng = Xoshiro(261)
    # the Berlin shape: ground wall, open sides and top, r = (2, 2, 4)
    pm = RegularGrid(Float64; size = (8, 8, 8), extent = (1.0, 1.0, 1.0), halo = (3, 3, 3))
    nest = nested_grid(pm, ((0.25, 0.625), (0.25, 0.625), (0.0, 0.5)), (2, 2, 4); halo = (3, 3, 3))
    cm = nest_child_mesh(nest)
    psim = BoussinesqSimulation(pm; bcs = FieldBoundaryConditions(:periodic, :periodic, :wall))
    for f in (psim.u, psim.v, psim.w)
        interior_view(f) .= randn(rng, size(interior_view(f)))
    end
    step!(psim, 1e-10); psim.t = 0.0             # project: ∇·u = 0 to round-off
    csim = BoussinesqSimulation(cm; bcs = nest_bcs(cm, :nest, :nest; bottom = :wall, top = :nest))
    _nb_init_child!(csim, psim, nest)
    run = NestedRun(psim, csim, nest; feedback = :one_way, advance_parent = domain_advance(step_rk3!),
                    advance_child = domain_advance(step_rk3!),
                    cfl_parent = CFLConstraints(C_advective = 0.4))
    # project the prolongated child once with its open faces (SSP-RK3 keeps
    # a non-solenoidal start state, ×1/9 per step)
    step!(csim, 1e-12); csim.t = psim.t
    U = maximum(abs, interior_view(psim.u))
    A = _nb_open_area(csim)
    worst_res = 0.0; worst_δ = 0.0; worst_div = 0.0
    st = only(open_boundary_states(csim))
    st.record = true
    for _ in 1:2
        step_nested!(run)
        worst_res = max(worst_res, maximum(abs(res) for (_, Q, res) in st.log))
        empty!(st.log)
        worst_δ = max(worst_δ, abs(open_boundary_ledger(csim).δ))
        divergence(csim.div_buf, csim.u, csim.v, csim.w)
        worst_div = max(worst_div, maximum(abs, interior_view(csim.div_buf)))
    end
    @test worst_res <= 1e-13 * U * A
    @test worst_div < 1e-10
    # flux-preserving prolongation + linear blend: a divergence-free parent
    # already gives a compatible boundary, δ is round-off
    @test worst_δ < 1e-13 * U
end

# A parent warming in time, θ = 300 + a t (set analytically, at rest). The
# child is nested only through its top: its top halo follows the trend and
# its column changes only by the diffusive flux through the top face.
@testset "top boundary: a parent θ trend reaches the child only through fluxes" begin
    a, κ = 1e-3, 5.0
    pm = RegularGrid(Float64; size = (4, 4, 16), extent = (200.0, 200.0, 800.0), halo = (3, 3, 3))
    nest = nested_grid(pm, ((0.0, 200.0), (0.0, 200.0), (0.0, 400.0)), (2, 2, 4))
    cm = nest_child_mesh(nest)
    psim = BoussinesqSimulation(pm; bcs = FieldBoundaryConditions(:periodic, :periodic, :wall))
    fill!(psim.θ.data, 300.0)
    csim = BoussinesqSimulation(cm; ν = κ,
                                bcs = nest_bcs(cm, :periodic, :periodic; bottom = :wall, top = :nest))
    fill!(csim.θ.data, 300.0)
    core = (s, dt; kw...) -> (s.t += dt; fill!(s.θ.data, 300.0 + a * s.t); s)
    halo_err = Ref(0.0); flux = Ref(0.0)
    kt = cm.Hz + cm.Nz
    probe! = (s, Δt) -> begin           # after the tendencies, with the halos the core read
        ht = view(s.θ.data, cm.Hx+1:cm.Hx+cm.Nx, cm.Hy+1:cm.Hy+cm.Ny, kt + 1)
        halo_err[] = max(halo_err[], maximum(abs, ht .- (300.0 + a * s.t)))
        inn = view(s.θ.data, cm.Hx+1:cm.Hx+cm.Nx, cm.Hy+1:cm.Hy+cm.Ny, kt)
        flux[] += Δt * κ * sum(ht .- inn) / cm.dz * cm.dx * cm.dy
        s
    end
    run = NestedRun(psim, csim, nest; feedback = :one_way, advance_parent = domain_advance(core),
                    advance_child = domain_advance(; body_force! = probe!),
                    cfl_parent = nothing, cfl_child = nothing, substeps = 3)
    V = cm.dx * cm.dy * cm.dz
    I0 = sum(interior_view(csim.θ) .- 300.0) * V          # anomaly integral
    for _ in 1:5
        step_nested!(run; dt = 5.0)
    end
    I1 = sum(interior_view(csim.θ) .- 300.0) * V
    @test halo_err[] < 1e-12
    @test abs((I1 - I0) - flux[]) / abs(I1 - I0) < 1e-10
    top_inc = sum(interior_view(csim.θ)[:, :, end] .- 300.0)
    bot_inc = sum(interior_view(csim.θ)[:, :, 1] .- 300.0)
    @test top_inc > 0 && top_inc > 100 * abs(bot_inc)     # warming from the top down
end

# A building cut by the child's east face: the open faces next to its solid
# cells carry u·n = 0 and are left out of the mass-flux correction, exactly.
@testset "faces inside solids: u·n = 0, excluded from δ" begin
    pm = RegularGrid(Float64; size = (16, 16, 8), extent = (800.0, 800.0, 400.0), halo = (3, 3, 3))
    nest = nested_grid(pm, ((200.0, 600.0), (200.0, 600.0), (0.0, 200.0)), 2)
    cm = nest_child_mesh(nest)
    psim = BoussinesqSimulation(pm; bcs = FieldBoundaryConditions(:periodic, :periodic, :wall))
    fill!(psim.u.data, 3.0)                               # eastward: outflow at the east face
    interior_view(psim.v) .= 0.2 .* randn(Xoshiro(5), size(interior_view(psim.v)))
    step!(psim, 1e-6); psim.t = 0.0
    sm = SolidMask(cm)                                    # x ∈ [550, 600], y ∈ [350, 450], z < 100
    mark_solid!(sm, (x, y, z) -> x > 550 && 350 < y < 450 && z < 100)
    csim = BoussinesqSimulation(cm; bcs = nest_bcs(cm, :nest, :nest; bottom = :wall, top = :nest))
    _nb_init_child!(csim, psim, nest)
    apply_solid_mask!(csim, sm)
    run = NestedRun(psim, csim, nest; feedback = :one_way, solid = sm,
                    advance_child = domain_advance(; post_predictor! = (s, h) -> apply_solid_mask!(s, sm)),
                    cfl_parent = CFLConstraints(C_advective = 0.4))
    st = only(open_boundary_states(csim))
    east = csim.bcs.east.planes[:u]
    step_nested!(run)
    ie = cm.Hx + cm.Nx + 1
    solid_rows = [(cm.Hy + a, cm.Hz + b) for a in 1:cm.Ny, b in 1:cm.Nz if !east.fluid[a, b]]
    @test length(solid_rows) == 4 * 4                     # 100 m × 100 m of 25 m faces
    @test all(csim.u.data[ie, j, k] == 0.0 for (j, k) in solid_rows)
    led = open_boundary_ledger(csim)
    A_fluid = sum(count(p.fluid) * (p.axis == 1 ? cm.dy * cm.dz : p.axis == 2 ? cm.dx * cm.dz :
                                    cm.dx * cm.dy) for p in st.planes if p.normal)
    @test led.δ ≈ led.Q / A_fluid rtol = 1e-14             # solid faces excluded from δ
    @test abs(led.residual) <= 1e-13 * 3.0 * _nb_open_area(csim)
    divergence(csim.div_buf, csim.u, csim.v, csim.w)
    @test maximum(abs, interior_view(csim.div_buf)) < 1e-10
end

# Refinement 1: the child is the parent's own grid over a sub-box, so the
# parent is the exact solution of the child's problem; with the projection
# step the child stays on it to round-off.
@testset "perfect parent: refinement 1, the child stays on the truth" begin
    L = 1.0
    pm = RegularGrid(Float64; size = (16, 16, 4), extent = (L, L, L / 4), halo = (3, 3, 3))
    nest = nested_grid(pm, ((0.25, 0.75), (0.25, 0.75), (0.0, L / 4)), (1, 1, 1))
    cm = nest_child_mesh(nest)
    psim = BoussinesqSimulation(pm; ν = 2e-3, θ₀ = 300.0, g_accel = 0.1)
    for k in axes(psim.u.data, 3), j in axes(psim.u.data, 2), i in axes(psim.u.data, 1)
        x = pm.x0 + (i - pm.Hx - 1) * pm.dx; xc = x + 0.5pm.dx
        y = pm.y0 + (j - pm.Hy - 0.5) * pm.dy; yf = y - 0.5pm.dy
        psim.u.data[i, j, k] = 0.4 + sin(2π * x) * cos(2π * y)
        psim.v.data[i, j, k] = -cos(2π * xc) * sin(2π * yf)
        psim.θ.data[i, j, k] = 300.0 + exp(-((xc - 0.4)^2 + (y - 0.5)^2) / 0.02)
    end
    interior_view(psim.w) .+= 0.05 .* randn(Xoshiro(11), size(interior_view(psim.w)))
    step!(psim, 1e-8); psim.t = 0.0
    csim = BoussinesqSimulation(cm; ν = 2e-3, θ₀ = 300.0, g_accel = 0.1,
                                bcs = nest_bcs(cm, :nest, :nest, :periodic))
    _nb_init_child!(csim, psim, nest)
    run = NestedRun(psim, csim, nest; feedback = :one_way, cfl_parent = nothing, cfl_child = nothing,
                    substeps = 1)
    cx, cy, cz = nest_covered_range(nest)
    rel(fc, fp; base = 0.0) = sqrt(sum(abs2, interior_view(fc) .- interior_view(fp)[cx, cy, cz]) /
                                   sum(abs2, interior_view(fp)[cx, cy, cz] .- base))
    for _ in 1:5
        step_nested!(run; dt = 4e-3)
    end
    @test max(rel(csim.u, psim.u), rel(csim.v, psim.v), rel(csim.θ, psim.θ; base = 300.0)) < 1e-10
end

# A θ pulse only in the child (parent θ = 0) leaves through the east face
# of a 1-D nest: the convective outflow (#329; Orlanski 1976) radiates it,
# the default Dirichlet outflow (parent θ imposed) traps and reflects it.
function _nb_outflow_pulse(mode; σ = 4.0, U = 1.0)
    pm = RegularGrid(Float64; size = (64, 1, 1), extent = (64.0, 1.0, 1.0), halo = (3, 3, 3))
    nest = nested_grid(pm, ((16.0, 48.0), (0.0, 1.0), (0.0, 1.0)), (2, 1, 1))
    cm = nest_child_mesh(nest)
    psim = BoussinesqSimulation(pm)
    fill!(psim.u.data, U)
    csim = BoussinesqSimulation(cm; bcs = nest_bcs(cm, :nest, :periodic, :periodic; outflow = mode, halo = 3))
    fill!(csim.u.data, U)
    ii, jj, kk = interior(cm)
    for i in ii
        csim.θ.data[i, jj, kk] .= exp(-(_nb_xc(cm, i) - 32.0)^2 / (2σ^2))
    end
    E0 = sum(abs2, interior_view(csim.θ))
    run = NestedRun(psim, csim, nest; feedback = :one_way, advance_parent = domain_advance(step_rk3!),
                    advance_child = domain_advance(step_rk3!),
                    cfl_parent = nothing, cfl_child = nothing, substeps = 2)
    for _ in 1:round(Int, 2 * 32.0 / (U * 0.25))          # two crossings of the child
        step_nested!(run; dt = 0.25)
    end
    return sum(abs2, interior_view(csim.θ)) / E0
end

@testset "outflow condition through nest_bcs (#329)" begin
    @test _nb_outflow_pulse(:convective) < 1e-3          # radiated out
    @test _nb_outflow_pulse(:dirichlet) > 0.1            # trapped by the imposed parent θ = 0
    cm = RegularGrid(Float64; size = (8, 8, 8), extent = (1.0, 1.0, 1.0), halo = (3, 3, 3))
    @test_throws ArgumentError nest_bcs(cm; halo = 4)   # advection halo checked
end

# A NestSponge relaxes the child velocity toward the prolongated parent
# (flux-consistent on parent cells, not divergence-free on child cells):
# relaxed on the predicted velocity, before the projection, the child stays
# divergence-free; relaxed after the step (before #329) it does not.
@testset "a NestSponge keeps the child divergence-free (#329)" begin
    rng = Xoshiro(329)
    pm = RegularGrid(Float64; size = (8, 8, 8), extent = (1.0, 1.0, 1.0), halo = (3, 3, 3))
    nest = nested_grid(pm, ((0.25, 0.75), (0.25, 0.75), (0.25, 0.75)), 2; halo = (3, 3, 3))
    cm = nest_child_mesh(nest)
    psim = BoussinesqSimulation(pm; bcs = FieldBoundaryConditions(:periodic))
    for f in (psim.u, psim.v, psim.w)
        interior_view(f) .= randn(rng, size(interior_view(f)))
    end
    step!(psim, 1e-10); psim.t = 0.0
    csim = BoussinesqSimulation(cm; bcs = nest_bcs(cm))
    _nb_init_child!(csim, psim, nest)
    run = NestedRun(psim, csim, nest; feedback = :one_way, cfl_parent = nothing, cfl_child = nothing,
                    sponge = NestSponge(; width = 3, α_max = 20.0))
    step!(csim, 1e-12); csim.t = psim.t
    for _ in 1:2
        step_nested!(run; dt = 2e-3)
    end
    divergence(csim.div_buf, csim.u, csim.v, csim.w)
    worst = maximum(abs, interior_view(csim.div_buf))
    apply_nest_sponge!(csim, boundary_time(run, run.t0 + run.Δt), run.sponge; dt = 2e-3, scalars = false)
    divergence(csim.div_buf, csim.u, csim.v, csim.w)
    @test worst < 1e-10
    @test maximum(abs, interior_view(csim.div_buf)) > 1e3 * worst   # the old placement is visible
end

# fill_nest_halos! is the public entry a driver uses after changing a
# child's boundary data outside a NestedRun step (#329, #349).
@testset "fill_nest_halos!: new boundary data reach the halos (#329)" begin
    pm = RegularGrid(Float64; size = (8, 8, 8), extent = (1.0, 1.0, 1.0), halo = (3, 3, 3))
    nest = nested_grid(pm, ((0.25, 0.75), (0.25, 0.75), (0.25, 0.75)), 2)
    cm = nest_child_mesh(nest)
    csim = BoussinesqSimulation(cm; bcs = nest_bcs(cm, :nest, :nest, :periodic))
    st = nest_interp_state(nest)
    fill!(st.θ.data, 301.0)
    apply_nest_boundary!(csim, st)
    ii, jj, kk = interior(cm)
    fill!(st.θ.data, 305.0)                       # the driver's new parent data
    @test csim.θ.data[first(ii) - 1, first(jj), first(kk)] == 301.0
    @test fill_nest_halos!(csim, st) === csim
    @test csim.θ.data[first(ii) - 1, first(jj), first(kk)] == 305.0   # copied
end

end # testset
