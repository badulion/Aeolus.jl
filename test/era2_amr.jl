using Test
using Printf
using Random
using Aeolus
using Aeolus.AdaptiveMesh
using Aeolus.Grids: interior, xnodes, ynodes, znodes
using Aeolus.Fields: set!, interior_view
using Aeolus.AdaptiveMesh: _xc, _yc, _zc, _xf, _yf, _zf, _coarse_bounds,
                           _level_size, _amr_advect!

# =========================================================================
# helpers
# =========================================================================

"Exact cell average of sin(2πx/L) over [a, b]."
_sinavg(a, b, L) = (L / (2π * (b - a))) * (cos(2π * a / L) - cos(2π * b / L))

"""
Exact cell average of f(x,y,z) = sin(2πx/L)·sin(2πy/L) over the cell whose
array index is (i,j,k) on mesh `m` (halos included).
"""
function _exact_avg(m, i, j, k, L)
    xa = m.x0 + (i - m.Hx - 1) * m.dx; xb = xa + m.dx
    ya = m.y0 + (j - m.Hy - 1) * m.dy; yb = ya + m.dy
    return _sinavg(xa, xb, L) * _sinavg(ya, yb, L)
end

"Fill a cell-centred field's interior with the exact cell averages of the test function."
function _fill_exact_interior!(f, L)
    m = f.mesh
    ii, jj, kk = interior(m)
    for k in kk, j in jj, i in ii
        f.data[i, j, k] = _exact_avg(m, i, j, k, L)
    end
    return f
end

"Max error over the halo cells only (where the prolongation wrote)."
function _halo_max_error(cf, L)
    m = cf.mesh
    ii, jj, kk = interior(m)
    e = 0.0
    for k in axes(cf.data, 3), j in axes(cf.data, 2), i in axes(cf.data, 1)
        (i in ii) && (j in jj) && (k in kk) && continue
        e = max(e, abs(cf.data[i, j, k] - _exact_avg(m, i, j, k, L)))
    end
    return e
end

"Taylor-Green-like divergence-free velocity, set on the staggered faces."
function _init_taylor_green!(sim, L)
    m = sim.mesh
    for i in axes(sim.u.data, 1), j in axes(sim.u.data, 2), k in axes(sim.u.data, 3)
        sim.u.data[i,j,k] =  sin(2π * _xf(m,i) / L) * cos(2π * _yc(m,j) / L)
        sim.v.data[i,j,k] = -cos(2π * _xc(m,i) / L) * sin(2π * _yf(m,j) / L)
        sim.w.data[i,j,k] = 0.0
        sim.θ.data[i,j,k] = sin(2π * _xc(m,i) / L) * sin(2π * _yc(m,j) / L)
    end
    return sim
end

"Build a fresh (parent, child, nest) triple for the flux-correction test."
function _make_nest_pair(; Np=16, r=2, L=1.0)
    pm = RegularGrid(Float64; size=(Np,Np,4), extent=(L,L,L), halo=(3,3,3))
    nest = nested_grid(pm, ((L/4, L/2), (L/4, L/2), (0.0, L)), (r, r, 1);
                       halo=(3,3,3))
    cm = nest_child_mesh(nest)
    psim = BoussinesqSimulation(pm)
    csim = BoussinesqSimulation(cm; bcs=nest_bcs(:nest, :nest, :periodic),
                                solver=Aeolus.PressureSolvers.FFTPoissonSolver(cm))
    _init_taylor_green!(psim, L)
    step!(psim, 1e-3)                       # project to machine-precision div-free
    st = nest_interp_state(nest)
    interpolate_parent_state!(st, psim, nest; order=3, region=:all)
    for (cf, sf) in ((csim.u, st.u), (csim.v, st.v), (csim.w, st.w), (csim.θ, st.θ))
        cf.data .= sf.data
    end
    return (psim, csim, nest)
end

# =========================================================================

@testset "Era II — Nesting & AMR" begin

# -------------------------------------------------------------------------
@testset "NestedGrid geometry + validation" begin
    pm = RegularGrid(Float64; size=(16,16,8), extent=(1.0,1.0,1.0), halo=(2,2,2))
    nest = nested_grid(pm, ((0.25,0.5), (0.25,0.5), (0.0,1.0)), 2)
    cm = nest_child_mesh(nest)
    @test refinement_ratio(nest) == (2,2,2)
    @test nest_offset(nest) == (5, 5, 1)
    @test (cm.Nx, cm.Ny, cm.Nz) == (8, 8, 16)
    @test cm.dx ≈ pm.dx / 2
    @test cm.x0 ≈ 0.25
    @test nest_covered_range(nest) == (5:8, 5:8, 1:8)

    # refinement 3
    n3 = nested_grid(pm, ((0.25,0.5), (0.25,0.5), (0.0,1.0)), 3)
    @test refinement_ratio(n3) == (3,3,3)
    @test nest_child_mesh(n3).Nx == 12

    # anisotropic refinement
    na = nested_grid(pm, ((0.25,0.5), (0.25,0.5), (0.0,1.0)), (2,3,1))
    @test (nest_child_mesh(na).Nx, nest_child_mesh(na).Ny, nest_child_mesh(na).Nz) ==
          (8, 12, 8)

    # misalignment must error clearly
    @test_throws ErrorException nested_grid(pm, ((0.26,0.5), (0.25,0.5), (0.0,1.0)), 2)
    @test_throws ErrorException nested_grid(pm, ((0.25,1.25), (0.25,0.5), (0.0,1.0)), 2)
    bad = RegularGrid(Float64; size=(7,8,8), extent=(0.25,0.25,1.0), halo=(2,2,2),
                      origin=(0.25,0.25,0.0))
    @test_throws ErrorException NestedGrid(pm, bad; refinement=2, offset=(5,5,1))
end

# -------------------------------------------------------------------------
@testset "Prolongation order (parent → child halo)" begin
    L = 1.0
    orders_measured = Dict{Int,Vector{Float64}}()
    for order in (2, 3, 4)
        errs = Float64[]
        for s in (1, 2, 4)
            Np = 8 * s
            pm = RegularGrid(Float64; size=(Np, Np, 4), extent=(L,L,L), halo=(4,4,4))
            nest = nested_grid(pm, ((0.25L, 0.5L), (0.25L, 0.5L), (0.0, L)), (2,2,1);
                               halo=(2,2,2))
            pf = Field(CellCenter(), pm)
            _fill_exact_interior!(pf, L)
            Aeolus.BoundaryConditions.fill_halos!(pf, FieldBoundaryConditions(:periodic))
            cf = Field(CellCenter(), nest_child_mesh(nest))
            interpolate_to_nest_boundary!(cf, pf, nest; order=order)
            push!(errs, _halo_max_error(cf, L))
        end
        p1 = log2(errs[1] / errs[2]); p2 = log2(errs[2] / errs[3])
        orders_measured[order] = [p1, p2]
        @printf("  prolongation order=%d : errors %.3e %.3e %.3e  ⇒ measured p = %.2f, %.2f\n",
                order, errs[1], errs[2], errs[3], p1, p2)
        # the asymptotic rate (finest pair) must reach the requested order
        @test p2 > order - 0.45
        @test errs[3] < errs[1]
    end
    # order 4 must be meaningfully better than order 2 at the same resolution
    @test orders_measured[4][2] > orders_measured[2][2] + 1.0
end

# -------------------------------------------------------------------------
@testset "Restriction is exactly conservative" begin
    Random.seed!(20251)
    for r in (2, 3)
        pm = RegularGrid(Float64; size=(12,12,6), extent=(1.0,1.0,1.0), halo=(2,2,2))
        nest = nested_grid(pm, ((0.25,0.5), (0.25,0.5), (0.0,1.0)), r)
        cm = nest_child_mesh(nest)
        pf = Field(CellCenter(), pm); cf = Field(CellCenter(), cm)
        cd = interior_view(cf); cd .= randn(size(cd))
        Vc = cm.dx * cm.dy * cm.dz
        Vp = pm.dx * pm.dy * pm.dz
        Ic = sum(cd) * Vc
        restrict_to_parent!(pf, cf, nest)
        cx, cy, cz = nest_covered_range(nest)
        Ip = sum(pf.data[pm.Hx .+ cx, pm.Hy .+ cy, pm.Hz .+ cz]) * Vp
        rel = abs(Ip - Ic) / abs(Ic)
        @printf("  restriction r=%d : ∫child = %.15e, ∫parent = %.15e, rel err = %.3e\n",
                r, Ic, Ip, rel)
        @test rel < 1e-13

        # face-located (flux) restriction is area-weighted: a constant u
        # restricts to the same constant.
        uf_c = Field(FaceX(), cm); uf_p = Field(FaceX(), pm)
        fill!(uf_c.data, 2.5)
        restrict_to_parent!(uf_p, uf_c, nest; include_boundary=true)
        for K in 1:nest.nz, J in 1:nest.ny, q in 0:nest.nx
            @test uf_p.data[pm.Hx + nest.ix + q, pm.Hy + nest.iy + J - 1,
                            pm.Hz + nest.iz + K - 1] ≈ 2.5
        end
    end
end

# -------------------------------------------------------------------------
@testset "Round trip on a constant field" begin
    for r in (2, 3), loc in (CellCenter(), FaceX(), FaceY(), FaceZ())
        pm = RegularGrid(Float64; size=(12,12,6), extent=(1.0,1.0,1.0), halo=(3,3,3))
        nest = nested_grid(pm, ((0.25,0.5), (0.25,0.5), (0.0,1.0)), r)
        cm = nest_child_mesh(nest)
        pf = Field(loc, pm); cf = Field(loc, cm)
        fill!(pf.data, 7.25)
        interpolate_parent_to_child!(cf, pf, nest; order=3)
        @test maximum(abs, cf.data .- 7.25) < 1e-14
        pback = Field(loc, pm)
        if loc isa CellCenter
            restrict_to_parent!(pback, cf, nest)
        else
            restrict_to_parent!(pback, cf, nest; include_boundary=true)
        end
        cx, cy, cz = nest_covered_range(nest)
        @test maximum(abs, pback.data[pm.Hx .+ cx, pm.Hy .+ cy, pm.Hz .+ cz] .- 7.25) < 1e-14
    end

    # flux preservation: the summed fine-face fluxes through a parent face
    # equal the parent flux, for a non-constant u.
    L = 1.0
    pm = RegularGrid(Float64; size=(16,16,4), extent=(L,L,L), halo=(3,3,3))
    nest = nested_grid(pm, ((0.25,0.5), (0.25,0.5), (0.0,L)), (2,2,1); halo=(3,3,3))
    cm = nest_child_mesh(nest)
    up = Field(FaceX(), pm); uc = Field(FaceX(), cm)
    for i in axes(up.data,1), j in axes(up.data,2), k in axes(up.data,3)
        up.data[i,j,k] = sin(2π*_xf(pm,i)/L) * cos(2π*_yc(pm,j)/L)
    end
    interpolate_parent_to_child!(uc, up, nest; order=3)
    mm = compute_nest_flux_mismatch(up, uc, nest)
    @printf("  flux-preserving prolongation: interface mismatch L1 = %.3e\n",
            mismatch_l1(mm))
    @test mismatch_l1(mm) < 1e-13
end

# -------------------------------------------------------------------------
@testset "Flux correction conserves mass (r=2 and r=3)" begin
    results = Dict{Int,Tuple{Float64,Float64}}()
    for r in (2, 3)
        p1, c1, n1 = _make_nest_pair(; Np=16, r=r)
        step_nested!(p1, c1, n1, 2e-3; flux_correction=true)
        e_with = nest_conservation_error(p1, c1, n1, :mass)

        p2, c2, n2 = _make_nest_pair(; Np=16, r=r)
        step_nested!(p2, c2, n2, 2e-3; flux_correction=false)
        e_without = nest_conservation_error(p2, c2, n2, :mass)

        results[r] = (e_with, e_without)
        @printf("  r=%d : mass error WITH flux correction = %.3e, WITHOUT = %.3e%s\n",
                r, e_with, e_without,
                e_with > 0 ? @sprintf(" (%.3g× larger)", e_without / e_with) :
                             " (with-correction error is exactly zero)")
        @test e_with < 1e-10
        @test e_without > 1e4 * max(e_with, 1e-16)
        @test e_without > 1e-8
        # the θ and momentum ledgers are populated too
        @test isfinite(nest_conservation_error(p1, c1, n1, :theta))
        @test isfinite(nest_conservation_error(p1, c1, n1, :momentum_x))
    end
    @test haskey(results, 2) && haskey(results, 3)
end

# -------------------------------------------------------------------------
@testset "Nested time stepping: substeps and clock" begin
    p, c, n = _make_nest_pair(; Np=16, r=2)
    t0p = p.t; t0c = c.t
    dt = 2e-3
    step_nested!(p, c, n, dt; substeps=2)
    @test p.t ≈ t0p + dt
    @test c.t ≈ t0c + dt
    @test isfinite(maximum(abs, interior_view(c.u)))

    p3, c3, n3 = _make_nest_pair(; Np=16, r=3)
    step_nested!(p3, c3, n3, dt)           # default substeps = max(r) = 3
    @test c3.t ≈ dt                        # child clock advanced exactly one dt
end

# -------------------------------------------------------------------------
@testset "Nest sponge reduces reflection" begin
    L = 1.0; U = 1.0
    Np = 24
    pm = RegularGrid(Float64; size=(Np, 4, 4), extent=(L, L/6, L/6), halo=(2,2,2))
    nest = nested_grid(pm, ((0.0, L), (0.0, L/6), (0.0, L/6)), (2,1,1))
    cm = nest_child_mesh(nest)

    function run_packet(sponge)
        psim = BoussinesqSimulation(pm)
        fill!(psim.u.data, U); fill!(psim.v.data, 0.0); fill!(psim.w.data, 0.0)
        fill!(psim.θ.data, 0.0)
        csim = BoussinesqSimulation(cm; bcs=nest_bcs(:nest, :periodic, :periodic),
                                    solver=Aeolus.PressureSolvers.FFTPoissonSolver(cm))
        fill!(csim.u.data, U); fill!(csim.v.data, 0.0); fill!(csim.w.data, 0.0)
        # a Gaussian θ packet travelling toward the east nest boundary
        σ = 0.05L; x0 = 0.30L
        for i in axes(csim.θ.data,1), j in axes(csim.θ.data,2), k in axes(csim.θ.data,3)
            csim.θ.data[i,j,k] = exp(-((_xc(cm,i) - x0)^2) / (2σ^2))
        end
        st = nest_interp_state(nest)
        interpolate_parent_state!(st, psim, nest; order=3, region=:all)
        dt = 0.3 * cm.dx / U
        nsteps = ceil(Int, 1.05L / (U * dt))
        for _ in 1:nsteps
            Aeolus.AdaptiveMesh.apply_nest_boundary!(csim, st)
            step_rk3!(csim, dt)
            sponge === nothing || apply_nest_sponge!(csim, st, sponge; dt=dt)
        end
        # reflected amplitude: |θ| left of mid-domain long after the packet left
        θi = interior_view(csim.θ)
        return maximum(abs, θi[1:cm.Nx÷2, :, :])
    end

    ref_none   = run_packet(nothing)
    sp = NestSponge(; width=8, α_max=25.0, profile=:cos2,
                    sides=(false, true, false, false, false, false))
    ref_sponge = run_packet(sp)
    factor = ref_none / max(ref_sponge, 1e-300)
    @printf("  reflected amplitude: no sponge = %.4e, with sponge = %.4e ⇒ reduction %.1fx\n",
            ref_none, ref_sponge, factor)
    @test ref_sponge < ref_none
    @test factor > 3.0
end

# =========================================================================
# AMR
# =========================================================================

# -------------------------------------------------------------------------
@testset "Refinement criteria detect the right cells" begin
    L = 1.0; N = 32
    m = RegularGrid(Float64; size=(N, N, 4), extent=(L,L,L), halo=(2,2,2))
    sim = BoussinesqSimulation(m)
    δ = 1.5 * m.dx
    set!(sim.θ, (x,y,z) -> tanh((x - 0.5L) / δ) + 0.02 * sin(2π * x / L))
    mask = refinement_criterion_gradient(sim; field=:θ, threshold=0.15)
    ifront = round(Int, 0.5L / m.dx)
    near = ifront-3:ifront+3
    far  = [i for i in 4:N-3 if abs(i - ifront) > 6]
    nnear = count(mask[near, :, :]); nfar = count(mask[far, :, :])
    @printf("  gradient criterion: %d flagged within ±3 cells of the front, %d in the smooth region\n",
            nnear, nfar)
    @test nnear > 0
    @test nfar == 0

    # vorticity criterion: zero for a uniform flow, positive in a shear layer
    sim2 = BoussinesqSimulation(m)
    fill!(sim2.u.data, 1.0)
    @test count(refinement_criterion_vorticity(sim2; threshold=0.2)) == 0
    for i in axes(sim2.u.data,1), j in axes(sim2.u.data,2), k in axes(sim2.u.data,3)
        sim2.u.data[i,j,k] = tanh((_yc(m,j) - 0.5L) / (2m.dy))
    end
    vm = refinement_criterion_vorticity(sim2; threshold=0.4)
    jmid = round(Int, 0.5L / m.dy)
    @printf("  vorticity criterion: %d cells flagged, %d of them within ±3 of the shear layer\n",
            count(vm), count(vm[:, max(1,jmid-3):min(N,jmid+3), :]))
    @test count(vm) > 0
    @test count(vm[:, max(1,jmid-3):min(N,jmid+3), :]) == count(vm)

    # Richardson criterion: nothing on a linear field, something on the front
    sim3 = BoussinesqSimulation(m)
    set!(sim3.θ, (x,y,z) -> 3x + 2y)
    @test count(refinement_criterion_richardson(sim3; threshold=0.01)) == 0
    rm = refinement_criterion_richardson(sim; threshold=0.01)
    @printf("  richardson criterion: %d cells flagged (front), all within ±6 of the front: %s\n",
            count(rm), string(count(rm[near, :, :]) == count(rm)))
    @test count(rm) > 0
    @test count(rm[max(1,ifront-6):min(N,ifront+6), :, :]) == count(rm)

    # combine
    u1 = combine_criteria(mask, rm; mode=:union)
    i1 = combine_criteria(mask, rm; mode=:intersection)
    @test count(u1) >= max(count(mask), count(rm))
    @test count(i1) <= min(count(mask), count(rm))
    @test count(combine_criteria(mask; mode=:union)) == count(mask)
end

# -------------------------------------------------------------------------
@testset "Berger-Rigoutsos clustering efficiency" begin
    Random.seed!(7717)
    N = 32
    mask = falses(N, N, 4)
    # three well-separated blobs plus scatter inside them
    for (cx, cy, rad) in ((7, 8, 3), (24, 22, 4), (10, 26, 2))
        for k in 1:4, j in 1:N, i in 1:N
            if (i-cx)^2 + (j-cy)^2 <= rad^2
                mask[i,j,k] = rand() < 0.95
            end
        end
    end
    min_eff = 0.7; maxb = 64
    boxes = cluster_flagged_cells(mask; min_efficiency=min_eff, max_blocks=maxb)
    eff = box_efficiency(mask, boxes)
    @printf("  Berger-Rigoutsos: %d flagged cells → %d blocks, efficiency = %.3f (target ≥ %.2f)\n",
            count(mask), length(boxes), eff, min_eff)
    @test length(boxes) <= maxb
    @test eff >= min_eff
    # boxes must be disjoint and cover every flagged cell
    cover = falses(size(mask))
    for b in boxes
        @test !any(cover[b[1], b[2], b[3]])      # disjoint
        cover[b[1], b[2], b[3]] .= true
    end
    @test all(cover[mask])

    # empty mask → no boxes
    @test isempty(cluster_flagged_cells(falses(8,8,8)))
    # a fully flagged mask → one perfect box
    fb = cluster_flagged_cells(trues(8,8,8))
    @test length(fb) == 1 && box_efficiency(trues(8,8,8), fb) == 1.0
    # block budget is respected even on a pathological mask
    scat = falses(N, N, 2)
    for _ in 1:200; scat[rand(1:N), rand(1:N), rand(1:2)] = true; end
    @test length(cluster_flagged_cells(scat; min_efficiency=0.99, max_blocks=12)) <= 12
end

# -------------------------------------------------------------------------
@testset "Proper nesting invariant after regrid!" begin
    L = 1.0
    base = RegularGrid(Float64; size=(16,16,4), extent=(L,L,L), halo=(2,2,2))
    vel = (x,y,z,t) -> (1.0, 0.0, 0.0)
    h = amr_hierarchy(base; refinement=(2,2,1), max_level=3, vars=[:θ],
                      velocity=vel, buffer=2)
    root = h.levels[1][1]
    set!(root.state[:θ], (x,y,z) -> exp(-((x-0.5L)^2 + (y-0.5L)^2) / (2*(0.06L)^2)))
    # refine wherever θ is appreciable -> nested boxes at both levels
    crit = b -> Array(interior_view(b.state[:θ])) .> 0.05
    regrid!(h, crit; buffer=2)
    @test length(h.levels) == 3
    @test !isempty(h.levels[2]) && !isempty(h.levels[3])
    @printf("  hierarchy: level cells = %s (total %d)\n",
            string([level_cells(h, l) for l in 1:length(h.levels)]), total_cells(h))
    @test check_proper_nesting(h; buffer=2)

    # explicitly verify the containment + buffer for every level-3 block
    for b in h.levels[3]
        pb = b.parent
        cb = _coarse_bounds(h, b)
        N2 = _level_size(h, 2)
        for d in 1:3
            @test first(cb[d]) >= first(pb.bounds[d])
            @test last(cb[d])  <= last(pb.bounds[d])
            @test (first(pb.bounds[d]) == 1) ||
                  (first(cb[d]) - first(pb.bounds[d]) >= 2)
            @test (last(pb.bounds[d]) == N2[d]) ||
                  (last(pb.bounds[d]) - last(cb[d]) >= 2)
        end
    end
    # a buffer that cannot fit simply yields no deeper level (no invalid blocks)
    h2 = amr_hierarchy(base; refinement=(2,2,1), max_level=3, vars=[:θ],
                       velocity=vel, buffer=2)
    set!(h2.levels[1][1].state[:θ], (x,y,z) -> exp(-((x-0.5L)^2+(y-0.5L)^2)/(2*(0.06L)^2)))
    regrid!(h2, crit; buffer=64)
    @test check_proper_nesting(h2; buffer=64)
end

# -------------------------------------------------------------------------
@testset "AMR conservation across a regrid event" begin
    L = 1.0
    base = RegularGrid(Float64; size=(16,16,4), extent=(L,L,L), halo=(2,2,2))
    Uv = (1.0, 0.5, 0.0)
    vel = (x,y,z,t) -> Uv
    h = amr_hierarchy(base; refinement=(2,2,1), max_level=2, vars=[:θ],
                      velocity=vel, buffer=2, order=3)
    σ = 0.07L
    set!(h.levels[1][1].state[:θ],
         (x,y,z) -> exp(-((x-0.3L)^2 + (y-0.4L)^2) / (2σ^2)))
    # value AND Löhner: the value filter bounds the refined region (the
    # scale-free Löhner indicator alone flags the whole Gaussian tail).
    crit = b -> combine_criteria(
        Array(interior_view(b.state[:θ])) .> 0.05,
        refinement_criterion_gradient(b; field=:θ, threshold=0.2); mode=:intersection)

    regrid!(h, crit; buffer=2)
    m0 = amr_total_mass(h)
    dt = 0.3 * base.dx / 1.2
    nregrid = 0
    for n in 1:12
        nb = length(h.levels) >= 2 ? map(b -> b.bounds, h.levels[2]) : []
        step_amr!(h, dt; regrid_interval=3, criterion=crit)
        na = length(h.levels) >= 2 ? map(b -> b.bounds, h.levels[2]) : []
        nb == na || (nregrid += 1)
    end
    m1 = amr_total_mass(h)
    rel = abs(m1 - m0) / abs(m0)
    @printf("  AMR mass: m0 = %.15e, m1 = %.15e, rel err = %.3e (%d regrid events, %d blocks)\n",
            m0, m1, rel, nregrid, length(h.levels) >= 2 ? length(h.levels[2]) : 0)
    @test nregrid >= 1
    @test rel < 1e-10

    # refluxing does real work: the same run with the flux correction
    # disabled must lose mass measurably.
    h2 = amr_hierarchy(base; refinement=(2,2,1), max_level=2, vars=[:θ],
                       velocity=vel, buffer=2, order=3)
    set!(h2.levels[1][1].state[:θ],
         (x,y,z) -> exp(-((x-0.3L)^2 + (y-0.4L)^2) / (2σ^2)))
    regrid!(h2, crit; buffer=2)
    n0 = amr_total_mass(h2)
    # monkey-patch: advance without accumulating flux registers → no reflux
    h2.advance! = (hh, b, ddt, tt) -> begin
        _amr_advect!(hh, b, ddt, tt)
        for v in hh.vars, ax in (:x, :y, :z)
            fill!(b.state[Aeolus.AdaptiveMesh._reg(v, ax)].data, 0)
        end
        b
    end
    for n in 1:12
        step_amr!(h2, dt; regrid_interval=3, criterion=crit)
    end
    n1 = amr_total_mass(h2)
    rel_nofix = abs(n1 - n0) / abs(n0)
    @printf("  AMR mass without refluxing: rel err = %.3e (vs %.3e with)\n", rel_nofix, rel)
    @test rel_nofix > 1e3 * max(rel, 1e-16)
end

# -------------------------------------------------------------------------
@testset "Refinement improves accuracy at a fraction of the cells" begin
    L = 1.0
    Ux, Uy = 1.0, 0.5
    vel = (x,y,z,t) -> (Ux, Uy, 0.0)
    σ = 0.07L; x0 = 0.30L; y0 = 0.40L
    tend = 0.20
    per(a) = mod(a, L)
    exact = (x,y,z) -> begin
        dx = x - per(x0 + Ux*tend); dy = y - per(y0 + Uy*tend)
        dx -= L*round(dx/L); dy -= L*round(dy/L)
        exp(-(dx^2 + dy^2) / (2σ^2))
    end
    init = (x,y,z) -> exp(-((x-x0)^2 + (y-y0)^2) / (2σ^2))

    "Uniform-grid reference run using the same block advance kernel."
    function uniform_run(N)
        m = RegularGrid(Float64; size=(N, N, 4), extent=(L,L,L), halo=(2,2,2))
        hh = amr_hierarchy(m; refinement=(2,2,1), max_level=1, vars=[:θ],
                           velocity=vel, buffer=2)
        set!(hh.levels[1][1].state[:θ], init)
        dt = 0.3 * m.dx / hypot(Ux, Uy)
        ns = max(1, round(Int, tend / dt)); dt = tend / ns
        for _ in 1:ns
            step_amr!(hh, dt; regrid_interval=0)
        end
        return (amr_error(hh, exact), total_cells(hh))
    end

    e_coarse, n_coarse = uniform_run(16)
    e_fine,   n_fine   = uniform_run(32)

    # (c) AMR: base 16³-class grid, level 2 tracking the Gaussian
    base = RegularGrid(Float64; size=(16,16,4), extent=(L,L,L), halo=(2,2,2))
    h = amr_hierarchy(base; refinement=(2,2,1), max_level=2, vars=[:θ],
                      velocity=vel, buffer=1, order=3)
    set!(h.levels[1][1].state[:θ], init)
    crit = b -> Array(interior_view(b.state[:θ])) .> 0.02
    regrid!(h, crit; buffer=1)
    dt = 0.3 * base.dx / hypot(Ux, Uy)
    ns = max(1, round(Int, tend / dt)); dt = tend / ns
    cellsum = 0; nsam = 0
    for _ in 1:ns
        step_amr!(h, dt; regrid_interval=2, criterion=crit)
        cellsum += total_cells(h); nsam += 1
    end
    e_amr = amr_error(h, exact)
    n_amr = cellsum / nsam

    @printf("  coarse 16³-class : L1 error = %.4e, cells = %d\n", e_coarse, n_coarse)
    @printf("  fine   32²×4     : L1 error = %.4e, cells = %d\n", e_fine, n_fine)
    @printf("  AMR (2 levels)   : L1 error = %.4e, cells = %.0f (mean)  = %.0f%% of fine\n",
            e_amr, n_amr, 100 * n_amr / n_fine)
    @test e_amr < e_coarse
    @test n_amr < n_fine
    # AMR must land much closer to the fine-grid error than to the coarse one
    @test (e_amr - e_fine) < 0.5 * (e_coarse - e_fine)
end

end # testset
