using Test
using Random
using Aeolus
using Aeolus.AdaptiveMesh
using Aeolus.Grids: interior
using Aeolus.Fields: set!, interior_view
using Aeolus.AdaptiveMesh: _xc, _yc, _xf, _yf, _coarse_bounds, _level_size, _amr_advect!

@testset "Nesting & AMR" begin

@testset "NestedGrid geometry + validation" begin
    pm = RegularGrid(Float64; size=(16,16,8), extent=(1.0,1.0,1.0), halo=(2,2,2))
    nest = nested_grid(pm, ((0.25,0.5), (0.25,0.5), (0.0,1.0)), (2,3,1))
    cm = nest_child_mesh(nest)
    @test refinement_ratio(nest) == (2,3,1) && nest_offset(nest) == (5, 5, 1)
    @test (cm.Nx, cm.Ny, cm.Nz) == (8, 12, 8) && cm.dx ≈ pm.dx / 2
    @test_throws ErrorException nested_grid(pm, ((0.26,0.5), (0.25,0.5), (0.0,1.0)), 2)  # misaligned
end

# Since N-nest-d (#261) the child's nest faces carry the parent's normal
# velocity, so the interface mismatch of a step is round-off; the flux
# correction still removes a genuine coarse/fine disagreement (a perturbed
# child interface flux, as two-way feedback without boundary data gives).
@testset "Flux correction removes a coarse/fine flux disagreement" begin
    L = 1.0
    pm = RegularGrid(Float64; size=(16,16,4), extent=(L,L,L), halo=(3,3,3))
    nest = nested_grid(pm, ((L/4, L/2), (L/4, L/2), (0.0, L)), (2, 2, 1); halo=(3,3,3))
    cm = nest_child_mesh(nest)
    up = Field(FaceX(), pm); uc = Field(FaceX(), cm)
    for i in axes(up.data,1), j in axes(up.data,2), k in axes(up.data,3)
        up.data[i,j,k] = sin(2π*_xf(pm,i)/L) * cos(2π*_yc(pm,j)/L)
    end
    interpolate_parent_to_child!(uc, up, nest; order=3)
    @test mismatch_l1(compute_nest_flux_mismatch(up, uc, nest)) < 1e-13   # flux-preserving
    rng = Xoshiro(32)
    for i in (cm.Hx + 1, cm.Hx + cm.Nx + 1)
        uc.data[i, cm.Hy+1:cm.Hy+cm.Ny, cm.Hz+1:cm.Hz+cm.Nz] .+= 1e-2 .* randn(rng, cm.Ny, cm.Nz)
    end
    mm = compute_nest_flux_mismatch(up, uc, nest)
    leak = mismatch_l1(mm)
    apply_flux_correction!(up, mm, nest)
    @test leak > 1e-5
    @test mismatch_l1(compute_nest_flux_mismatch(up, uc, nest)) < 1e-13 * leak / 1e-5
end

@testset "Refinement criteria detect the right cells" begin
    L = 1.0; N = 32
    m = RegularGrid(Float64; size=(N, N, 4), extent=(L,L,L), halo=(2,2,2))
    sim = BoussinesqSimulation(m)
    set!(sim.θ, (x,y,z) -> tanh((x - 0.5L) / (1.5 * m.dx)) + 0.02 * sin(2π * x / L))
    mask = refinement_criterion_gradient(sim; field=:θ, threshold=0.15)
    ifront = round(Int, 0.5L / m.dx)
    # the smooth part: away from the front and from the periodic wrap at
    # x = 0 = L, where tanh jumps from −1 to +1 (flagged, correctly)
    far = [i for i in 4:N-3 if abs(i - ifront) > 6]
    @test count(mask[ifront-3:ifront+3, :, :]) > 0
    @test count(mask[far, :, :]) == 0
    fill!(sim.u.data, 1.0)
    @test count(refinement_criterion_vorticity(sim; threshold=0.2)) == 0   # uniform flow
end

@testset "Berger-Rigoutsos clustering efficiency" begin
    Random.seed!(7717)
    N = 32
    mask = falses(N, N, 4)
    for (cx, cy, rad) in ((7, 8, 3), (24, 22, 4), (10, 26, 2))
        for k in 1:4, j in 1:N, i in 1:N
            (i-cx)^2 + (j-cy)^2 <= rad^2 && (mask[i,j,k] = rand() < 0.95)
        end
    end
    boxes = cluster_flagged_cells(mask; min_efficiency=0.7, max_blocks=64)
    @test length(boxes) <= 64 && box_efficiency(mask, boxes) >= 0.7
    # boxes are disjoint and cover every flagged cell
    cover = zeros(Int, size(mask))
    for b in boxes
        cover[b[1], b[2], b[3]] .+= 1
    end
    @test maximum(cover) == 1 && all(cover[mask] .== 1)
end

@testset "Proper nesting invariant after regrid!" begin
    L = 1.0
    base = RegularGrid(Float64; size=(16,16,4), extent=(L,L,L), halo=(2,2,2))
    h = amr_hierarchy(base; refinement=(2,2,1), max_level=3, vars=[:θ],
                      velocity=(x,y,z,t) -> (1.0, 0.0, 0.0), buffer=2)
    set!(h.levels[1][1].state[:θ], (x,y,z) -> exp(-((x-0.5L)^2 + (y-0.5L)^2) / (2*(0.06L)^2)))
    regrid!(h, b -> Array(interior_view(b.state[:θ])) .> 0.05; buffer=2)
    @test length(h.levels) == 3 && !isempty(h.levels[3])
    @test check_proper_nesting(h; buffer=2)
    # containment + buffer of every level-3 block, explicitly
    N2 = _level_size(h, 2)
    @test all(h.levels[3]) do b
        pb = b.parent; cb = _coarse_bounds(h, b)
        all(d -> first(cb[d]) >= first(pb.bounds[d]) && last(cb[d]) <= last(pb.bounds[d]) &&
                 (first(pb.bounds[d]) == 1 || first(cb[d]) - first(pb.bounds[d]) >= 2) &&
                 (last(pb.bounds[d]) == N2[d] || last(pb.bounds[d]) - last(cb[d]) >= 2), 1:3)
    end
end

@testset "AMR conservation across regrid events; refluxing does the work" begin
    L = 1.0
    base = RegularGrid(Float64; size=(16,16,4), extent=(L,L,L), halo=(2,2,2))
    vel = (x,y,z,t) -> (1.0, 0.5, 0.0)
    init = (x,y,z) -> exp(-((x-0.3L)^2 + (y-0.4L)^2) / (2*(0.07L)^2))
    # value AND Löhner: the value filter bounds the refined region
    crit = b -> combine_criteria(
        Array(interior_view(b.state[:θ])) .> 0.05,
        refinement_criterion_gradient(b; field=:θ, threshold=0.2); mode=:intersection)
    dt = 0.3 * base.dx / 1.2
    function run(reflux)
        h = amr_hierarchy(base; refinement=(2,2,1), max_level=2, vars=[:θ],
                          velocity=vel, buffer=2, order=3)
        set!(h.levels[1][1].state[:θ], init)
        regrid!(h, crit; buffer=2)
        reflux || (h.advance! = (hh, b, ddt, tt) -> begin     # no flux registers → no reflux
            _amr_advect!(hh, b, ddt, tt)
            for v in hh.vars, ax in (:x, :y, :z)
                fill!(b.state[Aeolus.AdaptiveMesh._reg(v, ax)].data, 0)
            end
            b
        end)
        m0 = amr_total_mass(h); nregrid = 0
        for n in 1:12
            nb = map(b -> b.bounds, h.levels[2])
            step_amr!(h, dt; regrid_interval=3, criterion=crit)
            nb == map(b -> b.bounds, h.levels[2]) || (nregrid += 1)
        end
        return abs(amr_total_mass(h) - m0) / abs(m0), nregrid
    end
    rel, nregrid = run(true)
    rel_nofix, _ = run(false)
    @test nregrid >= 1
    @test rel < 1e-10
    @test rel_nofix > 1e3 * max(rel, 1e-16)
end

end # testset
