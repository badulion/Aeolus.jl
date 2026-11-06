using Test
using Random
using Aeolus
using Aeolus.Nesting
using Aeolus.Grids: interior
using Aeolus.Fields: interior_view
using Aeolus.StretchedGrids: StretchedGrid

# =========================================================================
# #328 — nest hierarchies (several levels, siblings), non-aligned children
# (rational ratios, origins off the parent faces) and z-stretched children.
#
#   1. NestAxis geometry: common faces, conservation box, overlaps;
#   2. operators on the overlap path: block conservation, flux
#      preservation, prolongation order for rational ratios and a stretched
#      child; the integer path is its special case (aligned = non-aligned);
#   3. free stream to round-off through three levels (H3), two-way;
#   4. composite conservation of the H3 hierarchy (refluxing at every
#      level, with the negative control);
#   5. rollback per level, refusals.
# =========================================================================

# A periodic root (16³, Δ = 100 m) with uniform flow U and θ = θ₀ (+ a blob).
function _nh_root(; N = 16, L = 1600.0, U = (5.0, 2.0, 0.0), θ0 = 300.0, ν = 0.0, C_s = 0.0,
                  moisture = false)
    pm = RegularGrid(Float64; size = (N, N, N), extent = (L, L, L), halo = (3, 3, 3))
    p = BoussinesqSimulation(pm; ν, C_s, moisture, bcs = FieldBoundaryConditions(:periodic))
    _nh_fill!(p, U, θ0)
    return p
end

_nh_fill!(s, U, θ0) = (fill!(s.u.data, U[1]); fill!(s.v.data, U[2]); fill!(s.w.data, U[3]);
                       fill!(s.θ.data, θ0); s)

# A child sim on `nest`'s child mesh with six open faces, uniform flow.
function _nh_child(nest; U = (5.0, 2.0, 0.0), θ0 = 300.0, ν = 0.0, C_s = 0.0, scalars = (),
                   moisture = false)
    cm = nest_child_mesh(nest)
    c = BoussinesqSimulation(cm; ν, C_s, moisture, bcs = nest_bcs(cm, :nest; scalars, moisture))
    return _nh_fill!(c, U, θ0)
end

# θ = θ₀ + a Gaussian blob of amplitude A at (x0, y0, z0), width σ, in a sim.
function _nh_blob!(s; x0 = 700.0, y0 = 800.0, z0 = 800.0, σ = 150.0, A = 1.0, θ0 = 300.0)
    m = s.mesh
    zc = m isa StretchedGrid ? m.zc : [m.z0 + (k - m.Hz - 0.5) * m.dz for k in 1:m.Nz+2m.Hz]
    for k in axes(s.θ.data, 3), j in axes(s.θ.data, 2), i in axes(s.θ.data, 1)
        x = m.x0 + (i - m.Hx - 0.5) * m.dx; y = m.y0 + (j - m.Hy - 0.5) * m.dy
        s.θ.data[i, j, k] = θ0 + A * exp(-((x - x0)^2 + (y - y0)^2 + (zc[k] - z0)^2) / σ^2)
    end
    return s
end

# The custom faces of a stretched child over the parent layers 400…1200 m:
# 3, 3, 3, 2, 2, 2, 1, 1 child layers per 100 m parent layer (ratio 3 → 1).
function _nh_st_faces(z0 = 400.0, splits = (3, 3, 3, 2, 2, 2, 1, 1), Δp = 100.0)
    zf = Float64[z0]
    for (K, r) in enumerate(splits), m in 1:r
        push!(zf, z0 + (K - 1) * Δp + m * Δp / r)
    end
    return zf
end

_nh_st_mesh(; extent = 800.0, origin = 400.0, zf = _nh_st_faces()) =
    StretchedGrid(Float64; size = (16, 16, length(zf) - 1), extent = (extent, extent, zf[end] - zf[1]),
                  halo = (3, 3, 3), origin = (origin, origin, zf[1]), stretching = (:custom, zf))

# H3: three levels (#328 §5), the child over the middle half of the root at
# r = 2, the grandchild over the middle half of the child at r = 2.
# Returns (hierarchy, sims).
function _nh_config(cfg::Symbol; feedback = :one_way, core = step!, blob = false, U = (5.0, 2.0, 0.0),
                    reflux = true)
    cfg === :H3 || throw(ArgumentError("only H3 is set up here"))
    fb = feedback === :two_way ? TwoWayFeedback(; reflux) : :one_way
    p = _nh_root(; U)
    blob && _nh_blob!(p)
    adv = domain_advance(core)
    h = NestHierarchy(p; advance = adv, cfl = nothing)
    n1 = nested_grid(p.mesh, ((400.0, 1200.0), (400.0, 1200.0), (400.0, 1200.0)), 2)
    c = _nh_child(n1; U); blob && _nh_blob!(c)
    add_nest!(h, :root => c, n1; id = :child, advance = adv, cfl = nothing, feedback = fb)
    n2 = nested_grid(c.mesh, ((600.0, 1000.0), (600.0, 1000.0), (600.0, 1000.0)), 2)
    g = _nh_child(n2; U); blob && _nh_blob!(g)
    add_nest!(h, :child => g, n2; id = :grand, advance = adv, cfl = nothing, feedback = fb)
    return h, Any[p, c, g]
end

# Max relative deviation of a free stream.
_nh_dev(s, U, θ0) = (maximum(abs, interior_view(s.u) .- U[1]) / hypot(U...),
                     maximum(abs, interior_view(s.v) .- U[2]) / hypot(U...),
                     maximum(abs, interior_view(s.w) .- U[3]) / hypot(U...),
                     maximum(abs, interior_view(s.θ) .- θ0) / θ0)

# Max |∇·u| Δ / |U| on the interior of a sim (stretched-aware: max_divergence).
_nh_div(s, U) = max_divergence(s) * min(s.mesh.dx, s.mesh.dy) / hypot(U...)

# A core that fails once (section 6): after `left` attempts it puts a NaN
# into θ (a numerical failure: rolled back and retried) or calls a method
# that does not exist (a bug: rethrown).
mutable struct _NHInject{F}
    core::F
    left::Int                                # attempts left before the NaN
    kind::Symbol
end
function (c::_NHInject)(sim, dt; kw...)
    c.core(sim, dt; kw...)
    c.left -= 1
    if c.left == 0
        c.kind === :nan ? (sim.θ.data[5, 5, 5] = NaN) : (sim + 1)      # a bug: MethodError
    end
    return sim
end
Aeolus.Nesting.nest_stage_weights(c::_NHInject, args...) = Aeolus.Nesting.nest_stage_weights(c.core)

@testset "nest hierarchy, non-aligned and stretched children (#328)" begin

# -------------------------------------------------------------------------
@testset "NestAxis: common faces, conservation box, overlaps" begin
    pf = Float64[(i - 4) * 30.0 for i in 1:27]                 # Hp = 3, Np = 20, Δp = 30
    # r = 5/2: Δc = 12, child over parent faces 0 … 240 m (cells 1…8)
    cf = Float64[(i - 4) * 12.0 for i in 1:27]                 # Hc = 3, Nc = 20
    a = NestAxis(pf, 3, 20, cf, 3, 20)
    @test a.integer === nothing
    @test a.pbox == 1:8 && a.cbox == 1:20
    # common faces every 2 parent cells (60 m = 5 child cells)
    @test count(!=(0), a.cface) == 5
    @test a.cface[1] == 4 && a.cface[3] == 9 && a.cface[9] == 24
    # the overlaps of each box parent cell sum to its length, and of each
    # child cell to the child cell's
    @test all(n -> isapprox(sum(a.len[a.ptr[n]:a.ptr[n+1]-1]), 30.0; rtol = 1e-15), 1:8)
    cl = zeros(30)
    for q in eachindex(a.cidx); cl[a.cidx[q]] += a.len[q]; end
    @test all(c -> cl[c] ≈ 12.0, 4:23)
    # a face off the common faces: bracketed, linear weight
    @test a.cface[2] == 0 && a.fbr[2] == 6 && a.ftau[2] ≈ 0.5
    # child origin half a parent cell off: the box shrinks to the common faces
    cf2 = Float64[15 + (i - 4) * 15.0 for i in 1:27]           # r = 2, origin 15 m
    b = NestAxis(pf, 3, 20, cf2, 3, 20)
    @test b.pbox == 2:10 && b.cbox == 2:19
    # aligned integer: the fast path
    cf3 = Float64[30 + (i - 4) * 10.0 for i in 1:31]           # r = 3, offset 2
    c = NestAxis(pf, 3, 20, cf3, 3, 24)
    @test c.integer == (3, 2) && c.pbox == 2:9
    @test NestAxis(pf, 3, 20, cf3, 3, 24; integer = false).integer === nothing
    # snapping: a child face within 1e-9 Δc of a parent face takes its bits
    cf4 = copy(cf3); cf4[4] += 1e-12
    @test NestAxis(pf, 3, 20, cf4, 3, 24).cf[4] === pf[5]
end

# -------------------------------------------------------------------------
@testset "NestedGrid from the geometry: aligned picks the integer path, rational/off-face/stretched the overlap path" begin
    pm = RegularGrid(Float64; size = (16, 16, 16), extent = (1600.0, 1600.0, 1600.0), halo = (3, 3, 3))
    n1 = nested_grid(pm, ((400.0, 1200.0), (400.0, 1200.0), (400.0, 1200.0)), 2)
    n2 = NestedGrid(pm, nest_child_mesh(n1))
    @test n2.aligned && (n2.rx, n2.ry, n2.rz) == (2, 2, 2) && nest_offset(n2) == nest_offset(n1)
    @test nest_covered_range(n2) == nest_covered_range(n1)
    n3 = NestedGrid(pm, nest_child_mesh(n1); plan = :overlap)
    @test !n3.aligned && nest_covered_range(n3) == nest_covered_range(n1)
    na = nested_grid(pm, ((400.0, 1200.0), (400.0, 1200.0), (400.0, 1200.0)), (5 // 2, 2, 2))
    @test !na.aligned && refinement_ratio(na) == (5 // 2, 2, 2)
    @test nest_child_mesh(na).Nx == 20 && nest_covered_range(na) == (5:12, 5:12, 5:12)
    nb = nested_grid(pm, ((450.0, 1150.0), (400.0, 1200.0), (400.0, 1200.0)), 2)
    @test nest_covered_range(nb)[1] == 6:11                    # shrunk to the common faces
    ns = NestedGrid(pm, _nh_st_mesh())
    @test !ns.aligned && refinement_ratio(ns)[3] === nothing && nest_covered_range(ns)[3] == 5:12
    # refusals of the geometry
    coarse = RegularGrid(Float64; size = (4, 4, 4), extent = (800.0, 800.0, 800.0), halo = (3, 3, 3),
                         origin = (400.0, 400.0, 400.0))
    @test_throws ErrorException NestedGrid(pm, coarse)          # coarser than the parent
    out = RegularGrid(Float64; size = (16, 16, 16), extent = (800.0, 800.0, 800.0), halo = (3, 3, 3),
                      origin = (1000.0, 400.0, 400.0))
    @test_throws ErrorException NestedGrid(pm, out)             # leaves the parent interior
    @test_throws ArgumentError NestedGrid(pm, nest_child_mesh(n1); refinement = 2)
end

# -------------------------------------------------------------------------
# 2. Operators. Block conservation of the prolongation and restriction on
#    the overlap path; restrict∘prolong = id where every parent face is
#    common (the stretched child); flux preservation on the box planes.
_nh_smooth(x, y, z) = 2 + sin(2π * x / 1600) * cos(2π * y / 1600) + 0.3 * sin(2π * z / 800)

function _nh_set_cells!(f, g)
    m = f.mesh
    zc = m isa StretchedGrid ? m.zc : [m.z0 + (k - m.Hz - 0.5) * m.dz for k in 1:m.Nz+2m.Hz]
    for k in axes(f.data, 3), j in axes(f.data, 2), i in axes(f.data, 1)
        f.data[i, j, k] = g(m.x0 + (i - m.Hx - 0.5) * m.dx, m.y0 + (j - m.Hy - 0.5) * m.dy, zc[k])
    end
    return f
end

function _nh_box_integrals(p, c, nest)
    pm = nest.parent_mesh; cm = nest.child_mesh
    pb = nest_covered_range(nest); cb = map(a -> a.cbox, nest.axes)
    dzp = [Aeolus.Nesting._dzp(pm, k) for k in 1:pm.Nz]
    dzc = cm isa StretchedGrid ? [cm.dzc[cm.Hz + k] for k in 1:cm.Nz] : fill(cm.dz, cm.Nz)
    Ip = sum(p.data[pm.Hx + i, pm.Hy + j, pm.Hz + k] * pm.dx * pm.dy * dzp[k]
             for k in pb[3], j in pb[2], i in pb[1])
    Ic = sum(c.data[cm.Hx + i, cm.Hy + j, cm.Hz + k] * cm.dx * cm.dy * dzc[k]
             for k in cb[3], j in cb[2], i in cb[1])
    return Ip, Ic
end

@testset "overlap operators: block conservation, restrict∘prolong, flux preservation" begin
    pm = RegularGrid(Float64; size = (16, 16, 16), extent = (1600.0, 1600.0, 1600.0), halo = (3, 3, 3))
    cases = ("r = 5/2" => nested_grid(pm, ((400.0, 1200.0), (400.0, 1200.0), (400.0, 1200.0)), (5 // 2, 3 // 2, 2)),
             "off-face, r = 2" => nested_grid(pm, ((450.0, 1150.0), (350.0, 1250.0), (400.0, 1200.0)), 2),
             "r = 7/3, off-face" => nested_grid(pm, ((400.0 - 3 * 300 / 7, 400.0 + 18 * 300 / 7), (400.0, 1000.0),
                                                     (400.0, 1200.0)), (7 // 3, 7 // 3, 1)),
             "stretched z" => NestedGrid(pm, _nh_st_mesh()))
    worst_box = worst_plane = worst_corr = 0.0
    for (name, nest) in cases
        cm = nest_child_mesh(nest)
        p = _nh_set_cells!(Field(CellCenter(), pm), _nh_smooth); c = Field(CellCenter(), cm)
        interpolate_parent_to_child!(c, p, nest)
        Ip, Ic = _nh_box_integrals(p, c, nest)
        q = Field(CellCenter(), pm); q.data .= p.data .+ 1.0      # restriction overwrites the box
        restrict_to_parent!(q, c, nest)
        Iq, _ = _nh_box_integrals(q, c, nest)
        worst_box = max(worst_box, abs(Ic - Ip) / Ip, abs(Iq - Ic) / Ic)
        # flux preservation on the box planes: the parent u after the
        # interface flux correction carries the child's summed flux, and the
        # child u prolongated from a parent u has the parent's flux there up
        # to the block conservation of the tangential directions
        pu = Field(FaceX(), pm); _nh_set_cells!(pu, _nh_smooth)
        cu = Field(FaceX(), cm); interpolate_parent_to_child!(cu, pu, nest)
        mm = compute_nest_flux_mismatch(pu, cu, nest)
        S = maximum(abs, pu.data) * pm.dy * pm.dz * length(mm.west)     # flux scale of a plane
        # the summed mismatch of a plane vanishes (block conservation in y, z)
        worst_plane = max(worst_plane, abs(sum(mm.west)) / S, abs(sum(mm.east)) / S)
        apply_flux_correction!(pu, mm, nest)
        worst_corr = max(worst_corr, mismatch_l1(compute_nest_flux_mismatch(pu, cu, nest)) / S)
    end
    @test worst_box <= 1e-14
    @test worst_plane <= 1e-13 && worst_corr <= 1e-13
    # every parent face of the stretched child's overlap is a child face:
    # restriction undoes prolongation cell by cell
    nest = NestedGrid(pm, _nh_st_mesh())
    p = _nh_set_cells!(Field(CellCenter(), pm), _nh_smooth); c = Field(CellCenter(), nest_child_mesh(nest))
    interpolate_parent_to_child!(c, p, nest)
    q = Field(CellCenter(), pm); q.data .= p.data
    restrict_to_parent!(q, c, nest)
    e = maximum(abs, q.data .- p.data) / maximum(abs, p.data)
    @test e <= 1e-14
end

# Order of the prolongation on the overlap path: the child cell averages of
# a smooth field against the exact averages, at parent spacing Δ and Δ/2.
function _nh_prolong_error(N, r, order; stretched = false)
    L = 1.0
    pm = RegularGrid(Float64; size = (N, 4, 4), extent = (L, 4L / N, 4L / N), halo = (3, 3, 3))
    Δ = L / N
    # the child over the middle half in x, ratio r, origin a third of a parent cell off
    x0 = L / 4 + Δ / 3
    nc = round(Int, (L / 2) / (Δ / r))
    cm = RegularGrid(Float64; size = (nc, 4, 4), extent = (nc * Δ / r, 4L / N, 4L / N),
                     halo = (3, 3, 3), origin = (x0, 0.0, 0.0))
    nest = NestedGrid(pm, cm)
    k = 2π / L
    F(x) = -cos(k * x) / k                                          # primitive of sin(kx)
    p = Field(CellCenter(), pm)
    for i in axes(p.data, 1)
        a = (i - pm.Hx - 1) * Δ
        p.data[i, :, :] .= (F(a + Δ) - F(a)) / Δ
    end
    c = Field(CellCenter(), cm)
    interpolate_parent_to_child!(c, p, nest; order)
    err = 0.0
    for i in cm.Hx+1:cm.Hx+cm.Nx
        a = x0 + (i - cm.Hx - 1) * cm.dx
        err = max(err, abs(c.data[i, 5, 5] - (F(a + cm.dx) - F(a)) / cm.dx))
    end
    return err
end

@testset "prolongation order on the overlap path: rational ratios, origin off the faces" begin
    r, order = 5 // 2, 3
    e1 = _nh_prolong_error(24 * denominator(r), r, order)
    e2 = _nh_prolong_error(48 * denominator(r), r, order)
    @test log2(e1 / e2) >= order - 0.2
end

# A stretched child's z prolongation keeps its order: a field varying in z
# only, the child faces over the parent layers split 3, 2, 1, … .
function _nh_prolong_error_z(N, order)
    L = 1.0; Δ = L / N
    pm = RegularGrid(Float64; size = (4, 4, N), extent = (4Δ, 4Δ, L), halo = (3, 3, 3))
    K0 = N ÷ 4
    splits = Tuple(repeat([3, 2, 1, 2], N ÷ 8))
    zf = _nh_st_faces(K0 * Δ, splits, Δ)
    cm = StretchedGrid(Float64; size = (4, 4, length(zf) - 1), extent = (4Δ, 4Δ, zf[end] - zf[1]),
                       halo = (3, 3, 3), origin = (0.0, 0.0, zf[1]), stretching = (:custom, zf))
    nest = NestedGrid(pm, cm)
    k = 2π / L
    F(z) = -cos(k * z) / k
    p = Field(CellCenter(), pm)
    for kk in axes(p.data, 3)
        a = (kk - pm.Hz - 1) * Δ
        p.data[:, :, kk] .= (F(a + Δ) - F(a)) / Δ
    end
    c = Field(CellCenter(), cm)
    interpolate_parent_to_child!(c, p, nest; order)
    err = 0.0
    for kk in cm.Hz+1:cm.Hz+cm.Nz
        a = cm.zf[kk]; b = cm.zf[kk+1]
        err = max(err, abs(c.data[5, 5, kk] - (F(b) - F(a)) / (b - a)))
    end
    return err
end

@testset "prolongation order on a stretched child" begin
    order = 3
    e1 = _nh_prolong_error_z(32, order); e2 = _nh_prolong_error_z(64, order)
    @test log2(e1 / e2) >= order - 0.2
end

# -------------------------------------------------------------------------
@testset "aligned = non-aligned: the integer path is the overlap path's special case" begin
    pm = RegularGrid(Float64; size = (16, 16, 16), extent = (1600.0, 1600.0, 1600.0), halo = (3, 3, 3))
    n1 = nested_grid(pm, ((400.0, 1200.0), (400.0, 1200.0), (400.0, 1200.0)), 2)
    cm = nest_child_mesh(n1)
    n2 = NestedGrid(pm, cm)                        # inferred, aligned: the integer path
    n3 = NestedGrid(pm, cm; plan = :overlap)       # forced overlap path
    # a uniform StretchedGrid child: the overlap path in z
    n4 = NestedGrid(pm, StretchedGrid(cm))
    p = _nh_set_cells!(Field(CellCenter(), pm), _nh_smooth)
    pu = _nh_set_cells!(Field(FaceX(), pm), _nh_smooth)
    ref = Field(CellCenter(), cm); interpolate_parent_to_child!(ref, p, n1)
    refu = Field(FaceX(), cm); interpolate_parent_to_child!(refu, pu, n1)
    c2 = Field(CellCenter(), cm); interpolate_parent_to_child!(c2, p, n2)
    @test values_agree(c2.data, ref.data)                      #
    worst = 0.0
    for n in (n3, n4)
        cmn = nest_child_mesh(n)
        c = Field(CellCenter(), cmn); interpolate_parent_to_child!(c, p, n)
        cu = Field(FaceX(), cmn); interpolate_parent_to_child!(cu, pu, n)
        e1 = maximum(abs, c.data .- ref.data) / maximum(abs, ref.data)
        e2 = maximum(abs, cu.data .- refu.data) / maximum(abs, refu.data)
        q1 = Field(CellCenter(), pm); q1.data .= p.data; restrict_to_parent!(q1, ref, n1)
        qn = Field(CellCenter(), pm); qn.data .= p.data; restrict_to_parent!(qn, ref, n)
        e3 = maximum(abs, qn.data .- q1.data) / maximum(abs, q1.data)
        u1 = Field(FaceX(), pm); u1.data .= pu.data; restrict_to_parent!(u1, refu, n1)
        un = Field(FaceX(), pm); un.data .= pu.data; restrict_to_parent!(un, refu, n)
        e4 = maximum(abs, un.data .- u1.data) / maximum(abs, u1.data)
        worst = max(worst, e1, e2, e3, e4)
    end
    @test worst <= 1e-14
end

# -------------------------------------------------------------------------
# 3. Free stream. U = (5, 2, 0) m/s and θ = 300 K on every domain of the
#    three-level hierarchy must stay so to round-off (two-way, SSP-RK3).
@testset "free stream to round-off: H3, two-way" begin
    U = (5.0, 2.0, 0.0)
    h, sims = _nh_config(:H3; feedback = :two_way, core = step_rk3!, U)
    for _ in 1:5; step_nested!(h; dt = 4.0); end
    @test maximum(maximum(_nh_dev(s, U, 300.0)[1:3]) for s in sims) <= 1e-12
    @test maximum(_nh_dev(s, U, 300.0)[4] for s in sims) <= 1e-13
    @test maximum(_nh_div(s, U) for s in sims) <= 1e-12
    Q = 0.0
    for s in sims[2:end], st in Aeolus.BoundaryConditions.open_boundary_states(s)
        Q = max(Q, abs(Aeolus.BoundaryConditions.open_boundary_flux(st)) /
                   (hypot(U...) * 2 * (s.mesh.Lx * s.mesh.Ly + s.mesh.Lx * s.mesh.Lz + s.mesh.Ly * s.mesh.Lz)))
    end
    @test Q <= 1e-12
    @test all(s -> s.t ≈ h.t, sims)
end

# -------------------------------------------------------------------------
# 4. Composite conservation. A θ blob in the stream through every level of
#    H3; with refluxing at every link the hierarchy's composite ∫θ changes
#    only by round-off, without it it drifts (the negative control).
function _nh_conservation(cfg, core; reflux = true, nsteps = 10)
    h, sims = _nh_config(cfg; feedback = :two_way, core, blob = true, reflux)
    I0 = composite_integral(h, :θ)
    Ip = I0; per = 0.0
    for _ in 1:nsteps
        step_nested!(h; dt = 4.0)
        I = composite_integral(h, :θ)
        per = max(per, abs(I - Ip) / I0); Ip = I
    end
    return per, abs(Ip - I0) / I0
end

@testset "composite conservation of the H3 hierarchy, with and without refluxing" begin
    per, tot = _nh_conservation(:H3, step_rk3!)
    per0, _ = _nh_conservation(:H3, step_rk3!; reflux = false, nsteps = 5)
    @test per <= 1e-13 && tot <= 1e-12
    @test per0 > 100 * max(per, 1e-16)                 # the negative control
end

# -------------------------------------------------------------------------
# 6. Rollback per level: a NaN injected into one grandchild substep rolls
# back the grandchild only (the root and the child are the same as a
#    clean run; the grandchild's link records the retry); a bug in a core
#    (MethodError) propagates.
@testset "rollback per level; exceptions" begin
    function h3(inject)
        p = _nh_blob!(_nh_root())
        n1 = nested_grid(p.mesh, ((400.0, 1200.0), (400.0, 1200.0), (400.0, 1200.0)), 2)
        c = _nh_child(n1); _nh_blob!(c)
        n2 = nested_grid(c.mesh, ((600.0, 1000.0), (600.0, 1000.0), (600.0, 1000.0)), 2)
        g = _nh_child(n2); _nh_blob!(g)
        h = NestHierarchy(p; advance = domain_advance(step_rk3!), cfl = nothing)
        add_nest!(h, :root => c, n1; id = :child, advance = domain_advance(step_rk3!), cfl = nothing,
                  feedback = :two_way)
        gadv = inject === nothing ? domain_advance(step_rk3!) : domain_advance(inject)
        add_nest!(h, :child => g, n2; feedback = :one_way, id = :grand, advance = gadv, cfl = nothing)   # one-way
        return h, p, c, g
    end
    ha, pa, ca, ga = h3(nothing)
    hb, pb, cb, gb = h3(_NHInject(step_rk3!, 7, :nan))
    for _ in 1:3
        step_nested!(ha; dt = 4.0); step_nested!(hb; dt = 4.0)
    end
    @test values_agree(pa.θ.data, pb.θ.data) && values_agree(pa.u.data, pb.u.data)
    @test values_agree(ca.θ.data, cb.θ.data) && values_agree(ca.u.data, cb.u.data)
    @test all(isfinite, gb.θ.data)
    @test nest_link(hb, :grand).stats.child_retries == 1
    @test nest_link(hb, :child).stats.child_retries == 0 && nest_link(hb, :child).stats.parent_retries == 0
    @test nest_link(hb, :grand).stats.child_substeps == nest_link(ha, :grand).stats.child_substeps + 1
    hc, _, _, _ = h3(_NHInject(step_rk3!, 3, :method))
    @test_throws MethodError begin
        for _ in 1:3; step_nested!(hc; dt = 4.0); end
    end
end

# -------------------------------------------------------------------------
@testset "refusals" begin
    p = _nh_root()
    pm = p.mesh
    n1 = nested_grid(pm, ((400.0, 1000.0), (400.0, 1200.0), (400.0, 1200.0)), 2)
    # overlapping siblings
    h = NestHierarchy(p; cfl = nothing)
    add_nest!(h, :root => _nh_child(n1), n1; feedback = :one_way, id = :A, cfl = nothing)
    n2 = nested_grid(pm, ((900.0, 1400.0), (400.0, 1200.0), (400.0, 1200.0)), 2)
    err = try add_nest!(h, :root => _nh_child(n2), n2; feedback = :one_way, id = :B, cfl = nothing); nothing catch e; e end
    @test err isa ArgumentError && occursin("overlaps its sibling :A", err.msg)
    # two-way siblings with boxes less than one parent cell apart
    p2 = _nh_root(); h2 = NestHierarchy(p2; cfl = nothing)
    nA = nested_grid(p2.mesh, ((400.0, 1000.0), (400.0, 1200.0), (400.0, 1200.0)), 2)
    add_nest!(h2, :root => _nh_child(nA), nA; id = :A, cfl = nothing, feedback = :two_way)
    nB = nested_grid(p2.mesh, ((1000.0, 1400.0), (400.0, 1200.0), (400.0, 1200.0)), 2)
    err = try add_nest!(h2, :root => _nh_child(nB), nB; id = :B, cfl = nothing, feedback = :two_way); nothing catch e; e end
    @test err isa ArgumentError && occursin("less than one parent cell apart", err.msg)
    # … one-way they may touch
    @test add_nest!(h2, :root => _nh_child(nB), nB; feedback = :one_way, id = :B, cfl = nothing) === :B
    # a two-way child without a conservation box: r = 5/2 with no two
    # common faces inside (x extent 2 parent cells, origin off the faces)
    nn = nested_grid(pm, ((450.0, 650.0), (400.0, 1200.0), (400.0, 1200.0)), (5 // 2, 2, 2))
    @test isempty(nn.axes[1].pbox)
    p3 = _nh_root()
    err = try NestedRun(p3, _nh_child(nn), nn; feedback = :two_way); nothing catch e; e end
    @test err isa ArgumentError && occursin("conservation box", err.msg)
    @test NestedRun(_nh_root(), _nh_child(nn), nn; feedback = :one_way) isa NestedRun        # one-way is fine
    # a grandchild whose halo stencils reach the child's open boundary
    p4 = _nh_root(); h4 = NestHierarchy(p4; cfl = nothing)
    nc = nested_grid(p4.mesh, ((400.0, 1200.0), (400.0, 1200.0), (400.0, 1200.0)), 2)
    c4 = _nh_child(nc); add_nest!(h4, :root => c4, nc; feedback = :one_way, id = :child, cfl = nothing)
    ng = nested_grid(c4.mesh, ((400.0, 800.0), (600.0, 1000.0), (600.0, 1000.0)), 2)
    err = try add_nest!(h4, :child => _nh_child(ng), ng; feedback = :one_way, id = :g, cfl = nothing); nothing catch e; e end
    @test err isa ArgumentError && occursin("not properly nested", err.msg) && occursin("west", err.msg)
    @test !isempty(proper_nesting_violations(ng, c4, _nh_child(ng)))
    @test isempty(proper_nesting_violations(nc, p4, c4))         # a periodic root: no limit
    # a mixed-core two-way link
    p5 = _nh_root(); h5 = NestHierarchy(p5; cfl = nothing)
    na5 = nested_grid(p5.mesh, ((400.0, 1200.0), (400.0, 1200.0), (400.0, 1200.0)), 2)
    cm5 = nest_child_mesh(na5)
    an = AnelasticSimulation(cm5; ref = isothermal_reference(cm5; T_surf = 300.0),
                             bcs = nest_bcs(cm5, :nest))
    err = try add_nest!(h5, :root => an, na5; id = :an, cfl = nothing, feedback = :two_way); nothing catch e; e end
    @test err isa ArgumentError && occursin("same core family", err.msg)
    # a child coarser than its parent
    coarse = RegularGrid(Float64; size = (4, 4, 4), extent = (800.0, 800.0, 800.0), halo = (3, 3, 3),
                         origin = (400.0, 400.0, 400.0))
    err = try NestedGrid(pm, coarse); nothing catch e; e end
    @test err isa ErrorException && occursin("coarser than its parent", err.msg)
end

end # testset
