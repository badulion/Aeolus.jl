using Test
using Printf
using Random
using TOML
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
#   3. free stream to round-off: three levels (H3), siblings (S2), a
#      non-aligned child (NA), a stretched child (ST), one- and two-way;
#   4. composite conservation of the hierarchy (refluxing at every level,
#      with the negative control), sibling order, NA and ST;
#   5. nested against a single fine-domain reference, and convergence;
#   6. rollback per level, refusals, sync = :shared_min_dt;
#   7. the golden NestedRun checksums (bit for bit after the refactor).
# =========================================================================

const _NH_DIR = @__DIR__

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

# The four configurations of the design (#328 §5): H3 (three levels), S2
# (two siblings, r = 2 and r = 3, two parent cells apart), NA (r = 5/2 in x,
# the child's x edges 0.4 and 0.8 parent cells off the parent faces: the
# box shrinks to the common faces 400 … 1000 m), ST (a z-stretched child,
# ratio 3 → 1). Returns (hierarchy, sims).
function _nh_config(cfg::Symbol; feedback = :one_way, core = step!, blob = false, U = (5.0, 2.0, 0.0),
                    reflux = true, order_siblings = (1, 2))
    fb = feedback === :two_way ? TwoWayFeedback(; reflux) : :one_way
    p = _nh_root(; U)
    blob && _nh_blob!(p)
    adv = domain_advance(core)
    h = NestHierarchy(p; advance = adv, cfl = nothing)
    sims = Any[p]
    if cfg === :H3
        n1 = nested_grid(p.mesh, ((400.0, 1200.0), (400.0, 1200.0), (400.0, 1200.0)), 2)
        c = _nh_child(n1; U); blob && _nh_blob!(c)
        add_nest!(h, :root => c, n1; id = :child, advance = adv, cfl = nothing, feedback = fb)
        n2 = nested_grid(c.mesh, ((600.0, 1000.0), (600.0, 1000.0), (600.0, 1000.0)), 2)
        g = _nh_child(n2; U); blob && _nh_blob!(g)
        add_nest!(h, :child => g, n2; id = :grand, advance = adv, cfl = nothing, feedback = fb)
        push!(sims, c, g)
    elseif cfg === :S2
        nA = nested_grid(p.mesh, ((200.0, 600.0), (400.0, 1200.0), (400.0, 1200.0)), 2)
        nB = nested_grid(p.mesh, ((800.0, 1400.0), (400.0, 1100.0), (300.0, 1200.0)), 3)
        a = _nh_child(nA; U); b = _nh_child(nB; U)
        blob && (_nh_blob!(a); _nh_blob!(b))
        specs = ((:A, a, nA), (:B, b, nB))
        for i in order_siblings
            id, s, n = specs[i]
            add_nest!(h, :root => s, n; id, advance = adv, cfl = nothing, feedback = fb)
        end
        push!(sims, a, b)
    elseif cfg === :NA
        n = nested_grid(p.mesh, ((360.0, 1080.0), (400.0, 1200.0), (400.0, 1200.0)), (5 // 2, 2, 2))
        c = _nh_child(n; U); blob && _nh_blob!(c)
        add_nest!(h, :root => c, n; id = :child, advance = adv, cfl = nothing, feedback = fb)
        push!(sims, c)
    elseif cfg === :ST
        cm = _nh_st_mesh()
        n = NestedGrid(p.mesh, cm)
        c = _nh_child(n; U); blob && _nh_blob!(c)
        add_nest!(h, :root => c, n; id = :child, advance = adv, cfl = nothing, feedback = fb)
        push!(sims, c)
    end
    return h, sims
end

# Max relative deviation of a free stream.
_nh_dev(s, U, θ0) = (maximum(abs, interior_view(s.u) .- U[1]) / hypot(U...),
                     maximum(abs, interior_view(s.v) .- U[2]) / hypot(U...),
                     maximum(abs, interior_view(s.w) .- U[3]) / hypot(U...),
                     maximum(abs, interior_view(s.θ) .- θ0) / θ0)

# Max |∇·u| Δ / |U| on the interior of a sim (stretched-aware: max_divergence).
_nh_div(s, U) = max_divergence(s) * min(s.mesh.dx, s.mesh.dy) / hypot(U...)

# The grandchild's region of the quasi-1-D reference case (section 5).
const _NH_R = (2800.0, 3600.0)

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

# A tracer moved by a separate transport after each step (forward-Euler
# WENO-Z5, the Berlin slot pattern), which books its own interface flux
# (book_nest_flux!): in a hierarchy the middle domain books into the link
# above it and the link below it.
const _NH_WENO = advection_scheme(:weno_z5)
struct _NHSlot{F}
    f::F
end
function (c::_NHSlot)(sim, dt; kw...)
    step!(sim, dt; kw...)
    fill_halos!(sim.u, sim.bcs); fill_halos!(sim.v, sim.bcs); fill_halos!(sim.w, sim.bcs)
    fill_halos!(c.f, sim.θ_bcs)
    book_nest_flux!(sim, :co2, c.f, dt; scheme = :weno_z5)
    t = Field(CellCenter(), sim.mesh)
    _NH_WENO(t, sim.u, sim.v, sim.w, c.f)
    interior_view(c.f) .+= dt .* interior_view(t)
    return sim
end
Aeolus.Nesting.nest_stage_weights(::_NHSlot, args...) = (1.0,)

@testset "Era II — nest hierarchy, non-aligned and stretched children (#328)" begin

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
    for n in 1:8
        @test sum(a.len[a.ptr[n]:a.ptr[n+1]-1]) ≈ 30.0 rtol = 1e-15
    end
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
    for (name, nest) in cases
        cm = nest_child_mesh(nest)
        p = _nh_set_cells!(Field(CellCenter(), pm), _nh_smooth); c = Field(CellCenter(), cm)
        interpolate_parent_to_child!(c, p, nest)
        Ip, Ic = _nh_box_integrals(p, c, nest)
        q = Field(CellCenter(), pm); q.data .= p.data .+ 1.0      # restriction overwrites the box
        restrict_to_parent!(q, c, nest)
        Iq, _ = _nh_box_integrals(q, c, nest)
        @printf("  %-18s box ∫: prolongation %.1e, restriction %.1e (relative)\n", name,
                abs(Ic - Ip) / Ip, abs(Iq - Ic) / Ic)
        @test abs(Ic - Ip) <= 1e-14 * Ip
        @test abs(Iq - Ic) <= 1e-14 * Ic
        # flux preservation on the box planes: the parent u after the
        # interface flux correction carries the child's summed flux, and the
        # child u prolongated from a parent u has the parent's flux there up
        # to the block conservation of the tangential directions
        pu = Field(FaceX(), pm); _nh_set_cells!(pu, _nh_smooth)
        cu = Field(FaceX(), cm); interpolate_parent_to_child!(cu, pu, nest)
        mm = compute_nest_flux_mismatch(pu, cu, nest)
        S = maximum(abs, pu.data) * pm.dy * pm.dz * length(mm.west)     # flux scale of a plane
        # the summed mismatch of a plane vanishes (block conservation in y, z)
        @test abs(sum(mm.west)) <= 1e-13 * S && abs(sum(mm.east)) <= 1e-13 * S
        apply_flux_correction!(pu, mm, nest)
        @test mismatch_l1(compute_nest_flux_mismatch(pu, cu, nest)) <= 1e-13 * S
    end
    # every parent face of the stretched child's overlap is a child face:
    # restriction undoes prolongation cell by cell
    nest = NestedGrid(pm, _nh_st_mesh())
    p = _nh_set_cells!(Field(CellCenter(), pm), _nh_smooth); c = Field(CellCenter(), nest_child_mesh(nest))
    interpolate_parent_to_child!(c, p, nest)
    q = Field(CellCenter(), pm); q.data .= p.data
    restrict_to_parent!(q, c, nest)
    e = maximum(abs, q.data .- p.data) / maximum(abs, p.data)
    @printf("  stretched child: max |R P φ − φ| / max|φ| = %.1e\n", e)
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
    for r in (5 // 2, 3 // 2, 7 // 3), order in (2, 3, 4)
        e1 = _nh_prolong_error(24 * denominator(r), r, order)
        e2 = _nh_prolong_error(48 * denominator(r), r, order)
        ob = log2(e1 / e2)
        @printf("  r = %-5s order %d: error %.2e → %.2e, observed order %.2f\n", string(r), order, e1, e2, ob)
        @test ob >= order - 0.2
    end
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
    for order in (2, 3, 4)
        e1 = _nh_prolong_error_z(32, order); e2 = _nh_prolong_error_z(64, order)
        ob = log2(e1 / e2)
        @printf("  stretched child, order %d: error %.2e → %.2e, observed order %.2f\n", order, e1, e2, ob)
        @test ob >= order - 0.2
    end
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
    @test c2.data == ref.data                      # bit for bit
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
    @printf("  overlap path vs integer path (forced overlap; uniform StretchedGrid child): max relative difference %.1e\n", worst)
    @test worst <= 1e-14
    # a 20-step two-way run: integer path against the overlap path
    function run20(nest)
        p = _nh_blob!(_nh_root()); c = _nh_child(nest); _nh_blob!(c)
        run = NestedRun(p, c, nest; feedback = :two_way, cfl_parent = nothing, cfl_child = nothing,
                        advance_parent = domain_advance(step_rk3!), advance_child = domain_advance(step_rk3!))
        for _ in 1:20; step_nested!(run; dt = 4.0); end
        return p, c
    end
    pa, ca = run20(nested_grid(pm, ((400.0, 1200.0), (400.0, 1200.0), (400.0, 1200.0)), 2))
    pb, cb = run20(NestedGrid(pm, nest_child_mesh(nested_grid(pm, ((400.0, 1200.0), (400.0, 1200.0),
                                                                   (400.0, 1200.0)), 2)); plan = :overlap))
    d = max(maximum(abs, pa.θ.data .- pb.θ.data) / 300, maximum(abs, ca.θ.data .- cb.θ.data) / 300,
            maximum(abs, pa.u.data .- pb.u.data) / 5, maximum(abs, ca.u.data .- cb.u.data) / 5)
    @printf("  20 two-way RK3 steps, integer vs overlap path: max relative difference %.1e\n", d)
    @test d <= 1e-12
end

# -------------------------------------------------------------------------
@testset "stretched child: open-boundary mass flux, divergence, flux on the box planes" begin
    pm = RegularGrid(Float64; size = (16, 16, 16), extent = (1600.0, 1600.0, 1600.0), halo = (3, 3, 3))
    p = _nh_root(; U = (5.0, 2.0, 0.0))
    rng = Xoshiro(5)
    interior_view(p.u) .+= 0.5 .* randn(rng, size(interior_view(p.u)))
    interior_view(p.w) .+= 0.5 .* randn(rng, size(interior_view(p.w)))
    Aeolus.Simulations._project_velocity!(p)
    nest = NestedGrid(pm, _nh_st_mesh())
    c = _nh_child(nest)
    for (cf, pf) in ((c.u, p.u), (c.v, p.v), (c.w, p.w))
        interpolate_parent_to_child!(cf, pf, nest)
    end
    run = NestedRun(p, c, nest; cfl_parent = nothing, cfl_child = nothing)
    worst_div = 0.0; worst_Q = 0.0
    for _ in 1:10
        step_nested!(run; dt = 4.0)
        worst_div = max(worst_div, max_divergence(c))
        Q = sum(abs(Aeolus.BoundaryConditions.open_boundary_flux(st)) for st in
                Aeolus.BoundaryConditions.open_boundary_states(c))
        worst_Q = max(worst_Q, Q)
    end
    A = 6 * 800.0^2
    @printf("  stretched child, 10 one-way steps: net open-boundary flux %.1e |U|A, max |∇·u| %.1e 1/s\n",
            worst_Q / (5.4 * A), worst_div)
    @test worst_Q <= 1e-12 * 5.4 * A
    @test worst_div <= 1e-11
end

# -------------------------------------------------------------------------
# 3. Free stream. U = (5, 2, 0) m/s and θ = 300 K on every domain must stay
#    so to round-off, with both cores, one-way and two-way.
@testset "free stream to round-off: H3, S2, NA, ST, one- and two-way" begin
    U = (5.0, 2.0, 0.0)
    for cfg in (:H3, :S2, :NA, :ST), fb in (:one_way, :two_way), core in (step!, step_rk3!)
        h, sims = _nh_config(cfg; feedback = fb, core, U)
        for _ in 1:20; step_nested!(h; dt = 4.0); end
        dev = maximum(maximum(_nh_dev(s, U, 300.0)[1:3]) for s in sims)
        devθ = maximum(_nh_dev(s, U, 300.0)[4] for s in sims)
        dv = maximum(_nh_div(s, U) for s in sims)
        Q = 0.0
        for s in sims[2:end], st in Aeolus.BoundaryConditions.open_boundary_states(s)
            Q = max(Q, abs(Aeolus.BoundaryConditions.open_boundary_flux(st)) /
                       (hypot(U...) * 2 * (s.mesh.Lx * s.mesh.Ly + s.mesh.Lx * s.mesh.Lz + s.mesh.Ly * s.mesh.Lz)))
        end
        @printf("  %s %-7s %-9s: max |u−U|/|U| %.1e, |θ−θ₀|/θ₀ %.1e, |∇·u|Δ/|U| %.1e, net open flux %.1e |U|A\n",
                cfg, fb, core, dev, devθ, dv, Q)
        @test dev <= 1e-12
        @test devθ <= 1e-13
        @test dv <= 1e-12
        @test Q <= 1e-12
        @test all(s -> s.t ≈ h.t, sims)
    end
end

# -------------------------------------------------------------------------
# 4. Composite conservation. A θ blob in the stream through every level
#    (the H3 grandchild, both S2 siblings, the NA and ST children); with
#    refluxing at every link the hierarchy's composite ∫θ changes only by
#    round-off, without it it drifts (the negative control).
function _nh_conservation(cfg, core; reflux = true, nsteps = 40, order_siblings = (1, 2))
    h, sims = _nh_config(cfg; feedback = :two_way, core, blob = true, reflux, order_siblings)
    I0 = composite_integral(h, :θ)
    A0 = sum(abs, interior_view(sims[1].θ) .- 300.0) * 100.0^3           # blob scale
    Ip = I0; per = 0.0
    for _ in 1:nsteps
        step_nested!(h; dt = 4.0)
        I = composite_integral(h, :θ)
        per = max(per, abs(I - Ip) / I0); Ip = I
    end
    return per, abs(Ip - I0) / I0, h, sims
end

@testset "composite conservation of the hierarchy: H3, S2, NA, ST, with and without refluxing" begin
    for cfg in (:H3, :S2, :NA, :ST), core in (step!, step_rk3!)
        per, tot, h, _ = _nh_conservation(cfg, core)
        per0, tot0, _, _ = _nh_conservation(cfg, core; reflux = false, nsteps = 10)
        @printf("  %s %-9s: composite ∫θ per step %.1e, after 40 steps %.1e | without refluxing per step %.1e\n",
                cfg, core, per, tot, per0)
        @test per <= 1e-13
        @test tot <= 1e-12
        @test per0 > 100 * max(per, 1e-16)                 # the negative control
    end
end

@testset "sibling order does not change the result (bit for bit)" begin
    _, _, h1, s1 = _nh_conservation(:S2, step_rk3!; nsteps = 15, order_siblings = (1, 2))
    _, _, h2, s2 = _nh_conservation(:S2, step_rk3!; nsteps = 15, order_siblings = (2, 1))
    same = all(i -> s1[i].θ.data == s2[i].θ.data && s1[i].u.data == s2[i].u.data &&
                    s1[i].v.data == s2[i].v.data && s1[i].w.data == s2[i].w.data, 1:3)
    @test same
end

@testset "hierarchy conservation of qv and of a tracer booked by its transport (three levels)" begin
    U = (5.0, 2.0, 0.0)
    p = _nh_root(; U, moisture = true); _nh_blob!(p)
    n1 = nested_grid(p.mesh, ((400.0, 1200.0), (400.0, 1200.0), (400.0, 1200.0)), 2)
    c = _nh_child(n1; U, moisture = true, scalars = (:co2,)); _nh_blob!(c)
    n2 = nested_grid(c.mesh, ((600.0, 1000.0), (600.0, 1000.0), (600.0, 1000.0)), 2)
    g = _nh_child(n2; U, moisture = true, scalars = (:co2,)); _nh_blob!(g)
    tr = [Field(CellCenter(), s.mesh) for s in (p, c, g)]
    for (s, f) in zip((p, c, g), tr)
        m = s.mesh
        for k in axes(f.data, 3), j in axes(f.data, 2), i in axes(f.data, 1)
            x = m.x0 + (i - m.Hx - 0.5) * m.dx; y = m.y0 + (j - m.Hy - 0.5) * m.dy
            z = m.z0 + (k - m.Hz - 0.5) * m.dz
            f.data[i, j, k] = 400 + 50 * exp(-((x - 750)^2 + (y - 800)^2 + (z - 800)^2) / 150^2)
            s.qv.data[i, j, k] = 0.01 * (1 + exp(-((x - 850)^2 + (y - 800)^2 + (z - 800)^2) / 150^2))
        end
    end
    fb = TwoWayFeedback(tracer_fluxes = (:co2 => NestScalarFlux(applied_by = :transport),))
    h = NestHierarchy(p; advance = domain_advance(_NHSlot(tr[1])), cfl = nothing)
    add_nest!(h, :root => c, n1; id = :child, advance = domain_advance(_NHSlot(tr[2])), cfl = nothing,
              feedback = fb, scalars = (:co2 => (tr[2], tr[1]),))
    add_nest!(h, :child => g, n2; id = :grand, advance = domain_advance(_NHSlot(tr[3])), cfl = nothing,
              feedback = fb, scalars = (:co2 => (tr[3], tr[2]),))
    J0 = Dict(q => composite_integral(h, q) for q in (:θ, :qv, :co2))
    Jp = copy(J0); per = Dict(q => 0.0 for q in keys(J0))
    for _ in 1:30
        step_nested!(h; dt = 4.0)
        for q in keys(J0)
            J = composite_integral(h, q)
            per[q] = max(per[q], abs(J - Jp[q]) / J0[q]); Jp[q] = J
        end
    end
    for q in (:θ, :qv, :co2)
        @printf("  H3 two-way, moist, WENO tracer slot: composite ∫%s per step %.1e, after 30 steps %.1e\n",
                q, per[q], abs(Jp[q] - J0[q]) / J0[q])
        @test per[q] <= 1e-13
        @test abs(Jp[q] - J0[q]) <= 1e-12 * J0[q]
    end
end

@testset "anelastic pair with a rational ratio: composite ∫ρ̄θ in the mass measure" begin
    pm = RegularGrid(Float64; size = (12, 12, 12), extent = (1200.0, 1200.0, 1200.0), halo = (3, 3, 3))
    nest = nested_grid(pm, ((300.0, 900.0), (300.0, 900.0), (0.0, 600.0)), (5 // 2, 5 // 2, 2))
    @test !nest.aligned && nest_covered_range(nest) == (4:9, 4:9, 1:6)
    cm = nest_child_mesh(nest)
    rp = isothermal_reference(pm; T_surf = 300.0); rc = isothermal_reference(cm; T_surf = 300.0)
    pA = AnelasticSimulation(pm; ref = rp, bcs = FieldBoundaryConditions(:periodic, :periodic, :wall))
    cA = AnelasticSimulation(cm; ref = rc, bcs = nest_bcs(cm, :nest, :nest; bottom = :wall, top = :nest))
    rng = Xoshiro(3)
    for k in axes(pA.θ.data, 3), j in axes(pA.θ.data, 2), i in axes(pA.θ.data, 1)
        pA.θ.data[i, j, k] = rp.θ̄c_pad[k] + 0.5 * exp(-((i - 7)^2 + (j - 7)^2 + (k - 5)^2) / 5)
    end
    fill!(pA.u.data, 4.0)
    interior_view(pA.v) .+= 0.3 .* randn(rng, size(interior_view(pA.v)))
    Aeolus.Anelastic._project_mass_weighted!(pA)
    meas = nest_measure(nest, pA, cA)
    for (cf, pf) in ((cA.θ, pA.θ), (cA.u, pA.u), (cA.v, pA.v), (cA.w, pA.w))
        interpolate_parent_to_child!(cf, pf, nest; measure = meas)
    end
    run = NestedRun(pA, cA, nest; feedback = :two_way, cfl_parent = nothing, cfl_child = nothing)
    I0 = composite_integral(pA.θ, cA.θ, nest; measure = meas)
    for _ in 1:20
        step_nested!(run; dt = 5.0)
    end
    d = abs(composite_integral(pA.θ, cA.θ, nest; measure = meas) - I0) / I0
    @printf("  anelastic two-way, r = 5/2 horizontally, 20 steps: composite ∫ρ̄θ drift %.2e\n", d)
    @test d <= 1e-12
end

# -------------------------------------------------------------------------
# 5. A nested run against a single fine-domain reference. A quasi-1-D case:
#    a periodic channel in x (6.4 km, Δ = 100 m; y and z periodic, two
#    cells), a Gaussian θ blob advected by U = 5 m/s through nests that span
#    the parent in y and z, so only their x faces are open: the child over
#    [2400, 4000] m (r = 2), the grandchild over R = [2800, 3600] m (r = 2
#    again). SSP-RK3, centred advection, ν = 2 m²/s; every domain starts
#    from the exact profile. Errors are RMS over R against the exact
#    solution (advected and diffused).

# The exact solution: the Gaussian (centre x₀ at t = 0) advected at
# U = 5 m/s and spread by ν (σ_t² = σ² + 2νt), on the periodic channel L.
function _nh_blob1d(x, t, L, σ, ν, x0)
    d = mod(x - x0 - 5.0 * t + L / 2, L) - L / 2
    σt = sqrt(σ^2 + 2ν * t)
    return σ / σt * exp(-d^2 / (2 * σt^2))
end

function _nh_err1d(s, t, L, σ, ν, x0)
    m = s.mesh
    xs = [(i, m.x0 + (i - m.Hx - 0.5) * m.dx) for i in m.Hx+1:m.Hx+m.Nx]
    sel = [(i, x) for (i, x) in xs if _NH_R[1] <= x <= _NH_R[2]]
    return sqrt(sum((s.θ.data[i, m.Hy+1, m.Hz+1] - 300 - _nh_blob1d(x, t, L, σ, ν, x0))^2
                    for (i, x) in sel) / length(sel))
end

function _nh_1d(; N = 64, L = 6400.0, levels = ((2400.0, 4000.0, 2),), feedback = :one_way,
                T = 100.0, dt = 10.0 * 64 / N, σ = 100.0, ν = 2.0, x0 = 2600.0, outflow = :dirichlet)
    Δ = L / N
    mk(m, bcs) = (s = BoussinesqSimulation(m; ν, bcs); fill!(s.u.data, 5.0); s)
    setθ!(s) = (m = s.mesh; for i in axes(s.θ.data, 1)
                    s.θ.data[i, :, :] .= 300 + _nh_blob1d(m.x0 + (i - m.Hx - 0.5) * m.dx, 0.0, L, σ, ν, x0)
                end; s)
    pm = RegularGrid(Float64; size = (N, 2, 2), extent = (L, 2Δ, 2Δ), halo = (3, 3, 3))
    p = setθ!(mk(pm, FieldBoundaryConditions(:periodic)))
    if isempty(levels)                          # the uniform run
        for _ in 1:round(Int, T / dt); step_rk3!(p, dt); end
        return nothing, Any[p], [_nh_err1d(p, p.t, L, σ, ν, x0)]
    end
    h = NestHierarchy(p; advance = domain_advance(step_rk3!), cfl = nothing)
    parent = :root; pmesh = pm; sims = Any[p]
    for (n, (a, b, r)) in enumerate(levels)
        nest = nested_grid(pmesh, ((a, b), (0.0, pmesh.Ly), (0.0, pmesh.Lz)), (r, 1, 1))
        cm = nest_child_mesh(nest)
        c = setθ!(mk(cm, nest_bcs(cm, :nest, :periodic, :periodic; outflow)))
        id = Symbol(:L, n)
        add_nest!(h, parent => c, nest; id, advance = domain_advance(step_rk3!), cfl = nothing,
                  feedback, substeps = r)
        parent = id; pmesh = cm; push!(sims, c)
    end
    for _ in 1:round(Int, T / dt); step_nested!(h; dt); end
    errs = [s.mesh.x0 <= _NH_R[1] && s.mesh.x0 + s.mesh.Lx >= _NH_R[2] ? _nh_err1d(s, h.t, L, σ, ν, x0) : NaN
            for s in sims]
    return h, sims, errs
end

@testset "nested run against a fine reference; convergence" begin
    H3 = ((2400.0, 4000.0, 2), (2800.0, 3600.0, 2))
    # (a) σ = 100 m = 1 root cell: the blob (centre 2600 m) crosses the child
    #     and enters the grandchild; at t = 100 s its centre is at 3100 m
    _, _, ec = _nh_1d(; levels = ())                                      # coarse only, Δ = 100 m
    _, _, ef = _nh_1d(; N = 256, levels = ())                             # uniform Δ = 25 m
    for fb in (:one_way, :two_way)
        _, _, e2 = _nh_1d(; feedback = fb)
        _, _, e3 = _nh_1d(; levels = H3, feedback = fb)
        @printf("  %s, t = 100 s, RMS error over R: coarse only %.2e, uniform Δ = 25 m %.2e | child r = 2 %.2e | H3 child %.2e, grandchild %.2e\n",
                fb, ec[1], ef[1], e2[2], e3[2], e3[3])
        @test e2[2] <= 0.5 * ec[1]
        @test e3[3] <= 0.5 * ec[1]
    end
    # (b) the extra interface costs little: a blob starting inside the
    #     grandchild (centre 3000 m, t = 60 s) against a two-level run whose
    #     child has the grandchild's resolution (r = 4) over the child's region
    _, _, eb = _nh_1d(; levels = (), x0 = 3000.0, T = 60.0)
    for fb in (:one_way, :two_way)
        _, _, e3 = _nh_1d(; levels = H3, feedback = fb, x0 = 3000.0, T = 60.0)
        _, _, e4 = _nh_1d(; levels = ((2400.0, 4000.0, 4),), feedback = fb, x0 = 3000.0, T = 60.0)
        @printf("  %s, blob inside the grandchild, t = 60 s: coarse only %.2e | H3 grandchild %.2e | two-level child at r = 4 %.2e (ratio %.2f)\n",
                fb, eb[1], e3[3], e4[2], e3[3] / e4[2])
        @test e3[3] <= 1.2 * e4[2]
    end
    # (c) the residual after the blob has left the nests (σ = 300 m,
    #     resolved by the root; t = 1000 s: centre at 1200 m). The root's own
    #     dispersive wake over the nests' region is the floor.
    _, sc, _ = _nh_1d(; levels = (), T = 1000.0, σ = 300.0)
    pc = sc[1]; mc = pc.mesh
    wake = maximum(abs(pc.θ.data[i, mc.Hy+1, mc.Hz+1] - 300) for i in mc.Hx+1:mc.Hx+mc.Nx
                   if 2400 <= mc.x0 + (i - mc.Hx - 0.5) * mc.dx <= 4000)
    resid(fb, of) = (s = _nh_1d(; levels = H3, feedback = fb, T = 1000.0, σ = 300.0, outflow = of)[2];
                     maximum(maximum(abs, interior_view(c.θ) .- 300.0) for c in s[2:end]))
    r2 = resid(:two_way, :dirichlet); r1c = resid(:one_way, :convective); r1d = resid(:one_way, :dirichlet)
    @printf("  t = 1000 s, blob outside the nests: max |θ − 300| in the nests: two-way %.2e, one-way with convective outflow %.2e, one-way with Dirichlet outflow %.2e K (coarse-only run over the same region %.2e K)\n",
            r2, r1c, r1d, wake)
    @test r2 <= 1e-2
    @test r1c <= 1e-2
    # A one-way child with Dirichlet outflow faces receives the parent's
    # coarse blob at its outflow face while its own fine blob leaves: the
    # mismatch is reflected (centred advection) and stays in the nests at
    # ~5× the root's wake. The 1e-2 K target is not met there (#328 PR);
    # the radiation outflow above meets it.
    @test r1d <= 0.1
    # (d) convergence: every domain refined by 2 and 4 (ratios and blob fixed)
    es = [(_nh_1d(; N, levels = H3, feedback = :two_way, σ = 200.0)[3])[3] for N in (32, 64, 128)]
    o1 = log2(es[1] / es[2]); o2 = log2(es[2] / es[3])
    @printf("  H3 two-way convergence (grandchild RMS error, root N = 32, 64, 128): %.2e, %.2e, %.2e; observed order %.2f, %.2f\n",
            es..., o1, o2)
    @test o2 >= 1.8
end

# -------------------------------------------------------------------------
# 6. Rollback per level: a NaN injected into one grandchild substep rolls
#    back the grandchild only (the root and the child are bit-identical to a
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
        add_nest!(h, :child => g, n2; id = :grand, advance = gadv, cfl = nothing)   # one-way
        return h, p, c, g
    end
    ha, pa, ca, ga = h3(nothing)
    hb, pb, cb, gb = h3(_NHInject(step_rk3!, 7, :nan))
    for _ in 1:5
        step_nested!(ha; dt = 4.0); step_nested!(hb; dt = 4.0)
    end
    @test pa.θ.data == pb.θ.data && pa.u.data == pb.u.data
    @test ca.θ.data == cb.θ.data && ca.u.data == cb.u.data
    @test all(isfinite, gb.θ.data)
    @test nest_link(hb, :grand).stats.child_retries == 1
    @test nest_link(hb, :child).stats.child_retries == 0 && nest_link(hb, :child).stats.parent_retries == 0
    @test nest_link(hb, :grand).stats.child_substeps == nest_link(ha, :grand).stats.child_substeps + 1
    @printf("  NaN in grandchild substep 7: root and child bit-identical to the clean run; grandchild retries %d, substeps %d (clean %d)\n",
            nest_link(hb, :grand).stats.child_retries, nest_link(hb, :grand).stats.child_substeps,
            nest_link(ha, :grand).stats.child_substeps)
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
    add_nest!(h, :root => _nh_child(n1), n1; id = :A, cfl = nothing)
    n2 = nested_grid(pm, ((900.0, 1400.0), (400.0, 1200.0), (400.0, 1200.0)), 2)
    err = try add_nest!(h, :root => _nh_child(n2), n2; id = :B, cfl = nothing); nothing catch e; e end
    @test err isa ArgumentError && occursin("overlaps its sibling :A", err.msg)
    # two-way siblings with boxes less than one parent cell apart
    p2 = _nh_root(); h2 = NestHierarchy(p2; cfl = nothing)
    nA = nested_grid(p2.mesh, ((400.0, 1000.0), (400.0, 1200.0), (400.0, 1200.0)), 2)
    add_nest!(h2, :root => _nh_child(nA), nA; id = :A, cfl = nothing, feedback = :two_way)
    nB = nested_grid(p2.mesh, ((1000.0, 1400.0), (400.0, 1200.0), (400.0, 1200.0)), 2)
    err = try add_nest!(h2, :root => _nh_child(nB), nB; id = :B, cfl = nothing, feedback = :two_way); nothing catch e; e end
    @test err isa ArgumentError && occursin("less than one parent cell apart", err.msg)
    # … one-way they may touch
    @test add_nest!(h2, :root => _nh_child(nB), nB; id = :B, cfl = nothing) === :B
    # a two-way child without a conservation box: r = 5/2 with no two
    # common faces inside (x extent 2 parent cells, origin off the faces)
    nn = nested_grid(pm, ((450.0, 650.0), (400.0, 1200.0), (400.0, 1200.0)), (5 // 2, 2, 2))
    @test isempty(nn.axes[1].pbox)
    p3 = _nh_root()
    err = try NestedRun(p3, _nh_child(nn), nn; feedback = :two_way); nothing catch e; e end
    @test err isa ArgumentError && occursin("conservation box", err.msg)
    @test NestedRun(_nh_root(), _nh_child(nn), nn) isa NestedRun        # one-way is fine
    # a grandchild whose halo stencils reach the child's open boundary
    p4 = _nh_root(); h4 = NestHierarchy(p4; cfl = nothing)
    nc = nested_grid(p4.mesh, ((400.0, 1200.0), (400.0, 1200.0), (400.0, 1200.0)), 2)
    c4 = _nh_child(nc); add_nest!(h4, :root => c4, nc; id = :child, cfl = nothing)
    ng = nested_grid(c4.mesh, ((400.0, 800.0), (600.0, 1000.0), (600.0, 1000.0)), 2)
    err = try add_nest!(h4, :child => _nh_child(ng), ng; id = :g, cfl = nothing); nothing catch e; e end
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

# -------------------------------------------------------------------------
@testset "adaptive per-level steps; sync = :shared_min_dt" begin
    # adaptive: each level picks its own CFL step and lands on its parent's
    p = _nh_blob!(_nh_root())
    n1 = nested_grid(p.mesh, ((400.0, 1200.0), (400.0, 1200.0), (400.0, 1200.0)), 2)
    c = _nh_child(n1); _nh_blob!(c)
    n2 = nested_grid(c.mesh, ((600.0, 1000.0), (600.0, 1000.0), (600.0, 1000.0)), 2)
    g = _nh_child(n2); _nh_blob!(g)
    cfl = CFLConstraints(C_advective = 0.5)
    h = NestHierarchy(p; advance = domain_advance(step_rk3!), cfl)
    add_nest!(h, :root => c, n1; id = :child, advance = domain_advance(step_rk3!), feedback = :two_way)
    add_nest!(h, :child => g, n2; id = :grand, advance = domain_advance(step_rk3!), feedback = :two_way)
    I0 = composite_integral(h, :θ)
    for _ in 1:10; step_nested!(h); end
    sc = nest_link(h, :child).stats; sg = nest_link(h, :grand).stats
    @printf("  adaptive H3: root dt %.3f s, child substeps per root step %d–%d (CFL %.2f), grandchild per child step %d–%d (CFL %.2f); composite ∫θ drift %.1e\n",
            sc.dt_parent_max, sc.n_min, sc.n_max, sc.max_child_cfl, sg.n_min, sg.n_max, sg.max_child_cfl,
            abs(composite_integral(h, :θ) - I0) / I0)
    @test sc.n_min >= 2 && sg.n_min >= 2
    @test sc.max_child_cfl <= 0.5 + 1e-9 && sg.max_child_cfl <= 0.5 + 1e-9
    @test c.t == p.t && g.t == p.t
    @test abs(composite_integral(h, :θ) - I0) <= 1e-12 * I0
    # shared minimum dt: every domain one step of the smallest limit
    p2 = _nh_root(); c2 = _nh_child(n1)
    g2 = _nh_child(nested_grid(c2.mesh, ((600.0, 1000.0), (600.0, 1000.0), (600.0, 1000.0)), 2))
    h2 = NestHierarchy(p2; advance = domain_advance(step_rk3!), cfl, sync = :shared_min_dt)
    add_nest!(h2, :root => c2, n1; id = :child, advance = domain_advance(step_rk3!))
    add_nest!(h2, :child => g2, nested_grid(c2.mesh, ((600.0, 1000.0), (600.0, 1000.0), (600.0, 1000.0)), 2);
              id = :grand, advance = domain_advance(step_rk3!))
    for _ in 1:3; step_nested!(h2); end
    @test nest_link(h2, :child).stats.child_substeps == 3 && nest_link(h2, :grand).stats.child_substeps == 3
    # the grandchild (Δ = 25 m) sets the step of every domain
    @test nest_link(h2, :grand).stats.dt_child_max == nest_link(h2, :child).stats.dt_child_max ==
          nest_link(h2, :child).stats.dt_parent_max
    @test nest_link(h2, :grand).stats.dt_child_max <= 0.5 * 25.0 / 5.0 + 1e-12
    @test p2.t == c2.t == g2.t
end

# -------------------------------------------------------------------------
@testset "golden NestedRun: bit for bit after the hierarchy refactor" begin
    include(joinpath(_NH_DIR, "nest_golden_cases.jl"))
    ref = TOML.parsefile(joinpath(_NH_DIR, "data", "nest_golden.toml"))
    for (name, f) in nest_golden_cases()
        v = f()
        ok = v == Float64.(ref[name])
        ok || @printf("  golden %s: max |Δ| %.3e\n", name, maximum(abs, v .- ref[name]))
        @test ok
    end
end

end # testset
