using Test
using Printf
using Random
using Aeolus
using Aeolus.Nesting
using Aeolus.StretchedGrids: StretchedGrid, zfaces, cell_thickness, surface_refined_faces
using Aeolus.Grids: interior
using Aeolus.Fields: set!, interior_view, location

# =========================================================================
# N-nest-b (#256): nesting operators on a z-stretched parent, in the
# reference-mass measure of the sound-proof systems, and solid-aware.
# =========================================================================

# --- smooth separable test function (as in era2_nesting.jl), with exact
# --- cell / face averages on *any* vertical face distribution
const _KB = 2π
const _PHIB = (0.3, 1.1, 0.7)
_avgB(a, b, φ) = (cos(_KB * a + φ) - cos(_KB * b + φ)) / (_KB * (b - a))
_ptB(x, φ)     = sin(_KB * x + φ)

_zfB(m::RegularGrid)   = [m.z0 + (k - m.Hz - 1) * m.dz for k in 1:m.Nz+2m.Hz+1]
_zfB(m::StretchedGrid) = zfaces(m)

function _exactB(loc, m, zf, i, j, k)
    x0 = m.x0 + (i - m.Hx - 1) * m.dx; y0 = m.y0 + (j - m.Hy - 1) * m.dy
    fx = loc isa FaceX ? _ptB(x0, _PHIB[1]) : _avgB(x0, x0 + m.dx, _PHIB[1])
    fy = loc isa FaceY ? _ptB(y0, _PHIB[2]) : _avgB(y0, y0 + m.dy, _PHIB[2])
    fz = loc isa FaceZ ? _ptB(zf[k], _PHIB[3]) : _avgB(zf[k], zf[k+1], _PHIB[3])
    return fx * fy * fz
end

function _fill_exactB!(f)
    m = f.mesh; zf = _zfB(m); loc = location(f)
    for k in axes(f.data, 3), j in axes(f.data, 2), i in axes(f.data, 1)
        f.data[i, j, k] = _exactB(loc, m, zf, i, j, k)
    end
    return f
end

"Max halo error of a child field, split into (child-top halo, every other halo cell)."
function _halo_errsB(f)
    m = f.mesh; zf = _zfB(m); loc = location(f); ii, jj, kk = interior(m)
    etop = 0.0; eoth = 0.0
    for k in axes(f.data, 3), j in axes(f.data, 2), i in axes(f.data, 1)
        (i in ii) && (j in jj) && (k in kk) && continue
        e = abs(f.data[i, j, k] - _exactB(loc, m, zf, i, j, k))
        k > last(kk) ? (etop = max(etop, e)) : (eoth = max(eoth, e))
    end
    return etop, eoth
end

"""
Unit-box parent with `N` cells per unit: uniform `1/N` up to `z = 0.5`
(the child top), then geometric stretching by `ratio` per layer to
`z ≥ 1`. Locally self-similar under refinement, so the child-top halo
stencil (which reaches the stretched layers) sees the same node pattern
at every resolution, scaled by `1/N`.
"""
function _stretched_unit_parent(N; ratio=1.08, halo=(4, 4, 4))
    h = 1 / N
    zf = collect(0:h:0.5)
    while zf[end] < 1.0
        push!(zf, zf[end] + (zf[end] - zf[end-1]) * ratio)
    end
    return StretchedGrid(Float64; size=(N, N, length(zf) - 1), extent=(1.0, 1.0, zf[end]),
                         halo=halo, stretching=(:custom, zf))
end

_rows(pf, Hp, cov, ax) = ntuple(d -> Hp[d] .+ (first(cov[d]):last(cov[d]) + (d == ax ? 1 : 0)), 3)

# Reference state whose ρ̄ falls by about 30 % over 3 km: an isothermal
# atmosphere with scale height H = 3000 m / ln(1/0.7).
const _H30 = 3000 / log(1 / 0.7)
const _T30 = 9.80665 * _H30 / 287.04

@testset "Era II — Nesting: stretched parent, mass weighting, solid-aware (#256)" begin

# =========================================================================
# 1. Stretched parent
# =========================================================================

@testset "geometry: NestedGrid on a StretchedGrid parent" begin
    pm = _stretched_unit_parent(8)
    nest = nested_grid(pm, ((0.25, 0.5), (0.5, 0.75), (0.25, 0.5)), (2, 2, 4))
    cm = nest_child_mesh(nest)
    @test nest_parent_mesh(nest) === pm
    @test (cm.Nx, cm.Ny, cm.Nz) == (4, 4, 8)
    @test cm.dz ≈ 1 / 32 && cm.z0 == 0.25
    @test nest_covered_range(nest) == (3:4, 5:6, 3:4)
    # the child top must sit on a parent face …
    @test_throws ErrorException nested_grid(pm, ((0.25, 0.5), (0.5, 0.75), (0.25, 0.52)), 2)
    # … and every covered parent layer must hold exactly rz child layers:
    # an overlap reaching into the stretched layers is refused
    zf = zfaces(pm); ktop = findfirst(==(0.5), zf)
    @test_throws ErrorException nested_grid(pm, ((0.25, 0.5), (0.5, 0.75),
                                                 (0.25, zf[ktop + 1])), 2)
end

# -------------------------------------------------------------------------
# Halo prolongation order on the stretched parent. The interpolating
# polynomial is built on the actual parent face heights, so the
# reconstruction keeps its formal order p on a non-uniform stencil. The
# child-top halo is reported separately: its stencil reaches the layers
# stretched by 1.08, where the leading error constant changes from one
# resolution to the next until the stencil is asymptotically fine (the
# first stretched layers are 1.08, 1.17, 1.26 h); it is held to p − 0.6.
@testset "prolongation order on a stretched parent (ratio 1.08 above the overlap)" begin
    for r in (2, 3, 4), p in (2, 3)
        rt = Dict{String,Float64}(); ro = Dict{String,Float64}()
        for (lname, loc) in (("C", CellCenter()), ("u", FaceX()), ("v", FaceY()),
                             ("w", FaceZ()))
            et = Float64[]; eo = Float64[]
            for N in (8, 16, 32)
                pm = _stretched_unit_parent(N)
                nest = nested_grid(pm, ((0.25, 0.5), (0.5, 0.75), (0.25, 0.5)), r;
                                   halo=(3, 3, 3))
                pf = _fill_exactB!(Field(loc, pm))
                cf = Field(loc, nest_child_mesh(nest))
                interpolate_to_nest_boundary!(cf, pf, nest; order=p)
                a, b = _halo_errsB(cf); push!(et, a); push!(eo, b)
            end
            rt[lname] = log2(et[2] / et[3]); ro[lname] = log2(eo[2] / eo[3])
            @test ro[lname] > p - 0.45
            @test rt[lname] > p - 0.6
            @test et[3] < et[1] && eo[3] < eo[1]
        end
        @printf("  stretched parent r=%d p=%d : halo order C %.2f u %.2f v %.2f w %.2f | child-top halo C %.2f u %.2f v %.2f w %.2f\n",
                r, p, (ro[l] for l in ("C", "u", "v", "w"))...,
                (rt[l] for l in ("C", "u", "v", "w"))...)
    end
end

# -------------------------------------------------------------------------
@testset "exact conservation on the stretched parent: restrict ∘ prolong = id, ∫, fluxes" begin
    Random.seed!(256_1)
    worst_rp = 0.0; worst_int = 0.0; worst_flux = 0.0
    for r in (2, 3, (2, 2, 4)), p in (2, 3, 4)
        pm = _stretched_unit_parent(10; halo=(3, 3, 3))
        nest = nested_grid(pm, ((0.3, 0.6), (0.2, 0.5), (0.2, 0.5)), r; halo=(3, 3, 3))
        cm = nest_child_mesh(nest)
        cov = nest_covered_range(nest); Hp = (pm.Hx, pm.Hy, pm.Hz)
        for (ax, loc) in ((0, CellCenter()), (1, FaceX()), (2, FaceY()), (3, FaceZ()))
            pf = Field(loc, pm); pf.data .= 1 .+ randn(size(pf.data))
            cf = Field(loc, cm)
            interpolate_parent_to_child!(cf, pf, nest; order=p)
            back = Field(loc, pm)
            rng = _rows(pf, Hp, cov, ax)
            if ax == 0
                restrict_to_parent!(back, cf, nest)
                # the overlap integral: Σ parent φ V (actual layer thickness)
                # equals Σ child φ V
                dzc = cell_thickness(pm)
                Ip = sum(pf.data[i, j, k] * pm.dx * pm.dy * dzc[k] for i in rng[1],
                         j in rng[2], k in rng[3])
                Ic = sum(interior_view(cf)) * cm.dx * cm.dy * cm.dz
                worst_int = max(worst_int, abs(Ip - Ic) / abs(Ip))
                @test abs(Ip - Ic) / abs(Ip) < 1e-13
            else
                restrict_to_parent!(back, cf, nest; include_boundary=true)
                mm = compute_nest_flux_mismatch(pf, cf, nest)
                planes = ax == 1 ? (mm.west, mm.east) : ax == 2 ? (mm.south, mm.north) :
                                                         (mm.bottom, mm.top)
                Ap = ax == 3 ? pm.dx * pm.dy : (ax == 1 ? pm.dy : pm.dx) * maximum(cell_thickness(pm))
                fl = maximum(maximum(abs, P) for P in planes) / (maximum(abs, pf.data[rng...]) * Ap)
                worst_flux = max(worst_flux, fl)
                @test fl < 1e-13
            end
            rel = maximum(abs, back.data[rng...] .- pf.data[rng...]) /
                  maximum(abs, pf.data[rng...])
            worst_rp = max(worst_rp, rel)
            @test rel < 1e-14
        end
    end
    @printf("  stretched parent, random data: max |restrict(prolong f) − f|/max|f| = %.2e, overlap ∫ rel. error %.2e, face-flux mismatch %.2e\n",
            worst_rp, worst_int, worst_flux)
end

# -------------------------------------------------------------------------
# A StretchedGrid with uniform faces is the RegularGrid: the coordinate
# plans must reproduce the integer plans of N-nest-a to round-off.
@testset "uniform StretchedGrid parent ≡ RegularGrid parent" begin
    Random.seed!(256_2)
    rg = RegularGrid(Float64; size=(10, 10, 10), extent=(1.0, 1.0, 1.0), halo=(3, 3, 3))
    sg = StretchedGrid(rg)
    worst = 0.0
    for loc in (CellCenter(), FaceX(), FaceY(), FaceZ()), p in (2, 3, 4)
        ext = ((0.3, 0.6), (0.2, 0.5), (0.4, 0.7))
        nr = nested_grid(rg, ext, (2, 3, 4)); ns = nested_grid(sg, ext, (2, 3, 4))
        pr = Field(loc, rg); pr.data .= 1 .+ randn(size(pr.data))
        ps = Field(loc, sg); ps.data .= pr.data
        cr = Field(loc, nest_child_mesh(nr)); cs = Field(loc, nest_child_mesh(ns))
        interpolate_parent_to_child!(cr, pr, nr; order=p)
        interpolate_parent_to_child!(cs, ps, ns; order=p)
        worst = max(worst, maximum(abs, cr.data .- cs.data))
    end
    @printf("  uniform StretchedGrid vs RegularGrid parent: max |Δ child| = %.2e\n", worst)
    @test worst < 1e-13
end

# -------------------------------------------------------------------------
@testset "step_nested! with a stretched Boussinesq parent" begin
    zf = surface_refined_faces(dz_surface=50, z_uniform=500, z_top=1500, ratio=1.08)
    pm = StretchedGrid(Float64; size=(12, 12, length(zf) - 1), extent=(600.0, 600.0, 1500.0),
                       halo=(3, 3, 3), stretching=(:custom, zf))
    nest = nested_grid(pm, ((150.0, 300.0), (150.0, 300.0), (0.0, 400.0)), (3, 3, 4);
                       halo=(3, 3, 3))
    cm = nest_child_mesh(nest)
    psim = BoussinesqSimulation(pm; moisture=true,
                                bcs=FieldBoundaryConditions(:periodic, :periodic, :wall))
    csim = BoussinesqSimulation(cm; moisture=true,
                                bcs=nest_bcs(cm, :nest, :nest; bottom=:wall, top=:nest, moisture=true))
    fill!(psim.θ.data, 300.0); fill!(csim.θ.data, 300.0)
    fill!(psim.qv.data, 0.010); fill!(csim.qv.data, 0.0)
    step_nested!(psim, csim, nest, 0.5)
    ii, jj, kk = interior(cm)
    halo_qv = [csim.qv.data[i, j, k] for i in axes(csim.qv.data, 1), j in jj, k in kk
               if !(i in ii)]
    @test maximum(abs, halo_qv .- 0.010) < 1e-15
    @test maximum(abs, interior_view(csim.θ) .- 300) < 1e-12
    # a state at rest stays at rest, and nothing is created or lost
    for key in (:mass, :theta, :qv, :momentum_x)
        @test nest_conservation_error(psim, csim, nest, key) < 1e-14
    end
    @test maximum(abs, interior_view(psim.w)) < 1e-12
end

# -------------------------------------------------------------------------
# Flow through the nest on the stretched parent: a uniform free stream
# (u = 2 m/s, no buoyancy) carries a warm blob through the child for 30
# two-way steps. The free stream must be preserved to round-off on both
# grids, no vertical motion may appear, and θ must stay within the
# initial bounds up to the advection scheme's own dispersive undershoot.
# The child's nest faces are OpenBoundary faces fed by NestBoundaryData
# (N-nest-d, #261), whose boundary-normal interface plane is Dirichlet. The
# N-nest-a NestBC path (free interface faces), which amplified round-off at
# the nest corners on a *uniform* parent too (reported on #256), is retired.
@testset "free stream and a warm blob through the nest on a stretched parent (30 steps)" begin
    zf = surface_refined_faces(dz_surface=50, z_uniform=500, z_top=1500, ratio=1.08)
    pm = StretchedGrid(Float64; size=(12, 12, length(zf) - 1), extent=(600.0, 600.0, 1500.0),
                       halo=(3, 3, 3), stretching=(:custom, zf))
    nest = nested_grid(pm, ((150.0, 300.0), (150.0, 300.0), (0.0, 400.0)), (3, 3, 4);
                       halo=(3, 3, 3))
    cm = nest_child_mesh(nest)
    psim = BoussinesqSimulation(pm; bcs=FieldBoundaryConditions(:periodic, :periodic, :wall))
    csim = BoussinesqSimulation(cm; bcs=nest_bcs(cm, :nest, :nest; bottom=:wall, top=:nest))
    fill!(psim.u.data, 2.0)
    for k in axes(psim.θ.data, 3), j in axes(psim.θ.data, 2), i in axes(psim.θ.data, 1)
        psim.θ.data[i, j, k] = 300 + 0.5 * exp(-((i - 6)^2 + (j - 6)^2 + (k - 6)^2) / 4)
    end
    θmax0 = maximum(interior_view(psim.θ))
    interpolate_parent_to_child!(csim.θ, psim.θ, nest)
    interpolate_parent_to_child!(csim.u, psim.u, nest)
    st = ntuple(_ -> nest_interp_state(nest, psim, csim), 3)
    lo = Inf; hi = -Inf
    for n in 1:30
        step_nested!(psim, csim, nest, 2.0; state=st)
        lo = min(lo, minimum(interior_view(csim.θ))); hi = max(hi, maximum(interior_view(csim.θ)))
    end
    du = max(maximum(abs, interior_view(csim.u) .- 2), maximum(abs, interior_view(psim.u) .- 2))
    @printf("  stretched parent, 30 two-way steps of a free stream: max |u − 2| %.1e, max |w| %.1e, child θ range [%.4f, %.4f] (initial max %.4f)\n",
            du, maximum(abs, interior_view(csim.w)), lo, hi, θmax0)
    @test du < 1e-12                                     # free stream preserved
    @test maximum(abs, interior_view(csim.w)) < 1e-12
    @test maximum(abs, interior_view(psim.w)) < 1e-12
    @test hi - 300 > 0.3                                 # the blob did cross the child
    @test hi <= θmax0 + 1e-12                            # no new maximum
    @test lo > 300 - 0.01 * 0.5                          # undershoot < 1 % of the blob
    @test nest_conservation_error(psim, csim, nest, :mass) < 1e-14
end

# -------------------------------------------------------------------------
# The Berlin parent (#259, maintainer decision 2026-10-05): uniform 25 m
# to 500 m, stretched by 1.08 (capped at 100 m) to 3 km; refinement
# (2, 2, 4) and a ~400 m child. Quote the overlap geometry and check that
# the child-top halo stencil stays in the uniform region.
@testset "Berlin overlap geometry: parent faces at the 400 m child top" begin
    zf = surface_refined_faces(dz_surface=25, z_uniform=500, z_top=3000, ratio=1.08,
                               dz_max=100)
    pm = StretchedGrid(Float64; size=(8, 8, length(zf) - 1), extent=(250.0, 250.0, 3000.0),
                       halo=(3, 3, 3), stretching=(:custom, zf))
    nest = nested_grid(pm, ((62.5, 187.5), (62.5, 187.5), (0.0, 400.0)), (2, 2, 4);
                       halo=(3, 3, 3))
    cm = nest_child_mesh(nest)
    @test pm.Nz == 53
    @test (cm.dx, cm.dz, cm.Nz) == (15.625, 6.25, 64)
    @test nest.rz == 4 && nest_covered_range(nest)[3] == 1:16
    ktop = pm.Hz + 17                                   # storage face at the child top
    @test zfaces(pm)[ktop] == 400.0
    # every node of the order-3 stencils of the child-top halo is a face
    # of the uniform 25 m region (≤ 500 m)
    zs = Aeolus.Nesting._zplan(nest, :avg, 3, nothing)
    top = [zs[k] for k in cm.Hz+cm.Nz+1:cm.Nz+2cm.Hz]
    hi = maximum(s0 + 3 for (s0, _) in top)                 # last node (storage face)
    @test zfaces(pm)[hi] <= 500.0
    dzs = cell_thickness(pm)
    @printf("  Berlin parent: %d levels; faces around the child top: %s m; layers %s m; rz = %d; child-top halo stencil reaches z = %.0f m (uniform to 500 m, first stretched layer %.2f m)\n",
            pm.Nz, join(string.(zfaces(pm)[ktop-1:ktop+2]), ", "),
            join(string.(dzs[ktop-1:ktop+1]), ", "), nest.rz, zfaces(pm)[hi],
            dzs[pm.Hz + 21])
end

# =========================================================================
# 2. Reference-mass (anelastic) measure
# =========================================================================

function _anelastic_pair(; T_surf=_T30, child_T=T_surf, r=(2, 2, 4), ref_fn=nothing, scalars=())
    pm = RegularGrid(Float64; size=(8, 8, 24), extent=(1000.0, 1000.0, 3000.0), halo=(3, 3, 3))
    nest = nested_grid(pm, ((250.0, 500.0), (250.0, 750.0), (0.0, 1000.0)), r; halo=(3, 3, 3))
    cm = nest_child_mesh(nest)
    rp = ref_fn === nothing ? isothermal_reference(pm; T_surf) : ref_fn(pm)
    rc = ref_fn === nothing ? isothermal_reference(cm; T_surf=child_T) : ref_fn(cm)
    pA = AnelasticSimulation(pm; ref=rp)
    cA = AnelasticSimulation(cm; ref=rc, bcs=nest_bcs(cm, :nest, :nest; bottom=:wall, top=:nest, scalars))
    return pm, nest, cm, pA, cA
end

@testset "one reference profile: the shared-ReferenceState assertion" begin
    pm, nest, cm, pA, cA = _anelastic_pair()
    @test pA.ref.ρ̄[end] / pA.ref.ρ̄[1] ≈ 0.71 atol=0.01        # ρ̄ falls ~30 % over 3 km
    mis = nest_reference_mismatch(nest, pA.ref, cA.ref)
    # one isothermal profile sampled on both grids differs only by the
    # midpoint rule over a parent layer: (Δz_p² − Δz_c²) / (24 H²)
    pred = (125.0^2 - 31.25^2) / (24 * _H30^2)
    @printf("  shared profile: reference mismatch %.3e (midpoint-rule prediction %.3e)\n", mis, pred)
    @test isapprox(mis, pred; rtol=0.05)
    @test nest_measure(nest, pA, cA).kind === :mass
    # a different profile (child surface 1 K warmer) is refused
    _, nest2, _, pB, cB = _anelastic_pair(child_T=_T30 + 1)
    mis2 = nest_reference_mismatch(nest2, pB.ref, cB.ref)
    @printf("  child surface 1 K warmer: mismatch %.3e → refused\n", mis2)
    @test mis2 > 1e-3
    @test_throws ErrorException nest_measure(nest2, pB, cB)
    # a pseudo-incompressible child under an anelastic parent is refused
    cP = PseudoIncompressibleSimulation(cm; ref=cA.ref, bcs=nest_bcs(cm, :nest, :nest; bottom=:wall, top=:nest))
    @test_throws ErrorException nest_measure(nest, pA, cP)
end

@testset "mass weighting: ∫ρ̄θ dV and face mass fluxes exact; volume weighting is not" begin
    Random.seed!(256_3)
    pm, nest, cm, pA, cA = _anelastic_pair()
    m = nest_measure(nest, pA, cA)
    ρp = pA.ref.ρ̄c_pad; ρc = cA.ref.ρ̄c_pad
    cov = nest_covered_range(nest); Hp = (pm.Hx, pm.Hy, pm.Hz)
    worst_rp = 0.0; worst_int = 0.0; worst_flux = 0.0; ctrl_int = 0.0; ctrl_flux = 0.0
    for p in (2, 3, 4), (ax, loc) in ((0, CellCenter()), (1, FaceX()), (2, FaceY()), (3, FaceZ()))
        pf = Field(loc, pm); pf.data .= 300 .+ randn(size(pf.data))
        cf = Field(loc, cm)
        interpolate_parent_to_child!(cf, pf, nest; order=p, measure=m)
        rng = _rows(pf, Hp, cov, ax)
        back = Field(loc, pm); vol = Field(loc, pm)
        if ax == 0
            restrict_to_parent!(back, cf, nest; measure=m)
            restrict_to_parent!(vol, cf, nest)                    # negative control
            # ∫ρ̄θ dV over the overlap: the child's own ρ̄ against the
            # parent's (mass of a covered parent cell = Σ of its children)
            Ic = sum(ρc[k] * cm.dz * cf.data[i, j, k] for i in cm.Hx+1:cm.Hx+cm.Nx,
                     j in cm.Hy+1:cm.Hy+cm.Ny, k in cm.Hz+1:cm.Hz+cm.Nz) * cm.dx * cm.dy
            Ip(f) = sum((m.pc[k+1] - m.pc[k]) * f.data[i, j, k] for i in rng[1], j in rng[2],
                        k in rng[3]) * pm.dx * pm.dy
            worst_int = max(worst_int, abs(Ip(pf) - Ic) / Ic, abs(Ip(back) - Ic) / Ic)
            ctrl_int = max(ctrl_int, abs(Ip(vol) - Ic) / Ic)
            @test abs(Ip(pf) - Ic) / Ic < 1e-13        # prolongation conserves ∫ρ̄θ
            @test abs(Ip(back) - Ic) / Ic < 1e-13      # so does restriction
        else
            restrict_to_parent!(back, cf, nest; include_boundary=true, measure=m)
            restrict_to_parent!(vol, cf, nest; include_boundary=true)
            # the parent face mass flux ρ̄u·A equals Σ ρ̄u·A of its child faces
            mm = compute_nest_flux_mismatch(pf, cf, nest; measure=m)
            planes = ax == 1 ? (mm.west, mm.east) : ax == 2 ? (mm.south, mm.north) :
                                                     (mm.bottom, mm.top)
            Fs = maximum(abs, pf.data[rng...]) * pm.dx * pm.dz * maximum(ρp)
            fl = maximum(maximum(abs, P) for P in planes) / Fs
            worst_flux = max(worst_flux, fl)
            @test fl < 1e-13
            if ax != 3
                # explicit Σρ̄u·A on one interior parent face vs the restricted face
                A = pm.Hx + first(cov[1]) + 1; B = pm.Hy + first(cov[2]); Kp = first(cov[3])
                ia = ax == 1 ? cm.Hx + nest.rx + 1 : nothing
                flux_c = 0.0; mass_c = 0.0
                for kc in 1:nest.rz, t in 1:(ax == 1 ? nest.ry : nest.rx)
                    k = cm.Hz + (Kp - 1) * nest.rz + kc
                    i, j = ax == 1 ? (cm.Hx + nest.rx + 1, cm.Hy + t) :
                                     (cm.Hx + t, cm.Hy + nest.ry + 1)
                    w = ρc[k] * cm.dz * (ax == 1 ? cm.dy : cm.dx)
                    flux_c += w * cf.data[i, j, k]; mass_c += w
                end
                Pi, Pj = ax == 1 ? (A, B) : (pm.Hx + first(cov[1]), pm.Hy + first(cov[2]) + 1)
                flux_p = mass_c * back.data[Pi, Pj, pm.Hz + Kp]
                @test abs(flux_p - flux_c) / abs(flux_c) < 1e-13
                ctrl_flux = max(ctrl_flux, abs(mass_c * vol.data[Pi, Pj, pm.Hz + Kp] - flux_c) /
                                           abs(flux_c))
            end
        end
        rel = maximum(abs, back.data[rng...] .- pf.data[rng...]) / maximum(abs, pf.data[rng...])
        worst_rp = max(worst_rp, rel)
        @test rel < 1e-14
    end
    @printf("  ρ̄ falling 30%%/3 km: ∫ρ̄θ rel. error %.2e, face mass-flux mismatch %.2e, restrict∘prolong %.2e\n",
            worst_int, worst_flux, worst_rp)
    @printf("  negative control (volume-weighted restriction of the same data): ∫ρ̄θ rel. error %.2e, face mass flux rel. error %.2e\n",
            ctrl_int, ctrl_flux)
    @test ctrl_int > 1e3 * max(worst_int, 1e-16)
    @test ctrl_flux > 1e-7
    # a constant stays constant in the mass measure
    pf = Field(CellCenter(), pm); fill!(pf.data, 300.0); cf = Field(CellCenter(), cm)
    interpolate_parent_to_child!(cf, pf, nest; order=3, measure=m)
    @test maximum(abs, cf.data .- 300) / 300 < 1e-14
end

@testset "analytic limit: ρ̄ ≡ const gives the N-nest-a operators bit for bit" begin
    Random.seed!(256_4)
    # g = 0: the isothermal atmosphere has an infinite scale height, ρ̄ ≡ ρ₀
    pm, nest, cm, pA, cA = _anelastic_pair(ref_fn = m -> isothermal_reference(m; g_accel=0.0))
    @test all(==(pA.ref.ρ̄[1]), pA.ref.ρ̄c_pad) && all(==(pA.ref.ρ̄[1]), cA.ref.ρ̄c_pad)
    m = nest_measure(nest, pA, cA)
    @test m.kind === :volume
    for loc in (CellCenter(), FaceX(), FaceY(), FaceZ()), p in (2, 3)
        pf = Field(loc, pm); pf.data .= 300 .+ randn(size(pf.data))
        a = Field(loc, cm); b = Field(loc, cm)
        interpolate_parent_to_child!(a, pf, nest; order=p, measure=m)
        interpolate_parent_to_child!(b, pf, nest; order=p)
        @test a.data == b.data
        ra = Field(loc, pm); rb = Field(loc, pm)
        ra.data .= pf.data; rb.data .= pf.data
        restrict_to_parent!(ra, a, nest; measure=m); restrict_to_parent!(rb, a, nest)
        @test ra.data == rb.data
    end
    # continuity: a near-constant W (relative variation 1e-9 over the
    # column) takes the general mass-measure path and agrees with the
    # volume operators to O(1e-9)
    Wp = [1.2 * (1 + 1e-9 * k / 30) for k in 1:pm.Nz+2pm.Hz]
    Wc = [1.2 * (1 + 1e-9 * (k - cm.Hz - 0.5) * cm.dz / (30 * pm.dz) +
                  1e-9 * (pm.Hz + 0.5) / 30) for k in 1:cm.Nz+2cm.Hz]
    mm = nest_measure(nest, Wp, Wc)
    @test mm.kind === :mass
    pf = Field(CellCenter(), pm); pf.data .= 300 .+ randn(size(pf.data))
    a = Field(CellCenter(), cm); b = Field(CellCenter(), cm)
    interpolate_parent_to_child!(a, pf, nest; order=3, measure=mm)
    interpolate_parent_to_child!(b, pf, nest; order=3)
    d = maximum(abs, a.data .- b.data) / 300
    @printf("  near-constant W (1e-9 variation): mass vs volume prolongation differ by %.1e\n", d)
    @test d < 1e-8
end

# -------------------------------------------------------------------------
# Refluxing in the mass measure: the parent (closed) and the child advance
# ρ̄θ with different random mass-flux registers Φ = ∫ρ̄F dt; the composite
# ∫ρ̄θ dV drifts by the interface disagreement unless refluxed.
@testset "refluxing conserves the composite ∫ρ̄θ dV" begin
    Random.seed!(256_5)
    pm, nest, cm, pA, cA = _anelastic_pair(r=(2, 3, 4))
    m = nest_measure(nest, pA, cA)
    function regs(mesh, amp; closed)
        Φ = (Field(FaceX(), mesh), Field(FaceY(), mesh), Field(FaceZ(), mesh))
        N = (mesh.Nx, mesh.Ny, mesh.Nz); H = (mesh.Hx, mesh.Hy, mesh.Hz)
        for ax in 1:3
            rng = ntuple(a -> a == ax ? (H[a] + 1 : H[a] + N[a] + 1) : (H[a] + 1 : H[a] + N[a]), 3)
            Φ[ax].data[rng...] .= amp .* randn(length.(rng)...)
            if closed
                Φ[ax].data[ntuple(a -> a == ax ? (H[a] + 1:H[a] + 1) : rng[a], 3)...] .= 0
                Φ[ax].data[ntuple(a -> a == ax ? (H[a] + N[a] + 1:H[a] + N[a] + 1) : rng[a], 3)...] .= 0
            end
        end
        return Φ
    end
    # mass-form finite-volume update: W θ ΔV changes by −Σ Φ A
    function fv!(f, Φ, W)
        g = f.mesh; X, Y, Z = Φ[1].data, Φ[2].data, Φ[3].data
        for k in g.Hz+1:g.Hz+g.Nz, j in g.Hy+1:g.Hy+g.Ny, i in g.Hx+1:g.Hx+g.Nx
            f.data[i, j, k] -= ((X[i+1, j, k] - X[i, j, k]) / g.dx +
                                (Y[i, j+1, k] - Y[i, j, k]) / g.dy +
                                (Z[i, j, k+1] - Z[i, j, k]) / g.dz) / W[k]
        end
    end
    interior_view(pA.θ) .= 300 .+ randn(size(interior_view(pA.θ)))
    interpolate_parent_to_child!(cA.θ, pA.θ, nest; order=3, measure=m)
    I0 = composite_integral(pA.θ, cA.θ, nest; measure=m)
    # the parent's integral, uncovered cells in the parent's own mass
    # W_p Δz_p (what the parent conserves), covered cells in the measure's
    # mass Σ W_c Δz_c of their children (what prolongation conserves)
    cxr, cyr, czr = nest_covered_range(nest)
    pmass(i, j, k) = (i - pm.Hx in cxr && j - pm.Hy in cyr && k - pm.Hz in czr) ?
                     m.pc[k+1] - m.pc[k] : pA.ref.ρ̄c_pad[k] * pm.dz
    Ipar = sum(pmass(i, j, k) * pA.θ.data[i, j, k] for i in pm.Hx+1:pm.Hx+pm.Nx,
               j in pm.Hy+1:pm.Hy+pm.Ny, k in pm.Hz+1:pm.Hz+pm.Nz) * pm.dx * pm.dy
    @test abs(I0 - Ipar) / Ipar < 1e-13              # prolongation: composite = parent
    preg = regs(pm, 1.0; closed=true); creg = regs(cm, 1.0; closed=false)
    # the child's bottom is the parent's ground (a wall in both): no flux
    # through it. (Before #349 this face carried random child fluxes, and
    # the refluxing default `periodic = (true, true, true)` wrapped their
    # mismatch into the parent's *top* layer, the #329 bug; with the
    # parent's real axes the correction across a wall is dropped, so the
    # register must be physical.)
    @test nest_periodic_axes(pA) == (true, true, false)
    creg[3].data[:, :, cm.Hz+1] .= 0
    fv!(pA.θ, preg, pA.ref.ρ̄c_pad); fv!(cA.θ, creg, cA.ref.ρ̄c_pad)
    Imid = composite_integral(pA.θ, cA.θ, nest; measure=m)
    restrict_to_parent!(pA.θ, cA.θ, nest; measure=m)
    reflux!(pA.θ, preg, creg, nest; measure=m, periodic=nest_periodic_axes(pA))
    I1 = composite_integral(pA.θ, cA.θ, nest; measure=m)
    drift = abs(Imid - I0) / I0; rel = abs(I1 - I0) / I0
    @printf("  mass-measure refluxing: composite ∫ρ̄θ drift without %.2e, with %.2e\n", drift, rel)
    @test rel < 1e-13
    @test drift > 1e3 * max(rel, 1e-17)
end

# -------------------------------------------------------------------------
# The sound-proof core has no qv field: a moist anelastic nest carries qv
# as a tracer of the field set, transferred in the mass measure.
@testset "moist anelastic nest: qv as a tracer, step_nested! with MassWeightedSimulation" begin
    pm, nest, cm, pA, cA = _anelastic_pair(scalars=(:qv,))
    qp = Field(CellCenter(), pm); qc = Field(CellCenter(), cm)
    fill!(qp.data, 0.008); fill!(qc.data, 0.0)
    st = ntuple(_ -> nest_interp_state(nest, pA, cA; scalars=(:qv => (qc, qp),)), 3)
    @test nest_scalar_names(st[1]) == [:θ, :qv]
    @test st[1].measure.kind === :mass
    θ0 = composite_integral(pA, cA, nest, :theta; state=st[1])
    step_nested!(pA, cA, nest, 1.0; state=st, feedback=TwoWayFeedback(buffer=0))
    ii, jj, kk = interior(cm)
    hq = [qc.data[i, j, k] for i in axes(qc.data, 1), j in jj, k in kk if !(i in ii)]
    @test maximum(abs, hq .- 0.008) < 1e-15
    # the covered parent cells now hold the ρ̄-weighted child mean of qv (0
    # in the interior of a dry child)
    cx, cy, cz = nest_covered_range(nest)
    @test maximum(abs, interior_view(qp)[cx, cy, cz]) < 1e-15
    # a base state at rest: θ = θ̄(z) is restricted onto θ̄ of the parent to
    # the reference mismatch, and the composite ∫ρ̄θ moves by no more
    θ1 = composite_integral(pA, cA, nest, :theta; state=st[1])
    @printf("  anelastic step_nested!: composite ∫ρ̄θ rel. change %.2e (reference mismatch %.2e)\n",
            abs(θ1 - θ0) / θ0, nest_reference_mismatch(nest, pA.ref, cA.ref))
    @test abs(θ1 - θ0) / θ0 < 2 * nest_reference_mismatch(nest, pA.ref, cA.ref)
    for key in (:mass, :theta, :qv, :momentum_x)
        @test isfinite(nest_conservation_error(pA, cA, nest, key))
    end
end

# =========================================================================
# 3. Solid-aware restriction
# =========================================================================

@testset "solid-aware: a building half-filling a parent cell" begin
    pm = RegularGrid(Float64; size=(8, 8, 8), extent=(8.0, 8.0, 8.0), halo=(2, 2, 2))
    nest = nested_grid(pm, ((2.0, 6.0), (2.0, 6.0), (0.0, 4.0)), (2, 2, 4); halo=(2, 2, 2))
    cm = nest_child_mesh(nest)
    sm = SolidMask(cm)
    # parent cell (I,J,K) = (3,3,1) (child block x ∈ [2,3], y ∈ [2,3], z ∈ [0,1]):
    # its western half (x < 2.5) is a building; parent cell (4,4,2) is all building
    mark_solid!(sm, (x, y, z) -> (2 < x < 2.5 && 2 < y < 3 && z < 1) ||
                                 (3 < x < 4 && 3 < y < 4 && 1 < z < 2))
    χ = nest_fluid_mask(nest, sm)
    # u = a linear field on fluid faces, 0 on solid faces (as the IBM sets it)
    uc = Field(FaceX(), cm); θc = Field(CellCenter(), cm)
    for k in axes(uc.data, 3), j in axes(uc.data, 2), i in 2:size(uc.data, 1)
        x = cm.x0 + (i - cm.Hx - 1) * cm.dx; y = cm.y0 + (j - cm.Hy - 0.5) * cm.dy
        z = cm.z0 + (k - cm.Hz - 0.5) * cm.dz
        uc.data[i, j, k] = (χ[i, j, k] && χ[i-1, j, k]) ? 1 + 0.5x + 0.25y + 0.125z : 0.0
    end
    for k in axes(θc.data, 3), j in axes(θc.data, 2), i in axes(θc.data, 1)
        z = cm.z0 + (k - cm.Hz - 0.5) * cm.dz
        θc.data[i, j, k] = χ[i, j, k] ? 300 + 0.01z + 0.1i : 999.0   # 999 marks solid
    end
    up = Field(FaceX(), pm); fill!(up.data, -7.0)
    θp = Field(CellCenter(), pm); fill!(θp.data, -7.0)
    restrict_to_parent!(up, uc, nest; solid=sm)
    restrict_to_parent!(θp, θc, nest; solid=sm)
    # the u face x = 3 between parent cells (3,·) and (4,·) at J = 3, K = 1:
    # all its child faces are fluid → plain mean; the face x = 2.5 is inside
    # parent cell 3 (not a parent face). Check the parent face at x = 3:
    P(i, j, k) = (pm.Hx + i, pm.Hy + j, pm.Hz + k)
    # analytic fluid mean of u over the four (y, z) child faces at x = 3
    ex = sum(1 + 0.5 * 3 + 0.25 * y + 0.125 * z for y in (2.25, 2.75), z in (0.125, 0.375, 0.625, 0.875)) / 8
    @test up.data[P(4, 3, 1)...] ≈ ex rtol=1e-15
    # the parent cell (3,3,1) is half building: its scalar is the fluid mean
    fl = [(i, j, k) for i in cm.Hx+1:cm.Hx+2, j in cm.Hy+1:cm.Hy+2, k in cm.Hz+1:cm.Hz+4
          if χ[i, j, k]]
    @test length(fl) == 8                                  # half of the 16 children
    @test θp.data[P(3, 3, 1)...] ≈ sum(θc.data[c...] for c in fl) / 8 rtol=1e-15
    @test θp.data[P(3, 3, 1)...] < 999                     # no solid value leaked in
    # a fully solid parent cell keeps its value, and so do its interior faces
    @test θp.data[P(4, 4, 2)...] == -7.0
    # u faces of the fully solid cell (4,4,2): x = 3 (west) and x = 4 (east,
    # an interior parent face of the nest) have no fluid child face
    @test up.data[P(5, 4, 2)...] == -7.0
    # a face bordering a half-solid cell averages only its fluid child faces:
    # the western face (x = 2) of cell (3,3,1) is an interface plane (not
    # restricted); the u face at x = 3 of row (·,3,1) was checked above.
    # Faces on the solid boundary itself are excluded (u = 0 there, IBM):
    # with a naive restriction they would pull the mean towards 0.
    naive = Field(FaceX(), pm); restrict_to_parent!(naive, uc, nest)
    @test naive.data[P(5, 4, 2)...] == 0.0                 # averaged from solid faces
    # fluid-weighted scalar integral over the overlap conserved
    cx, cy, cz = nest_covered_range(nest)
    Ic = sum(θc.data[i, j, k] * cm.dx * cm.dy * cm.dz
             for i in cm.Hx+1:cm.Hx+cm.Nx, j in cm.Hy+1:cm.Hy+cm.Ny, k in cm.Hz+1:cm.Hz+cm.Nz
             if χ[i, j, k])
    Ip = 0.0
    for K in cz, J in cy, I in cx
        nf = count(χ[cm.Hx + (I - cx[1]) * 2 + a, cm.Hy + (J - cy[1]) * 2 + b,
                     cm.Hz + (K - cz[1]) * 4 + c] for a in 1:2, b in 1:2, c in 1:4)
        Ip += θp.data[P(I, J, K)...] * nf * cm.dx * cm.dy * cm.dz
    end
    @printf("  solid-aware restriction: fluid-weighted ∫θ rel. error %.2e\n", abs(Ip - Ic) / Ic)
    @test abs(Ip - Ic) / Ic < 1e-13
end

# -------------------------------------------------------------------------
# The real Berlin buildings (OpenStreetMap, the summer static driver) on the
# nested child: 1 km × 1 km, Δx = 15.625 m, 64 levels of 6.25 m (~400 m),
# refinement (2, 2, 4) to the 31.25 m × 25 m parent. Child velocities on
# solid faces are set to NaN: a single solid face in any parent average
# would make that parent value NaN.
@testset "solid-aware restriction on the real Berlin solid field" begin
    BC = Aeolus.Urban
    cfg = BC.site_options(load_run_config(joinpath(pkgdir(Aeolus), "examples", "urban", "berlin_summer.toml")))
    pm = RegularGrid(Float64; size=(32, 32, 20), extent=(1000.0, 1000.0, 500.0), halo=(3, 3, 3))
    nest = nested_grid(pm, ((0.0, 1000.0), (0.0, 1000.0), (0.0, 400.0)), (2, 2, 4);
                       halo=(3, 3, 3))
    cm = nest_child_mesh(nest)
    @test (cm.Nx, cm.Ny, cm.Nz, cm.dx, cm.dz) == (64, 64, 64, 15.625, 6.25)
    # the summer ground (its static driver, on this child mesh: the summer grid)
    st = BC.case_static_inputs(cfg, cm)
    city = st.city
    sm, _ = BC.solid_and_classes(st, cm; tag_walls=false)
    solid = sm.mask
    nsolid = count(solid)
    @test nsolid > 0
    χ = nest_fluid_mask(nest, solid)
    stats = Dict{Symbol,NTuple{3,Int}}()
    for (nm, loc, ax) in ((:u, FaceX(), 1), (:v, FaceY(), 2), (:w, FaceZ(), 3))
        cf = Field(loc, cm); cf.data .= 1.0
        e = ntuple(d -> d == ax ? 1 : 0, 3)
        for k in 2:size(cf.data, 3), j in 2:size(cf.data, 2), i in 2:size(cf.data, 1)
            (χ[i, j, k] && χ[i - e[1], j - e[2], k - e[3]]) || (cf.data[i, j, k] = NaN)
        end
        pf = Field(loc, pm); fill!(pf.data, 0.5)
        restrict_to_parent!(pf, cf, nest; solid)
        naive = Field(loc, pm); restrict_to_parent!(naive, cf, nest)
        # every restricted parent face is finite: no solid child face entered
        @test all(isfinite, pf.data)
        # the parent faces restriction owns: strictly inside the nest along
        # the face normal (the interface planes belong to flux correction)
        o = (nest.ix + pm.Hx, nest.iy + pm.Hy, nest.iz + pm.Hz); n = (nest.nx, nest.ny, nest.nz)
        vals = pf.data[ntuple(d -> d == ax ? (o[d]+1:o[d]+n[d]-1) : (o[d]:o[d]+n[d]-1), 3)...]
        nmixed = count(isnan, naive.data)               # parent faces a naive average poisons
        nkept  = count(==(0.5), vals)                   # parent faces with no fluid child: kept
        nfluid = count(==(1.0), vals)
        @test nmixed > 0
        # every owned face is either a fluid mean (exactly 1) or, with no
        # fluid child face, kept (0.5): nothing in between
        @test nkept + nfluid == length(vals)
        @test nkept > 0
        stats[nm] = (nmixed, nkept, nfluid)
    end
    # scalars: a fully built-over parent cell keeps its value
    θc = Field(CellCenter(), cm); θc.data .= ifelse.(solid, NaN, 300.0)
    θp = Field(CellCenter(), pm); fill!(θp.data, 290.0)
    restrict_to_parent!(θp, θc, nest; solid)
    @test all(isfinite, θp.data)
    cov = θp.data[pm.Hx+1:pm.Hx+32, pm.Hy+1:pm.Hy+32, pm.Hz+1:pm.Hz+16]
    @printf("  Berlin solid field: %d of %d child cells solid; parent faces a naive restriction would average from solid cells: u %d, v %d, w %d (solid-aware: 0); parent cells fully built over (value kept): %d; buildings %d\n",
            nsolid, cm.Nx * cm.Ny * cm.Nz, stats[:u][1], stats[:v][1], stats[:w][1],
            count(==(290.0), cov), length(city.buildings))
end

end # testset
