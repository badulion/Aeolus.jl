using Test
using Random
using Aeolus
using Aeolus.Nesting
using Aeolus.StretchedGrids: StretchedGrid, zfaces, cell_thickness
using Aeolus.Grids: interior
using Aeolus.Fields: set!, interior_view, location

# =========================================================================
# N-nest-b (#256): nesting operators on a z-stretched parent, in the
# reference-mass measure of the sound-proof systems, and solid-aware.
# =========================================================================

# --- smooth separable test function (as in nesting.jl), with exact
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
    @test (cm.Nx, cm.Ny, cm.Nz) == (4, 4, 8) && cm.dz ≈ 1 / 32
    @test nest_covered_range(nest) == (3:4, 5:6, 3:4)
    # every covered parent layer must hold exactly rz child layers: an overlap
    # reaching into the stretched layers is refused
    zf = zfaces(pm); ktop = findfirst(==(0.5), zf)
    @test_throws ErrorException nested_grid(pm, ((0.25, 0.5), (0.5, 0.75),
                                                 (0.25, zf[ktop + 1])), 2)
end

# The interpolating polynomial is built on the actual parent face heights,
# so the reconstruction keeps its order p on a non-uniform stencil.
@testset "prolongation order on a stretched parent (ratio 1.08 above the overlap)" begin
    errs = map((16, 32)) do N
        pm = _stretched_unit_parent(N)
        nest = nested_grid(pm, ((0.25, 0.5), (0.5, 0.75), (0.25, 0.5)), 2; halo=(3, 3, 3))
        pf = _fill_exactB!(Field(CellCenter(), pm))
        cf = Field(CellCenter(), nest_child_mesh(nest))
        interpolate_to_nest_boundary!(cf, pf, nest; order=3)
        _halo_errsB(cf)
    end
    @test log2(errs[1][2] / errs[2][2]) > 3 - 0.45     # every halo but the child top
    @test log2(errs[1][1] / errs[2][1]) > 3 - 0.6      # child top: stencil in the stretched layers
end

@testset "exact conservation on the stretched parent: restrict ∘ prolong = id, ∫, fluxes" begin
    Random.seed!(256_1)
    pm = _stretched_unit_parent(10; halo=(3, 3, 3))
    nest = nested_grid(pm, ((0.3, 0.6), (0.2, 0.5), (0.2, 0.5)), (2, 2, 4); halo=(3, 3, 3))
    cm = nest_child_mesh(nest)
    cov = nest_covered_range(nest); Hp = (pm.Hx, pm.Hy, pm.Hz)
    for (ax, loc) in ((0, CellCenter()), (3, FaceZ()))
        pf = Field(loc, pm); pf.data .= 1 .+ randn(size(pf.data))
        cf = Field(loc, cm)
        interpolate_parent_to_child!(cf, pf, nest; order=3)
        back = Field(loc, pm)
        rng = _rows(pf, Hp, cov, ax)
        if ax == 0
            restrict_to_parent!(back, cf, nest)
            dzc = cell_thickness(pm)          # Σ parent φ V (actual layer thickness) = Σ child φ V
            Ip = sum(pf.data[i, j, k] * pm.dx * pm.dy * dzc[k] for i in rng[1], j in rng[2], k in rng[3])
            Ic = sum(interior_view(cf)) * cm.dx * cm.dy * cm.dz
            @test abs(Ip - Ic) / abs(Ip) < 1e-13
        else
            restrict_to_parent!(back, cf, nest; include_boundary=true)
            mm = compute_nest_flux_mismatch(pf, cf, nest)
            @test maximum(maximum(abs, P) for P in (mm.bottom, mm.top)) /
                  (maximum(abs, pf.data[rng...]) * pm.dx * pm.dy) < 1e-13
        end
        @test maximum(abs, back.data[rng...] .- pf.data[rng...]) / maximum(abs, pf.data[rng...]) < 1e-14
    end
end

# =========================================================================
# 2. Reference-mass (anelastic) measure
# =========================================================================

function _anelastic_pair(; T_surf=_T30, child_T=T_surf, r=(2, 2, 4), ref_fn=nothing)
    pm = RegularGrid(Float64; size=(8, 8, 24), extent=(1000.0, 1000.0, 3000.0), halo=(3, 3, 3))
    nest = nested_grid(pm, ((250.0, 500.0), (250.0, 750.0), (0.0, 1000.0)), r; halo=(3, 3, 3))
    cm = nest_child_mesh(nest)
    rp = ref_fn === nothing ? isothermal_reference(pm; T_surf) : ref_fn(pm)
    rc = ref_fn === nothing ? isothermal_reference(cm; T_surf=child_T) : ref_fn(cm)
    pA = AnelasticSimulation(pm; ref=rp)
    cA = AnelasticSimulation(cm; ref=rc, bcs=nest_bcs(cm, :nest, :nest; bottom=:wall, top=:nest))
    return pm, nest, cm, pA, cA
end

@testset "one reference profile: the shared-ReferenceState assertion" begin
    pm, nest, cm, pA, cA = _anelastic_pair()
    # one isothermal profile sampled on both grids differs only by the
    # midpoint rule over a parent layer: (Δz_p² − Δz_c²) / (24 H²)
    @test isapprox(nest_reference_mismatch(nest, pA.ref, cA.ref),
                   (125.0^2 - 31.25^2) / (24 * _H30^2); rtol=0.05)
    @test nest_measure(nest, pA, cA).kind === :mass
    _, nest2, _, pB, cB = _anelastic_pair(child_T=_T30 + 1)   # child 1 K warmer: refused
    @test_throws ErrorException nest_measure(nest2, pB, cB)
    # ρ̄ ≡ const (g = 0) gives the volume operators
    _, n0, _, p0, c0 = _anelastic_pair(ref_fn = m -> isothermal_reference(m; g_accel=0.0))
    @test nest_measure(n0, p0, c0).kind === :volume
end

@testset "mass weighting: ∫ρ̄θ dV and face mass fluxes exact; volume weighting is not" begin
    Random.seed!(256_3)
    pm, nest, cm, pA, cA = _anelastic_pair()
    m = nest_measure(nest, pA, cA)
    ρp = pA.ref.ρ̄c_pad; ρc = cA.ref.ρ̄c_pad
    cov = nest_covered_range(nest); Hp = (pm.Hx, pm.Hy, pm.Hz)
    pf = Field(CellCenter(), pm); pf.data .= 300 .+ randn(size(pf.data))
    cf = Field(CellCenter(), cm)
    interpolate_parent_to_child!(cf, pf, nest; order=3, measure=m)
    rng = _rows(pf, Hp, cov, 0)
    back = Field(CellCenter(), pm); vol = Field(CellCenter(), pm)
    restrict_to_parent!(back, cf, nest; measure=m)
    restrict_to_parent!(vol, cf, nest)                    # negative control
    Ic = sum(ρc[k] * cm.dz * cf.data[i, j, k] for i in cm.Hx+1:cm.Hx+cm.Nx,
             j in cm.Hy+1:cm.Hy+cm.Ny, k in cm.Hz+1:cm.Hz+cm.Nz) * cm.dx * cm.dy
    Ip(f) = sum((m.pc[k+1] - m.pc[k]) * f.data[i, j, k] for i in rng[1], j in rng[2],
                k in rng[3]) * pm.dx * pm.dy
    @test abs(Ip(pf) - Ic) / Ic < 1e-13        # prolongation conserves ∫ρ̄θ
    @test abs(Ip(back) - Ic) / Ic < 1e-13      # so does restriction
    @test abs(Ip(vol) - Ic) / Ic > 1e3 * max(abs(Ip(back) - Ic) / Ic, 1e-16)
    # the parent face mass flux ρ̄u·A equals Σ ρ̄u·A of its child faces
    uf = Field(FaceX(), pm); uf.data .= 300 .+ randn(size(uf.data))
    uc = Field(FaceX(), cm)
    interpolate_parent_to_child!(uc, uf, nest; order=3, measure=m)
    mm = compute_nest_flux_mismatch(uf, uc, nest; measure=m)
    r1 = _rows(uf, Hp, cov, 1)
    @test maximum(maximum(abs, P) for P in (mm.west, mm.east)) /
          (maximum(abs, uf.data[r1...]) * pm.dx * pm.dz * maximum(ρp)) < 1e-13
end

# Refluxing in the mass measure: parent (closed) and child advance ρ̄θ with
# different random mass-flux registers; the composite ∫ρ̄θ dV drifts by the
# interface disagreement unless refluxed.
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
    preg = regs(pm, 1.0; closed=true); creg = regs(cm, 1.0; closed=false)
    # the child's bottom is the parent's ground (a wall in both): no flux
    # through it (#349: with the parent's real axes the correction across a
    # wall is dropped, so the register must be physical)
    creg[3].data[:, :, cm.Hz+1] .= 0
    fv!(pA.θ, preg, pA.ref.ρ̄c_pad); fv!(cA.θ, creg, cA.ref.ρ̄c_pad)
    Imid = composite_integral(pA.θ, cA.θ, nest; measure=m)
    restrict_to_parent!(pA.θ, cA.θ, nest; measure=m)
    reflux!(pA.θ, preg, creg, nest; measure=m, periodic=nest_periodic_axes(pA))
    rel = abs(composite_integral(pA.θ, cA.θ, nest; measure=m) - I0) / I0
    @test rel < 1e-13
    @test abs(Imid - I0) / I0 > 1e3 * max(rel, 1e-17)
end

# =========================================================================
# 3. Solid-aware restriction
# =========================================================================

@testset "solid-aware: a building half-filling a parent cell" begin
    pm = RegularGrid(Float64; size=(8, 8, 8), extent=(8.0, 8.0, 8.0), halo=(2, 2, 2))
    nest = nested_grid(pm, ((2.0, 6.0), (2.0, 6.0), (0.0, 4.0)), (2, 2, 4); halo=(2, 2, 2))
    cm = nest_child_mesh(nest)
    sm = SolidMask(cm)
    # parent cell (3,3,1): western half is a building; parent cell (4,4,2) all building
    mark_solid!(sm, (x, y, z) -> (2 < x < 2.5 && 2 < y < 3 && z < 1) ||
                                 (3 < x < 4 && 3 < y < 4 && 1 < z < 2))
    χ = nest_fluid_mask(nest, sm)
    θc = Field(CellCenter(), cm)
    for k in axes(θc.data, 3), j in axes(θc.data, 2), i in axes(θc.data, 1)
        θc.data[i, j, k] = χ[i, j, k] ? 300 + 0.01k + 0.1i : 999.0   # 999 marks solid
    end
    θp = Field(CellCenter(), pm); fill!(θp.data, -7.0)
    restrict_to_parent!(θp, θc, nest; solid=sm)
    P(i, j, k) = (pm.Hx + i, pm.Hy + j, pm.Hz + k)
    fl = [(i, j, k) for i in cm.Hx+1:cm.Hx+2, j in cm.Hy+1:cm.Hy+2, k in cm.Hz+1:cm.Hz+4 if χ[i, j, k]]
    @test length(fl) == 8                                  # half of the 16 children
    @test θp.data[P(3, 3, 1)...] ≈ sum(θc.data[c...] for c in fl) / 8 rtol=1e-15   # fluid mean
    @test θp.data[P(4, 4, 2)...] == -7.0                   # fully solid: value kept
    # fluid-weighted scalar integral over the overlap conserved
    cx, cy, cz = nest_covered_range(nest)
    Ic = sum(θc.data[i, j, k] for i in cm.Hx+1:cm.Hx+cm.Nx, j in cm.Hy+1:cm.Hy+cm.Ny,
             k in cm.Hz+1:cm.Hz+cm.Nz if χ[i, j, k])
    Ip = sum(θp.data[P(I, J, K)...] *
             count(χ[cm.Hx + (I - cx[1]) * 2 + a, cm.Hy + (J - cy[1]) * 2 + b, cm.Hz + (K - cz[1]) * 4 + c]
                   for a in 1:2, b in 1:2, c in 1:4) for K in cz, J in cy, I in cx)
    @test abs(Ip - Ic) / Ic < 1e-13
end

end # testset
