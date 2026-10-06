using Test
using Printf
using Random
using Aeolus
using Aeolus.Nesting
using Aeolus.Grids: interior
using Aeolus.Fields: set!, interior_view
using Aeolus.BoundaryConditions: FieldBoundaryConditions

# =========================================================================
# Phase 30 — parent ↔ child nesting operators (src/Nesting.jl, #250)
# =========================================================================

# =========================================================================
# helpers
# =========================================================================

"A cell-centred field on `m` holding a copy of `f`'s data (halos included)."
function _ccopy(f, m)
    g = Field(CellCenter(), m)
    g.data .= f.data
    return g
end

"Random-fill the interior of `f` with `base .+ amp .* randn`."
function _randfill!(f, base, amp)
    v = interior_view(f)
    v .= base .+ amp .* randn(size(v))
    return f
end

"""
Random face-flux registers `(Φx, Φy, Φz)` on mesh `m`, with zero flux
through the mesh's own boundary faces when `closed` (so a closed parent
conserves its integral exactly), random on every face otherwise.
"""
function _registers(m, amp; closed::Bool)
    Φ = (Field(FaceX(), m), Field(FaceY(), m), Field(FaceZ(), m))
    N = (m.Nx, m.Ny, m.Nz); H = (m.Hx, m.Hy, m.Hz)
    for ax in 1:3
        d = Φ[ax].data
        rng = ntuple(a -> a == ax ? (H[a] + 1 : H[a] + N[a] + 1) : (H[a] + 1 : H[a] + N[a]), 3)
        d[rng...] .= amp .* randn(length.(rng)...)
        if closed
            lo = ntuple(a -> a == ax ? (H[a] + 1 : H[a] + 1) : rng[a], 3)
            hi = ntuple(a -> a == ax ? (H[a] + N[a] + 1 : H[a] + N[a] + 1) : rng[a], 3)
            d[lo...] .= 0; d[hi...] .= 0
        end
    end
    return Φ
end

"Finite-volume update θ -= ∇·Φ (Φ = ∫F dt per unit area) on every interior cell."
function _fv_update!(f, Φ)
    m = f.mesh
    d = f.data; X, Y, Z = Φ[1].data, Φ[2].data, Φ[3].data
    for k in m.Hz+1:m.Hz+m.Nz, j in m.Hy+1:m.Hy+m.Ny, i in m.Hx+1:m.Hx+m.Nx
        d[i, j, k] -= (X[i+1, j, k] - X[i, j, k]) / m.dx +
                      (Y[i, j+1, k] - Y[i, j, k]) / m.dy +
                      (Z[i, j, k+1] - Z[i, j, k]) / m.dz
    end
    return f
end

# --- smooth separable test function f = Π_d sin(k x_d + φ_d), with exact
# --- cell / face averages so the prolongation error can be measured
const _K = 2π                              # one wavelength over L = 1
const _PHI = (0.3, 1.1, 0.7)
_avg1(a, b, φ) = (cos(_K * a + φ) - cos(_K * b + φ)) / (_K * (b - a))
_pt1(x, φ)     = sin(_K * x + φ)

"Exact value of the test function for the array cell (i,j,k) of a field at `loc` on `m`."
function _exact(loc, m, i, j, k)
    x0 = m.x0 + (i - m.Hx - 1) * m.dx; y0 = m.y0 + (j - m.Hy - 1) * m.dy
    z0 = m.z0 + (k - m.Hz - 1) * m.dz
    fx = loc isa FaceX ? _pt1(x0, _PHI[1]) : _avg1(x0, x0 + m.dx, _PHI[1])
    fy = loc isa FaceY ? _pt1(y0, _PHI[2]) : _avg1(y0, y0 + m.dy, _PHI[2])
    fz = loc isa FaceZ ? _pt1(z0, _PHI[3]) : _avg1(z0, z0 + m.dz, _PHI[3])
    return fx * fy * fz
end

function _fill_exact!(f)
    loc = Aeolus.Fields.location(f); m = f.mesh
    for k in axes(f.data, 3), j in axes(f.data, 2), i in axes(f.data, 1)
        f.data[i, j, k] = _exact(loc, m, i, j, k)
    end
    return f
end

"Max error over the halo cells only (where the prolongation wrote)."
function _halo_err(f)
    loc = Aeolus.Fields.location(f); m = f.mesh
    ii, jj, kk = interior(m)
    e = 0.0
    for k in axes(f.data, 3), j in axes(f.data, 2), i in axes(f.data, 1)
        (i in ii) && (j in jj) && (k in kk) && continue
        e = max(e, abs(f.data[i, j, k] - _exact(loc, m, i, j, k)))
    end
    return e
end

_rstr(r) = string(r)


@testset "Era II — Nesting operators (#250)" begin

# -------------------------------------------------------------------------
# Replaces the Phase 16 `NestSpec` scaffold test. The scaffold asserted a
# verbatim copy (child value = parent value), which holds only for
# refine = 1. Its intent — restricting a hotter child raises the covered
# parent cells to the child's value and leaves the rest alone — is
# re-asserted at r = 3, where the correct statement is "each covered parent
# cell becomes the volume mean of its 27 child cells".
@testset "restricting a hotter child (r = 3): covered cells → child volume mean" begin
    pm = RegularGrid(Float64; size=(8, 8, 8), extent=(1.0, 1.0, 1.0), halo=(2, 2, 2))
    nest = nested_grid(pm, ((0.25, 0.75), (0.25, 0.75), (0.25, 0.75)), 3)
    cm = nest_child_mesh(nest)
    @test (cm.Nx, cm.Ny, cm.Nz) == (12, 12, 12)
    parent = BoussinesqSimulation(pm)
    child  = BoussinesqSimulation(cm)
    set!(parent.θ, (x, y, z) -> 300.0)
    # a hotter child with structure inside each parent cell
    set!(child.θ, (x, y, z) -> 305.0 + 2.0 * sin(2π * x) * cos(2π * y) + z)
    restrict_to_parent!(parent.θ, child.θ, nest)
    cx, cy, cz = nest_covered_range(nest)
    @test (cx, cy, cz) == (3:6, 3:6, 3:6)
    θp = interior_view(parent.θ); θc = interior_view(child.θ)
    maxdev = 0.0
    for K in cz, J in cy, I in cx
        a = (I - first(cx)) * 3; b = (J - first(cy)) * 3; c = (K - first(cz)) * 3
        mean27 = sum(θc[a+1:a+3, b+1:b+3, c+1:c+3]) / 27
        maxdev = max(maxdev, abs(θp[I, J, K] - mean27))
    end
    @printf("  r=3 restriction: max |θ_parent − child volume mean| = %.3e K\n", maxdev)
    @test maxdev < 1e-12
    @test all(>(300.0), θp[cx, cy, cz])     # the covered cells got hotter
    # cells outside the nest are untouched, exactly
    outside = [θp[I, J, K] for K in 1:8, J in 1:8, I in 1:8
               if !((I in cx) && (J in cy) && (K in cz))]
    @test length(outside) == 8^3 - 4^3
    @test all(==(300.0), outside)
end

# -------------------------------------------------------------------------
@testset "field set: u, v, w, θ, qv (moist) and extra scalars" begin
    pm = RegularGrid(Float64; size=(12, 12, 6), extent=(1.0, 1.0, 1.0), halo=(3, 3, 3))
    nest = nested_grid(pm, ((0.25, 0.5), (0.25, 0.5), (0.0, 1.0)), (3, 3, 2); halo=(3, 3, 3))
    cm = nest_child_mesh(nest)
    pdry = BoussinesqSimulation(pm); cdry = BoussinesqSimulation(cm)
    pwet = BoussinesqSimulation(pm; moisture=true)
    cwet = BoussinesqSimulation(cm; moisture=true)
    co2c = Field(CellCenter(), cm); co2p = Field(CellCenter(), pm)

    @test nest_scalar_names(nest_interp_state(nest)) == [:θ]
    @test nest_scalar_names(nest_interp_state(nest, pdry, cdry)) == [:θ]
    s = nest_interp_state(nest, pwet, cwet; scalars=(:co2 => (co2c, co2p),))
    @test nest_scalar_names(s) == [:θ, :qv, :co2]
    @test s[:u] === s.u && s[:θ] === s.θ && s[:theta] === s.θ
    @test size(s[:qv].data) == size(s[:co2].data) == size(cwet.θ.data)
    @test_throws KeyError s[:nope]
    # a nest carries qv on both grids or on neither
    @test_throws ErrorException nest_interp_state(nest, pwet, cdry)
    # reserved, duplicated, mis-located and mis-meshed scalars are refused
    @test_throws ErrorException nest_interp_state(nest; scalars=(:θ => (co2c, co2p),))
    @test_throws ErrorException nest_interp_state(nest;
        scalars=(:a => (co2c, co2p), :a => (co2c, co2p)))
    @test_throws ErrorException nest_interp_state(nest;
        scalars=(:a => (Field(FaceX(), cm), co2p),))
    @test_throws ErrorException nest_interp_state(nest; scalars=(:a => (co2p, co2c),))
end

# -------------------------------------------------------------------------
# A tracer initialised equal to θ must be bit-identical to θ through every
# operator: the field set is iterated with one code path per member, so any
# difference would mean a member is treated differently (a missed halo
# fill, a skipped sponge, a forgotten restriction).
@testset "tracer equivalence: a tracer equal to θ stays bit-identical" begin
    Random.seed!(250_1)
    pm = RegularGrid(Float64; size=(12, 12, 8), extent=(1.0, 1.0, 1.0), halo=(3, 3, 3))
    nest = nested_grid(pm, ((0.25, 0.5), (0.25, 0.75), (0.25, 0.75)), (3, 2, 2);
                       halo=(3, 3, 3))
    cm = nest_child_mesh(nest)
    psim = BoussinesqSimulation(pm; moisture=true)
    csim = BoussinesqSimulation(cm; moisture=true,
                                bcs=nest_bcs(cm; moisture=true, scalars=(:tr,)))
    _randfill!(psim.θ, 300.0, 2.0); _randfill!(csim.θ, 301.0, 2.0)
    _randfill!(psim.qv, 0.008, 1e-3); _randfill!(csim.qv, 0.009, 1e-3)
    trp = _ccopy(psim.θ, pm); trc = _ccopy(csim.θ, cm)
    sc = (:tr => (trc, trp),)
    sa, sb, sbc = (nest_interp_state(nest, psim, csim; scalars=sc) for _ in 1:3)
    same(a, b) = a.data == b.data && all(isequal.(a.data, b.data))

    interpolate_parent_state!(sa, psim, nest; order=3, region=:all)
    @test same(sa[:tr], sa.θ)
    @test same(trp, psim.θ)                   # the parent halo fill matched too
    interpolate_parent_state!(sb, psim, nest; order=4, region=:halo)
    @test same(sb[:tr], sb.θ)
    Aeolus.Nesting._blend_state!(sbc, sa, sb, 0.3)
    @test same(sbc[:tr], sbc.θ)
    apply_nest_boundary!(csim, sbc)
    @test same(trc, csim.θ)
    apply_nest_sponge!(csim, sbc, NestSponge(; width=3, α_max=0.7); dt=0.5)
    @test same(trc, csim.θ)
    restrict_state_to_parent!(psim, csim, nest, sbc)
    @test same(trp, psim.θ)
    preg = _registers(pm, 0.01; closed=true); creg = _registers(cm, 0.01; closed=false)
    done = reflux_scalars!(psim, nest, sbc, Dict(:θ => (preg, creg), :tr => (preg, creg)))
    @test done == [:θ, :tr]
    @test same(trp, psim.θ)
    @test composite_integral(psim, csim, nest, :tr; state=sbc) ===
          composite_integral(psim, csim, nest, :theta)
    # the moist member was carried along, untouched by the tracer
    @test !same(sbc[:qv], sbc.θ)
end

# -------------------------------------------------------------------------
# Composite conservation of every scalar of the set through refluxing.
# The parent (closed box) and the child are advanced with *different*
# random face fluxes Φ = ∫F dt; the child's fluxes through the interface
# disagree with the parent's. Without refluxing the composite integral
# drifts by exactly that disagreement; Berger-Colella refluxing books it
# into the parent cells outside the nest and the composite ∫ is unchanged
# to round-off, for θ, qv and a tracer alike.
@testset "composite ∫θ, ∫qv, ∫tracer conserved by refluxing (r = (2,3,4))" begin
    Random.seed!(250_2)
    pm = RegularGrid(Float64; size=(10, 10, 8), extent=(1.0, 1.0, 0.8), halo=(3, 3, 3))
    nest = nested_grid(pm, ((0.3, 0.6), (0.2, 0.7), (0.2, 0.5)), (2, 3, 4); halo=(3, 3, 3))
    cm = nest_child_mesh(nest)
    psim = BoussinesqSimulation(pm; moisture=true)
    csim = BoussinesqSimulation(cm; moisture=true,
                                bcs=nest_bcs(cm; moisture=true, scalars=(:co2,)))
    trp = Field(CellCenter(), pm); trc = Field(CellCenter(), cm)
    sc = (:co2 => (trc, trp),)
    st = nest_interp_state(nest, psim, csim; scalars=sc)
    base = Dict(:θ => (300.0, 1.0), :qv => (0.01, 1e-3), :co2 => (420.0, 5.0))
    fields = Dict(:θ => (psim.θ, csim.θ), :qv => (psim.qv, csim.qv), :co2 => (trp, trc))
    regs = Dict{Symbol,Any}()
    for (n, (b, a)) in base
        pf, cf = fields[n]
        _randfill!(pf, b, a)
        # child initialised by conservative prolongation of the parent
        interpolate_parent_to_child!(cf, pf, nest; order=3)
        regs[n] = (_registers(pm, 0.05a; closed=true), _registers(cm, 0.05a; closed=false))
    end
    I0 = Dict(n => composite_integral(psim, csim, nest, n; state=st) for n in keys(base))
    # (i) prolongation: the composite ∫ equals the parent-only ∫
    Vp = pm.dx * pm.dy * pm.dz
    for n in keys(base)
        Ip = sum(interior_view(fields[n][1])) * Vp
        rel = abs(I0[n] - Ip) / abs(Ip)
        @printf("  prolongation %-3s: |∫composite − ∫parent| / ∫ = %.2e\n", n, rel)
        @test rel < 1e-13
    end
    # (ii) different coarse and fine fluxes; drift without refluxing
    for n in keys(base)
        pf, cf = fields[n]
        _fv_update!(pf, regs[n][1]); _fv_update!(cf, regs[n][2])
    end
    Imid = Dict(n => composite_integral(psim, csim, nest, n; state=st) for n in keys(base))
    # (iii) restriction + refluxing over the whole set
    restrict_state_to_parent!(psim, csim, nest, st)
    done = reflux_scalars!(psim, nest, st, regs)
    @test sort(done) == sort([:θ, :qv, :co2])
    for n in keys(base)
        I1 = composite_integral(psim, csim, nest, n; state=st)
        drift = abs(Imid[n] - I0[n]) / abs(I0[n])
        rel = abs(I1 - I0[n]) / abs(I0[n])
        @printf("  reflux %-3s: composite drift without refluxing %.2e, with %.2e\n",
                n, drift, rel)
        @test rel < 1e-13
        @test drift > 1e3 * max(rel, 1e-17)       # the test can see a leak
    end
end

# -------------------------------------------------------------------------
@testset "step_nested! carries qv and tracers across the interface" begin
    pm = RegularGrid(Float64; size=(12, 12, 4), extent=(1.0, 1.0, 1.0), halo=(3, 3, 3))
    nest = nested_grid(pm, ((0.25, 0.5), (0.25, 0.5), (0.0, 1.0)), (3, 3, 1); halo=(3, 3, 3))
    cm = nest_child_mesh(nest)
    psim = BoussinesqSimulation(pm; moisture=true)
    csim = BoussinesqSimulation(cm; moisture=true,
                                bcs=nest_bcs(cm, :nest, :nest, :periodic; moisture=true,
                                             scalars=(:co2,)))
    fill!(psim.θ.data, 300.0); fill!(csim.θ.data, 300.0)
    fill!(psim.qv.data, 0.010); fill!(csim.qv.data, 0.0)      # dry child, moist parent
    trp = Field(CellCenter(), pm); trc = Field(CellCenter(), cm)
    fill!(trp.data, 400.0); fill!(trc.data, 0.0)
    # buffer = 0: every covered cell is anterpolated (the default buffer of
    # two parent cells, N-nest-e #263, covers this 3×3 nest entirely)
    step_nested!(psim, csim, nest, 1e-3; scalars=(:co2 => (trc, trp),),
                 feedback=TwoWayFeedback(buffer=0))
    # halos of the child now hold the parent value (qv and tracer at rest)
    ii, jj, kk = interior(cm)
    halo_qv = [csim.qv.data[i, j, k] for i in axes(csim.qv.data, 1), j in jj, k in kk
               if !(i in ii)]
    halo_tr = [trc.data[i, j, k] for i in axes(trc.data, 1), j in jj, k in kk if !(i in ii)]
    @printf("  child x-halo after one nested step: qv ∈ [%.6g, %.6g], co2 ∈ [%.6g, %.6g]\n",
            extrema(halo_qv)..., extrema(halo_tr)...)
    @test maximum(abs, halo_qv .- 0.010) < 1e-15
    @test maximum(abs, halo_tr .- 400.0) < 1e-11
    # the restriction overwrote the covered parent cells with the child mean
    cx, cy, cz = nest_covered_range(nest)
    θc = interior_view(trc)
    @test interior_view(trp)[cx, cy, cz] ≈ fill(sum(θc) / length(θc), length.((cx, cy, cz))) atol=1e-12
    for key in (:mass, :theta, :qv, :co2, :momentum_x)
        @test isfinite(nest_conservation_error(psim, csim, nest, key))
    end
    @test nest_conservation_error(psim, csim, nest, :θ) ==
          nest_conservation_error(psim, csim, nest, :theta)
end

# -------------------------------------------------------------------------
# Any integer refinement per direction (odd, even, mixed). The halo error
# of the prolongation of a smooth field converges at the requested order p:
# the primitive-function reconstruction over p parent cells is exact for
# polynomials of degree p-1 (conservative directions), and so is the nodal
# Lagrange stencil with p nodes (the face-normal direction of a velocity).
@testset "prolongation order: r ∈ {2,3,4,5,(2,2,4)}, p ∈ {2,3,4}, all locations" begin
    rates = Dict{Tuple{Any,Int,String},Float64}()
    for r in (2, 3, 4, 5, (2, 2, 4)), p in (2, 3, 4)
        for (lname, loc) in (("C", CellCenter()), ("u", FaceX()), ("v", FaceY()),
                             ("w", FaceZ()))
            errs = Float64[]
            for Np in (8, 16, 32)
                pm = RegularGrid(Float64; size=(Np, Np, Np), extent=(1.0, 1.0, 1.0),
                                 halo=(4, 4, 4))
                nest = nested_grid(pm, ((0.25, 0.5), (0.5, 0.75), (0.25, 0.5)), r;
                                   halo=(3, 3, 3))
                pf = _fill_exact!(Field(loc, pm))
                cf = Field(loc, nest_child_mesh(nest))
                interpolate_to_nest_boundary!(cf, pf, nest; order=p)
                push!(errs, _halo_err(cf))
            end
            rate = log2(errs[2] / errs[3])
            rates[(r, p, lname)] = rate
            @test rate > p - 0.45
            @test errs[3] < errs[1]
        end
        @printf("  r=%-9s p=%d : measured halo order C %.2f  u %.2f  v %.2f  w %.2f\n",
                _rstr(r), p, (rates[(r, p, l)] for l in ("C", "u", "v", "w"))...)
    end
end

# -------------------------------------------------------------------------
@testset "exact conservation for any r: restrict ∘ prolong = id, face fluxes" begin
    Random.seed!(250_3)
    worst_rp = 0.0; worst_flux = 0.0
    for r in (2, 3, 4, 5, (2, 2, 4), (3, 2, 5)), p in (2, 3, 4)
        pm = RegularGrid(Float64; size=(10, 10, 10), extent=(1.0, 1.0, 1.0), halo=(3, 3, 3))
        nest = nested_grid(pm, ((0.3, 0.6), (0.2, 0.5), (0.4, 0.7)), r; halo=(3, 3, 3))
        cm = nest_child_mesh(nest)
        cov = nest_covered_range(nest)
        Hp = (pm.Hx, pm.Hy, pm.Hz)
        for (ax, loc) in ((0, CellCenter()), (1, FaceX()), (2, FaceY()), (3, FaceZ()))
            pf = Field(loc, pm); pf.data .= 1 .+ randn(size(pf.data))
            cf = Field(loc, cm)
            interpolate_parent_to_child!(cf, pf, nest; order=p)
            back = Field(loc, pm)
            # covered parent cells, plus both interface planes for a face field
            rng = ntuple(d -> Hp[d] .+ (first(cov[d]):last(cov[d]) + (d == ax ? 1 : 0)), 3)
            if ax == 0
                restrict_to_parent!(back, cf, nest)
            else
                restrict_to_parent!(back, cf, nest; include_boundary=true)
                # the summed child fluxes through every parent face of the
                # interface planes equal the parent flux
                mm = compute_nest_flux_mismatch(pf, cf, nest)
                Ap = ax == 1 ? pm.dy * pm.dz : ax == 2 ? pm.dx * pm.dz : pm.dx * pm.dy
                planes = ax == 1 ? (mm.west, mm.east) : ax == 2 ? (mm.south, mm.north) :
                                                         (mm.bottom, mm.top)
                fl = maximum(maximum(abs, P) for P in planes) /
                     (maximum(abs, pf.data[rng...]) * Ap)
                worst_flux = max(worst_flux, fl)
                @test fl < 1e-13
            end
            rel = maximum(abs, back.data[rng...] .- pf.data[rng...]) /
                  maximum(abs, pf.data[rng...])
            worst_rp = max(worst_rp, rel)
            @test rel < 1e-14
        end
    end
    @printf("  random data, all r/p/locations: max |restrict(prolong f) − f|/max|f| = %.2e, max face-flux mismatch / (max|u|·A) = %.2e\n", worst_rp, worst_flux)
end

# -------------------------------------------------------------------------
# A child face that lies on a parent face. In the face-normal direction
# the prolongation is nodal Lagrange, and at a node the Lagrange basis is
# exactly 0 or 1 — so with no tangential refinement the coincident child
# face is the parent face bit for bit; with tangential refinement its
# r×r face-area mean is the parent value to round-off.
@testset "odd-ratio coincident faces (r = 3, 5) reproduce the parent face" begin
    Random.seed!(250_4)
    worst_bit = 0.0; worst_mean = 0.0
    for r in (3, 5), p in (2, 3, 4), (ax, loc) in ((1, FaceX()), (2, FaceY()), (3, FaceZ()))
        pm = RegularGrid(Float64; size=(10, 10, 10), extent=(1.0, 1.0, 1.0), halo=(3, 3, 3))
        ext = ((0.3, 0.6), (0.2, 0.5), (0.4, 0.7))
        Hp = (pm.Hx, pm.Hy, pm.Hz); Hc = (3, 3, 3)
        pf = Field(loc, pm); pf.data .= 300 .+ randn(size(pf.data))
        # (a) refinement only along the face normal: bit for bit
        nest = nested_grid(pm, ext, ntuple(d -> d == ax ? r : 1, 3); halo=Hc)
        cf = Field(loc, nest_child_mesh(nest))
        interpolate_parent_to_child!(cf, pf, nest; order=p)
        nn = (nest.nx, nest.ny, nest.nz); oo = nest_offset(nest)
        psl(q) = ntuple(d -> d == ax ? (Hp[d] + oo[d] + q:Hp[d] + oo[d] + q) :
                                       (Hp[d] + oo[d]:Hp[d] + oo[d] + nn[d] - 1), 3)
        for q in 0:nn[ax]
            csl = ntuple(d -> d == ax ? (Hc[d] + q * r + 1:Hc[d] + q * r + 1) :
                                        (Hc[d] + 1:Hc[d] + nn[d]), 3)
            worst_bit = max(worst_bit,
                            maximum(abs, cf.data[csl...] .- pf.data[psl(q)...]) / 300)
            @test cf.data[csl...] == pf.data[psl(q)...]
        end
        # (b) isotropic r: the r×r mean over each coincident parent face
        nest_r = nested_grid(pm, ext, r; halo=Hc)
        cf_r = Field(loc, nest_child_mesh(nest_r))
        interpolate_parent_to_child!(cf_r, pf, nest_r; order=p)
        mean_back = Field(loc, pm)
        restrict_to_parent!(mean_back, cf_r, nest_r; include_boundary=true)
        for q in 0:nn[ax]
            d = maximum(abs, mean_back.data[psl(q)...] .- pf.data[psl(q)...]) /
                maximum(abs, pf.data[psl(q)...])
            worst_mean = max(worst_mean, d)
            @test d < 1e-15
        end
    end
    @printf("  coincident faces: normal-only refinement max rel diff = %.1e (bitwise); isotropic r×r face mean max rel diff = %.2e\n", worst_bit, worst_mean)
end

# -------------------------------------------------------------------------
# Bug caught by N-nest-c (#257): impose_interface overwrote the low-side
# interior face plane of every velocity component, also along directions
# that are not nest faces. For a child periodic in x the plane i = Hx+1 is
# the child's own interior face, so the parent's u was written into the
# child's interior every step. Since N-nest-d (#261) the boundary face of
# the normal velocity belongs to the OpenBoundary of a nest face, and a
# periodic direction has none: the same assertions hold by construction.
@testset "the boundary face is imposed only on nest faces" begin
    pm = RegularGrid(Float64; size=(8, 8, 8), extent=(1.0, 1.0, 1.0), halo=(3, 3, 3))
    nest = nested_grid(pm, ((0.0, 1.0), (0.0, 1.0), (0.25, 0.75)), 2; halo=(3, 3, 3))
    cm = nest_child_mesh(nest)
    psim = BoussinesqSimulation(pm); fill!(psim.u.data, 2.0); fill!(psim.w.data, 0.5)
    csim = BoussinesqSimulation(cm; bcs=nest_bcs(cm, :periodic, :periodic, :nest))
    @test nest_open_sides(csim) == (false, false, false, false, true, true)
    fill!(csim.u.data, 1.0); fill!(csim.w.data, 0.0)
    st = nest_interp_state(nest, psim, csim)
    interpolate_parent_state!(st, psim, nest; region=:all)
    apply_nest_boundary!(csim, st)
    ii, jj, kk = interior(cm)
    @test all(==(1.0), csim.u.data[first(ii), jj, kk])      # x periodic: untouched
    @test all(==(0.5), csim.w.data[ii, jj, first(kk)])      # z nest: imposed
    @test all(==(0.5), csim.w.data[ii, jj, last(kk) + 1])   # the top face too
    # the tangential velocity and θ halos hold the parent (Dirichlet)
    @test maximum(abs, csim.u.data[ii, jj, first(kk) - 1] .- 2.0) < 1e-14
    # a nest face needs the child mesh (it holds per-face buffers)
    @test_throws ArgumentError nest_bcs(:nest)
    # the run's field set and the child's open faces must agree
    cw = BoussinesqSimulation(cm; bcs=nest_bcs(cm, :periodic, :periodic, :nest;
                                               scalars=(:co2,)))
    @test_throws ArgumentError apply_nest_boundary!(cw, st)
end

# -------------------------------------------------------------------------
# #329 item 2: the refluxing correction of a child face that lies on the
# parent's own boundary goes where the parent's boundary conditions say —
# the wrapped cell across a periodic seam, nowhere across a wall (it used
# to wrap on every axis, so a child touching the parent top booked its top
# correction into the parent's lowest layer).
@testset "flux correction at a child face on the parent boundary (#329)" begin
    pm = RegularGrid(Float64; size=(8, 8, 8), extent=(1.0, 1.0, 1.0), halo=(3, 3, 3))
    @test nest_periodic_axes(BoussinesqSimulation(pm)) == (true, true, true)
    walled = BoussinesqSimulation(pm; bcs = FieldBoundaryConditions(:periodic, :periodic, :wall))
    @test nest_periodic_axes(walled) == (true, true, false)
    zwall = nest_periodic_axes(walled)
    Vp = pm.dx * pm.dy * pm.dz
    # a child touching the parent top
    nt = nested_grid(pm, ((0.25, 0.75), (0.25, 0.75), (0.5, 1.0)), 2)
    @test last(nest_covered_range(nt)[3]) == pm.Nz
    z0 = zeros(nt.nx, nt.ny)
    mt = NestFluxMismatch{Float64}(zeros(0, 0), zeros(0, 0), zeros(0, 0), zeros(0, 0),
                                   copy(z0), ones(nt.nx, nt.ny))
    θ = Field(CellCenter(), pm)
    apply_flux_correction!(θ, mt, nt; periodic = zwall)       # z is not periodic
    @test all(iszero, θ.data)                                  # bottom layer untouched
    apply_flux_correction!(θ, mt, nt)                          # default: triply periodic
    cx, cy, _ = nest_covered_range(nt)
    @test all(interior_view(θ)[cx, cy, 1] .≈ -1 / Vp)          # z-periodic parent: wraps
    @test count(!iszero, θ.data) == nt.nx * nt.ny
    # a child touching the parent's east edge
    ne = nested_grid(pm, ((0.5, 1.0), (0.25, 0.75), (0.25, 0.75)), 2)
    me = NestFluxMismatch{Float64}(zeros(ne.ny, ne.nz), ones(ne.ny, ne.nz), zeros(0, 0),
                                   zeros(0, 0), zeros(0, 0), zeros(0, 0))
    θ = Field(CellCenter(), pm)
    apply_flux_correction!(θ, me, ne; periodic = zwall)       # x-periodic: wraps to i = 1
    _, cy, cz = nest_covered_range(ne)
    @test all(interior_view(θ)[1, cy, cz] .≈ -1 / Vp)
    @test count(!iszero, θ.data) == ne.ny * ne.nz
    θ = Field(CellCenter(), pm)
    apply_flux_correction!(θ, me, ne; periodic = (false, true, false))
    @test all(iszero, θ.data)                                  # a wall: no cell beyond
end

# -------------------------------------------------------------------------
# #329 item 9: the 3rd- and 5th-order prolongation overshoots at a sharp
# scalar front (a negative tracer next to a clean parent cell). The
# limiters scale the children of each parent cell about the parent value:
# :positive (default for scalars) keeps them ≥ 0, :monotone inside the
# range of the parent neighbourhood; both conserve the parent value.
@testset "prolongation limiter on a sharp scalar front (#329)" begin
    pm = RegularGrid(Float64; size=(16, 16, 16), extent=(1.0, 1.0, 1.0), halo=(3, 3, 3))
    nest = nested_grid(pm, ((0.25, 0.75), (0.25, 0.75), (0.25, 0.75)), 2; halo=(3, 3, 3))
    cm = nest_child_mesh(nest)
    pf = Field(CellCenter(), pm)
    for k in axes(pf.data, 3), j in axes(pf.data, 2), i in axes(pf.data, 1)
        x = pm.x0 + (i - pm.Hx - 0.5) * pm.dx; y = pm.y0 + (j - pm.Hy - 0.5) * pm.dy
        pf.data[i, j, k] = (x + 0.3y >= 0.5) ? 1.0 : 0.0     # an oblique front, tracer 0 / 1
    end
    cx, cy, cz = nest_covered_range(nest)
    # the parent value against the mean of its r³ children (child interior)
    function cons(cf)
        worst = 0.0
        for K in 1:nest.nz, J in 1:nest.ny, I in 1:nest.nx
            s = 0.0
            for c in 1:2, b in 1:2, a in 1:2
                s += cf.data[cm.Hx + 2(I-1) + a, cm.Hy + 2(J-1) + b, cm.Hz + 2(K-1) + c]
            end
            p = pf.data[pm.Hx + cx[I], pm.Hy + cy[J], pm.Hz + cz[K]]
            worst = max(worst, abs(s / 8 - p))
        end
        return worst
    end
    for order in (3, 5)
        res = Dict{Symbol,Any}()
        for lim in (:none, :positive, :monotone)
            cf = Field(CellCenter(), cm)
            interpolate_parent_to_child!(cf, pf, nest; order, limiter = lim)
            res[lim] = (extrema(cf.data), cons(cf), cf)
        end
        @printf("  order %d front: range unlimited [%.3f, %.3f], :positive [%.3f, %.3f], :monotone [%.3f, %.3f]; |mean of children − parent| %.1e / %.1e / %.1e\n",
                order, res[:none][1]..., res[:positive][1]..., res[:monotone][1]...,
                res[:none][2], res[:positive][2], res[:monotone][2])
        @test res[:none][1][1] < -0.01                 # the unlimited overshoot is real
        @test res[:none][1][2] > 1.01
        @test res[:positive][1][1] >= 0.0              # positivity
        @test res[:monotone][1][1] >= 0.0 && res[:monotone][1][2] <= 1.0   # no new extrema
        @test res[:positive][2] < 1e-14 && res[:monotone][2] < 1e-14       # conservative
        # the low-level operator does not limit by default (a field may be signed)
        cd = Field(CellCenter(), cm); interpolate_parent_to_child!(cd, pf, nest; order)
        @test cd.data == res[:none][3].data
        # halo-only prolongation is limited the same way
        ch = Field(CellCenter(), cm)
        interpolate_to_nest_boundary!(ch, pf, nest; order, limiter = :positive)
        @test minimum(ch.data) >= 0.0
    end
    # away from zero the positivity limiter is inactive: bit for bit
    sm = Field(CellCenter(), pm)
    for k in axes(sm.data, 3), j in axes(sm.data, 2), i in axes(sm.data, 1)
        sm.data[i, j, k] = 300 + sin(2π * (i - pm.Hx - 0.5) / pm.Nx)
    end
    a = Field(CellCenter(), cm); b = Field(CellCenter(), cm)
    interpolate_parent_to_child!(a, sm, nest; order = 5, limiter = :none)
    interpolate_parent_to_child!(b, sm, nest; order = 5, limiter = :positive)
    @test a.data == b.data
    # the field set of a run: qv and the tracers limited, θ not
    psim = BoussinesqSimulation(pm; moisture = true)
    psim.qv.data .= pf.data .* 0.01; psim.θ.data .= 300 .+ pf.data
    trp = Field(CellCenter(), pm); trp.data .= pf.data
    csim = BoussinesqSimulation(cm; moisture = true)
    st = nest_interp_state(nest, psim, csim; scalars = (:tr => (Field(CellCenter(), cm), trp),))
    interpolate_parent_state!(st, psim, nest; order = 3, region = :all)
    @test minimum(st[:qv].data) >= 0 && minimum(st[:tr].data) >= 0
    @test minimum(st.θ.data) < 300 - 0.01                 # θ: no limiter by default
    interpolate_parent_state!(st, psim, nest; order = 3, region = :all, limit_scalars = :none)
    @test minimum(st[:tr].data) < -0.01
    # velocities: no limiter by default; :monotone clips to the stencil range
    pu = Field(FaceX(), pm); pu.data .= pf.data
    u0 = Field(FaceX(), cm); interpolate_parent_to_child!(u0, pu, nest; order = 3)
    u1 = Field(FaceX(), cm); interpolate_parent_to_child!(u1, pu, nest; order = 3, limiter = :monotone)
    @test minimum(u0.data) < 0 && minimum(u1.data) >= 0 && maximum(u1.data) <= 1
    @test_throws ArgumentError interpolate_parent_to_child!(u1, pu, nest; limiter = :positive)
    @test_throws ArgumentError interpolate_parent_to_child!(a, sm, nest; limiter = :minmod)
end

end # Era II — Nesting operators
