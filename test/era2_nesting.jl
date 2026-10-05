using Test
using Printf
using Random
using Aeolus
using Aeolus.Nesting
using Aeolus.Grids: interior
using Aeolus.Fields: set!, interior_view

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
    csim = BoussinesqSimulation(cm; moisture=true, bcs=nest_bcs(:nest))
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
    csim = BoussinesqSimulation(cm; moisture=true, bcs=nest_bcs(:nest))
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
    csim = BoussinesqSimulation(cm; moisture=true, bcs=nest_bcs(:nest, :nest, :periodic),
                                solver=Aeolus.PressureSolvers.FFTPoissonSolver(cm))
    fill!(psim.θ.data, 300.0); fill!(csim.θ.data, 300.0)
    fill!(psim.qv.data, 0.010); fill!(csim.qv.data, 0.0)      # dry child, moist parent
    trp = Field(CellCenter(), pm); trc = Field(CellCenter(), cm)
    fill!(trp.data, 400.0); fill!(trc.data, 0.0)
    step_nested!(psim, csim, nest, 1e-3; scalars=(:co2 => (trc, trp),))
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

end # Era II — Nesting operators
