using Test
using Random
using Aeolus
using Aeolus.Nesting
using Aeolus.Grids: interior
using Aeolus.Fields: set!, interior_view
using Aeolus.BoundaryConditions: FieldBoundaryConditions

# =========================================================================
# Phase 30 — parent ↔ child nesting operators (src/Nesting.jl, #250)
# =========================================================================

"Random-fill the interior of `f` with `base .+ amp .* randn`."
function _randfill!(f, base, amp)
    v = interior_view(f)
    v .= base .+ amp .* randn(size(v))
    return f
end

"""
Random face-flux registers `(Φx, Φy, Φz)` on mesh `m`, with zero flux
through the mesh's own boundary faces when `closed`.
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

# smooth separable f = Π_d sin(2π x_d + φ_d) with exact cell / face averages
const _PHI = (0.3, 1.1, 0.7)
_avg1(a, b, φ) = (cos(2π * a + φ) - cos(2π * b + φ)) / (2π * (b - a))
function _exact(loc, m, i, j, k)
    x0 = m.x0 + (i - m.Hx - 1) * m.dx; y0 = m.y0 + (j - m.Hy - 1) * m.dy
    z0 = m.z0 + (k - m.Hz - 1) * m.dz
    fx = loc isa FaceX ? sin(2π * x0 + _PHI[1]) : _avg1(x0, x0 + m.dx, _PHI[1])
    fy = loc isa FaceY ? sin(2π * y0 + _PHI[2]) : _avg1(y0, y0 + m.dy, _PHI[2])
    fz = loc isa FaceZ ? sin(2π * z0 + _PHI[3]) : _avg1(z0, z0 + m.dz, _PHI[3])
    return fx * fy * fz
end
"Max error of the prolongation over the halo cells only."
function _halo_err(f)
    loc = Aeolus.Fields.location(f); m = f.mesh
    ii, jj, kk = interior(m)
    return maximum(abs(f.data[i, j, k] - _exact(loc, m, i, j, k))
                   for k in axes(f.data, 3), j in axes(f.data, 2), i in axes(f.data, 1)
                   if !((i in ii) && (j in jj) && (k in kk)))
end

@testset "Nesting operators (#250)" begin

@testset "restriction (r = 3): covered cells → child volume mean, rest untouched" begin
    pm = RegularGrid(Float64; size=(8, 8, 8), extent=(1.0, 1.0, 1.0), halo=(2, 2, 2))
    nest = nested_grid(pm, ((0.25, 0.75), (0.25, 0.75), (0.25, 0.75)), 3)
    cm = nest_child_mesh(nest)
    θp = Field(CellCenter(), pm); θc = Field(CellCenter(), cm)
    set!(θp, (x, y, z) -> 300.0)
    set!(θc, (x, y, z) -> 305.0 + 2.0 * sin(2π * x) * cos(2π * y) + z)
    restrict_to_parent!(θp, θc, nest)
    cx, cy, cz = nest_covered_range(nest)
    @test (cx, cy, cz) == (3:6, 3:6, 3:6)
    P, C = interior_view(θp), interior_view(θc)
    @test maximum(abs(P[I, J, K] - sum(C[3(I-3)+1:3(I-2), 3(J-3)+1:3(J-2), 3(K-3)+1:3(K-2)]) / 27)
                  for K in cz, J in cy, I in cx) < 1e-12
    @test all(P[I, J, K] == 300.0 for K in 1:8, J in 1:8, I in 1:8
              if !((I in cx) && (J in cy) && (K in cz)))
end

@testset "prolongation order (halo of a smooth field)" begin
    # primitive-function reconstruction over p cells: order p; nodal Lagrange
    # with p nodes in the face-normal direction of a velocity
    for (r, p, loc) in ((3, 3, CellCenter()), ((2, 2, 4), 2, FaceX()))
        errs = map((16, 32)) do Np
            pm = RegularGrid(Float64; size=(Np, Np, Np), extent=(1.0, 1.0, 1.0), halo=(4, 4, 4))
            nest = nested_grid(pm, ((0.25, 0.5), (0.5, 0.75), (0.25, 0.5)), r; halo=(3, 3, 3))
            pf = Field(loc, pm)
            for k in axes(pf.data, 3), j in axes(pf.data, 2), i in axes(pf.data, 1)
                pf.data[i, j, k] = _exact(loc, pm, i, j, k)
            end
            cf = Field(loc, nest_child_mesh(nest))
            interpolate_to_nest_boundary!(cf, pf, nest; order=p)
            _halo_err(cf)
        end
        @test log2(errs[1] / errs[2]) > p - 0.45
    end
end

@testset "exact conservation: restrict ∘ prolong = id, child face fluxes = parent flux" begin
    Random.seed!(250_3)
    pm = RegularGrid(Float64; size=(10, 10, 10), extent=(1.0, 1.0, 1.0), halo=(3, 3, 3))
    nest = nested_grid(pm, ((0.3, 0.6), (0.2, 0.5), (0.4, 0.7)), (3, 2, 5); halo=(3, 3, 3))
    cm = nest_child_mesh(nest)
    cov = nest_covered_range(nest)
    Hp = (pm.Hx, pm.Hy, pm.Hz)
    worst_rp = 0.0; worst_flux = 0.0
    for (ax, loc) in ((0, CellCenter()), (1, FaceX()), (3, FaceZ()))
        pf = Field(loc, pm); pf.data .= 1 .+ randn(size(pf.data))
        cf = Field(loc, cm)
        interpolate_parent_to_child!(cf, pf, nest; order=3)
        back = Field(loc, pm)
        rng = ntuple(d -> Hp[d] .+ (first(cov[d]):last(cov[d]) + (d == ax ? 1 : 0)), 3)
        if ax == 0
            restrict_to_parent!(back, cf, nest)
        else
            restrict_to_parent!(back, cf, nest; include_boundary=true)
            mm = compute_nest_flux_mismatch(pf, cf, nest)
            Ap = ax == 1 ? pm.dy * pm.dz : pm.dx * pm.dy
            planes = ax == 1 ? (mm.west, mm.east) : (mm.bottom, mm.top)
            worst_flux = max(worst_flux, maximum(maximum(abs, P) for P in planes) /
                                         (maximum(abs, pf.data[rng...]) * Ap))
        end
        worst_rp = max(worst_rp, maximum(abs, back.data[rng...] .- pf.data[rng...]) /
                                 maximum(abs, pf.data[rng...]))
    end
    @test worst_rp < 1e-14
    @test worst_flux < 1e-13
end

# Parent (closed box) and child advanced with different random face fluxes:
# without refluxing the composite integral drifts by the interface
# disagreement; Berger-Colella refluxing books it into the parent cells
# outside the nest and the composite ∫ is unchanged to round-off.
@testset "composite ∫θ, ∫qv, ∫tracer conserved by refluxing (r = (2,3,4))" begin
    Random.seed!(250_2)
    pm = RegularGrid(Float64; size=(10, 10, 8), extent=(1.0, 1.0, 0.8), halo=(3, 3, 3))
    nest = nested_grid(pm, ((0.3, 0.6), (0.2, 0.7), (0.2, 0.5)), (2, 3, 4); halo=(3, 3, 3))
    cm = nest_child_mesh(nest)
    psim = BoussinesqSimulation(pm; moisture=true)
    csim = BoussinesqSimulation(cm; moisture=true,
                                bcs=nest_bcs(cm; moisture=true, scalars=(:co2,)))
    trp = Field(CellCenter(), pm); trc = Field(CellCenter(), cm)
    st = nest_interp_state(nest, psim, csim; scalars=(:co2 => (trc, trp),))
    @test nest_scalar_names(st) == [:θ, :qv, :co2]
    base = Dict(:θ => (300.0, 1.0), :qv => (0.01, 1e-3), :co2 => (420.0, 5.0))
    fields = Dict(:θ => (psim.θ, csim.θ), :qv => (psim.qv, csim.qv), :co2 => (trp, trc))
    regs = Dict{Symbol,Any}()
    for (n, (b, a)) in base
        pf, cf = fields[n]
        _randfill!(pf, b, a)
        interpolate_parent_to_child!(cf, pf, nest; order=3)   # conservative prolongation
        regs[n] = (_registers(pm, 0.05a; closed=true), _registers(cm, 0.05a; closed=false))
    end
    I0 = Dict(n => composite_integral(psim, csim, nest, n; state=st) for n in keys(base))
    Vp = pm.dx * pm.dy * pm.dz
    @test maximum(abs(I0[n] - sum(interior_view(fields[n][1])) * Vp) / abs(I0[n])
                  for n in keys(base)) < 1e-13
    for n in keys(base)
        _fv_update!(fields[n][1], regs[n][1]); _fv_update!(fields[n][2], regs[n][2])
    end
    Imid = Dict(n => composite_integral(psim, csim, nest, n; state=st) for n in keys(base))
    restrict_state_to_parent!(psim, csim, nest, st)
    @test sort(reflux_scalars!(psim, nest, st, regs)) == sort([:θ, :qv, :co2])
    rel = Dict(n => abs(composite_integral(psim, csim, nest, n; state=st) - I0[n]) / abs(I0[n])
               for n in keys(base))
    @test maximum(values(rel)) < 1e-13
    # the test can see a leak: without refluxing the drift is far larger
    @test minimum(abs(Imid[n] - I0[n]) / abs(I0[n]) / max(rel[n], 1e-17) for n in keys(base)) > 1e3
end

# #257: impose_interface overwrote the low-side interior face plane along
# directions that are not nest faces (a periodic child got the parent's u
# written into its interior every step)
@testset "the boundary face is imposed only on nest faces (#257)" begin
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
    @test all(==(0.5), csim.w.data[ii, jj, first(kk)])      # z nest: imposed (copied)
end

# #329: the refluxing correction of a child face on the parent's own boundary
# goes across a periodic seam, nowhere across a wall
@testset "flux correction at a child face on the parent boundary (#329)" begin
    pm = RegularGrid(Float64; size=(8, 8, 8), extent=(1.0, 1.0, 1.0), halo=(3, 3, 3))
    walled = BoussinesqSimulation(pm; bcs = FieldBoundaryConditions(:periodic, :periodic, :wall))
    zwall = nest_periodic_axes(walled)
    @test zwall == (true, true, false)
    Vp = pm.dx * pm.dy * pm.dz
    nt = nested_grid(pm, ((0.25, 0.75), (0.25, 0.75), (0.5, 1.0)), 2)   # touches the top
    mt = NestFluxMismatch{Float64}(zeros(0, 0), zeros(0, 0), zeros(0, 0), zeros(0, 0),
                                   zeros(nt.nx, nt.ny), ones(nt.nx, nt.ny))
    θ = Field(CellCenter(), pm)
    apply_flux_correction!(θ, mt, nt; periodic = zwall)
    @test all(iszero, θ.data)                                  # bottom layer untouched
    apply_flux_correction!(θ, mt, nt; periodic = (true, true, true))
    cx, cy, _ = nest_covered_range(nt)
    @test all(interior_view(θ)[cx, cy, 1] .≈ -1 / Vp)          # z-periodic parent: wraps
    @test_throws UndefKeywordError apply_flux_correction!(θ, mt, nt)   # #349: no silent default
end

# #329: the 3rd-order prolongation overshoots at a sharp scalar front; the
# limiters scale the children about the parent value (conservative)
@testset "prolongation limiter on a sharp scalar front (#329)" begin
    pm = RegularGrid(Float64; size=(8, 8, 8), extent=(1.0, 1.0, 1.0), halo=(3, 3, 3))
    nest = nested_grid(pm, ((0.25, 0.75), (0.25, 0.75), (0.25, 0.75)), 2; halo=(3, 3, 3))
    cm = nest_child_mesh(nest)
    pf = Field(CellCenter(), pm)
    for k in axes(pf.data, 3), j in axes(pf.data, 2), i in axes(pf.data, 1)
        x = pm.x0 + (i - pm.Hx - 0.5) * pm.dx; y = pm.y0 + (j - pm.Hy - 0.5) * pm.dy
        pf.data[i, j, k] = (x + 0.3y >= 0.5) ? 1.0 : 0.0     # oblique front, tracer 0 / 1
    end
    cx, cy, cz = nest_covered_range(nest)
    cons(cf) = maximum(abs(sum(cf.data[cm.Hx + 2(I-1) + a, cm.Hy + 2(J-1) + b, cm.Hz + 2(K-1) + c]
                               for c in 1:2, b in 1:2, a in 1:2) / 8 -
                           pf.data[pm.Hx + cx[I], pm.Hy + cy[J], pm.Hz + cz[K]])
                       for K in 1:nest.nz, J in 1:nest.ny, I in 1:nest.nx)
    res = Dict(map((:none, :positive, :monotone)) do lim
        cf = Field(CellCenter(), cm)
        interpolate_parent_to_child!(cf, pf, nest; order = 3, limiter = lim)
        lim => (extrema(cf.data), cons(cf))
    end)
    @test res[:none][1][1] < -0.01                 # the unlimited overshoot is real
    @test res[:positive][1][1] >= 0.0              # positivity
    @test res[:monotone][1][1] >= 0.0 && res[:monotone][1][2] <= 1.0   # no new extrema
    @test max(res[:positive][2], res[:monotone][2]) < 1e-14            # conservative
end

end # Nesting operators
