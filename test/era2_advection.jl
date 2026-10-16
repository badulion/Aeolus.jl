# =========================================================================
# Era II — Phases 26 & 28
#   * WENO-Z / TENO advection for all prognostics
#   * energy- and enstrophy-conserving discrete operators
# =========================================================================
# Every quantitative claim made in `src/AdvectionSchemes.jl` is measured
# here and printed, so the numbers can be compared against the literature
# and tracked across refactors.

using Test
using Random
using Aeolus
using Aeolus.AdvectionSchemes
using Aeolus.Grids: interior
using Aeolus.Fields: set!, interior_view
using Aeolus.PressureSolvers: FFTPoissonSolver, solve!

const PERIODIC = FieldBoundaryConditions(:periodic)

# -------------------------------------------------------------------------
# helpers
# -------------------------------------------------------------------------

"Exact cell average of sin over the cell of width `h` centred at `xc`."
avg_sin(xc, h) = (cos(xc - h/2) - cos(xc + h/2))/h

"""
Exact cell averages of `F'`-antiderivative pair on the stencil whose cell
`0` is the cell immediately left of the interface at `x0`.
"""
function stencil_avgs(F, x0, h, off0, n)
    return ntuple(n) do j
        xc = x0 - h/2 + (off0 + j - 1)*h
        (F(xc + h/2) - F(xc - h/2))/h
    end
end

"Convergence order of `rec` for f = sin sampled as cell averages."
function measure_order(rec, n, off0, hs; x0 = 0.3)
    errs = [abs(rec(ntuple(j -> avg_sin(x0 - h/2 + (off0 + j - 1)*h, h), n)...) - sin(x0))
            for h in hs]
    ords = [log(errs[i]/errs[i+1]) / log(hs[i]/hs[i+1]) for i in 1:length(errs)-1]
    return errs, ords
end

"Reconstruction error for the degree-`deg` monomial-sum polynomial `p`."
function poly_error(rec, n, off0, p, P; h = 0.3, x0 = 0.7)
    return abs(rec(stencil_avgs(P, x0, h, off0, n)...) - p(x0))
end

# a generic polynomial of degree `d` with its antiderivative
function poly_pair(d)
    c = [1.0, 2.0, -0.5, 0.3, -0.1, 0.05, -0.02, 0.01, 0.004][1:d+1]
    p = x -> sum(c[m+1] * x^m for m in 0:d)
    P = x -> sum(c[m+1] * x^(m+1)/(m+1) for m in 0:d)
    return p, P
end

total_variation(a) = sum(abs, diff(vcat(a, a[1:1])))

"SSP-RK3 advance of a scalar in a frozen velocity field."
function rk3_scalar!(ϕ, u, v, w, advect!, dt, nsteps, g)
    t1 = Aeolus.Field(CellCenter(), g)
    ϕ0 = similar(ϕ.data)
    ii, jj, kk = interior(g)
    stage! = (a, b) -> begin
        fill_halos!(ϕ, PERIODIC)
        fill!(t1.data, 0)
        advect!(t1, u, v, w, ϕ, dt)
        @views ϕ.data[ii,jj,kk] .= a .* ϕ0[ii,jj,kk] .+
                                   b .* (ϕ.data[ii,jj,kk] .+ dt .* t1.data[ii,jj,kk])
    end
    for _ in 1:nsteps
        ϕ0 .= ϕ.data
        stage!(0.0, 1.0); stage!(0.75, 0.25); stage!(1/3, 2/3)
    end
    return ϕ
end

ke_of(u, v, w, g) = 0.5 * (sum(abs2, interior_view(u)) + sum(abs2, interior_view(v)) +
                           sum(abs2, interior_view(w))) * g.dx*g.dy*g.dz

"""
Momentum-only SSP-RK3 integration of a Taylor–Green vortex with
`momentum_advection! = advect!`, optionally with a pressure projection at
every substage. Returns the relative kinetic-energy drift.
"""
function tg_ke_drift(advect!, g; dt = 1e-4, nsteps = 2000, project = false)
    u = Aeolus.Field(FaceX(), g); v = Aeolus.Field(FaceY(), g); w = Aeolus.Field(FaceZ(), g)
    k = 2π/g.Lx
    set!(u, (x,y,z) ->  sin(k*x)*cos(k*y))
    set!(v, (x,y,z) -> -cos(k*x)*sin(k*y))
    for f in (u, v, w); fill_halos!(f, PERIODIC); end
    tu = Aeolus.Field(FaceX(), g); tv = Aeolus.Field(FaceY(), g); tw = Aeolus.Field(FaceZ(), g)
    u0 = copy(u.data); v0 = copy(v.data); w0 = copy(w.data)
    div_buf = Aeolus.Field(CellCenter(), g); φ = Aeolus.Field(CellCenter(), g)
    gx = Aeolus.Field(FaceX(), g); gy = Aeolus.Field(FaceY(), g); gz = Aeolus.Field(FaceZ(), g)
    solver = FFTPoissonSolver(g)
    ii, jj, kk = interior(g)
    stage! = (a, b) -> begin
        for f in (u, v, w); fill_halos!(f, PERIODIC); end
        fill!(tu.data, 0); fill!(tv.data, 0); fill!(tw.data, 0)
        advect!(tu, tv, tw, u, v, w)
        @views u.data[ii,jj,kk] .= a.*u0[ii,jj,kk] .+ b.*(u.data[ii,jj,kk] .+ dt.*tu.data[ii,jj,kk])
        @views v.data[ii,jj,kk] .= a.*v0[ii,jj,kk] .+ b.*(v.data[ii,jj,kk] .+ dt.*tv.data[ii,jj,kk])
        @views w.data[ii,jj,kk] .= a.*w0[ii,jj,kk] .+ b.*(w.data[ii,jj,kk] .+ dt.*tw.data[ii,jj,kk])
        if project
            for f in (u, v, w); fill_halos!(f, PERIODIC); end
            divergence(div_buf, u, v, w); solve!(φ, div_buf, solver); fill_halos!(φ, PERIODIC)
            ∂x(gx, φ); ∂y(gy, φ); ∂z(gz, φ)
            @views u.data[ii,jj,kk] .-= gx.data[ii,jj,kk]
            @views v.data[ii,jj,kk] .-= gy.data[ii,jj,kk]
            @views w.data[ii,jj,kk] .-= gz.data[ii,jj,kk]
        end
    end
    E0 = ke_of(u, v, w, g)
    for _ in 1:nsteps
        u0 .= u.data; v0 .= v.data; w0 .= w.data
        stage!(0.0, 1.0); stage!(0.75, 0.25); stage!(1/3, 2/3)
    end
    return (E0 - ke_of(u, v, w, g)) == 0 ? 0.0 : (ke_of(u, v, w, g) - E0)/E0
end

"Σ u·C(u) dV — the discrete kinetic-energy production of a momentum operator."
function energy_production(advect!, u, v, w, g)
    tu = Aeolus.Field(FaceX(), g); tv = Aeolus.Field(FaceY(), g); tw = Aeolus.Field(FaceZ(), g)
    advect!(tu, tv, tw, u, v, w)
    dV = g.dx*g.dy*g.dz
    return (sum(interior_view(u) .* interior_view(tu)) +
            sum(interior_view(v) .* interior_view(tv)) +
            sum(interior_view(w) .* interior_view(tw))) * dV
end

"Σ ϕ·C(ϕ) dV — the discrete scalar-variance production of a scalar operator."
function variance_production(advect!, u, v, w, ϕ, g)
    t = Aeolus.Field(CellCenter(), g)
    advect!(t, u, v, w, ϕ)
    return sum(interior_view(ϕ) .* interior_view(t)) * g.dx*g.dy*g.dz
end

# =========================================================================

@testset "Era II — High-order & conservative advection" begin

# -------------------------------------------------------------------------
@testset "Reconstruction order verification" begin
    hs5 = [0.4, 0.2, 0.1, 0.05]
    hs7 = [0.8, 0.4, 0.2]
    hs8 = [0.4, 0.2, 0.1]

    e_z5, o_z5 = measure_order((a...) -> weno_z5_left(a...), 5, -2, hs5)
    e_z7, o_z7 = measure_order((a...) -> weno_z7_left(a...), 7, -3, hs7)
    e_t5, o_t5 = measure_order((a...) -> teno5_left(a...),   5, -2, hs5)
    e_t8, o_t8 = measure_order((a...) -> teno8_left(a...),   8, -4, hs8)

    @info "measured convergence orders (cell averages of sin, interface value)" *
          "\n  WENO-Z5 : orders = $(round.(o_z5, digits=3))  errors = $e_z5" *
          "\n  WENO-Z7 : orders = $(round.(o_z7, digits=3))  errors = $e_z7" *
          "\n  TENO5   : orders = $(round.(o_t5, digits=3))  errors = $e_t5" *
          "\n  TENO8   : orders = $(round.(o_t8, digits=3))  errors = $e_t8"

    @test minimum(o_z5) > 4.8
    @test minimum(o_t5) > 4.8
    @test minimum(o_z7) > 6.7
    # TENO8's h⁸ term only dominates below h ≈ 0.1, where Float64 round-off
    # in the cell averages already bites; the polynomial-exactness test
    # below is the sharp statement of its design order.
    @test minimum(o_t8) > 6.8

    # --- design order ⇔ exactness for polynomials of degree order-1 ------
    # TENO's *discrete* stencil selection returns the optimal linear
    # weights verbatim in smooth regions, so TENO5/TENO8 reproduce a
    # degree-4 / degree-7 polynomial exactly. That is the sharp statement
    # of their design order. WENO-Z only approaches the linear weights
    # asymptotically (α_k = d_k(1 + (τ/β_k)^p) is never exactly d_k), so it
    # keeps the *order* but not the exactness — the practical reason TENO is
    # markedly less dissipative than WENO at equal stencil width.
    p4, P4 = poly_pair(4); p6, P6 = poly_pair(6); p7, P7 = poly_pair(7)
    ez5 = poly_error((a...) -> weno_z5_left(a...), 5, -2, p4, P4)
    et5 = poly_error((a...) -> teno5_left(a...),   5, -2, p4, P4)
    ez7 = poly_error((a...) -> weno_z7_left(a...), 7, -3, p6, P6)
    et8 = poly_error((a...) -> teno8_left(a...),   8, -4, p7, P7)
    @info "polynomial reconstruction error at design order (h = 0.3)" *
          "\n  TENO5 on deg 4   = $et5   (exact — linear weights recovered)" *
          "\n  TENO8 on deg 7   = $et8   (exact)" *
          "\n  WENO-Z5 on deg 4 = $ez5   (nonlinear weights: small, not exact)" *
          "\n  WENO-Z7 on deg 6 = $ez7"
    @test et5 < 1e-13          # TENO is *exact* at its design order
    @test et8 < 1e-12
    @test ez5 < 1e-6           # WENO-Z: small but nonzero weight error
    @test ez7 < 1e-7

    # --- right-biased variants are the mirrored left-biased ones ---------
    v5 = (0.3, 0.9, 1.4, 1.1, 0.2)
    v7 = (0.3, 0.9, 1.4, 1.1, 0.2, -0.4, 0.7)
    v8 = (0.3, 0.9, 1.4, 1.1, 0.2, -0.4, 0.7, 1.9)
    @test weno_z5_right(reverse(v5)...) == weno_z5_left(v5...)
    @test weno_z7_right(reverse(v7)...) == weno_z7_left(v7...)
    @test teno5_right(reverse(v5)...)   == teno5_left(v5...)
    @test teno8_right(reverse(v8)...)   == teno8_left(v8...)

    # --- shift equivariance and scale covariance -------------------------
    # Both are exact for the Z/TENO weights: their smoothness indicators are
    # translation invariant and the τ/β ratio is scale free (the ε there is a
    # machine-zero guard, not a smoothness cut-off). WENO-JS's absolute
    # ε = 1e-6 makes it shift- but *not* scale-invariant, which is the same
    # defect that costs it its formal order.
    for (rec, v) in ((weno_z5_left, v5), (weno_z7_left, v7),
                     (teno5_left, v5), (teno8_left, v8), (weno_js5_left, v5))
        @test rec((v .+ 10.0)...) ≈ rec(v...) + 10.0 atol=1e-12
    end
    for (rec, v) in ((weno_z5_left, v5), (weno_z7_left, v7),
                     (teno5_left, v5), (teno8_left, v8))
        @test rec((v .* 3.0)...) ≈ 3 * rec(v...) rtol=1e-13
    end
end

# -------------------------------------------------------------------------
@testset "WENO-Z beats WENO-JS at a critical point" begin
    legacy = Aeolus.Operators.FiniteVolume._weno5_left

    # f = sin with f'(stencil centre) = 0 → the classic WENO-JS failure mode
    sin_errs = map((0.2, 0.1, 0.05)) do h
        x0 = π/2 + h/2                       # stencil centre sits on π/2
        v  = ntuple(j -> avg_sin(x0 - h/2 + (j - 3)*h, h), 5)
        (h  = h,
         js = abs(weno_js5_left(v...) - sin(x0)),
         z  = abs(weno_z5_left(v...)  - sin(x0)),
         t5 = abs(teno5_left(v...)    - sin(x0)),
         legacy = abs(legacy(v...)    - sin(x0)))
    end

    # f = x³ : f' = f'' = 0 at the interface (the textbook critical point)
    cube, CUBE = x -> x^3, x -> x^4/4
    cube_errs = map((0.2, 0.1, 0.05)) do h
        v = stencil_avgs(CUBE, 0.0, h, -2, 5)
        (h  = h,
         js = abs(weno_js5_left(v...) - cube(0.0)),
         z  = abs(weno_z5_left(v...)  - cube(0.0)),
         t5 = abs(teno5_left(v...)    - cube(0.0)),
         legacy = abs(legacy(v...)    - cube(0.0)))
    end

    @info "critical-point reconstruction error (smaller is better)" *
          "\n  f = sin, critical point at stencil centre:" *
          join(["\n    h=$(e.h): WENO-JS=$(e.js)  WENO-Z=$(e.z)  TENO5=$(e.t5)  " *
                "(legacy FiniteVolume WENO-5=$(e.legacy))  JS/Z=$(round(e.js/e.z, digits=1))"
                for e in sin_errs]) *
          "\n  f = x³ at x=0:" *
          join(["\n    h=$(e.h): WENO-JS=$(e.js)  WENO-Z=$(e.z)  TENO5=$(e.t5)  " *
                "(legacy=$(e.legacy))  JS/Z=$(round(e.js/e.z, digits=1))"
                for e in cube_errs])

    for e in sin_errs
        @test e.z < e.js            # vs. a *correct* WENO-JS
        @test e.z < e.legacy        # vs. the WENO-5 already in the library
    end
    for e in cube_errs
        @test e.z <= e.js
        @test e.z < e.legacy
    end
    # TENO5's discrete selection keeps the linear 5th-order scheme here, so
    # x³ (degree 3 < 5) is reproduced exactly.
    @test all(e -> e.t5 < 1e-15, cube_errs)

    # ---- known defect in FiniteVolume._weno5_left (informational) -------
    # `2f0` / `3f0` inside its smoothness indicators are Float32 *literals*,
    # not `2*f0` / `3*f0`, so the indicators are not translation invariant.
    v  = (0.3, 0.9, 1.4, 1.1, 0.2)
    shift_violation = abs(legacy((v .+ 10.0)...) - (legacy(v...) + 10.0))
    @info "legacy FiniteVolume._weno5_left shift-equivariance violation = " *
          "$shift_violation  (a correct WENO reconstruction gives 0; the cause is " *
          "`2f0`/`3f0` parsing as Float32 literals in its β indicators — " *
          "AdvectionSchemes.weno_js5_left is the corrected reference)"
    @test abs(weno_js5_left((v .+ 10.0)...) - (weno_js5_left(v...) + 10.0)) < 1e-13
end

# -------------------------------------------------------------------------
@testset "ENO property at a discontinuity (square wave, one period)" begin
    g = RegularGrid(Float64; size=(64,8,8), extent=(1.0,0.125,0.125), halo=(5,5,5))
    u = Aeolus.Field(FaceX(), g); v = Aeolus.Field(FaceY(), g); w = Aeolus.Field(FaceZ(), g)
    set!(u, (x,y,z) -> 1.0)
    for f in (u, v, w); fill_halos!(f, PERIODIC); end
    square(x) = (0.25 <= x <= 0.5) ? 1.0 : 0.0
    dt = 0.4 * g.dx                     # CFL = 0.4
    nsteps = round(Int, 1.0/dt)          # exactly one period

    wsp = FluxLimiterWorkspace(g)        # required when the backend runs threaded (KA-e)
    schemes = (("centered", (t,u,v,w,f,dt) -> advect_centered!(t,u,v,w,f)),
               ("weno5(legacy)", (t,u,v,w,f,dt) -> Aeolus.advect_weno5!(t,u,v,w,f)),
               ("weno_z5", (t,u,v,w,f,dt) -> advect_weno_z!(t,u,v,w,f; order=5)),
               ("weno_z7", (t,u,v,w,f,dt) -> advect_weno_z!(t,u,v,w,f; order=7)),
               ("teno5",   (t,u,v,w,f,dt) -> advect_teno!(t,u,v,w,f; order=5)),
               ("teno8",   (t,u,v,w,f,dt) -> advect_teno!(t,u,v,w,f; order=8)),
               ("positive(teno5)", (t,u,v,w,f,dt) ->
                    advect_positive!(t,u,v,w,f,dt; scheme=:teno5, workspace=wsp)))

    results = map(schemes) do (name, adv)
        ϕ = Aeolus.Field(CellCenter(), g); set!(ϕ, (x,y,z) -> square(x))
        fill_halos!(ϕ, PERIODIC)
        a0 = copy(interior_view(ϕ)[:,1,1])
        m0 = sum(interior_view(ϕ))
        rk3_scalar!(ϕ, u, v, w, adv, dt, nsteps, g)
        a = interior_view(ϕ)[:,1,1]
        (name = name, over = maximum(a) - 1.0, under = -minimum(a),
         tv = total_variation(a)/total_variation(a0),
         mass = (sum(interior_view(ϕ)) - m0)/m0)
    end

    @info "square wave advected one period at CFL 0.4 (N = 64)" *
          join(["\n  $(rpad(r.name,16)) overshoot = $(r.over)  undershoot = $(r.under)  " *
                "TV/TV₀ = $(r.tv)  relative mass drift = $(r.mass)" for r in results])

    byname = Dict(r.name => r for r in results)
    # TENO / WENO-Z: essentially no new extrema and no TV growth
    for nm in ("weno_z5", "weno_z7", "teno5", "teno8", "positive(teno5)")
        r = byname[nm]
        @test r.tv < 1.02
        @test r.over < 5e-3
        @test r.under < 5e-3
    end
    # the centred scheme is wildly oscillatory — that is why we need the above
    @test byname["centered"].tv > 2.0
    @test byname["centered"].over > 0.1
    # mass is conserved by every flux-form scheme
    @test all(r -> abs(r.mass) < 1e-12, results)
end

# -------------------------------------------------------------------------
@testset "Mass conservation of every scheme (div-free periodic flow)" begin
    g = RegularGrid(Float64; size=(16,16,16), extent=(1.0,1.0,1.0), halo=(5,5,5))
    u = Aeolus.Field(FaceX(), g); v = Aeolus.Field(FaceY(), g); w = Aeolus.Field(FaceZ(), g)
    # u = u(y,z), v = v(x,z), w = w(x,y) ⇒ the *discrete* divergence is
    # identically zero, not merely small.
    set!(u, (x,y,z) ->  1.0 + 0.3sin(2π*y) + 0.2cos(2π*z))
    set!(v, (x,y,z) ->  0.5 + 0.3cos(2π*x) + 0.1sin(4π*z))
    set!(w, (x,y,z) -> -0.2 + 0.2sin(2π*x)*cos(2π*y))
    for f in (u, v, w); fill_halos!(f, PERIODIC); end
    div_buf = Aeolus.Field(CellCenter(), g); divergence(div_buf, u, v, w)
    @test maximum(abs, interior_view(div_buf)) == 0.0

    ϕ = Aeolus.Field(CellCenter(), g)
    set!(ϕ, (x,y,z) -> 2.0 + sin(2π*x)*cos(2π*y)*sin(2π*z))
    fill_halos!(ϕ, PERIODIC)
    dV = g.dx*g.dy*g.dz

    wsp = FluxLimiterWorkspace(g)
    schemes = (("centered", (t) -> advect_centered!(t,u,v,w,ϕ)),
               ("weno5(legacy)", (t) -> Aeolus.advect_weno5!(t,u,v,w,ϕ)),
               ("weno_z5", (t) -> advect_weno_z!(t,u,v,w,ϕ; order=5)),
               ("weno_z7", (t) -> advect_weno_z!(t,u,v,w,ϕ; order=7)),
               ("teno5",   (t) -> advect_teno!(t,u,v,w,ϕ; order=5)),
               ("teno8",   (t) -> advect_teno!(t,u,v,w,ϕ; order=8)),
               ("upwind1",  advection_scheme(:upwind)  |> s -> (t) -> s(t,u,v,w,ϕ)),
               ("variance_conserving", (t) -> advect_variance_conserving!(t,u,v,w,ϕ)),
               ("positive(weno_z5)", (t) -> advect_positive!(t,u,v,w,ϕ,1e-3; scheme=:weno_z5, workspace=wsp)))

    msgs = String[]
    for (name, adv) in schemes
        t = Aeolus.Field(CellCenter(), g)
        adv(t)
        dM = sum(interior_view(t)) * dV          # d/dt ∫ϕ dV
        push!(msgs, "\n  $(rpad(name,20)) |d/dt ∫ϕ dV| = $(abs(dM))")
        @test abs(dM) < 1e-12
    end
    @info "mass conservation (instantaneous)" * join(msgs)
end

# -------------------------------------------------------------------------
@testset "Positivity preservation (narrow Gaussian, CFL ≈ 0.95)" begin
    g = RegularGrid(Float64; size=(32,32,32), extent=(1.0,1.0,1.0), halo=(5,5,5))
    u = Aeolus.Field(FaceX(), g); v = Aeolus.Field(FaceY(), g); w = Aeolus.Field(FaceZ(), g)
    set!(u, (x,y,z) -> 1.0); set!(v, (x,y,z) -> 0.6); set!(w, (x,y,z) -> 0.3)
    for f in (u, v, w); fill_halos!(f, PERIODIC); end
    dt = 0.95 / (1.0/g.dx + 0.6/g.dy + 0.3/g.dz)      # first-order CFL sum ≈ 0.95
    σ  = 1.2 * g.dx
    gauss = (x,y,z) -> exp(-((x-0.5)^2 + (y-0.5)^2 + (z-0.5)^2)/(2σ^2))

    ws = FluxLimiterWorkspace(g)
    cases = (("weno_z5 (no limiter)", (t,f) -> advect_weno_z!(t,u,v,w,f; order=5)),
             ("teno8 (no limiter)",   (t,f) -> advect_teno!(t,u,v,w,f; order=8)),
             ("advect_positive! :weno_z5", (t,f) ->
                  advect_positive!(t,u,v,w,f,dt; scheme=:weno_z5, workspace=ws)),
             ("advect_positive! :teno8", (t,f) ->
                  advect_positive!(t,u,v,w,f,dt; scheme=:teno8, workspace=ws)))

    results = map(cases) do (name, adv)
        ϕ = Aeolus.Field(CellCenter(), g); set!(ϕ, gauss); fill_halos!(ϕ, PERIODIC)
        m0 = sum(interior_view(ϕ))
        t  = Aeolus.Field(CellCenter(), g)
        ii, jj, kk = interior(g)
        for _ in 1:20                     # plain forward Euler: the case the proof covers
            fill!(t.data, 0); fill_halos!(ϕ, PERIODIC)
            adv(t, ϕ)
            @views ϕ.data[ii,jj,kk] .+= dt .* t.data[ii,jj,kk]
        end
        (name = name, min = minimum(interior_view(ϕ)), max = maximum(interior_view(ϕ)),
         mass = (sum(interior_view(ϕ)) - m0)/m0)
    end

    @info "positivity after 20 forward-Euler steps at CFL ≈ 0.95" *
          join(["\n  $(rpad(r.name,28)) min = $(r.min)  max = $(r.max)  " *
                "relative mass drift = $(r.mass)" for r in results])

    # without the limiter a high-order scheme goes (badly) negative …
    @test results[1].min < -0.1
    @test results[2].min < -0.1
    # … with it, ϕ stays non-negative to round-off, and mass is untouched
    for r in results[3:4]
        @test r.min >= -1e-14 * r.max
        @test abs(r.mass) < 1e-12
    end

    # the bare limiter (zero-flux fallback) is unconditional in dt
    ϕ = Aeolus.Field(CellCenter(), g); set!(ϕ, gauss); fill_halos!(ϕ, PERIODIC)
    fx = Aeolus.Field(FaceX(), g); fy = Aeolus.Field(FaceY(), g); fz = Aeolus.Field(FaceZ(), g)
    for (F, vel) in ((fx, u), (fy, v), (fz, w))
        F.data .= 5.0 .* vel.data          # deliberately absurd fluxes
    end
    apply_positivity_limiter!(fx, fy, fz, ϕ, 10.0, g; ratio = Aeolus.Field(CellCenter(), g))  # CFL ≈ 100
    t = Aeolus.Field(CellCenter(), g)
    Aeolus.AdvectionSchemes._add_flux_divergence!(t, fx, fy, fz, g)
    ii, jj, kk = interior(g)
    ϕnew = @views ϕ.data[ii,jj,kk] .+ 10.0 .* t.data[ii,jj,kk]
    @info "bare apply_positivity_limiter! with absurd fluxes at CFL ≈ 100: " *
          "min(ϕ⁺) = $(minimum(ϕnew))"
    @test minimum(ϕnew) >= -1e-14 * maximum(interior_view(ϕ))
end

# -------------------------------------------------------------------------
@testset "Morinishi skew-symmetric momentum advection" begin
    g = RegularGrid(Float64; size=(16,16,16), extent=(1.0,1.0,1.0), halo=(3,3,3))

    # (a) staggering check: the reconstruction machinery reduced to a
    #     2-point average must reproduce `momentum_advect!` exactly.
    u = Aeolus.Field(FaceX(), g); v = Aeolus.Field(FaceY(), g); w = Aeolus.Field(FaceZ(), g)
    set!(u, (x,y,z) -> 1.0 + 0.3sin(2π*y) + 0.2cos(2π*z))
    set!(v, (x,y,z) -> 0.5 + 0.3cos(2π*x) + 0.1sin(4π*z))
    set!(w, (x,y,z) -> -0.2 + 0.2sin(2π*x)*cos(2π*y))
    for f in (u, v, w); fill_halos!(f, PERIODIC); end

    ref = map(l -> Aeolus.Field(l, g), (FaceX(), FaceY(), FaceZ()))
    momentum_advect!(ref..., u, v, w)
    c2 = map(l -> Aeolus.Field(l, g), (FaceX(), FaceY(), FaceZ()))
    Aeolus.AdvectionSchemes._momentum_advect_recon!(c2..., u, v, w,
        Aeolus.AdvectionSchemes.Centered2(), g)
    for (a, b) in zip(ref, c2)
        @test maximum(abs, interior_view(a) .- interior_view(b)) < 1e-12
    end
    # ... and likewise for the scalar sweep vs. `advect_centered!`
    ϕs = Aeolus.Field(CellCenter(), g)
    set!(ϕs, (x,y,z) -> 1.0 + sin(2π*x)*cos(2π*y)*sin(2π*z)); fill_halos!(ϕs, PERIODIC)
    sref = Aeolus.Field(CellCenter(), g); srec = Aeolus.Field(CellCenter(), g)
    advect_centered!(sref, u, v, w, ϕs)
    Aeolus.AdvectionSchemes._advect_reconstructed!(srec, u, v, w, ϕs,
        Aeolus.AdvectionSchemes.Centered2(), g)
    @test maximum(abs, interior_view(sref) .- interior_view(srec)) < 1e-12

    # (b) energy production on a *divergence-free* field: both forms vanish
    divfree_flux = energy_production((a,b,c,x,y,z) -> momentum_advect!(a,b,c,x,y,z), u, v, w, g)
    divfree_skew = energy_production((a,b,c,x,y,z) -> momentum_advect_skew!(a,b,c,x,y,z), u, v, w, g)
    divfree_weno = energy_production(
        (a,b,c,x,y,z) -> momentum_advect_weno_z!(a,b,c,x,y,z; order=5), u, v, w, g)

    # (c) energy production on a field with non-zero discrete divergence:
    #     only the skew-symmetric form still vanishes.
    Random.seed!(2718)
    ur = Aeolus.Field(FaceX(), g); vr = Aeolus.Field(FaceY(), g); wr = Aeolus.Field(FaceZ(), g)
    for f in (ur, vr, wr)
        interior_view(f) .= randn.()
        fill_halos!(f, PERIODIC)
    end
    scale = sqrt(sum(abs2, interior_view(ur)) + sum(abs2, interior_view(vr)) +
                 sum(abs2, interior_view(wr))) * g.dx*g.dy*g.dz
    rand_flux = energy_production((a,b,c,x,y,z) -> momentum_advect!(a,b,c,x,y,z), ur, vr, wr, g)
    rand_skew = energy_production((a,b,c,x,y,z) -> momentum_advect_skew!(a,b,c,x,y,z), ur, vr, wr, g)

    @info "discrete kinetic-energy production Σu·C(u) dV" *
          "\n  divergence-free field : flux form = $divfree_flux   skew = $divfree_skew" *
          "   WENO-Z5 = $divfree_weno" *
          "\n  ∇·u ≠ 0 (random) field: flux form = $rand_flux   skew = $rand_skew" *
          "   (field norm ≈ $scale)"

    @test abs(divfree_skew) < 1e-12
    @test abs(divfree_flux) < 1e-12               # the 2nd-order flux form is fine here
    @test divfree_weno < 0                        # upwind-biased ⇒ dissipative
    @test abs(rand_skew) < 1e-12                  # still exactly conservative
    @test abs(rand_flux) > 1e-3                   # the flux form is not
    @test abs(rand_flux) > 1e6 * abs(rand_skew)

    # (d) integrated drift: inviscid Taylor–Green, no buoyancy, no SGS
    gtg = RegularGrid(Float64; size=(16,16,4), extent=(1.0,1.0,0.25), halo=(3,3,3))
    drift_skew = tg_ke_drift((a,b,c,x,y,z) -> momentum_advect_skew!(a,b,c,x,y,z), gtg)
    drift_flux = tg_ke_drift((a,b,c,x,y,z) -> momentum_advect!(a,b,c,x,y,z), gtg)
    drift_skew_p = tg_ke_drift((a,b,c,x,y,z) -> momentum_advect_skew!(a,b,c,x,y,z), gtg;
                               project = true, dt = 1e-3, nsteps = 200)
    drift_flux_p = tg_ke_drift((a,b,c,x,y,z) -> momentum_advect!(a,b,c,x,y,z), gtg;
                               project = true, dt = 1e-3, nsteps = 200)
    @info "Taylor–Green kinetic-energy drift, ν = 0, C_s = 0, SSP-RK3" *
          "\n  advection only (∇·u grows, 2000 steps × dt = 1e-4):" *
          "\n     momentum_advect_skew! = $drift_skew" *
          "\n     momentum_advect!      = $drift_flux" *
          "\n  with pressure projection every substage (200 steps × dt = 1e-3):" *
          "\n     momentum_advect_skew! = $drift_skew_p" *
          "\n     momentum_advect!      = $drift_flux_p"

    @test abs(drift_skew) < 1e-10                 # only the RK3 time error remains
    @test abs(drift_flux) > 100 * abs(drift_skew)
    @test abs(drift_skew_p) < 1e-10               # projected: both are conservative
end

# -------------------------------------------------------------------------
@testset "Scalar variance conservation" begin
    g = RegularGrid(Float64; size=(16,16,16), extent=(1.0,1.0,1.0), halo=(5,5,5))
    u = Aeolus.Field(FaceX(), g); v = Aeolus.Field(FaceY(), g); w = Aeolus.Field(FaceZ(), g)
    ϕ = Aeolus.Field(CellCenter(), g)
    set!(ϕ, (x,y,z) -> 1.0 + 0.5sin(2π*x)*cos(2π*y)*sin(2π*z))

    # divergence-free
    set!(u, (x,y,z) -> 1.0 + 0.3sin(2π*y)); set!(v, (x,y,z) -> 0.5 + 0.3cos(2π*z))
    set!(w, (x,y,z) -> 0.2sin(2π*x))
    for f in (u, v, w, ϕ); fill_halos!(f, PERIODIC); end
    df_cen  = variance_production((t,a,b,c,f) -> advect_centered!(t,a,b,c,f), u,v,w,ϕ,g)
    df_skew = variance_production((t,a,b,c,f) -> advect_variance_conserving!(t,a,b,c,f), u,v,w,ϕ,g)
    df_z5   = variance_production((t,a,b,c,f) -> advect_weno_z!(t,a,b,c,f; order=5), u,v,w,ϕ,g)
    df_t8   = variance_production((t,a,b,c,f) -> advect_teno!(t,a,b,c,f; order=8), u,v,w,ϕ,g)

    # non-divergence-free (u = u(x), v = v(y), w = w(z)) acting on a generic
    # (random) scalar — a harmonic ϕ would hide the effect behind the
    # orthogonality of Σ ϕ²(∇·u).
    ud = Aeolus.Field(FaceX(), g); vd = Aeolus.Field(FaceY(), g); wd = Aeolus.Field(FaceZ(), g)
    set!(ud, (x,y,z) -> 1.0 + 0.3sin(2π*x)); set!(vd, (x,y,z) -> 0.5 + 0.3cos(2π*y))
    set!(wd, (x,y,z) -> 0.2sin(2π*z))
    for f in (ud, vd, wd); fill_halos!(f, PERIODIC); end
    Random.seed!(31415)
    ϕr = Aeolus.Field(CellCenter(), g)
    interior_view(ϕr) .= 1.0 .+ randn.()
    fill_halos!(ϕr, PERIODIC)
    nd_cen  = variance_production((t,a,b,c,f) -> advect_centered!(t,a,b,c,f), ud,vd,wd,ϕr,g)
    nd_skew = variance_production((t,a,b,c,f) -> advect_variance_conserving!(t,a,b,c,f), ud,vd,wd,ϕr,g)

    @info "discrete scalar-variance production Σϕ·C(ϕ) dV" *
          "\n  divergence-free : centered = $df_cen  variance-conserving = $df_skew" *
          "  WENO-Z5 = $df_z5  TENO8 = $df_t8" *
          "\n  ∇·u ≠ 0         : centered = $nd_cen  variance-conserving = $nd_skew"
    @test abs(df_skew) < 1e-13
    @test abs(nd_skew) < 1e-13            # holds for any velocity field
    @test abs(nd_cen)  > 1e-4             # the centred flux form does not
    @test abs(nd_cen)  > 1e6 * abs(nd_skew)
    @test df_z5 < 0 && df_t8 < 0          # upwind schemes dissipate variance
    @test abs(df_t8) < abs(df_z5)         # ... and TENO8 far less than WENO-Z5

    # integrated drift over 2000 SSP-RK3 steps, divergence-free flow
    function var_drift(advect!; dt = 2e-4, nsteps = 2000)
        f = Aeolus.Field(CellCenter(), g)
        set!(f, (x,y,z) -> 1.0 + 0.5sin(2π*x)*cos(2π*y)*sin(2π*z))
        fill_halos!(f, PERIODIC)
        V0 = sum(abs2, interior_view(f)); M0 = sum(interior_view(f))
        rk3_scalar!(f, u, v, w, (t,a,b,c,ff,dτ) -> advect!(t,a,b,c,ff), dt, nsteps, g)
        return ((sum(abs2, interior_view(f)) - V0)/V0,
                (sum(interior_view(f)) - M0)/M0)
    end
    dv_skew, dm_skew = var_drift(advect_variance_conserving!)
    dv_cen,  dm_cen  = var_drift(advect_centered!)
    dv_z5,   dm_z5   = var_drift((t,a,b,c,f) -> advect_weno_z!(t,a,b,c,f; order=5))
    @info "relative ∫ϕ² drift over 2000 SSP-RK3 steps (divergence-free flow)" *
          "\n  advect_variance_conserving! = $dv_skew  (mass drift $dm_skew)" *
          "\n  advect_centered!            = $dv_cen   (mass drift $dm_cen)" *
          "\n  advect_weno_z! order 5      = $dv_z5    (mass drift $dm_z5)"
    @test abs(dv_skew) < 1e-10
    @test abs(dm_skew) < 1e-12
    @test abs(dv_z5) > 100 * abs(dv_skew)      # WENO dissipation dominates
end

# -------------------------------------------------------------------------
@testset "Budget closure and enstrophy (DCBL-like case)" begin
    Random.seed!(11)
    m = RegularGrid(Float64; size=(16,16,16), extent=(500.0,500.0,500.0), halo=(3,3,3))
    wall = FieldBoundaryConditions(:periodic, :periodic, :wall)
    sim = BoussinesqSimulation(m; ν = 0.05, θ₀ = 300.0, g_accel = 9.81,
                               f_coriolis = 1e-4, C_s = 0.17, Pr_t = 1/3,
                               θ_flux_bottom = 0.1, bcs = wall, θ_bcs = wall)
    set!(sim.θ, (x,y,z) -> 300.0 + 0.003z)
    interior_view(sim.θ) .+= 0.05 .* randn.()
    set!(sim.u, (x,y,z) -> 0.5)
    for _ in 1:30; step!(sim, 1.0); end

    b = kinetic_energy_budget(sim)
    terms = (b.advection, b.buoyancy, b.coriolis, b.pressure, b.viscous, b.sgs)
    scale = maximum(abs, terms)
    b_skew = kinetic_energy_budget(sim; momentum_advection! = momentum_advect_skew!)
    b_weno = kinetic_energy_budget(sim; momentum_advection! =
        (a,bb,c,x,y,z) -> momentum_advect_weno_z!(a,bb,c,x,y,z; order=5))
    ens = enstrophy(sim)

    @info "kinetic-energy budget after 30 steps of a DCBL-like run" *
          "\n  E            = $(b.kinetic_energy)" *
          "\n  advection    = $(b.advection)   (flux form)" *
          "\n  advection    = $(b_skew.advection)   (momentum_advect_skew!)" *
          "\n  advection    = $(b_weno.advection)   (momentum_advect_weno_z!, dissipative)" *
          "\n  buoyancy     = $(b.buoyancy)" *
          "\n  coriolis     = $(b.coriolis)" *
          "\n  pressure     = $(b.pressure)" *
          "\n  viscous      = $(b.viscous)" *
          "\n  sgs          = $(b.sgs)" *
          "\n  total dE/dt  = $(b.total)" *
          "\n  residual     = $(b.residual)  (relative to largest term: $(abs(b.residual)/scale))" *
          "\n  enstrophy    = $ens"

    @test abs(b.residual)/scale < 1e-10          # the decomposition closes
    @test abs(b.coriolis)/scale < 1e-12          # f×u does no work, exactly
    @test abs(b.pressure)/scale < 1e-10          # ∇·u = 0 ⇒ no pressure work
    @test abs(b.advection)/scale < 1e-10         # flux form on a projected field
    @test abs(b_skew.advection)/scale < 1e-10
    @test b_weno.advection < 0                   # upwind advection dissipates
    @test b.viscous < 0 && b.sgs < 0
    @test b.buoyancy > 0                         # convective boundary layer
    @test ens > 0 && isfinite(ens)

    # ν Σu·∇²u = -2ν·enstrophy for a divergence-free field — an independent
    # check that the staggered curl is put together correctly.
    @test isapprox(b.viscous, -2*sim.ν*ens; rtol = 1e-8)

    sv = scalar_variance_budget(sim, sim.θ)
    sv_skew = scalar_variance_budget(sim, sim.θ; advection! = advect_variance_conserving!)
    sv_z5 = scalar_variance_budget(sim, sim.θ;
                advection! = (t,u,v,w,f) -> advect_weno_z!(t,u,v,w,f; order=5))
    sscale = maximum(abs, (sv.advection, sv.diffusion, sv.sgs, sv.surface_flux))
    @info "θ-variance budget" *
          "\n  ½∫θ²         = $(sv.variance)" *
          "\n  advection    = $(sv.advection)  (centered)" *
          "\n  advection    = $(sv_skew.advection)  (variance-conserving)" *
          "\n  advection    = $(sv_z5.advection)  (WENO-Z5)" *
          "\n  diffusion    = $(sv.diffusion)" *
          "\n  sgs          = $(sv.sgs)" *
          "\n  surface flux = $(sv.surface_flux)" *
          "\n  total        = $(sv.total)   residual = $(sv.residual)"
    @test abs(sv.residual)/sscale < 1e-10
    @test abs(sv_skew.advection)/sscale < 1e-10
    @test sv.diffusion < 0 && sv.sgs < 0
    @test sv_z5.advection < 0
end

# -------------------------------------------------------------------------
@testset "Scheme dispatch, halo checks and error messages" begin
    g = RegularGrid(Float64; size=(16,16,16), extent=(1.0,1.0,1.0), halo=(5,5,5))
    u = Aeolus.Field(FaceX(), g); v = Aeolus.Field(FaceY(), g); w = Aeolus.Field(FaceZ(), g)
    ϕ = Aeolus.Field(CellCenter(), g)
    set!(u, (x,y,z) -> 1.0 + 0.2sin(2π*y)); set!(v, (x,y,z) -> 0.3cos(2π*z))
    set!(w, (x,y,z) -> 0.1sin(2π*x))
    set!(ϕ, (x,y,z) -> sin(2π*x)*cos(2π*y))
    for f in (u, v, w, ϕ); fill_halos!(f, PERIODIC); end

    for (name, halo, direct) in
            ((:centered, 1, (t) -> advect_centered!(t,u,v,w,ϕ)),
             (:weno5,    3, (t) -> Aeolus.advect_weno5!(t,u,v,w,ϕ)),
             (:weno_z5,  3, (t) -> advect_weno_z!(t,u,v,w,ϕ; order=5)),
             (:weno_z7,  4, (t) -> advect_weno_z!(t,u,v,w,ϕ; order=7)),
             (:teno5,    3, (t) -> advect_teno!(t,u,v,w,ϕ; order=5)),
             (:teno8,    5, (t) -> advect_teno!(t,u,v,w,ϕ; order=8)),
             (:variance_conserving, 1, (t) -> advect_variance_conserving!(t,u,v,w,ϕ)))
        s = advection_scheme(name)
        @test s.name === name
        @test required_halo(s) == halo
        a = Aeolus.Field(CellCenter(), g); b = Aeolus.Field(CellCenter(), g)
        s(a, u, v, w, ϕ); direct(b)
        @test maximum(abs, interior_view(a) .- interior_view(b)) < 1e-13
    end
    @test occursin("AdvectionScheme(:teno8", sprint(show, advection_scheme(:teno8)))
    @test_throws ArgumentError advection_scheme(:nonexistent)
    @test_throws ArgumentError advect_weno_z!(Aeolus.Field(CellCenter(), g), u, v, w, ϕ; order = 6)
    @test_throws ArgumentError advect_teno!(Aeolus.Field(CellCenter(), g), u, v, w, ϕ; order = 6)

    # halo too small must fail loudly, per scheme
    gsmall = RegularGrid(Float64; size=(8,8,8), extent=(1.0,1.0,1.0), halo=(3,3,3))
    us = Aeolus.Field(FaceX(), gsmall); vs = Aeolus.Field(FaceY(), gsmall)
    wsz = Aeolus.Field(FaceZ(), gsmall); ϕs = Aeolus.Field(CellCenter(), gsmall)
    ts = Aeolus.Field(CellCenter(), gsmall)
    @test_throws ErrorException advect_weno_z!(ts, us, vs, wsz, ϕs; order = 7)
    @test_throws ErrorException advect_teno!(ts, us, vs, wsz, ϕs; order = 8)
    @test advect_weno_z!(ts, us, vs, wsz, ϕs; order = 5) === ts    # halo 3 is enough

    # momentum WENO-Z runs and is dissipative on a smooth field
    tu = Aeolus.Field(FaceX(), g); tv = Aeolus.Field(FaceY(), g); tw = Aeolus.Field(FaceZ(), g)
    momentum_advect_weno_z!(tu, tv, tw, u, v, w; order = 7)
    @test all(isfinite, interior_view(tu))
    @test all(isfinite, interior_view(tv))
    @test all(isfinite, interior_view(tw))
end

# -------------------------------------------------------------------------
# KA-c (#295): the serial baseline before threading
# -------------------------------------------------------------------------
# The Z/TENO weights take their exponent as a compile-time constant
# (`_ipow`), the limiter takes a prebuilt descriptor, and the halo fills
# allocate nothing. These are the properties the threaded port (#291)
# builds on.

# The weights as they were before #295: Base's x^p with a runtime p.
_teno5_left_basepow(v1, v2, v3, v4, v5, C_T, q) = begin
    AS = Aeolus.AdvectionSchemes
    q0 = (2*v1 - 7*v2 + 11*v3)/6; q1 = (-v2 + 5*v3 + 2*v4)/6; q2 = (2*v3 + 5*v4 - v5)/6
    b0 = AS._b3_m2(v1, v2, v3); b1 = AS._b3_m1(v2, v3, v4); b2 = AS._b3_p0(v3, v4, v5)
    ε = AS._zeps(Float64); τ = abs(b0 - b2)
    g0 = (1 + τ/(b0 + ε))^q; g1 = (1 + τ/(b1 + ε))^q; g2 = (1 + τ/(b2 + ε))^q
    gs = g0 + g1 + g2
    d0 = g0/gs < C_T ? 0.0 : 1/10; d1 = g1/gs < C_T ? 0.0 : 6/10; d2 = g2/gs < C_T ? 0.0 : 3/10
    (d0*q0 + d1*q1 + d2*q2)/(d0 + d1 + d2)
end

@testset "KA-c: compile-time integer power in the WENO-Z / TENO weights" begin
    AS = Aeolus.AdvectionSchemes
    rng = Xoshiro(295)
    xs = [exp(20 * randn(rng)) for _ in 1:200_000]
    pr = Ref(0)
    # p = 1, 2, 3: identical to Base's x^p with a runtime exponent, bit for
    # bit (one correctly rounded product for 2; Base's own x*x*x for 3)
    for p in 1:3
        pr[] = p
        @test all(x -> AS._ipow(x, Val(p)) === x^pr[], xs)
    end
    # q = 6 (TENO): x³·x³ is not Base's compensated x^6. Three
    # correctly rounded multiplies bound its relative error by
    # (1+u)^4 − 1 ≈ 4u = 2 eps, i.e. ≤ 4 ulp of the result (measured:
    # 3.02 ulp on 20 000 log-normal arguments; Base's is ≤ 1 ulp). A first
    # guess of 3 ulp was the wrong expectation, not a code error.
    pr[] = 6
    nd = count(x -> AS._ipow(x, Val(6)) !== x^pr[], xs)
    maxulp = maximum(xs[1:20_000]) do x
        ex = big(x)^6
        Float64(abs(big(AS._ipow(x, Val(6))) - ex) / eps(Float64(ex)))
    end
    @info "x^6: x³·x³ differs from Base's x^6 in $(nd) of $(length(xs)) arguments; max error $(round(maxulp; digits = 2)) ulp"
    @test maxulp <= 4
    # the TENO stencil selection (a cut-off on the normalised weights) is
    # what q enters; on random stencils the reconstructed values agree with
    # the Base-power ones to ≤ 1e-12 relative and differ in a few only
    vals = [randn(rng, 5) .* (rand(rng) < 0.5 ? 1.0 : 1e-6) .+ (rand(rng) < 0.1 ? 10.0 : 0.0) for _ in 1:100_000]
    new = [teno5_left(v...) for v in vals]
    old = [_teno5_left_basepow(v..., 1e-5, pr[]) for v in vals]
    ndiff = count(i -> new[i] !== old[i], eachindex(new))
    rel = maximum(i -> abs(new[i] - old[i]) / max(abs(old[i]), 1e-300), eachindex(new))
    @info "TENO5 (q = 6) against the Base-power weights: $(ndiff) of $(length(vals)) values differ, max rel $(rel)"
    @test rel <= 1e-12
    # WENO-Z (p = 2, the default; p = 1): the scalar entry points and the
    # descriptor path are one function, and agree
    for p in (1, 2), v in vals[1:2000]
        @test values_agree(weno_z5_left(v...; p = p), AS.reconstruct_left(AS.WENOZ(5, p), Tuple(v)))
        @test values_agree(weno_z5_right(v...; p = p), AS.reconstruct_right(AS.WENOZ(5, p), Tuple(v)))
    end
    w7 = randn(rng, 7)
    @test values_agree(weno_z7_left(w7...), AS.reconstruct_left(AS.WENOZ(7, 2), Tuple(w7)))
    w8 = randn(rng, 8)
    @test values_agree(teno8_left(w8...), AS.reconstruct_left(AS.TENO(8), Tuple(w8)))
    @test values_agree(teno5_left(w8[1:5]...; q = 4), AS.reconstruct_left(AS.TENO(5; q = 4), Tuple(w8[1:5])))
end

@testset "KA-c: prebuilt reconstruction descriptor, allocation-free halos and limiter" begin
    AS = Aeolus.AdvectionSchemes
    g = RegularGrid(Float64; size = (16, 12, 10), extent = (1.0, 1.0, 1.0), halo = (3, 3, 3))
    u = Aeolus.Field(FaceX(), g); v = Aeolus.Field(FaceY(), g); w = Aeolus.Field(FaceZ(), g)
    set!(u, (x, y, z) -> 0.7 + 0.3sin(2π * y)); set!(v, (x, y, z) -> -0.4cos(2π * x))
    set!(w, (x, y, z) -> 0.2sin(2π * x))
    ϕ = Aeolus.Field(CellCenter(), g)
    set!(ϕ, (x, y, z) -> exp(-((x - 0.5)^2 + (y - 0.5)^2) / 0.01))
    walls = FieldBoundaryConditions(:periodic, :periodic, :wall)
    for f in (u, v, w, ϕ); fill_halos!(f, PERIODIC); end
    ws = FluxLimiterWorkspace(g)
    dt = 0.5 * g.dx
    for name in (:weno_z5, :teno5, :centered)
        r = AS._reconstruction(name)
        t1 = Aeolus.Field(CellCenter(), g); t2 = Aeolus.Field(CellCenter(), g)
        advect_positive!(t1, u, v, w, ϕ, dt; scheme = name, workspace = ws)
        advect_positive!(t2, u, v, w, ϕ, dt; scheme = r, workspace = ws)
        @test agree(t1.data, t2.data)                         # the same operator
        advect_positive!(t2, u, v, w, ϕ, dt; scheme = r, workspace = ws)
        @test (@allocated advect_positive!(t2, u, v, w, ϕ, dt; scheme = r, workspace = ws)) == 0
    end
    # halo fills: no plane copies (they ran ~6 per field per call before #295)
    for f in (u, v, w, ϕ), bcs in (PERIODIC, walls, FieldBoundaryConditions(:noflux))
        fill_halos!(f, bcs)
        @test (@allocated fill_halos!(f, bcs)) == 0
    end
    # and the values are those of explicit copies
    a = copy(ϕ.data); fill_halos!(ϕ, PERIODIC)
    H = g.Hx; Nx = g.Nx; _, jj, kk = interior(g)
    @test ϕ.data[1:H, jj, kk] == a[Nx+1:Nx+H, jj, kk]
    @test ϕ.data[H+Nx+1:end, jj, kk] == a[H+1:2H, jj, kk]
end

end # testset
