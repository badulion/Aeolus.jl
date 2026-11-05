# =========================================================================
# Era II — Phases 26 & 28
#   * WENO-Z / TENO advection for all prognostics
#   * energy- and enstrophy-conserving discrete operators
# =========================================================================

using Test
using Random
using Aeolus
using Aeolus.AdvectionSchemes
using Aeolus.Grids: interior
using Aeolus.Fields: set!, interior_view

const PERIODIC = FieldBoundaryConditions(:periodic)

"Exact cell average of sin over the cell of width `h` centred at `xc`."
avg_sin(xc, h) = (cos(xc - h/2) - cos(xc + h/2))/h

"Cell averages of F' on the n-cell stencil whose cell 0 is left of the interface x0."
stencil_avgs(F, x0, h, off0, n) =
    ntuple(j -> (xc = x0 - h/2 + (off0 + j - 1)*h; (F(xc + h/2) - F(xc - h/2))/h), n)

"Observed order of `rec` (n-point stencil) for sin, from two cell widths."
function measure_order(rec, n, off0; hs = (0.2, 0.1), x0 = 0.3)
    e = [abs(rec(stencil_avgs(x -> -cos(x), x0, h, off0, n)...) - sin(x0)) for h in hs]
    return log(e[1]/e[2]) / log(hs[1]/hs[2])
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
        advect!(t1, u, v, w, ϕ)
        @views ϕ.data[ii,jj,kk] .= a .* ϕ0[ii,jj,kk] .+
                                   b .* (ϕ.data[ii,jj,kk] .+ dt .* t1.data[ii,jj,kk])
    end
    for _ in 1:nsteps
        ϕ0 .= ϕ.data
        stage!(0.0, 1.0); stage!(0.75, 0.25); stage!(1/3, 2/3)
    end
    return ϕ
end

@testset "Era II — High-order & conservative advection" begin

@testset "Reconstruction: design order, TENO exactness, critical point" begin
    @test measure_order(weno_z5_left, 5, -2) > 4.8
    @test measure_order(teno5_left, 5, -2) > 4.8
    # TENO's discrete stencil selection returns the linear weights verbatim:
    # a degree-4 polynomial is reproduced exactly (WENO-Z only approaches them)
    c = [1.0, 2.0, -0.5, 0.3, -0.1]
    p4(x) = sum(c[m+1] * x^m for m in 0:4)
    P4(x) = sum(c[m+1] * x^(m+1)/(m+1) for m in 0:4)
    @test abs(teno5_left(stencil_avgs(P4, 0.7, 0.3, -2, 5)...) - p4(0.7)) < 1e-13
    # right-biased = mirrored left-biased
    v5 = (0.3, 0.9, 1.4, 1.1, 0.2)
    @test weno_z5_right(reverse(v5)...) == weno_z5_left(v5...)
    # WENO-Z beats WENO-JS where f' = 0 at the stencil centre
    h = 0.1; x0 = π/2 + h/2
    v = ntuple(j -> avg_sin(x0 - h/2 + (j - 3)*h, h), 5)
    @test abs(weno_z5_left(v...) - sin(x0)) < abs(weno_js5_left(v...) - sin(x0))
end

@testset "ENO property and mass at a discontinuity (square wave, one period)" begin
    g = RegularGrid(Float64; size=(64,4,4), extent=(1.0,1/16,1/16), halo=(3,3,3))
    u = Aeolus.Field(FaceX(), g); v = Aeolus.Field(FaceY(), g); w = Aeolus.Field(FaceZ(), g)
    set!(u, (x,y,z) -> 1.0)
    for f in (u, v, w); fill_halos!(f, PERIODIC); end
    dt = 0.4 * g.dx                      # CFL = 0.4
    nsteps = round(Int, 1.0/dt)          # one period
    run(adv) = begin
        ϕ = Aeolus.Field(CellCenter(), g); set!(ϕ, (x,y,z) -> 0.25 <= x <= 0.5 ? 1.0 : 0.0)
        a0 = copy(interior_view(ϕ)[:,1,1]); m0 = sum(interior_view(ϕ))
        rk3_scalar!(ϕ, u, v, w, adv, dt, nsteps, g)
        a = interior_view(ϕ)[:,1,1]
        (over = maximum(a) - 1, under = -minimum(a), tv = total_variation(a)/total_variation(a0),
         mass = (sum(interior_view(ϕ)) - m0)/m0)
    end
    t5 = run((t,u,v,w,f) -> advect_teno!(t,u,v,w,f; order=5))
    cen = run((t,u,v,w,f) -> advect_centered!(t,u,v,w,f))
    @test t5.tv < 1.02 && t5.over < 5e-3 && t5.under < 5e-3
    @test cen.tv > 2.0                   # the centred scheme oscillates
    @test abs(t5.mass) < 1e-12 && abs(cen.mass) < 1e-12
end

@testset "Positivity preservation (narrow Gaussian, CFL ≈ 0.95)" begin
    g = RegularGrid(Float64; size=(16,16,16), extent=(1.0,1.0,1.0), halo=(3,3,3))
    u = Aeolus.Field(FaceX(), g); v = Aeolus.Field(FaceY(), g); w = Aeolus.Field(FaceZ(), g)
    set!(u, (x,y,z) -> 1.0); set!(v, (x,y,z) -> 0.6); set!(w, (x,y,z) -> 0.3)
    for f in (u, v, w); fill_halos!(f, PERIODIC); end
    dt = 0.95 / (1.0/g.dx + 0.6/g.dy + 0.3/g.dz)
    σ  = 1.2 * g.dx
    gauss = (x,y,z) -> exp(-((x-0.5)^2 + (y-0.5)^2 + (z-0.5)^2)/(2σ^2))
    ws = FluxLimiterWorkspace(g)
    ii, jj, kk = interior(g)
    function euler(adv)
        ϕ = Aeolus.Field(CellCenter(), g); set!(ϕ, gauss)
        m0 = sum(interior_view(ϕ))
        t  = Aeolus.Field(CellCenter(), g)
        for _ in 1:20                     # forward Euler: the case the proof covers
            fill!(t.data, 0); fill_halos!(ϕ, PERIODIC)
            adv(t, ϕ)
            @views ϕ.data[ii,jj,kk] .+= dt .* t.data[ii,jj,kk]
        end
        (min = minimum(interior_view(ϕ)), max = maximum(interior_view(ϕ)),
         mass = (sum(interior_view(ϕ)) - m0)/m0)
    end
    raw = euler((t,f) -> advect_weno_z!(t,u,v,w,f; order=5))
    lim = euler((t,f) -> advect_positive!(t,u,v,w,f,dt; scheme=:weno_z5, workspace=ws))
    @test raw.min < -0.1                 # unlimited high order goes negative ...
    @test lim.min >= -1e-14 * lim.max    # ... the limiter keeps ϕ ≥ 0
    @test abs(lim.mass) < 1e-12
end

@testset "Skew-symmetric momentum and variance-conserving scalar advection" begin
    g = RegularGrid(Float64; size=(8,8,8), extent=(1.0,1.0,1.0), halo=(3,3,3))
    dV = g.dx*g.dy*g.dz
    Random.seed!(2718)
    # a field with non-zero discrete divergence: only the skew forms conserve
    ur = Aeolus.Field(FaceX(), g); vr = Aeolus.Field(FaceY(), g); wr = Aeolus.Field(FaceZ(), g)
    ϕ = Aeolus.Field(CellCenter(), g)
    for f in (ur, vr, wr, ϕ)
        interior_view(f) .= 1 .+ randn.()
        fill_halos!(f, PERIODIC)
    end
    eprod(adv!) = (t = map(l -> Aeolus.Field(l, g), (FaceX(), FaceY(), FaceZ()));
                   adv!(t..., ur, vr, wr);
                   sum(sum(interior_view(a) .* interior_view(b)) for (a, b) in zip((ur, vr, wr), t)) * dV)
    vprod(adv!) = (t = Aeolus.Field(CellCenter(), g); adv!(t, ur, vr, wr, ϕ);
                   (sum(interior_view(ϕ) .* interior_view(t)) * dV, sum(interior_view(t)) * dV))
    @test abs(eprod(momentum_advect_skew!)) < 1e-12
    @test abs(eprod(momentum_advect!)) > 1e-3          # the flux form does not conserve
    vs, ms = vprod(advect_variance_conserving!)
    vc, mc = vprod(advect_centered!)
    @test abs(vs) < 1e-12
    @test abs(vc) > 1e6 * abs(vs)
    # the flux forms conserve mass for any velocity (the skew form only if ∇·u = 0)
    @test abs(mc) < 1e-12
    vz, mz = vprod((t,a,b,c,f) -> advect_weno_z!(t,a,b,c,f; order=5))
    @test abs(mz) < 1e-12
end

@testset "Budget closure (DCBL-like case)" begin
    Random.seed!(11)
    m = RegularGrid(Float64; size=(8,8,8), extent=(500.0,500.0,500.0), halo=(3,3,3))
    wall = FieldBoundaryConditions(:periodic, :periodic, :wall)
    sim = BoussinesqSimulation(m; ν = 0.05, θ₀ = 300.0, g_accel = 9.81,
                               f_coriolis = 1e-4, C_s = 0.17, Pr_t = 1/3,
                               θ_flux_bottom = 0.1, bcs = wall, θ_bcs = wall)
    set!(sim.θ, (x,y,z) -> 300.0 + 0.003z)
    interior_view(sim.θ) .+= 0.05 .* randn.()
    set!(sim.u, (x,y,z) -> 0.5)
    for _ in 1:2; step!(sim, 1.0); end
    b = kinetic_energy_budget(sim)
    scale = maximum(abs, (b.advection, b.buoyancy, b.coriolis, b.pressure, b.viscous, b.sgs))
    @test abs(b.residual)/scale < 1e-10          # the decomposition closes
    @test abs(b.coriolis)/scale < 1e-12          # f×u does no work
    @test abs(b.pressure)/scale < 1e-10          # ∇·u = 0 ⇒ no pressure work
    @test b.viscous < 0 && b.sgs < 0
    # ν Σu·∇²u = -2ν·enstrophy for a divergence-free field
    @test isapprox(b.viscous, -2*sim.ν*enstrophy(sim); rtol = 1e-8)
    sv = scalar_variance_budget(sim, sim.θ)
    sscale = maximum(abs, (sv.advection, sv.diffusion, sv.sgs, sv.surface_flux))
    @test abs(sv.residual)/sscale < 1e-10
end

@testset "Scheme dispatch and error messages" begin
    g = RegularGrid(Float64; size=(8,8,8), extent=(1.0,1.0,1.0), halo=(3,3,3))
    u = Aeolus.Field(FaceX(), g); v = Aeolus.Field(FaceY(), g); w = Aeolus.Field(FaceZ(), g)
    ϕ = Aeolus.Field(CellCenter(), g)
    set!(u, (x,y,z) -> 1.0 + 0.2sin(2π*y)); set!(ϕ, (x,y,z) -> sin(2π*x)*cos(2π*y))
    for f in (u, v, w, ϕ); fill_halos!(f, PERIODIC); end
    s = advection_scheme(:teno5)
    @test required_halo(s) == 3 && required_halo(advection_scheme(:teno8)) == 5
    a = Aeolus.Field(CellCenter(), g); b = Aeolus.Field(CellCenter(), g)
    s(a, u, v, w, ϕ); advect_teno!(b, u, v, w, ϕ; order=5)
    @test maximum(abs, interior_view(a) .- interior_view(b)) < 1e-13
    @test_throws ArgumentError advection_scheme(:nonexistent)
    @test_throws ErrorException advect_teno!(a, u, v, w, ϕ; order = 8)   # halo 3 < 5
end

end # testset
