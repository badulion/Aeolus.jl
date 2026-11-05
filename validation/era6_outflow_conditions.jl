# N-open-c (#260, epic #246): radiation outflow conditions on open faces
# (`outflow = :dirichlet | :convective | :orlanski`), upwind-consistent
# scalar halos, and `boundary_scalar_flux`, the one boundary-flux function.
#
#   1. API: per-face modes, the halo check, inflow points keep the data
#   2. uniform flow through radiating faces: an exact solution, every core
#   3. scalar conservation: d/dt∫θ dV + ∮θ u·n dA = 0 per step, centred
#      (through step!) and WENO-Z5 / TENO5 (their own face fluxes)
#   4. scalar pulse through a convective outflow: ∫θ'² after 2 crossings
#   5. plane Poiseuille flow, Re = 50, with an open x axis
#   6. outgoing Lamb–Oseen vortex (test/open_boundary_vortex_harness.jl):
#      R for Dirichlet (perfect and uniform-U parent), convective, Orlanski
#   7. stability: 2000 steps with a time-interpolated parent, ∫|u|² vs parent
#
# References: Orlanski (1976), JCP 21, 251 (radiation condition, diagnosed
# phase speed); Miller & Thorpe (1981), QJRMS 107, 615 (clipping
# 0 ≤ c ≤ Δx/Δt, upstream radiation schemes); Pauley, Moin & Reynolds
# (1990), JFM 220, 397 (convective outflow condition in incompressible
# LES); Colonius (2004), Annu. Rev. Fluid Mech. 36, 315 (outgoing-vortex
# tests of non-reflecting conditions).

using Test
using Aeolus
using Aeolus.Grids: interior, CellCenter, FaceX, FaceY, FaceZ
using Aeolus.Fields: Field, interior_view, set!
using Aeolus.BoundaryConditions: OpenBoundary, RadiationPlane
using Aeolus.Simulations: BoussinesqSimulation, step_rk3!, max_divergence
using Aeolus.TimeIntegration: step_imex!, ARK2, Rosenbrock2, step_adaptive!, CFLConstraints
using Aeolus.Anelastic: AnelasticSimulation, PseudoIncompressibleSimulation
using Aeolus.Operators: advect_centered!
using Printf

include(joinpath(@__DIR__, "support", "open_boundary_vortex_harness.jl"))

# ---------------------------------------------------------------------------
@testset "N-open-c API: per-face outflow modes, halo check, inflow stays Dirichlet" begin
    m = RegularGrid(size = (8, 6, 5), extent = (80.0, 60.0, 50.0), halo = (3, 3, 3))
    data = ProfileBoundaryData(u = 2.0, v = 1.0, w = 0.0, θ = 300.0)
    # default: every face Dirichlet, no radiation state
    b0 = open_boundary_conditions(m, data)
    st0 = only(open_boundary_states(b0))
    @test all(==( :dirichlet), values(st0.outflow))
    @test all(r -> r === nothing, st0.radiation)
    @test !has_radiating_outflow(b0)
    # one mode for every face
    b1 = open_boundary_conditions(m, data; outflow = :convective)
    st1 = only(open_boundary_states(b1))
    @test all(==(:convective), values(st1.outflow))
    @test all(r -> r isa RadiationPlane, st1.radiation)
    @test has_radiating_outflow(b1)
    # per face: only the east face radiates
    b2 = open_boundary_conditions(m, data; outflow = (east = :orlanski,))
    st2 = only(open_boundary_states(b2))
    @test st2.outflow[:east] === :orlanski && st2.outflow[:west] === :dirichlet
    @test all((r !== nothing) == (p.side === :east) for (p, r) in zip(st2.planes, st2.radiation))
    @test all(r.mode === :orlanski for r in st2.radiation if r !== nothing)
    # errors: unknown mode, unknown side, a side that is not open, shallow halo
    @test_throws ArgumentError open_boundary_conditions(m, data; outflow = :sponge)
    @test_throws ArgumentError open_boundary_conditions(m, data; outflow = (eest = :convective,))
    @test_throws ArgumentError open_boundary_conditions(m, data; x = :periodic,
                                                        outflow = (east = :convective,))
    m2 = RegularGrid(size = (8, 6, 5), extent = (80.0, 60.0, 50.0))      # halo 2
    @test_throws ArgumentError open_boundary_conditions(m2, data; halo = 3)
    @test open_boundary_conditions(m2, data; halo = 2) isa FieldBoundaryConditions
    # the inflow faces keep the data exactly, the outflow faces radiate
    # (:outflow_only correction: the inflow faces are not shifted by δ)
    s = BoussinesqSimulation(m; bcs = open_boundary_conditions(m, data; outflow = :convective,
                                                              correction = :outflow_only))
    set!(s.u, (x, y, z) -> 2.0 + 0.1sin(2π * x / 80)); set!(s.v, 1.0); set!(s.θ, 300.0)
    for _ in 1:5; step_rk3!(s, 1.0); end
    st = only(open_boundary_states(s))
    pw = only(p for p in st.planes if p.side === :west && p.name === :u)
    ps = only(p for p in st.planes if p.side === :south && p.name === :θ)
    rw = st.radiation[findfirst(p -> p === pw, st.planes)]
    re = st.radiation[findfirst(p -> p.side === :east && p.name === :u, st.planes)]
    @test all(isnan, rw.c)                       # west: inflow everywhere
    @test all(c -> c > 0, re.c)                  # east: outflow everywhere
    @test all(==(2.0), pw.data)                  # the inflow face is the data
    @test all(==(300.0), ps.data)
end

# ---------------------------------------------------------------------------
@testset "uniform flow through radiating faces: every core, buoyancy and SGS" begin
    # A uniform stream is an exact solution; the radiation of a uniform
    # field returns it, so u = U, θ and p stay uniform to round-off.
    m = RegularGrid(size = (12, 10, 8), extent = (600.0, 500.0, 400.0), halo = (3, 3, 3))
    U, Vv = 5.0, 2.0
    data = ProfileBoundaryData(u = U, v = Vv, w = 0.0, θ = 300.0)
    check(s, θref) = (maximum(abs, interior_view(s.u) .- U) < 1e-12 &&
                      maximum(abs, interior_view(s.v) .- Vv) < 1e-12 &&
                      maximum(abs, interior_view(s.w)) < 1e-12 &&
                      maximum(abs, interior_view(s.θ) .- θref) < 1e-12 &&
                      (maximum(interior_view(s.p)) - minimum(interior_view(s.p))) < 1e-12)
    for mode in (:convective, :orlanski)
        for (name, adv!) in (("step!", s -> step!(s, 2.0)),
                             ("step_rk3!", s -> step_rk3!(s, 2.0)),
                             ("step_imex! ARK2", s -> step_imex!(s, 2.0, ARK2())),
                             ("step_imex! ROS2", s -> step_imex!(s, 2.0, Rosenbrock2())),
                             ("step_adaptive!", s -> step_adaptive!(s, CFLConstraints(dt_max = 2.0))))
            s = BoussinesqSimulation(m; bcs = open_boundary_conditions(m, data; outflow = mode),
                                     θ₀ = 300.0, C_s = 0.18, ν = 1e-5)
            set!(s.u, U); set!(s.v, Vv); set!(s.θ, 300.0)
            θ0 = copy(interior_view(s.θ))
            for _ in 1:100; adv!(s); end
            @test check(s, θ0)
        end
        for ctor in (AnelasticSimulation, PseudoIncompressibleSimulation)
            s = ctor(m; bcs = open_boundary_conditions(m, data; outflow = mode), C_s = 0.18)
            only(open_boundary_states(s)).data = ProfileBoundaryData(u = U, v = Vv, θ = s.ref.θ̄c_pad)
            set!(s.u, U); set!(s.v, Vv)
            θ0 = copy(interior_view(s.θ))
            for _ in 1:100; step!(s, 2.0); end
            @test check(s, θ0)
        end
    end
end

# ---------------------------------------------------------------------------
# A blob of θ leaving through a convective outflow in a sheared stream, on
# the state after 20 steps (boundaries engaged).
function _ob_conservation_case()
    Nx, Nz, H = 48, 24, 3
    m = RegularGrid(size = (Nx, 1, Nz), extent = (48.0, 1.0, 24.0), halo = (H, H, H))
    data = FunctionBoundaryData(u = (x, y, z, t) -> 1.0 + 0.3sin(2π * z / 24), v = 0.0, w = 0.0,
                                θ = (x, y, z, t) -> 0.5 + 0.2cos(2π * z / 24))
    bcs = open_boundary_conditions(m, data; x = :open, y = :periodic, top = :wall,
                                   outflow = :convective, halo = H)
    s = BoussinesqSimulation(m; bcs = bcs)
    set!(s.u, (x, y, z) -> 1.0 + 0.3sin(2π * z / 24) + 0.2exp(-((x - 30)^2 + (z - 12)^2) / 40))
    set!(s.θ, (x, y, z) -> 0.5 + exp(-((x - 44)^2 + (z - 10)^2) / 20))
    for _ in 1:20; step_rk3!(s, 0.2); end
    return s
end

@testset "boundary_scalar_flux: ∫θ budget closes per step, centred / WENO-Z5 / TENO5" begin
    s = _ob_conservation_case()
    m = s.mesh; dV = m.dx * m.dy * m.dz
    ii, jj, kk = interior(m)
    mass(f) = sum(view(f.data, ii, jj, kk)) * dV
    scale = sum(abs, view(s.θ.data, ii, jj, kk)) * dV
    dt = 0.2
    # the schemes' own operators, one forward-Euler step each
    for (scheme, op!) in ((:centered, advect_centered!),
                          (:weno_z5, (t, u, v, w, ϕ) -> advect_weno_z!(t, u, v, w, ϕ)),
                          (:teno5, (t, u, v, w, ϕ) -> advect_teno!(t, u, v, w, ϕ)))
        F = boundary_scalar_flux(s, s.θ; scheme = scheme)
        tend = Field(CellCenter(), m)
        op!(tend, s.u, s.v, s.w, s.θ)
        resid = abs(sum(view(tend.data, ii, jj, kk)) * dV * dt + dt * F.total) / scale
        @printf("  ∫θ budget, %-8s: F_west = %.6f, F_east = %.6f, residual %.2e\n",
                scheme, F.west, F.east, resid)
        @test resid < 1e-12
        @test F.west < 0 && F.east > 0            # inflow west, outflow east
        @test F.south == F.north == F.bottom == F.top == 0
    end
    # through the dynamical core: step! advects θ with the centred flux form
    F = boundary_scalar_flux(s, s.θ)
    S0 = mass(s.θ)
    step!(s, dt)
    resid = abs(mass(s.θ) - S0 + dt * F.total) / scale
    @printf("  ∫θ budget through step!: residual %.2e\n", resid)
    @test resid < 1e-12
    # analytic: a uniform θ₀ in a uniform stream U crosses the box unchanged,
    # F_east = −F_west = θ₀ U Ly Lz
    m3 = RegularGrid(size = (8, 4, 4), extent = (8.0, 4.0, 4.0), halo = (3, 3, 3))
    s3 = BoussinesqSimulation(m3; bcs = open_boundary_conditions(m3, ProfileBoundaryData(
             u = 2.0, v = 0.0, w = 0.0, θ = 3.0); y = :periodic, top = :wall,
             outflow = :convective))
    set!(s3.u, 2.0); set!(s3.θ, 3.0); step!(s3, 0.1)
    for sch in (:centered, :upwind, :weno_z5, :weno_z7, :teno5)
        sch === :weno_z7 && continue                 # needs halo 4
        F3 = boundary_scalar_flux(s3, s3.θ; scheme = sch)
        @test F3.east ≈ 3.0 * 2.0 * 16.0 rtol = 1e-13
        @test F3.west ≈ -3.0 * 2.0 * 16.0 rtol = 1e-13
        @test abs(F3.total) < 1e-12
    end
    # a halo shallower than the scheme needs at an open face is an error
    @test_throws ArgumentError boundary_scalar_flux(s3, s3.θ; scheme = :weno_z7)
    @test_throws ArgumentError boundary_scalar_flux(s3, s3.θ; scheme = :weno5)
end

# ---------------------------------------------------------------------------
# A Gaussian θ pulse (σ = 8Δx) in a uniform stream U through open x faces:
# inflow θ = 0, convective outflow. After two crossing times the pulse must
# have left without reflection. step_rk3!, centred θ advection.
function _ob_pulse(mode; σ = 8.0, N = 128, dt = 0.25, U = 1.0, x0 = 40.0)
    m = RegularGrid(size = (N, 1, 1), extent = (Float64(N), 1.0, 1.0), halo = (3, 3, 3))
    bcs = open_boundary_conditions(m, ProfileBoundaryData(u = U, v = 0.0, w = 0.0, θ = 0.0);
                                   y = :periodic, top = :wall, outflow = mode)
    s = BoussinesqSimulation(m; bcs = bcs)
    set!(s.u, U)
    g(x) = exp(-x^2 / (2σ^2))
    set!(s.θ, (x, y, z) -> g(x - x0))
    ii, jj, kk = interior(m)
    xc = [(i - 3.5) for i in ii]
    E0 = sum(abs2, view(s.θ.data, ii, jj, kk))
    L2mid = NaN; cface = NaN
    nmid = round(Int, (N / 2) / (U * dt))                  # moved N/2: still inside
    nface = round(Int, (N - x0 - σ) / (U * dt))           # the pulse's front half leaving
    st = only(open_boundary_states(s))
    kθ = findfirst(p -> p.side === :east && p.name === :θ, st.planes)
    for n in 1:round(Int, 2N / (U * dt))
        step_rk3!(s, dt)
        if n == nface && st.radiation[kθ] !== nothing
            cface = st.radiation[kθ].c[4, 4]               # phase speed of θ at the face
        end
        if n == nmid
            ex = g.(xc .- x0 .- U * s.t)
            L2mid = sqrt(sum(abs2, view(s.θ.data, ii, 4, 4) .- ex) / sum(abs2, ex))
        end
    end
    return sum(abs2, view(s.θ.data, ii, jj, kk)) / E0, L2mid, cface
end

@testset "scalar pulse through a radiating outflow: ∫θ'² after two crossings" begin
    res_c, L2_c, c_c = _ob_pulse(:convective)
    res_o, _, c_o    = _ob_pulse(:orlanski)
    res_d, _, _      = _ob_pulse(:dirichlet)
    @printf("  pulse σ = 8Δx: residual ∫θ'²/∫θ'²₀ convective %.2e, Orlanski %.2e, Dirichlet %.2e; interior L2 error (centred, after N/2) %.3f; phase speed at the face: convective %.3f, Orlanski %.3f (U = 1)\n",
            res_c, res_o, res_d, L2_c, c_c, c_o)
    @test c_c == 1.0                  # convective: the normal velocity
    @test abs(c_o - 1.0) < 0.05       # Orlanski: diagnosed ≈ U (centred scheme's phase error)
    @test res_c < 1e-4
    @test res_o < 1e-4
    @test res_d > 0.5                 # Dirichlet θ = 0 at the outflow traps and reflects it
    @test L2_c < 0.05                 # the centred interior scheme's own phase error
end

# ---------------------------------------------------------------------------
# Plane Poiseuille flow between no-slip walls, Re = U_max H/ν = 50, x open:
# parabolic inflow, convective outflow, from a plug of the same flux.
#
# The discrete steady state is the parabola shifted by U_max Δz²/H²: the
# no-slip halo is linear (u_ghost = −u_1), and the parabola that vanishes
# at the wall in that sense is u = 4U_max z(H−z)/H² + U_max Δz²/H². That
# profile is the inflow data and the reference; the continuous parabola
# differs from it by exactly U_max Δz²/H² (checked).
@testset "plane Poiseuille, Re = 50, open x: profile, dp/dx, outflow layer" begin
    Hc, Umax, Re = 1.0, 1.0, 50.0
    ν = Umax * Hc / Re
    Nx, Nz, Lx = 32, 16, 4.0
    dz = Hc / Nz
    upar(z) = 4Umax * z * (Hc - z) / Hc^2
    udisc(z) = upar(z) + Umax * dz^2 / Hc^2
    m = RegularGrid(size = (Nx, 1, Nz), extent = (Lx, Lx / Nx, Hc), halo = (3, 3, 3))
    bcs = open_boundary_conditions(m, ProfileBoundaryData(u = udisc, v = 0.0, w = 0.0, θ = 0.0);
                                   y = :periodic, bottom = :noslip, top = :noslip,
                                   outflow = :convective)
    s = BoussinesqSimulation(m; ν = ν, bcs = bcs)
    set!(s.u, 2Umax / 3)                                   # plug, same flux
    ii, jj, kk = interior(m)
    zc = reshape([(k - 3.5) * dz for k in kk], 1, 1, :)
    for _ in 1:2000; step_rk3!(s, 0.02); end                # t = 40 H/U_max
    u = view(s.u.data, ii, jj, kk)
    err = maximum(abs, u .- udisc.(zc))
    shift = maximum(abs, u .- upar.(zc))
    out_err = maximum(abs, view(s.u.data, last(ii)-4:last(ii)+1, jj, kk) .- udisc.(zc)) / Umax
    G = -8ν * Umax / Hc^2
    p = s.p.data
    dpdx = (p[ii[2:end], jj, kk] .- p[ii[1:end-1], jj, kk]) ./ m.dx
    n10 = round(Int, 0.1 * (Nx - 1))
    dp_err = maximum(abs, dpdx[1+n10:end-n10, :, :] ./ G .- 1)
    @printf("  Poiseuille Re=50: max|u − u_disc| = %.2e, dp/dx rel. error (middle 80%%) = %.2e, outflow layer (last 5 cells) = %.2e, |u − parabola| = %.6f (Δz²/H² = %.6f)\n",
            err, dp_err, out_err, shift, dz^2 / Hc^2)
    @test err < 1e-8
    @test dp_err < 1e-6
    @test out_err < 1e-2
    @test shift ≈ Umax * dz^2 / Hc^2 rtol = 1e-6
    @test max_divergence(s) < 1e-12
end

# ---------------------------------------------------------------------------
# The outgoing Lamb–Oseen vortex. R at the end of the run, when the vortex
# centre is 3 r_c past the outflow face.
@testset "outgoing Lamb–Oseen vortex (r_c = 8Δx): reflection R" begin
    h = VortexHarness()
    perfect = run_vortex_harness(h; outflow = :dirichlet, parent = :truth)
    dir_u   = run_vortex_harness(h; outflow = :dirichlet, parent = :uniform)
    conv    = run_vortex_harness(h; outflow = :convective, parent = :uniform)
    orl     = run_vortex_harness(h; outflow = :orlanski, parent = :uniform)
    @printf("  vortex R (end; max over the run): Dirichlet perfect parent %.2e (%.2e), Dirichlet uniform-U %.3f (%.3f), convective %.2e (%.2e), Orlanski %.2e (%.2e)\n",
            perfect.R, perfect.Rmax, dir_u.R, dir_u.Rmax, conv.R, conv.Rmax, orl.R, orl.Rmax)
    @test perfect.R < 1e-3
    @test conv.R < 1e-2
    @test orl.R < 1e-2
    @test dir_u.R >= 10 * conv.R
    @test dir_u.R >= 10 * orl.R
    @test max_divergence(conv.child) < 1e-10 && max_divergence(orl.child) < 1e-10
end

# ---------------------------------------------------------------------------
@testset "stability: 2000 steps, time-interpolated parent, ∫|u|² within 1 %" begin
    h = VortexHarness()
    for mode in (:dirichlet, :convective, :orlanski)
        r = run_vortex_harness(h; outflow = mode, parent = :truth, parent_every = 5,
                               nsteps = 2000, record_every = 20)
        d = maximum(abs, r.ke_child ./ r.ke_truth .- 1)
        @printf("  %-10s 2000 steps, parent every 5 steps: max |∫|u|²_child / ∫|u|²_parent − 1| = %.2e\n",
                mode, d)
        @test d < 0.01
        @test all(isfinite, r.child.u.data)
        @test max_divergence(r.child) < 1e-10
    end
end
