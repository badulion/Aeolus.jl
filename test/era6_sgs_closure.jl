# Era VI, P54-a (#167): the SGS closure contract.
#
#   * the velocity-gradient helper every Era VI closure reads, and the
#     strain stencil built from it;
#   * the built-in closures as wrappers: the pre-contract runs;
#   * the commit cadence: advance_sgs! once per accepted step on every core,
#     rolled back with the state when step_adaptive! rejects a step;
#   * θ-aware closures see the current state;
#   * the κ_t field is the scalar diffusivity (discrete eigenvalue, CFL,
#     conservation);
#   * the anelastic core takes the same closures.

using Test
using Aeolus
using Aeolus.Fields: interior_view, Field
using Aeolus.Grids: interior
using Random: Xoshiro, rand, randn
using Printf

const OPS = Aeolus.Operators

# The pre-contract strain stencil (Operators._strain_at at cc4a150), kept
# here verbatim as the reference.
function _strain_ref(u, v, w, i, j, k, idx, idy, idz)
    S11 = (u[i+1,j,k] - u[i,j,k]) * idx
    S22 = (v[i,j+1,k] - v[i,j,k]) * idy
    S33 = (w[i,j,k+1] - w[i,j,k]) * idz
    du_dy = ((u[i,j+1,k] + u[i+1,j+1,k]) - (u[i,j-1,k] + u[i+1,j-1,k])) * (0.25 * idy)
    dv_dx = ((v[i+1,j,k] + v[i+1,j+1,k]) - (v[i-1,j,k] + v[i-1,j+1,k])) * (0.25 * idx)
    du_dz = ((u[i,j,k+1] + u[i+1,j,k+1]) - (u[i,j,k-1] + u[i+1,j,k-1])) * (0.25 * idz)
    dw_dx = ((w[i+1,j,k] + w[i+1,j,k+1]) - (w[i-1,j,k] + w[i-1,j,k+1])) * (0.25 * idx)
    dv_dz = ((v[i,j,k+1] + v[i,j+1,k+1]) - (v[i,j,k-1] + v[i,j+1,k-1])) * (0.25 * idz)
    dw_dy = ((w[i,j+1,k] + w[i,j+1,k+1]) - (w[i,j-1,k] + w[i,j-1,k+1])) * (0.25 * idy)
    S12 = 0.5 * (du_dy + dv_dx)
    S13 = 0.5 * (du_dz + dw_dx)
    S23 = 0.5 * (dv_dz + dw_dy)
    S2 = 2 * (S11^2 + S22^2 + S33^2) + 4 * (S12^2 + S13^2 + S23^2)
    return S11, S22, S33, S12, S13, S23, sqrt(S2)
end


# ---- test closures -------------------------------------------------------

# Smagorinsky with memory: the commit count and the dt of every commit.
# `coef_calls` is instrumentation outside the memory (it is not rolled back).
mutable struct CountingSGS <: AbstractSGSClosure
    inner::StaticSmagorinsky{Float64,Nothing}
    n::Int
    dts::Vector{Float64}
    coef_calls::Int
end
CountingSGS(C_s) = CountingSGS(StaticSmagorinsky(C_s), 0, Float64[], 0)
Aeolus.Operators.sgs_coefficients!(f::SGSFields, c::CountingSGS, sim) =
    (c.coef_calls += 1; sgs_coefficients!(f, c.inner, sim))
Aeolus.Operators.advance_sgs!(c::CountingSGS, sim, dt) = (c.n += 1; push!(c.dts, dt); c)
Aeolus.Operators.sgs_state(c::CountingSGS) = (c.n, copy(c.dts))
Aeolus.Operators.restore_sgs_state!(c::CountingSGS, s) = (c.n = s[1]; c.dts = copy(s[2]); c)
memory(c::CountingSGS) = (c.n, c.dts)

# κ = κ₀ everywhere (a prescribed field when `κ` is an array), ν = 0.
struct PrescribedKappa{K} <: AbstractSGSClosure
    κ::K
end
function Aeolus.Operators.sgs_coefficients!(f::SGSFields, c::PrescribedKappa, sim)
    fill!(f.νh.data, 0)
    c.κ isa Real ? fill!(f.κh.data, c.κ) : (interior_view(f.κh) .= c.κ)
    return f
end

# Smagorinsky ν with a scalar diffusivity 5ν (not ν/Prₜ).
struct FiveKappa <: AbstractSGSClosure
    C_s::Float64
end
function Aeolus.Operators.sgs_coefficients!(f::SGSFields, c::FiveKappa, sim)
    compute_smagorinsky_νt!(f.νh, sim.u, sim.v, sim.w, c.C_s)
    f.κh.data .= 5 .* f.νh.data
    return f
end

# Records the θ (with halos) it is shown and sets ν from it.
mutable struct ThetaProbe <: AbstractSGSClosure
    seen::Vector{Array{Float64,3}}
end
function Aeolus.Operators.sgs_coefficients!(f::SGSFields, c::ThetaProbe, sim)
    push!(c.seen, copy(sim.θ.data))
    interior_view(f.νh) .= 1e-4 .* abs.(interior_view(sim.θ) .- 300)
    f.κh.data .= f.νh.data
    return f
end

# A discretely divergence-free random field: the staggered curl of a
# smoothed random vector potential (periodic), scaled to max|u| = umax.
function solenoidal!(sim, rng; umax = 1.0, smooth = 2)
    m = sim.mesh; N = (m.Nx, m.Ny, m.Nz)
    A = [randn(rng, N...) for _ in 1:3]
    sm(a) = (b = copy(a); for _ in 1:smooth, d in 1:3
                 b = (circshift(b, ntuple(i -> i == d ? 1 : 0, 3)) .+ 2 .* b .+
                      circshift(b, ntuple(i -> i == d ? -1 : 0, 3))) ./ 4
             end; b)
    Ax, Ay, Az = sm.(A)
    D(a, d) = circshift(a, ntuple(i -> i == d ? -1 : 0, 3)) .- a      # forward difference
    u = D(Az, 2) ./ m.dy .- D(Ay, 3) ./ m.dz
    v = D(Ax, 3) ./ m.dz .- D(Az, 1) ./ m.dx
    w = D(Ay, 1) ./ m.dx .- D(Ax, 2) ./ m.dy
    s = umax / maximum(abs, vcat(vec(u), vec(v), vec(w)))
    interior_view(sim.u) .= s .* u
    interior_view(sim.v) .= s .* v
    interior_view(sim.w) .= s .* w
    for f in (sim.u, sim.v, sim.w)
        fill_halos!(f, sim.bcs)
    end
    return sim
end

# 16³ periodic decaying turbulence with a θ perturbation and buoyancy.
function decaying(; C_s = 0.0, seed = 54, kw...)
    g = RegularGrid(; size = (16, 16, 16), extent = (1.0, 1.0, 1.0), halo = (3, 3, 3))
    sim = BoussinesqSimulation(g; C_s = C_s, ν = 1e-5, θ₀ = 300.0, kw...)
    rng = Xoshiro(seed)
    solenoidal!(sim, rng)
    interior_view(sim.θ) .= 300 .+ 0.5 .* randn(rng, 16, 16, 16)
    return sim
end
state(s) = (copy(interior_view(s.u)), copy(interior_view(s.v)), copy(interior_view(s.w)),
            copy(interior_view(s.θ)))

@testset "Era VI — SGS closure contract (P54-a, #167)" begin

    @testset "velocity-gradient helper: exact on linear fields; strain" begin
        dx, dy, dz = 15.625, 15.625, 6.25
        g = RegularGrid(; size = (8, 8, 8), extent = (8dx, 8dy, 8dz), halo = (2, 2, 2))
        A = [ 0.013 -0.021  0.034;
              0.008  0.017 -0.052;
             -0.027  0.041 -0.030]             # ∂u_a/∂x_b, s⁻¹ (trace-free)
        u = Field(FaceX(), g); v = Field(FaceY(), g); w = Field(FaceZ(), g)
        Hx, Hy, Hz = g.Hx, g.Hy, g.Hz
        # face positions of every stored value (halos included)
        xf(i) = (i - Hx - 1) * dx; xc(i) = (i - Hx - 0.5) * dx
        yf(j) = (j - Hy - 1) * dy; yc(j) = (j - Hy - 0.5) * dy
        zf(k) = (k - Hz - 1) * dz; zc(k) = (k - Hz - 0.5) * dz
        lin(a, x, y, z) = 0.7 + A[a, 1] * x + A[a, 2] * y + A[a, 3] * z
        for I in CartesianIndices(u.data)
            i, j, k = Tuple(I)
            u.data[I] = lin(1, xf(i), yc(j), zc(k))
            v.data[I] = lin(2, xc(i), yf(j), zc(k))
            w.data[I] = lin(3, xc(i), yc(j), zf(k))
        end
        ii, jj, kk = interior(g)
        err = 0.0
        for k in kk, j in jj, i in ii
            G = OPS._velocity_gradient_at(u.data, v.data, w.data, i, j, k, 1 / dx, 1 / dy, 1 / dz)
            err = max(err, maximum(abs(G[a][b] - A[a, b]) for a in 1:3, b in 1:3))
        end
        @test err < 1e-12
        @printf("  [gradient] linear field on (15.625, 15.625, 6.25) m: max|g − A| = %.1e s⁻¹\n", err)

        # _strain_at, now built on the helper, is the pre-contract stencil
        rng = Xoshiro(167)
        for f in (u, v, w)
            f.data .= randn(rng, size(f.data))
        end
        same = all(_strain_ref(u.data, v.data, w.data, i, j, k, 1 / dx, 1 / dy, 1 / dz) ===
                   OPS._strain_at(u.data, v.data, w.data, i, j, k, 1 / dx, 1 / dy, 1 / dz)
                   for k in kk, j in jj, i in ii)
        @test same
        # ... and the Smagorinsky kernel built on _strain_at is (C_s Δ)²|S|
        νt = Field(CellCenter(), g)
        compute_smagorinsky_νt!(νt, u, v, w, 0.17)
        Δ = (dx * dy * dz)^(1 / 3)
        @test all(νt.data[i, j, k] == (0.17 * Δ)^2 *
                  _strain_ref(u.data, v.data, w.data, i, j, k, 1 / dx, 1 / dy, 1 / dz)[7]
                  for k in kk, j in jj, i in ii)
    end
    dt = 2e-3
    steppers = (("step!", (s, h; kw...) -> step!(s, h; kw...)),
                ("step_rk3!", (s, h; kw...) -> step_rk3!(s, h; kw...)),
                ("step_imex!", (s, h; kw...) -> step_imex!(s, h, ARK2(); kw...)))

    @testset "the wrappers reproduce the built-in and legacy runs ($name)" for (name, adv!) in steppers
        run(sim, n; kw...) = (for _ in 1:n; adv!(sim, dt; kw...); end; sim)
        ref = run(decaying(C_s = 0.17), 50)                                 # built-in closure
        @test maximum(interior_view(ref.νt)) > 0
        st  = run(decaying(), 50; sgs = StaticSmagorinsky(0.17))            # C_s = 0 + closure
        leg = run(decaying(), 50; eddy_viscosity! = (νt, u, v, w) -> compute_smagorinsky_νt!(νt, u, v, w, 0.17))
        fun = run(decaying(), 50; sgs = (νt, u, v, w) -> compute_smagorinsky_νt!(νt, u, v, w, 0.17))
        @test values_agree(state(st), state(ref))
        @test values_agree(state(leg), state(ref))        # legacy 4-argument closure: identical u, v, w, θ
        @test values_agree(state(fun), state(ref))
        @test values_agree(interior_view(st.κt), interior_view(st.νt) ./ st.Pr_t)
        # the dynamic closure against today's eddy_viscosity! form
        ws1 = DynamicSmagorinskyWorkspace(ref.mesh); ws2 = DynamicSmagorinskyWorkspace(ref.mesh)
        dyn = run(decaying(), 50; sgs = DynamicSmagorinsky(ws1))
        old = run(decaying(), 50; eddy_viscosity! = (νt, u, v, w) ->
                      compute_dynamic_smagorinsky_νt!(νt, u, v, w; workspace = ws2))
        @test state(dyn) == state(old)
        @test ws1.cs2 == ws2.cs2
        @test state(dyn) != state(ref)        # the dynamic coefficient is not 0.17
        @test_throws ArgumentError adv!(decaying(), dt; sgs = StaticSmagorinsky(0.1),
                                        eddy_viscosity! = (νt, u, v, w) -> nothing)
    end

    @testset "commit cadence: advance_sgs! once per accepted step" begin
        n = 6
        for (name, adv!) in steppers
            c = CountingSGS(0.17)
            sim = decaying()
            for _ in 1:n
                adv!(sim, dt; sgs = c)
            end
            @test c.n == n                                   # RK3: n, not 3n
            @test c.dts == fill(dt, n)
            # coefficients: once per Euler step / IMEX step, once per RK3 substage
            @test c.coef_calls == (name == "step_rk3!" ? 3n : n)
            @printf("  [commit] %-10s %d steps: %d commits, %d coefficient evaluations\n",
                    name, n, c.n, c.coef_calls)
        end
        # adaptive: CFL-picked steps without retries
        c = CountingSGS(0.17); sim = decaying()
        k = CFLConstraints(dt_max = 5e-3)
        dts = [step_adaptive!(sim, k; sgs = c)[2] for _ in 1:n]
        @test c.n == n && c.dts == dts
        c = CountingSGS(0.17); sim = decaying()
        dts = [step_adaptive!(sim, k; sgs = c, scheme = ARK2())[2] for _ in 1:n]
        @test c.n == n && c.dts == dts
    end

    @testset "commit cadence: a rejected adaptive step is rolled back" begin
        # A uniform acceleration a makes max|u| grow by a·dt: with the cap at
        # u₀ + 0.75 a·dt the step of dt is rejected and dt/2 is accepted.
        a = 1.0; h = 0.02
        push_x!(s, Δt) = (interior_view(s.tu) .+= a; nothing)
        cap = 0.01 + 0.75 * a * h                       # the flow starts at max|u| = 0.01
        k = CFLConstraints(dt_min = 1e-6, dt_max = 1.0)
        fresh() = (s = decaying(; seed = 7); solenoidal!(s, Xoshiro(7); umax = 0.01); s)

        c1 = CountingSGS(0.17); s1 = fresh()
        _, used, nretry = step_adaptive!(s1, k; dt = h, sgs = c1, body_force! = push_x!,
                                         max_velocity = cap)
        @test nretry == 1 && used == h / 2
        c2 = CountingSGS(0.17); s2 = fresh()
        _, used2, nretry2 = step_adaptive!(s2, k; dt = h / 2, sgs = c2, body_force! = push_x!,
                                           max_velocity = cap)
        @test nretry2 == 0 && used2 == h / 2
        @test values_agree(memory(c1), memory(c2)) && values_agree(memory(c2), (1, [h / 2]))      # the direct short step
        @test values_agree(state(s1), state(s2))
        @test c1.coef_calls == 2                               # the rejected try did evaluate

        # the Berlin pattern: cover h in pieces; every commit carries its piece
        c3 = CountingSGS(0.17); s3 = fresh()
        remaining = h; pieces = Float64[]
        while remaining > 1e-12 * h
            _, u3, _ = step_adaptive!(s3, k; dt = remaining, sgs = c3, body_force! = push_x!,
                                      max_velocity = cap + a * h)   # the 2nd piece starts faster
            remaining -= u3; push!(pieces, u3)
        end
        @test c3.n == length(pieces) && c3.dts == pieces && sum(pieces) ≈ h
        # IMEX through step_adaptive! rolls back the same way
        c4 = CountingSGS(0.17); s4 = fresh()
        _, used4, nretry4 = step_adaptive!(s4, k; dt = h, sgs = c4, scheme = ARK2(),
                                           body_force! = push_x!, max_velocity = cap)
        @test nretry4 == 1 && memory(c4) == (1, [h / 2])
        @printf("  [commit] adaptive, forced retry: %d retry, commits %d (dts %s)\n",
                nretry, c1.n, c1.dts)
    end

    @testset "θ-aware closure sees the current θ" begin
        for (name, adv!) in steppers
            sim = decaying()
            p = ThetaProbe(Array{Float64,3}[])
            prev = nothing
            for _ in 1:3
                θnow = copy(interior_view(sim.θ))
                ncalls = length(p.seen)
                adv!(sim, dt; sgs = p)
                seen = p.seen[ncalls + 1]           # the first evaluation of this step
                m = sim.mesh
                @test seen[m.Hx+1:m.Hx+m.Nx, m.Hy+1:m.Hy+m.Ny, m.Hz+1:m.Hz+m.Nz] == θnow
                @test seen[1, m.Hy+1, m.Hz+1] == seen[m.Nx+1, m.Hy+1, m.Hz+1]   # halos filled (periodic)
                prev === nothing || @test θnow != prev
                prev = θnow
            end
        end
    end

    @testset "κ_t is the scalar diffusivity: discrete eigenvalue" begin
        # θ = sin(2πz/Lz), u = 0, κ ≡ κ₀ (≠ νt/Prₜ = 0): the mode is an exact
        # eigenvector of the discrete diffusion, eigenvalue −κ₀(4/dz²)sin²(π dz/Lz).
        g = RegularGrid(; size = (4, 4, 32), extent = (1.0, 1.0, 2.0), halo = (2, 2, 2))
        # explicit: h = 0.02 s (the 2Δz mode needs h·κ₀·4/dz² < 2; it is 1.2);
        # IMEX: h = 0.5 s, 25× past that limit
        κ0 = 6e-2
        λ = -κ0 * (4 / g.dz^2) * sin(π * g.dz / g.Lz)^2
        zc = [(k - 0.5) * g.dz for k in 1:g.Nz]
        prof = sin.(2π .* zc ./ g.Lz)
        amp(s) = sum(interior_view(s.θ)[1, 1, :] .* prof) / sum(abs2, prof)
        γ = ARK2().γ
        R_ark2(z) = (1 + (1 - γ) * z / (1 - γ * z)) / (1 - γ * z)
        cases = (("step!", 0.02, 100, z -> 1 + z,
                  (s, Δt) -> step!(s, Δt; sgs = PrescribedKappa(κ0))),
                 ("step_imex!", 0.5, 20, R_ark2,
                  (s, Δt) -> step_imex!(s, Δt, ARK2(); sgs = PrescribedKappa(κ0), tol = 1e-14, maxit = 500)))
        for (name, h, nst, R, adv!) in cases
            sim = BoussinesqSimulation(g)            # ν = 0, no buoyancy, C_s = 0
            for k in 1:g.Nz
                interior_view(sim.θ)[:, :, k] .= prof[k]
            end
            for _ in 1:nst
                adv!(sim, h)
            end
            rel = abs(amp(sim) / R(h * λ)^nst - 1)
            @test rel <= 1e-10
            @test amp(sim) < 0.75                    # it decayed (κ₀ is what acts)
            @test maximum(abs, interior_view(sim.u)) == 0
            @printf("  [κt] %-10s %3d steps of %.2f s, hλ = %.4f: amplitude %.4f vs R(hλ)^n, rel. error %.1e\n",
                    name, nst, h, h * λ, amp(sim), rel)
        end
    end

    @testset "CFL: the SGS limit is set by the larger of νt and κt" begin
        sim = decaying()
        k = CFLConstraints()
        d = cfl_diagnostics(sim, k; sgs = FiveKappa(0.17))
        m = sim.mesh
        νmax = maximum(interior_view(sim.νt)); κmax = maximum(interior_view(sim.κt))
        @test νmax > 0 && κmax == maximum(5 .* interior_view(sim.νt))
        Δ²harm = 1 / (1 / m.dx^2 + 1 / m.dy^2 + 1 / m.dz^2)
        @test d.κt_max == κmax && d.νt_max == νmax
        @test d.dt_sgs == k.C_diffusive * Δ²harm / κmax              # exactly
        @test d.dt_diff == k.C_diffusive * Δ²harm / (sim.ν + νmax)
        # the built-in closure: κt = νt/Prₜ, so the SGS limit is Prₜ × the viscous one
        s2 = decaying(C_s = 0.17)
        d2 = cfl_diagnostics(s2, k)
        @test d2.κt_max == maximum(interior_view(s2.νt)) / s2.Pr_t
        @test d2.dt_sgs == k.C_diffusive * Δ²harm / d2.κt_max
        # pure: computing the limit does not commit
        c = CountingSGS(0.17)
        for _ in 1:3
            cfl_diagnostics(sim, k; sgs = c)
        end
        @test c.n == 0 && c.coef_calls == 3
    end

    @testset "conservation: Σθ dV with a random positive κt, no-flux walls" begin
        g = RegularGrid(; size = (12, 10, 8), extent = (1.2, 1.0, 0.8), halo = (2, 2, 2))
        rng = Xoshiro(11)
        bcs = FieldBoundaryConditions(:periodic, :periodic, :wall)
        θb = FieldBoundaryConditions(:periodic, :periodic, :noflux)
        κ = 1e-3 .* (0.2 .+ rand(rng, 12, 10, 8))
        for (name, adv!) in (("step!", (s, Δt) -> step!(s, Δt; sgs = PrescribedKappa(κ))),
                             ("step_imex!", (s, Δt) -> step_imex!(s, Δt, ARK2(); sgs = PrescribedKappa(κ),
                                                                  tol = 1e-14, maxit = 500)))
            sim = BoussinesqSimulation(g; bcs = bcs, θ_bcs = θb)
            interior_view(sim.θ) .= 290 .+ randn(rng, 12, 10, 8)
            S0 = sum(interior_view(sim.θ)); V0 = sum(abs2, interior_view(sim.θ) .- S0 / length(κ))
            for _ in 1:100
                adv!(sim, 0.5)
            end
            S = sum(interior_view(sim.θ))
            V = sum(abs2, interior_view(sim.θ) .- S / length(κ))
            @test abs(S - S0) / abs(S0) <= 1e-13
            @test V < 0.5 * V0                                  # and it did diffuse
            @printf("  [conservation] %-10s 100 steps: |ΔΣθ|/Σθ = %.1e, variance %.3f → %.3f\n",
                    name, abs(S - S0) / abs(S0), V0, V)
        end
    end

    @testset "anelastic core: the same closures" begin
        g = RegularGrid(; size = (16, 16, 12), extent = (800.0, 800.0, 600.0), halo = (3, 3, 3))
        bcs = FieldBoundaryConditions(:periodic, :periodic, :wall)
        rng = Xoshiro(3)
        U = (0.3 .* randn(rng, 17 + 5, 16 + 6, 12 + 6), 0.3 .* randn(rng, 16 + 6, 17 + 5, 12 + 6),
             0.3 .* randn(rng, 16 + 6, 16 + 6, 13 + 5))
        set_uvw!(s) = (interior_view(s.u) .= U[1][1:16, 1:16, 1:12];
                       interior_view(s.v) .= U[2][1:16, 1:16, 1:12];
                       interior_view(s.w) .= U[3][1:16, 1:16, 1:12];
                       interior_view(s.w)[:, :, 1] .= 0; s)
        bq = set_uvw!(BoussinesqSimulation(g; bcs = bcs))
        an = set_uvw!(AnelasticSimulation(g; bcs = bcs))
        for s in (bq, an), f in (s.u, s.v, s.w)
            fill_halos!(f, s.bcs)
        end
        cb = DynamicSmagorinsky(g); ca = DynamicSmagorinsky(g)
        compute_sgs!(cb, bq); compute_sgs!(ca, an)
        @test values_agree(interior_view(an.νt), interior_view(bq.νt))        # same νt
        @test values_agree(interior_view(an.κt), interior_view(bq.κt))
        @test maximum(interior_view(an.νt)) > 0
        # ... and the step runs it (νt is formed at the start of the step)
        a1 = set_uvw!(AnelasticSimulation(g; bcs = bcs)); step!(a1, 0.5; sgs = DynamicSmagorinsky(g))
        @test values_agree(interior_view(a1.νt), interior_view(an.νt))
        # built-in C_s = 0.17 ≡ StaticSmagorinsky(0.17); no closure ≡ a zero closure
        b1 = set_uvw!(AnelasticSimulation(g; bcs = bcs, C_s = 0.17))
        b2 = set_uvw!(AnelasticSimulation(g; bcs = bcs))
        z1 = set_uvw!(AnelasticSimulation(g; bcs = bcs))
        z2 = set_uvw!(AnelasticSimulation(g; bcs = bcs))
        cnt = CountingSGS(0.17)
        b3 = set_uvw!(AnelasticSimulation(g; bcs = bcs))
        for _ in 1:5
            step!(b1, 0.5); step!(b2, 0.5; sgs = StaticSmagorinsky(0.17))
            step!(b3, 0.5; sgs = cnt)
            step!(z1, 0.5); step!(z2, 0.5; sgs = StaticSmagorinsky(0.0))
        end
        @test state(b1) == state(b2)
        @test state(z1) == state(z2)
        @test all(iszero, z1.νt.data) && all(iszero, z1.κt.data)   # C_s = 0, no closure: untouched
        @test state(b1) != state(z1)
        @test cnt.n == 5 && cnt.dts == fill(0.5, 5)
    end
end
