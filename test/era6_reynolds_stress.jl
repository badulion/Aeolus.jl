# Era VI, Phase 56 (P56-d, #188): the differential SGS Reynolds-stress
# closure (src/prognostic_sgs/reynolds_stress.jl): Deardorff (1973) with
# the LRR-IP pressure–strain (Launder, Reece & Rodi 1975), Daly–Harlow
# buoyancy, exact Coriolis, and Schumann (1977) realizability, on the
# shared stress (P55-b) and tensor-flux (P55-c) operators and the TKE
# closure's l and ε (P56-b).
#
# Numbers measured on this branch are quoted next to each assertion.

using Test
using Aeolus
using Aeolus.Fields: interior_view, set!
using Aeolus.Operators: advance_sgs!, sgs_state, restore_sgs_state!, add_sgs_scalar!,
                        add_sgs_momentum!, compute_sgs!, sgs_explicit_stress
using Aeolus.Simulations: kinetic_energy, step_rk3!
using Aeolus.TimeIntegration: step_imex!, ARK2
using LinearAlgebra: Symmetric, eigvals, det, tr, exp
using Random: Xoshiro
using Statistics: mean
using Printf

# the closed-form LRR-IP homogeneous-shear equilibrium (P = ε, l = Δ):
# τ₂₂/e = τ₃₃/e = 2/3 − (2/3)(1 − C₂)/C₁, τ₁₁/e = 2/3 + (2(1 − C₂) + (2/3)C₂ − 2/3)/C₁,
# (τ₁₂/e)² = ((1 − C₂)/C₁) τ₂₂/e — from P_ij + Φ_ij = (2/3)εδ_ij with P = ε
function lrr_shear_equilibrium(C1 = 1.8, C2 = 0.6)
    t22 = 2 / 3 - (2 / 3) * (1 - C2) / C1
    t11 = 2 / 3 + (2 * (1 - C2) + (2 / 3) * C2 - 2 / 3) / C1
    t12 = sqrt((1 - C2) / C1 * t22)
    return t11, t22, t12
end

# a box with linear shear u = S z and θ = θ₀ + Γ z between free-slip walls
function rsm_shear_box(S, Ri; Nz = 16, Δ = 10.0)
    m = RegularGrid(; size = (4, 4, Nz), extent = (4Δ, 4Δ, Nz * Δ))
    sim = BoussinesqSimulation(m; θ₀ = 300.0, bcs = FieldBoundaryConditions(:periodic, :periodic, :wall),
                               θ_bcs = FieldBoundaryConditions(:periodic, :periodic, :noflux))
    set!(sim.u, (x, y, z) -> S * z)
    Γ = Ri * S^2 * 300.0 / 9.81
    set!(sim.θ, (x, y, z) -> 300.0 + Γ * z)
    return sim
end

# a periodic box with a divergence-free multi-mode flow: u(y, z), v(z, x),
# w(x, y), so ∇·u = 0 exactly on the staggered grid
function rsm_flow_box(amp; N = 16, L = 160.0, halo = 3, seed = 1)
    m = RegularGrid(; size = (N, N, N), extent = (L, L, L), halo = (halo, halo, halo))
    sim = BoussinesqSimulation(m)
    rng = Xoshiro(seed)
    k = 2π / L
    modes = [(rand(rng, 1:3), rand(rng, 1:3), 2π * rand(rng), amp * randn(rng)) for _ in 1:6]
    set!(sim.u, (x, y, z) -> sum(a * sin(p * k * y + q * k * z + ph) for (p, q, ph, a) in modes))
    set!(sim.v, (x, y, z) -> sum(a * cos(p * k * z + q * k * x + ph) for (p, q, ph, a) in modes))
    set!(sim.w, (x, y, z) -> sum(a * sin(p * k * x + q * k * y + ph) for (p, q, ph, a) in modes))
    return sim
end

randpsd(rng) = (A = randn(rng, 3, 3); M = A * A'; (M[1, 1], M[2, 2], M[3, 3], M[1, 2], M[1, 3], M[2, 3]))
full(τ) = [τ[1] τ[4] τ[5]; τ[4] τ[2] τ[6]; τ[5] τ[6] τ[3]]
half_trace(t) = (t[1] + t[2] + t[3]) / 2

@testset "Era VI — differential SGS Reynolds-stress closure (P56-d, #188)" begin

@testset "1. trace: ½ tr of the RSM right-hand side is the TKE right-hand side" begin
    rng = Xoshiro(188)
    # (a) any state: ½ tr rhs = −τᵢₖ∂ₖuᵢ + βh₃ − ε; Φ and C trace-free
    worst = 0.0; worstΦC = 0.0
    for _ in 1:200
        τ = randpsd(rng); G = Tuple(randn(rng, 9)); dθ = Tuple(randn(rng, 3))
        r, cθ, β = rand(rng), rand(rng), rand(rng); Ω = Tuple(randn(rng, 3))
        t = reynolds_stress_terms(τ, G, dθ, r, cθ, β; Ω = Ω)
        e = half_trace(τ)
        Gm = [G[1] G[2] G[3]; G[4] G[5] G[6]; G[7] G[8] G[9]]
        P = -sum(full(τ) .* Gm)
        h3 = -cθ * (τ[5] * dθ[1] + τ[6] * dθ[2] + τ[3] * dθ[3])
        scale = abs(P) + abs(β * h3) + r * e + sum(abs, t.rhs)
        worst = max(worst, abs(half_trace(t.rhs) - (P + β * h3 - r * e)) / scale)
        worstΦC = max(worstΦC, (abs(half_trace(t.pressure_strain)) + abs(half_trace(t.coriolis))) / scale)
    end
    @info @sprintf("  trace identity, 200 random states: %.1e; tr Φ + tr C: %.1e", worst, worstΦC)
    @test worst < 1e-13
    @test worstΦC < 1e-13

    # (b) the eddy-viscosity state of a stratified shear, τ = (2/3)eI − 2ν_t S
    # (τ₁₃ = −ν_t S), with the TKE closure's l, ε, ν_t, κ_t: ½ tr of the RSM
    # rhs is the TKE rhs ν_t S² − κ_t N² − ε to round-off (the :tke
    # calibration of C_θ makes the isotropic Daly–Harlow flux the TKE κ_t)
    S, Δ, β = 0.05, 10.0, 9.81 / 300
    worst = 0.0
    for e in (1e-3, 0.01, 0.1), Ri in (-0.5, 0.0, 0.1, 0.2), lw in (Inf, 3.0), fac in (1.0, 3.9)
        N2 = Ri * S^2
        l, s = tke_length_scale(e, N2, Δ; l_wall = lw)
        cε = tke_dissipation_coefficient(l, Δ; factor = fac)
        ν, κ = tke_eddy_coefficients(e, l, Δ)
        rhs_tke = ν * S^2 - κ * N2 - cε * s * e
        r = cε * s
        cθ = daly_harlow_coefficient(e, l, Δ, r)
        τ = (2e / 3, 2e / 3, 2e / 3, 0.0, -ν * S, 0.0)
        G = (0.0, 0.0, S, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        rsm = half_trace(reynolds_stress_rhs(τ, G, (0.0, 0.0, N2 / β), r, cθ, β))
        worst = max(worst, abs(rsm - rhs_tke) / (ν * S^2 + abs(κ * N2) + cε * s * e))
    end
    @info @sprintf("  ½ tr RSM rhs vs TKE rhs (eddy-viscosity state, 48 cases): %.1e", worst)
    @test worst < 1e-13
    # the calibrated Daly–Harlow constant: C_θ = (3/2)(1 + 2l/Δ) c_k c_ε = 0.315 at l = Δ
    e = 0.1; l, s = tke_length_scale(e, 0.0, Δ); r = tke_dissipation_coefficient(l, Δ) * s
    @test daly_harlow_coefficient(e, l, Δ, r) * r ≈ 1.5 * 3 * 0.1 * 0.70 rtol = 1e-12
    @test daly_harlow_coefficient(e, l, Δ, r; C_θ = 0.3) ≈ 0.3 / r rtol = 1e-14
end

@testset "2. Rotta return to isotropy: db/dt = −(C₁ − 1)(ε/e) b" begin
    rng = Xoshiro(2)
    # (a) the right-hand side: d b/dt from the rhs equals −(C₁−1) r b exactly
    worst = 0.0
    for _ in 1:100
        τ = randpsd(rng); r = rand(rng)
        rhs = reynolds_stress_rhs(τ, ntuple(_ -> 0.0, 9), (0.0, 0.0, 0.0), r, 0.0, 0.0)
        e = half_trace(τ); de = half_trace(rhs)
        b, _, _ = stress_anisotropy(τ)
        db = ntuple(m -> rhs[m] / (2e) - τ[m] * de / (2e^2), 6)
        worst = max(worst, maximum(abs.(db .+ (1.8 - 1) * r .* b)) / (r * maximum(abs, b)))
    end
    @info @sprintf("  db/dt + (C₁−1) r b, 100 random states: %.1e", worst)
    @test worst < 1e-13

    # (b) the closed form at l = Δ: e(t) = (e₀^{-1/2} + c_ε t/(2Δ))^{-2}, and
    # since ∫(ε/e)dt = ln(e₀/e), b(t) = b₀ (e/e₀)^{C₁−1}; the local ODE by RK4
    Δ = 10.0; cε = tke_dissipation_coefficient(Δ, Δ)
    τ0 = (0.9, 0.3, 0.2, 0.1, -0.05, 0.02)
    f(τ) = (e = half_trace(τ); reynolds_stress_rhs(τ, ntuple(_ -> 0.0, 9), (0.0, 0.0, 0.0),
                                                   cε * sqrt(e) / Δ, 0.0, 0.0))
    τ = τ0; h = 0.05; T = 600.0
    for _ in 1:round(Int, T / h)
        k1 = f(τ); k2 = f(τ .+ (h / 2) .* k1); k3 = f(τ .+ (h / 2) .* k2); k4 = f(τ .+ h .* k3)
        τ = τ .+ (h / 6) .* (k1 .+ 2 .* k2 .+ 2 .* k3 .+ k4)
    end
    e0 = half_trace(τ0); eT = (e0^-0.5 + cε * T / (2Δ))^-2
    b0, _, _ = stress_anisotropy(τ0); bT, _, _ = stress_anisotropy(τ)
    errE = abs(half_trace(τ) - eT) / eT
    errB = maximum(abs.(bT .- b0 .* (eT / e0)^0.8)) / maximum(abs, b0)
    @info @sprintf("  Rotta closed form (RK4 of the rhs, 600 s): e %.1e, b %.1e", errE, errB)
    @test errE < 1e-6
    @test errB < 1e-6

    # (c) the 3-D closure (backward-Euler sources, lagged ε/e) converges to
    # the closed form at first order
    m = RegularGrid(; size = (4, 4, 4), extent = (40.0, 40.0, 40.0))
    sim = BoussinesqSimulation(m)
    errs = Float64[]
    for n in (60, 120, 240, 480)
        c = ReynoldsStressClosure(m; transport = false)
        for k in 1:6
            interior_view(c.τ[k]) .= τ0[k]
        end
        for _ in 1:n
            advance_sgs!(c, sim, T / n)
        end
        τc = ntuple(k -> interior_view(c.τ[k])[2, 2, 2], 6)
        bc, _, _ = stress_anisotropy(τc)
        push!(errs, max(abs(half_trace(τc) - eT) / eT,
                        maximum(abs.(bc .- b0 .* (eT / e0)^0.8)) / maximum(abs, b0)))
        @test reynolds_stress_statistics(c).clips == 0
    end
    ratios = errs[1:end-1] ./ errs[2:end]
    @info @sprintf("  Rotta decay, 3-D closure vs closed form: errors %s, ratios %s",
                   join((@sprintf("%.2e", x) for x in errs), ", "),
                   join((@sprintf("%.2f", x) for x in ratios), ", "))
    @test all(x -> 1.8 < x < 2.2, ratios)               # first order
    @test errs[end] < 0.01
end

@testset "3. homogeneous shear equilibrium (LRR-IP, P = ε)" begin
    t11, t22, t12 = lrr_shear_equilibrium()
    # the issue quotes the rounded 0.963 / 0.519 / 0.339; the closed form is
    # 0.962963 / 0.518519 / 0.339450 — 0.519 and 0.339 are 4.8e-4 and 4.5e-4
    # away, so "within 1e-4 of the rounded numbers" was an expectation error;
    # the closure is asserted against the closed form instead
    @test abs(t11 - 0.963) < 1e-4
    @test abs(t11 - 0.962963) < 1e-6 && abs(t22 - 0.518519) < 1e-6 && abs(t12 - 0.339450) < 1e-6
    S, Δ = 0.05, 10.0
    cε = tke_dissipation_coefficient(Δ, Δ)
    e_eq = (t12 * S * Δ / cε)^2                          # P = −τ₁₃S = ε = c_ε e^{3/2}/Δ
    sim = rsm_shear_box(S, 0.0)
    c = ReynoldsStressClosure(sim.mesh; e0 = 1e-3, transport = false)
    for _ in 1:4000
        advance_sgs!(c, sim, 5.0)
    end
    v(k) = interior_view(c.τ[k])[:, :, 3:14]             # away from the walls
    e = (v(1) .+ v(2) .+ v(3)) ./ 2
    # shear du/dz: x₁ streamwise, x₃ the gradient direction (τ₁₃ is the shear stress)
    @test maximum(abs.(v(1) ./ e .- t11)) < 1e-10
    @test maximum(abs.(v(2) ./ e .- t22)) < 1e-10
    @test maximum(abs.(v(3) ./ e .- t22)) < 1e-10
    @test maximum(abs.(.-v(5) ./ e .- t12)) < 1e-10
    @test maximum(abs.(v(4))) < 1e-14 && maximum(abs.(v(6))) < 1e-14
    @test maximum(abs.(e .- e_eq)) / e_eq < 1e-10
    # (quoted, not asserted: Tavoularis & Corrsin 1981 measured a larger
    # anisotropy with τ22 ≠ τ33, roughly b11 ≈ 0.20, b22 ≈ −0.14, b33 ≈ −0.06,
    # b12 ≈ −0.15; the IP model's τ22 = τ33 is a known deficiency)
    @info @sprintf("  shear equilibrium: τ11/e %.6f τ22/e %.6f τ33/e %.6f −τ13/e %.6f, e/e_eq − 1 = %.1e",
                   mean(v(1) ./ e), mean(v(2) ./ e), mean(v(3) ./ e), mean(.-v(5) ./ e),
                   maximum(abs.(e .- e_eq)) / e_eq)
    st = reynolds_stress_statistics(c)
    @test st.clips == 0
    # P = ε at the equilibrium (the interior; the wall cells see half the shear)
    b = reynolds_stress_budget(c, sim)
    @test b.shear > 0 && b.dissipation > 0
end

@testset "4. rotation: τ(t) = R(2Ωt) τ₀ R(2Ωt)ᵀ" begin
    m = RegularGrid(; size = (4, 4, 4), extent = (40.0, 40.0, 40.0))
    sim = BoussinesqSimulation(m)
    Ω = (0.3e-4, -0.2e-4, 1.0e-4)
    a = sqrt(sum(abs2, Ω))
    W = [0 -Ω[3] Ω[2]; Ω[3] 0 -Ω[1]; -Ω[2] Ω[1] 0]
    period = 2π / (2a)
    τ0 = (0.9, 0.3, 0.2, 0.1, -0.05, 0.02)
    # no dissipation (c_ε = 0): pure rotation
    c = ReynoldsStressClosure(m; transport = false, c_ε0 = 0, c_ε1 = 0, Ω = Ω)
    for k in 1:6
        interior_view(c.τ[k]) .= τ0[k]
    end
    nper = 50
    dt = period / nper
    err = 0.0; dtr = 0.0; ddet = 0.0
    tr0 = tr(full(τ0)); det0 = det(full(τ0))
    for n in 1:10nper
        advance_sgs!(c, sim, dt)
        R = exp(-2 * n * dt * W)                         # independent of Rodrigues
        ex = R * full(τ0) * R'
        τc = ntuple(k -> interior_view(c.τ[k])[1, 3, 2], 6)
        err = max(err, maximum(abs.(full(τc) .- ex)) / maximum(abs, ex))
        dtr = max(dtr, abs(tr(full(τc)) - tr0) / tr0)
        ddet = max(ddet, abs(det(full(τc)) - det0) / det0)
    end
    @info @sprintf("  rotation over 10 periods: |τ − Rτ₀Rᵀ| %.1e, tr %.1e, det %.1e", err, dtr, ddet)
    @test err < 1e-8
    @test dtr < 1e-12 && ddet < 1e-12
    # the closed-form rotation is the solution of the C_ij of the rhs (RK4)
    f(τ) = reynolds_stress_rhs(τ, ntuple(_ -> 0.0, 9), (0.0, 0.0, 0.0), 0.0, 0.0, 0.0; Ω = Ω)
    τ = τ0; h = period / 2000
    for _ in 1:2000
        k1 = f(τ); k2 = f(τ .+ (h / 2) .* k1); k3 = f(τ .+ (h / 2) .* k2); k4 = f(τ .+ h .* k3)
        τ = τ .+ (h / 6) .* (k1 .+ 2 .* k2 .+ 2 .* k3 .+ k4)
    end
    @test maximum(abs.(full(τ) .- full(rotate_stress(τ0, coriolis_rotation(Ω, period))))) < 1e-10
    @test maximum(abs.(full(τ) .- full(τ0))) < 1e-10     # back after one period of 2Ω
    # Ω = :auto takes (0, 0, f/2) from a simulation with Coriolis
    simf = BoussinesqSimulation(m; f_coriolis = 1e-4)
    ca = ReynoldsStressClosure(m; transport = false, c_ε0 = 0, c_ε1 = 0)
    cb = ReynoldsStressClosure(m; transport = false, c_ε0 = 0, c_ε1 = 0, Ω = (0, 0, 0.5e-4))
    for cc in (ca, cb), k in 1:6
        interior_view(cc.τ[k]) .= τ0[k]
    end
    advance_sgs!(ca, simf, 600.0); advance_sgs!(cb, sim, 600.0)
    @test all(k -> interior_view(ca.τ[k]) == interior_view(cb.τ[k]), 1:6)
end

@testset "5. stratification: equilibrium anisotropy against the algebraic root" begin
    S, Δ, θ₀, g = 0.05, 10.0, 300.0, 9.81
    β = g / θ₀
    # the steady LRR-IP + Daly–Harlow equations for u = Sz, θ = θ₀ + Γz,
    # written out by hand (unknowns τ₁₁, τ₂₂, τ₃₃, τ₁₃; τ₁₂ = τ₂₃ = 0)
    function residual(x, N2)
        t11, t22, t33, t13 = x
        e = (t11 + t22 + t33) / 2
        N = sqrt(N2)
        l = min(Δ, 0.76 * sqrt(e) / N)
        cε = 0.19 + 0.51 * l / Δ
        r = cε * sqrt(e) / l
        κ = (1 + 2l / Δ) * 0.1 * l * sqrt(e)
        cθ = 3κ / (2e)
        Γ = N2 / β
        P11, P13, Pk = -2t13 * S, -t33 * S, -t13 * S
        h1, h3 = -cθ * t13 * Γ, -cθ * t33 * Γ
        B33, B13, Bk = 2β * h3, β * h1, β * h3
        C1, C2, C3 = 1.8, 0.6, 0.5
        iso(Pd, Bd) = -(2 / 3) * r * e + C1 * r * (2e / 3) + C2 * (2Pk / 3) + C3 * (2Bk / 3)
        return [P11 - C1 * r * t11 - C2 * P11 + iso(P11, 0),
                -C1 * r * t22 + iso(0, 0),
                B33 - C1 * r * t33 - C3 * B33 + iso(0, B33),
                P13 + B13 - C1 * r * t13 - C2 * P13 - C3 * B13]
    end
    function newton(x, N2)
        for _ in 1:100
            F = residual(x, N2)
            J = zeros(4, 4)
            for k in 1:4
                d = zeros(4); d[k] = 1e-7 * max(abs(x[k]), 1e-6)
                J[:, k] = (residual(x .+ d, N2) .- F) ./ d[k]
            end
            dx = J \ F
            x = x .- dx
            maximum(abs, dx) < 1e-15 * maximum(abs, x) && break
        end
        return x
    end
    Ris = (0.02, 0.05, 0.08, 0.11)
    b33s = Float64[]; Nls = Float64[]
    t11, t22, t12 = lrr_shear_equilibrium()
    e_n = (t12 * S * Δ / 0.7)^2
    x = [t11, t22, t22, -t12] .* e_n
    for Ri in Ris
        N2 = Ri * S^2
        x = newton(x, N2)
        @test maximum(abs, residual(x, N2)) < 1e-14 * maximum(abs, x) * S
        sim = rsm_shear_box(S, Ri)
        c = ReynoldsStressClosure(sim.mesh; e0 = 1e-3, transport = false, θ₀ = θ₀, g_accel = g)
        for _ in 1:6000
            advance_sgs!(c, sim, 5.0)
        end
        τc = ntuple(k -> mean(interior_view(c.τ[k])[:, :, 4:13]), 6)
        @test maximum(abs.(collect(τc[[1, 2, 3, 5]]) .- x)) / maximum(abs, x) < 1e-6
        @test maximum(abs, τc[4]) < 1e-14 && maximum(abs, τc[6]) < 1e-14
        b, II, _ = stress_anisotropy(τc)
        e = half_trace(τc)
        l, _ = tke_length_scale(e, N2, Δ)
        push!(b33s, b[3]); push!(Nls, sqrt(N2) * l / sqrt(e))
        @test reynolds_stress_statistics(c).clips == 0
        @info @sprintf("  Ri = %.2f: e = %.4e m²/s², N l/√e = %.3f, b33 = %.4f, b11 = %.4f, II = %.4f, |τ − root|/|τ| = %.1e",
                       Ri, e, Nls[end], b[3], b[1], II,
                       maximum(abs.(collect(τc[[1, 2, 3, 5]]) .- x)) / maximum(abs, x))
    end
    @test all(<(0), b33s)                                # vertical fluctuations suppressed
    @test issorted(Nls) && issorted(abs.(b33s))          # |b33| grows with N l/√e
end

@testset "6. realizability and energy in the decaying box" begin
    sim = rsm_flow_box(0.1)
    c = ReynoldsStressClosure(sim.mesh)
    initialize_reynolds_stress!(c, sim; e_min = 1e-4)
    @test c.advection === :weno_z5
    dt = 0.1
    worst = 0.0; allreal = true
    V = sim.mesh.dx * sim.mesh.dy * sim.mesh.dz
    E0 = kinetic_energy(sim) + reynolds_stress_budget(c, sim).tke
    for n in 1:200
        KE0 = kinetic_energy(sim); e0 = reynolds_stress_budget(c, sim).tke
        step_rk3!(sim, dt; sgs = c)
        KE1 = kinetic_energy(sim); e1 = reynolds_stress_budget(c, sim).tke
        st = reynolds_stress_statistics(c)
        # Σ(KE + Σ e V) changes by −Σ ε V dt (+ the booked clips) + truncation
        res = ((KE1 + e1) - (KE0 + e0) + dt * st.dissipation - st.step_clipped_energy) / (KE0 + e0)
        worst = max(worst, abs(res))
        allreal &= all(i -> is_realizable(ntuple(k -> interior_view(c.τ[k])[i], 6)),
                       eachindex(interior_view(c.τ[1])))
    end
    st = reynolds_stress_statistics(c)
    E1 = kinetic_energy(sim) + reynolds_stress_budget(c, sim).tke
    @info @sprintf("  decaying box (16³, 200 RK3 steps of 0.1 s): max energy residual %.2e per step, %d clips (%.1e of the SGS energy), total energy %.4e → %.4e",
                   worst, st.clips, st.clipped_energy / reynolds_stress_budget(c, sim).tke, E0, E1)
    @test allreal
    @test worst <= 1e-6
    @test st.clipped_energy <= 1e-3 * reynolds_stress_budget(c, sim).tke
    @test E1 < E0

    # the exchange term on its own is exact: the resolved-KE tendency of the
    # stress divergence is minus the SGS production (dual gradient), here
    # with walls in z and a solid block (closed edges)
    m = RegularGrid(; size = (10, 8, 6), extent = (10.0, 8.0, 6.0))
    simw = BoussinesqSimulation(m; bcs = FieldBoundaryConditions(:periodic, :periodic, :wall),
                                θ_bcs = FieldBoundaryConditions(:periodic, :periodic, :noflux))
    rng = Xoshiro(6)
    for f in (simw.u, simw.v, simw.w)
        interior_view(f) .= randn(rng, size(interior_view(f)))
    end
    interior_view(simw.w)[:, :, 1] .= 0
    mask = falses(size(simw.θ.data))
    mask[m.Hx+3:m.Hx+5, m.Hy+3:m.Hy+4, m.Hz+1:m.Hz+2] .= true
    cw = ReynoldsStressClosure(m; transport = false, mask = mask)
    for k in 1:6
        interior_view(cw.τ[k]) .= k <= 3 ? 0.5 .+ rand(rng, size(interior_view(cw.τ[k]))) :
                                           0.1 .* randn(rng, size(interior_view(cw.τ[k])))
        cw.τ[k].data[mask] .= 0
    end
    for f in (simw.u, simw.v, simw.w)
        fill_halos!(f, simw.bcs)
    end
    f = compute_sgs!(cw, simw)
    tu = Field(FaceX(), m); tv = Field(FaceY(), m); tw = Field(FaceZ(), m)
    add_sgs_momentum!(tu, tv, tw, cw, f, simw)
    ke = sum(interior_view(simw.u) .* interior_view(tu)) + sum(interior_view(simw.v) .* interior_view(tv)) +
         sum(interior_view(simw.w) .* interior_view(tw))
    # Σ τ^dev:G with the dual gradient (the isotropic part pairs with the divergence)
    bnd = (false, false, true)
    pd = 0.0
    for k in 1:m.Nz, j in 1:m.Ny, i in 1:m.Nx
        I, J, K = m.Hx + i, m.Hy + j, m.Hz + k
        τc = ntuple(q -> cw.τ[q].data[I, J, K], 6)
        e = half_trace(τc)
        G = dual_velocity_gradient(simw.u, simw.v, simw.w, I, J, K; bounded = bnd, mask = mask)
        pd += (τc[1] - 2e / 3) * G[1] + (τc[2] - 2e / 3) * G[5] + (τc[3] - 2e / 3) * G[9] +
              τc[4] * (G[2] + G[4]) + τc[5] * (G[3] + G[7]) + τc[6] * (G[6] + G[8])
    end
    @info @sprintf("  stress divergence vs dual production (walls, solid block): %.1e", abs(ke - pd) / sum(abs, interior_view(tu)))
    @test abs(ke - pd) <= 1e-13 * sum(abs, interior_view(tu))
    @test abs(sum(interior_view(tu))) < 1e-12 && abs(sum(interior_view(tv))) < 1e-12
end

@testset "shared operators: Daly–Harlow flux through the P55-c tensor operator" begin
    # isotropic τ = (2/3)eI: the RSM θ tendency is the TKE closure's (κ_t = (1+2l/Δ)ν_t)
    sim = rsm_flow_box(0.5; N = 8, L = 80.0)
    rng = Xoshiro(7)
    interior_view(sim.θ) .= 300 .+ rand(rng, 8, 8, 8)
    fill_halos!(sim.θ, sim.θ_bcs)
    e = 0.02 .+ 0.01 .* rand(rng, 8, 8, 8)
    tke = TKEClosure(sim.mesh)
    interior_view(tke.e) .= e
    rsm = ReynoldsStressClosure(sim.mesh)
    for k in 1:3
        interior_view(rsm.τ[k]) .= 2 .* e ./ 3
    end
    for f in (sim.u, sim.v, sim.w)
        fill_halos!(f, sim.bcs)
    end
    t1 = Field(CellCenter(), sim.mesh); t2 = Field(CellCenter(), sim.mesh)
    f1 = compute_sgs!(tke, sim); κ1 = copy(sim.κt.data); ν1 = copy(sim.νt.data)
    add_sgs_scalar!(t1, sim.θ, tke, f1, sim)
    f2 = compute_sgs!(rsm, sim)
    add_sgs_scalar!(t2, sim.θ, rsm, f2, sim)
    # ν_t: the same function (e = ½ tr τ differs from the TKE's e in the last bit)
    @test maximum(abs.(interior_view(sim.νt) .- ν1[4:11, 4:11, 4:11])) <= 1e-15 * maximum(ν1)
    @test maximum(abs.(sim.κt.data[4:11, 4:11, 4:11] .- κ1[4:11, 4:11, 4:11])) <= 1e-15 * maximum(κ1)
    @test maximum(abs.(interior_view(t2) .- interior_view(t1))) <= 1e-12 * maximum(abs, interior_view(t1))
    # anisotropic τ: K = c_θ τ, exactly what add_tensor_diffusion_scalar! gives
    interior_view(rsm.τ[5]) .= 0.3 .* (2 .* e ./ 3)
    f2 = compute_sgs!(rsm, sim)
    t3 = Field(CellCenter(), sim.mesh); add_sgs_scalar!(t3, sim.θ, rsm, f2, sim)
    @test maximum(abs.(interior_view(t3) .- interior_view(t2))) > 1e-3 * maximum(abs, interior_view(t2))
    @test interior_view(rsm.K.K13) ≈ interior_view(rsm.K.K11) .* 0.3 rtol = 1e-14
end

@testset "realizability projection, rollback, IMEX refusal" begin
    # Schumann's conditions and the nearest PSD tensor
    @test is_realizable((1.0, 1.0, 1.0, 0.0, 0.0, 0.0))
    @test !is_realizable((1.0, 1.0, 1.0, 1.5, 0.0, 0.0))     # τ₁₂² > τ₁₁τ₂₂
    @test !is_realizable((1.0, -0.1, 1.0, 0.0, 0.0, 0.0))
    @test !is_realizable((1.0, 1.0, 1.0, 0.9, 0.9, -0.9))    # 2×2 minors ≥ 0, det < 0
    rng = Xoshiro(9)
    for _ in 1:50
        A = randn(rng, 3, 3); M = (A + A') / 2
        τ = (M[1, 1], M[2, 2], M[3, 3], M[1, 2], M[1, 3], M[2, 3])
        τp, changed = project_realizable(τ)
        λ = eigvals(Symmetric(M))
        @test changed == !is_realizable(τ)
        minimum(λ) < -1e-9 * maximum(abs, λ) && @test changed
        λp = eigvals(Symmetric(full(τp)))
        @test minimum(λp) >= -1e-12 * maximum(abs, λ)
        @test sort(λp) ≈ sort(max.(λ, 0)) atol = 1e-12 * maximum(abs, λ)
    end
    # a rejected adaptive step: the state rolls back bit for bit
    sim = rsm_flow_box(0.3; N = 8, L = 80.0)
    c = ReynoldsStressClosure(sim.mesh)
    initialize_reynolds_stress!(c, sim; e_min = 1e-3)
    before = map(f -> copy(f.data), c.τ); stb = reynolds_stress_statistics(c)
    s = sgs_state(c)
    step_rk3!(sim, 0.5; sgs = c)
    @test any(k -> c.τ[k].data != before[k], 1:6)
    restore_sgs_state!(c, s)
    @test all(k -> c.τ[k].data == before[k], 1:6)
    @test reynolds_stress_statistics(c) == stb
    # step_imex! cannot represent the stress: refused, not silently replaced
    @test sgs_explicit_stress(c)
    @test_throws ArgumentError step_imex!(sim, 0.5, ARK2(); sgs = c)
end

end
