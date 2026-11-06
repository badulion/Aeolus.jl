# Phase 56 (P56-e, #192): the TKE (P56-b) and Reynolds-stress
# (P56-d) SGS closures against filtered DNS, with Smagorinsky and dynamic
# Smagorinsky as baselines. The method and the fixtures are in
# the sgs_decay run type (formerly test/fixtures/sgs_validation.jl): one 48³ DNS of decaying isotropic
# turbulence (the P56-c fixture, Re_λ = 32 at t₀ = 2) continued for two
# turnovers T = L/u′ as is, with rotation (Ro = 0.2) and with a background
# stratification (Fr = 0.3); every LES (Δ = 4h on 12³) starts from the
# filtered DNS at t₀ (e and τᵢⱼ from the exact subfilter stress) and is
# compared with the filtered DNS at 16 equally spaced times.
#
# Measured (48³, seed 1; 44 s with JULIA_NUM_THREADS=2, quoted in
# docs/src/theory/sgs_tke.md, section "Validation against filtered DNS"):
#
#   base: u′ = 0.784, L = 1.382, T = 1.763, f = 2.835, N = 1.890
#
#   resolved KE error at t₀ + 2T    isotropic   rotating   stratified
#     Smagorinsky (C_s = 0.17)        +2.2 %     −11.9 %     −0.8 %
#     dynamic Smagorinsky             +9.2 %      +7.5 %    +11.6 %
#     TKE (Deardorff/Moeng)          −11.6 %     −22.6 %    +28.4 %
#     Reynolds stress (LRR-IP)       −11.7 %      +6.4 %    +15.9 %
#
#   e at t₀ + 2T: filtered DNS 0.0443, TKE 0.125, RSM 0.125 (isotropic).
#   DNS K(t₀+2T)/K(t₀): 0.295 isotropic, 0.389 rotating (full resolution).
#   Stratified, ⟨τ₃₃⟩/⟨τ₁₁⟩ over (t₀, t₀+2T]: DNS 0.70, RSM 0.76, TKE 1.00;
#   pointwise ρ(τ₃₃/τ₁₁) at t₀ + T/8: RSM 0.71, TKE 0.17;
#   II at t₀ + 2T: DNS 0.270, RSM 0.049, TKE 0.0012;
#   mean vertical SGS heat flux over (t₀, t₀+2T]: DNS −0.232,
#   TKE −0.315 (1.35×), RSM −0.347 (1.49×).
#
# Expectations of the issue that proved wrong at this Re_λ (the code was
# checked first: the LES start reproduces the filtered DNS exactly, see
# "start"), kept as @test_broken with the numbers:
#
#   * "resolved KE of every model within 10 % at t₀ + 2T" holds for both
#     Smagorinsky baselines but not for TKE (−11.6 %) and RSM (−11.7 %);
#   * "TKE: e within 25 % of ½⟨τₖₖ⟩" — e is 2.8× the DNS value.
#     Both have the same cause, asserted in "why": the Deardorff
#     constants (c_k = 0.1, c_ε = 0.70 at l = Δ; Moeng & Wyngaard 1988)
#     are inertial-range values, and Δ = 4h ≈ 8η is in the
#     near-dissipation range. With the exact e at t₀ the closure's
#     production c_kΔ√e·2⟨S̄S̄⟩ is 1.9× the exact transfer
#     Π = −⟨τᵢⱼS̄ᵢⱼ⟩, and its dissipation c_εe^{3/2}/Δ is 0.68× the viscous
#     dissipation of the subfilter scales 2ν(⟨SS⟩ − ⟨S̄S̄⟩): e grows for a
#     turnover while the DNS's decays, and ν_t follows it.
#   * "the RSM τ₃₃/τ₁₁ correlates with the DNS (ρ > 0.5)" holds pointwise
#     early on (0.71 at T/8, 0.56 at T/4, against 0.17, 0.20 for TKE) and
#     then decays for every model (0.02 at 2T) — the LES and the DNS
#     decorrelate pointwise within a turnover (τ₃₃ itself: 0.15 at 2T). It
#     is asserted at T/8 and T/4; the statistic that survives, the mean
#     ratio, is asserted over the whole run.
#   * "RSM's heat flux closer to the DNS than TKE's": both are within 2×,
#     but TKE is closer (1.35× vs 1.49×). The DNS flux oscillates with the
#     buoyancy period 2π/N = 1.9T and is counter-gradient for a third of
#     the run (up to +1.4); no down-gradient closure follows that, and the
#     RSM's slower e decay (0.029 vs 0.015 at 2T) makes its flux larger.
#
# The API it relies on (the step_rk3! hooks, the closures seeded from a
# stress) stays tested in test/sgs_validation.jl.

using Test
using Printf
using Aeolus

# the fixture (back in test/fixtures since ONE-l #475; it was the sgs_decay run type)
using Aeolus.Fields: interior_view
isdefined(@__MODULE__, :SGSValidation) || include(joinpath(@__DIR__, "fixtures", "sgs_validation.jl"))
using .SGSValidation
using .SGSValidation: VAL_RO, VAL_FR
import Aeolus.Operators: advance_sgs!

_pcorr(a, b) = (m = isfinite.(a) .& isfinite.(b); x = a[m] .- vmean(a[m]); y = b[m] .- vmean(b[m]);
                sum(x .* y) / sqrt(sum(abs2, x) * sum(abs2, y)))
_rel(a, b) = a / b - 1

@testset "P56-e: TKE and Reynolds-stress SGS against filtered DNS" begin
    wall = @elapsed begin
        base = validation_base(; N = 48)
        dns = Dict(c => run_dns_case(base, c; nsave = 16) for c in (:isotropic, :rotating, :stratified))
        models = (:smagorinsky, :dynamic, :tke, :rsm)
        les = Dict((c, m) => run_les(base, dns[c], m) for c in keys(dns) for m in models)
    end
    @printf("  P56-e: DNS spin-up %.1f s, DNS continuations %.1f s, LES %.1f s, total %.1f s\n",
            base.wall, sum(d.wall for d in values(dns)), sum(l.wall for l in values(les)), wall)
    iso, rot, str = dns[:isotropic], dns[:rotating], dns[:stratified]
    endK(c, m) = _rel(les[(c, m)].K[end], dns[c].snaps[end].K)

    @testset "fixture and start" begin
        # the P56-c fixture at 48³ (its own test checks it as a DNS)
        @test 25 < base.Re_λ < 40
        @test base.up / (base.f * base.L) ≈ VAL_RO
        @test base.up / (base.Nbv * base.L) ≈ VAL_FR
        # the three continuations share t₀
        @test iso.snaps[1].K == rot.snaps[1].K == str.snaps[1].K
        # every LES starts at the filtered DNS: resolved KE (after the
        # projection onto the LES grid's discrete divergence) and the SGS
        # energy of the exact stress, which the 2×2×2 centre sample keeps
        # positive semi-definite (no realizability correction needed)
        for m in models
            @test abs(_rel(les[(:isotropic, m)].K[1], iso.snaps[1].K)) < 0.02
        end
        @test les[(:isotropic, :tke)].e[1] ≈ iso.snaps[1].e_sgs rtol = 1e-12
        @test les[(:isotropic, :rsm)].e[1] ≈ iso.snaps[1].e_sgs rtol = 1e-12
        g = les[(:isotropic, :rsm)].sim.mesh
        c = ReynoldsStressClosure(g)
        _, nproj = initialize_reynolds_stress!(c, iso.snaps[1].τ)
        @test nproj == 0
        # the stratified DNS conserves K + PE up to dissipation: the
        # background term −Γw̄ and the buoyancy gθ′/θ₀ exchange energy
        # exactly, so K + PE falls at every step
        E = str.K .+ str.PE
        @test all(diff(E) .< 0)
        @test maximum(str.PE) > 0.1 * str.K[1]          # the exchange is not small
    end

    for c in (:isotropic, :rotating, :stratified)
        @printf("  P56-e %-10s KE error at t₀+2T: %s; e: DNS %.4f TKE %.4f RSM %.4f\n", c,
                join([@sprintf("%s %+.1f %%", m, 100endK(c, m)) for m in models], ", "),
                dns[c].snaps[end].e_sgs, les[(c, :tke)].e[end], les[(c, :rsm)].e[end])
    end

    @testset "isotropic decay" begin
        # baselines: within 10 % of the filtered DNS after two turnovers
        @test abs(endK(:isotropic, :smagorinsky)) < 0.10
        @test abs(endK(:isotropic, :dynamic)) < 0.10
        # expectation error at Re_λ = 32 (header): −11.6 %, −11.7 %
        @test_broken abs(endK(:isotropic, :tke)) < 0.10
        @test_broken abs(endK(:isotropic, :rsm)) < 0.10
        @test abs(endK(:isotropic, :tke)) < 0.15
        @test abs(endK(:isotropic, :rsm)) < 0.15
        # e against ½⟨τₖₖ⟩: expectation error, 2.8× (header)
        e_tke = les[(:isotropic, :tke)].e[end]; e_dns = iso.snaps[end].e_sgs
        @test_broken abs(_rel(e_tke, e_dns)) < 0.25
        # the trace of the RSM is the TKE closure's e equation with the exact
        # production: its SGS energy tracks the TKE closure's within 10 %
        @test abs(_rel(les[(:isotropic, :rsm)].e[end], e_tke)) < 0.10
    end

    @testset "why: the Deardorff constants at Δ ≈ 8η" begin
        sim = base.sim0; g = sim.mesh; Δ = 4g.dx
        uc = map(Aeolus.Operators._to_center_periodic, (sim.u, sim.v, sim.w))
        U = ntuple(m -> (F = Field(CellCenter(), g); interior_view(F) .= uc[m]; F), 3)
        Ub = ntuple(m -> (F = Field(CellCenter(), g); gaussian_filter!(F, U[m], Δ); F), 3)
        SS(S) = vmean(S.xx .^ 2 .+ S.yy .^ 2 .+ S.zz .^ 2 .+ 2 .* (S.xy .^ 2 .+ S.xz .^ 2 .+ S.yz .^ 2))
        S̄ = filtered_strain(Ub...)
        τ = SymmetricTensorField(g)
        subfilter_stress!(τ, sim.u, sim.v, sim.w, (o, f) -> gaussian_filter!(o, f, Δ))
        Π = vmean(sgs_dissipation(τ, S̄))
        e = 0.5 * vmean(τ.xx .+ τ.yy .+ τ.zz)
        l = 4 * (2π / 48)                                   # the LES grid's Δ
        νt, _ = tke_eddy_coefficients(e, l, l)
        P = νt * 2SS(S̄)
        ε_D = tke_dissipation_coefficient(l, l) * e^1.5 / l
        ε_sub = 2base.ν * (SS(filtered_strain(U...)) - SS(S̄))
        @printf("  P56-e why: Π = %.4f, P_TKE = %.4f (%.2f×), ε_D = %.4f, ε_sub = %.4f (%.2f×)\n",
                Π, P, P / Π, ε_D, ε_sub, ε_D / ε_sub)
        @test Π > 0                                       # forward transfer on average
        @test P / Π > 1.5                                 # measured 1.91
        @test ε_D / ε_sub < 0.8                           # measured 0.68
    end

    @testset "rotating decay" begin
        # Bardina, Ferziger & Rogallo (1985): rotation slows the decay
        @test rot.K[end] / rot.K[1] > iso.K[end] / iso.K[1] + 0.05
        @test rot.snaps[end].K > iso.snaps[end].K
        e_rsm = abs(endK(:rotating, :rsm)); e_smag = abs(endK(:rotating, :smagorinsky))
        @printf("  P56-e rotating: |KE error| RSM %.3f, Smagorinsky %.3f, dynamic %.3f, TKE %.3f\n",
                e_rsm, e_smag, abs(endK(:rotating, :dynamic)), abs(endK(:rotating, :tke)))
        @test e_rsm < e_smag
        # the RSM is the only closure that sees the rotation directly (the
        # Coriolis term of the stresses): it beats the TKE closure too
        @test e_rsm < abs(endK(:rotating, :tke))

        # tr τ is invariant under the Coriolis term in 3-D: the same
        # anisotropic stress field (the exact one at t₀) advanced at rest
        # with and without Ω — the components differ at O(1), the trace to
        # round-off
        g = les[(:rotating, :rsm)].sim.mesh
        sim = BoussinesqSimulation(g; ν = base.ν)
        trs = map(((0.0, 0.0, 0.0), (0.0, 0.0, base.f / 2))) do Ω
            c = ReynoldsStressClosure(g; Ω = Ω)
            initialize_reynolds_stress!(c, rot.snaps[1].τ)
            for _ in 1:20
                advance_sgs!(c, sim, 0.02)
            end
            c
        end
        tr(c) = interior_view(c.τ[1]) .+ interior_view(c.τ[2]) .+ interior_view(c.τ[3])
        d11 = maximum(abs, interior_view(trs[1].τ[1]) .- interior_view(trs[2].τ[1]))
        dtr = maximum(abs, tr(trs[1]) .- tr(trs[2]))
        @printf("  P56-e Coriolis: max |Δτ₁₁| = %.3e, max |Δ tr τ| = %.3e\n", d11, dtr)
        @test d11 > 1e-3 * maximum(tr(trs[1]))
        @test dtr < 1e-12 * maximum(tr(trs[1]))
    end

    @testset "stratified decay" begin
        n = length(str.snaps)
        r_dns = [s.r33 for s in str.snaps]
        r_rsm = les[(:stratified, :rsm)].r33; r_tke = les[(:stratified, :tke)].r33
        # the DNS's subfilter stress becomes anisotropic, vertical suppressed
        @test minimum(r_dns) < 0.5
        # pointwise correlation of τ₃₃/τ₁₁ while the LES still follows the
        # DNS (T/8, T/4; header), and RSM above TKE there
        ρ(m, i) = _pcorr(les[(:stratified, m)].τs[i].zz ./ les[(:stratified, m)].τs[i].xx,
                         str.snaps[i].τ.zz ./ str.snaps[i].τ.xx)
        @printf("  P56-e stratified ρ(τ₃₃/τ₁₁): RSM %s\n", join([@sprintf("%.2f", ρ(:rsm, i)) for i in 1:n], " "))
        for i in (2, 3)
            @test ρ(:rsm, i) > 0.5
            @test ρ(:rsm, i) > ρ(:tke, i)
        end
        # the mean anisotropy over the run: RSM closer to the DNS than TKE
        mr(x) = vmean(x[2:end])
        @printf("  P56-e stratified ⟨τ₃₃⟩/⟨τ₁₁⟩: DNS %.3f, RSM %.3f, TKE %.3f\n", mr(r_dns), mr(r_rsm), mr(r_tke))
        @test abs(mr(r_rsm) - mr(r_dns)) < abs(mr(r_tke) - mr(r_dns))
        @test mr(r_rsm) < 0.85                           # vertical suppression
        @test abs(mr(r_tke) - 1) < 0.03                  # an eddy viscosity cannot
        # Lumley II at t₀ + 2T
        II_d = str.snaps[end].II_mean
        II_r = les[(:stratified, :rsm)].II[end]; II_t = les[(:stratified, :tke)].II[end]
        @printf("  P56-e stratified II: DNS %.4f, RSM %.4f, TKE %.4f\n", II_d, II_r, II_t)
        @test abs(II_r - II_d) < abs(II_t - II_d)
        # vertical SGS heat flux, mean over (t₀, t₀ + 2T]
        h_d = vmean([s.hw_mean for s in str.snaps[2:end]])
        h_t = vmean(les[(:stratified, :tke)].hw[2:end]); h_r = vmean(les[(:stratified, :rsm)].hw[2:end])
        @printf("  P56-e stratified ⟨w′θ′⟩_sgs: DNS %.3f, TKE %.3f (%.2f×), RSM %.3f (%.2f×)\n",
                h_d, h_t, h_t / h_d, h_r, h_r / h_d)
        @test h_d < 0                                     # down-gradient on average
        @test 0.5 < h_t / h_d < 2
        @test 0.5 < h_r / h_d < 2
        @test maximum(s.hw_mean for s in str.snaps) > 0   # counter-gradient phases
        # expectation error (header): TKE is the closer one
        @test_broken abs(h_r - h_d) < abs(h_t - h_d)
    end
end
