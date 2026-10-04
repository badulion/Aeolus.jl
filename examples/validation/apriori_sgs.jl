# =========================================================================
# A-priori SGS-stress analysis on filtered DNS — Phase 56 (P56-c, #178)
#
# Decaying isotropic turbulence computed by Aeolus (the fixture of the CI
# test, test/fixtures/dns_decay.jl, here at 96³ by default), filtered with
# a Gaussian at Δ = 4h, 8h and 16h. For each width the exact subfilter
# stress is compared with the Smagorinsky model (at the dissipation-matched
# C_s), the Bardina scale-similarity model and the Clark gradient model,
# and the dynamic C_s is evaluated (a) on the DNS grid with the
# second-moment test-to-grid ratio and (b) on the LES grid of spacing Δ
# (Fourier coarse-grained field, default ratio 3).
#
#   julia --project=. examples/validation/apriori_sgs.jl [N]
#
# writes examples/validation/output/apriori_sgs_<N>.md (correlation tables)
# =========================================================================

using Aeolus
using Aeolus.Fields: interior_view
using Printf
include(joinpath(@__DIR__, "..", "..", "test", "fixtures", "dns_apriori.jl"))   # + dns_decay.jl

N = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 96

d = dns_decay(; N)
@printf("DNS %d³: ν = %.4g, %d steps, %.0f s; Re_λ %.1f; k_max η ≥ %.2f (%.2f at t = %.1f); S = %.3f; max |budget err| %.1e\n",
        N, d.ν, d.nsteps, d.wall, d.Re_λ, d.kmax_eta, d.kmax_eta_end, d.t[end], d.S[end], maximum(abs, d.budget_err))

sim = d.sim; g = sim.mesh; h = g.dx
bcs = FieldBoundaryConditions(:periodic)
rows = String[]
cs_rows = String[]
for r in (4, 8, 16)
    Δ = r * h
    filt = (o, f) -> gaussian_filter!(o, f, Δ)
    ub = ntuple(_ -> Field(CellCenter(), g), 3)
    τe = SymmetricTensorField(g)
    subfilter_stress!(τe, sim.u, sim.v, sim.w, filt; ubar = ub)
    S = filtered_strain(ub...)
    cs = dissipation_matched_cs(τe, S, Δ)
    τs = SymmetricTensorField(g); smagorinsky_stress!(τs, S, cs, Δ)
    τb = SymmetricTensorField(g); bardina_stress!(τb, ub..., filt)
    λ1, _, _ = stress_eigvals(τe)
    minλ = minimum(λ1 ./ (τe.xx .+ τe.yy .+ τe.zz))
    for (name, τm) in (("Smagorinsky", τs), ("Bardina", τb), ("gradient", clark_gradient_stress(ub, Δ, h)))
        m = apriori_metrics(τm, τe, S)
        push!(rows, @sprintf("| %2dh | %-11s | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f | %.1f%% | %.3f | %.4f |",
                             r, name, m.ρ.xx, m.ρ.yy, m.ρ.zz, m.ρ.xy, m.ρ.xz, m.ρ.yz, m.ρ_Π, m.Π_ratio,
                             100m.backscatter_exact, m.II, m.III))
    end
    # dynamic C_s, (a) DNS grid, second-moment ratio
    ff = (Field(FaceX(), g), Field(FaceY(), g), Field(FaceZ(), g))
    for (c, f) in zip(ff, (sim.u, sim.v, sim.w))
        gaussian_filter!(c, f, Δ); fill_halos!(c, bcs)
    end
    ws = DynamicSmagorinskyWorkspace(g)
    compute_dynamic_smagorinsky_νt!(Field(CellCenter(), g), ff...; workspace = ws, filter_ratio = sqrt(1 + 8h^2 / Δ^2))
    cs_fine = germano_lilly_cs(ws) * h / Δ
    # (b) LES grid of spacing Δ
    cs_coarse = N ÷ r >= 8 ? first(dynamic_cs_on_coarse_grid(sim, r, Δ)) : NaN
    coarse = isnan(cs_coarse) ? "— (grid < 8³)" : @sprintf("%.3f (%+.1f%%)", cs_coarse, 100(cs_coarse / cs - 1))
    push!(cs_rows, @sprintf("| %2dh | %.1f | %.3f | %.3f (%+.1f%%) | %s | %.1e |", r, Δ / d.η, cs,
                            cs_fine, 100(cs_fine / cs - 1), coarse, minλ))
end

out = joinpath(@__DIR__, "output", "apriori_sgs_$(N).md")
mkpath(dirname(out))
open(out, "w") do io
    println(io, "# A-priori SGS stress, filtered DNS $(N)³ (Gaussian filter)\n")
    @printf(io, "DNS: ν = %.4g, t = %.1f, %d RK3 steps, %.0f s; Re_λ = %.1f; k_max η ≥ %.2f; derivative skewness %.3f; max |dK/dt/(−ε) − 1| = %.1e.\n\n",
            d.ν, d.t[end], d.nsteps, d.wall, d.Re_λ, d.kmax_eta, d.S[end], maximum(abs, d.budget_err))
    println(io, "## Correlations with the exact stress (deviatoric)\n")
    println(io, "| Δ | model | ρ11 | ρ22 | ρ33 | ρ12 | ρ13 | ρ23 | ρ_Π | ⟨Π⟩ ratio | backscatter | II | III |")
    println(io, "| --: | :-- | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: |")
    foreach(r -> println(io, r), rows)
    println(io, "\nII, III: mean Lumley invariants of the exact stress; backscatter: fraction of points with Π_exact < 0.\n")
    println(io, "## Smagorinsky coefficient\n")
    println(io, "| Δ | Δ/η | C_s (Π-matched) | dynamic, DNS grid | dynamic, LES grid | min λ₁/tr τ |")
    println(io, "| --: | --: | --: | --: | --: | --: |")
    foreach(r -> println(io, r), cs_rows)
end
print(read(out, String))
println("wrote ", out)
