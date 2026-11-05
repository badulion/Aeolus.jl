# The a-priori quantities of one Gaussian filter width Δ = r h on the
# state of `sim` (P56-c, #178): the exact subfilter stress τe, the filtered
# velocity and strain, the dissipation-matched C_s and the a-priori metrics
# of the Smagorinsky, Bardina and Clark gradient models. Shared by
# test/era6_apriori_sgs.jl (a cheap DNS state) and
# validation/era6_apriori_sgs.jl (the 96³ fixture).

using Aeolus
using Aeolus.Fields: interior_view

"""
    clark_gradient_stress(ub, Δ, h) -> SymmetricTensorField

The Clark gradient model ``τ_{ij} ≈ (Δ^2/12)\\,∂_k\\bar u_i\\,∂_k\\bar u_j``
of the cell-centred filtered velocity `ub` (periodic centred differences
of spacing `h`): the leading term of the Taylor expansion of the exact
subfilter stress of a Gaussian filter of width `Δ` (Clark, Ferziger &
Reynolds 1979, *J. Fluid Mech.* 91, 1).
"""
function clark_gradient_stress(ub, Δ, h)
    a = map(interior_view, ub)
    D(f, k) = (s = ntuple(i -> i == k ? 1 : 0, 3); (circshift(f, .-s) .- circshift(f, s)) ./ (2h))
    G = [D(a[i], k) for i in 1:3, k in 1:3]
    τ = SymmetricTensorField{Float64}(size(a[1])...)
    for (n, (i, j)) in enumerate(((1,1),(2,2),(3,3),(1,2),(1,3),(2,3)))
        Aeolus.Operators.components(τ)[n] .= (Δ^2 / 12) .* (G[i,1] .* G[j,1] .+ G[i,2] .* G[j,2] .+ G[i,3] .* G[j,3])
    end
    return τ
end

"""
    apriori_case(sim, r) -> NamedTuple

The a-priori quantities of the Gaussian filter of width `Δ = r h` (h the
grid spacing of `sim`) on the current velocity of `sim`: `(; Δ, filt, ub,
τe, S, cs, smag, bard, grad)`, the last three `apriori_metrics` of the
Smagorinsky (dissipation-matched C_s), Bardina and Clark gradient models
against the exact stress `τe`.
"""
function apriori_case(sim, r)
    g = sim.mesh; h = g.dx
    Δ = r * h
    filt = (o, f) -> gaussian_filter!(o, f, Δ)
    ub = ntuple(_ -> Field(CellCenter(), g), 3)
    τe = SymmetricTensorField(g)
    subfilter_stress!(τe, sim.u, sim.v, sim.w, filt; ubar = ub)
    S = filtered_strain(ub...)
    cs = dissipation_matched_cs(τe, S, Δ)
    τs = SymmetricTensorField(g); smagorinsky_stress!(τs, S, cs, Δ)
    τb = SymmetricTensorField(g); bardina_stress!(τb, ub..., filt)
    return (; Δ, filt, ub, τe, S, cs, smag = apriori_metrics(τs, τe, S), bard = apriori_metrics(τb, τe, S),
                grad = apriori_metrics(clark_gradient_stress(ub, Δ, h), τe, S))
end
