# The diagnosed plane-mean momentum budget (moved from the SBL run type's
# `SBLMomentumBudget`, P58-d #194, into the core budgets: ONE-c #452).
#
# Where the stage budgets measure what each stage entry did, this budget
# evaluates the physical terms of the plane-mean momentum equation from the
# state, per level `k`, for `U` and `V`:
#
#   ΔU_k = ∫ Σ_terms dt     (e.g. turbulent flux divergence, Coriolis,
#                            pressure gradient, advection, sponge)
#
# The run that owns it evaluates each term in every (sub)step and books it
# with the step's weight (`book_momentum!`); the surface stress enters
# level 1 of one named term (`surface_term`, the turbulent flux divergence:
# the wall stress is the flux through the surface face). Comparison terms
# (`diagnostics`, e.g. the core's own advection + SGS tendency) are booked
# alike but are not part of the closure. `momentum_closure` returns the
# change of the plane means, every term and the residual.

"""
    MomentumBudget(U0, V0, t0; terms, diagnostics = (), surface_term = first(terms))

The plane-mean momentum budget started from the plane-mean profiles `U0`,
`V0` (per level) at time `t0`. Fields:

* `terms` — the names of the closure terms, in the order they are summed;
* `u[term]`, `v[term]` — per level, ``∫`` term ``dt`` (m s⁻¹);
* `diagnostics` — names of comparison terms (`u`, `v` alike, not summed);
* `surface_u`, `surface_v` — ``∫ -τ_{wall}/Δz\\,dt`` into level 1, added
  to `surface_term`.
"""
mutable struct MomentumBudget
    U0::Vector{Float64}
    V0::Vector{Float64}
    t0::Float64
    terms::Vector{Symbol}
    diagnostics::Vector{Symbol}
    surface_term::Symbol
    u::Dict{Symbol,Vector{Float64}}
    v::Dict{Symbol,Vector{Float64}}
    surface_u::Float64
    surface_v::Float64
end

function MomentumBudget(U0::AbstractVector, V0::AbstractVector, t0::Real; terms, diagnostics = (),
                        surface_term::Symbol = first(terms))
    Nz = length(U0)
    length(V0) == Nz || throw(DimensionMismatch("MomentumBudget: U0 and V0 need the same levels"))
    names = collect(Symbol, terms); diag = collect(Symbol, diagnostics)
    surface_term in names || throw(ArgumentError("MomentumBudget: surface_term :$surface_term is not a term"))
    isempty(intersect(names, diag)) || throw(ArgumentError("MomentumBudget: a term is also a diagnostic"))
    u = Dict(t => zeros(Nz) for t in vcat(names, diag))
    v = Dict(t => zeros(Nz) for t in vcat(names, diag))
    return MomentumBudget(collect(Float64, U0), collect(Float64, V0), Float64(t0), names, diag,
                          surface_term, u, v, 0.0, 0.0)
end

"""
    book_momentum!(mb, term, k, du, dv) -> mb

Add `du`, `dv` (m s⁻¹: the term times the weighted step length) to level
`k` of `term` (a closure term or a diagnostic).
"""
@inline function book_momentum!(mb::MomentumBudget, term::Symbol, k::Integer, du::Real, dv::Real)
    @inbounds mb.u[term][k] += du
    @inbounds mb.v[term][k] += dv
    return mb
end

"""
    book_surface_momentum!(mb, du, dv) -> mb

Add `du`, `dv` (m s⁻¹) of the surface stress to level 1
(``-τ_{wall}/Δz`` times the weighted step length).
"""
book_surface_momentum!(mb::MomentumBudget, du::Real, dv::Real) =
    (mb.surface_u += du; mb.surface_v += dv; mb)

"""
    momentum_closure(mb, U, V; ks = eachindex(U)) -> NamedTuple

The budget on the levels `ks` for the current plane-mean profiles `U`,
`V`: `ΔU`, `ΔV`, the terms `u[term]`, `v[term]` (the surface stress added
to level 1 of `surface_term`; diagnostics included), the residuals
`res_u = ΔU − Σ terms` and `res_v`, the largest term magnitude `largest`
and `rel = max|res| / largest`.
"""
function momentum_closure(mb::MomentumBudget, U::AbstractVector, V::AbstractVector; ks = eachindex(U))
    ΔU = U .- mb.U0
    ΔV = V .- mb.V0
    tu = Dict(t => copy(x) for (t, x) in mb.u)
    tv = Dict(t => copy(x) for (t, x) in mb.v)
    tu[mb.surface_term][1] += mb.surface_u           # the surface stress enters level 1
    tv[mb.surface_term][1] += mb.surface_v
    su = copy(tu[mb.terms[1]]); sv = copy(tv[mb.terms[1]])
    for t in mb.terms[2:end]                          # summed in the terms' order
        su .+= tu[t]; sv .+= tv[t]
    end
    ru = ΔU .- su
    rv = ΔV .- sv
    largest = maximum(max(maximum(abs, tu[t][ks]), maximum(abs, tv[t][ks])) for t in mb.terms)
    res = max(maximum(abs, ru[ks]), maximum(abs, rv[ks]))
    return (ΔU = ΔU[ks], ΔV = ΔV[ks], u = Dict(t => x[ks] for (t, x) in tu),
            v = Dict(t => x[ks] for (t, x) in tv), res_u = ru[ks], res_v = rv[ks],
            largest = largest, rel = res / largest)
end
