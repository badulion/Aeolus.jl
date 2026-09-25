# Phase 45, issue #13 — sub-grid supersaturation and condensational growth.
#
# Part-file of `TurbulentMicrophysics`; included from
# src/TurbulentMicrophysics.jl.
#
# Diffusional growth obeys r dr/dt = G S (Phase 41, `grow_droplet`), i.e.
# dr²/dt = 2 G S: at a single supersaturation every droplet gains the SAME
# r², and the radius spectrum narrows (small droplets gain more radius).
# In a turbulent cloud each droplet follows its own path through the
# sub-grid supersaturation field, and what it gains is 2G ∫ S dt along
# that path. The spread of that integral widens the r² distribution:
#
#     Var(r²) = (2G)² Var(∫₀ᵗ S' dt').
#
# S' is modelled per droplet as an Ornstein-Uhlenbeck process with the
# sub-grid standard deviation σ_S and a Lagrangian correlation time τ,
# for which (e.g. Sardina et al. 2015; Paoli & Shariff 2009)
#
#     Var(∫₀ᵗ S' dt') = 2 σ_S² τ [t − τ (1 − e^{−t/τ})]
#                     → σ_S² t²   (t ≪ τ, "frozen" sub-volumes)
#                     → 2 σ_S² τ t (t ≫ τ, diffusive: σ_{r²} ∝ √t).
#
# The water that condenses on a droplet is taken from the cell's vapour,
# so however the fluctuations spread the spectrum the cell's total water
# is exactly what it was: vapour and heat exchange go through the
# Phase 41 `apply_condensate_tendency!`, unchanged. The fluctuations only
# decide WHICH droplets the condensate goes to.

export SupersaturationFluctuations, supersaturation_fluctuations,
       advance_fluctuations!, local_supersaturation,
       supersaturation_integral_variance,
       sgs_condensation_growth!, sgs_condensation_step!

"""
    SupersaturationFluctuations

Per-droplet sub-grid supersaturation state for stochastic condensational
growth. Droplet `i` sees `S̄ + σ_S ξ[i]`, where `ξ[i]` is a unit-variance
Ornstein-Uhlenbeck process with correlation time `τ` (s; `Inf` freezes
each droplet in its sub-volume).

  * `ξ`   — unit-variance fluctuation per superdroplet (indexed like
            `sd.radius`)
  * `σ_S` — standard deviation of the supersaturation (fractional)
  * `τ`   — Lagrangian correlation time of `S'` (s)
  * `rng` — random-number generator driving the process

Build with [`supersaturation_fluctuations`](@ref).
"""
mutable struct SupersaturationFluctuations{R<:AbstractRNG}
    ξ::Vector{Float64}
    σ_S::Float64
    τ::Float64
    rng::R
end

"""
    supersaturation_fluctuations(n; σ_S, τ = Inf, rng = MersenneTwister(0))
    supersaturation_fluctuations(sd::SuperdropletSet; kwargs...)
        -> SupersaturationFluctuations

Draw the initial fluctuations of `n` droplets (or one per superdroplet of
`sd`) from the stationary distribution, `ξ ~ N(0, 1)`. `σ_S` is typically
[`SubgridCloud`](@ref)`.σ_S` from issue #12, and `τ` the sub-grid eddy
turnover time (`Δ^{2/3} ε^{-1/3}` for filter width `Δ` and dissipation
rate `ε`). Using the same `rng` seed for several `σ_S` gives the same
unit-noise realisation scaled by `σ_S` (common random numbers).
"""
function supersaturation_fluctuations(n::Integer; σ_S::Real, τ::Real = Inf,
                                      rng::AbstractRNG = MersenneTwister(0))
    σ_S >= 0 || throw(ArgumentError("supersaturation_fluctuations: σ_S must be ≥ 0"))
    τ > 0 || throw(ArgumentError("supersaturation_fluctuations: τ must be > 0"))
    return SupersaturationFluctuations(randn(rng, n), float(σ_S), float(τ), rng)
end

supersaturation_fluctuations(sd::SuperdropletSet; kw...) =
    supersaturation_fluctuations(nsuperdroplets(sd); kw...)

"""
    advance_fluctuations!(fl, dt; indices = nothing) -> fl

Exact Ornstein-Uhlenbeck update over `dt`:
`ξ ← ρ ξ + √(1−ρ²) N(0,1)` with `ρ = exp(−dt/τ)`. Stationary for any
`dt` (the variance stays 1); `τ = Inf` leaves `ξ` unchanged. With
`indices`, only those droplets advance — the ones whose time has actually
moved on (e.g. the droplets of the one cell being stepped).
"""
function advance_fluctuations!(fl::SupersaturationFluctuations, dt::Real;
                               indices::Union{Nothing,AbstractVector{Int}} = nothing)
    isfinite(fl.τ) || return fl
    ρ = exp(-float(dt) / fl.τ)
    a = sqrt(max(1 - ρ^2, 0.0))
    idx = indices === nothing ? eachindex(fl.ξ) : indices
    @inbounds for i in idx
        fl.ξ[i] = ρ * fl.ξ[i] + a * randn(fl.rng)
    end
    return fl
end

"""
    local_supersaturation(fl, S̄) -> Vector

Supersaturation `S̄ + σ_S ξ[i]` seen by each droplet.
"""
local_supersaturation(fl::SupersaturationFluctuations, S̄::Real) =
    float(S̄) .+ fl.σ_S .* fl.ξ

"""
    supersaturation_integral_variance(σ_S, τ, t) -> Var(∫₀ᵗ S' dt')

Variance of the time integral of an Ornstein-Uhlenbeck supersaturation
with standard deviation `σ_S` and correlation time `τ`,

```math
\\mathrm{Var}\\!\\int_0^t S'\\,dt' = 2σ_S^2 τ\\left[t - τ\\left(1 - e^{-t/τ}\\right)\\right],
```

which is `σ_S² t²` for `τ = Inf`. Multiplied by `(2G)²`, with `G` the
growth coefficient of `r dr/dt = G S`, it is the variance of `r²`
acquired by an initially monodisperse population (Sardina et al. 2015,
eq. 5; Paoli & Shariff 2009). For `t ≫ τ` it grows linearly — the
spectral width `σ_{r²}` grows as `√t`, without bound, which is the
mechanism by which turbulence broadens droplet spectra that pure
condensation would narrow.
"""
function supersaturation_integral_variance(σ_S::Real, τ::Real, t::Real)
    isfinite(τ) || return float(σ_S)^2 * float(t)^2
    x = float(t) / τ
    # t − τ(1 − e^{−x}) = τ (x − 1 + e^{−x}); expm1 keeps it exact for x ≪ 1
    return 2 * float(σ_S)^2 * τ^2 * (x + expm1(-x))
end

# Sub-steps needed to resolve the correlation time: the droplet holds its
# S for one sub-step, which reproduces ∫S'dt only if h ≪ τ. With h = τ/10
# the error in the integral variance is (h/τ)²/12 ≈ 0.08 %.
# Capped like Phase 41's `_phase_substeps`, so a tiny τ cannot demand an
# unbounded loop (or overflow `ceil(Int, ·)`).
_ou_substeps(σ_S::Real, τ::Real, Δt) =
    (σ_S > 0 && isfinite(τ)) ? Int(clamp(ceil(10 * Δt / τ), 1, 100_000)) : 1
_ou_substeps(fl::SupersaturationFluctuations, Δt) = _ou_substeps(fl.σ_S, fl.τ, Δt)

"""
    sgs_condensation_growth!(sd, fl, S̄, T, p, dt; indices = nothing,
                             substeps = 0, mc, c) -> Δm

Grow every superdroplet at the prescribed mean state (`S̄`, `T`, `p`)
over `dt`, with droplet `i` in the local supersaturation
`S̄ + σ_S ξ[i]` ([`local_supersaturation`](@ref)), then advance the
fluctuations by `dt`. The growth is Phase 41's `condensation_growth!`;
returns the real water mass condensed (kg). Does not touch a host cell
— see [`sgs_condensation_step!`](@ref). With `indices`, only those
droplets grow and only their fluctuations advance.

Each droplet holds its `S` for the whole `dt`, so `dt` should be well
below `τ` (`sgs_condensation_step!` sub-steps to `τ/10` itself).
`fl.σ_S` is a single value; when stepping several cells with different
sub-grid spreads, set `fl.σ_S` to the cell's value before each call.
"""
function sgs_condensation_growth!(sd::SuperdropletSet, fl::SupersaturationFluctuations,
                                  S̄::Real, Temp::Real, p::Real, dt::Real;
                                  indices::Union{Nothing,AbstractVector{Int}} = nothing,
                                  substeps::Integer = 0,
                                  mc::MicrophysicalConstants = DEFAULT_MICRO,
                                  c::ThermodynamicConstants = DEFAULT_CONSTANTS)
    length(fl.ξ) == nsuperdroplets(sd) || throw(DimensionMismatch(
        "sgs_condensation_growth!: $(length(fl.ξ)) fluctuations for " *
        "$(nsuperdroplets(sd)) superdroplets"))
    Δm = condensation_growth!(sd, local_supersaturation(fl, S̄), Temp, p, dt;
                              indices = indices, substeps = substeps, mc = mc, c = c)
    # advance only the droplets that were grown: a caller stepping cell by
    # cell (`sgs_condensation_step!(...; index = k)` for every k) must not
    # age every droplet's fluctuation once per cell, which would shorten
    # the effective correlation time to τ / (number of cells).
    advance_fluctuations!(fl, dt; indices = indices)
    return Δm
end

"""
    sgs_condensation_step!(sd, cell, dt, fl; index = nothing, nsub = 0,
                           substeps = 0, mc, c) -> (Δm, Δq_vap, ΔT)

Two-way coupled condensation with sub-grid supersaturation fluctuations:
Phase 41's `condensation_step!` in which each superdroplet grows in
`S̄ + σ_S ξ[i]`, `S̄` being the cell supersaturation. Vapour removal and
latent heating go through `apply_condensate_tendency!` with the total
condensed mass, so the cell's total water (`cell_total_water`) is
conserved to round-off whatever `σ_S`; the fluctuations only redistribute
the condensate among droplets, broadening the spectrum.

Sub-stepping is `condensation_step!`'s phase-relaxation count, raised
when `σ_S > 0` so that each sub-step is at most `τ/10`. With `σ_S = 0` the
result is **bit for bit** `condensation_step!(sd, cell, dt)`.
"""
function sgs_condensation_step!(sd::SuperdropletSet, cell::EulerianCell, dt::Real,
                                fl::SupersaturationFluctuations;
                                index::Union{Nothing,Integer} = nothing,
                                nsub::Integer = 0, substeps::Integer = 0,
                                mc::MicrophysicalConstants = DEFAULT_MICRO,
                                c::ThermodynamicConstants = DEFAULT_CONSTANTS)
    idx = index === nothing ? collect(eachindex(sd.radius)) :
                              findall(==(Int(index)), sd.cell)
    isempty(idx) && return (0.0, 0.0, 0.0)
    Δt = float(dt)
    ns = nsub > 0 ? Int(nsub) :
         max(_phase_substeps(sd, cell, Δt, idx, mc, c), _ou_substeps(fl, Δt))
    h = Δt / ns

    Δm_tot = 0.0; Δq_tot = 0.0; ΔT_tot = 0.0
    for _ in 1:ns
        p = cell_pressure(cell, 0.0, c)
        S̄ = cell_supersaturation(cell, c)
        Δm = sgs_condensation_growth!(sd, fl, S̄, cell.temperature, p, h;
                                      indices = idx, substeps = substeps,
                                      mc = mc, c = c)
        q_liq = total_water_mass(sd, mc; cell = index) / cell_air_mass(cell)
        Δq, ΔT = apply_condensate_tendency!(cell, Δm, q_liq, c)
        Δm_tot += Δm; Δq_tot += Δq; ΔT_tot += ΔT
    end
    return (Δm_tot, Δq_tot, ΔT_tot)
end
