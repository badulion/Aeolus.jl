# Phase 43, issue #5 — prognostic ice-nucleating-particle species.
#
# Units are SI throughout, matching `AerosolCloud.ARGMode`: number in m⁻³,
# mass in kg m⁻³, radius in m, surface area in m² m⁻³.

# =========================================================================
# 1. Log-normal surface-area moments
# =========================================================================
#
# `AerosolCloud` already provides the 0th and 3rd moments above a cut-off
# (`number_above_radius`, `mass_above_radius`). Surface-site-density
# (INAS) freezing needs the 2nd. The algebra is the same: the r²-weighted
# distribution of a log-normal is log-normal with the same σ_g and a
# median shifted to r_g exp(2 ln²σ_g).

"""
    DEMOTT_CUTOFF_RADIUS = 0.25e-6

Dry radius (m) of the DeMott et al. (2010, 2015) aerosol cut-off: their
INP parameterisations are written in terms of `N_{>0.5 μm}`, the number
of particles with *diameter* above 0.5 μm, i.e. radius above 0.25 μm.
"""
const DEMOTT_CUTOFF_RADIUS = 0.25e-6

"""
    mode_surface_area(m::ARGMode) -> m² m⁻³

Total particle surface area of a log-normal mode,

```math
A = 4π N r_g^2 \\exp(2\\ln^2σ_g).
```

This is the quantity immersion-freezing INAS parameterisations
(Niemand et al. 2012; Ullrich et al. 2017) multiply by the ice-nucleation
active site density `n_s(T)`.
"""
mode_surface_area(m::ARGMode) = 4π * m.N * m.r_g^2 * exp(2 * log(m.σ_g)^2)
mode_surface_area(m::LogNormalMode) = mode_surface_area(arg_mode(m))

"""
    surface_area_above_radius(m::ARGMode, r_cut) -> m² m⁻³

Surface area carried by particles of dry radius above `r_cut`,

```math
A(>r) = \\frac{A}{2}\\,\\mathrm{erfc}\\!\\left(
        \\frac{\\ln(r/r_g)}{\\sqrt2 \\ln σ_g} - \\frac{2\\lnσ_g}{\\sqrt2}\\right),
```

the second-moment companion of `AerosolCloud.number_above_radius`
(shift 0) and `AerosolCloud.mass_above_radius` (shift `3lnσ_g/√2`). It is
exact for the log-normal.
"""
function surface_area_above_radius(m::ARGMode, r_cut::Real)
    A = mode_surface_area(m)
    r_cut <= 0 && return A
    lnσ = log(m.σ_g)
    u = log(float(r_cut) / m.r_g) / (sqrt(2.0) * lnσ) - 2 * lnσ / sqrt(2.0)
    return A * 0.5 * erfc_series(u)
end
surface_area_above_radius(m::LogNormalMode, r_cut::Real) =
    surface_area_above_radius(arg_mode(m), r_cut)

# =========================================================================
# 2. Quadrature over a log-normal for size-dependent per-particle rates
# =========================================================================
#
# Most sinks act particle by particle with a size-dependent probability
# p(r): settling out of a layer (1 - exp(-v(r)Δt/Δz)), scavenging, and
# stochastic freezing (1 - exp(-J(T) 4πr² Δt)). The removed number and
# mass are
#
#     ΔN = N ∫ p(r_g e^{lnσ z})              φ(z) dz
#     ΔM = M ∫ p(r_g e^{lnσ (z + 3 lnσ)})    φ(z) dz
#
# because the mass distribution is log-normal with median r_g e^{3ln²σ}.
# Writing each moment in its own standard-normal variable makes p ≡ c
# integrate to exactly c for every moment, so a size-independent sink
# conserves r_g to rounding. The trapezoid rule on a Gaussian is
# spectrally accurate for smooth p; p may be steep (freezing
# probabilities switch on over a few kelvin, i.e. a factor of a few in
# r), so the node spacing is fine (h = 0.025σ) and the range ±10σ.

const _QZ_H = 0.025
const _QZ = collect(-10.0:_QZ_H:10.0)
const _QW = let w = [_QZ_H * exp(-0.5 * z^2) / sqrt(2π) for z in _QZ]
    w ./ sum(w)          # normalise so ∫φ = 1 to rounding
end

# ∫ p(r) weighted by the k-th moment of the log-normal (k = 0 number,
# 2 surface, 3 mass), as a fraction of that moment.
function _moment_fraction(p, r_g::Real, σ_g::Real, k::Integer)
    lnσ = log(σ_g)
    shift = k * lnσ
    s = 0.0
    @inbounds for j in eachindex(_QZ)
        pj = Float64(p(r_g * exp(lnσ * (_QZ[j] + shift))))
        s += _QW[j] * clamp(pj, 0.0, 1.0)
    end
    return s
end

# =========================================================================
# 3. The prognostic INP population
# =========================================================================

"""
    INPPopulation{T}

Prognostic ice-nucleating particles in one air parcel or grid cell,
organised as log-normal modes (the Phase 37/42 modal representation)
with two moments per population:

  * `species`         — the `AerosolSpecies` of each mode (`:mineral_dust`,
                        `:biological`, …); freezing parameterisations
                        dispatch on it
  * `σ_g`, `ρ_p`, `κ` — fixed shape parameter, dry density (kg m⁻³) and
                        hygroscopicity of each mode
  * `N_avail`, `M_avail` — number (m⁻³) and dry mass (kg m⁻³) of INP that
                        have **not** nucleated ice: interstitial or
                        immersed in droplets, still able to freeze
  * `N_cons`, `M_cons`   — INP **consumed** by nucleation: now inside ice
                        crystals, unable to nucleate again until released

plus cumulative budget counters (`N_src`, `N_nuc`, `N_rel`, `N_wet`,
`N_sed`, `N_cons_lost`, their mass twins, and the initial `N_init`,
`M_init`) that make conservation checkable — see [`inp_budget`](@ref).

**Why available/consumed and not just a number.** Singular (time-
independent) freezing parameterisations — DeMott et al. (2010, 2015),
Niemand et al. (2012), Tobo et al. (2013) — give the number of INP that
*have* frozen by the time the parcel reaches temperature `T`. Applied to
a diagnostic INP field every time step, they re-freeze the same
particles and produce a steady ice source where the real atmosphere
produces a one-off burst. Carrying the consumed population turns the
parameterisation into the correct *increment*,
`ΔN = max(0, N_frozen(T) - N_cons)`, so holding `T` fixed nucleates
nothing after the first step. That is the mechanism by which INP
depletion sustains mixed-phase Arctic stratus (Solomon et al. 2015).

**Modal closure.** Each population's median radius is diagnosed from
its number and mass with `σ_g` held fixed, exactly as in
`AerosolCloud.AerosolReservoir` (Whitby & McMurry 1997). A
size-selective sink (sedimentation of the coarse tail, nucleation on the
largest particles first) removes more mass per particle than average,
so the diagnosed median radius of what remains shrinks.

`AerosolReservoir` itself is not reused as the container because its
two populations mean *interstitial* versus *activated as CCN*; the INP
split is orthogonal (an available INP may be interstitial or immersed),
and the INP budget needs the species identity that the reservoir drops.
"""
mutable struct INPPopulation{T<:AbstractFloat}
    species::Vector{AerosolSpecies}
    σ_g::Vector{T}
    ρ_p::Vector{T}
    κ::Vector{T}
    N_avail::Vector{T}
    M_avail::Vector{T}
    N_cons::Vector{T}
    M_cons::Vector{T}
    # cumulative budget
    N_init::Vector{T}
    M_init::Vector{T}
    N_src::Vector{T}
    M_src::Vector{T}
    N_nuc::Vector{T}
    M_nuc::Vector{T}
    N_rel::Vector{T}
    M_rel::Vector{T}
    N_wet::Vector{T}
    M_wet::Vector{T}
    N_sed::Vector{T}
    M_sed::Vector{T}
    N_cons_lost::Vector{T}
    M_cons_lost::Vector{T}
end

"""
    inp_population(modes::AbstractVector{<:LogNormalMode}) -> INPPopulation
    inp_population(modes::AbstractVector{<:ARGMode}, species) -> INPPopulation

Initialise an INP population from Phase 37 optical modes (which carry
their species; units are converted with `AerosolCloud.arg_mode`) or from
SI activation modes plus a species per mode (an `AerosolSpecies`, a
symbol, or a vector of either). Everything starts available.

```julia
dust = lognormal_mode(:mineral_dust, 1.0, 0.4, 2.0)    # 1 cm⁻³, r_g 0.4 μm
bio  = lognormal_mode(:biological, 0.01, 1.0, 1.6)
inp  = inp_population([dust, bio])
```
"""
function inp_population(modes::AbstractVector{<:ARGMode}, species)
    n = length(modes)
    sp = species isa AbstractVector ? species : fill(species, n)
    length(sp) == n ||
        error("inp_population: need one species per mode ($n), got $(length(sp))")
    sps = AerosolSpecies[s isa AerosolSpecies ? s : aerosol_species(Symbol(s))
                         for s in sp]
    z() = zeros(Float64, n)
    inp = INPPopulation{Float64}(sps, z(), z(), z(), z(), z(), z(), z(),
                                 z(), z(), z(), z(), z(), z(), z(), z(),
                                 z(), z(), z(), z(), z(), z())
    for (i, m) in enumerate(modes)
        inp.σ_g[i] = m.σ_g
        inp.ρ_p[i] = m.ρ_p
        inp.κ[i] = m.κ
        inp.N_avail[i] = m.N
        inp.M_avail[i] = mode_dry_mass(m)
        inp.N_init[i] = inp.N_avail[i]
        inp.M_init[i] = inp.M_avail[i]
    end
    return inp
end

function inp_population(modes::AbstractVector{<:LogNormalMode})
    return inp_population([arg_mode(m) for m in modes],
                          [m.species for m in modes])
end

"Number of modes in an INP population."
inp_nmodes(inp::INPPopulation) = length(inp.N_avail)
Base.length(inp::INPPopulation) = inp_nmodes(inp)

"Species of mode `i` (an `AerosolSpecies`)."
inp_species(inp::INPPopulation, i::Integer) = inp.species[i]

Base.show(io::IO, inp::INPPopulation) = print(io,
    "INPPopulation(", inp_nmodes(inp), " modes ",
    [s.name for s in inp.species], ", N_avail=",
    round.(inp.N_avail .* 1e-3; sigdigits = 3), " L⁻¹, N_cons=",
    round.(inp.N_cons .* 1e-3; sigdigits = 3), " L⁻¹)")

# --- population selection ------------------------------------------------

function _moments(inp::INPPopulation, i::Integer, population::Symbol)
    if population === :available
        return inp.N_avail[i], inp.M_avail[i]
    elseif population === :consumed
        return inp.N_cons[i], inp.M_cons[i]
    elseif population === :total
        return inp.N_avail[i] + inp.N_cons[i], inp.M_avail[i] + inp.M_cons[i]
    else
        error("population must be :available, :consumed or :total, got :$population")
    end
end

_modes_range(inp::INPPopulation, mode::Integer) =
    mode == 0 ? (1:inp_nmodes(inp)) : (mode:mode)

"""
    inp_median_radius(inp, i; population = :available) -> m

Median dry radius of mode `i`, diagnosed from its number and mass with
`σ_g` fixed (the modal closure, identical to
`AerosolCloud.reservoir_dry_radius`). Zero for an empty population.
"""
function inp_median_radius(inp::INPPopulation, i::Integer;
                           population::Symbol = :available)
    N, M = _moments(inp, i, population)
    return AerosolCloud._rg_from_moments(N, M, inp.ρ_p[i], inp.σ_g[i])
end

"""
    inp_modes(inp; population = :available) -> Vector{ARGMode}

The chosen population as `ARGMode`s with re-diagnosed median radii, so
every Phase 42 modal diagnostic (`number_above_radius`,
`mass_above_radius`, `arg_activation`, …) applies to the INP directly.
Empty modes get a placeholder radius of 1 nm and zero number.
"""
function inp_modes(inp::INPPopulation; population::Symbol = :available)
    return [ARGMode{Float64}(first(_moments(inp, i, population)),
                             max(inp_median_radius(inp, i; population), 1e-9),
                             inp.σ_g[i], inp.κ[i], inp.ρ_p[i])
            for i in 1:inp_nmodes(inp)]
end

_mode(inp::INPPopulation, i::Integer, population::Symbol) =
    ARGMode{Float64}(first(_moments(inp, i, population)),
                     max(inp_median_radius(inp, i; population), 1e-9),
                     inp.σ_g[i], inp.κ[i], inp.ρ_p[i])

# =========================================================================
# 4. Diagnostics
# =========================================================================

"""
    inp_number(inp; mode = 0, population = :available) -> m⁻³

INP number concentration of one mode (`mode = i`) or summed over all
(`mode = 0`). `population` is `:available`, `:consumed` or `:total`.
"""
inp_number(inp::INPPopulation; mode::Integer = 0,
           population::Symbol = :available) =
    sum((first(_moments(inp, i, population)) for i in _modes_range(inp, mode));
        init = 0.0)

"""
    inp_mass(inp; mode = 0, population = :available) -> kg m⁻³

INP dry mass concentration, per mode or summed.
"""
inp_mass(inp::INPPopulation; mode::Integer = 0,
         population::Symbol = :available) =
    sum((last(_moments(inp, i, population)) for i in _modes_range(inp, mode));
        init = 0.0)

"""
    inp_surface_area(inp; mode = 0, population = :available) -> m² m⁻³

Total INP surface area from the log-normal second moment
([`mode_surface_area`](@ref)), per mode or summed. This is the `A` in
INAS immersion freezing, `N_ice = N (1 - exp(-n_s(T) A/N))`.
"""
inp_surface_area(inp::INPPopulation; mode::Integer = 0,
                 population::Symbol = :available) =
    sum((mode_surface_area(_mode(inp, i, population))
         for i in _modes_range(inp, mode)); init = 0.0)

"""
    inp_number_above_radius(inp, r_cut; mode = 0, population = :available) -> m⁻³

Number of INP with dry radius above `r_cut` (m), analytic for each
log-normal mode (`AerosolCloud.number_above_radius`). With
`r_cut = DEMOTT_CUTOFF_RADIUS` this is the `N_{>0.5 μm}` of DeMott
et al. (2010, 2015).
"""
inp_number_above_radius(inp::INPPopulation, r_cut::Real; mode::Integer = 0,
                        population::Symbol = :available) =
    sum((number_above_radius(_mode(inp, i, population), r_cut)
         for i in _modes_range(inp, mode)); init = 0.0)

"""
    inp_surface_area_above_radius(inp, r_cut; mode = 0, population = :available) -> m² m⁻³

Surface area of INP above dry radius `r_cut`, analytic
([`surface_area_above_radius`](@ref)).
"""
inp_surface_area_above_radius(inp::INPPopulation, r_cut::Real;
                              mode::Integer = 0,
                              population::Symbol = :available) =
    sum((surface_area_above_radius(_mode(inp, i, population), r_cut)
         for i in _modes_range(inp, mode)); init = 0.0)

"""
    inp_budget(inp; mode = 0) -> NamedTuple

Cumulative number and mass budget of one mode or of all modes:

  * `N_total`, `M_total` — available + consumed now
  * `N_init`, `M_init`   — at construction
  * `N_src`, `M_src`     — net sources (emission, entrainment)
  * `N_nuc`, `M_nuc`     — moved available → consumed by nucleation
  * `N_rel`, `M_rel`     — moved consumed → available by ice loss
  * `N_wet`, `M_wet`     — removed by scavenging
  * `N_sed`, `M_sed`     — removed by sedimentation / precipitation
  * `N_cons_lost`, `M_cons_lost` — the part of `wet + sed` that came from
    the consumed population (INP leaving inside falling ice)

The two identities that must hold to rounding are

```math
N_\\mathrm{avail} + N_\\mathrm{cons} = N_\\mathrm{init} + N_\\mathrm{src}
    - N_\\mathrm{wet} - N_\\mathrm{sed}
```

```math
N_\\mathrm{cons} = N_\\mathrm{cons,0} + N_\\mathrm{nuc} - N_\\mathrm{rel}
    - N_\\mathrm{cons,lost}
```

and the same for mass; [`inp_budget_residual`](@ref) evaluates them.
"""
function inp_budget(inp::INPPopulation; mode::Integer = 0)
    rg = _modes_range(inp, mode)
    s(v) = sum(@view v[rg]; init = 0.0)
    return (N_avail = s(inp.N_avail), M_avail = s(inp.M_avail),
            N_cons = s(inp.N_cons), M_cons = s(inp.M_cons),
            N_total = s(inp.N_avail) + s(inp.N_cons),
            M_total = s(inp.M_avail) + s(inp.M_cons),
            N_init = s(inp.N_init), M_init = s(inp.M_init),
            N_src = s(inp.N_src), M_src = s(inp.M_src),
            N_nuc = s(inp.N_nuc), M_nuc = s(inp.M_nuc),
            N_rel = s(inp.N_rel), M_rel = s(inp.M_rel),
            N_wet = s(inp.N_wet), M_wet = s(inp.M_wet),
            N_sed = s(inp.N_sed), M_sed = s(inp.M_sed),
            N_cons_lost = s(inp.N_cons_lost),
            M_cons_lost = s(inp.M_cons_lost))
end

"""
    inp_budget_residual(inp; mode = 0) -> (N_total, M_total, N_cons, M_cons)

Relative residuals of the two budget identities in [`inp_budget`](@ref)
(total and consumed; number and mass), each normalised by the larger of
the initial and current totals. All four are rounding-level (≲ 1e-14)
for any sequence of the source/sink operators in this file; anything
larger is a bookkeeping bug. The consumed identity assumes the
population started with nothing consumed, which is how
[`inp_population`](@ref) builds it.
"""
function inp_budget_residual(inp::INPPopulation; mode::Integer = 0)
    b = inp_budget(inp; mode)
    scaleN = max(b.N_init + abs(b.N_src), b.N_total, floatmin())
    scaleM = max(b.M_init + abs(b.M_src), b.M_total, floatmin())
    rN = (b.N_total - (b.N_init + b.N_src - b.N_wet - b.N_sed)) / scaleN
    rM = (b.M_total - (b.M_init + b.M_src - b.M_wet - b.M_sed)) / scaleM
    cN = (b.N_cons - (b.N_nuc - b.N_rel - b.N_cons_lost)) / scaleN
    cM = (b.M_cons - (b.M_nuc - b.M_rel - b.M_cons_lost)) / scaleM
    return (N_total = rN, M_total = rM, N_cons = cN, M_cons = cM)
end

# =========================================================================
# 5. Sources: emission and entrainment
# =========================================================================

"""
    inp_emit!(inp, i, ΔN, ΔM) -> ΔN
    inp_emit!(inp, i, source::ARGMode, Δt) -> ΔN
    inp_emit!(inp, i, source::LogNormalMode, Δt) -> ΔN

Source hook: add number `ΔN` (m⁻³) and dry mass `ΔM` (kg m⁻³) to the
*available* population of mode `i`. The `ARGMode` form treats
`source.N` as a number emission rate (m⁻³ s⁻¹) with the size
distribution of `source`, so `ΔN = N Δt` and `ΔM = mode_dry_mass(source) Δt`;
the `LogNormalMode` form converts from cm⁻³ s⁻¹ and μm first.

Only the two moments are added. Under the fixed-`σ_g` modal closure a
source whose median radius differs from the resident population's
shifts the diagnosed median between the two — which is the intended
behaviour of a two-moment scheme; a source with a very different `σ_g`
belongs in a separate mode.

Negative increments are rejected; use a sink operator to remove INP.
"""
function inp_emit!(inp::INPPopulation, i::Integer, ΔN::Real, ΔM::Real)
    (ΔN >= 0 && ΔM >= 0) || error("inp_emit!: increments must be non-negative")
    (ΔN > 0) == (ΔM > 0) ||
        error("inp_emit!: number and mass must both be zero or both positive")
    inp.N_avail[i] += ΔN; inp.M_avail[i] += ΔM
    inp.N_src[i] += ΔN; inp.M_src[i] += ΔM
    return float(ΔN)
end

function inp_emit!(inp::INPPopulation, i::Integer, source::ARGMode, Δt::Real)
    Δt >= 0 || error("inp_emit!: Δt must be non-negative")
    return inp_emit!(inp, i, source.N * Δt, mode_dry_mass(source) * Δt)
end
inp_emit!(inp::INPPopulation, i::Integer, source::LogNormalMode, Δt::Real) =
    inp_emit!(inp, i, arg_mode(source), Δt)

"""
    inp_entrain!(inp, N_env, M_env, f) -> ΔN

Entrainment hook for parcel and single-column models: mix a fraction
`f ∈ [0, 1]` of environmental air, carrying available INP number
`N_env[i]` (m⁻³) and mass `M_env[i]` (kg m⁻³) per mode, into the
parcel. The available population relaxes linearly,
`X ← (1-f) X + f X_env`.

Consumed INP are *not* diluted here: they are inside ice crystals, and
the host's ice category (which is transported and entrained with its
own tendencies) owns them — remove them with the ice via
[`inp_scavenge!`](@ref) or [`inp_sediment!`](@ref) with
`population = :consumed`, or return them with [`inp_release!`](@ref).

In an LES this is resolved transport and the hook is not used. The net
change (which is negative when the environment is cleaner) is booked as
a source, so the budget identities still close.
"""
function inp_entrain!(inp::INPPopulation, N_env::AbstractVector,
                      M_env::AbstractVector, f::Real)
    0 <= f <= 1 || error("inp_entrain!: f must be in [0, 1]")
    n = inp_nmodes(inp)
    (length(N_env) == n && length(M_env) == n) ||
        error("inp_entrain!: need one environmental value per mode ($n)")
    total = 0.0
    for i in 1:n
        (N_env[i] >= 0 && M_env[i] >= 0) ||
            error("inp_entrain!: environmental INP must be non-negative")
        ΔN = f * (N_env[i] - inp.N_avail[i])
        ΔM = f * (M_env[i] - inp.M_avail[i])
        inp.N_avail[i] += ΔN; inp.M_avail[i] += ΔM
        inp.N_src[i] += ΔN; inp.M_src[i] += ΔM
        total += ΔN
    end
    return total
end

inp_entrain!(inp::INPPopulation, env::INPPopulation, f::Real) =
    inp_entrain!(inp, env.N_avail, env.M_avail, f)

# =========================================================================
# 6. Nucleation sinks: available → consumed
# =========================================================================

# Move (ΔN, ΔM) from available to consumed in mode i, clipped so neither
# population goes negative, and keeping a non-empty population's number
# and mass jointly positive (all-or-nothing when the number is exhausted).
function _transfer_to_consumed!(inp::INPPopulation, i::Integer,
                                ΔN::Real, ΔM::Real)
    ΔN = clamp(float(ΔN), 0.0, inp.N_avail[i])
    ΔM = clamp(float(ΔM), 0.0, inp.M_avail[i])
    if ΔN >= inp.N_avail[i]
        ΔN = inp.N_avail[i]; ΔM = inp.M_avail[i]
    elseif ΔN == 0
        ΔM = 0.0
    elseif ΔM >= inp.M_avail[i]
        # Cannot take all the mass but leave particles behind: fall back to
        # an average-mass transfer for the remainder.
        ΔM = inp.M_avail[i] * ΔN / inp.N_avail[i]
    end
    inp.N_avail[i] -= ΔN; inp.M_avail[i] -= ΔM
    inp.N_cons[i] += ΔN; inp.M_cons[i] += ΔM
    inp.N_nuc[i] += ΔN; inp.M_nuc[i] += ΔM
    return ΔN
end

# Radius above which a log-normal holds number `target` (< N): invert
# N(>r) = N/2 erfc(u) by bisection in u (erfc is monotone).
function _radius_holding(m::ARGMode, target::Real)
    frac = target / m.N
    frac >= 1 && return 0.0
    lo, hi = -10.0, 10.0
    for _ in 1:200
        mid = 0.5 * (lo + hi)
        0.5 * erfc_series(mid) > frac ? (lo = mid) : (hi = mid)
        hi - lo < 1e-13 && break
    end
    u = 0.5 * (lo + hi)
    return m.r_g * exp(u * sqrt(2.0) * log(m.σ_g))
end

"""
    inp_consume!(inp, i, ΔN; selection = :largest) -> ΔN_moved
    inp_consume!(inp, i, p::Function) -> ΔN_moved

Nucleation sink: move INP from the available to the consumed population
of mode `i`. Consumed INP cannot nucleate again, so repeated calls can
never freeze more particles than were ever available.

*Number form* — a freezing scheme has computed how many INP nucleated
(`ΔN`, m⁻³). Which particles is set by `selection`:

  * `:largest` — the coarsest particles go first: `ΔN` is converted to
    the cut-off radius that holds exactly that number
    (`AerosolCloud.number_above_radius` inverted) and the mass above
    that radius (`AerosolCloud.mass_above_radius`) moves with it. The
    appropriate choice for surface- or size-threshold schemes (INAS,
    DeMott `N_{>0.5μm}`), where nucleation ability rises steeply with
    size.
  * `:random`  — size-independent: mass moves at the population mean,
    `ΔM = M ΔN/N`, leaving the median radius unchanged.

*Probability form* — `p(r)` is the probability that a particle of dry
radius `r` (m) nucleates during the step, e.g. stochastic immersion
freezing `p(r) = 1 - exp(-J_het(T) 4πr² Δt)`. Number and mass moved are
the log-normal integrals of `p` over the available population,
evaluated in each moment's own standard-normal variable so that a
constant `p ≡ c` moves exactly `c N` and `c M`.

`ΔN` is clipped to the available number; the return value is what was
actually moved.
"""
function inp_consume!(inp::INPPopulation, i::Integer, ΔN::Real;
                      selection::Symbol = :largest)
    ΔN >= 0 || error("inp_consume!: ΔN must be non-negative")
    N = inp.N_avail[i]
    (N > 0 && ΔN > 0) || return 0.0
    ΔN >= N && return _transfer_to_consumed!(inp, i, N, inp.M_avail[i])
    if selection === :largest
        m = _mode(inp, i, :available)
        r_cut = _radius_holding(m, ΔN)
        ΔM = mass_above_radius(m, r_cut)
    elseif selection === :random
        ΔM = inp.M_avail[i] * ΔN / N
    else
        error("inp_consume!: selection must be :largest or :random")
    end
    return _transfer_to_consumed!(inp, i, ΔN, ΔM)
end

function inp_consume!(inp::INPPopulation, i::Integer, p::Function)
    N = inp.N_avail[i]
    N > 0 || return 0.0
    r_g = inp_median_radius(inp, i)
    fN = _moment_fraction(p, r_g, inp.σ_g[i], 0)
    fM = _moment_fraction(p, r_g, inp.σ_g[i], 3)
    return _transfer_to_consumed!(inp, i, fN * N, fM * inp.M_avail[i])
end

"""
    inp_frozen_fraction(inp, i, p; population = :total) -> (fN, fM)

Number and mass fractions of mode `i` for which the per-particle
probability `p(r)` holds, without changing the population. With the
default `:total` population this is what a singular parameterisation
says *has* frozen by now, regardless of when.
"""
function inp_frozen_fraction(inp::INPPopulation, i::Integer, p::Function;
                             population::Symbol = :total)
    N, _ = _moments(inp, i, population)
    N > 0 || return (0.0, 0.0)
    r_g = inp_median_radius(inp, i; population)
    return (_moment_fraction(p, r_g, inp.σ_g[i], 0),
            _moment_fraction(p, r_g, inp.σ_g[i], 3))
end

"""
    inp_nucleate_singular!(inp, i, N_frozen) -> ΔN
    inp_nucleate_singular!(inp, i, p::Function) -> ΔN

Apply a *singular* (time-independent, deterministic) freezing
parameterisation without double counting.

Singular schemes — DeMott et al. (2010, 2015), Niemand et al. (2012),
Ullrich et al. (2017), Tobo et al. (2013) — state how many of the
parcel's INP are frozen at the current temperature: `N_frozen(T)`, or a
per-particle frozen indicator `p(r) = 1 - exp(-n_s(T) 4πr²)`. The
population that statement applies to is the **total** (available +
consumed), because particles that froze earlier would freeze again at
this temperature. The nucleation increment is therefore

```math
ΔN = \\min\\!\\left[N_\\mathrm{avail},\\ \\max(0,\\ N_\\mathrm{frozen} - N_\\mathrm{cons})\\right]
```

and likewise for mass with the `p`-weighted mass fraction. At constant
temperature a second call returns exactly zero; cooling further
nucleates only the difference; warming nucleates nothing (it does not
un-freeze anything — melting is the host's business, see
[`inp_release!`](@ref)).

The number form uses `:largest` selection for the mass, matching the
monotone size dependence of every singular scheme above.
"""
function inp_nucleate_singular!(inp::INPPopulation, i::Integer, N_frozen::Real)
    N_frozen >= 0 || error("inp_nucleate_singular!: N_frozen must be non-negative")
    ΔN = min(inp.N_avail[i], max(0.0, N_frozen - inp.N_cons[i]))
    ΔN > 0 || return 0.0
    return inp_consume!(inp, i, ΔN; selection = :largest)
end

function inp_nucleate_singular!(inp::INPPopulation, i::Integer, p::Function)
    Ntot, Mtot = _moments(inp, i, :total)
    Ntot > 0 || return 0.0
    fN, fM = inp_frozen_fraction(inp, i, p; population = :total)
    ΔN = min(inp.N_avail[i], max(0.0, fN * Ntot - inp.N_cons[i]))
    ΔN > 0 || return 0.0
    ΔM = min(inp.M_avail[i], max(0.0, fM * Mtot - inp.M_cons[i]))
    return _transfer_to_consumed!(inp, i, ΔN, ΔM)
end

"""
    inp_release!(inp, f; mode = 0) -> ΔN

Return a fraction `f ∈ [0, 1]` of the consumed INP to the available
population — the host calls this when that fraction of the ice that
carries them sublimates or melts completely without precipitating.

Whether a released INP should be able to nucleate again is a physical
question (pre-activation: particles that have held ice often nucleate
at warmer temperatures the second time; Kanji et al. 2017, §5). This
operator makes them available again; with a singular scheme applied
through [`inp_nucleate_singular!`](@ref) they will re-freeze once the
parcel is again as cold as when they first froze — which is the ice
recycling Solomon et al. (2015) found necessary to sustain observed
Arctic ice number.
"""
function inp_release!(inp::INPPopulation, f::Real; mode::Integer = 0)
    0 <= f <= 1 || error("inp_release!: f must be in [0, 1]")
    total = 0.0
    for i in _modes_range(inp, mode)
        ΔN = f * inp.N_cons[i]; ΔM = f * inp.M_cons[i]
        inp.N_cons[i] -= ΔN; inp.M_cons[i] -= ΔM
        inp.N_avail[i] += ΔN; inp.M_avail[i] += ΔM
        inp.N_rel[i] += ΔN; inp.M_rel[i] += ΔM
        total += ΔN
    end
    return total
end

# =========================================================================
# 7. Removal sinks: scavenging and sedimentation
# =========================================================================

# Remove per-particle probability p(r) from one population of mode i,
# booking the loss to (Nctr, Mctr). Returns the number removed.
function _remove!(inp::INPPopulation, i::Integer, p, population::Symbol,
                  Nctr::Vector, Mctr::Vector)
    if population === :available
        N, M = inp.N_avail[i], inp.M_avail[i]
    elseif population === :consumed
        N, M = inp.N_cons[i], inp.M_cons[i]
    else
        error("population must be :available or :consumed")
    end
    N > 0 || return 0.0
    r_g = AerosolCloud._rg_from_moments(N, M, inp.ρ_p[i], inp.σ_g[i])
    if p isa Real
        fN = fM = clamp(Float64(p), 0.0, 1.0)
    else
        fN = _moment_fraction(p, r_g, inp.σ_g[i], 0)
        fM = _moment_fraction(p, r_g, inp.σ_g[i], 3)
    end
    ΔN = fN * N
    ΔM = fM * M
    if fN >= 1 || ΔN >= N
        ΔN, ΔM = N, M
    end
    if population === :available
        inp.N_avail[i] -= ΔN; inp.M_avail[i] -= ΔM
    else
        inp.N_cons[i] -= ΔN; inp.M_cons[i] -= ΔM
        inp.N_cons_lost[i] += ΔN; inp.M_cons_lost[i] += ΔM
    end
    Nctr[i] += ΔN; Mctr[i] += ΔM
    return ΔN
end

function _populations(population::Symbol)
    population === :both && return (:available, :consumed)
    population in (:available, :consumed) && return (population,)
    error("population must be :available, :consumed or :both")
end

"""
    inp_scavenge!(inp, Λ, Δt; mode = 0, population = :available) -> ΔN

Wet scavenging with scavenging coefficient `Λ` (s⁻¹) over a step `Δt`:
each particle survives with probability `exp(-Λ Δt)` — the exact
solution of `dN/dt = -Λ N`, so the operator is unconditionally positive
for any `Δt`.

`Λ` may be a number (size-independent: number and mass decay together
and the median radius is unchanged) or a function `Λ(r)` of dry radius
in m (e.g. below-cloud impaction scavenging, which has the
Greenfield-gap minimum near 0.1–1 μm; Seinfeld & Pandis 2016, ch. 20).
`population = :consumed` removes INP that are inside ice with a
precipitation removal rate; `:both` applies the same `Λ` to both.
"""
function inp_scavenge!(inp::INPPopulation, Λ, Δt::Real; mode::Integer = 0,
                       population::Symbol = :available)
    Δt >= 0 || error("inp_scavenge!: Δt must be non-negative")
    p = Λ isa Real ? (Λ >= 0 ? 1 - exp(-Λ * Δt) :
                      error("inp_scavenge!: Λ must be non-negative")) :
                     (r -> 1 - exp(-max(Λ(r), 0.0) * Δt))
    total = 0.0
    for pop in _populations(population), i in _modes_range(inp, mode)
        total += _remove!(inp, i, p, pop, inp.N_wet, inp.M_wet)
    end
    return total
end

"""
    inp_settling_velocity(inp, i; moment = 0, population = :available,
                          micro = DEFAULT_MICRO) -> m s⁻¹

Moment-weighted gravitational settling speed of mode `i`:
`moment = 0` gives the number-weighted mean `⟨v⟩_N` (what sets the
number flux), `moment = 3` the mass-weighted `⟨v⟩_M`. The single-
particle speed is `Superdroplet.stokes_settling_velocity` (Stokes drag
with Cunningham slip) at the mode's dry density.

Without slip `v = k r²`, and the log-normal moments give the closed forms

```math
⟨v⟩_N = k r_g^2 e^{2\\ln^2σ_g}, \\qquad ⟨v⟩_M = k r_g^2 e^{8\\ln^2σ_g},
```

so for `σ_g = 2` mass settles `e^{6 ln²2} = 18` times faster than
number — the reason dust plumes coarsen at their base and fine-mode
dust travels intercontinental distances.
"""
function inp_settling_velocity(inp::INPPopulation, i::Integer; moment::Integer = 0,
                               population::Symbol = :available,
                               micro::MicrophysicalConstants = DEFAULT_MICRO)
    N, M = _moments(inp, i, population)
    N > 0 || return 0.0
    r_g = inp_median_radius(inp, i; population)
    lnσ = log(inp.σ_g[i])
    ρ = inp.ρ_p[i]
    s = 0.0
    @inbounds for j in eachindex(_QZ)
        s += _QW[j] * stokes_settling_velocity(r_g * exp(lnσ * (_QZ[j] + moment * lnσ)),
                                               ρ, micro)
    end
    return s
end

"""
    inp_sediment!(inp, Δz, Δt; mode = 0, micro = DEFAULT_MICRO) -> ΔN

Gravitational sedimentation of *available* INP out of a well-mixed layer
of depth `Δz` (m) over `Δt` (s), with no inflow from above (a transport-
free sink; in a column model the host passes the outflow down as the
next layer's `inp_emit!` source). A particle of radius `r` survives with
probability `exp(-v(r)Δt/Δz)` — exact for a well-mixed box and positive
for any `Δt` — so the coarse tail is lost first, number and mass are
removed with different effective speeds, and the diagnosed median radius
of what remains shrinks.

Consumed INP fall with their ice crystal, not at the dry-particle speed;
remove them with `inp_scavenge!(…; population = :consumed)` using the
host's ice sedimentation rate.
"""
function inp_sediment!(inp::INPPopulation, Δz::Real, Δt::Real;
                       mode::Integer = 0,
                       micro::MicrophysicalConstants = DEFAULT_MICRO)
    Δz > 0 || error("inp_sediment!: Δz must be positive")
    Δt >= 0 || error("inp_sediment!: Δt must be non-negative")
    total = 0.0
    for i in _modes_range(inp, mode)
        ρ = inp.ρ_p[i]
        p = r -> 1 - exp(-stokes_settling_velocity(r, ρ, micro) * Δt / Δz)
        total += _remove!(inp, i, p, :available, inp.N_sed, inp.M_sed)
    end
    return total
end
