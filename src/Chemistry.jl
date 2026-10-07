module Chemistry

# Era I chemistry.
#
#   1. `apply_linear_chemistry!` — a single tracer with first-order decay
#      and a source (Phase 20).
#   2. A reduced urban NOx–O3–VOC mechanism in mixing ratios (ppb), with
#      rate constants from the JPL / IUPAC evaluations and the MCM
#      clear-sky NO2 photolysis frequency. The NO–NO2–O3 triad is
#      integrated analytically, so NOx and Ox are conserved exactly by it.
#   3. Dry deposition by the resistance model v_d = 1/(r_a + r_b + r_c),
#      with r_a supplied by the caller (from SurfaceExchange) and r_b, r_c
#      from here.
#   4. Below-cloud wet deposition by rain.
#   5. The conversion of an emission mass flux to a mixing-ratio flux.
#
# Advection is not done here: the caller moves the tracers (in the Berlin
# case the :transport stage does it).
#
# The Berlin example (examples/urban/couplings/aq_chem.jl) used to carry
# its own copy of 2–4, as a toy in ppb-scaled units. That copy is gone and
# the example calls this module (#135).

using ..Grids: RegularGrid, interior, CellCenter
using ..Fields: Field, interior_view
using ..Backends: Backends, plane_reduce!, plane_map!, ordered_sum, column_sum

export apply_linear_chemistry!,
       air_number_density, ppb_rate_constant, emission_ppb_flux,
       k_NO_O3, k_NO2_OH, k_VOC_O3, k_VOC_OH, j_NO2,
       nox_triad, photostationary_NO2, leighton_ratio,
       photochemistry_cell, apply_photochemistry!,
       schmidt_number, quasi_laminar_resistance, surface_resistance,
       dry_deposition_velocity, apply_dry_deposition!,
       rain_scavenging_coefficient, apply_wet_deposition!

# =========================================================================
# 1. Linear chemistry (Phase 20)
# =========================================================================

"""
    apply_linear_chemistry!(tc, c, τ; source = nothing)

Add first-order chemical relaxation `-c/τ` (plus an optional source
`S`) to the tracer tendency `tc`. If `source` is a scalar, it's applied
uniformly; if it's a 3D array (interior-sized), the per-cell values are
added.

Positive `τ` implies decay (photolysis, dry deposition); pass `τ = Inf`
for an inert tracer with only the source term.
"""
function apply_linear_chemistry!(tc::Field{CellCenter},
                                 c::Field{CellCenter},
                                 τ::Real;
                                 source = nothing)
    invτ = τ == Inf ? zero(eltype(tc.data)) : 1 / eltype(tc.data)(τ)
    ii, jj, kk = interior(tc.mesh)
    tcd = tc.data; cd = c.data
    if source === nothing
        @inbounds for k in kk, j in jj, i in ii
            tcd[i,j,k] -= invτ * cd[i,j,k]
        end
    elseif source isa Real
        s = eltype(tcd)(source)
        @inbounds for k in kk, j in jj, i in ii
            tcd[i,j,k] += s - invτ * cd[i,j,k]
        end
    elseif source isa AbstractArray{<:Real,3}
        size(source) == (length(ii), length(jj), length(kk)) ||
            error("apply_linear_chemistry!: source array size mismatch")
        @inbounds for (kk_, kv) in enumerate(kk),
                       (jj_, jv) in enumerate(jj),
                       (ii_, iv) in enumerate(ii)
            tcd[iv,jv,kv] += source[ii_,jj_,kk_] - invτ * cd[iv,jv,kv]
        end
    else
        error("apply_linear_chemistry!: source must be Nothing, Real, or 3-D Array")
    end
    return tc
end

# =========================================================================
# 2. Units
# =========================================================================

const _K_BOLTZMANN = 1.380649e-23      # J K⁻¹ (SI, exact)
const _R_GAS       = 8.314462618       # J mol⁻¹ K⁻¹ (SI, exact)

"""
    air_number_density(p, T) -> molecules cm⁻³

Number density of air, ``M = p/(k_B T)``, in molecules cm⁻³ (`p` in Pa,
`T` in K). At 101 325 Pa and 298.15 K it is 2.46 × 10¹⁹ cm⁻³.
"""
air_number_density(p::Real, T::Real) = 1e-6 * p / (_K_BOLTZMANN * T)

"""
    ppb_rate_constant(k, p, T) -> ppb⁻¹ s⁻¹

Convert a bimolecular rate constant `k` (cm³ molecule⁻¹ s⁻¹) to mixing
ratios: ``k_{ppb} = k M × 10^{-9}``, so that
``d[\\mathrm{A}]/dt = -k_{ppb} [\\mathrm{A}][\\mathrm{B}]`` with both in ppb.
"""
ppb_rate_constant(k::Real, p::Real, T::Real) = k * air_number_density(p, T) * 1e-9

"""
    emission_ppb_flux(F, M_molar; p = 101325, T = 293.15) -> ppb m s⁻¹

Convert a surface emission mass flux `F` (kg m⁻² s⁻¹) of a gas with
molar mass `M_molar` (kg mol⁻¹) to a mixing-ratio flux: the molar flux
divided by the molar density of air ``p/(RT)``, times 10⁹. Spread over a
layer of depth ``Δz`` for a time ``Δt``, it raises the mixing ratio by
``F_{ppb} Δt/Δz`` ppb.
"""
emission_ppb_flux(F::Real, M_molar::Real; p::Real = 101325.0, T::Real = 293.15) =
    1e9 * (F / M_molar) / (p / (_R_GAS * T))

# =========================================================================
# 3. Rate constants and photolysis
# =========================================================================

"""
    k_NO_O3(T) -> cm³ molecule⁻¹ s⁻¹

NO + O3 → NO2 + O2: ``k = 3.0×10^{-12} \\exp(-1500/T)``
(Burkholder et al. 2019, JPL Publication 19-5). 1.9 × 10⁻¹⁴ at 298 K.
"""
k_NO_O3(T::Real) = 3.0e-12 * exp(-1500 / T)

"""
    k_NO2_OH(T, M) -> cm³ molecule⁻¹ s⁻¹

NO2 + OH (+M) → HNO3, the termolecular JPL form
``k = \\frac{k_0 M}{1 + k_0 M/k_∞} F_c^{\\{1 + [\\log_{10}(k_0 M/k_∞)]^2\\}^{-1}}``
with ``k_0 = 1.8×10^{-30} (T/300)^{-3}``, ``k_∞ = 2.8×10^{-11}``,
``F_c = 0.6`` (Burkholder et al. 2019). `M` is the air number density
(cm⁻³, [`air_number_density`](@ref)). About 1.05 × 10⁻¹¹ at the surface.
"""
function k_NO2_OH(T::Real, M::Real)
    k0M = 1.8e-30 * (T / 300)^(-3.0) * M
    kinf = 2.8e-11
    x = k0M / kinf
    return k0M / (1 + x) * 0.6^(1 / (1 + log10(x)^2))
end

"""
    k_VOC_O3(T) -> cm³ molecule⁻¹ s⁻¹

Ozonolysis of the lumped reactive VOC, taken as isoprene:
``k = 1.03×10^{-14} \\exp(-1995/T)`` (IUPAC; Atkinson et al. 2006).
1.3 × 10⁻¹⁷ at 298 K.
"""
k_VOC_O3(T::Real) = 1.03e-14 * exp(-1995 / T)

"""
    k_VOC_OH(T) -> cm³ molecule⁻¹ s⁻¹

OH oxidation of the lumped reactive VOC, taken as isoprene:
``k = 2.7×10^{-11} \\exp(390/T)`` (IUPAC; Atkinson et al. 2006).
1.0 × 10⁻¹⁰ at 298 K.
"""
k_VOC_OH(T::Real) = 2.7e-11 * exp(390 / T)

"""
    j_NO2(μ₀) -> s⁻¹

Clear-sky NO2 photolysis frequency (NO2 + hν → NO + O(³P), the O(³P)
forming O3 at once) as a function of the cosine of the solar zenith
angle, in the MCM form ``J = l\\,μ_0^m \\exp(-n/μ_0)`` with
``l = 1.165×10^{-2}``, ``m = 0.244``, ``n = 0.267`` (MCM v3, J4; Saunders
et al. 2003). Zero for ``μ_0 ≤ 0``; 8.9 × 10⁻³ s⁻¹ with the sun overhead.
It ignores clouds, aerosol and shading, so it is an upper bound in a
street canyon.
"""
j_NO2(μ₀::Real) = μ₀ > 0 ? 1.165e-2 * μ₀^0.244 * exp(-0.267 / μ₀) : zero(float(μ₀))

# =========================================================================
# 4. The NO–NO2–O3 triad (analytic)
# =========================================================================

"""
    photostationary_NO2(NOx, Ox, J, k) -> ppb

The photostationary NO2 for total ``NO_x = NO + NO_2`` and
``O_x = O_3 + NO_2`` (ppb): the smaller root of
``k (NO_x - y)(O_x - y) = J y``, the state in which
``J [NO_2] = k [NO][O_3]``. `k` in ppb⁻¹ s⁻¹ ([`ppb_rate_constant`](@ref)).
At night (``J = 0``) it is ``\\min(NO_x, O_x)``: all the NO titrated,
or all the O3.
"""
function photostationary_NO2(N::Real, X::Real, J::Real, k::Real)
    k > 0 || return zero(float(N))
    B = k * (N + X) + J
    D = sqrt(max(B^2 - 4 * k^2 * N * X, zero(B)))
    s = B + D
    return s > 0 ? min(2 * k * N * X / s, min(N, X)) : zero(s)
end

"""
    leighton_ratio(NO, NO2, O3, J, k) -> φ

The Leighton ratio ``φ = J[NO_2]/(k[NO][O_3])``: 1 in the photostationary
state of the triad alone, above 1 where peroxy radicals also convert NO to
NO2, below 1 in fresh NO emissions that have not yet reached the
photostationary state.
"""
leighton_ratio(NO::Real, NO2::Real, O3::Real, J::Real, k::Real) = J * NO2 / (k * NO * O3)

"""
    nox_triad(NO, NO2, O3, dt, J, k) -> (NO, NO2, O3)

Exact solution over `dt` of

    NO2 + hν → NO + O3     (J,  s⁻¹)
    NO + O3  → NO2 + O2    (k,  ppb⁻¹ s⁻¹)

With ``NO_x = NO + NO_2`` and ``O_x = O_3 + NO_2`` invariant, ``y = [NO_2]``
obeys the Riccati equation ``\\dot y = k (y - y_1)(y - y_2)`` with roots
``y_1 ≤ \\min(NO_x, O_x) ≤ y_2``, whose solution is

``y(t) = y_1 + \\frac{(y_0 - y_1) E}{1 - (y_0 - y_1)(1 - E)/r}``,
``r = y_2 - y_1``, ``E = e^{-k r t}``

(the ``r → 0`` limit, night-time titration with ``NO_x = O_x``, is taken
through `expm1`). It is unconditionally stable, positive, relaxes to the
photostationary state [`photostationary_NO2`](@ref) on the time scale
``1/(k r)``, and conserves ``NO_x`` and ``O_x`` to round-off for any `dt`.
"""
function nox_triad(NO::Real, NO2::Real, O3::Real, dt::Real, J::Real, k::Real)
    N = NO + NO2
    X = O3 + NO2
    y0 = NO2
    if k <= 0
        y = y0 * exp(-J * dt)
    else
        B = k * (N + X) + J
        D = sqrt(max(B^2 - 4 * k^2 * N * X, zero(B)))
        s = B + D
        s > 0 || return (NO, NO2, O3)              # nothing to react
        y1 = 2 * k * N * X / s                      # stable root, no cancellation
        a = y0 - y1
        if D > 0
            E = exp(-D * dt)                        # k r = D
            φ = -expm1(-D * dt) * k / D             # (1 - E)/r
        else
            E = one(D)
            φ = k * dt
        end
        y = y1 + a * E / (1 - a * φ)
    end
    y = clamp(y, zero(y), min(N, X))
    return (N - y, y, X - y)
end

"""
    photochemistry_cell(NO, NO2, O3, VOC, HNO3, dt; J, k1, k3, L4, L5, ν = 2)
        -> (NO, NO2, O3, VOC, HNO3, P_Ox, L_Ox)

Advance one cell of the reduced urban mechanism by `dt` (mixing ratios in
ppb):

    R1  NO2 + hν  → NO + O3               J   (s⁻¹)
    R2  NO + O3   → NO2 + O2              k1  (ppb⁻¹ s⁻¹)
    R3  VOC + O3  → products              k3  (ppb⁻¹ s⁻¹)
    R4  VOC + OH  → ν (NO → NO2) + …      L4 = k_VOC_OH [OH]  (s⁻¹)
    R5  NO2 + OH  → HNO3                  L5 = k_NO2_OH [OH]  (s⁻¹)

OH is prescribed (the mechanism has no HOx budget). In R4 the peroxy
radical RO2 and the HO2 it yields each convert one NO to NO2, so each VOC
oxidised converts ``ν = 2`` NO (limited by the NO available): this is the
ozone production of the mechanism.

Operator splitting (first order): R3–R5 as pseudo-first-order losses over
`dt` with the partners frozen (exact exponentials), then the triad R1–R2
exactly by [`nox_triad`](@ref). All species stay ≥ 0. Returns the new
state and the cell's Ox production ``P_{Ox}`` (R4) and loss ``L_{Ox}``
(R3 + R5), in ppb, so that
``Δ(O_3 + NO_2) = P_{Ox} - L_{Ox}`` and ``Δ(NO + NO_2) = -ΔHNO_3`` exactly.
"""
function photochemistry_cell(NO::Real, NO2::Real, O3::Real, VOC::Real, HNO3::Real, dt::Real;
                             J::Real, k1::Real, k3::Real = 0.0, L4::Real = 0.0,
                             L5::Real = 0.0, ν::Real = 2.0)
    # R3 + R4: VOC loss, split between ozonolysis and OH
    λ = k3 * O3 + L4
    dV = λ > 0 ? -VOC * expm1(-λ * dt) : zero(VOC)
    dO3 = λ > 0 ? min(O3, dV * k3 * O3 / λ) : zero(O3)
    conv = λ > 0 ? min(NO, ν * dV * L4 / λ) : zero(NO)
    VOC -= dV
    O3 -= dO3
    NO -= conv
    NO2 += conv
    # R5: NO2 + OH → HNO3
    dN = L5 > 0 ? -NO2 * expm1(-L5 * dt) : zero(NO2)
    NO2 -= dN
    HNO3 += dN
    # R1 + R2 exactly
    NO, NO2, O3 = nox_triad(NO, NO2, O3, dt, J, k1)
    return (NO, NO2, O3, max(VOC, zero(VOC)), HNO3, conv, dO3 + dN)
end

_atk(x::Real, k) = x
_atk(x::AbstractVector, k) = x[k]

"""
    apply_photochemistry!(NO, NO2, O3, VOC, HNO3, dt; μ₀ | J, T, p, OH = 0, ν = 2)
        -> (P_Ox, L_Ox, HNO3_formed)

Advance 3-D arrays of mixing ratios (ppb, e.g. `interior_view`s) by `dt`
with [`photochemistry_cell`](@ref). The photolysis frequency is `J`
(s⁻¹) or, if `J` is not given, [`j_NO2`](@ref)`(μ₀)`. Temperature `T` (K),
pressure `p` (Pa) and the prescribed OH number density `OH` (cm⁻³) are
scalars or per-level vectors (length `size(NO, 3)`); the rate constants
are evaluated once per level.

Returns the domain sums (ppb × cells) of Ox production, Ox loss and HNO3
formed, for budget checks.
"""
function apply_photochemistry!(NO::AbstractArray{<:Real,3}, NO2::AbstractArray{<:Real,3},
                               O3::AbstractArray{<:Real,3}, VOC::AbstractArray{<:Real,3},
                               HNO3::AbstractArray{<:Real,3}, dt::Real;
                               μ₀::Real = 0.0, J::Real = j_NO2(μ₀),
                               T = 293.15, p = 101325.0, OH = 0.0, ν::Real = 2.0)
    size(NO) == size(NO2) == size(O3) == size(VOC) == size(HNO3) ||
        throw(DimensionMismatch("apply_photochemistry!: species arrays differ in size"))
    Nx, Ny, Nz = size(NO)
    # the rate constants once per level
    k1 = zeros(Nz); k3 = zeros(Nz); L4 = zeros(Nz); L5 = zeros(Nz)
    for k in 1:Nz
        Tk, pk, OHk = _atk(T, k), _atk(p, k), _atk(OH, k)
        M = air_number_density(pk, Tk)
        k1[k] = k_NO_O3(Tk) * M * 1e-9
        k3[k] = k_VOC_O3(Tk) * M * 1e-9
        L4[k] = k_VOC_OH(Tk) * OHk
        L5[k] = k_NO2_OH(Tk, M) * OHk
    end
    # the cell update fused with the sums of what it produced, in the
    # canonical (thread-invariant) plane order (KA-b, #296)
    P = zeros(Nz); L = zeros(Nz); H = zeros(Nz)
    plane_reduce!((P, L, H), Backends.backend(NO), _photochemistry_at,
                  (1:Nx, 1:Ny, 1:Nz), NO, NO2, O3, VOC, HNO3, dt, J, k1, k3, L4, L5, ν)
    return (P_Ox = ordered_sum(P), L_Ox = ordered_sum(L), HNO3_formed = ordered_sum(H))
end

@inline function _photochemistry_at(i, j, k, NO, NO2, O3, VOC, HNO3, dt, J, k1, k3, L4, L5, ν)
    @inbounds begin
        h0 = HNO3[i, j, k]
        a, b, c, d, e, pr, ls = photochemistry_cell(NO[i, j, k], NO2[i, j, k], O3[i, j, k],
                                                    VOC[i, j, k], h0, dt;
                                                    J = J, k1 = k1[k], k3 = k3[k], L4 = L4[k],
                                                    L5 = L5[k], ν = ν)
        NO[i, j, k] = a; NO2[i, j, k] = b; O3[i, j, k] = c
        VOC[i, j, k] = d; HNO3[i, j, k] = e
        return (Float64(pr), Float64(ls), Float64(e - h0))
    end
end

# =========================================================================
# 5. Dry deposition (resistance model)
# =========================================================================

# Molecular diffusivity of water vapour over that of the species,
# D_H2O/D_x (Wesely 1989, Table 2).
const _DH2O_RATIO = Dict{Symbol,Float64}(:NO => 1.3, :NO2 => 1.6, :O3 => 1.6, :HNO3 => 1.9)

"""
    schmidt_number(species; T = 293.15, p = 101325) -> Sc

Schmidt number ``Sc = ν/D`` of a gas in air. ``ν`` is the kinematic
viscosity of air (Sutherland's law for μ over the ideal-gas density);
``D = D_{H_2O}/(D_{H_2O}/D_x)``, with the water-vapour diffusivity
``D_{H_2O} = 2.178×10^{-5} (T/273.15)^{1.81} (p_0/p)`` m² s⁻¹ (Massman
1998) and the ratio ``D_{H_2O}/D_x`` of Wesely (1989, Table 2) for
`:NO` 1.3, `:NO2` 1.6, `:O3` 1.6, `:HNO3` 1.9. O3 at 20 °C: 1.0.
"""
function schmidt_number(species::Symbol; T::Real = 293.15, p::Real = 101325.0)
    haskey(_DH2O_RATIO, species) || throw(ArgumentError("schmidt_number: unknown species :$species"))
    μ = 1.458e-6 * T^1.5 / (T + 110.4)             # Sutherland, Pa s
    ν = μ / (p / (287.05 * T))
    D = 2.178e-5 * (T / 273.15)^1.81 * (101325.0 / p) / _DH2O_RATIO[species]
    return ν / D
end

"""
    quasi_laminar_resistance(u_star; Sc, Pr = 0.72, κ = 0.4) -> s m⁻¹

Quasi-laminar (sub-layer) resistance to gas transfer over a surface,
``r_b = \\frac{2}{κ u_*}(Sc/Pr)^{2/3}`` (Wesely & Hicks 1977; Seinfeld &
Pandis 2016, §19.3). `u_star` is floored at 0.01 m s⁻¹.
"""
quasi_laminar_resistance(u_star::Real; Sc::Real, Pr::Real = 0.72, κ::Real = 0.4) =
    2 / (κ * max(u_star, 0.01)) * (Sc / Pr)^(2 / 3)

# Surface (canopy) resistance r_c, s m⁻¹, by species and surface kind.
# The NO/NO2/O3 values are the reciprocals of the total deposition
# velocities the Berlin example used to carry (0.2–7 mm s⁻¹), now read as
# the surface-limited conductance; they are of the order of Wesely's (1989)
# midsummer values (O3 over grass ≈ 100–200 s m⁻¹, over water ≈ 2000),
# not a fit. HNO3 sticks to any surface: Wesely's floor of 10 s m⁻¹.
const _R_C = Dict{Symbol,NamedTuple}(
    :O3   => (vegetation = 1 / 0.007, water = 1 / 0.001, paved = 1 / 0.0005, glass = 1 / 0.0002),
    :NO2  => (vegetation = 1 / 0.002, water = 1 / 0.001, paved = 1 / 0.002,  glass = 1 / 0.001),
    :NO   => (vegetation = 1 / 0.002, water = 1 / 0.001, paved = 1 / 0.002,  glass = 1 / 0.001),
    :HNO3 => (vegetation = 10.0,      water = 10.0,      paved = 10.0,       glass = 10.0),
    :VOC  => (vegetation = Inf,       water = Inf,       paved = Inf,        glass = Inf),
)

"""
    surface_resistance(species, surface) -> s m⁻¹

Surface (canopy) resistance ``r_c`` of `species` (`:NO`, `:NO2`, `:O3`,
`:HNO3`, `:VOC`) over a surface of kind `:vegetation`, `:water`, `:paved`
(streets, roofs, walls) or `:glass`.

| species | vegetation | water | paved | glass |
|:------- | ---------: | ----: | ----: | ----: |
| O3      | 143        | 1000  | 2000  | 5000  |
| NO, NO2 | 500        | 1000  | 500   | 1000  |
| HNO3    | 10         | 10    | 10    | 10    |
| VOC     | ∞          | ∞     | ∞     | ∞     |

The O3 and NOx values are an order-of-magnitude set in the range of
Wesely's (1989) midsummer resistances, not a fit to measurements; HNO3
uses Wesely's floor of 10 s m⁻¹; the lumped VOC (isoprene) does not
deposit.
"""
function surface_resistance(species::Symbol, surface::Symbol)
    haskey(_R_C, species) || throw(ArgumentError("surface_resistance: unknown species :$species"))
    t = _R_C[species]
    hasproperty(t, surface) || throw(ArgumentError("surface_resistance: unknown surface :$surface (use :vegetation, :water, :paved, :glass)"))
    return getproperty(t, surface)
end

"""
    dry_deposition_velocity(r_a, r_b, r_c) -> m s⁻¹

``v_d = 1/(r_a + r_b + r_c)``: aerodynamic, quasi-laminar and surface
resistances in series (s m⁻¹). `r_a = 1/(C_H U)` is the scalar
aerodynamic resistance of the surface layer, e.g. from
`SurfaceExchange.solve_surface_layer`.
"""
dry_deposition_velocity(r_a::Real, r_b::Real, r_c::Real) = 1 / (r_a + r_b + r_c)

"""
    apply_dry_deposition!(c, v_d, dt; dz, k = 1) -> removed

Remove dry-deposited mass from the lowest air cell of every column of the
3-D array `c` (or `Field`): ``c ← c\\, e^{-v_d Δt/Δz}``, the exact
solution of ``dc/dt = -v_d c/Δz`` over the step (surface flux
``F = -v_d c``). `v_d` is a scalar or an `(Nx, Ny)` matrix (m s⁻¹); `k` is
the level of the first air cell, a scalar or an `(Nx, Ny)` integer
matrix (roof-aware). Returns ``Σ_{ij} Δc\\,Δz`` (units of `c` × m), the
column-integrated amount removed summed over columns.
"""
function apply_dry_deposition!(c::AbstractArray{<:Real,3}, v_d, dt::Real; dz::Real, k = 1,
                               backend::Backends.AbstractBackend = Backends.backend(c))
    Nx, Ny, _ = size(c)
    # one work item per column (KA-m, #301), each returning what it removed;
    # the total folded in column-major order, the old loop's (the same bits)
    return column_sum(backend, _dry_deposition_column!, (1:Nx, 1:Ny), c, v_d, dt, dz, k)
end

@inline function _dry_deposition_column!(i, j, c, v_d, dt, dz, k)
    @inbounds begin
        kk = k isa Integer ? k : k[i, j]
        1 <= kk <= size(c, 3) || return 0.0
        vd = v_d isa Real ? v_d : v_d[i, j]
        c0 = c[i, j, kk]
        c1 = c0 * exp(-vd * dt / dz)
        c[i, j, kk] = c1
        return (c0 - c1) * dz
    end
end
apply_dry_deposition!(c::Field, v_d, dt::Real; kw...) =
    apply_dry_deposition!(interior_view(c), v_d, dt; kw...)

# =========================================================================
# 6. Wet deposition (below-cloud scavenging by rain)
# =========================================================================

"""
    rain_scavenging_coefficient(species) -> (kg/kg)⁻¹ s⁻¹

Coefficient `s` of the below-cloud scavenging rate ``Λ = s\\,q_r`` (s⁻¹)
used by [`apply_wet_deposition!`](@ref), with ``q_r`` the rain mixing
ratio (kg kg⁻¹).

* `:HNO3`: 1.0, so that ``Λ ≈ 10^{-4}`` s⁻¹ at ``q_r = 10^{-4}`` (about
  2 mm h⁻¹ of rain) — the order of magnitude of below-cloud scavenging of
  a highly soluble gas (Seinfeld & Pandis 2016, ch. 20). Order of
  magnitude only, not a fit.
* `:NO`, `:NO2`, `:O3`, `:VOC`: 0. Their Henry's-law solubilities are so
  low that rain removes a negligible fraction.
"""
function rain_scavenging_coefficient(species::Symbol)
    species === :HNO3 && return 1.0
    species in (:NO, :NO2, :O3, :VOC) && return 0.0
    throw(ArgumentError("rain_scavenging_coefficient: unknown species :$species"))
end

"""
    apply_wet_deposition!(c, qr, dt; scav) -> removed

Below-cloud scavenging by rain, ``dc/dt = -Λ c`` with ``Λ = s\\,q_r``,
integrated exactly over `dt`: ``c ← c\\,e^{-s q_r Δt}``. `c` and `qr`
(rain mixing ratio, kg kg⁻¹) are 3-D arrays of the same size or `Field`s;
`scav` is `s` in (kg/kg)⁻¹ s⁻¹ ([`rain_scavenging_coefficient`](@ref)).
Cells without rain are untouched. Returns ``Σ Δc`` over all cells (units
of `c` × cells).
"""
function apply_wet_deposition!(c::AbstractArray{<:Real,3}, qr::AbstractArray{<:Real,3},
                               dt::Real; scav::Real)
    size(qr) == size(c) ||
        throw(DimensionMismatch("apply_wet_deposition!: qr size $(size(qr)) vs $(size(c))"))
    scav == 0 && return 0.0
    qr isa Array && (any(>(0), qr) || return 0.0)   # no rain: nothing to scavenge (exact)
    # the update fused with the sum of what it removed: one running sum per
    # plane in cell order, planes in parallel, then ascending k
    # (thread-invariant; KA-b, #296). Not the eight lanes of plane_reduce!:
    # with `exp` called only in the raining cells, the lanes cost 1.3x the
    # old loop at 1 thread, the cell-order plane sum does not (ka_port.md).
    parts = plane_map!(zeros(size(c, 3)), Backends.backend(c), _wet_deposition_plane, axes(c),
                       c, qr, scav, dt)
    return ordered_sum(parts)
end

# One plane k of `apply_wet_deposition!`, i fastest. Plain arrays and
# unit-range views of them (the `Field` interiors) are swept along
# contiguous rows of their parents with linear indices: a 3-index store
# into `c` makes the compiler reload the array's size at every cell
# (measured 1.5x slower). A full plane of a plain array is one contiguous
# run. The sweep is its own function so that the plane bookkeeping around
# it does not cost it registers.
const _LinearRows = Union{Array{<:Real,3},
                          SubArray{<:Real,3,<:Array{<:Real,3},<:NTuple{3,UnitRange{Int}}}}

# the parent of `A` and the parent's linear index of point (i, j, k) of `A`
@inline _lin(A::Array, i, j, k) = (A, i + size(A, 1) * ((j - 1) + size(A, 2) * (k - 1)))
@inline function _lin(A::SubArray, i, j, k)
    P = parent(A); pi, pj, pk = parentindices(A)
    return _lin(P, pi[i], pj[j], pk[k])
end

function _wet_deposition_plane(k, ir, jr, c::_LinearRows, qr::_LinearRows, scav, dt)
    s = 0.0
    if c isa Array && qr isa Array && ir == axes(c, 1) && jr == axes(c, 2)
        cp, ic = _lin(c, 1, 1, k); qp, iq = _lin(qr, 1, 1, k)
        return _wet_deposition_run!(s, cp, qp, ic, iq, length(ir) * length(jr), scav, dt)
    end
    i1 = first(ir)
    for j in jr
        cp, ic = _lin(c, i1, j, k); qp, iq = _lin(qr, i1, j, k)
        s = _wet_deposition_run!(s, cp, qp, ic, iq, length(ir), scav, dt)
    end
    return s
end
# n consecutive cells from parent index ic of `cp` (iq of `qp`), adding to s
@noinline function _wet_deposition_run!(s, cp, qp, ic, iq, n, scav, dt)
    ic == iq || return _wet_deposition_run2!(s, cp, qp, ic, iq, n, scav, dt)
    @inbounds for I in ic:ic+n-1           # the same layout: one index
        q = qp[I]
        q > 0 || continue
        c0 = cp[I]
        c1 = c0 * exp(-scav * q * dt)
        cp[I] = c1
        s += Float64(c0 - c1)
    end
    return s
end
@noinline function _wet_deposition_run2!(s, cp, qp, ic, iq, n, scav, dt)
    @inbounds for r in 0:n-1
        q = qp[iq + r]
        q > 0 || continue
        c0 = cp[ic + r]
        c1 = c0 * exp(-scav * q * dt)
        cp[ic + r] = c1
        s += Float64(c0 - c1)
    end
    return s
end
function _wet_deposition_plane(k, ir, jr, c, qr, scav, dt)
    s = 0.0
    @inbounds for j in jr, i in ir
        q = qr[i, j, k]
        q > 0 || continue
        c0 = c[i, j, k]
        c1 = c0 * exp(-scav * q * dt)
        c[i, j, k] = c1
        s += Float64(c0 - c1)
    end
    return s
end
apply_wet_deposition!(c::Field, qr, dt::Real; kw...) =
    apply_wet_deposition!(interior_view(c), qr isa Field ? interior_view(qr) : qr, dt; kw...)

end # module

# ---------------------------------------------------------------------------
# Two additional Phase-20 modules live in their own files: Particles, ML,
# DataAssimilation. They share the pragmatic-scaffold spirit.
# ---------------------------------------------------------------------------
