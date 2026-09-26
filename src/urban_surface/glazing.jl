# Glazing: angle-dependent shortwave optics of uncoated glass panes and
# multi-pane units, the gas-gap heat transfer between panes, and the
# window facet energy balance. Included into `module UrbanSurface`.
#
# Physics owned elsewhere and delegated to:
#   * the Fresnel equations      — SurfaceOptics.fresnel_polarized_reflectance
#   * glass thermal properties   — the :glass row of the thermal material table
#   * σ, still-air conductivity, gravity, the gas constant — SurfaceExchange /
#     MoistThermodynamics through `SurfaceConstants`
#   * the surface energy balance (Rn, H, LE, MOST) — the SurfaceExchange contract
#   * the Robin face elimination — `_boundary_coefficients` / `_face_temperature`
#     of the conduction core above

# =========================================================================
# 7. Glass panes: Fresnel + Beer–Lambert slab optics
# =========================================================================

"""
    CLEAR_FLOAT_REFRACTIVE_INDEX = 1.526

Broadband solar refractive index of soda-lime float glass (Duffie &
Beckman 2013, §5.1 and Example 5.1.1). The normal-incidence reflectance
of one air–glass interface is ``((n-1)/(n+1))^2 = 0.04336``.
"""
const CLEAR_FLOAT_REFRACTIVE_INDEX = 1.526

"""
    CLEAR_FLOAT_EXTINCTION = 32.0  # m⁻¹

Broadband solar extinction coefficient ``K`` of ordinary ("greenish
edge") float glass. Duffie & Beckman (2013, §5.2) give ``K`` from about
4 m⁻¹ for water-white (low-iron) glass to about 32 m⁻¹ for glass with a
greenish cast of edge — the ordinary clear float of building glazing.
It is taken from the book, not fitted: with it a 3 mm pane has
``T_{sol} = 0.833``, ``R_{sol} = 0.076`` against the 0.83 / 0.08 of the
Vitro Clear data sheet (ISO 9050), and inverting that data sheet row
with [`glass_pane_from_normal`](@ref) returns ``K = 31.6`` m⁻¹.
"""
const CLEAR_FLOAT_EXTINCTION = 32.0

"""
    UNCOATED_GLASS_EMISSIVITY = 0.837

Corrected (hemispherical) thermal emissivity of uncoated soda-lime glass,
the standard value of EN 673 / EN 12898 and ISO 15099 (Vitro's data
sheet quotes 0.84). It governs the pane–pane radiative exchange in a gap
and the long-wave balance of the outer face.

Note: the broadband emissivity of the `SurfaceOptics` `:glass` entry is
0.94, a value for a generic glass-clad surface in the urban radiation
library. A pane's emissivity is a property of its surface finish (a
low-E coating has 0.02–0.2), so it is a [`GlassPane`](@ref) field.
"""
const UNCOATED_GLASS_EMISSIVITY = 0.837

"""
    GlassPane(thickness; n = 1.526, K = 32.0, emissivity = 0.837,
              emissivity_back = emissivity, material = :glass)

One uncoated, homogeneous, specular glass pane of `thickness` metres:
real refractive index `n`, broadband solar extinction coefficient `K`
(m⁻¹), thermal emissivities of the outer (`emissivity`) and inner face,
and the [`ThermalMaterial`](@ref) (default the `:glass` row of the
table) whose ``ρ c d`` is the pane's heat capacity and whose ``k``
its conduction resistance.

Optics are those of [`pane_optics`](@ref). To build a pane from a data
sheet's normal-incidence solar transmittance and reflectance use
[`glass_pane_from_normal`](@ref).
"""
struct GlassPane
    thickness::Float64
    n::Float64
    K::Float64
    emissivity::Float64
    emissivity_back::Float64
    material::ThermalMaterial{Float64}
end

function GlassPane(thickness::Real; n::Real = CLEAR_FLOAT_REFRACTIVE_INDEX,
                   K::Real = CLEAR_FLOAT_EXTINCTION,
                   emissivity::Real = UNCOATED_GLASS_EMISSIVITY,
                   emissivity_back::Real = emissivity,
                   material::Union{Symbol, ThermalMaterial} = :glass)
    thickness > 0 || throw(ArgumentError("GlassPane: thickness must be positive"))
    n > 1 || throw(ArgumentError("GlassPane: n must exceed 1"))
    K >= 0 || throw(ArgumentError("GlassPane: K must be ≥ 0"))
    (0 < emissivity <= 1 && 0 < emissivity_back <= 1) ||
        throw(ArgumentError("GlassPane: emissivities must lie in (0, 1]"))
    m = material isa Symbol ? thermal_material(material) : material
    mf = ThermalMaterial{Float64}(m.name, m.conductivity, m.density, m.specific_heat,
                                  m.optical, m.source)
    return GlassPane(float(thickness), float(n), float(K), float(emissivity),
                     float(emissivity_back), mf)
end

"""
    glass_pane_from_normal(thickness, T_normal, R_normal; kwargs...) -> GlassPane

Invert a measured normal-incidence solar transmittance `T_normal` and
reflectance `R_normal` of an uncoated pane (a manufacturer's ISO 9050 /
EN 410 data sheet row) for its refractive index and extinction
coefficient, as LBNL WINDOW does for uncoated glass (Furler 1991;
WINDOW Technical Documentation 2018, Eqs. 7-26 to 7-29):

```math
β = T_0^2 - R_0^2 + 2R_0 + 1,\\qquad
ρ_0 = \\frac{β - \\sqrt{β^2 - 4(2-R_0)R_0}}{2(2-R_0)},
```
```math
n = \\frac{1+\\sqrt{ρ_0}}{1-\\sqrt{ρ_0}},\\qquad
τ = \\frac{R_0 - ρ_0}{ρ_0 T_0},\\qquad K = -\\ln(τ)/d,
```

with ``ρ_0`` the single-interface reflectivity and ``τ`` the internal
transmittance. The pane then reproduces `(T_normal, R_normal)` exactly
and its angular behaviour follows from first principles.
`kwargs` go to [`GlassPane`](@ref).
"""
function glass_pane_from_normal(thickness::Real, T0::Real, R0::Real; kwargs...)
    (0 < T0 < 1 && 0 < R0 < 1 && T0 + R0 <= 1) ||
        throw(ArgumentError("glass_pane_from_normal: need 0 < T, R and T + R ≤ 1"))
    β = T0^2 - R0^2 + 2R0 + 1
    ρ0 = (β - sqrt(β^2 - 4 * (2 - R0) * R0)) / (2 * (2 - R0))
    n = (1 + sqrt(ρ0)) / (1 - sqrt(ρ0))
    τ = (R0 - ρ0) / (ρ0 * T0)
    (0 < τ <= 1 + 1e-12) ||
        throw(ArgumentError("glass_pane_from_normal: (T, R) = ($T0, $R0) is not an " *
                            "uncoated absorbing slab (internal transmittance $τ)"))
    K = max(-log(min(τ, 1.0)) / thickness, 0.0)
    return GlassPane(thickness; n = n, K = K, kwargs...)
end

# Incoherent slab of interface reflectance r and internal transmittance τ,
# summing the multiple internal reflections (Duffie & Beckman Eq. 5.3.1-3):
#   T = (1-r)² τ / (1 - r² τ²),   R = r + r (1-r)² τ² / (1 - r² τ²),   A = 1 - R - T
@inline function _slab(r::Real, τ::Real)
    den = 1 - r^2 * τ^2
    T = (1 - r)^2 * τ / den
    R = r + r * (1 - r)^2 * τ^2 / den
    return (R, T, max(1 - R - T, 0.0))
end

# Per-polarisation (R, T, A) of a pane: ((Rs, Ts, As), (Rp, Tp, Ap)).
function _pane_sp(p::GlassPane, θ::Real)
    Rs, Rp, ct = fresnel_polarized_reflectance(θ, p.n)
    τ = ct > 0 ? exp(-p.K * p.thickness / ct) : 0.0
    return (_slab(Rs, τ), _slab(Rp, τ))
end

"""
    pane_optics(pane, θ) -> (R = …, T = …, A = …)

Solar reflectance, transmittance and absorptance of an uncoated pane for
a direct beam at incidence angle `θ` (radians, from the pane normal).
Each polarisation is treated separately — interface reflectance ``ρ``
from the Fresnel equations (`SurfaceOptics.fresnel_polarized_reflectance`),
internal transmittance ``τ = \\exp(-K d / \\cos θ_t)`` along the refracted
path — and the incoherent multiple reflections are summed,

```math
T = \\frac{(1-ρ)^2 τ}{1-ρ^2τ^2},\\quad
R = ρ + \\frac{ρ(1-ρ)^2τ^2}{1-ρ^2τ^2},\\quad A = 1 - R - T,
```

before averaging the two polarisations (unpolarised sunlight; Duffie &
Beckman 2013 §5.3; ISO 15099 §6). `R + T + A = 1` exactly.
"""
function pane_optics(p::GlassPane, θ::Real)
    s, pp = _pane_sp(p, θ)
    return (R = (s[1] + pp[1]) / 2, T = (s[2] + pp[2]) / 2, A = (s[3] + pp[3]) / 2)
end

# =========================================================================
# 8. Fill gases and gaps (ISO 15099)
# =========================================================================

"""
    FillGas(name, k, μ, cp, M)

A glazing fill gas with ISO 15099 (Annex B, Table B.1) linear fits in the
absolute temperature `T` (K): conductivity ``k = k_a + k_b T``
(W m⁻¹ K⁻¹), dynamic viscosity ``μ = μ_a + μ_b T`` (Pa s), specific
heat ``c_p = c_a + c_b T`` (J kg⁻¹ K⁻¹), each a 2-tuple, and molar
mass `M` (kg kmol⁻¹). The density is the ideal gas at the gap pressure.

Two are provided: [`GLAZING_AIR`](@ref) and [`GLAZING_ARGON`](@ref).
"""
struct FillGas
    name::Symbol
    k::NTuple{2, Float64}
    μ::NTuple{2, Float64}
    cp::NTuple{2, Float64}
    M::Float64
end

"""
Air, ISO 15099:2003 Table B.1. The conductivity fit gives 0.02406
W m⁻¹ K⁻¹ at 273.15 K, the still-air value `SurfaceConstants.k_air`
(0.024) of the Era V contract.
"""
const GLAZING_AIR = FillGas(:air, (2.873e-3, 7.760e-5), (3.723e-6, 4.940e-8),
                            (1002.7370, 1.2324e-2), 28.97)
"Argon, ISO 15099:2003 Table B.1."
const GLAZING_ARGON = FillGas(:argon, (2.285e-3, 5.149e-5), (3.379e-6, 6.451e-8),
                              (521.9285, 0.0), 39.948)

"Conductivity of a fill gas at `T` (K), W m⁻¹ K⁻¹ (ISO 15099 Table B.1 fit)."
gas_conductivity(g::FillGas, T::Real) = g.k[1] + g.k[2] * T
"Dynamic viscosity of a fill gas at `T` (K), Pa s (ISO 15099 Table B.1 fit)."
gas_viscosity(g::FillGas, T::Real) = g.μ[1] + g.μ[2] * T
"Specific heat of a fill gas at `T` (K), J kg⁻¹ K⁻¹ (ISO 15099 Table B.1 fit)."
gas_heat_capacity(g::FillGas, T::Real) = g.cp[1] + g.cp[2] * T

"""
    GlazingGap(width; gas = GLAZING_AIR, height = 1.0, pressure = 101325.0)

A sealed gas gap of `width` metres between two panes, filled with
`gas` ([`FillGas`](@ref)) at `pressure` (Pa); `height` (m) is the gap's
vertical extent, which enters the convection correlation through the
aspect ratio.
"""
struct GlazingGap
    width::Float64
    gas::FillGas
    height::Float64
    pressure::Float64
end
function GlazingGap(width::Real; gas::FillGas = GLAZING_AIR, height::Real = 1.0,
                    pressure::Real = 101325.0)
    (width > 0 && height > 0 && pressure > 0) ||
        throw(ArgumentError("GlazingGap: width, height and pressure must be positive"))
    return GlazingGap(float(width), gas, float(height), float(pressure))
end

"""
    gap_nusselt(Ra, aspect) -> Nu

Nusselt number of a vertical gas cavity (ISO 15099:2003 §5.3.3.3,
from Wright 1996), `aspect` = height / width:

```math
Nu_1 = \\begin{cases}
0.0673838\\,Ra^{1/3} & Ra > 5\\times10^4\\\\
0.028154\\,Ra^{0.4134} & 10^4 < Ra ≤ 5\\times10^4\\\\
1 + 1.7596678\\times10^{-10}\\,Ra^{2.2984755} & Ra ≤ 10^4
\\end{cases},\\qquad
Nu_2 = 0.242\\,(Ra/A)^{0.272},
```

``Nu = \\max(Nu_1, Nu_2)``. ``Nu → 1`` (pure conduction) as ``Ra → 0``.
"""
function gap_nusselt(Ra::Real, aspect::Real)
    Ra = max(Ra, 0.0)
    Nu1 = Ra > 5e4 ? 0.0673838 * Ra^(1 / 3) :
          Ra > 1e4 ? 0.028154 * Ra^0.4134 :
                     1 + 1.7596678e-10 * Ra^2.2984755
    Nu2 = 0.242 * (Ra / aspect)^0.272
    return max(Nu1, Nu2)
end

"""
    gap_conductance(gap, T1, T2, ε1, ε2; c = DEFAULT_SURFACE_CONSTANTS)
        -> (h_c = …, h_r = …)

Heat-transfer coefficients (W m⁻² K⁻¹) across a glazing gap between pane
faces at `T1`, `T2` (K) with emissivities `ε1`, `ε2`:

  * convective ``h_c = Nu\\,k/d`` with [`gap_nusselt`](@ref) at
    ``Ra = ρ^2 d^3 g β c_p |ΔT| / (μ k)``, ``β = 1/T_m``, gas properties
    at the mean temperature ``T_m`` and ``ρ = pM/(R T_m)``;
  * radiative, two infinite grey parallel plates,
    ``h_r = σ (T_1^2+T_2^2)(T_1+T_2) / (1/ε_1 + 1/ε_2 - 1)``, so that
    ``h_r (T_1 - T_2)`` is the exact net exchange.

σ, `g` and the gas constant come from the Era V contract `c`.
"""
function gap_conductance(gap::GlazingGap, T1::Real, T2::Real, ε1::Real, ε2::Real;
                         c::SurfaceConstants = DEFAULT_SURFACE_CONSTANTS)
    Tm = (T1 + T2) / 2
    gas = gap.gas
    k = gas_conductivity(gas, Tm)
    μ = gas_viscosity(gas, Tm)
    cp = gas_heat_capacity(gas, Tm)
    ρ = gap.pressure * (gas.M * 1e-3) / (c.thermo.gas_constant * Tm)
    d = gap.width
    Ra = ρ^2 * d^3 * SurfaceExchange.grav(c) * cp * abs(T1 - T2) / (Tm * μ * k)
    hc = gap_nusselt(Ra, gap.height / d) * k / d
    hr = c.σ_SB * (T1^2 + T2^2) * (T1 + T2) / (1 / ε1 + 1 / ε2 - 1)
    return (h_c = hc, h_r = hr)
end

# =========================================================================
# 9. Glazing units: multi-pane optics
# =========================================================================

"""
    Glazing(panes, gaps)

A glazing unit: `panes` (outside first) separated by `gaps`
(`length(gaps) == length(panes) - 1`). Convenience constructors:
[`single_glazing`](@ref), [`double_glazing`](@ref),
[`triple_glazing`](@ref).
"""
struct Glazing
    panes::Vector{GlassPane}
    gaps::Vector{GlazingGap}
    function Glazing(panes::AbstractVector{GlassPane}, gaps::AbstractVector{GlazingGap})
        isempty(panes) && throw(ArgumentError("Glazing: need at least one pane"))
        length(gaps) == length(panes) - 1 ||
            throw(ArgumentError("Glazing: need length(panes) - 1 gaps"))
        return new(collect(panes), collect(gaps))
    end
end
Glazing(pane::GlassPane) = Glazing([pane], GlazingGap[])

"Single glazing: one clear pane of `thickness` m (kwargs to [`GlassPane`](@ref))."
single_glazing(thickness::Real = 0.004; kwargs...) = Glazing(GlassPane(thickness; kwargs...))

"""
    double_glazing(; thickness = 0.004, gap = 0.016, gas = GLAZING_AIR, height = 1.0, kwargs...)

Two identical clear panes around one gas gap (the 4-16-4 unit by
default). `kwargs` go to [`GlassPane`](@ref).
"""
double_glazing(; thickness::Real = 0.004, gap::Real = 0.016, gas::FillGas = GLAZING_AIR,
               height::Real = 1.0, kwargs...) =
    Glazing([GlassPane(thickness; kwargs...), GlassPane(thickness; kwargs...)],
            [GlazingGap(gap; gas = gas, height = height)])

"Three identical clear panes, two gaps (4-12-4-12-4 by default); see [`double_glazing`](@ref)."
triple_glazing(; thickness::Real = 0.004, gap::Real = 0.012, gas::FillGas = GLAZING_AIR,
               height::Real = 1.0, kwargs...) =
    Glazing([GlassPane(thickness; kwargs...) for _ in 1:3],
            [GlazingGap(gap; gas = gas, height = height) for _ in 1:2])

Base.length(g::Glazing) = length(g.panes)

# Net-radiation solution of a stack of symmetric layers for one
# polarisation. D_k / U_k are the inward / outward fluxes in gap k
# (k = 0: outside, k = N: interior), unit incidence from outside, black
# interior:
#   D_0 = 1,  U_N = 0,
#   D_j = T_j D_{j-1} + R_j U_j,   U_{j-1} = R_j D_{j-1} + T_j U_j,
#   absorbed_j = A_j (D_{j-1} + U_j).
function _stack!(A::AbstractVector, layers)
    N = length(layers)
    # Grazing incidence: the outer interface reflects everything and the
    # inter-pane system is degenerate (0 = 0); the answer is exact.
    if layers[1][1] >= 1 - 1e-12
        A .= 0
        return (1.0, 0.0)
    end
    M = zeros(2N + 2, 2N + 2); b = zeros(2N + 2)
    iD(k) = k + 1; iU(k) = N + 2 + k
    M[1, iD(0)] = 1; b[1] = 1
    M[2, iU(N)] = 1
    for j in 1:N
        R, T, _ = layers[j]
        r = 2j + 1
        M[r, iD(j)] = 1; M[r, iD(j - 1)] = -T; M[r, iU(j)] = -R
        M[r + 1, iU(j - 1)] = 1; M[r + 1, iD(j - 1)] = -R; M[r + 1, iU(j)] = -T
    end
    x = M \ b
    for j in 1:N
        A[j] = layers[j][3] * (x[iD(j - 1)] + x[iU(j)])
    end
    return (x[iU(0)], x[iD(N)])
end

"""
    glazing_optics(g::Glazing, θ) -> (R = …, T = …, A = [A_1, …, A_N])

Solar reflectance, transmittance and per-pane absorptance of a glazing
unit for a direct beam at incidence angle `θ` (radians) from outside.
The inter-reflections between panes are solved exactly for each
polarisation (net-radiation method, the linear-system form of the
recursion of ISO 15099 §6 / WINDOW Eqs. 7-15 to 7-20) with the room
treated as black, then averaged. `R + T + ΣA = 1` to round-off.
"""
function glazing_optics(g::Glazing, θ::Real)
    N = length(g)
    sp = [_pane_sp(p, θ) for p in g.panes]
    As = zeros(N); Ap = zeros(N)
    Rs, Ts = _stack!(As, [x[1] for x in sp])
    Rp, Tp = _stack!(Ap, [x[2] for x in sp])
    return (R = (Rs + Rp) / 2, T = (Ts + Tp) / 2, A = (As .+ Ap) ./ 2)
end

"""
    glazing_optics_hemispherical(g; nquad = 48) -> (R = …, T = …, A = […])

Optics for isotropic (diffuse) irradiance, the cosine-weighted
hemispherical average ``P_{hem} = 2\\int_0^{π/2} P(θ)\\cos θ \\sin θ\\,dθ
= \\int_0^1 P(μ)\\,2μ\\,dμ`` (WINDOW Eq. 7-36), by `nquad`-point
Gauss–Legendre quadrature in ``μ = \\cos θ``.
"""
function glazing_optics_hemispherical(g::Glazing; nquad::Int = 48)
    x, w = _gauss_legendre(nquad)
    R = 0.0; T = 0.0; A = zeros(length(g))
    for i in eachindex(x)
        μ = (x[i] + 1) / 2
        wt = w[i] / 2 * 2μ
        o = glazing_optics(g, acos(μ))
        R += wt * o.R; T += wt * o.T; A .+= wt .* o.A
    end
    return (R = R, T = T, A = A)
end

# =========================================================================
# 10. The window thermal network: U-value and SHGC
# =========================================================================
#
# One node per pane, at its mid-plane, with heat capacity ρ c d. Between
# node j and the pane face lies the half-pane conductance K_j = 2k/d; a
# gap couples the faces of adjacent panes through h_c + h_r (evaluated at
# the node temperatures — the temperature drop across a pane is below
# 0.1 K). Absorbed shortwave is deposited at the nodes. The same network
# serves the steady U-value and SHGC and the transient window facet, so
# the facet's steady state reproduces the SHGC by construction.

_pane_K(p::GlassPane) = 2 * p.material.conductivity / p.thickness

# Series conductances between successive pane nodes.
function _gap_links!(Gl::AbstractVector, g::Glazing, Tn::AbstractVector,
                     c::SurfaceConstants)
    for j in eachindex(g.gaps)
        p1, p2 = g.panes[j], g.panes[j + 1]
        h = gap_conductance(g.gaps[j], Tn[j], Tn[j + 1], p1.emissivity_back, p2.emissivity; c = c)
        Gl[j] = 1 / (1 / _pane_K(p1) + 1 / (h.h_c + h.h_r) + 1 / _pane_K(p2))
    end
    return Gl
end

# Steady node temperatures for boundary conditions `outer`, `inner`
# (ColumnBoundary, fluxes into the glazing) and node sources S (W m⁻²).
# Gap conductances are iterated to consistency.
function _steady_network(g::Glazing, outer::ColumnBoundary, inner::ColumnBoundary,
                         S::AbstractVector; c::SurfaceConstants = DEFAULT_SURFACE_CONSTANTS,
                         Tn = nothing, links = nothing, iters::Int = 50)
    N = length(g)
    go, so = _boundary_coefficients(outer, _pane_K(g.panes[1]))
    gi, si = _boundary_coefficients(inner, _pane_K(g.panes[N]))
    T = Tn === nothing ? fill((outer.T_env + inner.T_env) / 2, N) : collect(float.(Tn))
    Gl = zeros(max(N - 1, 0))
    a, b, cc, d = zeros(N), zeros(N), zeros(N), zeros(N)
    for it in 1:iters
        links === nothing ? _gap_links!(Gl, g, T, c) : (Gl .= links)
        for j in 1:N
            a[j] = j > 1 ? -Gl[j - 1] : 0.0
            cc[j] = j < N ? -Gl[j] : 0.0
            b[j] = (j > 1 ? Gl[j - 1] : 0.0) + (j < N ? Gl[j] : 0.0)
            d[j] = S[j]
        end
        b[1] += go; d[1] += go * outer.T_env + so
        b[N] += gi; d[N] += gi * inner.T_env + si
        Tprev = copy(T)
        _thomas!(T, a, b, cc, d)
        (links !== nothing || maximum(abs.(T .- Tprev)) < 1e-10) && break
    end
    F_in = gi * (inner.T_env - T[N]) + si        # into the glazing at the room face
    return (T = T, links = copy(_gap_links!(Gl, g, T, c)), q_room = -F_in)
end

"""
    glazing_u_value(g; T_out = 273.15, T_in = 293.15, h_out = 25.0, h_in = 7.7,
                    c = DEFAULT_SURFACE_CONSTANTS) -> W m⁻² K⁻¹

Centre-of-glass thermal transmittance of a glazing unit between outdoor
and indoor environments at `T_out`, `T_in` through combined (convective
+ radiative) surface coefficients `h_out`, `h_in`, with no sun: the
steady heat flow ``q/(T_{in} - T_{out})`` of the pane network, gap
coefficients ([`gap_conductance`](@ref)) iterated to the resulting pane
temperatures. Defaults are the EN 673 conventions (``h_e = 25``,
``h_i = 7.7`` W m⁻² K⁻¹).
"""
function glazing_u_value(g::Glazing; T_out::Real = 273.15, T_in::Real = 293.15,
                         h_out::Real = 25.0, h_in::Real = 7.7,
                         c::SurfaceConstants = DEFAULT_SURFACE_CONSTANTS)
    T_in != T_out || throw(ArgumentError("glazing_u_value: need T_in ≠ T_out"))
    sol = _steady_network(g, convective_boundary(T_out, h_out), convective_boundary(T_in, h_in),
                          zeros(length(g)); c = c)
    return -sol.q_room / (T_in - T_out)
end

"""
    inward_flowing_fractions(g; h_out = 23.0, h_in = 7.7, T_out = 273.15,
                             T_in = 293.15, c) -> Vector

Fraction ``N_j`` of the shortwave absorbed in pane `j` that reaches the
room in steady state: the unit-source response of the pane network with
both environments held fixed. Gap coefficients are frozen at the
no-sun steady state between `T_out` and `T_in`, which linearises the
problem as EN 410 does with its gap conductance Λ. Defaults are the
EN 410 surface coefficients (``h_e = 23``, ``h_i = 7.7``). For a single
pane ``N = h_i' / (h_i' + h_e')`` with the half-pane conductance in
series with each coefficient.
"""
function inward_flowing_fractions(g::Glazing; h_out::Real = 23.0, h_in::Real = 7.7,
                                  T_out::Real = 273.15, T_in::Real = 293.15,
                                  c::SurfaceConstants = DEFAULT_SURFACE_CONSTANTS)
    N = length(g)
    ob, ib = convective_boundary(T_out, h_out), convective_boundary(T_in, h_in)
    ref = _steady_network(g, ob, ib, zeros(N); c = c)
    Nk = zeros(N)
    for j in 1:N
        S = zeros(N); S[j] = 1.0
        sol = _steady_network(g, convective_boundary(0.0, h_out), convective_boundary(0.0, h_in),
                              S; c = c, links = ref.links)
        Nk[j] = sol.q_room
    end
    return Nk
end

"""
    glazing_shgc(g, θ; kwargs...) -> SHGC
    glazing_shgc(g, :hemispherical; kwargs...) -> SHGC

Solar heat gain coefficient (total solar energy transmittance, the EN 410
``g``-value) at incidence angle `θ` (radians) or for diffuse irradiance:

```math
\\mathrm{SHGC}(θ) = T(θ) + \\sum_j N_j A_j(θ)
```

(Klems 2002, Eq. 1; ISO 15099 §7) with the transmittance and per-pane
absorptances of [`glazing_optics`](@ref) and the inward-flowing fractions
of [`inward_flowing_fractions`](@ref) (`kwargs` go there).
"""
function glazing_shgc(g::Glazing, θ; kwargs...)
    o = θ === :hemispherical ? glazing_optics_hemispherical(g) : glazing_optics(g, θ)
    Nk = inward_flowing_fractions(g; kwargs...)
    return o.T + sum(Nk .* o.A)
end

# =========================================================================
# 11. Window facets — AbstractSurface implementers
# =========================================================================

"""
    WindowFacet(glazing; T = 293.15, T_interior = T, h_interior = 1/0.13,
                interior = convective_boundary(T_interior, h_interior),
                cos_incidence = 1.0, diffuse_fraction = 1.0,
                z0m = 0.01, z0h = z0m / 10, h_convective = nothing, θ = 0.5,
                c = DEFAULT_SURFACE_CONSTANTS, nquad = 48)

A glazed facet: the pane network of a [`Glazing`](@ref) whose outer face
is closed by the surface energy balance of the Era V contract and whose
inner face exchanges with the room through `interior` (a
[`ColumnBoundary`](@ref); by default ISO 6946 ``R_{si} = 0.13``, the
same hook as [`UrbanFacet`](@ref)).

**Shortwave.** Incident shortwave `sw_down` (on the facet plane) is split
into a direct beam at the incidence angle `acos(cos_incidence)` and a
diffuse part `diffuse_fraction`; each is reflected, absorbed per pane and
transmitted by [`glazing_optics`](@ref) /
[`glazing_optics_hemispherical`](@ref). Set the geometry each step with
[`set_solar_incidence!`](@ref) (the radiosity / solar-geometry coupling
supplies it). With the default `diffuse_fraction = 1` the facet uses its
hemispherical optics.

  * `surface_albedo(w)` is the effective solar reflectance of the
    glazing at the current geometry, so `net_radiation` and the
    atmosphere see the correct reflected shortwave;
  * the absorbed shortwave heats the panes;
  * the transmitted shortwave enters the room:
    [`transmitted_shortwave`](@ref) (W m⁻² of glazing) is the solar gain
    a building energy model receives, in addition to the conductive /
    convective [`interior_heat_flux`](@ref).

**Energy bookkeeping.** `SurfaceFluxes.G = R_n - H - LE` is everything
the facet takes from the outside — storage in the panes, the heat
conducted to the room, *and* the transmitted shortwave:
``G\\,Δt = ΔE_{panes} + (Q_{interior} + SW_{trans})\\,Δt`` to round-off.

The panes are lumped nodes stepped by the θ-scheme; the outer face is
linearised about the current skin temperature (one Newton step per time
step, as in [`UrbanFacet`](@ref)), and the gap coefficients are
evaluated at the start of the step. Turbulent exchange: Monin–Obukhov
through `SurfaceExchange.turbulent_fluxes`, or ``H = h (T_s - T_{air})``
with a given `h_convective`. Facets are dry.
"""
mutable struct WindowFacet{T<:AbstractFloat} <: AbstractSurface
    glazing::Glazing
    T_panes::Vector{T}
    interior::ColumnBoundary{T}
    T_skin::T
    z0m::T
    z0h::T
    h_convective::Union{Nothing, T}
    θ::T
    cos_incidence::T
    diffuse_fraction::T
    hemispherical::NamedTuple{(:R, :T, :A), Tuple{Float64, Float64, Vector{Float64}}}
    constants::SurfaceConstants
    G::T
    Q_interior::T
    sw_transmitted::T
    sw_absorbed::T
end

function WindowFacet(g::Glazing; T::Real = 293.15, T_interior::Real = T,
                     h_interior::Real = 1 / ISO6946_SURFACE_RESISTANCE.inside_horizontal,
                     interior::Union{Nothing, ColumnBoundary} = nothing,
                     cos_incidence::Real = 1.0, diffuse_fraction::Real = 1.0,
                     z0m::Real = 0.01, z0h::Real = z0m / 10, h_convective = nothing,
                     θ::Real = 0.5, c::SurfaceConstants = DEFAULT_SURFACE_CONSTANTS,
                     nquad::Int = 48)
    _check_theta(θ, "WindowFacet")
    ibc = interior === nothing ? convective_boundary(T_interior, h_interior) : interior
    isinf(ibc.h) && throw(ArgumentError("WindowFacet: the interior boundary must be " *
                                        "finite (a room, not a fixed pane temperature)"))
    ibc = ColumnBoundary{Float64}(ibc.T_env, ibc.h, ibc.q)
    hem = glazing_optics_hemispherical(g; nquad = nquad)
    w = WindowFacet{Float64}(g, fill(float(T), length(g)), ibc, float(T), float(z0m),
                             float(z0h), h_convective === nothing ? nothing : float(h_convective),
                             float(θ), 1.0, 1.0, hem, c, 0.0, 0.0, 0.0, 0.0)
    set_solar_incidence!(w, cos_incidence; diffuse_fraction = diffuse_fraction)
    return w
end

"""
    set_solar_incidence!(w::WindowFacet, cos_incidence; diffuse_fraction = w.diffuse_fraction)

Set the cosine of the direct-beam incidence angle on the facet plane
(`≤ 0`: the sun is behind the facet, no direct beam) and the diffuse
fraction of the incident shortwave, for the following steps.
"""
function set_solar_incidence!(w::WindowFacet, cos_incidence::Real;
                              diffuse_fraction::Real = w.diffuse_fraction)
    (0 <= diffuse_fraction <= 1) ||
        throw(ArgumentError("set_solar_incidence!: diffuse_fraction must lie in [0, 1]"))
    w.cos_incidence = clamp(cos_incidence, 0.0, 1.0)
    w.diffuse_fraction = w.cos_incidence > 0 ? diffuse_fraction : 1.0
    return w
end

"""
    window_optics(w::WindowFacet) -> (R = …, T = …, A = […])

Effective solar reflectance, transmittance and per-pane absorptance of
the facet at its current geometry: the diffuse-fraction-weighted mix of
the hemispherical and direct-beam optics.
"""
function window_optics(w::WindowFacet)
    fd = w.diffuse_fraction
    h = w.hemispherical
    fd >= 1 && return h
    d = glazing_optics(w.glazing, acos(w.cos_incidence))
    return (R = fd * h.R + (1 - fd) * d.R, T = fd * h.T + (1 - fd) * d.T,
            A = fd .* h.A .+ (1 - fd) .* d.A)
end

surface_temperature(w::WindowFacet) = w.T_skin
roughness_lengths(w::WindowFacet) = (z0m = w.z0m, z0h = w.z0h, z0q = w.z0h)
surface_albedo(w::WindowFacet) = window_optics(w).R
surface_emissivity(w::WindowFacet) = w.glazing.panes[1].emissivity

"Heat from the inner pane into the room over the last step (W m⁻², conduction/convection + long-wave through `h_interior`); excludes the transmitted shortwave."
interior_heat_flux(w::WindowFacet) = w.Q_interior

"""
    transmitted_shortwave(w::WindowFacet) -> W m⁻²

Shortwave transmitted into the room over the last step, per unit
glazing area — the direct solar gain a building energy model deposits
on its interior surfaces.
"""
transmitted_shortwave(w::WindowFacet) = w.sw_transmitted

"Pane temperatures (K), outside first."
pane_temperatures(w::WindowFacet) = w.T_panes

"""
    window_heat_content(w; T_ref = 0) -> J m⁻²

Sensible heat stored in the panes, ``Σ_j ρ_j c_j d_j (T_j - T_{ref})``.
"""
window_heat_content(w::WindowFacet; T_ref::Real = 0) =
    sum(j -> _pane_C(w.glazing.panes[j]) * (w.T_panes[j] - T_ref), eachindex(w.T_panes))

_pane_C(p::GlassPane) = volumetric_heat_capacity(p.material) * p.thickness

set_interior_boundary!(w::WindowFacet, bc::ColumnBoundary) =
    (isinf(bc.h) && throw(ArgumentError("WindowFacet: interior boundary must be finite"));
     w.interior = ColumnBoundary{Float64}(bc.T_env, bc.h, bc.q); w)
set_interior_temperature!(w::WindowFacet, T_in::Real) =
    (w.interior = ColumnBoundary{Float64}(float(T_in), w.interior.h, w.interior.q); w)

function _window_fluxes(w::WindowFacet, forcing::SurfaceForcing; kwargs...)
    c = w.constants
    if w.h_convective === nothing
        return turbulent_fluxes(w, forcing; c = c, kwargs...)
    end
    Rn = net_radiation(w, forcing; c = c)
    H = w.h_convective * (w.T_skin - forcing.T_air)
    z = zero(Rn)
    return SurfaceFluxes(promote(Rn, H, z, Rn - H, z, z)...)
end

"""
    step_surface!(w::WindowFacet, forcing, dt; δT = 0.01, kwargs...) -> SurfaceFluxes

Advance a window facet by `dt` seconds. The shortwave is partitioned by
[`window_optics`](@ref); the outer-face balance *without* the shortwave,
``F(T_s) = εL↓ - εσT_s^4 - H - LE``, is linearised about the current skin
temperature and closes the pane network as a Robin condition through the
half-pane conductance; the pane temperatures are advanced by the
θ-scheme with the absorbed shortwave as node sources. Returns the
θ-weighted fluxes: ``R_n = (1-R)\\,SW↓ + LW_{net}``, and
``G = R_n - H - LE`` exactly (see [`WindowFacet`](@ref) for what `G`
contains). `kwargs` go to `SurfaceExchange.turbulent_fluxes`.
"""
function step_surface!(w::WindowFacet{T}, forcing::SurfaceForcing, dt::Real;
                       δT::Real = 0.01, kwargs...) where {T}
    dt > 0 || throw(ArgumentError("step_surface!: dt must be positive"))
    g = w.glazing
    N = length(g)
    opt = window_optics(w)
    sw = forcing.sw_down
    S = opt.A .* sw
    sw_net = (1 - opt.R) * sw                 # the shortwave part of Rn
    # Outer-face balance without the shortwave, and its derivative.
    Ts = w.T_skin
    fl0 = _window_fluxes(w, forcing; kwargs...)
    w.T_skin = Ts + δT
    fl1 = _window_fluxes(w, forcing; kwargs...)
    w.T_skin = Ts
    LW0 = fl0.Rn - sw_net
    dLW = (fl1.Rn - fl0.Rn) / δT
    dH = (fl1.H - fl0.H) / δT
    dLE = (fl1.LE - fl0.LE) / δT
    F0 = LW0 - fl0.H - fl0.LE
    Fp = dLW - dH - dLE
    Fp < 0 || (Fp = min(dLW, -eps(T)))
    outer = ColumnBoundary{T}(Ts, -Fp, F0)
    K1 = _pane_K(g.panes[1]); KN = _pane_K(g.panes[N])
    go, so = _boundary_coefficients(outer, K1)
    gi, si = _boundary_coefficients(w.interior, KN)
    Gl = _gap_links!(zeros(max(N - 1, 0)), g, w.T_panes, w.constants)
    θ = w.θ
    Tn = w.T_panes
    Fo_old = go * (outer.T_env - Tn[1]) + so
    Fi_old = gi * (w.interior.T_env - Tn[N]) + si
    a, b, cc, d = zeros(T, N), zeros(T, N), zeros(T, N), zeros(T, N)
    for j in 1:N
        m = _pane_C(g.panes[j]) / dt
        R = zero(T); diag = zero(T)
        if j > 1
            R += Gl[j - 1] * (Tn[j - 1] - Tn[j]); diag += Gl[j - 1]; a[j] = -θ * Gl[j - 1]
        end
        if j < N
            R += Gl[j] * (Tn[j + 1] - Tn[j]); diag += Gl[j]; cc[j] = -θ * Gl[j]
        end
        rhs = S[j]
        if j == 1
            diag += go; R += Fo_old; rhs += θ * (go * outer.T_env + so)
        end
        if j == N
            diag += gi; R += Fi_old; rhs += θ * (gi * w.interior.T_env + si)
        end
        b[j] = m + θ * diag
        d[j] = m * Tn[j] + (1 - θ) * R + rhs
    end
    Tf_old = _face_temperature(outer, K1, Tn[1])
    _thomas!(Tn, a, b, cc, d)
    Fo_new = go * (outer.T_env - Tn[1]) + so
    Fi_new = gi * (w.interior.T_env - Tn[N]) + si
    Fo = θ * Fo_new + (1 - θ) * Fo_old
    Fi = θ * Fi_new + (1 - θ) * Fi_old
    Tf_new = _face_temperature(outer, K1, Tn[1])
    x = θ * Tf_new + (1 - θ) * Tf_old - Ts
    LW = LW0 + dLW * x
    LE = fl0.LE + dLE * x
    H = LW - LE - Fo
    Rn = sw_net + LW
    E = LE / surface_latent_heat(w; c = w.constants)
    w.T_skin = Tf_new
    w.sw_transmitted = opt.T * sw
    w.sw_absorbed = sum(S)
    w.Q_interior = -Fi
    w.G = Rn - H - LE
    return SurfaceFluxes{T}(T(Rn), T(H), T(LE), T(w.G), T(fl0.τ), T(E))
end
