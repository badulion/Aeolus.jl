# Glazing and window facets

Windows are where sunshine gets into a building without first being
turned into heat on an outer surface. The glazing reflects part of the
incident shortwave, absorbs part in the glass, and transmits the rest
straight into the room. All three fractions depend on the incidence
angle: at grazing incidence clear glass is a mirror. The single glass
class of the urban example had one fixed albedo and one fixed
transmission. `src/urban_surface/glazing.jl`, part of `UrbanSurface`,
replaces it with the first-principles optics of LBNL WINDOW and ISO 15099
and a window facet that implements the Era V surface interface.

## 1. One pane: Fresnel plus Beer–Lambert

A pane is a homogeneous, specular, weakly absorbing slab of thickness
``d``, refractive index ``n`` and solar extinction coefficient ``K``.
For a beam at incidence angle ``θ``, Snell's law gives the refraction
angle ``\sin θ_t = \sin θ / n``. The Fresnel equations give the
reflectivity of each interface for the two polarisations,

```math
ρ_s = \Bigl(\frac{\cos θ - n\cos θ_t}{\cos θ + n\cos θ_t}\Bigr)^2,\qquad
ρ_p = \Bigl(\frac{n\cos θ - \cos θ_t}{n\cos θ + \cos θ_t}\Bigr)^2 .
```

This is `SurfaceOptics.fresnel_polarized_reflectance`, the package's only
implementation of the Fresnel equations; the water albedo uses it too.
Along the refracted path the internal transmittance is
``τ = \exp(-Kd/\cos θ_t)``. Summing the incoherent multiple reflections
between the two faces gives, for each polarisation,

```math
T = \frac{(1-ρ)^2 τ}{1 - ρ^2τ^2},\qquad
R = ρ + \frac{ρ(1-ρ)^2τ^2}{1-ρ^2τ^2},\qquad A = 1 - R - T ,
```

and sunlight, which is unpolarised, takes the mean of the two
(Duffie & Beckman 2013, §5.1–5.3). Each polarisation is summed
separately *before* averaging. Using the averaged ``ρ`` in the sum
instead, as WINDOW's Eq. 7-23 does, under-predicts ``T`` at oblique
incidence: by 0.008 at the Brewster angle and 0.018 at 70° for a 4 mm
pane. The two components differ most there, with ``ρ_p → 0`` while
``ρ_s`` does not.

**Parameters.** Duffie & Beckman give ``n = 1.526`` and ``K`` from about
4 m⁻¹ (water-white, low-iron glass) to about 32 m⁻¹ (ordinary float with
a greenish edge). The defaults are these book values, not fitted ones.
`glass_pane_from_normal` performs the inversion WINDOW uses for uncoated
glass (Furler 1991). It takes a data sheet's normal-incidence ``(T_0, R_0)``,
solves for ``ρ_0``, then ``n = (1+\sqrt{ρ_0})/(1-\sqrt{ρ_0})`` and
``τ = (R_0-ρ_0)/(ρ_0T_0)``. The resulting pane reproduces the data sheet
exactly. Inverting the Vitro 3 mm clear row (0.83, 0.08) gives
``n = 1.543`` and ``K = 31.6`` m⁻¹, close to the book values.

**Limitation: a grey pane.** One broadband ``K`` is a Beer–Lambert law
for the whole solar spectrum. Real float glass absorbs mainly in the
Fe²⁺ band near 1 μm, and once that band is saturated, extra thickness
removes little more. The grey model therefore matches building-glass
thicknesses (3–8 mm) but under-predicts the transmittance of very thick
glass by up to 0.051 at 19 mm (see §6).

## 2. Multi-pane units

A unit of ``N`` panes (outside first) separated by gas gaps is solved
with the net-radiation method. Let ``D_k`` and ``U_k`` be the inward and
outward fluxes in gap ``k``, where gap 0 is outside and gap ``N`` is the
room, treated as black. Then

```math
D_0 = 1,\quad U_N = 0,\quad
D_j = T_jD_{j-1} + R_jU_j,\quad U_{j-1} = R_jD_{j-1} + T_jU_j ,
```

and pane ``j`` absorbs ``A_j(D_{j-1} + U_j)``. This linear system is
equivalent to the recursions of ISO 15099 §6 and WINDOW Eqs. 7-15 to
7-20. It is solved per polarisation. ``R + T + ΣA_j = 1`` holds to
round-off. For diffuse irradiance, the optics are the cosine-weighted
hemispherical average ``P_{hem} = \int_0^1 P(μ)\,2μ\,dμ`` (WINDOW
Eq. 7-36), computed by Gauss–Legendre quadrature in ``μ = \cos θ``.

## 3. Heat transfer across the gaps

The fill gas has the temperature-dependent conductivity, viscosity and
heat capacity of ISO 15099 Annex B (air and argon are provided). Its
density is that of an ideal gas at the gap pressure. Convection in a
vertical cavity follows the ISO 15099 correlation (from Wright 1996),
``Nu = \max(Nu_1(Ra), 0.242(Ra/A)^{0.272})`` with ``h_c = Nu\,k/d``, which
reduces to pure conduction as ``Ra → 0``. The ISO 15099 air conductivity
at 0 °C equals the contract's still-air `SurfaceConstants.k_air`, and a
test asserts this. Radiation between the panes is the exact grey
parallel-plate exchange,

```math
h_r = \frac{σ(T_1^2+T_2^2)(T_1+T_2)}{1/ε_1 + 1/ε_2 - 1},
```

with the EN 673 emissivity of uncoated glass, 0.837, as the default.

## 4. The pane network: U-value and SHGC

Each pane is a node at its mid-plane with heat capacity ``ρcd``, taken
from the `:glass` row of the thermal material table. A half-pane
conductance ``2k/d`` separates the node from each face. Adjacent nodes
are coupled through a half pane, the gap coefficient ``h_c + h_r`` and
another half pane in series. The outer and inner boundaries are the same
Robin conditions (`ColumnBoundary`) as the opaque facets use. Absorbed
shortwave is deposited at the nodes.

Three results come from this one network:

* **U-value** (`glazing_u_value`). The steady heat flow through the unit
  between outdoor and indoor air, with the gap coefficients iterated to
  the pane temperatures. The defaults are the EN 673 surface
  coefficients ``h_e = 25`` and ``h_i = 7.7`` W m⁻² K⁻¹.
* **Inward-flowing fractions** ``N_j`` (`inward_flowing_fractions`). The
  fraction of the heat absorbed in pane ``j`` that reaches the room in
  steady state.
* **SHGC** (`glazing_shgc`). The total solar energy transmittance
  ``\mathrm{SHGC}(θ) = T(θ) + Σ_j N_jA_j(θ)`` (Klems 2002; ISO 15099 §7;
  EN 410's ``g``). The default surface coefficients are EN 410's,
  ``h_e = 23`` and ``h_i = 3.6 + 4.4\,ε/0.837 = 8.0`` W m⁻² K⁻¹.

## 5. The window facet

`WindowFacet <: AbstractSurface` makes the network time-dependent. The
incident shortwave on the facet plane is split into a direct beam at the
current incidence angle and a diffuse part (`set_solar_incidence!`). The
optics of §1–2 then partition it:

* ``R\,SW↓`` is reflected. `surface_albedo(w)` returns ``R``, so
  `net_radiation` and the atmosphere see the correct reflected
  shortwave.
* ``A_j\,SW↓`` heats pane ``j``.
* ``T\,SW↓`` enters the room. It is reported by `transmitted_shortwave`,
  the direct solar gain a building energy model deposits on its interior
  surfaces. The conductive, convective and long-wave gain through
  ``h_{in}`` is reported separately by `interior_heat_flux`.

The outer-face balance without the shortwave,
``F(T_s) = εL↓ - εσT_s^4 - H - LE``, is linearised about the current
skin temperature. The pane temperatures are advanced by the θ-scheme,
exactly as for an opaque `UrbanFacet`. The contract's
``G = R_n - H - LE`` is everything the facet takes from outside:
storage in the panes, the heat conducted into the room, and the
transmitted shortwave. It satisfies
``G\,Δt = ΔE_{panes} + (Q_{interior} + SW_{trans})\,Δt`` to round-off.

## 6. Validation

| check | reference | result |
|:------|:----------|:-------|
| Fresnel limits, Brewster, Snell | analytic | 1e-13 |
| non-absorbing stack of ``N`` plates, ``T = (1-ρ)/(1+(2N-1)ρ)`` | Duffie & Beckman Eq. 5.1.6 | 1e-13, ``N`` = 1–3 |
| data-sheet inversion round trip | WINDOW Eqs. 7-26 to 7-29 | exact (1e-12) |
| clear float, normal incidence, 3–8 mm | Vitro Clear, ISO 9050 | ``T`` within 0.015, ``R`` within 0.011 |
| clear float, 10–19 mm | same | ``T`` 0.016–0.051 low (grey model) |
| 3 mm angular ``T(θ)``, ``R(θ)``, 0–80° | WINDOW clear-glass regression (Table 7-1) | ``T`` ≤ 0.007, ``R`` ≤ 0.012 |
| single 3 mm clear SHGC | NFRC (LLumar data sheet) 0.86 | 0.857 |
| 6-13-6 clear/clear SHGC | Vitro IGU data (NFRC, WINDOW 7.3) 0.70 | 0.682 (book ``K``), 0.691 (inverted Vitro pane) |
| 6-13-6 U, NFRC winter, air / argon | Vitro 2.669 / 2.555 W m⁻² K⁻¹ | 2.707 / 2.547 (see note) |
| facet steady state vs SHGC, 1–3 panes | the steady network | ≤ 0.0008 |
| facet steady state vs U-value | the steady network | 4 significant figures |
| facet energy bookkeeping, 24 h diurnal | exact | 2e-9 J m⁻² |

The WINDOW angular regression is a single curve for all thicknesses,
whereas the physical path length grows as ``1/\cos θ_t``. For 6 mm glass
the model therefore falls below the regression towards grazing
incidence: by 0.026 in ``T`` and 0.038 in ``R`` at 80°. The direction is
the one the path-length argument predicts.

The published argon U-value is for a 90 % argon / 10 % air fill; the
model gap is pure argon. Blending the air and argon property fits 90/10
(a crude stand-in for the ISO 15099 mixing rules, which are not
implemented) raises the model U by about 0.017 W m⁻² K⁻¹, so the
like-for-like model value is about 2.56 (+0.3 %) rather than 2.547.

No field observations of façade temperatures are used here. That
comparison (BUBBLE / CAPITOUL) is the Phase 49 validation issue.

## References

* Duffie, J. A. & Beckman, W. A. (2013). *Solar Engineering of Thermal
  Processes*, 4th ed., Wiley. §5.1–5.3.
* Furler, R. A. (1991). Angular dependence of optical properties of
  homogeneous glasses. *ASHRAE Transactions* 97(2).
* ISO 15099:2003. *Thermal performance of windows, doors and shading
  devices — Detailed calculations.* §5.3 (gaps), §6 (optics), Annex B
  (gas properties).
* EN 410:2011. *Glass in building — Determination of luminous and solar
  characteristics of glazing.* EN 673:2011. *Glass in building —
  Determination of thermal transmittance (U value).*
* Klems, J. H. (2002). Solar heat gain through fenestration systems
  containing shading: summary of procedures for estimating performance
  from minimal data. LBNL-48672.
* LBNL (2018). *WINDOW Technical Documentation*, §7 (glazing system
  optical performance).
* Vitro Architectural Glass (2020). *Clear Technical Product Data*; and
  *Performance Data for One-Inch Insulating Glass Units*.
* Wright, J. L. (1996). A correlation to quantify convective heat
  transfer between vertical window glazings. *ASHRAE Transactions* 102(1).
