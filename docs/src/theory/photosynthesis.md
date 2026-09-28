# Leaf photosynthesis

Transpiration is the largest latent-heat flux over vegetated land, and
stomata open to let CO₂ in. Before Era V can make evaporation respond to
light, CO₂, humidity and soil water, it needs the rate at which a leaf
*uses* CO₂. This page develops that rate as implemented in
`src/Photosynthesis.jl`; stomatal conductance, the land column and the
resolved canopy build on it.

## 1. Three limitations of a C3 leaf

Farquhar, von Caemmerer & Berry (1980) describe C3 photosynthesis as
the smallest of three biochemical capacities. Rubisco either carboxylates
RuBP (fixing CO₂) or oxygenates it (photorespiration, releasing half a
CO₂ per oxygenation). The ratio of the two is set by the CO₂ and O₂ at
the enzyme, so net carboxylation vanishes at the *photorespiratory
compensation point*

```math
\Gamma^* = \frac{o_i}{2\,S_{c/o}},
```

with ``S_{c/o}`` the Rubisco specificity factor. ``\Gamma^*`` is
proportional to O₂, which is why low-O₂ air raises C3 photosynthesis
(the Warburg effect).

**Rubisco-limited** — enough RuBP, too little enzyme:

```math
A_c = V_{cmax}\,\frac{c_i - \Gamma^*}{c_i + K_c(1 + o_i/K_o)} .
```

**RuBP-regeneration (light)-limited** — the electron transport rate
``J`` limits how fast RuBP is regenerated; four electrons per
carboxylation and eight per oxygenation give

```math
A_j = J\,\frac{c_i - \Gamma^*}{4 c_i + 8 \Gamma^*},
\qquad
\Theta J^2 - (I + J_{max}) J + I J_{max} = 0,
\quad I = \tfrac12 \Phi_{PSII} Q,
```

``J`` being the smaller root: linear in absorbed photons ``Q`` at low
light (one photon per photosystem per electron, hence the ½), saturating
at ``J_{max}``.

**Export (TPU)-limited** — the chloroplast cannot export triose
phosphate faster than ``T_p``: ``A_p = 3T_p``.

Net assimilation subtracts mitochondrial respiration in the light,
``A_n = A - R_d``.

### Analytic limits the tests assert

| Limit | Result |
| :-- | :-- |
| Darkness, ``Q = 0`` | ``J = 0 \Rightarrow A = 0,\ A_n = -R_d`` exactly |
| ``c_i = \Gamma^*`` | ``A = 0`` at any light |
| ``c_i = 0`` (Rubisco-limited) | ``A = -V_{cmax}\Gamma^*/K_m`` |
| CO₂ compensation point (Rubisco) | ``\Gamma = (\Gamma^* V_{cmax} + K_m R_d)/(V_{cmax} - R_d)`` |
| Initial quantum yield | ``dA/dQ \to \tfrac12\Phi_{PSII}(c_i-\Gamma^*)/(4c_i+8\Gamma^*)`` |
| Rubisco → RuBP transition | ``c_i = (J K_m - 8\Gamma^* V_{cmax})/(4V_{cmax} - J)`` |
| Light saturation | ``J \to J_{max}`` |

with ``K_m = K_c(1+o_i/K_o)``. At 25 °C with CLM's constants
(``\Gamma^* = 42.75``) the compensation point of a leaf with
``V_{cmax25} = 60`` is 54.2 µmol mol⁻¹, and the initial quantum yield at
``c_i = 280`` is 0.069 mol CO₂ per mol absorbed photons.

**TPU with CLM defaults.** ``A_j`` is bounded by ``J_{max}/4``
(``c_i \to \infty``). CLM sets ``T_{p25} = 0.167 V_{cmax25}`` so that
``3T_p = 0.501 V_{cmax}``, and ``J_{max25} = 1.97 V_{cmax25}``, so
``J_{max}/4 = 0.4925 V_{cmax} < 3T_p``: at 25 °C the default C3 leaf is
*never* TPU-limited. TPU limitation appears when
``J_{max25}/V_{cmax25} > 2.004``, e.g. for a Kattge-Knorr leaf grown
below ~ 16 °C.

## 2. Co-limitation

The transitions between limitations are not sharp in real leaves.
Following Collatz et al. (1991) and CLM, the gross rate is a nested
smooth minimum, each the smaller root of a quadratic,

```math
\Theta_{cj} A_i^2 - (A_c + A_j) A_i + A_c A_j = 0,\qquad
\Theta_{ip} A^2 - (A_i + A_p) A + A_i A_p = 0,
```

with ``\Theta_{cj} = 0.98``, ``\Theta_{ip} = 0.95`` (C3). ``\Theta = 1``
recovers the Farquhar minimum and ``\Theta = 0`` the rectangular
hyperbola. The root is evaluated as
``2ab/[(a+b) + \sqrt{(a+b)^2 - 4\Theta ab}]``, free of cancellation.

Because this smooth minimum is homogeneous of degree one, the C3 rate is
written ``A_i = (c_i - \Gamma^*)\,\mathrm{colim}(V_{cmax}/(c_i+K_m),\,
J/(4c_i+8\Gamma^*))``. Above ``\Gamma^*`` this is identical to CLM;
below it, it continues the curve to the physically correct negative
gross rate (photorespiration exceeding carboxylation), where CLM clips.

## 3. C4 leaves

C4 plants concentrate CO₂ around Rubisco with PEP carboxylase, which
removes photorespiration and saturates the leaf near ambient CO₂.
Collatz et al. (1992) replace the three rates with

```math
A_c = V_{cmax},\qquad A_j = \alpha Q,\qquad A_p = k_p\,c_i ,
```

co-limited with ``\Theta_{cj} = 0.80``, ``\Theta_{ip} = 0.95`` (CLM) or
0.83 / 0.93 (the paper). The CO₂ compensation point ``R_d/k_p`` is
≈ 1.25 µmol mol⁻¹ for CLM parameters, against ≈ 54 for C3.

## 4. Temperature

Michaelis constants and ``\Gamma^*`` follow Arrhenius functions with the
activation energies of Bernacchi et al. (2001),
``f(T) = \exp[\Delta H_a(1 - T_{25}/T)/(R T_{25})]``. Capacities
(``V_{cmax}``, ``J_{max}``, ``T_p``, ``R_d``) additionally deactivate at
high temperature:

```math
f_H(T) = \frac{1 + \exp[(T_{25}\Delta S - \Delta H_d)/(R T_{25})]}
              {1 + \exp[(T\,\Delta S - \Delta H_d)/(R T)]},
\qquad
T_{opt} = \frac{\Delta H_d}{\Delta S - R\ln[\Delta H_a/(\Delta H_d - \Delta H_a)]}.
```

Two parameter sets are provided. **Bonan et al. (2011)** (CLM4.5 Table
8.2) does not acclimate; its ``V_{cmax}`` optimum is 33.3 °C. **Kattge &
Knorr (2007)** lets the entropy term follow the growth temperature
``t_g`` (°C, clamped to 11–35), ``\Delta S = 668.39 - 1.07 t_g`` for
``V_{cmax}`` and ``659.70 - 0.75 t_g`` for ``J_{max}``, and lowers
``J_{max25}/V_{cmax25} = 2.59 - 0.035 t_g``; the ``V_{cmax}`` optimum
moves from 29.2 °C (grown at 11 °C) to 41.5 °C (grown at 35 °C).

**A caveat worth knowing.** The deactivation term of the Bonan set is
not negligible at 25 °C — it already removes 13 % of ``V_{cmax}`` there —
and normalising to 25 °C hides that. The peaked curve therefore lies
10–15 % *above* Bernacchi's pure Arrhenius fit between 10 and 20 °C and
falls below it above 25 °C (−18 % at 30 °C). The activation energies
agree; the shapes are different by construction.

C4 capacities use CLM's ``Q_{10} = 2`` with low- and high-temperature
inhibition (CLM4.5 eqs. 8.12–8.14), not renormalised at 25 °C (the C4
``V_{cmax}`` factor there is 0.871); the optimum is 36.2 °C.

## 5. Plant functional types

`CLM45_PFT_PHOTOSYNTHESIS` carries CLM4.5 Table 8.1: pathway, canopy-top
``V_{cmax25}``, leaf C:N, fraction of leaf N in Rubisco and specific leaf
area for 14 natural and 5 crop types (rainfed rows; the irrigated
rows are identical). ``V_{cmax25}`` in that table is
itself derived, ``V_{cmax25} = F_{LNR} F_{NR} a_{R25} / (CN_L\,SLA_0)``;
recomputing it from the other columns reproduces every row to within
0.1 %, which is how the transcription is checked.

## 6. What is and is not validated

| Check | Reference | Result |
| :-- | :-- | :-- |
| ``K_c, K_o, \Gamma^*`` temperature shape, 5–40 °C | Bernacchi et al. (2001) fits | max rel. error 1.3 × 10⁻⁴ |
| ``K_c, K_o, \Gamma^*`` absolute, 5–40 °C | Bernacchi et al. (2001) fits | < 1 % (max 0.54 %: the fit's two-decimal ``c`` does not reproduce Bernacchi's own tabulated 25 °C values exactly) |
| ``V_{cmax}``, ``R_d`` activation part | Bernacchi et al. (2001) fits | < 1 % |
| ``T_{opt}`` closed form | numerical maximum | < 2 mK |
| CLM4.5 Table 8.1 transcription | eq. 8.17 from same row | max 9.3 × 10⁻⁴ |
| Analytic limits of §1, §3 | closed forms | 10⁻⁸ – 10⁻³ |

Not validated here: measured A–ci curves or leaf gas-exchange
observations. The phase's observational target — FLUXNET latent and
sensible heat — needs the stomatal conductance and land column that
build on this module.

## References

- Farquhar, von Caemmerer & Berry (1980), *Planta* 149, 78–90.
- von Caemmerer (2000), *Biochemical Models of Leaf Photosynthesis*, CSIRO.
- Collatz et al. (1991), *Agric. For. Meteorol.* 54, 107–136.
- Collatz, Ribas-Carbo & Berry (1992), *Aust. J. Plant Physiol.* 19, 519–538.
- Bernacchi et al. (2001), *Plant Cell Environ.* 24, 253–259.
- Medlyn et al. (2002), *Plant Cell Environ.* 25, 1167–1179.
- Kattge & Knorr (2007), *Plant Cell Environ.* 30, 1176–1190.
- Bonan et al. (2011), *J. Geophys. Res.* 116, G02014.
- Oleson et al. (2013), CLM4.5 Technical Note, NCAR/TN-503+STR, §8.
