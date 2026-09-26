# Stomatal conductance

Stomata are pores that let CO₂ into a leaf, and water vapour out through
the same opening. The [leaf photosynthesis](photosynthesis.md) page gives
the assimilation rate ``A_n(c_i)`` at a given intercellular CO₂. This page
closes the loop: how open the stomata are, the CO₂ inside the leaf that
results, and the transpiration that comes with it. The code is the
second half of `src/Photosynthesis.jl` (Phase 47, #20).

## 1. Diffusion paths

CO₂ travels from the air (``c_a``) across the leaf boundary layer to the
leaf surface (``c_s``), then through the stomatal pore to the
intercellular spaces (``c_i``). With no storage at the surface, the three
fluxes are equal (CLM5 eq. 2.9.19):

```math
A_n = \frac{c_a - c_i}{1.4/g_b + 1.6/g_s}
    = \frac{g_b}{1.4}(c_a - c_s) = \frac{g_s}{1.6}(c_s - c_i).
```

Here ``g_s`` and ``g_b`` are the stomatal and boundary-layer
conductances *to water vapour*, in mol m⁻² s⁻¹. 1.6 is the ratio of the
molecular diffusivities of H₂O and CO₂ in air. 1.4 ≈ 1.6^{2/3} is the
same ratio for a laminar boundary layer, where the dependence on
diffusivity is weaker. With CO₂ in µmol mol⁻¹ the flux is in
µmol m⁻² s⁻¹ and pressure drops out.

Water vapour leaves along the same path in reverse. The leaf interior is
saturated at leaf temperature, ``e_i = e_{sat}(T_\ell)``, where ``e_{sat}``
is `MoistThermodynamics.saturation_vapor_pressure`, so

```math
E = \frac{e_i - e_a}{p}\,\frac{g_s g_b}{g_s + g_b},\qquad
e_s = \frac{g_b e_a + g_s e_i}{g_b + g_s}
```

(CLM5 eqs. 2.9.20, 2.9.23). The humidity at the leaf surface therefore
depends on ``g_s`` itself: opening the stomata humidifies the leaf
surface.

### Boundary layer

For forced convection over a leaf of dimension ``d`` the boundary-layer
conductance scales as ``\sqrt{U/d}`` (laminar flat plate). CLM5 (eq.
2.5.122) takes

```math
g_b = C_v \sqrt{U/d_{leaf}},\qquad C_v = 0.01\ \mathrm{m\,s^{-1/2}},\quad d_{leaf} = 0.04\ \mathrm{m},
```

and converts it to molar units with ``p/(RT)``. At ``U = 1`` m s⁻¹ and
20 °C this is 2.08 mol m⁻² s⁻¹, typically ten times the stomatal
conductance. That is why well-ventilated leaves are close to ``c_s ≈ c_a``.

## 2. Stomatal closures

**Ball-Berry** (Ball, Woodrow & Berry 1987; Collatz et al. 1991; CLM4.5):

```math
g_s = g_0 + g_1\,\frac{A_n h_s}{c_s},
```

where ``h_s = e_s/e_i`` is the relative humidity at the leaf surface.
Substituting ``e_s(g_s)`` gives a quadratic in ``g_s``:

```math
g_s^2 + (g_b - g_0 - a)\,g_s - g_b\,(g_0 + a\,e_a/e_i) = 0,\qquad a = g_1 A_n / c_s .
```

This is the quadratic of CLM's `ci_func`. The product of its roots is
negative, so it has exactly one positive root, and the code evaluates
that root in a form free of cancellation. CLM defaults: ``g_1 = 9``,
``g_0 = 0.01`` mol m⁻² s⁻¹ for C3; ``g_1 = 4``, ``g_0 = 0.04`` for C4.

**Medlyn** (Medlyn et al. 2011; CLM5 eq. 2.9.1). This form comes from
optimal stomatal behaviour, which maximises carbon gain minus a
marginal water cost:

```math
g_s = g_0 + 1.6\left(1 + \frac{g_1}{\sqrt{D_s}}\right)\frac{A_n}{c_s},
```

with ``D_s`` in kPa. Since ``D_s = e_i - e_s = g_b D_a/(g_b + g_s)``, the
substitution ``x = \sqrt{g_b + g_s}`` turns the closure into

```math
x^2 - c_2 x - (g_b + g_0 + c_1) = 0,\qquad
c_1 = 1.6 A_n / c_s,\quad c_2 = g_1 c_1 / \sqrt{g_b D_a},
```

which again has exactly one positive root. Squaring the closure instead
gives CLM5's quadratic (eqs. 2.9.24–2.9.26). Its larger root is the same
``g_s``; the tests check agreement to 3 × 10⁻¹⁶.

For ``g_0 = 0`` and ``g_b → ∞`` the Medlyn closure implies

```math
\frac{c_i}{c_a} = \frac{g_1}{g_1 + \sqrt{D}},
```

independent of ``A_n``, light, temperature and ``c_a``. This is the
model's defining prediction and the strongest analytic test of the
coupled solver.

**No net uptake.** For ``A_n ≤ 0`` both closures return ``g_0``. Both
are continuous at ``A_n = 0``.

**Deficit bound (a deliberate difference from CLM5).** The Medlyn
closure diverges as ``D_s → 0``. CLM5 bounds the *air* deficit
``e_i - e_a`` at 50 Pa. That leaves the leaf-surface deficit unbounded
for a weakly coupled leaf, where the humidification feedback can run
away: at 5 °C, 98 % RH and ``g_b = 0.1`` mol m⁻² s⁻¹ the tests found
``g_s ≈ 10^3`` mol m⁻² s⁻¹. Aeolus evaluates the closure at
``\max(D_s, D_{min})`` with ``D_{min}`` = 50 Pa. This caps ``g_s`` and is
identical to CLM5 whenever ``D_s ≥ D_{min}``.

## 3. The coupled solution

Given ``c_i``, the four relations determine everything else:
``A_n = f(c_i)`` from the Farquhar or Collatz biochemistry,
``c_s = c_a - 1.4A_n/g_b``, ``g_s`` from the closure, and a new
intercellular CO₂. The solution is the root of

```math
R(c_i) = c_i - \left[c_a - A_n(c_i)\left(\frac{1.4}{g_b} + \frac{1.6}{g_s}\right)\right].
```

``R(0) < 0``, because a leaf cannot fix carbon with no CO₂ inside. At the
upper end ``c_i = c_a``, ``R ≥ 0`` whenever the leaf has net uptake
there. Otherwise, as in darkness, the root lies above ``c_a``, at the
``c_i`` that respiration implies. `leaf_gas_exchange` brackets the root
and solves it with the Illinois method, the same routine
`SurfaceExchange` uses for Monin–Obukhov. It is the bracketed
counterpart of CLM5's hybrid secant/Brent solver, which replaced CLM4's
fixed-point iteration because that iteration did not always converge
(Sun et al. 2012, cited in CLM5 §2.9.6). Typical cases need 5–20
iterations. Near the compensation point up to ~100 are needed, because
``g_s`` drops to ``g_0`` as ``A_n`` changes sign and ``R`` becomes steep
there.

## 4. Soil-water stress

The plant wetness factor ``β ∈ [0, 1]`` (CLM's `btran`) is an **input**.
The land column (#21) computes it from soil matric potential.
`apply_water_stress` multiplies it into the parameters. By default it
scales ``V_{cmax}``, ``R_d`` and ``g_0``, as CLM4.5/CLM5 do. It can
also scale the slope ``g_1`` (`stress_targets = (:g1,)`), which is the
alternative used in CABLE-style studies. The stomatal intercept is
always at least 1 µmol m⁻² s⁻¹, CLM's floor, so the system stays well
posed at ``β = 0``.

## 5. Per-PFT parameters

`pft_stomatal_model(pft)` returns the CLM5 Medlyn closure, with ``g_1``
from Table 2.9.1 (`CLM5_MEDLYN_G1`, the CABLE values of De Kauwe et al.
2015) and ``g_0 = 10^{-4}`` mol m⁻² s⁻¹. With `model = :ball_berry` it
returns the CLM4.5 closure. The generic C3 crop and winter-cereal rows
of CLM4.5 have no Table 2.9.1 entry. They are left out rather than
guessed.

## 6. What is and is not validated

All numbers below were measured by `test/era5_stomata.jl`. Unless
stated otherwise the conditions are 25 °C, 400 W m⁻² PAR (85 % absorbed),
``U = 1`` m s⁻¹ and ``D = 1.5`` kPa.

| Check | Reference | Result |
| :-- | :-- | :-- |
| Ball-Berry root | defining equation / CLM `ci_func` quadratic | 4 × 10⁻¹⁶ / 2 × 10⁻¹² |
| Medlyn root | defining equation / CLM5 eqs. 2.9.24–26 | 5 × 10⁻¹⁵ / 3 × 10⁻¹⁶ |
| Medlyn ``g_s - g_0 ∝ D^{-1/2}``, ``∝ A``, ``∝ 1/c_s`` | closed form | log-slope −0.5 to 10⁻¹² |
| ``c_i/c_a = g_1/(g_1+\sqrt{D})`` (64 cases) | Medlyn et al. (2011) | max abs. error 9.4 × 10⁻⁶ (the ``g_0`` floor) |
| ``A_n = g_c(c_a - c_i)`` = BL = pore flux (3456 cases) | CLM5 eq. 2.9.19 | 2.4 × 10⁻¹¹ relative; all converged |
| H₂O flux continuity | CLM5 eq. 2.9.20 | 1.1 × 10⁻¹¹ (roundoff) |
| ``c_i/c_a``, C3 PFTs × {Medlyn, Ball-Berry} | "≈ 0.7" for C3 | mean 0.719, range 0.638–0.811 |
| ``c_i/c_a``, C4 | below C3 | 0.366–0.568 |
| ``g_s`` at 800 vs 400 µmol mol⁻¹, C3 broadleaf | stomata close | −25 % (Ball-Berry), −32 % (Medlyn) |
| ``g_s`` at 567 vs 400 µmol mol⁻¹, C3 broadleaf | stomata close | −10 % / −14 % |
| Boundary layer, ``U = 1``, 20 °C | CLM5 eq. 2.5.122 | 2.079 mol m⁻² s⁻¹ (exact) |

With the CLM5 broadleaf slope ``g_1 = 4.45`` the Medlyn ratio is 0.78 at
1.5 kPa. That is above the textbook 0.7. The needleleaf ``g_1 = 2.35``
gives 0.65, and Ball-Berry gives 0.69.

Not validated here: measured leaf gas exchange, and the phase's
observational target of FLUXNET latent and sensible heat. The latter
needs the canopy and soil column of #21.

## References

- Ball, Woodrow & Berry (1987), in *Progress in Photosynthesis Research* IV, 221–224.
- Collatz et al. (1991), *Agric. For. Meteorol.* 54, 107–136.
- Medlyn et al. (2011), *Glob. Change Biol.* 17, 2134–2144.
- De Kauwe et al. (2015), *Geosci. Model Dev.* 8, 431–452.
- Oleson et al. (2013), CLM4.5 Technical Note, NCAR/TN-503+STR, §8.
- Lawrence et al. (2018), CLM5 Technical Note, §2.5.3, §2.9.3, §2.9.6, Table 2.9.1.
