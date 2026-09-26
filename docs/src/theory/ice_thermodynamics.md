# Lake and sea ice

Floating ice is a slab that loses heat upward by conduction, gains it
from the water below, and changes thickness wherever the two do not
balance. This page develops the model in `src/IceThermodynamics.jl`: the
energy-conserving multi-layer thermodynamics of Bitz & Lipscomb (1999,
"BL99"), which reduces exactly to pure lake ice when the salinity is
zero.

## 1. Sea ice is a mushy layer

Sea ice holds brine in pockets that stay at the liquidus,
``T_m = -μS`` (°C, ``μ`` = 0.054 K ppt⁻¹; Assur 1958). Warming the ice
melts the pocket walls, so the effective heat capacity is (Untersteiner
1961; Ono 1967)

```math
c(T,S) = c_0 + \frac{L\,μ S}{T^2},
```

which exceeds ``c_0`` thirty-fold at -0.5 °C for S = 3.2 ppt. Integrating
``ρc`` from ``T`` to the liquidus gives the *energy of melting*, the
energy that turns a unit volume into brine (BL99 eq. 5):

```math
q(S,T) = ρ_i\left[c_0 (T_m - T) + L\left(1 - \frac{T_m}{T}\right)\right].
```

For ``S = 0`` it is the familiar ``ρ_i[c_0(0 - T) + L]``; for ``S > 0``
it falls to zero at ``T = T_m``. At -2 °C and 3.2 ppt it is ``0.925\,ρ_i
L`` (BL99 quote 0.92). The conductivity is lowered by the brine,
``k = k_0 + βS/T`` (Untersteiner 1964, β = 0.1172 W m⁻¹).

BL99's point: a model that stores heat in brine pockets through ``c(T,S)``
but melts the surface with ``q = ρ_i L`` counts that heat twice; it
over-predicts the MU71 equilibrium thickness by 50–124 cm.

## 2. Enthalpy form and the discretisation

Each of the ``N`` equal ice layers carries its enthalpy ``E = -q``
(J m⁻³, zero for brine at the liquidus); the snow layer carries
``E_s = ρ_s(c_0 T - L)``. The heat equation

```math
\frac{∂E}{∂t} = \frac{∂}{∂z}\left(k \frac{∂T}{∂z}\right) + I_0 κ e^{-κ z}
```

(penetrating shortwave ``I_0 = i_0(1-α)SW↓`` on snow-free ice, Beer's law
with ``κ`` = 1.5 m⁻¹) is stepped backward-Euler. For given conductances
the new temperatures solve a tridiagonal system, nonlinear through
``E(T)``; Newton linearises ``E`` with ``ρc(T,S)`` and, when the surface is
not melting, the surface balance

```math
F(T_s) = (1-α)(1-i_0)SW↓ + εLW↓ - εσT_s^4 - H(T_s) - LE(T_s) = F_{c}(T_s, T_1)
```

with its derivative. When ``F`` at the melting point exceeds the
conductive flux, ``T_s`` is pinned there and the surplus melts the
surface. Layers driven past their liquidus are pinned at it.

The enthalpies are then updated **from the interface fluxes actually
used**, ``E_j ← E_j + Δt(F_{j+1/2} - F_{j-1/2} + A_j)/Δz_j``. Energy is
therefore conserved to round-off whatever the Newton tolerance; the
temperature is only an intermediate. Melting consumes exactly ``-E`` per
unit volume (snow first, then ice from the top; from the bottom when
the ocean flux ``F_w`` exceeds the basal conduction ``F_{cb}``); basal
accretion adds ice of enthalpy ``E(T_f, S_{new})`` at the rate

```math
\frac{dh}{dt} = \frac{F_{cb} - F_w}{q(T_f, S_{new})}.
```

Snow pushed below the waterline (``ρ_i h + ρ_s h_s > ρ_w h``) becomes
ice, keeping mass and enthalpy. Finally the layers are remapped
conservatively to ``N`` equal layers under a salinity profile fixed in
relative depth (BL99). The skin, snow and ice interfaces use half-cell
fluxes (``2k/Δz``) and harmonic-mean conductances.

## 3. Stefan's law and the Neumann solution

With the surface held ``ΔT`` below the freezing point and no ocean heat,
a linear (quasi-steady) profile gives Stefan's (1891) law

```math
h^2 = h_0^2 + \frac{2kΔT}{ρ_i L}\,t ,
```

the basis of degree-day lake-ice forecasts. Keeping the sensible heat of
the ice gives the exact Neumann solution ``h = 2λ\sqrt{αt}`` with
``λe^{λ^2}\mathrm{erf}\,λ = St/\sqrt{π}``, ``St = c_0ΔT/L``,
``λ ≈ \sqrt{St/2}(1 - St/6)``: for ΔT = 10 K the ice is 1.0 % thinner
than Stefan predicts.

| check | result |
| :-- | :-- |
| Stefan limit (``c_0 → 1`` J kg⁻¹ K⁻¹), 90 d, ΔT = 10 K | 1.0639 m vs 1.0635 m (0.035 %); ``h^2-h_0^2 ∝ t^{0.997}`` |
| Neumann, 61 d, ΔT = 10 K, N = 10 / 40 | +0.11 % / +0.05 % |

## 4. The MU71 benchmark

Maykut & Untersteiner (1971) forced a column with Fletcher's (1965)
climatological fluxes at ~80 °N, 2 W m⁻² from the ocean and a snowfall
schedule; BL99 (and Semtner 1976) use the same protocol. The forcing,
as distributed with the BL99 reference code, is in
`test/data/mu71_forcing.csv`.

**Against the BL99 reference code** (run in GNU Octave 11.1, same
configuration: N = 10, Δt = 12 h, and the code's own ``c_0`` = 2054
J kg⁻¹ K⁻¹): the first annual cycle agrees to a few millimetres
(mean/max/min 263.8/293.0/242.5 cm vs 264.1/293.6/242.8). Over decades
the two part: the reference settles at 265 cm, this model at ~250 cm.
The cause is one boundary stencil. With N = 10 the reference code
estimates the conductive flux at a snow surface by a quadratic through
the snow midpoint and the first ice-layer midpoint — across the
snow/ice interface, where ``k`` jumps from 0.31 to ~2 W m⁻¹ K⁻¹ and the
profile has a kink a quadratic cannot represent. Replacing that one
stencil by the half-cell flux used here brings the reference code to
248.5 cm after 20 years (ours 250.4 cm; all-first-order reference
250.0 cm). The unmodified reference also drifts towards our value as its
layers are refined (265.7, 259.1, 254.7 cm at N = 10, 30, 60, year 20),
while this model changes by 1.5 cm (250.4, 249.5, 249.0 cm).

The same holds in the published Table-1 configuration (Δt = 4 h, i₀
through snow after Bettge et al. 1996, ``T_f`` = −1.95 °C, 100 years):
the unmodified reference code gives 281.0 cm — BL99's published 281 —
and 262.6 cm with only the snow-surface stencil changed; this model
gives 264.2 cm.

**Against the published BL99 numbers** (Table 1 configuration; ``c_0``
here is the contract's 2108 J kg⁻¹ K⁻¹, the paper's 2110):

| case | BL99 (cm) | this model (cm) |
| :-- | --: | --: |
| vertically varying S, α_i = 0.63 | 281 | 264 |
| isosaline 3.2 ppt, α_i = 0.63 | 400 | 356 |
| isosaline 3.2 ppt, α_i = 0.60 | 259 | 237 |
| first-cycle amplitude, varying S | 42 | 44.5 |
| first-cycle amplitude, isosaline α_i = 0.60 | 41 | 45.7 |
| Δh for α_i − 0.01 (Table 2) | 27 thinner | 25.9 thinner |
| Δh for F_L + 1 W m⁻² (Table 2) | −48 | −44.9 |

The equilibria are 6–11 % thinner — the stencil effect above — while the
sensitivities, which the stencil hardly touches, agree within 7 %. The
isosaline case is less completely explained: the stencil-corrected
reference gives 368 cm (unmodified 418 cm) against our 356 cm.

No observations are used for validation; MU71 itself is a model
benchmark forced by climatology.

## 5. Coupling

`IceColumn` implements `SurfaceExchange.AbstractSurface`: skin
temperature, roughness lengths, albedo (dry snow 0.80, melting snow
0.75, bare ice 0.63, open water 0.06), emissivity, saturation over ice
and the latent heat of sublimation. `step_surface!` iterates the
Monin–Obukhov fluxes inside the Newton solve, so the returned
``R_n - H - LE - G`` closes to ~1e-12 W m⁻², and removes the sublimated
mass (snow first) with its enthalpy. Open water freezes over when it
loses heat at its freezing point; ice thinner than 0.1 mm melts into the
water. Rain is assumed to drain without exchanging heat.

## References

- Bitz, C. M. & Lipscomb, W. H. (1999). *J. Geophys. Res.*, 104(C7), 15669–15677.
- Maykut, G. A. & Untersteiner, N. (1971). *J. Geophys. Res.*, 76, 1550–1575.
- Semtner, A. J. (1976). *J. Phys. Oceanogr.*, 6, 379–389.
- Untersteiner, N. (1961, 1964). *Arch. Meteorol. Geophys. Bioklim.* A12; *J. Geophys. Res.*, 69, 4755–4766.
- Bettge, T. W. et al. (1996). The NCAR CSM sea ice model. NCAR TN-425+STR.
- Stefan, J. (1891). *Ann. Phys.*, 278, 269–286.
- Carslaw, H. S. & Jaeger, J. C. (1959). *Conduction of Heat in Solids*, §11.2.
