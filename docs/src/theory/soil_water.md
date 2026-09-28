# Soil water: Richards' equation in a CLM-style column

Phase 46 gives the land surface a soil: a column of layers below the
skin in which water infiltrates, redistributes, drains and is pulled up
by evaporation and roots. This page documents the water-transport part
implemented in `src/SoilColumn.jl` (issue #15); soil heat and freeze–thaw
(#16) and the water table (#17) extend the same column.

## 1. Layers

Depth ``z`` is positive downward from the soil surface. The default
column is CLM5's (Lawrence et al. 2019; CLM5.0 Technical Note,
Table 2.2.3): 20 hydrologically active layers whose thicknesses grow
from 2 cm at the surface to 1.14 m at the base,

```math
\Delta z_j =
\begin{cases}
0.02\,j & j \le 4\\
0.08 + 0.04\,(j-4) & 5 \le j \le 13\\
0.44 + 0.10\,(j-13) & j \ge 14
\end{cases}
```

reaching 8.6 m, with the nodes at the layer centres. The older
exponential CLM2–4.5 structure ``z_i = 0.025[\exp(0.5(i-0.5)) - 1]``
(10 layers to 3.43 m; Oleson et al. 2004) and a uniform grid for
benchmarks are also provided. Both CLM grids are tested against the
published tables.

## 2. Hydraulic functions

The matric potential ``\psi`` (m of water, negative when unsaturated)
and the volumetric water content ``\theta`` are linked by a retention
curve, and the hydraulic conductivity ``K`` falls steeply as the soil
dries. Three families are implemented behind `AbstractHydraulicModel`:

| model | retention | conductivity | source |
| :-- | :-- | :-- | :-- |
| `ClappHornberger` | ``\psi = \psi_s(\theta/\theta_s)^{-b}`` | ``K_s(\theta/\theta_s)^{2b+3}`` | Clapp & Hornberger (1978); Cosby et al. (1984) pedotransfer |
| `VanGenuchtenMualem` | ``S_e = [1+(\alpha|\psi|)^n]^{-m}`` | ``K_s S_e^{1/2}[1-(1-S_e^{1/m})^m]^2`` | van Genuchten (1980), Mualem (1976); Carsel & Parrish (1988) textures |
| `GardnerExponential` | ``\theta - \theta_r \propto e^{\alpha\psi}`` | ``K_s e^{\alpha\psi}`` | Gardner (1958) — the exact-benchmark soil |

Texture tables: `CLAPP_HORNBERGER_1978` (11 USDA classes) and
`CARSEL_PARRISH_1988` (12 classes), stored in the papers' units and
converted by `clapp_hornberger_texture` / `carsel_parrish_texture`.

**Air entry in van Genuchten soils.** For ``n < 2`` (loams, clays) the
Mualem conductivity has an infinite slope at saturation: a loam loses
40 % of ``K_s`` within 2 cm of saturation. Ippisch et al. (2006) show
this is non-physical, and in the solver it made the Picard iteration
chatter indefinitely once the surface layers reached saturation under
ponding (thousands of sub-steps per 5 minutes). Following Vogel et al.
(2001) the model takes an air-entry potential ``\psi_e``, below which
``S_e`` is rescaled by ``S_c = [1+(\alpha|\psi_e|)^n]^{-m}``;
`carsel_parrish_texture` uses ``\psi_e = -2`` cm by default
(``\psi_e = 0`` recovers the published curves). The retention curve
moves by at most ``(\theta_s-\theta_r)(1-S_c)``, at ``\psi_e``: under
0.6 % of the pore space for the loams and clays, but 1.3–2.3 % for the
coarse classes (sand, loamy sand, sandy loam, sandy clay loam) — a
deliberate departure from the published curves in the wettest 2 cm.

## 3. Temperature and ice

Hydraulic conductivity is ``K = k\rho g/\mu``; the intrinsic
permeability ``k`` does not depend on temperature, the viscosity of
water does — by a factor 1.8 between 0 and 20 °C. Layer conductivities
are multiplied by ``\mu(T_{ref})/\mu(T)`` with ``\mu(T)`` from the
correlation of Kestin et al. (1978) (within 0.075 % of the CRC
values at seven temperatures, 0–80 °C; the test tolerance is 0.3 %) and ``T_{ref} = 20`` °C, the laboratory temperature of the
texture tables. A soil at 5 °C therefore needs a wetter profile to carry
the same drainage flux, which the unit-gradient test verifies.

Pore ice reduces conductivity further: the column carries `θ_ice` and
`T_soil` and multiplies ``K`` by `ice_impedance_factor(imp, θ_ice, θ_s)`
of an `AbstractIceImpedance` (``1`` for `NoIceImpedance`; CLM's
``10^{-6\,θ_{ice}/θ_s}`` for `PowerLawIceImpedance`). With ice present the
retention curve applies to the *total* water
``θ_{liq} + (ρ_{ice}/ρ_w)θ_{ice}``, so freezing does not change ψ — see
[soil heat and freeze–thaw](soil_heat.md), where the temperatures and
ice contents come from.

## 4. Richards' equation and its discretisation

With sinks ``s`` (root uptake), in mixed form,

```math
\frac{\partial\theta}{\partial t}
 = \frac{\partial}{\partial z}\left[K(\psi)\left(\frac{\partial\psi}{\partial z} - 1\right)\right] - s ,
\qquad q = K\left(1 - \frac{\partial\psi}{\partial z}\right)\ \text{(downward flux)} .
```

**Space.** Finite volumes on the layers: node values ``\psi_i``, fluxes
at the interfaces
``q_{i+1/2} = K_{i+1/2}[1 - (\psi_{i+1}-\psi_i)/(z_{i+1}-z_i)]``. The
interface conductivity is by default the logarithmic mean
``K_{i+1/2} = (K_{i+1}-K_i)/\ln(K_{i+1}/K_i)``, which is exact when
``K`` varies exponentially with ``\psi`` between nodes. On the exact
Gardner benchmark (§6) it converges at second order, while the
arithmetic mean of Celia et al. (1990) is first order: the steep
gradients next to a prescribed-head boundary make its boundary flux
``O(\Delta z)`` wrong, and that error pollutes the whole profile.

**Time.** Backward Euler, linearised by the *modified Picard* iteration
of Celia et al. (1990): with ``m`` the iterate,

```math
\theta^{n+1,m+1} \approx \theta^{n+1,m} + C^m(\psi^{m+1}-\psi^m),\qquad C = d\theta/d\psi,
```

and ``K`` frozen at ``m``, each iterate is one tridiagonal solve. The
mixed form is what makes the scheme mass-conservative where the
``\psi``-form is not (Celia's point). Here conservation is made exact:
after convergence the new water content is formed from the converged
fluxes themselves,

```math
\theta_i^{n+1} = \theta_i^n + \frac{\Delta t}{\Delta z_i}\left(q_{i-1/2} - q_{i+1/2} - s_i\right),
```

so the change of storage equals the boundary fluxes to round-off, for
any tolerance; the tolerance only bounds the mismatch between
``\theta`` and ``\theta(\psi)`` (which the next step's residual then
corrects). Convergence is declared, as in HYDRUS, when water contents of
unsaturated layers change by less than `tol` (default ``10^{-7}``) and
potentials of saturated layers by less than `tol_ψ` (``10^{-4}`` m).
Saturated layers store ``\theta_s + S_s(\psi-\psi_e)`` with a small
elastic storage ``S_s = 10^{-5}`` m⁻¹, which keeps a fully saturated
column well posed.

**Stiffness.** Near saturation the capacity ``C`` collapses and the
problem becomes elliptic; the Picard iteration then needs short steps.
Each call of `step_soil_water!` is split into sub-steps that halve when
the iteration fails and grow by half after fast convergence; after 8
iterations the update is under-relaxed (``\omega = 1/2``).

## 5. Boundary conditions

**Top — `AtmosphericTop`.** The net supply ``P - E`` is imposed as a
flux. If absorbing it would require a surface potential above
``\psi_{pond} = 0`` the boundary switches to the head ``\psi = 0`` and the
excess becomes infiltration-excess runoff; if delivering the evaporative
demand would require a surface potential below ``\psi_{dry}`` (CLM's
``-10^5`` m) it switches to that head and evaporation becomes
soil-limited (stage-2 drying). `HeadTop(ψ)` prescribes the surface
potential directly.

**Bottom — `AbstractSoilBottomBoundary`.** Free drainage ``q = K_N``
(CLM's default), an impermeable base, or a prescribed head. A boundary
supplies the linearisation ``q = a + b\,\psi_N`` and may keep its own
state, updated by `finalize_bottom!` after each accepted sub-step — the
interface through which the [water table](water_table.md) (#17),
`UnconfinedAquifer`, is coupled.

**Coupling.** `step_soil_water!(col, forcing, fluxes, dt)` takes the
Era V `SurfaceForcing` (rain) and `SurfaceFluxes` (evaporation ``E``)
and converts kg m⁻² s⁻¹ with the contract's ``\rho_w``;
`moisture_availability(col)` is the Lee & Pielke (1992) ``\beta`` of the
top layer relative to field capacity (−33 kPa).

## 6. Validation

**Exact solution (Gardner soil).** For ``K = K_s e^{\alpha\psi}`` and
``\theta - \theta_r = (\theta_s-\theta_r)K/K_s``, Richards' equation is
linear in ``K`` (Srivastava & Yeh 1991; Tracy 2006). With ``\zeta`` the
height above the base of a column of length ``L``,

```math
c\,\partial_t K = \partial_\zeta^2 K + \alpha\,\partial_\zeta K ,\qquad c = \alpha(\theta_s-\theta_r)/K_s .
```

Start from ``K = K_r`` everywhere, hold the base at ``K_r`` and raise the
top to ``K_t`` at ``t = 0``. The steady part is
``K_r + A(1-e^{-\alpha\zeta})/(1-e^{-\alpha L})``, ``A = K_t - K_r``; the
substitution ``K = e^{-\alpha\zeta/2}u`` turns the transient into the
heat equation with decay ``\alpha^2/4``, and projecting the initial
deficit on ``\sin(k_n\zeta)``, ``k_n = n\pi/L``, gives

```math
K = K_r + A\frac{1-e^{-\alpha\zeta}}{1-e^{-\alpha L}}
  + e^{-\alpha\zeta/2}\sum_{n\ge1} \frac{2A k_n(-1)^n e^{\alpha L/2}}{L\lambda_n}\sin(k_n\zeta)\,e^{-\lambda_n t/c},
\qquad \lambda_n = k_n^2 + \frac{\alpha^2}{4}.
```

For a 1 m column (``\alpha = 2`` m⁻¹, ``K_s = 10^{-5}`` m s⁻¹, from
``\psi = -5`` m, surface held at 0):

| measure | result |
| :-- | :-- |
| max ``|\Delta\theta|`` at 12 h, log mean, 50 / 100 layers | 1.36e-4 / 3.86e-5 (order 1.8) |
| same, arithmetic mean | 7.64e-3 / 3.77e-3 (order 1.0) |
| max ``|\Delta\theta|`` at 3 h, 200 layers, Δt = 10 / 5 / 2.5 s | 8.5e-5 / 4.4e-5 / 2.4e-5 (first order, backward Euler) |

**Sorptivity.** With gravity off and the surface saturated, Boltzmann
similarity makes the absorption exactly ``I = S\sqrt t``. For the Gardner
soil (constant diffusivity) ``S = 2\Delta\theta\sqrt{D/\pi}``: the solver
gives 0.9993 and 0.9998 of it after 2 and 6 h. For the loam, the
numerical ``S`` is constant to 0.1 % between 1 and 6 h and 0.7 % above
Parlange's (1975) approximation, inside the rigorous bounds. (Parlange's
integral must run to the *boundary* potential, not the air-entry value:
the saturated segment ``\psi_e < \psi < 0`` still transmits water. The
first version of the test stopped at ``\psi_e`` and disagreed by 12 %.)

**Philip infiltration.** Vertical ponded infiltration into the loam
exceeds horizontal absorption by ``A t`` with ``A/K_s`` = 0.43–0.49 over
10 min–3 h, inside Philip's ``[1/3, 2/3]``; the ponded rate falls to
1.006 ``K_s`` after 5 days.

**Steady states and conservation.** A layered CLM5 column on bedrock
relaxes to hydrostatic equilibrium (total-head spread 3e-10 m); free
drainage under a steady input reaches the unit-gradient content
``\theta_s(q/K_sf_T)^{1/(2b+3)}`` to 1e-6. Over 30 days of mixed rain,
evaporation, dew and root extraction the worst per-step budget error is
1.3e-15 m, and the Celia et al. (1990) problem has a mass-balance ratio
of ``1 + 3\times10^{-15}`` at Δt = 1 h.

**Not validated against observations.** The phase target — soil
moisture profiles at FLUXNET sites — needs the vegetated land column
(#21) and the thermal model (#16); this issue is validated against the
exact, similarity and benchmark results above only.

## References

- Carsel, R. F. & Parrish, R. S. (1988). *Water Resour. Res.*, 24, 755–769.
- Celia, M. A., Bouloutas, E. T. & Zarba, R. L. (1990). *Water Resour. Res.*, 26, 1483–1496.
- Clapp, R. B. & Hornberger, G. M. (1978). *Water Resour. Res.*, 14, 601–604.
- Cosby, B. J. et al. (1984). *Water Resour. Res.*, 20, 682–690.
- Gardner, W. R. (1958). *Soil Sci.*, 85, 228–232.
- Ippisch, O., Vogel, H.-J. & Bastian, P. (2006). *Adv. Water Resour.*, 29, 1780–1789.
- Kestin, J., Sokolov, M. & Wakeham, W. A. (1978). *J. Phys. Chem. Ref. Data*, 7, 941–948.
- Lawrence, D. M. et al. (2019). *J. Adv. Model. Earth Syst.*, 11, 4245–4287.
- Lee, T. J. & Pielke, R. A. (1992). *J. Appl. Meteorol.*, 31, 480–484.
- Mualem, Y. (1976). *Water Resour. Res.*, 12, 513–522.
- Oleson, K. W. et al. (2004). NCAR/TN-461+STR.
- Parlange, J.-Y. (1975). *Soil Sci. Soc. Am. Proc.*, 39, 415–418.
- Philip, J. R. (1957). The theory of infiltration: 4. Sorptivity and algebraic infiltration equations. *Soil Sci.*, 84, 257–264.
- Srivastava, R. & Yeh, T.-C. J. (1991). *Water Resour. Res.*, 27, 753–762.
- Tracy, F. T. (2006). *Water Resour. Res.*, 42, W08503.
- van Genuchten, M. Th. (1980). *Soil Sci. Soc. Am. J.*, 44, 892–898.
- Vogel, T., van Genuchten, M. Th. & Císlerová, M. (2001). *Adv. Water Resour.*, 24, 133–144.
