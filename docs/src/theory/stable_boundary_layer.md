# Stable boundary layers: GABLS1 and the stratified channel

P55-f (#191) validates the stability-corrected closure of P55-e
(`StabilityCorrected`, [Smagorinsky-Lilly SGS → Stability
correction](sgs.md#Stability-correction)) on two stably stratified,
horizontally homogeneous flows. It adds no physics to `src/`: both cases
are assembled from the Era V and Era VI modules, and only the
wall-resolved, θ-walled variant of the P54-d channel builder is new
(`channel_case(bottom_bc = :noslip, θ_walls = …)`).

## The shared SBL driver

`examples/sbl/driver.jl` is the one stable-boundary-layer driver; P58-d
(#194) extends it for GABLS2/3/4 and CASES-99. A periodic Boussinesq LES
over a cooled surface, every piece delegated:

| piece | provided by |
| :-- | :-- |
| geostrophic pressure gradient ``(-fV_g, +fU_g)`` | `LargeScaleForcing.GeostrophicPressureGradient` (#171), `apply_forcing!` from `body_force!`, the core's Coriolis term |
| surface temperature | `SurfaceTemperatureSeries` (#171), `surface_temperature(series, t)` |
| surface heat flux ``w'θ'_0 = -u_*θ_*`` | `SurfaceExchange.solve_surface_layer` per column, first level at Δz/2 |
| surface stress | `WallModels.MOSTWallLaw(sf, z0m, L)` with the ``L`` of that solve, `apply_wall_stress!` in every RK3 substage (the law returns the solver's ``u_*`` bit for bit) |
| SGS closure | any eddy-viscosity closure in `StabilityCorrected` (`sgs_stability = :brown94` default, `:lilly`, `:none`), stress-form divergence (`sgs_stress = :symmetric`) so that ``ν_v`` reaches the momentum |
| top damping | the core's Rayleigh sponge, retargeted at ``(U_g, V_g)`` with its own `add_sponge!(…; f_ref)` profile, as the Berlin `:geostrophic` coupling does |

The only case-specific "physics" is the GABLS1 surface-layer functions,
``φ_m = 1 + 4.8ζ``, ``φ_h = 1 + 7.8ζ`` (`SBLLogLinear`), an implementer of
the SurfaceExchange stability contract, so the solver and the wall law
use it unchanged. The surface layer is solved once per step from the
start-of-step state at the mid-step surface temperature; the heat flux is
held over the step (the RK3 weights sum to one), so exactly
``⟨w'θ'_0⟩Δt`` enters the column.

## GABLS1

The first GEWEX stable-boundary-layer LES intercomparison (Beare et al.
2006; protocol as quoted by Basu & Porté-Agel 2006): a 400 m cube,
``U_g = 8`` m s⁻¹, ``f = 1.39 × 10^{-4}`` s⁻¹ (73° N), ``z_0 = 0.1`` m,
θ = 265 K to 100 m and +0.01 K m⁻¹ above, the surface cooled from 265 K
at 0.25 K h⁻¹ for 9 h, random θ perturbations of ±0.1 K below 50 m, a
Rayleigh layer above 300 m, statistics of hours 8–9. The BL height ``h``
is the height where the mean total stress falls to 5 % of its surface
value, divided by 0.95 (Kosović & Curry 2000).

The reference ranges (`examples/sbl/data/gabls1_reference.csv`) are those
Basu & Porté-Agel (2006, §4c) quote from the intercomparison: surface
momentum flux 0.06–0.08 m² s⁻² (``u_* = 0.245``–0.283 m s⁻¹) and buoyancy
flux −3.5 to −5.5 × 10⁻⁴ m² s⁻³ (``w'θ'_0 = -0.0094`` to −0.0148 K m s⁻¹,
``H ≈ -12`` to −19 W m⁻²), ``h ≈ 200`` m; their LASD runs give
``h = 205/185`` m and ``u_* = 0.283/0.276`` m s⁻¹ at 32³/64³. No source
available to this change quotes the ensemble's low-level-jet maximum, so
it is reported without a reference.

### Results

Reduced CI test (`test/era6_gabls1.jl`, 32³ at 12.5 m, 0–1.25 h,
statistics over 1.0–1.25 h):

| | `:brown94` | `:none` |
| :-- | :-- | :-- |
| θ budget, max over steps of ``|Δ∫⟨θ⟩dz - ∫⟨w'θ'_0⟩dt|`` / ``|∫⟨w'θ'_0⟩dt|`` | 1.7e-9 | 1.7e-9 |
| ``u_*`` (m s⁻¹), mean [min, max] | 0.272 [0.263, 0.285] | 0.270 [0.259, 0.281] |
| ``w'θ'_0`` (K m s⁻¹), mean (max) | −0.0049 (−0.0047) | −0.0050 (−0.0047) |
| ``h`` (m) | 178.7 | 178.2 |
| mean ``κ_v/κ_h`` below ``h`` | 0.63 | 1 |
| SGS share of the heat flux, lower half of the BL | 0.113 | 0.145 |
| time-stepping cost (s, loaded machine) | 74 | 52 |

Full runs (`examples/sbl/gabls1.jl`, hours 8–9):

| closure | grid | ``h`` (m) | ``u_*`` (m s⁻¹) | ``w'θ'_0`` (K m s⁻¹) | ``H`` (W m⁻²) | LLJ max (m s⁻¹) at z | ``L`` (m) |
| :-- | :-- | --: | --: | --: | --: | :-- | --: |
| Smagorinsky + Brown 1994 | 32³ | 222.5 | 0.304 | −0.0153 | −19.9 | 9.37 at 206 m | 119 |
| Smagorinsky + Lilly | 32³ | 213.8 | 0.300 | −0.0152 | −19.8 | 9.40 at 206 m | 115 |
| Smagorinsky, uncorrected | 32³ | 210.2 | 0.295 | −0.0155 | −20.3 | 9.29 at 194 m | 107 |
| reference range | | ≈ 200 (LASD 205) | 0.245–0.283 (LASD 0.283) | −0.0148 to −0.0094 | −19.3 to −12.3 | — | (LASD 114) |

| closure | grid | ``h`` (m) | ``u_*`` (m s⁻¹) | ``w'θ'_0`` (K m s⁻¹) | ``H`` (W m⁻²) | LLJ max (m s⁻¹) at z | ``L`` (m) | steps, wall time |
| :-- | :-- | --: | --: | --: | --: | :-- | --: | :-- |
| Smagorinsky + Brown 1994 | 64³ | 196.4 | 0.284 | −0.0134 | −17.5 | 9.49 at 191 m | 110 | 62 677, 6618 s |
| Smagorinsky + Lilly | 64³ | 195.9 | 0.283 | −0.0134 | −17.5 | 9.48 at 191 m | 109 | 62 285, 6361 s |
| Smagorinsky, uncorrected | 64³ | 191.1 | 0.281 | −0.0134 | −17.5 | 9.45 at 184 m | 106 | 61 914, 5467 s |
| reference range | | ≈ 200 (LASD 185) | 0.245–0.283 (LASD 0.276) | −0.0148 to −0.0094 | −19.3 to −12.3 | — | (LASD 114) | |

At 64³ (Δ = 6.25 m; single-thread processes, three in parallel on a
loaded machine, the corrected closure 21 % slower) every closure is
inside the ensemble's heat-flux range and within 2–5 % of ``h ≈ 200`` m;
``u_*`` is at the upper end of the range (0.281 inside; 0.283 and 0.284
on and 0.4 % above its upper limit 0.283), and the ``u_*``, ``h`` and ``L``
of the LASD reference run are matched within 3 %, 7 % and 7 %. Refining
from 32³ to 64³ lowers ``u_*`` by 5–7 %, ``|w'θ'_0|`` by 12–14 % and ``h``
by 8–12 %. The corrected and uncorrected profiles (figure
`examples/sbl/output/gabls1_N64.svg`) differ only near the BL top, where
the corrected run has a sharper capping inversion.

At 32³, by contrast, every closure is slightly outside the range on the strong side:
``u_*`` 4–7 % above its upper end and ``|w'θ'_0|`` 3–5 % above it.

**An expectation that proved wrong.** The issue expected the corrected
closure to give a lower BL than the uncorrected one. At CI resolution
the two agree within their scatter (178.7 ± 3.4 m and 178.2 ± 3.9 m over
1.0–1.25 h), and after 9 h the corrected run is *deeper* at both
resolutions (222.5 against 210.2 m at 32³, 196.4 against 191.1 m at
64³). The code does what it should — ``κ_v/κ_h = 0.63`` below ``h`` and
the SGS share of the heat flux falls from 0.145 to 0.113 — but at 12.5 m
the resolved eddies carry 85–90 % of the flux, and a weaker SGS sink
leaves them more energetic. The test asserts the correction's action
(``κ_v/κ_h < 0.8``, SGS share reduced by more than 10 %), not the ordering
of ``h``.

## The stably stratified channel

The wall-resolved channel of García-Villalba & del Álamo (2011, Phys.
Fluids 23, 045104; the issue's "JFM 680, 75" is this paper): isothermal
walls with the top hotter, a constant pressure gradient, ``Pr = 0.71``,
the friction Richardson number

```math
Ri_τ = \frac{g\,Δρ\,δ}{ρ_0 u_τ^2} = \frac{g\,(θ_{top} - θ_{bottom})\,δ}{θ_0\,u_τ^2}.
```

`channel_case(; Re_τ, bottom_bc = :noslip, θ_walls, Pr, θ₀, g_accel,
sgs_stress)` builds it on the P54-d harness: `NoSlipWall`s (P59-a) instead
of the wall model; θ an active Boussinesq scalar with diffusivity
``ν/Pr`` (the core diffuses θ with ν; the difference is added) and wall
conduction ``(ν/Pr)(θ_w - θ_1)/(Δz/2)`` — the Dirichlet ghost
``2θ_w - θ_1`` — into the wall cells, with mirrored θ halos so the SGS flux
through a wall face is zero (``κ_t → 0`` at a resolved wall);
`stratified_channel_walls(Ri_τ)` gives the walls and
`channel_richardson(case)` the inverse. Laminar checks: the conduction
flux is the laminar ``κΔθ/2δ`` (Nu = 1) to 1e-12, and steady Poiseuille
flow has the total stress ``u_τ^2(1 - z/δ)`` to 1e-6.

The DNS at ``Re_τ = 550`` is not affordable wall-resolved here (about
three times the points per direction of ``Re_τ = 180``), so the
validation uses the lowest Re_τ of the DNS, 180, where its bulk Reynolds
and Nusselt numbers are tabulated (Ahmed et al. 2021, Table I, columns
"GV11"; `examples/sbl/data/stratified_channel_gv11_re180.csv`):
``U_b^+ = 15.67, 16.91, 19.09`` and ``Nu = 6.03, 4.02, 2.80`` at
``Ri_τ = 0, 18, 60``. The closure is the plane-averaged dynamic
Smagorinsky model (Germano et al. 1991), which needs no wall damping;
static Smagorinsky without damping is not usable at a resolved wall here
(at 32³ and ``C_s = 0.1`` its SGS wall flux exceeds the molecular one and
the flow relaminarises: ``⟨τ_w⟩ = 0.32 u_τ^2``).

As in the DNS, the stratification is switched on in a developed neutral
channel (`stratified_channel_spinup`: θ passive, no buoyancy), and θ is
rescaled to the walls of each run.

Reduced CI test (`test/era6_stratified_channel.jl`, 24×16×32 on
2πδ × πδ × 2δ, Δx⁺ = 47, Δy⁺ = 35, Δz⁺ = 11.25; neutral spin-up 6 δ/u_τ,
then each run to 16 δ/u_τ, statistics over 10–16 δ/u_τ; 2.5 min for the
file on a loaded machine):

| | ``Ri_τ = 0`` (passive θ) | 18, none | 18, corrected | DNS 0 / 18 |
| :-- | --: | --: | --: | :-- |
| ``U_b^+`` | 14.80 | 15.74 | 15.58 | 15.67 / 16.91 |
| Nu | 7.05 | 4.79 | 4.89 | 6.03 / 4.02 |
| Nu(18)/Nu(0) | | 0.680 | 0.694 | 0.667 |
| ``⟨τ_w⟩`` (molecular, u_τ²) | 0.92 | 0.86 | 0.93 | |
| momentum budget residual (u_τ²) | 0.0019 | 0.0063 | 0.0023 | |

The momentum budget is ``τ(z) = τ(0) + ∫_0^z(∂⟨U⟩/∂t - G)\,dz'`` of the
time-averaged total stress (resolved + SGS + viscous), unsteadiness
included (``U_b`` still drifts by 0.1–0.4 over the window). The
stratification cuts the wall heat transfer to 0.68–0.69 of the neutral
value (DNS 0.667) and raises the bulk velocity by 6.4 % (uncorrected) and
5.3 % (corrected), against 7.9 % in the DNS. The coarse LES under-predicts
``U_b^+`` by 5.5 % and over-predicts Nu by 17 %.

The highest Ri_τ of the CI test is 18. At ``Ri_τ = 60`` this grid does not
hold a turbulent state after the switch-on: over 10–16 δ/u_τ the
uncorrected run relaminarises (Nu = 1.55, ``⟨τ_w⟩ = 0.45 u_τ^2``, ``U_b``
still rising by 3.3 u_τ) and the corrected one partly (Nu = 2.65,
``U_b`` +2.2 u_τ), so no statistics of that window mean anything; started
from noise instead, the same grid stays turbulent for 16 δ/u_τ (Nu = 3.20
both, ratio 0.45 against the DNS 0.464) but accelerates throughout
(``U_b`` +0.7 u_τ over the window). ``Ri_τ = 60`` is run at 32×32×48 for
longer in `examples/sbl/stratified_channel.jl`.

Example run (`examples/sbl/stratified_channel.jl`, 32×32×48 on
2πδ × πδ × 2δ, Δx⁺ = 35, Δy⁺ = 18, Δz⁺ = 7.5; neutral spin-up 10 δ/u_τ,
each run to 40 δ/u_τ, statistics over 20–40 δ/u_τ; six single-thread
processes of 10–15 min each on a loaded machine):

| ``Ri_τ`` | closure | ``U_b^+`` | DNS | Nu | DNS | ``⟨τ_w⟩`` (u_τ²) | budget (u_τ²) | ``ΔU_b`` over the window |
| --: | :-- | --: | --: | --: | --: | --: | --: | --: |
| 0 | none | 14.98 | 15.67 | 6.96 | 6.03 | 0.946 | 0.0047 | +0.02 |
| 0 | corrected | 14.88 | 15.67 | 6.87 | 6.03 | 0.995 | 0.0027 | −0.15 |
| 18 | none | 16.67 | 16.91 | 4.40 | 4.02 | 0.936 | 0.0079 | +0.23 |
| 18 | corrected | 17.36 | 16.91 | 4.57 | 4.02 | 1.025 | 0.0139 | −0.78 |
| 60 | none | 25.85 | 19.09 | 1.23 | 2.80 | 0.532 | 0.0079 | +9.37 |
| 60 | corrected | 25.85 | 19.09 | 1.23 | 2.80 | 0.532 | 0.0079 | +9.37 |

At ``Ri_τ = 0`` and 18 the finer grid is closer to the DNS than the CI
grid: ``U_b^+`` within 4.4 % (neutral) and 1.4 %/2.7 % (``Ri_τ = 18``), Nu
over-predicted by 15 % and 9 %/14 %; Nu(18)/Nu(0) = 0.63/0.67 against the
DNS 0.667. **A miss:** at ``Ri_τ = 60`` both closures relaminarise after
the switch-on (Nu = 1.23, ``⟨τ_w⟩ = 0.53 u_τ^2``, the flow still
accelerating by 9.4 u_τ over the window), where the DNS stays turbulent
(Nu = 2.80). The two runs are nearly identical because the SGS
viscosity vanishes in the laminarised flow. We attribute the miss to
resolution, not to a code error: neither grid (Δx⁺ = 47 or 35) holds the
near-wall cycle against this stratification; matching the DNS at
``Ri_τ = 60`` needs a finer, DNS-like wall resolution not affordable here. This
is the regime P59-e (#195) studies.

**An expectation that proved wrong.** The issue expected the corrected
closure's mean profile to be closer to the DNS than the uncorrected one.
At CI resolution it is not: at ``Ri_τ = 18`` the corrected run gives
``U_b^+ = 15.58`` and Nu = 4.89 against 15.74 and 4.79 uncorrected (DNS
16.91 and 4.02), the uncorrected one closer by 1–2 % in both. With a
dynamic base the correction applies no ``f_m`` to the momentum (the dynamic
coefficient already sees the stratification), only
``Pr_t(Ri) = Pr_t/(1 - 1.2Ri)`` below ``Ri_c`` (Brown et al. 1994) and the
anisotropic length scales; the test reports the numbers and does not assert
the ordering.

## Berlin

No new physics, so no new coupling: this validates the P55-e closure that
every Berlin `:sgs` provider runs by default (`sgs_stability = :brown94`).
For comparison with GABLS1: the mean ``κ_v/κ_h`` (over cells with ``κ_h > 0``, the
same statistic as the Berlin `:sgs` report) is 0.63 below ``h`` in the
reduced GABLS1 run (32³, 1.0–1.25 h). In the Alexanderplatz flagship
(06:00 → 03:00, measured by P55-e on its branch) the night mean is
0.016, with 91.3 % of the fluid cells at ``Ri ≥ Ri_c`` at 03:00: the
urban night is far more strongly corrected than the GABLS1 SBL, whose
turbulence the 0.25 K h⁻¹ cooling keeps continuous. A cold start at
18:00 on this branch (`CaseConfig(:alexanderplatz; start_h = 18)`, the
default `:brown94`) gives 0.72, 0.77, 0.97 and 1.02 at 21:00, 21:30,
22:00 and 22:30 with 31–37 % supercritical cells: the evening-initialised column has not
built the stable night layer, and its convectively enhanced cells
(``κ_v > κ_h`` where ``Ri < 0``) enter the same mean, so the spun-up
flagship value is the comparable one.

## References

  * Ahmed, M. A., Bae, H. J., Thompson, A. F. & McKeon, B. J. (2021).
    Resolvent analysis of stratification effects on wall-bounded shear
    flows. *Phys. Rev. Fluids*, 6, 084804 (arXiv:2101.11771).
  * Basu, S. & Porté-Agel, F. (2006). Large-eddy simulation of stably
    stratified atmospheric boundary layer turbulence: a scale-dependent
    dynamic modeling approach. *J. Atmos. Sci.*, 63, 2074-2091.
  * Beare, R. J. et al. (2006). An intercomparison of large-eddy
    simulations of the stable boundary layer. *Boundary-Layer Meteorol.*,
    118, 247-272.
  * Brown, A. R., Derbyshire, S. H. & Mason, P. J. (1994). Large-eddy
    simulation of stable atmospheric boundary layers with a revised
    stochastic subgrid model. *Q. J. R. Meteorol. Soc.*, 120, 1485-1512.
  * García-Villalba, M. & del Álamo, J. C. (2011). Turbulence modification
    by stable stratification in channel flow. *Phys. Fluids*, 23, 045104.
  * Germano, M., Piomelli, U., Moin, P. & Cabot, W. H. (1991). A dynamic
    subgrid-scale eddy viscosity model. *Phys. Fluids A*, 3, 1760-1765.
  * Kosović, B. & Curry, J. A. (2000). A large eddy simulation study of a
    quasi-steady, stably stratified atmospheric boundary layer. *J. Atmos.
    Sci.*, 57, 1052-1068.
