# The Berlin case: turbulence and SGS

## Turbulence diagnostics (#170)

`turbulence_diag` (`components/urban/turbulence_diag.jl`, addon, on by default)
is the Berlin coupling of the stability contract
`TurbulenceStability` (see [the theory page](../theory/turbulence_stability.md)).
At the output cadence it evaluates, on copies of `u, v, w, θ, q_v` whose
halos it fills with the core's boundary conditions (the model state is
not touched):

* the pointwise ``N^2`` of ``θ_v = θ(1 + 0.61 q_v)``
  (`buoyancy_frequency_squared!`, the z-interpolation of the core's
  buoyancy), ``|S|^2`` (`strain_rate_squared!`, the Smagorinsky stencil)
  and ``Ri_g`` (`gradient_richardson!`);
* per level, over the fluid cells: the mean ``N^2`` (`N2_mean`), the
  Richardson number of the means ``⟨N^2⟩/⟨|S|^2⟩`` (`Ri_g_mean`; the mean
  of the pointwise ratio is dominated by shear-free cells, where the
  ``S^2`` floor makes it huge) and the mean ``f_m(Ri_g)`` (`f_m_mean`).
  These are written as `(z, time)` profiles (`OutputField` dims `:z`)
  and summarised in the log line.

Option `stability`: `:brown` (`BrownEtAl1994`, the LEM / MONC functions,
default), `:lilly`, `:short_tail`, `:long_tail`, `:most`
(`MOSTConsistent` with the surface exchange's BDBH functions; the smoke
variant). `budget!` is a no-op: the coupling changes no state. P59-c
(#180) extends this addon with the transition diagnostics (below).

**Measured** (`:smoke`, after its 12 steps, noon, capping inversion at
60 m, `Γ = 3 K km⁻¹` above). In the first two levels above the
inversion (z = 65.6 m and 71.9 m) `Ri_g` of the means is 0.273 and
0.196 and the mean `f_m` (Brown et al.) 0.049 and 0.0057; with
`stability = :most` the same cells give 0.101 and 0.069 (no critical
Richardson number). Below the inversion θ is uniform, and the vapour
lapse of the initial state (``q_v ∝ e^{-z/800\,{\rm m}}``) makes
``θ_v`` decrease with height: ``N^2 ≈ -7 × 10^{-5}`` s⁻² at step 0, and
−2 to −16 × 10⁻⁵ s⁻² after 12 steps of noon heating, so `Ri_g` < 0 and
`f_m` = 2.5–10. The same vapour lapse removes about 2/3 of the dry
``gΓ/θ_0 = 1.0 × 10^{-4}`` s⁻² above the inversion (3.3 × 10⁻⁵ at step 0).
Cost per evaluation (compiled): 0.15 ms on the 16³ smoke grid
(`:most`: 0.89 ms, a root find per cell), 3.6 ms on the 64 × 64 × 32
flagship grid, once per output record (every 1800 s).

### Transition diagnostics (#180)

P59-c extends the same addon (there is no second diagnostic addon;
`modules_used = (:TurbulenceStability, :Transition)`) with the
laminar/turbulent diagnostics of `Aeolus.Transition` (see
[the theory page](../theory/transition.md)), written as `(z, time)`
profiles at the output cadence:

* `gamma_turb` — the turbulent fraction ``γ(z)`` of the fluid cells,
  `turbulent_indicator!` with ``|ω'|/⟨|ω|⟩ >`` `gamma_threshold`
  (option, default 0.2) and the plane normalisation
  (`gamma_normalization = :plane`, or `:volume`); solid cells are never
  turbulent;
* `Ri_supercritical_fraction` — the fraction of fluid cells with
  ``Ri_g > 1/4`` (`supercritical_fraction` of the `gradient_richardson!`
  field above: Transition does not define a Richardson number);
* `Re_grid` — ``|U_h| Δ/(ν + ⟨ν_t⟩)`` (`grid_reynolds_number`) with the
  plane-mean horizontal wind speed, ``Δ = (Δx Δy Δz)^{1/3}`` and `cfg.ν`.

The log line adds canopy (levels with buildings) and free-stream (above
the highest roof + 1, below the sponge) means
(`transition_summary`). `smoke_check` adds ``0 ≤ γ ≤ 1``, no turbulent
solid cell, the supercritical fraction in [0, 1] and a finite positive
grid Reynolds number.

**Measured** (`:smoke`, after 12 steps). Summer noon: ``γ`` = 0.74–0.99
on the nine canopy levels (mean 0.871), 0.02 at z = 59 m and 0 above the
60 m inversion, where the initial state is still laminar (free-stream
mean 0.438 because it includes the 53 m and 59 m levels); the
supercritical fraction is 0 in the canopy and 0.61 / 0.10 in the two
levels above the inversion (domain mean 0.044). Winter (`:winter`
initial state) at 06:00: domain-mean supercritical fraction 0.244, with
0.88–1.0 in the four levels from 53 to 72 m — drier air, so the vapour
lapse removes less of the inversion's ``N^2``. Grid Reynolds number at
``ν = 2`` m² s⁻¹: 4.0–5.7 on the smoke grid (Δ = 15.6 m × 15.6 m × 6.25 m);
on the 64 × 64 × 32 flagship grid after 60 steps it peaks at 5.7 (2.4 at
the ground, 0.14 at the top) — below the "about 8" estimated in the
issue. Cost per evaluation of the whole addon: 0.28–0.55 ms on the smoke
grid and 7.0 ms on the flagship grid (3.6 ms before #180), once per
output record.

### Inflow fetch (#266) and the inflow generator (#268)

N-inflow-d measures the development length behind a synthetic inflow on
the periodic half channel (the harness `validation/fixtures/inflow_fetch.jl`, theory:
[Development length behind the inflow](../theory/transition.md#Development-length-behind-the-inflow-(N-inflow-d))).
Its diagnostics (`FetchStatistics`, `development_length`) are not coupled;
the generator itself is, on the nested child's lateral faces: the addon
`:inflow_turbulence` (N-inflow-e, #268, [nesting page](berlin_nesting.md#Synthetic-inflow-turbulence-(#268))).
The Berlin child's fetch, day and night, is measured with the 24 h
validation (#270); until then the analysis buffer of the observation
diagnostics is an estimate (140 m).

### Flat-plate transition (#185): stated exception

The spatially developing boundary layer of P59-d (fringe region,
`FreeStreamBoundary` top, Blasius inflow, `FreeStreamDisturbance`) is not
coupled: the Berlin domain is laterally periodic with geostrophic forcing
and has no inflow direction, so a fringe and an inflow generator have no
place in it. Inflow boundaries arrived with nesting (#265), and the
inflow generator is coupled there (#268, above). Transition stays coupled
through `turbulence_diag` above. The
one shared piece that does run in every Berlin step is the damping
profile: the Rayleigh sponge (`sponge_α_max` = 0.05 s⁻¹ from 75 m in
`:smoke`) now takes its rate from `Operators.DampingProfile`, the
implementation the fringe also uses. The smoke run's final state is
bit-identical with the verbatim Phase-7 kernel reinstalled; the sponge's
column kinetic-energy sink ``∫α|\mathbf u_h|^2dz`` after the 12 smoke
steps is 0.139 m³ s⁻³. No Berlin coupling was switched off.

## Prognostic SGS TKE (#182)

`sgs_tke` (`components/urban/sgs_tke.jl`) is the `:tke` provider of the `:sgs`
slot and its default: `PrognosticSGS.TKEClosure`, the Deardorff (1980) /
Moeng (1984) closure (see [the theory page](../theory/sgs_tke.md)). It is
PALM's urban-LES default and the only closure here whose mixing responds
prognostically to stratification (its ``0.76\sqrt e/N`` length scale; no
stability function is applied on top of it).

Not wrapping it in `StabilityCorrected` is by design. `:tke` has none of
that wrapper's ``κ_h ≠ κ_v`` anisotropy (one ``Δ``, as in PALM): at
``dx/dz = 2.5`` it mixes 3.39× more vertically and 1.84× less
horizontally than `StabilityCorrected` Smagorinsky of equal ``C_s``, and
no `:isentropic` tensor is offered
([details](../theory/sgs_tke.md#The-model)). Its ``N^2`` (and the
`StabilityCorrected` providers' and `turbulence_diag`'s) is that of cloudy
air: the `ql` tracer's loading, and the saturated ``N^2`` in fog (#239).

* The closure is the `sgs` of every dynamics step, so the cores evaluate
  ``ν_t = c_k l\sqrt e``, ``κ_t = (1 + 2l/Δ)ν_t`` from the committed ``e``
  and commit the sources (shear production, the exact dual of the SGS
  momentum operator; buoyancy production with the ``θ_v`` of the `qv`
  tracer; the implicit sinks) once per accepted step; `step_adaptive!`
  rolls ``e`` back with the state.
* ``e`` is the tracer `:e` (`advect = true`, `positive = true`, m² s⁻²,
  written to the NetCDF file). The `:transport` stage advects it with the
  case's scheme and limiter and diffuses it with ``2ν_t`` (its
  `diffusivity`), mask-aware; the closure runs with `transport = false`.
* ``l`` is capped at ``κ`` × `IBM.wall_distance` of the building mask and
  the floor (option `wall_cap = true`), ``c_ε`` gets Moeng's 3.9 above the
  floor and the roofs (`IBM.first_level_cells`), ``e = 0`` inside the
  buildings.
* The first air cell of each column (`bus.k_air`) holds
  ``e_s = u_*^2/\sqrt{c_k c_ε}`` with the friction velocity of the MOST
  surface exchange (`bus.u_star`, option `surface = true`).
* `init!` seeds ``e`` with the local equilibrium of the initial flow, at
  least `e_min = 1e-4` m² s⁻² (``e = 0`` is a fixed point).
* `budget!` books nothing: ``e`` carries no surface energy or water (its
  dissipation heating is neglected, as with every closure here).
* `smoke_check`: ``e ≥ 0``, ``ν_t ≥ 0``, ``e = 0`` in the buildings,
  ``κ_t ≥ ν_t``, and (Boussinesq core) ``Σ P_s V`` equal to the SGS sink of
  `kinetic_energy_budget(sim; sgs = closure)` to 1e-8. The smoke variant
  runs it without the wall cap and the surface value.

The default changed from `:smagorinsky` to `:tke` because the coverage
ratchet then required a new src module to run by default and no frozen
list could grow (the coverage lists were removed on 2026-10-06, #320); whether it stays the default is decided at review (#46).
Berlin tests that pinned the Smagorinsky default now select
`:smagorinsky` explicitly where they test its bit-for-bit equivalences
(MLSGS, the commit counter, the stale-halo demonstration).

**Measured** (summer flagship, 64 × 64 × 32, adaptive step, 30 min from
12:00 and from 02:00 local, `:tke` vs `:smagorinsky` with `cfg.C_s = 0.1`;
"street canopy" is the fluid cells of levels 1–3, 0–18.75 m):

| | noon `:tke` | noon `:smagorinsky` | 02:00 `:tke` | 02:00 `:smagorinsky` |
| :-- | --: | --: | --: | --: |
| street-canopy mean ``e`` (m² s⁻²) | 0.0110 | — | 0.0011 | — |
| street-canopy mean ``ν_t`` (m² s⁻¹) | 0.0275 | 0.0326 | 0.0086 | 0.0174 |
| domain mean ``ν_t`` (m² s⁻¹) | 0.0756 | 0.0220 | 0.0412 | 0.0118 |
| whole step (ms) | 161.6 | 131.9 | 131.9 | 116.1 |

At night the street-canopy ``ν_t`` is 51 % below Smagorinsky's (16 % at
noon) and the canopy ``e`` is a tenth of its noon value: the stability
length and the wall cap act where the static closure only sees the
strain. Higher up the TKE closure mixes more: its neutral equilibrium is
Smagorinsky with ``C_s = 0.194``, against the case's ``C_s = 0.1``
(``(0.194/0.1)^2 = 3.8``). No clip was booked. Cost per call on this grid:
`advance_sgs!` 7.2–7.4 ms, the TKE coefficients 0.9 ms (Smagorinsky
1.0 ms); the `:e` tracer adds 8–11 ms to `scalar_advection` (80.1 vs
69.1, 71.8 vs 63.7 ms/step); whole steps +30 ms (+23 %) at noon and
+16 ms (+14 %) at night (loaded machine, sequential runs).

## Differential Reynolds stress (#188)

`sgs_reynolds_stress` (`components/urban/sgs_reynolds_stress.jl`) is the
`:reynolds_stress` provider of the `:sgs` slot. It is an option, a genuine
alternative to the default `:tke`, and it uses the same module
(`PrognosticSGS`), so no coverage list changes. The closure is
`ReynoldsStressClosure` (see the Reynolds-stress section of
[the theory page](../theory/sgs_tke.md)).

* The closure is the `sgs` of every dynamics step. The cores take three
  things from the committed stresses: ``ν_t``, the Daly–Harlow tensor
  ``K = c_θτ`` (θ goes through the P55-c tensor operator), and the
  momentum ``−∂_jτ^{dev}_{ij}`` through the P55-b stress operator.
  `advance_sgs!` is the source stage, run once per accepted step: implicit
  sources (sub-stepped for growing modes), exact Coriolis, the
  realizability projection and the booking. `step_adaptive!` rolls it back
  with the state. A separate case stage could not be rolled back.
* The six stresses are the tracers `:tau11 … :tau23` (m² s⁻², written to
  the NetCDF file). The `:transport` stage advects them, with the
  diagonals `positive` and therefore flux-limited, and diffuses them with
  ``2ν_t`` (their `diffusivity`), mask-aware. The closure runs with
  `transport = false`.
* The wall cap ``κ d_{wall}`` (option `wall_cap`) and Moeng's 3.9 come from
  the TKE closure's functions. ``τ = 0`` in the buildings, and an edge that
  touches a building or the ground carries no stress.
* The first air cell of each column (`bus.k_air`, option `surface`) holds
  the LRR-IP constant-stress layer:
  * ``τ_{13}, τ_{23} = −u_*^2(\cos α, \sin α)`` against the cell's wind,
    with `bus.u_star` from the MOST exchange;
  * ``e_s = u_*^2/0.3395``.
* `budget!` books nothing, because the stresses carry no surface energy or
  water.
* **Incompatible** with `timestepper = :imex`: `step_imex!` represents the
  SGS momentum by an implicit eddy viscosity, and it refuses the closure
  (`Operators.sgs_explicit_stress`).
* **Stated exception.** The other tracers (qv, ql, the species, the
  pollutant) are diffused by the `:transport` stage with the isotropic
  ``κ_t = \operatorname{tr}K/3`` (`sim.κt`), not with the full tensor.
* `smoke_check`:
  * ``τ`` is positive semi-definite in every cell and ``τ = 0`` in the
    buildings;
  * the realizability corrections are below 1e-3 of the SGS energy;
  * ``Σ e V > 0`` and ``ν_t ≥ 0``;
  * the closure commits once per accepted step;
  * on the Boussinesq core, ``Σ P V`` equals the SGS sink of
    `kinetic_energy_budget(sim; sgs = closure)` to 1e-6.

  The smoke variant runs the closure without the wall cap and without the
  surface value. That variant caught a code bug: a single backward-Euler
  source step inverted a growing mode at a street corner, and the
  projection then added more SGS energy than the domain held. The fix is
  the sub-stepping.

**Measured.** Summer flagship, 64 × 64 × 32, adaptive step, 30 min from
12:00 and from 02:00 local time. "Canopy" means the fluid cells of the
levels that are more than 5 % solid, as for `:tke`.

| | noon `:reynolds_stress` | noon `:tke` | 02:00 `:reynolds_stress` | 02:00 `:tke` |
| :-- | --: | --: | --: | --: |
| canopy mean ``e`` (m² s⁻²) | 0.0133 | 0.0099 | 0.0025 | 0.0013 |
| canopy anisotropy ``II = b_{ij}b_{ji}`` | 0.068 | — | 0.066 | — |
| canopy ``τ_{33}/τ_{11}`` | 1.35 | — | 1.35 | — |
| canopy mean ``ν_t`` (m² s⁻¹) | 0.043 | 0.034 | 0.017 | 0.012 |
| domain mean ``ν_t`` (m² s⁻¹) | 0.096 | 0.074 | 0.056 | 0.040 |
| realizability corrections (30 min) | 203 383 | — | 502 282 | — |
| `step!` (ms/step) | 141.5 | 69.7 | 118.0 | 35.8 |
| `scalar_advection` (ms/step) | 196.7 | 140.9 | 171.1 | 94.6 |
| whole step (ms) | 462.4 | 344.1 | 388.8 | 193.4 |

The canopy stresses are anisotropic, with ``II ≈ 0.07``; in the neutral
shear equilibrium of the LRR-IP model ``II = 0.091``. The vertical
component dominates: ``τ_{33}/τ_{11} = 1.35`` at noon, where the canopy is
convective and buoyancy produces ``τ_{33}``.

The 02:00 value is the same, and that is physical in this case, not a
defect. The canopy at night is still unstably stratified: the city
surfaces stay warmer than the air, and a 5-minute diagnostic run from
02:00 has θ falling by 0.22 K over the lowest 75 m. Above the roofs, where
the night air is stable, ``τ_{33}/τ_{11} = 0.43`` (level 24).

The realizability corrections come from the component-wise transport. In
the last step, 127 cells reached the source update already unrealizable
at noon and 126 were corrected; at night the numbers were 632 and 616, or
0.1–0.5 % of the 131 072 cells. Over 30 min the corrections added
2.1e3 m⁵ s⁻² at noon and 1.5e3 m⁵ s⁻² at night. That is 0.09 % and
0.16 % of the final SGS energy ``Σ e V`` (2.38e6 and 8.85e5 m⁵ s⁻²).

The closure is a cost option. Its `advance_sgs!` adds 72–82 ms per step
to `step!`, from a 6×6 solve in every cell. The five extra tracers add
56–76 ms to `scalar_advection`. The whole step is 1.3× the `:tke` cost at
noon and 2.0× at night. These were sequential runs on a heavily loaded
machine, so the ratios are rough.

### A-priori metrics, outputs and a 1 h comparison (P56-e, #192)

No new physics. Both prognostic providers now add a-priori metrics to
their diagnostics (`apriori.*`, #338; a report line before). It comes from `sgs_test_filter_metrics` in
`components/urban/sgs.jl`, which compares the closure's subfilter stress over the
canopy fluid cells with the resolved (Germano) stress at the test-filter
scale. That stress is ``L_{ij} = \widehat{\bar u_i\bar u_j} − \hat{\bar u}_i\hat{\bar u}_j``,
computed by `subfilter_stress!` of the LES velocity under the 3-cell
masked box filter (P56-c). The line reports:

* ``k_{test} = ½⟨L_{kk}⟩``;
* the ratio ``e/k_{test}``;
* the off-diagonal correlation ``ρ(τ, L)``;
* the median II of ``L`` and of the model stress;
* the fraction of cells whose model stress lies outside the Lumley
  triangle.

For `:tke` the model stress is the implied ``\tfrac23 e δ − 2ν_t\bar S``;
for `:reynolds_stress` it is ``τ``.

There are new `output_fields`. `:tke` writes `II_sgs`; its ``e`` is
already the tracer `e`. `:reynolds_stress` writes `e` (``½ τ_{kk}``) and
`II_sgs`. The smoke checks now also require ``k_{test} > 0``, a finite
ratio, TKE II ≥ 0, and RSM II in ``[0, 2/3]``.

**Measured.** 1 h runs of the summer flagship (64 × 64 × 32, adaptive
step) from 02:00 and from 12:00. The four runs ran at the same time on a
shared machine, so the costs compare with each other.

| | 02:00 `:tke` | 02:00 `:reynolds_stress` | 12:00 `:tke` | 12:00 `:reynolds_stress` |
| :-- | --: | --: | --: | --: |
| canopy mean ``e`` (m² s⁻²) | 0.001 | 0.0014 | 0.012 | 0.016 |
| canopy mean ``ν_t`` (m² s⁻¹) | 0.008 | 0.011 | 0.036 | — |
| ``k_{test}`` (m² s⁻²) | 0.0043 | 0.0042 | 0.0203 | 0.0202 |
| ``e/k_{test}`` | 0.18 | 0.33 | 0.58 | 0.78 |
| ``ρ(τ, L)`` | 0.07 | 0.25 | 0.21 | 0.13 |
| median II: ``L`` / model | 0.294 / 0.008 | 0.295 / 0.079 | 0.252 / 0.006 | 0.255 / 0.077 |
| model cells outside the Lumley triangle | 10 % | 0 % | 2 % | 0 % |
| realizability corrections (1 h) | — | 917 442 | — | 361 796 |
| steps | 1397 | 1379 | 2097 | 2094 |
| wall time (s) | 261 | 510 | 382 | 748 |
| `step!` (ms/step) | 33.1 | 112.5 | 32.4 | 108.9 |
| `scalar_advection` (ms/step) | 89.3 | 157.5 | 89.2 | 159.0 |

The canopy ``ν_t`` of the noon Reynolds-stress run was not recorded; the
night value is the one the issue asks for. The `:tke` values are the
report's three-decimal ones.

* Under both closures the resolved stress at the test filter is far more
  anisotropic (median II ≈ 0.25–0.30) than either closure's stress. The
  Reynolds-stress closure is ten times closer (0.08 against 0.006–0.008).
  In 2–10 % of the canopy cells the TKE closure's implied stress is not
  realizable.
* At night the Reynolds-stress stress correlates better with ``L``
  (0.25 against 0.07). At noon TKE's does (0.21 against 0.13).
* The Reynolds-stress closure costs 2.0× the whole step. It also makes
  ~10⁶ realizability corrections per hour at night.

**Recommendation for the maintainer.** Keep `:tke` as the default. The
Reynolds-stress closure wins clearly only where its physics is
structural. It reproduces the stress anisotropy in stratified decay
(``⟨τ_{33}⟩/⟨τ_{11}⟩`` 0.76 against DNS 0.70; TKE 1.00) and has the
smallest KE error under rotation. It does not win on resolved KE in
isotropic decay, and its heat flux is not closer to the DNS (see the
[validation](../theory/sgs_tke.md)). At 2× the cost that is not a clear
win, so this PR does not change the default.

### Relaminarisation of the stable layer (#195)

P59-e extends the same addon once more (no new module, no list change):
`relam_fraction` — per level, and `relam_frac` over the whole stable
layer, the fraction of the fluid cells with ``Ri_g > 1/4`` that are
relaminarised, i.e. γ-laminar **and** carry ``ν_t/ν < 10^{-2}``
(`Transition.relaminarised_fraction`; ``ν_t`` the larger of the active
closure's ``ν_h`` and ``ν_v``, `Operators.sgs_fields`; ``ν`` = `cfg.ν`).
The laminar mask uses the volume normalisation (option
`relam_normalization = :volume`): a quiet, shear-free free stream is
laminar against the domain-mean ``|ω|``, while the plane-normalised
indicator, being scale-free, marks its weak disturbances turbulent. The
log line adds the percentage and the number of stable cells. For a
laminar-consistent closure (`Operators.laminar_consistent`: `:sigma`)
`smoke_check` asserts that ``ν_t ≡ 0`` (bitwise) in the parallel flow of
the current plane-mean wind on the Berlin mesh (a `deepcopy` of the
closure on a scratch core, so the case's closure is untouched). The
issue asked for ``ν_t = 0`` "in laminar cells" of the running flow; the
γ-laminar cells of the smoke run are not exactly laminar (weakly 3-D),
so σ's ``ν_t`` there is small but not zero (12 steps: max 0.117 ν, mean
0.007 ν over the 251 plane-γ laminar cells, the same as Smagorinsky's
0.086 ν / 0.010 ν) — the exact statement is the parallel-flow one. AMD
and Vreman (`:amd`, `:vreman`, P55-d) do not declare
`laminar_consistent` yet; their own `smoke_check` asserts the
laminar-reference switch-off.

**Measured** — `:winter` from 05:00 to 06:00 (1800 steps of 2 s, before
the 08:15 sunrise), 64 × 64 × 32: the lowest 50 m are convective (the
warm city under 271 K air: ``N^2 = -7 × 10^{-4}`` s⁻² at 9 m) and an
elevated stable layer lies above (supercritical fraction 0.48–0.69
between 60 and 140 m). Of the 49 056 stable fluid cells 4.0 % are
relaminarised with the default Smagorinsky closure and 4.1 % (48 750
cells) with `:sigma` — mostly at the ground level (0.42) and in the
quiet top levels inside the sponge (0.61 and 0.91 at z = 184 and 197 m);
the stable layer between them is γ-turbulent. The background viscosity
``ν = 2`` m² s⁻¹ dominates the result: with ``ν = 0.2`` m² s⁻¹ (not the
default; same hour) the stable cells drop to 15 392 and only 0.1 % of
them are relaminarised (grid Reynolds number 28.7 against 3.8 in the
canopy). Cost of the added evaluation: one more `laminar_mask!` per
output record (the addon measured 34.6 ms per call with Smagorinsky,
43.2 ms with `:sigma`, on the flagship grid under a shared load).
