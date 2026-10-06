# The Berlin case: ledgers and tests

## The budget ledger

`case.ledger` integrates, step by step, the surface energy balance of
every surface class and the water budget of the air. The core calls
`ledger_begin_step!` before `:clock` (it snapshots `bus.Ts` into
`ledger.Ts_start` and the water in the air) and `ledger_end_step!` after
`:sanity` (it calls every coupling's `budget!` and books the change of
the water in the air). A coupling reports only what it did during that
step:

| call | meaning |
| :-- | :-- |
| `add_surface!(l, term, F, sfc, dt)` | a W m⁻² map `F` on the columns, over `dt` s, sorted by surface class |
| `add_water!(l, term, m)` | `m` kg m⁻² of domain added to the air (negative: removed) |
| `water_tracer!(l, :qi)` | count a new tracer as water in the air (not one that is part of another, like P3's rime mass) |
| `ledger_water(case, q)` | ``Σ_k ρ_k Σ_{ij} q Δz / (N_x N_y)``, for measuring a change of a tracer |
| `l.dt` | the length of the current step (s); use it, not `cfg.dt`, to turn the step's energy into a flux — they differ under the adaptive stepper |

**Surface energy.** `:Rn` (net radiation into the surface), `:H`,
`:LE`, `:G` (sensible, latent and ground heat flux out of it) and `:dS`
(storage change of the surface layer) are the core terms. Every other
term name is an extra flux *into* the surface. `surface_budget(ledger)`
returns per class, and for the whole domain,

    residual = Rn − H − LE − G − ΔS        (what the four-term balance misses)
    closure  = residual + Σ extras          (round-off if nothing is missed)

The residual is a physical baseline: the terms this model has that the
conventional balance does not. The closure is bookkeeping and the smoke
test asserts it to 1e-6 W m⁻² (measured ≤ 3.5e-10).

Who books what in the default case: `surface_exchange` books, on each
surface step, the whole interval's `Rn`, `H`, `LE` and `G` (the heat
conducted into a soil column; a slab's storage `C_s ΔT` goes to `ΔS`
instead), and every step the change of the extra energy it holds for
the next surface step (in `ΔS`); `fog_dew` the dew's latent heat
(`dew_heat`); `buildings_citymean` the window sun that leaves the wall
cells (`window_sun`); `glass` the shortwave transmitted into the
building (`glass_transmission`). Before #129 the slab booked the same
terms per step and its Ts clamp as `clamp`; nothing is clamped now.

**Water.** `water_budget(ledger)` gives the change of water in the air
`ΔW` (measured by the ledger over `qv`, `ql`, `qr` and every
`water_tracer!`), the booked sources and `residual = ΔW − Σ sources`.
Sources in the default case: `evaporation` (surface exchange; negative
where the surface takes up dew), `building_vapour`,
`dew` (deposition, negative), `precipitation` (Kessler rain out of the
bottom), `qv_clamp` (the Kessler coupling's clamp), `transport` (the
change of water the transport stage makes; zero for a conservative
scheme, and measured at ~1e-19 kg m⁻² at ρ = 1 since #132) and
`sgs_diffusion` (the same for the tracers' SGS diffusion, #177; ~1e-19). The ledger
integrates the water with `ledger.ρ`, one density
per level, the reference density `case.ref.ρ̄` (#131). A coupling that
puts a flux into the air with a different density than the ledger's
shows up as a water residual, so a coupling converts every flux with
`ref_density(case, k)`.

The incompressible core does not conserve that ρ̄-weighted water: ∇·u = 0
conserves Σ q, so vapour carried up into thinner air loses ρ̄-weighted
mass. The transport stage therefore books its change of water in two
parts (`ledger_water_split`): `transport`, the column-mean ρ̄ times its
change of Σ q (zero for a conservative scheme; before #132 the halo
leak), and `density_redistribution`, the rest. Under a sound-proof core
(#133) the transport is W-weighted and conserves Σ W q, and the split is
by that weight: `transport` is c Σ W dq (c = ρ̄_m/W_m), zero for a
conservative scheme, and `density_redistribution` Σ(ρ̄_k − c W_k) dq,
exactly 0 on the anelastic core (W = ρ̄). Over the 1 h flagship run
(06:00–07:00, before #132's halo fix) with moist buoyancy on,
`density_redistribution` is
−8.1e-4 kg m⁻² (72 % of the slab's evaporation, 1.13e-3) and
`transport` −7.0e-4; with it off, −6.3e-4 and +1.5e-5. Without the
split both would be booked as `transport` (−1.5e-3, against −2.7e-4 at
ρ = 1 in the #127 baseline), and fixing the #132 halo leak could not
bring it to zero. Kessler
sedimentation has the same effect inside its `precipitation` term
(rain falling into denser levels), which #143 owns.

The water integral runs over every grid cell, solid ones included.
Before #129 the slab's evaporation and the dew exchange acted on level 1
of every column, which under a roof is inside the building; the ledger
closed either way, so a closed water ledger does not show where the
water went. Since #129 both act on the first air cell (`bus.k_air`), and
the surface exchange's smoke check counts writes into solid cells
(zero).

`run_case!` prints `ledger_report(case)` at the end of a verbose run.
The cost is about 1 ms/step on the flagship grid (the two water
integrals), plus about 0.4 ms/step for the Kessler coupling's own
bookkeeping.

### Baseline (measured, #127)

(Before the reference state of #131. With it the slab's fluxes carry
ρ̄_1 ≈ 1.20 instead of 1, so in the same hour the domain H is −23.2
instead of −19.7 W m⁻², and the moist buoyancy changes the flow; see
the #131 PR for its ledger.)

Flagship configuration, 1 h (`AEOLUS_NSTEPS=1800`, 06:00–07:00 local,
20 June, 64 × 64 × 32, dt = 2 s), mean W m⁻² over the hour per class:

| class | cols | Rn | H | LE | G | ΔS | Rn − H − LE − G − ΔS | extras | closure |
| :-- | --: | --: | --: | --: | --: | --: | --: | :-- | --: |
| street | 1690 | 327.47 | −18.45 | 0.88 | −4.72 | 350.04 | −0.27 | dew heat 0.27 | 1.7e-11 |
| roof | 874 | 309.18 | −15.83 | 0.44 | 1.51 | 323.21 | −0.16 | dew heat 0.16 | 2.3e-11 |
| park | 511 | −9.95 | −23.51 | 0.04 | −15.07 | 33.13 | −4.55 | dew heat 4.55 | 1.1e-12 |
| water | 2 | 389.85 | −9.36 | 18.23 | 0 | 380.98 | 0.00 | — | −3.5e-10 |
| wall | 978 | 172.33 | −24.03 | 0.05 | −12.26 | 145.08 | 63.49 | dew heat 3.39, window sun −66.89 | 1.4e-10 |
| glass | 41 | 309.22 | −4.53 | 4.31 | 21.27 | 43.87 | 244.29 | glass transmission −244.29 | 9.6e-12 |
| domain | 4096 | 244.28 | −19.71 | 0.53 | −6.22 | 252.79 | **16.89** | dew 1.52, glass −2.45, window sun −15.97 | 4.6e-11 |

Water over the same hour (kg m⁻² of domain): ΔW = −7.75e-4 =
evaporation +7.72e-4, building vapour +9.38e-4, dew −2.21e-3,
precipitation 0, clamp 0, transport **−2.70e-4**; residual −1.4e-14.
After the stale-halo fix of #132 the same hour gives ΔW = −5.72e-4 =
evaporation +7.57e-4, building vapour +9.12e-4, dew −2.24e-3,
transport **+6.9e-19**; residual −1.3e-14 (the surface-energy table
changes by at most 0.1 W m⁻² in any entry).

What the numbers say about the current (pre-#128/#129) physics:

* **The park has negative net radiation in the morning sun** (−9.95
  W m⁻² while the streets get +327). This is the compounding tree shade
  of #128: `bus.sw` over the park is multiplied by exp(−0.5·LAI) every
  step between radiation updates, so it is effectively zero. **Fixed in
  #128**: see "After #128" below.
* **H is negative everywhere** (the air heats the surfaces): the case
  starts with `Ts0 = 285 K` under air at θ = 293 K, and the slab's
  constant-coefficient flux (#129) is weak. The surfaces warm by storage
  (ΔS ≈ Rn) instead.
* **The transport was not conservative**: over the hour it removed
  2.7e-4 kg m⁻² of water, 35 % of what the surfaces evaporate. This was
  the stale-halo leak below, fixed by #132.
* The domain residual of 16.9 W m⁻² is the window sun (−16.0), the
  glass transmission (−2.4) and the dew's latent heat (+1.5): energy
  that goes into the buildings or comes from condensation, which the
  four-term balance does not see.

On master (`915bd49`, after #128–#133, adaptive step) the same hour
gives a domain Rn of 222.28 W m⁻² and a residual of 19.02 W m⁻²: window
sun −16.57, glass −2.50, dew heat +0.06. The dew term has almost gone because since #129
the RadiationFog dew acts only over the β = 0 surfaces (roofs, glass);
the soil and water surfaces take up dew as negative evaporation. This
was measured with a reconstruction of the method above, not the
original script, so it is indicative.

**The `:smoke` fog seed (#127, corrected in #131).** The #127 seed set
`q_v = q_s(θ, p)`, treating θ as the temperature, while the fog
adjustment uses `T = θ Π(p)`, about 1.1 K warmer at 1013 hPa. The air
was at 93.5 % RH and the fog evaporated completely on the first
`:fog_dew` call. Since #131 the seed is `q_s(θ Π̄_k, p̄_k)` and the fog
persists. Remeasured on master (`915bd49`, `test/berlin_smoke.jl`
configuration, 16 × 16 × 16, every addon on, 24 s at noon: with the
adaptive default that is 9 steps, the first of 2.2 s, mean 2.67 s, not
12 of 2 s):

| | summer | winter |
| :-- | --: | --: |
| ql left after the first `:fog_dew` call, called directly on the seeded state (dt of the first step, 2.2 s) / within step 1 | 99.91 % / 99.13 % | 100.0 % / 98.1 % |
| ql left after 24 s (9 steps) | 84.1 % | 75.1 % |
| domain Rn (W m⁻²) | 568.41 | 70.10 |
| domain residual Rn − H − LE − G − ΔS (W m⁻²) | −4.87 | +7.86 |
| of which dew heat / window sun / glass (W m⁻²) | +14.99 / −6.31 / −3.81 | +0.09 / −6.92 / −1.03 |

The other summer terms are H −1.30, LE −1.24, G 363.36, ΔS 212.46
W m⁻², closure −2.9e-13. The dew heat now sits only on the β = 0
surfaces: a class mean of +55.6 W m⁻² on roofs and +56.3 on glass. The
same model without fog (the pre-#131 seed, whose fog evaporates at
once) has a summer Rn of 602.30, so the fog darkens the smoke window by
about 34 W m⁻².

Earlier values of this paragraph (residual −34.4 / −33.9 W m⁻² with
dew heat +44.4 at #127–#132; Rn 34.8 instead of 388.7 and residual
−53.2 at #131) do not reproduce on master. #128–#132 changed the
shade, the cloud shadow, the surfaces and the seed in between; which of
these changes accounts for the difference was not isolated. Even the
pre-#131 seed now gives a summer residual of −3.14 W m⁻² (dew heat
+13.34, window sun −6.31, glass −3.88), winter +8.0, with 0 % of the
fog left after the first call and after 24 s (that reproduction uses p̄_k, since the 8 km scale-height pressure
it originally used is gone).

### After #128 (shade applied once, radiosity class map)

Same 1 h flagship configuration. The park shortwave (mean over the 511
park columns) is 116.05 W m⁻² averaged over the hour, against 0.62
before; the park / open-street ratio is 0.288–0.298 on every step
(before: 0.298 on a radiation step, then 0.086, 0.025, 0.007, …, 0 to
three decimals after 180 steps). The ratio is a little above
exp(−0.5·2.5) = 0.2865 because the unshaded effective irradiance of the
park columns is 0.4–4 % above that of the open streets (the park is less
shaded by buildings in the low morning sun). Per class, W m⁻² over the
hour:

| class | Rn before | Rn after | ΔS after | dew heat after |
| :-- | --: | --: | --: | --: |
| park | −9.95 | 80.47 | 119.74 | 3.43 |
| street | 327.47 | 327.63 | 350.21 | 0.28 |
| water | 389.85 | 389.95 | 381.12 | — |
| domain | 244.28 | 255.71 | 263.72 | 1.39 |

The domain residual Rn − H − LE − G − ΔS is 17.07 W m⁻² (16.89 before);
it is still the window sun, glass transmission and dew heat. The park and
water ground facets now carry their own material in the radiosity solve
(511 park and 2 water facets; before, all 3181 ground facets were
street). For flat ground this changes the absorbed shortwave but not the
effective irradiance `absorbed / (1 − α)` the slab receives, so the park
shortwave at the first radiation step moves only from 99.409 to 99.427
W m⁻²; the brighter park raises the reflected light on its neighbours
(street 333.190 → 333.212).

Around noon (900 steps, 11:45–12:15, otherwise the same flagship
configuration) the park / open-street ratio is 0.2840–0.2860 on every
step against exp(−1.25) = 0.2865, and the park shortwave is 234.83
W m⁻² averaged over the half hour (1.36 before). Park Rn goes from
−10.12 to +174.01 W m⁻²; domain Rn from 495.61 to 518.76.

In the `:smoke` configuration the summer domain residual was −33.9
W m⁻² after #128; on master it is −4.87 W m⁻² (see the table under the
#127 baseline).

## Smoke test and coverage test

**`test/berlin_smoke.jl`** runs the case in `CaseConfig(:smoke)`: a
250 m × 250 m × 100 m window of the real OSM file (16 × 16 × 16,
dx = 15.6 m, 312.5 m south of the anchor, 69 building and 50 park
columns, tallest roof 31 m) at local noon, 12 steps of 2 s, radiation
every 8 s, soil every 4 s. Before the first step it seeds a fog layer
from a fixed RNG seed (`Xoshiro(127)`): air saturated at the reference
state (`q_v = q_s(θ Π̄_k, p̄_k)`) with ql = 0.2 g/kg
± 10 % in the lowest three levels and rain qr < 5e-5 in levels 2–3, so
dew, fog, sedimentation and precipitation all do real work. The
flagship's pollutant stacks are outside this window, so the smoke runs
put one stack in it, 40 m up.

The run matrix is derived from the registry (`smoke_matrix()` in the
test file):

1. **summer, everything on** — every additive coupling on, including
   those off by default, and the default provider of every slot;
2. **winter, everything on** — the same with the `:winter` initial
   state;
3. one run per **non-default provider** of every exclusive slot;
4. one run per entry of every coupling's **`smoke_variants`**.

A run that selects a coupling declared `incompatible` with an additive
coupling switches that additive coupling off; one incompatible with the
default provider of another slot switches that slot to its first
compatible provider (so `core = anelastic` runs with `pressure_solver =
density_weighted_cg`, and that solver's own run with the anelastic
core). Every run must finish
without a blow-up, leave every field and tracer finite, pass every
`smoke_check` of its couplings, and close the surface-energy ledger per
class (1e-6 W m⁻²) and the water ledger (1e-12 kg m⁻²). The first run
also asserts that the domain gains radiative energy at noon (Rn > 0,
ΔS > 0) and that the transport conserves water (|transport| <
1e-12 kg m⁻²; measured ~1e-19 since #132). Finally every registered
coupling must have run in at least one configuration, and the whole
test must take less than 300 s.

At #127 the matrix was 2 runs (no slot had a second provider yet); #130
added the column radiation's `surface_sw = :bird` variant, and #132 the
non-default providers of the numerics slots (`momentum_advection =
weno_z`, `pressure_solver = multigrid` and `mg_cg`, `sgs = dynamic` and
`mlsgs`) and the scalar-advection variants (TENO5; WENO-Z7 with halo 4).
Run on its own the test took about 66 s at #132, almost all of it
first-call compilation (each run takes 0.3–1 s once compiled, the
multigrid runs a few seconds). The `:core` and `:timestepper` slots of #133
add six (anelastic, pseudo-incompressible, `pressure_solver =
density_weighted_cg`, adaptive, IMEX, adaptive + ARK2): 18 runs.

**`test/berlin_coverage.jl`** checks that every submodule of `Aeolus`
is either listed in some registered coupling's `modules_used`, or in
`BERLIN_EXEMPT` with a reason, or in `BERLIN_PENDING` with the issue
that wires it in. It ratchets: a new src module fails the suite until
it is coupled or exempted, and a pending module that some coupling now
uses fails it until it is removed from `BERLIN_PENDING`, so that list
only shrinks. It also checks that each coupling's file calls into every
module it lists (an exported name, or a `Module.` qualification, in the
code with comments and docstrings removed); reviewed indirect uses go in
`BERLIN_INDIRECT`. At #127: 55 modules, 15 coupled, 11 exempt
(the core's infrastructure and the Parallel / Backends / SpectralElement
scaffolds), 30 pending; #130 coupled RadiationCore, GasOptics and
TwoStream (27 pending).

**The flagship guardrails.** The Berlin case is the flagship: it runs
every component, on by default, and new physics lands in it in the PR
that adds the physics. `test/berlin_coverage.jl` enforces that with four
reviewed lists, each frozen against a literal snapshot in the same file
(`*_FROZEN`; the test requires list = snapshot, so an entry can only be
deleted, from both; adding one is a snapshot edit a reviewer sees and
only the maintainer agrees to):

| list | holds | at #160 |
| :-- | :-- | :-- |
| `BERLIN_EXEMPT` | non-physics modules; each reason starts `infrastructure:` or `scaffold:` | 11 |
| `BERLIN_PENDING` | modules not wired in yet, with the owning issue | 24 |
| `BERLIN_DEFAULT_OFF` | addon couplings off by default, with the measured cost or conflict | 0 |
| `BERLIN_OPTION_ONLY` | modules only a non-default slot provider runs, with why it is an alternative | 2 (Multigrid: `pressure_solver = :multigrid / :mg_cg`; ML: `sgs = :mlsgs`) |

and with these testsets:

* **frozen lists … may only shrink** — each list equals its snapshot;
  exemption reasons start with `infrastructure:` / `scaffold:`.
* **every addon on by default, every slot's default provider in the
  default run** — every `slot = :addon` coupling has `is_default = true`
  unless it is in `BERLIN_DEFAULT_OFF`; every exclusive slot has exactly
  one default provider and it is not an `:off` / `:none` / `:disabled`
  switch (`chem_off` is a non-default provider of `:chemistry`); and
  `selected_couplings` of `CaseConfig()` and of every `SCENARIOS` preset
  contains every such addon and default provider.
* **a module only a non-default provider runs is a reviewed exception** —
  a module some coupling lists counts as coupled (the smoke test runs
  every provider), but if the default `CaseConfig()` run does not use
  it, it must be in `BERLIN_OPTION_ONLY`; an entry the default run now
  uses is stale and fails too.

`test/berlin_smoke.jl` adds one more: the "summer / winter, everything
on" configurations of `smoke_matrix()` must select every addon and every
default provider, so an `incompatible` declaration cannot silently drop
a coupling from the runs that are meant to have everything.

Each guardrail was checked against a deliberately broken tree (not
committed): a new `src/DummyPhysics.jl` added to `BERLIN_PENDING` or
`BERLIN_EXEMPT` fails "BERLIN_PENDING has grown: DummyPhysics" (resp.
EXEMPT, plus the `infrastructure:` / `scaffold:` check);
`is_default = false` on `veg_drag` fails "Addon couplings off by
default: veg_drag" and "CaseConfig() does not run: veg_drag"; `veg_drag`
moved into the `:sgs` slot as a non-default provider fails "Modules the
default Berlin run does not use (only a non-default provider does):
PlantCanopy (by veg_drag)"; making `chem_off` the `:chemistry` default
fails "slot :chemistry defaults to an off switch (chem_off)"; and
declaring `cloud_shadow` incompatible with `fog_dew` fails 'smoke run
"summer, everything on" does not run: cloud_shadow, fog_dew'.
