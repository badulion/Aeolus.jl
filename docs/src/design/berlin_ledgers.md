# The Berlin case: ledgers and tests

## The budget ledger

`case.ledger` integrates, step by step, the surface energy balance of
every surface class, the water budget of the air and the heat of the
air ([below](#The-heat-of-the-air), #253). The core calls
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

**Summation order (KA-b, #296).** Every ledger sum has one fixed order,
independent of the thread count. The water integrals build on
`level_sums(q)`, the per-level plane sums ``S_k = Σ_{ij} q`` in the
canonical order of `plane_partials!` ([kernel port](ka_port.md#The-reduction-contract)),
combined over ``k`` in ascending order; the heat of the air takes its
fluid and solid level sums the same way (`air_heat_levels!`).
`add_surface!` sums the step's column contributions per class in
column-major order and adds that step total to the accumulated term.
The couplings' own column totals — the surface exchange's evaporation,
the `aq_chem` emissions and dry deposition, the parent's area sums of
its tile fluxes — are `column_sum` totals (KA-m, #301): one work item
per column, the columns' terms added in column-major order from zero,
the same bits serially and threaded.

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

`run_case!` logs the ledgers at the end of the run as `Info` events
(`event = :ledger`, one per class, the water and each air-heat layer and
term, the values as key-values; #338), and the run monitor writes their
means and closures to `diagnostics.csv` as `ledger.*`
([Diagnostics](berlin_case.md#Diagnostics)). `ledger_report(case)` is
the same as a table, for scripts.
The cost is about 1 ms/step on the flagship grid (the two water
integrals), plus about 0.4 ms/step for the Kessler coupling's own
bookkeeping.

### The heat of the air

`ledger.heat`, an `AirHeatLedger`, closes the heat of the air

``H_{air} = Σ_k ρ̄_k c_p Π̄_k Σ_{ij ∈ fluid} θ_{ijk} Δz_k / (N_x N_y)``  (J m⁻²),

the dry enthalpy ``c_p T`` (``T = θ Π̄``) of the air above a unit of
ground, with the case's one reference state (`case.ref`) and without the
solid cells. The meter (`AirHeatMeter`, `core/air_heat.jl`) and the
ledger (`core/air_heat_ledger.jl`) are functions of (mesh or sim,
reference state, solid mask) and of the θ field, not of the Berlin
`Case`, so the parent run of the nesting epic (#246) keeps its own with
the same code.

**Every heater converts with the same weight.** A heat flux `q` (W m⁻²
into level `k`) raises θ by ``q Δt / (ρ̄_k c_p Π̄_k Δz_k)``:
`ref_θ_per_heat(case, k)` is that conversion, the inverse of the
meter's weight. `surface_exchange` and `traffic_heat` call it;
`buildings_citymean` applies the same ``1/(ρ̄_k c_p Π̄_k)`` inline in
`apply_building_heat!` (its `ρ`, `cp`, `π` keywords, passed from
`case.ref`), and `column_radiation`, whose heating rate is already
``dT/dt`` (K s⁻¹), divides it by ``Π̄_k`` only. Every heater therefore
uses the same weight, but the conversion is written in three places,
not delegated to one function. Before #253 they divided by ``ρ̄ c_p`` only and the
air received Π̄₁ ≈ 1.004 times the flux at the ground.
The meter weights θ with the dry ``c_{pd}``; the surface exchange
converts with the moist ``c_p`` of the first air cell, so the ledger
books its heating as ``H Δt\, c_{pd}/c_p``, about 0.8 % below the
surface's ``H Δt`` at ``q_v ≈ 0.01`` (exactly ``H Δt`` in dry air). The
ledger still closes, because it books what it measures.

**Attribution is automatic.** Couplings book nothing. Every schedule
entry `timed_run!` runs, and the dynamics step, is an attribution
*context* `(stage, coupling)`: the ledger measures ``H_{air}`` per level
when the context closes (the value it opened with is the previous
close) and books the difference. Contexts nest: the `:body_force` and
`:post_predictor` entries run inside `step!` and are taken out of the
dynamics' share. The dynamics' rest is split into
`(:dynamics, :transport)`, what the core's transport conserves
(``Σ W θ ΔV``, ``W = 1`` on the Boussinesq core), whose column total is
the numerics residual, `(:dynamics, :redistribution)`, the change of
``H_{air}`` because the transport conserves ``Σ W θ`` and not
``Σ ρ̄ Π̄ θ`` (moving warm air up into levels of lower ρ̄Π̄ lowers
``H``; the heat analogue of `density_redistribution`), and
`(:dynamics, :solid_exchange)`, the heat the transport and diffusion move
into the building cells (θ is carried through the immersed boundary, so
the solid cells' θ evolves and exchanges heat with the air through the
numerics alone). A change of θ outside every context is still caught:
as `(:core, :unattributed)` within a step, as `(:output, :between_steps)`
between two steps. The smoke test asserts that neither is booked.

**Booked terms.** `add_heat!(ledger, term, J; k)` books `J` J m⁻² under
`(stage, term)` from inside a running context and takes it out of that
context's own share, so a context's entry becomes the residual of the
terms named inside it. The dynamics books `:top_flux`, the heat through
the fixed-gradient θ top ([The θ top](#The-θ-top), N-parent-b, #254), this
way; the nesting epic will add `:boundary_heat`, `:relaxation`,
`:mass_correction` (N-open-d) and `:feedback_heat` (N-berlin-twoway, #267: booked in the parent's ledgers as `(:nest_feedback, :feedback_heat)` and `(:nest_feedback, :feedback_water)`, see [Nesting](berlin_nesting.md#Two-way-option-(#267))). Called outside a step it is
an error: that heat would have no place in the balance.

**Layers and windows.** Every term is kept per level and per local hour
of the step's start, so `heat_budget(ledger; window = (20, 5), layer =
:canopy)` returns the 20:00–05:00 integrals of #238 for the canopy
(`(0, cfg.canopy_top)`; `NaN`, the default, takes the tallest roof),
or for `layer = (0, 100)`, `:column`. For a sub-layer
`(:dynamics, :transport)` is the heat the dynamics carried across the
layer's top. `ledger_report` prints the whole-run table (column,
canopy, 0–100 m); `heat_report(ledger; window = (20, 5))` the night one.

In the Alexanderplatz scene the tallest roof is the Fernsehturm, clipped
at the 200 m domain top (the next tallest roof is 143.75 m, the
90th percentile of the built columns 37.5 m, the mean 21.2 m), so the
default `:canopy` is the whole column there; set `cfg.canopy_top` for the
street canyons.

**Tests** (the former `test/berlin_air_heat.jl` and `test/berlin_smoke.jl`,
removed in #320 with the rest of the Berlin suite; the heat content of
the discretely hydrostatic reference against this analytic value is now
asserted in `test/era2_anelastic.jl`, the ledger closure by the smoke runs
of `test/urban_checks.jl`). The meter
against the analytic heat of a uniform-θ hydrostatic layer
(``c_p^2 θ p_0/(R g)\,[Π^{c_p/R+1}/(c_p/R+1)]``: 3.1e-8 at Nz = 16,
2.0e-9 at Nz = 64, second order). A closed box (every θ source off):
the transport's column change is ≤ 2e-16 of ``H`` per step; ``H``
itself changes by up to 3.7e-7 per step, all of it `redistribution`.
(The issue expected ``|ΔH_{air}|/H_{air} ≤ 10^{-13}`` per step for a
flux-form θ advection. That holds for what the flux-form Boussinesq
transport conserves, ``Σ θ ΔV``; for ``H``, which weights θ with ρ̄Π̄,
the expectation was wrong.) H₀ = 100 W m⁻² into the lowest level of a
flat, motionless variant: the attributed term and ``ΔH_{air}`` equal
H₀Δt to 3.9e-13 (relative). Every smoke run (everything on included) closes to
≤ 1e-10 W m⁻² (measured ≤ 1.9e-14 in all 39 smoke runs).

**Cost.** One measurement is one pass over the interior (20 µs on
64 × 64 × 32); the flagship step opens ≈ 29 contexts, 31.1
measurements per step: 1.06 ms/step warm (0.7 % of a 150 ms step;
1.27 ms of 178 ms over the 24 h run on a shared machine). The issue's
target was < 1 ms/step; the measurements are memory-bound, so the rest
would need fewer contexts.

**Night budget of the flagship (measured, #253).** The 24 h
Alexanderplatz run (20 June 2024 from 06:00, every default coupling,
adaptive step, 61 902 steps), 20:00–05:00 local (32 401 s), mean W m⁻²
over the window (J m⁻² = W m⁻² × 32 401 s):

| term | column 0–200 m | 0–100 m | 0–37.5 m |
| :-- | --: | --: | --: |
| storage ``ΔH_{air}`` | −52.79 | −26.35 | −9.26 |
| `column_tendency / column_radiation` | −52.70 | −42.73 | −32.61 |
| `anthropogenic / buildings_citymean` | −10.31 | −10.71 | −14.80 |
| `anthropogenic / traffic_heat` | +8.00 | +8.00 | +8.00 |
| `surface_flux / surface_exchange` | +0.12 | +0.12 | +0.11 |
| `dynamics / transport` (across the layer top) | −5e-11 | +17.28 | +28.44 |
| `dynamics / solid_exchange` | +1.70 | +1.39 | +1.26 |
| `dynamics / redistribution` | +0.40 | +0.30 | +0.34 |
| `fog_dew / fog_dew` | 0 | 0 | 0 |
| closure | −7.2e-15 | −1.8e-14 | −1.1e-14 |

The default `:canopy` is the whole column here (the Fernsehturm); 37.5 m
is the 90th percentile of the roof heights. At night the air of the
column loses 52.8 W m⁻², almost all of it to the column radiation; the
surface exchange gives the air almost nothing (+0.12 W m⁻²), the
city-mean buildings take heat out of it (−10.3) and the traffic heat
puts 8 W m⁻² in. Inside the canyons the transport brings down from
above most of what the radiation removes (+28.4 of −32.6 W m⁻² below
37.5 m). This is the #238 budget and the baseline of N-berlin-validate
(#270).

### The θ top

Since N-parent-b (#254) the θ condition of the top face is
`cfg.θ_top = :initial_gradient`, on by default: `FluxBC(Γ)` with
``Γ = ∂θ/∂z`` of the initial profile across the top face
(`θ_top_gradient(cfg)`, 3 × 10⁻³ K m⁻¹ above the 100 m inversion), the
fixed-gradient top of PALM and DALES. Before it the top was a closed
no-flux lid, still available as `θ_top = :noflux` (a number sets Γ in
K m⁻¹). The physics is `src/` (`BoundaryConditions.FluxBC`, the halo fill
of `src/boundary_conditions/gradient_top.jl`; theory in
[Large-scale forcing](../theory/large_scale_forcing.md#The-fixed-gradient-top));
the case code is `core/top_boundary.jl`. `qv` and the tracers keep their
no-flux top.

**Ledger.** The heat the top face carries is booked as
`(:dynamics, :top_flux)` at the top level. The composed body-force hook
captures ``Γ\,ρ̄_N c_p Π̄_N ⟨ν + ½(κ_{v,N} + κ_{v,N+1})⟩`` in every dynamics
stage (the SGS closure has just filled ``κ_v``), and each accepted
(sub)step books it times its length (`dynamics_step!`, each piece of
`step_adaptive!`, `step_imex!`). What is left to `(:dynamics, :transport)`
over the column is round-off: 3.5e-11 W m⁻² (adaptive) and 8.4e-10 W m⁻²
(fixed step) on the smoke case against a top flux of 7.18 W m⁻², 3.1e-6
J m⁻² over the 24 h flagship against 6.2e5 J m⁻². With the IMEX stepper
the residual is the CG tolerance (−1.0e-2 W m⁻² at `tol = 1e-8`,
−8.6e-7 at `1e-12`; −1.4e-3 with the lid), on the anelastic core the
core's unweighted diffusion (2.0e-2 W m⁻², 1.8e-2 with the lid).

**Measured (flagship, 24 h from 06:00, every default coupling, adaptive
step).** Two runs that differ only in `θ_top`:

| | `:initial_gradient` (default) | `:noflux` (before #254) |
| :-- | --: | --: |
| steps | 62 724 | 61 903 |
| top-face heat flux, 24 h mean | +7.21 W m⁻² | 0 |
| top-face heat flux, 20–05 h | +7.14 W m⁻² | 0 |
| column storage 20–05 h | −49.94 W m⁻² | −52.79 W m⁻² |
| column radiation 20–05 h | −55.06 W m⁻² | −52.70 W m⁻² |
| canopy air θ (fluid cells below 37.5 m), 20–05 h mean | 313.53 K | 312.54 K |
| canopy air θ at 20:00 / 05:00 | 317.02 / 309.88 K | 316.21 / 308.74 K |

The flux is downgradient, into the domain (the free atmosphere above is
warmer), and nearly constant: ``ρ̄_N c_pΠ̄_N νΓ`` = 7.09 W m⁻² with the
case's molecular ``ν`` = 2 m² s⁻¹ is almost all of it, the SGS
``κ_v`` at the top adding 0.12 W m⁻² over the 24 h. The nightly canopy air is
0.99 K warmer (20–05 h mean), 0.81 K at 20:00 rising to 1.14 K at 05:00,
and cools 0.33 K less during the night (7.13 against 7.46 K). In the
afternoon the convective layer fills the 200 m domain, and a fixed
gradient then imposes ``κΓ`` regardless of the real entrainment: the
open top of the nested run (N-open-b, N-berlin-nest #265) replaces it.

**Stated exception: no θ damping in the periodic child.** θ is not
Rayleigh-damped in the periodic 200 m child. Its velocity sponge starts
at 150 m, and a θ damping toward a constant target there would clamp the
top 50 m (#238, approach B). With `lateral = :open` (#262) the top is
open and θ is relaxed over the same layer toward the boundary data's
profile, the state the open top imposes (`cfg.θ_relaxation`, see
[Open boundaries](#Open-boundaries)); in the nested run that is the
parent's. `LargeScaleForcing.ReferenceDamping` is the one θ-damping type
in both places.

### Open boundaries

`cfg.lateral = :open` (N-open-d, #262; `core/open_boundary.jl`) gives the
case open west, east, south, north and top faces fed by analytic data (with
the nest switched off, `physics = Dict(:nesting => false)`; the nested
default of #265 feeds the same faces from the parent, see
[Nesting](berlin_nesting.md#Nesting-(#265))):

* **Data.** `ProfileBoundaryData(u = U_g, v = V_g, w = 0, θ = initial_θ)`
  — the `:geostrophic` coupling's wind (also when it is off) and the
  initial θ profile (`case_boundary_data`). The normal velocity of every
  face is mass-corrected by the uniform global correction (N-open-b),
  the outflow is `cfg.outflow` (`:dirichlet`, default; `:convective`,
  `:orlanski`, N-open-c).
* **Sponge.** The core's Rayleigh sponge relaxes `u, v, w` toward the
  same data (`sponge_target(data, mesh)`, the one damping kernel; a
  sponge below an open top without a target is an error), so the
  `:geostrophic` coupling's own retargeting of the periodic case is off.
  θ is relaxed toward the data's profile over the sponge layer
  (`cfg.θ_relaxation = true`, `ReferenceDamping` in the body-force hook):
  in the gravity-wave test of #254 damping θ as well as the velocity cut
  the reflection of a layer one vertical wavelength deep from 0.17 to
  0.034.
* **Geometry.** `cfg.boundary_frame` cells (default 3, 47 m in the smoke
  window, 47 m on the flagship grid) next to each open side are kept free
  of buildings; a building that still touches a face (frame 0, or the
  Fernsehturm through the top) is solid there:
  `mark_open_boundary_solids!` gives its boundary points zero velocity and
  takes them out of the mass-flux correction.
* **Neighbours.** No coupling wraps across the box any more: every
  former `mod1` reads through `neighbor_index` (`centre_wind` for the
  cell-centred wind of the surface exchange, the deposition and the facade
  wall model; the wall tagging; the a-priori stress centring); masks and
  tracer halos take `case_scalar_bcs` (zero gradient at an open face; a
  tracer has no boundary data, so it leaves with its own value and enters
  with the value next to the face); the pressure solvers read the
  all-Neumann axes from the case's `bcs`.

**Ledger.** The open faces carry heat and water. Inside the dynamics
context the composed body-force hook captures, per level, with the state
and the SGS diffusivity the core's tendencies were built from:

| term | what |
| :-- | :-- |
| `(:dynamics, :boundary_heat)` | ``-\sum ρ̄_k c_p Π̄_k (θ\,u·n - K ∂_nθ) A`` through the four sides and the top (`boundary_scalar_flux`: the core's centred θ face values, ``K = ν + ½(κ_{in} + κ_{out})``), without … |
| `(:dynamics, :mass_correction)` | … the part the global correction ``δ`` carries, ``δ\,ρ̄c_pΠ̄\,θ_{face} A`` on every face it adjusted |
| `(:dynamics, :relaxation)` | the θ relaxation toward the data |
| water `:boundary` | the vapour and cloud water the faces carry, `boundary_scalar_flux` with the transport's own scheme (`scalar_advection`) |

Each accepted (sub)step books them times its length (`book_dynamics_heat!`,
also called for the fixed-gradient top). The mass-correction term is
large next to its velocity because it multiplies the absolute θ (≈ 293 K):
``|δ|/U`` = 7.8e-5 carries 19 W m⁻² in the convective run below, of the
same sign as the net mass the correction removes; only the sum of the two
terms is the boundary's heat. For the forward-Euler steps of the default
`:adaptive` and the `:fixed` stepper the booked heat is the heat the step
moved; under `:imex` it is the last stage's rate times dt (a residual in
`(:dynamics, :transport)`, as for the top flux: −2.4e-2 W m⁻² on the
open smoke window); on the anelastic core the column transport keeps
the core's unweighted-diffusion residual of #254 (2.2e-2 W m⁻² open,
1.8e-2 with the lid). The smoke check asserts < 1e-6 W m⁻² for the
forward-Euler Boussinesq steps and < 0.1 W m⁻² otherwise. The diffusive part is the
closures' ``κ_h``/``κ_v`` form: the cross terms of the tensor closure
(`sgs = :reynolds_stress`) at the faces are not in it.

**Measured (smoke window, 16 × 16 × 16, every coupling on, 24 s from
12:00).**

| | periodic | `:open`, Dirichlet | `:open`, convective |
| :-- | --: | --: | --: |
| steps (adaptive) | 8 | 9 | 9 |
| surface sensible heat H (domain) | −2.12 W m⁻² | −2.87 W m⁻² | −2.85 W m⁻² |
| `top_flux` (fixed-gradient top) | 7.18 W m⁻² | — | — |
| open faces: top | — | 7.19 W m⁻² | 14.55 W m⁻² |
| open faces: four sides | — | −0.17 W m⁻² | −3.34 W m⁻² |
| `boundary_heat` | — | 7.02 W m⁻² | −7.98 W m⁻² |
| `mass_correction` | — | 0 | 19.19 W m⁻² |
| `relaxation` | — | 0.023 W m⁻² | 0.025 W m⁻² |
| max ``|δ|/U`` | — | 0 | 7.8e-5 |
| `(:dynamics, :transport)` (column) | −1.1e-10 W m⁻² | 4.3e-10 W m⁻² | −2.0e-10 W m⁻² |
| air-heat closure | 0 | −4.7e-15 W m⁻² | 3.8e-14 W m⁻² |
| water `:boundary` | — | 6.6e-5 kg m⁻² | 9.8e-5 kg m⁻² |
| water residual | 1.1e-15 kg m⁻² | 1.5e-15 kg m⁻² | 1.3e-15 kg m⁻² |

The open top's heat is the diffusion of the Dirichlet θ halo (the initial
profile, 3 K km⁻¹ above the inversion) through the top face: 7.19 W m⁻²,
the 7.18 W m⁻² of the periodic case's fixed-gradient top. With Dirichlet
outflow the data balance (``δ = 0``); the convective outflow radiates the
interior's building-perturbed wind at the faces and the correction
removes the imbalance, which moves heat through the top as well. The
column transport stays a numerics residual (1e-10 W m⁻²) in every
configuration.

**Cost.** The fixed stepper, the same buildings (`boundary_frame = 0`),
the faster of two alternating rounds after a warm-up, on a shared
machine:

| grid | periodic | `:open` | ratio |
| :-- | --: | --: | --: |
| flagship 64 × 64 × 32, whole step | 116.2 ms | 113.3 ms | 0.98 |
| flagship, dynamics (`step!`) | 20.5 ms | 18.1 ms | 0.88 |
| smoke 16 × 16 × 16, whole step (3 rounds of 24 steps) | 18.5 ms | 18.7 ms | 1.01 |

The open faces' work (the buffer update and mass correction, the
boundary-heat capture, the θ relaxation) costs no more than the
measurement noise; the all-Neumann cosine solve of the open box is if
anything cheaper than the periodic-x, y solve. The former
`test/berlin_open_boundaries.jl` printed a smoke-window measurement.

**Tests** (the former `test/berlin_open_boundaries.jl`, removed in #320;
the +1 K anomaly's analytic face flux, upwind and centred, with the
diffusive part, and the wall tagging across an open face are asserted in
`test/era6_open_near_boundary.jl`): the default stays periodic;
the open faces, data, sponge target, solver axes, frame and solid points;
one dynamics step through the case's own hook books 0.0 J m⁻² for an
inflow equal to the interior and ``ρ̄c_pΠ̄(UΔT/2 + νΔT/Δx)AΔt`` = 604.525
J m⁻² to 1e-12 for a +1 K inflow anomaly (the core's centred face value is
the mean of the halo and the first cell; the diffusive flux across the
jump is ``νΔT/Δx`` with the case's ``ν`` = 2 m² s⁻¹). The smoke matrix runs
`geostrophic: lateral = :open, Dirichlet outflow` and `…, convective
outflow` with every coupling on: air-heat closure ≤ 1.1e-13 W m⁻², water
residual ≤ 4.3e-16 kg m⁻², column transport < 1e-6 W m⁻² (smoke check).
The src tests are in `test/era6_open_near_boundary.jl` (theory:
[Near the open faces](../theory/fractional_step.md#Near-the-open-faces-(N-open-d))).

### Baseline (measured, #127)

(Before the reference state of #131. With it the slab's fluxes carry
ρ̄_1 ≈ 1.20 instead of 1, so in the same hour the domain H is −23.2
instead of −19.7 W m⁻², and the moist buoyancy changes the flow; see
the #131 PR for its ledger.)

Flagship configuration, 1 h (1800 steps, 06:00–07:00 local,
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
persists. Remeasured on master (`915bd49`, smoke-run
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

## Smoke test

The smoke runs were `test/berlin_smoke.jl` until #320 removed the Berlin
test suite; they were then the `smoke` part of the by-hand
`examples/urban/berlin_checks.jl`, and since #438 are
**`test/urban_checks.jl`** (full tier; `julia --project=. test/run.jl --only
urban_checks.jl`), while every run writes its ledger closure to
`output/results.toml`. They
run the case in `examples/urban/configs/berlin_smoke.toml`
(each run of the matrix is that file plus `--set` overrides, CFG-c #358): a
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

The run matrix is derived from the registry (`smoke_specs()` in
`test/support/smoke_matrix.jl`, written as overrides by `smoke_overrides`):

1. **summer, everything on** — every additive coupling on, including
   those off by default, and the default provider of every slot;
2. **winter, everything on** — the same with the values
   `berlin_winter.toml` sets (its initial state);
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
coupling must have run in at least one configuration. (As a test it also
had to take less than 300 s; the by-hand script has no time limit.)

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

The coverage test `test/berlin_coverage.jl` and its lists (`BERLIN_EXEMPT`,
`BERLIN_PENDING`, `BERLIN_DEFAULT_OFF`, `BERLIN_OPTION_ONLY`) were removed
(maintainer decision 2026-10-06), and with #320 the rest of the Berlin
suite: the checks that the "everything on" runs select every addon and
default provider and that the matrix is derived from the registry
(plumbing of the former flagship rule) are gone; the every-coupling-ran
check stays in the smoke runs.
