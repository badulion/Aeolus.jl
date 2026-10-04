# Wall models

An LES of the atmospheric surface layer or of a high-Reynolds-number
channel cannot resolve the near-wall eddies: the first grid point sits
in the logarithmic layer, where the resolved velocity gradient is far
too weak to carry the wall stress through the sub-grid viscosity. A
*wall model* supplies that stress from the resolved velocity at the
first point (Piomelli & Balaras 2002). In Aeolus every modelled wall
stress comes from one module, `Aeolus.WallModels` (Phase 57, #169); the
channel harness, the stable-boundary-layer cases and the Berlin ground
call it rather than writing their own.

## Equilibrium laws

An equilibrium law relates the wall-parallel speed ``U`` at the distance
``h`` from the wall to the friction velocity ``u_τ = \sqrt{τ_w/ρ}``,
assuming the layer between the wall and ``h`` is in equilibrium (no
pressure gradient, no advection). The wall stress is
``τ_w/ρ = u_τ^2`` (`wall_stress`, kinematic) and acts against the
velocity, ``τ_{w,i} = τ_w u_i/U``.

**Smooth wall — Spalding (1961).** One formula covers the viscous
sublayer, the buffer layer and the log layer:

```math
y^+ = u^+ + e^{-κB}\left[e^{κu^+} - 1 - κu^+ - \frac{(κu^+)^2}{2} - \frac{(κu^+)^3}{6}\right],
\qquad u^+ = U/u_τ,\quad y^+ = h u_τ/ν .
```

It tends to ``u^+ = y^+`` as ``y^+ → 0`` and to the log law
``u^+ = \ln(y^+)/κ + B`` as ``y^+ → ∞``. Given ``U``, ``h`` and ``ν``,
`friction_velocity` solves ``u^+ y^+(u^+) = Uh/ν`` by Newton's method,
safeguarded by the bracket ``0 < u^+ ≤ \sqrt{Uh/ν}`` (the left side is
convex and increasing, so the iteration converges monotonically once it
is above the root). Measured: the round trip ``y^+ → u^+ → y^+`` over
``y^+ ∈ [10^{-2}, 10^6]`` (200 points) is exact to 8.7e-15, and
``u_τ`` is recovered to 9.3e-16. The defaults are κ = 0.41, B = 5.2;
`LEE_MOSER_2015` has the constants of the ``Re_τ ≈ 5200`` channel DNS,
κ = 0.384, B = 4.27.

Spalding's formula is *not* the log law at moderate ``y^+``: truncating
the exponential after the cubic leaves it above the log law by
``δu^+ ≈ [e^{-κB}(1 + x + x^2/2 + x^3/6) - u^+]/(dy^+/du^+)``,
``x = κu^+``. This is 2.5e-3 at ``y^+ = 10^4``, 7.2e-4 at ``10^5`` and
1.4e-4 at ``10^6`` (measured, equal to the estimate to 0.1 %). The
issue's original expectation, 1e-3 at ``y^+ = 10^4``, was wrong for
this formula.

In the viscous sublayer (``h^+ < 1``) the law is the linear stress
``τ_w = νU/h``, the stress `CutCell.wall_shear_stress` computes from a
probe at the same distance; at the probes of a flat cut-cell floor
(``h^+ ≤ 0.40``) the two agree to 9.5e-6.

**Rough wall, neutral — `RoughLogLaw(z0)`.**
``u_τ = κU/\ln(h/z_0)``, independent of ν.

**Rough wall, stratified — `MOSTWallLaw(sf, z0m, L)`.** Monin–Obukhov
similarity for momentum at a held Obukhov length ``L``:

```math
u_τ = \frac{κ\,\max(U, U_{min})}{\ln(h/z_{0m}) - ψ_m(h/L) + ψ_m(z_{0m}/L)} .
```

Nothing here is restated: κ is `SurfaceExchange.von_karman`, and the
denominator is the stability-corrected log term of
`SurfaceExchange.solve_surface_layer` with the stability functions'
`psi_m`, so at the solver's ζ the law returns the solver's ``u_*`` bit
for bit (tested at ζ = ±0.5), and with ``L = ±∞`` it is the rough log
law (1e-12). A surface model that returns only ``u_*`` hands its
stability over through `matching_obukhov_length`, the ``L`` for which
the law gives back that ``u_*`` (the root of the monotone
``Φ_m(ζ) = κU/u_*``, by the solver's Illinois iteration).

## The operator

`apply_wall_stress!(sim, wm, Δt)` applies the stress of a law to the
first velocity level next to the z-walls of a `RegularGrid`, as a
`post_predictor!` hook. The speed of a column is taken at the centre of
its wall cell (``h = Δz/2``); a face shared by two columns gets the mean
of the two columns' stresses, a column contributing only at its own wall
level, so a level mask can describe an immersed floor (the roof-aware
Berlin ground). Three formulations:

| `formulation` | stress | reference |
| :-- | :-- | :-- |
| `:local` | ``τ_w(U)\,u_i/U`` | Schumann (1975), Grötzbach (1987) |
| `:schumann` | ``⟨τ_w⟩\,u_i/⟨U⟩``, ``⟨τ_w⟩ = τ_w(⟨U⟩)`` | Schumann (1975) |
| `:moeng` | ``⟨τ_w⟩[⟨U⟩(u_i - ⟨u_i⟩) + U⟨u_i⟩]/(⟨U⟩\,|⟨\mathbf u⟩|)`` | Moeng (1984) |

Each is affine in the face velocity, ``τ_{w,i} = a u_i + b``, and is
applied implicitly,

```math
u_i ← \frac{u_i - Δt\,b/Δz}{1 + Δt\,a/Δz},
```

so the stress is evaluated with the end-of-step velocity. The explicit
form is unstable for ``Δt\,a/Δz > 2``; this one decays monotonically and
never changes sign for any Δt (tested at 100× the explicit limit for the
rough and the Spalding law). The momentum it removes is exactly
``Σ τ_{w,i} ΔA Δt`` (measured ≤ 1.3e-15 relative), and nothing else
changes: `w`, the other levels and the halo and finite-volume code are
untouched, so on a free-slip `ImpermeableWall` the wall model is the
only momentum flux through the wall face.

## Immersed-boundary walls

A direct-forcing immersed boundary (`Aeolus.IBM`) zeroes every velocity
face of a solid cell. A fluid face whose tangential neighbour is such a
face then sees, through the viscous and eddy-viscosity stencils, the
flux ``(ν + ν_{t,e})(0 - q)/Δ``: an unresolved no-slip wall a distance
``Δ`` away, with no log law. Tessicini et al. (2002) and Chester,
Meneveau & Parlange (2007) replace it with the stress of a wall law at
the first fluid point; Aeolus does the same in its equilibrium form
(`src/wall_models/immersed.jl`, P57-d, #179).

`immersed_walls(sim, sm, law)` finds every (face, wall) pair of the mask:

| face | walls | tangential partner interpolated to the face |
| :-- | :-- | :-- |
| u | ±y | w (4 faces) |
| v | ±x | w (4 faces) |
| w | ±x | v (4 faces) |
| w | ±y | u (4 faces) |

The ±z walls of u and v (ground, roofs) are the z-walls of
`apply_wall_stress!`. The law (any `AbstractWallLaw`, or a function of
the solid cell behind the wall) is evaluated at ``h = Δ_d/2``: the solid
cell's face is the wall, as for the roofs of the z-wall operator, and
the face centre is half a cell from it. Two hooks act on every step:

1. **`remove_stencil_wall_flux!`** (`body_force!`): after the tendencies
   are formed, the stencil's own wall flux
   ``(ν + ν_{t,e})(q_N - q_F)/Δ_d^2`` (``ν_{t,e}`` the mean of the four
   cell-centre ``ν_t`` around the edge) is subtracted, from the same
   start-of-step velocity and ``ν_t`` the stencil used.
2. **`apply_immersed_wall_stress!`** (`post_predictor!`, after the IBM's
   direct forcing): ``q ← q/(1 + Δt \sum_{walls} a/Δ_d)``,
   ``a = τ_w(U)/U``, implicit as for the z-walls; a corner face (two
   walls) gets the sum.

The correction of a step is therefore the modelled stress minus the
stencil stress, and the momentum the step removes at the walls is
exactly ``Σ τ_w ΔA Δt`` (measured to 1e-14 for u, v and the w faces of
an L-shaped corner, `test/era6_wall_models_immersed.jl`). With the IMEX
stepper the diffusion is implicit; the explicit removal is then
first-order in time.

**Rotated channel.** A channel at ``Re_τ = 1000`` whose walls in ``y``
are an IBM mask reaches the bulk velocity of the same channel with
z-walls and the z-wall operator to round-off (3e-15), ``U_b^+ = 19.18``
at 16 cells across ``2δ`` (first face at ``h^+ = 62.5``), 4.0 % below
Lee & Moser (2015), 19.98. With the wall model off — the stencil's
no-slip flux, as the Berlin facades had before #179 — it is 13.86. The
eddy viscosity there is the parabolic ``ν_t = κ u_τ d(1 - d/δ)`` (the
exact profile is then the log law), not an LES. The reference itself
falls below Lee & Moser as the grid is refined (``U_b^+`` = 19.38,
19.18, 18.86, 18.58, 18.10 for 12, 16, 24, 32, 48 cells): the
log-layer mismatch of a wall model matched at the first point (Kawai &
Larsson 2012), which matching away from the wall (P57-b) addresses.

## Cut-cell walls

On a cut-cell geometry (`Aeolus.CutCell`) the ghost-cell stencils
already carry a probe off the true surface along its normal.
`wall_shear_stress(u, v, w, stencils, geom, law; ν)` — a method of
`CutCell.wall_shear_stress` — evaluates the law there: the tangential
speed at the image point, at the probe distance ``h``. `law = nothing`
is the resolved ν-linear stress ``ν u_t/h``. `wall_slip_velocity` turns
the modelled stress into the slip wall value
``u_s = u_t - τ_w h/ν_{eff}`` that makes the linear ghost-cell
reconstruction (`reconstruct_wall_velocity!`) carry it; in the viscous
sublayer it is zero (no-slip).

On a channel inclined at 30° (``Re_τ = 1000``, ``Δ = δ/15``) the
plane-mean modelled stress is 0.990 ``δ\,dp/dx`` with the image point
two cells out (`probe_factor = 2`, ``h^+ = 133``); the ν-linear stress
is 0.127 of it. At the default `probe_factor = 1` the trilinear
interpolation of the concave log profile at one cell gives 0.931: for
wall modelling the probe should be built at least two cells out.

## Wall-bounded benchmark cases

`src/wall_models/channel_cases.jl` (P54-d, #174) is the shared harness
every Era VI closure is validated on. It adds no wall physics: the
wall stress is `apply_wall_stress!` above, the SGS model is whatever
`AbstractSGSClosure` is passed, and the diagnostics are Operators'
`phi_m_profile`, `total_stress_profile` and `streamwise_spectrum`.

**`half_channel_case`** — the neutral ABL of Porté-Agel, Meneveau &
Parlange (2000) and Bou-Zeid, Meneveau & Parlange (2005): a
``2πH × 2πH × H`` box, periodic in x and y, driven by the mean pressure
gradient ``-∂p/∂x/ρ = u_*^2/H`` (a body force on ``u``), free-slip top,
`RoughLogLaw(z0)` wall stress at the bottom (``z_0/H = 10^{-4}``),
inviscid, no Coriolis and **no wall-damping function**. In statistically
steady state the plane-mean momentum budget is

```math
τ_{res}(z) + τ_{sgs}(z) = u_*^2\Big(1 - \frac{z}{H}\Big), \qquad τ_{wall} = u_*^2 ,
```

and in the surface layer ``Φ_m = (κz/u_*)∂⟨U⟩/∂z = 1``.

**`channel_case`** — the smooth two-wall channel at ``Re_τ = u_τδ/ν``
(``2πδ × πδ × 2δ``, `SpaldingLaw` at both walls, forcing ``u_τ^2/δ``),
for the log-layer-mismatch study (P57-b) and the stratified and
relaminarising channels (P55-f, P59-e).

Both return a `WallBoundedCase`; `run_channel!` advances it with SSP-RK3
(`step_rk3!` now takes `step!`'s `body_force!`/`post_predictor!` hooks)
and returns time averages. `advance_channel!` records the stress the wall
operator applied in each step, with the SSP-RK3 weights 1/6, 1/6, 2/3 of
its three substages, so the plane-mean momentum budget closes per step:
``L_z Δ⟨u⟩ = (G L_z - τ_b - τ_t)Δt`` to 1e-11 relative (RK3, Euler, one
and two walls, local and Schumann).

The **staggered-consistent Φ_m** differences the plane-mean speed of two
cell centres and places the result on the w-face between them, so for
the exact log law it is ``(z/Δz)\ln[(z + Δz/2)/(z - Δz/2)] =
1 + (Δz/z)^2/12 + (Δz/z)^4/80 + …`` — `ln 3 = 1.099` at the first face;
the test checks this to 1e-12 and the observed order (2.042, 2.010,
2.003). The **stress profile** is formed with the stencils the solver
integrates (the centred flux product of `momentum_advect!`, the edge
viscosity of `add_eddy_viscosity_u!`), so its divergence equals the
plane-mean tendency to round-off (1e-12). **E11** is Parseval-normalised
(``Σ E_{11} Δk_1 = ⟨u'^2⟩`` to 1e-12).

Measured in `test/era6_neutral_abl.jl` (seed 1; about 60 s in all):

| case | quantity | measured | expected |
| :-- | :-- | :-- | :-- |
| half channel 24³, `StaticSmagorinsky(0.1)`, 8-20 H/u* | ⟨τ_w⟩ | 0.972 u*² (12 seeds: 0.961-0.991) | u*² ± 5 % |
| same | max ``\|τ_{tot} - u_*^2(1 - z/H)\|`` | 0.027 u*² (12 seeds: ≤ 0.040) | ≤ 0.1 u*² |
| then `StaticSmagorinsky(0.17)`, 25-30 H/u* | max Φ_m, first 3 faces | 1.621 (0.760, 1.134, 1.621; 12 seeds: 1.45-1.65) | > 1.3; ≈ 1.5-2 (Mason & Thomson 1992) |
| channel 16×8×16, Re_τ = 1000, `StaticSmagorinsky(0.065)`, 2-10 δ/u_τ | ⟨τ_w⟩, both walls | 0.994 u_τ² (bottom 0.977, top 1.011) | u_τ² ± 2 % |

Two expectations of the plan proved wrong and were changed, with the
tolerances kept. (1) A 10 H/u* run averaged over the last 5 is too
short: the mean wall stress over 5-10 H/u* scattered over 0.946-1.027 u*²
across 10 seeds, because the flow is still accelerating toward its
Smagorinsky equilibrium and 5 H/u* averages of a 24³ box fluctuate by a
few per cent; averaging over 8-20 H/u* gives 0.961-0.991. (2) At 24³ a
`StaticSmagorinsky(0.17)` run started from noise stays laminar-like
(``τ_w`` 0.68-0.93 u*² for 40 H/u*), so the overshoot is measured after
switching the closure of the developed C_s = 0.1 flow. In the channel,
C_s = 0.1 at 16×8×16 is likewise too dissipative (⟨τ_w⟩ 0.77-0.86 u_τ²
over 2-10 δ/u_τ); the default is the channel value 0.065 of Moin & Kim
(1982), which gives 0.994-1.007 over four seeds.

The initial state is the law's profile plus seeded uniform noise of
amplitude `noise·u*` (default 3, rms 1.7 u*); with `noise = 0.3` the
24³ half channel never becomes turbulent in 30 H/u* (``τ_w`` ≈ 0.5-0.7
u*²): the eddy viscosity damps weak noise.

`examples/validation/neutral_abl.jl` runs `StaticSmagorinsky(0.1)` at 32³
and 48³ for 50 H/u* and plots Φ_m(z) and E11(k1):
averaged over 30-50 H/u*, ⟨τ_w⟩ = 1.002 u*² (32³, 28,157 steps, 188 s)
and 1.001 u*² (48³, 45,999 steps, 904 s) — the 2-3 % deficit of the
shorter test window is the start-up transient, gone by 30 H/u*. The
Smagorinsky overshoot peaks at Φ_m = 1.58 (32³, z/H = 0.094) and 1.49
(48³, z/H = 0.062), at the third face in both, and Φ_m falls below 1
above z/H ≈ 0.25 (32³) and 0.15 (48³): the Porté-Agel et al. (2000,
Fig. 7) shape. E11 follows k₁^(-5/3) only up to k₁H ≈ 8 at 48³ and then
falls by four decades to the Nyquist wavenumber (k₁H = 24): the static
model's excess dissipation near the filter scale, which the
scale-dependent model of P54-e is meant to reduce.

Hooks reserved for later issues (documented, refusing non-default
values): `filter` (P57-b), `surface_heat_flux` (P58-e), `bottom_bc =
:noslip` and `θ_walls` (P55-f, P59-e).

The cases are idealised validation set-ups and are not coupled to the
Berlin run (a stated exception, #174): the wall physics they exercise is
`apply_wall_stress!`, which is already the default Berlin ground stress,
and their diagnostics live in Operators (infrastructure, exempt).

## Log-layer mismatch: matching height and input filtering (P57-b)

An equilibrium law fed with the velocity of the wall-adjacent cell gives
a wall stress that is consistent with that cell, but the LES velocity in
the first one or two cells is itself wrong: the eddies there are not
resolved and the SGS model is at its least accurate. The result is the
*log-layer mismatch* (LLM): the mean profile has a kink next to the wall
and is shifted against the log law above it, which shows as Φ_m ≠ 1 at
the first faces and, in a pressure-driven channel, as an error in the
bulk velocity. Two remedies, both in `src/wall_models/matching.jl`
(`MatchingInput`), change only *what goes into the law*; the stress is
still applied at the wall face by `apply_wall_stress!` (its `wind` and
`height` keywords):

* **Matching height off the first cell** (Kawai & Larsson 2012). The
  velocity is sampled at ``h_{wm}`` (`h_wm` in metres, linearly
  interpolated between cell centres, or `h_wm_cells = n`, the n-th cell
  centre at ``(n-½)Δz``) and the law is evaluated there:
  ``τ_w = τ_w(U(h_{wm}), h_{wm})``, applied along the first-cell
  velocity, ``τ_{w,i} = τ_w\,u_i/U_1``.
* **Input filtering.** In time, the exponential filter of Yang, Park &
  Moin (2017),
  ``\tilde u ← \tilde u + ε(u - \tilde u)``, ``ε = 1 - e^{-Δt/T_f}``,
  advanced once per accepted step (so all Runge-Kutta substages see the
  same input); the weight is the exact solution of
  ``T_f \dot{\tilde u} = u - \tilde u`` for a held input, so a step is
  followed by ``1 - e^{-t/T_f}`` whatever the Δt (measured 1.1e-16 over
  60 random steps). Along the wall, the 3 × 3 box of
  `Operators.box_filter!` (`n = (3, 3, 1)`, the dynamic model's test
  filter; second moment ``2Δ^2/3``), the discrete form of the 2Δ input
  filter of Bou-Zeid, Meneveau & Parlange (2005). It preserves the plane
  mean (1e-12) and has the exact transfer function
  ``(1 + 2\cos kΔ)/3`` per direction (1e-13).

**Input wind, corrected.** Before P57-b, `apply_wall_stress!` with a
`wind` used the coefficient ``τ_w(U_{in})/U_{in}``, i.e. a stress of
magnitude ``τ_w(U_{in})\,U_1/U_{in}``; now it is ``τ_w(U_{in})/U_1``,
so the magnitude is the law's at the input wind, as documented. The
#169 test encoded the old formula; both were wrong. Nothing used `wind`
before this phase.

**Channel, Re_τ = 1000** (`test/era6_wall_models_channel.jl`: 24 × 12 ×
16, `StaticSmagorinsky(0.1)`, `SpaldingLaw`, averaged over 5-8 δ/u_τ,
seed 1; Lee & Moser U_b⁺ = 19.99):

| input | U_b⁺ (error) | Φ_m(Δz) | max\|Φ_m − 1\|, 0.1δ < z < 0.3δ |
| :-- | --: | --: | --: |
| first cell | 19.845 (−0.73 %) | 1.323 | 0.323 |
| ``h_{wm} = 3Δz`` + time filter (``T_f = 0.5\,δ/u_τ``) | 19.691 (−1.49 %) | 0.969 | 0.036 |

Over seeds 1-6 first-cell Φ_m(Δz) is 1.26-1.44 and 0.97-1.18 with the
fix; max|Φ_m − 1| with the fix 0.04-0.22. The fix meets the 5 % U_b⁺
target, but **not** the expected "error reduced at least 2×": first-cell
matching is already within 1 % (−0.75 to +0.02 %), because its mismatch
is a kink in the two wall cells (``U^+ - U^+_{LM} = -0.51, +0.21``) that
nearly cancels in the bulk integral, while U_b⁺ is set by the coarse
grid's outer-layer deficit (−0.8 at the centre), which the wall-model
input cannot change. That expectation is a broken test.

**Four Reynolds numbers** (`examples/validation/log_layer_mismatch.jl`,
same grid, averaged over 10-20 δ/u_τ; reference data from the headers of
the published Lee & Moser profiles, `test/data/lee_moser_2015.jl`):

| Re_τ | U_b⁺ LM | Dean (1978) | first cell | ``h_{wm} = 3Δz`` + ``T_f`` | Φ_m(Δz) first / fix |
| --: | --: | --: | --: | --: | --: |
| 182 | 15.69 | 15.40 | +5.58 % | −1.24 % | 1.70 / 1.47 |
| 544 | 18.40 | 18.00 | −0.50 % | −1.19 % | 1.35 / 1.11 |
| 1000 | 19.99 | 19.64 | −0.59 % | −1.40 % | 1.34 / 1.04 |
| 5186 | 24.10 | 24.85 | −1.41 % | −1.45 % | 1.24 / 0.93 |

All four are within 5 % with the fix. Only at Re_τ = 182, where the first
cell centre is at y⁺ = 11 (buffer layer), does first-cell matching miss
by more than 5 %; there the fix reduces the error 4.5×. With
`DynamicSmagorinsky` (plane averaging) at Re_τ = 1000 the error goes from
−2.99 % to −1.22 %. The result is insensitive to the filter scale:
``T_f = 0.1, 0.5, 2\,δ/u_τ`` give −1.45, −1.44, −1.30 %.

**Neutral rough half channel** (24³, `RoughLogLaw`, z0/H = 1e-4, 6-12
H/u*): with ``h_{wm} = 3Δz`` and both filters, Φ_m at the first face
drops from 1.189 to 1.106 (by 0.07-0.10 over seeds 1-4), but the peak,
1.65-1.72 at z ≈ 3Δz, does not move beyond the seed scatter (1.685 →
1.694 for seed 1). That peak is the Smagorinsky overshoot of the closure
(Mason & Thomson 1992), not a wall-model error: the issue's expectation
that the input fix lowers it is a broken test, and the remedy is the
closure (P54-e, #186).

## In the Berlin case

The ground of the flagship is this operator with a `MOSTWallLaw` per
column: `z0m` from the surface's roughness (the per-class table), `L`
from the last surface step, ``u_τ`` re-evaluated at the instantaneous
first-air wind every step (see [The Berlin case](../design/berlin_case.md)).
With ``L`` and the wind frozen it reproduced the drag the example used
to carry (``u_*^2/U`` held between surface steps) exactly.

Since P57-b the wind the law is evaluated at is time-filtered by default
(`input_T_f = 10 s`, of the order of the eddy time ``z_{ref}/u_*`` of
the 6.25 m first cell). The matching height stays at the first air cell:
three cells above a street (18.75 m) lie in or above the canyon, where
no log law holds; `h_wm_cells = 3` over park and water columns is a
smoke variant. In `CaseConfig(:smoke)` (1 h, last 30 min) the street
wall stress has almost no sub-minute fluctuation for the filter to
remove (RMS of the part faster than 60 s: 2.2e-6 m² s⁻², 0.03 % of the
mean 7.8e-3), and the filter changes the ground mean ``u_*`` by +0.7 %
(0.05225 → 0.05260 m s⁻¹); it costs 0.07 ms per step on a 64 × 64
plane.

The facades are the immersed-boundary walls above (addon
`wall_model_facades`, #179): a `RoughLogLaw` with the wall (or glass)
roughness of the surface exchange's class table, the stencil's no-slip
flux removed. In the 1 h smoke run the facades carry 9.3 % of the
surface drag; the stencil drag they replace was 33× the modelled one.

## Not yet here

The thin-boundary-layer equations (P57-c) and dynamic roughness (P57-e)
add files under `src/wall_models/`. The cut-cell method is not in the
Berlin case: the case has no cut-cell provider yet (CutCell is pending
on #152).

## References

  * Spalding, D. B. (1961). A single formula for the "law of the wall".
    *J. Appl. Mech.*, 28, 455-458.
  * Schumann, U. (1975). Subgrid scale model for finite difference
    simulations of turbulent flows in plane channels and annuli.
    *J. Comput. Phys.*, 18, 376-404.
  * Grötzbach, G. (1987). Direct numerical and large eddy simulation of
    turbulent channel flows. In *Encyclopedia of Fluid Mechanics*,
    vol. 6, 1337-1391.
  * Moeng, C.-H. (1984). A large-eddy-simulation model for the study of
    planetary boundary-layer turbulence. *J. Atmos. Sci.*, 41,
    2052-2062.
  * Piomelli, U. & Balaras, E. (2002). Wall-layer models for large-eddy
    simulations. *Annu. Rev. Fluid Mech.*, 34, 349-374.
  * Tessicini, F., Iaccarino, G., Fatica, M., Wang, M. & Verzicco, R.
    (2002). Wall modeling for large-eddy simulation using an immersed
    boundary method. *Annual Research Briefs 2002*, Center for
    Turbulence Research, Stanford, 181-187.
  * Chester, S., Meneveau, C. & Parlange, M. B. (2007). Modeling
    turbulent flow over fractal trees with renormalized numerical
    simulation. *J. Comput. Phys.*, 225, 427-448.
  * Kawai, S. & Larsson, J. (2012). Wall-modeling in large eddy
    simulation: length scales, grid resolution, and accuracy. *Phys.
    Fluids*, 24, 015105.

  * Porté-Agel, F., Meneveau, C. & Parlange, M. B. (2000). A
    scale-dependent dynamic model for large-eddy simulation: application
    to a neutral atmospheric boundary layer. *J. Fluid Mech.*, 415,
    261-284.
  * Bou-Zeid, E., Meneveau, C. & Parlange, M. B. (2005). A scale-dependent
    Lagrangian dynamic model for large eddy simulation of complex
    turbulent flows. *Phys. Fluids*, 17, 025105.
  * Mason, P. J. & Thomson, D. J. (1992). Stochastic backscatter in
    large-eddy simulations of boundary layers. *J. Fluid Mech.*, 242,
    51-78.
  * Moin, P. & Kim, J. (1982). Numerical investigation of turbulent
    channel flow. *J. Fluid Mech.*, 118, 341-377.
  * Lee, M. & Moser, R. D. (2015). Direct numerical simulation of
    turbulent channel flow up to Re_τ ≈ 5200. *J. Fluid Mech.*, 774,
    395-415.
  * Kawai, S. & Larsson, J. (2012). Wall-modeling in large eddy
    simulation: length scales, grid resolution, and accuracy.
    *Phys. Fluids*, 24, 015105.
  * Yang, X. I. A., Park, G. I. & Moin, P. (2017). Log-layer mismatch
    and modeling of the fluctuating wall stress in wall-modeled
    large-eddy simulations. *Phys. Rev. Fluids*, 2, 104601.
  * Dean, R. B. (1978). Reynolds number dependence of skin friction and
    other bulk flow variables in two-dimensional rectangular duct flow.
    *J. Fluids Eng.*, 100, 215-223.
