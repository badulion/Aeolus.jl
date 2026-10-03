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

## Non-equilibrium wall model: the thin-boundary-layer equations (P57-c)

An equilibrium law assumes the stress is constant between the wall and
the matching point, so the wall stress follows the matching-point speed
and nothing else. The non-equilibrium model of Balaras, Benocci &
Piomelli (1996) and Wang & Moin (2002) instead integrates the
thin-boundary-layer equations (TBLE) on an embedded wall-normal grid
under every wall column,

```math
\frac{∂u_i}{∂t} + \mathrm{conv}_i = -\frac{∂p}{∂x_i}
  + \frac{∂}{∂y}\left[(ν + ν_{t,wm})\frac{∂u_i}{∂y}\right],
\qquad u_i(0) = 0,\quad u_i(h_{wm}) = u_i^{LES}(h_{wm}),
```

with the eddy viscosity of Cabot & Moin (2000)

```math
ν_{t,wm} = κ_{wm}\,y\,u_τ\left(1 - e^{-y^+/A^+}\right)^2, \qquad A^+ = 17,
```

(``κ_{wm}u_τ(y + z_0)`` over a rough wall). `TBLEWallModel` is an
`AbstractWallModel`: the upper boundary condition is the velocity
sampled by P57-b's `MatchingInput` (matching height, time and
wall-parallel filters), and the stress is applied by the same implicit
face loop as the equilibrium `WallModel`. The eddy viscosity is a
generic (`AbstractTBLEEddyViscosity`, `tble_eddy_viscosity`) so P58 can
add buoyancy by a subtype; the test file adds a channel mixing length
that way.

`terms` selects the equation:

| `terms` | equation |
| :-- | :-- |
| `:equilibrium_ode` | ``∂_y[(ν + ν_t)∂_y u_i] = 0``, solved directly (the ODE of the equilibrium models) |
| `:pressure_gradient` | unsteady, with ``∂p/∂x_i`` (the mean forcing plus the resolved ``p`` at the matching cell) |
| `:full` | plus ``u∂_x u_i + v∂_y u_i + w∂_y u_i`` from the neighbouring columns (Park & Moin 2014): first-order upwind, ``w`` from continuity |

**Discretisation.** 30 nodes with geometric stretching (ratio 1.2; the
first node at ``8.5·10^{-4}h``), finite volumes, backward Euler (any Δt)
and one tridiagonal solve per component. The flux of a segment is
``G(u_{j+1} - u_j)`` with the conductance ``G = [∫dy/(ν + ν_t)]^{-1}``
integrated exactly (4-point Gauss–Legendre), so the steady unforced
solution is the quadrature of the equilibrium ODE at the nodes whatever
the grid, and with ``ν_t ≡ 0`` the scheme is exact for quadratic
profiles. The wall stress is the first segment's flux carried to the wall
with the wall value of ``∂τ/∂y = ∂p/∂x``. ``ν_t`` depends on the
column's own ``u_τ``: each step is iterated (secant on
``u_τ = |τ_w(u_τ)|^{1/2}``, each iterate a full implicit solve).

**Coupling.** In every substage `apply_wall_stress!` applies the
columns' current stress ``\boldsymbol τ_w``, split as
``τ_{w,i} = a u_i + b_i`` with the implicit part
``a = \max(\boldsymbol τ_w·\hat{\mathbf u}, 0)/U`` along the resolved
first-cell velocity and the explicit rest ``b_i`` (a pressure gradient
can turn the stress away from the flow). `update_wall_model!` advances
the columns once per accepted step with the end-of-step velocity and
pressure. The first use primes them with the equilibrium solution.

### Measured (`test/era6_wall_models_tble.jl`)

| test | result |
| :-- | :-- |
| equilibrium vs `SpaldingLaw()` (κ = 0.41), ``u_τ`` at ``h^+`` = 30 / 300 / 3000 | −3.0 % / −0.003 % / +0.22 % |
| equilibrium ODE vs adaptive quadrature of ``U = u_τ^2∫dy/(ν+ν_t)`` | ≤ 7e-11 |
| Stokes' first problem (``ν_t ≡ 0``), ``τ_w = νU/\sqrt{πνt}`` for ``t`` = 10…1000 ``Δy^2/ν`` | error ≤ 2.7e-3, ratio 4.00-4.19 under halving |
| Poiseuille (``ν_t ≡ 0``, stretched grid) | profile 1e-10, ``τ = τ_w + y\,∂p/∂x`` to 1e-12 |
| column momentum, 200 random steps (`:pressure_gradient`, `:full`) | ≤ 6e-16 |
| channel Re_τ = 1000, 24×12×16, Smagorinsky 0.1, ``h_{wm} = 3Δz``, time filter, `:pressure_gradient` | U_b⁺ 19.253 (−3.68 % vs Lee & Moser), ⟨τ_w⟩ 1.054 |
| … the same with `:full` / `:equilibrium_ode` (example runs) | −3.38 % / −1.58 % (the equilibrium `SpaldingLaw`: −1.49 %) |
| ``∂p/∂x`` tripled (1-D resolved-channel reference, ``h_{wm} = 0.1δ``): ``τ_w`` half-way to ``3τ_0`` | reference 5.960, TBLE 5.965, equilibrium 6.255 δ/u_τ |
| … rise of ``τ_w`` at 0.25 δ/u_τ | reference 0.072, TBLE 0.071, equilibrium 0.051 ``u_τ^2`` |

The response test follows He & Seddighi (2013) qualitatively: after a
step increase of the flow rate the wall stress first rises in a thin
Stokes-like layer driven by the pressure gradient, before the core
accelerates. The TBLE sees the pressure gradient and reproduces that
early rise (within 1.4 %); the equilibrium law sees only the matching
speed and lags it.

**Expectation that proved wrong: within 1 % of Spalding at h⁺ = 30.**
At ``h^+ = 30`` the matching point is in the buffer layer, where the
van Driest-damped viscosity (A⁺ = 17) and Spalding's fit are two
different profiles: −3.0 % (A⁺ = 19: −6.6 %; A⁺ = 25: −14 %). In the
log layer they agree to 0.2 %. Kept as a broken test.

**Dynamic κ (Wang & Moin 2002).** `dynamic_kappa = :total` makes the
TBLE eddy viscosity at ``h_{wm}`` equal the LES's effective one there,
(resolved + SGS stress)/(resolved mean shear), in the plane mean;
`:sgs` matches the SGS viscosity alone (Wang & Moin's form). A first
version matched the *stress* instead; that is degenerate: in a
statistically steady flow both sides equal ``τ_w + h\,∂p/∂x`` by the
momentum balance whatever κ is, and κ drifted to the clamp. With the
viscosity matching, the channel above gives κ_wm = 0.27 (averaged over
5-8 δ/u_τ, not fed back), close to ``0.41(1 - h/δ) = 0.26`` — the
outer-layer decrease of the stress that the mixing-length viscosity
lacks — and to Wang & Moin's ≈ 0.2. Fed back into `:pressure_gradient`,
which has no resolved convection to carry the difference, it lowers the
viscosity near the wall as well: U_b⁺ +6.5 % at 8 δ/u_τ and +15.9 % over
16-24 δ/u_τ (κ_wm 0.27, ⟨τ_w⟩ 0.99); with `:full` +7.4 % at 8 δ/u_τ. So
the default is `:off` (κ_wm = κ of the law), and
`κ_feedback = false` reports the dynamic value without using it.

**Not modelled.** No Monin–Obukhov correction in ``ν_{t,wm}`` (neutral).

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

Since P57-c the option `wall_model = :nonequilibrium` of the
`surface_exchange` coupling replaces the ground's `MOSTWallLaw` by a
`TBLEWallModel` (default stays `:equilibrium`): a 30-point column under
every ground and roof column, rough (the surface's own ``z_{0m}``),
driven by the same time-filtered first-air-cell wind and by the resolved
pressure gradient at the first air level (one-sided next to a building),
with `tble_terms = :pressure_gradient`. It is neutral: the Monin–Obukhov
correction of the equilibrium law is not carried into the TBLE. A smoke
variant runs it. Measured on the flagship (`CaseConfig(:summer)`,
64 × 64 × 32, 100 steps after 300 steps of spin-up): the coupling's own
stages cost 15.6 ms per step more (9.4 % of the 166.5 ms equilibrium
step; the whole step measured 3.3 % slower on a loaded machine), so it
is not proposed as the default (the issue's bar was 5 %). On the 1690
street columns the TBLE and the equilibrium stress at the same input
differ in sign on 11.1 % and in sign or by more than 30 % on 77.3 %
(median ratio 1.03, interquartile 0.48-2.16; mean 4.5e-3 against
3.0e-3 m² s⁻²): the resolved pressure gradient at the first air level
of a street canyon (median 3.3e-3, 90th percentile 7.4e-3 m s⁻²) times
the 3.1 m matching height is of the order of the stress itself, which
is exactly the non-equilibrium the law cannot see.

## Not yet here

Facade walls (P57-d) and dynamic roughness (P57-e) add files under
`src/wall_models/`; buoyancy in the TBLE eddy viscosity is P58.

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
  * Balaras, E., Benocci, C. & Piomelli, U. (1996). Two-layer
    approximate boundary conditions for large-eddy simulations.
    *AIAA J.*, 34, 1111-1119.
  * Cabot, W. & Moin, P. (2000). Approximate wall boundary conditions in
    the large-eddy simulation of high Reynolds number flow. *Flow
    Turbul. Combust.*, 63, 269-291.
  * Wang, M. & Moin, P. (2002). Dynamic wall modeling for large-eddy
    simulation of complex turbulent flows. *Phys. Fluids*, 14,
    2043-2051.
  * Park, G. I. & Moin, P. (2014). An improved dynamic non-equilibrium
    wall-model for large eddy simulation. *Phys. Fluids*, 26, 015108.
  * He, S. & Seddighi, M. (2013). Turbulence in transient channel flow.
    *J. Fluid Mech.*, 715, 60-102.
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
