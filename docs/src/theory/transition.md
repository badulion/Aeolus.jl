# Transition and wall-bounded flow

Era VI resolves the near-wall layer instead of modelling it wherever a
case is small enough to afford it: laminar and transitional channels,
the wall-resolved stratified channel, and the laminar checks of the
turbulence closures. This page collects the wall treatment those runs
share. Wall-*modelled* runs (the Berlin case among them) keep the
free-slip wall and add the wall-model stress on top.

Phase 59 (#180) also gives the repository one owner for the
laminar/turbulent and intermittency diagnostics, the module
`Aeolus.Transition`. The stable-boundary-layer collapse (P58-e), the
CASES-99 "hockey stick" (P58-d) and relaminarisation (P59-e) use these
functions and do not redefine them. Transition defines no Richardson
number: ``N^2`` and ``Ri_g`` come from the
[stability contract](turbulence_stability.md) and its pointwise fields
(`buoyancy_frequency_squared!`, `gradient_richardson!`).

## Walls

### The no-slip / moving-wall condition

[`NoSlipWall(u_wall, v_wall)`](@ref Aeolus.BoundaryConditions.NoSlipWall)
is the single no-slip wall type. On the staggered grid the horizontal
velocities `u @ FaceX`, `v @ FaceY` are cell-centred in ``z``, so the wall
lies half a cell below the first interior value ``u_1``. The halo is the
odd reflection about the wall velocity,

```math
u_{1-h} = 2u_\mathrm{wall} - u_h, \qquad h = 1,\dots,H_z,
```

so the face average ``\tfrac12(u_0 + u_1)`` seen by every stencil is
exactly ``u_\mathrm{wall}`` and a linear profile through the wall is
extended exactly. The viscous operator in the wall cell becomes

```math
\nu\,\frac{u_2 - u_1}{\Delta z^2} - \nu\,\frac{2(u_1 - u_\mathrm{wall})}{\Delta z^2},
```

the standard second-order ghost-cell treatment (Ferziger & Perić 2002,
§8.10). The vertical velocity and every cell-centred field (``p, θ, ν_t``)
are treated exactly as by the free-slip `ImpermeableWall`: ``w = 0`` on the
wall face with an antisymmetric halo, mirror (Neumann) for scalars. The
pressure problem therefore does not change, and the cosine-FFT solve
stays exact.

A no-slip wall can face a free-slip one, which gives an open channel:

```julia
FieldBoundaryConditions(:periodic, :periodic; bottom = :noslip, top = :wall)
FieldBoundaryConditions(:periodic, :periodic; bottom = :noslip, top = NoSlipWall(U, 0))  # Couette lid
```

A time-dependent wall (Stokes' second problem) reassigns `sim.bcs` with
the new wall velocity before each step.

### Wall stress

`wall_shear_stress(sim; side = :bottom, component = :u)` is a method of
the cut-cell generic [`wall_shear_stress`](@ref Aeolus.CutCell.wall_shear_stress),
not a new name. It returns the plane mean of ``ν\,∂u/∂z`` across the wall
face, formed from the same halo the viscous operator uses:
``ν\,2(u_1 - u_\mathrm{wall})/Δz`` on a no-slip wall and exactly zero on a
free-slip one. With periodic ``x, y`` and ``w = 0`` on the walls, advection
and the pressure gradient drop out of the plane-integrated momentum
budget, which is

```math
\frac{d}{dt}\int_0^{L_z} \bar u\,dz = τ(\mathrm{top}) - τ(\mathrm{bottom}) + G L_z .
```

### Implicit diffusion with a moving wall

A moving wall makes the discrete diffusion operator affine,
``∇·(ν∇φ) = L_0 φ + g``, where ``g`` is the wall term. The CG solve of
[`solve_implicit_diffusion!`](@ref Aeolus.TimeIntegration.solve_implicit_diffusion!)
iterates with the stationary wall (``L_0``, symmetric positive definite),
using [`homogeneous_bcs`](@ref Aeolus.BoundaryConditions.homogeneous_bcs),
and adds ``Δt\,g`` to the right-hand side. ARK2 stages solve for the state,
so the wall term is kept. Rosenbrock stages solve for increments, whose
wall value is zero, so it is dropped. `:wall` runs have no wall term and
take the same code path as before (`homogeneous_bcs(bcs) === bcs`).

The places that special-case a wall were all audited. A missed one would
silently fall back to free slip:

| Site | Treatment of `NoSlipWall` |
| :--- | :--- |
| `fill_halos!` (`_fill_z!`) | new wall-pair methods; the `ImpermeableWall` pair keeps its own |
| `FieldBoundaryConditions`, `nest_bcs` | `:noslip` symbol, `bottom`/`top` keywords; a mixed z pair must be two walls, anything else is an `ArgumentError` |
| pressure solver choice (`_default_solver`) | non-periodic → `CosineFFTPoissonSolver`, unchanged |
| IMEX implicit diffusion | affine wall term (`homogeneous_bcs`) |
| IMEX pinned `w` row (`_constraint_index`) | **behaviour change:** was "`bottom` or `top` is an `ImpermeableWall`", now `is_wall(bcs.bottom)`. The pinned row is the bottom wall face `Hz + 1`; the top wall face sits in the halo, so only the bottom decides. The old symbol constructor could not build a case where the two differ |
| `apply_cutcell_velocity_bc!` | acts on apertures only; the wall comes through `fill_halos!` |
| anelastic / pseudo-incompressible cores | through `fill_halos!`; only z-periodic is rejected |

### Verification

All runs below are on `(4, 4, Nz)` grids with the full 3-D
fractional-step driver (`step!`, ``Δt = 0.3\,Δz^2/ν``), in
`test/era6_walls.jl`.

| Flow | Reference | Measured |
| :--- | :--- | :--- |
| Couette, Re = 100, Nz = 16, from rest | ``u = Uz/h``, ``τ = νU/h`` | ``‖u - Uz/h‖_∞ = 2.7\times10^{-15}U``; ``τ = νU/h`` to ``10^{-15}`` |
| Plane Poiseuille, Nz = 16/32/64 | ``u = \frac{G}{2ν} z(h-z)``, ``u_b = Gh^2/(12ν)`` | L2 order 2.0000 / 2.0000; ``u_b`` off by ``4.9\times10^{-4}`` at Nz = 64; ``τ_b = Gh/2`` to ``10^{-10}`` |
| Stokes I, Nz = 128, ``t = 0.1h^2/ν`` | ``U\,\mathrm{erfc}(z/2\sqrt{νt})`` | ``L_∞`` error ``2.3\times10^{-5}U``; wall stress ``-1.7839`` vs ``-ν U/\sqrt{πνt} = -1.7841`` |
| Stokes II, ``ω = 2π``, 6th period | decay length ``δ = \sqrt{2ν/ω}`` | ``δ`` fitted to ``2.8\times10^{-4}``; phase lag ``z/δ`` to 2 % |
| 3-D random flow, lid moving in ``x`` and ``y`` | plane momentum budget | residual ``8.7\times10^{-18}`` per step |
| IMEX Couette (ARK2, Rosenbrock2, ``Δt ≈ 13×`` explicit limit) | explicit steady state | ``1.1\times10^{-14}U``, ``2.6\times10^{-15}U`` |

``\mathrm{erfc}`` is `GasOptics.erfc_series` (one source of truth). The
Stokes I domain is ``2h`` deep, so the semi-infinite solution is the exact
reference: the top's image term is ``\mathrm{erfc}(2h/\sqrt{4νt}) ≈ 8\times10^{-6}``.
In a domain only ``h`` deep, the top wall reflects ``2.5\,\%\,U`` by
``t = 0.1h^2/ν``. That 2.5 % is a finite-domain effect of the reference,
not a solver error. A `:wall` run (explicit, ARK2 and Rosenbrock2, SGS on)
is bitwise identical to the code before the change. That comparison
against the previous `master` was made by hand. The suite does not
automate it. It checks only that `:wall` keeps the original `fill_halos!`
methods and that `_constraint_index` and `homogeneous_bcs` leave it
unchanged.

### The Berlin case

The Berlin ground is not a `NoSlipWall`. It is a rough, wall-modelled
surface: the MOST stress ``ρu_*^2`` is applied as an implicit drag over a
free-slip `:wall`, see [The Berlin case](../design/berlin_case.md). A
resolved no-slip wall at ``Δx = 15.6`` m would put the first velocity
point ``Δz/2 ≈ 3`` m from the wall, with a viscous sublayer of
``ν/u_* ∼ 10^{-4}`` m. `BoundaryConditions` is exempt infrastructure in
the coverage test, so no Berlin list changes.

## Shear instability and intermittency

### The tanh shear layer

The parallel shear layer ``U(z) = U\tanh(z/δ)`` is inviscidly unstable to
waves ``e^{iα(x - ct)}``. Its most unstable wavenumber is ``αδ = 0.4446``,
with growth rate ``σ = αc_i = 0.1897\,U/δ`` (Michalke 1964). With a
stable stratification ``N^2(z) = J (U/δ)^2 \operatorname{sech}^2(z/δ)``,
the local gradient Richardson number is
``Ri_g(z) = J\cosh^2(z/δ)``. Its minimum, at the centre, is the bulk
parameter ``J``. For this profile the neutral curve is
``J_c(α) = α^2 (1 - α^2)`` (Drazin 1958), with maximum ``J_c = 1/4`` at
``αδ = 1/\sqrt 2``. This agrees with the Miles–Howard theorem: a
parallel flow with ``Ri_g ≥ 1/4`` everywhere is linearly stable (Miles
1961; Howard 1961).

`test/era6_transition_kh.jl` checks these results with the Boussinesq
core in 2-D (``N_y = 1``), with free-slip walls at ``z = ±L_z/2``
(``≥ 8.9δ``) and ``Re = Uδ/ν = 1000``. The seed is
``w' = 10^{-4} U \sin(αx)\,e^{-z^2/δ^2}``. The base state is held against
its own viscous diffusion by the body force ``-ν∇^2(\text{base})``, which
is the frozen parallel flow that the linear theory assumes. The time
stepping is projected SSP-RK3 (`step_rk3!` with `body_force!`) at CFL 0.6.

| case | grid | measured | reference |
| :-- | :-- | :-- | :-- |
| ``J = 0``, ``αδ = 0.4446`` | 128 × 1 × 256 | ``σ = 0.1864\,U/δ`` (``R^2 = 0.99995``) | 0.1897 inviscid (Michalke 1964): 1.76 % below |
| ``J = 0``, ``αδ = 0.7071`` | 128 × 1 × 256 | ``σ = 0.1396\,U/δ`` | — (sets the e-folding time ``τ_0 = 7.2\,δ/U``) |
| ``J = 0.15``, ``αδ = 0.7071`` | 128 × 1 × 256 | ``σ = 0.0391\,U/δ`` | unstable: ``J < J_c = 1/4`` |
| ``J = 0.30``, ``αδ = 0.7071`` | 128 × 1 × 256 | ``A(5τ_0)/A(τ_0) = 0.223``, fitted ``σ = -0.054`` | stable: ``Ri_g > 1/4`` everywhere |
| ``J = 0``, ``αδ = 0.4446``, no SGS | 64 × 1 × 128 | ``σ = 0.1820`` | quoted |
| the same, Smagorinsky ``C_s = 0.1`` | 64 × 1 × 128 | ``σ = 0.1769`` (−2.8 %) | quoted |

The 1.76 % shortfall at ``J = 0`` is the sum of the viscous reduction at
``Re = 1000`` and the discretisation error of the second-order operators
at ``Δx = 0.11δ``. At ``64 × 128`` (``Δx = 0.22δ``) the shortfall grows
to 4.1 %, so most of it is numerical. The Smagorinsky closure removes
another 2.8 %, because its ``ν_t`` acts on the laminar shear of the base
flow. Comparing it with the σ closure, which is zero for a pure shear,
is P59-b's job.

### The turbulent fraction γ(z)

A cell is turbulent when the vorticity fluctuation exceeds a fraction of
the mean vorticity magnitude of its level,

```math
\frac{|ω'|}{⟨|ω|⟩} > \text{threshold}, \qquad ω' = ω - ⟨ω⟩,
```

and ``γ(z)`` is the fraction of the level's fluid cells that are
turbulent (`turbulent_fraction`). This is the plane-averaged indicator
of Brandt, Schlatter & Henningson (2004). ``ω`` is computed at the cell
centre with the same velocity-gradient stencil as the Smagorinsky
strain rate. A laminar parallel shear flow has a uniform ``ω`` in each
plane, so ``γ = 0`` exactly. The default threshold 0.2 marks almost all
of a developed isotropic field: if ``ω'`` were a Gaussian vector, only
about 1 % of cells would fall below ``0.2⟨|ω|⟩ = 0.32σ``. Measured on a
decaying 32³ random solenoidal field, ``γ ≥ 0.984`` on every level. On a
synthetic field that is random in one half of the domain and at rest in
the other, ``γ`` = 0.513–0.515. The excess over 0.5 is the vortex sheet
at the two cuts, about 1.8 cell columns each. `laminar_mask!` is the
complement within the fluid, and solid cells are never turbulent.
`normalization = :volume` divides by the domain-mean ``|ω|`` instead.
Use it when the levels of interest have very different vorticity levels.

In the Kelvin–Helmholtz billow of `examples/transition/kelvin_helmholtz.jl`
(``J = 0.05``, ``Re = 1000``, free base state), ``γ`` of the layer
``|z| ≤ 4δ`` stays exactly 0 through the linear stage. It rises once the
billow rolls up (0.78 at ``t = 60\,δ/U``, when ``A = 0.075U``) and peaks
at 0.92 at saturation (``t = 75``):

![rms w](../assets/transition/kh_amplitude.svg)
![γ of the shear layer](../assets/transition/kh_gamma.svg)
![spanwise vorticity at t = 60, 90, 140](../assets/transition/kh_vorticity.png)

### Time-series intermittency

The intermittency factor is the fraction of time a signal spends above a
threshold (Townsend 1976): `intermittency_factor(series; threshold)`
counts samples, and `intermittency_factor(t, series; threshold)` weights
each sample by the interval it starts. An on/off signal with duty cycle
0.3 gives 0.3 exactly.

### Flux Richardson number, buoyancy Reynolds number, V_TKE

```math
Ri_f = -\frac{B}{P}, \qquad B = \frac{g}{θ_0}⟨w'θ'⟩, \qquad
P = -⟨u'w'⟩\partial_z U - ⟨v'w'⟩\partial_z V
```

(Ellison 1957). Its irreversible part is the mixing efficiency (Ivey &
Imberger 1991). `flux_richardson` uses the resolved plus the SGS fluxes
(``τ_{i3} = -2ν_t S_{i3}``, ``⟨w'θ'⟩_{\rm sgs} = -(ν_t/Pr_t)\partial_zθ``)
and matches the hand values of an analytic field to ``10^{-10}``. In the
billow example, the cumulative ``-∫B\,dt/∫P\,dt`` of the layer is about
0.15 through the linear stage, 0.345 at saturation and 0.394 at
``t = 160\,δ/U``:

![cumulative Ri_f](../assets/transition/kh_rif.svg)

The buoyancy Reynolds number is ``Re_b = ε/(νN^2)`` (Gibson 1980; Smyth &
Moum 2000). The turbulence velocity scale of Sun et al. (2012) is
``V_{TKE} = \sqrt{(σ_u^2 + σ_v^2 + σ_w^2)/2}``. The grid Reynolds number
``UΔ/(ν + ν_t)`` and the fraction of cells with ``Ri_g > 1/4``
(`supercritical_fraction`, which counts cells of a `gradient_richardson!`
field) are the transition diagnostics of the Berlin
[`turbulence_diag` addon](../design/berlin_case.md).

### Growth-rate fit

`growth_rate(t, A)` fits ``\ln A = σt + c`` by least squares. Without an
explicit window it picks the longest run of samples that meets two
conditions: ``R^2 ≥ 0.999``, and the rates of the run's three thirds
agree to 1 %. ``R^2`` alone is not enough, because it grows with the span
of ``\ln A``. A window that includes the seed adjustment and the onset of
saturation still passes 0.999, but its rate is biased: 0.1859 instead of
0.1864 in the KH test, and 2.8 % low on a synthetic saturating series,
where the thirds criterion is 0.3 % low.

## Flat-plate transition

P59-d (#185) adds the spatially developing zero-pressure-gradient
flat-plate boundary layer: the Blasius solution, the fringe region that
makes such a layer possible on the `x`-periodic FFT grid, a free-stream
top boundary, free-stream turbulence, and the integral diagnostics that
locate transition. Everything lives in `Aeolus.Transition`, except the
damping profile (`Operators`) and the top boundary (`BoundaryConditions`).

### The Blasius layer

[`blasius_solution`](@ref Aeolus.Transition.blasius_solution) integrates
``f''' + \tfrac12 f f'' = 0``, ``f(0) = f'(0) = 0``, ``f'(∞) = 1`` with
classical RK4 (12 000 steps on ``0 ≤ η ≤ 12``) and finds ``s = f''(0)``
by a secant iteration on ``f'(η_{\max}; s) - 1``. Five iterations give
``f''(0) = 0.3320573362`` (Howarth 1938: 0.332057), the displacement
constant ``∫(1 - f')dη = 1.7207876575``, ``θ`` constant ``0.6641146724
= 2f''(0)`` to 1e-10 (the momentum-integral identity, checked by an
independent Simpson quadrature), ``η_{99} = 4.90999`` and
``H = 2.59110``. [`blasius_profile`](@ref Aeolus.Transition.blasius_profile)
is the cubic Hermite interpolant of the table (``f'``, ``f''`` and
``f''' = -\tfrac12 f f''`` are all tabulated, so each interpolant is C¹),
and [`blasius_velocity`](@ref Aeolus.Transition.blasius_velocity) returns
``u = U f'``, ``w = \tfrac12\sqrt{νU/x}\,(ηf' - f)``.

### The fringe region

A spatially developing layer grows downstream, so a periodic domain needs
a region that returns the outflow to the inflow. The fringe of
Nordström, Nordin & Henningson (1999) adds
``-λ(x)(\mathbf u - \mathbf u_{\rm target})`` with

```math
λ(x) = λ_{\max}\Big[S\Big(\frac{x - x_{\rm start}}{Δ_{\rm rise}}\Big)
                    - S\Big(\frac{x - x_{\rm end}}{Δ_{\rm fall}} + 1\Big)\Big],
\qquad
S(x) = \frac{1}{1 + \exp\big(\frac{1}{x - 1} + \frac{1}{x}\big)}\ (0 < x < 1),
```

``S = 0`` below and ``1`` above (their eq. 2.3). The rate comes from the
`Operators.DampingProfile` with a `NordstromRamp`, the same type, rate
function and kernel (`_add_damping!`) that the Rayleigh sponge
[`add_sponge!`](@ref Aeolus.Operators.add_sponge!) uses with its
`SinSquaredRamp`: one damping-profile implementation for every
relaxation term. `add_sponge!` is bit for bit unchanged on all three mesh
types (asserted against the verbatim Phase-7 kernel; the Berlin smoke run
gives an identical state with either kernel).

The target is the Blasius layer at the
[`fringe_distance`](@ref Aeolus.Transition.fringe_distance) — the
distance from the leading edge of the upstream periodic image inside the
fringe — so the profile the fringe releases is already developing at the
right rate and its ``w`` satisfies continuity. A target frozen at the
inflow ``x_{in}`` released a layer with ``H = 2.37`` instead of 2.59 at
the inflow, and ``c_f`` stayed 4–5 % low for half the plate.

### The free-stream boundary

A closed lid cannot hold a growing layer: the free stream must
accelerate by ``Δδ^*/H`` and the layer feels a favourable pressure
gradient. On the 2-D test grid (``H = 16δ^*_0``) a free-slip lid gave
``c_f`` 16–28 % above Blasius, even normalised by the local edge
velocity. [`FreeStreamBoundary`](@ref Aeolus.BoundaryConditions.FreeStreamBoundary)
is the far-field condition of the spatial DNS of Spalart (1988) and
Brandt et al. (2004): zero shear for ``u, v`` and a prescribed
transpiration ``w = W(x)`` on the top face. The pressure problem stays a
Neumann problem (the face value enters the divergence of the top cell),
so the cosine FFT solver remains exact, provided the net transpiration
vanishes. [`blasius_freestream`](@ref Aeolus.Transition.blasius_freestream)
prescribes the displacement velocity
``W_∞ = U\,dδ^*/dx = 0.8604\,U/\sqrt{Re_x}`` and returns the plate's
outflow through a bump confined to the fringe rise, upstream of the
plateau where the velocity is pinned.

The momentum budget of one explicit step closes exactly:
``\frac{d}{dt}∫u\,dV = ∫F_{\rm fringe}dV - τ_w A - ∮u\,w_{\rm top}dA``,
residual 5e-12 (2e-13 relative) with the lid and with the free-stream
top.

### Laminar plate

[`flat_plate_setup`](@ref Aeolus.Transition.flat_plate_setup) assembles
the case in units of the inflow ``δ^*_0`` and ``U_∞``. In 2-D at
``Re_{δ^*_0} = 200`` (256 × 64 cells on 512 × 16 ``δ^*_0``, plate
``Re_x`` = 1.35e4 to 1.0e5), started from a uniform stream and run for
1200 RK3 steps (12 s), the layer develops from the fringe inflow to
``c_f/c_{f,\rm Blasius} ∈ [0.9982, 1.0001]`` and ``H ∈ [2.574, 2.583]``
over the middle 60 % of the plate; the target is recovered at the fringe
exit to 5.2e-4 ``U``. The issue's σ-model variant of this check waits for
P59-b (#184), which has not merged.

### Free-stream turbulence and bypass transition

[`FreeStreamDisturbance`](@ref Aeolus.Transition.FreeStreamDisturbance)
represents isotropic turbulence by random Fourier modes
(Jacobs & Durbin 2001; Brandt et al. 2004) with the von Kármán spectrum
``E(κ) = \tfrac23 a (κL)^4/(b + (κL)^2)^{17/6}\,Lq``. The velocity is the
grid curl of a vector potential sampled on the cell edges, so its
discrete divergence is zero to round-off (3.9e-16 measured); the energy
is scaled to ``q = \tfrac32 (Tu\,U)^2`` and the spectrum length
calibrated so the discrete mode set has the requested ``L_{11}``
(measured on a 512 × 32 × 32 box with 400 modes: Tu 0.0497 for 0.05,
``L_{11}`` 1.042 for 1.0 with seed 1; one realisation scatters with the
seed — 0.82–1.43 over seeds 1–16, standard deviation 0.14 — and the
ensemble mean over those 16 seeds is 1.027, which is what the test
asserts). The disturbances enter through the fringe
target as Taylor's frozen turbulence
([`disturbance_fringe_update`](@ref Aeolus.Transition.disturbance_fringe_update)),
kept out of the inflow layer by a damping of the vector potential.

The onset is located by
[`transition_onset`](@ref Aeolus.Transition.transition_onset): the
minimum of ``c_f`` upstream of the point from which ``c_f`` stays 10 %
above laminar. In the coarse 3-D sanity test (96 × 24 × 48,
``Re_{δ^*_0} = 300``) Tu = 6 % raises the outlet ``c_f`` to 3.47 times
Blasius while the undisturbed plate stays within 1.5 %.

The T3A-like example (`examples/transition/flat_plate_t3a.jl`, Tu = 3.3 %
as in Roach & Brierley 1992) runs 288 × 32 × 48 cells on
600 × 36 × 24 ``δ^*_0`` (``Re_x`` = 3.0e4 to 1.9e5) without an SGS model —
a Smagorinsky ``ν_t`` from the laminar wall shear alone would be several
times ``ν`` and keep the layer laminar.

Measured (1500 RK3 steps at ``Δt = 1\,δ^*_0/U``, statistics over the
last 600; 332 s, 0.22 s per step on 4 threads):

| quantity | result | reference |
| :-- | :-- | :-- |
| free-stream Tu (streamwise) | 3.29 % at the inflow, 2.66 % at ``Re_x = 10^5``, 2.77 % at the plate end | 3.3 % requested |
| onset (minimum of ``c_f``) | ``x = 54\,δ^*_0``, ``Re_x = 4.66\times10^4`` | — |
| ``Re_θ`` at onset, measured ``θ`` | 190 | Mayle (1991) 190 (+0 %), Abu-Ghannam & Shaw (1980) 200 (−5 %) |
| ``Re_θ`` at onset, Blasius ``θ`` at that ``Re_x`` | 143 | −25 % / −29 % |
| downstream ``c_f/(0.0576\,Re_x^{-1/5})``, ``Re_x`` 1.2e5–1.8e5 | 1.26–1.39 (mean 1.30) | issue target: within 20 % — **not met** |
| downstream ``H`` | 1.41–1.45 | ≈ 1.4 |

The onset is within the ±30 % band of both correlations whichever ``θ`` is
used (the free-stream turbulence thickens the pre-transitional layer, so
the measured ``θ`` at onset is 33 % above Blasius). The turbulent ``c_f``
is 26–39 % above the correlation. The coarse, model-free calculation
over-predicts it, and the excess falls with resolution (1.20–1.26 with
``N_z = 96``, ``Δz = 0.25\,δ^*_0``, in a 1500-step trial run). The
short plate (a virtual-origin effect) is not the reason: against the
``Re_θ`` form ``0.024\,Re_θ^{-1/4}`` (Kays & Crawford) at the measured
``Re_θ`` = 437–578 the ratio is 1.33–1.45.

![c_f of the T3A-like plate](../assets/transition/t3a_cf.svg)

The boundary-layer integrals of a disturbed layer are taken to its
[`boundary_layer_edge`](@ref Aeolus.Transition.boundary_layer_edge): the
fringe-fed free stream carries a ≈ 0.5 % mean deficit, and integrating
to the top inflated ``θ`` by 50 % at the inflow.

### The Berlin case

Stated exception (#185): the Berlin domain is laterally periodic with
geostrophic forcing and has no inflow direction, so a fringe, a
free-stream top and an inflow generator have no place in it; inflow
boundaries arrive with nesting and ERA5 driving (Phase 68). Transition
stays coupled through the `turbulence_diag` addon (P59-c), and the
Rayleigh sponge of every Berlin step now takes its rate from the shared
damping profile (state bit-identical to the Phase-7 kernel in the smoke
run). No Berlin list changed.

## References

* Brandt, L., Schlatter, P. & Henningson, D. S. (2004). Transition in
  boundary layers subject to free-stream turbulence. *J. Fluid Mech.*,
  517, 167-198.
* Drazin, P. G. (1958). The stability of a shear layer in an unbounded
  heterogeneous inviscid fluid. *J. Fluid Mech.*, 4, 214-224.
* Ellison, T. H. (1957). Turbulent transport of heat and momentum from an
  infinite rough plane. *J. Fluid Mech.*, 2, 456-466.
* Gibson, C. H. (1980). Fossil temperature, salinity, and vorticity
  turbulence in the ocean. *Marine Turbulence*, Elsevier, 221-257.
* Howard, L. N. (1961). Note on a paper of John W. Miles. *J. Fluid
  Mech.*, 10, 509-512.
* Ivey, G. N. & Imberger, J. (1991). On the nature of turbulence in a
  stratified fluid. Part I. *J. Phys. Oceanogr.*, 21, 650-658.
* Michalke, A. (1964). On the inviscid instability of the hyperbolic-
  tangent velocity profile. *J. Fluid Mech.*, 19, 543-556.
* Miles, J. W. (1961). On the stability of heterogeneous shear flows.
  *J. Fluid Mech.*, 10, 496-508.
* Smyth, W. D. & Moum, J. N. (2000). Length scales of turbulence in
  stably stratified mixing layers. *Phys. Fluids*, 12, 1327-1342.
* Sun, J., Mahrt, L., Banta, R. M. & Pichugina, Y. L. (2012). Turbulence
  regimes and turbulence intermittency in the stable boundary layer
  during CASES-99. *J. Atmos. Sci.*, 69, 338-351.
* Townsend, A. A. (1976). *The Structure of Turbulent Shear Flow*, 2nd
  ed. Cambridge University Press.
* Ferziger, J. H. & Perić, M. (2002). *Computational Methods for Fluid
  Dynamics*, 3rd ed., Springer, §8.10.
* Pope, S. B. (2000). *Turbulent Flows*, Cambridge University Press, §7.1.
* Schlichting, H. (1979). *Boundary-Layer Theory*, 7th ed., McGraw-Hill,
  ch. V (Stokes' first and second problems).
* Abu-Ghannam, B. J. & Shaw, R. (1980). Natural transition of boundary
  layers — the effects of turbulence, pressure gradient, and flow
  history. *J. Mech. Eng. Sci.*, 22, 213-228.
* Blasius, H. (1908). Grenzschichten in Flüssigkeiten mit kleiner
  Reibung. *Z. Math. Phys.*, 56, 1-37.
* Howarth, L. (1938). On the solution of the laminar boundary layer
  equations. *Proc. R. Soc. Lond. A*, 164, 547-579.
* Jacobs, R. G. & Durbin, P. A. (2001). Simulations of bypass
  transition. *J. Fluid Mech.*, 428, 185-212.
* Kays, W. M. & Crawford, M. E. (1993). *Convective Heat and Mass
  Transfer*, 3rd ed., McGraw-Hill.
* Mayle, R. E. (1991). The role of laminar-turbulent transition in gas
  turbine engines. *J. Turbomach.*, 113, 509-537.
* Nordström, J., Nordin, N. & Henningson, D. (1999). The fringe region
  technique and the Fourier method used in the direct numerical
  simulation of spatially evolving viscous flows. *SIAM J. Sci.
  Comput.*, 20(4), 1365-1393.
* Roach, P. E. & Brierley, D. H. (1992). The influence of a turbulent
  free-stream on zero pressure gradient transitional boundary layer
  development. Part I. In *Numerical Simulation of Unsteady Flows and
  Transition to Turbulence*, Cambridge University Press, 319-347.
* Spalart, P. R. (1988). Direct simulation of a turbulent boundary
  layer up to R_θ = 1410. *J. Fluid Mech.*, 187, 61-98.
