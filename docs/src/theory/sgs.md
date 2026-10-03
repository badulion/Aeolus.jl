# Smagorinsky-Lilly SGS closure

Aeolus closes the filtered momentum equation with the eddy-viscosity
Smagorinsky-Lilly model (see [LES filtering & closure](les_filtering.md)
for motivation). This page walks through the discrete implementation.

## The model

Local eddy viscosity is determined by the resolved strain rate:

```math
\nu_t = (C_s\,\Delta)^2\, |\bar S|, \qquad
|\bar S| = \sqrt{2\,\bar S_{ij}\,\bar S_{ij}}
```

where the strain-rate tensor is

```math
\bar S_{ij} = \frac12\!\left(\frac{\partial \bar u_i}{\partial x_j}
                           + \frac{\partial \bar u_j}{\partial x_i}\right)
```

and the filter width is

```math
\Delta = (\Delta x\,\Delta y\,\Delta z)^{1/3}.
```

For the scalar (potential temperature) the SGS diffusivity is
`κ_t = ν_t / Prₜ`, kept in its own field `sim.κt` (see [Closure
contract](#Closure-contract): other closures set `κ_t` independently).

## Discretization on the C-grid

The six independent components of the symmetric tensor `S_{ij}` live at
different places on the staggered mesh:

  * **Diagonal** `S₁₁ = ∂u/∂x`, `S₂₂ = ∂v/∂y`, `S₃₃ = ∂w/∂z` live
    naturally at cell centers (they are 2-point face → cell differences
    of the corresponding velocity components — the same stencils used
    by the divergence operator).
  * **Off-diagonal** `S_{12}`, `S_{13}`, `S_{23}` live at cell *edges*
    (x-y, x-z, y-z edges respectively).

Aeolus computes `ν_t` at cell centers. This requires interpolating the
off-diagonal components to the cell center — a standard 4-point edge
average.

The kernel `_compute_smagorinsky_νt!` reads:

```julia
S11 = (u[i+1,j,k] - u[i,j,k]) / dx           # cell center
S22 = (v[i,j+1,k] - v[i,j,k]) / dy           # cell center
S33 = (w[i,j,k+1] - w[i,j,k]) / dz           # cell center

# ∂u/∂y at cell center: average two face-to-face differences in y,
# each interpolated in x to the cell center.
du_dy = ((u[i,j+1,k] + u[i+1,j+1,k]) -
         (u[i,j-1,k] + u[i+1,j-1,k])) / (4 dy)
dv_dx = ((v[i+1,j,k] + v[i+1,j+1,k]) -
         (v[i-1,j,k] + v[i-1,j+1,k])) / (4 dx)
S12 = 0.5 * (du_dy + dv_dx)

# ... similarly for S13 and S23 ...

|S|² = 2*(S11² + S22² + S33²) + 4*(S12² + S13² + S23²)
νt[i,j,k] = (C_s * Δ)^2 * sqrt(|S|²)
```

## Applying the SGS tendency to momentum

Once `νt` is known at cell centers, the SGS contribution to the
momentum equation is

```math
\frac{\partial u_i}{\partial t} \supset
\frac{\partial}{\partial x_j}\!\left(\nu_t \frac{\partial u_i}{\partial x_j}\right).
```

Note this is the **diagonal-only** or **eddy-viscosity flux form** — we
retain the term with `∂uᵢ/∂xⱼ` and drop the transposed piece
`∂uⱼ/∂xᵢ` from the full stress divergence. For a well-resolved LES with
modest `νₜ` variation, the omitted piece is small (and it disappears
exactly for incompressible flow when `νₜ` is constant). PALM makes the
same choice.

For u-momentum at `FaceX`:

```math
\frac{\partial}{\partial x}\!\left(\nu_t \frac{\partial u}{\partial x}\right)\bigg|_{i-1/2}
= \frac{\nu_{t,i}\,(u_{i+1}-u_i)/\Delta x - \nu_{t,i-1}\,(u_i-u_{i-1})/\Delta x}{\Delta x}
```

with `νₜ` already at cell centers. The y- and z-flux terms need `νₜ`
at x-y and x-z edges respectively — again a 4-point average.

## Applying the SGS tendency to a scalar

The scalar tendency is

```math
\frac{\partial \theta}{\partial t} \supset
\frac{\partial}{\partial x_j}\!\left(\frac{\nu_t}{Pr_t}
                                     \frac{\partial \theta}{\partial x_j}\right).
```

Because `θ` is cell-centered, its diffusive flux lives at faces. The
kernel `_add_eddy_diffusion_scalar!` interpolates `νₜ` linearly from
cell centers to each face.

## Choice of coefficients

  * `Cₛ = 0.17` for isotropic free turbulence (Lilly 1967). Reduced to
    ~0.1 near walls; the constant-`Cₛ` form is used everywhere (no
    near-wall damping), with the wall stress from a
    [wall model](wall_models.md).
  * `Prₜ = 1/3` is standard for atmospheric ABL (Deardorff 1972).
    Engineering flows often use `Prₜ ≈ 0.7 – 1`.

Both are constructor parameters on
[`BoussinesqSimulation`](@ref Aeolus.Simulations.BoussinesqSimulation).

## Dynamic coefficient (Germano–Lilly)

[`compute_dynamic_smagorinsky_νt!`](@ref Aeolus.Operators.compute_dynamic_smagorinsky_νt!)
computes ``C_s`` from the resolved flow instead of prescribing it
(Germano et al. 1991, Lilly 1992). The resolved stress at a coarser
*test* filter ``\widetilde{\cdot}`` (a 3 × 3 × 3 box of cell-centred
values) is known exactly, and the Germano identity equates it to the
difference of the model at the two filter levels:

```math
L_{ij} = \widetilde{u_i u_j} - \tilde u_i \tilde u_j = C_s^2 M_{ij}, \qquad
M_{ij} = 2Δ^2\left(\widetilde{|S| S_{ij}} - α^2 |\tilde S|\, \tilde S_{ij}\right),
```

with ``α = \hat Δ/Δ``. The least-squares solution, averaged over each
horizontal level (the homogeneous directions), is
``C_s^2(z) = ⟨L^d_{ij} M_{ij}⟩ / ⟨M_{ij} M_{ij}⟩``, clipped to
``[0, 0.5^2]`` (no backscatter). `averaging = :local` uses the pointwise
ratio instead.

**Filter-width ratio.** The discrete box of three cells has second moment
``2Δ^2/3``; combined with the implicit grid filter (a top hat of width
``Δ``, second moment ``Δ^2/12``) the effective test-level width is
``\sqrt{Δ^2 + 12 \cdot 2Δ^2/3} = 3Δ``, so `filter_ratio = 3` is the
default. With 2 (the value often quoted for the trapezoidal filter) the
coefficient is too large for this filter:

| field (32³, triply periodic, ``u_{rms} = 1``) | ``α = 3`` | ``α = 2`` |
| :-- | --: | --: |
| random phases, ``E(k) ∝ k^{-5/3}`` | 0.012 | — |
| same field after ``t = 1`` (≈ 1 turnover) of LES | 0.129 | 0.220 |
| after ``t = 3`` | 0.154 | 0.267 |

The initial field is drawn by
[`isotropic_random_field!`](@ref Aeolus.Operators.isotropic_random_field!)
(Phase 54). Up to #168 it came from a test-local generator (continuous
projection, amplitude ``∝ k^{-5/3}`` per mode); that field gave 0.018,
0.129 / 0.222 and 0.153 / 0.265. The difference is the random
realisation, not the closure: with the old field the refactored
dynamic model reproduces the old numbers bit for bit.

Lilly's inertial-range value is 0.17. The random-phase field has the
right spectrum but no energy transfer to small scales, and the dynamic
procedure — which measures that transfer — correctly gives
``C_s ≈ 0``; it switches off in laminar shear too (``L_{ij}M_{ij} = 0``
for a linear velocity profile), where the constant-coefficient model
still dissipates. These are the tests in `test/era2_sgs_numerics.jl`.

**Using it.** `step!(sim, dt; sgs = DynamicSmagorinsky(ws))` replaces
the built-in closure (the eddy-viscosity terms are added where the
built-in ones are); it is bit for bit the pre-contract
`eddy_viscosity! = (νt, u, v, w) -> compute_dynamic_smagorinsky_νt!(νt,
u, v, w; workspace = ws)`, which still works. `ML.MLSGS` closures plug in
the same way. The Berlin case selects it with `physics[:sgs] = :dynamic`,
on any dynamical core.

## Lagrangian dynamic averaging (Meneveau, Lund & Cabot 1996)

The plane average of the Germano identity needs horizontally homogeneous
statistics, which an urban canopy does not have. Before P54-b (#173)
the plane average also ran over the solid cells of the immersed
buildings. [`LagrangianDynamicSmagorinsky`](@ref Aeolus.Operators.LagrangianDynamicSmagorinsky)
averages the identity backwards in time along fluid pathlines instead
(MLC96), so the coefficient is local and needs no homogeneous direction.
The averages obey

```math
\mathcal I_{LM}^{n+1}(\mathbf x) = ε\,[L^d_{ij}M_{ij}]^{n+1}(\mathbf x)
  + (1-ε)\,\mathcal I_{LM}^{n}(\mathbf x - \mathbf u^n Δt),
\qquad
\mathcal I_{MM}^{n+1}(\mathbf x) = ε\,[M_{ij}M_{ij}]^{n+1}(\mathbf x)
  + (1-ε)\,\mathcal I_{MM}^{n}(\mathbf x - \mathbf u^n Δt),
```

```math
ε = \frac{Δt/T}{1+Δt/T}, \qquad
T = θ\,Δ\,(\mathcal I_{LM}\mathcal I_{MM})^{-1/8},\ θ = 1.5, \qquad
C_s^2 = \mathcal I_{LM}/\mathcal I_{MM}.
```

This is the exponentially weighted pathline average of MLC96 eqs.
2.10–2.13, in the time discretisation of their eqs. 3.1–3.3. The details:

* The upstream values come from trilinear interpolation. Periodic
  directions wrap, and in a bounded direction the departure point is
  clamped to the boundary cell.
* ``\mathcal I_{LM}`` is clipped at 0.
* ``T`` uses the upstream memories. Where ``\mathcal I_{LM}`` has been
  clipped to 0, ``T`` is evaluated with ``\mathcal I_{LM} ≥
  10^{-8}\mathcal I_{MM}``, so the point can recover; with ``T = ∞`` it
  would freeze. Where ``\mathcal I_{MM} = 0``, ``ε = 1``.
* Initialisation (MLC96 §3): ``\mathcal I_{MM} = M_{ij}M_{ij}`` and
  ``\mathcal I_{LM} = C_0^2\mathcal I_{MM}`` with ``C_0 = 0.16``.
* ``C_s^2`` is clipped to ``[0, 0.5^2]``. No damping function is applied
  anywhere.

**One Germano kernel.** ``L^d_{ij}M_{ij}`` and ``M_{ij}M_{ij}`` come from
`_germano_identity!`. That kernel also computes the plane and local
averages of
[`compute_dynamic_smagorinsky_νt!`](@ref Aeolus.Operators.compute_dynamic_smagorinsky_νt!),
and its test filter is the stencil of
[`box_filter!`](@ref Aeolus.Operators.box_filter!). The refactor is bit
for bit: LM, MM, ``C_s^2`` and ``ν_t`` on the evolved field below are
`==` the pre-refactor values for both averagings and both filter ratios.

**Contract.** The memory advances only in `advance_sgs!`, once per
accepted step, from the state after the step. The departure point uses
the cell-centred velocity committed with the previous memory,
``\mathbf u^n``. `sgs_coefficients!` is pure: it takes ``C_s^2`` from the
committed memory and ``|S|`` from the current state. So the three RK3
substages and the CFL estimate see one coefficient, and between commits a
coefficient call costs only a strain evaluation. `sgs_state` copies the
memory, and a rejected adaptive step is rolled back bit for bit.

**Solids.** `mask` is a cell mask, e.g. `IBM.SolidMask.mask`. Its halos
are filled from the boundary conditions. The plane / local
`DynamicSmagorinsky` take the same mask.

* ``ν_t = 0`` and the memories are 0 in solid cells.
* The dynamic procedure reads the velocity with the no-slip value 0 on
  every face that touches a solid.
* The test filter runs over fluid neighbours only, renormalised by their
  number.
* A departure point whose nearest cell is solid takes the value of the
  cell itself.
* Solid corners of the trilinear stencil are dropped and the remaining
  weights renormalised.

Nothing stored inside an obstacle reaches a fluid coefficient: test 6
below sets every solid face to NaN and to 1e30 and gets the same bits.

Measured (`test/era6_lagrangian_sgs.jl`):

| test | result |
| :-- | :-- |
| relaxation to a frozen source, ``u = 0``, fixed ``T`` (40 steps) | ``\big\lvert\lvert I^n - S\rvert - (1-ε)^n\lvert I^0-S\rvert\big\rvert ≤ 1.4×10^{-15}`` |
| pathline transport, ``ε = 0``, 32³ periodic, CFL 0.4, 50 steps | domain sum to 1.4e-14, centroid moves ``\mathbf U t`` to 3.7e-13 m, max 0.911 → 0.171 (never grows) |
| laminar shear ``u = 3z`` | ``L^d_{ij}M_{ij} = 0`` exactly; ``C_s^2/C_0^2`` decays monotonically to 1.4e-8 after ``St = 45`` |
| isotropic, 32³ (field of the table above), ``t = 1`` | mean ``C_s`` 0.1228 against the plane 0.1291 (−4.8 %) |
| same, ``t = 3`` | 0.1500 against 0.1539 (−2.5 %) |
| 200 steps with the closure driving the flow | ``\mathcal I_{LM} ≥ 0``, ``\min \mathcal I_{MM} = 1.33``, ``ν_t ≥ 0``, clip fraction 0, ``\mathcal I_{LM} = 0`` in 0.59 % of cells |
| wall-mounted cube, 32×32×16 periodic channel, 200 steps | ``ν_t = 0`` in solids, no NaN/Inf, ``C_s^2 ∈ [0, 0.25]`` next to the cube (mean ``C_s`` 0.098 there, 0.152 over all fluid) |

As in MLC96 fig. 6, the Lagrangian coefficient follows the plane value in
isotropic turbulence. In laminar shear it does not drop to 0 at once, as
the plane value does. It relaxes there, with ``T`` growing as
``\mathcal I_{LM} → 0``. ``C_s^2`` and ``ν_t`` therefore reach 0 only as
``t → ∞``; the test asserts ``< 10^{-6}`` of the initial value. In the
CBC decay below the Lagrangian model is slightly closer to the
measurements than the plane model.

In the Berlin case it is the default averaging of the `:dynamic` SGS
provider. There, `CaseConfig(:smoke)` after 100 fixed steps of 2 s gives
these mean ``C_s`` over air cells:

| averaging | canopy | free stream | max ``ν_t`` (m²/s) |
| :-- | --: | --: | --: |
| Lagrangian | 0.037 | 0.033 | 0.38 |
| plane (solids now excluded) | 0.032 | 0.018 | 0.037 |

There is no clipping in either run.

## Closure contract

Era VI (#167) defines the interface between an SGS closure and the
dynamical cores **once**, in `src/operators/sgs_closures.jl`, so that
closures that need more than the velocity — buoyancy-aware mixing, a
prognostic TKE, Reynolds-stress transport, a Lagrangian-averaged dynamic
coefficient — plug into every core the same way. A closure is an
[`AbstractSGSClosure`](@ref Aeolus.Operators.AbstractSGSClosure) passed as
`sgs = closure` to `step!`, `step_rk3!`, `step_imex!`, `step_adaptive!`
and the anelastic / pseudo-incompressible `step!`.

| generic | contract |
| :-- | :-- |
| `sgs_coefficients!(f, c, sim)` | fill the eddy viscosity and diffusivity `f = SGSFields(νh, νv, κh, κv)` (isotropic: `νv === νh`, `κv === κh`; by default `sim.νt`, `sim.κt`) from the current state — `u, v, w`, `θ`, `qv` with filled halos, the mesh, `θ₀`, `g` — and the closure's committed memory. **Pure.** |
| `advance_sgs!(c, sim, dt)` | commit the memory over an accepted step of length `dt`, after the projection. Default no-op. |
| `sgs_state(c)`, `restore_sgs_state!(c, s)` | snapshot / restore the memory around a step that may be rejected. Default no-op. |
| `add_sgs_momentum!(tu, tv, tw, c, f, sim)` | the stress divergence; default the Laplacian eddy-viscosity form with `νh` |
| `add_sgs_scalar!(tϕ, ϕ, c, f, sim)` | the scalar flux divergence; default `∂ⱼ(κh ∂ⱼϕ)` |

**Why `sgs_coefficients!` must be pure.** The cores evaluate the
coefficients more often than once per step: once per RK3 substage, once
for the adaptive stepper's CFL limit, and again on every retried try of
a rejected step. Before the contract the closure hook `f(νt, u, v, w)`
was called at each of these, so a closure with memory would have
advanced it two to four times per step and never rolled it back. Under
the contract the memory changes only in `advance_sgs!`, which every core
calls exactly once per accepted step (RK3: once with `dt` after the
third substage, not three times), and `step_adaptive!` brackets each try
with `sgs_state` / `restore_sgs_state!`, so after a rejection the memory
is bit for bit that of a run that took the shorter step directly.

**κ_t is a field.** The simulations carry `κt` next to `νt`. The θ
tendency is formed with it, the CFL limit is
``\Delta t_{sgs} = C_{diff}\, Δ^2_{harm} / \max(ν_{t,\max}, κ_{t,\max})``
(rather than ``ν_{t,\max} \max(1, 1/Pr_t)``), and the IMEX implicit
operator freezes ``ν + κ_t`` for the scalars. The Smagorinsky-type
closures ([`AbstractEddyViscosityClosure`](@ref
Aeolus.Operators.AbstractEddyViscosityClosure): `StaticSmagorinsky`,
`DynamicSmagorinsky`, `ML.MLSGS`, and `FunctionSGS`, which wraps a
legacy `f(νt, u, v, w)`) set ``κ_t = ν_t/Pr_t``; they apply it as
``(1/Pr_t)\,∂_j(ν_t ∂_jθ)`` through the same kernel, the order of
operations of the pre-contract code, so the default run is unchanged bit
for bit (face averages of ``ν_t/Pr_t`` are not bitwise ``1/Pr_t`` times
face averages of ``ν_t``). A closure with its own ``κ_t`` (a stability
function, a TKE closure) gets the three-argument
[`add_eddy_diffusion_scalar!`](@ref Aeolus.Operators.add_eddy_diffusion_scalar!)
by default. qv and the passive tracers keep their current treatment
(P56-a, #177).

**One gradient stencil.** `Operators._velocity_gradient_at` returns the
full resolved gradient ``g_{ij} = ∂u_i/∂x_j`` at a cell centre (diagonal:
the face-to-face difference; off-diagonal: the centred difference over
two cells of the component averaged to the centre); the strain rate of
both Smagorinsky closures is built from it, and the Era VI closures that
need the gradient (AMD / Vreman, σ, the gradient Richardson number, TKE
shear production) read the same tensor.

**What was measured** (`test/era6_sgs_closure.jl`):

| property | result |
| :-- | :-- |
| `StaticSmagorinsky(C_s)`, a legacy `f(νt, u, v, w)` vs the built-in `C_s` run, 50 steps of 16³ decaying turbulence, `step!` / `step_rk3!` / `step_imex!` | identical `u, v, w, θ` (`==`) |
| `DynamicSmagorinsky(ws)` vs `eddy_viscosity! = (…) -> compute_dynamic_smagorinsky_νt!(…)` | identical (`==`), same `cs2` |
| commits over 6 steps: Euler / RK3 / IMEX / adaptive | 6 / 6 (18 coefficient evaluations) / 6 / 6 |
| adaptive, one forced retry | memory `(1, [dt/2])` = the run that took `dt/2` directly, state bitwise equal |
| `θ = sin(2πz/L_z)`, constant `κ₀ ≠ ν_t/Pr_t`: amplitude vs the discrete amplification factor, explicit (100 steps) / ARK2 (20 steps, 25× the explicit limit) | 1.8e-15 / 4.4e-16 relative |
| `κ_t = 5ν_t`: `dt_sgs` | `C_diff Δ²_harm / κ_{t,max}` exactly |
| random positive `κ_t`, no-flux walls, 100 steps: ``Σθ`` | conserved to 0 (Euler), 6e-14 (ARK2) relative |
| linear velocity field on (15.625, 15.625, 6.25) m: `_velocity_gradient_at` | 5.6e-17 s⁻¹ from the prescribed tensor |
| anelastic vs Boussinesq core, `DynamicSmagorinsky`, same velocity | identical `ν_t`, `κ_t` (`==`) |

In the Berlin case the SGS slot's providers are closure objects passed as
`sgs`, so `:dynamic` and `:mlsgs` now also run on the sound-proof cores
(see [The Berlin case](../design/berlin_case.md#Numerics-slots)).

## Decaying grid turbulence (Comte-Bellot and Corrsin)

The Era VI benchmark for SGS closures (Phase 54) is the wind-tunnel decay
of Comte-Bellot & Corrsin (1971, CBC): grid mesh ``M = 5.08`` cm, mean
speed ``U_0 = 10`` m/s, spectra measured at ``tU_0/M = 42, 98, 171``
(their table 3, in `test/data/cbc1971_table3.txt`; the tabulated spectra
integrate to ``\tfrac32 u'^2`` of their table 4 within 4.2, 1.3 and
0.3 %). Following Meneveau, Lund & Cabot (1996) and Kang, Chester &
Meneveau (2003) the flow is simulated in a frame moving with ``U_0``: a
triply periodic box of side ``11M`` at ``ν = 1.5×10^{-5}`` m²/s,
initialised with the station-42 spectrum
([`isotropic_random_field!`](@ref Aeolus.Operators.isotropic_random_field!),
then two cycles of evolve-and-[`rescale`](@ref Aeolus.Operators.rescale_to_spectrum!)
so the cascade is established), run to the later stations with SSP-RK3.
The LES is compared with CBC *filtered to the grid cut-off*: TKE is the
shell sum over ``k ≤ k_c = (N/2)Δk``. CBC is an experiment, not DNS; the
harness (`cbc_benchmark(closure; N)` in `test/cbc_benchmark_driver.jl`)
accepts any spectrum function, so a tabulated DNS spectrum can replace it.

Measured (three random seeds; `test/era6_cbc_decay.jl` at 32³,
`examples/validation/cbc_decay.jl` at 64³):

| ``N^3`` | closure | ``K_{98}/K_{42}`` vs CBC | ``K_{171}/K_{42}`` vs CBC | RMS ``\log_{10}E``, ``k ≤ \tfrac23 k_c`` (98 / 171) | ``E(k_c)``/CBC (98 / 171) |
| :-- | :-- | --: | --: | --: | --: |
| 32³ | dynamic (plane)     | +1.3 % | −15.7 % | 0.085 / 0.099 | 1.23 / 1.00 |
| 32³ | dynamic (Lagrangian, #173) | +0.7 % | −14.5 % | 0.076 / 0.087 | 1.18 / 1.00 |
| 32³ | Smagorinsky 0.17    | +2.5 % | −14.2 % | 0.088 / 0.097 | 1.27 / 1.08 |
| 32³ | no model            | +78.3 % | +121.1 % | 0.216 / 0.316 | 5.25 / 10.4 |
| 64³ | dynamic (plane)     | −13.4 % | −23.5 % | 0.062 / 0.118 | 0.72 / 0.61 |
| 64³ | Smagorinsky 0.17    | −14.2 % | −24.3 % | 0.065 / 0.124 | 0.67 / 0.52 |
| 64³ | no model            | +32.7 % | +15.7 % | 0.226 / 0.346 | 5.66 / 8.02 |

CI cost at 32³: about 18 s for the four closures × three seeds. The
Lagrangian model runs through `step_rk3!(sim, dt; sgs)`, so its memory is
committed once per step; it was not run at 64³.

Without a model the energy piles up at the cut-off (5–10 × CBC), which
is what the test uses to show it can fail. With either eddy-viscosity
closure the first interval is right at 32³ but the second decays too
fast: the deficit is in the energy-containing shells (LES/CBC 0.65–0.71
at ``k ≈ 0.2–0.3`` cm⁻¹ at station 171), where the measured large scales
barely decay and the ``11M`` box holds only a few dozen modes. At 64³
the closures are also too dissipative at the small scales. The
proposed ±10 % on both TKE ratios (a figure from spectral-code LES) is
therefore not met; the test asserts ±20 % at station 171 and keeps ±10 %
as a broken test, so a closure that reaches it (Lagrangian dynamic,
AMD, …: one line each in the test) shows up. Starting from the random-
phase field without the rescale cycles instead gives +21 % / −3 % at
32³ (no transfer during the first turnover).

## Limitations and planned upgrades

The current implementation is deliberately the *simplest* working
Smagorinsky-Lilly. Known deficiencies (all fixable in later phases):

  * No wall damping (Van Driest / Piomelli). Needed only when walls
    exist — Phase 1b.
  * The constant coefficient is too dissipative in transitional / stably
    stratified regimes. The dynamic coefficient below is the alternative.
  * The **stability correction** for stably stratified flow
    (Deardorff / Lilly closure with a Richardson-number reduction) is
    not applied — it will matter when we do stably stratified boundary
    layers.
  * The SGS closure itself has no wall treatment: at a free-slip z-wall
    its flux through the wall face is zero. The wall stress — smooth
    (Spalding), rough (log law) or Monin–Obukhov — is supplied by the
    wall model, `apply_wall_stress!` of `Aeolus.WallModels` (Phase 57,
    #169; see [Wall models](wall_models.md)); the Berlin ground uses it.

## References

  * Smagorinsky, J. (1963), Mon. Wea. Rev. 91, 99.
  * Lilly, D. K. (1967), *The representation of small-scale turbulence
    in numerical simulation experiments*, IBM Sci. Comp. Symp.
  * Deardorff, J. W. (1972), *Numerical investigation of neutral and
    unstable planetary boundary layers*, J. Atmos. Sci. 29, 91.
  * Moin, P. & Kim, J. (1982), *Numerical investigation of turbulent
    channel flow*, JFM 118, 341.
  * Pope, S. B. (2000), *Turbulent Flows*, ch. 13.
  * Germano, M., Piomelli, U., Moin, P. & Cabot, W. H. (1991), *A dynamic
    subgrid-scale eddy viscosity model*, Phys. Fluids A 3, 1760.
  * Lilly, D. K. (1992), *A proposed modification of the Germano subgrid-scale
    closure method*, Phys. Fluids A 4, 633.
  * Lund, T. S. (1997), *On the use of discrete filters for large eddy
    simulation*, CTR Annual Research Briefs, 83.
  * Meneveau, C., Lund, T. S. & Cabot, W. H. (1996), *A Lagrangian dynamic
    subgrid-scale model of turbulence*, J. Fluid Mech. 319, 353 (the
    stateful closure the commit contract is designed for).
  * Deardorff, J. W. (1980), *Stratocumulus-capped mixed layers derived
    from a three-dimensional model*, Boundary-Layer Meteorol. 18, 495
    (prognostic SGS TKE).
  * Comte-Bellot, G. & Corrsin, S. (1971), *Simple Eulerian time
    correlation of full- and narrow-band velocity signals in
    grid-generated, 'isotropic' turbulence*, JFM 48, 273.
  * Kang, H. S., Chester, S. & Meneveau, C. (2003), *Decaying turbulence
    in an active-grid-generated flow and comparisons with large-eddy
    simulation*, JFM 480, 129.
