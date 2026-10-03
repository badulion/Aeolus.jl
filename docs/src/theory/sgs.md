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
    ~0.1 near walls — but Aeolus currently has only periodic BCs, so
    the constant-`Cₛ` form is used.
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
| 32³ | Smagorinsky 0.17    | +2.5 % | −14.2 % | 0.088 / 0.097 | 1.27 / 1.08 |
| 32³ | no model            | +78.3 % | +121.1 % | 0.216 / 0.316 | 5.25 / 10.4 |
| 32³ | σ-model 1.35 (#184)  | +6.6 % | −11.1 % | 0.096 / 0.105 | 1.43 / 1.31 |
| 64³ | dynamic (plane)     | −13.4 % | −23.5 % | 0.062 / 0.118 | 0.72 / 0.61 |
| 64³ | Smagorinsky 0.17    | −14.2 % | −24.3 % | 0.065 / 0.124 | 0.67 / 0.52 |
| 64³ | no model            | +32.7 % | +15.7 % | 0.226 / 0.346 | 5.66 / 8.02 |

CI cost at 32³: 14 s for the three closures × three seeds.

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

## σ-model

The σ-model of Nicoud, Baya Toda, Cabrit, Bose & Lee (2011) is a
closure of the [contract](#Closure-contract),
[`SigmaModel`](@ref Aeolus.Operators.SigmaModel)`(C_σ = 1.35)`, P59-b
(#184). It builds the eddy viscosity from the singular values
``σ_1 ≥ σ_2 ≥ σ_3 ≥ 0`` of the resolved velocity gradient
``g_{ij} = ∂u_i/∂x_j``:

```math
ν_t = (C_σ Δ)^2\, D_σ, \qquad
D_σ = \frac{σ_3\,(σ_1 - σ_2)\,(σ_2 - σ_3)}{σ_1^2}, \qquad
Δ = (Δx\,Δy\,Δz)^{1/3},
```

with ``κ_t = ν_t/Pr_t``. The design goal is an operator that is zero
wherever the flow cannot be turbulent. Smagorinsky's ``|S|`` sees only
the strain, so it is on in any laminar shear. ``D_σ`` vanishes in these
cases (Nicoud et al. 2011, Sec. II):

* ``σ_3 = 0`` whenever ``g`` has rank ≤ 2: any two-dimensional flow, pure
  shear, solid-body rotation;
* ``σ_1 = σ_2`` or ``σ_2 = σ_3``: isotropic expansion or contraction,
  axisymmetric strain.

At a no-slip wall ``u, v = O(z)`` and ``w = O(z^2)``, so ``σ_1 = O(1)``,
``σ_2 = O(z)``, ``σ_3 = O(z^2)`` and ``D_σ = O(z^3)``. This is the
near-wall behaviour of the true SGS dissipation, and it comes without a
damping function. ``C_σ = 1.35`` is Nicoud et al.'s value from isotropic
turbulence.

**Computing the singular values.** ``σ_i^2`` are the eigenvalues of
``G = g^Tg``. They are found in closed form, with no LAPACK call per
cell, from the trigonometric solution of the characteristic cubic (Smith
1961; the form Nicoud et al. take from Hasan, Basri & Kopp 2010). The
cubic is written in the deviator ``G' = G - \tfrac13 \mathrm{tr}G\,I``,
with ``α_1 = \mathrm{tr}(G'^2)/6`` and ``α_2 = \det G'/2``, so neither is
formed by cancellation. ``D_σ`` is a product of eigenvalue *gaps*, and
near a double eigenvalue the cubic gives a gap only to ``\sqrt ε``,
because the arccos argument is near ±1. A plain implementation therefore
returns ``ν_t ≈ 10^{-8}(C_σΔ)^2|g|`` for axisymmetric strain instead of
zero. Two steps fix this:

1. Where the smaller gap is below 1 % of the spread, the simple
   eigenvalue is deflated. Its eigenvector is the largest cross product
   of two rows of ``G − λI``, and the close pair is recomputed from the
   2 × 2 compression of ``G`` on the orthogonal complement, whose gap is
   accurate to ``ε\|G\|`` (Kopp 2008).
2. The smallest singular value is ``σ_3 = |\det g|/(σ_1σ_2)``, which is
   exactly zero for a rank-2 ``g`` rather than the square root of a
   round-off eigenvalue.

``σ_1 − σ_2`` is formed as ``(λ_1 − λ_2)/(σ_1 + σ_2)``. Round-off can
still leave ``σ_3`` a few ulp above ``σ_2`` when the two coincide; the
operator is clipped at zero there. The gradient is
`_velocity_gradient_at`, the one stencil of the contract, so there is no
second stencil.

**What was measured** (`test/era6_sigma_sgs.jl`; KH in
`test/era6_transition_kh.jl`; CBC in `test/era6_cbc_decay.jl`):

| property | σ-model | Smagorinsky |
| :-- | :-- | :-- |
| 16³ linear fields, ``\max ν_t/(Δ^2\|g\|)``: uniform shear / solid-body rotation / isotropic expansion / axisymmetric strain (also rotated off the grid axes) | 0 / 0 / 1.7e-30 / 5.0e-16 (rotated: 3.3e-18 / 1.5e-30 / 1.1e-15) | 0.0289 / 0 / 0.0409 / 0.0409 (``C_s = 0.17``) |
| 2-D flow (16³, ``w = 0``, no ``z`` dependence) | exactly 0 | max 1.44e-3 |
| closed form vs `svdvals`, 1e4 random ``g`` over six decades | ``\max\|Δσ\|/σ_1`` = 2.5e-14 | — |
| near-degenerate ``Q\,\mathrm{diag}(a, a(1+δ), b)\,Q^T``, δ = 1e-1 … 1e-8 | ``\max\|ΔD_σ\|/σ_1 ≤`` 2.1e-15 | — |
| clip: degenerate ``g`` (two equal singular values), 1e4 | raw ``D_σ < 0`` in 431, ``\|D_σ\| ≤`` 3.8e-16 ``σ_1``, clipped to 0 | — |
| ``d\log ν_t/d\log z``, first 8 of 64 cells, Taylor wall field | 2.955 | 0.0025 |
| laminar Couette, Re = 100, ``N_z = 32``, from rest | wall stress ``νU/h`` to 2.8e-14, ``ν_t ≡ 0`` | stress excess ``(C_sΔ)^2(U/h)/ν`` = 4.5156e-2, measured 4.5156e-2 |
| decaying HIT 32³ (``ν = 10^{-4}``), exponent ``n`` of ``K ∝ t^{-n}`` on ``t ∈ [1, 3]``, seeds 1 / 2 | 0.5823 / 0.4746 | 0.5840 / 0.4848 |
| CBC 32³ (3 seeds): ``K_{98}/K_{42}``, ``K_{171}/K_{42}`` vs CBC; RMS ``\log_{10}E`` (98 / 171); ``E(k_c)``/CBC | +6.6 %, −11.1 %; 0.096 / 0.105; 1.43 / 1.31 | +2.5 %, −14.2 %; 0.088 / 0.097; 1.27 / 1.08 |
| KH 64×1×128 growth rate (no SGS 0.1820) | 0.1820 (``ν_t ≡ 0``: the layer is 2-D) | 0.1769 (−2.82 %, ``C_s = 0.1``) |
| kernel cost, 64×64×32 | 6.7 ms | 0.85 ms |

The issue expected Smagorinsky to be nonzero in all five laminar cases.
That expectation is wrong for solid-body rotation, which has no strain,
so Smagorinsky is zero there too. The test asserts that.

**Berlin.** The `:sgs` slot's provider `:sigma`
(`examples/urban/couplings/sgs_sigma.jl`) is off by default; flipping
the default from Smagorinsky is the maintainer's call. After 100 steps
of the `:smoke` window, the mean ``ν_t`` is 2.7e-2 m²/s in the canopy,
against 2.4e-2 for Smagorinsky on the same flow. Above the sponge start
it is 5.3e-4 against 2.0e-2 m²/s. The Transition indicator finds that
free stream laminar (γ = 0). It is not *exactly* parallel, though: weak
3-D disturbances from the canopy reach it, below the γ threshold, so
``σ_3 ≠ 0`` and ``ν_t`` is small rather than zero. For an exactly
parallel profile on the Berlin mesh it is zero to the bit
(`test/berlin_numerics.jl`). The smoke step costs 16.4 ms against
15.8 ms with Smagorinsky.

## Limitations and planned upgrades

The current implementation is deliberately the *simplest* working
Smagorinsky-Lilly. Known deficiencies (all fixable in later phases):

  * No wall damping (Van Driest / Piomelli) for Smagorinsky. Needed only
    when walls exist. The [σ-model](#σ-model) needs none: its ``ν_t``
    falls off as ``z^3`` at a no-slip wall (measured slope 2.955).
  * The constant coefficient is too dissipative in transitional / stably
    stratified regimes. The dynamic coefficient and the σ-model, which is
    zero in laminar shear and in 2-D flow, are the alternatives.
  * The σ-model costs about 8 times the Smagorinsky kernel (6.7 against
    0.85 ms on 64×64×32). It is zero only in *exactly* laminar
    (parallel, 2-D, pure-strain) flow; a laminar but weakly 3-D free
    stream gets a small ``ν_t`` (Berlin: 2.6 % of Smagorinsky's above
    the sponge start). It is not stability-aware.
  * The **stability correction** for stably stratified flow
    (Deardorff / Lilly closure with a Richardson-number reduction) is
    not applied — it will matter when we do stably stratified boundary
    layers.
  * No **near-surface roughness** treatment (MOST) — Phase 1b.

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
  * Comte-Bellot, G. & Corrsin, S. (1971), *Simple Eulerian time
    correlation of full- and narrow-band velocity signals in
    grid-generated, 'isotropic' turbulence*, JFM 48, 273.
  * Meneveau, C., Lund, T. S. & Cabot, W. H. (1996), *A Lagrangian
    dynamic subgrid-scale model of turbulence*, JFM 319, 353.
  * Kang, H. S., Chester, S. & Meneveau, C. (2003), *Decaying turbulence
    in an active-grid-generated flow and comparisons with large-eddy
    simulation*, JFM 480, 129.
  * Deardorff, J. W. (1980), *Stratocumulus-capped mixed layers derived
    from a three-dimensional model*, Boundary-Layer Meteorol. 18, 495
    (prognostic SGS TKE).
  * Nicoud, F., Baya Toda, H., Cabrit, O., Bose, S. & Lee, J. (2011),
    *Using singular values to build a subgrid-scale model for large eddy
    simulations*, Phys. Fluids 23, 085106 (the σ-model).
  * Hasan, K. M., Basri, A. H. & Kopp, J. (2010), as cited by Nicoud et
    al. (2011) for the closed-form eigenvalues; Smith, O. K. (1961),
    *Eigenvalues of a symmetric 3 × 3 matrix*, Commun. ACM 4, 168.
  * Kopp, J. (2008), *Efficient numerical diagonalization of Hermitian
    3 × 3 matrices*, Int. J. Mod. Phys. C 19, 523.
