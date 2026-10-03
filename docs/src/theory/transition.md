# Laminar–turbulent transition

Era VI (Phase 59, #180) gives the repository one owner for the
laminar/turbulent and intermittency diagnostics, the module
`Aeolus.Transition`. The stable-boundary-layer collapse (P58-e), the
CASES-99 "hockey stick" (P58-d) and relaminarisation (P59-e) use these
functions and do not redefine them. Transition defines no Richardson
number: ``N^2`` and ``Ri_g`` come from the
[stability contract](turbulence_stability.md) and its pointwise fields
(`buoyancy_frequency_squared!`, `gradient_richardson!`).

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
