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

## In the Berlin case

The ground of the flagship is this operator with a `MOSTWallLaw` per
column: `z0m` from the surface's roughness (the per-class table), `L`
from the last surface step, ``u_τ`` re-evaluated at the instantaneous
first-air wind every step (see [The Berlin case](../design/berlin_case.md)).
With ``L`` and the wind frozen it reproduced the drag the example used
to carry (``u_*^2/U`` held between surface steps) exactly.

## Not yet here

Matching height and input filtering (P57-b), the thin-boundary-layer
equations (P57-c), facade walls (P57-d) and dynamic roughness (P57-e)
add files under `src/wall_models/`.

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
  * Lee, M. & Moser, R. D. (2015). Direct numerical simulation of
    turbulent channel flow up to Re_τ ≈ 5200. *J. Fluid Mech.*, 774,
    395-415.
