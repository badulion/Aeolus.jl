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

## In the Berlin case

The ground of the flagship is this operator with a `MOSTWallLaw` per
column: `z0m` from the surface's roughness (the per-class table), `L`
from the last surface step, ``u_τ`` re-evaluated at the instantaneous
first-air wind every step (see [The Berlin case](../design/berlin_case.md)).
With ``L`` and the wind frozen it reproduced the drag the example used
to carry (``u_*^2/U`` held between surface steps) exactly.

## Not yet here

Matching height and input filtering (P57-b), the thin-boundary-layer
equations (P57-c) and dynamic roughness (P57-e) add files under
`src/wall_models/`. The cut-cell method is not in the Berlin case: the
case has no cut-cell provider yet (CutCell is pending on #152).

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
  * Lee, M. & Moser, R. D. (2015). Direct numerical simulation of
    turbulent channel flow up to Re_τ ≈ 5200. *J. Fluid Mech.*, 774,
    395-415.
