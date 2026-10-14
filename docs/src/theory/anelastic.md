# Sound-proof cores: the weighted flux form

The anelastic core (Lipps & Hemler 1982) and the pseudo-incompressible core
(Durran 1989) replace the Boussinesq constraint ``∇·\mathbf{u} = 0`` by

```math
∇·(W\,\mathbf{u}) = 0, \qquad W = \bar ρ(z) \;\text{(anelastic)}, \quad W = \bar ρ(z)\,\bar θ(z) \;\text{(pseudo-incompressible)},
```

with a prescribed base state ``\bar ρ(z)``, ``\bar θ(z)`` (`Aeolus.Anelastic.ReferenceState`).
Every transport term is then the divergence of a ``W``-weighted flux:

```math
\frac{∂u_i}{∂t} = -\frac{1}{W}\,∂_j\!\left(W u_j u_i\right) - ∂_i φ + b\,δ_{i3}
                 + \frac{1}{W}\,∂_j\!\left(W τ_{ij}\right) + F_i,
\qquad
\frac{∂θ}{∂t} = -\frac{1}{W}\,∇·\!\left(W \mathbf{u} θ\right) + \frac{1}{W}\,∇·\!\left(W κ_t ∇θ\right).
```

In this form ``\sum W θ\,ΔV`` and ``\sum W u_i\,ΔV`` change only through the
boundaries, and the SGS stress removes the weighted kinetic energy
``\tfrac12 \sum W |\mathbf{u}|^2 ΔV``, never adds to it.

## Discretisation

``W`` depends on ``z`` only, so only the vertical fluxes carry it. On the
staggered grid the weights are stored per storage level as a cell-centre
vector ``W_c[k]`` and a bottom-face vector ``W_f[k]``
(`Anelastic.constraint_weights`):

| tendency at | vertical fluxes at | vertical part of the divergence |
| :-- | :-- | :-- |
| ``θ``, ``u``, ``v`` (centre levels ``k``) | the z faces ``k``, ``k+1`` | ``\bigl(W_f[k+1] F_{k+1} - W_f[k] F_k\bigr) / (W_c[k]\,Δz)`` |
| ``w`` (face ``k``) | the centres ``k-1``, ``k`` | ``\bigl(W_c[k] F_k - W_c[k-1] F_{k-1}\bigr) / (W_f[k]\,Δz)`` |

The horizontal parts are those of the Boussinesq operators. These are the
weights of the advection (`weighted_momentum_advect!`,
`weighted_scalar_advect!`) and of the projection, so every flux of a step is
in the same form. With ``W ≡ 1`` each weighted kernel is the Boussinesq
kernel bit for bit (`test/components_both_cores.jl`); the Boussinesq core
calls them with no weights.

Since ONE-j (#460) the sound-proof cores use this form for every flux:

* **the SGS stress and scalar flux** (new in ONE-j) — the Laplacian and the stress
  (`sgs_stress = :symmetric`) forms of every eddy-viscosity closure and of
  the diagonal (anisotropic) closures: `Operators.flux_weights(sim)`
  hands the core's weights to `add_sgs_momentum!` and `add_sgs_scalar!`.
  A closure with a full-tensor scalar flux or its own stress (the isentropic
  tensor, the Reynolds-stress closure) has no weighted form, and a
  sound-proof core refuses it (`Operators.sgs_weighted_form`);
* **a prescribed kinematic surface flux** ``F`` (`θ_flux_bottom`,
  `θ_flux_top`, the `SurfaceThetaFlux` component), unchanged: the
  weighted form ``(W_f/W_c)\,F/Δz`` is the Boussinesq ``F/Δz``, because the
  wall faces of a `ReferenceState` carry the weight of the adjacent cell
  (``W_f = W_c`` there, as its halo fill does). So
  ``\frac{d}{dt}\sum W θ\,ΔV = (W_\mathrm{bot} F_\mathrm{bot} - W_\mathrm{top} F_\mathrm{top})\,A``.

The molecular term ``ν∇^2`` stays unweighted: its ``O(∂_z \ln \bar ρ)``
correction is negligible at molecular ``ν``.

Before ONE-j the SGS fluxes were in the Boussinesq form on the sound-proof
cores. In the dry convective boundary layer of
`examples/dry_convective_bl/dry_convective_bl.toml` (32³, 30 min, anelastic,
isentropic 300 K base state) the ``\bar ρ``-weighted θ budget then missed
the surface input by ``1.4 × 10^{-3}`` of it; in the weighted form it
closes to ``2.5 × 10^{-14}``.

## Time stepping

Both cores take the same three schemes:

* **forward Euler** with the projection (`step!`);
* **SSP-RK3** (`step_rk3!`): three Euler substages of ``Δt``, each with its
  weighted projection, combined as
  ``u^{(2)} = \tfrac34 u^n + \tfrac14 E(u^{(1)})``,
  ``u^{n+1} = \tfrac13 u^n + \tfrac23 E(u^{(2)})`` (Shu & Osher 1988). A
  combination of fields with ``∇·(W\mathbf u) = 0`` keeps it, so the step
  ends at the projection's tolerance. The SGS memory is committed once per
  step. Open faces are refused on this step (their boundary buffers are not
  combined);
* **the adaptive step** (`step_adaptive!`): the step from the core's own
  CFL limits (`adaptive_dt`, with a model's component limits after the
  Coriolis limit), the snapshot of the state and the closure's memory, and
  the post-step guard (`step_guard`), which checks the core's own
  constraint ``\max |(1/W)∇·(W\mathbf u)|`` — the plain ``∇·\mathbf u =
  -w\,∂_z \ln W`` is not zero on these cores by design.

## Checkpoints

A sound-proof core is checkpointed like the Boussinesq one
(`IO.checkpoint!`, `IO.restart!`): ``u, v, w, p, θ`` and the last eddy
coefficients ``ν_t, κ_t`` with their halos, the clock, the closure's memory
and the components' states. The base state is recorded
(`IO.checkpoint_reference`: the weighting and ``\bar θ``, ``\bar ρ`` at the
centres and faces) and compared on restart like the scalar parameters: a
restart onto another base state is refused, or warned with
`allow_mismatch = true`. A restart is the uninterrupted run bit for bit on
every scheme.

## References

* Lipps, F. B. & Hemler, R. S. (1982), *J. Atmos. Sci.* 39, 2192–2210.
* Durran, D. R. (1989), *J. Atmos. Sci.* 46, 1453–1461.
* Shu, C.-W. & Osher, S. (1988), *J. Comput. Phys.* 77, 439–471.
