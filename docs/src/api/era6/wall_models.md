# Wall models — WallModels

`Aeolus.WallModels` (Phase 57, #169) owns every modelled wall stress of
Era VI. Theory: [Wall models](../../theory/wall_models.md).

- **Laws**: `AbstractWallLaw` with `friction_velocity`, `wall_stress`
  (kinematic, m² s⁻²) and `wall_drag` (``τ_w/U``); `SpaldingLaw`
  (smooth wall; `LEE_MOSER_2015` constants), `RoughLogLaw` and
  `MOSTWallLaw` (κ and ``ψ_m`` delegated to `SurfaceExchange`), and
  `matching_obukhov_length`.
- **Operator**: `apply_wall_stress!` on the z-walls of a `RegularGrid`,
  implicit, `:local` / `:schumann` / `:moeng`, with Bool or level masks.
  A `post_predictor!` hook; the Berlin ground drag is this operator.
- **Benchmark cases** (P54-d, #174): `half_channel_case` (neutral ABL
  half channel) and `channel_case` (two-wall channel at a given Re_τ),
  a `WallBoundedCase` advanced by `advance_channel!` / `run_channel!`
  (time-averaged U, Φ_m, stress budget, ⟨τ_w⟩), with the step size of
  `channel_dt`. Closure, wall law, formulation and stepper are keyword
  arguments; the diagnostics are Operators' (see [Era VI](../era6.md)).
- **Wall-resolved, stratified channel** (P55-f, #191):
  `channel_case(bottom_bc = :noslip, θ_walls = (θ_b, θ_t))` with
  `stratified_channel_walls(Ri_τ)` and `channel_richardson(case)`; the
  validation against the DNS of García-Villalba & del Álamo (2011) is in
  [Stable boundary layers](../../theory/stable_boundary_layer.md).
- **Flux-cooled half channel** (P58-e, #193): `half_channel_case(surface_heat_flux = Q)`
  with `MOSTFluxWallLaw`, the flux-prescribed Monin–Obukhov wall law
  that exerts no stress beyond the maximum sustainable heat flux
  (`SurfaceExchange.max_sustainable_heat_flux`; theory:
  [Maximum sustainable heat flux](../../theory/turbulence_stability.md#Maximum-sustainable-heat-flux-(P58-e))).
- **Flux-cooled open channel** (P59-e, #195):
  `channel_case(bottom_bc = :noslip, walls = :bottom, extent = (…, …, δ),
  surface_heat_flux = Q)`, the resolved stable channel of Nieuwstadt
  (2005); the relaminarisation study is in
  [Transition → Relaminarisation](../../theory/transition.md#Relaminarisation-in-strongly-stable-stratification).

```@autodocs
Modules = [Aeolus.WallModels]
```
