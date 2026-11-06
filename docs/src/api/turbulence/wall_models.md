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
- **Immersed walls** (P57-d, #179): `immersed_walls` (the tangential
  walls of an IBM solid mask), `remove_stencil_wall_flux!` (a
  `body_force!` hook: the stencil's own facade flux out) and
  `apply_immersed_wall_stress!` (a `post_predictor!` hook: the modelled
  stress, implicit, corners summed). The Berlin facade drag is this.
  `immersed_wall_drag!` is the one evaluation of the law per (face,
  wall) pair (the drag rate ``a``); the TKE closure books ``a q^2`` as
  the facade shear production (#232, `attach_immersed_walls!`).
- **Cut-cell walls** (P57-d, #179): the method
  `wall_shear_stress(u, v, w, stencils, geom, law; ν)` of
  `CutCell.wall_shear_stress` (modelled stress at the image-point
  probe), `wall_slip_velocity` and `slip_wall_value` (the slip wall
  value of `reconstruct_wall_velocity!`).
- **Benchmark cases** (P54-d, #174; the test fixture
  `test/fixtures/channel_cases.jl` since ONE-l, #475): `half_channel_case` (neutral ABL
  half channel) and `channel_case` (two-wall channel at a given Re_τ),
  a `WallBoundedCase` advanced by `advance_channel!` / `run_channel!`
  (time-averaged U, Φ_m, stress budget, ⟨τ_w⟩), with the step size of
  `channel_dt`. Closure, wall law, formulation and stepper are keyword
  arguments; the diagnostics are Operators' (see [Era VI](../turbulence.md)).
- **Wall-model input** (P57-b, #183): `MatchingInput` (matching height
  `h_wm` / `h_wm_cells`, exponential time filter `T_f`, 3 × 3
  wall-parallel filter), `matching_input!` (the `wind` / `height` of
  `apply_wall_stress!`), `update_input_filter!`, `matching_distance`,
  `exponential_filter_weight`, `wall_parallel_filter!`; the case
  builders take `h_wm`, `h_wm_cells`, `filter`, `T_f`. Theory:
  [Wall models → Log-layer mismatch](../../theory/wall_models.md).
- **Non-equilibrium wall model** (P57-c, #189): `AbstractWallModel`
  (`WallModel` and `TBLEWallModel` share the operator's implicit face
  loop); `TBLEWallModel` (the thin-boundary-layer equations under every
  wall column, `terms = :equilibrium_ode | :pressure_gradient | :full`,
  dynamic ``κ_{wm}`` of Wang & Moin 2002) with `update_wall_model!`,
  `tble_kappa`, `tble_columns`; the column solver `TBLEColumn`,
  `tble_solve!`, `tble_grid`, `tble_column_momentum`; the eddy-viscosity
  generic `AbstractTBLEEddyViscosity` / `tble_eddy_viscosity` with
  `DampedMixingLength` and `NoEddyViscosity`. `channel_case(wall_model =
  :nonequilibrium, tble = (...))`. Theory:
  [Wall models → Non-equilibrium wall model](../../theory/wall_models.md).

- **Dynamic roughness** (P57-e, #190): `SubfilterHeight` /
  `subfilter_height` (sub-filter height statistics at the grid and
  test-filter scale), `DynamicRoughness` (``z_0 = \sqrt{z_{0,f}^2 +
  α^2σ_Δ^2}``, α from total-drag invariance, Anderson & Meneveau 2011),
  `dynamic_alpha`, `update_dynamic_roughness!`, `roughness_drag_residual`,
  `roughness_surface_drag`, `surface_gradient_drag`, and the parameter
  provider `with_roughness` / `dynamic_wall_laws(!)` for `RoughLogLaw` and
  `MOSTWallLaw`; `half_channel_case(; roughness)`. Theory:
  [Wall models → Dynamic roughness](../../theory/wall_models.md).
- **Wall-resolved, stratified channel** (P55-f, #191):
  `channel_case(bottom_bc = :noslip, θ_walls = (θ_b, θ_t))` with
  `stratified_channel_walls(Ri_τ)` and `channel_richardson(case)`; its
  validation against the DNS of García-Villalba & del Álamo (2011), made
  with the former `sbl` run type, is recorded in
  [Stable boundary layers](../../theory/stable_boundary_layer.md) (the
  channel cases were removed by ONE-g, #456).
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
