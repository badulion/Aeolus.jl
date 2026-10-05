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
- **Cut-cell walls** (P57-d, #179): the method
  `wall_shear_stress(u, v, w, stencils, geom, law; ν)` of
  `CutCell.wall_shear_stress` (modelled stress at the image-point
  probe), `wall_slip_velocity` and `slip_wall_value` (the slip wall
  value of `reconstruct_wall_velocity!`).
- **Benchmark cases** (P54-d, #174): `half_channel_case` (neutral ABL
  half channel) and `channel_case` (two-wall channel at a given Re_τ),
  a `WallBoundedCase` advanced by `advance_channel!` / `run_channel!`
  (time-averaged U, Φ_m, stress budget, ⟨τ_w⟩), with the step size of
  `channel_dt`. Closure, wall law, formulation and stepper are keyword
  arguments; the diagnostics are Operators' (see [Era VI](../era6.md)).
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

```@autodocs
Modules = [Aeolus.WallModels]
```
