# The options of an urban site — everything that defines one urban run.
#
# They are plain data: grid, geometry window, clock, dynamics parameters,
# initial state, cadences (in SECONDS, never in steps, so the schedule
# survives a change of dt or an adaptive stepper) and the urban components
# it enables with their options (`components`). They are read from the
# model's resolved configuration (`site_options`, site/config_schema.jl);
# their defaults and their TOML form are the urban site's sections of the
# model's schema (CFG-c #358, ONE-e #454); the Berlin case is the TOML files
# examples/urban/berlin_*.toml.

"""
    site_options(cfg::RunConfig) -> NamedTuple

The options of the urban site of the model's resolved configuration
`cfg` (`[urban] enabled`, its one domain), what [`build_case`](@ref)
consumes: a `NamedTuple` (ONE-k, #474;
formerly the `CaseConfig` struct with Julia-side defaults and
constructors). Its values are those of the urban site's sections of the
model's schema (`site/config_schema.jl`, CFG-c #358, ONE-e #454;
[`SITE_FIELDS`](@ref) gives the key of every field and `aeolus options`
lists them); a run changes them in its TOML file or with `--set`
overrides.

Groups of fields (`site_options(cfg).<field>`):

* **grid** — `size`, `extent` (m), `halo`. The default child is
  1 km × 1 km × 400 m on 64 × 64 × 64 cells (15.6 m × 6.25 m): deepened
  from 200 m for the nest (maintainer decision of 2026-10-05, epic #246)
  so that the Fernsehturm (368 m) fits and the child top sits in the
  parent's uniform 25 m layers;
* **geometry** — `static`, the static driver the ground is read from
  (CFG-f #361; AeolusPrep.jl writes it from OpenStreetMap or GeoJSON,
  ONE-h #457), `parent_static`, the nest parent's (its land-use tiles;
  `nothing`: from `landuse` in memory), `landuse`, the parent's land-use
  table (`aeolusprep fetch-landuse` writes it), the anchor `lat0`/`lon0`
  of the projection, `offset` (m, the window centre relative to the
  anchor), `elevation` (m above sea level, for the solar position),
  `wall_subdivisions` of the radiosity scene. A driver whose grid or
  anchor differs from the case's is an error (no resampling);
* **clock** — `date = (y, m, d)`, `start_h` (local hour at t = 0), `utc_offset` (h, local − UTC), `dt`
  (s), `nsteps`;
* **cadences** — `cadence::Dict{Symbol,Float64}`, periods in seconds.
  `:radiation`, `:soil`, `:surface` (optional) and `:output` (in TOML:
  `[urban] radiation_period`, `soil_period`, `surface_period` and
  `[output] cadence`);
* **dynamics** — `ν`, `θ₀`, `g`, `f_coriolis`, `C_s`, `Pr_t`,
  `sponge_α_max`, `sponge_frac` (sponge starts at `sponge_frac·Lz`; it
  damps the velocity only; in a nested case toward the parent's mean wind
  over the child, #265), `θ_top` (the θ condition of a closed top face, not
  used by a nested or open case:
  `:initial_gradient` (default), the fixed-gradient top `FluxBC(Γ)` with
  the initial ∂θ/∂z at the top face; `:noflux`, a closed lid; or a
  gradient in K/m — see `site/top_boundary.jl`, #254),
  `sgs_stability` (the stability correction every eddy-viscosity `:sgs`
  provider wraps its closure in, `Operators.StabilityCorrected`, P55-e
  #187: `:brown94` (default, Brown et al. 1994), `:lilly`, `:long_tail`,
  `:short_tail`, `:most`, or `:none` for the uncorrected closure);
* **lateral boundaries** (`site/open_boundary.jl`) — by default the
  case is **nested** (N-berlin-nest, #265): the `:nesting` coupling runs
  an Aeolus parent (its settings in `[nesting]`) whose state
  feeds the four sides and the top; `lateral` is then not used. With the
  nest switched off (`[nesting] enabled = false`), `lateral`
  chooses them (N-open-d, #262): `:periodic` (default) or `:open` (the
  four sides and the top fed by `ProfileBoundaryData` of the geostrophic
  wind and the initial θ profile). `outflow` (`:dirichlet`, default,
  `:convective` or `:orlanski`: the outflow condition of the open or nest
  faces, N-open-c #260), `boundary_frame` (cells: buildings closer than
  this to an open face are removed; default 3) and `θ_relaxation` (default
  `true`: θ is relaxed over the velocity sponge layer toward the data's
  profile, in a nested case toward the parent's mean profile over the
  child); `analysis_buffer` (m, default [`ANALYSIS_BUFFER_DEFAULT`](@ref)):
  the columns closer than this to a lateral open face are left out of the
  diagnostics compared with observations ([`analysis_columns`](@ref)),
  where the inflow turbulence is still developing (#268);
* **initial state** — `sounding`: the radiosonde CSV the initial θ(z),
  qv(z) and surface pressure come from (default `nothing`; the Berlin
  case: the Lindenberg WMO 10393 launches of 29/30 June 2019,
  `examples/urban/data/lindenberg_10393_20190629.csv`), interpolated in
  time to the case start; see
  [`initial_θ`](@ref) and `site/reference_state.jl`); `utc_offset` (h,
  local clock = UTC + `utc_offset`; 2 = CEST) places the case start on
  the sounding's UTC time axis. `sounding = nothing` selects the
  **idealised** profile instead (unless `dynamic` names a sounding
  driver): `θ_surface`, `θ_inversion_z`,
  `θ_lapse` (θ(z) = θ_surface + max(0, z − θ_inversion_z)·θ_lapse),
  `qv0`, `qv_scale` (qv(z) = qv0·exp(−z/qv_scale)); those five fields are
  used only then. `p_surface` (Pa at the ground; `nothing` = from the
  sounding, 101325 Pa for the idealised profile; see
  [`case_surface_pressure`](@ref)). `dynamic`: the dynamic driver
  (CFG-g, #362; `nothing`, the default: none) — a sounding driver
  (AeolusPrep.jl, `aeolusprep dynamic --from-sounding`), which then
  replaces `sounding` (the same tables, read from one file), or, with the
  offline nest (`[nesting] parent_driver = "offline"`), the
  parent-made driver of the nest faces (`child_boundaries` of the
  online run); see
  [`case_sounding`](@ref). `u0`, `Ts0` and `Ts_water_offset`
  (surface temperature, water `Ts0 + offset`), `T_soil0`;
* **components** — `components::Dict{Symbol,Any}`: the urban components
  the site enables (ONE-e, #454; formerly the registry's `physics` and
  `options`), `component_name => NamedTuple` of every option of its
  section (the `default_options` with the section's values on top), in no
  order: `build_case` builds them in the order of
  [`URBAN_COMPONENTS`](@ref);
* **ledger** — `canopy_top` (m): top of the `:canopy` layer of the
  air-heat budget (`heat_budget(…; layer = :canopy)`); `NaN` (default)
  takes the tallest roof of the scene;
* **output** — the NetCDF output is the run's writer's (`[output]`:
  the variables of the site's catalogue it selects, `site/output.jl`);
  `child_boundaries` (`nothing`: off) — with the online nest, the dynamic
  driver the run writes the child's boundary slabs to at every parent
  step (CFG-g, #362), the file an offline child reads (`aeolus run` of
  the online config writes it);
  what is printed is decided by the installed logger and its level (#347,
  in TOML `[logging]`): the setup summary is `Debug`, the run's start,
  output frames (with each coupling's diagnostics) and the end-of-run
  events (run, stage timing, ledgers) are `Info`, a blowup is an
  `Error`, all structured (`subsystem = :urban`, `event = …`, the
  values as key-values; site/diagnostics.jl, #338);
* **backend** — `backend::AbstractBackend` (default `CPUBackend()`): the
  compute backend of the run (`Aeolus.Backends`, KA-a #294). The core
  coupling calls `init_threads!(cfg.backend)` before building the
  simulation, so every field is allocated on it and its kernels run on
  it; `CPUBackend(serial = true)` is the single-task reference of the
  thread-invariance tests.
"""
function site_options end

"Cadence period (s) named `key` in `cfg`; errors if the config has none."
function cadence_period(cfg::NamedTuple, key::Symbol)
    haskey(cfg.cadence, key) ||
        error("the site has no cadence :$key (known: $(join(sort!(collect(keys(cfg.cadence))), ", ")))")
    return cfg.cadence[key]
end
