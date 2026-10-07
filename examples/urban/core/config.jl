# CaseConfig — everything that defines one Berlin run.
#
# A config is plain data: grid, geometry window, clock, dynamics
# parameters, initial state, cadences (in SECONDS, never in steps, so the
# schedule survives a change of dt or an adaptive stepper) and the
# `physics` / `options` dictionaries the coupling registry reads to decide
# which couplings run and how. Its defaults and its TOML form are the
# `berlin` run-configuration schema (core/config_schema.jl, CFG-c #358);
# the scenarios are the shipped TOML files in examples/urban/configs/.

"""
    CaseConfig(; kwargs...)
    CaseConfig(scenario::Symbol; kwargs...)
    CaseConfig(cfg::RunConfig; run_output = nothing)

Configuration of a Berlin (Alexanderplatz) case, the struct
[`build_case`](@ref) consumes. Its defaults are those of the `berlin`
run-configuration schema (`core/config_schema.jl`, CFG-c #358; `aeolus
options berlin` lists them with the TOML key of every field): the first
form is the all-defaults config with `kwargs` set on top. The second
loads the shipped scenario file `examples/urban/configs/berlin_<scenario>.toml`
(see [`SCENARIO_FILES`](@ref)) and then applies `kwargs` (`physics` and
`options` merge per coupling over the file's). The third converts a
loaded [`RunConfig`](@ref Aeolus.Config.RunConfig) of the `berlin` driver.

Groups of fields:

* **grid** — `size`, `extent` (m), `halo`. The default child is
  1 km × 1 km × 400 m on 64 × 64 × 64 cells (15.6 m × 6.25 m): deepened
  from 200 m for the nest (maintainer decision of 2026-10-05, epic #246)
  so that the Fernsehturm (368 m) fits and the child top sits in the
  parent's uniform 25 m layers;
* **geometry** — `static`, the static driver the ground is read from
  (CFG-f #361, `aeolus preprocess static` writes it; `nothing`: the
  `preprocess` source is rasterised in memory instead), `parent_static`,
  the nest parent's (its land-use tiles; `nothing`: from `landuse` in
  memory), the anchor `lat0`/`lon0` of the OSM projection, `offset` (m,
  the window centre relative to the anchor), `elevation` (m above sea
  level, for the solar position), `wall_subdivisions` of the radiosity
  scene. A driver whose grid or anchor differs from the case's is an
  error (no resampling);
* **preprocess** — what the static drivers are made from: `osm_path`
  (the OpenStreetMap file), `min_height` (buildings lower than this are
  dropped), `park_lai` (leaf-area index of park cells), `landuse` (the
  parent's land-use table, `data/berlin_landuse_6km.csv`);
* **clock** — `date = (y, m, d)`, `start_h` (local hour at t = 0), `utc_offset` (h, local − UTC), `dt`
  (s), `nsteps`;
* **cadences** — `cadence::Dict{Symbol,Float64}`, periods in seconds.
  `:radiation`, `:soil` and `:output` are used by the initial couplings;
  a coupling may add its own key (in TOML: `[cadence]`, whose keys are
  the ones the couplings declare, and `[output] cadence`);
* **dynamics** — `ν`, `θ₀`, `g`, `f_coriolis`, `C_s`, `Pr_t`,
  `sponge_α_max`, `sponge_frac` (sponge starts at `sponge_frac·Lz`; it
  damps the velocity only; in a nested case toward the parent's mean wind
  over the child, #265), `θ_top` (the θ condition of a closed top face, not
  used by a nested or open case:
  `:initial_gradient` (default), the fixed-gradient top `FluxBC(Γ)` with
  the initial ∂θ/∂z at the top face; `:noflux`, a closed lid; or a
  gradient in K/m — see `core/top_boundary.jl`, #254),
  `sgs_stability` (the stability correction every eddy-viscosity `:sgs`
  provider wraps its closure in, `Operators.StabilityCorrected`, P55-e
  #187: `:brown94` (default, Brown et al. 1994), `:lilly`, `:long_tail`,
  `:short_tail`, `:most`, or `:none` for the uncorrected closure);
* **lateral boundaries** (`core/open_boundary.jl`) — by default the
  case is **nested** (N-berlin-nest, #265): the `:nesting` coupling runs
  an Aeolus parent (its settings in `options[:nesting]`) whose state
  feeds the four sides and the top; `lateral` is then not used. With the
  nest switched off (`physics = Dict(:nesting => false)`), `lateral`
  chooses them (N-open-d, #262): `:periodic` (default) or `:open` (the
  four sides and the top fed by `ProfileBoundaryData` of the geostrophic
  wind and the initial θ profile). `outflow` (`:dirichlet`, default,
  `:convective` or `:orlanski`: the outflow condition of the open or nest
  faces, N-open-c #260), `boundary_frame` (cells: buildings closer than
  this to an open face are removed; default 3) and `θ_relaxation` (default
  `true`: θ is relaxed over the velocity sponge layer toward the data's
  profile, in a nested case toward the parent's mean profile over the
  child);
* **initial state** — `sounding`: the radiosonde CSV the initial θ(z),
  qv(z) and surface pressure come from (default: the Lindenberg WMO 10393
  launches of 29/30 June 2019, `data/lindenberg_10393_20190629.csv`,
  interpolated in time to the case start; see
  [`initial_θ`](@ref) and `core/reference_state.jl`); `utc_offset` (h,
  local clock = UTC + `utc_offset`; 2 = CEST) places the case start on
  the sounding's UTC time axis. `sounding = nothing` selects the
  **idealised** profile instead: `θ_surface`, `θ_inversion_z`,
  `θ_lapse` (θ(z) = θ_surface + max(0, z − θ_inversion_z)·θ_lapse),
  `qv0`, `qv_scale` (qv(z) = qv0·exp(−z/qv_scale)); those five fields are
  used only then. `p_surface` (Pa at the ground; `nothing` = from the
  sounding, 101325 Pa for the idealised profile; see
  [`case_surface_pressure`](@ref)). `u0`, `Ts0` and `Ts_water_offset`
  (surface temperature, water `Ts0 + offset`), `T_soil0`;
* **physics** — `physics::Dict{Symbol,Any}`: for an exclusive slot,
  `slot => provider` (e.g. `:microphysics => :kessler`); for an additive
  coupling, `name => true/false`. Anything absent takes the registry
  default;
* **options** — `options::Dict{Symbol,Any}`: `coupling_name => NamedTuple`
  merged over that coupling's `default_options`;
* **ledger** — `canopy_top` (m): top of the `:canopy` layer of the
  air-heat budget (`heat_budget(…; layer = :canopy)`); `NaN` (default)
  takes the tallest roof of the scene;
* **output** — `output_path` (`nothing` disables NetCDF output; in TOML
  `[output] netcdf`, the file name, in `[output] path`, a directory: the
  run folder's `output/` under `aeolus run`, else `examples/urban/output/`);
  what is printed is decided by the installed logger and its level (#347,
  in TOML `[logging]`): the setup summary is `Debug`, the run's start,
  output frames (with each coupling's diagnostics) and the end-of-run
  events (run, stage timing, ledgers) are `Info`, a blowup is an
  `Error`, all structured (`subsystem = :berlin`, `event = …`, the
  values as key-values; core/diagnostics.jl, #338);
* **backend** — `backend::AbstractBackend` (default `CPUBackend()`): the
  compute backend of the run (`Aeolus.Backends`, KA-a #294). The core
  coupling calls `init_threads!(cfg.backend)` before building the
  simulation, so every field is allocated on it and its kernels run on
  it; `CPUBackend(serial = true)` is the single-task reference of the
  thread-invariance tests.
"""
mutable struct CaseConfig
    name::Symbol
    scenario::Symbol
    # grid
    size::NTuple{3,Int}
    extent::NTuple{3,Float64}         # 400 m deep: the nested child (#246)
    halo::NTuple{3,Int}               # 3: WENO-Z5 moisture transport
    # geometry
    static::Union{Nothing,String}
    parent_static::Union{Nothing,String}
    lat0::Float64
    lon0::Float64
    offset::NTuple{2,Float64}
    elevation::Float64
    wall_subdivisions::Int
    # the static drivers' sources (CFG-f, #361)
    osm_path::String
    min_height::Float64
    park_lai::Float64
    landuse::String
    # clock
    date::NTuple{3,Int}               # 2019-06-29: clear-sky heat-wave day (#247)
    utc_offset::Float64               # local = UTC + 2 h (CEST)
    start_h::Float64
    dt::Float64
    nsteps::Int
    cadence::Dict{Symbol,Float64}
    # dynamics
    ν::Float64
    θ₀::Float64
    g::Float64
    f_coriolis::Float64
    C_s::Float64
    Pr_t::Float64
    sgs_stability::Symbol
    sponge_α_max::Float64
    sponge_frac::Float64
    θ_top::Union{Symbol,Float64}
    # lateral boundaries (N-open-d, #262)
    lateral::Symbol
    outflow::Symbol
    boundary_frame::Int
    θ_relaxation::Bool
    # initial state: the observed sounding (#247) …
    sounding::Union{Nothing,String}
    # … or, with sounding = nothing, the idealised profile
    θ_surface::Float64
    θ_inversion_z::Float64
    θ_lapse::Float64
    u0::Float64
    qv0::Float64
    qv_scale::Float64
    Ts0::Float64
    Ts_water_offset::Float64
    T_soil0::Float64
    p_surface::Union{Nothing,Float64}  # nothing: from the sounding
    # coupling selection
    physics::Dict{Symbol,Any}
    options::Dict{Symbol,Any}
    # output
    canopy_top::Float64
    output_path::Union{Nothing,String}
    # compute backend (KA-a, #294)
    backend::AbstractBackend
end

# The defaults live in the `berlin` configuration schema
# (core/config_schema.jl), the one place they are written: `CaseConfig(;
# kw...)` is the all-defaults config with `kw` set on top.
function CaseConfig(; kw...)
    c = CaseConfig(default_run_config())
    for (k, v) in kw
        hasfield(CaseConfig, k) || throw(ArgumentError("CaseConfig has no field $k" *
            " (fields: $(join(string.(fieldnames(CaseConfig)), ", ")))"))
        setfield!(c, k, convert(fieldtype(CaseConfig, k), v))
    end
    return c
end

"Cadence period (s) named `key` in `cfg`; errors if the config has none."
function cadence_period(cfg::CaseConfig, key::Symbol)
    haskey(cfg.cadence, key) ||
        error("CaseConfig has no cadence :$key (known: $(join(sort!(collect(keys(cfg.cadence))), ", ")))")
    return cfg.cadence[key]
end
