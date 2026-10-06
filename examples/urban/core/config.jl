# CaseConfig — everything that defines one Berlin run.
#
# A config is plain data: grid, geometry window, clock, dynamics
# parameters, initial state, cadences (in SECONDS, never in steps, so the
# schedule survives a change of dt or an adaptive stepper) and the
# `physics` / `options` dictionaries the coupling registry reads to decide
# which couplings run and how. Scenarios (`scenarios.jl`) are named
# presets of this struct.

"""
    CaseConfig(; kwargs...)
    CaseConfig(scenario::Symbol; kwargs...)

Configuration of a Berlin (Alexanderplatz) case. The second form starts
from a named scenario preset (see [`SCENARIOS`](@ref)) and then applies
`kwargs` on top.

Groups of fields:

* **grid** — `size`, `extent` (m), `halo`;
* **geometry** — `osm_path`, the anchor `lat0`/`lon0` of the OSM
  projection, `offset` (m, the window centre relative to the anchor),
  `min_height` (buildings lower than this are dropped), `elevation` (m
  above sea level, for the solar position), `wall_subdivisions` of the
  radiosity scene, `park_lai` (leaf-area index of park cells);
* **clock** — `date = (y, m, d)`, `start_h` (local hour at t = 0), `utc_offset` (h, local − UTC), `dt`
  (s), `nsteps`;
* **cadences** — `cadence::Dict{Symbol,Float64}`, periods in seconds.
  `:radiation`, `:soil` and `:output` are used by the initial couplings;
  a coupling may add its own key;
* **dynamics** — `ν`, `θ₀`, `g`, `f_coriolis`, `C_s`, `Pr_t`,
  `sponge_α_max`, `sponge_frac` (sponge starts at `sponge_frac·Lz`; it
  damps the velocity only), `θ_top` (the θ condition of the top face:
  `:initial_gradient` (default), the fixed-gradient top `FluxBC(Γ)` with
  the initial ∂θ/∂z at the top face; `:noflux`, a closed lid; or a
  gradient in K/m — see `core/top_boundary.jl`, #254),
  `sgs_stability` (the stability correction every eddy-viscosity `:sgs`
  provider wraps its closure in, `Operators.StabilityCorrected`, P55-e
  #187: `:brown94` (default, Brown et al. 1994), `:lilly`, `:long_tail`,
  `:short_tail`, `:most`, or `:none` for the uncorrected closure);
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
* **output** — `output_path` (`nothing` disables NetCDF output),
  `verbose`.
"""
Base.@kwdef mutable struct CaseConfig
    name::Symbol                      = :alexanderplatz
    scenario::Symbol                  = :summer
    # grid
    size::NTuple{3,Int}               = (64, 64, 32)
    extent::NTuple{3,Float64}         = (1000.0, 1000.0, 200.0)
    halo::NTuple{3,Int}               = (3, 3, 3)      # 3: WENO-Z5 moisture transport
    # geometry
    osm_path::String                  = joinpath(DATA_DIR, "alexanderplatz_real.osm")
    lat0::Float64                     = 52.5220
    lon0::Float64                     = 13.4100
    offset::NTuple{2,Float64}         = (0.0, 0.0)
    min_height::Float64               = 8.0
    elevation::Float64                = 34.0
    wall_subdivisions::Int            = 1
    park_lai::Float64                 = 2.5
    # clock
    date::NTuple{3,Int}               = (2019, 6, 29)   # clear-sky heat-wave day (#247)
    utc_offset::Float64               = 2.0             # local = UTC + 2 h (CEST)
    start_h::Float64                  = 6.0
    dt::Float64                       = 2.0
    nsteps::Int                       = 43_200
    cadence::Dict{Symbol,Float64}     = Dict(:radiation => 600.0, :soil => 30.0,
                                             :output => 1800.0)
    # dynamics
    ν::Float64                        = 2.0
    θ₀::Float64                       = 293.0
    g::Float64                        = 9.81
    f_coriolis::Float64               = 1e-4
    C_s::Float64                      = 0.1
    Pr_t::Float64                     = 1 / 3
    sgs_stability::Symbol             = :brown94
    sponge_α_max::Float64             = 0.05
    sponge_frac::Float64              = 0.75
    θ_top::Union{Symbol,Float64}      = :initial_gradient
    # initial state: the observed sounding (#247) …
    sounding::Union{Nothing,String}   = joinpath(DATA_DIR, "lindenberg_10393_20190629.csv")
    # … or, with sounding = nothing, the idealised profile
    θ_surface::Float64                = 293.0
    θ_inversion_z::Float64            = 100.0
    θ_lapse::Float64                  = 3e-3
    u0::Float64                       = 1.0
    qv0::Float64                      = 0.010
    qv_scale::Float64                 = 800.0
    Ts0::Float64                      = 285.0
    Ts_water_offset::Float64          = 5.0
    T_soil0::Float64                  = 288.0
    p_surface::Union{Nothing,Float64} = nothing         # nothing: from the sounding
    # coupling selection
    physics::Dict{Symbol,Any}         = Dict{Symbol,Any}()
    options::Dict{Symbol,Any}         = Dict{Symbol,Any}()
    # output
    canopy_top::Float64               = NaN
    output_path::Union{Nothing,String} = joinpath(URBAN_DIR, "output", "alexanderplatz.nc")
    verbose::Bool                     = true
end

"Cadence period (s) named `key` in `cfg`; errors if the config has none."
function cadence_period(cfg::CaseConfig, key::Symbol)
    haskey(cfg.cadence, key) ||
        error("CaseConfig has no cadence :$key (known: $(join(sort!(collect(keys(cfg.cadence))), ", ")))")
    return cfg.cadence[key]
end
