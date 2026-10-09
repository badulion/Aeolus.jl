"""
    Aeolus.ValidationRuns

The run types of the validation cases (EX-c, #440): each is an
[`AbstractRunDriver`](@ref Aeolus.Config.AbstractRunDriver) selected by
`[run] model` and configured only through its schema, so a validation
case is a TOML file (`examples/validation/configs/`) run with

    julia --project -m Aeolus run examples/validation/configs/<case>.toml

and its numbers are diagnostics of the run (`output/results.toml`, or the
a-priori record of the `dns` run type). The scripts left in
`examples/validation/` only read those files and print or plot them.

| driver | what it builds and runs |
| :-- | :-- |
| `channel` | the wall-modelled channel LES of `WallModels` (`half_channel_case` / `channel_case`, `run_channel!`), with the streamwise spectra averaged over the statistics window |
| `column` | single-column and box cases with no 3-D dynamics, by `[column] case`: `dycoms_rf01`, `parisfog_2011`, `kessler_box`, `droplet_parcel` |
| `cbc_decay` | the Comte-Bellot & Corrsin (1971) decay benchmark ([`cbc_benchmark`](@ref)) |
| `patchy_surface` | the convective boundary layer over hot and cool strips ([`patchy_les`](@ref), [`patchy_run!`](@ref)) |
| `plant_canopy` | the steady 1-D canopy column of `PlantCanopy`: wind, or the in-canopy temperature of the leaf energy balance ([`canopy_wind_profile`](@ref), [`canopy_temperature_profile`](@ref)) |
| `dynamic_roughness` | the a-priori dynamic roughness of a staggered cube array ([`cube_array_roughness`](@ref)) |
| `dns` | the decaying-turbulence DNS of the filtered-DNS fixture ([`dns_decay`](@ref)) and its a-priori record ([`dns_apriori_finish`](@ref)) |

The harnesses these run types are built on (the CBC benchmark, the DNS
fixture, the patchy-surface LES, the RF01 column) are the ones the module
tests use: one source of truth for the test suite and the examples.

Included at the end of `Aeolus.jl`, after every export: it sees the
package's public API as `using Aeolus` does.
"""
module ValidationRuns

using ..Aeolus
using ..Aeolus: current_domain
using ..Config: Config, AbstractRunDriver, RunConfig, ConfigError, Section, Optional, OpenTable, OneOf,
                Path, section, option!, register_driver!, write_results, no_checkpoints,
                kwdef_section, kwdef_value, selected_case_sections, load_run_config
using ..IO: IO, RunFolder, CheckpointSchedule, with_monitor, output_dir, resume!, maybe_checkpoint!
using ..Fields: set!, interior_view
using ..Grids: interior
import ..Simulations
using Printf
import Random
import TOML

"The directory of the shipped validation configs, `examples/validation/configs/`."
const VALIDATION_CONFIG_DIR = normpath(joinpath(@__DIR__, "..", "examples", "validation", "configs"))

include("validation_runs/cbc.jl")
include("validation_runs/dns.jl")
include("validation_runs/channel.jl")
include("validation_runs/rf01.jl")
include("validation_runs/canopy.jl")
include("validation_runs/roughness.jl")
include("validation_runs/column.jl")
include("validation_runs/patchy_surface.jl")

# the CBC benchmark (P54-c, #168)
export CBC_M, CBC_U0, CBC_NU, CBC_STATIONS, CBC_LBOX, cbc_table, cbc_spectrum, cbc_benchmark, cbc_summary,
       cbc_reference,
       CBCDriver
# the filtered-DNS fixture and its a-priori record (P56-c, #178)
export DNS_KP, DNS_NU48, DNS_T_END, velocity_derivative_skewness, dns_start, dns_running, dns_step!,
       dns_summary, dns_decay, coarse_grain, dynamic_cs_on_coarse_grid, germano_lilly_cs,
       clark_gradient_stress, dns_checks, apriori_statistics, dns_apriori_record, write_dns_apriori,
       read_dns_apriori, ADVISORY_CHECKS, dns_apriori_finish, DNSDriver, dns_kwargs, dns_config
# the channel LES (P54-d, P57-b)
export ChannelDriver, spectrum_levels
# the column cases
export ColumnDriver, RF01_DEFAULTS, rf01_column, rf01_radiation, parcel_ou_sigma_r2
# the plant canopy column (Phase 48) and the dynamic roughness of a cube array (P57-e)
export PlantCanopyDriver, canopy_wind_profile, canopy_temperature_profile, canopy_lad,
       DynamicRoughnessDriver, staggered_cube_heights, cube_array_roughness
# the patchy-surface LES (Phase 53, #43)
export PatchSkin, PatchyLES, patchy_strip_classes, patchy_les, patchy_forcing, patchy_column_fluxes!,
       patchy_step!, patchy_heat_content, patchy_zi, patchy_wstar, patchy_sample, patchy_run!,
       patchy_amplitude, PatchySurfaceDriver

end # module
