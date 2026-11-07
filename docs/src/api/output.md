# Output

`Aeolus.Output` (ONE-c, #452) is the one output writer of the model. It
replaces three writers: the `simulation` run type's `fields_<step>.nc`,
the urban case's NetCDF file and the SBL run type's `series.csv`.

Every component publishes an **output catalogue**, a list of
[`OutputVariable`](@ref Aeolus.Output.OutputVariable)s. Each variable has
a name, units, dimensions (`:xyz`, `:xy`, `:z`, `:k` — a spectrum over the shell wavenumbers —, `:scalar`) and a getter.
The model's catalogue is the core's variables plus every enabled
component's ([`domain_catalogue`](@ref Aeolus.Components.domain_catalogue)),
and `[output] variables` selects from it. Selecting a variable that no
enabled component publishes is a configuration error.

[`OutputWriter`](@ref Aeolus.Output.OutputWriter) writes the selection
into one NetCDF-4 file (`output/output.nc` by default) as two record
streams:

- the **records** (unlimited dimension `time`): 3-D fields, surface maps,
  profiles, and per-record scalars, written every `[output] cadence`;
- the **time series** (unlimited dimension `time_series`): scalars such as
  the domain means and the budgets, written every `series_cadence`.

With `averaging = "mean"` every value is the time mean over its interval.
The writer accumulates it every step, weighted by the step length, and
the variable carries `cell_methods`. A checkpoint stores the open
intervals ([`output_means`](@ref Aeolus.Output.output_means)), so a run
resumed from it writes the same means as the uninterrupted run.

```toml
[output]
variables = ["u", "v", "w", "theta", "theta_profile", "theta_mean", "budget_energy"]
cadence = 600.0          # s, the records
series_cadence = 60.0    # s, the time series (0: every step)
averaging = "instant"    # or "mean"
```

The default set is `u`, `v`, `w`, `theta` and the time series of the
domain means of u, v, w and θ.

## Temperatures (#536)

The prognostic θ is a potential temperature. The absolute temperatures
are ``T = θ\,Π̄``, with the Exner function ``Π̄(z)`` of the domain's
thermodynamic reference
([`thermodynamic_reference`](@ref Aeolus.Components.thermodynamic_reference),
[`reference_exner`](@ref Aeolus.Components.reference_exner)): the hydrostatic
state of a Boussinesq domain's initial θ profile from its surface
pressure (`[dynamics] p_surface`, else the dynamic driver's, else
1013.25 hPa; PALM's `hyp`), an urban site's own reference, or the anelastic
core's base state. It is the same reference the tiled surface and the
column radiation convert their fluxes with. (A buoyancy-free Boussinesq
run whose θ is not positive has none; selecting a temperature there is a
configuration error.) All are in K; a CF
`standard_name` is written where one applies.

| name | dims | published by | |
| :-- | :-- | :-- | :-- |
| `temperature` | 3-D | core | ``θ Π̄_k`` (`air_temperature`) |
| `temperature_profile`, `temperature_mean` | z, scalar | core | its horizontal and domain means |
| `surface_theta` | scalar / 2-D | `[surface] model = "most"` / `"tiles"` | the surface potential temperature: the prescribed ``θ_s`` (named `surface_temperature` before #536); the tiles' area-weighted ``T_{s,i}/Π̄(z_{s,i})`` |
| `surface_temperature` | scalar / 2-D | `"most"` / `"tiles"` | the absolute surface temperature: ``θ_s Π̄_s``; the tiles' area-weighted skin temperature (`surface_temperature`) |
| `surface_temperature_<class>` | 2-D | `"tiles"` | the skin temperature of the street, roof, park, water or wall tile; NaN where the column has none |
| `theta_2m`, `temperature_2m` | scalar / 2-D | `"most"` / `"tiles"` | the screen level: ``θ_{2m}`` and ``θ_{2m}\,Π̄(2\,\mathrm{m})`` (`air_temperature`); for the tiles area-weighted over the tiles |

The tiles' 2 m value follows PALM's `calc_2m_temperature` per tile:
Monin–Obukhov similarity from the tile's own surface layer when its
forcing level is above 2 m, else linear interpolation between the
tile's surface and the level centres; a tile's surface is at its
displacement height ``z_{s,i}``. Details in
[the parent design page](../design/berlin_parent.md#Surface-and-screen-level-temperatures-(#536)).

Selecting `surface_temperature` where no surface publishes it is refused
with a message that names `surface_theta`: the name was given a new
meaning in #536, without a deprecation period. A `budget_*` variable switches on the
budget it writes ([Budgets](budgets.md)). Its terms are written as the
group `budget_<q>__<stage>__<name>` (cumulative since the start), with
`__storage` and `__closure`.

```@autodocs
Modules = [Aeolus.Output]
```
