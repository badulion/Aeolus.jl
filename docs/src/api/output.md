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
the variable carries `cell_methods`.

```toml
[output]
variables = ["u", "v", "w", "theta", "theta_profile", "theta_mean", "budget_energy"]
cadence = 600.0          # s, the records
series_cadence = 60.0    # s, the time series (0: every step)
averaging = "instant"    # or "mean"
```

The default set is `u`, `v`, `w`, `theta` and the time series of the
domain means of u, v, w and θ. A `budget_*` variable switches on the
budget it writes ([Budgets](budgets.md)). Its terms are written as the
group `budget_<q>__<stage>__<name>` (cumulative since the start), with
`__storage` and `__closure`.

```@autodocs
Modules = [Aeolus.Output]
```
