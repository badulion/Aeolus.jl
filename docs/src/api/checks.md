# Checks

`Aeolus.Checks` (ONE-d, #453) runs the model's checks inside the run.
The core and the components **publish** named checks. A run's `[checks]`
section **enables** them, each required or advisory and with a
tolerance. The run writes the results to `output/results.toml`, and
`aeolus run` exits with code **3** when a required check failed. Without
`[checks]` nothing is evaluated and no `results.toml` is written.

A check measures one error value ≥ 0 and passes when the value is at
most its tolerance:

| check | published by | value | default tolerance |
| :-- | :-- | :-- | :-- |
| `finite` | the core | non-finite interior values of u, v, w, p, θ (qv, νt, κt) at the end | 0 |
| `energy_closure` | the core's energy budget | (\|closure\| + \|unattributed\|) / elapsed, W m⁻² | 1e-6 |
| `water_closure` | the water budget | the same, kg m⁻² s⁻¹ | 1e-12 |
| `momentum_closure` | the momentum budget | the larger of x and y, kg m⁻¹ s⁻² | 1e-9 |
| `tracer_closure` | the tracer budget | the same, kg m⁻² s⁻¹ per unit mixing ratio | 1e-12 |
| `<name>` | a component (`checks(c)`, `check_names(C)`) | its own | its own |
| `reference_<quantity>` | `[checks.reference]` | distance outside the reference band, or the RMS of a profile | 0 |

A closure check measures its budget: enabling it turns the budget on
([Budgets](budgets.md)), just as selecting the `budget_*` output does,
with the same requirements (`water_closure` needs moisture, and so on).
The "unattributed" part is the change that no stage entry booked:
`(:core, :unattributed)` and `(:output, :between_steps)`. A closure check
therefore also fails when something changed the air outside the
pipeline. The budgets measure the state without changing it: a run with
checks is bit for bit the run without them.

```toml
[checks]
finite = { required = true }
energy_closure = { required = true }                 # default tolerance 1e-6 W m-2
momentum_closure = { required = false, tolerance = 1e-12 }
reference_file = "data/gabls1_reference.csv"         # CSV or NetCDF, hashed under [inputs]

[checks.reference.u_star]                            # a quantity of the reference file
required = true
variable = "u_star"                                  # a time-series (scalar) or profile output variable
window = [28800.0, 32400.0]                          # the mean over hours 8-9; default: the end of the run
tolerance = 0.0                                      # widens the band [low, high]
```

The reference file is a CSV quantity table (`quantity,low,high` or
`quantity,value`, as in `examples/sbl/data/*_reference.csv`), a CSV of
profiles (`z,<quantity>,…`), or a NetCDF file
([`read_reference`](@ref Aeolus.Checks.read_reference)). The compared
variable comes from the domain's output catalogue
([Output](output.md)), whether or not `[output]` writes it. A scalar
fails outside `[low − tolerance, high + tolerance]`. A profile fails
when the RMS of model − reference at the reference heights inside the
column exceeds the tolerance.

`results.toml` holds `passed` (every required check passed), the lists
`failed_required` and `failed_advisory`, and one table per check under
`[checks.<name>]`: `required`, `evaluated`, `passed`, `value`,
`tolerance`, `units`, `description`, and the check's own numbers (the
closure, unattributed change and storage per second; the model value and
the band; the window and its step count). An advisory failure only warns.

A run that blows up records the sanity check `finite` as failed (with the
error, step and time). It does so even when only other checks were
enabled, and records every other check as not evaluated. It exits
with 1, like any failed run. A resumed run continues the window means:
they are checkpointed with the run.

```@autodocs
Modules = [Aeolus.Checks]
```
