"""
    Aeolus.RunDrivers

Run types of the example cases (EX-d, #441): each is a run driver
([`AbstractRunDriver`](@ref Aeolus.Config.AbstractRunDriver)) selected by
`[run] model` and configured only through its schema, so an example is a
TOML file (plus its input drivers), run with

    julia --project -m Aeolus run examples/<dir>/configs/<case>.toml

One submodule per driver:

  * [`AdvectionDriver`](@ref Aeolus.RunDrivers.AdvectionDriver), `"advection"`:
    a passive scalar in a prescribed (kinematic) flow;
  * [`CBLDriver`](@ref Aeolus.RunDrivers.CBLDriver), `"cbl"`: the free
    convective boundary layer with the prognostic SGS TKE closure, and the
    inflow statistics of it as a nest parent;
  * [`InflowFetchDriver`](@ref Aeolus.RunDrivers.InflowFetchDriver), `"inflow_fetch"`:
    the development length behind a synthetic inflow;
  * [`ObstacleChannelDriver`](@ref Aeolus.RunDrivers.ObstacleChannelDriver),
    `"obstacle_channel"`: a periodic channel over immersed obstacles read
    from a static driver, held at a bulk velocity;
  * [`ShearLayerDriver`](@ref Aeolus.RunDrivers.ShearLayerDriver), `"shear_layer"`:
    the Kelvin–Helmholtz billow of a stratified tanh shear layer;
  * [`FlatPlateDriver`](@ref Aeolus.RunDrivers.FlatPlateDriver), `"flat_plate"`:
    bypass transition of a flat-plate boundary layer under free-stream
    turbulence;
  * [`SGSDecayDriver`](@ref Aeolus.RunDrivers.SGSDecayDriver), `"sgs_decay"`:
    the SGS closures against filtered DNS of rotating and stratified decay.

Their numbers go to `results.toml` (and CSV files) in the run folder's
`output/`; the scripts beside the configs only read those files.
"""
module RunDrivers

include("run_drivers/advection.jl")
include("run_drivers/cbl.jl")
include("run_drivers/inflow_fetch.jl")
include("run_drivers/obstacle_channel.jl")
include("run_drivers/shear_layer.jl")
include("run_drivers/flat_plate.jl")
include("run_drivers/sgs_decay.jl")

"The `[run] model` names of the run types in this module."
const DRIVER_NAMES = ["advection", "cbl", "inflow_fetch", "obstacle_channel", "shear_layer", "flat_plate", "sgs_decay"]

end # module
