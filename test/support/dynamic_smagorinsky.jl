# The dynamic Smagorinsky coefficient of a simulation's current velocity,
# shared by test/era2_sgs_numerics.jl and validation/era2_sgs_numerics.jl.

using Aeolus
using Statistics: mean

"Plane-mean C_s, the workspace (with `cs2`) and ν_t of the dynamic Smagorinsky model on `sim`'s velocity."
function _dynamic_cs(sim; kw...)
    ws = DynamicSmagorinskyWorkspace(sim.mesh)
    νt = Aeolus.Fields.Field(CellCenter(), sim.mesh)
    for f in (sim.u, sim.v, sim.w)
        fill_halos!(f, sim.bcs)
    end
    compute_dynamic_smagorinsky_νt!(νt, sim.u, sim.v, sim.w; workspace = ws, kw...)
    return mean(sqrt.(ws.cs2)), ws, νt
end
