# The SGS coefficients of AMD (ν_t, κ_t), Vreman and Smagorinsky 0.17 of the
# current state of `sim` (interiors), for test/era6_amd_vreman.jl and
# validation/era6_amd_vreman.jl.
using Aeolus.Fields: interior_view, Field

function closures!(sim; βg = 0.0)
    νa = Field(CellCenter(), sim.mesh); κa = Field(CellCenter(), sim.mesh)
    νv = Field(CellCenter(), sim.mesh); νs = Field(CellCenter(), sim.mesh)
    compute_amd_coefficients!(νa, κa, sim.u, sim.v, sim.w, sim.θ; buoyancy = βg)
    compute_vreman_νt!(νv, sim.u, sim.v, sim.w)
    compute_smagorinsky_νt!(νs, sim.u, sim.v, sim.w, 0.17)
    return interior_view(νa), interior_view(κa), interior_view(νv), interior_view(νs)
end
