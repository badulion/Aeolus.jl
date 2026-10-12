# The case's budgets: the case-specific half of `Aeolus.BudgetSystem` (ONE-c,
# #452). The container, the stage budgets, the surface and water
# bookkeeping and the reports are the core's; here are the case's step
# hooks and the integrals that need the case (its tracers, mesh and
# reference density).
#
# The budgets are the case's `Domain`'s (core/stages.jl): the pipeline
# attributes the heat of the air to every stage entry, the case's
# `UrbanSite` snapshots the surface temperature and the water in the air at
# the step's start and books the change of air water at its end, and
# every coupling's `budget!(c, case, budgets)` runs after `:budgets`. A
# coupling reports only what it did during that step (`add_surface!`,
# `add_water!`, `add_heat!`: see `src/budgets/budgets.jl`).

"""
    air_water(case) -> kg m⁻²

Water in the air per unit domain area: ``Σ_{tracers} Σ_{cells} ρ_k q\\,Δz
/ (N_x N_y)`` over `case.budgets.water_tracers` that are registered.
"""
function air_water(case)
    W = 0.0
    for name in case.budgets.water_tracers
        haskey(case.tracers, name) || continue
        W += water_content(case.budgets, case.mesh, interior_view(tracer(case, name)))
    end
    return W
end

"""
    reset_budgets!(case) -> budgets

Zero every accumulated term and start integrating from the current
state (called by [`build_case`](@ref)).
"""
function reset_budgets!(case)
    l = case.budgets
    empty!(l.terms); empty!(l.surface); empty!(l.water)
    l.water_air = 0.0
    l.elapsed = 0.0
    l.nsteps = 0
    l.ncol = [count(==(UInt8(k)), case.geom.sfc) for k in 1:length(SURFACE_CLASS_NAMES)]
    budget_start!(energy(l), LevelMeter(case.sim, case.ref, case.geom.sm.mask);
                  canopy_top = canopy_top(case))
    return l
end

"""
    canopy_top(case) -> m

Top of the `:canopy` layer of the air-heat budget: `case.cfg.canopy_top`
if it is set (finite), otherwise the tallest roof of the scene (the top
of its highest solid cell), or one level for a scene without buildings.

In the Alexanderplatz scene the tallest roof is the Fernsehturm, clipped
at the domain top (200 m on the 64 × 64 × 32 grid), so the default
`:canopy` layer is the whole column there; set `cfg.canopy_top` (e.g. to
the 90th percentile of the roof heights, 37.5 m) for the street canyons.
"""
function canopy_top(case)
    isfinite(case.cfg.canopy_top) && return Float64(case.cfg.canopy_top)
    return max(maximum(building_height(case.geom, case.mesh); init = 0.0),
               Float64(level_thickness(case.mesh)[1]))
end
