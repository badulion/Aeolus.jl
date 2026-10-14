# The site's output catalogue (ONE-c #452; ONE-k #474): what the run's one
# writer (`Aeolus.Output`, the core's `[output]`) can write of an urban
# site. `[output] variables` selects from it and the core's catalogue
# (`Components.domain_catalogue`: u, v, w, p, theta, their profiles and
# means, the budgets).
#
# The site publishes the frame (`hour`, `solar_elevation`, time series),
# every tracer registered with `output = true`, the bus fields Ts / sw_in /
# lw_in, and whatever the enabled couplings return from `output_fields`
# (their catalogues of `OutputVariable`s, getters of the site); the static
# geometry is written once. The renderers (`alexanderplatz_animation.jl`,
# `alexanderplatz_surface_diagnostics.jl`) read the file by variable name.

"""
    core_output_fields(case) -> Vector{OutputVariable}

The site's own variables (getters of the site): the local `hour` and the
`solar_elevation` (time series), every tracer registered with `output =
true` (3-D), and the surface temperature `Ts`, the shortwave `sw_in` and
the downwelling longwave `lw_in` at the surface (maps). The velocity,
θ and pressure are the core's (`Components.core_outputs`).
"""
function core_output_fields(case)
    iv(f) = interior_view(f)
    fs = OutputVariable[
        OutputVariable("hour", "h", "local hour of day", :scalar, c -> Float64(c.clock.hour)),
        OutputVariable("solar_elevation", "degree", "solar elevation angle", :scalar,
                       c -> Float64(c.clock.sun.elevation))]
    for t in case.tracers
        t.output || continue
        f = t.field
        push!(fs, OutputVariable(string(t.name), t.units, t.long_name, :xyz, c -> iv(f)))
    end
    append!(fs, OutputVariable[
        OutputVariable("Ts",    "K",     "surface temperature",                 :xy, c -> c.bus.Ts),
        OutputVariable("sw_in", "W m-2", "shortwave irradiance at the surface", :xy, c -> c.bus.sw),
        OutputVariable("lw_in", "W m-2", "downwelling longwave at the surface", :xy, c -> c.bus.lw)])
    return fs
end

"The static geometry the output file carries (written once; getters of the site)."
static_output_fields(case) = OutputVariable[
    OutputVariable("sky_view_factor", "1", "sky view factor, area-weighted per column", :xy, c -> c.geom.svf),
    OutputVariable("building_height", "m", "building height", :xy, c -> building_height(c.geom, c.mesh)),
    OutputVariable("surface_class", "1", "surface class (1 street 2 roof 3 park 4 water 5 wall 6 glass)", :xy,
                   c -> c.geom.sfc)]

# a variable of the site as one of its domain's catalogue (the writer's
# context is the domain)
_on_domain(v::OutputVariable) =
    OutputVariable(v.name, v.units, v.long_name, v.dims, d -> v.getter(urban_case(d));
                   eltype = v.eltype, group = v.group)

"""
    Components.outputs(case::UrbanSite) -> Vector{OutputVariable}

The site's output catalogue: [`core_output_fields`](@ref) and every
enabled coupling's `output_fields`, as variables of its domain.
"""
function Components.outputs(case::UrbanSite)
    fields = core_output_fields(case)
    for c in case.couplings
        append!(fields, output_fields(c, case))
    end
    return OutputVariable[_on_domain(v) for v in check_catalogue(fields)]
end

"The site's static geometry ([`static_output_fields`](@ref)), written once into its domain's output file."
Components.static_outputs(case::UrbanSite) = OutputVariable[_on_domain(v) for v in static_output_fields(case)]

"The names of the site's output catalogue ([`Components.outputs`](@ref))."
output_names(case::UrbanSite) = [v.name for v in Components.outputs(case)]
