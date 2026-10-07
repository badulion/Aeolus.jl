# The sounding and case-table preprocessor of the dynamic driver (CFG-g,
# #362): a radiosonde sounding, or the forcing tables of a case (GABLS,
# CASES-99), as `DynamicInputs` on the "seconds since init_time" axis.
# Nothing is resampled: the sounding's height-time tables and the case's
# tables are written with their own nodes, so a run reading the file sees
# the tables it would have built from the CSV, value for value.

"""
    init_time_string(y, m, d, hour_utc) -> String

The ISO-8601 UTC time (`"yyyy-mm-ddTHH:MM:SSZ"`) of fractional hour
`hour_utc` of the date `(y, m, d)`: the `init_time` attribute of a dynamic
driver.
"""
function init_time_string(y::Integer, m::Integer, d::Integer, hour_utc::Real)
    t = Dates.DateTime(y, m, d) + Dates.Millisecond(round(Int64, 3_600_000 * hour_utc))
    return Dates.format(t, "yyyy-mm-ddTHH:MM:SS") * "Z"
end

"""
    init_day_seconds(init_time) -> s

Seconds from 00 UTC of the date of `init_time` (ISO-8601 UTC) to
`init_time`: the offset that puts a driver's "seconds since init_time"
tables on a day-based axis (a sounding whose epoch is the start date).
"""
function init_day_seconds(init_time::AbstractString)
    t = Dates.DateTime(rstrip(init_time, 'Z'))
    return Dates.value(t - Dates.DateTime(Dates.Date(t))) / 1000
end

"""
    surface_pressure_at(profiles, t, z_launch, z_site; c = MoistThermodynamics.DEFAULT_CONSTANTS) -> Pa

The pressure at a site of elevation `z_site` (m above sea level) from the
sounding `profiles` ([`sounding_profiles`](@ref
Aeolus.LargeScaleForcing.sounding_profiles)) at time `t` (on the tables'
axis), launched at `z_launch`: the launch-site surface pressure reduced
hypsometrically with the surface virtual temperature,
``p = p_L \\exp\\bigl(g_0 (z_L - z)/(R_d T_v)\\bigr)``.

Reference: Wallace & Hobbs (2006), *Atmospheric Science*, 2nd ed., §3.2.2.
"""
function surface_pressure_at(P, t::Real, z_launch::Real, z_site::Real; c = DEFAULT_CONSTANTS)
    pL = forcing_value(P.p, 0.0, t)
    T = forcing_value(P.θ, 0.0, t) * exner_function(pL)
    Tv = virtual_temperature(T, PhasePartition(forcing_value(P.qv, 0.0, t)))
    R_d = c.gas_constant / c.molmass_dry
    return pL * exp(c.grav * (z_launch - z_site) / (R_d * Tv))
end

"""
    sounding_dynamic_inputs(snd::Sounding; t0, init_time, z_top = 3000.0, p_surface = nothing)
        -> DynamicInputs

The dynamic-driver form of a radiosonde sounding: its height-time tables
([`sounding_profiles`](@ref Aeolus.LargeScaleForcing.sounding_profiles)
up to `z_top`) as `ls_profile_theta`, `_qv`, `_u`, `_v`, `_p` on the axis
"seconds since init_time" (`t0` is the run start on the sounding's axis,
[`sounding_time`](@ref Aeolus.LargeScaleForcing.sounding_time)), and
`p_surface` (Pa) as `init_p_surface`. There are no `init_*` profiles: the
initial state is the tables at `t = 0` ([`initial_profile`](@ref)), the
time interpolation between the two launches that bracket the start.
"""
function sounding_dynamic_inputs(snd::Sounding; t0::Real, init_time::AbstractString, z_top::Real = 3000.0,
                                 p_surface = nothing)
    P = sounding_profiles(snd; z_top = z_top)
    tabs = Dict{Symbol,ForcingTable{Float64}}(
        :profile_theta => shift_table(P.θ, -t0), :profile_qv => shift_table(P.qv, -t0),
        :profile_u => shift_table(P.u, -t0), :profile_v => shift_table(P.v, -t0),
        :profile_p => shift_table(P.p, -t0))
    attrs = Dict{String,Any}("dynamic_source" => "sounding", "sounding_station" => snd.station,
                             "sounding_elevation" => snd.elevation, "sounding_launches" => length(snd.launches))
    return DynamicInputs(; init_time, p_surface, tables = tabs, attributes = attrs)
end

"""
    forcing_dynamic_inputs(; U_g = nothing, V_g = nothing, extra = (), surface = nothing,
                           t_start = 0.0, init_time = "", attributes = Dict()) -> DynamicInputs

The dynamic-driver form of a case's large-scale forcing (the case-table
preprocessor: GABLS, CASES-99): the inverse of [`dynamic_forcing`](@ref).
`U_g`, `V_g` and the quantities of the `extra` forcings
([`AdvectiveTendencies`](@ref Aeolus.LargeScaleForcing.AdvectiveTendencies),
[`Subsidence`](@ref Aeolus.LargeScaleForcing.Subsidence),
[`ProfileNudging`](@ref Aeolus.LargeScaleForcing.ProfileNudging)) that
are `ForcingTable`s become `ls_*` tables; a number or a function stays
with the case (it is not a table). `surface` is a `(t, temperature,
pressure)` series. Times move from the case's model time to seconds since
`init_time`, the run start `t_start`.
"""
function forcing_dynamic_inputs(; U_g = nothing, V_g = nothing, extra = (), surface = nothing,
                                t_start::Real = 0.0, init_time::AbstractString = "",
                                attributes::AbstractDict = Dict{String,Any}())
    tabs = Dict{Symbol,ForcingTable{Float64}}()
    _put(k, q) = q isa ForcingTable && (tabs[k] = shift_table(q, -t_start))
    _put(:u_g, U_g); _put(:v_g, V_g)
    attrs = Dict{String,Any}(attributes)
    attrs["dynamic_source"] = get(attrs, "dynamic_source", "case_table")
    for F in extra
        if F isa AdvectiveTendencies
            _put(:theta_adv, F.θ); _put(:qv_adv, F.q); _put(:u_adv, F.u); _put(:v_adv, F.v)
        elseif F isa Subsidence
            _put(:w_subs, F.w_s)
            F.explicit && (attrs["subsidence_fields"] = join(String.(F.fields), " "))
            attrs["subsidence_form"] = String(F.form)
        elseif F isa ProfileNudging
            F.weight === nothing || error("forcing_dynamic_inputs: a ProfileNudging weight has no driver form")
            for (f, k) in ((:u, :nudge_u), (:v, :nudge_v), (:θ, :nudge_theta), (:qv, :nudge_qv))
                haskey(F.targets, f) && _put(k, F.targets[f])
            end
            attrs["nudging_tau"] = Float64(F.τ)
        else
            error("forcing_dynamic_inputs: $(nameof(typeof(F))) has no dynamic-driver form")
        end
    end
    sfc = surface === nothing ? nothing :
          (t = Float64.(surface.t) .- t_start, temperature = surface.temperature, pressure = surface.pressure)
    return DynamicInputs(; init_time, tables = tabs, surface = sfc, attributes = attrs)
end
