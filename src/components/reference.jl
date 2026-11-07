# The thermodynamic reference of a domain (#536): the one hydrostatic
# profile T = θ Π̄ and every flux conversion use.
#
# Where it comes from (one source of truth, nothing is computed twice):
#
#   * a Boussinesq domain built from a configuration: `drivers.reference`,
#     the hydrostatic state of its initial θ profile from its surface
#     pressure (`Urban.domain_reference`, built by the model's config for
#     every Boussinesq domain, PALM's `hyp` of `init_3d_model`); an urban
#     site's domain: the site's own `CaseReference`;
#   * the anelastic core: its base state `sim.ref`
#     (`Anelastic.ReferenceState`).
#
# Both carry `π̄`, `θ̄` (per level centre) and `cp`. A bare Boussinesq core
# (a `Domain` made in Julia without a reference driver) has none.

"""
    thermodynamic_reference(d::Domain) -> reference or nothing

The thermodynamic reference profiles of domain `d` (`π̄`, `θ̄` at the
level centres, `cp`; see `src/components/reference.jl`): its
`drivers.reference` (a Boussinesq domain built from a configuration, an
urban site's `CaseReference`), else the anelastic core's base state
`sim.ref`, else `nothing` (a bare Boussinesq core). The urban surface and
radiation components (`Urban.air_reference`) and the absolute-temperature
outputs (`temperature`, `temperature_2m`, `surface_temperature`, #536)
take it from here.
"""
function thermodynamic_reference(d::Domain)
    r = get(d.drivers, :reference, nothing)
    r !== nothing && return r
    s = d.state
    return hasproperty(s, :ref) ? s.ref : nothing
end

"""
    reference_exner(ref, z_centres, z; g) -> Π̄(z)

The reference Exner function at height `z` (m above the ground) of the
profile `ref.π̄` given at the level centres `z_centres`: linear between
two centres; below the first centre (and above the last) the hydrostatic
dry-adiabatic relation ``dΠ/dz = -g/(c_p θ̄)`` of that level,
``Π̄(z) = π̄_1 + g (z_1 - z)/(c_p θ̄_1)`` — exact for a constant θ̄
between the ground and ``z_1``, the dry-adiabatic hydrostatic profile
PALM's barometric formula gives (`init_3d_model.f90`; its 2 m Exner value,
`calc_2m_temperature`). With `z = 0` this is the surface Exner value
``Π̄_s``. `g` is the gravity of the dynamics.
"""
function reference_exner(ref, zc::AbstractVector, z::Real; g::Real)
    π̄, θ̄, cp = ref.π̄, ref.θ̄, ref.cp
    n = length(zc)
    z <= zc[1] && return π̄[1] + g * (zc[1] - z) / (cp * θ̄[1])
    z >= zc[n] && return π̄[n] - g * (z - zc[n]) / (cp * θ̄[n])
    k = searchsortedlast(zc, z)
    w = (z - zc[k]) / (zc[k+1] - zc[k])
    return (1 - w) * π̄[k] + w * π̄[k+1]
end

"""
    reference_exner(d::Domain, z) -> Π̄(z)

[`reference_exner`](@ref) of the [`thermodynamic_reference`](@ref) of
domain `d` at height `z` (m above the ground), with its level centres and
the gravity the reference was integrated with (an anelastic base state's
`g`, else that of the core). An `ArgumentError` when `d` has no reference.
"""
function reference_exner(d::Domain, z::Real)
    r = required_reference(d)
    return reference_exner(r, last(cell_centres(d.state.mesh)), z; g = hasproperty(r, :g) ? r.g : d.state.g_accel)
end

"""
    required_reference(d::Domain) -> reference

The [`thermodynamic_reference`](@ref) of `d`, or an `ArgumentError` that
says why an absolute temperature cannot be had without one.
"""
function required_reference(d::Domain)
    r = thermodynamic_reference(d)
    r === nothing &&
        throw(ArgumentError("domain $(d.id): the Boussinesq core has no thermodynamic reference, so θ cannot become " *
                            "an absolute temperature (a domain built from a configuration has one: the hydrostatic " *
                            "state of its initial θ profile; in Julia pass drivers = (; reference = " *
                            "Urban.domain_reference(mesh, θ, p_surface)))"))
    return r
end
