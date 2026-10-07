# N-parent-b (#254) — Rayleigh damping of a prognostic field toward a
# reference: the θ (and qv) damping layer of a free-atmosphere top.
#
# Included inside `module LargeScaleForcing`, after the forcings above.
# The arithmetic is the one damping kernel `Operators.FiniteVolume._add_damping!`
# (the velocity sponge and the fringe region use the same kernel); this
# file only resolves the target and books the tendency integral.

"""
    ReferenceDamping(field::Symbol, profile::DampingProfile, target)

Rayleigh damping of the prognostic field `field` of a simulation (`:θ`,
`:qv`, or a velocity component `:u`, `:v`, `:w`) toward `target`, with the
height-dependent rate `λ(z)` of `profile` (a
[`DampingProfile`](@ref Aeolus.Operators.FiniteVolume.DampingProfile), usually
[`sponge_profile`](@ref Aeolus.Operators.FiniteVolume.sponge_profile)`(α_max, z_start, z_top)`):

```math
\\left.\\frac{\\partial φ}{\\partial t}\\right|_{d} = -λ(z)\\,\\bigl(φ - φ_{ref}\\bigr).
```

It is applied with [`apply_forcing!`](@ref) from the `body_force!` hook of
a core's `step!`, so the core's time integrator integrates it like every
other tendency (an RK3 stage sees it at the stage value). It is the one
place θ or qv is Rayleigh-damped: the cores' own sponge damps the
velocity only.

`target` is

* a number, a level vector (one value per interior level, length `Nz`),
  a `Field` or a halo-padded 3-D array — passed to the kernel as is
  (see [`add_sponge!`](@ref Aeolus.Operators.add_sponge!));
* a [`ForcingTable`](@ref) or a function `(z, t) -> value` — evaluated at
  the level heights at the time `t` of the call (the stage time);
* `:mean` — the horizontal (slab) mean of the field at each level, at the
  time of the call: only the deviations from the slab mean are damped,
  the mean profile is left free (DALES, Heus et al. 2010, Eqs. 40-41).
  Every level's integral is then zero to round-off.

After each application, `damping_tendency(F)` returns the tendency it
added: `levels[k]`, the sum over the columns of level `k` (field units
s⁻¹), and `integral`, its volume integral ``∫ ∂_tφ|_d\\,dV`` (field
units m³ s⁻¹) — the term a budget (the air-heat ledger) books for it.

Works on a `RegularGrid` and a `StretchedGrid` (the profile is sampled at
the actual level heights).

References: Klemp, J. B. & Lilly, D. K. (1978), *J. Atmos. Sci.*, 35,
78-107 (the damping layer); Durran, D. R. (2010), *Numerical Methods for
Fluid Dynamics*, 2nd ed., §8.4; Skamarock, W. C. et al. (2019), *A
Description of the Advanced Research WRF Model Version 4*, NCAR/TN-556,
§4.4.3 (damping θ as well as the velocity); Heus, T. et al. (2010),
*Geosci. Model Dev.*, 3, 415-444 (DALES; damping toward the slab mean);
Maronga, B. et al. (2015), *Geosci. Model Dev.*, 8, 2515-2551 (PALM).
"""
struct ReferenceDamping{P<:DampingProfile, X} <: AbstractLargeScaleForcing
    field::Symbol
    profile::P
    target::X
    buf::Vector{Float64}                 # evaluated level target (table, function, :mean)
    levels::Vector{Float64}              # Σ_columns of the tendency per level, last call
    integral::Base.RefValue{Float64}     # ∫ tendency dV, last call
    function ReferenceDamping(field::Symbol, profile::P, target::X) where {P<:DampingProfile, X}
        if target isa Symbol
            target === :mean ||
                throw(ArgumentError("ReferenceDamping: the symbolic target is :mean, got :$target"))
        end
        return new{P,X}(field, profile, target, Float64[], Float64[], Ref(0.0))
    end
end

"""
    damping_tendency(F::ReferenceDamping) -> (; levels, integral)

The tendency the last [`apply_forcing!`](@ref) of `F` added: per level
summed over the columns (`levels`, field units s⁻¹) and its volume
integral (`integral`, field units m³ s⁻¹).
"""
damping_tendency(F::ReferenceDamping) = (levels = F.levels, integral = F.integral[])

_tendency_name(field::Symbol) = Symbol(:t, field)

function _damped_pair(sim, F::ReferenceDamping)
    name = F.field
    tname = _tendency_name(name)
    (hasproperty(sim, name) && hasproperty(sim, tname)) ||
        throw(ArgumentError("ReferenceDamping: the simulation has no field :$name with tendency :$tname"))
    if name === :qv && hasproperty(sim, :has_moisture) && !sim.has_moisture
        throw(ArgumentError("ReferenceDamping(:qv, …): the simulation carries no moisture"))
    end
    return getproperty(sim, name), getproperty(sim, tname)
end

# Level thicknesses Δz_k of the interior levels (cell thickness on a
# StretchedGrid, dz on a RegularGrid).
_level_dz(m, k) = hasproperty(m, :dzc) ? m.dzc[m.Hz + k] : m.dz

# The target the kernel gets: a level vector for the evaluated forms.
_resolve_target(F::ReferenceDamping, target, f, z, t) = target
function _resolve_target(F::ReferenceDamping, target::Union{ForcingTable,Function}, f, z, t)
    @inbounds for k in eachindex(z)
        F.buf[k] = forcing_value(target, z[k], t)
    end
    return F.buf
end
function _resolve_target(F::ReferenceDamping, ::Symbol, f, z, t)
    m = f.mesh
    # the plane means, one work item per level (KA-k, #307; the former loop)
    launch_weighted!(backend(f.data), m.Nx * m.Ny, _plane_mean_level!, (1:m.Nz,), F.buf, f.data, m)
    return F.buf
end

function apply_forcing!(sim, F::ReferenceDamping, t, Δt; kw...)
    f, tf = _damped_pair(sim, F)
    m = sim.mesh
    Nz = m.Nz
    length(F.buf) == Nz || resize!(F.buf, Nz)
    length(F.levels) == Nz || resize!(F.levels, Nz)
    fill!(F.levels, 0.0)
    z = damping_coordinates(m, location(f))
    target = _resolve_target(F, F.target, f, z, t)
    _add_damping!(tf, f, F.profile, z, 3, target, m; levels = F.levels)
    s = 0.0
    @inbounds for k in 1:Nz
        s += F.levels[k] * _level_dz(m, k)
    end
    F.integral[] = s * m.dx * m.dy
    return sim
end
