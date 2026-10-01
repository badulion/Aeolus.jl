# Legacy snow flag: a per-column Bool that flips on below 273 K with
# precipitation and off above 275 K. Dormant in a summer run (no
# precipitation is passed); the winter-surface issue replaces it with the
# Snowpack mosaic and snow optics.

"""
    SnowState(Nx, Ny) -> Matrix{Bool}
    SnowState(mesh)  -> Matrix{Bool}

Allocate the boolean snow / ice cover flag matrix, size `(Nx, Ny)`,
initialized to `false` (no snow). When the flag is `true` for a cell,
callers should treat that cell as having snow-like radiative properties
(high albedo, high emissivity) and zero latent flux from soil moisture.
"""
SnowState(Nx::Integer, Ny::Integer) = falses(Int(Nx), Int(Ny))
SnowState(mesh)                     = falses(mesh.Nx, mesh.Ny)

"""
    update_snow_state!(snow, Ts, precip = 0)

Flip the per-cell snow / ice flag in place using a simple two-threshold
rule (avoids the ping-pong you get with a single threshold):

  * cell becomes snowy when `Ts < 273 K` **and** `precip > 0`
  * cell becomes bare  when `Ts > 275 K` (unconditionally)

`precip` may be a scalar (uniform precipitation) or a matrix the same
shape as `Ts`. Returns `snow`.
"""
function update_snow_state!(snow::AbstractMatrix{Bool},
                            Ts::AbstractMatrix{<:Real},
                            precip = 0)
    Nx, Ny = size(snow)
    is_field = precip isa AbstractMatrix
    p_scalar = is_field ? 0.0 : Float64(precip)
    @inbounds for j in 1:Ny, i in 1:Nx
        Tsij = Float64(Ts[i, j])
        p    = is_field ? Float64(precip[i, j]) : p_scalar
        if snow[i, j]
            # currently snowy: melt only above the upper threshold
            Tsij > 275.0 && (snow[i, j] = false)
        else
            # currently bare: freeze only if cold AND wet
            (Tsij < 273.0 && p > 0.0) && (snow[i, j] = true)
        end
    end
    return snow
end

# Snow-modified radiative constants for the surface energy balance.
const _SNOW_ALBEDO     = 0.75      # fresh-to-slightly-aged snow
const _SNOW_EMISSIVITY = 0.99      # snow radiates almost as a black body

"""
    snow_modify_params!(alb_view, emis_view, alb_orig, emis_orig, snow;
                        snow_albedo = 0.75, snow_emissivity = 0.99)

Populate `alb_view` and `emis_view` (both `AbstractMatrix{Float64}` of
size `(Nx, Ny)`) with the *effective* albedo and emissivity for the
current time step: cells with `snow[i, j] == true` are overridden with
the snow values; the rest are copied straight from `alb_orig` /
`emis_orig`.

This is the recommended way to plug snow into `step_surfaces!` without
mutating the class-level `SurfaceParams` struct. Typical call site:

    for j in 1:Ny, i in 1:Nx
        alb_orig[i, j]  = params.albedo[sfc[i, j]]
        emis_orig[i, j] = params.emissivity[sfc[i, j]]
    end
    snow_modify_params!(alb_eff, emis_eff, alb_orig, emis_orig, snow)

Returns `(alb_view, emis_view)`.
"""
function snow_modify_params!(alb_view::AbstractMatrix{Float64},
                             emis_view::AbstractMatrix{Float64},
                             alb_orig::AbstractMatrix{<:Real},
                             emis_orig::AbstractMatrix{<:Real},
                             snow::AbstractMatrix{Bool};
                             snow_albedo::Real = _SNOW_ALBEDO,
                             snow_emissivity::Real = _SNOW_EMISSIVITY)
    Nx, Ny = size(snow)
    a_snow = Float64(snow_albedo)
    e_snow = Float64(snow_emissivity)
    @inbounds for j in 1:Ny, i in 1:Nx
        if snow[i, j]
            alb_view[i, j]  = a_snow
            emis_view[i, j] = e_snow
        else
            alb_view[i, j]  = Float64(alb_orig[i, j])
            emis_view[i, j] = Float64(emis_orig[i, j])
        end
    end
    return (alb_view, emis_view)
end

"""
    SnowLegacy

Slot `:snow`, provider `:legacy_flag`: [`update_snow_state!`](@ref) every
step after the soil, with zero precipitation (KNOWN: nothing feeds it).
"""
struct SnowLegacy <: AbstractCoupling
    snow::BitMatrix
end

coupling_name(::Type{SnowLegacy}) = :snow_legacy
slot(::Type{SnowLegacy}) = :snow
provider(::Type{SnowLegacy}) = :legacy_flag
stages(::Type{SnowLegacy}) = ((:surface_update, 30, EVERY_STEP),)
build(::Type{SnowLegacy}, case, o) = SnowLegacy(SnowState(case.mesh))

run_stage!(c::SnowLegacy, case, ::Val{:surface_update}, dt) =
    (update_snow_state!(c.snow, case.bus.Ts, 0); nothing)

register!(SnowLegacy)
