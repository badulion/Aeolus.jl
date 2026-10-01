# Shortwave modifier applied to the surface irradiance every step:
# Beer's-law shade under the park trees. (The cloud shadow that used to
# live here is `cloud_shadow.jl`, the slant-path
# SolarGeometry.cloud_shadow_transmittance!, since #130.)
#
# The shade acts on the UNMODIFIED irradiance of the last radiation
# update, never on its own previous output. `bus.sw` is recomputed only
# at the radiation cadence, so the coupling keeps that field (`sw_base`)
# and rebuilds `bus.sw = shade(sw_base)` every step. Before #128 the
# shade multiplied `bus.sw` in place, so the attenuation compounded
# between radiation updates (park shortwave × 0.287 per step, 7e-28 of
# its value after 50 steps). The cloud shadow, which also runs at the
# radiation cadence, is already in that field.
#
# KNOWN (later issues): the shade is a single big-leaf Beer's law on the
# effective irradiance of the radiosity solve, not a resolved canopy
# (#140, #145).

"""
    apply_tree_shade!(sw_arr, veg_mask, LAI_arr; k_leaf = 0.5)

Attenuate the shortwave flux reaching the ground beneath tree canopies
using Beer's law with an extinction coefficient `k_leaf` (typical
`0.4..0.6` for a randomly-distributed broadleaf canopy):

    SW_ground ← SW_ground · exp(-k_leaf · LAI)

Modifies `sw_arr` in place at every cell where `veg_mask[i, j]` is
`true`. Non-vegetated cells are untouched. Returns `sw_arr`.
"""
function apply_tree_shade!(sw_arr::AbstractMatrix{<:Real},
                           veg_mask::AbstractMatrix{Bool},
                           LAI_arr::AbstractMatrix{<:Real};
                           k_leaf::Real = 0.5)
    Nx, Ny = size(sw_arr)
    k = Float64(k_leaf)
    @inbounds for j in 1:Ny, i in 1:Nx
        if veg_mask[i, j]
            lai = Float64(LAI_arr[i, j])
            sw_arr[i, j] *= exp(-k * lai)
        end
    end
    return sw_arr
end

"""
    SWModifiers

Additive coupling: tree shade ([`apply_tree_shade!`](@ref), park columns,
LAI `cfg.park_lai`, option `k_leaf`), every step, on the irradiance of
the last radiation update.

`sw_base` holds that irradiance. The coupling detects a new radiation
update by `bus.sw` no longer equalling `sw_out`, the field it wrote on its
previous step, so it needs no knowledge of which provider or cadence
produced the irradiance. The shade is therefore applied exactly once to
each radiation field: over the park, `bus.sw / sw_base = exp(−k_leaf·LAI)`
on every step.
"""
mutable struct SWModifiers <: AbstractCoupling
    veg_mask::BitMatrix
    LAI::Matrix{Float64}
    k_leaf::Float64
    sw_base::Matrix{Float64}   # irradiance of the last radiation update
    sw_out::Matrix{Float64}    # what this coupling wrote to bus.sw last step
end

coupling_name(::Type{SWModifiers}) = :sw_modifiers
default_options(::Type{SWModifiers}) = (k_leaf = 0.5,)
stages(::Type{SWModifiers}) = ((:sw_modifiers, 10, EVERY_STEP),)

function build(::Type{SWModifiers}, case, o)
    veg_mask, LAI_map = park_lai_map(case)
    sz = size(case.bus.sw)
    # NaN: the first call always takes bus.sw as the new base
    return SWModifiers(veg_mask, LAI_map, o.k_leaf, zeros(sz), fill(NaN, sz))
end

function run_stage!(c::SWModifiers, case, ::Val{:sw_modifiers}, dt)
    sw = case.bus.sw
    # a radiation update (by any provider) has rewritten bus.sw since our
    # last step: that is the new unmodified irradiance
    sw == c.sw_out || copyto!(c.sw_base, sw)
    copyto!(sw, c.sw_base)
    # L2: tree shade attenuates SW under vegetation
    apply_tree_shade!(sw, c.veg_mask, c.LAI; k_leaf = c.k_leaf)
    copyto!(c.sw_out, sw)
    return nothing
end

"""
    sw_modifier_factor(c::SWModifiers, i, j) -> Float64

The factor the shade applies to the unmodified irradiance in column
`(i, j)`: `exp(−k_leaf·LAI)` on park columns, 1 elsewhere.
"""
sw_modifier_factor(c::SWModifiers, i, j) =
    c.veg_mask[i, j] ? exp(-c.k_leaf * c.LAI[i, j]) : 1.0

function smoke_check(c::SWModifiers, case)
    sw = case.bus.sw
    fac = [sw_modifier_factor(c, i, j) for i in axes(sw, 1), j in axes(sw, 2)]
    return ["shaded shortwave ≥ 0" => all(>=(0), sw),
            "park columns in the window" => any(c.veg_mask),
            # applied once to the radiation field, not compounded
            "bus.sw = factor × unmodified irradiance (rel. 1e-12)" =>
                all(isapprox.(sw, fac .* c.sw_base; rtol = 1e-12, atol = 1e-12)),
            "unmodified irradiance kept ≥ bus.sw" => all(c.sw_base .>= sw)]
end

register!(SWModifiers)
