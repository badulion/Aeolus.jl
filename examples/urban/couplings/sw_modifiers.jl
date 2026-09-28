# Shortwave modifier applied to the surface irradiance every step:
# Beer's-law shade under the park trees. (The cloud shadow that used to
# live here is `cloud_shadow.jl`, the slant-path
# SolarGeometry.cloud_shadow_transmittance!, since #130.)
#
# KNOWN DEFECT, kept so the refactor is bit-for-bit (the shortwave
# modifier issue fixes it): the shade multiplies `bus.sw` IN PLACE on
# every step, while `bus.sw` is recomputed only at the radiation cadence,
# so the attenuation compounds — park shortwave is multiplied by
# exp(-0.5·2.5) = 0.287 per step and is effectively zero between
# radiation updates.

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
LAI `cfg.park_lai`, option `k_leaf`) on `bus.sw`, every step. See the
KNOWN DEFECT in this file.
"""
struct SWModifiers <: AbstractCoupling
    veg_mask::BitMatrix
    LAI::Matrix{Float64}
    k_leaf::Float64
end

coupling_name(::Type{SWModifiers}) = :sw_modifiers
default_options(::Type{SWModifiers}) = (k_leaf = 0.5,)
stages(::Type{SWModifiers}) = ((:sw_modifiers, 10, EVERY_STEP),)

function build(::Type{SWModifiers}, case, o)
    veg_mask, LAI_map = park_lai_map(case)
    return SWModifiers(veg_mask, LAI_map, o.k_leaf)
end

function run_stage!(c::SWModifiers, case, ::Val{:sw_modifiers}, dt)
    # L2: tree shade attenuates SW under vegetation
    apply_tree_shade!(case.bus.sw, c.veg_mask, c.LAI; k_leaf = c.k_leaf)
    return nothing
end

smoke_check(c::SWModifiers, case) =
    ["shaded shortwave ≥ 0" => all(>=(0), case.bus.sw),
     "park columns in the window" => any(c.veg_mask)]

register!(SWModifiers)
