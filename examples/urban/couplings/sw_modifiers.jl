# Shortwave modifiers applied to the surface irradiance every step:
# Beer's-law shade under the park trees and a cloud shadow from the
# column-integrated cloud water.
#
# KNOWN DEFECTS, kept so the refactor is bit-for-bit (the shortwave
# modifier issue fixes them): both multiply `bus.sw` IN PLACE on every
# step, while `bus.sw` is recomputed only at the radiation cadence, so
# the attenuation compounds — park shortwave is multiplied by
# exp(-0.5·2.5) = 0.287 per step and is effectively zero between
# radiation updates, and the cloud shadow compounds the same way wherever
# there is condensate. The cloud shadow is also not a slant path
# (SolarGeometry.cloud_shadow_transmittance! is the replacement).

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
    apply_cloud_shadow!(sw_arr, ql_col_int, sw_direct_beam)

Attenuate the direct-beam SW component of `sw_arr` by the column-integrated
cloud water. Uses Beer's law with an extinction coefficient tuned so that
1 g m⁻² integrated ql gives ~50 % attenuation.

`ql_col_int::Matrix{Float64}` is the column integral of `ql` (kg m⁻²).
`sw_direct_beam` is the direct-beam value that would appear in a clear sky.
"""
function apply_cloud_shadow!(sw_arr::AbstractMatrix, ql_col_int::AbstractMatrix,
                             sw_direct_beam::Real; k_ext::Real = 700.0)
    Nx, Ny = size(sw_arr)
    @inbounds for j in 1:Ny, i in 1:Nx
        τ  = k_ext * ql_col_int[i, j]
        tr = exp(-τ)
        # Assume `sw_arr[i, j]` currently holds direct + diffuse.
        # Attenuate only the direct-beam portion; diffuse stays.
        # Extract the current direct component by rescaling:
        # sw_arr = direct*(1-shade) + diffuse. We approximate direct as
        # sw_direct_beam · (sw_arr - diffuse_floor) / sw_direct_beam. For a
        # demo cloud effect we just apply tr uniformly on the direct part.
        sw_arr[i, j] = sw_arr[i, j] * tr + sw_arr[i, j] * (1 - tr) * 0.15
    end
    return sw_arr
end

"""
    SWModifiers

Additive coupling: tree shade ([`apply_tree_shade!`](@ref), park columns,
LAI `cfg.park_lai`, option `k_leaf`) then cloud shadow
([`apply_cloud_shadow!`](@ref) with the column-integrated `ql`, option
`k_ext`) on `bus.sw`, every step. See the KNOWN DEFECTS in this file.
"""
struct SWModifiers <: AbstractCoupling
    veg_mask::BitMatrix
    LAI::Matrix{Float64}
    k_leaf::Float64
    k_ext::Float64
end

coupling_name(::Type{SWModifiers}) = :sw_modifiers
default_options(::Type{SWModifiers}) = (k_leaf = 0.5, k_ext = 700.0)
stages(::Type{SWModifiers}) = ((:sw_modifiers, 10, EVERY_STEP),)

function build(::Type{SWModifiers}, case, o)
    veg_mask, LAI_map = park_lai_map(case)
    return SWModifiers(veg_mask, LAI_map, o.k_leaf, o.k_ext)
end

function run_stage!(c::SWModifiers, case, ::Val{:sw_modifiers}, dt)
    sw = case.bus.sw
    # L2: tree shade attenuates SW under vegetation
    apply_tree_shade!(sw, c.veg_mask, c.LAI; k_leaf = c.k_leaf)
    # R5: cloud shadow from column-integrated ql
    ql_col = dropdims(sum(interior_view(tracer(case, :ql)); dims = 3) .* case.mesh.dz;
                      dims = 3)
    apply_cloud_shadow!(sw, ql_col, case.bus.sw_beam; k_ext = c.k_ext)
    return nothing
end

register!(SWModifiers)
