# Shortwave modifiers applied to the surface irradiance every step:
# Beer's-law shade under the park trees and a cloud shadow from the
# column-integrated cloud water.
#
# The modifiers act on the UNMODIFIED irradiance of the last radiation
# update, never on their own previous output. `bus.sw` is recomputed only
# at the radiation cadence, so the coupling keeps that field (`sw_base`)
# and rebuilds `bus.sw = cloud(shade(sw_base))` every step. Before #128
# both modifiers multiplied `bus.sw` in place, so the attenuation
# compounded between radiation updates (park shortwave × 0.287 per step,
# 7e-28 of its value after 50 steps).
#
# KNOWN (later issues): the cloud shadow is not a slant path and its
# extinction is tuned, not derived (SolarGeometry.cloud_shadow_transmittance!,
# #130); the shade is a single big-leaf Beer's law on the effective
# irradiance of the radiosity solve, not a resolved canopy (#140, #145).

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
`k_ext`), every step, on the irradiance of the last radiation update.

`sw_base` holds that irradiance. The coupling detects a new radiation
update by `bus.sw` no longer equalling `sw_out`, the field it wrote on its
previous step, so it needs no knowledge of which provider or cadence
produced the irradiance. The modifiers are therefore applied exactly once
to each radiation field: over the park, with no cloud water,
`bus.sw / sw_base = exp(−k_leaf·LAI)` on every step.
"""
mutable struct SWModifiers <: AbstractCoupling
    veg_mask::BitMatrix
    LAI::Matrix{Float64}
    k_leaf::Float64
    k_ext::Float64
    sw_base::Matrix{Float64}   # irradiance of the last radiation update
    sw_out::Matrix{Float64}    # what this coupling wrote to bus.sw last step
    ql_col::Matrix{Float64}    # column-integrated ql, kg m⁻²
end

coupling_name(::Type{SWModifiers}) = :sw_modifiers
default_options(::Type{SWModifiers}) = (k_leaf = 0.5, k_ext = 700.0)
stages(::Type{SWModifiers}) = ((:sw_modifiers, 10, EVERY_STEP),)

function build(::Type{SWModifiers}, case, o)
    veg_mask, LAI_map = park_lai_map(case)
    sz = size(case.bus.sw)
    # NaN: the first call always takes bus.sw as the new base
    return SWModifiers(veg_mask, LAI_map, o.k_leaf, o.k_ext,
                       zeros(sz), fill(NaN, sz), zeros(sz))
end

"""
    column_liquid_water!(ql_col, ql, dz) -> ql_col

`ql_col[i, j] = Σ_k ql[i, j, k]·dz` (kg m⁻² for ρ = 1 kg m⁻³, the
Boussinesq reference density of the case).
"""
function column_liquid_water!(ql_col::AbstractMatrix, ql::AbstractArray{<:Real,3}, dz::Real)
    fill!(ql_col, 0)
    @inbounds for k in axes(ql, 3), j in axes(ql, 2), i in axes(ql, 1)
        ql_col[i, j] += ql[i, j, k]
    end
    ql_col .*= dz
    return ql_col
end

function run_stage!(c::SWModifiers, case, ::Val{:sw_modifiers}, dt)
    sw = case.bus.sw
    # a radiation update (by any provider) has rewritten bus.sw since our
    # last step: that is the new unmodified irradiance
    sw == c.sw_out || copyto!(c.sw_base, sw)
    copyto!(sw, c.sw_base)
    # L2: tree shade attenuates SW under vegetation
    apply_tree_shade!(sw, c.veg_mask, c.LAI; k_leaf = c.k_leaf)
    # R5: cloud shadow from column-integrated ql
    column_liquid_water!(c.ql_col, interior_view(tracer(case, :ql)), case.mesh.dz)
    apply_cloud_shadow!(sw, c.ql_col, case.bus.sw_beam; k_ext = c.k_ext)
    copyto!(c.sw_out, sw)
    return nothing
end

"""
    sw_modifier_factor(c::SWModifiers, i, j) -> Float64

The factor the modifiers apply to the unmodified irradiance in column
`(i, j)` given the current column water `c.ql_col`: the Beer's-law shade
`exp(−k_leaf·LAI)` on park columns times the cloud factor
`tr + 0.15·(1 − tr)`, `tr = exp(−k_ext·ql_col)`.
"""
function sw_modifier_factor(c::SWModifiers, i, j)
    shade = c.veg_mask[i, j] ? exp(-c.k_leaf * c.LAI[i, j]) : 1.0
    tr = exp(-c.k_ext * c.ql_col[i, j])
    return shade * (tr + 0.15 * (1 - tr))
end

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
