# R6 — Window / glass material for the urban flagship example.
#
# Adds a sixth surface class (`SURFACE_GLASS = 6`) representing curtain
# walls / large window façades. Glass differs from the existing WALL
# class in three ways:
#
#   1. Very high specular albedo — most direct SW is reflected.
#   2. Partial SW transmission into the building interior, so a
#      fraction `(1 - α) · τ` of the incoming SW never becomes a
#      sensible flux to the outside atmosphere.
#   3. Slightly reduced LW emissivity (a real coated glazing).
#
# The class integer is 6, one past `SURFACE_WALL = 5`, so a
# `SurfaceParams` from [`glass_surface_params`](@ref) can be used with
# `step_surfaces!` as-is.

include(joinpath(@__DIR__, "urban_geometry.jl"))
include(joinpath(@__DIR__, "urban_physics.jl"))

export SURFACE_GLASS, promote_to_glass!, glass_surface_params,
       apply_glass_sw_transmission!

"Surface-class code for window / glass façade cells."
const SURFACE_GLASS = UInt8(6)

"""
    promote_to_glass!(sfc, glass_mask) -> sfc

Overwrite every cell of the surface-class array `sfc::AbstractMatrix{UInt8}`
where `glass_mask[i, j] == true` with [`SURFACE_GLASS`](@ref). The
`glass_mask` must have the same shape as `sfc`. Returns `sfc`.

Typical use: run `tag_building_adjacent!` to get wall cells, then mark
the wall subset that carries a glazed façade (e.g. detected from OSM
`building:material=glass` or a hand-drawn mask).
"""
function promote_to_glass!(sfc::AbstractMatrix{UInt8},
                           glass_mask::AbstractMatrix{Bool})
    size(sfc) == size(glass_mask) ||
        error("promote_to_glass!: sfc $(size(sfc)) and glass_mask $(size(glass_mask)) size mismatch")
    Nx, Ny = size(sfc)
    @inbounds for j in 1:Ny, i in 1:Nx
        if glass_mask[i, j]
            sfc[i, j] = SURFACE_GLASS
        end
    end
    return sfc
end

"""
    glass_surface_params() -> SurfaceParams

Return a [`SurfaceParams`](@ref) extending the five default urban
classes with a sixth glass entry:

  * `C_s = 1e5`  J/m²/K — low areal heat capacity (thin pane).
  * `albedo = 0.35` — high specular reflectance.
  * `emissivity = 0.85` — partial LW emissivity for coated glazing.

Suitable to pass to `step_surfaces!` in place of `default_surface_params()`
when a `SURFACE_GLASS`-tagged surface-class map is used.
"""
function glass_surface_params()
    base = default_surface_params()
    C_s  = vcat(base.C_s,        1.0e5)
    alb  = vcat(base.albedo,     0.35)
    emis = vcat(base.emissivity, 0.85)
    return SurfaceParams(C_s, alb, emis)
end

"""
    apply_glass_sw_transmission!(sensible, sfc, sw_arr; transmission = 0.3,
                                 albedo = 0.35)

For every ground cell tagged [`SURFACE_GLASS`](@ref), remove
`transmission · (1 - albedo) · sw_arr[i, j]` W/m² from the sensible-flux
matrix `sensible`. Physically: the transmitted fraction of the SW that
enters the (non-albedo) portion of the incident beam continues *into*
the building interior instead of becoming a convective flux to the
outside atmosphere. Non-glass cells are untouched.

Call this after `apply_surface_flux!` (which fills the sensible-flux
matrix) but before the sensible flux is fed back into `step_surfaces!`
so both budgets remain consistent.
"""
function apply_glass_sw_transmission!(sensible::AbstractMatrix,
                                      sfc::AbstractMatrix{UInt8},
                                      sw_arr::AbstractMatrix;
                                      transmission::Real = 0.3,
                                      albedo::Real = 0.35)
    size(sensible) == size(sfc) == size(sw_arr) ||
        error("apply_glass_sw_transmission!: shape mismatch " *
              "sensible=$(size(sensible)) sfc=$(size(sfc)) sw=$(size(sw_arr))")
    Nx, Ny = size(sensible)
    factor = transmission * (1.0 - albedo)
    @inbounds for j in 1:Ny, i in 1:Nx
        if sfc[i, j] == SURFACE_GLASS
            sensible[i, j] -= factor * sw_arr[i, j]
        end
    end
    return sensible
end
