# R6 — Window / glass material for the Berlin case.
#
# Adds a sixth surface class (`SURFACE_GLASS = 6`) representing curtain
# walls / large window façades in the example's one-temperature-per-cell
# surface scheme.
#
# Since Phase 49 (#27) this file holds NO glazing physics of its own (the
# pane heat capacity and the transmittance moved to UrbanSurface in ONE-e,
# #454). The
# optics (Fresnel + absorption, multi-pane inter-reflection, angular and
# hemispherical averages), the pane heat capacity and the emissivity all
# come from `Aeolus.UrbanSurface` — the same code as the production
# `WindowFacet`. It used to carry three hand-set numbers (albedo 0.35,
# transmission 0.3, emissivity 0.85) and a transmission correction with
# the wrong sign; see `apply_glass_sw_transmission!` below.
#
# The class integer is 6, one past `SURFACE_WALL = 5`, so a
# `SurfaceParams` from [`glass_surface_params`](@ref) extends the class
# tables as-is (since #129 the surface exchange builds a `SlabLand` from it). For a resolved glazed facet (pane temperatures,
# transmitted solar gain into the room) use `WindowFacet` directly.

# KNOWN (the per-building zones / WindowFacet issue replaces this file):
# the glass cells are a SYNTHETIC mask — every 5th/4th roof cell — because
# OSM carries no glazing tag, so the "glass" sits on roofs, not facades.

"Surface-class code for window / glass façade cells."
const SURFACE_GLASS = UInt8(6)

"""
    example_glazing() -> Glazing

The glazing the example's glass class stands for: an uncoated 4-16-4
air-filled double-glazed unit of clear float
(`Aeolus.UrbanSurface.double_glazing()`).
"""
example_glazing() = double_glazing()

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
    glass_surface_params(; glazing = example_glazing()) -> SurfaceParams

Return a [`SurfaceParams`](@ref) extending the five default urban
classes with a sixth glass entry, every number derived from `glazing`
by `Aeolus.UrbanSurface`:

  * `C_s` — the areal heat capacity of the panes, ``Σ ρ c d`` from the
    `:glass` row of the thermal material table (1.5e4 J m⁻² K⁻¹ for
    4-16-4, not the 1e5 the add-on used to assume);
  * `albedo` — the hemispherical (diffuse) solar *reflectance* of the
    unit, `glazing_optics_hemispherical(glazing).R` (0.19 for 4-16-4);
  * `emissivity` — the outer pane's thermal emissivity (0.837).

The transmitted part of the shortwave is *not* in the albedo: the
case's glass surface (a `SlabLand`, #129) would absorb it into the cell. Remove it with
[`apply_glass_sw_transmission!`](@ref).
"""
function glass_surface_params(; glazing::Glazing = example_glazing())
    base = default_surface_params()
    C_glass = glazing_areal_heat_capacity(glazing)
    hem = glazing_optics_hemispherical(glazing)
    C_s  = vcat(base.C_s,        C_glass)
    alb  = vcat(base.albedo,     hem.R)
    emis = vcat(base.emissivity, glazing.panes[1].emissivity)
    return SurfaceParams(C_s, alb, emis)
end

"""
    apply_glass_sw_transmission!(loss, sfc, sw_arr; glazing = example_glazing(),
                                 cos_incidence = nothing)

For every cell tagged [`SURFACE_GLASS`](@ref), add the shortwave the
glazing transmits into the building, ``T\\,SW``, to the surface-loss
matrix `loss` (`bus.sensible`, which the surface exchange takes from the
cell's energy budget as `Q_extra = -loss`). ``T`` is the glazing's solar transmittance: the
hemispherical value by default, or the direct-beam value at the
incidence angle `acos(cos_incidence)` (for the example's horizontal
glass cells, the cosine of the solar zenith angle). With the albedo of
[`glass_surface_params`](@ref) the glass cell then absorbs exactly
``(1 - R - T)\\,SW``, the unit's total pane absorptance.

Correction: the pre-#27 version *subtracted* `transmission·(1-albedo)·SW`
from this loss term, which *added* the transmitted shortwave to the glass
cell's heating instead of removing it.
"""
function apply_glass_sw_transmission!(loss::AbstractMatrix,
                                      sfc::AbstractMatrix{UInt8},
                                      sw_arr::AbstractMatrix;
                                      glazing::Glazing = example_glazing(),
                                      cos_incidence::Union{Nothing, Real} = nothing)
    size(loss) == size(sfc) == size(sw_arr) ||
        error("apply_glass_sw_transmission!: shape mismatch " *
              "loss=$(size(loss)) sfc=$(size(sfc)) sw=$(size(sw_arr))")
    τ = glazing_solar_transmittance(glazing; cos_incidence = cos_incidence)
    Nx, Ny = size(loss)
    @inbounds for j in 1:Ny, i in 1:Nx
        if sfc[i, j] == SURFACE_GLASS
            loss[i, j] += τ * sw_arr[i, j]
        end
    end
    return loss
end

"""
    GlassClass

Slot `:glazing`, provider `:glass_class`. At `init!` promotes a synthetic
set of roof cells (`i % 5 == 0 && j % 4 == 0`) to [`SURFACE_GLASS`](@ref)
and extends `bus.params` with the glass class
([`glass_surface_params`](@ref)); every step, in `:sources` after
the buildings and traffic, the transmitted shortwave leaves the glass
cells' surface budget ([`apply_glass_sw_transmission!`](@ref)).
"""
struct GlassClass{G} <: AbstractCoupling
    glazing::G
end

coupling_name(::Type{<:GlassClass}) = :glass
slot(::Type{<:GlassClass}) = :glazing
provider(::Type{<:GlassClass}) = :glass_class
modules_used(::Type{<:GlassClass}) = (:UrbanSurface,)
stages(::Type{<:GlassClass}) = ((:sources, 30, EVERY_STEP),)
build(::Type{<:GlassClass}, case, o) = GlassClass(example_glazing())

function init!(c::GlassClass, case)
    sfc = case.geom.sfc
    Nx, Ny = size(sfc)
    glass_mask = falses(Nx, Ny)
    for j in 1:Ny, i in 1:Nx
        if sfc[i, j] == SURFACE_ROOF && (i % 5 == 0) && (j % 4 == 0)
            glass_mask[i, j] = true
        end
    end
    promote_to_glass!(sfc, glass_mask)
    case.bus.params = glass_surface_params(; glazing = c.glazing)
    return nothing
end

run_stage!(c::GlassClass, case, ::Val{:sources}, dt) =
    (apply_glass_sw_transmission!(case.bus.sensible, case.geom.sfc, case.bus.sw;
                                  glazing = c.glazing); nothing)

# Ledger: the shortwave transmitted through the glass leaves the glass
# cells' surface budget (it heats the building, not the slab).
function budget!(c::GlassClass, case, l)
    T = apply_glass_sw_transmission!(zeros(size(case.bus.sw)), case.geom.sfc, case.bus.sw;
                                     glazing = c.glazing)
    add_surface!(l, :glass_transmission, -T, case.geom.sfc, l.dt)
    return nothing
end

function smoke_check(c::GlassClass, case)
    opt = glazing_optics_hemispherical(c.glazing)
    α = case.bus.params.albedo[SURFACE_GLASS]
    return ["glass cells in the window" => any(==(SURFACE_GLASS), case.geom.sfc),
            "glazing R + T + ΣA = 1 and the glass albedo is R" =>
                isapprox(opt.R + opt.T + sum(opt.A), 1; atol = 1e-9) && α == opt.R]
end

register!(GlassClass)
