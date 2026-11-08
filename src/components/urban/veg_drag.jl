# Park canopy drag: a uniform crown over the lowest `k_height` layers of
# the park columns, applied implicitly after the predictor, so it is
# stable at any dt. The physics is `PlantCanopy.vegetation_canopy_drag` /
# `apply_canopy_drag!` (issue #22; moved there in ONE-e, #454).

"""
    VegDrag

Additive coupling: the resolved park canopy drag of `Aeolus.PlantCanopy`
in `:post_predictor` (after the IBM). Options `cd` (leaf drag
coefficient, 0.2) and `k_height` (crown depth in layers, 3 → 18.75 m).
The LAI is the static driver's `lai` on park columns ([`park_lai_map`](@ref)).
"""
struct VegDrag{D} <: UrbanComponent
    drag::D
end

component_name(::Type{<:VegDrag}) = :veg_drag
modules_used(::Type{<:VegDrag}) = (:PlantCanopy,)
default_options(::Type{<:VegDrag}) = (cd = 0.2, k_height = 3)
stages(::Type{<:VegDrag}) = ((:post_predictor, 20, EVERY_STEP),)

function build(::Type{<:VegDrag}, case, o)
    veg_mask, LAI_map = park_lai_map(case)
    return VegDrag(vegetation_canopy_drag(case.mesh, veg_mask, LAI_map;
                                          cd = o.cd, k_height = o.k_height))
end

run_stage!(c::VegDrag, case, ::Val{:post_predictor}, dt) =
    (apply_canopy_drag!(case.sim, c.drag, dt); nothing)


# in the fused sweep (#533 step 6): the same drag on the sweep's delayed rows
FiniteVolume.post_sweep_term(c::VegDrag, d, Δt) = Aeolus.PlantCanopy.canopy_drag_term(c.drag, d.state, Δt)
Components.post_after_sweep(::VegDrag) = true     # (footprint 1: a domain runs it after the sweep)
