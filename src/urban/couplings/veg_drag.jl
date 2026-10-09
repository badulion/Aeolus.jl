# Park canopy drag: a uniform crown over the lowest `k_height` layers of
# the park columns, applied implicitly after the predictor
# (Aeolus.PlantCanopy, issue #22), so it is stable at any dt. The old
# explicit tendency shipped disabled after an instability at dt = 2 s that
# #22 could not reproduce (dt·cd·LAI·|U| <= 0.05 here); cause undiagnosed.

"""
    vegetation_canopy_drag(mesh, veg_mask, LAI_arr; cd=0.2, k_height=nothing)
        -> Aeolus.PlantCanopy.ResolvedCanopyDrag

Build the resolved canopy drag for the vegetated columns of the city.
Each column with `veg_mask[i, j]` and `LAI_arr[i, j] > 0` gets a uniform
leaf-area-density crown `UniformLAD(LAI, h)` with the canopy height
`h = k_height · dz` (`k_height === nothing` → `Nz ÷ 4` layers, the old
default). The physics — LAD profiles, the implicit drag factor and the
staggered-grid application — lives in `Aeolus.PlantCanopy`; this is
only the city-specific wiring.

Build it once, outside the time loop, and apply it with
[`apply_vegetation_drag!`](@ref) from the `post_predictor!` hook.
"""
function vegetation_canopy_drag(mesh, veg_mask::AbstractMatrix{Bool},
                                LAI_arr::AbstractMatrix{<:Real};
                                cd::Real = 0.2,
                                k_height::Union{Nothing, Integer} = nothing)
    Nz = mesh.Nz
    k_top = clamp(k_height === nothing ? max(1, Nz ÷ 4) : Int(k_height), 1, Nz)
    h = k_top * mesh.dz
    profiles = Union{Nothing, UniformLAD{Float64}}[
        (veg_mask[i, j] && LAI_arr[i, j] > 0) ? UniformLAD(LAI_arr[i, j], h) : nothing
        for i in axes(veg_mask, 1), j in axes(veg_mask, 2)]
    return ResolvedCanopyDrag(mesh, canopy_lad_field(mesh, profiles); C_d = cd)
end

"""
    apply_vegetation_drag!(sim, drag, dt)

Apply the canopy drag `-C_d a |U| U` *implicitly* — delegates to
`Aeolus.PlantCanopy.apply_canopy_drag!`, which multiplies the predicted
velocity by `1 / (1 + dt C_d a |U|)`. Call it from the `post_predictor!`
hook of `step!`; `drag` comes from [`vegetation_canopy_drag`](@ref).

History: this used to be an explicit tendency `-cd · LAI · |u| · u` added
in `body_force!`. It used LAI (m² m⁻²) where the leaf-area *density*
(m² m⁻³) belongs and read `u`, `v`, `w` co-located at the same index on
the staggered grid. The example shipped with it disabled after an
instability at dt = 2 s was reported; re-running it in #22 (cd = 0.01 and
0.03, 1 h at dt = 2 s) did not reproduce that, so its cause is undiagnosed.
The implicit form is stable at any dt (issue #22).
"""
apply_vegetation_drag!(sim, drag::ResolvedCanopyDrag, dt::Real) =
    apply_canopy_drag!(sim, drag, dt)

"""
    VegDrag

Additive coupling: the resolved park canopy drag of `Aeolus.PlantCanopy`
in `:post_predictor` (after the IBM). Options `cd` (leaf drag
coefficient, 0.2) and `k_height` (crown depth in layers, 3 → 18.75 m).
The LAI is the static driver's `lai` on park columns ([`park_lai_map`](@ref)).
"""
struct VegDrag{D} <: AbstractCoupling
    drag::D
end

coupling_name(::Type{<:VegDrag}) = :veg_drag
modules_used(::Type{<:VegDrag}) = (:PlantCanopy,)
default_options(::Type{<:VegDrag}) = (cd = 0.2, k_height = 3)
stages(::Type{<:VegDrag}) = ((:post_predictor, 20, EVERY_STEP),)

function build(::Type{<:VegDrag}, case, o)
    veg_mask, LAI_map = park_lai_map(case)
    return VegDrag(vegetation_canopy_drag(case.mesh, veg_mask, LAI_map;
                                          cd = o.cd, k_height = o.k_height))
end

run_stage!(c::VegDrag, case, ::Val{:post_predictor}, dt) =
    apply_vegetation_drag!(case.sim, c.drag, dt)

register!(VegDrag)
