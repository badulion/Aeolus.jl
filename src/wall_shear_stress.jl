# P59-a (#172) — plane-mean wall shear stress of a channel simulation.
#
# A method of `CutCell.wall_shear_stress` (the per-stencil cut-cell
# diagnostic), so the exported name is not duplicated. Included at the
# top level of `Aeolus`, after `CutCell` and `Simulations` are loaded.

"""
    wall_shear_stress(sim::BoussinesqSimulation; side = :bottom, component = :u) -> T

Plane-mean molecular wall shear stress `ν ∂u/∂z` (m² s⁻², kinematic) on
the `side = :bottom` or `:top` z-wall, for the velocity `component = :u`
or `:v`.

The gradient is the one the discrete viscous operator uses: the halos are
filled with `sim.bcs`, and the one-sided difference across the wall face
is `(u₁ − u_ghost)/Δz`. On a [`NoSlipWall`](@ref) this is
`2(u₁ − u_wall)/Δz`, second-order accurate at the wall; on a free-slip
[`ImpermeableWall`](@ref) it is exactly zero. The sign is that of
`∂u/∂z`: a fluid moving faster than the bottom wall gives a positive
bottom stress, and the plane-integrated momentum budget of the column is

    d/dt ∫ ū dz = τ(top) − τ(bottom) + ∫ forcing dz .

Only the molecular stress `ν` is included (the SGS closure is not a wall
model; a wall-modelled run reports its own stress).

Reference: Pope (2000), *Turbulent Flows*, §7.1.
"""
function CutCell.wall_shear_stress(sim::Simulations.BoussinesqSimulation{T};
                                   side::Symbol = :bottom,
                                   component::Symbol = :u) where {T}
    side in (:bottom, :top) ||
        throw(ArgumentError("side must be :bottom or :top, got :$side"))
    component in (:u, :v) ||
        throw(ArgumentError("component must be :u or :v, got :$component"))
    bc = side === :bottom ? sim.bcs.bottom : sim.bcs.top
    BoundaryConditions.is_wall(bc) ||
        throw(ArgumentError("wall_shear_stress: the $side face is a " *
                            "$(nameof(typeof(bc))), not a wall"))
    f = component === :u ? sim.u : sim.v
    fill_halos!(f, sim.bcs)
    m = sim.mesh
    ii, jj, _ = Grids.interior(m)
    k_in, k_gh = side === :bottom ? (m.Hz + 1, m.Hz) : (m.Hz + m.Nz, m.Hz + m.Nz + 1)
    d = f.data
    s = zero(T)
    @inbounds for j in jj, i in ii
        s += d[i, j, k_in] - d[i, j, k_gh]
    end
    dudz = s / (length(ii) * length(jj) * m.dz)
    return T(sim.ν) * (side === :bottom ? dudz : -dudz)
end
