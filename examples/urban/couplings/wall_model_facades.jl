# The building facades as walls of Aeolus.WallModels (#179): every fluid
# velocity face next to a facade (u across ±y, v across ±x, w across ±x
# and ±y) gets the stress of a rough-wall log law, and the stencil's own
# no-slip flux through the facade is removed, so the two are not counted
# twice.
#
# Before #179 the facades had no wall model: the direct-forcing IBM zeroes
# the velocity inside the buildings and the facade flux was whatever the
# viscous and eddy-viscosity stencils made of (ν + ν_t)(u_air − 0)/Δ — an
# unresolved no-slip, at ν = 2 m² s⁻¹ (cfg.ν) and Δ = 15.6 m. The ground
# and the roofs (the z-walls) are the surface exchange's MOSTWallLaw
# (#169); this coupling owns only the facades.

"""
    WallModelFacades

Additive coupling, on by default: the facade wall model of
`Aeolus.WallModels` (`immersed_walls`, `remove_stencil_wall_flux!`,
`apply_immersed_wall_stress!`) on the IBM solid mask (`Aeolus.IBM`).

* **Walls.** At `init!`, `immersed_walls` on `case.geom.sm`: every
  fluid face whose tangential neighbour face is zeroed by the IBM.
* **Law.** `RoughLogLaw(z0m)` (neutral; κ from `SurfaceExchange`) at
  ``h = Δ/2``, the face centre's distance from the facade. `z0m` is read
  **from the bus** (`bus.z0m_class`, the per-class table of the surface
  exchange): the `SURFACE_GLASS` value for a facade of a column the
  glass coupling tagged, the `SURFACE_WALL` value otherwise.
* **Stages.** `:body_force` (40): `remove_stencil_wall_flux!` takes the
  stencil's ``(ν + ν_{t,e})(q_N - q_F)/Δ^2`` through every facade out of
  the momentum tendency (with ``ν_t`` when the step has an eddy
  viscosity: the built-in Smagorinsky or an `eddy_viscosity!` closure of
  the `:sgs` slot). `:post_predictor` (12, after the `ibm` direct forcing
  at 10, before the ground's wall stress at 15):
  `apply_immersed_wall_stress!` applies the modelled stress implicitly.

Momentum is not in the ledger (the surface-energy and water budgets), so
there is no `budget!`; `report` prints the facade share of the surface
drag.
"""
mutable struct WallModelFacades <: AbstractCoupling
    iw::Any                  # ImmersedWalls (built at init!)
    z0m_wall::Float64
    z0m_glass::Float64
    sgs::Bool                # the step has an eddy viscosity whose facade flux is removed
end

coupling_name(::Type{WallModelFacades}) = :wall_model_facades
modules_used(::Type{WallModelFacades}) = (:WallModels, :IBM)
stages(::Type{WallModelFacades}) = ((:body_force, 40, EVERY_STEP),
                                    (:post_predictor, 12, EVERY_STEP))

build(::Type{WallModelFacades}, case, o) = WallModelFacades(nothing, NaN, NaN, false)

function init!(c::WallModelFacades, case)
    sm::SolidMask = case.geom.sm
    z0 = case.bus.z0m_class
    haskey(z0, SURFACE_WALL) ||
        error("wall_model_facades: no wall roughness on the bus (bus.z0m_class); " *
              "it is set by the :surface_exchange provider")
    c.z0m_wall = z0[SURFACE_WALL]
    c.z0m_glass = get(z0, SURFACE_GLASS, c.z0m_wall)
    sfc = case.geom.sfc
    law_wall, law_glass = RoughLogLaw(c.z0m_wall), RoughLogLaw(c.z0m_glass)
    # the facade's material: the class of the building column behind it
    lawof(i, j, k) = sfc[i, j] == SURFACE_GLASS ? law_glass : law_wall
    c.iw = immersed_walls(case.sim, sm, lawof)
    if case.cfg.verbose
        nu, nv, nw = immersed_wall_count(c.iw)
        @printf(case.io, "Facade wall model: %d u, %d v, %d w face-walls; z0m wall %.3g m, glass %.3g m\n",
                nu, nv, nw, c.z0m_wall, c.z0m_glass)
    end
    return nothing
end

# (the step's keywords are composed after init!, so the eddy-viscosity
# test is made here)
function run_stage!(c::WallModelFacades, case, ::Val{:body_force}, dt)
    sim = case.sim
    c.sgs = (hasproperty(sim, :has_sgs) && sim.has_sgs) || haskey(case.step_kwargs, :eddy_viscosity!)
    remove_stencil_wall_flux!(sim, c.iw; sgs = c.sgs)
    return nothing
end

run_stage!(c::WallModelFacades, case, ::Val{:post_predictor}, dt) =
    (apply_immersed_wall_stress!(case.sim, c.iw, dt); nothing)

"""
    facade_drag_share(c::WallModelFacades, case) -> (facade, ground, share)

The kinematic drag of the last step (m⁴ s⁻²): the facades' ``\\sum τ_w
ΔA`` and the ground's (roofs included) ``\\sum u_*^2 Δx Δy`` over the
columns with a surface, and the facade share of their sum.
"""
function facade_drag_share(c::WallModelFacades, case)
    bus, m = case.bus, case.mesh
    ground = sum(bus.u_star[I]^2 for I in eachindex(bus.u_star) if bus.surface[I] !== nothing;
                 init = 0.0) * m.dx * m.dy
    facade = c.iw.stats[:τA]
    return (facade = facade, ground = ground, share = facade / (facade + ground))
end

"""
    mean_canopy_wind(case) -> (u, v)

The mean horizontal wind over the air cells of the levels with buildings
(cell-centred, from the staggered faces).
"""
function mean_canopy_wind(case)
    m, sm = case.mesh, case.geom.sm
    u, v = interior_view(case.sim.u), interior_view(case.sim.v)
    kr = max(highest_solid_level(case), 1)
    su = sv = 0.0; n = 0
    for k in 1:kr, j in 1:m.Ny, i in 1:m.Nx
        sm.mask[m.Hx+i, m.Hy+j, m.Hz+k] && continue
        su += (u[i, j, k] + u[mod1(i + 1, m.Nx), j, k]) / 2
        sv += (v[i, j, k] + v[i, mod1(j + 1, m.Ny), k]) / 2
        n += 1
    end
    return (u = su / n, v = sv / n)
end

function report(c::WallModelFacades, case)
    d = facade_drag_share(c, case)
    F = c.iw.stats[:F]; Fs = c.iw.stats[:F_stencil]
    A = c.iw.stats[:area]
    return @sprintf("   facade wall model: ⟨τ_w⟩ %.2e m²/s² over %.0f m², %.1f %% of the surface drag; stencil drag removed |F| %.2e, modelled %.2e m⁴/s²",
                    d.facade / A, A, 100 * d.share, hypot(Fs[1], Fs[2], Fs[3]), hypot(F...))
end

function smoke_check(c::WallModelFacades, case)
    iw, m, sm = c.iw, case.mesh, case.geom.sm
    mask = sm.mask
    e = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
    # every face the model writes is a fluid face: no solid cell on either side
    solid_face(f) = (mask[f.I, f.J, f.K] ||
                     mask[f.I - e[f.c][1], f.J - e[f.c][2], f.K - e[f.c][3]])
    in_solid = count(f -> f.c != 3 || f.K > m.Hz + 1 ? solid_face(f) : true, iw.faces)
    W = mean_canopy_wind(case)
    F = iw.stats[:F]
    return ["the facades have walls (u, v and w faces)" => all(>(0), immersed_wall_count(iw)),
            "facade τ_w ≥ 0 (every drag coefficient, and Σ τ_w ΔA)" =>
                iw.stats[:τ_min] >= 0 && iw.stats[:τA] >= 0,
            "no facade stress is written into a solid cell's face" => in_solid == 0,
            "the total facade drag opposes the mean canopy wind" => F[1] * W.u + F[2] * W.v < 0,
            "the stencil's facade flux was removed (nonzero, opposing the flow too)" =>
                iw.stats[:F_stencil][1] * W.u + iw.stats[:F_stencil][2] * W.v < 0]
end

register!(WallModelFacades)
