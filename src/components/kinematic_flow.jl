# A prescribed, steady (kinematic) velocity (ONE-l, #475; the `advection`
# run type of EX-d #441, formerly examples/scalar_advection_2d.jl).

"""
    KinematicFlow(u, v, w)

A prescribed, steady velocity: the dynamics do not change it, and the
scalars (θ, and qv with moisture) are carried by it with the core's
scalar advection and time scheme. `[kinematic_flow] kind`:

  * `"solid_body_rotation"` — ``u = −ω (y − y_c)``, ``v = ω (x − x_c)``,
    ``w = 0`` (`omega`, `center`), the rotation of the plane about the
    axis ``(x_c, y_c)``; on the staggered grid ``u`` depends on ``y``
    only and ``v`` on ``x`` only, so it is discretely divergence-free
    and the projection leaves it unchanged;
  * `"uniform"` — the constant `velocity` ``(u, v, w)``;
  * `"none"` (the default) — off.

The velocity is set when the component is built (over the initial
state), at the start of every step (stage 1, `:clock`), after every
predictor (stage 8, `:post_predictor`, inside every substage) and after
the core step (stage 11, `:surface_update`: SSP-RK3 combines the
substages' velocities, which may round off the prescribed values), so
each substage's tendencies and the projection, and the state after the
step, carry exactly the prescribed field. It is stateless: a restart rebuilds it from the configuration.
The core still computes the momentum tendencies (advection, diffusion,
buoyancy and every component's), which the prescription then discards;
with `dynamics.theta0 = 0` θ is a passive scalar.
"""
struct KinematicFlow{A<:AbstractArray} <: AbstractComponent
    u::A
    v::A
    w::A
end

component_name(::Type{<:KinematicFlow}) = :kinematic_flow
section(::Type{<:KinematicFlow}) = "kinematic_flow"
schema(::Type{<:KinematicFlow}) =
    Config.section("kinematic_flow"; doc = "A prescribed, steady velocity that carries the scalars (no dynamics).") do s
        Config.option!(s, "kind", Symbol; default = :none, allowed = [:none, :solid_body_rotation, :uniform],
                       doc = "none (off); solid_body_rotation (u = −ω (y − y_c), v = ω (x − x_c), w = 0) or uniform (velocity)")
        Config.option!(s, "omega", Float64; default = 0.0, units = "s^-1",
                       doc = "solid_body_rotation: the angular velocity ω")
        Config.option!(s, "center", NTuple{2,Float64}; default = (0.0, 0.0), units = "m",
                       doc = "solid_body_rotation: the axis (x_c, y_c)")
        Config.option!(s, "velocity", NTuple{3,Float64}; default = (0.0, 0.0, 0.0), units = "m s^-1",
                       doc = "uniform: (u, v, w)")
    end
enabled(::Type{<:KinematicFlow}, opts) = opts.kind !== :none

function build(::Type{<:KinematicFlow}, opts, domain)
    s = domain.state
    if opts.kind === :solid_body_rotation
        ω = Float64(opts.omega); xc, yc = opts.center
        set!(s.u, (x, y, z) -> -ω * (y - yc)); set!(s.v, (x, y, z) -> ω * (x - xc)); set!(s.w, (x, y, z) -> 0.0)
    else
        U, V, W = opts.velocity
        set!(s.u, (x, y, z) -> U); set!(s.v, (x, y, z) -> V); set!(s.w, (x, y, z) -> W)
    end
    fill_halos!(s.u, s.bcs); fill_halos!(s.v, s.bcs); fill_halos!(s.w, s.bcs)
    return KinematicFlow(copy(s.u.data), copy(s.v.data), copy(s.w.data))
end
stages(::KinematicFlow) = (Stage(:clock), Stage(:post_predictor), Stage(:surface_update))
# it prescribes the velocity: the budgets measure it on the momentum only (#539)
budget_writes(::KinematicFlow, ::Val) = (:momentum_x, :momentum_y)

function _prescribe!(c::KinematicFlow, d)
    s = d.state
    copyto!(s.u.data, c.u); copyto!(s.v.data, c.v); copyto!(s.w.data, c.w)
    return nothing
end
run_stage!(c::KinematicFlow, d, ::Val{:clock}, dt) = _prescribe!(c, d)
run_stage!(c::KinematicFlow, d, ::Val{:post_predictor}, dt) = _prescribe!(c, d)
run_stage!(c::KinematicFlow, d, ::Val{:surface_update}, dt) = _prescribe!(c, d)

# The prescription in the fused sweep (#533 step 6): each predicted row is
# overwritten with the prescribed one right after its predictor
# (pointwise, footprint 0). The halos keep their stage-start values, the
# prescribed ones the stage started from; the projection refills them.
struct KinematicFlowTerm{A<:AbstractArray} <: FiniteVolume.PostTerm
    u::A
    v::A
    w::A
end
FiniteVolume.footprint(::KinematicFlowTerm) = (0, 0, 0)
@inline function FiniteVolume.post_row!(t::KinematicFlowTerm, ::Val{2}, j, k, i1, i2, st)
    u, v, w, pu, pv, pw = st.u, st.v, st.w, t.u, t.v, t.w
    @inbounds @simd for i in i1:i2
        u[i, j, k] = pu[i, j, k]
        v[i, j, k] = pv[i, j, k]
        w[i, j, k] = pw[i, j, k]
    end
    return nothing
end
FiniteVolume.post_sweep_term(c::KinematicFlow, d, Δt) = KinematicFlowTerm(c.u, c.v, c.w)
