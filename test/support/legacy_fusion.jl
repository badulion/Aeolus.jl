# SC-d (#404): the Boussinesq step cores as they were before the kernel
# fusion (branch point 9de7c553 = SC-a #401 + SC-b #402 + master),
# transcribed verbatim from src/Simulations.jl: `_step_euler!` (step!),
# `_euler_projection!`, `_project_velocity!` and `_step_rk3!`, each a
# sequence of single-purpose passes (zero the tendencies, advection,
# Laplacian + axpy, ∂φ + axpy, the RK3 combinations over the storage).
# The reference of test/ka_fusion.jl . Nothing here is used
# by the package.
# ---------------------------------------------------------------------------
module LegacyFusion
using Aeolus
using Aeolus.Backends: Backends, range_map!
using Aeolus.Fields: Field
using Aeolus.Simulations: BoussinesqSimulation, RK3Workspace, rk3_workspace, axpy!, zero_storage!,
                          copy_storage!, pressure_from_potential!, _Lin2r, _note_solve!, _wall,
                          fill_state_halos!
using Aeolus.Operators: divergence, momentum_advect!, laplacian, ∂x, ∂y, ∂z, advect_centered!,
                        add_buoyancy!, add_coriolis!, add_surface_flux!, compute_sgs!, add_sgs_momentum!,
                        add_sgs_scalar!, add_eddy_diffusion_scalar!, advance_sgs!, _active_sgs,
                        _check_sgs_stress, sgs_tracer_diffusivity!
using Aeolus.Simulations: add_velocity_sponge!
using Aeolus.BoundaryConditions: fill_halos!, update_open_boundaries!, prepare_open_boundaries!,
                                 assert_open_face_correction, open_boundary_snapshot,
                                 combine_open_boundaries!, mark_open_boundary_stage!
using Aeolus.PressureSolvers: solve!

"""
    step!(sim, dt; kwargs...)
    step_rk3!(sim, dt; kwargs...)

The pre-fusion `step!` and `step_rk3!` (without the step record).
"""
step!(sim::BoussinesqSimulation{T}, dt::Real; kwargs...) where {T} = _step_euler!(sim, T(dt); kwargs...)
step_rk3!(sim::BoussinesqSimulation{T}, dt::Real; kwargs...) where {T} = _step_rk3!(sim, T(dt); kwargs...)

function _step_euler!(sim::BoussinesqSimulation{T}, dt::Real;
                      body_force!::Union{Nothing, Function} = nothing,
                      post_predictor!::Union{Nothing, Function} = nothing,
                      momentum_advection::F = momentum_advect!,
                      sgs = nothing,
                      eddy_viscosity! = nothing,
                      sgs_stress::Symbol = :laplacian) where {T, F}
    Δt = T(dt)
    closure = _active_sgs(sim, sgs, eddy_viscosity!)
    prepare_open_boundaries!(sim)          # no-op unless an open face is unfilled
    _euler_projection!(sim, Δt, closure, body_force!, post_predictor!, momentum_advection,
                       _check_sgs_stress(sgs_stress))
    # the one commit of the closure's memory, after the projection
    closure === nothing || advance_sgs!(closure, sim, Δt)
    return sim
end

function _add_sgs_tracer!(tϕ, ϕ, closure, f, sim, scratch)
    κ = sgs_tracer_diffusivity!(scratch, closure, f, sim, ϕ)
    if κ === f.κh
        add_sgs_scalar!(tϕ, ϕ, closure, f, sim)
    else
        add_eddy_diffusion_scalar!(tϕ, ϕ, κ)
    end
    return tϕ
end

function _euler_projection!(sim::BoussinesqSimulation{T}, Δt::T,
                            closure::C, body_force!, post_predictor!,
                            momentum_advection::F,
                            sgs_stress::Symbol = :laplacian,
                            t_bc = sim.t + Δt) where {T, C, F}
    # 1. Halo fill so all stencils see valid neighbours.
    fill_state_halos!(sim)
    mark_open_boundary_stage!(sim)   # radiating outflow: stage-start layers (N-open-c)

    # 2. Zero tendencies and add momentum + scalar advection.
    zero_storage!(sim.tu); zero_storage!(sim.tv); zero_storage!(sim.tw)
    zero_storage!(sim.tθ)
    sim.has_moisture && zero_storage!(sim.tqv)
    momentum_advection(sim.tu, sim.tv, sim.tw, sim.u, sim.v, sim.w)
    advect_centered!(sim.tθ, sim.u, sim.v, sim.w, sim.θ)
    sim.has_moisture && advect_centered!(sim.tqv, sim.u, sim.v, sim.w, sim.qv)

    # 3. Molecular viscosity / diffusion.
    if sim.has_viscosity
        laplacian(sim.lap_u, sim.u); laplacian(sim.lap_v, sim.v); laplacian(sim.lap_w, sim.w)
        laplacian(sim.lap_θ, sim.θ)
        axpy!(sim.tu, sim.ν, sim.lap_u)
        axpy!(sim.tv, sim.ν, sim.lap_v)
        axpy!(sim.tw, sim.ν, sim.lap_w)
        axpy!(sim.tθ, sim.ν, sim.lap_θ)
        if sim.has_moisture                 # qv: the same molecular diffusivity as θ (#177)
            laplacian(sim.lap_θ, sim.qv)
            axpy!(sim.tqv, sim.ν, sim.lap_θ)
        end
    end

    # 4. SGS closure (the built-in Smagorinsky, or the `sgs` keyword): the
    #    coefficients from the current state, then the tendencies. Every
    #    transported scalar gets the closure's scalar flux: θ and, when
    #    moist, qv (P56-a, #177 — before it qv was advected but never
    #    diffused, a code bug).
    if closure !== nothing
        t_sgs = time_ns()
        f = compute_sgs!(closure, sim)
        add_sgs_momentum!(sim.tu, sim.tv, sim.tw, closure, f, sim; stress = sgs_stress)
        add_sgs_scalar!(sim.tθ, sim.θ, closure, f, sim)
        sim.has_moisture && _add_sgs_tracer!(sim.tqv, sim.qv, closure, f, sim, sim.lap_θ)
        sim.workspaces.rec.wall_sgs += _wall(t_sgs)
    end

    # 5. Buoyancy on w-momentum. Moist buoyancy uses virtual θ when qv is on.
    if sim.has_buoyancy
        if sim.has_moisture
            add_buoyancy!(sim.tw, sim.θ, sim.qv, sim.θ₀, sim.g_accel)
        else
            add_buoyancy!(sim.tw, sim.θ, sim.θ₀, sim.g_accel)
        end
    end

    # 6. Coriolis on horizontal momenta.
    if sim.has_coriolis
        add_coriolis!(sim.tu, sim.tv, sim.u, sim.v, sim.f_coriolis)
    end

    # 6a. Prescribed surface fluxes on θ (Phase 7).
    if sim.has_surface_flux
        add_surface_flux!(sim.tθ, sim.θ_flux_bottom, sim.θ_flux_top)
    end

    # 6b. Rayleigh sponge on velocity (Phase 7), toward `sim.sponge_target`
    #     (zero for a closed top; the boundary data's state for an open one).
    sim.has_sponge && add_velocity_sponge!(sim)

    # 6c. User body force / IBM penalty — last tendency contribution.
    body_force! === nothing || body_force!(sim, Δt)

    # 7. Predictor: u* = u + Δt F  (in-place).
    axpy!(sim.u, Δt, sim.tu)
    axpy!(sim.v, Δt, sim.tv)
    axpy!(sim.w, Δt, sim.tw)

    # 7b. Post-predictor hook — stiff corrections (implicit IBM, mask
    # enforcement, ...). Runs after u* is formed and before the Poisson
    # solve so the projection folds any velocity correction into a
    # divergence-free result.
    post_predictor! === nothing || post_predictor!(sim, Δt)

    # 7c. Open boundaries (N-open-b): boundary data at the end of the step,
    # mass-corrected so the projection's Neumann problem is compatible. The
    # halo fill below writes the corrected normal velocity into u*.
    update_open_boundaries!(sim, t_bc, Δt)

    # 8.–10. Chorin projection of the predicted velocity.
    _project_velocity!(sim)

    # 11. Advance θ (no projection needed for scalar).
    axpy!(sim.θ, Δt, sim.tθ)
    sim.has_moisture && axpy!(sim.qv, Δt, sim.tqv)

    # 12. Recover physical pressure and advance clock.
    pressure_from_potential!(sim.p, sim.φ, Δt)
    sim.t += Δt

    return sim
end

function _project_velocity!(sim::BoussinesqSimulation{T}) where {T}
    # 8. Divergence of predicted velocity.
    fill_halos!((sim.u, sim.v, sim.w), sim.bcs)
    divergence(sim.div_buf, sim.u, sim.v, sim.w)

    # 9. Solve ∇²φ = ∇·u*.
    t_solve = time_ns()
    solve!(sim.φ, sim.div_buf, sim.solver)
    _note_solve!(sim.workspaces.rec, sim.solver, t_solve)   # iterations, residual, wall (#336)

    # 10. Correct velocity.
    fill_halos!(sim.φ, sim.bcs)
    ∂x(sim.dφdx, sim.φ); ∂y(sim.dφdy, sim.φ); ∂z(sim.dφdz, sim.φ)
    assert_open_face_correction(sim.bcs, sim.dφdx, sim.dφdy, sim.dφdz)   # ∂φ/∂n = 0 on open faces
    axpy!(sim.u, -one(T), sim.dφdx)
    axpy!(sim.v, -one(T), sim.dφdy)
    axpy!(sim.w, -one(T), sim.dφdz)
    return sim
end

function _step_rk3!(sim::BoussinesqSimulation{T}, dt::Real;
                    sgs = nothing, eddy_viscosity! = nothing,
                    body_force!::Union{Nothing, Function} = nothing,
                    post_predictor!::Union{Nothing, Function} = nothing,
                    sgs_stress::Symbol = :laplacian,
                    workspace::RK3Workspace = rk3_workspace(sim)) where {T}
    Δt = T(dt)
    closure = _active_sgs(sim, sgs, eddy_viscosity!)
    _check_sgs_stress(sgs_stress)
    prepare_open_boundaries!(sim)
    t0 = sim.t
    bc0 = open_boundary_snapshot(sim)      # nothing for closed boundaries
    if bc0 !== nothing                     # uⁿ carries its boundary state into the combinations
        fill_state_halos!(sim, sim.θ_bcs)
    end

    # Save current state (into the workspace: no allocation per step, #295;
    # copy and combinations are kernels over the whole storage, KA-j #306)
    ws = workspace
    u0 = copy_storage!(ws.u0, sim.u.data); v0 = copy_storage!(ws.v0, sim.v.data)
    w0 = copy_storage!(ws.w0, sim.w.data); θ0 = copy_storage!(ws.θ0, sim.θ.data)
    moist = sim.has_moisture
    q0 = moist ? copy_storage!(ws.q0, sim.qv.data) : θ0      # qv is combined like θ (#177)

    _rk3_substep!(sim, Δt, closure, body_force!, post_predictor!, sgs_stress,
                  t0 + Δt)                                                        # u1 = uⁿ + Δt L(uⁿ), then project
    # u1 stays in sim (the next substage advances it in place)

    _rk3_substep!(sim, Δt, closure, body_force!, post_predictor!, sgs_stress,
                  t0 + 2Δt)                                                       # u1 → u1 + Δt L(u1), then project
    # combine to u2 = 3/4 uⁿ + 1/4 (…)
    c2 = _Lin2r(0.75, 0.25)
    combine_storage!(c2, sim.u.data, u0); combine_storage!(c2, sim.v.data, v0)
    combine_storage!(c2, sim.w.data, w0); combine_storage!(c2, sim.θ.data, θ0)
    moist && combine_storage!(c2, sim.qv.data, q0)
    combine_open_boundaries!(sim, 0.75, bc0, 0.25)

    _rk3_substep!(sim, Δt, closure, body_force!, post_predictor!, sgs_stress,
                  t0 + 3Δt / 2)                                                   # u2 → u2 + Δt L(u2), then project
    # combine to uⁿ⁺¹ = 1/3 uⁿ + 2/3 (…)
    inv3 = one(T)/3; two3 = 2*inv3
    c3 = _Lin2r(inv3, two3)
    combine_storage!(c3, sim.u.data, u0); combine_storage!(c3, sim.v.data, v0)
    combine_storage!(c3, sim.w.data, w0); combine_storage!(c3, sim.θ.data, θ0)
    moist && combine_storage!(c3, sim.qv.data, q0)
    combine_open_boundaries!(sim, inv3, bc0, two3)

    # sim.t was advanced 3× by _rk3_substep!; correct to one nominal step
    sim.t = sim.t - 2Δt
    closure === nothing || advance_sgs!(closure, sim, Δt)   # once per step
    return sim
end

_rk3_substep!(sim::BoussinesqSimulation, Δt, closure, body_force! = nothing,
              post_predictor! = nothing, sgs_stress::Symbol = :laplacian,
              t_bc = sim.t + Δt) =
    _euler_projection!(sim, Δt, closure, body_force!, post_predictor!, momentum_advect!,
                       sgs_stress, t_bc)

# `out = a·snap + b·out` over the whole storage (the RK3 stage combinations)
function combine_storage!(f, out::AbstractArray{T,3}, snap::AbstractArray{T,3}) where {T}
    range_map!(Backends.backend(out), f, out, map(r -> UnitRange{Int}(r), axes(out)), out, snap)
    return out
end

end # module
