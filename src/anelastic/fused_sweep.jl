# The fused sweep of the sound-proof cores (#533 plan step 8). Included into
# `module Anelastic`, after `_mass_weighted_step!`.
#
# A substage of `step_anelastic!` / `step_pseudo_incompressible!` (and of
# their `step_rk3!`) runs its tendencies and the predictors of u, v, w and
# θ in one fused pass, the term interface of the Boussinesq core
# (`FiniteVolume.SweepTerm`), when every term is a sweep term: the base
# term is the W-weighted flux-form advection with the molecular diffusion
# (`WeightedBaseTendencies`), the buoyancy g(θ − θ̄)/θ̄ is
# `AnelasticBuoyancy`, and the pipeline's components come through the
# body-force hook (`Simulations.hook_terms`). Otherwise the passes of
# `_mass_weighted_step!` (the reference), and the simulation's report names
# the terms (`Simulations.SWEEP_FALLBACKS`).

using ..Operators.FiniteVolume: FiniteVolume, SweepTerm, BaseTerm, SweepRow, SWEEP_U, SWEEP_V, SWEEP_W,
                                SWEEP_Θ, _visc, grid_metrics

"""
    WeightedBaseTendencies(ν, Wc, Wf, idx, idy, idz)

The base term of the sound-proof cores in the fused sweep (#533): the
`W`-weighted flux-form advection `−(1/W) ∂(W uⱼuᵢ)/∂xⱼ` of u, v, w
([`weighted_momentum_advect!`](@ref)) and `−(1/W) ∂(W uⱼθ)/∂xⱼ` of θ
([`weighted_scalar_advect!`](@ref)), plus the molecular `ν∇²` (`ν =
nothing`: none), with the weights `Wc`, `Wf` of
[`constraint_weights`](@ref) and the inverse spacings `idx, idy, idz`.
Each value is the expression of those whole-field passes (the same
point functions).
"""
struct WeightedBaseTendencies{V,T,A<:AbstractVector{T}} <: BaseTerm
    ν::V
    Wc::A
    Wf::A
    idx::T
    idy::T
    idz::T
end
FiniteVolume.footprint(::WeightedBaseTendencies) = (1, 1, 1)
FiniteVolume.sweep_inputs(t::WeightedBaseTendencies) = (t.Wc, t.Wf)

@inline function FiniteVolume.init_point!(t::WeightedBaseTendencies, tb, i, j, k, st)
    u, v, w, m = st.u, st.v, st.w, st.m
    Wc, Wf, idx, idy, idz, ν = t.Wc, t.Wf, t.idx, t.idy, t.idz, t.ν
    tb[i, SWEEP_U] = _visc(ν, zero(eltype(u)) - _wmom_u(i, j, k, u, v, w, Wc, Wf, idx, idy, idz), i, j, k, u, m, FaceX())
    tb[i, SWEEP_V] = _visc(ν, zero(eltype(v)) - _wmom_v(i, j, k, u, v, w, Wc, Wf, idx, idy, idz), i, j, k, v, m, FaceY())
    tb[i, SWEEP_W] = _visc(ν, zero(eltype(w)) - _wmom_w(i, j, k, u, v, w, Wc, Wf, idx, idy, idz), i, j, k, w, m, FaceZ())
    θ = st.θ
    tb[i, SWEEP_Θ] = _visc(ν, zero(eltype(θ)) - _wscal(i, j, k, u, v, w, θ, Wc, Wf, idx, idy, idz), i, j, k, θ, m,
                           CellCenter())
    return nothing
end

# the row form for the ring-buffer sweep: momentum and θ in separate loops
# (as the Boussinesq base term; @noinline, the arrays as arguments)
@inline FiniteVolume.init_row!(t::WeightedBaseTendencies, buf, a, b, s, j, k, ii, st) =
    (_wtrow_mom!(buf, a, b, s, j, k, ii, st.u, st.v, st.w, t.Wc, t.Wf, t.idx, t.idy, t.idz, st.m, t.ν);
     _wtrow_θ!(buf, a, b, s, j, k, ii, st.u, st.v, st.w, st.θ, t.Wc, t.Wf, t.idx, t.idy, t.idz, st.m, t.ν))
@noinline function _wtrow_mom!(buf, a, b, s, j, k, ii, u, v, w, Wc, Wf, idx, idy, idz, m, ν)
    @inbounds @simd ivdep for i in ii
        buf[i, a, b, 1, s] = _visc(ν, zero(eltype(u)) - _wmom_u(i, j, k, u, v, w, Wc, Wf, idx, idy, idz),
                                   i, j, k, u, m, FaceX())
        buf[i, a, b, 2, s] = _visc(ν, zero(eltype(v)) - _wmom_v(i, j, k, u, v, w, Wc, Wf, idx, idy, idz),
                                   i, j, k, v, m, FaceY())
        buf[i, a, b, 3, s] = _visc(ν, zero(eltype(w)) - _wmom_w(i, j, k, u, v, w, Wc, Wf, idx, idy, idz),
                                   i, j, k, w, m, FaceZ())
    end
    return nothing
end
@noinline function _wtrow_θ!(buf, a, b, s, j, k, ii, u, v, w, θ, Wc, Wf, idx, idy, idz, m, ν)
    @inbounds @simd ivdep for i in ii
        buf[i, a, b, 4, s] = _visc(ν, zero(eltype(θ)) - _wscal(i, j, k, u, v, w, θ, Wc, Wf, idx, idy, idz),
                                   i, j, k, θ, m, CellCenter())
    end
    return nothing
end

"""
    AnelasticBuoyancy(θ̄f, g)

The buoyancy of the sound-proof cores as a term of the fused sweep (#533):
`g (θ − θ̄(z))/θ̄(z)` at the w points, with `θ` averaged to the face and
`θ̄f` the reference profile at the faces (`ref.θ̄f_pad`), the expression of
[`add_anelastic_buoyancy!`](@ref). Reads `θ[k−1]`: footprint `(0, 0, 1)`.
"""
struct AnelasticBuoyancy{T,A<:AbstractVector{T}} <: SweepTerm
    θ̄f::A
    g::T
end
FiniteVolume.footprint(::AnelasticBuoyancy) = (0, 0, 1)
FiniteVolume.sweep_inputs(t::AnelasticBuoyancy) = (t.θ̄f,)
@inline FiniteVolume.add_tendency!(t::AnelasticBuoyancy, tb, i, j, k, st) =
    (tb[i, SWEEP_W] += _anelastic_buoyancy(i, j, k, st.θ, t.θ̄f, t.g); nothing)

# What keeps a substage of a sound-proof core off the fused sweep (only
# when reporting): its terms without a fused form yet.
function Simulations._sweep_blockers(sim::MassWeightedSimulation, closure, body_force!, post_predictor!,
                                     momentum_advection)
    out = Pair{String,String}[]
    has_open_boundaries(sim) && push!(out, Simulations._blocker("builtin.anelastic_open_boundaries"))
    closure === nothing || push!(out, Simulations._blocker("builtin.anelastic_sgs"))
    sim.has_coriolis && push!(out, Simulations._blocker("builtin.anelastic_coriolis"))
    sim.has_surface_flux && push!(out, Simulations._blocker("builtin.anelastic_surface_flux"))
    sim.has_sponge && push!(out, Simulations._blocker("builtin.anelastic_sponge"))
    body_force! === nothing || append!(out, Simulations.hook_blockers(body_force!, sim))
    Simulations._hook_books_per_point(sim, body_force!) && push!(out, Simulations._blocker("builtin.budgets_per_point"))
    if post_predictor! !== nothing        # (no post terms in these cores yet, #533 step 6)
        append!(out, Simulations.post_hook_blockers(post_predictor!, sim))
        push!(out, Simulations._blocker("builtin.anelastic_post_predictor"))
    end
    return unique!(out)
end

# Steps 2-7 and 11 of `_mass_weighted_step!` in the fused sweep, then the
# projection, the pressure and the clock; `false` (the substage counted
# as unfused) when a term has no fused form. The base term and the
# buoyancy are chosen by branches, so every call below is concrete.
function _try_fused_mw!(sim::MassWeightedSimulation{S,T}, Δt::T, closure, body_force!, post_predictor!,
                        Wc, Wf) where {S,T}
    r = sim.workspaces.fused
    if r.loop_fusion === :never
        r.unfused += 1
        return false
    end
    m = sim.mesh
    idx, idy, idz = 1 / m.dx, 1 / m.dy, 1 / m.dz
    if sim.has_viscosity
        base = WeightedBaseTendencies(sim.ν, Wc, Wf, idx, idy, idz)
        return sim.has_buoyancy ?
            _fused_mw!(sim, Δt, closure, body_force!, post_predictor!, Wc, Wf, base,
                       (AnelasticBuoyancy(sim.ref.θ̄f_pad, sim.ref.g),)) :
            _fused_mw!(sim, Δt, closure, body_force!, post_predictor!, Wc, Wf, base, ())
    else
        base = WeightedBaseTendencies(nothing, Wc, Wf, idx, idy, idz)
        return sim.has_buoyancy ?
            _fused_mw!(sim, Δt, closure, body_force!, post_predictor!, Wc, Wf, base,
                       (AnelasticBuoyancy(sim.ref.θ̄f_pad, sim.ref.g),)) :
            _fused_mw!(sim, Δt, closure, body_force!, post_predictor!, Wc, Wf, base, ())
    end
end

# the term tuple of a substage, or `nothing` when one term has no fused form
function _mw_sweep_terms(sim::MassWeightedSimulation, closure, body_force!, post_predictor!, base, extra)
    has_open_boundaries(sim) && return nothing
    closure === nothing || return nothing
    (sim.has_coriolis || sim.has_surface_flux || sim.has_sponge) && return nothing
    post_predictor! === nothing || return nothing
    body_force! === nothing && return (base, extra...)
    ht = Simulations.hook_terms(body_force!, sim)
    ht === nothing && return nothing
    Simulations._books_per_point(sim, ht) && return nothing
    return (base, extra..., ht...)
end

function _fused_mw!(sim::MassWeightedSimulation{S,T}, Δt::T, closure, body_force!, post_predictor!,
                    Wc, Wf, base, extra) where {S,T}
    r = sim.workspaces.fused
    terms = _mw_sweep_terms(sim, closure, body_force!, post_predictor!, base, extra)
    if terms === nothing
        r.loop_fusion === :always && Simulations._refuse_unfused(sim, closure, body_force!, post_predictor!, nothing)
        Simulations._note_unfused!(sim, closure, body_force!, post_predictor!, nothing)
        return false
    end
    r.fused += 1
    bufs = Simulations._fused_buffers!(sim, terms, 4)
    Simulations._prepare_terms!(terms, sim, Δt)
    ax = Simulations._Axpy(Δt)
    Simulations.fused_predictor!(sim, bufs, terms, ax, (sim.θ, ax, nothing), nothing, nothing)
    Simulations._finish_sweep_terms!(terms, sim, Δt)   # the budget term books (#533 step 7)
    _project_mass_weighted!(sim, Wc, Wf)        # 8.-10. (does not read θ, updated in the sweep)
    pressure_from_potential!(sim.p, sim.φ, Δt)  # 12.
    sim.t += Δt
    return true
end
