# Nesting: a mixed core pair (NEST-p2, #482) — a Boussinesq domain nested
# in a sound-proof (anelastic) one, or the reverse, one-way or two-way,
# with a conservative density conversion in the boundary data and the
# feedback.
#
# The two cores conserve different things: the Boussinesq core the volume
# flux ∇·u = 0 (its mass is ρ₀ V, ρ₀ the constant density of its buoyancy
# reference), the anelastic core the mass flux ∇·(ρ̄u) = 0. Across the
# interface the conserved quantity is the mass flux, and a scalar's flux is
# that mass flux times the (intensive) scalar. The nest therefore
#
#   * measures each side with its own density, W = ρ₀ or W = ρ̄(z): the
#     mass measure `nest_measure(nest, W_p, W_c)`, whose covered parent
#     layers carry the child's mass (so an intensive scalar — θ, qv, a
#     tracer — is prolongated and restricted as a mass-weighted mean, a
#     constant staying that constant);
#   * converts the velocities: a parent layer K whose mass is
#     W_p,K Δz_K and whose child mass is M_c,K = Σ W_c Δz_c sends
#     u_c with Σ_c W_c u_c Δz_c = W_p,K u_K Δz_K — the parent velocity
#     scaled by r_K = W_p,K Δz_K / M_c,K before the conservative
#     prolongation (r_f = W_p,f / W_c,f at a horizontal face for w) —
#     and takes back u_K = Σ_c W_c u_c Δz_c / (W_p,K Δz_K) in the
#     anterpolation and the interface flux correction (the inverse
#     scaling around the feedback);
#   * books the flux registers of both sides as mass fluxes (the
#     Boussinesq side's weighted by ρ₀), so the refluxing conserves the
#     composite Σ W θ V.
#
# A same-core pair has no conversion (`nothing`): its operators are
# untouched, bit for bit.

"""
    NestCoreConversion

The density conversion of a mixed core nest (see `nesting/mixed_cores.jl`):
`ρ0` the Boussinesq side's constant density, the cell and face weights of
each side (`Wp`, `Wpf`, `Wc`, `Wcf`, storage-indexed: `ρ̄` of the
anelastic core, `ρ₀` of the Boussinesq one), the velocity factors `rc`
(per parent storage level) and `rf` (per parent storage face), and the
scratch copies of the parent velocity the scaling saves.
"""
struct NestCoreConversion
    ρ0::Float64
    Wp::Vector{Float64}
    Wpf::Vector{Float64}
    Wc::Vector{Float64}
    Wcf::Vector{Float64}
    rc::Vector{Float64}
    rf::Vector{Float64}
    save::NTuple{3,Array{Float64,3}}
end

"The density weights `(cell, face)` (storage-indexed) of a core in a mixed nest: `ρ̄` of a sound-proof core (its constraint weight), `ρ₀` of a Boussinesq one."
_core_weights(sim::MassWeightedSimulation, ρ0) = map(collect, constraint_weights(sim))
function _core_weights(sim::BoussinesqSimulation, ρ0)
    m = sim.mesh
    return (fill(Float64(ρ0), m.Nz + 2m.Hz), fill(Float64(ρ0), m.Nz + 2m.Hz))   # faces: storage k is the bottom face of cell k
end

# linear interpolation (constant beyond the ends) of y(x) at x0
function _interp1(x::AbstractVector, y::AbstractVector, x0::Real)
    x0 <= x[1] && return y[1] + (y[2] - y[1]) / (x[2] - x[1]) * (x0 - x[1])
    x0 >= x[end] && return y[end] + (y[end] - y[end-1]) / (x[end] - x[end-1]) * (x0 - x[end])
    i = searchsortedlast(x, x0)
    t = (x0 - x[i]) / (x[i+1] - x[i])
    return (1 - t) * y[i] + t * y[i+1]
end

"""
    nest_core_conversion(nest, parent, child; rho0) -> Union{Nothing,NestCoreConversion}

The density conversion of the nest's core pair: `nothing` for a
same-core pair (two Boussinesq simulations, or two sound-proof ones of
one reference profile); for a mixed pair a [`NestCoreConversion`](@ref)
with `rho0` the Boussinesq side's constant density (`p_ref / (R_d θ₀)`,
the density the model's budgets give the Boussinesq core). A mixed pair
without `rho0` (`NaN`) has no conversion: it can then only be nested
one-way, with the volume measure (the transfers before NEST-p2). Only the
anelastic weighting (`W = ρ̄`, a density) can meet a Boussinesq core.
"""
nest_core_conversion(nest::NestedGrid, p, c; rho0 = NaN) =
    _same_core(p, c) || isnan(rho0) ? nothing : _core_conversion(nest, p, c, Float64(rho0))

function _core_conversion(nest::NestedGrid, p, c, ρ0::Float64)
    for s in (p, c)
        s isa MassWeightedSimulation && !(s.weighting isa AnelasticWeighting) &&
            throw(ArgumentError("a mixed core nest pairs a Boussinesq core with the anelastic one (W = ρ̄); " *
                                "the $(nameof(typeof(s.weighting))) weight is not a density"))
    end
    ρ0 > 0 ||
        throw(ArgumentError("a mixed core nest needs the Boussinesq side's density (boussinesq_density > 0, " *
                            "p_ref / (R_d θ₀)), got $ρ0"))
    Wp, Wpf = _core_weights(p, ρ0)
    Wc, Wcf = _core_weights(c, ρ0)
    meas = nest_measure(nest, Wp, Wc)
    pm = nest.parent_mesh; cm = nest.child_mesh
    ph = _zfaces_storage(pm)
    zc_p = [(ph[K] + ph[K+1]) / 2 for K in 1:length(ph)-1]
    chf = _child_zfaces(nest, ph)
    zc_c = [(chf[k] + chf[k+1]) / 2 for k in 1:length(chf)-1]
    # cell factors: exact mass ratios on the covered layers, the density
    # ratio elsewhere (the prolongation stencils beyond the child's top)
    rc = [Wp[K] / _interp1(zc_c, Wc, zc_p[K]) for K in eachindex(zc_p)]
    if meas.kind === :mass
        Pb = pm.Hz + nest.iz
        for K in Pb:Pb+nest.nz-1
            rc[K] = meas.pw[K] / (meas.pc[K+1] - meas.pc[K])
        end
    end
    rf = [Wpf[F] / _interp1(view(chf, eachindex(Wcf)), Wcf, ph[F]) for F in eachindex(Wpf)]
    save = (similar(p.u.data, Float64), similar(p.v.data, Float64), similar(p.w.data, Float64))
    return NestCoreConversion(ρ0, Wp, Wpf, Wc, Wcf, rc, rf, save)
end

"The mass measure of a mixed pair (each side weighted by its own density)."
_conversion_measure(nest::NestedGrid, cv::NestCoreConversion) = nest_measure(nest, cv.Wp, cv.Wc)

# Scale the parent's velocity in place by the conversion factors (u, v by
# the cell factor of their level, w by the face factor), keeping copies of
# the original data (halos included).
function _scale_parent!(p, cv::NestCoreConversion)
    copyto!(cv.save[1], p.u.data); copyto!(cv.save[2], p.v.data); copyto!(cv.save[3], p.w.data)
    _scale_levels!(p.u.data, cv.rc); _scale_levels!(p.v.data, cv.rc); _scale_levels!(p.w.data, cv.rf)
    return p
end
function _scale_levels!(d::AbstractArray{<:Real,3}, r::Vector{Float64})
    @inbounds for k in axes(d, 3)
        rk = r[min(k, length(r))]
        for j in axes(d, 2), i in axes(d, 1)
            d[i, j, k] *= rk
        end
    end
    return d
end

# Put the original parent velocity back, bit for bit.
function _restore_parent!(p, cv::NestCoreConversion)
    copyto!(p.u.data, cv.save[1]); copyto!(p.v.data, cv.save[2]); copyto!(p.w.data, cv.save[3])
    return p
end

# After the feedback changed some scaled faces: each face the feedback set
# takes its value divided by the factor (the mass flux back in the parent's
# own density); a face it left alone (its scaled value unchanged) takes its
# original value back, bit for bit.
function _unscale_parent!(p, cv::NestCoreConversion)
    _unscale_levels!(p.u.data, cv.save[1], cv.rc)
    _unscale_levels!(p.v.data, cv.save[2], cv.rc)
    _unscale_levels!(p.w.data, cv.save[3], cv.rf)
    return p
end
function _unscale_levels!(d::AbstractArray{<:Real,3}, orig::AbstractArray{<:Real,3}, r::Vector{Float64})
    @inbounds for k in axes(d, 3)
        rk = r[min(k, length(r))]
        for j in axes(d, 2), i in axes(d, 1)
            o = orig[i, j, k]
            d[i, j, k] = d[i, j, k] == o * rk ? o : d[i, j, k] / rk
        end
    end
    return d
end

# The flux-register weights of a side: its own constraint weights for a
# same-core pair (`nothing`: the Boussinesq volume flux), its density for a
# mixed pair (the Boussinesq side's ρ₀ too).
_register_weights(sim, ::Nothing) = _register_weights(sim)
_register_weights(sim::BoussinesqSimulation, cv::NestCoreConversion) = _core_weights(sim, cv.ρ0)
_register_weights(sim::MassWeightedSimulation, cv::NestCoreConversion) = constraint_weights(sim)
