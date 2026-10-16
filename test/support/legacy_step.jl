# KA-j (#306): the step-core loops as they were before the port (master
# c8399fe = KA-d + KA-g + master), transcribed verbatim: the weighted
# flux-form operators and the anelastic buoyancy of src/Anelastic.jl, the
# field algebra of the step drivers (broadcasts over interior views), and
# the separate post-step checks and CFL maxima of src/TimeIntegration.jl.
# The reference of test/ka_step.jl and of the KA-j timings.
# Nothing here is used by the package.
# ---------------------------------------------------------------------------
module LegacyStep
using Aeolus
using Aeolus.Grids: RegularGrid, AbstractMesh, interior
using Aeolus.Fields: Field, interior_view
using Aeolus.Grids: CellCenter, FaceX, FaceY, FaceZ
using Aeolus.Anelastic: ReferenceState
using Aeolus.Simulations: BoussinesqSimulation, max_divergence, _dzc
using Aeolus.Backends: interior_all_finite, interior_max

# ===================== src/Anelastic.jl ======================================
function weighted_divergence!(out::Field{CellCenter,T},
                              u::Field{FaceX,T}, v::Field{FaceY,T}, w::Field{FaceZ,T},
                              Wc::Vector{T}, Wf::Vector{T}, g::RegularGrid{T};
                              normalize::Bool = false) where {T}
    idx = 1 / g.dx; idy = 1 / g.dy; idz = 1 / g.dz
    ii, jj, kk = interior(g)
    ud = u.data; vd = v.data; wd = w.data; od = out.data
    @inbounds for k in kk
        wc = Wc[k]; wfb = Wf[k]; wft = Wf[k+1]
        scale = normalize ? 1 / wc : one(T)
        for j in jj, i in ii
            d = wc * ((ud[i+1,j,k] - ud[i,j,k]) * idx +
                      (vd[i,j+1,k] - vd[i,j,k]) * idy) +
                (wft * wd[i,j,k+1] - wfb * wd[i,j,k]) * idz
            od[i,j,k] = scale * d
        end
    end
    return out
end

function weighted_momentum_advect!(tu::Field{FaceX,T}, tv::Field{FaceY,T},
                                   tw::Field{FaceZ,T},
                                   u::Field{FaceX,T}, v::Field{FaceY,T},
                                   w::Field{FaceZ,T},
                                   Wc::Vector{T}, Wf::Vector{T},
                                   g::RegularGrid{T}) where {T}
    idx = 1 / g.dx; idy = 1 / g.dy; idz = 1 / g.dz
    ii, jj, kk = interior(g)
    ud = u.data; vd = v.data; wd = w.data

    # --- u-momentum, west face of cell (i,j,k) --------------------------
    @inbounds for k in kk
        iwc = 1 / Wc[k]; wfb = Wf[k]; wft = Wf[k+1]
        for j in jj, i in ii
            uce = 0.5*(ud[i,   j, k] + ud[i+1, j, k])
            ucw = 0.5*(ud[i-1, j, k] + ud[i,   j, k])
            d_uu_dx = (uce*uce - ucw*ucw) * idx

            vun = 0.5*(vd[i-1, j+1, k] + vd[i, j+1, k])
            uvn = 0.5*(ud[i,   j,   k] + ud[i, j+1, k])
            vus = 0.5*(vd[i-1, j,   k] + vd[i, j,   k])
            uvs = 0.5*(ud[i,   j-1, k] + ud[i, j,   k])
            d_vu_dy = (vun*uvn - vus*uvs) * idy

            wut = 0.5*(wd[i-1, j, k+1] + wd[i, j, k+1])
            uwt = 0.5*(ud[i,   j, k  ] + ud[i, j, k+1])
            wub = 0.5*(wd[i-1, j, k  ] + wd[i, j, k  ])
            uwb = 0.5*(ud[i,   j, k-1] + ud[i, j, k  ])
            d_wu_dz = (wft*wut*uwt - wfb*wub*uwb) * idz * iwc

            tu.data[i, j, k] -= d_uu_dx + d_vu_dy + d_wu_dz
        end
    end

    # --- v-momentum, south face of cell (i,j,k) -------------------------
    @inbounds for k in kk
        iwc = 1 / Wc[k]; wfb = Wf[k]; wft = Wf[k+1]
        for j in jj, i in ii
            uve = 0.5*(ud[i+1, j-1, k] + ud[i+1, j, k])
            vue = 0.5*(vd[i,   j,   k] + vd[i+1, j, k])
            uvw = 0.5*(ud[i,   j-1, k] + ud[i,   j, k])
            vuw = 0.5*(vd[i-1, j,   k] + vd[i,   j, k])
            d_uv_dx = (uve*vue - uvw*vuw) * idx

            vcn = 0.5*(vd[i, j,   k] + vd[i, j+1, k])
            vcs = 0.5*(vd[i, j-1, k] + vd[i, j,   k])
            d_vv_dy = (vcn*vcn - vcs*vcs) * idy

            wvt = 0.5*(wd[i, j-1, k+1] + wd[i, j, k+1])
            vwt = 0.5*(vd[i, j,   k  ] + vd[i, j, k+1])
            wvb = 0.5*(wd[i, j-1, k  ] + wd[i, j, k  ])
            vwb = 0.5*(vd[i, j,   k-1] + vd[i, j, k  ])
            d_wv_dz = (wft*wvt*vwt - wfb*wvb*vwb) * idz * iwc

            tv.data[i, j, k] -= d_uv_dx + d_vv_dy + d_wv_dz
        end
    end

    # --- w-momentum, bottom face of cell (i,j,k) ------------------------
    @inbounds for k in kk
        iwf = 1 / Wf[k]; wcb = Wc[k-1]; wct_ = Wc[k]
        for j in jj, i in ii
            uwe = 0.5*(ud[i+1, j, k-1] + ud[i+1, j, k])
            wue = 0.5*(wd[i,   j, k  ] + wd[i+1, j, k])
            uww = 0.5*(ud[i,   j, k-1] + ud[i,   j, k])
            wuw = 0.5*(wd[i-1, j, k  ] + wd[i,   j, k])
            d_uw_dx = (uwe*wue - uww*wuw) * idx

            vwn = 0.5*(vd[i, j+1, k-1] + vd[i, j+1, k])
            wvn = 0.5*(wd[i, j,   k  ] + wd[i, j+1, k])
            vws = 0.5*(vd[i, j,   k-1] + vd[i, j,   k])
            wvs = 0.5*(wd[i, j-1, k  ] + wd[i, j,   k])
            d_vw_dy = (vwn*wvn - vws*wvs) * idy

            wtop = 0.5*(wd[i, j, k  ] + wd[i, j, k+1])
            wbot = 0.5*(wd[i, j, k-1] + wd[i, j, k  ])
            d_ww_dz = (wct_*wtop*wtop - wcb*wbot*wbot) * idz * iwf

            tw.data[i, j, k] -= d_uw_dx + d_vw_dy + d_ww_dz
        end
    end

    return (tu, tv, tw)
end

function weighted_scalar_advect!(tϕ::Field{CellCenter,T},
                                 u::Field{FaceX,T}, v::Field{FaceY,T}, w::Field{FaceZ,T},
                                 ϕ::Field{CellCenter,T},
                                 Wc::Vector{T}, Wf::Vector{T},
                                 g::RegularGrid{T}) where {T}
    idx = 1 / g.dx; idy = 1 / g.dy; idz = 1 / g.dz
    ii, jj, kk = interior(g)
    ud = u.data; vd = v.data; wd = w.data; f = ϕ.data
    @inbounds for k in kk
        iwc = 1 / Wc[k]; wfb = Wf[k]; wft = Wf[k+1]
        for j in jj, i in ii
            uϕe = ud[i+1,j,k] * 0.5 * (f[i,  j,k] + f[i+1,j,k])
            uϕw = ud[i,  j,k] * 0.5 * (f[i-1,j,k] + f[i,  j,k])
            vϕn = vd[i,j+1,k] * 0.5 * (f[i,j,  k] + f[i,j+1,k])
            vϕs = vd[i,j,  k] * 0.5 * (f[i,j-1,k] + f[i,j,  k])
            wϕt = wft * wd[i,j,k+1] * 0.5 * (f[i,j,k  ] + f[i,j,k+1])
            wϕb = wfb * wd[i,j,k  ] * 0.5 * (f[i,j,k-1] + f[i,j,k  ])
            tϕ.data[i,j,k] -= (uϕe - uϕw) * idx + (vϕn - vϕs) * idy +
                              (wϕt - wϕb) * idz * iwc
        end
    end
    return tϕ
end

function add_anelastic_buoyancy!(tw::Field{FaceZ,T}, θ::Field{CellCenter,T},
                                 ref::ReferenceState{T},
                                 g::RegularGrid{T} = ref.mesh) where {T}
    ii, jj, kk = interior(g)
    gacc = ref.g
    θ̄f = ref.θ̄f_pad
    @inbounds for k in kk
        θ̄k = θ̄f[k]
        coef = gacc / θ̄k
        for j in jj, i in ii
            θface = 0.5 * (θ.data[i,j,k-1] + θ.data[i,j,k])
            tw.data[i,j,k] += coef * (θface - θ̄k)
        end
    end
    return tw
end

# ===================== the field algebra (Simulations / TimeIntegration / Anelastic)
"In-place y[interior] += a · x[interior]."
function _axpy_interior!(y::Field, a::Number, x::Field, m::AbstractMesh)
    ii, jj, kk = interior(m)
    @inbounds @views y.data[ii, jj, kk] .+= a .* x.data[ii, jj, kk]
    return y
end

function _combine2!(out::Field, a, A::Field, b, B::Field, m::AbstractMesh)
    ii, jj, kk = interior(m)
    @inbounds @views out.data[ii, jj, kk] .= a .* A.data[ii, jj, kk] .+
                                             b .* B.data[ii, jj, kk]
    return out
end

function _combine3!(out::Field, a, A::Field, b, B::Field, c, C::Field, m::AbstractMesh)
    ii, jj, kk = interior(m)
    @inbounds @views out.data[ii, jj, kk] .= a .* A.data[ii, jj, kk] .+
                                             b .* B.data[ii, jj, kk] .+
                                             c .* C.data[ii, jj, kk]
    return out
end

function _combine4!(out::Field, a, A::Field, b, B::Field, c, C::Field,
                    d, D::Field, m::AbstractMesh)
    ii, jj, kk = interior(m)
    @inbounds @views out.data[ii, jj, kk] .= a .* A.data[ii, jj, kk] .+
                                             b .* B.data[ii, jj, kk] .+
                                             c .* C.data[ii, jj, kk] .+
                                             d .* D.data[ii, jj, kk]
    return out
end

# the RK3 stage combinations of step_rk3! (whole storage)
rk3_combine2!(a, a0) = (a .= 0.75 .* a0 .+ 0.25 .* a; a)
rk3_combine3!(a, a0, inv3, two3) = (a .= inv3 .* a0 .+ two3 .* a; a)

# p = φ/Δt
function pressure!(p, φ, Δt, m)
    ii, jj, kk = interior(m)
    @inbounds @views p.data[ii, jj, kk] .= φ.data[ii, jj, kk] ./ Δt
    return p
end

# ===================== src/TimeIntegration.jl: the post-step checks =========
function _all_finite(sim::BoussinesqSimulation)
    interior_all_finite(sim.u) || return false
    interior_all_finite(sim.v) || return false
    interior_all_finite(sim.w) || return false
    interior_all_finite(sim.θ) || return false
    interior_all_finite(sim.p) || return false
    sim.has_moisture && !interior_all_finite(sim.qv) && return false
    return true
end
_max_speed(sim) = max(maximum(abs, interior_view(sim.u)), maximum(abs, interior_view(sim.v)),
                      maximum(abs, interior_view(sim.w)))
# the reason step_adaptive! rejected a step, before the fused guard
function guard_reason(sim, div_tol, u_cap)
    _all_finite(sim) || return :nonfinite
    d = max_divergence(sim)
    (!isfinite(d) || d > div_tol) && return :divergence
    _max_speed(sim) > u_cap && return :velocity
    return :ok
end

# the CFL maxima: the advective rate loop and one maximum per coefficient
function max_advective_rate(ud, vd, wd, m)
    ii, jj, kk = interior(m)
    idx = 1 / m.dx; idy = 1 / m.dy
    r = -Inf
    for k in kk, j in jj, i in ii
        idz = m isa RegularGrid ? 1 / m.dz : 1 / _dzc(m, k)
        au = max(abs(ud[i, j, k]), abs(ud[i+1, j, k]))
        av = max(abs(vd[i, j, k]), abs(vd[i, j+1, k]))
        aw = max(abs(wd[i, j, k]), abs(wd[i, j, k+1]))
        r  = max(r, au * idx + av * idy + aw * idz)
    end
    return max(0.0, r)
end
cfl_maxima(sim, f) = (max_advective_rate(sim.u.data, sim.v.data, sim.w.data, sim.mesh),
                      maximum(interior_view(f.νh)), maximum(interior_view(f.νv)),
                      maximum(interior_view(f.κh)), maximum(interior_view(f.κv)))
end # module
