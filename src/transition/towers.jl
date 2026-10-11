# ONE-g (#456): virtual towers and the "hockey stick" of the stable
# boundary layer (Sun et al. 2012), moved from the `sbl` run type (P58-d,
# #194) into the physics module; the `Towers` component of the model
# (Aeolus.Components) is its adapter.

"""
    TowerSampler

Virtual towers: every column of the domain is a tower with instruments at
the cell centres nearest to the requested heights. Over each averaging
`window` (s; Sun et al. 2012 use 5-min segments) the time means and
variances of `u, v, w` are formed per tower and level, and each segment
gives one sample of the mean wind speed ``V = |\\bar{\\mathbf u}|`` and of
the turbulence velocity scale ``V_{TKE} = \\sqrt{e}``
([`turbulence_velocity_scale`](@ref)), with ``e = (σ_u^2 + σ_v^2 +
σ_w^2)/2`` plus the mean SGS energy ``e_{sgs} = (ν_t/(C_k Δ))^2``
(Deardorff 1980, ``ν_t = C_k Δ \\sqrt{e_{sgs}}``, ``C_k = 0.1``). The
samples are binned in `V` per level (`count`, `sumV`, `sumE`, `sumE2`).
Build with [`tower_sampler`](@ref), feed with [`sample_towers!`](@ref)
after every step, read with [`tower_bins`](@ref) and
[`hockey_stick_threshold`](@ref).
"""
mutable struct TowerSampler
    ks::Vector{Int}; z::Vector{Float64}
    window::Float64; τ::Float64
    acc::Array{Float64,4}                 # (Nx, Ny, nlev, 7): u, v, w, uu, vv, ww, e_sgs
    edges::Vector{Float64}
    count::Matrix{Int}; sumV::Matrix{Float64}; sumE::Matrix{Float64}; sumE2::Matrix{Float64}
    nseg::Int
end

"""
    tower_sampler(mesh, heights; window = 300, Vmax = 16, dV = 0.25) -> TowerSampler

Virtual towers in every column of `mesh` at the cell centres nearest to
`heights` (m), with `window`-second segments, binning the wind speed in
`dV` bins up to `Vmax`.
"""
function tower_sampler(m, heights; window::Real = 300.0, Vmax::Real = 16.0, dV::Real = 0.25)
    ks = [clamp(round(Int, h / m.dz + 0.5), 1, m.Nz) for h in heights]
    edges = collect(0.0:dV:Vmax)
    nb = length(edges) - 1; nl = length(ks)
    return TowerSampler(ks, [(k - 0.5) * m.dz for k in ks], Float64(window), 0.0,
                        zeros(m.Nx, m.Ny, nl, 7), edges, zeros(Int, nb, nl), zeros(nb, nl),
                        zeros(nb, nl), zeros(nb, nl), 0)
end

"""
    reset_segment!(T) -> T

Discard the current (incomplete) segment of `T`: the towers record only
in the hours asked for, and a segment starts afresh when they resume.
"""
reset_segment!(T::TowerSampler) = (T.τ = 0.0; fill!(T.acc, 0); T)

"""
    sample_towers!(T, u, v, w, νt, dt) -> T

Add the state at the end of a step of length `dt` (the velocities, at
their faces, and the eddy viscosity `νt` of the last SGS evaluation) to
the current segment of the towers `T`; close the segment (one sample per
tower and level) when it reaches `T.window`.
"""
function sample_towers!(T::TowerSampler, u::Field, v::Field, w::Field, νt::Field, dt::Real)
    m = mesh(u)
    Δ = cbrt(m.dx * m.dy * m.dz); Ck = 0.1
    ud, vd, wd, nd = u.data, v.data, w.data, νt.data
    a = T.acc
    @inbounds for (l, k) in enumerate(T.ks), j in 1:m.Ny, i in 1:m.Nx
        I, J, K = m.Hx + i, m.Hy + j, m.Hz + k
        uc = 0.5 * (ud[I, J, K] + ud[I+1, J, K])
        vc = 0.5 * (vd[I, J, K] + vd[I, J+1, K])
        wc = 0.5 * (wd[I, J, K] + wd[I, J, K+1])
        es = (nd[I, J, K] / (Ck * Δ))^2
        a[i, j, l, 1] += dt * uc; a[i, j, l, 2] += dt * vc; a[i, j, l, 3] += dt * wc
        a[i, j, l, 4] += dt * uc * uc; a[i, j, l, 5] += dt * vc * vc; a[i, j, l, 6] += dt * wc * wc
        a[i, j, l, 7] += dt * es
    end
    T.τ += dt
    if T.τ >= T.window - 1e-9
        b = T.acc ./ T.τ
        nb = length(T.edges) - 1; dV = T.edges[2] - T.edges[1]
        for l in eachindex(T.ks), j in 1:m.Ny, i in 1:m.Nx
            ū, v̄, w̄ = b[i, j, l, 1], b[i, j, l, 2], b[i, j, l, 3]
            e = 0.5 * (max(b[i, j, l, 4] - ū^2, 0) + max(b[i, j, l, 5] - v̄^2, 0) +
                       max(b[i, j, l, 6] - w̄^2, 0)) + b[i, j, l, 7]
            V = hypot(ū, v̄); E = turbulence_velocity_scale(e)
            bin = floor(Int, V / dV) + 1
            1 <= bin <= nb || continue
            T.count[bin, l] += 1; T.sumV[bin, l] += V; T.sumE[bin, l] += E; T.sumE2[bin, l] += E^2
        end
        T.nseg += 1
        reset_segment!(T)
    end
    return T
end

"""
    tower_bins(T, l; min_count = 20) -> (V, V_TKE, σ, n)

The bin means of `V` and `V_TKE` (and the standard deviation and count of
`V_TKE`) at tower level `l`, for the bins with at least `min_count`
samples.
"""
function tower_bins(T::TowerSampler, l::Integer; min_count::Integer = 20)
    sel = findall(>=(min_count), T.count[:, l])
    n = T.count[sel, l]
    V = T.sumV[sel, l] ./ n; E = T.sumE[sel, l] ./ n
    σ = sqrt.(max.(T.sumE2[sel, l] ./ n .- E .^ 2, 0))
    return (V = V, V_TKE = E, σ = σ, n = n)
end

"""
    hockey_stick_threshold(V, E; weights = ones) -> (V_s, slope1, slope2, intercept, sse)

The threshold wind speed of the "hockey stick" of Sun et al. (2012): the
break point `V_s` of the continuous two-segment line ``E = a + s_1 V``
for ``V < V_s``, ``E = a + s_1 V_s + s_2 (V - V_s)`` above, fitted by
weighted least squares to the bin means `(V, E)`. The break point is
searched on a fine grid between the second and the second-to-last bin
(each segment keeps at least two bins); for each candidate the line is
linear in ``(a, s_1, s_2)`` and solved exactly.

Reference: Sun, J., Mahrt, L., Banta, R. M. & Pichugina, Y. L. (2012),
J. Atmos. Sci. 69, 338–351.
"""
function hockey_stick_threshold(V::AbstractVector, E::AbstractVector; weights = ones(length(V)))
    n = length(V); n >= 4 || throw(ArgumentError("hockey_stick_threshold needs at least 4 bins"))
    p = sortperm(V); V = V[p]; E = E[p]; w = weights[p]
    best = (V_s = NaN, slope1 = NaN, slope2 = NaN, intercept = NaN, sse = Inf)
    for Vs in range(V[2], V[end-1]; length = 400)
        X = hcat(ones(n), min.(V, Vs), max.(V .- Vs, 0))
        W = sqrt.(w)
        β = (W .* X) \ (W .* E)
        r = E .- X * β
        sse = sum(w .* r .^ 2)
        sse < best.sse && (best = (V_s = Vs, slope1 = β[2], slope2 = β[3], intercept = β[1], sse = sse))
    end
    return best
end

"""
    tower_thresholds(T; min_count = 20) -> Vector{Float64}

The hockey-stick threshold `V_s` ([`hockey_stick_threshold`](@ref)) at
every tower level of `T`, fitted to the bins of at least `min_count`
samples weighted by their counts; `NaN` at a level with fewer than four
such bins.
"""
function tower_thresholds(T::TowerSampler; min_count::Integer = 20)
    return [begin
                f = tower_bins(T, l; min_count = min_count)
                length(f.V) >= 4 ? hockey_stick_threshold(f.V, f.V_TKE; weights = Float64.(f.n)).V_s : NaN
            end for l in eachindex(T.ks)]
end
