# N-inflow-d (#266) — how far behind a synthetic inflow the turbulence is
# still out of equilibrium: the streamwise statistics of an inflow–outflow
# domain and its development length.
#
# Included inside `module Transition`, after inflow_turbulence.jl.
#
#   FetchStatistics        span- and time-averaged moments per (x, z) column
#   accumulate_fetch!      one sample (time weight Δt)
#   fetch_profiles         U(x,z), resolved TKE(x,z), −u'w'(x,z), variances
#   development_length     the distance from the inflow plane beyond which a
#                          profile stays within a band around its target
#
# No new physics: the generators are N-inflow-a's, their application
# N-inflow-c's; the stress through the solver's own stencil is
# `Operators.total_stress_profile` (with its `irange`).

"""
    FetchStatistics(sim) -> fs

Running span- and time-averaged moments of the resolved velocity of `sim`
(a simulation with `u, v, w` on a `RegularGrid`) per streamwise column
``i`` and level ``k``, for a flow that is homogeneous in ``y`` and in time
but develops in ``x`` (an inflow–outflow domain). Add samples with
[`accumulate_fetch!`](@ref), read them with [`fetch_profiles`](@ref).

The **variances are taken at each staggered component's own points**, as
`Operators.resolved_tke_profile` and [`ParentResolvedStatistics`](@ref)
do: ``u`` on the two x-faces of the column, ``v`` on its y-faces, ``w`` on
the bottom and top face of the level, each about the time-and-span mean
of its own face; the column value is the mean of the two faces'
variances (a wall face, ``w ≡ 0``, contributes zero). Interpolating to
the cell centre first would multiply the variance of a mode by
``\\cos^2(kΔ/2)``. The **shear stress** ``\\overline{u'w'}`` needs both
components at one point: it is formed from the cell-centred
(two-point-mean) ``u`` and ``w``. Fluctuations are deviations from the
time-and-span mean of the same point, so a steady, spanwise-uniform
developing flow has zero TKE.

Applied to a horizontally periodic precursor, the same statistics
averaged over ``x`` are the target the developing flow is compared with
(same operator, same grid), so no definition difference enters the
comparison.
"""
mutable struct FetchStatistics{G}
    mesh::G
    yper::Bool                       # y periodic: face Ny + 1 is face 1
    W::Float64                       # Σ time weights
    nsamples::Int
    Suf::Array{Float64,3}            # (Nx+1) × Ny × Nz: Σ u per x-face point
    Suu::Matrix{Float64}             # (Nx+1) × Nz:      Σ_j u²
    Svf::Array{Float64,3}            # Nx × nyf × Nz:    Σ v per y-face point
    Svv::Matrix{Float64}             # Nx × Nz:          Σ_faces wgt v²
    Swf::Array{Float64,3}            # Nx × Ny × (Nz+1): Σ w per z-face point
    Sww::Matrix{Float64}             # Nx × (Nz+1):      Σ_j w²
    Sucp::Array{Float64,3}           # Nx × Ny × Nz:     Σ centre u, w per point
    Swcp::Array{Float64,3}
    Suwc::Matrix{Float64}            # Nx × Nz:          Σ_j centre u w
end

function FetchStatistics(sim)
    g = mesh(sim.u)
    g isa RegularGrid || throw(ArgumentError("FetchStatistics: the simulation must be on a RegularGrid"))
    Nx, Ny, Nz = g.Nx, g.Ny, g.Nz
    yper = sim.bcs.south isa Periodic
    nyf = yper ? Ny : Ny + 1
    return FetchStatistics(g, yper, 0.0, 0,
                           zeros(Nx + 1, Ny, Nz), zeros(Nx + 1, Nz),
                           zeros(Nx, nyf, Nz), zeros(Nx, Nz),
                           zeros(Nx, Ny, Nz + 1), zeros(Nx, Nz + 1),
                           zeros(Nx, Ny, Nz), zeros(Nx, Ny, Nz), zeros(Nx, Nz))
end

# y-face weights of a column: each cell averages its two faces, so an end
# face of a non-periodic y range counts half
_yface_weight(fs::FetchStatistics, jf) =
    (!fs.yper && (jf == 1 || jf == fs.mesh.Ny + 1)) ? 0.5 : 1.0

Base.show(io::IO, fs::FetchStatistics) =
    print(io, "FetchStatistics(", fs.mesh.Nx, " × ", fs.mesh.Nz, " columns, ", fs.nsamples,
          " samples, weight ", fs.W, ")")

"""
    accumulate_fetch!(fs::FetchStatistics, sim, Δt = 1) -> fs

Add the current velocity of `sim` to `fs` with the time weight `Δt` (the
step just taken, so unequal adaptive steps are weighted correctly). The
halos are filled first (the open-boundary faces ``i = N_x + 1`` and the
wall faces are read).
"""
function accumulate_fetch!(fs::FetchStatistics, sim, Δt::Real = 1.0)
    Δt > 0 || throw(ArgumentError("accumulate_fetch!: Δt must be > 0"))
    g = fs.mesh
    gs = mesh(sim.u)
    (gs.Nx, gs.Ny, gs.Nz) == (g.Nx, g.Ny, g.Nz) ||
        throw(DimensionMismatch("accumulate_fetch!: the simulation is not on the statistics' mesh"))
    fill_halos!(sim.u, sim.bcs); fill_halos!(sim.v, sim.bcs); fill_halos!(sim.w, sim.bcs)
    Nx, Ny, Nz, Hx, Hy, Hz = g.Nx, g.Ny, g.Nz, g.Hx, g.Hy, g.Hz
    u, v, w = sim.u.data, sim.v.data, sim.w.data
    a = Float64(Δt)
    @inbounds for k in 1:Nz
        K = Hz + k
        for j in 1:Ny
            J = Hy + j
            for i in 1:Nx+1
                x = u[Hx+i, J, K]
                fs.Suf[i, j, k] += a * x
                fs.Suu[i, k] += a * x^2
            end
            for i in 1:Nx
                I = Hx + i
                uc = (u[I, J, K] + u[I+1, J, K]) / 2
                wc = (w[I, J, K] + w[I, J, K+1]) / 2
                fs.Sucp[i, j, k] += a * uc
                fs.Swcp[i, j, k] += a * wc
                fs.Suwc[i, k] += a * uc * wc
            end
        end
        for jf in axes(fs.Svf, 2), i in 1:Nx
            x = v[Hx+i, Hy+jf, K]
            fs.Svf[i, jf, k] += a * x
            fs.Svv[i, k] += a * _yface_weight(fs, jf) * x^2
        end
    end
    @inbounds for kf in 1:Nz+1, j in 1:Ny, i in 1:Nx
        x = w[Hx+i, Hy+j, Hz+kf]
        fs.Swf[i, j, kf] += a * x
        fs.Sww[i, kf] += a * x^2
    end
    fs.W += a
    fs.nsamples += 1
    return fs
end

"""
    fetch_profiles(fs::FetchStatistics) -> NamedTuple

The statistics of `fs` per column `i` (cell-centre ``x_i``, from the west
face of the mesh) and level `k` (cell-centre ``z_k``), as `Nx × Nz`
matrices (m s⁻¹, m² s⁻²):

| field | quantity |
| :-- | :-- |
| `x`, `z` | cell-centre coordinates (m, mesh origin) |
| `U` | mean streamwise velocity (cell-centred) |
| `uu`, `vv`, `ww` | variances at the native points, mean of the column's two faces |
| `tke` | resolved TKE ``\\tfrac12(\\overline{u'^2} + \\overline{v'^2} + \\overline{w'^2})`` |
| `uw` | ``-\\overline{u'w'}`` (cell-centred), positive for a downward momentum flux |

Each moment is about the time mean of its own point (face or centre),
then averaged over ``y``; see [`FetchStatistics`](@ref).
"""
function fetch_profiles(fs::FetchStatistics)
    fs.W > 0 || throw(ArgumentError("fetch_profiles: no samples"))
    g = fs.mesh
    Nx, Ny, Nz = g.Nx, g.Ny, g.Nz
    W = fs.W
    # u: variance per x-face, about each point's time mean, averaged over y
    uf = zeros(Nx + 1, Nz)
    @inbounds for k in 1:Nz, i in 1:Nx+1
        s = 0.0
        for j in 1:Ny
            s += (fs.Suf[i, j, k] / W)^2
        end
        uf[i, k] = fs.Suu[i, k] / (W * Ny) - s / Ny
    end
    uu = [(uf[i, k] + uf[i+1, k]) / 2 for i in 1:Nx, k in 1:Nz]
    # v: variance per y-face, weighted mean over the faces
    vv = zeros(Nx, Nz)
    nw = sum(_yface_weight(fs, jf) for jf in axes(fs.Svf, 2))
    @inbounds for k in 1:Nz, i in 1:Nx
        s = 0.0
        for jf in axes(fs.Svf, 2)
            s += _yface_weight(fs, jf) * (fs.Svf[i, jf, k] / W)^2
        end
        vv[i, k] = (fs.Svv[i, k] / W - s) / nw
    end
    # w: per z-face, averaged over the level's bottom and top faces
    wf = zeros(Nx, Nz + 1)
    @inbounds for kf in 1:Nz+1, i in 1:Nx
        s = 0.0
        for j in 1:Ny
            s += (fs.Swf[i, j, kf] / W)^2
        end
        wf[i, kf] = fs.Sww[i, kf] / (W * Ny) - s / Ny
    end
    ww = [(wf[i, k] + wf[i, k+1]) / 2 for i in 1:Nx, k in 1:Nz]
    # centre u, w and −u'w' about each point's mean
    U = zeros(Nx, Nz); uw = zeros(Nx, Nz)
    @inbounds for k in 1:Nz, i in 1:Nx
        su = 0.0; sp = 0.0
        for j in 1:Ny
            mu = fs.Sucp[i, j, k] / W; mw = fs.Swcp[i, j, k] / W
            su += mu; sp += mu * mw
        end
        U[i, k] = su / Ny
        uw[i, k] = -(fs.Suwc[i, k] / (W * Ny) - sp / Ny)
    end
    clamp0(x) = max(x, 0.0)                              # round-off of a vanishing variance
    uu .= clamp0.(uu); vv .= clamp0.(vv); ww .= clamp0.(ww)
    x = [g.x0 + (i - 0.5) * g.dx for i in 1:Nx]
    z = [g.z0 + (k - 0.5) * g.dz for k in 1:Nz]
    return (x = x, z = z, U = U, uu = uu, vv = vv, ww = ww,
            tke = 0.5 .* (uu .+ vv .+ ww), uw = uw)
end

"""
    development_length(profile_x, target; frac = 0.9, x0 = first(x)) -> L

The development (adjustment, recovery) length of a streamwise profile:
the distance from the inflow plane `x0` beyond which the profile stays
within the band ``[\\text{frac}\\cdot T, T/\\text{frac}]`` around its
target ``T > 0``. `profile_x = (x, f)`: positions (increasing) and values
(e.g. the span- and time-averaged resolved TKE at one height,
[`fetch_profiles`](@ref)); `target` is a number (the periodic precursor's
value). `frac = 0.9` is "within 10 %" (the upper bound ``T/0.9`` is
11 %, the symmetric band in ``\\ln f``).

The first sample after which **every** later sample lies in the band is
found; the entry point is then refined by linear interpolation between
that sample and the one before it (the crossing of the band edge it
left). A profile that is in the band from its first sample gives
``x_1 - x_0``; one that ends outside it gives `Inf`. This is the
"recovery distance" of the inflow-generator literature (Xie & Castro
2008; Kim, Castro & Xie 2013; Wu 2017, AIAA review) with an explicit
threshold, measured with the profile's own resolution.

References: Xie, Z.-T. & Castro, I. P. (2008). *Flow Turbul. Combust.*,
81, 449-470. Kim, Y., Castro, I. P. & Xie, Z.-T. (2013). Divergence-free
turbulence inflow conditions for large-eddy simulations with
incompressible flow solvers. *Comput. Fluids*, 84, 56-68. Wu, X. (2017).
Inflow turbulence generation methods. *Annu. Rev. Fluid Mech.*, 49, 23-49.
"""
function development_length(profile_x, target::Real; frac::Real = 0.9, x0 = nothing)
    x, f = profile_x
    length(x) == length(f) || throw(DimensionMismatch("development_length: x and f differ in length"))
    isempty(x) && throw(ArgumentError("development_length: empty profile"))
    issorted(x) || throw(ArgumentError("development_length: x must be increasing"))
    target > 0 || throw(ArgumentError("development_length: target must be > 0"))
    0 < frac < 1 || throw(ArgumentError("development_length: frac must lie in (0, 1)"))
    xs = x0 === nothing ? Float64(first(x)) : Float64(x0)
    lo, hi = frac * target, target / frac
    inband(v) = lo <= v <= hi
    inband(f[end]) || return Inf
    n = length(f)
    m = n
    while m > 1 && inband(f[m-1])
        m -= 1
    end
    m == 1 && return Float64(x[1]) - xs
    # f[m-1] is outside, f[m] inside: the crossing of the edge it left
    a, b = Float64(f[m-1]), Float64(f[m])
    edge = a < lo ? lo : hi
    τ = (edge - a) / (b - a)
    return Float64(x[m-1]) + τ * (Float64(x[m]) - Float64(x[m-1])) - xs
end
