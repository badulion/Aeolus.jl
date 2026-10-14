# Filtered-DNS fixture of Era VI (P56-c, #178): decaying isotropic
# turbulence computed by Aeolus itself, from a fixed seed.
#
# Included by test/era6_apriori_sgs.jl, the thread-invariance worker, the
# benchmarks and the SGS validation fixture test/fixtures/sgs_validation.jl,
# which starts from this DNS (it was the `sgs_decay` run type until ONE-l, #475).
#
# Set-up. A triply periodic 2π box at N³ (96³ in the test and the
# example; 48³ was the test's size before the review of #208) is filled
# by `isotropic_random_field!` with the spectrum
# E(k) ∝ k⁴ exp(−2(k/k_p)²), k_p = 1.5, scaled to u' = 1 (K = 3/2), and
# integrated with the projected SSP-RK3 of `step_rk3!` (2nd-order
# energy-conserving centred advection, no SGS model, CFL 0.5) to
# t = 2.0. The dissipation peak moves later as Re_λ grows (t ≈ 0.99,
# 1.45, 1.85, 2.41 at 32³, 48³, 64³, 96³): t = 2 is past it at 48³ and on
# its plateau (1.8 % below) at 96³. The derivative skewness reaches ≈ −0.5
# within t ≈ 0.4 and stays in −0.6 … −0.35 afterwards.
# The viscosity ν = 0.018 at 48³ (scaled as N^{-4/3} for other N, which
# keeps k_max η fixed) is the smallest for which k_max η ≥ 1.5 holds over
# the whole run (minimum 1.54 at the dissipation peak at 48³, 1.60 at 96³).
#
# It is a DNS only if it is checked as one, so `dns_decay` records:
#   * the derivative skewness S = ⟨(∂u/∂x)³⟩/⟨(∂u/∂x)²⟩^{3/2}, from the
#     exact staggered differences, averaged over the three components;
#   * the discrete energy budget: dK/dt (one RK3 step) against
#     ε = −ν⟨u_i ∇²u_i⟩, the viscous sink of the discrete Laplacian
#     (= ν⟨∂_j u_i ∂_j u_i⟩ = 2ν⟨S_ij S_ij⟩ for periodic solenoidal flow);
#   * k_max η and Re_λ = u'λ/ν with λ = (15νu'²/ε)^{1/2}.
#
# The filtered fields are formed in the test (Gaussian, Δ = 4h and 8h).
# `coarse_grain` resamples a filtered staggered field onto the grid of
# spacing r·h by Fourier truncation (exact for the retained modes), which
# is how an LES on that grid would see the filtered DNS; the dynamic
# Smagorinsky procedure is applied there.

using Aeolus
using Aeolus.Fields: interior_view
using Aeolus.Grids: interior
using FFTW: fft, ifft, fftfreq
using Printf
import Random

const DNS_KP = 1.5
const DNS_NU48 = 0.018
const DNS_T_END = 2.0

_dns_spectrum(k) = k^4 * exp(-2 * (k / DNS_KP)^2)

# The per-step history runs on the deterministic reductions of KA-b (#296,
# #400): every sum below is a canonical plane sum (`plane_reduce!`,
# `interior_dot`) combined by `ordered_sum`, and the CFL speed a
# `range_max`, so the history is threaded and bit-identical at every thread
# count. Before #400 they were `sum` over broadcast temporaries and
# `maximum(abs, …)`, serial; the sums now add in the canonical order, which
# moves S and ε in their last bits (the max is exact: dt keeps its bits).

# ε of the discrete viscous operator: −ν⟨u_i ∇²u_i⟩ (halos filled first).
function _dns_epsilon(sim, lap)
    for f in (sim.u, sim.v, sim.w)
        fill_halos!(f, sim.bcs)
    end
    I3 = interior(sim.mesh); b = backend(sim.u.data)
    s = 0.0
    for (f, l) in zip((sim.u, sim.v, sim.w), lap)
        laplacian(l, f)
        s += interior_dot(b, f.data, l.data, I3)
    end
    return -sim.ν * s / length(interior_view(sim.u))
end

# (g², g³) of the forward difference g = (a[p + e_d] − a[p])/h at a point
@inline function _dns_dudx_moments(i, j, k, a, di, dj, dk, h)
    @inbounds g = (a[i + di, j + dj, k + dk] - a[i, j, k]) / h
    g2 = g * g
    return (g2, g2 * g)
end

"""
    velocity_derivative_skewness(sim) -> S

``S = ⟨(∂_1u_1)^3⟩/⟨(∂_1u_1)^2⟩^{3/2}`` with the moments averaged over the
three components ``∂_1u_1, ∂_2u_2, ∂_3u_3``, each the exact staggered
difference at the cell centres (periodic: the halos are filled first).
Both moments of a component come from one threaded sweep
(`plane_reduce!` with a tuple body, in the canonical order of KA-b), so
`S` is the same at every thread count.
"""
function velocity_derivative_skewness(sim)
    I3 = interior(sim.mesh); b = backend(sim.u.data)
    n = prod(length, I3)
    p2 = zeros(length(I3[3])); p3 = zeros(length(I3[3]))
    num = 0.0; den = 0.0
    for (f, d, h) in ((sim.u, 1, sim.mesh.dx), (sim.v, 2, sim.mesh.dy), (sim.w, 3, sim.mesh.dz))
        fill_halos!(f, sim.bcs)
        plane_reduce!((p2, p3), b, _dns_dudx_moments, I3, f.data,
                      Int(d == 1), Int(d == 2), Int(d == 3), h)
        num += ordered_sum(p3) / n; den += ordered_sum(p2) / n
    end
    return (num / 3) / (den / 3)^1.5
end

# max |u_i| over the interior, for the CFL step: one threaded sweep of the
# three components; max is exact, so these are the bits of any serial max.
@inline _dns_absmax3(i, j, k, u, v, w) = @inbounds (abs(u[i, j, k]), abs(v[i, j, k]), abs(w[i, j, k]))
_dns_umax(sim) = max(range_max(backend(sim.u.data), _dns_absmax3, interior(sim.mesh),
                               sim.u.data, sim.v.data, sim.w.data)...)

"""
    dns_start(; N = 48, seed = 1, ν = 0.018 (48/N)^{4/3}, spectrum = E(k) ∝ k⁴ e^{-2(k/1.5)²})
        -> NamedTuple

The initial state of the decaying-turbulence DNS (see the file header):
the simulation `sim` on the triply periodic 2π box at `N`³ filled by
`isotropic_random_field!` with `spectrum` from `Xoshiro(seed)` and scaled
to u' = 1, and its history `hist = (t, K, ε, S, budget_err)` (one entry per
step, the initial one included). Advance it with [`dns_step!`](@ref)
while [`dns_running`](@ref); [`dns_summary`](@ref) is the result of
[`dns_decay`](@ref).
"""
function dns_start(; N::Integer = 48, seed::Integer = 1, ν::Real = DNS_NU48 * (48 / N)^(4 / 3),
                   spectrum = _dns_spectrum)
    L = 2π
    mesh = RegularGrid(; size = (N, N, N), extent = (L, L, L), halo = (3, 3, 3))
    sim = BoussinesqSimulation(mesh; ν = ν)
    lap = (Field(FaceX(), mesh), Field(FaceY(), mesh), Field(FaceZ(), mesh))
    isotropic_random_field!(sim, spectrum; rng = Random.Xoshiro(seed))
    s = sqrt(1.5 / (kinetic_energy(sim) / L^3))           # K = 3/2: u' = 1
    for f in (sim.u, sim.v, sim.w)
        f.data .*= s
        fill_halos!(f, sim.bcs)
    end
    t = [0.0]; K = [kinetic_energy(sim) / L^3]; ε = [_dns_epsilon(sim, lap)]
    S = [velocity_derivative_skewness(sim)]
    hist = (t = t, K = K, ε = ε, S = S, budget_err = Float64[])
    return (; sim, mesh, ν = Float64(ν), lap, hist)
end

"""
    dns_running(st; t_end = 2.0, past_peak = 0, t_max = 3t_end) -> Bool

Whether the DNS state `st` ([`dns_start`](@ref)) has not reached `t_end`
or, with `past_peak > 0`, is not yet `past_peak` past the maximum of ε
(capped at `t_max`).
"""
function dns_running(st; t_end::Real = DNS_T_END, past_peak::Real = 0, t_max::Real = 3t_end)
    sim = st.sim; t = st.hist.t; ε = st.hist.ε
    return sim.t < t_end - 1e-12 ||
           (past_peak > 0 && sim.t < min(t[argmax(ε)] + past_peak, t_max) - 1e-12)
end

"""
    dns_step!(st; cfl = 0.5, t_end = 2.0, t_max = 3t_end) -> dt

One projected SSP-RK3 step (`step_rk3!`, no SGS model) of the DNS state
`st` at the advective step `cfl Δx/max|u|` (clipped to `t_end`, then to
`t_max`), with the history entries of the new time.
"""
function dns_step!(st; cfl::Real = 0.5, t_end::Real = DNS_T_END, t_max::Real = 3t_end)
    sim = st.sim; mesh = st.mesh; (; t, K, ε, S, budget_err) = st.hist
    L = 2π
    umax = _dns_umax(sim)
    dt = min(cfl * mesh.dx / umax, (sim.t < t_end - 1e-12 ? t_end : t_max) - sim.t)
    step_rk3!(sim, dt)
    push!(t, sim.t); push!(K, kinetic_energy(sim) / L^3); push!(ε, _dns_epsilon(sim, st.lap))
    push!(S, velocity_derivative_skewness(sim))
    push!(budget_err, -(K[end] - K[end-1]) / dt / (0.5 * (ε[end] + ε[end-1])) - 1)
    return dt
end

"""
    dns_summary(st, wall) -> NamedTuple

What [`dns_decay`](@ref) returns for the DNS state `st` (`wall` seconds).
"""
function dns_summary(st, wall)
    (; sim, mesh, ν) = st; (; t, K, ε, S, budget_err) = st.hist
    kmax = π / mesh.dx
    η = (ν^3 ./ ε) .^ 0.25
    u_rms = sqrt(2K[end] / 3)
    λ = sqrt(15ν * u_rms^2 / ε[end])
    return (; sim, mesh, ν, t, K, ε, S, budget_err, kmax_eta = minimum(kmax .* η),
              kmax_eta_end = kmax * η[end], Re_λ = u_rms * λ / ν, u_rms, λ, η = η[end],
              nsteps = length(t) - 1, wall)
end

"""
    dns_decay(; N = 48, seed = 1, t_end = 2.0, cfl = 0.5, ν = 0.018 (48/N)^{4/3},
              past_peak = 0, t_max = 3t_end, checkpoint = nothing) -> NamedTuple

Run the decaying-turbulence DNS fixture (see the file header): [`dns_start`](@ref),
then [`dns_step!`](@ref) while [`dns_running`](@ref). Returns
the simulation `sim` at the final time, the history `t, K, ε, S` (one
entry per RK3 step), `budget_err` (per step, dK/dt / (−ε̄) − 1 with ε̄ the
mean of the start and end values), `kmax_eta` (its minimum over the
run), and at the final time `Re_λ`, `u_rms`, `λ`, `η`, plus `nsteps` and
`wall` (s).

With `past_peak > 0` the run continues beyond `t_end` until it is
`past_peak` time units past the maximum of ε (capped at `t_max`); the
peak moves later as Re_λ grows (file header), so a fixed `t_end` is
before it at large N. The default (`past_peak = 0`) stops at `t_end`.

`checkpoint` (an `IO.CheckpointSchedule`, #245) resumes the run from the
newest checkpoint of its folder (the fields and the history above) and
writes periodic checkpoints; the resumed run is the uninterrupted one bit
for bit (`wall` then counts this process only).
"""
function dns_decay(; N::Integer = 48, seed::Integer = 1, t_end::Real = DNS_T_END, cfl::Real = 0.5,
                   ν::Real = DNS_NU48 * (48 / N)^(4 / 3), past_peak::Real = 0, t_max::Real = 3t_end,
                   checkpoint = nothing)
    st = nothing
    wall = @elapsed begin
        st = dns_start(; N, seed, ν)
        checkpoint === nothing || resume!(checkpoint, st.sim; components = st.hist)
        while dns_running(st; t_end, past_peak, t_max)
            dns_step!(st; cfl, t_end, t_max)
            checkpoint === nothing || maybe_checkpoint!(checkpoint, st.sim; step = length(st.hist.t) - 1,
                                                        components = st.hist)
        end
    end
    return dns_summary(st, wall)
end

"""
    coarse_grain(f::Field, r) -> Array

Resample the (triply periodic, already filtered) staggered field `f` onto
the grid `r` times coarser at the same staggered location, by keeping the
Fourier modes ``|m_d| < N/(2r)`` and shifting their phase to the coarse
sample positions (a face direction keeps its origin; a centred direction
moves from ``h/2`` to ``rh/2``). Exact for a field band-limited to the
coarse grid; otherwise it adds a sharp cut-off at the coarse Nyquist
wavenumber.
"""
function coarse_grain(f::Field{Loc}, r::Integer) where {Loc}
    g = f.mesh
    a = Array(interior_view(f))
    Nf = size(a); Nc = Nf .÷ r
    all(Nc .* r .== Nf) || throw(ArgumentError("coarse_grain: $Nf not divisible by $r"))
    face = Loc === FaceX ? 1 : Loc === FaceY ? 2 : Loc === FaceZ ? 3 : 0
    h = (g.dx, g.dy, g.dz); Lb = (g.Lx, g.Ly, g.Lz)
    δ = ntuple(d -> d == face ? 0.0 : (r - 1) * h[d] / 2, 3)
    F = fft(a)
    G = zeros(ComplexF64, Nc)
    mc = ntuple(d -> fftfreq(Nc[d], Nc[d]), 3)
    for c in 1:Nc[3], b in 1:Nc[2], a_ in 1:Nc[1]
        m = (mc[1][a_], mc[2][b], mc[3][c])
        any(d -> 2abs(m[d]) >= Nc[d], 1:3) && continue      # drop the coarse Nyquist planes
        fi = ntuple(d -> mod(Int(m[d]), Nf[d]) + 1, 3)
        ph = sum(2π * m[d] / Lb[d] * δ[d] for d in 1:3)
        G[a_, b, c] = F[fi...] * cis(ph)
    end
    return real.(ifft(G)) .* (prod(Nc) / prod(Nf))
end

_field_like(f::Field{L}) where {L} = Field(L(), f.mesh)

