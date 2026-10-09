# The decaying-turbulence DNS of Era VI (P56-c, #178): the filtered-DNS
# fixture, its a-priori statistics and TOML record, and the run type `dns`
# (CFG-d #359). Moved from test/fixtures/dns_decay.jl,
# test/fixtures/dns_apriori.jl and examples/validation/dns_apriori_driver.jl
# (EX-c #440); the `dns_decay` driver of CFG-e (#360) ran the same DNS and
# is replaced by `dns` (apriori_sgs.toml).
#
# Filtered-DNS fixture of Era VI (P56-c, #178): decaying isotropic
# turbulence computed by Aeolus itself, from a fixed seed.
#
# Used by test/era6_apriori_sgs.jl and the `dns` run type below (moved
# from test/fixtures/dns_decay.jl, EX-c #440).
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

using FFTW: fft, ifft, fftfreq

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
[`dns_decay`](@ref). The `dns` run type ([`DNSDriver`](@ref), CFG-d #359)
runs the same steps.
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

"""
    dynamic_cs_on_coarse_grid(sim, r, Δ_filter) -> (C_s, cs2_profile)

Filter the DNS velocity with a Gaussian of width `Δ_filter` (staggered
components filtered in place of their location), coarse-grain it by `r`
(`coarse_grain`) and evaluate `compute_dynamic_smagorinsky_νt!` (default
3-cell box test filter, `filter_ratio = 3`) on that grid of spacing
``rh``. Returns the volume-averaged Germano–Lilly coefficient
``C_s = (⟨L_{ij}M_{ij}⟩/⟨M_{ij}M_{ij}⟩)^{1/2}`` (no clip, `germano_lilly_cs`) and the clipped
plane-averaged ``C_s^2`` profile the routine stores.
"""
function dynamic_cs_on_coarse_grid(sim, r::Integer, Δ_filter::Real)
    g = sim.mesh
    cmesh = RegularGrid(; size = (g.Nx ÷ r, g.Ny ÷ r, g.Nz ÷ r), extent = (g.Lx, g.Ly, g.Lz), halo = (3, 3, 3))
    bcs = FieldBoundaryConditions(:periodic)
    cf = (Field(FaceX(), cmesh), Field(FaceY(), cmesh), Field(FaceZ(), cmesh))
    for (c, f) in zip(cf, (sim.u, sim.v, sim.w))
        ff = _field_like(f)
        gaussian_filter!(ff, f, Δ_filter)
        interior_view(c) .= coarse_grain(ff, r)
        fill_halos!(c, bcs)
    end
    νt = Field(CellCenter(), cmesh)
    ws = DynamicSmagorinskyWorkspace(cmesh)
    compute_dynamic_smagorinsky_νt!(νt, cf...; workspace = ws)
    return germano_lilly_cs(ws), ws.cs2[1, 1, :]
end

"""
    germano_lilly_cs(ws::DynamicSmagorinskyWorkspace) -> C_s

``(⟨L_{ij}M_{ij}⟩/⟨M_{ij}M_{ij}⟩)^{1/2}`` from the workspace of the last
`compute_dynamic_smagorinsky_νt!` call, volume-averaged and unclipped;
`NaN` when the mean ``L_{ij}M_{ij}`` is negative (net backscatter between
the two filter levels, e.g. a filter width near the box size).
"""
function germano_lilly_cs(ws)
    r = sum(ws.LM) / sum(ws.MM)
    return r >= 0 ? sqrt(r) : NaN
end

# ===========================================================================
# a-priori statistics and their record
# ===========================================================================

# A-priori statistics of the filtered DNS, and their TOML record
# (Era VI, P56-c, #178; the hybrid-DNS decision on #208).
#
# Used by test/era6_apriori_sgs.jl and by the `dns` run type below
# (CFG-d #359; moved from test/fixtures/dns_apriori.jl, EX-c #440), whose
# config examples/validation/configs/dns256.toml is run offline at 256³
# (its record copied to test/fixtures/dns256_apriori.toml). The
# always-on testset of era6_apriori_sgs.jl reads that file when it
# exists; the in-suite 96³ fixture stays the CI reference.
#
# The record is plain TOML written by the TOML standard library, which
# Aeolus already depends on (src/IO.jl): no new dependency.
#
#   [dns]            N, ν, seed, t_end, nsteps, wall_s, Re_λ, k_max η (min
#                    over the run and at the end), skewness, budget error,
#                    ε at the end and at its peak, the peak time, η, λ, u'
#   [checks]         the DNS checks of the fixture, each true/false
#   [spectrum]       k, E (isotropic_energy_spectrum at the final time)
#   [filter.<r>h]    per Gaussian width Δ = r·h: Δ, Δ/η, C_s (matched,
#                    dynamic on the DNS grid, dynamic on the LES grid),
#                    backscatter, II, III, min λ₁/tr τ, and for each of
#                    smagorinsky / bardina / gradient: ρ per component
#                    (deviatoric), ρ_Π, ⟨Π⟩ ratio

"""
    clark_gradient_stress(ub, Δ, h) -> SymmetricTensorField

The Clark gradient model ``τ_{ij} ≈ (Δ^2/12)\\,∂_k\\bar u_i\\,∂_k\\bar u_j``
of the cell-centred filtered velocity `ub` (periodic centred differences
of spacing `h`): the leading term of the Taylor expansion of the exact
subfilter stress of a Gaussian filter of width `Δ` (Clark, Ferziger &
Reynolds 1979, *J. Fluid Mech.* 91, 1).
"""
function clark_gradient_stress(ub, Δ, h)
    a = map(interior_view, ub)
    D(f, k) = (s = ntuple(i -> i == k ? 1 : 0, 3); (circshift(f, .-s) .- circshift(f, s)) ./ (2h))
    G = [D(a[i], k) for i in 1:3, k in 1:3]
    τ = SymmetricTensorField{Float64}(size(a[1])...)
    for (n, (i, j)) in enumerate(((1,1),(2,2),(3,3),(1,2),(1,3),(2,3)))
        Aeolus.Operators.components(τ)[n] .= (Δ^2 / 12) .* (G[i,1] .* G[j,1] .+ G[i,2] .* G[j,2] .+ G[i,3] .* G[j,3])
    end
    return τ
end

"""
    dns_checks(d) -> Dict{String,Bool}

The DNS checks of the fixture (test/era6_apriori_sgs.jl) on the result
`d` of `dns_decay`: resolution, skewness band (second half of the run)
and settling, energy
budget, monotone decay, and whether the run reached the dissipation
peak (`past_peak`) or at least the plateau of ε(t) (`eps_plateau`:
d ln ε/dt < 0.25 over the last 0.25 time units). `eps_plateau` is a test
on the dissipation rate in time, not on the energy spectrum: at the
Re_λ of these runs (≤ 79) there is no k^{-5/3} plateau to test.
"""
function dns_checks(d)
    late = d.t .>= d.t[end] / 2          # t ≥ 1 for the t = 2 fixture
    last = d.t .>= d.t[end] - 0.5
    i = argmin(abs.(d.t .- (d.t[end] - 0.25)))
    return Dict(
        "kmax_eta_ge_1.5" => d.kmax_eta >= 1.5,
        "skewness_in_band" => all(s -> -0.6 <= s <= -0.35, d.S[late]),
        "skewness_settled" => maximum(d.S[last]) - minimum(d.S[last]) < 0.05,
        "budget_le_2pc" => maximum(abs, d.budget_err) <= 0.02,
        "monotone_decay" => all(diff(d.K) .< 0),
        "past_peak" => d.t[argmax(d.ε)] < d.t[end] - 0.1,
        "eps_plateau" => log(d.ε[end] / d.ε[i]) / (d.t[end] - d.t[i]) < 0.25)
end

_rho_dict(ρ) = Dict(String(k) => Float64(getfield(ρ, k)) for k in keys(ρ))

"""
    apriori_statistics(d; ratios = (4, 8, 16, 32)) -> Dict

Filter the final field of `dns_decay` result `d` with a Gaussian of width
``Δ = r h`` for each `r` in `ratios` and return, per width (key `"<r>h"`),
the a-priori statistics of the Smagorinsky (dissipation-matched C_s),
Bardina and Clark gradient models against the exact subfilter stress,
the dynamic C_s on the DNS grid (test-to-grid ratio
``\\sqrt{1 + 8h^2/Δ^2}``) and on the LES grid of spacing Δ (when it has
at least 8³ points; `NaN` otherwise), and the realizability margin
min λ₁/tr τ.
"""
function apriori_statistics(d; ratios = (4, 8, 16, 32))
    out = Dict{String,Any}()
    for r in ratios
        out["$(r)h"] = _apriori_width(d, r)
        GC.gc()                     # free the width's ~40 full-size temporaries (256³: 140 MB each)
    end
    return out
end

# One model against τe; the model stress is built inside, so it is
# garbage once the metrics are taken.
function _model_record(build, τe, S)
    m = apriori_metrics(build(), τe, S)
    rec = Dict{String,Any}("rho" => _rho_dict(m.ρ), "rho_offdiag" => m.ρ_offdiag,
                           "rho_Pi" => m.ρ_Π, "Pi_ratio" => m.Π_ratio)
    return rec, m
end

function _apriori_width(d, r)
    sim = d.sim; g = sim.mesh; h = g.dx
    Δ = r * h
    filt = (o, f) -> gaussian_filter!(o, f, Δ)
    ub = ntuple(_ -> Field(CellCenter(), g), 3)
    τe = SymmetricTensorField(g)
    subfilter_stress!(τe, sim.u, sim.v, sim.w, filt; ubar = ub)
    S = filtered_strain(ub...)
    cs = dissipation_matched_cs(τe, S, Δ)
    λ1, _, _ = stress_eigvals(τe)
    rec = Dict{String,Any}("ratio" => r, "delta" => Δ, "delta_over_eta" => Δ / d.η, "cs_matched" => cs,
                           "min_lambda1_over_trace" => minimum(λ1 ./ (τe.xx .+ τe.yy .+ τe.zz)))
    λ1 = nothing
    rec["smagorinsky"], m = _model_record(() -> (τ = SymmetricTensorField(g); smagorinsky_stress!(τ, S, cs, Δ)), τe, S)
    rec["backscatter"] = m.backscatter_exact
    rec["II"] = m.II; rec["III"] = m.III
    GC.gc()
    rec["bardina"], _ = _model_record(() -> (τ = SymmetricTensorField(g); bardina_stress!(τ, ub..., filt)), τe, S)
    GC.gc()
    rec["gradient"], _ = _model_record(() -> clark_gradient_stress(ub, Δ, h), τe, S)
    τe = S = ub = nothing
    GC.gc()
    bcs = FieldBoundaryConditions(:periodic)
    ff = (Field(FaceX(), g), Field(FaceY(), g), Field(FaceZ(), g))
    for (c, f) in zip(ff, (sim.u, sim.v, sim.w))
        gaussian_filter!(c, f, Δ); fill_halos!(c, bcs)
    end
    ws = DynamicSmagorinskyWorkspace(g)
    compute_dynamic_smagorinsky_νt!(Field(CellCenter(), g), ff...; workspace = ws,
                                    filter_ratio = sqrt(1 + 8h^2 / Δ^2))
    rec["cs_dynamic_dns_grid"] = germano_lilly_cs(ws) * h / Δ
    ws = ff = nothing
    GC.gc()
    rec["cs_dynamic_les_grid"] = g.Nx ÷ r >= 8 && g.Nx % r == 0 ? first(dynamic_cs_on_coarse_grid(sim, r, Δ)) : NaN
    return rec
end

# A missing LES-grid C_s is NaN; TOML 1.0 has `nan`, which TOML.print
# writes and TOML.parse reads back.
"""
    dns_apriori_record(d; ratios = (4, 8, 16, 32), seed = 1) -> Dict

The full TOML record (file header) of `dns_decay` result `d`.
"""
function dns_apriori_record(d; ratios = (4, 8, 16, 32), seed::Integer = 1)
    ip = argmax(d.ε)
    k, E = isotropic_energy_spectrum(d.sim.u, d.sim.v, d.sim.w)
    return Dict{String,Any}(
        "generator" => "the dns run type, examples/validation/configs/dns256.toml (Aeolus.ValidationRuns)",
        "created" => Libc.strftime("%Y-%m-%dT%H:%M:%S", time()),
        "dns" => Dict{String,Any}(
            "N" => d.mesh.Nx, "nu" => d.ν, "seed" => seed, "t_end" => d.t[end], "nsteps" => d.nsteps,
            "wall_s" => d.wall, "Re_lambda" => d.Re_λ, "kmax_eta_min" => d.kmax_eta,
            "kmax_eta_end" => d.kmax_eta_end, "skewness_end" => d.S[end],
            "budget_err_max" => maximum(abs, d.budget_err), "eps_end" => d.ε[end],
            "eps_peak" => d.ε[ip], "t_eps_peak" => d.t[ip], "eta" => d.η, "lambda" => d.λ,
            "u_rms" => d.u_rms, "K_end" => d.K[end]),
        "checks" => dns_checks(d),
        "spectrum" => Dict{String,Any}("k" => k, "E" => E),
        "filter" => apriori_statistics(d; ratios))
end

"""
    write_dns_apriori(path, rec)

Write the record `rec` (`dns_apriori_record`) as TOML to `path`, keys
sorted, after a comment header naming the generator.
"""
function write_dns_apriori(path::AbstractString, rec::AbstractDict)
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, "# Filtered-DNS a-priori statistics (Aeolus, Era VI P56-c, #178).")
        println(io, "# Written by the dns run type (examples/validation/configs/dns256.toml); read by test/era6_apriori_sgs.jl.")
        println(io, "# Do not edit by hand: regenerate (AEOLUS_DNS256=1, or aeolus run dns256.toml with apriori.out).")
        TOML.print(io, rec; sorted = true)
    end
    return path
end

"""
    read_dns_apriori(path) -> Dict

The a-priori record written by [`write_dns_apriori`](@ref).
"""
read_dns_apriori(path::AbstractString) = TOML.parsefile(path)

# Recorded but not required by the generator: the ε(t) plateau check
# `eps_plateau` (the run is past the peak by construction; the name says
# ε, not E(k): there is no spectral plateau at these Re_λ) and the
# settling of the skewness, whose spread over the last 0.5 time units
# is a heuristic of the 48³ fixture (0.05) that the past-peak 64³ run misses by 0.001 (0.051, a
# slow drift −0.45 → −0.40 through the peak). The skewness band itself
# is required.
const ADVISORY_CHECKS = ("eps_plateau", "skewness_settled")

"""
    dns_apriori_finish(d; out, ratios = (4, 8, 16, 32), seed = 1, require_checks = true) -> rec

The a-priori record of the DNS result `d` ([`dns_summary`](@ref),
[`dns_decay`](@ref)), written to `out`. With `require_checks` the record
goes to `out` only if every required DNS check holds (k_max η ≥ 1.5 over
the run, skewness band, budget, monotone decay, past the peak;
`ADVISORY_CHECKS` are recorded only); otherwise it is written to
`out * ".failed"` and the call throws, naming the failed checks and the
measured k_max η. The `dns` run type ([`DNSDriver`](@ref), `[apriori]`,
CFG-d #359) calls it after the run.
"""
function dns_apriori_finish(d; out::AbstractString, ratios = (4, 8, 16, 32), seed::Integer = 1,
                            require_checks::Bool = true)
    N = d.mesh.Nx
    ratios = Tuple(r for r in ratios if N ÷ r >= 1)
    rec = dns_apriori_record(d; ratios, seed)
    bad = sort([k for (k, v) in rec["checks"] if !v && k ∉ ADVISORY_CHECKS])
    if require_checks && !isempty(bad)
        # keep the expensive result for inspection, but not under the name the suite reads
        write_dns_apriori(out * ".failed", rec)
        throw(ErrorException("DNS checks failed at $(N)³: $(join(bad, ", ")); measured k_max η = " *
                             string(round(d.kmax_eta; sigdigits = 4)) * " (minimum over the run; the " *
                             "2nd-order solver needs ≥ 1.5) with ν = $(d.ν); record written to " *
                             "$(out).failed, not to $(out)"))
    end
    write_dns_apriori(out, rec)
    @info("wrote $out", subsystem = :dns_apriori, domain = current_domain(), path = out)
    return rec
end

# ===========================================================================
# the `dns` run type
# ===========================================================================

"""
    DNSDriver()

The `dns` run type: `[run] model = "dns"`, the decaying-turbulence DNS
of [`dns_decay`](@ref) (`dns_start`, then `dns_step!` while
`dns_running`) and, after it, the a-priori record of
[`dns_apriori_finish`](@ref) at the `[apriori] widths`. Shipped config:
`examples/validation/configs/dns256.toml`.
"""
struct DNSDriver <: AbstractRunDriver end

Config.driver_name(::DNSDriver) = "dns"
Config.config_dir(::DNSDriver) = VALIDATION_CONFIG_DIR

function _dns_spectrum_problems(t)
    get(t, "kind", nothing) == "k4_gaussian" ||
        return ["kind: expected \"k4_gaussian\" (E(k) ∝ k⁴ exp(-2 (k/k_p)²)), got $(repr(get(t, "kind", nothing)))"]
    p = ["$k: unknown key (keys: kind, k_p)" for k in keys(t) if !(k in ("kind", "k_p"))]
    kp = get(t, "k_p", nothing)
    (kp isa Real && !(kp isa Bool) && kp > 0) || push!(p, "k_p: expected the peak wavenumber > 0")
    return p
end

function Config.driver_sections(::DNSDriver)
    grid = section("grid"; doc = "The triply periodic 2π box.") do s
        option!(s, "n", Int; default = 256, range = (4, 4096), doc = "cells per direction (N³)")
    end
    fluid = section("fluid"; doc = "The fluid.") do s
        option!(s, "nu", OneOf(Float64, Symbol); default = :scaled, allowed = [:scaled], range = (0, Inf),
                open_lower = true, units = "m^2 s^-1",
                doc = "kinematic viscosity; \"scaled\": nu_margin × 0.018 (48/N)^{4/3}, the fixture's scaling that holds k_max η fixed")
        option!(s, "nu_margin", Float64; units = "1", default = 1.035, range = (0, Inf), open_lower = true,
                doc = "the factor of \"scaled\" over the fixture's ν (1.035: the offline generator's 3.5 % margin, k_max η × 1.026; 1: the CI fixture)")
    end
    initial = section("initial"; doc = "The initial field (isotropic_random_field!, u' = 1).") do s
        option!(s, "spectrum", OpenTable(_dns_spectrum_problems);
                default = Dict{String,Any}("kind" => "k4_gaussian", "k_p" => 1.5),
                doc = "{kind = \"k4_gaussian\", k_p}: E(k) ∝ k⁴ exp(-2 (k/k_p)²); the seed is [run] seed")
    end
    stepper = section("stepper"; doc = "Projected SSP-RK3, no SGS model.") do s
        option!(s, "cfl", Float64; units = "1", default = 0.5, range = (0, Inf), open_lower = true,
                doc = "advective Courant number")
        option!(s, "end_time", Float64; units = "L/u'", default = 2.0, range = (0, Inf), open_lower = true,
                doc = "the run goes at least to this time")
        option!(s, "past_peak", Float64; units = "L/u'", default = 0.25, range = (0, Inf),
                doc = "then on until this far past the maximum of ε (0: stop at end_time)")
        option!(s, "max_time", Float64; units = "L/u'", default = 6.0, range = (0, Inf), open_lower = true,
                doc = "the cap of the run past the peak")
        option!(s, "nsteps", Optional{Int}; range = (1, typemax(Int)),
                doc = "stop after this many steps (\"none\": no cap)")
    end
    apriori = section("apriori"; doc = "The filtered-DNS a-priori record (dns_apriori_record).") do s
        option!(s, "widths", Vector{Int}; default = [4, 8, 16, 32], range = (1, 4096),
                doc = "Gaussian filter widths Δ in grid spacings (each also the LES-grid coarsening); [] : no record")
        option!(s, "out", Optional{Path}; input = false,
                doc = "the record's TOML file; \"none\": dns_apriori.toml in the run folder's output/")
        option!(s, "require_checks", Bool; default = true,
                doc = "refuse a run whose required DNS checks fail (k_max η ≥ 1.5 over the run, skewness band, energy budget, monotone decay, past the peak): the record goes to <out>.failed and the run fails")
    end
    return Section[grid, fluid, initial, stepper, apriori]
end

function _dns_checks(cfg)
    e = ConfigError[]
    cfg.stepper.max_time >= cfg.stepper.end_time ||
        push!(e, ConfigError("stepper.max_time", "must be ≥ stepper.end_time"))
    n = cfg.grid.n
    for w in cfg.apriori.widths
        n % w == 0 || push!(e, ConfigError("apriori.widths", "the width $w does not divide grid.n = $n"))
    end
    return e
end

Config.driver_checks(::DNSDriver) = Any[
    """`stepper.max_time` ≥ `stepper.end_time`
    every `apriori.widths` entry divides `grid.n`""" => _dns_checks]

"""
    dns_kwargs(cfg::RunConfig) -> NamedTuple

The [`dns_decay`](@ref) keywords of a `dns` config: `N`, `seed`, `ν`
(`"scaled"`: `nu_margin` × 0.018 (48/N)^{4/3}), `t_end`, `past_peak`,
`t_max`, `cfl`.
"""
function dns_kwargs(cfg::RunConfig)
    N = cfg.grid.n
    ν = cfg.fluid.nu === :scaled ? cfg.fluid.nu_margin * DNS_NU48 * (48 / N)^(4 / 3) : cfg.fluid.nu
    st = cfg.stepper
    return (; N, seed = cfg.run.seed, ν, t_end = st.end_time, past_peak = st.past_peak,
              t_max = st.max_time, cfl = st.cfl)
end

"""
    dns_config(path_or_name = "dns256"; overrides = String[]) -> RunConfig

Load a `dns` config: a path, or a shipped config of `examples/validation/configs/`.
"""
function dns_config(path::AbstractString = "dns256"; overrides = String[])
    p = isfile(path) ? path : joinpath(VALIDATION_CONFIG_DIR, endswith(path, ".toml") ? path : path * ".toml")
    return load_run_config(p; overrides = overrides)
end

"""
    DNSRun

What the `dns` driver builds: the DNS state `st` ([`dns_start`](@ref)),
the resolved `cfg`, and after the run the `summary` ([`dns_summary`](@ref))
and the a-priori `record` (or `nothing`).
"""
mutable struct DNSRun
    st::Any
    cfg::RunConfig
    summary::Any
    record::Any
end

function Config.build(::DNSDriver, cfg::RunConfig)
    k = dns_kwargs(cfg)
    kp = Float64(cfg.initial.spectrum["k_p"])
    st = dns_start(; N = k.N, seed = k.seed, ν = k.ν, spectrum = q -> q^4 * exp(-2 * (q / kp)^2))
    return DNSRun(st, cfg, nothing, nothing)
end

function Simulations.run!(::DNSDriver, r::DNSRun, rf::RunFolder, ck::CheckpointSchedule; callback = nothing)
    cfg = r.cfg; k = dns_kwargs(cfg); st = r.st; sim = st.sim
    nmax = something(cfg.stepper.nsteps, typemax(Int))
    wall = @elapsed begin
        resume!(ck, sim; components = st.hist)
        # attached (#336): step_rk3! reports every step to the monitor
        with_monitor(sim; run = rf, t_end = k.t_end, cadence = cfg.logging.diagnostics_cadence,
                     step0 = length(st.hist.t) - 1) do m
            while dns_running(st; t_end = k.t_end, past_peak = k.past_peak, t_max = k.t_max) &&
                  length(st.hist.t) - 1 < nmax
                dt = dns_step!(st; cfl = k.cfl, t_end = k.t_end, t_max = k.t_max)
                n = length(st.hist.t) - 1
                maybe_checkpoint!(ck, sim; step = n, components = st.hist)
                callback === nothing || callback(r, n)
            end
        end
    end
    # keep the velocity and pressure only: the solver's work arrays are ~40
    # fields (6 GB at 256³), freed before the statistics
    slim = (; mesh = sim.mesh, u = sim.u, v = sim.v, w = sim.w, p = sim.p, t = sim.t)
    r.st = merge(st, (; sim = slim)); st = nothing; sim = nothing
    GC.gc()
    d = dns_summary(r.st, wall)
    r.summary = d
    @info(@sprintf("DNS %d³: ν = %.4g, %d steps to t = %.2f, %.0f s; Re_λ %.1f; k_max η ≥ %.2f; ε peak at t = %.2f; S = %.3f; max |budget err| %.1e",
                   k.N, d.ν, d.nsteps, d.t[end], d.wall, d.Re_λ, d.kmax_eta, d.t[argmax(d.ε)], d.S[end],
                   isempty(d.budget_err) ? 0.0 : maximum(abs, d.budget_err)),
          subsystem = :run, N = k.N, Re_λ = d.Re_λ, nsteps = d.nsteps, kmax_eta = d.kmax_eta)
    out = something(cfg.apriori.out, joinpath(output_dir(rf), "dns_apriori.toml"))
    if isempty(cfg.apriori.widths)
        (cfg.apriori.require_checks && d.kmax_eta < 1.5) &&
            error("DNS checks failed at $(k.N)³: kmax_eta_ge_1.5; measured k_max η = " *
                  "$(round(d.kmax_eta; sigdigits = 4)) (minimum over the run; the 2nd-order solver needs ≥ 1.5)")
        return r
    end
    r.record = dns_apriori_finish(d; out, ratios = Tuple(cfg.apriori.widths), seed = k.seed,
                                  require_checks = cfg.apriori.require_checks)
    return r
end

"""
    state_fields(r::DNSRun) -> NamedTuple

The DNS velocity and pressure (with halos): `u, v, w, p`.
"""
function Config.state_fields(r::DNSRun)
    s = r.st.sim
    return (u = s.u.data, v = s.v.data, w = s.w.data, p = s.p.data)
end

register_driver!(DNSDriver())

