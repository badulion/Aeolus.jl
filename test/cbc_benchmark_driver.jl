# Comte-Bellot & Corrsin (1971) decaying grid turbulence: the shared LES
# benchmark harness of Era VI (P54-c, #168).
#
# Included by test/era6_cbc_decay.jl and examples/validation/cbc_decay.jl.
# A closure is benchmarked with one call, `cbc_benchmark(closure; N)`, so
# later closures (LASD, AMD/Vreman, σ, TKE) add one line each to the test.
#
# Set-up (Kang, Chester & Meneveau 2003, §2; Meneveau, Lund & Cabot 1996):
# the flow is observed in a frame moving with the mean speed U0 = 10 m/s
# behind a grid of mesh M = 5.08 cm, so time t ↔ downstream distance
# x = U0 t. A triply periodic box of side 11 M (the reference length of
# Kang et al. is 11M/2π) is initialised with the spectrum measured at
# tU0/M = 42 and run to the stations tU0/M = 98 and 171 at the molecular
# viscosity of air, ν = 1.5e-5 m² s⁻¹. The run is dimensional; this is the
# same as the normalised form of the references (L_ref = 11M/2π,
# u_ref = √(3/2)·22.2 cm/s), which only rescales units.
#
# CBC is a wind-tunnel experiment, not DNS. The comparison is against the
# measured spectra *filtered to the grid cut-off*: the reference TKE is
# Σ_n E_CBC(nΔk) Δk over the shells the grid resolves (n ≤ N/2), the same
# shell sum the spectrum diagnostic forms for the LES.

using Aeolus
using Aeolus.Fields: interior_view
using Printf
import Random

const CBC_M = 0.0508                  # grid mesh [m]
const CBC_U0 = 10.0                   # mean speed [m/s]
const CBC_NU = 1.5e-5                 # air [m²/s]
const CBC_STATIONS = (42, 98, 171)    # tU0/M
const CBC_LBOX = 11 * CBC_M           # periodic box side [m]

"""
    cbc_table() -> (k, E, urms)

Table 3 of Comte-Bellot & Corrsin (1971) in SI units: `k` [rad m⁻¹],
`E[:, s]` [m³ s⁻²] at station `s` = 1, 2, 3 (tU0/M = 42, 98, 171;
`NaN` where the table has no value), and the streamwise r.m.s. velocity
of their table 4 [m s⁻¹].
"""
function cbc_table(path = joinpath(@__DIR__, "data", "cbc1971_table3.txt"))
    rows = [parse.(Float64, split(l)) for l in eachline(path) if !startswith(l, "#") && !isempty(strip(l))]
    A = reduce(vcat, permutedims.(rows))
    k = A[:, 1] .* 100                     # cm⁻¹ → m⁻¹
    E = A[:, 2:4] .* 1e-6                  # cm³ s⁻² → m³ s⁻²
    return k, E, (0.222, 0.128, 0.0895)
end

"""
    cbc_spectrum(s) -> E(k)

The measured spectrum at station `s` as a function, piecewise linear in
log k – log E between the tabulated points and continued with the slope
of the first (last) segment below (above) the table. At 32³ only the
first shell (k = 0.112 cm⁻¹, below the first tabulated 0.20 cm⁻¹ at the
first two stations) is extrapolated.
"""
function cbc_spectrum(s::Integer; table = cbc_table())
    k, E, _ = table
    ok = .!isnan.(E[:, s])
    lk = log.(k[ok]); lE = log.(E[ok, s])
    n = length(lk)
    return function (kk)
        kk <= 0 && return 0.0
        x = log(kk)
        i = x <= lk[1] ? 1 : x >= lk[n] ? n - 1 : searchsortedlast(lk, x)
        i = clamp(i, 1, n - 1)
        exp(lE[i] + (lE[i+1] - lE[i]) / (lk[i+1] - lk[i]) * (x - lk[i]))
    end
end

# One SSP-RK3 step (Shu–Osher), each stage a projected `step!` with the
# closure; the same combination as `step_rk3!`, which on this master does
# not forward the `eddy_viscosity!` keyword. (After the Era VI closure
# contract, #167, `step_rk3!(sim, dt; sgs)` replaces this.)
# A closure object (stateful closures included) goes through `step_rk3!`,
# which commits its memory once per step (the Era VI contract, #167).
_cbc_rk3!(sim, dt, kw::NamedTuple{(:sgs,)}) = step_rk3!(sim, dt; sgs = kw.sgs)
function _cbc_rk3!(sim, dt, kw)
    u0 = copy(sim.u.data); v0 = copy(sim.v.data); w0 = copy(sim.w.data)
    step!(sim, dt; kw...)
    step!(sim, dt; kw...)
    sim.u.data .= 0.75 .* u0 .+ 0.25 .* sim.u.data
    sim.v.data .= 0.75 .* v0 .+ 0.25 .* sim.v.data
    sim.w.data .= 0.75 .* w0 .+ 0.25 .* sim.w.data
    step!(sim, dt; kw...)
    sim.u.data .= u0 ./ 3 .+ (2 / 3) .* sim.u.data
    sim.v.data .= v0 ./ 3 .+ (2 / 3) .* sim.v.data
    sim.w.data .= w0 ./ 3 .+ (2 / 3) .* sim.w.data
    sim.t -= 2dt
    return sim
end

function _cbc_advance!(sim, t_end, cfl, skw)
    n = 0
    while sim.t < t_end - 1e-12
        umax = maximum(f -> maximum(abs, interior_view(f)), (sim.u, sim.v, sim.w))
        _cbc_rk3!(sim, min(cfl * sim.mesh.dx / umax, t_end - sim.t), skw)
        n += 1
    end
    return n
end

# closure → (constructor keywords, step! keywords)
_cbc_closure(::Nothing, mesh) = ((;), (;))
_cbc_closure(C_s::Real, mesh) = ((C_s = C_s,), (;))
_cbc_closure(f::Function, mesh) = ((;), (eddy_viscosity! = f,))
function _cbc_closure(s::Symbol, mesh)
    s === :none && return _cbc_closure(nothing, mesh)
    s === :smagorinsky && return _cbc_closure(0.17, mesh)
    s === :sigma && return _cbc_closure(SigmaModel(1.35), mesh)      # P59-b (#184)
    s === :tke && return ((;), (sgs = TKEClosure(mesh),))       # P56-b (#182), Moeng 1984
    if s === :dynamic
        ws = DynamicSmagorinskyWorkspace(mesh)
        return _cbc_closure((νt, u, v, w) -> compute_dynamic_smagorinsky_νt!(νt, u, v, w; workspace = ws), mesh)
    end
    s === :amd && return _cbc_closure(AMDClosure(), mesh)          # C = 0.3 (P55-d, #176)
    s === :vreman && return _cbc_closure(VremanClosure(0.17), mesh) # c = 2.5·0.17² (P55-d, #176)
    s === :lagrangian && return ((;), (sgs = LagrangianDynamicSmagorinsky(mesh),))
    throw(ArgumentError("unknown closure $(repr(s)): use :none, :smagorinsky, :dynamic, :lagrangian, :tke, " *
                        ":amd, :vreman, :sigma, a C_s, f(νt, u, v, w) or an AbstractSGSClosure"))
end
# a closure object: a fresh copy per realisation (its memory must not leak between seeds)
_cbc_closure(c::AbstractSGSClosure, mesh) = ((;), (sgs = deepcopy(c),))

"""
    cbc_benchmark(closure; N = 32, seeds = 1:3, spinup = 2, t_spin, cfl = 0.5) -> NamedTuple

Run the Comte-Bellot–Corrsin decay at N³ with `closure`:

* `:none` / `nothing` — no SGS model (molecular ν only);
* `:smagorinsky` or a number `C_s` — the built-in static Smagorinsky;
* `:dynamic` — plane-averaged dynamic Smagorinsky (Germano–Lilly);
* `:tke` — the Deardorff (1980) / Moeng (1984) prognostic SGS TKE closure
  (`TKEClosure`, P56-b), ``e`` seeded with its local equilibrium
  (`initialize_tke!`) and carried through the evolve-and-rescale cycles;
* `:amd`, `:vreman` — `AMDClosure()` (C = 0.3), `VremanClosure(0.17)`;
* `:lagrangian` — [`LagrangianDynamicSmagorinsky`](@ref) (Meneveau, Lund &
  Cabot 1996), stepped with `step_rk3!(sim, dt; sgs)` so its memory is
  committed once per step;
* `:sigma` — the σ-model, `SigmaModel(1.35)` (Nicoud et al. 2011; #184);
* an `AbstractSGSClosure` — any closure of the Era VI contract (copied per
  realisation), stepped the same way;
* a function `f(νt, u, v, w)` — any closure accepted by `step!`'s
  `eddy_viscosity!`.

Initial field: [`isotropic_random_field!`](@ref) with the station-1
spectrum, then `spinup` cycles of "evolve for `t_spin` with the closure,
[`rescale_to_spectrum!`](@ref) back to station 1" so the run starts with
the phase correlations of a cascade (a random-phase field transfers no
energy for about one turnover). `t_spin` defaults to one station interval,
(98 − 42)M/U0 = 0.284 s. Time stepping: SSP-RK3, dt = `cfl`·Δ/max|u|,
landing exactly on the stations. Spectra are ensemble averaged over the
random `seeds` (the energy-containing shells hold only a few dozen
modes, so one realisation scatters).

Returns `K_ratio` (LES K(98)/K(42), K(171)/K(42)), `K_ratio_ref` (the
same for CBC filtered to the grid cut-off), `K_err` (relative errors),
`rms_log_err` (RMS of log10(E_LES/E_CBC) over shells 1 … ⌊N/3⌋, i.e.
k ≤ 2k_c/3, at stations 2 and 3), `E_kc_ratio` (E_LES/E_CBC in the
cut-off shell k_c = (N/2)Δk, stations 2 and 3), the spectra `k`, `E_les`,
`E_ref` (3 stations each), the steps per realisation and the wall time.
"""
function cbc_benchmark(closure; N::Integer = 32, cfl::Real = 0.5, seeds = 1:3,
                       spinup::Integer = 2,
                       t_spin::Real = (CBC_STATIONS[2] - CBC_STATIONS[1]) * CBC_M / CBC_U0)
    table = cbc_table()
    Eref = [cbc_spectrum(s; table) for s in 1:3]
    runs = []
    wall = @elapsed for seed in seeds
        push!(runs, _cbc_run(closure, N, cfl, seed, spinup, t_spin, Eref))
    end
    k = runs[1].k
    E_les = [sum(r.E[s] for r in runs) ./ length(runs) for s in 1:3]
    nc = N ÷ 2
    Δk = k[2] - k[1]
    shells = 2:nc+1                                   # n = 1 … N/2
    E_ref = [Eref[s].(k) for s in 1:3]
    K(E) = sum(E[shells]) * Δk
    K_ratio = (K(E_les[2]) / K(E_les[1]), K(E_les[3]) / K(E_les[1]))
    K_ratio_ref = (K(E_ref[2]) / K(E_ref[1]), K(E_ref[3]) / K(E_ref[1]))
    nlog = floor(Int, 2nc / 3)
    rms(s) = sqrt(sum(abs2, log10.(E_les[s][2:nlog+1] ./ E_ref[s][2:nlog+1])) / nlog)
    return (; K_ratio, K_ratio_ref, K_err = K_ratio ./ K_ratio_ref .- 1,
              rms_log_err = (rms(2), rms(3)),
              E_kc_ratio = (E_les[2][nc+1] / E_ref[2][nc+1], E_les[3][nc+1] / E_ref[3][nc+1]),
              k, E_les, E_ref, nsteps = runs[1].nsteps, wall, N, nseeds = length(runs))
end

function _cbc_run(closure, N, cfl, seed, spinup, t_spin, Eref)
    mesh = RegularGrid(; size = (N, N, N), extent = (CBC_LBOX, CBC_LBOX, CBC_LBOX), halo = (3, 3, 3))
    ckw, skw = _cbc_closure(closure, mesh)
    sim = BoussinesqSimulation(mesh; ν = CBC_NU, ckw...)
    isotropic_random_field!(sim, Eref[1]; rng = Random.Xoshiro(seed))
    haskey(skw, :sgs) && skw.sgs isa TKEClosure && initialize_tke!(skw.sgs, sim)
    # evolve-and-rescale: build the phase correlations of a cascade, then
    # put the station-1 spectrum back (Rogallo 1981; Kang et al. 2003)
    for _ in 1:spinup
        _cbc_advance!(sim, sim.t + t_spin, cfl, skw)
        rescale_to_spectrum!(sim.u, sim.v, sim.w, Eref[1])
        for f in (sim.u, sim.v, sim.w)
            fill_halos!(f, sim.bcs)
        end
    end
    sim.t = 0.0
    t_station = [(s - CBC_STATIONS[1]) * CBC_M / CBC_U0 for s in CBC_STATIONS]
    k, E0 = isotropic_energy_spectrum(sim.u, sim.v, sim.w)
    E = [E0]; nsteps = 0
    for ts in t_station[2:3]
        nsteps += _cbc_advance!(sim, ts, cfl, skw)
        push!(E, isotropic_energy_spectrum(sim.u, sim.v, sim.w)[2])
    end
    return (; k, E, nsteps)
end

function cbc_summary(name, r)
    @sprintf("%-12s N=%d×%d  K98/K42 %.3f (CBC %.3f, %+.1f%%)  K171/K42 %.3f (CBC %.3f, %+.1f%%)  rms log10 E %.3f / %.3f  E(kc)/CBC %.2f / %.2f  %d steps %.1f s",
             name, r.N, r.nseeds, r.K_ratio[1], r.K_ratio_ref[1], 100r.K_err[1], r.K_ratio[2], r.K_ratio_ref[2], 100r.K_err[2],
             r.rms_log_err..., r.E_kc_ratio..., r.nsteps, r.wall)
end
