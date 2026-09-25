# Phase 43, issue #7 — mixed-phase Arctic stratus in a 1-D column.
#
# The classic test of an ice scheme in mixed-phase cloud is whether an
# Arctic stratocumulus deck *persists*: observed decks keep a supercooled
# liquid layer for days with ice concentrations of ~0.1–1 L⁻¹, whereas a
# scheme that makes too much ice glaciates the deck through the
# Wegener–Bergeron–Findeisen process within hours (Harrington et al. 1999;
# Morrison et al. 2011, 2012; Ovchinnikov et al. 2014).
#
# This is NOT an LES. It is a single column with prescribed turbulent
# mixing, in the set-up of the ISDAC intercomparison (Ovchinnikov et al.
# 2014; flight 31, 26 April 2008, over sea ice north of Alaska):
#
#   * initial θ_l and q_t profiles and surface pressure of the ISDAC case
#     (initial LWP 14.4 g m⁻² here; ~14 g m⁻² reported for the case by
#     Paukert & Hoose 2014);
#   * no surface fluxes (sea ice), droplet number fixed at 200 cm⁻³
#     (P3's `N_c_prescribed`), as in ISDAC.
#
# NOT from ISDAC: the longwave. ISDAC runs use a full radiation code
# (RRTMG). Here cloud-top cooling is the idealised DYCOMS-form
# F = F0 e^{-κ LWP↓} + F1 e^{-κ LWP↑} with F0 = 72 W m⁻², F1 = 15 W m⁻²,
# κ = 170 m² kg⁻¹ — values of unverified provenance, chosen as a
# plausible Arctic cloud-top cooling, not taken from the ISDAC
# specification.
#
# Simplifications, stated because they matter:
#
#   * turbulence is a prescribed eddy diffusivity K(z) over the ISDAC
#     elevated mixed layer (400–825 m), zero across the inversion. The
#     ISDAC large-scale subsidence (w = -D z, D = 5e-6 s⁻¹, constant above
#     825 m) is applied as upwind advection of every scalar, with the top
#     boundary held at the initial profile. With K = 0 across the
#     inversion, subsidence is what carries free-tropospheric air into the
#     mixed layer: entrainment happens at exactly w_e = D z_inv ≈ 4 mm s⁻¹,
#     the rate that keeps the inversion stationary (the mixed-layer
#     closure of Lilly 1968), rather than at the rate resolved turbulence
#     would produce. That entrainment is also the INP source (Solomon et
#     al. 2015: entrained free-tropospheric INP sustain cloud ice);
#   * ice comes from prognostic INP (IceNucleation: Niemand et al. 2012
#     immersion freezing on a dust mode) consumed on nucleation, NOT from
#     the ISDAC prescription of a fixed ice number; the dust number is
#     chosen to give ice of order 0.1–1 L⁻¹ (ISDAC flight 31 observed
#     ~0.35 L⁻¹; the intercomparison prescribed 1 L⁻¹ with a 4 L⁻¹
#     sensitivity run);
#   * microphysics is the full P3 of this package (process rates,
#     Hallett–Mossop, saturation adjustment, sedimentation of ice with
#     per-moment fall speeds), with consumed INP falling with the ice
#     number and released where ice sublimates.
#
# Moist conserved variables for mixing: the liquid-water static energy
# s_l = c_pm T + g z - L_v (q_c + q_r) - L_s q_i
# (`MoistThermodynamics.liquid_water_static_energy`) and each water
# species; after mixing T is recovered from s_l and the cloud is
# re-adjusted by P3's `p3_condensation`, which conserves θ_li rather than
# s_l (issue #82). The resulting energy drift is not quantified here.

"""
    ArcticStratusColumn(; kwargs...)

Configuration of the 1-D mixed-phase Arctic stratus case
([`run_arctic_stratus`](@ref)).

| field | default | meaning |
| :-- | :-- | :-- |
| `nz`, `dz` | 48, 25 m | levels and layer thickness (domain 0–1200 m) |
| `dt` | 30 s | time step |
| `hours` | 8 | length of the run |
| `p_sfc` | 1020 hPa | surface pressure (ISDAC) |
| `z_ml_bot`, `z_inv` | 400, 825 m | elevated mixed layer (ISDAC) |
| `K_ml` | 30 m² s⁻¹ | eddy diffusivity inside the mixed layer |
| `F0`, `F1`, `κ_lw` | 72, 15 W m⁻², 170 m² kg⁻¹ | idealised longwave parameterisation (not ISDAC, which uses RRTMG; values unverified) |
| `D_sub` | 5e-6 s⁻¹ | large-scale divergence: `w = -D_sub z` below `z_inv`, `-D_sub z_inv` above (ISDAC) |
| `dust_N` | 5 cm⁻³ | number of the dust INP mode (r_g 0.4 μm, σ_g 2) |
| `inp_scale` | 1 | multiplies `dust_N` (100 for the glaciation experiment) |
| `hallett_mossop` | `P3_DEFAULT_HM` | rime splintering (`nothing` = off) |
| `pp` | `P3_DEFAULT_PROCESS` | P3 process parameters (N_c = 200 cm⁻³) |
| `output_every` | 600 s | interval of the diagnostic time series |
"""
Base.@kwdef struct ArcticStratusColumn
    nz::Int = 48
    dz::Float64 = 25.0
    dt::Float64 = 30.0
    hours::Float64 = 8.0
    p_sfc::Float64 = 1.02e5
    z_ml_bot::Float64 = 400.0
    z_inv::Float64 = 825.0
    K_ml::Float64 = 30.0
    F0::Float64 = 72.0
    F1::Float64 = 15.0
    κ_lw::Float64 = 170.0
    D_sub::Float64 = 5e-6
    dust_N::Float64 = 5.0
    inp_scale::Float64 = 1.0
    hallett_mossop::Union{Nothing,P3HallettMossop{Float64}} = P3_DEFAULT_HM
    pp::P3ProcessParameters{Float64} = P3_DEFAULT_PROCESS
    output_every::Float64 = 600.0
end

"""
    isdac_initial_profile(z) -> (θ_l [K], q_t [kg kg⁻¹])

Initial liquid-water potential temperature and total water of the ISDAC
intercomparison case (Ovchinnikov et al. 2014, §2), `z` in m:

    θ_l = 265 + 0.004 (z - 400)          q_t = 1.5 - 0.00075 (z - 400) g/kg   z < 400
    θ_l = 265                            q_t = 1.5 g/kg                        400 ≤ z < 825
    θ_l = 266 + (z - 825)^0.3            q_t = 1.2 g/kg                        825 ≤ z < 2045
    θ_l = 271 + (z - 2000)^0.33          q_t = 0.5 - 0.000075 (z - 2045) g/kg z ≥ 2045
"""
function isdac_initial_profile(z::Real)
    z = float(z)
    if z < 400
        return (265 + 0.004 * (z - 400), 1e-3 * (1.5 - 0.00075 * (z - 400)))
    elseif z < 825
        return (265.0, 1.5e-3)
    elseif z < 2045
        return (266 + (z - 825)^0.3, 1.2e-3)
    end
    return (271 + (z - 2000)^0.33, 1e-3 * (0.5 - 0.000075 * (z - 2045)))
end

# --- small helpers ---------------------------------------------------------

_partition(s::P3ParcelState) = PhasePartition(p3_total_water(s), s.q_c + s.q_r, s.q_i)
_rho(s::P3ParcelState, c) = air_density(s.T, s.p, _partition(s), c)

# Temperature from s_l at fixed water species (s_l is linear in T because
# the Kirchhoff latent heats are).
function _T_from_sl(sl, z, q::PhasePartition, c)
    bv = c.cp_vapor - c.cp_liquid
    bs = c.cp_vapor - c.cp_ice
    av = c.LH_v0 - bv * c.T_triple
    as = c.LH_s0 - bs * c.T_triple
    cpm = heat_capacity_moist(q, c)
    return (sl - c.grav * z + av * q.liq + as * q.ice) / (cpm - bv * q.liq - bs * q.ice)
end

# Rescale the INP of one population by the density ratio f (number and
# mass together, so the modal median radius is unchanged).
function _scale_inp!(inp::INPPopulation, f)
    inp.N_avail .*= f; inp.M_avail .*= f
    inp.N_cons .*= f; inp.M_cons .*= f
    return inp
end

# Large-scale subsidence, ∂ψ/∂t = -w ∂ψ/∂z with w ≤ 0, first-order
# upwind (from above), inflow ψ_top through the model top. Advective,
# not flux form: the column divergence of w is balanced by horizontal
# convergence that brings no new air properties, so this is not
# conservative — the caller books the change as a large-scale source.
function _subside!(ψ::Vector{Float64}, w::Vector{Float64}, ψ_top, dz, dt)
    n = length(ψ)
    old = copy(ψ)
    for k in 1:n
        above = k == n ? ψ_top : old[k + 1]
        ψ[k] = old[k] - w[k] * dt / dz * (above - old[k])
    end
    return ψ
end

# Implicit (backward-Euler) diffusion of ψ with interface diffusivities
# K[k] between cells k-1 and k (K[1] = K[nz+1] = 0: no flux through the
# surface or the top), flux form with density weighting — conservative.
function _diffuse!(ψ::Vector{Float64}, K::Vector{Float64}, ρ::Vector{Float64},
                   ρf::Vector{Float64}, dz, dt)
    n = length(ψ)
    a = zeros(n); b = zeros(n); cc = zeros(n)
    for k in 1:n
        lo = K[k] * ρf[k] * dt / (ρ[k] * dz^2)
        hi = K[k + 1] * ρf[k + 1] * dt / (ρ[k] * dz^2)
        a[k] = -lo; cc[k] = -hi; b[k] = 1 + lo + hi
    end
    # Thomas algorithm
    for k in 2:n
        w = a[k] / b[k - 1]
        b[k] -= w * cc[k - 1]
        ψ[k] -= w * ψ[k - 1]
    end
    ψ[n] /= b[n]
    for k in n-1:-1:1
        ψ[k] = (ψ[k] - cc[k] * ψ[k + 1]) / b[k]
    end
    return ψ
end

"""
    run_arctic_stratus(cfg = ArcticStratusColumn(); record_profiles = false) -> NamedTuple

Integrate the 1-D mixed-phase stratus column described by `cfg`
([`ArcticStratusColumn`](@ref); set-up and simplifications in the source
header of `src/icenuc/stratus_column.jl` and the theory page). Each step
of `dt`:

 1. longwave heating from the current LWP profile, then implicit mixing
    of s_l, the water species, ice moments and INP with the prescribed
    `K(z)`; T recovered from s_l;
 2. P3 source terms per level ([`p3_step_with_rates`](@ref), with
    Hallett–Mossop and the saturation adjustment); INP of sublimated or
    melted ice released ([`p3_inp_release!`](@ref));
 3. ice nucleation per level ([`p3_ice_nucleation`](@ref)) on the
    level's prognostic INP;
 4. sedimentation: ice with [`p3_sediment!`](@ref) at the P3 mass- and
    number-weighted speeds ([`p3_ice_sedimentation_velocities`](@ref)),
    the consumed INP falling with ice number; rain with
    [`p3_sediment_rain!`](@ref).

Returns a NamedTuple with time series (every `output_every` s) `t` (h),
`LWP`, `IWP` (g m⁻²), `Ni_cloud` (mean ice number in the liquid layer,
L⁻¹), `Ni_max` (L⁻¹), `precip_ice` (accumulated surface ice, kg m⁻²),
plus `z`, final `states`, the INP populations and the budget totals
`inp_total0`, `inp_total`, `inp_surface`, `inp_subsidence` (m⁻²; column
available+consumed INP initially and at the end, consumed INP that
reached the surface, and the net INP brought in by subsidence) and
`water0`, `water`, `precip`, `water_subsidence` (kg m⁻²), so that
`inp_total = inp_total0 + inp_subsidence - inp_surface` and
`water = water0 + water_subsidence - precip` close the budgets.
"""
function run_arctic_stratus(cfg::ArcticStratusColumn = ArcticStratusColumn();
                            record_profiles::Bool = false)
    pp = cfg.pp
    c = pp.thermo
    nz, dz, dt = cfg.nz, cfg.dz, cfg.dt
    z = [(k - 0.5) * dz for k in 1:nz]

    # --- initial state: hydrostatic pressure, T from θ_l with the cloud
    # condensed by P3's saturation adjustment ------------------------------
    states = Vector{P3ParcelState{Float64}}(undef, nz)
    p = cfg.p_sfc * 1.0
    pz = zeros(nz)
    for k in 1:nz
        θl, qt = isdac_initial_profile(z[k])
        # half a layer (first cell) or a full layer up from the previous level
        h = k == 1 ? z[1] : dz
        Tguess = θl * (p / c.p_ref)^0.286
        Tv = Tguess * (1 + 0.61 * qt)
        p *= exp(-c.grav * h / (MoistThermodynamics.R_d(c) * Tv))
        pz[k] = p
        T0 = θl * exner_function(p, PhasePartition(qt), c)
        states[k] = p3_condensation(P3ParcelState(T = T0, p = p, q_v = qt), pp)
    end
    ρ = [_rho(s, c) for s in states]          # held fixed for transport
    ρf = zeros(nz + 1)
    for k in 2:nz
        ρf[k] = 0.5 * (ρ[k - 1] + ρ[k])
    end
    # eddy diffusivity at interfaces (face k between cells k-1 and k)
    Kf = zeros(nz + 1)
    for k in 2:nz
        zf = (k - 1) * dz
        (cfg.z_ml_bot <= zf < cfg.z_inv) && (Kf[k] = cfg.K_ml)
    end

    # --- INP: a dust mode in every level --------------------------------
    dust = lognormal_mode(:mineral_dust, cfg.dust_N * cfg.inp_scale, 0.4, 2.0)
    inps = [inp_population([dust]) for _ in 1:nz]
    schemes = default_freezing_schemes(inps[1])
    colINP(f) = sum(f(inps[k]) * dz for k in 1:nz)
    inp_total0 = colINP(i -> i.N_avail[1] + i.N_cons[1])
    water0 = sum(ρ[k] * p3_total_water(states[k]) * dz for k in 1:nz)

    # subsidence (m s⁻¹, negative = downward) and the fixed top inflow
    wsub = [-cfg.D_sub * min(zk, cfg.z_inv) for zk in z]
    top_in = nothing
    ls_water = 0.0; ls_inp = 0.0

    nsteps = round(Int, cfg.hours * 3600 / dt)
    every = max(1, round(Int, cfg.output_every / dt))
    ts = Float64[]; LWP = Float64[]; IWP = Float64[]
    Nic = Float64[]; Nimax = Float64[]; Pice = Float64[]
    profiles = NamedTuple[]
    precip = 0.0; precip_ice = 0.0; inp_surface = 0.0

    function record!(t)
        push!(ts, t / 3600)
        push!(LWP, 1e3 * sum(ρ[k] * (states[k].q_c + states[k].q_r) * dz for k in 1:nz))
        push!(IWP, 1e3 * sum(ρ[k] * states[k].q_i * dz for k in 1:nz))
        cl = [k for k in 1:nz if states[k].q_c > 1e-5]
        push!(Nic, isempty(cl) ? 0.0 :
              1e-3 * sum(ρ[k] * states[k].N_i for k in cl) / length(cl))
        push!(Nimax, 1e-3 * maximum(ρ[k] * states[k].N_i for k in 1:nz))
        push!(Pice, precip_ice)
        record_profiles && push!(profiles, (t = t / 3600,
            q_c = [s.q_c for s in states], q_i = [s.q_i for s in states],
            N_i = [ρ[k] * states[k].N_i for k in 1:nz], T = [s.T for s in states]))
    end
    record!(0.0)

    sl = zeros(nz); fields = [zeros(nz) for _ in 1:12]
    for step in 1:nsteps
        # --- 1. radiation + mixing ---------------------------------------
        # LWP above/below each interface (kg m⁻²)
        lw = [ρ[k] * states[k].q_c * dz for k in 1:nz]
        below = cumsum(lw)
        total = below[end]
        F = zeros(nz + 1)
        for k in 1:nz+1
            Lb = k == 1 ? 0.0 : below[k - 1]
            F[k] = cfg.F0 * exp(-cfg.κ_lw * (total - Lb)) + cfg.F1 * exp(-cfg.κ_lw * Lb)
        end
        for k in 1:nz
            s = states[k]
            q = _partition(s)
            sl[k] = liquid_water_static_energy(s.T, z[k], q, c) -
                    dt * (F[k + 1] - F[k]) / (ρ[k] * dz)
            f = (s.q_v, s.q_c, s.q_r, s.N_r, s.q_i, s.N_i, s.q_rim, s.B_rim,
                 inps[k].N_avail[1] / ρ[k], inps[k].M_avail[1] / ρ[k],
                 inps[k].N_cons[1] / ρ[k], inps[k].M_cons[1] / ρ[k])
            for j in 1:12
                fields[j][k] = f[j]
            end
        end
        # subsidence: upwind advective form, inflow at the top from the
        # initial profile (the free troposphere is a fixed reservoir)
        top_in === nothing && (top_in = (sl[nz], [fields[j][nz] for j in 1:12]))
        w0 = sum(ρ[k] * (fields[1][k] + fields[2][k] + fields[3][k] + fields[5][k]) * dz for k in 1:nz)
        i0 = sum(ρ[k] * (fields[9][k] + fields[11][k]) * dz for k in 1:nz)
        _subside!(sl, wsub, top_in[1], dz, dt)
        for j in 1:12
            _subside!(fields[j], wsub, top_in[2][j], dz, dt)
        end
        ls_water += sum(ρ[k] * (fields[1][k] + fields[2][k] + fields[3][k] + fields[5][k]) * dz for k in 1:nz) - w0
        ls_inp += sum(ρ[k] * (fields[9][k] + fields[11][k]) * dz for k in 1:nz) - i0
        _diffuse!(sl, Kf, ρ, ρf, dz, dt)
        for j in 1:12
            _diffuse!(fields[j], Kf, ρ, ρf, dz, dt)
        end
        for k in 1:nz
            qv, qc, qr, Nr, qi, Ni, qrim, Brim = (fields[j][k] for j in 1:8)
            q = PhasePartition(qv + qc + qr + qi, qc + qr, qi)
            T = _T_from_sl(sl[k], z[k], q, c)
            Nc = qc > 0 ? states[k].N_c : 0.0
            states[k] = P3ParcelState(T = T, p = pz[k], q_v = qv, q_c = qc, N_c = Nc,
                                      q_r = qr, N_r = Nr, q_i = qi, N_i = Ni,
                                      q_rim = qrim, B_rim = Brim)
            i = inps[k]
            i.N_avail[1] = fields[9][k] * ρ[k]; i.M_avail[1] = fields[10][k] * ρ[k]
            i.N_cons[1] = fields[11][k] * ρ[k]; i.M_cons[1] = fields[12][k] * ρ[k]
        end

        # --- 2-3. microphysics and nucleation, level by level -------------
        for k in 1:nz
            s0 = states[k]
            s1, r = p3_step_with_rates(s0, dt, pp; hallett_mossop = cfg.hallett_mossop)
            p3_inp_release!(inps[k], s0, r, dt)
            if s1.q_i == 0 && inps[k].N_cons[1] > 0
                inp_release!(inps[k], 1.0)       # ice removed entirely
            end
            # INP are carried per kg of air (stored as m⁻³ at the fixed
            # transport density ρ[k]); nucleation sees them per m³ at the
            # cell's current density, so that one INP is one P3 crystal.
            f = _rho(s1, c) / ρ[k]
            _scale_inp!(inps[k], f)
            s2, _ = p3_ice_nucleation(s1, inps[k], dt; pp, schemes)
            _scale_inp!(inps[k], 1 / f)
            states[k] = s2
        end

        # --- 4. sedimentation ----------------------------------------------
        qi = [s.q_i for s in states]; qrim = [s.q_rim for s in states]
        Brim = [s.B_rim for s in states]; Ni = [s.N_i for s in states]
        nc = [inps[k].N_cons[1] / ρ[k] for k in 1:nz]
        mc = [inps[k].M_cons[1] / ρ[k] for k in 1:nz]
        Tair = [s.T for s in states]
        vel(f) = begin
            VN, VM = p3_ice_sedimentation_velocities(f[1], f[2], f[3], f[4], ρ;
                                                     T_air = Tair, μ = pp.μ_i, p = pp.ice)
            (VM, VM, VM, VN, VN, VN)
        end
        surf, _ = p3_sediment!((qi, qrim, Brim, Ni, nc, mc), vel, ρ, dz, dt)
        precip_ice += surf[1]; precip += surf[1]; inp_surface += surf[5]
        qr = [s.q_r for s in states]; Nr = [s.N_r for s in states]
        Pr, _, _ = p3_sediment_rain!(qr, Nr, ρ, dz, dt; μ_r = pp.μ_r)
        precip += Pr
        for k in 1:nz
            s = states[k]
            states[k] = P3ParcelState(T = s.T, p = s.p, q_v = s.q_v, q_c = s.q_c,
                                      N_c = s.N_c, q_r = qr[k], N_r = Nr[k],
                                      q_i = qi[k], N_i = Ni[k], q_rim = qrim[k],
                                      B_rim = Brim[k])
            inps[k].N_cons[1] = max(nc[k], 0.0) * ρ[k]
            inps[k].M_cons[1] = max(mc[k], 0.0) * ρ[k]
        end

        step % every == 0 && record!(step * dt)
    end

    water = sum(ρ[k] * p3_total_water(states[k]) * dz for k in 1:nz)
    return (t = ts, LWP = LWP, IWP = IWP, Ni_cloud = Nic, Ni_max = Nimax,
            precip_ice = Pice, z = z, states = states, inps = inps, ρ = ρ,
            profiles = profiles,
            inp_total0 = inp_total0,
            inp_total = colINP(i -> i.N_avail[1] + i.N_cons[1]),
            inp_surface = inp_surface, inp_subsidence = ls_inp,
            water0 = water0, water = water, precip = precip,
            water_subsidence = ls_water)
end

"""
    glaciation_time(run; fraction = 0.1) -> hours (or Inf)

First output time at which the liquid water path of a
[`run_arctic_stratus`](@ref) result falls below `fraction` of its
initial value — the deck has glaciated. `Inf` if it never does.
"""
function glaciation_time(run; fraction::Real = 0.1)
    L0 = run.LWP[1]
    i = findfirst(<(fraction * L0), run.LWP)
    return i === nothing ? Inf : run.t[i]
end
