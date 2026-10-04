# Surface–atmosphere exchange (Aeolus.SurfaceExchange): Monin–Obukhov
# fluxes between every column's surface and the first AIR cell above it.
#
# Per column the surface is an `AbstractSurface` implementer, built from
# the per-class factories couplings register on the bus (the ground
# coupling registers SoilThermalColumns for streets, the building-adjacent
# cells and the park); classes nobody claims (roofs, water, glass until
# #139/#141/#144) get a `Radiation.SlabLand` with the class's heat
# capacity, albedo and emissivity. Every surface step, each surface is
# advanced with `step_surface!` under a `SurfaceForcing` built from its
# first air cell — above the roof on building columns — and returns
# `SurfaceFluxes` computed by `SurfaceExchange.turbulent_fluxes`. The
# fluxes are then held as tendencies until the next surface step:
# H → θ, E → qv of that air cell every step. The surface stress is the
# ground's wall model (Aeolus.WallModels, #169): a MOSTWallLaw per
# column with the Obukhov length of the last surface step, evaluated
# every step at the instantaneous wind of the first air cell and applied
# by apply_wall_stress! as an implicit drag on its horizontal velocity.
#
# This replaces the slab scheme of #126 (constant C_h = 1e-3 at a fixed
# 3 m/s, fluxes into k = 1 of EVERY column — inside the building on roof
# columns — no momentum drag, and clamps on H, LE and Ts). Nothing is
# clamped here; energy other couplings add to or take from the surfaces
# (bus.sensible: dew heat, window sun, glazing transmission) is collected
# every step and delivered as `Q_extra` at the next surface step, so the
# ledger closes to round-off.

"The ground's wall law: Monin–Obukhov with the default stability functions."
const GroundWallLaw = typeof(MOSTWallLaw(0.05, Inf))

"""
    SurfaceExchangeMOST

Slot `:surface_exchange`, provider `:most` (default).

* **Surfaces.** At `init!`, one `AbstractSurface` per column that has
  air above it, from the factory of its class
  ([`register_surface_factory!`](@ref)) or, for a class without one, a
  `SlabLand` (`C_s`, albedo, emissivity from `bus.params`; β = 1 on
  water, 0 elsewhere; roughness from the options).
* **Air level.** `bus.k_air` is the first non-solid level of each column,
  so roof fluxes go into the air above the roof. The forcing height is
  the centre of that cell above the surface, `z_ref = Δz/2`; the air
  temperature and pressure are ``T = θ\\,\\bar Π_k`` and ``\\bar p_k`` of
  the case's reference state (#131), the conversion every other coupling
  uses.
* **Cadence.** The surfaces are stepped every `period` seconds
  (`cfg.cadence[:surface]` if present, otherwise `cfg.cadence[:soil]`):
  on step 1 in `:surface_flux`, then in `:surface_update` on the step at
  which the air reaches the time the surfaces were advanced to. Each
  surface step covers the next `period` seconds, and the air receives
  its fluxes for exactly that long, so the air gets what the surface
  gave away. With a fixed `dt` dividing `period` that is every step's
  `dt`. With a variable `dt` (the adaptive time stepper, #133) a step
  can run past the end of a surface interval: it delivers the held
  fluxes only up to that end, and the time beyond it is delivered with
  the next surface step's fluxes, which cover it (`dt ≤ period` is
  assumed; the adaptive stepper caps `dt` at the shortest cadence).
* **Air side**, every step in `:surface_flux`: ``Δθ = H\\,τ/(ρ_k c_p Δz)``,
  ``Δq_v = E\\,τ/(ρ_k Δz)`` in the first air cell, `τ` the air time this
  step delivers (``τ = Δt`` with a fixed step), with ``ρ_k`` the
  reference density ``\\bar ρ_k`` of that level (`case.ref`, the density
  the ledger integrates with) and ``c_p`` the moist heat capacity of that air.
* **Momentum**, every step in `:post_predictor` (after the IBM): the
  ground is a wall of `Aeolus.WallModels`. Every surface step sets, per
  column, a `MOSTWallLaw` with the column's `z0m` and the Obukhov
  length `L` of that step (`matching_obukhov_length`: the `L` for which
  the law gives back the step's ``u_*``); `apply_wall_stress!` then
  applies its stress over the `k_air` level mask, implicitly (each
  first-air-cell face multiplied by ``1/(1 + Δt\\,τ_w/(U\\,Δz))``, stable
  for any `Δt`), along the resolved wind; a face between two columns
  gets the mean of the two columns' stresses (a column contributes only
  at its own air level). The law is evaluated **every step** at the
  instantaneous first-air-cell wind (L held from the surface step), and
  `bus.u_star` is that step's ``u_τ`` (before #169 the stress
  ``u_*^2/U`` of the surface step was held between surface steps).
* **Extra energy.** `bus.sensible` (energy other couplings take from the
  surfaces, W m⁻²) is collected every step in `:surface_update` and
  handed to the surfaces as `Q_extra` at their next step.
* **Maximum sustainable heat flux** (P58-e, #193), every surface step:
  per column, `SurfaceExchange.surface_mshf_ratio` of the MOST fluxes of
  the skin at the start of the step (`turbulent_fluxes`), one 1-D
  maximisation at the column's wind, `z_ref`, `z0m` and ``θ_{va}``
  (Van de Wiel et al. 2012). `H_max` (W m⁻²) and the ratio of the downward
  buoyancy flux to the MSHF are output fields; the report prints the
  mean ratio of the stable street, roof and park columns and the share
  of them beyond the MSHF point (``z_{ref}/L`` past the peak of the flux
  curve: the very stable, collapsed branch); the smoke
  check asserts ratio ≤ 1 + 1e-8 everywhere (and < 1.1 for the fluxes
  the surface returned, which a soil skin linearises in ``T_{skin}``).

Options: `U_min` (m s⁻¹, the minimum wind of the surface-layer solve),
`z0m` (per class, m), `z0h_ratio` (``z_{0h} = z_{0m}``·ratio),
`period` (s, overrides the cadence), `wall_formulation` (`:local`,
default; `:schumann` or `:moeng`, the plane-mean forms of
`apply_wall_stress!`, over all ground and roof columns).
"""
mutable struct SurfaceExchangeMOST <: AbstractCoupling
    P::Float64                 # surface step period (s)
    U_min::Float64
    z0m::Dict{UInt8,Float64}
    z0h_ratio::Float64
    stepped::Bool
    t_next::Float64            # the surfaces are advanced up to this simulation time
    owed::Float64              # air time (s) past the end of the previous surface interval, not yet delivered
    τ_interval::Float64        # air time the current surface interval's fluxes have been delivered for (s)
    cp::Matrix{Float64}        # c_p of the air cell, for H → θ
    laws::Matrix{GroundWallLaw} # the ground's wall law per column (z0m, L of the last surface step)
    wm::WallModel{Matrix{GroundWallLaw},Float64} # the WallModels wall of the ground (per-column laws)
    L::Matrix{Float64}         # Obukhov length of the last surface step (m)
    U_wall::Matrix{Float64}    # wind bus.u_star is the wall law at (m s⁻¹): this step's, or the surface step's
    formulation::Symbol        # apply_wall_stress! formulation
    pending::Matrix{Float64}   # extra energy into the surface not yet delivered (J m⁻²)
    Q::Matrix{Float64}         # Q_extra of the last surface step (W m⁻²)
    # this step, for the ledger (J m⁻²; zero on steps without a surface step
    # except dS, which also carries the change of `pending`)
    eRn::Matrix{Float64}
    eH::Matrix{Float64}
    eLE::Matrix{Float64}
    eG::Matrix{Float64}
    edS::Matrix{Float64}
    evap::Float64              # water put into the air this step, kg m⁻² of domain
    # checks
    solid_writes::Int          # air-side increments that landed in a solid cell (must be 0)
    store_err::Float64         # max |Δ(surface store)/P − G| over columns and surface steps (W m⁻²)
    nsurface_steps::Int
    interval_err::Float64      # max |delivered air time − P| over the completed surface intervals (s)
    n_split::Int               # steps that ran past the end of a surface interval (variable dt only)
    # P58-e (#193): maximum sustainable heat flux per column, last surface step
    H_max::Matrix{Float64}     # ρ c_p Q_max (W m⁻²; SurfaceExchange.max_sustainable_heat_flux)
    mshf_ratio::Matrix{Float64} # downward buoyancy flux / Q_max (< 0: upward, unstable)
    mshf_beyond::Matrix{Bool}  # z_ref/L beyond the MSHF point: the very stable (collapsed) branch
    mshf_worst::Float64        # max ratio of the MOST fluxes of the step's start, all columns and steps
    mshf_worst_delivered::Float64 # the same for the fluxes step_surface! returned (linearised soil skin)
    mshf_time::Float64         # wall time spent in the MSHF maximisations (s)
end

coupling_name(::Type{SurfaceExchangeMOST}) = :surface_exchange
slot(::Type{SurfaceExchangeMOST}) = :surface_exchange
provider(::Type{SurfaceExchangeMOST}) = :most
modules_used(::Type{SurfaceExchangeMOST}) = (:SurfaceExchange, :Radiation, :MoistThermodynamics,
                                              :WallModels)
default_options(::Type{SurfaceExchangeMOST}) =
    (U_min = 0.1, z0h_ratio = 0.1, period = nothing, wall_formulation = :local,
     # momentum roughness per class (m): the sub-grid roughness of the
     # surface itself — buildings and park trees are resolved (IBM, drag)
     z0m = Dict(SURFACE_STREET => 0.05, SURFACE_ROOF => 0.05, SURFACE_PARK => 0.1,
                SURFACE_WATER => 1e-4, SURFACE_WALL => 0.05, UInt8(6) => 0.01))
smoke_variants(::Type{SurfaceExchangeMOST}) =
    ((label = "Schumann wall stress", options = Dict(:surface_exchange => (wall_formulation = :schumann,))),
     (label = "Moeng wall stress", options = Dict(:surface_exchange => (wall_formulation = :moeng,))))
stages(::Type{SurfaceExchangeMOST}) = ((:surface_flux, 10, EVERY_STEP),
                                       (:post_predictor, 15, EVERY_STEP),
                                       (:surface_update, 10, EVERY_STEP))

"""
    first_air_level(sm, mesh) -> Matrix{Int}

The first non-solid level of every column of the solid mask `sm` (the
buildings are solid from the ground up); `Nz + 1` for a column that is
solid to the lid.
"""
function first_air_level(sm, mesh)
    Nx, Ny, Nz = mesh.Nx, mesh.Ny, mesh.Nz
    Hx, Hy, Hz = mesh.Hx, mesh.Hy, mesh.Hz
    k_air = fill(Nz + 1, Nx, Ny)
    for j in 1:Ny, i in 1:Nx
        for k in 1:Nz
            if !sm.mask[Hx+i, Hy+j, Hz+k]
                k_air[i, j] = k
                break
            end
        end
    end
    return k_air
end

function build(::Type{SurfaceExchangeMOST}, case, o)
    mesh, bus = case.mesh, case.bus
    Nx, Ny = mesh.Nx, mesh.Ny
    P = o.period === nothing ?
        get(case.cfg.cadence, :surface, cadence_period(case.cfg, :soil)) : Float64(o.period)
    P > 0 || error("surface_exchange: the surface step period must be > 0, got $P")
    o.wall_formulation in (:local, :schumann, :moeng) ||
        error("surface_exchange: wall_formulation must be :local, :schumann or :moeng, got $(o.wall_formulation)")
    bus.k_air = first_air_level(case.geom.sm, mesh)
    fill!(bus.z_ref, mesh.dz / 2)
    z() = zeros(Nx, Ny)
    # neutral until the first surface step
    laws = GroundWallLaw[MOSTWallLaw(0.05, Inf; U_min = o.U_min) for _ in 1:Nx, _ in 1:Ny]
    return SurfaceExchangeMOST(P, o.U_min, Dict{UInt8,Float64}(o.z0m), o.z0h_ratio,
                               false, 0.0, 0.0, 0.0, z(), laws, WallModel(laws), fill(Inf, Nx, Ny),
                               z(), Symbol(o.wall_formulation),
                               z(), z(), z(), z(), z(), z(), z(),
                               0.0, 0, 0.0, 0, 0.0, 0,
                               fill(NaN, Nx, Ny), fill(NaN, Nx, Ny), fill(false, Nx, Ny),
                               -Inf, -Inf, 0.0)
end

"The `SlabLand` for a column of class `cls` nobody registered a factory for."
function fallback_slab(c::SurfaceExchangeMOST, case, I)
    p, cls = case.bus.params, case.geom.sfc[I]
    z0m = get(c.z0m, cls, 0.05)
    return SlabLand(; T = case.bus.Ts[I], C_s = p.C_s[cls], albedo = p.albedo[cls],
                    emissivity = p.emissivity[cls], z0m = z0m, z0h = c.z0h_ratio * z0m,
                    β = cls == SURFACE_WATER ? 1.0 : 0.0)
end

function init!(c::SurfaceExchangeMOST, case)
    bus, Nz = case.bus, case.mesh.Nz
    sfc = case.geom.sfc
    for I in CartesianIndices(sfc)
        if bus.k_air[I] > Nz
            bus.surface[I] = nothing
            bus.dew_mask[I] = false
            continue
        end
        fac = get(bus.factories, sfc[I], nothing)
        s = fac === nothing ? fallback_slab(c, case, I) : fac[2](case, I)
        bus.surface[I] = s
        bus.Ts[I] = surface_temperature(s)
        bus.dew_mask[I] = moisture_availability(s) == 0
    end
    if case.cfg.verbose
        kinds = Dict{String,Int}()
        for s in bus.surface
            s === nothing && continue
            k = string(nameof(typeof(s)))
            kinds[k] = get(kinds, k, 0) + 1
        end
        @printf(case.io, "Surfaces: %s; every %.0f s\n",
                join(("$v $k" for (k, v) in sort!(collect(kinds))), ", "), c.P)
    end
    return nothing
end

"""
    step_all_surfaces!(c, case)

Advance every column's surface by `c.P` under the forcing of its first
air cell and store the fluxes it returns (held until the next call).
"""
function step_all_surfaces!(c::SurfaceExchangeMOST, case)
    bus, mesh = case.bus, case.mesh
    θ = interior_view(case.sim.θ)
    u, v = interior_view(case.sim.u), interior_view(case.sim.v)
    qv = interior_view(tracer(case, :qv))
    Nx, Ny, dz = mesh.Nx, mesh.Ny, mesh.dz
    P = c.P
    # the air has received the fluxes of the interval that ends now for
    # τ_interval seconds; it must be P (not checked before the first step)
    c.stepped && (c.interval_err = max(c.interval_err, abs(c.τ_interval - P)))
    c.τ_interval = 0.0
    @inbounds for j in 1:Ny, i in 1:Nx
        s = bus.surface[i, j]
        s === nothing && continue
        k = bus.k_air[i, j]
        q = qv[i, j, k]
        uc = (u[i, j, k] + u[mod1(i + 1, Nx), j, k]) / 2
        vc = (v[i, j, k] + v[i, mod1(j + 1, Ny), k]) / 2
        U = sqrt(uc^2 + vc^2)
        qa = PhasePartition(q)
        T_air = θ[i, j, k] * ref_exner(case, k)
        p = ref_pressure(case, k)
        f = SurfaceForcing(; sw_down = bus.sw[i, j], lw_down = bus.lw[i, j], T_air = T_air,
                           q_air = q, U = U, p = p, z_ref = bus.z_ref[i, j])
        Q = c.pending[i, j] / P
        c.Q[i, j] = Q
        c.pending[i, j] = 0.0
        E0 = surface_store_energy(s)
        # P58-e: the Monin–Obukhov fluxes of the skin at the start of the
        # step, before step_surface! (which may linearise them in T_skin)
        c.mshf_time += @elapsed (f_most = _most_fluxes(s, f, c.U_min))
        fl = _step_one(s, f, P, Q, c.U_min)
        E1 = surface_store_energy(s)
        E0 === nothing || (c.store_err = max(c.store_err, abs((E1 - E0) / P - fl.G)))
        bus.T_air[i, j], bus.q_air[i, j], bus.U[i, j] = T_air, q, U
        bus.Rn[i, j], bus.H[i, j], bus.LE[i, j] = fl.Rn, fl.H, fl.LE
        bus.G[i, j], bus.E[i, j] = fl.G, fl.E
        ρ = air_density(T_air, p, qa)
        us = sqrt(fl.τ / ρ)
        bus.u_star[i, j] = us
        # the ground's wall law: the Obukhov length for which MOST gives
        # back this step's u* (WallModels; z0m of the surface itself)
        z0m = roughness_lengths(s).z0m
        L = matching_obukhov_length(BusingerDyerBeljaarsHoltslag(), us, U, bus.z_ref[i, j], z0m;
                                    U_min = c.U_min)
        c.L[i, j] = L
        c.laws[i, j] = MOSTWallLaw(z0m, L; U_min = c.U_min)
        # P58-e: how far the column is from its maximum sustainable heat
        # flux (Van de Wiel et al. 2012) — one 1-D maximisation per column.
        # MOST bounds the fluxes it defines (f_most) by Q_max exactly; the
        # fluxes a soil column returns are its first-order extrapolation in
        # T_skin to the step's mean skin temperature, which can overshoot
        # the maximum of the concave H(T_skin) by O(ΔT²) (tracked apart).
        c.mshf_time += @elapsed begin
            r = surface_mshf_ratio(BusingerDyerBeljaarsHoltslag(), f_most, f, z0m; U_min = c.U_min)
            rd = surface_mshf_ratio(BusingerDyerBeljaarsHoltslag(), fl, f, z0m; U_min = c.U_min,
                                    Q_max = r.Q_max)
        end
        c.H_max[i, j] = r.H_max
        c.mshf_ratio[i, j] = r.ratio
        c.mshf_beyond[i, j] = r.ratio > 0 && bus.z_ref[i, j] / L > r.ζ_mshf
        c.mshf_worst = max(c.mshf_worst, r.ratio)
        c.mshf_worst_delivered = max(c.mshf_worst_delivered, rd.ratio)
        c.U_wall[i, j] = U          # u* is the law at this wind until the next step
        c.cp[i, j] = heat_capacity_moist(qa)
        bus.Ts[i, j] = surface_temperature(s)
        # ledger: the whole interval P is booked on this step
        c.eRn[i, j] = fl.Rn * P
        c.eH[i, j] = fl.H * P
        c.eLE[i, j] = fl.LE * P
        if skin_has_storage(s)
            c.edS[i, j] += fl.G * P
        else
            c.eG[i, j] = fl.G * P
        end
        # the extra energy delivered leaves the pending store
        c.edS[i, j] -= Q * P
    end
    c.stepped = true
    c.t_next += P
    c.nsurface_steps += 1
    return nothing
end

# function barrier: one dynamic dispatch per column
_step_one(s, f, P, Q, U_min) = step_surface!(s, f, P; Q_extra = Q, U_min = U_min)
_most_fluxes(s, f, U_min) = turbulent_fluxes(s, f; U_min = U_min)

function run_stage!(c::SurfaceExchangeMOST, case, ::Val{:surface_flux}, dt)
    c.stepped || step_all_surfaces!(c, case)          # step 1: the first surface step
    τ = air_delivery_time!(c, case.sim.t, dt)
    bus, mesh = case.bus, case.mesh
    θ = interior_view(case.sim.θ)
    qv = interior_view(tracer(case, :qv))
    sm = case.geom.sm
    Hx, Hy, Hz, dz = mesh.Hx, mesh.Hy, mesh.Hz, mesh.dz
    evap = 0.0
    nsolid = 0
    @inbounds for I in CartesianIndices(bus.k_air)
        bus.surface[I] === nothing && continue
        i, j = Tuple(I)
        k = bus.k_air[I]
        sm.mask[Hx+i, Hy+j, Hz+k] && (nsolid += 1)
        ρk = ref_density(case, k)
        θ[i, j, k] += bus.H[I] * τ / (ρk * c.cp[I] * dz)
        qv[i, j, k] += bus.E[I] * τ / (ρk * dz)
        evap += bus.E[I] * τ
    end
    c.evap = evap / length(bus.k_air)
    c.solid_writes += nsolid
    return nothing
end

"""
    air_delivery_time!(c, t, dt) -> τ

The air time (s) for which the step from `t` to `t + dt` hands the held
surface fluxes to the air: the part of the step inside the current
surface interval (which ends at `c.t_next`) plus `c.owed`, the part of
the previous step that ran past the end of the previous interval. If
this step runs past `c.t_next`, the excess becomes the new `c.owed`. So
each interval's fluxes reach the air for exactly the interval's length
`P`. With a fixed `dt` dividing `P` it returns `dt` (bit for bit).
"""
function air_delivery_time!(c::SurfaceExchangeMOST, t, dt)
    left = c.t_next - t                       # time to the end of the surface interval
    if dt <= left + 1e-9 * c.P                # (the firing tolerance of :surface_update)
        τ = c.owed + dt
        c.owed = 0.0
    else
        τ = c.owed + left
        c.owed = dt - left
        c.n_split += 1
    end
    c.τ_interval += τ
    return τ
end

function run_stage!(c::SurfaceExchangeMOST, case, ::Val{:post_predictor}, dt)
    # the ground is a WallModels wall at every column's first air level
    # (k_air > Nz: no surface, no wall); u_τ from this step's wind
    apply_wall_stress!(case.sim, c.wm, dt; side = :bottom, mask = case.bus.k_air,
                       formulation = c.formulation, ustar = case.bus.u_star,
                       wind_out = c.U_wall)
    return nothing
end

function run_stage!(c::SurfaceExchangeMOST, case, ::Val{:surface_update}, dt)
    bus = case.bus
    # collect the extra energy other couplings took from the surfaces this step
    @inbounds for I in eachindex(bus.sensible)
        # (a column solid to the lid keeps it pending: it has no surface)
        e = -bus.sensible[I] * dt
        c.pending[I] += e
        c.edS[I] += e
        bus.sensible[I] = 0.0
    end
    t_new = case.sim.t
    t_new >= c.t_next - 1e-9 * c.P && step_all_surfaces!(c, case)
    return nothing
end

# Ledger. Surface: on a surface step the whole interval's Rn, H, LE and G
# (into the soil; a slab's G is its storage and goes to ΔS) are booked;
# ΔS also carries the change of the pending extra energy, so the
# per-class closure is round-off. Water: what this step put into qv.
function budget!(c::SurfaceExchangeMOST, case, l)
    sfc = case.geom.sfc
    for (term, e) in ((:Rn, c.eRn), (:H, c.eH), (:LE, c.eLE), (:G, c.eG), (:dS, c.edS))
        add_surface!(l, term, e, sfc, 1.0)
        fill!(e, 0.0)
    end
    add_water!(l, :evaporation, c.evap)
    return nothing
end

"Columns with a surface of type `T` whose class is `cls` (any class if `nothing`)."
_cols(case, T, cls = nothing) =
    [I for I in CartesianIndices(case.bus.k_air)
     if case.bus.surface[I] isa T && (cls === nothing || case.geom.sfc[I] == cls)]

function report(c::SurfaceExchangeMOST, case)
    bus, sfc = case.bus, case.geom.sfc
    have = [I for I in eachindex(bus.surface) if bus.surface[I] !== nothing]
    roof = [I for I in have if sfc[I] == SURFACE_ROOF]
    ground = [I for I in have if sfc[I] != SURFACE_ROOF]
    m(x, idx) = isempty(idx) ? NaN : mean(x[idx])
    # P58-e: the stable columns' downward buoyancy flux over their MSHF,
    # per class (|H|/H_max; at night every column is stable)
    stable(cls) = [I for I in have if sfc[I] == cls && c.mshf_ratio[I] > 0]
    rs = map((SURFACE_STREET, SURFACE_ROOF, SURFACE_PARK)) do cls
        idx = stable(cls)
        isempty(idx) ? "–" : @sprintf("%.2f (%.0f%% beyond)", mean(c.mshf_ratio[idx]),
                                      100 * count(I -> c.mshf_beyond[I], idx) / length(idx))
    end
    return @sprintf("           surface: H %6.1f LE %6.1f G %6.1f W/m²  u* roof %.2f ground %.2f m/s\n",
                    m(bus.H, have), m(bus.LE, have), m(bus.G, have),
                    m(bus.u_star, roof), m(bus.u_star, ground)) *
           @sprintf("              MSHF: |H|/H_max (stable cols) street %s roof %s park %s; max %.3f (delivered %.3f); %.1f ms/surface step",
                    rs..., c.mshf_worst, c.mshf_worst_delivered, 1e3 * c.mshf_time / max(c.nsurface_steps, 1))
end

function output_fields(c::SurfaceExchangeMOST, case)
    bus = case.bus
    masked(x) = [bus.surface[I] === nothing ? NaN : x[I] for I in eachindex(IndexCartesian(), x)]
    return OutputField[
        OutputField("H_sfc", "W m-2", "surface sensible heat flux (upward), last surface step",
                    :xy, cs -> masked(bus.H)),
        OutputField("LE_sfc", "W m-2", "surface latent heat flux (upward), last surface step",
                    :xy, cs -> masked(bus.LE)),
        OutputField("u_star", "m s-1", "surface friction velocity", :xy, cs -> masked(bus.u_star)),
        OutputField("H_max_sfc", "W m-2", "maximum sustainable heat flux (Van de Wiel et al. 2012), last surface step",
                    :xy, cs -> masked(c.H_max)),
        OutputField("mshf_ratio", "1", "downward surface buoyancy flux / maximum sustainable heat flux (<0: unstable)",
                    :xy, cs -> masked(c.mshf_ratio)),
        OutputField("mshf_beyond", "1", "1 where z_ref/L is beyond the MSHF point (very stable, collapsed branch)",
                    :xy, cs -> masked(Float64.(c.mshf_beyond))),
        OutputField("beta_sfc", "1", "surface moisture availability",
                    :xy, cs -> [s === nothing ? NaN : moisture_availability(s) for s in bus.surface])]
end

function smoke_check(c::SurfaceExchangeMOST, case)
    bus, mesh, sm = case.bus, case.mesh, case.geom.sm
    Hx, Hy, Hz, Nz = mesh.Hx, mesh.Hy, mesh.Hz, mesh.Nz
    level_ok = all(CartesianIndices(bus.k_air)) do I
        i, j = Tuple(I); k = bus.k_air[I]
        all(kk -> sm.mask[Hx+i, Hy+j, Hz+kk], 1:k-1) && (k > Nz || !sm.mask[Hx+i, Hy+j, Hz+k])
    end
    have = [I for I in eachindex(bus.surface) if bus.surface[I] !== nothing]
    roofs = [I for I in have if bus.k_air[I] > 1]
    return ["forcing level is the first air cell: solid below, air at k_air" => level_ok,
            "roof columns exchange above the roof (k_air > 1)" => !isempty(roofs),
            "no surface flux was put into a solid cell" => c.solid_writes == 0,
            "the surfaces were stepped more than once" => c.nsurface_steps >= 2,
            "the air received each completed surface interval's fluxes for exactly P (1e-9 P)" =>
                c.interval_err <= 1e-9 * c.P,
            "per column, the surface stored G·P (|Δstore/P − G| < 1e-6 W/m²)" =>
                c.store_err < 1e-6,
            "u* > 0; above its neutral value κU/ln(z_ref/z0m) where H > 5 and E ≥ 0 (unstable), below it where H < −5 and E ≤ 0 (stable), U the wind of this step's wall law" =>
                all(I -> _ustar_ok(bus, I, c.U_wall[I], c.U_min), have),
            "u* is the ground's MOSTWallLaw at this step's wind (1e-12)" =>
                all(I -> abs(bus.u_star[I] - friction_velocity(c.laws[I], c.U_wall[I], bus.z_ref[I])) <=
                         1e-12 * bus.u_star[I], have),
            # the fluxes that go into the air are those of the surface step
            "Rn − H − LE = G − Q_extra per column (1e-9 W/m²)" =>
                all(I -> abs(bus.Rn[I] - bus.H[I] - bus.LE[I] - bus.G[I] + c.Q[I]) <=
                         1e-9 * (1 + abs(bus.Rn[I])), have),
            "H has the sign of T_skin − θ_air referred to the surface, where they differ by > 1 K and |H| > 1 W/m²" =>
                all(I -> _h_sign_ok(bus, I, case), have),
            "the MSHF was evaluated for every column with a surface (0 < H_max < ∞)" =>
                all(I -> 0 < c.H_max[I] < Inf, have),
            "|H| ≤ H_max (1 + 1e-8) for every column: the downward MOST buoyancy flux ≤ MSHF (Van de Wiel et al. 2012), every surface step" =>
                c.mshf_worst <= 1 + 1e-8,
            "the delivered fluxes (a soil skin's linearisation in T_skin) exceed H_max by < 10 % (measured ≤ 3.1 %)" =>
                c.mshf_worst_delivered < 1.1]
end

function _ustar_ok(bus, I, U, U_min)
    us = bus.u_star[I]
    us > 0 || return false
    neutral = 0.4 * max(U, U_min) / log(bus.z_ref[I] / roughness_lengths(bus.surface[I]).z0m)
    bus.H[I] > 5 && bus.E[I] >= 0 && return us >= neutral * (1 - 1e-9)
    bus.H[I] < -5 && bus.E[I] <= 0 && return us <= neutral * (1 + 1e-9)
    return true
end

function _h_sign_ok(bus, I, case)
    s = bus.surface[I]
    # The fluxes are linearised about the skin temperature at the start of
    # the surface step and evaluated at the step's mean skin temperature;
    # the skin compared here is the one at the END of the step, and a
    # soil column's massless skin can move by a kelvin in one step (dew
    # heat, the soil catching up). Near-zero fluxes of a collapsed stable
    # layer can also come out of the linearisation with either sign. So
    # the test is restricted to |T_s − θ_a| > 1 K and |H| > 1 W/m² (a
    # first version with 0.5 K and no |H| bound failed on 6 of 4094
    # flagship columns, all of those kinds).
    θa = bus.T_air[I] + dry_adiabatic_lapse_rate(PhasePartition(bus.q_air[I])) * bus.z_ref[I]
    d = surface_temperature(s) - θa
    return abs(d) <= 1 || abs(bus.H[I]) <= 1 || sign(bus.H[I]) == sign(d)
end

register!(SurfaceExchangeMOST)
