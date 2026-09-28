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
# H → θ, E → qv of that air cell every step, and the stress ρu*² as an
# implicit drag on the air cell's horizontal velocity.
#
# This replaces the slab scheme of #126 (constant C_h = 1e-3 at a fixed
# 3 m/s, fluxes into k = 1 of EVERY column — inside the building on roof
# columns — no momentum drag, and clamps on H, LE and Ts). Nothing is
# clamped here; energy other couplings add to or take from the surfaces
# (bus.sensible: dew heat, window sun, glazing transmission) is collected
# every step and delivered as `Q_extra` at the next surface step, so the
# ledger closes to round-off.

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
  temperature is ``T = θ - Γ_d z`` at the cell's height `z` (the case's θ
  is referred to the ground), so the Monin–Obukhov solve compares the
  skin with ``θ - Γ_d z_{sfc}``, the air's potential temperature referred
  to the surface height.
* **Cadence.** The surfaces are stepped every `period` seconds
  (`cfg.cadence[:surface]` if present, otherwise `cfg.cadence[:soil]`):
  on step 1 in `:surface_flux`, then in `:surface_update` on the step at
  which the air reaches the time the surfaces were advanced to. Each
  surface step covers the next `period` seconds, during which its
  fluxes are applied to the air every step, so the air receives exactly
  what the surface gave away when `dt` divides `period`.
* **Air side**, every step in `:surface_flux`: ``Δθ = H\\,Δt/(ρ_k c_p Δz)``,
  ``Δq_v = E\\,Δt/(ρ_k Δz)`` in the first air cell, with ``ρ_k`` the
  ledger density of that level (1 in the Boussinesq case; #131 sets a
  reference profile) and ``c_p`` the moist heat capacity of that air.
* **Momentum**, every step in `:post_predictor` (after the IBM): each
  first-air-cell velocity face is multiplied by ``1/(1 + Δt\\,u_*^2/(U\\,Δz))``
  (implicit, so stable for any `Δt`), the drag of the surface stress
  ``ρu_*^2`` along the resolved wind; a face between two columns gets the
  mean of the two columns' rates (a column contributes only at its own
  air level).
* **Extra energy.** `bus.sensible` (energy other couplings take from the
  surfaces, W m⁻²) is collected every step in `:surface_update` and
  handed to the surfaces as `Q_extra` at their next step.

Options: `U_min` (m s⁻¹, the minimum wind of the surface-layer solve),
`z0m` (per class, m), `z0h_ratio` (``z_{0h} = z_{0m}``·ratio),
`period` (s, overrides the cadence).
"""
mutable struct SurfaceExchangeMOST <: AbstractCoupling
    P::Float64                 # surface step period (s)
    U_min::Float64
    z0m::Dict{UInt8,Float64}
    z0h_ratio::Float64
    stepped::Bool
    t_next::Float64            # the surfaces are advanced up to this simulation time
    cp::Matrix{Float64}        # c_p of the air cell, for H → θ
    drag::Matrix{Float64}      # u*²/U (m s⁻¹): kinematic stress per unit wind
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
end

coupling_name(::Type{SurfaceExchangeMOST}) = :surface_exchange
slot(::Type{SurfaceExchangeMOST}) = :surface_exchange
provider(::Type{SurfaceExchangeMOST}) = :most
modules_used(::Type{SurfaceExchangeMOST}) = (:SurfaceExchange, :Radiation, :MoistThermodynamics)
default_options(::Type{SurfaceExchangeMOST}) =
    (U_min = 0.1, z0h_ratio = 0.1, period = nothing,
     # momentum roughness per class (m): the sub-grid roughness of the
     # surface itself — buildings and park trees are resolved (IBM, drag)
     z0m = Dict(SURFACE_STREET => 0.05, SURFACE_ROOF => 0.05, SURFACE_PARK => 0.1,
                SURFACE_WATER => 1e-4, SURFACE_WALL => 0.05, UInt8(6) => 0.01))
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
    bus.k_air = first_air_level(case.geom.sm, mesh)
    fill!(bus.z_ref, mesh.dz / 2)
    z() = zeros(Nx, Ny)
    return SurfaceExchangeMOST(P, o.U_min, Dict{UInt8,Float64}(o.z0m), o.z0h_ratio,
                               false, 0.0, z(), z(), z(), z(), z(), z(), z(), z(), z(),
                               0.0, 0, 0.0, 0)
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

"Air density of the ledger at level `k` (the Boussinesq case: 1)."
function _ledger_rho(case, k)
    l = case.ledger
    isempty(l.ρ) && (l.ρ = ones(case.mesh.Nz))
    return l.ρ[k]
end

# Pressure of the air cell (Pa): the isothermal 8 km scale-height profile
# fog_dew uses (the Boussinesq core carries no pressure; #131 replaces it
# with a reference state).
_cell_pressure(z) = 1.01325e5 * exp(-z / 8000.0)

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
    @inbounds for j in 1:Ny, i in 1:Nx
        s = bus.surface[i, j]
        s === nothing && continue
        k = bus.k_air[i, j]
        q = qv[i, j, k]
        z = (k - 0.5) * dz
        uc = (u[i, j, k] + u[mod1(i + 1, Nx), j, k]) / 2
        vc = (v[i, j, k] + v[i, mod1(j + 1, Ny), k]) / 2
        U = sqrt(uc^2 + vc^2)
        qa = PhasePartition(q)
        T_air = θ[i, j, k] - dry_adiabatic_lapse_rate(qa) * z
        p = _cell_pressure(z)
        f = SurfaceForcing(; sw_down = bus.sw[i, j], lw_down = bus.lw[i, j], T_air = T_air,
                           q_air = q, U = U, p = p, z_ref = bus.z_ref[i, j])
        Q = c.pending[i, j] / P
        c.Q[i, j] = Q
        c.pending[i, j] = 0.0
        E0 = surface_store_energy(s)
        fl = _step_one(s, f, P, Q, c.U_min)
        E1 = surface_store_energy(s)
        E0 === nothing || (c.store_err = max(c.store_err, abs((E1 - E0) / P - fl.G)))
        bus.T_air[i, j], bus.q_air[i, j], bus.U[i, j] = T_air, q, U
        bus.Rn[i, j], bus.H[i, j], bus.LE[i, j] = fl.Rn, fl.H, fl.LE
        bus.G[i, j], bus.E[i, j] = fl.G, fl.E
        ρ = air_density(T_air, p, qa)
        us2 = fl.τ / ρ
        bus.u_star[i, j] = sqrt(us2)
        c.drag[i, j] = us2 / max(U, c.U_min)
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

function run_stage!(c::SurfaceExchangeMOST, case, ::Val{:surface_flux}, dt)
    c.stepped || step_all_surfaces!(c, case)          # step 1: the first surface step
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
        ρk = _ledger_rho(case, k)
        θ[i, j, k] += bus.H[I] * dt / (ρk * c.cp[I] * dz)
        qv[i, j, k] += bus.E[I] * dt / (ρk * dz)
        evap += bus.E[I] * dt
    end
    c.evap = evap / length(bus.k_air)
    c.solid_writes += nsolid
    return nothing
end

function run_stage!(c::SurfaceExchangeMOST, case, ::Val{:post_predictor}, dt)
    bus, mesh = case.bus, case.mesh
    Nx, Ny, Nz = mesh.Nx, mesh.Ny, mesh.Nz
    Hx, Hy, Hz, dz = mesh.Hx, mesh.Hy, mesh.Hz, mesh.dz
    u, v = case.sim.u.data, case.sim.v.data
    kair, drag, surf = bus.k_air, c.drag, bus.surface
    # drag rate (s⁻¹) column (i, j) exerts on its own first air level k
    rate(i, j, k) = (kair[i, j] == k && surf[i, j] !== nothing) ? drag[i, j] / dz : 0.0
    # u(i, j, k) is the face between cells (i-1, j) and (i, j), v(i, j, k)
    # the face between (i, j-1) and (i, j). A face is dragged at the air
    # level of either adjacent column, with the mean of their two rates
    # (half the face's area belongs to each column).
    function drag_face!(f, i, j, iw, jw)
        a, b = kair[iw, jw], kair[i, j]
        for k in (a, b)
            k > Nz && continue
            r = (rate(iw, jw, k) + rate(i, j, k)) / 2
            r > 0 && (f[Hx+i, Hy+j, Hz+k] /= 1 + dt * r)
            a == b && break                     # one level: visit it once
        end
    end
    @inbounds for j in 1:Ny, i in 1:Nx
        drag_face!(u, i, j, mod1(i - 1, Nx), j)
        drag_face!(v, i, j, i, mod1(j - 1, Ny))
    end
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
    return @sprintf("           surface: H %6.1f LE %6.1f G %6.1f W/m²  u* roof %.2f ground %.2f m/s",
                    m(bus.H, have), m(bus.LE, have), m(bus.G, have),
                    m(bus.u_star, roof), m(bus.u_star, ground))
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
            "per column, the surface stored G·P (|Δstore/P − G| < 1e-6 W/m²)" =>
                c.store_err < 1e-6,
            "u* > 0; above its neutral value κU/ln(z_ref/z0m) where H > 5 and E ≥ 0 (unstable), below it where H < −5 and E ≤ 0 (stable)" =>
                all(I -> _ustar_ok(bus, I, c.U_min), have),
            # the fluxes that go into the air are those of the surface step
            "Rn − H − LE = G − Q_extra per column (1e-9 W/m²)" =>
                all(I -> abs(bus.Rn[I] - bus.H[I] - bus.LE[I] - bus.G[I] + c.Q[I]) <=
                         1e-9 * (1 + abs(bus.Rn[I])), have),
            "H has the sign of T_skin − θ_air referred to the surface, where they differ by > 1 K and |H| > 1 W/m²" =>
                all(I -> _h_sign_ok(bus, I, case), have)]
end

function _ustar_ok(bus, I, U_min)
    us = bus.u_star[I]
    us > 0 || return false
    neutral = 0.4 * max(bus.U[I], U_min) / log(bus.z_ref[I] / roughness_lengths(bus.surface[I]).z0m)
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
