# The legacy slab surface scheme: one skin temperature per column with an
# areal heat capacity per class, a constant-coefficient bulk sensible
# flux and a moisture-availability latent flux.
#
# KNOWN DEFECTS, kept so the refactor is bit-for-bit; the surface-exchange
# issue replaces this whole file with SurfaceExchange (MOST) over
# SoilThermalColumn / UrbanFacet skins:
#   * constant C_h = 1e-3 and U = 3 m/s, ignoring the resolved wind;
#   * |H| clamped to 250 W/m², LE to [0, 400] W/m², Ts to [270, 340] K;
#   * sensible, latent (and dew) fluxes go into k = 1 of EVERY column,
#     which on roof columns is inside the building;
#   * no surface momentum drag.

"""
    step_surfaces!(Ts, sfc, params, sw_in_arr, lw_in, sensible_arr, dt)

Advance the surface temperature array by `dt`. Each cell's balance:

    C_s dT/dt = (1 - α) SW↓ + ε LW↓ - ε σ T⁴ - H

  * `sw_in_arr::AbstractMatrix` — SW at each cell;
  * `lw_in` — downward LW, a scalar or a matrix;
  * `sensible_arr::AbstractMatrix` — the surface loss term [W/m²]
    (positive means the surface loses energy).
"""
function step_surfaces!(Ts::AbstractMatrix, sfc::AbstractMatrix{UInt8},
                        params::SurfaceParams,
                        sw_in_arr::AbstractMatrix,
                        lw_in,
                        sensible_arr::AbstractMatrix, dt::Real)
    Nx, Ny = size(Ts)
    lw_scalar = lw_in isa AbstractMatrix ? nothing : Float64(lw_in)
    @inbounds for j in 1:Ny, i in 1:Nx
        c = sfc[i, j]
        α = params.albedo[c]; ε = params.emissivity[c]; C = params.C_s[c]
        lw = lw_scalar === nothing ? lw_in[i, j] : lw_scalar
        net = (1 - α) * sw_in_arr[i, j] + ε * lw -
              ε * STEFAN * Ts[i, j]^4 - sensible_arr[i, j]
        Ts[i, j] += net * dt / C
    end
    return Ts
end

"""
    apply_surface_flux!(out, sim, Ts, sfc, params; ρcp = 1005.0, U_char = 3.0,
                        C_h = 5e-3, dt = 1.0, H_cap = 400.0) -> out

Bulk sensible flux `H = ρcp · C_h · U_char · (Ts − θ₁)`, clamped to
`±H_cap`, written into `out` (W/m², positive upward) and injected into
the first interior layer of `sim.θ` over `dt`.
"""
function apply_surface_flux!(out::AbstractMatrix, sim, Ts::AbstractMatrix,
                             sfc::AbstractMatrix{UInt8}, params::SurfaceParams;
                             ρcp::Real = 1005.0, U_char::Real = 3.0,
                             C_h::Real = 5e-3, dt::Real = 1.0,
                             H_cap::Real = 400.0)
    Nx, Ny = size(Ts)
    Hx, Hy, Hz = sim.mesh.Hx, sim.mesh.Hy, sim.mesh.Hz
    dz = sim.mesh.dz
    @inbounds for j in 1:Ny, i in 1:Nx
        θ_air = sim.θ.data[Hx+i, Hy+j, Hz+1]
        H_flx = ρcp * C_h * U_char * (Ts[i, j] - θ_air)
        H_flx = clamp(H_flx, -H_cap, H_cap)
        Fθ = H_flx / ρcp
        out[i, j] = H_flx
        sim.θ.data[Hx+i, Hy+j, Hz+1] += Fθ * dt / dz
    end
    return out
end

"""
    q_saturation(T_K, p_Pa = 1.01325e5) -> kg/kg

Saturation specific humidity. Delegates to the Era IV contract
(`Aeolus.MoistThermodynamics.saturation_specific_humidity`), so that
evaporation here and dew in the fog/dew coupling agree exactly on where
saturation is.
"""
q_saturation(T_K::Real, p::Real = 1.01325e5) =
    Aeolus.MoistThermodynamics.saturation_specific_humidity(T_K, p)

# Per-class moisture-availability factor β ∈ [0, 1]: 1 means saturated
# surface (open water), 0 means desert.
function _moisture_availability(cls::UInt8)
    cls == SURFACE_WATER  ? 1.0 :
    cls == SURFACE_PARK   ? 0.6 :
    cls == SURFACE_WALL   ? 0.05 :
    cls == SURFACE_ROOF   ? 0.05 :
                            0.15    # street (asphalt after rain: some LE)
end

"""
    apply_latent_flux!(LE, sim, Ts, sfc; ρ = 1.0, C_e = 1e-3, U_char = 3.0,
                       qv_air = 0.008, LE_cap = 400.0, dt = 1.0,
                       qv_field = nothing) -> LE

Evaporative cooling `LE = ρ L_v C_e U β max(0, q_sat(Ts) − q_v)`, clamped
to `[0, LE_cap]`, written into `LE` (W/m²); with `qv_field` the matching
vapour source goes into its first layer (`q_v` is then that layer,
otherwise `qv_air`). `β` is the per-class moisture availability.
"""
function apply_latent_flux!(LE::AbstractMatrix, sim, Ts::AbstractMatrix,
                            sfc::AbstractMatrix{UInt8};
                            ρ::Real = 1.0, C_e::Real = 1e-3,
                            U_char::Real = 3.0,
                            qv_air::Real = 0.008,
                            LE_cap::Real = 400.0,
                            dt::Real = 1.0,
                            qv_field::Union{Nothing, Any} = nothing)
    Nx, Ny = size(Ts)
    dz = sim.mesh.dz
    @inbounds for j in 1:Ny, i in 1:Nx
        c   = sfc[i, j]
        β   = _moisture_availability(c)
        qsat = q_saturation(Ts[i, j])
        qv_local = qv_field === nothing ? qv_air : interior_view(qv_field)[i, j, 1]
        Δq  = max(0.0, qsat - qv_local)
        H_L = ρ * _L_v * C_e * U_char * β * Δq
        H_L = clamp(H_L, 0.0, LE_cap)
        LE[i, j] = H_L
        if qv_field !== nothing
            interior_view(qv_field)[i, j, 1] += dt * H_L / (ρ * _L_v * dz)
        end
    end
    return LE
end

"""
    SurfaceSlab

Slot `:surface_exchange`, provider `:slab_bulk`: the legacy slab scheme.

* `:surface_flux` — bulk sensible flux into θ (C_h = 1e-3, |H| ≤ 250),
  which starts the step's `bus.sensible`;
* `:anthropogenic` @ 90 — latent flux into qv, added to `bus.sensible`.
  It runs after the buildings, traffic and glazing only to reproduce the
  pre-refactor floating-point order; conceptually it is a surface flux;
* `:surface_update` — `step_surfaces!`;
* `:sanity` — the Ts clamp to `[Ts_min, Ts_max]` (a known defect: it
  hides energy; the ledger will count what it clips).
"""
struct SurfaceSlab <: AbstractCoupling
    LE::Matrix{Float64}
    H::Matrix{Float64}        # this step's bulk sensible flux (W/m², for the ledger)
    clip::Matrix{Float64}     # energy the Ts clamp added this step (W/m², for the ledger)
    C_h::Float64
    H_cap::Float64
    qv_air::Float64
    Ts_min::Float64
    Ts_max::Float64
end

coupling_name(::Type{SurfaceSlab}) = :surface_slab
slot(::Type{SurfaceSlab}) = :surface_exchange
provider(::Type{SurfaceSlab}) = :slab_bulk
modules_used(::Type{SurfaceSlab}) = (:MoistThermodynamics,)
default_options(::Type{SurfaceSlab}) = (C_h = 1e-3, H_cap = 250.0, qv_air = 0.010,
                                        Ts_min = 270.0, Ts_max = 340.0)
stages(::Type{SurfaceSlab}) = ((:surface_flux, 10, EVERY_STEP),
                               (:anthropogenic, 90, EVERY_STEP),
                               (:surface_update, 10, EVERY_STEP),
                               (:sanity, 10, EVERY_STEP))
build(::Type{SurfaceSlab}, case, o) =
    SurfaceSlab(zeros(case.mesh.Nx, case.mesh.Ny), zeros(case.mesh.Nx, case.mesh.Ny),
                zeros(case.mesh.Nx, case.mesh.Ny), o.C_h, o.H_cap, o.qv_air, o.Ts_min, o.Ts_max)

function run_stage!(c::SurfaceSlab, case, ::Val{:surface_flux}, dt)
    o, bus = c, case.bus
    apply_surface_flux!(bus.sensible, case.sim, bus.Ts, case.geom.sfc, bus.params;
                        dt = dt, H_cap = o.H_cap, C_h = o.C_h)
    copyto!(c.H, bus.sensible)
    return nothing
end

function run_stage!(c::SurfaceSlab, case, ::Val{:anthropogenic}, dt)
    o, bus = c, case.bus
    apply_latent_flux!(c.LE, case.sim, bus.Ts, case.geom.sfc; dt = dt,
                       qv_field = tracer(case, :qv), qv_air = o.qv_air)
    @. bus.sensible += c.LE
    return nothing
end

function run_stage!(c::SurfaceSlab, case, ::Val{:surface_update}, dt)
    bus = case.bus
    step_surfaces!(bus.Ts, case.geom.sfc, bus.params, bus.sw, bus.lw, bus.sensible, dt)
    return nothing
end

function run_stage!(c::SurfaceSlab, case, ::Val{:sanity}, dt)
    Ts, C = case.bus.Ts, case.bus.params.C_s
    sfc = case.geom.sfc
    @inbounds for I in eachindex(Ts)
        Tc = clamp(Ts[I], c.Ts_min, c.Ts_max)
        c.clip[I] = C[sfc[I]] * (Tc - Ts[I]) / dt
        Ts[I] = Tc
    end
    return nothing
end

"""
    slab_net_radiation(Ts, sfc, params, sw, lw) -> Matrix

Net radiation absorbed by the slab, ``(1-α)SW + εLW↓ - εσT_s^4`` (W m⁻²),
exactly the radiative part of [`step_surfaces!`](@ref).
"""
function slab_net_radiation(Ts, sfc, params::SurfaceParams, sw, lw)
    Rn = similar(Ts, Float64)
    @inbounds for I in eachindex(Ts)
        k = sfc[I]
        α, ε = params.albedo[k], params.emissivity[k]
        Rn[I] = (1 - α) * sw[I] + ε * lw[I] - ε * STEFAN * Ts[I]^4
    end
    return Rn
end

# Ledger: the slab owns the surface storage, so it books Rn, H, LE, the
# storage change C_s ΔTs over the whole step (the soil exchange and the
# clamp included) and the clamp's energy as an extra term. Water: the
# evaporation it put into qv.
function budget!(c::SurfaceSlab, case, l)
    bus, sfc, C = case.bus, case.geom.sfc, case.bus.params.C_s
    dt = case.cfg.dt
    add_surface!(l, :Rn, slab_net_radiation(l.Ts_start, sfc, bus.params, bus.sw, bus.lw), sfc, dt)
    add_surface!(l, :H, c.H, sfc, dt)
    add_surface!(l, :LE, c.LE, sfc, dt)
    dS = similar(bus.Ts)
    @inbounds for I in eachindex(dS)
        dS[I] = C[sfc[I]] * (bus.Ts[I] - l.Ts_start[I]) / dt
    end
    add_surface!(l, :dS, dS, sfc, dt)
    add_surface!(l, :clamp, c.clip, sfc, dt)
    # apply_latent_flux! adds LE·dt/(ρ L_v dz) to qv at k = 1, ρ = 1
    add_water!(l, :evaporation, sum(c.LE) * dt / _L_v / length(c.LE))
    return nothing
end

function smoke_check(c::SurfaceSlab, case)
    Ts = case.bus.Ts
    return ["Ts within the clamp [$(c.Ts_min), $(c.Ts_max)] K" =>
                all(T -> c.Ts_min <= T <= c.Ts_max, Ts),
            "|H| ≤ H_cap = $(c.H_cap) W/m²" => maximum(abs, c.H) <= c.H_cap,
            "0 ≤ LE ≤ 400 W/m² (no dew through the latent flux)" =>
                all(x -> 0 <= x <= 400, c.LE),
            "H has the sign of Ts − θ₁ where |Ts − θ₁| > 0.5 K" =>
                _h_sign_ok(c, case)]
end

# The bulk flux is ρcp C_h U (Ts − θ₁) evaluated before the step; at the
# end of the step θ₁ and Ts have moved by well under 0.5 K in a 2 s step, so the
# sign still holds wherever the difference is larger than 0.5 K.
function _h_sign_ok(c::SurfaceSlab, case)
    θ1 = view(interior_view(case.sim.θ), :, :, 1)
    Ts = case.bus.Ts
    return all(I -> abs(Ts[I] - θ1[I]) <= 0.5 || sign(c.H[I]) == sign(Ts[I] - θ1[I]),
               eachindex(Ts))
end

register!(SurfaceSlab)
