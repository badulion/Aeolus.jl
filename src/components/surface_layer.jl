# The Monin–Obukhov surface layer over a prescribed surface temperature
# (ONE-g, #456): the heat flux and the wall stress of every column, the
# `wall!` closure and the surface solve of the former `sbl` run type
# (P55-f #191, P58-d #194).

"""
    SurfaceLayer

The `:surface` slot's Monin–Obukhov surface layer, `[surface] model =
"most"` (ONE-m, #476; formerly `[surface_layer] enabled`):
every column solves similarity theory (`SurfaceExchange.solve_surface_layer`)
between the first-level wind and θ (at ``Δz/2``) and the prescribed
surface temperature ``θ_s(t)``, with the roughness lengths `z0m`, `z0h`
and the stability functions of `stability`. From that one solution

  * the kinematic heat flux ``w'θ'_0 = -u_*θ_*`` (`kinematic_heat_flux`)
    enters the first level as ``w'θ'_0/Δz`` in stage 6 (`:tendencies`) of
    every (sub)step;
  * the momentum flux is the wall law `WallModels.MOSTWallLaw` of the
    column with the solution's Obukhov length (it returns the solver's
    ``u_*`` bit for bit), applied implicitly to the first level by
    `WallModels.apply_wall_stress!` in stage 8 (`:post_predictor`).

The solve is once per step (stage 3, `:surface_flux`) from the
start-of-step state at the mid-step surface temperature; the flux is held
over the step (its SSP-RK3 weights sum to one, so exactly ``⟨w'θ'_0⟩Δt``
enters each column), the wall law is applied in every substage with the
start-of-step ``L``.

The surface temperature `temperature` is `{kind = "constant", value}`,
`{kind = "cooling_rate", initial, rate_per_hour}` (from the run's start)
or `{kind = "driver"}`: the dynamic driver's `sfc_temperature` series (a
potential temperature; its times read at the model time minus `[time]
start`). Its outputs ([`outputs`](@ref)) are the boundary-layer
diagnostics of the stable-boundary-layer cases.

References: Monin & Obukhov (1954); Beare et al. (2006), Boundary-Layer
Meteorol. 118, 247 (GABLS1); Basu & Porté-Agel (2006), J. Atmos. Sci. 63,
2074.
"""
struct SurfaceLayer{S<:AbstractStabilityFunctions, TS, W, L} <: AbstractComponent
    sf::S
    z0m::Float64
    z0h::Float64
    Ts::TS                       # SurfaceTemperatureSeries
    ρcp::Float64
    wm::W                        # WallModel of `laws`
    laws::L                      # Matrix{MOSTWallLaw}: the per-column law of the current step
    wθ::Matrix{Float64}          # per-column kinematic surface heat flux (K m s⁻¹, upward +)
    ustar::Matrix{Float64}       # per-column u_* of the solve
    Lob::Matrix{Float64}         # per-column Obukhov length
    rk3::Bool                    # weights of the applied stress: SSP-RK3 (1/6, 1/6, 2/3) or one Euler step
    τ::Vector{Float64}           # plane-mean kinematic stress applied: the last step's (τx, τy), this step's sums
    nsub::Base.RefValue{Int}     # the substage of the current step
    t_surf::Base.RefValue{Float64}   # the time of the last solve
    cache::Base.RefValue{Any}        # (time, profiles) of the outputs of one record
end

component_name(::Type{<:SurfaceLayer}) = :surface_layer
section(::Type{<:SurfaceLayer}) = "surface"
slot(::Type{<:SurfaceLayer}) = :surface

const _SL_TEMPERATURE_KEYS = Dict("constant" => ["value"], "cooling_rate" => ["initial", "rate_per_hour"],
                                  "driver" => String[])
const _SL_STABILITY_KEYS = Dict("businger_dyer_log_linear" => ["beta", "beta_m", "beta_h", "gamma_m", "gamma_h"],
                                "businger_dyer_beljaars_holtslag" => ["gamma_m", "gamma_h", "a", "b", "c", "d"])

_sl_isnum(x) = x isa Real && !(x isa Bool) && isfinite(x)

function _sl_temperature_problems(t)
    kind = get(t, "kind", nothing)
    haskey(_SL_TEMPERATURE_KEYS, kind) ||
        return ["kind: expected one of " * join(repr.(sort!(collect(keys(_SL_TEMPERATURE_KEYS)))), ", ") * ", got $(repr(kind))"]
    want = _SL_TEMPERATURE_KEYS[kind]
    probs = String[]
    for k in sort!(collect(keys(t)))
        k == "kind" || k in want || push!(probs, "$k: unknown key for kind = \"$kind\" (keys: $(join(want, ", ")))")
    end
    for k in want
        _sl_isnum(get(t, k, nothing)) || push!(probs, "$k: required, a finite number")
    end
    return probs
end

function _sl_stability_problems(t)
    fam = get(t, "family", nothing)
    haskey(_SL_STABILITY_KEYS, fam) ||
        return ["family: expected one of " * join(repr.(sort!(collect(keys(_SL_STABILITY_KEYS)))), ", ") * ", got $(repr(fam))"]
    probs = String[]
    for k in sort!(collect(keys(t)))
        k == "family" && continue
        k in _SL_STABILITY_KEYS[fam] || (push!(probs, "$k: unknown key for family = \"$fam\""); continue)
        _sl_isnum(t[k]) || push!(probs, "$k: expected a finite number")
    end
    return probs
end

schema(::Type{<:SurfaceLayer}) =
    Config.slot_section("surface"; key = "model") do sec
      Config.provider!(sec, "most";
                       doc = "The Monin-Obukhov surface layer over a prescribed surface temperature (SurfaceExchange, " *
                             "WallModels.MOSTWallLaw): the heat flux and the wall stress of every column.") do s
        Config.option!(s, "z0m", Float64; default = 0.1, range = (0, Inf), open_lower = true, units = "m",
                       doc = "momentum roughness length")
        Config.option!(s, "z0h", Config.Optional{Float64}; range = (0, Inf), open_lower = true, units = "m",
                       doc = "scalar roughness length (the height of the prescribed θ); \"none\": z0m")
        Config.option!(s, "temperature", Config.OpenTable(_sl_temperature_problems);
                       default = Dict{String,Any}("kind" => "constant", "value" => 300.0),
                       doc = "the surface θ (K): {kind = \"constant\", value}, {kind = \"cooling_rate\", initial, " *
                             "rate_per_hour} (from the run's start), or {kind = \"driver\"}: the dynamic driver's " *
                             "sfc_temperature series (a potential temperature)")
        Config.option!(s, "stability", Config.OpenTable(_sl_stability_problems);
                       default = Dict{String,Any}("family" => "businger_dyer_beljaars_holtslag"),
                       doc = "the stability functions: {family = \"businger_dyer_log_linear\", beta | beta_m, beta_h, " *
                             "gamma_m, gamma_h} or {family = \"businger_dyer_beljaars_holtslag\", gamma_m, gamma_h, a, b, c, d}; " *
                             "keys left out take the family's defaults")
        Config.option!(s, "rho_cp", Float64; default = 1206.0, range = (0, Inf), open_lower = true,
                       units = "J m^-3 K^-1", doc = "ρ c_p of the surface_heat_flux output (W m⁻²; diagnostic only)")
      end
    end
enabled(::Type{<:SurfaceLayer}, opts) = opts.model === :most

"""
    stability_functions(t::AbstractDict) -> AbstractStabilityFunctions

The `SurfaceExchange` stability functions of a `[surface] stability`
table (`model = "most"`): `BusingerDyerLogLinear` (`beta` sets both slopes, `beta_m`,
`beta_h` each) or `BusingerDyerBeljaarsHoltslag`, keys left out at the
family's defaults.
"""
function stability_functions(t::AbstractDict)
    d = Dict{Symbol,Float64}()
    for (k, v) in t
        k == "family" || (d[Symbol(k)] = Float64(v))
    end
    if t["family"] == "businger_dyer_log_linear"
        β = get(d, :beta, 5.0)
        return BusingerDyerLogLinear(; γ_m = get(d, :gamma_m, 16.0), γ_h = get(d, :gamma_h, 16.0), β = β,
                                     β_m = get(d, :beta_m, β), β_h = get(d, :beta_h, β))
    end
    return BusingerDyerBeljaarsHoltslag(; (Symbol(replace(String(k), "gamma" => "γ")) => v for (k, v) in d)...)
end

function _surface_series(t, domain)
    t0 = clock(domain)
    kind = t["kind"]
    if kind == "constant"
        v = Float64(t["value"])
        return SurfaceTemperatureSeries([t0, t0 + 86400.0], [v, v])
    elseif kind == "cooling_rate"
        T0 = Float64(t["initial"]); r = Float64(t["rate_per_hour"]) / 3600
        return SurfaceTemperatureSeries([t0, t0 + 86400.0], [T0, T0 - r * 86400])
    end
    dyn = get(domain.drivers, :dynamic, nothing)
    dyn === nothing && throw(ConfigError("surface.temperature",
                                         "kind = \"driver\" reads the domain's dynamic driver, which the domain has not"))
    s = dyn.surface
    (s === nothing || s.temperature === nothing) &&
        throw(ConfigError("surface.temperature", "the dynamic driver has no sfc_temperature series"))
    return SurfaceTemperatureSeries(s.t .+ t0, s.temperature)
end

function build(::Type{<:SurfaceLayer}, opts, domain)
    m = grid(domain)
    sf = stability_functions(opts.stability)
    z0h = something(opts.z0h, opts.z0m)
    laws = [MOSTWallLaw(sf, opts.z0m, Inf) for _ in 1:m.Nx, _ in 1:m.Ny]
    z = zeros(m.Nx, m.Ny)
    return SurfaceLayer(sf, Float64(opts.z0m), Float64(z0h), _surface_series(opts.temperature, domain),
                        Float64(opts.rho_cp), WallModel(laws), laws, copy(z), copy(z), fill(Inf, m.Nx, m.Ny),
                        domain.scheme === :rk3, zeros(4), Ref(0), Ref(clock(domain)), Ref{Any}(nothing))
end

stages(::SurfaceLayer) = (Stage(:surface_flux), Stage(:tendencies), Stage(:post_predictor))

# cell-centre wind and θ of the first level of column (i, j)
@inline function _first_level(s, i, j)
    m = s.mesh
    I, J, K = m.Hx + i, m.Hy + j, m.Hz + 1
    uc = 0.5 * (s.u.data[I, J, K] + s.u.data[I+1, J, K])
    vc = 0.5 * (s.v.data[I, J, K] + s.v.data[I, J+1, K])
    return uc, vc, s.θ.data[I, J, K]
end

# stage 3: the surface layer of every column, from the start-of-step state
# at the mid-step surface temperature
function run_stage!(c::SurfaceLayer, d, ::Val{:surface_flux}, dt)
    s = d.state; m = s.mesh
    t = clock(d) + dt / 2
    θs = surface_temperature(c.Ts, t)
    c.t_surf[] = t
    z1 = m.dz / 2
    # the first-level wind of column Nx (Ny) reads the periodic halo: refill
    # it from the current interior (the last RK3 combination left it stale)
    fill_halos!(s.u, s.bcs); fill_halos!(s.v, s.bcs)
    @inbounds for j in 1:m.Ny, i in 1:m.Nx
        uc, vc, θ = _first_level(s, i, j)
        sol = solve_surface_layer(c.sf; U = sqrt(uc^2 + vc^2), z_m = z1, θ_a = θ, θ_s = θs,
                                  z0m = c.z0m, z0h = c.z0h)
        c.wθ[i, j] = kinematic_heat_flux(sol)
        c.ustar[i, j] = sol.u_star
        c.Lob[i, j] = sol.L
        c.laws[i, j] = MOSTWallLaw(c.sf, c.z0m, sol.L)
    end
    c.τ[3] = 0.0; c.τ[4] = 0.0; c.nsub[] = 0
    return nothing
end

# the u* of the step's solve: the surface value of a TKE closure ([sgs] surface_value)
surface_friction_velocity(c::SurfaceLayer, d) = c.ustar

# stage 6: the heat flux into the first level, every (sub)step
function run_stage!(c::SurfaceLayer, d, ::Val{:tendencies}, dt)
    s = d.state; m = s.mesh
    K = m.Hz + 1
    tθ = s.tθ.data
    @inbounds for j in 1:m.Ny, i in 1:m.Nx
        tθ[m.Hx+i, m.Hy+j, K] += c.wθ[i, j] / m.dz
    end
    return nothing
end

const _SL_RK3_WEIGHTS = (1 / 6, 1 / 6, 2 / 3)

# stage 8: the wall stress on the predicted velocity, every (sub)step; the
# plane-mean stress the step applied, weighted as the scheme combines its substages
function run_stage!(c::SurfaceLayer, d, ::Val{:post_predictor}, dt)
    s = d.state; m = s.mesh
    c.nsub[] += 1
    # the first substage of an attempt (a retried adaptive step repeats the
    # core step without the stages before it) starts the step's sums afresh
    (!c.rk3 || mod1(c.nsub[], 3) == 1) && (c.τ[3] = 0.0; c.τ[4] = 0.0)
    w = c.rk3 ? _SL_RK3_WEIGHTS[mod1(c.nsub[], 3)] : 1.0
    F = apply_wall_stress!(s, c.wm, dt; side = :bottom)
    A = m.Lx * m.Ly
    c.τ[3] += w * (-F.Fx) / A
    c.τ[4] += w * (-F.Fy) / A
    if !c.rk3 || c.nsub[] % 3 == 0          # the step's last substage
        c.τ[1] = c.τ[3]; c.τ[2] = c.τ[4]
    end
    return nothing
end

# ---------------------------------------------------------------------------
# diagnostics: the outputs
# ---------------------------------------------------------------------------

"The plane-mean surface friction velocity of the last step: √|τ| of the applied wall stress."
sl_ustar(c::SurfaceLayer) = sqrt(hypot(c.τ[1], c.τ[2]))

"""
    sl_profiles(c, d) -> NamedTuple

The plane-mean profiles of domain `d` with the surface layer `c`: at the
cell centres `z` the wind `U`, `V`, the speed `S` and `θ`; at the w-faces
`zf` the total (resolved + SGS) downward momentum flux `τx`, `τy` and its
magnitude `τ` (`Operators.total_stress_profile` with the SGS closure's
last vertical viscosity and the applied wall stress at the surface) and
the total upward heat flux `wθ` (`Operators.total_heat_flux_profile`, the
surface flux at the bottom face).
"""
function sl_profiles(c::SurfaceLayer, d)
    s = d.state; m = s.mesh
    fill_halos!(s.u, s.bcs); fill_halos!(s.v, s.bcs); fill_halos!(s.w, s.bcs); fill_halos!(s.θ, s.θ_bcs)
    clo = sgs_closure(d)
    νv, κv = if clo === nothing
        s.νt, s.κt
    else
        f = sgs_fields(clo, s); (f.νv, f.κv)
    end
    fill_halos!(νv, s.bcs); fill_halos!(κv, s.bcs)
    px = total_stress_profile(s.u, s.v, s.w, νv; ν = 0.0, τ_wall_bottom = c.τ[1])
    py = total_stress_profile(s.u, s.v, s.w, νv; ν = 0.0, τ_wall_bottom = c.τ[2], component = :y)
    # the cell-centre plane means of u (x-faces) and v (y-faces) are their face means (periodic x, y)
    U = _profile(s.u); V = _profile(s.v); θ = _profile(s.θ)
    q = total_heat_flux_profile(s.w, s.θ, κv; surface = sum(c.wθ) / length(c.wθ))
    z = collect(last(cell_centres(m)))
    return (z = z, U = U, V = V, S = sqrt.(U .^ 2 .+ V .^ 2), θ = θ,
            zf = px.z, τx = px.total, τy = py.total, τ = sqrt.(px.total .^ 2 .+ py.total .^ 2), wθ = q.total)
end

# the profiles of the outputs: computed once per model time (the outputs of
# one record, or one step of a mean, share them)
function _cached_profiles(c::SurfaceLayer, d)
    t = clock(d)
    x = c.cache[]
    (x !== nothing && x[1] == t && x[2] == c.t_surf[]) && return x[3]
    p = sl_profiles(c, d)
    c.cache[] = (t, c.t_surf[], p)
    return p
end

# the profile y at height zq (linear between cell centres, held beyond)
function _at_height(z, y, zq)
    k = clamp(searchsortedlast(z, zq), 1, length(z) - 1)
    return y[k] + (y[k+1] - y[k]) * (zq - z[k]) / (z[k+1] - z[k])
end

"""
    sl_screen(c, d; z_θ = 2, z_U = 10) -> (θ_screen, U_screen)

Plane means of the screen-level potential temperature at `z_θ` (2 m) and
wind speed at `z_U` (10 m). Below the first model level (``Δz/2``) each
column's profile is Monin–Obukhov similarity with the ``u_*``, ``θ_* =
-w'θ'_0/u_*`` and ``L`` of its last solve (`SurfaceExchange.similarity_theta`,
`similarity_wind`); at or above it the resolved plane-mean profile is
interpolated linearly.
"""
function sl_screen(c::SurfaceLayer, d; z_θ::Real = 2.0, z_U::Real = 10.0)
    m = grid(d); z1 = m.dz / 2
    θs = surface_temperature(c.Ts, c.t_surf[])
    n = length(c.wθ)
    p = (z_θ < z1 && z_U < z1) ? nothing : _cached_profiles(c, d)
    θscr = z_θ < z1 ?
        sum(similarity_theta(c.sf, z_θ; θ_s = θs, θ_star = -c.wθ[i] / c.ustar[i], L = c.Lob[i], z0h = c.z0h)
            for i in eachindex(c.wθ)) / n : _at_height(p.z, p.θ, z_θ)
    Uscr = z_U < z1 ?
        sum(similarity_wind(c.sf, z_U; u_star = c.ustar[i], L = c.Lob[i], z0m = c.z0m) for i in eachindex(c.ustar)) / n :
        _at_height(p.z, p.S, z_U)
    return (θ_screen = θscr, U_screen = Uscr)
end

"The Obukhov length of the plane means: ``L = -⟨u_*⟩^3 θ_s / (κ g ⟨w'θ'_0⟩)`` (`SurfaceExchange.obukhov_length`)."
function sl_obukhov(c::SurfaceLayer)
    n = length(c.wθ)
    u = sum(c.ustar) / n; wθ = sum(c.wθ) / n
    θs = surface_temperature(c.Ts, c.t_surf[])
    return wθ == 0 ? Inf : obukhov_length(u, -wθ / u, θs)
end

const _SL_OUTPUTS = ("u_star", "surface_wtheta", "surface_heat_flux", "obukhov_length", "surface_temperature",
                     "theta_2m", "wind_10m", "bl_height", "jet_speed", "jet_height", "inversion_height",
                     "wind_speed_profile", "stress_profile")

output_names(::Type{<:SurfaceLayer}, opts) = _SL_OUTPUTS


function outputs(c::SurfaceLayer)
    sl(d) = c
    mean_of(a) = sum(a) / length(a)
    return OutputVariable[
        OutputVariable("u_star", "m s-1", "surface friction velocity (the applied wall stress, plane mean)", :scalar,
                       d -> sl_ustar(sl(d))),
        OutputVariable("surface_wtheta", "K m s-1", "surface kinematic heat flux, plane mean", :scalar,
                       d -> mean_of(sl(d).wθ)),
        OutputVariable("surface_heat_flux", "W m-2", "surface sensible heat flux, plane mean (rho_cp w'theta')", :scalar,
                       d -> sl(d).ρcp * mean_of(sl(d).wθ)),
        OutputVariable("obukhov_length", "m", "Obukhov length of the plane-mean u* and heat flux", :scalar,
                       d -> sl_obukhov(sl(d))),
        OutputVariable("surface_temperature", "K", "prescribed surface potential temperature", :scalar,
                       d -> surface_temperature(sl(d).Ts, clock(d))),
        OutputVariable("theta_2m", "K", "screen-level (2 m) potential temperature, plane mean", :scalar,
                       d -> sl_screen(sl(d), d).θ_screen),
        OutputVariable("wind_10m", "m s-1", "10 m wind speed, plane mean", :scalar,
                       d -> sl_screen(sl(d), d).U_screen),
        OutputVariable("bl_height", "m", "boundary-layer height: the total stress falls to 5 % of its surface value, / 0.95",
                       :scalar, d -> (p = _cached_profiles(sl(d), d); stress_bl_height(p.zf, p.τ))),
        OutputVariable("jet_speed", "m s-1", "maximum of the plane-mean wind speed (the low-level jet)", :scalar,
                       d -> maximum(_cached_profiles(sl(d), d).S)),
        OutputVariable("jet_height", "m", "height of the maximum of the plane-mean wind speed", :scalar,
                       d -> (p = _cached_profiles(sl(d), d); p.z[argmax(p.S)])),
        OutputVariable("inversion_height", "m", "height of the largest theta gradient above 300 m (NaN below)", :scalar,
                       d -> (p = _cached_profiles(sl(d), d); inversion_height(p.z, p.θ))),
        OutputVariable("wind_speed_profile", "m s-1", "speed of the plane-mean wind", :z,
                       d -> _cached_profiles(sl(d), d).S),
        OutputVariable("stress_profile", "m2 s-2", "total (resolved + SGS) downward momentum flux magnitude, at the cell centres",
                       :z, d -> _centred(_cached_profiles(sl(d), d).τ))]
# (its total heat-flux profile is the core's heat_flux_profile, with its surface flux at the bottom face)
end

# a restart continues the surface layer of the last step bit for bit
function checkpoint_state(c::SurfaceLayer)
    return (laws_L = [l.L for l in c.laws], wθ = copy(c.wθ), ustar = copy(c.ustar), Lob = copy(c.Lob),
            τ = copy(c.τ), nsub = c.nsub[], t_surf = c.t_surf[])
end
function restore_checkpoint_state!(c::SurfaceLayer, s)
    for i in eachindex(c.laws)
        c.laws[i] = MOSTWallLaw(c.sf, c.z0m, s.laws_L[i])
    end
    c.wθ .= s.wθ; c.ustar .= s.ustar; c.Lob .= s.Lob; c.τ .= s.τ
    c.nsub[] = s.nsub; c.t_surf[] = s.t_surf
    return c
end

# the plane mean of the per-column surface flux of the step (the TKE closure's surface buoyancy)
wall_theta_flux(c::SurfaceLayer, d) = (sum(c.wθ) / length(c.wθ), 0.0)
