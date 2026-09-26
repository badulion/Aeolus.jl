# Urban-physics helpers for the Tier 3 flagship example.
#
# Ships:
#   * Per-surface-class `SlabLand`-like state via a 2-D array of
#     surface temperatures.
#   * `SurfaceParams` — heat capacities + albedos + roughness by class.
#   * `solar_zenith(lat, day, hour)` + `solar_declination(day)`
#     - simple astronomical model
#   * `sw_direct(zenith, sw_top)` - direct SW at surface (Beer's law
#     with a fixed optical depth).
#   * `step_surfaces!(Ts, sfc, params, sw_in, lw_in, sensible, dt)`
#     - advance each cell's surface temperature.
#   * `apply_surface_flux!(sim, Ts, sfc, params, θ_air_1)` -
#     push sensible flux from ground into the atmosphere.
#   * `add_geostrophic_wind!(sim, U_g, V_g, τ_relax)` - relax mean
#     u, v toward a geostrophic wind.

include(joinpath(@__DIR__, "urban_geometry.jl"))
using Aeolus

export SurfaceParams, default_surface_params,
       solar_declination, solar_zenith, solar_elev_azim, sun_direction,
       sw_direct,
       init_surface_state, step_surfaces!,
       apply_surface_flux!, apply_latent_flux!, add_geostrophic_wind!,
       mean_surface_temperature,
       compute_shadow_mask!, compute_sky_view_factor,
       apply_cloud_shadow!, wall_directional_factor,
       CityBuildings, city_buildings, step_city_buildings!, apply_building_heat!,
       add_traffic_heat!,
       q_saturation

# ---------------------------------------------------------------------------
# Per-class physical parameters
# ---------------------------------------------------------------------------

"""
    SurfaceParams(class_index -> (C_s, albedo, emissivity))

Look-up of thermal parameters per surface class. Values are typical
warm-summer daytime numbers.
"""
struct SurfaceParams
    C_s::Vector{Float64}        # areal heat capacity [J/m²/K]
    albedo::Vector{Float64}
    emissivity::Vector{Float64}
end

function default_surface_params(; μ₀::Real = 0.7)
    # Class index: 1=street, 2=roof, 3=park, 4=water, 5=wall (adjacent).
    #
    # Albedo and emissivity are NOT hand-entered here. They come from
    # `Aeolus.Radiosity.default_urban_materials`, which derives them by
    # band-averaging the Phase 36 spectral reflectance library over a
    # mosaic of real materials per class. Until Era III this file carried
    # its own copy of the five numbers, `Radiosity` carried a second copy
    # and `SurfaceOptics` carried the underlying curves — three places to
    # change and nothing keeping them consistent.
    #
    # `μ₀` (cosine solar zenith) only affects the water class, whose
    # Fresnel reflectance runs from ~0.03 overhead to ~0.29 at grazing
    # incidence. For a diurnal run, rebuild the params as the sun moves;
    # the default corresponds to a sun about 45° above the horizon.
    #
    # Heat capacity stays here: it is a bulk thermal property of the
    # surface layer, not an optical one, so it has no home in the
    # radiation modules. Walls have HIGH effective heat capacity (thick
    # masonry), water effectively enormous.
    mats = default_urban_materials(; μ₀ = μ₀)
    C_s  = [3e5, 1.8e5, 5e5, 8e6, 4e5]
    return SurfaceParams(C_s, copy(mats.albedo), copy(mats.emissivity))
end

# ---------------------------------------------------------------------------
# Solar geometry
# ---------------------------------------------------------------------------

"Solar declination (deg) for day-of-year `d`."
function solar_declination(d::Real)
    return 23.44 * sin(2π * (284 + d) / 365)
end

"""
    solar_zenith(lat_deg, day, hour) -> cos(zenith)

Return `max(0, cos(z))` where `z` is the solar zenith angle at
latitude `lat_deg`, day-of-year `day`, local solar time `hour` (0..24).
"""
function solar_zenith(lat_deg::Real, day::Real, hour::Real)
    δ   = deg2rad(solar_declination(day))
    lat = deg2rad(lat_deg)
    H   = deg2rad(15 * (hour - 12))
    μ   = sin(lat) * sin(δ) + cos(lat) * cos(δ) * cos(H)
    return max(0.0, μ)
end

"""
    solar_elev_azim(lat_deg, day, hour) -> (elev_rad, azim_rad)

Solar elevation and azimuth (radians). Elevation `= π/2 - zenith`.
Azimuth is measured from **north**, clockwise, positive to the east
(so noon azimuth ≈ 0 in the northern hemisphere before June solstice
and π after; use `atan2` form for numerical stability).
"""
function solar_elev_azim(lat_deg::Real, day::Real, hour::Real)
    δ   = deg2rad(solar_declination(day))
    lat = deg2rad(lat_deg)
    H   = deg2rad(15 * (hour - 12))
    sin_e = sin(lat)*sin(δ) + cos(lat)*cos(δ)*cos(H)
    elev  = asin(clamp(sin_e, -1, 1))
    cos_e = cos(elev)
    # atan2-based azimuth measured clockwise from north
    sin_a = -sin(H) * cos(δ) / max(cos_e, 1e-12)
    cos_a = (sin(δ) - sin_e * sin(lat)) / max(cos_e * cos(lat), 1e-12)
    azim  = atan(sin_a, cos_a)      # in (-π, π]
    return (elev, azim)
end

"""
    sun_direction(elev_rad, azim_rad) -> (dx, dy, dz)

Unit vector pointing from the ground *toward* the sun in (east, north, up)
Cartesian coordinates. `dz > 0` when the sun is above the horizon.
"""
function sun_direction(elev::Real, azim::Real)
    ce = cos(elev); se = sin(elev)
    dx = ce * sin(azim)     # east
    dy = ce * cos(azim)     # north
    dz = se                 # up
    return (dx, dy, dz)
end

"Direct-beam SW at surface [W/m²], reduced by an air-mass factor."
function sw_direct(μ::Real, sw_top::Real; τ_air::Real = 0.2)
    μ <= 0 && return 0.0
    return sw_top * μ * exp(-τ_air / μ)
end

# ---------------------------------------------------------------------------
# Surface temperature state (2-D array)
# ---------------------------------------------------------------------------

"""
    init_surface_state(sfc, T0) -> Ts

Initialize a `(Nx, Ny)` surface-temperature array with a uniform initial
temperature.
"""
init_surface_state(sfc, T0::Real) = fill(Float64(T0), size(sfc))

const STEFAN = 5.67e-8

"""
    step_surfaces!(Ts, sfc, params, sw_in_arr, lw_in, sensible_arr, dt)

Advance the surface temperature array by `dt`. Each cell's balance:

    C_s dT/dt = (1 - α) SW↓ + ε LW↓ - ε σ T⁴ - H

  * `sw_in_arr::AbstractMatrix` — SW at each cell (allows shading /
    orientation effects; here we pass a spatially uniform value or a
    solar-angle-modulated matrix).
  * `lw_in::Real` — atmospheric downward LW (uniform).
  * `sensible_arr::AbstractMatrix` — sensible heat flux from surface
    into air [W/m²] (positive means surface loses energy).
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

# ---------------------------------------------------------------------------
# Surface → atmosphere flux injection
# ---------------------------------------------------------------------------

"""
    apply_surface_flux!(sim, Ts, sfc, params; ρcp = 1005.0, U_char = 3.0)

Inject a horizontal-cell-dependent sensible heat flux into the first
interior z-layer of `sim.θ`. Uses a simple bulk exchange law

    H = ρcp · Cₕ · U · (Tₛ - θ_air)

with a class-dependent transfer coefficient `Cₕ = 0.005`. Returns the
2-D flux array so it can be fed into `step_surfaces!` on the next
substep.
"""
function apply_surface_flux!(sim, Ts::AbstractMatrix, sfc::AbstractMatrix{UInt8},
                             params::SurfaceParams;
                             ρcp::Real = 1005.0, U_char::Real = 3.0,
                             C_h::Real = 5e-3, dt::Real = 1.0,
                             H_cap::Real = 400.0)
    Nx, Ny = size(Ts)
    Hx, Hy, Hz = sim.mesh.Hx, sim.mesh.Hy, sim.mesh.Hz
    dz = sim.mesh.dz
    sensible = zeros(Float64, Nx, Ny)
    @inbounds for j in 1:Ny, i in 1:Nx
        θ_air = sim.θ.data[Hx+i, Hy+j, Hz+1]
        H_flx = ρcp * C_h * U_char * (Ts[i, j] - θ_air)
        # Cap the flux to keep the coupled system away from stiff regimes.
        H_flx = clamp(H_flx, -H_cap, H_cap)
        Fθ = H_flx / ρcp
        sensible[i, j] = H_flx
        sim.θ.data[Hx+i, Hy+j, Hz+1] += Fθ * dt / dz
    end
    return sensible
end

# ---------------------------------------------------------------------------
# Background geostrophic wind — Rayleigh nudging on horizontally-avg u, v
# ---------------------------------------------------------------------------

"""
    add_geostrophic_wind!(sim, U_g, V_g, τ_relax)

Add `-(⟨u⟩ - U_g)/τ` to every u-cell's tendency (and similarly for v)
so the horizontal mean wind is pulled toward the geostrophic value
`U_g` on a time scale `τ_relax`. Preserves turbulent fluctuations
because only the horizontal mean is nudged.

Call this from inside a `body_force!` callback passed to `step!`.
"""
function add_geostrophic_wind!(sim, U_g::Real, V_g::Real, τ_relax::Real)
    Nx, Ny = sim.mesh.Nx, sim.mesh.Ny
    Hx, Hy, Hz = sim.mesh.Hx, sim.mesh.Hy, sim.mesh.Hz
    Nz = sim.mesh.Nz
    invτ = 1 / τ_relax
    @inbounds for k in 1:Nz
        ū = 0.0; v̄ = 0.0
        for j in 1:Ny, i in 1:Nx
            ū += sim.u.data[Hx+i, Hy+j, Hz+k]
            v̄ += sim.v.data[Hx+i, Hy+j, Hz+k]
        end
        ū /= (Nx * Ny); v̄ /= (Nx * Ny)
        for j in 1:Ny, i in 1:Nx
            sim.tu.data[Hx+i, Hy+j, Hz+k] -= invτ * (ū - U_g)
            sim.tv.data[Hx+i, Hy+j, Hz+k] -= invτ * (v̄ - V_g)
        end
    end
    return sim
end

# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

"""
    compute_shadow_mask!(shade, sm, mesh, sun_dir)

Populate `shade::Matrix{Bool}` (size `(Nx, Ny)`) so that `shade[i, j]`
is `true` when the ground cell `(i, j)` is in the shadow of a solid
cell along the direction `sun_dir = (dx, dy, dz)` (unit vector,
pointing toward the sun).

Uses a per-cell 3-D ray march: from the top of ground cell `(i, j, 1)`
step along `sun_dir` at cell-diagonal resolution until either the
ray exits the domain top or hits a solid cell.

If the sun is below the horizon (`dz ≤ 0`), the whole map is set
`true` (shadow everywhere → no direct SW).
"""
function compute_shadow_mask!(shade::AbstractMatrix{Bool},
                              sm, mesh, sun_dir::NTuple{3, <:Real})
    dx_sun, dy_sun, dz_sun = sun_dir
    Nx, Ny, Nz = mesh.Nx, mesh.Ny, mesh.Nz
    Hx, Hy, Hz = mesh.Hx, mesh.Hy, mesh.Hz
    dx, dy, dz = mesh.dx, mesh.dy, mesh.dz

    if dz_sun <= 1e-6
        fill!(shade, true)
        return shade
    end

    # Step size: half a cell diagonal for accuracy.
    step_len = 0.5 * min(dx, dy, dz)
    sx = dx_sun * step_len
    sy = dy_sun * step_len
    sz = dz_sun * step_len

    # Maximum ray length: worst case = domain diagonal.
    max_len = sqrt((mesh.Lx)^2 + (mesh.Ly)^2 + (mesh.Lz)^2)
    max_steps = ceil(Int, max_len / step_len) + 4

    @inbounds for j in 1:Ny, i in 1:Nx
        # Ray origin: top of the highest solid cell in this column
        # (so roof cells start above the building, not inside it).
        k_top = 0
        for k in Nz:-1:1
            if sm.mask[Hx + i, Hy + j, Hz + k]; k_top = k; break; end
        end
        z_start = (k_top + 1) * dz + 0.5 * dz
        x = (i - 0.5) * dx
        y = (j - 0.5) * dy
        z = z_start
        blocked = false
        for _ in 1:max_steps
            x += sx; y += sy; z += sz
            (z >= mesh.Lz) && break
            xp = mod(x, mesh.Lx)
            yp = mod(y, mesh.Ly)
            i2 = clamp(Int(floor(xp / dx)) + 1, 1, Nx)
            j2 = clamp(Int(floor(yp / dy)) + 1, 1, Ny)
            k2 = clamp(Int(floor(z  / dz)) + 1, 1, Nz)
            if sm.mask[Hx + i2, Hy + j2, Hz + k2]
                blocked = true; break
            end
        end
        shade[i, j] = blocked
    end
    return shade
end

"""
    compute_sky_view_factor(sm, mesh; n_zenith = 6, n_azim = 24)

Return `svf::Matrix{Float64}` of size `(Nx, Ny)` giving the fraction of
the upper hemisphere unobstructed by solids, per ground cell. Uses
Tregenza-style sampling: `n_zenith` × `n_azim` rays over the hemisphere,
weighted by the cosine of the zenith angle so the returned value
approximates `∫ cos(z)/π · V(z, φ) dΩ` — the geometric weight the SVF
carries in the direct-diffuse SW and outgoing LW balances.
"""
function compute_sky_view_factor(sm, mesh; n_zenith::Int = 6, n_azim::Int = 24)
    Nx, Ny, Nz = mesh.Nx, mesh.Ny, mesh.Nz
    Hx, Hy, Hz = mesh.Hx, mesh.Hy, mesh.Hz
    dx, dy, dz = mesh.dx, mesh.dy, mesh.dz
    step_len = 0.5 * min(dx, dy, dz)
    max_steps = ceil(Int, sqrt(mesh.Lx^2 + mesh.Ly^2 + mesh.Lz^2) / step_len) + 4

    # Precompute the ray directions and their cos(z) weights.
    dirs   = Vector{NTuple{3, Float64}}()
    weights = Float64[]
    total_weight = 0.0
    for zi in 1:n_zenith
        # midpoint of each zenith band
        z_frac = (zi - 0.5) / n_zenith
        elev   = 0.5π * (1 - z_frac)     # elev from 0 (horizon) to π/2 (zenith)
        ce = cos(elev); se = sin(elev)
        w  = se                          # cos(zenith) = sin(elev)
        for ai in 1:n_azim
            φ = 2π * (ai - 0.5) / n_azim
            push!(dirs, (ce * cos(φ), ce * sin(φ), se))
            push!(weights, w)
            total_weight += w
        end
    end

    svf = Matrix{Float64}(undef, Nx, Ny)
    @inbounds for j in 1:Ny, i in 1:Nx
        # Same top-of-stack ray-origin trick as compute_shadow_mask!
        k_top = 0
        for k in Nz:-1:1
            if sm.mask[Hx + i, Hy + j, Hz + k]; k_top = k; break; end
        end
        z0 = (k_top + 1) * dz + 0.5 * dz
        weight_visible = 0.0
        for (di, (rx, ry, rz)) in enumerate(dirs)
            sx = rx * step_len; sy = ry * step_len; sz = rz * step_len
            x = (i - 0.5) * dx
            y = (j - 0.5) * dy
            z = z0
            blocked = false
            for _ in 1:max_steps
                x += sx; y += sy; z += sz
                (z >= mesh.Lz) && break
                xp = mod(x, mesh.Lx); yp = mod(y, mesh.Ly)
                i2 = clamp(Int(floor(xp / dx)) + 1, 1, Nx)
                j2 = clamp(Int(floor(yp / dy)) + 1, 1, Ny)
                k2 = clamp(Int(floor(z  / dz)) + 1, 1, Nz)
                if sm.mask[Hx + i2, Hy + j2, Hz + k2]
                    blocked = true; break
                end
            end
            blocked || (weight_visible += weights[di])
        end
        svf[i, j] = weight_visible / total_weight
    end
    return svf
end

# =========================================================================
# Post-Phase-20 physics — Batch A
# =========================================================================

# --- R3. Wall directional SW ---------------------------------------------

"""
    wall_directional_factor(sfc, sm, mesh, sun_dir) -> Matrix{Float64}

For each **wall-tagged** ground cell, compute a factor in `[0, 1]`
representing how much of the direct solar beam its neighbouring vertical
building face receives (dot product with the wall outward normal). For
non-wall cells the factor is `1.0`. Combined with the shadow mask this
gives a full azimuth-dependent SW on vertical building faces.
"""
function wall_directional_factor(sfc::AbstractMatrix{UInt8}, sm,
                                 mesh, sun_dir::NTuple{3, <:Real})
    dx_s, dy_s, _ = sun_dir
    Nx, Ny = size(sfc)
    Hx, Hy, Hz = mesh.Hx, mesh.Hy, mesh.Hz
    fac = ones(Float64, Nx, Ny)
    @inbounds for j in 1:Ny, i in 1:Nx
        sfc[i, j] == SURFACE_WALL || continue
        # Wall normal = average of neighbour-→-cell vectors from each
        # solid neighbour, then normalized.
        nx = 0.0; ny = 0.0
        for (di, dj) in ((-1,0), (1,0), (0,-1), (0,1))
            ni = mod1(i + di, Nx)
            nj = mod1(j + dj, Ny)
            has_solid = false
            for k in 1:mesh.Nz
                if sm.mask[Hx + ni, Hy + nj, Hz + k]; has_solid = true; break; end
            end
            if has_solid
                nx += di
                ny += dj
            end
        end
        # Normal points from building into fluid ⇒ opposite sign
        nx = -nx; ny = -ny
        mag = sqrt(nx*nx + ny*ny)
        if mag > 0
            nx /= mag; ny /= mag
            dot = nx * dx_s + ny * dy_s     # sun direction projected onto normal
            fac[i, j] = max(0.0, dot)       # walls only get SW on sunlit side
        end
    end
    return fac
end

# --- R5. Cloud shadow ---------------------------------------------------

"""
    apply_cloud_shadow!(sw_arr, ql_col_int, sw_direct_beam)

Attenuate the direct-beam SW component of `sw_arr` by the column-integrated
cloud water. Uses Beer's law with an extinction coefficient tuned so that
1 g m⁻² integrated ql gives ~50 % attenuation.

`ql_col_int::Matrix{Float64}` is the column integral of `ql` (kg m⁻²).
`sw_direct_beam` is the direct-beam value that would appear in a clear sky.
"""
function apply_cloud_shadow!(sw_arr::AbstractMatrix, ql_col_int::AbstractMatrix,
                             sw_direct_beam::Real; k_ext::Real = 700.0)
    Nx, Ny = size(sw_arr)
    @inbounds for j in 1:Ny, i in 1:Nx
        τ  = k_ext * ql_col_int[i, j]
        tr = exp(-τ)
        # Assume `sw_arr[i, j]` currently holds direct + diffuse.
        # Attenuate only the direct-beam portion; diffuse stays.
        # Extract the current direct component by rescaling:
        # sw_arr = direct*(1-shade) + diffuse. We approximate direct as
        # sw_direct_beam · (sw_arr - diffuse_floor) / sw_direct_beam. For a
        # demo cloud effect we just apply tr uniformly on the direct part.
        sw_arr[i, j] = sw_arr[i, j] * tr + sw_arr[i, j] * (1 - tr) * 0.15
    end
    return sw_arr
end

# --- L1. Latent heat / evapotranspiration --------------------------------

const _R_v      = 461.5     # J kg⁻¹ K⁻¹ (water-vapor gas constant)
const _R_d      = 287.05
const _L_v      = 2.45e6    # J kg⁻¹ latent heat of vaporization
const _P_ref    = 1.01325e5 # Pa
const _EPS_R    = _R_d / _R_v

"""
    q_saturation(T_K, p_Pa = 1.01325e5) -> kg/kg

Saturation specific humidity at temperature `T_K` (K) and pressure `p_Pa`
(Pa). Delegates to the Era IV thermodynamic contract
(`Aeolus.MoistThermodynamics.saturation_specific_humidity`) rather than a
local Tetens fit, so that evaporation here and dew in `apply_fog_dew!`
agree exactly on where saturation is — with two different fits (they
differ by up to 0.2 % at 305 K) there is a band of surface temperatures
where a cell both evaporates and collects dew in the same step.
"""
q_saturation(T_K::Real, p::Real = _P_ref) =
    Aeolus.MoistThermodynamics.saturation_specific_humidity(T_K, p)

# Per-class moisture-availability factor β ∈ [0, 1]: 1 means saturated
# surface (open water), 0 means desert. Streets, roofs ≈ dry after
# sunrise; parks moderate; water = 1.
function _moisture_availability(cls::UInt8)
    cls == SURFACE_WATER  ? 1.0 :
    cls == SURFACE_PARK   ? 0.6 :
    cls == SURFACE_WALL   ? 0.05 :
    cls == SURFACE_ROOF   ? 0.05 :
                            0.15    # street (asphalt after rain: some LE)
end

"""
    apply_latent_flux!(sim, Ts, sfc; ρ = 1.0, C_e = 1e-3, U_char = 3.0,
                        qv_air = 0.008, LE_cap = 400.0)

Adds an *evaporative-cooling* term to the surface energy balance and a
matching qv source to the first atmospheric layer:

    LE = ρ · L_v · C_e · U · β · max(0, q_sat(Ts) - qv_air)

`β` is the class-dependent moisture availability. Returns the 2-D
latent-flux matrix `[Nx, Ny]` (W/m²) so the caller can pass it into
`step_surfaces!` as an additional cooling term (subtract from Ts's
energy budget).
"""
function apply_latent_flux!(sim, Ts::AbstractMatrix,
                            sfc::AbstractMatrix{UInt8};
                            ρ::Real = 1.0, C_e::Real = 1e-3,
                            U_char::Real = 3.0,
                            qv_air::Real = 0.008,
                            LE_cap::Real = 400.0,
                            dt::Real = 1.0,
                            qv_field::Union{Nothing, Any} = nothing)
    Nx, Ny = size(Ts)
    Hx, Hy, Hz = sim.mesh.Hx, sim.mesh.Hy, sim.mesh.Hz
    dz = sim.mesh.dz
    LE = zeros(Float64, Nx, Ny)
    @inbounds for j in 1:Ny, i in 1:Nx
        c   = sfc[i, j]
        β   = _moisture_availability(c)
        qsat = q_saturation(Ts[i, j])
        qv_local = qv_field === nothing ? qv_air :
                   Aeolus.Fields.interior_view(qv_field)[i, j, 1]
        Δq  = max(0.0, qsat - qv_local)
        H_L = ρ * _L_v * C_e * U_char * β * Δq
        H_L = clamp(H_L, 0.0, LE_cap)
        LE[i, j] = H_L
        # Corresponding qv source at first layer:  dqv/dt += LE/(ρ L_v dz)
        if qv_field !== nothing
            Aeolus.Fields.interior_view(qv_field)[i, j, 1] +=
                dt * H_L / (ρ * _L_v * dz)
        end
    end
    return LE
end

# --- L4. Anthropogenic heat ---------------------------------------------
#
# Buildings: the output of the Phase 50 building energy model (issue #29),
# replacing the prescribed two-peak diurnal profile this file used to carry
# (class-dependent 2-40 W/m², with no source for the numbers). Traffic: a
# constant urban-area mean, see `add_traffic_heat!`.
#
# The slab surface scheme of this example has no building interior, so the
# city's buildings are represented by ONE city-mean `BuildingZone` whose
# volume, footprint and wall area are the scene's totals, run with the DOE
# medium-office schedules (#33). Its walls and roof are conduction columns
# that exchange with the canyon air through the ISO 6946 outside
# coefficient but absorb no sunshine: the slabs already absorb it, and a
# first version with sol-air walls counted it twice (-8.7 W/m² returned to
# the air in a one-hour run). A version that held the envelope's outer face
# at the slab skin temperature was tried next and dropped: that stiff
# interface, exchanged explicitly every 10 min, swung the slab budget by
# -64 then +27 W/m² (domain mean) between two radiation updates.
# The release is mapped onto the grid: HVAC exhaust and the roof envelope
# at the roofs (exhaust weighted by the floors above each roof cell); air
# exchange, glazing, lost gains and the wall envelope into the canyon (the
# SURFACE_WALL cells next to buildings). Sunshine transmitted by the
# windows is taken out of the wall cells' surface budget (`S`), which would
# otherwise absorb it too. Totals are conserved: Σ (H + LE) dx dy = the
# zone's direct release, Σ S dx dy = -Q_solar.
# `UrbanFacetScene` + `step_buildings!` resolve each building behind its own
# facets; switching this example to them is #18 / #30 scope.
#
# A BUG THE OLD CODE HAD: `add_anthropogenic_heat!` added Q_F to the
# `sensible` matrix, which `step_surfaces!` subtracts from the SURFACE
# budget, after `apply_surface_flux!` had already given the air its heat. So
# the prescribed anthropogenic heat never reached the air and instead cooled
# the surfaces by up to 40 W/m². The building release is now put into the
# first air layer directly (`apply_building_heat!`), and no longer touches
# `sensible`.

"""
    CityBuildings

State of the city-mean building of the example: the `BuildingZone`, its
`IdealLoadsHVAC`, `ThermostatSchedule` and `InternalGains`, and the maps
`H`, `LE` (W m⁻² of plan area) of the heat it put directly into the air
over the last building step, and `S` (≤ 0), the window-transmitted sun
taken from the wall cells' surface budget. See [`city_buildings`](@ref).
"""
mutable struct CityBuildings
    zone::BuildingZone{Float64}
    hvac::IdealLoadsHVAC
    thermostat::ThermostatSchedule
    gains::InternalGains
    floors::Matrix{Float64}          # floors above each column (0 off buildings)
    kair::Matrix{Int}                # first air layer above each column
    wall_cells::Vector{CartesianIndex{2}}
    H::Matrix{Float64}
    LE::Matrix{Float64}
    S::Matrix{Float64}
    release::Union{Nothing, BuildingHeatRelease{Float64}}
    hvac_budget::Union{Nothing, HVACBudget{Float64}}
end

"""
    city_buildings(heightmap, sfc, dx, dy, dz, wall_area; floor_height = 3.96,
                   glazing = 0.33, start_weekday = 4, T_air = 297.15) -> CityBuildings

The city-mean office building of the example. `wall_area` is the total
wall area (m², e.g. the sum over the radiosity scene's wall facets).

Assumptions, none of them a survey of Alexanderplatz: DOE medium-office
schedules, internal gains and thermostat (`office_internal_gains`,
`office_thermostat`); floor-to-floor height 3.96 m and window-to-wall
ratio 0.33, chosen to be close to the DOE medium-office reference
building (Deru et al. 2011) but not checked against it here; the illustrative
`UrbanSurface` example wall and roof; window U = 2.8 W m⁻² K⁻¹;
infiltration 0.3 h⁻¹; a packaged DX unit (COP 3.0) and a gas boiler.
`start_weekday = 4`: the run starts on a Thursday (20 June 2024).
"""
function city_buildings(heightmap::AbstractMatrix, sfc::AbstractMatrix{UInt8},
                        dx::Real, dy::Real, dz::Real, wall_area::Real;
                        floor_height::Real = 3.96, glazing::Real = 0.33,
                        start_weekday::Integer = 4, T_air::Real = 297.15)
    cell = dx * dy
    built = heightmap .> 0
    A_roof = count(built) * cell
    V = sum(heightmap[built]) * cell
    floor_area = V / floor_height
    # outdoor air through 1/R_se, no absorbed sun (the slabs take it)
    walls = EnvelopeElement(example_wall_layers(), (1 - glazing) * wall_area;
                            absorptance = 0.0, T = T_air, name = :walls)
    roof = EnvelopeElement(example_roof_layers(), A_roof; h_inside = 1 / 0.10,
                           absorptance = 0.0, T = T_air, name = :roof)
    slab = EnvelopeElement([WallLayer(:xps_insulation, 0.05), WallLayer(:concrete, 0.20)], A_roof;
                           exterior = :ground, T_ground = 285.15, h_inside = 1 / 0.17,
                           T = T_air, name = :slab)
    n_floors = V / A_roof / floor_height
    floors = InternalMass([WallLayer(:concrete, 0.20)], max(n_floors - 1, 0.01) * A_roof;
                          sides = 2, T = T_air, name = :floors)
    zone = BuildingZone(; volume = V, floor_area = floor_area,
                        envelope = AbstractZoneElement[walls, roof, slab],
                        internal_mass = [floors], window_area = glazing * wall_area,
                        window_U = 2.8, infiltration_ach = 0.3, T_air = T_air)
    hvac = IdealLoadsHVAC(cop_rated = 3.0, curves = DX_CURVES_PACKAGED_AC, heating = :boiler)
    fl = [built[I] ? heightmap[I] / floor_height : 0.0 for I in CartesianIndices(heightmap)]
    kair = [clamp(round(Int, heightmap[I] / dz) + 1, 1, typemax(Int)) for I in CartesianIndices(heightmap)]
    wc = [I for I in CartesianIndices(sfc) if sfc[I] == SURFACE_WALL]
    Nx, Ny = size(heightmap)
    return CityBuildings(zone, hvac, office_thermostat(; start_weekday),
                         office_internal_gains(floor_area; start_weekday), fl, kair, wc,
                         zeros(Nx, Ny), zeros(Nx, Ny), zeros(Nx, Ny), nothing, nothing)
end

"""
    step_city_buildings!(cb, t, dt, T_out, q_out; sw_wall = 0, shgc = 0.4)

Advance the city-mean building by `dt` seconds starting `t` seconds after
00:00 of day 1, against canyon air at `T_out` (K) and `q_out` (kg/kg),
with `sw_wall` (W m⁻²) the mean irradiance on the walls (for the
window-transmitted sun), and refresh `cb.H`, `cb.LE`, `cb.S`. `shgc` of
the glazing is an assumption.
"""
function step_city_buildings!(cb::CityBuildings, t::Real, dt::Real, T_out::Real, q_out::Real;
                              sw_wall::Real = 0.0, shgc::Real = 0.4)
    z = cb.zone
    # start from the outdoor humidity, so the first hours are not a
    # vapour exchange with an arbitrary initial indoor value
    cb.release === nothing && (z.q_air = q_out)
    r = internal_gains(cb.gains, t, dt; T_air = zone_air_temperature(z))
    f = ZoneForcing(r; T_out = T_out, q_out = q_out, Q_solar = shgc * z.window_area * sw_wall)
    hv = scheduled_hvac(cb.hvac, cb.thermostat, t)
    q0 = zone_specific_humidity(z)
    b, hb = step_zone!(z, f, dt, hv)
    rel = building_heat_release(z, f, b; dt = dt, gains = r, hvac = hv, hvac_budget = hb,
                                q_before = q0)
    cb.release = rel; cb.hvac_budget = hb
    # what the walls and roof give to the air (`Q_envelope_air` split)
    walls, roof = z.envelope[1], z.envelope[2]
    Q_wall = -walls.area * walls.F_outer
    Q_roof = -roof.area * roof.F_outer
    fill!(cb.H, 0.0); fill!(cb.LE, 0.0); fill!(cb.S, 0.0)
    cell_floors = sum(cb.floors)
    nroof = count(>(0), cb.floors)
    Nx, Ny = size(cb.H)
    cell_area = roof.area / nroof
    for I in CartesianIndices(cb.floors)
        cb.floors[I] > 0 || continue
        w = cb.floors[I] / cell_floors
        cb.H[I] += (w * rel.Q_exhaust_sensible + Q_roof / nroof) / cell_area
        cb.LE[I] += w * rel.Q_exhaust_latent / cell_area
    end
    canyon_H = rel.Q_ventilation_sensible + rel.Q_windows + rel.Q_lost + Q_wall
    canyon_LE = rel.Q_ventilation_latent + rel.Q_latent_vented
    nw = length(cb.wall_cells)
    for I in cb.wall_cells
        cb.H[I] += canyon_H / nw / cell_area
        cb.LE[I] += canyon_LE / nw / cell_area
        cb.S[I] -= rel.Q_solar / nw / cell_area
    end
    return cb
end

"""
    apply_building_heat!(sim, qv_field, sensible, cb, dt; ρcp = 1005.0, ρ = 1.0)

Put the city-mean building's direct release into the first air layer of
each column (above the roof for building columns): θ gets `H/ρcp`, the
vapour field `LE/(ρ L_v)`, per `dz`, over `dt`. `S` goes to the slab
surfaces: `sensible` is their loss term, so it is reduced by `S`.
"""
function apply_building_heat!(sim, qv_field, sensible::AbstractMatrix, cb::CityBuildings, dt::Real;
                              ρcp::Real = 1005.0, ρ::Real = 1.0)
    sensible .-= cb.S
    Hx, Hy, Hz = sim.mesh.Hx, sim.mesh.Hy, sim.mesh.Hz
    dz = sim.mesh.dz; Nz = sim.mesh.Nz
    qv = Aeolus.Fields.interior_view(qv_field)
    @inbounds for I in CartesianIndices(cb.H)
        i, j = Tuple(I)
        k = min(cb.kair[I], Nz)
        sim.θ.data[Hx+i, Hy+j, Hz+k] += cb.H[I] / ρcp * dt / dz
        qv[i, j, k] += cb.LE[I] / (ρ * _L_v) * dt / dz
    end
    return sim
end

"""
    add_traffic_heat!(sim, sfc, dt; Q_urban = 8.0, ρcp = 1005.0)

Vehicle heat: a constant `Q_urban` W m⁻² of urban area, released over the
road cells (streets and the building-adjacent cells). The default 8.0 W
m⁻² is the traffic heat Bueno et al. (2012, Geosci. Model Dev. 5, 433,
Table 3) prescribe for the Toulouse centre (their Table 4 uses the same
value for Athens); the tables give no source for it, and it is not a
Berlin inventory. It is not a building quantity and is not
from the building energy model.
"""
function add_traffic_heat!(sim, sfc::AbstractMatrix{UInt8}, dt::Real;
                           Q_urban::Real = 8.0, ρcp::Real = 1005.0)
    road = [I for I in CartesianIndices(sfc) if sfc[I] == SURFACE_STREET || sfc[I] == SURFACE_WALL]
    isempty(road) && return sim
    q = Q_urban * length(sfc) / length(road)
    Hx, Hy, Hz = sim.mesh.Hx, sim.mesh.Hy, sim.mesh.Hz
    for I in road
        i, j = Tuple(I)
        sim.θ.data[Hx+i, Hy+j, Hz+1] += q / ρcp * dt / sim.mesh.dz
    end
    return sim
end

# --- R4. Sharpened LW inter-reflection ----------------------------------
# (Existing radiation already blends `320 · svf + 380 · (1-svf)` — we
# refine by making the "wall LW" temperature-dependent on nearby Ts.)

"""
    lw_from_walls(Ts_mean, ε = 0.95) -> W/m²

Approximate longwave emitted by surrounding walls at temperature `Ts_mean`.
Used to weight the non-sky portion of downward LW inside a canyon.
"""
lw_from_walls(Ts_mean::Real, ε::Real = 0.95) = ε * STEFAN * Ts_mean^4

"Class-averaged surface temperature."
function mean_surface_temperature(Ts::AbstractMatrix, sfc::AbstractMatrix{UInt8},
                                  cls::Integer)
    s = 0.0; n = 0
    @inbounds for I in eachindex(Ts)
        if sfc[I] == cls; s += Ts[I]; n += 1; end
    end
    return n == 0 ? NaN : s / n
end
