# Column radiation: the domain-mean LES column, stitched onto a standard
# atmosphere above the domain, through the Era III radiation stack
# (RadiationCore columns, GasOptics correlated-k gas optics, TwoStream
# solver and Mie cloud optics) at the radiation cadence.
#
# What it gives the case:
#   * the sky longwave at the surface, `bus.L_sky`, for the radiosity
#     solve (replaces `lw_from_walls`, which took 0.95 σ T̄s⁴ of the
#     surfaces as the sky);
#   * the direct / diffuse shortwave for the radiosity solve, `bus.dni`,
#     `bus.dhi` (option `surface_sw = :bird` leaves them to the radiosity
#     coupling's Bird clear sky);
#   * the radiative heating profile of the LES layers, applied to θ every
#     step (replaces the prescribed −2 K/day of `cooling_prescribed`).
#
# The surface fluxes come from the column WITHOUT the LES condensate:
# the 3-D cloud shadow (`cloud_shadow.jl`) attenuates the direct beam per
# column, so taking the fog out of the column here is what keeps it from
# being counted twice. The heating profile and the sky longwave come from
# the column WITH it (the domain-mean fog absorbs, emits and cools at its
# top). See docs/src/design/berlin_surface_radiation.md.
#
# The column's radiation itself (the stitched column, the optics, the
# solves and the heating profile) is `Aeolus.ColumnRadiation` (moved there
# from this file in ONE-e, #454), which the nest parent uses too; this file
# gathers the LES column and the surface, and hands the results to the bus.

"""
    ColumnRadiation

Slot `:column_radiation`, provider `:twostream` (default): the
`ColumnRadiationModel` `m` of `Aeolus.ColumnRadiation` on the case's
domain-mean column, and `ghi_bird`, the Bird clear-sky GHI of the
radiosity coupling's sky (for the log).

At the `:radiation` cadence (and step 1), before the radiosity solve:

1. the fluid-cell mean profiles of θ, qv and ql on the LES levels become
   a hydrostatic column (`column_from_profiles`, surface pressure
   `surface_pressure(case)` of the reference state), which is stitched (`stitch_columns`) under
   `us_standard_column` (`nz_upper` layers to 30 km, its surface
   temperature chosen so that it matches the LES top temperature, the
   humidity profile of the scenario's season);
2. the surface is the domain: radiative temperature `(Σε Ts⁴ / Σε)^{1/4}`
   of the columns, mean emissivity, and the effective albedo of the last
   radiosity solve (class-mean albedo before the first one);
3. correlated-k gas optics (`correlated_k_optics`) plus Mie cloud optics
   (`mie_cloud_optics`, `r_eff_liq`) → `solve_lw!` and, with the sun up,
   `solve_sw!`, scaled by the Earth–Sun distance factor; all through the
   multi-column forms (KA-n, #305: optics `τ[k, g, column]`, one work item
   per layer for the optics and per g-point for the solves) on one column;
4. `bus.L_sky` = downwelling longwave at the surface; `bus.dni`,
   `bus.dhi` = direct-normal and diffuse shortwave at the surface of the
   same column with the LES condensate removed; `heat` = the LES-layer
   heating rate (K/s) from the net-flux divergence with cloud.

Every step (`:column_tendency`) θ += `heat`·dt/Π̄_k on every LES level
(`heat` is dT/dt and T = θΠ̄; the same 1/(ρ̄c_pΠ̄) as `ref_θ_per_heat`
with ρ̄c_p already divided out), in the fluid cells only (`Radiation.apply_radiative_cooling!` with the
solid mask): the heating rate is a property of the air, and heating the
immersed buildings' cells leaks into the air next to the walls.
"""
mutable struct ColumnRadiation{M<:ColumnRadiationModel} <: UrbanComponent
    m::M                          # the column's radiation (Aeolus.ColumnRadiation)
    ghi_bird::Float64             # Bird GHI with the radiosity coupling's sky, for the log
end

component_name(::Type{<:ColumnRadiation}) = :column_radiation
slot(::Type{<:ColumnRadiation}) = :column_radiation
provider(::Type{<:ColumnRadiation}) = :twostream
modules_used(::Type{<:ColumnRadiation}) =
    (:RadiationCore, :GasOptics, :TwoStream, :SolarGeometry, :Radiation)
default_options(::Type{<:ColumnRadiation}) =
    (nz_upper = 60, r_eff_liq = 10.0, surface_sw = :twostream, p_surf = nothing)
stages(::Type{<:ColumnRadiation}) = ((:radiation, 5, Cadence(:radiation; first = true)),
                                    (:column_tendency, 10, EVERY_STEP))
smoke_variants(::Type{<:ColumnRadiation}) =
    ((label = "surface shortwave from Bird", options = Dict(:column_radiation => (surface_sw = :bird,))),)

function build(::Type{<:ColumnRadiation}, case, o)
    # the case's surface pressure (the reference state, #131), unless overridden
    ps = o.p_surf === nothing ? surface_pressure(case) : Float64(o.p_surf)
    return ColumnRadiation(column_radiation_solver(case.mesh.Nz, ps; nz_upper = o.nz_upper,
                                                   r_eff_liq = o.r_eff_liq, surface_sw = o.surface_sw),
                           NaN)
end

"""
    fluid_mean_profile(case, f) -> Vector

Mean of the cell-centred field `f` over the fluid (non-solid) cells of
every LES level.
"""
function fluid_mean_profile(case, f)
    mask, m = case.geom.sm.mask, case.mesh
    Hx, Hy, Hz = m.Hx, m.Hy, m.Hz
    return _fluid_mean(interior_view(f), mask, Hx, Hy, Hz)
end

function _fluid_mean(a, mask, Hx, Hy, Hz)
    Nx, Ny, Nz = size(a)
    # the masked sums and counts per level in the canonical, thread-invariant
    # plane order (KA-b, #296)
    S, n = zeros(Nz), zeros(Nz)
    plane_reduce!((S, n), Aeolus.Backends.backend(a), _fluid_value_at, (1:Nx, 1:Ny, 1:Nz),
                  a, mask, Hx, Hy, Hz)
    return [n[k] > 0 ? S[k] / n[k] : NaN for k in 1:Nz]
end

@inline function _fluid_value_at(i, j, k, a, mask, Hx, Hy, Hz)
    @inbounds mask[Hx + i, Hy + j, Hz + k] && return (0.0, 0.0)
    return (Float64(@inbounds a[i, j, k]), 1.0)
end

"""
    les_column(c, case; condensate = true) -> AtmosphericColumn

The stitched radiation column of the case now: fluid-mean LES profiles
on the LES levels (with or without their `ql`) under a standard
atmosphere, with the domain surface's radiative temperature.
"""
function les_column(c::ColumnRadiation, case; condensate::Bool = true)
    m, bus, sfc = case.mesh, case.bus, case.geom.sfc
    θ  = fluid_mean_profile(case, case.sim.θ)
    qv = fluid_mean_profile(case, tracer(case, :qv))
    ql = condensate ? max.(fluid_mean_profile(case, tracer(case, :ql)), 0.0) : zeros(m.Nz)
    ε = bus.params.emissivity
    num = 0.0; den = 0.0
    @inbounds for I in eachindex(bus.Ts)
        e = ε[sfc[I]]
        num += e * bus.Ts[I]^4; den += e
    end
    T_rad = (num / den)^(1 / 4)
    return stitched_column(c.m, θ, qv, ql, fill(Float64(m.dz), m.Nz); T_rad = T_rad,
                           humid = season_humidity(case.cfg))
end

"The humidity profile of the standard atmosphere above the column for the scenario's season."
season_humidity(cfg) = cfg.scenario === :winter ? :subarctic_winter : :midlatitude_summer

"Domain-mean surface emissivity and shortwave albedo for the column."
function column_surface(case)
    bus, sfc = case.bus, case.geom.sfc
    ε = sum(bus.params.emissivity[s] for s in sfc) / length(sfc)
    α = bus.swsol === nothing ? sum(bus.params.albedo[s] for s in sfc) / length(sfc) :
                                bus.swsol.effective_albedo
    return ε, α
end

function run_stage!(c::ColumnRadiation, case, ::Val{:radiation}, dt)
    bus = case.bus
    sp = solar_position_at(case.cfg, case.clock.hour)
    ε, α = column_surface(case)
    column_radiation!(c.m, les_column(c, case), sp, ε, α;
                      clear = () -> les_column(c, case; condensate = false))
    fcl = c.m.fsw_clear
    if fcl !== nothing
        dir = fcl.flux_dn_dir[1]
        if c.m.surface_sw === :twostream
            bus.dni = dir / c.m.μ₀
            bus.dhi = fcl.flux_dn[1] - dir
        else
            bus.dni = NaN; bus.dhi = NaN
        end
        rs = find_coupling(case, :radiosity_sun)
        c.ghi_bird = rs === nothing ? NaN : clear_sky_irradiance(sp, rs.sky).ghi
    else
        bus.dni = c.m.surface_sw === :twostream ? 0.0 : NaN
        bus.dhi = c.m.surface_sw === :twostream ? 0.0 : NaN
        c.ghi_bird = 0.0
    end
    bus.L_sky = c.m.flw.flux_dn[1]
    return nothing
end

# the heating is dT/dt; θ = T/Π̄ gets it divided by Π̄_k (#253)
function run_stage!(c::ColumnRadiation, case, ::Val{:column_tendency}, dt)
    c.m.buf .= c.m.heat .* dt ./ case.ref.π̄
    apply_radiative_cooling!(case.sim.θ, c.m.buf; solid = case.geom.sm.mask)
    c.m.dt_applied = dt
    return nothing
end

# Apply the stage's increment to a copy of θ and compare what changed
# with heating·dt/Π̄: exactly that in every fluid cell, nothing in a solid
# one. (Measured on the field, so it tests the masked application, not
# the bookkeeping.)
function _applied_increment_ok(c::ColumnRadiation, case)
    θ = deepcopy(case.sim.θ)
    dt = c.m.dt_applied > 0 ? c.m.dt_applied : 1.0
    c.m.buf .= c.m.heat .* dt ./ case.ref.π̄
    apply_radiative_cooling!(θ, c.m.buf; solid = case.geom.sm.mask)
    m, mask = case.mesh, case.geom.sm.mask
    a, a0 = interior_view(θ), interior_view(case.sim.θ)
    ok = true
    @inbounds for k in 1:m.Nz, j in 1:m.Ny, i in 1:m.Nx
        d = a[i, j, k] - a0[i, j, k]
        if mask[m.Hx + i, m.Hy + j, m.Hz + k]
            ok &= d == 0                                  # untouched
        else
            ok &= abs(d - c.m.heat[k] * dt / case.ref.π̄[k]) <= 1e-12   # ≈ 20 ulp of θ ≈ 300 K
        end
    end
    return ok
end

"""
    diagnostics(c::ColumnRadiation, case)

The sky longwave on the surface `L_sky`, the clear-sky global horizontal
irradiance of the column `ghi` and of the Bird model `ghi_bird` (W m⁻²),
and the LES-layer mean heating `heating`, `heating_lw`, `heating_sw`
(K day⁻¹, `ColumnRadiation.column_heating_kday`). `NaN` before the first radiation
call.
"""
function diagnostics(c::ColumnRadiation, case)
    h = per_state(() -> c.m.col === nothing ? (NaN, NaN, NaN) : column_heating_kday(c.m), case)
    return Diagnostic[
        Diagnostic("L_sky", "W m-2", () -> c.m.col === nothing ? NaN : case.bus.L_sky),
        Diagnostic("ghi", "W m-2", () -> c.m.col === nothing ? NaN :
                                         c.m.fsw_clear === nothing ? 0.0 : c.m.fsw_clear.flux_dn[1]),
        Diagnostic("ghi_bird", "W m-2", () -> c.m.col === nothing ? NaN : c.ghi_bird),
        Diagnostic("heating", "K day-1", () -> h()[1]),
        Diagnostic("heating_lw", "K day-1", () -> h()[2]),
        Diagnostic("heating_sw", "K day-1", () -> h()[3])]
end

function smoke_check(c::ColumnRadiation, case)
    col, Nz = c.m.col, case.mesh.Nz
    cp = Aeolus.RadiationCore.CP_DRY_AIR
    # ∫ ρ c_p (dT/dt) dz over the LES layers is the net flux into them
    ΔF = c.m.flw.flux_net[Nz + 1] - c.m.flw.flux_net[1]
    c.m.fsw === nothing || (ΔF += c.m.fsw.flux_net[Nz + 1] - c.m.fsw.flux_net[1])
    Q = sum(col.ρ[k] * cp * c.m.heat[k] * col.dz[k] for k in 1:Nz)
    σ = STEFAN_BOLTZMANN
    Tmax = max(maximum(col.T_lev), maximum(col.T_lay))
    checks = [
        "LES-layer heating ∫ρc_p Ḣ dz = net flux convergence (rel. 1e-10)" =>
            isapprox(Q, ΔF; rtol = 1e-10, atol = 1e-10),
        "θ increment applied = heating·dt/Π̄ in every fluid cell, 0 in the solid ones" =>
            _applied_increment_ok(c, case),
        "0 < L_sky < σ T_max⁴ of the column" => 0 < case.bus.L_sky < σ * Tmax^4,
        "sky longwave on the bus is the column's" => case.bus.L_sky == c.m.flw.flux_dn[1],
        "radiation ran" => c.m.ncalls > 0]
    if c.m.fsw !== nothing
        sp = c.m.sun
        # the column receives S₀ (distance-corrected) μ₀ at its top
        push!(checks, "TOA insolation = S₀ f μ₀ (rel. 1e-10)" =>
              isapprox(c.m.fsw.flux_dn[end], SOLAR_CONSTANT * earth_sun_distance_factor(sp) * c.m.μ₀; rtol = 1e-10))
        push!(checks, "surface direct ≤ global ≤ TOA" =>
              0 <= c.m.fsw_clear.flux_dn_dir[1] <= c.m.fsw_clear.flux_dn[1] <= c.m.fsw_clear.flux_dn[end])
        push!(checks, "condensate does not add surface direct beam" =>
              c.m.fsw.flux_dn_dir[1] <= c.m.fsw_clear.flux_dn_dir[1] * (1 + 1e-12))
        # an independent clear-sky model: Bird with no aerosol (the column
        # has none) and the column's precipitable water, ozone and albedo.
        # Measured: −0.4 % at μ₀ = 0.86, −3.1 % at 0.58, −4.6 % at 0.27
        # (TwoStream falls below Bird as the sun gets low), so 5 % is
        # asserted for a high sun (the issue's noon criterion) and 10 %
        # below μ₀ = 0.5.
        pw = sum(col.vmr[:h2o][k] * 18.01528 / 28.9647 * col.ρ[k] * col.dz[k] for k in eachindex(col.dz))
        o3 = sum(col.vmr[:o3][k] * 47.9982 / 28.9647 * col.ρ[k] * col.dz[k] for k in eachindex(col.dz)) /
             2.1415e-2                                       # kg/m² → atm-cm (1 atm-cm = 2.1415e-2 kg/m²)
        bird = clear_sky_irradiance(sp, BirdClearSky(; aod380 = 0.0, aod500 = 0.0, ozone = o3,
                                    precipitable_water = pw / 10, albedo = column_surface(case)[2])).ghi
        tol = c.m.μ₀ >= 0.5 ? 0.05 : 0.10
        push!(checks, "clear-LES surface GHI within $(round(Int, 100tol)) % of an aerosol-free Bird sky" =>
              abs(c.m.fsw_clear.flux_dn[1] / bird - 1) < tol)
    end
    return checks
end

