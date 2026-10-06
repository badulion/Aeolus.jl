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

"""
    ColumnRadiation

Slot `:column_radiation`, provider `:twostream` (default).

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
mutable struct ColumnRadiation{GS, GL, CS, CL, SS, SL} <: AbstractCoupling
    gas_sw::GS
    gas_lw::GL
    cloud_sw::CS
    cloud_lw::CL
    solver_sw::SS
    solver_lw::SL
    nz_upper::Int
    r_eff_liq::Float64
    p_surf::Float64
    surface_sw::Symbol
    col::Any                      # last stitched column (with the LES condensate)
    fsw::Any                      # its fluxes (sw: nothing at night)
    flw::Any
    fsw_clear::Any                # shortwave fluxes without the LES condensate
    heat::Vector{Float64}         # K/s on the LES levels, lw + sw
    heat_lw::Vector{Float64}
    heat_sw::Vector{Float64}
    buf::Vector{Float64}
    μ₀::Float64
    sun::Any                      # solar position of the last radiation call
    ghi_bird::Float64             # Bird GHI with the radiosity coupling's sky, for the log
    dt_applied::Float64           # dt of the last :column_tendency
    ncalls::Int
    ws_sw::TwoStreamWorkspace{Float64}   # scratch of the multi-column solves (KA-n, #305)
    ws_lw::TwoStreamWorkspace{Float64}
end

coupling_name(::Type{<:ColumnRadiation}) = :column_radiation
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
    return column_radiation_solver(case.mesh.Nz, ps; nz_upper = o.nz_upper,
                                   r_eff_liq = o.r_eff_liq, surface_sw = o.surface_sw)
end

"""
    column_radiation_solver(Nz, p_surf; nz_upper = 60, r_eff_liq = 10, surface_sw = :twostream)
        -> ColumnRadiation

The column-radiation state for a model with `Nz` levels and ground
pressure `p_surf` (Pa): correlated-k gas optics, Mie cloud optics and the
two-stream solvers of the shortwave and longwave bands. The one
constructor: the `:column_radiation` coupling of the Berlin child builds
it from the case, the nesting parent (`parent/case.jl`, #259) from its
own grid, and both then call [`column_radiation!`](@ref).
"""
function column_radiation_solver(Nz::Integer, p_surf::Real; nz_upper::Integer = 60,
                                 r_eff_liq::Real = 10.0, surface_sw::Symbol = :twostream)
    surface_sw in (:twostream, :bird) ||
        error("column_radiation: surface_sw must be :twostream or :bird, got $surface_sw")
    sw, lw = shortwave_bands(), longwave_bands()
    return ColumnRadiation(correlated_k_optics(sw), correlated_k_optics(lw),
                           mie_cloud_optics(sw), mie_cloud_optics(lw),
                           two_stream_solver(sw), two_stream_solver(lw),
                           Int(nz_upper), Float64(r_eff_liq), Float64(p_surf), surface_sw,
                           nothing, nothing, nothing, nothing,
                           zeros(Nz), zeros(Nz), zeros(Nz), zeros(Nz),
                           0.0, nothing, NaN, 0.0, 0,
                           TwoStreamWorkspace{Float64}(), TwoStreamWorkspace{Float64}())
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
    return stitched_column(c, θ, qv, ql, fill(Float64(m.dz), m.Nz); T_rad = T_rad,
                           humid = season_humidity(case.cfg))
end

"The humidity profile of the standard atmosphere above the column for the scenario's season."
season_humidity(cfg) = cfg.scenario === :winter ? :subarctic_winter : :midlatitude_summer

"""
    stitched_column(c, θ, qv, ql, dz; T_rad, humid = :midlatitude_summer) -> AtmosphericColumn

The radiation column of a model's mean profiles: `θ`, `qv`, `ql` (K,
kg kg⁻¹) on layers of thickness `dz` (m, a vector: uniform for the child,
stretched for the parent), hydrostatic from `c.p_surf`
(`column_from_profiles`), with the surface radiative temperature `T_rad`,
under `us_standard_column` (`c.nz_upper` layers to 30 km, its surface
temperature chosen so that its lapse rate continues from the model top
temperature; humidity profile `humid`).
"""
function stitched_column(c::ColumnRadiation, θ, qv, ql, dz::AbstractVector; T_rad::Real,
                         humid::Symbol = :midlatitude_summer)
    low = column_from_profiles(θ, dz; p_surf = c.p_surf, qv = max.(qv, 0.0), ql = ql,
                               T_sfc = T_rad, r_eff_liq = c.r_eff_liq)
    # the standard atmosphere above, with its lapse rate continued from the model top
    T_top = low.T_lev[end]
    up = us_standard_column(; nz = c.nz_upper, z_top = 30_000.0,
                            T_sfc = T_top + 6.5e-3 * sum(dz), humid = humid)
    return stitch_columns(low, up)
end

"Domain-mean surface emissivity and shortwave albedo for the column."
function column_surface(case)
    bus, sfc = case.bus, case.geom.sfc
    ε = sum(bus.params.emissivity[s] for s in sfc) / length(sfc)
    α = bus.swsol === nothing ? sum(bus.params.albedo[s] for s in sfc) / length(sfc) :
                                bus.swsol.effective_albedo
    return ε, α
end

# The optics of the columns `cols` in the multi-column layout τ[k, g, c]
# (KA-n, #305): correlated-k gas optics, plus the Mie cloud optics when a
# column has condensate. The case passes one column (the domain or plane
# mean); a column's bits are those of the single-column optics.
function _optics(gas, cloud, cols)
    length(cols) == 1 || error("column_radiation: _optics combines the cloud optics of one column")
    spec = gas.spec
    n = length(cols[1].dz)
    o = allocate_column_optics(Float64, n, length(cols), spec)
    compute_optics!(o, gas, cols)
    col = cols[1]
    if any(>(0), col.ql) || any(>(0), col.qi)
        oc = allocate_column_optics(Float64, n, length(cols), spec)
        compute_optics!(oc, cloud, cols)
        combine_optics!(o, oc)
    end
    return o
end

# The single-column fluxes of column 1 of multi-column fluxes (no copy).
_column_fluxes(f) = RadiativeFluxes{Float64,1}(vec(f.flux_up), vec(f.flux_dn),
                                               vec(f.flux_dn_dir), vec(f.flux_net))

function _solve_sw(c, col, μ, α, fdist)
    cols = [col]
    f = allocate_fluxes(Float64, (1,), length(col.dz) + 1)
    solve_sw!(f, c.solver_sw, _optics(c.gas_sw, c.cloud_sw, cols); μ₀ = μ,
              sfc_albedo_direct = α, workspace = c.ws_sw)
    f1 = _column_fluxes(f)
    for a in (f1.flux_up, f1.flux_dn, f1.flux_dn_dir, f1.flux_net)
        a .*= fdist
    end
    return f1
end

"""
    column_radiation!(c, col, sp, ε, α; clear = () -> col) -> c

One radiation call on the stitched column `col` ([`stitched_column`](@ref))
at the solar position `sp`, for a surface of emissivity `ε` and
shortwave albedo `α`: correlated-k optics (plus Mie cloud optics where
`col` has condensate), `solve_lw!` and, with the sun up, `solve_sw!`
scaled by the Earth–Sun distance factor. Fills `c.heat_lw`, `c.heat_sw`,
`c.heat` (K/s, dT/dt of the model layers, from the net-flux divergence),
`c.col`, `c.flw`, `c.fsw` (`nothing` at night), `c.fsw_clear` — the
shortwave of the column `clear()` returns (the column without the model's
condensate; only built when `col` has any) — `c.μ₀` and `c.sun`.

The one radiation call of the Berlin case: the child's `:radiation` stage
and the nesting parent (#259) both call it; the surface downwelling
longwave is `c.flw.flux_dn[1]` and the direct / diffuse shortwave are
`c.fsw_clear.flux_dn_dir[1]` and `c.fsw_clear.flux_dn[1] − dir`.
"""
function column_radiation!(c::ColumnRadiation, col, sp, ε::Real, α::Real; clear = () -> col)
    Nz = length(c.heat)
    μ = sp.elevation > 0 ? sp.μ₀ : 0.0
    cols = [col]
    flw2 = allocate_fluxes(Float64, (1,), length(col.dz) + 1)
    solve_lw!(flw2, c.solver_lw, _optics(c.gas_lw, c.cloud_lw, cols), cols; sfc_emissivity = ε,
              workspace = c.ws_lw)
    flw = _column_fluxes(flw2)
    heating_rate!(c.heat_lw, _les_levels(flw, Nz), col.dz[1:Nz], col.ρ[1:Nz])
    fdist = earth_sun_distance_factor(sp)
    if μ > 0
        fsw = _solve_sw(c, col, μ, α, fdist)
        heating_rate!(c.heat_sw, _les_levels(fsw, Nz), col.dz[1:Nz], col.ρ[1:Nz])
        # the surface beam without the model condensate (the 3-D shadow applies it)
        fcl = any(>(0), @view col.ql[1:Nz]) ? _solve_sw(c, clear(), μ, α, fdist) : fsw
    else
        fsw = nothing; fcl = nothing
        fill!(c.heat_sw, 0.0)
    end
    @. c.heat = c.heat_lw + c.heat_sw
    c.col, c.flw, c.fsw, c.fsw_clear, c.μ₀, c.sun = col, flw, fsw, fcl, μ, sp
    c.ncalls += 1
    return c
end

function run_stage!(c::ColumnRadiation, case, ::Val{:radiation}, dt)
    bus = case.bus
    sp = solar_position_at(case.cfg, case.clock.hour)
    ε, α = column_surface(case)
    column_radiation!(c, les_column(c, case), sp, ε, α;
                      clear = () -> les_column(c, case; condensate = false))
    fcl = c.fsw_clear
    if fcl !== nothing
        dir = fcl.flux_dn_dir[1]
        if c.surface_sw === :twostream
            bus.dni = dir / c.μ₀
            bus.dhi = fcl.flux_dn[1] - dir
        else
            bus.dni = NaN; bus.dhi = NaN
        end
        rs = find_coupling(case, :radiosity_sun)
        c.ghi_bird = rs === nothing ? NaN : clear_sky_irradiance(sp, rs.sky).ghi
    else
        bus.dni = c.surface_sw === :twostream ? 0.0 : NaN
        bus.dhi = c.surface_sw === :twostream ? 0.0 : NaN
        c.ghi_bird = 0.0
    end
    bus.L_sky = c.flw.flux_dn[1]
    return nothing
end

# The fluxes on the LES levels 1 … Nz+1 only (heating_rate! wants a
# RadiativeFluxes whose layers are the ones it fills).
function _les_levels(f, Nz)
    r = 1:Nz + 1
    return RadiativeFluxes{Float64,1}(f.flux_up[r], f.flux_dn[r], f.flux_dn_dir[r], f.flux_net[r])
end

# the heating is dT/dt; θ = T/Π̄ gets it divided by Π̄_k (#253)
function run_stage!(c::ColumnRadiation, case, ::Val{:column_tendency}, dt)
    c.buf .= c.heat .* dt ./ case.ref.π̄
    apply_radiative_cooling!(case.sim.θ, c.buf; solid = case.geom.sm.mask)
    c.dt_applied = dt
    return nothing
end

# Apply the stage's increment to a copy of θ and compare what changed
# with heating·dt/Π̄: exactly that in every fluid cell, nothing in a solid
# one. (Measured on the field, so it tests the masked application, not
# the bookkeeping.)
function _applied_increment_ok(c::ColumnRadiation, case)
    θ = deepcopy(case.sim.θ)
    dt = c.dt_applied > 0 ? c.dt_applied : 1.0
    c.buf .= c.heat .* dt ./ case.ref.π̄
    apply_radiative_cooling!(θ, c.buf; solid = case.geom.sm.mask)
    m, mask = case.mesh, case.geom.sm.mask
    a, a0 = interior_view(θ), interior_view(case.sim.θ)
    ok = true
    @inbounds for k in 1:m.Nz, j in 1:m.Ny, i in 1:m.Nx
        d = a[i, j, k] - a0[i, j, k]
        if mask[m.Hx + i, m.Hy + j, m.Hz + k]
            ok &= d == 0                                  # untouched
        else
            ok &= abs(d - c.heat[k] * dt / case.ref.π̄[k]) <= 1e-12   # ≈ 20 ulp of θ ≈ 300 K
        end
    end
    return ok
end

"LES-layer mean heating rate in K/day, ρ·dz-weighted: (total, longwave, shortwave)."
function les_heating_kday(c::ColumnRadiation, case)
    Nz = case.mesh.Nz
    w = [c.col.ρ[k] * c.col.dz[k] for k in 1:Nz]
    avg(h) = 86400 * sum(w .* h) / sum(w)
    return (avg(c.heat), avg(c.heat_lw), avg(c.heat_sw))
end

function report(c::ColumnRadiation, case)
    c.col === nothing && return nothing
    h, hl, hs = les_heating_kday(c, case)
    ghi = c.fsw_clear === nothing ? 0.0 : c.fsw_clear.flux_dn[1]
    return @sprintf("column radiation: L_sky %.1f W/m², GHI %.1f (Bird %.1f), heating %+.2f K/day (lw %+.2f, sw %+.2f)",
                    case.bus.L_sky, ghi, c.ghi_bird, h, hl, hs)
end

function smoke_check(c::ColumnRadiation, case)
    col, Nz = c.col, case.mesh.Nz
    cp = Aeolus.RadiationCore.CP_DRY_AIR
    # ∫ ρ c_p (dT/dt) dz over the LES layers is the net flux into them
    ΔF = c.flw.flux_net[Nz + 1] - c.flw.flux_net[1]
    c.fsw === nothing || (ΔF += c.fsw.flux_net[Nz + 1] - c.fsw.flux_net[1])
    Q = sum(col.ρ[k] * cp * c.heat[k] * col.dz[k] for k in 1:Nz)
    σ = STEFAN_BOLTZMANN
    Tmax = max(maximum(col.T_lev), maximum(col.T_lay))
    checks = [
        "LES-layer heating ∫ρc_p Ḣ dz = net flux convergence (rel. 1e-10)" =>
            isapprox(Q, ΔF; rtol = 1e-10, atol = 1e-10),
        "θ increment applied = heating·dt/Π̄ in every fluid cell, 0 in the solid ones" =>
            _applied_increment_ok(c, case),
        "0 < L_sky < σ T_max⁴ of the column" => 0 < case.bus.L_sky < σ * Tmax^4,
        "sky longwave on the bus is the column's" => case.bus.L_sky == c.flw.flux_dn[1],
        "radiation ran" => c.ncalls > 0]
    if c.fsw !== nothing
        sp = c.sun
        # the column receives S₀ (distance-corrected) μ₀ at its top
        push!(checks, "TOA insolation = S₀ f μ₀ (rel. 1e-10)" =>
              isapprox(c.fsw.flux_dn[end], SOLAR_CONSTANT * earth_sun_distance_factor(sp) * c.μ₀; rtol = 1e-10))
        push!(checks, "surface direct ≤ global ≤ TOA" =>
              0 <= c.fsw_clear.flux_dn_dir[1] <= c.fsw_clear.flux_dn[1] <= c.fsw_clear.flux_dn[end])
        push!(checks, "condensate does not add surface direct beam" =>
              c.fsw.flux_dn_dir[1] <= c.fsw_clear.flux_dn_dir[1] * (1 + 1e-12))
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
        tol = c.μ₀ >= 0.5 ? 0.05 : 0.10
        push!(checks, "clear-LES surface GHI within $(round(Int, 100tol)) % of an aerosol-free Bird sky" =>
              abs(c.fsw_clear.flux_dn[1] / bird - 1) < tol)
    end
    return checks
end

register!(ColumnRadiation)
