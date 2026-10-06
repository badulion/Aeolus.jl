# Urban air quality: the NOx–O3–VOC mechanism of Aeolus.Chemistry on five
# advected tracers (NO, NO2, O3, VOC, HNO3, in ppb), with traffic and
# biogenic emissions into the first air cell of each column, resistance-
# model dry deposition whose aerodynamic resistance comes from
# Aeolus.SurfaceExchange, and rain scavenging of HNO3.
#
# All the chemistry, the deposition physics and the unit conversions live
# in src/Chemistry.jl (#135). This file holds only what is specific to the
# case: the emission inventory, the class → surface-kind and roughness
# maps, and the initial state.
#
# Assumptions to know about (all stated, none tuned):
#   * the emission inventory below is an order-of-magnitude estimate, not
#     a Berlin inventory (see `default_options`);
#   * OH is prescribed, OH_max × J(NO2)/J(NO2, overhead); the mechanism has
#     no HOx budget;
#   * J(NO2) is the clear-sky value everywhere: no shading, no clouds;
#   * T and p per level: the level-mean θ over air cells times the
#     reference Exner function, and the reference pressure p̄ (#131);
#   * r_a is computed here from the first air cell with
#     SurfaceExchange.solve_surface_layer and a per-class roughness equal
#     to the one #129 (PR #158) gives its surfaces. Once that PR is in,
#     the surface-exchange bus carries u* and the roughness per column and
#     this coupling should read them instead of solving again.

# Surface kinds of Chemistry.surface_resistance, by surface class
# (1 street, 2 roof, 3 park, 4 water, 5 wall, 6 glass).
const _AQ_SURFACE_KIND = (:paved, :paved, :vegetation, :water, :paved, :glass)

"The dry-depositing species of the mechanism (VOC does not deposit)."
const AQ_DEPOSITING = (:NO, :NO2, :O3, :HNO3)
"All species of the mechanism, in the order of `apply_photochemistry!`."
const AQ_SPECIES = (:NO, :NO2, :O3, :VOC, :HNO3)

"""
    traffic_profile(hour) -> 0…1

Diurnal shape of the traffic emissions: a night floor of 0.15, a daytime
plateau (06:30–21:00) and rush-hour peaks at 08:00 and 17:30, so that the
morning peak is ≈ 1 and midday ≈ 0.7. An assumed shape of the kind urban
weekday traffic counts show, not a Berlin count.
"""
function traffic_profile(hour::Real)
    day = 1 / (1 + exp(-(hour - 6.5) / 0.5)) / (1 + exp((hour - 21.0) / 0.7))
    rush = exp(-((hour - 8.0) / 1.5)^2) + exp(-((hour - 17.5) / 1.5)^2)
    return 0.15 + 0.55 * day + 0.3 * rush
end

"""
    AQChem

Slot `:chemistry`, provider `:nox_o3_voc` (default). Every step in
`:chemistry`:

1. emissions into the first air cell of each column (`k_air`, above the
   roof on building columns): traffic NOx on street and building-adjacent
   columns, isoprene on park columns;
2. [`apply_photochemistry!`](@ref) on the whole domain;
3. dry deposition [`apply_dry_deposition!`](@ref) of NO, NO2, O3 and HNO3
   from the first air cell with ``v_d = 1/(r_a + r_b + r_c)``;
4. wet deposition [`apply_wet_deposition!`](@ref) of HNO3 by the rain
   `qr` (the other species are too insoluble).

The species are registered tracers (`advect = true`), so the
`:transport` stage moves them. In `:surface_update`, on the surface
cadence, ``r_a = 1/(C_H U)`` and ``u_*`` are recomputed per column from
`solve_surface_layer` (first-air-cell wind and θ, surface skin Ts) and
the deposition velocities with them.
"""
mutable struct AQChem{F} <: AbstractCoupling
    NO::F; NO2::F; O3::F; VOC::F; HNO3::F
    k_air::Matrix{Int}               # first air level of each column (Nz+1 if none)
    kind::Matrix{Symbol}             # surface kind per column
    z0m::Matrix{Float64}
    r_a::Matrix{Float64}             # s/m, scalar aerodynamic resistance
    u_star::Matrix{Float64}
    v_d::Dict{Symbol,Matrix{Float64}}
    p_col::Vector{Float64}
    T_col::Vector{Float64}
    opts::NamedTuple
    # emission fluxes at traffic_profile = 1 / full sun, ppb m s⁻¹
    F_NO::Float64; F_NO2::Float64; F_VOC::Float64
    # diagnostics, summed over the run (ppb m, over the domain area)
    emitted::Dict{Symbol,Float64}
    dry::Dict{Symbol,Float64}
    wet::Dict{Symbol,Float64}
    # last step (domain sums, ppb × cells) for the budget smoke check
    last::NamedTuple
    J::Float64
    OH::Float64
end

coupling_name(::Type{<:AQChem}) = :aq_chem
slot(::Type{<:AQChem}) = :chemistry
provider(::Type{<:AQChem}) = :nox_o3_voc
modules_used(::Type{<:AQChem}) = (:Chemistry, :SurfaceExchange)
stages(::Type{<:AQChem}) = ((:chemistry, 10, EVERY_STEP),
                            (:surface_update, 90, Cadence(:soil)))
# a night run: J = OH = 0, pure NO + O3 titration and no photochemical source
smoke_variants(::Type{<:AQChem}) = ((label = "night titration", start_h = 0.5),)

"""
Options (all mixing ratios in ppb):

* `background` — initial NOx, Ox, VOC, HNO3; NO/NO2/O3 start in the
  photostationary state of the start hour. NOx 5.5 and Ox 45 ppb are
  of the order of a summer urban background, VOC (isoprene) 0.5 ppb.
* `nox_street` — traffic NOx emission per m² of street / building-adjacent
  ground at the morning peak, kg m⁻² s⁻¹ as NO2 mass: 2 µg m⁻² s⁻¹. The
  order of magnitude of a busy inner-city road network averaged over its
  area (≈ 0.3 g NOx km⁻¹ per vehicle, 1–2 thousand vehicles h⁻¹ on a
  30 m wide road gives 3–6 µg m⁻² s⁻¹ on the carriageway). An estimate,
  not an inventory.
* `no2_fraction` — primary NO2 share of the NOx emission (molar), 0.1.
* `isoprene_park` — park isoprene emission in full sun, kg m⁻² s⁻¹:
  0.3 µg m⁻² s⁻¹ (≈ 1 mg m⁻² h⁻¹, the order of mixed deciduous stands),
  scaled by J(NO2)/J(NO2, overhead) as a light proxy.
* `OH_max` — prescribed midday OH, molecules cm⁻³ (5 × 10⁶, the order of
  summer boundary-layer values).
* `ν` — NO → NO2 conversions per VOC oxidised by OH (RO2 and HO2), 2.
* `z0m` — momentum roughness per class (street, roof, park, water, wall,
  glass), m; `z0h = z0m/10`. The values #129 gives its surfaces.
* `output_3d` — species written as 3-D fields (the near-surface layer of
  every species is always written).
"""
default_options(::Type{<:AQChem}) =
    (background = (NOx = 5.5, Ox = 45.0, VOC = 0.5, HNO3 = 0.5),
     nox_street = 2.0e-9, no2_fraction = 0.1, isoprene_park = 3.0e-10,
     OH_max = 5.0e6, ν = 2.0,
     z0m = (0.05, 0.05, 0.1, 1e-4, 0.05, 0.01),
     output_3d = (:NO2, :O3))

function build(::Type{<:AQChem}, case, o)
    mesh, geom = case.mesh, case.geom
    Nx, Ny, Nz = mesh.Nx, mesh.Ny, mesh.Nz
    names = Dict(:NO => "nitric oxide", :NO2 => "nitrogen dioxide", :O3 => "ozone",
                 :VOC => "reactive VOC (isoprene)", :HNO3 => "nitric acid")
    f = Dict(s => ensure_tracer!(case, s; advect = true, positive = true,
                                 output = s in o.output_3d, units = "ppb",
                                 long_name = names[s] * " mixing ratio")
             for s in AQ_SPECIES)
    Hx, Hy, Hz = mesh.Hx, mesh.Hy, mesh.Hz
    k_air = [something(findfirst(k -> !geom.sm.mask[Hx+i, Hy+j, Hz+k], 1:Nz), Nz + 1)
             for i in 1:Nx, j in 1:Ny]
    cls = geom.sfc
    kind = [_AQ_SURFACE_KIND[clamp(Int(cls[i, j]), 1, 6)] for i in 1:Nx, j in 1:Ny]
    z0m = [Float64(o.z0m[clamp(Int(cls[i, j]), 1, 6)]) for i in 1:Nx, j in 1:Ny]
    p_col = copy(ref_pressure(case))
    # emissions → ppb m/s at the surface
    F_NOx = emission_ppb_flux(o.nox_street, 0.046)            # as NO2 mass
    F_VOC = emission_ppb_flux(o.isoprene_park, 0.06812)       # isoprene C5H8
    return AQChem(f[:NO], f[:NO2], f[:O3], f[:VOC], f[:HNO3], k_air, kind, z0m,
                  fill(Inf, Nx, Ny), zeros(Nx, Ny),
                  Dict(s => zeros(Nx, Ny) for s in AQ_DEPOSITING),
                  p_col, zeros(Nz), o,
                  (1 - o.no2_fraction) * F_NOx, o.no2_fraction * F_NOx, F_VOC,
                  Dict(s => 0.0 for s in AQ_SPECIES), Dict(s => 0.0 for s in AQ_DEPOSITING),
                  Dict(s => 0.0 for s in AQ_DEPOSITING),
                  (dOx = 0.0, dNOx = 0.0, P_Ox = 0.0, L_Ox = 0.0, HNO3 = 0.0, rain = false),
                  0.0, 0.0)
end

# J(NO2) and OH at the current time
function _aq_sun!(c::AQChem, case)
    μ = solar_position_at(case.cfg, case.clock.hour).μ₀
    c.J = j_NO2(μ)
    c.OH = c.opts.OH_max * c.J / j_NO2(1.0)
    return c
end

# level-mean temperature over the air cells: T = θ Π̄. The masked sums and
# counts per level in the canonical, thread-invariant order (KA-b, #296).
function _aq_temperature!(c::AQChem, case)
    θ = case.sim.θ
    m = case.mesh
    Nz = m.Nz
    S, n = zeros(Nz), zeros(Nz)
    plane_reduce!((S, n), Aeolus.Backends.backend(θ.data), _aq_air_θ_at,
                  (1:m.Nx, 1:m.Ny, 1:Nz), θ.data, c.k_air, m.Hx, m.Hy, m.Hz)
    for k in 1:Nz
        θk = n[k] > 0 ? S[k] / n[k] : case.cfg.θ₀
        c.T_col[k] = θk * ref_exner(case, k)
    end
    return c
end

@inline function _aq_air_θ_at(i, j, k, θ, k_air, Hx, Hy, Hz)
    @inbounds k_air[i, j] <= k || return (0.0, 0.0)
    return (@inbounds(θ[Hx+i, Hy+j, Hz+k]), 1.0)
end

function init!(c::AQChem, case)
    _aq_sun!(c, case)
    _aq_temperature!(c, case)
    b = c.opts.background
    for k in axes(c.T_col, 1)
        Tk, pk = c.T_col[k], c.p_col[k]
        k1 = ppb_rate_constant(k_NO_O3(Tk), pk, Tk)
        y = photostationary_NO2(b.NOx, b.Ox, c.J, k1)
        view(interior_view(c.NO), :, :, k) .= b.NOx - y
        view(interior_view(c.NO2), :, :, k) .= y
        view(interior_view(c.O3), :, :, k) .= b.Ox - y
    end
    interior_view(c.VOC) .= b.VOC
    interior_view(c.HNO3) .= b.HNO3
    _aq_deposition_velocities!(c, case)
    return nothing
end

"""
    _aq_deposition_velocities!(c, case)

Per column: solve the surface layer between the skin (`bus.Ts`) and the
first air cell (wind at the cell centre, θ, qv), take ``r_a = 1/(C_H U)``
and ``u_*``, and set ``v_d = 1/(r_a + r_b + r_c)`` for every depositing
species.
"""
function _aq_deposition_velocities!(c::AQChem, case)
    u, v, θ = interior_view(case.sim.u), interior_view(case.sim.v), interior_view(case.sim.θ)
    qv = haskey(case.tracers, :qv) ? interior_view(tracer(case, :qv)) : nothing
    Nx, Ny, Nz = size(θ)
    z_m = case.mesh.dz / 2                     # cell centre above the resolved surface
    Ts = case.bus.Ts
    Sc = Dict(s => schmidt_number(s) for s in AQ_DEPOSITING)
    for j in 1:Ny, i in 1:Nx
        k = c.k_air[i, j]
        if k > Nz
            c.r_a[i, j] = Inf; c.u_star[i, j] = 0.0
            for s in AQ_DEPOSITING; c.v_d[s][i, j] = 0.0; end
            continue
        end
        U = hypot(centre_wind(case.sim, i, j, k)...)   # BC-aware faces (#262)
        q = qv === nothing ? 0.0 : qv[i, j, k]
        z0 = c.z0m[i, j]
        # θ is referred to 1000 hPa (#131): the skin's potential temperature
        # is Ts / Π̄ at its air cell
        sol = solve_surface_layer(; U = U, z_m = z_m, θ_a = θ[i, j, k],
                                  θ_s = Ts[i, j] / ref_exner(case, k),
                                  q_a = q, q_s = q, z0m = z0, z0h = z0 / 10)
        ra = 1 / (sol.C_H * sol.U)
        c.r_a[i, j] = ra
        c.u_star[i, j] = sol.u_star
        for s in AQ_DEPOSITING
            rb = quasi_laminar_resistance(sol.u_star; Sc = Sc[s])
            c.v_d[s][i, j] = dry_deposition_velocity(ra, rb, surface_resistance(s, c.kind[i, j]))
        end
    end
    return c
end

function run_stage!(c::AQChem, case, ::Val{:surface_update}, dt)
    _aq_deposition_velocities!(c, case)
    return nothing
end

_aq_sp(c::AQChem, s::Symbol) = interior_view(getfield(c, s))

function _aq_emit!(c::AQChem, case, dt)
    dz, sfc = case.mesh.dz, case.geom.sfc
    NO, NO2, VOC = _aq_sp(c, :NO), _aq_sp(c, :NO2), _aq_sp(c, :VOC)
    Nz = size(NO, 3)
    ftr = traffic_profile(case.clock.hour)
    fsun = c.J / j_NO2(1.0)
    eNO, eNO2, eVOC = 0.0, 0.0, 0.0
    # surface sums: the columns' contributions folded in column-major order,
    # the canonical order of a 2-D reduction (serial; KA-b, #296). The dry
    # deposition total (`apply_dry_deposition!`) is the same kind of sum.
    @inbounds for j in axes(NO, 2), i in axes(NO, 1)
        k = c.k_air[i, j]
        k <= Nz || continue
        cl = sfc[i, j]
        if cl == SURFACE_STREET || cl == SURFACE_WALL
            a = c.F_NO * ftr * dt / dz; b = c.F_NO2 * ftr * dt / dz
            NO[i, j, k] += a; NO2[i, j, k] += b
            eNO += a * dz; eNO2 += b * dz
        elseif cl == SURFACE_PARK
            a = c.F_VOC * fsun * dt / dz
            VOC[i, j, k] += a
            eVOC += a * dz
        end
    end
    A = length(c.k_air)
    c.emitted[:NO] += eNO / A; c.emitted[:NO2] += eNO2 / A; c.emitted[:VOC] += eVOC / A
    return nothing
end

function run_stage!(c::AQChem, case, ::Val{:chemistry}, dt)
    _aq_sun!(c, case)
    _aq_temperature!(c, case)
    _aq_emit!(c, case, dt)
    NO, NO2, O3, VOC, HNO3 = (_aq_sp(c, s) for s in AQ_SPECIES)
    # the species totals in the canonical, thread-invariant order (KA-b, #296)
    tot(s) = interior_sum(getfield(c, s))
    Ox0 = tot(:O3) + tot(:NO2); N0 = tot(:NO) + tot(:NO2)
    b = apply_photochemistry!(NO, NO2, O3, VOC, HNO3, dt; J = c.J, T = c.T_col, p = c.p_col,
                              OH = c.OH, ν = c.opts.ν)
    dOx = tot(:O3) + tot(:NO2) - Ox0
    dN = tot(:NO) + tot(:NO2) - N0
    A, dz = length(c.k_air), case.mesh.dz
    for s in AQ_DEPOSITING
        c.dry[s] += apply_dry_deposition!(_aq_sp(c, s), c.v_d[s], dt; dz = dz, k = c.k_air) / A
    end
    rain = false
    if haskey(case.tracers, :qr)
        qr = interior_view(tracer(case, :qr))
        rain = any(>(0), qr)
        for s in AQ_DEPOSITING
            sc = rain_scavenging_coefficient(s)
            sc > 0 && (c.wet[s] += apply_wet_deposition!(_aq_sp(c, s), qr, dt; scav = sc) * dz / A)
        end
    end
    c.last = (dOx = dOx, dNOx = dN, P_Ox = b.P_Ox, L_Ox = b.L_Ox, HNO3 = b.HNO3_formed, rain = rain)
    return nothing
end

"""
    aq_leighton(c, case) -> (φ, φ_ext)

Medians over the air cells with NO, O3 > 0.01 ppb of the Leighton ratio
``φ = J[NO_2]/(k_1[NO][O_3])`` and of ``φ/φ_{ext}``, where
``φ_{ext} = 1 + ν k_{VOC,OH}[OH][VOC]/(k_1[NO][O_3])`` is the steady state
that includes the peroxy-radical NO → NO2 conversion. `NaN` at night.
"""
function aq_leighton(c::AQChem, case)
    c.J > 0 || return (φ = NaN, φ_ext = NaN)
    NO, NO2, O3, VOC = _aq_sp(c, :NO), _aq_sp(c, :NO2), _aq_sp(c, :O3), _aq_sp(c, :VOC)
    φs = Float64[]; rs = Float64[]
    for k in axes(NO, 3)
        Tk, pk = c.T_col[k], c.p_col[k]
        k1 = ppb_rate_constant(k_NO_O3(Tk), pk, Tk)
        L4 = k_VOC_OH(Tk) * c.OH
        for j in axes(NO, 2), i in axes(NO, 1)
            c.k_air[i, j] <= k || continue
            (NO[i, j, k] > 0.01 && O3[i, j, k] > 0.01) || continue
            φ = leighton_ratio(NO[i, j, k], NO2[i, j, k], O3[i, j, k], c.J, k1)
            push!(φs, φ)
            push!(rs, φ / (1 + c.opts.ν * L4 * VOC[i, j, k] / (k1 * NO[i, j, k] * O3[i, j, k])))
        end
    end
    isempty(φs) && return (φ = NaN, φ_ext = NaN)
    med(x) = (s = sort(x); n = length(s); isodd(n) ? s[(n + 1) ÷ 2] : (s[n ÷ 2] + s[n ÷ 2 + 1]) / 2)
    return (φ = med(φs), φ_ext = med(rs))
end

"Mean over the columns of `f` in the first air cell."
function _aq_surface_layer(c::AQChem, f)
    a = interior_view(f)
    Nz = size(a, 3)
    return [c.k_air[i, j] <= Nz ? a[i, j, c.k_air[i, j]] : NaN for i in axes(a, 1), j in axes(a, 2)]
end

function output_fields(c::AQChem, case)
    fs = OutputField[]
    for s in AQ_SPECIES
        f = getfield(c, s)
        push!(fs, OutputField("$(s)_sfc", "ppb", "$s mixing ratio in the first air cell", :xy,
                              _ -> _aq_surface_layer(c, f)))
    end
    push!(fs, OutputField("vd_O3", "m s-1", "O3 dry deposition velocity", :xy, _ -> c.v_d[:O3]))
    push!(fs, OutputField("r_a", "s m-1", "aerodynamic resistance, surface to first air cell", :xy,
                          _ -> min.(c.r_a, 1e6)))
    return fs
end

function report(c::AQChem, case)
    NO, NO2, O3 = (_aq_surface_layer(c, getfield(c, s)) for s in (:NO, :NO2, :O3))
    ok = .!isnan.(NO)
    m(x) = sum(x[ok]) / count(ok)
    L = aq_leighton(c, case)
    return @sprintf("chem: sfc NO %.2f NO2 %.2f O3 %.1f ppb, Leighton φ %.3f (φ/φ_ext %.3f), J %.2e",
                    m(NO), m(NO2), m(O3), L.φ, L.φ_ext, c.J)
end

function smoke_check(c::AQChem, case)
    # ≥ 0 to the round-off the transport's positivity limiter leaves
    # (measured −3e-21 ppb where a species is 0, e.g. NO at night)
    fin = all(s -> all(x -> isfinite(x) && x >= -1e-12, _aq_sp(c, s)), AQ_SPECIES)
    l = c.last
    Ox = interior_sum(c.O3) + interior_sum(c.NO2)
    checks = ["NO, NO2, O3, VOC, HNO3 finite and ≥ 0 (to 1e-12 ppb)" => fin,
              # the mechanism's own budgets over the last photochemistry call:
              # ΔOx = P − L and ΔNOx = −ΔHNO3 (the triad conserves both exactly)
              "Ox budget closes in the photochemistry (rel. 1e-10)" =>
                  abs(l.dOx - (l.P_Ox - l.L_Ox)) <= 1e-10 * Ox,
              "NOx budget closes in the photochemistry (rel. 1e-10)" =>
                  abs(l.dNOx + l.HNO3) <= 1e-10 * Ox,
              # series resistances: v_d below the aerodynamic conductance
              "0 ≤ v_d ≤ 1/r_a for every species" =>
                  all(s -> all(i -> 0 <= c.v_d[s][i] <= 1 / c.r_a[i] * (1 + 1e-12), eachindex(c.r_a)),
                      AQ_DEPOSITING),
              "dry deposition removed O3 and HNO3" => c.dry[:O3] > 0 && c.dry[:HNO3] > 0,
              "no wet deposition of the insoluble NO, NO2, O3" =>
                  c.wet[:NO] == 0 && c.wet[:NO2] == 0 && c.wet[:O3] == 0]
    l.rain && push!(checks, "rain (qr > 0) scavenged HNO3" => c.wet[:HNO3] > 0)
    if c.J > 0
        L = aq_leighton(c, case)
        # daylight: the triad keeps NO2/NO near its photostationary state
        # (relaxation ≈ 40 s); fresh NO pulls φ below 1 and peroxy
        # radicals push it above, so only a loose band is asserted.
        push!(checks, "daylight Leighton ratio within [0.8, 1.25] (median $(round(L.φ, digits = 3)))" =>
                  0.8 <= L.φ <= 1.25)
    end
    return checks
end

# ---------------------------------------------------------------------------
# Legacy PM path of the toy aerosol (aq_aerosol.jl, replaced by #147): a
# class table of PM deposition velocities, applied at k = 1 through the src
# `apply_dry_deposition!`.
# ---------------------------------------------------------------------------

"""
    deposition_velocity(:PM, cls::UInt8) -> m/s

PM dry deposition velocity of the toy aerosol by surface class (park
0.005, water 0.001, glass 0.0005, otherwise 0.001 m/s). Kept only for
`aq_aerosol.jl` until #147 unifies the aerosol.
"""
@inline function deposition_velocity(species::Symbol, cls::UInt8)
    species === :PM || throw(ArgumentError("deposition_velocity: only :PM (gases use Chemistry.dry_deposition_velocity)"))
    cls == SURFACE_PARK  && return 0.005
    cls == SURFACE_WATER && return 0.001
    cls == 0x06          && return 0.0005   # SURFACE_GLASS
    return 0.001
end

function Aeolus.Chemistry.apply_dry_deposition!(field::Field, sfc::AbstractMatrix{UInt8},
                                                species::Symbol, dt::Real; dz::Real)
    vd = [deposition_velocity(species, cl) for cl in sfc]
    Aeolus.Chemistry.apply_dry_deposition!(interior_view(field), vd, dt; dz = dz, k = 1)
    return field
end

register!(AQChem)

"""
    NoChem

Slot `:chemistry`, provider `:off`: no reactive species at all. The
chemistry adds about 24–31 % to the cost of a flagship step (measured on
the 64 × 64 × 32 case on master `915bd49` with the adaptive step on a
loaded machine: 233.7 → 289.1 ms/step, of which 50 ms is the WENO
transport of the five species and 12 ms the `aq_chem` stage; see
`docs/src/design/berlin_air_quality.md`), so runs that do not need air quality
can switch it off with
`physics = Dict(:chemistry => :off)`.
"""
struct NoChem <: AbstractCoupling end
coupling_name(::Type{NoChem}) = :chem_off
slot(::Type{NoChem}) = :chemistry
provider(::Type{NoChem}) = :off
is_default(::Type{NoChem}) = false

register!(NoChem)
