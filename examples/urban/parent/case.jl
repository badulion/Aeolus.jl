# The Berlin parent case (N-parent-e, #259): build, step, run.
#
# The parent assembles its physics from src and from the child's case code;
# it carries no copy of either:
#
#   core        BoussinesqSimulation on the StretchedGrid (#151), moist, f
#               of the child; the core's Rayleigh sponge damps u, v, w over
#               the top `damping_frac`;
#   SGS         the child's default `:sgs` provider, PrognosticSGS's TKE
#               closure (on the stretched grid since this issue), l capped
#               at κz, Moeng's c_ε factor in the first level, the surface
#               value from the tiles' u*;
#   time        TimeIntegration.step_adaptive! with the child's AdaptiveStep
#               constraints (`default_options(AdaptiveStep)`, the advective
#               Courant number 0.9), each step SSP-RK3 (`step_rk3!`);
#   initial     θ, qv from the child's `initial_θ` / `initial_qv` (the
#               Lindenberg sounding of #247 interpolated to the case start),
#               u, v = the geostrophic wind;
#   forcing     LargeScaleForcing: `GeostrophicPressureGradient` of the
#               sounding wind above z_g; `ProfileNudging` of θ, qv toward the
#               time-interpolated sounding (τ = 6 h) above 1.2 × z_i (#248);
#               `Subsidence` w_s = −D z (#248); `ReferenceDamping` of θ
#               toward the sounding over the top 25 % (#254);
#   top         the fixed-gradient θ top (#254, `θ_top_gradient` of the
#               child's initial profile at the parent's top face);
#   surface     the tiled urban mosaic (#249, `build_parent_surface`, the
#               child's class-to-surface factory) under the parent's air,
#               or a prescribed kinematic flux (the idealised tests);
#   radiation   the child's column radiation (`column_radiation!` on the
#               parent's mean column, `stitched_column`), the sun of the
#               child's clock (`solar_position_at`);
#   ledgers     two `AirHeatLedger`s (#253): the heat of the air
#               ``Σ ρ̄ c_p Π̄ θ ΔV / A`` and the water of the air
#               ``Σ ρ̄ q_v ΔV / A`` (the same meter with Π̄ = c_p = 1). Every
#               change is attributed: the surface and radiation stages by
#               their contexts, the dynamics split into transport and
#               redistribution, with the damping, nudging, subsidence and
#               top-flux terms booked from inside it.

"""
    ParentCase

A built Berlin parent run: `pc` ([`ParentConfig`](@ref)), `mesh`, `sim`
(the Boussinesq core), `ref` (the hydrostatic reference state, a
[`CaseReference`](@ref)), `closure` (the SGS closure or `nothing`),
`forcing` (named tuple of the large-scale forcings), `surface` (the
[`ParentSurface`](@ref) or `nothing`), `rad` (the `ColumnRadiation` state
or `nothing`), the ledgers `heat` and `water` ([`AirHeatLedger`](@ref)),
`top` (the [`TopHeatFlux`](@ref) or `nothing`), the time-stepping state,
the surface-flux state, the sounding time `t0` of the start and `diag`,
the diagnostics recorded every `output_period`.
"""
mutable struct ParentCase{M, S, C}
    pc::ParentConfig
    mesh::M
    sim::S
    ref::CaseReference
    closure::C
    forcing::NamedTuple
    surface::Any
    child::Any
    rad::Any
    heat::AirHeatLedger
    water::AirHeatLedger
    top::Union{Nothing,TopHeatFlux}
    constraints::CFLConstraints{Float64}
    dt_prev::Float64
    t0::Float64                 # sounding time (s) of the start; NaN without a sounding
    zi_obs::Float64             # largest parcel z_i of the sounding's launches (m)
    zi_run::Base.RefValue{Float64}   # running maximum of the diagnosed z_i (m), the nudging height
    # surface state
    θsrc::Array{Float64,3}      # K/s per column and canopy level, held between surface steps
    qsrc::Array{Float64,3}      # kg kg⁻¹ s⁻¹
    drag::Array{Float64,3}      # m s⁻² (kinematic stress / canopy depth)
    ustar::Matrix{Float64}
    t_surf::Float64             # end of the current surface interval (s)
    sfc_mean::Dict{Symbol,Float64}    # domain-mean J m⁻² of Rn, H, LE, G since the start
    sw_down::Float64
    lw_down::Float64
    t_rad::Float64              # time (s) of the next radiation call
    stage::Base.RefValue{Int}   # stage of the current dynamics step (1 for Euler, 1-3 for RK3)
    top_acc::Base.RefValue{Float64}   # stage-weighted top heat flux of the step (W m⁻²)
    # per-step scratch of the dynamics hooks
    lev::Dict{Symbol,Vector{Float64}}
    wθ::Vector{Float64}         # meter weights (J m⁻² per K of level sum)
    wq::Vector{Float64}         # water weights (kg m⁻² per kg/kg of level sum)
    # bookkeeping
    nsteps::Int
    retries::Int
    dt_hist::Vector{Float64}
    timing::Dict{Symbol,Float64}
    diag::Vector{Any}
    io::IO
end

# ---------------------------------------------------------------------------
# Reference state, initial state, sounding-derived forcing
# ---------------------------------------------------------------------------

"The initial θ(z) of the parent: `pc.θ_init`, else the child's `initial_θ`."
parent_θ_init(pc::ParentConfig) = pc.θ_init === nothing ?
    (θi = initial_θ(pc.child); z -> θi(0.0, 0.0, z)) : pc.θ_init
"The initial qv(z) of the parent: `pc.qv_init`, else the child's `initial_qv`."
parent_qv_init(pc::ParentConfig) = pc.qv_init === nothing ?
    (qi = initial_qv(pc.child); z -> qi(0.0, 0.0, z)) : pc.qv_init

"""
    parent_reference(pc, mesh) -> CaseReference

The hydrostatic reference state of the parent on its (stretched) levels:
θ̄ the initial profile ([`parent_θ_init`](@ref)), p̄ from
[`hydrostatic_pressure`](@ref) up from the child's ground pressure
(`case_surface_pressure`), ``Π̄ = (p̄/p_{ref})^{R_d/c_{pd}}``,
``T̄ = θ̄Π̄``, ``ρ̄ = p̄/(R_d T̄)``, with the dry-air constants of
`MoistThermodynamics` — the constants of the child's reference state.
"""
function parent_reference(pc::ParentConfig, mesh)
    c = Aeolus.MoistThermodynamics.DEFAULT_CONSTANTS
    R_d = c.gas_constant / c.molmass_dry
    κ = R_d / c.cp_dry
    θf = parent_θ_init(pc)
    zc = level_centres(mesh)
    ps = case_surface_pressure(pc.child)
    p̄ = hydrostatic_pressure(θf, zc, ps; g = pc.child.g)
    θ̄ = θf.(zc)
    π̄ = (p̄ ./ c.p_ref) .^ κ
    T̄ = θ̄ .* π̄
    ρ̄ = p̄ ./ (R_d .* T̄)
    return CaseReference(p̄, ρ̄, θ̄, T̄, π̄, nothing, ps, Float64(c.cp_dry))
end

"""
    sounding_zi_max(cfg) -> m

The largest parcel boundary-layer height (`boundary_layer_height`,
`:parcel`, on θ_v) of the launches of the child's sounding: 1171 m for the
10:45 UTC release of 29 June 2019 (#247). 0 without a sounding.
"""
function sounding_zi_max(cfg::CaseConfig)
    S = case_sounding(cfg)
    S === nothing && return 0.0
    MT = Aeolus.MoistThermodynamics
    P = S.profiles
    zi = 0.0
    for l in S.sounding.launches
        θv = [MT.virtual_temperature(forcing_value(P.θ, z, l.t), MT.PhasePartition(forcing_value(P.qv, z, l.t)))
              for z in P.z]
        zi = max(zi, boundary_layer_height(P.z, θv; method = :parcel))
    end
    return zi
end

"""
    parent_geostrophic_wind(pc, t0, z_g) -> (U_g, V_g)

The geostrophic wind of the parent as functions `(z, t)` (t the
simulation time): the sounding wind (`sounding_profiles` u, v,
linear in time between the launches) at ``\\max(z, z_g)`` — the wind
above the boundary layer is taken as geostrophic, and held constant with
height below `z_g` (no thermal wind inside the boundary layer). `z_g` is
`pc.z_g`, or 1.2 × [`sounding_zi_max`](@ref).
"""
function parent_geostrophic_wind(pc::ParentConfig, t0::Real, z_g::Real)
    S = case_sounding(pc.child)
    S === nothing && return ((z, t) -> 0.0, (z, t) -> 0.0)
    P = S.profiles
    return ((z, t) -> forcing_value(P.u, max(z, z_g), t0 + t),
            (z, t) -> forcing_value(P.v, max(z, z_g), t0 + t))
end

"Sounding table `name` as a function `(z, t)` of the simulation time."
function sounding_target(pc::ParentConfig, t0::Real, name::Symbol)
    S = case_sounding(pc.child)
    tab = getproperty(S.profiles, name)
    return (z, t) -> forcing_value(tab, z, t0 + t)
end

# ---------------------------------------------------------------------------
# Building
# ---------------------------------------------------------------------------

"""
    coarsen_landuse(pl, b) -> ParentLanduse

The parent land-use table on a grid `b` times coarser per axis (for the
reduced-resolution parents of the tests): class counts summed over each
`b × b` block (`r` becomes `b r`), frontal area density averaged, the
mean height roof-area weighted, the maximum height the block maximum.
"""
function coarsen_landuse(pl::ParentLanduse, b::Integer)
    b == 1 && return pl
    (pl.nx % b == 0 && pl.ny % b == 0) || error("coarsen_landuse: $(pl.nx) × $(pl.ny) not divisible by $b")
    nx, ny = pl.nx ÷ b, pl.ny ÷ b
    counts = zeros(Int, nx, ny, size(pl.counts, 3))
    λf, hm, hx = zeros(nx, ny), zeros(nx, ny), zeros(nx, ny)
    for J in 1:ny, I in 1:nx
        is, js = (I-1)*b+1:I*b, (J-1)*b+1:J*b
        counts[I, J, :] .= vec(sum(view(pl.counts, is, js, :); dims = (1, 2)))
        λf[I, J] = sum(view(pl.λf, is, js)) / b^2
        w = view(pl.counts, is, js, SURFACE_ROOF)
        hm[I, J] = sum(w) > 0 ? sum(w .* view(pl.hmean, is, js)) / sum(w) : 0.0
        hx[I, J] = maximum(view(pl.hmax, is, js))
    end
    return ParentLanduse(nx, ny, pl.dx * b, pl.x0, pl.y0, pl.r * b, counts, λf, hm, hx,
                         vcat(pl.header, "# coarsened by $b × $b (coarsen_landuse)"))
end

"""
    build_parent(pc::ParentConfig; child = nothing, io = stdout) -> ParentCase

Build the parent: the mesh ([`parent_mesh`](@ref)), the reference state,
the Boussinesq core with the sponge and the θ top, the initial state, the
SGS closure, the large-scale forcings, the surface (for `:tiles` the child
case `child` — built from `pc.child` if not given — provides the
class-to-surface factory and the class map under its footprint), the
column radiation and the ledgers.
"""
function build_parent(pc::ParentConfig; child = nothing, io::IO = stdout)
    cfg = pc.child
    mesh = parent_mesh(pc)
    Nx, Ny, Nz = mesh.Nx, mesh.Ny, mesh.Nz
    H = pc.extent[3]
    ref = parent_reference(pc, mesh)
    zc = level_centres(mesh)
    dzc = level_thickness(mesh)

    # the θ top (#254) and the core
    θtop = pc.θ_top === :initial_gradient ?
        FluxBC((parent_θ_init(pc)(H + dzc[end] / 2) - parent_θ_init(pc)(H - dzc[end] / 2)) / dzc[end]) :
        pc.θ_top === :noflux ? NoFlux() :
        error("ParentConfig.θ_top must be :initial_gradient or :noflux, got $(repr(pc.θ_top))")
    z_damp = (1 - pc.damping_frac) * H
    sim = BoussinesqSimulation(mesh; ν = 0.0, θ₀ = cfg.θ₀, g_accel = cfg.g, f_coriolis = cfg.f_coriolis,
                               C_s = pc.sgs === :smagorinsky ? cfg.C_s : 0.0, Pr_t = cfg.Pr_t,
                               sponge_α_max = pc.damping_rate, sponge_z_start = z_damp,
                               moisture = pc.moisture,
                               bcs = FieldBoundaryConditions(:periodic, :periodic, :wall),
                               θ_bcs = FieldBoundaryConditions(:periodic, :periodic; bottom = :noflux,
                                                               top = θtop))

    # sounding time axis, observed z_i, geostrophic wind
    S = case_sounding(cfg)
    t0 = S === nothing ? NaN : case_start_time(cfg)
    zi_obs = sounding_zi_max(cfg)
    z_g = isnan(pc.z_g) ? 1.2 * zi_obs : pc.z_g
    Ug, Vg = parent_geostrophic_wind(pc, isnan(t0) ? 0.0 : t0, z_g)

    # initial state
    θf, qf = parent_θ_init(pc), parent_qv_init(pc)
    set!(sim.θ, (x, y, z) -> θf(z))
    pc.moisture && set!(sim.qv, (x, y, z) -> qf(z))
    if pc.u_init === :geostrophic
        set!(sim.u, (x, y, z) -> Ug(z, 0.0)); set!(sim.v, (x, y, z) -> Vg(z, 0.0))
    else
        set!(sim.u, (x, y, z) -> Float64(pc.u_init))
    end
    rng = Random.Xoshiro(pc.seed)
    θi = interior_view(sim.θ)
    for k in 1:Nz
        zc[k] < pc.perturbation_depth || break
        for j in 1:Ny, i in 1:Nx
            θi[i, j, k] += pc.perturbation * (2rand(rng) - 1)
        end
    end

    # forcings (θ, qv: the tendencies they add are booked from inside the step)
    sounding = S !== nothing
    geo = pc.geostrophic && sounding ? GeostrophicPressureGradient(Ug, Vg) : nothing
    zi_run = Ref(0.0)
    nudge = isfinite(pc.nudging_τ) && sounding ?
        ProfileNudging(pc.moisture ? (θ = sounding_target(pc, t0, :θ), qv = sounding_target(pc, t0, :qv)) :
                                     (θ = sounding_target(pc, t0, :θ),),
                       pc.nudging_τ;
                       weight = NudgingWeight(s -> max(zi_obs, zi_run[]); factor = pc.nudging_factor,
                                              depth = pc.nudging_depth)) : nothing
    D = pc.divergence
    subs = D != 0 ? Subsidence((z, t) -> -D * z; form = :local) : nothing
    damp = pc.damping_rate > 0 ?
        ReferenceDamping(:θ, Aeolus.Operators.sponge_profile(pc.damping_rate, z_damp, H),
                         sounding && pc.θ_init === nothing ? sounding_target(pc, t0, :θ) :
                         (z, t) -> θf(z)) : nothing
    forcing = (geostrophic = geo, nudging = nudge, subsidence = subs, damping = damp)

    # surface
    surface = nothing
    if pc.surface === :tiles
        child === nothing && (child = build_case(cfg; io = devnull))
        pl = read_parent_landuse()
        b = pl.nx ÷ Nx
        (b * Nx == pl.nx && pl.dx * b ≈ mesh.dx) ||
            error("build_parent: the land use ($(pl.nx) cells of $(pl.dx) m) does not coarsen to $Nx cells of $(mesh.dx) m")
        pl = coarsen_landuse(pl, b)
        surface = build_parent_surface(child, pl; z_centres = zc)
    elseif pc.surface !== :prescribed
        error("ParentConfig.surface must be :tiles or :prescribed, got $(repr(pc.surface))")
    end
    kmax = surface === nothing ? 1 :
        maximum(max(1, maximum(col.ktile) - 1) for col in surface.columns if col !== nothing)
    ustar = fill(NaN, Nx, Ny)

    # SGS: the child's default provider, the TKE closure
    closure = nothing
    if pc.sgs === :tke
        wd = [zc[k] for i in 1:Nx, j in 1:Ny, k in 1:Nz]          # l ≤ κ z
        fl = falses(Nx, Ny, Nz); fl[:, :, 1] .= true                # Moeng's c_ε factor
        closure = TKEClosure(mesh; wall_distance = wd, first_level = fl,
                             surface = surface === nothing ? nothing : (u_star = ustar, k = ones(Int, Nx, Ny)),
                             transport = true)
        initialize_tke!(closure, sim; e_min = 1e-4)
    elseif pc.sgs !== :smagorinsky
        error("ParentConfig.sgs must be :tke or :smagorinsky, got $(repr(pc.sgs))")
    end

    rad = pc.radiation ? column_radiation_solver(Nz, ref.p_surface) : nothing

    # ledgers: heat of the air, water of the air
    hm = AirHeatMeter(mesh, ref.ρ̄, ref.π̄, ref.cp)
    wm = AirHeatMeter(mesh, ref.ρ̄, ones(Nz), 1.0)
    heat = heat_start!(AirHeatLedger(), hm; canopy_top = 100.0)
    water = heat_start!(AirHeatLedger(), wm; canopy_top = 100.0)
    top = θtop isa FluxBC ? TopHeatFlux(Float64(θtop.f), ref.ρ̄[Nz] * ref.cp * ref.π̄[Nz], 0.0, 0.0) : nothing

    o = default_options(AdaptiveStep)
    periods = filter(isfinite, [pc.surface === :tiles ? pc.surface_period : Inf,
                                pc.radiation ? pc.radiation_period : Inf, pc.output_period])
    k = CFLConstraints(; C_advective = pc.C_advective, C_diffusive = o.C_diffusive,
                       C_buoyancy = o.C_buoyancy, C_coriolis = o.C_coriolis, dt_min = o.dt_min,
                       dt_max = min(o.dt_max, minimum(periods; init = Inf)), growth_max = o.growth_max)

    lev = Dict{Symbol,Vector{Float64}}(s => zeros(Nz) for s in
              (:a, :b, :θnudging, :θsubsidence, :θdamping, :qnudging, :qsubsidence,
               :acc_θnudging, :acc_θsubsidence, :acc_θdamping, :acc_qnudging, :acc_qsubsidence))
    pcase = ParentCase(pc, mesh, sim, ref, closure, forcing, surface, child, rad, heat, water, top,
                       k, cfg.dt, isnan(t0) ? 0.0 : t0, zi_obs, zi_run,
                       zeros(Nx, Ny, kmax), zeros(Nx, Ny, kmax), zeros(Nx, Ny, kmax), ustar,
                       -Inf, Dict{Symbol,Float64}(:Rn => 0.0, :H => 0.0, :LE => 0.0, :G => 0.0, :E => 0.0,
                                            :delivered_H => 0.0, :delivered_E => 0.0),
                       0.0, 350.0, 0.0, Ref(0), Ref(0.0), lev, copy(hm.w), copy(wm.w),
                       0, 0, Float64[], Dict{Symbol,Float64}(), Any[], io)
    pc.verbose && @printf(io, "Berlin parent: %d × %d × %d (Δx %.2f m, Δz %.1f–%.1f m), surface %s, sgs %s, z_g %.0f m, z_i(obs) %.0f m\n",
                          Nx, Ny, Nz, mesh.dx, minimum(dzc), maximum(dzc), pc.surface, pc.sgs, z_g, zi_obs)
    return pcase
end

# ---------------------------------------------------------------------------
# Horizontal means, z_i
# ---------------------------------------------------------------------------

"Plane means of the interior of a cell-centred field, per level."
function plane_mean(f)
    a = interior_view(f)
    return [sum(view(a, :, :, k)) / (size(a, 1) * size(a, 2)) for k in axes(a, 3)]
end

"""
    mean_heat_flux(sim, κ) -> (zf, flux)

The plane-mean total kinematic heat flux (K m s⁻¹) of any core `sim` at
its interior w-faces (heights `zf` above the bottom): resolved
``⟨w'θ'⟩`` (θ averaged to the face) plus the SGS ``−⟨κ_v ∂θ/∂z⟩`` with the
cell-centred diffusivity `κ` (the centre spacing of a stretched grid in
the gradient) — the profile whose minimum is the `:min_flux` z_i
(Sullivan et al. 1998). Works on a `RegularGrid` and a `StretchedGrid`.
"""
function mean_heat_flux(sim, κ)
    m = sim.mesh
    Hx, Hy, Hz, Nx, Ny, Nz = m.Hx, m.Hy, m.Hz, m.Nx, m.Ny, m.Nz
    θd, wd, κd = sim.θ.data, sim.w.data, κ.data
    zfk(K) = hasproperty(m, :zf) ? m.zf[K] - m.z0 : (K - Hz - 1) * m.dz
    dzfk(K) = hasproperty(m, :dzf) ? m.dzf[K] : m.dz
    zf = Float64[zfk(Hz + k) for k in 2:Nz]
    flux = zeros(Nz - 1)
    n = Nx * Ny
    @inbounds for k in 2:Nz
        K = Hz + k
        sw = 0.0; sθ = 0.0; swθ = 0.0; sg = 0.0
        idz = 1 / dzfk(K)
        for j in Hy+1:Hy+Ny, i in Hx+1:Hx+Nx
            θf = 0.5 * (θd[i, j, K-1] + θd[i, j, K])
            w = wd[i, j, K]
            sw += w; sθ += θf; swθ += w * θf
            sg -= 0.5 * (κd[i, j, K-1] + κd[i, j, K]) * (θd[i, j, K] - θd[i, j, K-1]) * idz
        end
        flux[k-1] = swθ / n - (sw / n) * (sθ / n) + sg / n
    end
    return zf, flux
end

"The plane-mean total heat flux of the parent ([`mean_heat_flux`](@ref) with the closure's vertical κ)."
parent_heat_flux(p::ParentCase) =
    mean_heat_flux(p.sim, p.closure === nothing ? p.sim.κt : sgs_fields(p.closure, p.sim).κv)

"""
    parent_zi(p; method = :min_flux) -> m

The boundary-layer height of the parent now: `:min_flux` (the height of
the minimum of [`parent_heat_flux`](@ref), the CBL definition) or
`:parcel` (on the plane-mean θ_v), both `boundary_layer_height`.
"""
function parent_zi(p::ParentCase; method::Symbol = :min_flux)
    if method === :min_flux
        zf, fl = parent_heat_flux(p)
        return boundary_layer_height(zf, fl; method = :min_flux)
    end
    θ = plane_mean(p.sim.θ)
    q = p.pc.moisture ? plane_mean(p.sim.qv) : zeros(length(θ))
    MT = Aeolus.MoistThermodynamics
    θv = [MT.virtual_temperature(θ[k], MT.PhasePartition(max(q[k], 0.0))) for k in eachindex(θ)]
    return boundary_layer_height(level_centres(p.mesh), θv; method = method)
end

# ---------------------------------------------------------------------------
# Surface: tiles under the parent's air, or a prescribed flux
# ---------------------------------------------------------------------------

function _cell_speed(sim, i, j, k)
    m = sim.mesh
    I, J, K = m.Hx + i, m.Hy + j, m.Hz + k
    return hypot(0.5 * (sim.u.data[I, J, K] + sim.u.data[I+1, J, K]),
                 0.5 * (sim.v.data[I, J, K] + sim.v.data[I, J+1, K]))
end

"""
    step_parent_surface!(p, period) -> p

Advance every tile of every parent column by `period` seconds under the
parent's air (its own column: ``T = θ Π̄_k``, ``q_v``, the cell-centred wind
speed, ``p̄_k`` at each tile's forcing level, `parent_tile_forcings`) and
the radiation of the last radiation call, and turn the tiles' fluxes into
the sources the air receives until the next surface step. A tile forced at
level ``k_i`` (the first level above its roofs; 1 for park and water)
releases its heat, vapour and momentum flux into the levels below it,
``1 … \\max(1, k_i - 1)`` — the canopy air of the unresolved buildings —
uniformly per volume over their depth ``Z_i``:
``\\dot θ_k = f_i H_i / (Z_i ρ̄_k c_p Π̄_k)`` (the `ref_θ_per_heat`
conversion of the child), ``\\dot q_k = f_i E_i / (Z_i ρ̄_k)``, and a
deceleration ``f_i τ_i/(Z_i ρ̄_k)`` along the wind. u* (for the TKE surface
value) is ``\\sqrt{Σ f_i τ_i / ρ̄_1}``.
"""
function step_parent_surface!(p::ParentCase, period::Real)
    ps, sim, m, ref = p.surface, p.sim, p.mesh, p.ref
    Hx, Hy, Hz = m.Hx, m.Hy, m.Hz
    zf = Float64[m.zf[Hz+k] for k in 1:m.Nz+1]
    fill!(p.θsrc, 0.0); fill!(p.qsrc, 0.0); fill!(p.drag, 0.0)
    kmax = size(p.θsrc, 3)
    U_min = p.pc.U_min
    sRn = sH = sLE = sG = sE = 0.0
    θd = sim.θ.data
    qd = p.pc.moisture ? sim.qv.data : nothing
    fill_halos!(sim.u, sim.bcs); fill_halos!(sim.v, sim.bcs)
    for J in 1:m.Ny, I in 1:m.Nx
        col = ps.columns[I, J]
        kf = maximum(col.ktile)
        air = [SurfaceForcing(; sw_down = p.sw_down, lw_down = p.lw_down,
                              T_air = θd[Hx+I, Hy+J, Hz+k] * ref.π̄[k],
                              q_air = qd === nothing ? 0.0 : max(qd[Hx+I, Hy+J, Hz+k], 0.0),
                              U = max(_cell_speed(sim, I, J, k), U_min), p = ref.p̄[k], z_ref = 10.0)
               for k in 1:kf]
        agg = step_parent_column!(ps, col, air, period; U_min = U_min)
        sRn += agg.Rn; sH += agg.H; sLE += agg.LE; sG += agg.G; sE += agg.E
        tf = last_tile_fluxes(_mosaic(col))
        fr = mosaic_fractions(col.surface)
        τsum = 0.0
        for (n, f) in enumerate(tf)
            nl = max(1, col.ktile[n] - 1)
            Z = zf[nl+1]
            for k in 1:min(nl, kmax)
                p.θsrc[I, J, k] += fr[n] * f.H / (Z * ref.ρ̄[k] * ref.cp * ref.π̄[k])
                p.qsrc[I, J, k] += fr[n] * f.E / (Z * ref.ρ̄[k])
                p.drag[I, J, k] += fr[n] * f.τ / (Z * ref.ρ̄[k])
            end
            τsum += fr[n] * f.τ
        end
        p.ustar[I, J] = sqrt(max(τsum, 0.0) / ref.ρ̄[1])
    end
    n = m.Nx * m.Ny
    for (key, v) in ((:Rn, sRn), (:H, sH), (:LE, sLE), (:G, sG), (:E, sE))
        p.sfc_mean[key] += v / n * period
    end
    return p
end

# add θ, qv sources (per s) over τ seconds into the canopy levels
function _apply_sources!(p::ParentCase, τ::Real)
    m = p.mesh
    Hx, Hy, Hz = m.Hx, m.Hy, m.Hz
    θd = p.sim.θ.data
    qd = p.pc.moisture ? p.sim.qv.data : nothing
    @inbounds for k in axes(p.θsrc, 3)
        sθ = 0.0; sq = 0.0
        for j in 1:m.Ny, i in 1:m.Nx
            θd[Hx+i, Hy+j, Hz+k] += p.θsrc[i, j, k] * τ
            qd === nothing || (qd[Hx+i, Hy+j, Hz+k] += p.qsrc[i, j, k] * τ)
            sθ += p.θsrc[i, j, k]; sq += p.qsrc[i, j, k]
        end
        # what the sources delivered, by the ledger's weights (J m⁻², kg m⁻²)
        p.sfc_mean[:delivered_H] += p.wθ[k] * sθ * τ
        p.sfc_mean[:delivered_E] += p.wq[k] * sq * τ
    end
    return p
end

"""
    parent_surface_stage!(p, dt) -> p

Deliver the surface fluxes over the step `[t, t + dt]`: the held tile
sources up to the end of the current surface interval, then a surface
step ([`step_parent_surface!`](@ref)) and its sources for the rest (the
air receives exactly what the tiles gave away over the time it covers);
for `surface = :prescribed`, the step mean of `Q_surface(t)` and
`E_surface(t)` (`forcing_mean`) into the lowest level.
"""
function parent_surface_stage!(p::ParentCase, dt::Real)
    sim, m = p.sim, p.mesh
    t = sim.t
    if p.pc.surface === :prescribed
        dz1 = level_thickness(m)[1]
        Q = forcing_mean((z, s) -> p.pc.Q_surface(s), 0.0, t, t + dt)
        E = p.pc.moisture ? forcing_mean((z, s) -> p.pc.E_surface(s), 0.0, t, t + dt) : 0.0
        a = view(interior_view(sim.θ), :, :, 1)
        a .+= Q * dt / dz1
        p.pc.moisture && (view(interior_view(sim.qv), :, :, 1) .+= E * dt / dz1)
        p.sfc_mean[:H] += Q * dt * p.ref.ρ̄[1] * p.ref.cp * p.ref.π̄[1]
        return p
    end
    P = p.pc.surface_period
    s = t
    te = t + dt
    while s < te - 1e-9 * dt
        if s >= p.t_surf - 1e-9 * P
            step_parent_surface!(p, P)
            p.t_surf = s + P
        end
        τ = min(te, p.t_surf) - s
        _apply_sources!(p, τ)
        s += τ
    end
    return p
end

"""
    apply_parent_drag!(p, sim, Δt) -> p

The post-predictor hook: the momentum flux of the tiles as an implicit
deceleration of the canopy levels, ``u ← u / (1 + Δt\\,a/U)`` with
``a`` the per-volume kinematic stress of [`step_parent_surface!`](@ref)
(the mean of the two columns a face separates) and ``U ≥ U_{min}`` the
local speed: the stress magnitude held from the surface step, along the
instantaneous wind, stable for any Δt.
"""
function apply_parent_drag!(p::ParentCase, sim, Δt)
    m = p.mesh
    Hx, Hy, Hz, Nx, Ny = m.Hx, m.Hy, m.Hz, m.Nx, m.Ny
    ud, vd = sim.u.data, sim.v.data
    a = p.drag
    U_min = p.pc.U_min
    @inbounds for k in axes(a, 3)
        K = Hz + k
        for j in 1:Ny, i in 1:Nx
            im = i == 1 ? Nx : i - 1
            jm = j == 1 ? Ny : j - 1
            I, J = Hx + i, Hy + j
            # u at the west face of (i, j): v averaged to it
            au = 0.5 * (a[im, j, k] + a[i, j, k])
            vf = 0.25 * (vd[I-1, J, K] + vd[I, J, K] + vd[I-1, J+1, K] + vd[I, J+1, K])
            Uu = max(hypot(ud[I, J, K], vf), U_min)
            ud[I, J, K] /= 1 + Δt * au / Uu
            av = 0.5 * (a[i, jm, k] + a[i, j, k])
            uf = 0.25 * (ud[I, J-1, K] + ud[I+1, J-1, K] + ud[I, J, K] + ud[I+1, J, K])
            Uv = max(hypot(vd[I, J, K], uf), U_min)
            vd[I, J, K] /= 1 + Δt * av / Uv
        end
    end
    return p
end

# ---------------------------------------------------------------------------
# Radiation
# ---------------------------------------------------------------------------

"""
    parent_radiation!(p) -> p

One call of the child's column radiation ([`column_radiation!`](@ref)) on
the parent's plane-mean column ([`stitched_column`](@ref) with the
stretched layer thicknesses), at the sun of the child's clock: the
surface is the tile-mean emissivity and albedo and the emissivity-weighted
radiative temperature of all tiles. The downwelling shortwave and
longwave at the ground force the tiles until the next call.
"""
function parent_radiation!(p::ParentCase)
    c, m, cfg = p.rad, p.mesh, p.pc.child
    hour = hour_of_day(cfg, p.sim.t)
    sp = solar_position_at(cfg, hour)
    θ = plane_mean(p.sim.θ)
    qv = p.pc.moisture ? plane_mean(p.sim.qv) : zeros(m.Nz)
    if p.surface === nothing
        ε, α, Trad = 0.95, 0.15, θ[1] * p.ref.π̄[1]
    else
        cols = [col for col in p.surface.columns if col !== nothing]
        εs = [surface_emissivity(col.surface) for col in cols]
        ε = sum(εs) / length(cols)
        α = sum(surface_albedo(col.surface) for col in cols) / length(cols)
        Trad = (sum(e * surface_temperature(col.surface)^4 for (e, col) in zip(εs, cols)) / sum(εs))^(1 / 4)
    end
    dz = level_thickness(m)
    col = stitched_column(c, θ, qv, zeros(m.Nz), dz; T_rad = Trad, humid = season_humidity(cfg))
    column_radiation!(c, col, sp, ε, α)
    p.lw_down = c.flw.flux_dn[1]
    p.sw_down = c.fsw_clear === nothing ? 0.0 : c.fsw_clear.flux_dn[1]
    return p
end

# ---------------------------------------------------------------------------
# The dynamics: hooks, step, booking
# ---------------------------------------------------------------------------

function _level_sums!(out, f)
    a = interior_view(f)
    @inbounds for k in axes(a, 3)
        out[k] = sum(view(a, :, :, k))
    end
    return out
end

# The stage weights of the parent's time step: the net effect of a
# tendency evaluated at each stage is Σ w_s Δt T_s — one stage of weight 1
# for the forward-Euler projection, (1/6, 1/6, 2/3) for SSP-RK3
# (u³ = u⁰ + Δt(L⁰/6 + L¹/6 + 2L²/3) for the Shu–Osher form of step_rk3!).
const _RK3_WEIGHTS = (1 / 6, 1 / 6, 2 / 3)
_stage_weight(p::ParentCase, s::Int) = p.pc.stepper === :rk3 ? _RK3_WEIGHTS[s] : 1.0

const _BOOKED = (:θnudging, :θsubsidence, :θdamping, :qnudging, :qsubsidence)

# the θ (and qv) forcings, each one's per-level tendency sum recorded and
# accumulated with the stage weight
function _parent_body_force!(p::ParentCase, sim, Δt)
    F, L = p.forcing, p.lev
    stage = (p.stage[] += 1)
    w = _stage_weight(p, stage)
    if stage == 1
        for n in _BOOKED
            fill!(L[Symbol(:acc_, n)], 0.0)
        end
        p.top_acc[] = 0.0
    end
    t = sim.t
    F.geostrophic === nothing || apply_forcing!(sim, F.geostrophic, t, Δt)
    moist = p.pc.moisture
    for (name, f) in ((:nudging, F.nudging), (:subsidence, F.subsidence), (:damping, F.damping))
        f === nothing && continue
        _level_sums!(L[:a], sim.tθ)
        moist && name !== :damping && _level_sums!(L[:b], sim.tqv)
        apply_forcing!(sim, f, t, Δt)
        Lθ = L[Symbol(:θ, name)]
        _level_sums!(Lθ, sim.tθ); Lθ .-= L[:a]
        L[Symbol(:acc_θ, name)] .+= w .* Lθ
        if moist && name !== :damping
            Lq = L[Symbol(:q, name)]
            _level_sums!(Lq, sim.tqv); Lq .-= L[:b]
            L[Symbol(:acc_q, name)] .+= w .* Lq
        end
    end
    if p.top !== nothing
        capture_top_flux!(p.top, sim, p.closure)
        p.top_acc[] += w * p.top.rate
    end
    return sim
end

# book what the accepted (sub)step of length h added through the forcings
function _book_forcings!(p::ParentCase, h::Real)
    F, L = p.forcing, p.lev
    for name in (:nudging, :subsidence, :damping)
        getproperty(F, name) === nothing && continue
        add_heat!(p.heat, name, p.wθ .* L[Symbol(:acc_θ, name)] .* h)
        if p.pc.moisture && name !== :damping
            add_heat!(p.water, name, p.wq .* L[Symbol(:acc_q, name)] .* h)
        end
    end
    if p.top !== nothing
        J = p.top_acc[] * h
        p.top.total += J
        add_heat!(p.heat, :top_flux, J; k = p.mesh.Nz)
    end
    return p
end

"""
    step_parent!(p) -> dt

One step of the parent: the step length from `adaptive_dt` (the child's
constraints, with the SGS closure's diffusive limit), the surface fluxes
over it, the radiation (every `radiation_period`), then the dynamics with
`step_adaptive!` (retried in pieces if it must), with every change of the
heat and the water of the air attributed in the two ledgers.
"""
function step_parent!(p::ParentCase)
    sim, pc = p.sim, p.pc
    t = sim.t
    hour = hour_of_day(pc.child, t)
    tw = time_ns()
    d = cfl_diagnostics(sim, p.constraints; sgs = p.closure)
    dt = adaptive_dt(sim, p.constraints; dt_prev = p.dt_prev, diagnostics = d)
    # land on the output times
    tn = (floor(t / pc.output_period + 1e-9) + 1) * pc.output_period
    dt = min(dt, tn - t)
    θ, q = sim.θ, (pc.moisture ? sim.qv : sim.θ)
    heat_begin_step!(p.heat, θ, hour)
    pc.moisture && heat_begin_step!(p.water, q, hour)

    # surface
    t1 = time_ns()
    heat_open!(p.heat, :surface_flux, pc.surface, θ)
    pc.moisture && heat_open!(p.water, :surface_flux, pc.surface, q)
    parent_surface_stage!(p, dt)
    heat_close!(p.heat, θ)
    pc.moisture && heat_close!(p.water, q)
    p.timing[:surface] = get(p.timing, :surface, 0.0) + (time_ns() - t1) * 1e-9

    # radiation
    if p.rad !== nothing
        t1 = time_ns()
        if t >= p.t_rad - 1e-6
            parent_radiation!(p)
            p.t_rad = (floor(t / pc.radiation_period + 1e-9) + 1) * pc.radiation_period
        end
        heat_open!(p.heat, :column_tendency, :column_radiation, θ)
        Aeolus.Radiation.apply_radiative_cooling!(sim.θ, p.rad.heat .* dt ./ p.ref.π̄)
        heat_close!(p.heat, θ)
        p.rad.dt_applied = dt
        p.timing[:radiation] = get(p.timing, :radiation, 0.0) + (time_ns() - t1) * 1e-9
    end

    # dynamics
    t1 = time_ns()
    heat_open!(p.heat, :dynamics, :core, θ; split = true)
    pc.moisture && heat_open!(p.water, :dynamics, :core, q; split = true)
    bf = (s, Δt) -> _parent_body_force!(p, s, Δt)
    pp = p.surface === nothing ? nothing : (s, Δt) -> apply_parent_drag!(p, s, Δt)
    # every attempt (a retry too) starts its stage count afresh
    stepper = pc.stepper === :rk3 ?
        ((s, h; kw...) -> (p.stage[] = 0; step_rk3!(s, h; kw..., sgs = p.closure))) :
        ((s, h; kw...) -> (p.stage[] = 0; step!(s, h; kw..., sgs = p.closure)))
    remaining, nret, piece_min = Float64(dt), 0, Inf
    while remaining > 1e-12 * dt
        _, used, r = step_adaptive!(sim, p.constraints; dt = remaining, body_force! = bf,
                                    post_predictor! = pp, sgs = p.closure, stepper = stepper)
        _book_forcings!(p, used)
        remaining -= used; nret += r
        piece_min = min(piece_min, used)
    end
    heat_close!(p.heat, θ)
    pc.moisture && heat_close!(p.water, q)
    p.timing[:dynamics] = get(p.timing, :dynamics, 0.0) + (time_ns() - t1) * 1e-9

    heat_end_step!(p.heat, θ, dt)
    pc.moisture && heat_end_step!(p.water, q, dt)
    p.dt_prev = nret == 0 ? dt : piece_min
    p.retries += nret
    p.nsteps += 1
    push!(p.dt_hist, dt)
    p.timing[:total] = get(p.timing, :total, 0.0) + (time_ns() - tw) * 1e-9
    # the nudging height follows the boundary layer (cheap: one profile)
    if p.forcing.nudging !== nothing && (p.nsteps % 10 == 0)
        p.zi_run[] = max(p.zi_run[], parent_zi(p; method = :parcel))
    end
    return dt
end

"""
    parent_record!(p) -> NamedTuple

Record the diagnostics of the current state in `p.diag`: time (s, local
hour), z_i (`:min_flux` and `:parcel`), the plane-mean θ, qv, u, v
profiles, the mean surface fluxes since the last record, the ledgers'
cumulative terms (J m⁻², kg m⁻²), the air heat storage and its closure,
dt statistics and the wall time.
"""
function parent_record!(p::ParentCase)
    sim = p.sim
    hb = heat_budget(p.heat)
    wb = p.pc.moisture ? heat_budget(p.water) : nothing
    r = (t = sim.t, hour = hour_of_day(p.pc.child, sim.t),
         zi = parent_zi(p), zi_parcel = parent_zi(p; method = :parcel),
         θ = plane_mean(sim.θ), qv = p.pc.moisture ? plane_mean(sim.qv) : Float64[],
         u = plane_mean(sim.u), v = plane_mean(sim.v),
         sfc = copy(p.sfc_mean), heat = hb, water = wb,
         nsteps = p.nsteps, retries = p.retries, wall = get(p.timing, :total, 0.0),
         sw_down = p.sw_down, lw_down = p.lw_down)
    push!(p.diag, r)
    return r
end

"""
    run_parent!(p; hours = p.pc.hours, record = true) -> p

Run the parent for `hours` of simulated time, recording
[`parent_record!`](@ref) at the start and every `output_period`.
"""
function run_parent!(p::ParentCase; hours::Real = p.pc.hours, record::Bool = true)
    pc = p.pc
    t_end = p.sim.t + 3600hours
    record && isempty(p.diag) && parent_record!(p)
    while p.sim.t < t_end - 1e-6
        step_parent!(p)
        all(isfinite, interior_view(p.sim.θ)) || error("run_parent!: non-finite θ at t = $(p.sim.t) s")
        if record && abs(p.sim.t / pc.output_period - round(p.sim.t / pc.output_period)) < 1e-6
            r = parent_record!(p)
            pc.verbose && @printf(p.io, "  %5.2f h (%05.2f LT): z_i %6.0f m (parcel %6.0f), θ(12.5 m) %.2f K, H %.1f W/m², %d steps, %.0f s wall\n",
                                  r.t / 3600, r.hour, r.zi, r.zi_parcel, r.θ[1],
                                  length(p.diag) > 1 ? (r.sfc[:H] - p.diag[end-1].sfc[:H]) / pc.output_period : 0.0,
                                  r.nsteps, r.wall)
            pc.verbose && flush(p.io)
        end
    end
    return p
end
