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
* **Air side**, every step in `:surface_flux`:
  ``Δθ = H\\,τ/(\\bar ρ_k c_p \\bar Π_k Δz)`` (`ref_θ_per_heat`),
  ``Δq_v = E\\,τ/(\\bar ρ_k Δz)`` in the first air cell, `τ` the air time
  this step delivers (``τ = Δt`` with a fixed step), with ``\\bar ρ_k`` and
  ``\\bar Π_k`` the reference density and Exner function of that level
  (`case.ref`, the state the ledgers integrate with) and ``c_p`` the moist
  heat capacity of that air. The air-heat ledger weights θ with the dry
  ``c_{pd}``, so it books this heating as ``H τ\\, c_{pd}/c_p``, about
  0.8 % below ``H τ`` at ``q_v ≈ 0.01``.
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
* **Wall-model input** (P57-b, #183; `WallModels.MatchingInput`): the
  wind the law is evaluated at is the first-air-cell wind passed through
  the exponential time filter of Yang, Park & Moin (2017), time scale
  `input_T_f` (default 10 s, the order of the eddy time ``z_{ref}/u_*``
  of the 6.25 m first cell in a daytime wind; `0` switches it off),
  advanced once per accepted step in `:surface_update`. The stress keeps
  the direction of the instantaneous first-cell wind. The matching
  height stays at the first air cell (`h_wm_cells = 1`): three cells
  above a street (18.75 m) lie in or above the canyon, where no log law
  holds. `h_wm_cells = n` samples the `n`-th air cell (``(n-½)Δz``) of
  the columns whose class is in `h_wm_classes` (default park and water,
  the open surfaces; a column whose `n`-th cell is not air falls back to
  its first) — a smoke variant.
* **Non-equilibrium wall model** (P57-c, #189; option `wall_model =
  :nonequilibrium`, default `:equilibrium`): the ground stress is that of
  a `WallModels.TBLEWallModel` instead of the `MOSTWallLaw` — the
  thin-boundary-layer equations on a 30-point column under every ground
  and roof column, between the surface (u = 0, the column's own `z0m`,
  rough-wall eddy viscosity ``κu_τ(y + z_{0m})``) and the matching point
  of the same `MatchingInput` (time-filtered wind), with the pressure
  gradient of the resolved `p` at the first air level (one-sided next to
  a building) and the terms of `tble_terms` (default
  `:pressure_gradient`). The columns' stress is applied by the same
  `apply_wall_stress!` face loop in `:post_predictor`; the columns are
  advanced once per step in `:surface_update`; `bus.u_star` is their
  ``u_τ`` (set again after a surface step, whose MOST ``u_*`` stays
  internal to its heat and moisture fluxes). **Neutral:** the Monin–Obukhov stability correction of the
  equilibrium law is not carried into the TBLE (its eddy viscosity has no
  buoyancy term until P58). The surface-layer heat and moisture fluxes
  are unchanged (MOST). A smoke variant runs it.
* **Extra energy.** `bus.sensible` (energy other couplings take from the
  surfaces, W m⁻²) is collected every step in `:surface_update` and
  handed to the surfaces as `Q_extra` at their next step.
* **Maximum sustainable heat flux** (P58-e, #193), every surface step:
  per column, `SurfaceExchange.surface_mshf_ratio` of the MOST fluxes of
  the skin at the start of the step (`turbulent_fluxes`), one 1-D
  maximisation at the column's wind, `z_ref`, `z0m` and ``θ_{va}``
  (Van de Wiel et al. 2012). `H_max` (W m⁻²) and the ratio of the downward
  buoyancy flux to the MSHF are output fields; the diagnostics give the
  mean ratio of the stable street, roof and park columns and the share
  of them beyond the MSHF point (``z_{ref}/L`` past the peak of the flux
  curve: the very stable, collapsed branch); the smoke
  check asserts ratio ≤ 1 + 1e-8 everywhere (and < 1.1 for the fluxes
  the surface returned, which a soil skin linearises in ``T_{skin}``).

* **Dynamic roughness** (P57-e, #190; option `z0m = :dynamic`, off by
  default): the ground wall law's momentum roughness becomes the dynamic
  roughness of Anderson & Meneveau (2011), `WallModels.DynamicRoughness`:
  ``z_0 = \\sqrt{z_{0,f}^2 + α^2σ_Δ^2}``, ``σ_Δ`` the RMS of the OSM
  building heights *below* the grid — the footprints rasterised on a grid
  `dynamic_refine` times finer, with their true heights, about the
  immersed-boundary staircase the LES resolves (``(k_{air}-1)Δz``), i.e.
  the staircase loss — and ``z_{0,f}`` each column's own roughness
  (the per-class table, which stays the roughness of the surface-layer
  solve, so H, LE and the Obukhov length are unchanged). α is solved at
  every surface step (the `period` cadence, not every step) from the
  first-air-cell wind, test filter `dynamic_filter` (`:spectral`,
  default, or `:box`), resolved-relief drag coefficient `dynamic_C_d`.
  The default stays the table: the dynamic value's Berlin calibration has
  not been reviewed, and with buildings resolved at 15.6 m the sub-filter
  relief is only staircase loss. A smoke variant.

Options: `U_min` (m s⁻¹, the minimum wind of the surface-layer solve),
`z0m` (per class, m; or `:dynamic`, above), `z0h_ratio` (``z_{0h} = z_{0m}``·ratio),
`period` (s, overrides the cadence), `wall_formulation` (`:local`,
default; `:schumann` or `:moeng`, the plane-mean forms of
`apply_wall_stress!`, over all ground and roof columns), `input_T_f`
(s, the time filter of the wall-model input; 0 = off), `h_wm_cells`
and `h_wm_classes` (the matching height of the open columns),
`wall_model` (`:equilibrium` or `:nonequilibrium`) and `tble_terms`
(`:pressure_gradient`, `:full` or `:equilibrium_ode`),
`dynamic_refine`, `dynamic_filter`, `dynamic_C_d` (the dynamic roughness).
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
    input::MatchingInput       # the wall-model input (matching height, time filter), P57-b
    plain_input::Bool          # first air cell, no filter: the pre-P57-b call, bit for bit
    h_wall::Matrix{Float64}    # height (m above the surface) bus.u_star is the wall law at
    h_match::Matrix{Float64}   # matching height of the last wall-stress evaluation (m)
    n_h_fallback::Int          # open columns whose h_wm_cells-th cell is not air (matched at the first)
    tble::Union{Nothing,TBLEWallModel} # the non-equilibrium ground wall model (P57-c), or nothing
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
    # P57-e (#190): the dynamic roughness of the ground (z0m = :dynamic), or nothing
    dynamic::Bool
    dyn::Any                   # WallModels.DynamicRoughness, built in init!
    dyn_opts::NamedTuple
    t_dyn::Float64             # wall time spent in the dynamic solve (s)
    ucs::Matrix{Float64}       # first-air-cell wind of the dynamic solve (scratch, #295)
    vcs::Matrix{Float64}
    # P58-e (#193): maximum sustainable heat flux per column, last surface step
    H_max::Matrix{Float64}     # ρ c_p Q_max (W m⁻²; SurfaceExchange.max_sustainable_heat_flux)
    mshf_ratio::Matrix{Float64} # downward buoyancy flux / Q_max (< 0: upward, unstable)
    mshf_beyond::Matrix{Bool}  # z_ref/L beyond the MSHF point: the very stable (collapsed) branch
    mshf_worst::Float64        # max ratio of the MOST fluxes of the step's start, all columns and steps
    mshf_worst_delivered::Float64 # the same for the fluxes step_surface! returned (linearised soil skin)
    mshf_time::Float64         # time spent in the MSHF maximisations, summed over columns (s)
    # KA-m (#301): the columns grouped by concrete surface type (one launch
    # per group), and per-column terms of the surface step (rows: columns
    # in column-major order; store_err, mshf_worst, mshf_worst_delivered,
    # mshf time) folded after the launches
    groups::Vector{Any}
    colterms::Matrix{Float64}
end

coupling_name(::Type{SurfaceExchangeMOST}) = :surface_exchange
slot(::Type{SurfaceExchangeMOST}) = :surface_exchange
provider(::Type{SurfaceExchangeMOST}) = :most
modules_used(::Type{SurfaceExchangeMOST}) = (:SurfaceExchange, :Radiation, :MoistThermodynamics,
                                              :WallModels)
# the surface step's period: `[cadence] surface` if set, else `soil` (see `period`)
cadence_keys(::Type{SurfaceExchangeMOST}) = (:surface, :soil)
default_options(::Type{SurfaceExchangeMOST}) =
    (U_min = 0.1, z0h_ratio = 0.1, period = nothing, wall_formulation = :local,
     # the wall-model input (P57-b): time filter on, matching at the first air cell
     input_T_f = 10.0, h_wm_cells = 1, h_wm_classes = (SURFACE_PARK, SURFACE_WATER),
     # the ground wall model (P57-c): the MOST law, or the TBLE columns
     wall_model = :equilibrium, tble_terms = :pressure_gradient,
     # P57-e: z0m = :dynamic → the dynamic roughness of the OSM height map
     dynamic_refine = 4, dynamic_filter = :spectral, dynamic_C_d = 1.0,
     # momentum roughness per class (m): the sub-grid roughness of the
     # surface itself — buildings and park trees are resolved (IBM, drag)
     z0m = Dict(SURFACE_STREET => 0.05, SURFACE_ROOF => 0.05, SURFACE_PARK => 0.1,
                SURFACE_WATER => 1e-4, SURFACE_WALL => 0.05, UInt8(6) => 0.01))
smoke_variants(::Type{SurfaceExchangeMOST}) =
    ((label = "Schumann wall stress", options = Dict(:surface_exchange => (wall_formulation = :schumann,))),
     (label = "Moeng wall stress", options = Dict(:surface_exchange => (wall_formulation = :moeng,))),
     (label = "matching height 3 cells over park and water",
      options = Dict(:surface_exchange => (h_wm_cells = 3,))),
     (label = "non-equilibrium (TBLE) ground wall model",
      options = Dict(:surface_exchange => (wall_model = :nonequilibrium,))),
     # the spectral test filter needs a periodic surface: the closed box
     # (nest off); the nested default takes the box filter (#265)
     (label = "dynamic z0 from the OSM height map (Anderson & Meneveau 2011), periodic box",
      options = Dict(:surface_exchange => (z0m = :dynamic,)), physics = Dict(:nesting => false)),
     (label = "dynamic z0, box test filter, nested",
      options = Dict(:surface_exchange => (z0m = :dynamic, dynamic_filter = :box))))
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
    o.input_T_f >= 0 || error("surface_exchange: input_T_f must be ≥ 0, got $(o.input_T_f)")
    (o.h_wm_cells isa Integer && o.h_wm_cells >= 1) ||
        error("surface_exchange: h_wm_cells must be an integer ≥ 1, got $(o.h_wm_cells)")
    cells, nfb = matching_cells(case.geom, bus.k_air, mesh, o.h_wm_cells, o.h_wm_classes)
    input = MatchingInput(; h_wm_cells = cells, T_f = o.input_T_f)
    plain = o.input_T_f == 0 && all(==(1), cells)
    o.wall_model in (:equilibrium, :nonequilibrium) ||
        error("surface_exchange: wall_model must be :equilibrium or :nonequilibrium, got $(o.wall_model)")
    dynamic = o.z0m === :dynamic
    o.z0m isa Symbol && !dynamic &&
        error("surface_exchange: z0m must be a per-class Dict or :dynamic, got :$(o.z0m)")
    dynamic && o.wall_model !== :equilibrium &&
        error("surface_exchange: z0m = :dynamic is a roughness of the equilibrium wall law (wall_model = :equilibrium)")
    table = dynamic ? default_options(SurfaceExchangeMOST).z0m : o.z0m
    dynamic && !(o.dynamic_refine isa Integer && o.dynamic_refine >= 1) &&
        error("surface_exchange: dynamic_refine must be an integer ≥ 1")
    tble = nothing
    if o.wall_model === :nonequilibrium
        o.wall_formulation === :local ||
            error("surface_exchange: wall_model = :nonequilibrium needs wall_formulation = :local")
        # z0 per column is set in init!, from each surface's own roughness
        tble = TBLEWallModel(; terms = Symbol(o.tble_terms), z0 = zeros(Nx, Ny))
    end
    # the per-class roughness table goes on the bus: the facade wall model
    # (#179) reads the wall and glass z0m from it (with z0m = :dynamic, the
    # default table, which also stays the floor and the surface-layer z0m)
    z0m = Dict{UInt8,Float64}(table)
    bus.z0m_class = z0m
    return SurfaceExchangeMOST(P, o.U_min, z0m, o.z0h_ratio,
                               false, 0.0, 0.0, 0.0, z(), laws, WallModel(laws), fill(Inf, Nx, Ny),
                               z(), Symbol(o.wall_formulation), input, plain,
                               fill(mesh.dz / 2, Nx, Ny), fill(mesh.dz / 2, Nx, Ny), nfb, tble,
                               z(), z(), z(), z(), z(), z(), z(),
                               0.0, 0, 0.0, 0, 0.0, 0,
                               dynamic, nothing,
                               (refine = Int(o.dynamic_refine), filter = Symbol(o.dynamic_filter),
                                C_d = Float64(o.dynamic_C_d)), 0.0, z(), z(),
                               fill(NaN, Nx, Ny), fill(NaN, Nx, Ny), fill(false, Nx, Ny),
                               -Inf, -Inf, 0.0, Any[], zeros(Nx * Ny, 4))
end

"""
    osm_height_map(city, mesh, r) -> Matrix{Float64}

The building height (m) of the OSM `city` on a grid `r` times finer than
`mesh` in x and y (`(r Nx, r Ny)`, cell centres), with each footprint's
true height (the tallest where footprints overlap) — the surface the
IBM staircase of `rasterize_city` represents at the grid scale.
"""
function osm_height_map(city, mesh, r::Integer)
    Nf, Mf = mesh.Nx * r, mesh.Ny * r
    dxf, dyf = mesh.Lx / Nf, mesh.Ly / Mf
    xs = [mesh.x0 + (i - 0.5) * dxf for i in 1:Nf]
    ys = [mesh.y0 + (j - 0.5) * dyf for j in 1:Mf]
    hf = zeros(Nf, Mf)
    for b in city.buildings
        isempty(b.polygon) && continue
        xmin, xmax, ymin, ymax = polygon_bbox(b.polygon)
        i0 = max(1, floor(Int, (xmin - mesh.x0) / dxf)); i1 = min(Nf, ceil(Int, (xmax - mesh.x0) / dxf) + 1)
        j0 = max(1, floor(Int, (ymin - mesh.y0) / dyf)); j1 = min(Mf, ceil(Int, (ymax - mesh.y0) / dyf) + 1)
        for j in j0:j1, i in i0:i1
            point_in_polygon(xs[i], ys[j], b.polygon) && (hf[i, j] = max(hf[i, j], b.height))
        end
    end
    return hf
end

"""
    build_dynamic_roughness(c, case) -> DynamicRoughness

The dynamic roughness of the ground (`z0m = :dynamic`): the sub-filter
statistics of [`osm_height_map`](@ref) about the IBM staircase
``(k_{air}-1)Δz``, each column's surface `z0m` as the floor, the plane
means over the columns with a surface.
"""
function build_dynamic_roughness(c::SurfaceExchangeMOST, case)
    mesh, bus = case.mesh, case.bus
    Nz, dz = mesh.Nz, mesh.dz
    r = c.dyn_opts.refine
    hf = osm_height_map(case.geom.city, mesh, r)
    resolved = [min(k - 1, Nz) * dz for k in bus.k_air]
    # the sub-filter statistics are periodic only on a periodic case (#262);
    # the spectral test filter needs a periodic surface
    periodic = !lateral_open(case.cfg)
    periodic || c.dyn_opts.filter !== :spectral ||
        error("surface_exchange: z0m = :dynamic with dynamic_filter = :spectral needs a periodic " *
              "surface; with lateral = :open use dynamic_filter = :box")
    st = subfilter_height(hf, r; dx = mesh.dx, dy = mesh.dy, resolved = resolved,
                          filter = c.dyn_opts.filter, periodic = periodic)
    have = [bus.surface[I] !== nothing for I in CartesianIndices(bus.k_air)]
    floor_ = [have[I] ? roughness_lengths(bus.surface[I]).z0m : 0.05
              for I in CartesianIndices(bus.k_air)]
    return DynamicRoughness(st; z0_floor = floor_, C_d = c.dyn_opts.C_d, mask = have)
end

"""
    matching_cells(geom, k_air, mesh, n, classes) -> (cells, nfallback)

The matching cell count of the ground wall model per column: `n` for the
columns of a class in `classes` whose air levels `k_air … k_air+n-1` are
all air (inside the domain), 1 elsewhere; `nfallback` counts the columns
of those classes that fall back to 1.
"""
function matching_cells(geom, k_air, mesh, n, classes)
    Nx, Ny, Nz, Hx, Hy, Hz = mesh.Nx, mesh.Ny, mesh.Nz, mesh.Hx, mesh.Hy, mesh.Hz
    cells = ones(Int, Nx, Ny)
    n == 1 && return cells, 0
    nfb = 0
    for j in 1:Ny, i in 1:Nx
        geom.sfc[i, j] in classes || continue
        k = k_air[i, j]
        ok = k + n - 1 <= Nz && all(kk -> !geom.sm.mask[Hx+i, Hy+j, Hz+kk], k:k+n-1)
        ok ? (cells[i, j] = n) : (nfb += 1)
    end
    return cells, nfb
end

"""
    class_roughness(case, site::SurfaceSite) -> (z0m, z0h)

The roughness lengths (m) of a surface of class `site.class`: `site.z0m`
if set (a parent tile over unresolved buildings), otherwise the
`:surface_exchange` provider's per-class `z0m` table (0.05 m for a class
not in it, or when no provider is selected); `site.z0h` if set,
otherwise ``z_{0h} = z_{0m}`` times its `z0h_ratio` (0.1 without a
provider). Every surface factory takes its roughness from here.
"""
function class_roughness(case, site::SurfaceSite)
    se = find_coupling(case, :surface_exchange)
    z0m = site.z0m !== nothing ? site.z0m :
          se === nothing ? 0.05 : get(se.z0m, site.class, 0.05)
    ratio = se === nothing ? 0.1 : se.z0h_ratio
    return z0m, site.z0h !== nothing ? site.z0h : ratio * z0m
end

"The `SlabLand` for a site of a class nobody registered a factory for."
function fallback_slab(case, site::SurfaceSite)
    p, cls = case.bus.params, site.class
    z0m, z0h = class_roughness(case, site)
    return SlabLand(; T = site.T_skin, C_s = p.C_s[cls], albedo = p.albedo[cls],
                    emissivity = p.emissivity[cls], z0m = z0m, z0h = z0h,
                    β = cls == SURFACE_WATER ? 1.0 : 0.0)
end

"""
    class_surface(case, site::SurfaceSite) -> AbstractSurface

**The** class-to-surface factory of the Berlin case: the surface the
coupling registered for `site.class` builds
([`register_surface_factory!`](@ref)), or the surface exchange's
`SlabLand` fallback for a class nobody claims. The surface exchange
builds every child column with it, and the parent's tiled surface
(`parent/surface.jl`, #249) every tile of every parent column, so a
coupling that takes over a class (the park, #140; water, #141) changes
both at once.
"""
function class_surface(case, site::SurfaceSite)
    fac = get(case.bus.factories, site.class, nothing)
    return fac === nothing ? fallback_slab(case, site) : fac[2](case, site)
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
        s = class_surface(case, SurfaceSite(sfc[I]; T_skin = bus.Ts[I], column = I))
        bus.surface[I] = s
        bus.Ts[I] = surface_temperature(s)
        bus.dew_mask[I] = moisture_availability(s) == 0
        c.tble === nothing || (c.tble.z0[I] = roughness_lengths(s).z0m)
    end
    c.groups = surface_groups(bus.surface)
    c.dynamic && (c.dyn = build_dynamic_roughness(c, case))
    @debug(begin
               kinds = Dict{String,Int}()
               for s in bus.surface
                   s === nothing && continue
                   k = string(nameof(typeof(s)))
                   kinds[k] = get(kinds, k, 0) + 1
               end
               @sprintf("Surfaces: %s; every %.0f s",
                        join(("$v $k" for (k, v) in sort!(collect(kinds))), ", "), c.P)
           end, subsystem = :urban, domain = current_domain(), coupling = :surface_exchange)
    return nothing
end

"""
    SurfaceGroup{S}

The columns of one concrete surface type `S`: the surfaces themselves
(`surfs[n]`, the objects of `bus.surface`) and their columns (`ids[n]`).
[`step_all_surfaces!`](@ref) launches one `foreach_column!` per group, so
every launch calls `step_surface!` on one concrete type (no dynamic
dispatch per column; the layout a GPU backend will need, KA-m #301).
"""
struct SurfaceGroup{S}
    surfs::Vector{S}
    ids::Vector{CartesianIndex{2}}
end

"The columns of `surface` grouped by concrete type, each group in column-major order."
function surface_groups(surface::AbstractMatrix)
    by = Dict{DataType,Vector{CartesianIndex{2}}}()
    order = DataType[]
    for I in CartesianIndices(surface)
        s = surface[I]
        s === nothing && continue
        T = typeof(s)
        haskey(by, T) || (by[T] = CartesianIndex{2}[]; push!(order, T))
        push!(by[T], I)
    end
    return Any[SurfaceGroup{T}(T[surface[I] for I in by[T]], by[T]) for T in order]
end

# The work of one surface column in grid-point equivalents (a soil column's
# implicit heat and water solve plus two MOST root solves and two MSHF
# maximisations, tens of µs): the serial threshold of the launch counts it.
const SURFACE_WEIGHT = 1024

"""
    step_all_surfaces!(c, case)

Advance every column's surface by `c.P` under the forcing of its first
air cell and store the fluxes it returns (held until the next call).

The columns are independent (each owns its surface, its workspace and its
entries of the bus and of `c`), so they run as one `foreach_column!` work
item each, one launch per concrete surface type ([`SurfaceGroup`](@ref)).
What the old loop accumulated across columns — the worst storage error,
the worst MSHF ratios, the MSHF time — each column writes to its row of
`c.colterms`, folded in column-major order afterwards: the same bits as
the serial loop, at every thread count (KA-m, #301).
"""
function step_all_surfaces!(c::SurfaceExchangeMOST, case)
    bus, mesh = case.bus, case.mesh
    θ = interior_view(case.sim.θ)
    qv = interior_view(tracer(case, :qv))
    P = c.P
    # the air has received the fluxes of the interval that ends now for
    # τ_interval seconds; it must be P (not checked before the first step)
    c.stepped && (c.interval_err = max(c.interval_err, abs(c.τ_interval - P)))
    c.τ_interval = 0.0
    # P57-e: the dynamic roughness from this surface step's first-air-cell wind
    dyn = c.dyn
    if dyn !== nothing
        t0 = time()
        _dynamic_roughness_step!(dyn, c, case)
        c.t_dyn += time() - t0
    end
    T = c.colterms
    view(T, :, 1:3) .= -Inf          # max folds: a column without a surface adds nothing
    view(T, :, 4) .= 0.0
    dz0 = dyn === nothing ? nothing : dyn.z0
    b = Aeolus.Backends.backend(bus.Ts)
    for g in c.groups
        _step_surface_group!(b, g, c, case, bus, θ, qv, P, dz0)
    end
    c.store_err = foldl(max, view(T, :, 1); init = c.store_err)
    c.mshf_worst = foldl(max, view(T, :, 2); init = c.mshf_worst)
    c.mshf_worst_delivered = foldl(max, view(T, :, 3); init = c.mshf_worst_delivered)
    c.mshf_time += ordered_sum(view(T, :, 4))
    c.stepped = true
    c.t_next += P
    c.nsurface_steps += 1
    return nothing
end

# the first-air-cell wind of every column (pointwise), then the dynamic α
function _dynamic_roughness_step!(dyn, c, case)
    bus, mesh = case.bus, case.mesh
    foreach_column!(Aeolus.Backends.backend(c.ucs), _centre_wind_column!, (1:mesh.Nx, 1:mesh.Ny),
                    c.ucs, c.vcs, case.sim, bus.k_air, mesh.Nz)
    update_dynamic_roughness!(dyn, c.ucs, c.vcs, bus.z_ref)
    return nothing
end

@inline function _centre_wind_column!(i, j, ucs, vcs, sim, k_air, Nz)
    @inbounds k = min(k_air[i, j], Nz)
    @inbounds ucs[i, j], vcs[i, j] = centre_wind(sim, i, j, k)   # BC-aware (#262)
    return nothing
end

_step_surface_group!(b, g::SurfaceGroup, c, case, bus, θ, qv, P, dz0) =
    foreach_column!(b, _surface_column!, (1:length(g.ids),), g.surfs, g.ids, c, case, bus,
                    θ, qv, P, dz0; weight = SURFACE_WEIGHT)

# One surface column (the body of the old serial loop). Writes only column
# I of the bus and of `c`, its own surface, and row `lin` of c.colterms.
function _surface_column!(n, surfs, ids, c, case, bus, θ, qv, P, dz0)
    @inbounds begin
        I = ids[n]
        s = surfs[n]
        s === bus.surface[I] ||
            error("surface_exchange: bus.surface[$I] was replaced after init! (the surface groups are stale)")
        i, j = Tuple(I)
        lin = LinearIndices(bus.k_air)[I]
        k = bus.k_air[i, j]
        q = qv[i, j, k]
        uc, vc = centre_wind(case.sim, i, j, k)                  # BC-aware (#262)
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
        tm = @elapsed (f_most = _most_fluxes(s, f, c.U_min))
        fl = _step_one(s, f, P, Q, c.U_min)
        E1 = surface_store_energy(s)
        E0 === nothing || (c.colterms[lin, 1] = abs((E1 - E0) / P - fl.G))
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
        # (z0m = :dynamic: the law's roughness is the dynamic one; L is the
        # stability of this surface step, recovered with the surface's z0m)
        c.laws[i, j] = MOSTWallLaw(dz0 === nothing ? z0m : dz0[i, j], L; U_min = c.U_min)
        # the law reproduces the surface step's u* with the table z0m; with the
        # dynamic z0 bus.u_star is the wall law's at this wind and height
        dz0 === nothing || (bus.u_star[i, j] = friction_velocity(c.laws[i, j], U, bus.z_ref[i, j]))
        # P58-e: how far the column is from its maximum sustainable heat
        # flux (Van de Wiel et al. 2012) — one 1-D maximisation per column.
        # MOST bounds the fluxes it defines (f_most) by Q_max exactly; the
        # fluxes a soil column returns are its first-order extrapolation in
        # T_skin to the step's mean skin temperature, which can overshoot
        # the maximum of the concave H(T_skin) by O(ΔT²) (tracked apart).
        tm += @elapsed begin
            r = surface_mshf_ratio(BusingerDyerBeljaarsHoltslag(), f_most, f, z0m; U_min = c.U_min)
            rd = surface_mshf_ratio(BusingerDyerBeljaarsHoltslag(), fl, f, z0m; U_min = c.U_min,
                                    Q_max = r.Q_max)
        end
        c.H_max[i, j] = r.H_max
        c.mshf_ratio[i, j] = r.ratio
        c.mshf_beyond[i, j] = r.ratio > 0 && bus.z_ref[i, j] / L > r.ζ_mshf
        c.colterms[lin, 2] = r.ratio
        c.colterms[lin, 3] = rd.ratio
        c.colterms[lin, 4] = tm
        c.U_wall[i, j] = U          # u* is the law at this wind and height until the next step
        c.h_wall[i, j] = bus.z_ref[i, j]
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
    # one work item per (i, j) column, returning its evaporation and whether
    # it wrote into a solid cell; the totals folded in column-major order —
    # the canonical order of a 2-D sum, the old loop's (KA-m, #301)
    Nx, Ny = size(bus.k_air)
    evap, nsolid = column_sum(Aeolus.Backends.backend(bus.E), _deliver_column!, (1:Nx, 1:Ny),
                              θ, qv, bus, c.cp, sm.mask, case, Hx, Hy, Hz, dz, τ)
    c.evap = evap / length(bus.k_air)
    c.solid_writes += Int(nsolid)
    return nothing
end

@inline function _deliver_column!(i, j, θ, qv, bus, cp, mask, case, Hx, Hy, Hz, dz, τ)
    @inbounds begin
        bus.surface[i, j] === nothing && return (0.0, 0.0)
        k = bus.k_air[i, j]
        solid = mask[Hx+i, Hy+j, Hz+k]
        ρk = ref_density(case, k)
        θ[i, j, k] += bus.H[i, j] * τ / dz * ref_θ_per_heat(case, k; cp = cp[i, j])
        qv[i, j, k] += bus.E[i, j] * τ / (ρk * dz)
        return (bus.E[i, j] * τ, solid ? 1.0 : 0.0)
    end
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
    # (k_air > Nz: no surface, no wall); u_τ from this step's (filtered) wind
    if c.tble !== nothing
        # the non-equilibrium wall model: the TBLE columns' stress (P57-c)
        apply_wall_stress!(case.sim, c.tble, dt; side = :bottom, mask = case.bus.k_air,
                           input = c.input, ustar = case.bus.u_star, wind_out = c.U_wall)
        cols, act = tble_columns(c.tble, :bottom)
        @inbounds for I in eachindex(act)
            act[I] && (c.h_wall[I] = c.h_match[I] = cols[I].y[end])
        end
    elseif c.plain_input
        apply_wall_stress!(case.sim, c.wm, dt; side = :bottom, mask = case.bus.k_air,
                           formulation = c.formulation, ustar = case.bus.u_star,
                           wind_out = c.U_wall)
    else
        wind, height = matching_input!(c.input, case.sim; side = :bottom, mask = case.bus.k_air)
        apply_wall_stress!(case.sim, c.wm, dt; side = :bottom, mask = case.bus.k_air,
                           formulation = c.formulation, ustar = case.bus.u_star,
                           wind_out = c.U_wall, wind = wind, height = height)
        @inbounds for I in eachindex(height)
            case.bus.k_air[I] <= case.mesh.Nz && (c.h_wall[I] = c.h_match[I] = height[I])
        end
    end
    return nothing
end

function run_stage!(c::SurfaceExchangeMOST, case, ::Val{:surface_update}, dt)
    bus = case.bus
    # the time filter of the wall-model input: once per accepted step
    update_input_filter!(c.input, case.sim, dt; side = :bottom, mask = bus.k_air)
    # the TBLE columns of the non-equilibrium ground, to the end of the step
    c.tble === nothing ||
        update_wall_model!(c.tble, case.sim, dt; side = :bottom, mask = bus.k_air, input = c.input)
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
    # with the TBLE ground, bus.u_star is the stress the ground applies (the
    # columns' u_τ at the end of the step), not the surface step's MOST u*
    c.tble === nothing || _tble_ustar!(c, case)
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

"""
    diagnostics(c::SurfaceExchangeMOST, case)

Over the columns with a surface: the mean `H`, `LE`, `G` (W m⁻²) and
the mean friction velocity of the roofs `ustar_roof` and the ground
`ustar_ground` (m s⁻¹), the latter two over the
[`analysis_columns`](@ref) only (outside `cfg.analysis_buffer` of the
open faces: they are compared with observations, #270); with the dynamic roughness its `z0_alpha` and
the street's mean `z0_street` (m). P58-e: the stable columns' downward
buoyancy flux over their maximum sustainable heat flux, per class,
`mshf_ratio_<class>` (mean |H|/H_max) and `mshf_beyond_<class>` (the
fraction beyond it) for street, roof and park, the largest
`mshf_worst` and what was delivered `mshf_worst_delivered`; and the wall
time of the MSHF maximisations per surface step `wall.mshf_ms_per_step`.
"""
function diagnostics(c::SurfaceExchangeMOST, case)
    bus, sfc = case.bus, case.geom.sfc
    have = [I for I in eachindex(bus.surface) if bus.surface[I] !== nothing]
    # u* is compared with observations (#270): outside the inflow fetch (#268)
    ana = analysis_columns(case)
    roof = [I for I in have if sfc[I] == SURFACE_ROOF && ana[I]]
    ground = [I for I in have if sfc[I] != SURFACE_ROOF && ana[I]]
    m(x, idx) = () -> isempty(idx) ? NaN : mean(x[idx])
    ds = Diagnostic[
        Diagnostic("H", "W m-2", m(bus.H, have); description = "mean sensible heat flux, upward"),
        Diagnostic("LE", "W m-2", m(bus.LE, have); description = "mean latent heat flux, upward"),
        Diagnostic("G", "W m-2", m(bus.G, have); description = "mean ground heat flux"),
        Diagnostic("ustar_roof", "m s-1", m(bus.u_star, roof)),
        Diagnostic("ustar_ground", "m s-1", m(bus.u_star, ground))]
    if c.dyn !== nothing
        street = [I for I in have if sfc[I] == SURFACE_STREET]
        append!(ds, [Diagnostic("z0_alpha", "1", () -> c.dyn.α),
                     Diagnostic("z0_street", "m", () -> isempty(street) ? NaN : mean(c.dyn.z0[street]))])
    end
    # the stable columns of a class: those whose MSHF ratio is set (> 0)
    for (cls, name) in ((SURFACE_STREET, "street"), (SURFACE_ROOF, "roof"), (SURFACE_PARK, "park"))
        cols = [I for I in have if sfc[I] == cls]
        stable() = [I for I in cols if c.mshf_ratio[I] > 0]
        push!(ds, Diagnostic("mshf_ratio_$name", "1",
                             () -> (idx = stable(); isempty(idx) ? NaN : mean(c.mshf_ratio[idx]));
                             description = "|H|/H_max, mean over the stable $name columns"))
        push!(ds, Diagnostic("mshf_beyond_$name", "1",
                             () -> (idx = stable(); isempty(idx) ? NaN :
                                    count(I -> c.mshf_beyond[I], idx) / length(idx));
                             description = "fraction of the stable $name columns beyond the MSHF"))
    end
    append!(ds, [Diagnostic("mshf_worst", "1", () -> c.mshf_worst),
                 Diagnostic("mshf_worst_delivered", "1", () -> c.mshf_worst_delivered),
                 Diagnostic("wall.mshf_ms_per_step", "ms",
                            () -> 1e3 * c.mshf_time / max(c.nsurface_steps, 1);
                            description = "wall time of the MSHF maximisations per surface step")])
    return ds
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
            _wall_checks(c, case, have)...,
            "the wall law is evaluated at the centre of the column's matching air cell, (n − ½)Δz (1e-12)" =>
                all(I -> abs(c.h_match[I] - (c.input.cells[I] - 0.5) * case.mesh.dz) <= 1e-12 * case.mesh.dz, have),
            "matching cells are air: solid-free from k_air to k_air + n − 1" =>
                all(I -> _matching_air(sm, mesh, I, bus.k_air[I], c.input.cells[I]), have),
            "the time-filtered wall input is primed and finite (or the filter is off)" =>
                (c.input.T_f == 0 || (c.input.primed && all(isfinite, c.input.uf) && all(isfinite, c.input.vf))),
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
            # An empirical bound on the O(ΔT²) overshoot of the linearised
            # skin, not a physical limit. The former < 10 % was measured on
            # the idealised smoke window at 1 m/s (≤ 3.1 %); the nested
            # windows (#265) measured 21 % (a street column on the inflow
            # face, where the parent's air arrives unadjusted to the skins)
            # and 28 % (an interior column under the summer sounding's
            # 3.4 m/s geostrophic wind, which the nest now drives the child
            # with: stronger fluxes, a larger skin change over the 4 s
            # surface period)
            "the delivered fluxes (a soil skin's linearisation in T_skin) exceed H_max by < 50 % (measured ≤ 3.1 % periodic at 1 m/s, ≤ 28 % nested)" =>
                c.mshf_worst_delivered < 1.5,
            "the wall law's z0m is the surface's (table) or, with z0m = :dynamic, √(z0f² + α²σ_Δ²) of the dynamic roughness, solved every surface step (1e-12)" =>
                (c.dyn === nothing ? all(I -> c.laws[I].z0m == roughness_lengths(bus.surface[I]).z0m, have) :
                 c.dyn.nsolves == c.nsurface_steps &&
                 all(I -> c.laws[I].z0m == c.dyn.z0[I] &&
                          abs(c.dyn.z0[I] - sqrt(c.dyn.z0_floor[I]^2 + c.dyn.α^2 * c.dyn.stats.σ2[I])) <=
                          1e-12 * c.dyn.z0[I], have)),
            "dynamic z0 (if on) is finite, ≥ the column's table z0m and below the matching height" =>
                (c.dyn === nothing || all(I -> isfinite(c.dyn.z0[I]) &&
                                               c.dyn.z0[I] >= roughness_lengths(bus.surface[I]).z0m &&
                                               c.dyn.z0[I] < c.h_wall[I], have))]
end

function _tble_ustar!(c::SurfaceExchangeMOST, case)
    cols, act = tble_columns(c.tble, :bottom)
    @inbounds for I in eachindex(act)
        act[I] || continue
        case.bus.u_star[I] = cols[I].u_τ
        c.U_wall[I] = hypot(cols[I].u[end], cols[I].v[end])
        c.h_wall[I] = cols[I].y[end]
    end
    return nothing
end

# the checks of the ground's wall model: the MOST law (equilibrium) or the
# TBLE columns (non-equilibrium, P57-c)
function _wall_checks(c::SurfaceExchangeMOST, case, have)
    bus = case.bus
    if c.tble === nothing
        return ["u* > 0; above its neutral value κU/ln(h/z0m) where H > 5 and E ≥ 0 (unstable), below it where H < −5 and E ≤ 0 (stable), U and h the wind and height of this step's wall law" =>
                    all(I -> _ustar_ok(bus, I, c.U_wall[I], c.U_min, c.h_wall[I], c.laws[I].z0m), have),
                "u* is the ground's MOSTWallLaw at this step's wind and matching height (1e-12)" =>
                    all(I -> abs(bus.u_star[I] - friction_velocity(c.laws[I], c.U_wall[I], c.h_wall[I])) <=
                             1e-12 * bus.u_star[I], have)]
    end
    cols, act = tble_columns(c.tble, :bottom)
    return ["TBLE ground: every column with a surface has a TBLE column, and only those" =>
                all(I -> act[I] == (bus.k_air[I] <= case.mesh.Nz), eachindex(act)),
            "TBLE ground: u* > 0 and finite, the columns' |τ_w|^½ (1e-12)" =>
                all(I -> bus.u_star[I] > 0 && isfinite(bus.u_star[I]) &&
                         abs(bus.u_star[I] - sqrt(hypot(cols[I].τx, cols[I].τy))) <= 1e-12 * bus.u_star[I], have),
            "TBLE ground: the column profiles are finite and 0 at the surface" =>
                all(I -> all(isfinite, cols[I].u) && all(isfinite, cols[I].v) &&
                         cols[I].u[1] == 0 && cols[I].v[1] == 0, have),
            "TBLE ground: each column's roughness is its surface's z0m" =>
                all(I -> cols[I].z0 == roughness_lengths(bus.surface[I]).z0m, have),
            "TBLE ground: the columns were advanced every step after the first" =>
                c.tble.walls[:bottom].nupdates >= 1]
end

function _matching_air(sm, mesh, I, k, n)
    i, j = Tuple(CartesianIndices((mesh.Nx, mesh.Ny))[I])
    return all(kk -> !sm.mask[mesh.Hx+i, mesh.Hy+j, mesh.Hz+kk], k:k+n-1)
end

function _ustar_ok(bus, I, U, U_min, h = bus.z_ref[I], z0m = roughness_lengths(bus.surface[I]).z0m)
    us = bus.u_star[I]
    us > 0 || return false
    neutral = 0.4 * max(U, U_min) / log(h / z0m)
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
