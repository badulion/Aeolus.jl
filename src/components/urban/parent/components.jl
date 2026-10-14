# The urban nest parent's physics as components of a domain (NEST-p1, #481).
#
# `ParentCase` (case.jl) used to wire its physics by hand. Each piece is
# now a component of an ordinary `Components.Domain`, built from options
# and the domain (`Components.build`) and run by the stage pipeline:
#
#   TiledUrbanSurface           the `:surface` slot: the tiled urban mosaic of the
#                          coarse columns (surface.jl, #249), whose land use
#                          is the tile form of the parent's static driver
#                          (`STATIC_TILES_SCHEMA`); stage 3 delivers the
#                          tiles' heat and vapour into the canopy levels,
#                          stage 8 their momentum flux;
#   PrescribedSurfaceFlux  the `:surface` slot: prescribed kinematic heat and
#                          vapour fluxes Q(t), E(t) into the first level (the
#                          idealised parents);
#   MeanColumnRadiation    the `:radiation` slot: `ColumnRadiation` on the
#                          plane-mean column every `period`, its heating
#                          every step, its downwelling flux to the surface;
#   Components.LargeScale  the geostrophic pressure gradient, the nudging with
#                          its boundary-layer-dependent weight
#                          (`BoundaryLayerTracker`), the subsidence and the
#                          θ reference damping;
#   Components.SGS         `[sgs] model = "tke"` with the exact wall distance
#                          of the stretched grid and the surface value of the
#                          tiles' u*.
#
# Each works on both dynamical cores. `parent_domain` (case.jl) builds the
# parent's domain from them; `step_parent!` runs the same component entries
# in the pipeline's order inside the parent's air-heat and water ledgers.
# The physics is in the modules (SurfaceTiles, ColumnRadiation,
# LargeScaleForcing, PrognosticSGS); these are the adapters.

# Urban has a `build`, `stages` and `run_stage!` of its own (the urban
# components' interface): the component interface's are qualified here.

# ---------------------------------------------------------------------------
# the reference profiles of the air: the parent's hydrostatic state on the
# Boussinesq core, the base state on the anelastic one
# ---------------------------------------------------------------------------

"""
    air_reference(opts, domain) -> reference

The reference profiles (`p̄`, `ρ̄`, `π̄` per level and `cp`) a surface or
radiation component converts its fluxes with: `opts.reference` (the
parent's [`CaseReference`](@ref)), or on the anelastic core its base state
(`sim.ref`). A Boussinesq domain needs `reference`.
"""
function air_reference(opts, domain)
    r = Components.option(opts, :reference, nothing)
    r !== nothing && return r
    s = domain.state
    s isa BoussinesqSimulation &&
        throw(ArgumentError("the Boussinesq core has no thermodynamic reference: give `reference` (a CaseReference)"))
    return s.ref
end

_moist(sim) = hasproperty(sim, :has_moisture) && sim.has_moisture

# interior face heights (m, length Nz + 1)
_face_heights(m) = hasproperty(m, :zf) ? Float64[m.zf[m.Hz+k] for k in 1:m.Nz+1] :
                                         Float64[m.z0 + (k - 1) * m.dz for k in 1:m.Nz+1]

# ---------------------------------------------------------------------------
# TiledUrbanSurface
# ---------------------------------------------------------------------------

"""
    TiledUrbanSurface

The `:surface` slot's tiled urban mosaic (NEST-p1, #481; the parent
surface of #249): every column of the domain is a [`ParentColumn`](@ref)
of the [`ParentSurface`](@ref) `surface`, one tile per land-use class with
its area fraction, the unresolved buildings' Macdonald roughness and
displacement. Its land use is the tile form of the domain's static driver
(`InputDrivers.read_static_tiles`, `STATIC_TILES_SCHEMA`); the tiles are
those of a class-to-surface factory ([`class_surface`](@ref)).

Stage 3 (`:surface_flux`) delivers over the step `[t, t + Δt]` the sources
held from the last surface step, and every `period` seconds steps every
tile under the air of its column ([`step_parent_column!`](@ref): ``T = θ
Π̄_k``, ``q_v``, the cell-centre wind, ``p̄_k`` at each tile's forcing
level; the downwelling radiation of the domain's `:radiation` component)
and turns the tiles' fluxes into the sources of the canopy levels below
each tile's forcing level ``k_i``, uniformly per volume over their depth
``Z_i``: ``\\dot θ_k = f_i H_i/(Z_i ρ̄_k c_p Π̄_k)``, ``\\dot q_k = f_i
E_i/(Z_i ρ̄_k)`` and a deceleration ``f_i τ_i/(Z_i ρ̄_k)``. Stage 8
(`:post_predictor`) applies that deceleration implicitly along the
instantaneous wind, ``u ← u/(1 + Δt a/U)`` with ``U ≥ U_{min}``. The
friction velocity ``u_* = \\sqrt{Σ f_i τ_i/ρ̄_1}`` is the surface value of
a TKE closure ([`surface_friction_velocity`](@ref)).

Fields: `surface`, `ref` ([`air_reference`](@ref)), `period` (s), `U_min`
(m s⁻¹), the held sources `θsrc`, `qsrc`, `drag` (per column and canopy
level), `ustar`, `t_surf` (the end of the current surface interval), the
downwelling `sw_down`, `lw_down` (W m⁻²; set by the radiation, the
options' values until then), `sfc_mean` (domain-mean J m⁻² of `Rn`, `H`,
`LE`, `G`, `E` and what the air received, `delivered_H`, `delivered_E`),
and the level weights `wθ`, `wq` of the delivered amounts.
"""
mutable struct TiledUrbanSurface{R} <: AbstractComponent
    surface::ParentSurface
    ref::R
    period::Float64
    U_min::Float64
    θsrc::Array{Float64,3}
    qsrc::Array{Float64,3}
    drag::Array{Float64,3}
    ustar::Matrix{Float64}
    t_surf::Base.RefValue{Float64}
    sw_down::Base.RefValue{Float64}
    lw_down::Base.RefValue{Float64}
    sfc_mean::Dict{Symbol,Float64}
    wθ::Vector{Float64}
    wq::Vector{Float64}
end

Components.component_name(::Type{<:TiledUrbanSurface}) = :tiled_surface
Components.section(::Type{<:TiledUrbanSurface}) = "surface"
Components.slot(::Type{<:TiledUrbanSurface}) = :surface
Components.schema(::Type{<:TiledUrbanSurface}) =
    Config.slot_section("surface"; key = "model") do sec
        Config.provider!(sec, "tiles"; doc = "The tiled urban mosaic of the domain's static driver (its tile form).") do s
            Config.option!(s, "period", Float64; default = 60.0, range = (0, Inf), open_lower = true, units = "s",
                           doc = "step of the surface tiles")
            Config.option!(s, "u_min", Float64; default = 0.5, range = (0, Inf), units = "m s^-1",
                           doc = "wind floor of the surface layer")
            Config.option!(s, "sw_down", Float64; default = 0.0, units = "W m^-2",
                           doc = "downwelling shortwave without a :radiation component (and before its first call)")
            Config.option!(s, "lw_down", Float64; default = 350.0, units = "W m^-2",
                           doc = "downwelling longwave without a :radiation component (and before its first call)")
        end
    end

const _SFC_KEYS = (:Rn, :H, :LE, :G, :E, :delivered_H, :delivered_E)

"""
    build(TiledUrbanSurface, opts, domain)

The tiled surface of `domain`: its land use `opts.landuse` (a
[`ParentLanduse`](@ref); else the domain's static driver's tile form,
`domain.drivers.static_tiles`, a `StaticTiles`), its tiles from the
class-to-surface factory `opts.classes` (a built urban site:
[`class_surface`](@ref)) with the factory's child under its footprint,
or the [`ParentSurface`](@ref) `opts.surface` as built; `period`,
`u_min`, `sw_down`, `lw_down` and `reference` ([`air_reference`](@ref)).
"""
function Components.build(::Type{<:TiledUrbanSurface}, opts, domain)
    m = Components.grid(domain)
    ps = Components.option(opts, :surface, nothing)
    if ps === nothing
        pl = Components.option(opts, :landuse, nothing)
        if pl === nothing
            t = get(domain.drivers, :static_tiles, nothing)
            t === nothing && throw(ArgumentError("TiledUrbanSurface: the domain has no static driver with tiles"))
            pl = landuse_of_tiles(t)
        end
        ps = build_parent_surface(opts.classes, pl; z_centres = level_centres(m))
    end
    (size(ps.columns) == (m.Nx, m.Ny)) ||
        throw(DimensionMismatch("TiledUrbanSurface: $(size(ps.columns)) columns on a $(m.Nx) × $(m.Ny) grid"))
    ref = air_reference(opts, domain)
    kmax = maximum(max(1, maximum(col.ktile) - 1) for col in ps.columns if col !== nothing)
    z = zeros(m.Nx, m.Ny, kmax)
    hm = LevelMeter(m, ref.ρ̄, ref.π̄, ref.cp)
    wm = LevelMeter(m, ref.ρ̄, ones(m.Nz), 1.0)
    return TiledUrbanSurface(ps, ref, Float64(Components.option(opts, :period, 60.0)),
                        Float64(Components.option(opts, :u_min, 0.5)), copy(z), copy(z), copy(z),
                        fill(NaN, m.Nx, m.Ny), Ref(-Inf), Ref(Float64(Components.option(opts, :sw_down, 0.0))),
                        Ref(Float64(Components.option(opts, :lw_down, 350.0))),
                        Dict{Symbol,Float64}(k => 0.0 for k in _SFC_KEYS), copy(hm.w), copy(wm.w))
end

"The land use of the tile form of a static driver (`StaticTiles`) as a [`ParentLanduse`](@ref)."
landuse_of_tiles(t::StaticTiles) =
    ParentLanduse(length(t.x), length(t.y), t.dx, t.x0, t.y0, t.refinement, Int.(t.class_count),
                  t.lambda_f, t.h_mean, t.h_max, copy(t.notes))

Components.stages(::TiledUrbanSurface) = (Stage(:surface_flux), Stage(:post_predictor))
Components.surface_friction_velocity(c::TiledUrbanSurface, d) = c.ustar

"""
    set_downwelling!(surface, sw, lw)

Give a surface component the downwelling short- and longwave (W m⁻²) of
the radiation's last call; a no-op for a surface that takes none.
"""
set_downwelling!(::Any, sw, lw) = nothing
set_downwelling!(c::TiledUrbanSurface, sw, lw) = (c.sw_down[] = sw; c.lw_down[] = lw; nothing)

"""
    surface_optics(surface, θ1, π̄1) -> (ε, α, T_rad)

The emissivity, albedo and radiative temperature (K) of a domain's surface
for the column radiation: the tile means of a [`TiledUrbanSurface`](@ref) (the
emissivity-weighted radiative temperature of all tiles), else 0.95, 0.15
and the first level's air temperature ``θ_1 Π̄_1``.
"""
surface_optics(::Any, θ1, π̄1) = (0.95, 0.15, θ1 * π̄1)
function surface_optics(c::TiledUrbanSurface, θ1, π̄1)
    cols = [col for col in c.surface.columns if col !== nothing]
    εs = [surface_emissivity(col.surface) for col in cols]
    ε = sum(εs) / length(cols)
    α = sum(surface_albedo(col.surface) for col in cols) / length(cols)
    Trad = (sum(e * surface_temperature(col.surface)^4 for (e, col) in zip(εs, cols)) / sum(εs))^(1 / 4)
    return ε, α, Trad
end

function _cell_speed(sim, i, j, k)
    m = sim.mesh
    I, J, K = m.Hx + i, m.Hy + j, m.Hz + k
    return hypot(0.5 * (sim.u.data[I, J, K] + sim.u.data[I+1, J, K]),
                 0.5 * (sim.v.data[I, J, K] + sim.v.data[I, J+1, K]))
end

# the work of one column (up to six tiles, each a surface step) in
# grid-point equivalents, for the launch's serial threshold
const TILE_COLUMN_WEIGHT = 4096

"""
    step_tiles!(c::TiledUrbanSurface, sim, period) -> c

Advance every tile of every column by `period` seconds under the air of
its column and turn the tiles' fluxes into the sources the air receives
until the next surface step (see [`TiledUrbanSurface`](@ref)).
"""
function step_tiles!(c::TiledUrbanSurface, sim, period::Real)
    m = sim.mesh
    zf = _face_heights(m)
    fill!(c.θsrc, 0.0); fill!(c.qsrc, 0.0); fill!(c.drag, 0.0)
    kmax = size(c.θsrc, 3)
    θd = sim.θ.data
    qd = _moist(sim) ? sim.qv.data : nothing
    fill_halos!(sim.u, sim.bcs); fill_halos!(sim.v, sim.bcs)
    # one work item per column (KA-m, #301): a column steps its own tiles
    # and writes only its own sources and u*; the area sums of its fluxes
    # are its partials, folded in column-major order (the same bits at
    # every thread count)
    sRn, sH, sLE, sG, sE = column_sum(Aeolus.Backends.backend(c.θsrc), _tile_column!,
                                      (1:m.Nx, 1:m.Ny), c, sim, θd, qd, zf, kmax, period;
                                      weight = TILE_COLUMN_WEIGHT)
    n = m.Nx * m.Ny
    for (key, v) in ((:Rn, sRn), (:H, sH), (:LE, sLE), (:G, sG), (:E, sE))
        c.sfc_mean[key] += v / n * period
    end
    return c
end

function _tile_column!(I, J, c, sim, θd, qd, zf, kmax, period)
    m, ref, ps = sim.mesh, c.ref, c.surface
    Hx, Hy, Hz = m.Hx, m.Hy, m.Hz
    col = ps.columns[I, J]
    kf = maximum(col.ktile)
    U_min = c.U_min
    air = [SurfaceForcing(; sw_down = c.sw_down[], lw_down = c.lw_down[],
                          T_air = θd[Hx+I, Hy+J, Hz+k] * ref.π̄[k],
                          q_air = qd === nothing ? 0.0 : max(qd[Hx+I, Hy+J, Hz+k], 0.0),
                          U = max(_cell_speed(sim, I, J, k), U_min), p = ref.p̄[k], z_ref = 10.0)
           for k in 1:kf]
    agg = step_parent_column!(ps, col, air, period; U_min = U_min)
    tf = last_tile_fluxes(_mosaic(col))
    fr = mosaic_fractions(col.surface)
    τsum = 0.0
    for (n, f) in enumerate(tf)
        nl = max(1, col.ktile[n] - 1)
        Z = zf[nl+1]
        for k in 1:min(nl, kmax)
            c.θsrc[I, J, k] += fr[n] * f.H / (Z * ref.ρ̄[k] * ref.cp * ref.π̄[k])
            c.qsrc[I, J, k] += fr[n] * f.E / (Z * ref.ρ̄[k])
            c.drag[I, J, k] += fr[n] * f.τ / (Z * ref.ρ̄[k])
        end
        τsum += fr[n] * f.τ
    end
    c.ustar[I, J] = sqrt(max(τsum, 0.0) / ref.ρ̄[1])
    return (agg.Rn, agg.H, agg.LE, agg.G, agg.E)
end

# add θ, qv sources (per s) over τ seconds into the canopy levels
function _apply_sources!(c::TiledUrbanSurface, sim, τ::Real)
    m = sim.mesh
    Hx, Hy, Hz = m.Hx, m.Hy, m.Hz
    θd = sim.θ.data
    qd = _moist(sim) ? sim.qv.data : nothing
    @inbounds for k in axes(c.θsrc, 3)
        sθ = 0.0; sq = 0.0
        for j in 1:m.Ny, i in 1:m.Nx
            θd[Hx+i, Hy+j, Hz+k] += c.θsrc[i, j, k] * τ
            qd === nothing || (qd[Hx+i, Hy+j, Hz+k] += c.qsrc[i, j, k] * τ)
            sθ += c.θsrc[i, j, k]; sq += c.qsrc[i, j, k]
        end
        # what the sources delivered, by the level weights (J m⁻², kg m⁻²)
        c.sfc_mean[:delivered_H] += c.wθ[k] * sθ * τ
        c.sfc_mean[:delivered_E] += c.wq[k] * sq * τ
    end
    return c
end

# stage 3: the held sources up to the end of the current surface interval,
# then a surface step and its sources for the rest of the step (the air
# receives exactly what the tiles gave away over the time it covers)
function Components.run_stage!(c::TiledUrbanSurface, d, ::Val{:surface_flux}, dt)
    sim = d.state
    P = c.period
    s = Float64(sim.t)
    te = s + dt
    while s < te - 1e-9 * dt
        if s >= c.t_surf[] - 1e-9 * P
            step_tiles!(c, sim, P)
            c.t_surf[] = s + P
        end
        τ = min(te, c.t_surf[]) - s
        _apply_sources!(c, sim, τ)
        s += τ
    end
    return nothing
end

# stage 8: the tiles' momentum flux as an implicit deceleration of the
# canopy levels (the mean of the two columns a face separates), along the
# instantaneous wind: the stress magnitude held from the surface step,
# stable for any Δt
function Components.run_stage!(c::TiledUrbanSurface, d, ::Val{:post_predictor}, Δt)
    sim = d.state
    m = sim.mesh
    Hx, Hy, Hz, Nx, Ny = m.Hx, m.Hy, m.Hz, m.Nx, m.Ny
    ud, vd = sim.u.data, sim.v.data
    a = c.drag
    U_min = c.U_min
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
    return nothing
end

Aeolus.IO.checkpoint_state(c::TiledUrbanSurface) = Aeolus.IO.structural_state(c)
Aeolus.IO.restore_checkpoint_state!(c::TiledUrbanSurface, s::Aeolus.IO.StructuralState) = Aeolus.IO.restore_structural_state!(c, s)

# ---------------------------------------------------------------------------
# PrescribedSurfaceFlux
# ---------------------------------------------------------------------------

"""
    PrescribedSurfaceFlux(Q, E, ref)

The `:surface` slot's prescribed kinematic fluxes (NEST-p1, #481; the
idealised parents of #259): the heat flux `Q(t)` (K m s⁻¹) and, on a
moist core, the vapour flux `E(t)` (kg kg⁻¹ m s⁻¹), functions of the
model time — a [`ConstantForcing`](@ref), a [`SeriesForcing`](@ref) or
any Julia function. In stage 3 (`:surface_flux`) their exact means over
the step (`LargeScaleForcing.forcing_mean`) enter the first level, ``Δθ_1
= \\bar Q Δt/Δz_1``; no momentum flux. `sfc_mean[:H]` counts the heat in
J m⁻² (``ρ̄_1 c_p Π̄_1`` of the reference, [`air_reference`](@ref)). It
reports no wall flux to a TKE closure (the parents' closure has taken
none from it since #259).
"""
struct PrescribedSurfaceFlux{Q,E,R} <: AbstractComponent
    Q::Q
    E::E
    ref::R
    sfc_mean::Dict{Symbol,Float64}
end

Components.component_name(::Type{<:PrescribedSurfaceFlux}) = :prescribed_surface_flux
Components.section(::Type{<:PrescribedSurfaceFlux}) = "surface"
Components.slot(::Type{<:PrescribedSurfaceFlux}) = :surface
Components.schema(::Type{<:PrescribedSurfaceFlux}) =
    Config.slot_section("surface"; key = "model") do sec
        Config.provider!(sec, "prescribed"; doc = "Prescribed kinematic heat and vapour fluxes into the first level " *
                                                  "(the functions of time q_surface, e_surface of the build options).")
    end

Components.build(::Type{<:PrescribedSurfaceFlux}, opts, domain) =
    PrescribedSurfaceFlux(opts.q_surface, opts.e_surface, air_reference(opts, domain),
                          Dict{Symbol,Float64}(k => 0.0 for k in _SFC_KEYS))

Components.stages(::PrescribedSurfaceFlux) = (Stage(:surface_flux),)

function Components.run_stage!(c::PrescribedSurfaceFlux, d, ::Val{:surface_flux}, dt)
    sim = d.state
    t = sim.t
    dz1 = level_thickness(sim.mesh)[1]
    moist = _moist(sim)
    Q = forcing_mean((z, s) -> c.Q(s), 0.0, t, t + dt)
    E = moist ? forcing_mean((z, s) -> c.E(s), 0.0, t, t + dt) : 0.0
    a = view(interior_view(sim.θ), :, :, 1)
    a .+= Q * dt / dz1
    moist && (view(interior_view(sim.qv), :, :, 1) .+= E * dt / dz1)
    c.sfc_mean[:H] += Q * dt * c.ref.ρ̄[1] * c.ref.cp * c.ref.π̄[1]
    return nothing
end

Aeolus.IO.checkpoint_state(c::PrescribedSurfaceFlux) = (sfc_mean = copy(c.sfc_mean),)
Aeolus.IO.restore_checkpoint_state!(c::PrescribedSurfaceFlux, s::NamedTuple) = (merge!(c.sfc_mean, s.sfc_mean); c)

# ---------------------------------------------------------------------------
# MeanColumnRadiation
# ---------------------------------------------------------------------------

"""
    SiteSun(cfg)

The sun of an urban site's clock as a function of the model time `t`:
`solar_position_at(cfg, hour_of_day(cfg, t))` (the site's date, location
and local clock).
"""
struct SiteSun
    cfg::NamedTuple
end
(s::SiteSun)(t) = solar_position_at(s.cfg, hour_of_day(s.cfg, t))

"""
    MeanColumnRadiation

The `:radiation` slot's column radiation of the plane-mean column
(NEST-p1, #481; the parent's radiation since #259): every `period`
seconds `ColumnRadiation.column_radiation!` on the domain's plane-mean
θ and q_v (`stitched_column` on the domain's levels, under a standard
atmosphere of humidity `humid`), at the sun `sun(t)` ([`SiteSun`](@ref)),
over the surface of the domain's `:surface` component
([`surface_optics`](@ref)). Its downwelling short- and longwave at the
ground force that surface until the next call ([`set_downwelling!`](@ref)),
and its heating ``\\dot T`` warms θ every step by ``\\dot T Δt / Π̄``
(`Radiation.apply_radiative_cooling!`).

It runs in stage 3 (`:surface_flux`) after the surface (stage order 1):
the surface steps under the downwelling flux of the last call, then the
radiation sees the surface it has just stepped, as the parent has done
since #259; its heating is applied there too, before the core step. The
shortwave at the ground is that of the column without the model's
condensate (`fsw_clear`).
"""
mutable struct MeanColumnRadiation{M,S} <: AbstractComponent
    model::M
    sun::S
    humid::Symbol
    period::Float64
    π̄::Vector{Float64}
    t_rad::Base.RefValue{Float64}
    sw_down::Base.RefValue{Float64}
    lw_down::Base.RefValue{Float64}
end

Components.component_name(::Type{<:MeanColumnRadiation}) = :mean_column_radiation
Components.section(::Type{<:MeanColumnRadiation}) = "radiation"
Components.slot(::Type{<:MeanColumnRadiation}) = :radiation
Components.schema(::Type{<:MeanColumnRadiation}) =
    Config.slot_section("radiation"; key = "scheme") do sec
        Config.provider!(sec, "mean_column"; doc = "Two-stream radiation of the plane-mean column (ColumnRadiation).") do s
            Config.option!(s, "period", Float64; default = 600.0, range = (0, Inf), open_lower = true, units = "s",
                           doc = "period of the column radiation")
            Config.option!(s, "humid", Symbol; default = :midlatitude_summer,
                           allowed = [:midlatitude_summer, :subarctic_winter],
                           doc = "the humidity of the standard atmosphere above the domain")
        end
    end

"""
    build(MeanColumnRadiation, opts, domain)

The column radiation of `domain`: `period`, `humid`, `sun` (a function of
the model time, [`SiteSun`](@ref)), `p_surface` (Pa; default the
reference's, [`air_reference`](@ref)) and `reference`.
"""
function Components.build(::Type{<:MeanColumnRadiation}, opts, domain)
    m = Components.grid(domain)
    ref = air_reference(opts, domain)
    ps = Components.option(opts, :p_surface, nothing)
    ps === nothing && (ps = hasproperty(ref, :p_surface) ? ref.p_surface : ref.p̄[1])
    return MeanColumnRadiation(column_radiation_solver(m.Nz, ps), opts.sun,
                               Components.option(opts, :humid, :midlatitude_summer),
                               Float64(Components.option(opts, :period, 600.0)), collect(Float64, ref.π̄),
                               Ref(0.0), Ref(0.0), Ref(350.0))
end

Components.stages(::MeanColumnRadiation) = (Stage(:surface_flux; order = 1),)

"""
    radiate!(c::MeanColumnRadiation, d) -> c

One call of the column radiation of `c` on the plane-mean column of the
domain `d`, and its downwelling flux to the domain's surface.
"""
function radiate!(c::MeanColumnRadiation, d)
    sim = d.state
    m = sim.mesh
    sp = c.sun(sim.t)
    θ = Components.plane_mean(sim.θ)
    qv = _moist(sim) ? Components.plane_mean(sim.qv) : zeros(m.Nz)
    sfc = Components.slot_component(d, :surface)
    ε, α, Trad = surface_optics(sfc, θ[1], c.π̄[1])
    dz = level_thickness(m)
    col = stitched_column(c.model, θ, qv, zeros(m.Nz), dz; T_rad = Trad, humid = c.humid)
    column_radiation!(c.model, col, sp, ε, α)
    c.lw_down[] = c.model.flw.flux_dn[1]
    c.sw_down[] = c.model.fsw_clear === nothing ? 0.0 : c.model.fsw_clear.flux_dn[1]
    set_downwelling!(sfc, c.sw_down[], c.lw_down[])
    return c
end

function Components.run_stage!(c::MeanColumnRadiation, d, ::Val{:surface_flux}, dt)
    sim = d.state
    t = sim.t
    if t >= c.t_rad[] - 1e-6
        radiate!(c, d)
        c.t_rad[] = (floor(t / c.period + 1e-9) + 1) * c.period
    end
    apply_radiative_cooling!(sim.θ, c.model.heat .* dt ./ c.π̄)
    c.model.dt_applied = dt
    return nothing
end

Aeolus.IO.checkpoint_state(c::MeanColumnRadiation) = Aeolus.IO.structural_state(c)
Aeolus.IO.restore_checkpoint_state!(c::MeanColumnRadiation, s::Aeolus.IO.StructuralState) = Aeolus.IO.restore_structural_state!(c, s)
