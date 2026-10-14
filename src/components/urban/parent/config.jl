# The Berlin parent case (N-parent-e, #259): configuration.
#
# The parent is a deep, periodic, idealised diurnal Aeolus run around the
# child domain: 6 × 6 × 3 km, Δx = 2 Δx_child = 31.25 m, Δz = 4 Δz_child =
# 25 m uniform up to 500 m (so the ~400 m deepened child, maintainer
# decision of 2026-10-05, sits in the uniform part), then stretched by 1.08
# per level to at most 100 m at 3 km (53 levels, `surface_refined_faces`
# of #151). Everything the parent shares with the child — the date, the
# clock, the location, θ₀, f, the initial state and its sounding — comes
# from the child's site options (`site_options`).

"""
    ParentConfig(child::NamedTuple = site_options(default_run_config()); kwargs...)

Configuration of the Berlin nesting parent. The defaults are those of
the parent's configuration schema (`[nesting] parent` of an urban
site, `site/config_schema.jl`, CFG-c #358, ONE-e #454); in TOML the
keys are the ASCII names of the fields (`nudging_tau`, `theta_top`,
`q_surface`, …, `aeolus options`). The date, the clock
(`start_h`, `utc_offset`), the location (`lat0`, `lon0`, `elevation`),
`θ₀`, `g`, `f_coriolis`, the sounding and the initial θ(z), qv(z) and
ground pressure are the child's (`initial_θ`, `initial_qv`,
`case_surface_pressure` of `child`), so the two domains start from one
state.

Groups of keywords:

* **placement** — `center` (m, the horizontal centre of the parent box
  relative to the OSM anchor; `nothing`, the default, centres it on the
  child, `child.offset`). The parent mesh is placed in the child mesh's
  coordinates ([`parent_origin`](@ref)), so the child nests in it at its
  own origin, and the land use is cropped to the box
  ([`crop_landuse`](@ref)).
* **grid** — `extent` (m, default 6000 × 6000 × 3000), `refinement`
  `(rx, ry, rz)`: the parent cell is `rx × ry × rz` child cells (default
  `(2, 2, 4)`: Δx = 31.25 m, Δz = 25 m); `vertical`: `:stretched`
  (default; Δz uniform up to `z_uniform` = 500 m, then growing by
  `ratio` = 1.08 per level up to `dz_max` = 100 m) or `:uniform` (Δz
  everywhere; with `refinement = (4, 4, 8)` this is the 62.5 m × 50 m
  fallback of the issue, with `(2, 2, 4)` the full-resolution uniform
  alternative, 192 × 192 × 120); `halo`.
* **surface** — `surface`: `:tiles` (default; the tiled urban mosaic of
  #249, [`build_parent_surface`](@ref), forced by the parent's air and
  radiation) or `:prescribed` (a kinematic heat flux `Q_surface(t)`,
  K m s⁻¹, and moisture flux `E_surface(t)`, kg kg⁻¹ m s⁻¹, both
  functions of the simulation time in s: a [`ConstantForcing`](@ref)
  (`{kind = "constant", value}` in TOML), a [`SeriesForcing`](@ref)
  (`{kind = "series", t, value}`, linear in time) or any Julia function,
  which makes the run not replayable; no momentum flux — the idealised
  CBL tests); `surface_period` (s, the tiles' step), `U_min`
  (m s⁻¹, the wind floor of the surface layer).
* **radiation** — `radiation` (`true`: the column radiation of the child,
  `column_radiation!`, on the parent's mean column every `radiation_period`
  s).
* **large-scale forcing** — `geostrophic` (the `GeostrophicPressureGradient`
  of the sounding wind above `z_g`, see [`parent_geostrophic_wind`](@ref);
  without a sounding, the child's `:geostrophic` coupling options
  `(U_g, V_g)`, so the two domains are driven by one wind),
  `nudging_τ` (s; `Inf` = off) with `nudging_factor`, `nudging_depth`
  (the `NudgingWeight` above `nudging_factor` × the running maximum of the
  diagnosed z_i), `divergence` (s⁻¹, the large-scale divergence of the
  subsidence ``w_s = -D z``; default 0 = off, see below), `damping_rate` (s⁻¹) and
  `damping_frac` (the θ `ReferenceDamping` toward the sounding and the
  core's velocity sponge over the top `damping_frac` of the domain),
  `θ_top` (`:initial_gradient`, the fixed-gradient top of #254, or
  `:noflux`).

  The subsidence default is `divergence = 0` (maintainer decision of
  2026-10-06). The epic's planned D = 3e-6 s⁻¹ stays available, but it is
  currently unstable: the 24 h Lindenberg run diverges during the
  convective day (`step_adaptive!` reaches `dt_min`), while D = 0
  completes 24 h on the fallback grid with the ledger closed to
  1e-14 W/m². Tracked in issue #312; the 24 h validation and the final
  choice of D are deferred to issue #315 (after
  the KernelAbstractions port).
* **initial state** — from `child`, unless `θ_init` / `qv_init` (functions
  of z: a [`ProfileTable`](@ref), `{kind = "table", z, values}` in TOML,
  or any Julia function) override it (the idealised tests); `u_init`
  (`:geostrophic`, a number or a function of z), `perturbation` (K, uniform random amplitude in the lowest
  `perturbation_depth` m), `seed`.
* **time** — `hours` (simulated length), `output_period` (s, the
  diagnostics' cadence), the `AdaptiveStep` constraints of the child
  (`TimeIntegration.CFLConstraints` from its `default_options`, `dt_max`
  capped at the shortest period).
* `sgs` — `:tke` (the child's default `:sgs` provider, the Deardorff /
  Moeng closure, here on the stretched grid) or `:smagorinsky` (the
  core's built-in closure, `C_s` of the child).
* `stepper` — `:rk3` (default: `step_rk3!`, SSP-RK3 with a projection per
  stage, under `step_adaptive!`) or `:euler` (the child's forward-Euler
  projection `step!`). The forward-Euler step with the core's centred
  advection is unconditionally unstable for advection; the child's SGS
  diffusion damps it, but at the parent's 31.25 m and 62.5 m grids it did
  not: a grid-scale cold anomaly at the second level grew to −17 K within
  140 s at 31.25 m, and both 62.5 m runs failed after 5 h (the
  projection's divergence check, at the step floor). SSP-RK3 is stable for
  centred advection up to a Courant number of √3.
* `C_advective` — the advective Courant number of the step pick (default
  0.9 with RK3; the child's forward-Euler `AdaptiveStep` uses 0.5). Below
  1 so that the positive TKE transport (`advect_positive!`, first-order
  CFL ≤ 1 per stage) keeps its guarantee; the other constraints are the
  child's `default_options(AdaptiveStep)`.
* `moisture` — carry qv (default true).

What is printed is the installed logger's choice (#347): the build summary
is a `Debug` record, each output of [`run_parent!`](@ref) an `Info` one.
"""
mutable struct ParentConfig
    child::NamedTuple
    # placement
    center::Union{Nothing,NTuple{2,Float64}}   # nothing: on the child
    # grid
    extent::NTuple{3,Float64}
    refinement::NTuple{3,Int}
    vertical::Symbol
    z_uniform::Float64
    ratio::Float64
    dz_max::Float64
    halo::NTuple{3,Int}
    # surface
    surface::Symbol
    Q_surface::Any                # a function of t (s): ConstantForcing, SeriesForcing or any Julia function
    E_surface::Any
    surface_period::Float64
    U_min::Float64
    # radiation
    radiation::Bool
    radiation_period::Float64
    # large-scale forcing
    geostrophic::Bool
    z_g::Float64                  # NaN: 1.2 × the sounding's largest z_i
    nudging_τ::Float64
    nudging_factor::Float64
    nudging_depth::Float64
    divergence::Float64           # 3e-6 diverges by day, #312
    damping_rate::Float64
    damping_frac::Float64
    θ_top::Symbol
    # initial state
    θ_init::Any                   # nothing (the child's sounding) or a function of z
    qv_init::Any
    u_init::Any                   # :geostrophic, a number or a function of z
    perturbation::Float64
    perturbation_depth::Float64
    seed::Int
    # time
    hours::Float64
    output_period::Float64
    # numerics
    sgs::Symbol
    stepper::Symbol
    C_advective::Float64
    moisture::Bool
end

# The defaults live in the parent's configuration schema
# (`parent_section` in site/config_schema.jl), the one place they are
# written; `ParentConfig(; kw...)` starts from them.
function ParentConfig(; child::NamedTuple = site_options(default_run_config()), kw...)
    d = parent_defaults()
    for (k, v) in kw
        k === :child && continue
        haskey(d, k) || throw(ArgumentError("ParentConfig has no field $k" *
                                            " (fields: $(join(sort!(string.(collect(keys(d)))), ", ")))"))
        d[k] = v
    end
    return ParentConfig(child, (d[f] for f in fieldnames(ParentConfig)[2:end])...)
end

ParentConfig(child::NamedTuple; kw...) = ParentConfig(; child = child, kw...)

"""
    ConstantForcing(value)

A surface forcing of the parent that does not change in time:
`f(t) = value`. The TOML form `{kind = "constant", value = …}` of
`q_surface` / `e_surface` (CFG-c, #358).
"""
struct ConstantForcing
    value::Float64
end
(f::ConstantForcing)(t) = f.value

"""
    SeriesForcing(t, value)

A surface forcing of the parent given at the times `t` (s, strictly
increasing): linear in time between them, constant beyond the ends. The
TOML form `{kind = "series", t = […], value = […]}` (CFG-c, #358).
"""
struct SeriesForcing
    t::Vector{Float64}
    value::Vector{Float64}
end
(f::SeriesForcing)(t) = _interp1(f.t, f.value, t)
Base.:(==)(a::SeriesForcing, b::SeriesForcing) = a.t == b.t && a.value == b.value
Base.isequal(a::SeriesForcing, b::SeriesForcing) = isequal(a.t, b.t) && isequal(a.value, b.value)
Base.hash(a::SeriesForcing, h::UInt) = hash(a.value, hash(a.t, hash(:SeriesForcing, h)))

"""
    ProfileTable(z, values)

A vertical profile of the parent's initial state, linear in `z` (m,
strictly increasing) between the given heights and constant beyond the
ends. The TOML form `{kind = "table", z = […], values = […]}` of
`theta_init`, `qv_init` and `u_init` (CFG-c, #358).
"""
struct ProfileTable
    z::Vector{Float64}
    values::Vector{Float64}
end
(p::ProfileTable)(z) = _interp1(p.z, p.values, z)
Base.:(==)(a::ProfileTable, b::ProfileTable) = a.z == b.z && a.values == b.values
Base.isequal(a::ProfileTable, b::ProfileTable) = isequal(a.z, b.z) && isequal(a.values, b.values)
Base.hash(a::ProfileTable, h::UInt) = hash(a.values, hash(a.z, hash(:ProfileTable, h)))

"Linear interpolation of `(xs, ys)` at `x`, constant beyond the ends."
function _interp1(xs, ys, x)
    x <= xs[1] && return ys[1]
    x >= xs[end] && return ys[end]
    i = searchsortedlast(xs, x)
    w = (x - xs[i]) / (xs[i+1] - xs[i])
    return (1 - w) * ys[i] + w * ys[i+1]
end

"Horizontal and surface vertical spacing (m) of the parent: the child's spacing × the refinement."
function parent_spacing(pc::ParentConfig)
    c = pc.child
    dxc, dyc, dzc = c.extent ./ c.size
    rx, ry, rz = pc.refinement
    return rx * dxc, ry * dyc, rz * dzc
end

"""
    parent_faces(pc) -> Vector{Float64}

Interior face heights (m, length `Nz + 1`) of the parent: for
`vertical = :stretched`, `surface_refined_faces(dz_surface = Δz, z_uniform,
z_top = extent[3], ratio, dz_max)` (#151; 53 levels for the defaults:
20 × 25 m to 500 m, then 33 stretched, the largest 99.6 m); for
`:uniform`, `0:Δz:extent[3]`.
"""
function parent_faces(pc::ParentConfig)
    _, _, dz = parent_spacing(pc)
    H = pc.extent[3]
    if pc.vertical === :stretched
        return surface_refined_faces(; dz_surface = dz, z_uniform = pc.z_uniform, z_top = H,
                                     ratio = pc.ratio, dz_max = pc.dz_max)
    elseif pc.vertical === :uniform
        n = round(Int, H / dz)
        n * dz ≈ H || error("ParentConfig: extent[3] = $H is not a multiple of Δz = $dz")
        return collect(range(0.0, H; length = n + 1))
    end
    error("ParentConfig.vertical must be :stretched or :uniform, got $(repr(pc.vertical))")
end

"""
    parent_size(pc) -> (Nx, Ny, Nz)

The grid size of the parent; checks that the horizontal extent is a
whole number of parent cells (192 × 192 × 53 for the defaults).
"""
function parent_size(pc::ParentConfig)
    dx, dy, _ = parent_spacing(pc)
    Nx, Ny = round(Int, pc.extent[1] / dx), round(Int, pc.extent[2] / dy)
    (Nx * dx ≈ pc.extent[1] && Ny * dy ≈ pc.extent[2]) ||
        error("ParentConfig: extent $(pc.extent[1:2]) is not a whole number of $dx × $dy m cells")
    return Nx, Ny, length(parent_faces(pc)) - 1
end

"The horizontal centre of the parent box relative to the OSM anchor (m): `pc.center`, or the child's centre."
parent_center(pc::ParentConfig) = pc.center === nothing ? pc.child.offset : pc.center

"""
    parent_origin(pc) -> (x0, y0, 0)

The origin of the parent mesh in the coordinates of the child mesh
(whose origin is the child's south-west corner, `child.offset −
child.extent/2` from the anchor): `center − extent/2 − (offset −
extent_child/2)`. For the defaults (the 6 km parent centred on the 1 km
child) `(−2500, −2500, 0)`, so the child covers parent cells 81–112 in x
and y (`Nesting.nested_grid` finds them).
"""
function parent_origin(pc::ParentConfig)
    c, cfg = parent_center(pc), pc.child
    return (c[1] - pc.extent[1] / 2 - (cfg.offset[1] - cfg.extent[1] / 2),
            c[2] - pc.extent[2] / 2 - (cfg.offset[2] - cfg.extent[2] / 2), 0.0)
end

"""
    parent_mesh(pc) -> StretchedGrid

The parent's mesh: a `StretchedGrid` with the faces of
[`parent_faces`](@ref) (`:custom`), or `:uniform` stretching for the
uniform alternative (bit for bit the `RegularGrid` operators), placed at
[`parent_origin`](@ref).
"""
function parent_mesh(pc::ParentConfig)
    N = parent_size(pc)
    o = parent_origin(pc)
    if pc.vertical === :uniform
        return StretchedGrid(; size = N, extent = pc.extent, halo = pc.halo, origin = o)
    end
    return StretchedGrid(; size = N, extent = pc.extent, halo = pc.halo, origin = o,
                         stretching = (:custom, parent_faces(pc)))
end

"Level-centre heights (m) of the interior levels of a mesh."
level_centres(m) = Float64[m.zc[m.Hz+k] for k in 1:m.Nz]
