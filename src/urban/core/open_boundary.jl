# The lateral boundaries of the case (N-open-d, #262; nested, #265).
#
# With the `:nesting` coupling on (the default, N-berlin-nest #265) the
# four sides and the top are the nest faces of an Aeolus parent:
# `OpenBoundary` faces fed by `Nesting.NestBoundaryData`, built by the
# coupling's `sim_options` (couplings/nesting.jl). `cfg.lateral` is then
# not consulted for the boundary conditions. With the nest switched off
# (`physics = Dict(:nesting => false)`) `cfg.lateral` chooses them:
#
#   :periodic  (default) periodic in x and y, walls at the bottom and the
#              top for the velocity; θ with the top of `cfg.θ_top`
#              (core/top_boundary.jl). The case before #262, bit for bit.
#   :open      the four sides and the top are open (N-open-b, #255): the
#              normal velocity, the tangential velocity and θ of every
#              open face come from `ProfileBoundaryData` — the geostrophic
#              wind (U_g, V_g) of the `:geostrophic` coupling's options and
#              the initial θ profile `initial_θ(cfg)`, or the same values
#              read as the steady planes of the sounding driver
#              `cfg.dynamic` when it has them (CFG-g, #362) — with the uniform
#              global mass-flux correction, and the outflow condition
#              `cfg.outflow` (`:dirichlet`, `:convective` or `:orlanski`,
#              N-open-c #260). The core's Rayleigh sponge relaxes u, v, w
#              toward the same data (`sponge_target`), and, with
#              `cfg.θ_relaxation` (default), θ toward the data's profile
#              over the same layer (`LargeScaleForcing.ReferenceDamping`).
#              Buildings within `cfg.boundary_frame` cells of an open face
#              are removed (core/geometry.jl); a building that still
#              touches a face is solid there (`mark_open_boundary_solids!`).
#
# The nested default feeds the same open faces from the parent run; `:open`
# remains the parent-free alternative (analytic profile data).
#
# The open faces carry heat. The air-heat ledger books, inside the
# dynamics context and level by level:
#
#   (:dynamics, :boundary_heat)   the advective (the core's centred θ
#                                 fluxes) and diffusive (ν + the SGS κ)
#                                 heat through the lateral and top faces,
#                                 from `boundary_scalar_flux`, minus …
#   (:dynamics, :mass_correction) … the part the global mass-flux
#                                 correction δ carries;
#   (:dynamics, :relaxation)      the θ relaxation toward the data.
#
# Each is captured in the body-force hook of every dynamics (sub)step, where
# the state and the SGS diffusivities are those the core's tendencies were
# just built from, and booked when the (sub)step is accepted
# (`book_dynamics_heat!`), as the fixed-gradient top's flux is. For the
# forward-Euler `step!` of the default (`:adaptive`) and `:fixed` steppers
# the booked heat is the heat the step moved; for a multi-stage step
# (`:imex`) it is the last stage's rate × dt.

"""
    case_nested(cfg) -> Bool

Whether `cfg` runs the `:nesting` coupling (the default, #265): its sides
and top are then the nest faces of the parent run.
"""
case_nested(cfg::CaseConfig) = any(C -> coupling_name(C) === :nesting, selected_couplings(cfg))

"Whether the lateral boundaries of `cfg` are open: nested (the default) or `cfg.lateral = :open`."
function lateral_open(cfg::CaseConfig)
    cfg.lateral in (:periodic, :open) ||
        error("CaseConfig.lateral must be :periodic or :open, got $(repr(cfg.lateral))")
    return cfg.lateral === :open || case_nested(cfg)
end

"""
    ANALYSIS_BUFFER_DEFAULT

The default `cfg.analysis_buffer` (m): 140 m, the canopy-top inflow
fetch of the Alexanderplatz child **estimated** from the rule N-inflow-d
(#266) measured: behind a digital-filter inflow the turbulence recovers
to 10 % in 1.2 local eddy-turnover distances, ``L = 1.2\\,U k/ε`` (6.6δ
at 0.22δ and 12.3δ at 0.47δ of the half channel,
`docs/src/theory/transition.md`). With the neutral surface-layer values
``U = (u_*/κ)\\ln((z-d)/z_0)``, ``k = 5.48u_*^2`` (Panofsky & Dutton
1984: σ_{u,v,w} = 2.39, 1.92, 1.25 u_*) and ``ε = u_*^3/(κ(z-d))`` this is
``L = 6.58\\,(z-d)\\ln((z-d)/z_0)``, independent of ``u_*``; at the mean
roof height of the flagship scene (h = 21.6 m, λ_p = 0.223, λ_f = 0.151;
Macdonald et al. 1998: d = 9.6 m, z_0 = 2.0 m) L = 141 m. (At 100 m the
same rule gives 2.3 km, more than the child: there the child keeps what
the generator and the parent give it.) The Berlin fetch measured on the
nested child, day and night (#268 step 2), replaces it when the 24 h runs
are made (#270). Macdonald, R. W., Griffiths, R. F. & Hall, D. J. (1998),
*Atmos. Environ.* 32, 1857-1864.
"""
const ANALYSIS_BUFFER_DEFAULT = 140.0

"""
    analysis_columns(cfg, mesh) -> BitMatrix
    analysis_columns(case) -> BitMatrix

The `Nx × Ny` columns the diagnostics compared with observations use:
those whose centre lies at least `cfg.analysis_buffer` (m) from every
lateral open face (all four sides when the case is nested or
`lateral = :open`; none in the periodic box, where every column counts).
Behind an inflow face the turbulence the face receives is still
developing over a fetch (N-inflow-d #266, N-inflow-e #268), so a column
inside it is not the equilibrium flow over its own surface. Every side
is cut, not only the current inflow faces: the wind turns over the day.
"""
function analysis_columns(cfg::CaseConfig, m)
    b = Float64(cfg.analysis_buffer)
    A = trues(m.Nx, m.Ny)
    (lateral_open(cfg) && b > 0) || return A
    for j in 1:m.Ny, i in 1:m.Nx
        x, y = (i - 0.5) * m.dx, (j - 0.5) * m.dy
        A[i, j] = min(x, m.Lx - x, y, m.Ly - y) >= b
    end
    return A
end
analysis_columns(case) = analysis_columns(case.cfg, case.mesh)

"""
    case_boundary_data(cfg) -> ProfileBoundaryData

The boundary data of the `:open` case: `u = U_g`, `v = V_g` (the
`:geostrophic` coupling's options, also when it is switched off), `w = 0`
and θ the initial profile [`initial_θ`](@ref) of `cfg`.
"""
function case_boundary_data(cfg::CaseConfig)
    g = coupling_options(GeostrophicCoupling, cfg)
    θi = initial_θ(cfg)
    return ProfileBoundaryData(u = Float64(g.U_g), v = Float64(g.V_g), w = 0.0,
                               θ = z -> θi(0.0, 0.0, z))
end

"""
    case_boundary_conditions(cfg, mesh) -> NamedTuple

The boundary keywords of the case's core: `bcs` and `θ_bcs`, plus, for
`cfg.lateral = :open`, the `sponge_target` of the open top. One call
builds one open-boundary set, shared by `bcs` and `θ_bcs`. A nested case
(the default) gets none here: the `:nesting` coupling's `sim_options`
are the one definition of its nest faces (couplings/nesting.jl).
"""
function case_boundary_conditions(cfg::CaseConfig, mesh)
    case_nested(cfg) && return (;)
    if !lateral_open(cfg)
        return (bcs   = FieldBoundaryConditions(:periodic, :periodic, :wall),
                θ_bcs = FieldBoundaryConditions(:periodic, :periodic; bottom = :noflux,
                                                top = θ_top_bc(cfg)))
    end
    data = case_boundary_data(cfg)
    b = open_boundary_conditions(mesh, data; outflow = cfg.outflow, halo = 1)
    _open_planes_from_driver!(b, cfg, mesh)
    return (bcs = b, θ_bcs = b, sponge_target = sponge_target(data, mesh))
end

# The open case's faces from the steady planes of its sounding driver
# (CFG-g, #362: `aeolusprep dynamic --from-sounding` of an open
# config writes them, AeolusPrep.jl), when the driver has planes. The sponge and θ
# relaxation keep the profile data's targets.
function _open_planes_from_driver!(b, cfg::CaseConfig, mesh)
    path = initial_driver(cfg)
    path === nothing && return b
    ox, oy = anchor_xy(cfg, mesh)
    d = read_dynamic_driver(path; mesh = mesh, origin_lat = cfg.lat0, origin_lon = cfg.lon0,
                            origin_x = ox, origin_y = oy)
    f = d.boundary
    f === nothing && return b
    length(f.time) == 1 ||
        error("the dynamic driver $path has $(length(f.time)) boundary records; an open case without the " *
              "nest takes steady planes (one record): time-dependent planes feed the nest (parent_driver = \"offline\")")
    check_planes(f, only(open_boundary_states(b)).planes)
    set_boundary_data!(b, f.data)
    return b
end

"""
    case_scalar_bcs(cfg) -> FieldBoundaryConditions

The halo conditions of the case's tracers and of every mask built from
the solid mask (`scalar_advection`, the SGS couplings, the pollutant):
periodic laterally and no flux at the ground and the top for a periodic
case; for an open one the mirror (zero-gradient) halo on every face — a
tracer without boundary data leaves with its own value and enters with
the value next to the face, and a mask does not wrap across the box.
"""
case_scalar_bcs(cfg::CaseConfig) = lateral_open(cfg) ? FieldBoundaryConditions(:noflux) :
                                   FieldBoundaryConditions(:periodic, :periodic, :noflux)

"""
    tracer_bcs(case) -> FieldBoundaryConditions

The halo conditions of the advected tracers: [`case_scalar_bcs`](@ref),
except in a nested case, where they are the core's own open boundary set
(`sim.θ_bcs`): a tracer of the nest's field set gets the parent's value
(or, if the parent does not carry it, zero on inflow and its own value on
outflow — the `:inflow` halo), every other array the mirror, as with
`case_scalar_bcs` (couplings/nesting.jl, #265).
"""
tracer_bcs(case) = case_nested(case.cfg) ? case.sim.θ_bcs : case_scalar_bcs(case.cfg)

"""
    OpenBoundaryHeat

The heat the open faces of a `:open` case move, for the air-heat ledger
(see the file header): per level `k` (W m⁻² of domain) the rates of the
last dynamics (sub)step — `rate` (advective + diffusive, without the
mass-flux correction's part), `crate` (the correction's part) and
`relax` (the θ relaxation) — and the run totals (J m⁻²): `total`,
`ctotal`, `rtotal`, and the advective + diffusive heat split by face,
`lateral` (the four sides) and `top` (correction included); `δmax` is
the largest |δ| (m s⁻¹) of the mass-flux correction seen. `wc`, `wf`,
`wfd` are the storage-indexed ρ̄c_pΠ̄ weights `boundary_scalar_flux` takes
(lateral faces; the top face's advective and diffusive weight), `damping`
the θ `ReferenceDamping` (or `nothing`).
"""
mutable struct OpenBoundaryHeat{D}
    wc::Vector{Float64}
    wf::Vector{Float64}
    wfd::Vector{Float64}
    heat::Vector{Float64}       # ρ̄ c_p Π̄ per interior level
    A::Float64                  # ground area of the domain (m²)
    lev::Vector{Float64}
    clev::Vector{Float64}
    rate::Vector{Float64}
    crate::Vector{Float64}
    relax::Vector{Float64}
    damping::D
    total::Float64
    ctotal::Float64
    rtotal::Float64
    rlat::Float64               # W m⁻² through the four sides, last capture
    rtop::Float64               # W m⁻² through the top, last capture
    lateral::Float64
    top::Float64
    δmax::Float64
end

"The open-boundary heat meter of `case` (`nothing` for a periodic case)."
function open_boundary_heat(cfg::CaseConfig, sim, ref)
    has_open_boundaries(sim) || return nothing
    m = sim.mesh
    Nz, Hz = m.Nz, m.Hz
    heat = [ref.ρ̄[k] * ref.cp * ref.π̄[k] for k in 1:Nz]
    pad(v) = [v[clamp(k - Hz, 1, Nz)] for k in 1:Nz+2Hz]
    wc = pad(heat)
    kt = Hz + Nz + 1
    wf = zeros(Nz + 2Hz); wfd = zeros(Nz + 2Hz)
    wfd[kt] = heat[Nz]
    # the top face's advective flux enters the top level as W_f u θ / W_c on
    # a sound-proof core (its weighted flux form); 1 on the Boussinesq core
    W = boundary_flux_weights(sim)
    wf[kt] = W === nothing ? heat[Nz] : heat[Nz] * W[2][kt] / W[1][Hz+Nz]
    damping = nothing
    if cfg.θ_relaxation && sim.has_sponge
        z_top = m.z0 + m.Lz
        θi = initial_θ(cfg)
        zc = Aeolus.Operators.damping_coordinates(m, CellCenter())
        damping = ReferenceDamping(:θ, Aeolus.Operators.sponge_profile(sim.sponge_α_max,
                                                                       sim.sponge_z_start, z_top),
                                   [θi(0.0, 0.0, z) for z in zc])
    end
    return OpenBoundaryHeat(wc, wf, wfd, heat, m.Lx * m.Ly, zeros(Nz), zeros(Nz), zeros(Nz),
                            zeros(Nz), zeros(Nz), damping, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
end

"""
    capture_boundary_heat!(ob::OpenBoundaryHeat, sim, closure) -> ob

In the body-force hook of a dynamics (sub)step: apply the θ relaxation
toward the boundary data (if any) and record, per level, the heat rate of
the open faces with the state and the SGS diffusivity (`closure`, or the
core's built-in Smagorinsky) the step's tendencies were built from.
"""
function capture_boundary_heat!(ob::OpenBoundaryHeat, sim, closure)
    m = sim.mesh
    fill!(ob.lev, 0.0); fill!(ob.clev, 0.0)
    cl = Aeolus.Operators._active_sgs(sim, closure, nothing)
    κh, κv = cl === nothing ? (nothing, nothing) :
             (f = sgs_fields(cl, sim); (f.κh, f.κv))
    ν = sim.has_viscosity ? Float64(sim.ν) : 0.0
    F = boundary_scalar_flux(sim, sim.θ; scheme = :centered, bcs = sim.θ_bcs,
                             weights = (ob.wc, ob.wf), fill = false,
                             diffusivity = (ν = ν, κh = κh, κv = κv, weights = (ob.wc, ob.wfd)),
                             levels = ob.lev, correction_levels = ob.clev)
    ob.rlat = -(F.west + F.east + F.south + F.north) / ob.A
    ob.rtop = -F.top / ob.A
    for st in open_boundary_states(sim)
        ob.δmax = max(ob.δmax, abs(st.δ))
    end
    @inbounds for k in eachindex(ob.lev)
        # outward flux → heat into the level (W m⁻² of domain)
        ob.rate[k] = -(ob.lev[k] - ob.clev[k]) / ob.A
        ob.crate[k] = -ob.clev[k] / ob.A
    end
    if ob.damping !== nothing
        apply_forcing!(sim, ob.damping, sim.t, 0.0)
        lv = damping_tendency(ob.damping).levels
        V = m.dx * m.dy * m.dz
        @inbounds for k in eachindex(ob.relax)
            ob.relax[k] = ob.heat[k] * lv[k] * V / ob.A
        end
    end
    return ob
end

"""
    book_boundary_heat!(case, h) -> case

Book the open faces' heat over an accepted dynamics (sub)step of `h`
seconds in the air-heat ledger, level by level: `(:dynamics,
:boundary_heat)`, `(:dynamics, :mass_correction)` and, with the θ
relaxation, `(:dynamics, :relaxation)`. No-op for a periodic case.
"""
function book_boundary_heat!(case, h::Real)
    ob = case.open
    ob === nothing && return case
    J = ob.rate .* h; Jc = ob.crate .* h
    ob.total += sum(J); ob.ctotal += sum(Jc)
    ob.lateral += ob.rlat * h; ob.top += ob.rtop * h
    hl = energy(case.budgets)
    if hl.active
        book!(hl, :boundary_heat, J)
        book!(hl, :mass_correction, Jc)
    end
    if ob.damping !== nothing
        Jr = ob.relax .* h
        ob.rtotal += sum(Jr)
        hl.active && book!(hl, :relaxation, Jr)
    end
    return case
end

"""
    book_dynamics_heat!(case, h) -> case

Book the heat the dynamics moved through the domain's faces over an
accepted (sub)step of `h` seconds: the fixed-gradient top's
([`book_top_flux!`](@ref)) and the open faces'
([`book_boundary_heat!`](@ref)). Called by every `:timestepper` provider.
"""
book_dynamics_heat!(case, h::Real) = (book_top_flux!(case, h); book_boundary_heat!(case, h))

"""
    centre_wind(sim, i, j, k) -> (uc, vc)

The horizontal wind at the centre of interior cell `(i, j, k)`: the mean
of its two `u` faces and of its two `v` faces, the far face read with
[`neighbor_index`](@ref) on the core's velocity conditions — the wrapped
face across the periodic seam (the pre-#262 `mod1`, bit for bit), the
open face's prescribed normal velocity at an open side. The one place the
couplings (surface exchange, air-quality deposition, the facade wall
model's canopy wind) centre the wind.
"""
@inline function centre_wind(sim, i::Integer, j::Integer, k::Integer)
    m, b = sim.mesh, sim.bcs
    ud, vd = sim.u.data, sim.v.data
    ip = neighbor_index(m, b, i + 1, 1; face = true)
    jp = neighbor_index(m, b, j + 1, 2; face = true)
    I, J, K = m.Hx + i, m.Hy + j, m.Hz + k
    return ((ud[I, J, K] + ud[m.Hx+ip, J, K]) / 2, (vd[I, J, K] + vd[I, m.Hy+jp, K]) / 2)
end
