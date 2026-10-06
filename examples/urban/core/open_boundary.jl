# The lateral boundaries of the case (N-open-d, #262).
#
# `cfg.lateral` chooses them:
#
#   :periodic  (default) periodic in x and y, walls at the bottom and the
#              top for the velocity; θ with the top of `cfg.θ_top`
#              (core/top_boundary.jl). The case before #262, bit for bit.
#   :open      the four sides and the top are open (N-open-b, #255): the
#              normal velocity, the tangential velocity and θ of every
#              open face come from `ProfileBoundaryData` — the geostrophic
#              wind (U_g, V_g) of the `:geostrophic` coupling's options and
#              the initial θ profile `initial_θ(cfg)` — with the uniform
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
# The `:open` option is transitional: N-berlin-nest (#265) feeds the same
# open faces from the parent run and makes the nested configuration the
# default.
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

"Whether the lateral boundaries of `cfg` are open."
function lateral_open(cfg::CaseConfig)
    cfg.lateral in (:periodic, :open) ||
        error("CaseConfig.lateral must be :periodic or :open, got $(repr(cfg.lateral))")
    return cfg.lateral === :open
end

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
builds one open-boundary set, shared by `bcs` and `θ_bcs`.
"""
function case_boundary_conditions(cfg::CaseConfig, mesh)
    if !lateral_open(cfg)
        return (bcs   = FieldBoundaryConditions(:periodic, :periodic, :wall),
                θ_bcs = FieldBoundaryConditions(:periodic, :periodic; bottom = :noflux,
                                                top = θ_top_bc(cfg)))
    end
    data = case_boundary_data(cfg)
    b = open_boundary_conditions(mesh, data; outflow = cfg.outflow, halo = 1)
    return (bcs = b, θ_bcs = b, sponge_target = sponge_target(data, mesh))
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
    hl = case.ledger.heat
    if hl.active
        add_heat!(hl, :boundary_heat, J)
        add_heat!(hl, :mass_correction, Jc)
    end
    if ob.damping !== nothing
        Jr = ob.relax .* h
        ob.rtotal += sum(Jr)
        hl.active && add_heat!(hl, :relaxation, Jr)
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
