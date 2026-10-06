# Transport of the registered tracers (Aeolus.AdvectionSchemes), forward
# Euler in time, in the `:scalar_advection` slot. Replaces the former
# `transport_weno` coupling (the WENO-Z5 scheme is still the default).
#
# qv and ql are ACTIVE (fog/dew feed latent heat back into θ and Ts), so
# they need a stable, non-oscillatory scheme; tracers registered with
# `positive = true` go through the flux-limited `advect_positive!`. The
# former forward-Euler + 2nd-order centred update was unconditionally
# unstable and blew the run up once condensation was coupled.
#
# Velocity halos (#132): `step!` corrects the interior velocity after the
# pressure solve but leaves the halos at their predictor values, and on
# step 1 they were never filled. A flux-form scheme on those halos is not
# conservative across the periodic seam and the walls: in the :smoke
# window the first transport step ADDED 7.5e-3 kg/m² of water (0.8 % of
# the water in the air), and over 1 h of the flagship the transport
# removed 2.70e-4 kg/m², 35 % of the surface evaporation. The coupling
# therefore refreshes the u, v, w halos before it advects
# (`refresh_halos = true`); the transport then conserves water to
# round-off.
#
# Sound-proof cores (#133). Under the anelastic / pseudo-incompressible
# core the velocity satisfies ∇·(W u) = 0 (W = ρ̄ or ρ̄θ̄), not ∇·u = 0, so
# the plain flux form −∇·(u q) neither keeps a uniform q uniform nor
# conserves Σ ρ̄ q (it conserves Σ q). The consistent tracer equation is
# ∂(W q)/∂t = −∇·(W u q). The coupling then advects with the mass flux:
# the same scheme on the velocity (W_face/W_min)·u, and the increment is
# scaled by W_min/W_k. That conserves Σ W q exactly (flux form), keeps a
# uniform q uniform to the projection tolerance, and keeps the positivity
# limiter valid (the limited outflow is scaled by W_min/W_k ≤ 1). On the
# anelastic core W = ρ̄ is the ledger's density, so the transport books
# no density redistribution.
#
# Known residual: a few wall-nook cells still gain vapour, because the
# IBM correction applied after the projection leaves the velocity
# slightly convergent there (the flux form still conserves the total).
#
# A tracer may name its own diffusivity (`ensure_tracer!(…; diffusivity)`):
# the SGS TKE `e` of the `:sgs => :tke` provider is diffused with 2ν_t, the
# TKE flux of the Deardorff closure (#182).
#
# SGS diffusion (P56-a, #177). Before #177 nothing in the case diffused a
# tracer: θ got the closure's scalar flux inside step!, but qv, ql, the
# chemistry species and the pollutant saw only the implicit dissipation of
# the advection scheme — a missing-physics bug. Every tracer this stage
# advects is now also diffused with the closure's κt of the last dynamics
# step (`sim.κt`, the same diffusivity θ gets; with a stability-corrected
# closure, P55-e, κh = sim.κt on the x, y faces and its κv on the z faces,
# as for θ), after its advective update
# (Lie splitting, so each half keeps q ≥ 0 on its own), through the one
# src kernel `add_eddy_diffusion_scalar!` with the case's solid mask (no
# flux into or out of buildings, solid cells untouched) and, on a
# sound-proof core, the constraint weights (conserving Σ W q). The
# pollutant (its own :tracers stage) uses the same helper.
#
# Open lateral boundaries (`cfg.lateral = :open`, N-open-d #262). The tracer
# and mask halos come from `case_scalar_bcs`: the mirror (zero-gradient)
# halo on every face of an open case — a tracer has no boundary data, so it
# leaves with its own value and enters with the value next to the face. The
# water the open faces carry is booked as the water ledger's `:boundary`
# term, from `boundary_scalar_flux` with the advection's own scheme and
# face velocities (the flux through an open face is not a non-conservation
# of the scheme), split from `transport` / `density_redistribution` level by
# level. The SGS diffusion carries nothing through a mirror halo.
#
# Nested case (the default, #265): the tracer halos are the core's open set
# (`tracer_bcs`): qv (and the TKE `e`) take the parent's values on the nest
# faces, the tracers the parent does not carry zero on inflow and their own
# value on outflow; the `:boundary` term books what that carries.
#
# Two-way nest (#267, the M3 wiring of #327): the slot books the interface
# flux it applies to every tracer of a refluxing two-way nest
# (`book_nest_flux!`, a no-op otherwise): the advective flux with its own
# scheme and velocity at the state it advects, before the update, and the
# SGS diffusive flux with the tracer's κ at the advected state, before the
# diffusion. Where the positivity limiter scales an interface flux the
# booking is the unlimited one (a known residual of the slim option).

"Reconstructions `advect_positive!` accepts (the flux limiter needs one)."
const POSITIVE_SCHEMES = (:centered, :weno5, :weno_js5, :weno_z5, :weno_z7, :teno5, :teno8)

"""
    ScalarAdvection

Slot `:scalar_advection`, provider `:advection_schemes`: every step, each
tracer registered with `advect = true` (registration order) is advanced
over `dt` with the scheme `scheme` of `Aeolus.AdvectionSchemes`
(`advection_scheme(scheme)`, any of `:centered, :upwind, :weno5,
:weno_js5, :weno_z5, :weno_z7, :teno5, :teno8, :variance_conserving`),
with the halos of `case_scalar_bcs` (periodic lateral and no-flux vertical
by default; zero-gradient on every face of an open case, #262). Tracers flagged `positive`
use the flux-limited `advect_positive!` with the same reconstruction
(`positivity = false` switches that off). The mesh halo must be at least
`required_halo` of the scheme (`:weno_z7` needs 4, `:teno8` 5; the case
default is 3).

Options: `scheme = :weno_z5`, `positivity = true`, `refresh_halos = true`
(fill the velocity halos before advecting; `false` reproduces the
non-conservative pre-#132 transport), `sgs_diffusion = true` (diffuse every
advected tracer with the SGS closure's `κt`, see [`TracerSGSDiffusion`](@ref);
`false` reproduces the pre-#177 transport, which diffused no tracer).

On a sound-proof core (`core = :anelastic`, `:pseudo_incompressible`)
the transport is `∂(W q)/∂t = −∇·(W u q)` with the core's constraint
weight `W` (`Anelastic.constraint_weights`): the scheme is applied to the
mass-flux velocity `(W/W_min) u` and the increment scaled by `W_min/W_k`,
so `Σ W q` is conserved and a uniform tracer stays uniform. Option
`weighted = false` uses the plain velocity there too (the pre-#133
transport, for comparison).
"""
struct ScalarAdvection{F, W, B, S, R} <: AbstractCoupling
    tend::F
    ws::W
    bcs::B
    scheme::S                     # AdvectionScheme (the unlimited operator)
    recon::R                      # its reconstruction descriptor for advect_positive! (built once, #295), or nothing
    positivity::Bool
    refresh_halos::Bool
    dW::Base.RefValue{Float64}    # this step's change of air water, ρ̄-mean part (kg/m²; 0 if conservative)
    dWρ::Base.RefValue{Float64}   # this step's ρ̄-redistribution part (kg/m²; see ledger_water_split)
    weighted::Any                 # nothing (Boussinesq core), or the W-weighted transport of a sound-proof core
    weighted_option::Bool
    diffusion::Any                # TracerSGSDiffusion, or nothing (sgs_diffusion = false)
    open::Any                     # OpenFaceWater of an open case (#262), or nothing
    dWb::Base.RefValue{Float64}   # this step's water carried in through the open faces (kg/m²)
end

"""
    OpenFaceWater

The water an open case's faces move ([`ScalarAdvection`](@ref), #262):
the scheme name `boundary_scalar_flux` reconstructs with, its weights —
1 on the lateral faces, and on the top face the factor the advected
increment of the top level carries (`W_f/W_c` on a sound-proof core) —
and per-level buffers.
"""
struct OpenFaceWater
    scheme::Symbol
    weights::Tuple{Vector{Float64},Vector{Float64}}
    lev::Vector{Float64}
    S::Vector{Float64}
end

function OpenFaceWater(case, scheme::Symbol, wt)
    mesh = case.mesh
    Hz, Nz = mesh.Hz, mesh.Nz
    try
        Aeolus.AdvectionSchemes._flux_reconstruction(scheme)
    catch
        error("scalar_advection: the open faces' water (#262) is booked with boundary_scalar_flux, " *
              "which has no face reconstruction for scheme :$scheme")
    end
    wc = ones(Nz + 2Hz); wf = ones(Nz + 2Hz)
    if wt !== nothing          # the W-weighted transport: the top face enters level Nz as W_f/W_c
        Wc, Wf = Aeolus.Anelastic.constraint_weights(case.sim)
        wf[Hz+Nz+1] = Wf[Hz+Nz+1] / Wc[Hz+Nz]
    end
    return OpenFaceWater(scheme, (wc, wf), zeros(Nz), zeros(Nz))
end

"""
    TracerSGSDiffusion

SGS diffusion of a registered tracer with the closure's eddy diffusivity
(P56-a, #177): `q ← q + dt · ∇·(κt ∇q)` with `κt = sim.κt` of the last
dynamics step (zero before the first step, and with `C_s = 0`), formed by
`Operators.add_eddy_diffusion_scalar!` with

  * `mask` — the case's solid mask with the halos of `case_scalar_bcs`
    (periodic laterally; mirrored at open faces, #262): no flux
    through a face next to a building, solid cells untouched;
  * `weights` — on a sound-proof core the constraint weights `(Wc, Wf)`
    (`Anelastic.constraint_weights`), so the update is
    `(1/W) ∇·(W κt ∇q)` and conserves `Σ W q` (on the anelastic core that is
    the ledger's `Σ ρ̄ q`); `nothing` on the Boussinesq core (conserves `Σ q`).

Halos of the tracer: `case_scalar_bcs` (periodic laterally, no-flux at the
ground and the top; zero-gradient on every face of an open case).
[`sgs_diffuse!`](@ref) applies it; the `:transport` stage calls it for every
advected tracer and the pollutant coupling for its tracer. Every passive
tracer gets the same shared `κt`, also under the `:amd` provider, whose
per-scalar diffusivity is deliberately not used (one turbulent Schmidt
number; #234, see the SGS theory page).
"""
struct TracerSGSDiffusion{F, M, W}
    tend::F
    mask::M
    weights::W
    bcs::FieldBoundaryConditions
    dW::Base.RefValue{Float64}        # this step's air-water change, conserved part (kg/m²; ~0)
    dWρ::Base.RefValue{Float64}       # this step's ρ̄-redistribution part (kg/m²)
    resid::Base.RefValue{Float64}     # max relative change of the conserved total, this step
    Sq::Vector{Float64}               # scratch: per-level Σ|q| (canonical plane sums)
    St::Vector{Float64}               # scratch: per-level Σ tendency
end

function TracerSGSDiffusion(case)
    mesh = case.mesh
    bcs = case_scalar_bcs(case.cfg)
    mf = Field(CellCenter(), mesh)
    mf.data .= case.geom.sm.mask
    fill_halos!(mf, bcs)                       # periodic copies across the seam (mirror at open faces)
    sim = case.sim
    w = sim isa Aeolus.Anelastic.MassWeightedSimulation ?
        Aeolus.Anelastic.constraint_weights(sim) : nothing
    return TracerSGSDiffusion(Field(CellCenter(), mesh), Array{Bool}(mf.data .> 0.5), w, bcs,
                              Ref(0.0), Ref(0.0), Ref(0.0), zeros(mesh.Nz), zeros(mesh.Nz))
end

"""
    sgs_diffuse!(d::TracerSGSDiffusion, case, q, dt; water = false, κ = case.sim.κt, κv = κ) -> q

Advance the tracer field `q` by the SGS diffusion over `dt` (see
[`TracerSGSDiffusion`](@ref)) with the eddy diffusivity `κ` (the
closure's `κt`; a tracer can name another, see `tracer_diffusivity`) on
the x and y faces and `κv` on the z faces (the stability-corrected
closure's κv, P55-e, see `tracer_vertical_diffusivity`; `κv === κ` is the
isotropic operator, bit for bit). With `water = true` the increment is booked
in `d.dW` / `d.dWρ` (`ledger_water_split`), and the relative change of the
conserved total (`Σ q` over fluid cells, or `Σ W q`) is recorded in
`d.resid`.
"""
function sgs_diffuse!(d::TracerSGSDiffusion, case, q, dt; water::Bool = false, κ = case.sim.κt,
                     κv = κ)
    sim, mesh = case.sim, case.mesh
    Hx, Hy, Hz = mesh.Hx, mesh.Hy, mesh.Hz
    Nx, Ny, Nz = mesh.Nx, mesh.Ny, mesh.Nz
    fill_halos!(q, d.bcs)
    fill!(d.tend.data, 0)
    if κv === κ
        add_eddy_diffusion_scalar!(d.tend, q, κ; mask = d.mask, weights = d.weights)
    else
        add_eddy_diffusion_scalar!(d.tend, q, κ, κv; mask = d.mask, weights = d.weights)
    end
    tend = view(d.tend.data, Hx+1:Hx+Nx, Hy+1:Hy+Ny, Hz+1:Hz+Nz)
    qi = interior_view(q)
    Wk = d.weights === nothing ? nothing : view(d.weights[1], Hz+1:Hz+Nz)
    # the conserved total (solid tendencies are zero, so the sum over all
    # cells is the sum over fluid cells): the per-level sums of |q| and of
    # the tendency in one canonical, thread-invariant pass (KA-b, #296),
    # weighted and combined over k in ascending order
    Sq, St = d.Sq, d.St
    plane_reduce!((Sq, St), Aeolus.Backends.backend(q.data), _abs_and_tend_at,
                  Aeolus.Grids.interior(mesh), q.data, d.tend.data)
    tot = 0.0; dtot = 0.0
    @inbounds for k in 1:Nz
        wk = Wk === nothing ? 1.0 : Wk[k]
        tot += wk * Sq[k]; dtot += wk * St[k]
    end
    tot > 0 && (d.resid[] = max(d.resid[], abs(dt * dtot) / tot))
    qi .+= dt .* tend
    if water
        flat, redis = ledger_water_split_levels(case, St, Wk === nothing ? nothing : collect(Wk))
        d.dW[] += dt * flat; d.dWρ[] += dt * redis
    end
    return q
end

@inline _abs_and_tend_at(i, j, k, q, t) = @inbounds (abs(q[i, j, k]), t[i, j, k])

"The tracer diffusion of the enabled `:scalar_advection` coupling, or `nothing`."
function tracer_sgs_diffusion(case)
    c = find_coupling(case, :scalar_advection)
    return c === nothing ? nothing : c.diffusion
end

"""
    WeightedTransport

The mass-flux velocity of a sound-proof core for [`ScalarAdvection`](@ref):
`U = (W/W_min) u` (cell weight `Wc` on the x / y faces, face weight `Wf`
on the z faces, as in `Anelastic.weighted_divergence!`), the per-level
scale `s_k = W_min / Wc_k` of the increment, and the cell weights `W`
(interior levels) the ledger split uses.
"""
struct WeightedTransport{FU, FV, FW}
    U::FU
    V::FV
    Wv::FW
    rc::Vector{Float64}           # Wc / W_min per storage level (x, y faces)
    rf::Vector{Float64}           # Wf / W_min per storage level (z faces)
    s::Vector{Float64}            # W_min / Wc per interior level
    W::Vector{Float64}            # Wc per interior level
end

function WeightedTransport(sim, mesh)
    Wc, Wf = Aeolus.Anelastic.constraint_weights(sim)
    Hz, Nz = mesh.Hz, mesh.Nz
    Wmin = min(minimum(Wc[Hz+1:Hz+Nz]), minimum(Wf[Hz+1:Hz+Nz+1]))
    return WeightedTransport(Field(FaceX(), mesh), Field(FaceY(), mesh), Field(FaceZ(), mesh),
                             Wc ./ Wmin, Wf ./ Wmin, Wmin ./ Wc[Hz+1:Hz+Nz], Wc[Hz+1:Hz+Nz])
end

transport_weighting(sim::BoussinesqSimulation, mesh) = nothing
transport_weighting(sim::Aeolus.Anelastic.MassWeightedSimulation, mesh) = WeightedTransport(sim, mesh)

"Fill the mass-flux velocity of `wt` from the (halo-filled) core velocity."
function mass_flux_velocity!(wt::WeightedTransport, sim)
    rc = reshape(wt.rc, 1, 1, :)
    rf = reshape(wt.rf, 1, 1, :)
    wt.U.data .= sim.u.data .* rc
    wt.V.data .= sim.v.data .* rc
    wt.Wv.data .= sim.w.data .* rf
    return wt
end

coupling_name(::Type{<:ScalarAdvection}) = :scalar_advection
slot(::Type{<:ScalarAdvection}) = :scalar_advection
provider(::Type{<:ScalarAdvection}) = :advection_schemes
modules_used(::Type{<:ScalarAdvection}) = (:AdvectionSchemes, :Operators)
default_options(::Type{<:ScalarAdvection}) =
    (scheme = :weno_z5, positivity = true, refresh_halos = true, weighted = true,
     sgs_diffusion = true)
stages(::Type{<:ScalarAdvection}) = ((:transport, 10, EVERY_STEP),)

function build(::Type{<:ScalarAdvection}, case, o)
    mesh = case.mesh
    scheme = Aeolus.AdvectionSchemes.advection_scheme(o.scheme)
    H = Aeolus.AdvectionSchemes.required_halo(scheme)
    min(mesh.Hx, mesh.Hy, mesh.Hz) >= H ||
        error("scalar_advection: scheme :$(o.scheme) needs halo ≥ $H, the case has $((mesh.Hx, mesh.Hy, mesh.Hz)) (set CaseConfig halo)")
    o.positivity && !(o.scheme in POSITIVE_SCHEMES) &&
        error("scalar_advection: the positivity limiter takes one of $(POSITIVE_SCHEMES), not :$(o.scheme); set positivity = false")
    wt = o.weighted ? transport_weighting(case.sim, mesh) : nothing
    # the limiter's reconstruction as a concrete descriptor, built once:
    # from the name it would be rebuilt (a dynamic dispatch) every call (#295)
    recon = o.positivity ? Aeolus.AdvectionSchemes._reconstruction(o.scheme) : nothing
    return ScalarAdvection(Field(CellCenter(), mesh), FluxLimiterWorkspace(mesh),
                           tracer_bcs(case),
                           scheme, recon, o.positivity, o.refresh_halos, Ref(0.0), Ref(0.0),
                           wt, o.weighted,
                           o.sgs_diffusion ? TracerSGSDiffusion(case) : nothing,
                           lateral_open(case.cfg) ? OpenFaceWater(case, Symbol(o.scheme), wt) : nothing,
                           Ref(0.0))
end

function run_stage!(c::ScalarAdvection, case, ::Val{:transport}, dt)
    sim, mesh = case.sim, case.mesh
    Hx, Hy, Hz = mesh.Hx, mesh.Hy, mesh.Hz
    Nx, Ny, Nz = mesh.Nx, mesh.Ny, mesh.Nz
    if c.refresh_halos
        fill_halos!(sim.u, sim.bcs)
        fill_halos!(sim.v, sim.bcs)
        fill_halos!(sim.w, sim.bcs)
    end
    c.dW[] = 0.0; c.dWρ[] = 0.0; c.dWb[] = 0.0
    d = c.diffusion
    d === nothing || (d.dW[] = 0.0; d.dWρ[] = 0.0; d.resid[] = 0.0)
    wt = c.weighted
    u, v, w = wt === nothing ? (sim.u, sim.v, sim.w) :
                               (mass_flux_velocity!(wt, sim); (wt.U, wt.V, wt.Wv))
    for t in case.tracers
        t.advect || continue
        q = t.field
        fill_halos!(q, c.bcs)
        iswater = t.name in case.ledger.water_tracers
        limited = t.positive && c.positivity
        fill!(c.tend.data, 0)
        # the two-way nest's flux registers (#267): what this update moves
        # through the interface planes, at the state it moves it from
        book_nest_flux!(sim, t.name, q, dt; scheme = c.scheme.name, velocity = (u, v, w))
        if limited
            advect_positive!(c.tend, u, v, w, q, dt;
                             scheme = c.recon, workspace = c.ws, bcs = c.bcs)
        else
            c.scheme(c.tend, u, v, w, q)
        end
        # the water the open faces carry this step, per level (#262): the
        # face fluxes the transport used — the limited ones of
        # advect_positive! on the Boussinesq core (the positivity limiter
        # scales a boundary flux where the cell next to an outflow face
        # would go negative, #265), the reconstructed ones otherwise
        ob = c.open
        fb = rb = 0.0
        if ob !== nothing && iswater
            fill!(ob.lev, 0.0)
            boundary_scalar_flux(sim, q; scheme = ob.scheme, bcs = c.bcs, weights = ob.weights,
                                 fill = false, levels = ob.lev,
                                 face_fluxes = limited && wt === nothing ? (c.ws.Fx, c.ws.Fy, c.ws.Fz) : nothing)
            Vc = mesh.dx * mesh.dy * mesh.dz
            @inbounds for k in 1:Nz
                ob.S[k] = -dt * ob.lev[k] / Vc
            end
            fb, rb = ledger_water_split_levels(case, ob.S, wt === nothing ? nothing : wt.W)
            c.dWb[] += fb + rb
        end
        tend = view(c.tend.data, Hx+1:Hx+Nx, Hy+1:Hy+Ny, Hz+1:Hz+Nz)
        # weighted: −∇·(W u q)/W_min → −∇·(W u q)/W_k
        wt === nothing || (tend .*= reshape(wt.s, 1, 1, :))
        interior_view(q) .+= dt .* tend
        if iswater
            flat, redis = ledger_water_split(case, tend, wt === nothing ? nothing : wt.W)
            # the open faces' share is booked as :boundary (#262)
            c.dW[] += dt * flat - fb; c.dWρ[] += dt * redis - rb
        end
        # SGS diffusion with κt (κh, κv of a stability-corrected closure),
        # after the advective update (#177)
        if d !== nothing
            κ, κv = tracer_diffusivity(t, case), tracer_vertical_diffusivity(t, case)
            fill_halos!(q, d.bcs)       # the halos sgs_diffuse! diffuses with
            book_nest_flux!(sim, t.name, q, dt; diffusivity = (κh = κ, κv = κv))
            sgs_diffuse!(d, case, q, dt; water = iswater, κ = κ, κv = κv)
        end
    end
    return nothing
end

# Ledger: a flux-form scheme on a divergence-free velocity with periodic /
# no-flux boundaries changes no domain total, so anything it does change
# is booked as `transport` — a measure of non-conservation (zero to
# round-off with refreshed halos). Through the faces of an open case it
# changes the total by the boundary flux, booked as `boundary` (#262). Under a sound-proof core the conserved
# total is Σ W q (see the file header) and `transport` measures that. The ledger integrates with the
# reference density ρ̄_k (#131), which an incompressible transport does
# not conserve: moving vapour up into thinner air loses ρ̄-weighted water.
# That part is booked separately as `density_redistribution`
# (ledger_water_split), so `transport` stays the scheme's own
# non-conservation.
# The SGS diffusion books its conserved-part change as `sgs_diffusion`
# (zero to round-off: a flux form with no-flux walls and solids) and its
# ρ̄-redistribution with the transport's.
function budget!(c::ScalarAdvection, case, l)
    add_water!(l, :transport, c.dW[]); add_water!(l, :density_redistribution, c.dWρ[])
    c.open === nothing || add_water!(l, :boundary, c.dWb[])
    d = c.diffusion
    if d !== nothing
        add_water!(l, :sgs_diffusion, d.dW[]); add_water!(l, :density_redistribution, d.dWρ[])
    end
    return nothing
end

function smoke_check(c::ScalarAdvection, case)
    W = sum(t -> ledger_water(case, interior_view(t.field)),
            (t for t in case.tracers.list if t.name in case.ledger.water_tracers))
    return ["advected positive tracers ≥ 0 (limiter, to round-off)" =>
                all(t -> !(t.advect && t.positive && c.positivity) ||
                         minimum(interior_view(t.field)) >= -1e-15, case.tracers.list),
            "flux form conserves the air water (|ΔW| < 1e-12 W per step)" =>
                !c.refresh_halos || abs(c.dW[]) < 1e-12 * W,
            # under a sound-proof core the transport is W-weighted, and on the
            # anelastic core W is the ledger's ρ̄: no density redistribution
            "anelastic core: ρ̄-weighted transport, no density redistribution (< 1e-12 W)" =>
                !(case.sim isa Aeolus.Anelastic.AnelasticSimulation) || !c.weighted_option ||
                (c.weighted !== nothing && abs(c.dWρ[]) < 1e-12 * W),
            "SGS tracer diffusion alone conserves Σ q V (Σ W q V) to 1e-12" =>
                c.diffusion === nothing ||
                (c.diffusion.resid[] < 1e-12 && abs(c.diffusion.dW[]) < 1e-12 * W),
            "κt ≥ 0 (the diffusivity the tracers see)" =>
                minimum(interior_view(case.sim.κt)) >= 0]
end

smoke_variants(::Type{<:ScalarAdvection}) =
    ((label = "TENO5", options = Dict(:scalar_advection => (scheme = :teno5,))),
     (label = "WENO-Z7, halo 4", options = Dict(:scalar_advection => (scheme = :weno_z7,)),
      halo = (4, 4, 4)))

register!(ScalarAdvection)
