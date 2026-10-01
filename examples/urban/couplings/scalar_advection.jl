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

"Reconstructions `advect_positive!` accepts (the flux limiter needs one)."
const POSITIVE_SCHEMES = (:centered, :weno5, :weno_js5, :weno_z5, :weno_z7, :teno5, :teno8)

"""
    ScalarAdvection

Slot `:scalar_advection`, provider `:advection_schemes`: every step, each
tracer registered with `advect = true` (registration order) is advanced
over `dt` with the scheme `scheme` of `Aeolus.AdvectionSchemes`
(`advection_scheme(scheme)`, any of `:centered, :upwind, :weno5,
:weno_js5, :weno_z5, :weno_z7, :teno5, :teno8, :variance_conserving`),
periodic lateral and no-flux vertical halos. Tracers flagged `positive`
use the flux-limited `advect_positive!` with the same reconstruction
(`positivity = false` switches that off). The mesh halo must be at least
`required_halo` of the scheme (`:weno_z7` needs 4, `:teno8` 5; the case
default is 3).

Options: `scheme = :weno_z5`, `positivity = true`, `refresh_halos = true`
(fill the velocity halos before advecting; `false` reproduces the
non-conservative pre-#132 transport).

On a sound-proof core (`core = :anelastic`, `:pseudo_incompressible`)
the transport is `∂(W q)/∂t = −∇·(W u q)` with the core's constraint
weight `W` (`Anelastic.constraint_weights`): the scheme is applied to the
mass-flux velocity `(W/W_min) u` and the increment scaled by `W_min/W_k`,
so `Σ W q` is conserved and a uniform tracer stays uniform. Option
`weighted = false` uses the plain velocity there too (the pre-#133
transport, for comparison).
"""
struct ScalarAdvection{F, W, B, S} <: AbstractCoupling
    tend::F
    ws::W
    bcs::B
    scheme::S                     # AdvectionScheme (the unlimited operator)
    positivity::Bool
    refresh_halos::Bool
    dW::Base.RefValue{Float64}    # this step's change of air water, ρ̄-mean part (kg/m²; 0 if conservative)
    dWρ::Base.RefValue{Float64}   # this step's ρ̄-redistribution part (kg/m²; see ledger_water_split)
    weighted::Any                 # nothing (Boussinesq core), or the W-weighted transport of a sound-proof core
    weighted_option::Bool
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
modules_used(::Type{<:ScalarAdvection}) = (:AdvectionSchemes,)
default_options(::Type{<:ScalarAdvection}) =
    (scheme = :weno_z5, positivity = true, refresh_halos = true, weighted = true)
stages(::Type{<:ScalarAdvection}) = ((:transport, 10, EVERY_STEP),)

function build(::Type{<:ScalarAdvection}, case, o)
    mesh = case.mesh
    scheme = Aeolus.AdvectionSchemes.advection_scheme(o.scheme)
    H = Aeolus.AdvectionSchemes.required_halo(scheme)
    min(mesh.Hx, mesh.Hy, mesh.Hz) >= H ||
        error("scalar_advection: scheme :$(o.scheme) needs halo ≥ $H, the case has $((mesh.Hx, mesh.Hy, mesh.Hz)) (set CaseConfig halo)")
    o.positivity && !(o.scheme in POSITIVE_SCHEMES) &&
        error("scalar_advection: the positivity limiter takes one of $(POSITIVE_SCHEMES), not :$(o.scheme); set positivity = false")
    return ScalarAdvection(Field(CellCenter(), mesh), FluxLimiterWorkspace(mesh),
                           FieldBoundaryConditions(:periodic, :periodic, :noflux),
                           scheme, o.positivity, o.refresh_halos, Ref(0.0), Ref(0.0),
                           o.weighted ? transport_weighting(case.sim, mesh) : nothing, o.weighted)
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
    c.dW[] = 0.0; c.dWρ[] = 0.0
    wt = c.weighted
    u, v, w = wt === nothing ? (sim.u, sim.v, sim.w) :
                               (mass_flux_velocity!(wt, sim); (wt.U, wt.V, wt.Wv))
    for t in case.tracers
        t.advect || continue
        q = t.field
        fill_halos!(q, c.bcs)
        fill!(c.tend.data, 0)
        if t.positive && c.positivity
            advect_positive!(c.tend, u, v, w, q, dt;
                             scheme = c.scheme.name, workspace = c.ws, bcs = c.bcs)
        else
            c.scheme(c.tend, u, v, w, q)
        end
        tend = view(c.tend.data, Hx+1:Hx+Nx, Hy+1:Hy+Ny, Hz+1:Hz+Nz)
        # weighted: −∇·(W u q)/W_min → −∇·(W u q)/W_k
        wt === nothing || (tend .*= reshape(wt.s, 1, 1, :))
        interior_view(q) .+= dt .* tend
        if t.name in case.ledger.water_tracers
            flat, redis = ledger_water_split(case, tend, wt === nothing ? nothing : wt.W)
            c.dW[] += dt * flat; c.dWρ[] += dt * redis
        end
    end
    return nothing
end

# Ledger: a flux-form scheme on a divergence-free velocity with periodic /
# no-flux boundaries changes no domain total, so anything it does change
# is booked as `transport` — a measure of non-conservation (zero to
# round-off with refreshed halos). Under a sound-proof core the conserved
# total is Σ W q (see the file header) and `transport` measures that. The ledger integrates with the
# reference density ρ̄_k (#131), which an incompressible transport does
# not conserve: moving vapour up into thinner air loses ρ̄-weighted water.
# That part is booked separately as `density_redistribution`
# (ledger_water_split), so `transport` stays the scheme's own
# non-conservation.
budget!(c::ScalarAdvection, case, l) =
    (add_water!(l, :transport, c.dW[]); add_water!(l, :density_redistribution, c.dWρ[]); nothing)

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
                (c.weighted !== nothing && abs(c.dWρ[]) < 1e-12 * W)]
end

smoke_variants(::Type{<:ScalarAdvection}) =
    ((label = "TENO5", options = Dict(:scalar_advection => (scheme = :teno5,))),
     (label = "WENO-Z7, halo 4", options = Dict(:scalar_advection => (scheme = :weno_z7,)),
      halo = (4, 4, 4)))

register!(ScalarAdvection)
