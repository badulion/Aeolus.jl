# Transport of the registered tracers: WENO-Z5 with the positivity
# limiter (Aeolus.AdvectionSchemes, Phase 26), forward Euler in time.
#
# qv and ql are ACTIVE (fog/dew feed latent heat back into θ and Ts), so
# they need a stable, non-oscillatory scheme. The former forward-Euler +
# 2nd-order centred update is unconditionally unstable; as a passive
# field its grid-scale noise was simply clipped at the 0.03 clamp (RH
# reached 2.07 in 8,395 cells within 1000 steps), but as soon as
# condensation was coupled that noise became latent heat and the run blew
# up within 500 steps. Cloud water must move with the air too, or
# condensate stays put while its vapour is carried away.
#
# Known residual: a few wall-nook cells (e.g. (64,58,1)) still gain
# vapour, because the IBM correction applied after the projection leaves
# the velocity slightly convergent there.

"""
    TransportWENO

Slot `:scalar_advection`, provider `:weno_z5`: every step, each tracer
registered with `advect = true` (registration order) is advected with
`advect_positive!(...; scheme = :weno_z5)` over `dt` with periodic
lateral and no-flux vertical halos. Needs a mesh halo ≥ 3.
"""
struct TransportWENO{F, W, B} <: AbstractCoupling
    tend::F
    ws::W
    bcs::B
    dW::Base.RefValue{Float64}   # this step's change of air water, ρ̄-mean part (kg/m²; 0 if conservative)
    dWρ::Base.RefValue{Float64}  # this step's ρ̄-redistribution part (kg/m²; see ledger_water_split)
end

coupling_name(::Type{<:TransportWENO}) = :transport_weno
slot(::Type{<:TransportWENO}) = :scalar_advection
provider(::Type{<:TransportWENO}) = :weno_z5
modules_used(::Type{<:TransportWENO}) = (:AdvectionSchemes,)
stages(::Type{<:TransportWENO}) = ((:transport, 10, EVERY_STEP),)

function build(::Type{<:TransportWENO}, case, o)
    mesh = case.mesh
    min(mesh.Hx, mesh.Hy, mesh.Hz) >= 3 || error("transport_weno: WENO-Z5 needs halo ≥ 3")
    return TransportWENO(Field(CellCenter(), mesh), FluxLimiterWorkspace(mesh),
                         FieldBoundaryConditions(:periodic, :periodic, :noflux), Ref(0.0), Ref(0.0))
end

function run_stage!(c::TransportWENO, case, ::Val{:transport}, dt)
    sim, mesh = case.sim, case.mesh
    Hx, Hy, Hz = mesh.Hx, mesh.Hy, mesh.Hz
    Nx, Ny, Nz = mesh.Nx, mesh.Ny, mesh.Nz
    c.dW[] = 0.0; c.dWρ[] = 0.0
    for t in case.tracers
        t.advect || continue
        q = t.field
        fill_halos!(q, c.bcs)
        fill!(c.tend.data, 0)
        advect_positive!(c.tend, sim.u, sim.v, sim.w, q, dt;
                         scheme = :weno_z5, workspace = c.ws, bcs = c.bcs)
        tend = view(c.tend.data, Hx+1:Hx+Nx, Hy+1:Hy+Ny, Hz+1:Hz+Nz)
        interior_view(q) .+= dt .* tend
        if t.name in case.ledger.water_tracers
            flat, redis = ledger_water_split(case, tend)
            c.dW[] += dt * flat; c.dWρ[] += dt * redis
        end
    end
    return nothing
end

# Ledger: a flux-form scheme on a divergence-free velocity with periodic /
# no-flux boundaries changes no domain total, so anything it does change
# is booked as `transport` — a measure of non-conservation (KNOWN: the
# stale velocity halos of #132 make it nonzero). The ledger integrates
# with the reference density ρ̄_k (#131), which an incompressible
# transport does not conserve: moving vapour up into thinner air loses
# ρ̄-weighted water. That part is booked separately as
# `density_redistribution` (ledger_water_split), so `transport` stays the
# scheme's own non-conservation.
budget!(c::TransportWENO, case, l) =
    (add_water!(l, :transport, c.dW[]); add_water!(l, :density_redistribution, c.dWρ[]); nothing)

smoke_check(c::TransportWENO, case) =
    ["advected tracers ≥ 0 (positivity limiter, to round-off)" =>
         all(t -> !t.advect || minimum(interior_view(t.field)) >= -1e-15, case.tracers.list)]

register!(TransportWENO)
