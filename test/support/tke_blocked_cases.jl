# The cases of test/tke_blocked.jl (#554): a TKE closure on a random flow,
# one update from a fixed start, the agreement of the budget statistics.
# Included by test/tke_blocked.jl and by its -t 4 subprocess.

using Aeolus
using Aeolus.Fields: interior_view, set!
using Aeolus.IBM: SolidMask, mark_solid!
using Aeolus.WallModels: immersed_walls, RoughLogLaw
using Random: Xoshiro, randn

# A closure on a random flow: θ stably stratified with noise (N² of both
# signs), e with zeros and a large range (the limiter active), a first
# level, wall distance; optionally a solid mask with wall-modelled
# immersed walls and a surface value.
#
# `buoyancy` (#556): the variant of N² the closure computes: `:dry` (θ),
# `:moist` (θ_v, a moist core with a random qv), `:cloudy` (θ_ρ, and the
# saturated N² where a condensate field qc > 0: about a third of the
# cells) or `:none` (g = 0: N² = 0).
function tkb_case(g; transport = true, dissipation = :semi_implicit, mask = false, walls = false,
                  surface = false, bcs = FieldBoundaryConditions(:periodic, :periodic, :wall),
                  buoyancy = :dry)
    buoyancy in (:dry, :moist, :cloudy, :none) || throw(ArgumentError("buoyancy = $buoyancy"))
    moist = buoyancy in (:moist, :cloudy)
    sim = BoussinesqSimulation(g; θ₀ = 300.0, bcs = bcs, θ_flux_bottom = 0.05, moisture = moist)
    rng = Xoshiro(554)
    n = (g.Nx, g.Ny, g.Nz)
    for f in (sim.u, sim.v, sim.w)
        interior_view(f) .= 2.0 .* randn(rng, size(interior_view(f)))
    end
    interior_view(sim.θ) .= 300 .+ 0.02 .* randn(rng, n) .+ reshape(0.004 .* (1:g.Nz), 1, 1, :)
    moist && (interior_view(sim.qv) .= 0.008 .+ 1e-4 .* randn(rng, n))
    qc = nothing
    if buoyancy === :cloudy
        qc = Field(CellCenter(), g)
        interior_view(qc) .= 1e-4 .* rand(rng, n...) .* (rand(rng, n...) .< 0.35)
    end
    sm = nothing
    if mask
        sm = SolidMask(g)
        mark_solid!(sm, (x, y, z) -> (0.3 < x / (g.Nx * g.dx) < 0.5) && (0.2 < y / (g.Ny * g.dy) < 0.6) &&
                                     z < 0.4 * g.Nz * g.dz)
    end
    fl = falses(n); fl[:, :, 1] .= true
    sv = surface ? (u_star = 0.3 .+ 0.1 .* rand(rng, g.Nx, g.Ny), k = ones(Int, g.Nx, g.Ny)) : nothing
    c = TKEClosure(g; mask = sm === nothing ? nothing : sm.mask, transport = transport,
                   dissipation = dissipation, first_level = fl, surface = sv,
                   wall_distance = 1.0 .+ 30 .* rand(rng, n...),
                   walls = walls ? immersed_walls(sim, sm, RoughLogLaw(0.1)) : nothing,
                   qc = qc, p = qc === nothing ? nothing : 9.5e4,
                   g_accel = buoyancy === :none ? 0.0 : nothing)
    initialize_tke!(c, sim; e_min = 1e-3)
    interior_view(c.e) .*= 4 .* rand(rng, n...) .* (rand(rng, n...) .> 0.15)
    return sim, c
end

# e and the statistics after one update by `f!`, from the same start. The
# scratch the blocked pass reads at the boundary (ν, κt, the limiter's
# ratio; N² and D, which the pass computes row by row without walls, #556)
# is first set to NaN: a reference run on the same closure leaves
# correct values in their interior, which would hide a boundary cell the
# pass failed to compute itself (a NaN read reaches e).
function tkb_run!(f!, c, sim, e0, Δt)
    copyto!(c.e.data, e0); fill!(c.stats, 0)
    fill!(c.ν.data, NaN); fill!(c.κt.data, NaN)
    fill!(c.N2.data, NaN); fill!(c.D.data, NaN)      # computed by the pass itself (#556)
    c.ws === nothing || fill!(c.ws.ratio.data, NaN)
    f!(c, sim, Δt)
    return copy(c.e.data), tke_statistics(c)
end

tkb_stats_agree(a, b) = a.clips == b.clips && a.steps == b.steps &&
    all(isapprox(getfield(a, f), getfield(b, f); rtol = 1e-12, atol = 1e-12 * abs(a.shear))
        for f in (:clipped_energy, :surface_energy, :shear, :buoyancy, :dissipation,
                  :buoyant_destruction, :transport, :wall))
