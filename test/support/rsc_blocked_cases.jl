# The cases of test/reynolds_stress_blocked.jl (#560): a Reynolds-stress
# closure on a random flow, one update from a fixed start, the agreement of
# the booked statistics. Included by test/reynolds_stress_blocked.jl and by
# its -t 4 subprocess.

using Aeolus
using Aeolus.Fields: interior_view, set!
using Aeolus.IBM: SolidMask, mark_solid!
using Random: Xoshiro, randn

# A closure on a random flow: θ stably stratified with noise (N² of both
# signs), the stresses from the local equilibrium, scaled at random with
# zeros and random off-diagonals (so the limiter and the realizability
# projection act), a first level, wall distance; optionally a solid mask,
# rotation and a surface value.
function rsc_case(g; transport = true, advection = :auto, mask = false, rotation = false,
                  surface = false, bcs = FieldBoundaryConditions(:periodic, :periodic, :wall))
    sim = BoussinesqSimulation(g; θ₀ = 300.0, bcs = bcs, θ_flux_bottom = 0.05)
    rng = Xoshiro(560)
    n = (g.Nx, g.Ny, g.Nz)
    for f in (sim.u, sim.v, sim.w)
        interior_view(f) .= 0.5 .* randn(rng, size(interior_view(f)))
    end
    interior_view(sim.θ) .= 300 .+ 0.02 .* randn(rng, n) .+ reshape(0.004 .* (1:g.Nz), 1, 1, :)
    sm = nothing
    if mask
        sm = SolidMask(g)
        mark_solid!(sm, (x, y, z) -> (0.3 < x / (g.Nx * g.dx) < 0.5) && (0.2 < y / (g.Ny * g.dy) < 0.6) &&
                                     z < 0.4 * g.Nz * g.dz)
    end
    fl = falses(n); fl[:, :, 1] .= true
    sv = surface ? (u_star = 0.3 .+ 0.1 .* rand(rng, g.Nx, g.Ny), k = ones(Int, g.Nx, g.Ny)) : nothing
    c = ReynoldsStressClosure(g; mask = sm === nothing ? nothing : sm.mask, transport = transport,
                              advection = advection, first_level = fl, surface = sv,
                              wall_distance = 1.0 .+ 30 .* rand(rng, n...),
                              Ω = rotation ? (0.0, 3e-5, 5e-5) : :auto)
    initialize_reynolds_stress!(c, sim; e_min = 1e-3)
    for m in 1:3
        interior_view(c.τ[m]) .*= (0.5 .+ rand(rng, n...)) .* (rand(rng, n...) .> 0.1)
    end
    for m in 4:6
        interior_view(c.τ[m]) .= 0.3 .* randn(rng, n...) .* sqrt.(interior_view(c.τ[1]) .* interior_view(c.τ[2]))
    end
    return sim, c
end

# the six stresses and the statistics after one update by `f!`, from the
# same start. The scratch the blocked pass does not write in the interior
# (ν, the tendencies, the limiter's ratio) is first set to NaN: a reference
# run on the same closure leaves correct values there, which would hide a
# boundary cell the pass failed to compute itself (a NaN read reaches τ).
function rsc_run!(f!, c, sim, τ0, Δt)
    foreach((f, a) -> copyto!(f.data, a), c.τ, τ0); fill!(c.stats, 0)
    fill!(c.ν.data, NaN); fill!(c.K2.data, NaN)
    foreach(t -> fill!(t.data, NaN), c.tend)
    c.ws === nothing || fill!(c.ws.ratio.data, NaN)
    f!(c, sim, Δt)
    return map(f -> copy(f.data), c.τ), reynolds_stress_statistics(c)
end

# the largest difference of the six stresses relative to the largest stress
rsc_diff(a, b) = maximum(maximum(abs.(x .- y)) for (x, y) in zip(a, b)) / maximum(maximum(abs, y) for y in b)

# the integer counts exactly, the budget sums within a relative 1e-12 (the
# same cell values summed in a different grouping) of the largest of them
function rsc_stats_agree(a, b)
    a.clips == b.clips && a.steps == b.steps && a.step_clips == b.step_clips &&
        a.step_split_cells == b.step_split_cells && a.step_unrealizable_input == b.step_unrealizable_input ||
        return false
    sc = max(abs(b.shear), abs(b.dissipation), abs(b.buoyancy), abs(b.transport), 1e-300)
    return all(isapprox(getfield(a, f), getfield(b, f); rtol = 1e-12, atol = 1e-12 * sc)
               for f in (:shear, :buoyancy, :dissipation, :transport, :surface_energy)) &&
           isapprox(a.clipped_energy, b.clipped_energy; rtol = 1e-12, atol = 1e-12 * abs(b.clipped_energy) + 1e-300)
end
