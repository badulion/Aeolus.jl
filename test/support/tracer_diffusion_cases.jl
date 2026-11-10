# The cases of test/tracer_diffusion_blocked.jl (#559): the urban tracer
# SGS diffusion (`Urban.TracerSGSDiffusion`, `sgs_diffuse!`) on a random
# tracer and diffusivity, without building an urban case: `sgs_diffuse!`
# reads only the case's `sim` (its κt and loop-fusion setting), `mesh` and
# `budgets`. Included by test/tracer_diffusion_blocked.jl and by its -t 4
# subprocess.

using Aeolus
using Aeolus.Fields: interior_view
using Aeolus.BoundaryConditions: fill_halos!
using Random: Xoshiro, rand

const TDU = Aeolus.Urban

# A case on grid `g`: a Boussinesq or anelastic core (the anelastic one
# with a stable profile, so the constraint weights vary with height), a
# solid mask of two blocks (one touching the ground, one crossing the
# periodic x seam), the tracer and mask halos `bcs`, a random tracer with
# zeros and a random κt; κv a second random field when `aniso`. The
# ledger's ρ̄ decreases with height (the water split's redistribution
# part is then not zero).
function tdc_case(g; core = :boussinesq, bcs = FieldBoundaryConditions(:periodic, :periodic, :noflux),
                  aniso = false, solid = true, seed = 559)
    sim = if core === :boussinesq
        BoussinesqSimulation(g; θ₀ = 300.0)
    else
        θf(z) = 300 + 3e-3 * z
        Aeolus.Anelastic.AnelasticSimulation(g; ref = Aeolus.Anelastic.reference_from_profile(g, θf, 101325.0))
    end
    rng = Xoshiro(seed)
    n = (g.Nx, g.Ny, g.Nz)
    mf = Field(CellCenter(), g)
    if solid
        x(i) = (i - 0.5) / g.Nx; y(j) = (j - 0.5) / g.Ny; z(k) = (k - 0.5) / g.Nz
        for k in 1:g.Nz, j in 1:g.Ny, i in 1:g.Nx
            b1 = 0.3 < x(i) < 0.55 && 0.2 < y(j) < 0.6 && z(k) < 0.45
            b2 = (x(i) < 0.12 || x(i) > 0.9) && 0.6 < y(j) < 0.8 && 0.2 < z(k) < 0.7
            mf.data[g.Hx + i, g.Hy + j, g.Hz + k] = (b1 || b2) ? 1.0 : 0.0
        end
    end
    fill_halos!(mf, bcs)
    mask = Array{Bool}(mf.data .> 0.5)
    w = sim isa Aeolus.Anelastic.MassWeightedSimulation ? Aeolus.Anelastic.constraint_weights(sim) : nothing
    q = Field(CellCenter(), g)
    interior_view(q) .= rand(rng, n...) .* (rand(rng, n...) .> 0.1) .* 1e-2
    interior_view(sim.κt) .= 5 .* rand(rng, n...)
    fill_halos!(sim.κt, bcs)
    κv = nothing
    if aniso
        κv = Field(CellCenter(), g)
        interior_view(κv) .= 2 .* rand(rng, n...)
        fill_halos!(κv, bcs)
    end
    budgets = Aeolus.BudgetSystem.Budgets()
    budgets.ρ = [1.2 * exp(-k / (2g.Nz)) for k in 1:g.Nz]
    case = (sim = sim, mesh = g, budgets = budgets)
    d = tdc_diffusion(g, mask, w, bcs)
    return case, d, q, κv
end

# a TracerSGSDiffusion of the fields above, with the blocked pass's
# scratch for `ns` slabs (the default: one per thread)
function tdc_diffusion(g, mask, w, bcs; ns = nothing)
    t = Field(CellCenter(), g)
    nsl = ns === nothing ? TDU._tracer_slabs(t) : ns
    sc = Aeolus.Operators._eddy_diffusion_scratch(nothing, t.data, nsl, g.Ny, g.Nz)
    return TDU.TracerSGSDiffusion(t, mask, w, bcs, Ref(0.0), Ref(0.0), Ref(0.0), zeros(g.Nz), zeros(g.Nz), sc)
end

# q after one sgs_diffuse! of `q0` (water booked), and the booked figures
function tdc_run!(case, d, q, q0, dt; κv = nothing)
    copyto!(q.data, q0)
    d.dW[] = 0.0; d.dWρ[] = 0.0; d.resid[] = 0.0
    κ = case.sim.κt
    TDU.sgs_diffuse!(d, case, q, dt; water = true, κ = κ, κv = κv === nothing ? κ : κv)
    return (q = copy(q.data), dW = d.dW[], dWρ = d.dWρ[], resid = d.resid[], Sq = copy(d.Sq), St = copy(d.St))
end

# The blocked result `b` against the separate loops `s`: q within 4 eps of
# max q (the same expressions, so 0 is expected; the bound allows a compiler
# contraction to a fused multiply-add), the per-level sums of |q| within a
# relative 1e-13 and of the tendency within 1e-13 of Σ|tendency| scale (a
# different summation order: rows in ascending j against the plane
# reduction's lanes), the booked water within 1e-12 of the step's water
# scale. Returns the largest errors too.
function tdc_agree(b, s, case, dt)
    eq = maximum(abs.(b.q .- s.q)) / maximum(abs, s.q)
    eSq = maximum(abs.(b.Sq .- s.Sq)) / maximum(abs, s.Sq)
    stscale = max(maximum(abs, s.St), 1e-300)
    eSt = maximum(abs.(b.St .- s.St)) / stscale
    # the water scale of the step: dt Σ_k ρ_k |St_k| Δz / (Nx Ny)
    g = case.mesh
    wscale = dt * sum(case.budgets.ρ .* abs.(s.St)) * g.dz / (g.Nx * g.Ny) + 1e-300
    eW = max(abs(b.dW - s.dW), abs(b.dWρ - s.dWρ)) / wscale
    ok = eq <= 4eps() && eSq <= 1e-13 && eSt <= 1e-13 && eW <= 1e-12 &&
         (b.resid <= 1e-12) == (s.resid <= 1e-12)
    return ok, (q = eq, Sq = eSq, St = eSt, W = eW)
end
