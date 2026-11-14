# Moist buoyancy: virtual temperature and condensate loading on w, in the
# :body_force hook of step!. The physics is
# `Operators.moist_buoyancy_perturbation!` / `add_moist_buoyancy!`
# (src/operators/moist_buoyancy.jl, ONE-e #454); this file wires it in.
#
# The Boussinesq core carries θ only and adds the dry buoyancy itself
# (`sim.moisture = false`: the case's water lives in the tracer registry,
# not in the core). Condensate is every tracer the ledger counts as water
# other than qv (ql and qr today; P3's qi etc. join through
# `water_tracer!`), so a new microphysics provider is loaded automatically.

"""
    MoistBuoyancy

Additive coupling (on by default): every step, in `:body_force`, the
virtual-temperature and condensate-loading buoyancy
``g (θ_ρ − θ)′/θ₀`` of [`moist_buoyancy_perturbation!`](@ref) /
[`add_moist_buoyancy!`](@ref). `Δ` keeps the last ``(θ_ρ − θ)′`` (K) for
the output field `b_moist` and the diagnostics.
"""
struct MoistBuoyancy{V} <: UrbanComponent
    Δ::Array{Float64,3}
    fluid::Array{Bool,3}
    θ₀::Float64
    g::Float64
    qc::Vector{V}                 # condensate_views(case), collected once in init! (#295)
end

component_name(::Type{<:MoistBuoyancy}) = :moist_buoyancy
modules_used(::Type{<:MoistBuoyancy}) = (:MoistThermodynamics,)
stages(::Type{<:MoistBuoyancy}) = ((:body_force, 20, EVERY_STEP),)
Components.budget_writes(::MoistBuoyancy, ::Val{:body_force}) = ()   # tw alone, no budget quantity (#539)

function build(::Type{<:MoistBuoyancy}, case, o)
    m = case.mesh
    sm = case.geom.sm.mask
    fluid = [!sm[m.Hx+i, m.Hy+j, m.Hz+k] for i in 1:m.Nx, j in 1:m.Ny, k in 1:m.Nz]
    V = typeof(interior_view(case.sim.θ))       # the tracers' interior views are of this type
    return MoistBuoyancy(zeros(m.Nx, m.Ny, m.Nz), fluid, case.cfg.θ₀, case.cfg.g, V[])
end

# every coupling has registered its tracers by now: collect the condensate
# views once instead of in a fresh vector every step
init!(c::MoistBuoyancy, case) = (append!(empty!(c.qc), condensate_views(case)); nothing)

"The interior views of the condensate tracers: the ledger's water tracers other than qv."
condensate_views(case) =
    [interior_view(tracer(case, n)) for n in case.budgets.water_tracers
     if n !== :qv && haskey(case.tracers, n)]

function run_stage!(c::MoistBuoyancy, case, ::Val{:body_force}, dt)
    moist_buoyancy_perturbation!(c.Δ, interior_view(case.sim.θ), interior_view(tracer(case, :qv)),
                                 c.qc, c.fluid)
    add_moist_buoyancy!(case.sim.tw, c.Δ, c.θ₀, c.g)
    return nothing
end

output_fields(c::MoistBuoyancy, case) =
    [OutputVariable("b_moist", "m s-2",
                 "moist buoyancy perturbation g (theta_rho - theta)'/theta_0 (virtual + condensate loading)",
                 :xyz, cs -> c.Δ .* (c.g / c.θ₀))]

"""
    diagnostics(c::MoistBuoyancy, case)

The moist buoyancy perturbation `b_rms` and `b_maxabs` (m s⁻²), and of
the first level the largest relative humidity `rh1_max` (1) and the
smallest dew-point depression `dewpoint_depression1_min` (K)
([`moist_level1`](@ref)).
"""
function diagnostics(c::MoistBuoyancy, case)
    l1 = per_state(() -> moist_level1(c, case), case)
    return Diagnostic[
        Diagnostic("b_rms", "m s-2", () -> sqrt(sum(abs2, c.Δ) / max(count(c.fluid), 1)) * c.g / c.θ₀),
        Diagnostic("b_maxabs", "m s-2", () -> maximum(abs, c.Δ) * c.g / c.θ₀),
        Diagnostic("rh1_max", "1", () -> l1().rh; description = "relative humidity, first level"),
        Diagnostic("dewpoint_depression1_min", "K", () -> l1().dd; description = "T - Td, first level")]
end

"""
    moist_level1(c::MoistBuoyancy, case) -> (rh, dd)

The largest relative humidity `rh` and the smallest dew-point depression
`dd` (K) over the fluid cells of the first level, at the reference
pressure and Exner function of the level.
"""
function moist_level1(c::MoistBuoyancy, case)
    # relative humidity and dew-point depression of the lowest air layer
    θ1 = view(interior_view(case.sim.θ), :, :, 1)
    q1 = view(interior_view(tracer(case, :qv)), :, :, 1)
    p1, Π1 = ref_pressure(case, 1), ref_exner(case, 1)
    rh, dd = 0.0, Inf
    for I in eachindex(θ1)
        c.fluid[I, 1] || continue
        T = θ1[I] * Π1
        rh = max(rh, Aeolus.MoistThermodynamics.relative_humidity(T, p1, q1[I]))
        dd = min(dd, T - Aeolus.MoistThermodynamics.dewpoint_temperature(p1, q1[I]))
    end
    return (rh = rh, dd = dd)
end

function smoke_check(c::MoistBuoyancy, case)
    θ = interior_view(case.sim.θ)
    qv = interior_view(tracer(case, :qv))
    qc = condensate_views(case)
    Δ = similar(c.Δ)
    moist_buoyancy_perturbation!(Δ, θ, qv, qc, c.fluid)
    # θ_ρ − θ = θ ((1/ε − 1) q_v − q_c) exactly, with 1/ε − 1 = 0.608 from
    # the molar masses: the virtual term and the condensate loading
    cst = Aeolus.MoistThermodynamics.DEFAULT_CONSTANTS
    a = cst.molmass_dry / cst.molmass_water - 1
    zero_mean, form_ok = true, true
    for k in axes(Δ, 3)
        f = view(c.fluid, :, :, k)
        n = count(f); n == 0 && continue
        ref = [θ[I, k] * (a * qv[I, k] - sum(q[I, k] for q in qc; init = 0.0))
               for I in CartesianIndices(f)]
        m = sum(ref[I] for I in CartesianIndices(f) if f[I]) / n
        scale = maximum(abs(ref[I] - m) for I in CartesianIndices(f) if f[I])
        zero_mean &= abs(sum(Δ[I, k] for I in CartesianIndices(f) if f[I])) <=
                     1e-12 * n * max(scale, 1e-6)
        # the floor is the round-off of θ_ρ − θ, a difference of two ≈ 300 K
        # numbers: with nearly uniform vapour on a level the relative
        # tolerance alone falls below it
        form_ok &= all(I -> !f[I] || abs(Δ[I, k] - (ref[I] - m)) <=
                                     1e-9 * max(scale, 1e-6) + 16 * eps(θ[I, k]),
                       CartesianIndices(f))
    end
    return ["(θ_ρ − θ)′ has zero mean over the fluid cells of every level (no net force)" => zero_mean,
            "(θ_ρ − θ)′ = [θ (0.608 q_v − q_c)]′ (virtual temperature + condensate loading)" => form_ok,
            "no moist buoyancy in solid cells" => all(I -> c.fluid[I] || Δ[I] == 0, eachindex(Δ)),
            "0.608 = 1/ε − 1 from the MoistThermodynamics molar masses" => abs(a - 0.608) < 1e-3]
end


# --- the fused sweep (#533) ---------------------------------------------------
# The perturbation (θ_ρ − θ)′ needs the level means over the fluid cells of
# the stage-start state, so it is computed before the sweep, into `Δ`, as
# `run_stage!` computes it; the sweep adds g/θ₀ times its value at the w
# face (the expression of `add_moist_buoyancy!`, which skips the bottom face).
struct MoistBuoyancySweep{C<:MoistBuoyancy, Q} <: FVK.SweepTerm
    c::C
    qv::Q                         # the interior view of the qv tracer
    coef::Float64
    Hx::Int
    Hy::Int
    Hz::Int
end
"""
    moist_buoyancy_sweep(c::MoistBuoyancy, mesh, qv) -> SweepTerm

The term of the fused sweep (#533) that adds the moist buoyancy of `c`
(the qv tracer's interior view `qv`) on `mesh`: the same values as its
`run_stage!` (`moist_buoyancy_perturbation!`, `add_moist_buoyancy!`).
"""
moist_buoyancy_sweep(c::MoistBuoyancy, m, qv) = MoistBuoyancySweep(c, qv, c.g / c.θ₀, m.Hx, m.Hy, m.Hz)
FVK.sweep_term(c::MoistBuoyancy, d) =
    moist_buoyancy_sweep(c, d.state.mesh, interior_view(tracer(urban_case(d), :qv)))
FVK.footprint(::MoistBuoyancySweep) = (0, 0, 0)
FVK.sweep_inputs(t::MoistBuoyancySweep) = (t.c.Δ,)
function FVK.prepare!(t::MoistBuoyancySweep, sim, Δt)
    c = t.c
    moist_buoyancy_perturbation!(c.Δ, interior_view(sim.θ), t.qv, c.qc, c.fluid)
    return nothing
end
@inline function FVK.add_row!(t::MoistBuoyancySweep, buf, a, b, s, j, k, ii, st)
    kl = k - t.Hz
    kl >= 2 && kl <= size(t.c.Δ, 3) && _moist_b_row!(buf, a, b, s, j - t.Hy, kl, ii, t.Hx, t.c.Δ, t.coef)
    return nothing
end
# the point kernel of the per-point (GPU) form (#533 step 8)
@inline function FVK.add_tendency!(t::MoistBuoyancySweep, tb, i, j, k, st)
    kl = k - t.Hz
    if kl >= 2 && kl <= size(t.c.Δ, 3)
        Δ = t.c.Δ
        @inbounds tb[i, FVK.SWEEP_W] += t.coef * 0.5 * (Δ[i-t.Hx, j-t.Hy, kl-1] + Δ[i-t.Hx, j-t.Hy, kl])
    end
    return nothing
end
@noinline function _moist_b_row!(buf, a, b, s, jl, kl, ii, Hx, Δ, coef)
    @inbounds @simd ivdep for i in ii
        buf[i, a, b, FVK.SWEEP_W, s] += coef * 0.5 * (Δ[i-Hx, jl, kl-1] + Δ[i-Hx, jl, kl])
    end
    return nothing
end
