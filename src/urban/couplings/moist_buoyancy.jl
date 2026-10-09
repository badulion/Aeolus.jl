# Moist buoyancy: virtual temperature and condensate loading on w
# (Aeolus.MoistThermodynamics), in the :body_force hook of step!.
#
# The Boussinesq core carries θ only and adds the dry buoyancy
# g (θ − θ₀)/θ₀ itself (`sim.moisture = false`: the case's water lives in
# the tracer registry, not in the core). The buoyancy of moist, cloudy air
# is that of its density potential temperature,
#
#     θ_ρ = θ (1 + q_v/ε − q_t),      q_t = q_v + Σ condensate,
#
# (`MoistThermodynamics.density_temperature`, which is linear in its
# temperature argument, so it gives θ_ρ from θ exactly). This coupling adds
# the difference, g (θ_ρ − θ)′/θ₀ ≈ g (0.608 q_v′ − q_c′), at every
# interior w face. The prime is the deviation from the horizontal mean of
# the fluid cells of each level: a horizontally uniform force is a pure
# pressure gradient in the incompressible core (the projection removes it
# exactly), so subtracting it changes only the diagnosed pressure and
# keeps the term the size of the physics (1e-3 m s⁻² for 0.2 g/kg of fog)
# instead of the 0.06 m s⁻² of 10 g/kg of mean vapour.
#
# Condensate is every tracer the ledger counts as water other than qv
# (ql and qr today; P3's qi etc. join through `water_tracer!`), so a new
# microphysics provider is loaded automatically.

"""
    moist_buoyancy_perturbation!(Δ, θ, qv, qc, fluid) -> Δ

Fill the interior-sized array `Δ` with ``(θ_ρ − θ)′`` (K): per cell
``θ_ρ − θ`` from `MoistThermodynamics.density_temperature(θ, q)` with
`q = PhasePartition(qv + Σqc, Σqc)`, minus its mean over the fluid cells
(`fluid[i,j,k]`) of the same level. Solid cells get 0. `qc` is a
collection of interior-sized condensate arrays (may be empty).
"""
function moist_buoyancy_perturbation!(Δ::AbstractArray{<:Real,3}, θ::AbstractArray{<:Real,3},
                                      qv::AbstractArray{<:Real,3}, qc::AbstractVector,
                                      fluid::AbstractArray{Bool,3})
    Nx, Ny, Nz = size(Δ)
    # the cell values fused with the per-level fluid sums and counts, in the
    # canonical, thread-invariant plane order (KA-b, #296)
    S, n = zeros(Nz), zeros(Nz)
    plane_reduce!((S, n), Aeolus.Backends.backend(Δ), _moist_Δ_at, (1:Nx, 1:Ny, 1:Nz),
                  Δ, θ, qv, qc, fluid)
    # the deviation, one work item per level (KA-k, #307): the level's mean
    # is a local, so the masked update vectorises as the former loop did (a
    # pointwise body reloaded the mean at every point: 3.8x)
    Aeolus.Backends.launch_weighted!(Aeolus.Backends.backend(Δ), Nx * Ny, _moist_Δ_dev!, (1:Nz,),
                                     Δ, S, n, fluid)
    return Δ
end

@inline function _moist_Δ_dev!(k, Δ, S, n, fluid)
    Nx, Ny = size(Δ, 1), size(Δ, 2)
    @inbounds begin
        m = n[k] > 0 ? S[k] / n[k] : 0.0
        for j in 1:Ny, i in 1:Nx
            fluid[i, j, k] && (Δ[i, j, k] -= m)
        end
    end
    return nothing
end

@inline function _moist_Δ_at(i, j, k, Δ, θ, qv, qc, fluid)
    @inbounds begin
        if fluid[i, j, k]
            c = 0.0
            for q in qc
                c += q[i, j, k]
            end
            θk = θ[i, j, k]
            d = Aeolus.MoistThermodynamics.density_temperature(
                    θk, Aeolus.MoistThermodynamics.PhasePartition(qv[i, j, k] + c, c)) - θk
            Δ[i, j, k] = d
            return (Float64(d), 1.0)
        else
            Δ[i, j, k] = 0.0
            return (0.0, 0.0)
        end
    end
end

"""
    add_moist_buoyancy!(tw, Δ, θ₀, g) -> tw

Add ``g\\,Δ/θ₀`` to the w-momentum tendency `tw` (a `FaceZ` field) at the
interior faces `k = 2 … Nz`, with `Δ` interpolated arithmetically from
the two adjacent cell centres (as the core interpolates θ for its dry
buoyancy). The wall faces `k = 1` and `Nz + 1` carry no flow.
"""
function add_moist_buoyancy!(tw, Δ::AbstractArray{<:Real,3}, θ₀::Real, g::Real)
    m = tw.mesh
    Hx, Hy, Hz = m.Hx, m.Hy, m.Hz
    Nx, Ny, Nz = size(Δ)
    coef = g / θ₀
    d = tw.data
    # pointwise over the interior w faces, in storage indices (KA-k, #307)
    Aeolus.Backends.launch!(Aeolus.Backends.backend(d), _moist_b_body!,
                            (Hx+1:Hx+Nx, Hy+1:Hy+Ny, Hz+2:Hz+Nz), d, Δ, coef, Hx, Hy, Hz)
    return tw
end

@inline function _moist_b_body!(I, J, K, d, Δ, coef, Hx, Hy, Hz)
    @inbounds begin
        i, j, k = I - Hx, J - Hy, K - Hz
        d[I, J, K] += coef * 0.5 * (Δ[i, j, k-1] + Δ[i, j, k])
    end
    return nothing
end

"""
    MoistBuoyancy

Additive coupling (on by default): every step, in `:body_force`, the
virtual-temperature and condensate-loading buoyancy
``g (θ_ρ − θ)′/θ₀`` of [`moist_buoyancy_perturbation!`](@ref) /
[`add_moist_buoyancy!`](@ref). `Δ` keeps the last ``(θ_ρ − θ)′`` (K) for
the output field `b_moist` and the diagnostics.
"""
struct MoistBuoyancy{V} <: AbstractCoupling
    Δ::Array{Float64,3}
    fluid::Array{Bool,3}
    θ₀::Float64
    g::Float64
    qc::Vector{V}                 # condensate_views(case), collected once in init! (#295)
end

coupling_name(::Type{<:MoistBuoyancy}) = :moist_buoyancy
modules_used(::Type{<:MoistBuoyancy}) = (:MoistThermodynamics,)
stages(::Type{<:MoistBuoyancy}) = ((:body_force, 20, EVERY_STEP),)

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
    [interior_view(tracer(case, n)) for n in case.ledger.water_tracers
     if n !== :qv && haskey(case.tracers, n)]

function run_stage!(c::MoistBuoyancy, case, ::Val{:body_force}, dt)
    moist_buoyancy_perturbation!(c.Δ, interior_view(case.sim.θ), interior_view(tracer(case, :qv)),
                                 c.qc, c.fluid)
    add_moist_buoyancy!(case.sim.tw, c.Δ, c.θ₀, c.g)
    return nothing
end

output_fields(c::MoistBuoyancy, case) =
    [OutputField("b_moist", "m s-2",
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

register!(MoistBuoyancy)
