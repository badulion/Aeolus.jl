# The moist buoyancy perturbation (ONE-e, #454; moved from the urban
# case's `moist_buoyancy` coupling, which only wired it into its step).
#
# A core that carries θ only adds the dry buoyancy g (θ − θ₀)/θ₀. The
# buoyancy of moist, cloudy air is that of its density potential
# temperature,
#
#     θ_ρ = θ (1 + q_v/ε − q_t),      q_t = q_v + Σ condensate,
#
# (`MoistThermodynamics.density_temperature`, linear in its temperature
# argument, so it gives θ_ρ from θ exactly). The difference
# g (θ_ρ − θ)′/θ₀ ≈ g (0.608 q_v′ − q_c′) is added at every interior w
# face. The prime is the deviation from the horizontal mean of the fluid
# cells of each level: a horizontally uniform force is a pure pressure
# gradient in an incompressible or anelastic core (the projection removes
# it exactly), so subtracting it changes only the diagnosed pressure and
# keeps the term the size of the physics (1e-3 m s⁻² for 0.2 g/kg of fog)
# instead of the 0.06 m s⁻² of 10 g/kg of mean vapour.
#
# Reference: Emanuel, K. A. (1994), Atmospheric Convection, §4.3 (density
# temperature).

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
    plane_reduce!((S, n), Backends.backend(Δ), _moist_Δ_at, (1:Nx, 1:Ny, 1:Nz),
                  Δ, θ, qv, qc, fluid)
    # the deviation, one work item per level (KA-k, #307): the level's mean
    # is a local, so the masked update vectorises as the former loop did (a
    # pointwise body reloaded the mean at every point: 3.8x)
    launch_weighted!(Backends.backend(Δ), Nx * Ny, _moist_Δ_dev!, (1:Nz,),
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
            d = density_temperature(
                    θk, PhasePartition(qv[i, j, k] + c, c)) - θk
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
    launch!(Backends.backend(d), _moist_b_body!,
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
