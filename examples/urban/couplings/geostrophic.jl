# Background geostrophic wind: Rayleigh nudging of the horizontal-mean
# u, v toward (U_g, V_g) in the body-force hook.

"""
    add_geostrophic_wind!(sim, U_g, V_g, τ_relax)

Add `-(⟨u⟩ - U_g)/τ` to every u-cell's tendency (and similarly for v)
so the horizontal mean wind is pulled toward the geostrophic value
`U_g` on a time scale `τ_relax`. Preserves turbulent fluctuations
because only the horizontal mean is nudged.

Call this from inside a `body_force!` callback passed to `step!`.
"""
function add_geostrophic_wind!(sim, U_g::Real, V_g::Real, τ_relax::Real)
    Nx, Ny = sim.mesh.Nx, sim.mesh.Ny
    Hx, Hy, Hz = sim.mesh.Hx, sim.mesh.Hy, sim.mesh.Hz
    Nz = sim.mesh.Nz
    invτ = 1 / τ_relax
    @inbounds for k in 1:Nz
        ū = 0.0; v̄ = 0.0
        for j in 1:Ny, i in 1:Nx
            ū += sim.u.data[Hx+i, Hy+j, Hz+k]
            v̄ += sim.v.data[Hx+i, Hy+j, Hz+k]
        end
        ū /= (Nx * Ny); v̄ /= (Nx * Ny)
        for j in 1:Ny, i in 1:Nx
            sim.tu.data[Hx+i, Hy+j, Hz+k] -= invτ * (ū - U_g)
            sim.tv.data[Hx+i, Hy+j, Hz+k] -= invτ * (v̄ - V_g)
        end
    end
    return sim
end

"""
    GeostrophicCoupling

Additive coupling: [`add_geostrophic_wind!`](@ref) in `:body_force` with
options `U_g`, `V_g` (m/s) and `τ` (s, default 1800).
"""
struct GeostrophicCoupling <: AbstractCoupling
    U_g::Float64
    V_g::Float64
    τ::Float64
end

coupling_name(::Type{GeostrophicCoupling}) = :geostrophic
default_options(::Type{GeostrophicCoupling}) = (U_g = 1.0, V_g = 0.0, τ = 1800.0)
stages(::Type{GeostrophicCoupling}) = ((:body_force, 10, EVERY_STEP),)
build(::Type{GeostrophicCoupling}, case, o) = GeostrophicCoupling(o.U_g, o.V_g, o.τ)

run_stage!(c::GeostrophicCoupling, case, ::Val{:body_force}, dt) =
    add_geostrophic_wind!(case.sim, c.U_g, c.V_g, c.τ)

register!(GeostrophicCoupling)
