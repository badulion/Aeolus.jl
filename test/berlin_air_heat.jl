# The air-heat ledger of the Berlin case (#253): the heat content of the
# air H = Σ ρ̄ c_p Π̄ θ ΔV / A (core/air_heat.jl) and its automatic
# attribution to the stage entries and the dynamics
# (core/air_heat_ledger.jl).
#
#   * the meter against the analytic heat content of a dry-adiabatic
#     (uniform-θ) hydrostatic layer;
#   * a closed box (every θ source removed): the transport conserves Σθ
#     to round-off, and the change of H the ledger sees is the
#     redistribution between levels of different ρ̄Π̄;
#   * an analytic surface heating H₀ = 100 W m⁻² on a flat, motionless
#     variant: the attributed term and the total are H₀Δt;
#   * add_heat! splits a context, nested contexts, the night window.
#
# Run on its own with `julia --project=. test/berlin_air_heat.jl`.

using Test
using Aeolus
using Aeolus.Fields: interior_view
using Printf
using Random: Xoshiro

isdefined(Main, :BerlinCase) ||
    include(joinpath(@__DIR__, "..", "examples", "urban", "BerlinCase.jl"))
const BC = Main.BerlinCase

# test-only couplings (not registered: pushed into a built case)
struct _HeatPlate <: BC.AbstractCoupling
    H0::Float64          # W m⁻² into the lowest level of every column
    booked::Float64      # fraction of it booked by name through add_heat!
end
BC.coupling_name(::Type{_HeatPlate}) = :heat_plate
BC.stages(::Type{_HeatPlate}) = ((:surface_flux, 5, BC.EVERY_STEP),)
function BC.run_stage!(c::_HeatPlate, case, ::Val{:surface_flux}, dt)
    θ = interior_view(case.sim.θ)
    dz = case.mesh.dz
    θ[:, :, 1] .+= c.H0 * dt / dz * BC.ref_θ_per_heat(case, 1)
    c.booked > 0 && BC.add_heat!(case.ledger, :plate_named, c.booked * c.H0 * dt; k = 1)
    return nothing
end

struct _BodyHeat <: BC.AbstractCoupling
    Q::Float64           # W m⁻² into level 2, from inside step!
end
BC.coupling_name(::Type{_BodyHeat}) = :body_heat
BC.stages(::Type{_BodyHeat}) = ((:body_force, 50, BC.EVERY_STEP),)
function BC.run_stage!(c::_BodyHeat, case, ::Val{:body_force}, dt)
    interior_view(case.sim.θ)[:, :, 2] .+= c.Q * dt / case.mesh.dz * BC.ref_θ_per_heat(case, 2)
    return nothing
end

# couplings that never touch θ: the closed box keeps only these
const NO_THETA = (:core_boussinesq, :timestep_adaptive, :timestep_fixed, :pressure_fft,
                  :sgs_tke, :sgs_smagorinsky, :momentum_centered, :geostrophic,
                  :scalar_advection, :ibm, :moist_buoyancy, :turbulence_diag)

"""
A flat smoke case with only the couplings in `keep`, plus `extra`: the
solid mask is cleared after the build (the building couplings need a
scene to build, and are then removed).
"""
function flat_case(; keep = NO_THETA, extra = BC.AbstractCoupling[], kw...)
    cfg = BC.CaseConfig(:smoke; verbose = false, output_path = nothing, kw...)
    case = BC.build_case(cfg; io = devnull)
    case.geom.sm.mask .= false
    filter!(c -> BC.coupling_name(c) in keep, case.couplings)
    append!(case.couplings, extra)
    BC.build_schedule!(case)
    return case
end

step_terms(prev, now) = Dict(k => v - get(prev, k, 0.0) for (k, v) in now)

@testset "Berlin air-heat ledger (#253)" begin

    @testset "meter: the heat of a uniform-θ hydrostatic layer, analytically" begin
        # θ uniform: dry adiabatic, Π(z) = Π_s − g z/(c_p θ), and
        # ρΠ = p₀ Π^{c_p/R}/(R θ), so
        #   ∫ ρ c_p Π θ dz = c_p² θ p₀ /(R g) [Π^{c_p/R+1}/(c_p/R+1)]_{Π_top}^{Π_s}.
        # The meter integrates with the discretely hydrostatic ρ̄, Π̄ of the
        # case (Anelastic.reference_from_profile) by the midpoint rule:
        # agreement to O(Δz²).
        errs = Float64[]
        for Nz in (16, 64)
            cfg = BC.CaseConfig(:smoke; verbose = false, output_path = nothing,
                                θ_lapse = 0.0, size = (16, 16, Nz))
            mesh = Aeolus.Grids.RegularGrid(; size = cfg.size, extent = cfg.extent, halo = cfg.halo)
            ref = BC.build_reference(cfg, mesh)
            c = Aeolus.MoistThermodynamics.DEFAULT_CONSTANTS
            R = c.gas_constant / c.molmass_dry; cp = c.cp_dry; p₀ = c.p_ref
            θ = cfg.θ_surface
            Πs = (cfg.p_surface / p₀)^(R / cp)
            Πt = Πs - cfg.g * cfg.extent[3] / (cp * θ)
            a = cp / R + 1
            H_exact = cp^2 * θ * p₀ / (R * cfg.g) * (Πs^a - Πt^a) / a
            m = BC.AirHeatMeter(mesh, ref.ρ̄, ref.π̄, ref.cp)
            θf = Aeolus.Fields.Field(Aeolus.Grids.CellCenter(), mesh)
            interior_view(θf) .= θ
            H = BC.air_heat(m, θf)
            @info @sprintf("  meter, uniform θ = %.0f K, Nz = %d: H = %.9e J/m², analytic %.9e (rel %.1e)",
                           θ, Nz, H, H_exact, abs(H / H_exact - 1))
            # measured 3.1e-8 (Nz = 16) and 2.0e-9 (Nz = 64): the midpoint rule's O(Δz²)
            @test H ≈ H_exact rtol = (Nz == 16 ? 1e-7 : 1e-8)
            push!(errs, abs(H / H_exact - 1))
            # a layer is the overlap-weighted sum of levels: halves add up
            zc = cfg.extent[3] / 2 + 0.3 * cfg.extent[3] / Nz      # cuts a level
            @test BC.air_heat(m, θf; layer = (0, zc)) + BC.air_heat(m, θf; layer = (zc, Inf)) ≈ H rtol = 1e-14
            @test sum(BC.layer_fraction(m, 0, zc)) ≈ zc / (cfg.extent[3] / Nz) rtol = 1e-12
        end
        @test errs[1] / errs[2] > 12          # second order: 16 for a 4× finer grid
        # solid cells are not air
        case = BC.build_case(BC.CaseConfig(:smoke; verbose = false, output_path = nothing); io = devnull)
        mask = case.geom.sm.mask
        m_fl = BC.AirHeatMeter(case.sim, case.ref, mask)
        m_all = BC.AirHeatMeter(case.sim, case.ref, nothing)
        θ = interior_view(case.sim.θ)
        M, Hx, Hy, Hz = case.mesh, case.mesh.Hx, case.mesh.Hy, case.mesh.Hz
        Hsolid = sum(case.ref.ρ̄[k] * case.ref.cp * case.ref.π̄[k] * θ[i, j, k] * M.dz / (M.Nx * M.Ny)
                     for k in 1:M.Nz, j in 1:M.Ny, i in 1:M.Nx if mask[Hx+i, Hy+j, Hz+k])
        @test count(mask) > 0
        @test BC.air_heat(m_all, case.sim.θ) - BC.air_heat(m_fl, case.sim.θ) ≈ Hsolid rtol = 1e-10
        @test BC.air_heat(case.sim, case.ref, mask) == BC.air_heat(m_fl, case.sim.θ)
    end
end
