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
#   * add_heat! splits a context, nested contexts, the night window;
#   * the fixed-gradient θ top (#254): the dynamics books the top-face
#     flux as (:dynamics, :top_flux), leaving round-off to the transport.
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
            Πs = (ref.p_surface / p₀)^(R / cp)          # the ground pressure the reference integrated from
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

    @testset "closed box: the transport conserves Σθ; H changes only by redistribution" begin
        # a closed box: the θ top is the no-flux lid (the case default since
        # #254 is the fixed-gradient top, whose flux is tested below)
        case = flat_case(; u0 = 1.0, θ_top = :noflux)
        @test Set(BC.coupling_name.(case.couplings)) ⊆ Set(NO_THETA)
        # structure: θ noise in the lowest 8 levels and a random w
        rng = Xoshiro(253)
        θ = interior_view(case.sim.θ)
        θ[:, :, 1:8] .+= 0.5 .* (rand(rng, size(θ, 1), size(θ, 2), 8) .- 0.5)
        interior_view(case.sim.w)[:, :, 2:end] .+= 0.2 .* (rand(rng, size(θ, 1), size(θ, 2), size(θ, 3) - 1) .- 0.5)
        H0 = BC.air_heat(case.sim, case.ref)
        prev = Dict{Tuple{Symbol,Symbol},Float64}()
        worst_tr, worst_tot, sum_red = 0.0, 0.0, 0.0
        other = 0.0
        for n in 1:12
            Hb = BC.air_heat(case.sim, case.ref)
            BC.step_case!(case, 2.0)
            b = BC.heat_budget(case.ledger)
            d = step_terms(prev, b.terms); prev = b.terms
            ΔH = BC.air_heat(case.sim, case.ref) - Hb
            tr = get(d, (:dynamics, :transport), 0.0)
            red = get(d, (:dynamics, :redistribution), 0.0)
            worst_tr = max(worst_tr, abs(tr) / Hb)
            worst_tot = max(worst_tot, abs(ΔH) / Hb)
            sum_red += red
            # nothing but the dynamics changed θ: every other entry books 0
            other += sum(abs(v) for (k, v) in d if k[1] !== :dynamics; init = 0.0)
            @test abs(ΔH - (tr + red)) <= 1e-15 * Hb    # round-off of two summation orders
        end
        b = BC.heat_budget(case.ledger)
        @info @sprintf("  closed box, 12 steps: max |transport|/H %.1e per step, max |ΔH|/H %.1e per step (redistribution %.3e J/m² in 24 s), closure %.1e W/m²",
                       worst_tr, worst_tot, sum_red, b.closure / b.elapsed)
        # The issue's expectation, |ΔH_air|/H_air ≤ 1e-13 per step, holds
        # for what the flux-form transport conserves (Σ θ ΔV on the
        # Boussinesq core). H itself weights θ with ρ̄Π̄, which falls with
        # height, so vertical motion changes it: that part is
        # `redistribution`, physical in a Boussinesq model, not round-off.
        @test worst_tr <= 1e-13
        @test other == 0
        @test worst_tot > 1e-9             # the redistribution is real (measured 3.7e-7 per step)
        @test abs(b.closure / b.elapsed) <= 1e-10
        @test !haskey(b.terms, (:core, :unattributed))
        @test !haskey(b.terms, (:dynamics, :solid_exchange))     # no solid cells
    end

    @testset "analytic surface heating H₀ = 100 W m⁻² on a flat, motionless variant" begin
        H0 = 100.0
        # no motion (u0 = 0, no Coriolis, no viscosity, Smagorinsky at zero
        # strain): θ changes only by the plate
        keep = (:core_boussinesq, :timestep_fixed, :pressure_fft, :sgs_smagorinsky, :momentum_centered)
        case = flat_case(; keep = keep, extra = BC.AbstractCoupling[_HeatPlate(H0, 0.0)], u0 = 0.0,
                         f_coriolis = 0.0, ν = 0.0, qv0 = 0.0,
                         physics = Dict{Symbol,Any}(:timestepper => :fixed, :sgs => :smagorinsky))
        Hs = BC.air_heat(case.sim, case.ref)
        BC.run_case!(case; nsteps = 12)
        b = BC.heat_budget(case.ledger)
        E = H0 * b.elapsed
        plate = b.terms[(:surface_flux, :heat_plate)]
        @info @sprintf("  H₀ = 100 W/m² over %.0f s: attributed %.12e, ΔH_air %.12e, H₀Δt %.12e J/m² (rel %.1e, %.1e)",
                       b.elapsed, plate, b.storage, E, abs(plate / E - 1), abs(b.storage / E - 1))
        @test b.elapsed == 24.0
        @test plate ≈ E rtol = 1e-10
        @test b.storage ≈ E rtol = 1e-10
        @test BC.air_heat(case.sim, case.ref) - Hs ≈ E rtol = 1e-10
        # all of it in the lowest level: the canopy and 0–100 m layers see it too
        @test BC.heat_budget(case.ledger; layer = (0, 100)).storage ≈ E rtol = 1e-10
        @test abs(BC.heat_budget(case.ledger; layer = (case.mesh.dz, Inf)).storage) <= 1e-9 * E
        @test abs(b.closure) <= 1e-10 * b.elapsed
    end

    @testset "add_heat! splits a context; nested body-force contexts leave the dynamics" begin
        keep = (:core_boussinesq, :timestep_fixed, :pressure_fft, :sgs_smagorinsky, :momentum_centered)
        case = flat_case(; keep = keep,
                         extra = BC.AbstractCoupling[_HeatPlate(100.0, 0.25), _BodyHeat(40.0)],
                         u0 = 0.0, f_coriolis = 0.0, ν = 0.0, qv0 = 0.0,
                         physics = Dict{Symbol,Any}(:timestepper => :fixed, :sgs => :smagorinsky))
        BC.run_case!(case; nsteps = 6)
        b = BC.heat_budget(case.ledger)
        el = b.elapsed
        @test b.terms[(:surface_flux, :plate_named)] ≈ 0.25 * 100 * el rtol = 1e-12
        @test b.terms[(:surface_flux, :heat_plate)] ≈ 0.75 * 100 * el rtol = 1e-10
        @test b.terms[(:body_force, :body_heat)] ≈ 40 * el rtol = 1e-10
        @test abs(b.terms[(:dynamics, :transport)]) <= 1e-9 * 40 * el
        @test b.storage ≈ 140 * el rtol = 1e-10
        @test abs(b.closure) <= 1e-10 * el
        # the body heat went into level 2: the layer of level 2 sees it
        dz = case.mesh.dz
        lv2 = BC.heat_budget(case.ledger; layer = (dz, 2dz))
        @test lv2.terms[(:body_force, :body_heat)] ≈ 40 * el rtol = 1e-10
        @test get(lv2.terms, (:surface_flux, :heat_plate), 0.0) == 0
        # outside a step there is no context to book into
        @test_throws ErrorException BC.add_heat!(case.ledger, :late, 1.0)
        # an unlocated term counts in the column only
        hl = case.ledger.heat
        BC.heat_begin_step!(hl, case.sim.θ, 12.0)
        BC.heat_open!(hl, :anthropogenic, :test, case.sim.θ)
        BC.add_heat!(case.ledger, :somewhere, 5.0)
        BC.heat_close!(hl, case.sim.θ)
        BC.heat_end_step!(hl, case.sim.θ, 0.0)
        b2 = BC.heat_budget(case.ledger)
        @test b2.terms[(:anthropogenic, :somewhere)] == 5.0
        @test b2.terms[(:anthropogenic, :test)] == -5.0          # θ did not change
        @test abs(b2.closure) <= 1e-10 * el
        @test BC.heat_budget(case.ledger; layer = (0, 50)).unlocated == 0.0   # 5 − 5
    end

    @testset "night window: hour bins of the step start" begin
        @test BC._hour_bins((20, 5)) == [21, 22, 23, 24, 1, 2, 3, 4, 5]
        @test BC._hour_bins((6, 9)) == [7, 8, 9]
        @test length(BC._hour_bins(nothing)) == 24
        keep = (:core_boussinesq, :timestep_fixed, :pressure_fft, :sgs_smagorinsky, :momentum_centered)
        # 04:59:44 + 12 × 2 s: 8 steps start before 05:00, 4 after
        case = flat_case(; keep = keep, extra = BC.AbstractCoupling[_HeatPlate(100.0, 0.0)],
                         u0 = 0.0, f_coriolis = 0.0, ν = 0.0, qv0 = 0.0, start_h = 5 - 16 / 3600,
                         physics = Dict{Symbol,Any}(:timestepper => :fixed, :sgs => :smagorinsky))
        BC.run_case!(case; nsteps = 12)
        night = BC.heat_budget(case.ledger; window = (20, 5))
        day = BC.heat_budget(case.ledger; window = (5, 20))
        all_ = BC.heat_budget(case.ledger)
        @test night.elapsed == 16.0 && day.elapsed == 8.0
        @test night.storage ≈ 100 * 16 rtol = 1e-10
        @test night.storage + day.storage ≈ all_.storage rtol = 1e-14
        @test occursin("local 20-05 h", BC.heat_report(case.ledger; window = (20, 5)))
    end

    @testset "fixed-gradient θ top: the dynamics books the top flux (#254)" begin
        cfg0 = BC.CaseConfig(:smoke; verbose = false, output_path = nothing)
        # Γ is the initial lapse rate above the inversion (60 m < 100 m top)
        @test BC.θ_top_gradient(cfg0) ≈ cfg0.θ_lapse rtol = 1e-10       # a difference of θ ≈ 300 K over 6 m
        @test BC.θ_top_bc(cfg0) isa FluxBC
        @test BC.θ_top_bc(BC.CaseConfig(:smoke; θ_top = :noflux)) isa NoFlux
        @test BC.θ_top_bc(BC.CaseConfig(:smoke; θ_top = 0.01)).f == 0.01
        @test_throws ErrorException BC.θ_top_bc(BC.CaseConfig(:smoke; θ_top = :open))
        # the idealised summer profile: the top (200 m) is above the inversion (100 m)
        @test BC.θ_top_gradient(BC.CaseConfig(; sounding = nothing)) ≈ 3e-3 rtol = 1e-10
        # the default flagship starts from the Lindenberg sounding (#247): Γ is
        # the slope of the sounding's θ profile across the top face (4.14 K/km
        # measured), positive — a stable free atmosphere above the domain
        let c = BC.CaseConfig(), S = BC.case_sounding(c), θi = BC.initial_θ(c)
            H = c.extent[3]; zs = S.profiles.z
            i = searchsortedlast(zs, H)
            z1, z2 = zs[i], zs[i + 1]
            Γs = (θi(0.0, 0.0, z2) - θi(0.0, 0.0, z1)) / (z2 - z1)
            @test BC.θ_top_gradient(c) ≈ Γs rtol = 1e-8
            @test BC.θ_top_gradient(c) > 0
        end
        for phys in (Dict{Symbol,Any}(), Dict{Symbol,Any}(:timestepper => :fixed))
            case = BC.build_case(BC.CaseConfig(:smoke; verbose = false, output_path = nothing,
                                               physics = phys); io = devnull)
            @test case.sim.θ_bcs.top isa FluxBC
            @test case.top !== nothing
            BC.run_case!(case; nsteps = 12)
            b = BC.heat_budget(case.ledger)
            top = b.terms[(:dynamics, :top_flux)]
            tr = b.terms[(:dynamics, :transport)]
            # downgradient into the domain: at least the molecular part
            # ρ̄ c_p Π̄ ν Γ at the top level
            Nz = case.mesh.Nz
            mol = case.ref.ρ̄[Nz] * case.ref.cp * case.ref.π̄[Nz] * case.cfg.ν * cfg0.θ_lapse
            @info @sprintf("  fixed-gradient top (%s): top flux %.4f W/m² (molecular part %.4f), transport residual %.1e W/m², closure %.1e W/m²",
                           isempty(phys) ? "adaptive" : "fixed", top / b.elapsed, mol, tr / b.elapsed, b.closure / b.elapsed)
            @test top / b.elapsed >= mol * (1 - 1e-12)
            # the booked flux is the flux the core applied: what is left for the
            # transport is round-off (8e-10 W/m² measured, against 7.2 W/m²)
            @test abs(tr) <= 1e-8 * abs(top)
            @test abs(b.closure) <= 1e-10 * b.elapsed
            # the flux enters the top level only
            below = BC.heat_budget(case.ledger; layer = (0.0, case.mesh.Lz - case.mesh.dz))
            @test get(below.terms, (:dynamics, :top_flux), 0.0) == 0
            @test case.top.total ≈ top rtol = 1e-12
        end
    end

    @testset "canopy layer: the tallest roof, or cfg.canopy_top" begin
        for ct in (NaN, 30.0)
            case = BC.build_case(BC.CaseConfig(:smoke; verbose = false, output_path = nothing,
                                               canopy_top = ct); io = devnull)
            BC.run_case!(case; nsteps = 2)
            hl = case.ledger.heat
            roof = maximum(BC.building_height(case.geom, case.mesh))
            @test roof > 0
            @test hl.canopy_top == (isnan(ct) ? roof : ct)
            a = BC.heat_budget(case.ledger; layer = :canopy)
            b = BC.heat_budget(case.ledger; layer = (0.0, hl.canopy_top))
            @test a.storage == b.storage && a.terms == b.terms
        end
    end
end
