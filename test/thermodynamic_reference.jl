# The thermodynamic reference of a domain and the absolute air temperature
# outputs (#536): `Components.thermodynamic_reference`, `reference_exner`
# and the core's `temperature`, `temperature_profile`, `temperature_mean`.
#
#   * Π̄ of the reference (`Urban.domain_reference`, the one construction)
#     against the analytic dry-adiabatic hydrostatic profile of a constant
#     θ, Π(z) = Π_s − g z/(c_p θ), on a stretched and a uniform grid, and
#     `reference_exner` at the ground and at 2 m;
#   * T = θ Π̄ of the 3-D field, its profile and its domain mean, to round-off;
#   * where the reference comes from: the anelastic base state, a Boussinesq
#     domain's driver, none for a bare Boussinesq core (left out of the
#     catalogue, the getter refused with a message);
#   * the configuration: every Boussinesq domain built from a config has
#     the reference, so `temperature` is a valid selection; the CF
#     standard_name reaches the NetCDF file.

using Test
using Aeolus
using Aeolus.Components
using Aeolus.Fields: interior_view, set!
using Aeolus.Anelastic: AnelasticSimulation, isentropic_reference
using Aeolus.Output: select_variables, catalogue_names
const TR = Aeolus.Components

const TR_C = Aeolus.MoistThermodynamics.DEFAULT_CONSTANTS
const TR_RD = TR_C.gas_constant / TR_C.molmass_dry
const TR_KAPPA = TR_RD / TR_C.cp_dry
tr_bcs() = (bcs = FieldBoundaryConditions(:periodic, :periodic, :wall),
            θ_bcs = FieldBoundaryConditions(:periodic, :periodic, :noflux))

"Π(z) of a hydrostatic atmosphere of constant θ from p_s: Π_s − g z/(c_p θ)."
tr_exner(z, θ, ps; g = 9.81) = (ps / TR_C.p_ref)^TR_KAPPA - g * z / (TR_C.cp_dry * θ)

@testset "reference Π̄ of a constant θ: the analytic dry-adiabatic hydrostatic profile" begin
    θ, ps = 300.0, 101325.0
    # a stretched grid: the continuous hydrostatic equation by RK4 (1000
    # substeps a level), whose solution for a constant θ is exactly linear
    # in Π: what is left is the RK4 truncation and round-off, ≤ 1e-12
    faces = [0.0, 10.0, 25.0, 45.0, 75.0, 120.0, 180.0, 260.0, 360.0]
    sg = StretchedGrid(; size = (4, 4, 8), extent = (100.0, 100.0, 360.0), halo = (2, 2, 2), stretching = (:custom, faces))
    r = Aeolus.Urban.domain_reference(sg, z -> θ, ps)
    zc = [(faces[k] + faces[k+1]) / 2 for k in 1:8]
    @test r.π̄ ≈ tr_exner.(zc, θ, ps) rtol = 1e-12
    @test TR.reference_exner(r, zc, 0.0; g = 9.81) ≈ tr_exner(0.0, θ, ps) rtol = 1e-12
    @test TR.reference_exner(r, zc, 2.0; g = 9.81) ≈ tr_exner(2.0, θ, ps) rtol = 1e-12
    @test TR.reference_exner(r, zc, 100.0; g = 9.81) ≈ tr_exner(100.0, θ, ps) rtol = 1e-12   # between centres: Π linear
    # a uniform grid: the discrete hydrostatic balance of the anelastic base
    # state (`reference_from_profile`, a log-mean face density), second-order
    # accurate: its departure from the continuous profile is O(Δz²); measured
    # 2.1e-7, 8.4e-7, 1.4e-5 relative at Δz = 25, 50, 200 m: at 50 m the
    # tolerance 3e-6
    m = RegularGrid(; size = (4, 4, 10), extent = (400.0, 400.0, 500.0), halo = (2, 2, 2))
    ru = Aeolus.Urban.domain_reference(m, z -> θ, ps)
    zu = [(k - 0.5) * 50.0 for k in 1:10]
    @test ru.π̄ ≈ tr_exner.(zu, θ, ps) rtol = 3e-6
    @test TR.reference_exner(ru, zu, 0.0; g = 9.81) ≈ tr_exner(0.0, θ, ps) rtol = 3e-6
    # T̄ falls with the dry-adiabatic lapse rate g/c_p (the same O(Δz²): 5.6e-7)
    @test (ru.T̄[1] - ru.T̄[end]) / (zu[end] - zu[1]) ≈ 9.81 / TR_C.cp_dry rtol = 3e-6
end

@testset "temperature = θ Π̄: the 3-D field, its profile and domain mean ($core)" for core in (:boussinesq, :anelastic)
    m = RegularGrid(; size = (5, 4, 6), extent = (500.0, 400.0, 600.0), halo = (2, 2, 2))
    θf(x, y, z) = 295.0 + 0.004z + sin(x / 70) + cos(y / 50)
    sim, d = if core === :boussinesq
        s = BoussinesqSimulation(m; θ₀ = 300.0, tr_bcs()...)
        r = Aeolus.Urban.domain_reference(m, z -> 295.0 + 0.004z, 98000.0)
        s, Domain(s; drivers = (; reference = r), validate = false)
    else
        s = AnelasticSimulation(m; ref = isentropic_reference(m; θ_const = 300.0), tr_bcs()...)
        s, Domain(s; validate = false)
    end
    set!(sim.θ, θf)
    r = thermodynamic_reference(d)
    @test r === (core === :boussinesq ? d.drivers.reference : sim.ref)
    cat = domain_catalogue(d)
    T, Tp, Tm = (only(select_variables(cat, [n])) for n in ("temperature", "temperature_profile", "temperature_mean"))
    @test T.dims === :xyz && Tp.dims === :z && Tm.dims === :scalar
    @test all(v -> v.units == "K" && v.standard_name == "air_temperature", (T, Tp, Tm))
    θ = interior_view(sim.θ)
    # one multiplication per cell: equal to round-off (1 ulp of ~300 K)
    @test T.getter(d) ≈ [θ[i, j, k] * r.π̄[k] for i in 1:5, j in 1:4, k in 1:6] rtol = 4eps()
    @test Tp.getter(d) ≈ [sum(θ[:, :, k]) / 20 * r.π̄[k] for k in 1:6] rtol = 1e-14
    @test Tm.getter(d) ≈ sum(θ[i, j, k] * r.π̄[k] for i in 1:5, j in 1:4, k in 1:6) / 120 rtol = 1e-14
    # the air is cooler than θ above the 1000 hPa level and T falls with height
    Tprof = Tp.getter(d)
    core === :boussinesq && @test all(T.getter(d) .< θ)
    @test all(diff(Tprof) .< 0)
end

@testset "a bare Boussinesq core has no reference: no temperature in its catalogue" begin
    m = RegularGrid(; size = (4, 4, 4), extent = (400.0, 400.0, 400.0), halo = (2, 2, 2))
    d = Domain(BoussinesqSimulation(m; θ₀ = 300.0, tr_bcs()...); validate = false)
    @test thermodynamic_reference(d) === nothing
    @test !("temperature" in catalogue_names(domain_catalogue(d)))
    err = try reference_exner(d, 2.0); nothing catch e; e end
    @test err isa ArgumentError && occursin("no thermodynamic reference", sprint(showerror, err))
end

@testset "config: a Boussinesq domain gets its reference; temperature written with its standard_name" begin
    cfg = load_run_config(IOBuffer(""); overrides = ["grid.size=[8,8,8]", "fftw.planner=\"estimate\"",
                                                     "time.duration=\"none\"", "time.nsteps=2",
                                                     "dynamics.p_surface=95000.0",
                                                     "output.variables=[\"temperature\",\"temperature_profile\",\"temperature_mean\",\"theta_profile\"]"])
    r = run_config(cfg; root = mktempdir(), console = devnull, repo = mktempdir())
    Aeolus.NCIO.NCDataset(joinpath(r.folder.path, "output", "output.nc")) do ds
        @test ds["temperature"].attrib["standard_name"] == "air_temperature"
        @test ds["temperature"].attrib["units"] == "K"
        Tp = Float64.(ds["temperature_profile"][:, end])
        z = Float64.(ds["z"][:])
        θp = Float64.(ds["theta_profile"][:, end])
        T = Float64.(ds["temperature"][:, :, :, end])
        # T = θ Π̄(z), Π̄ that of the initial θ from 950 hPa: for the default
        # constant initial θ the analytic profile, to the O(Δz²) of the
        # discrete balance (1.4e-5 at the default Δz = 200 m, measured
        # above) and the Float32 storage (6e-8): 3e-5
        @test Tp ≈ θp .* tr_exner.(z, θp[1], 95000.0) rtol = 3e-5
        @test all(isfinite, T)
    end
end
