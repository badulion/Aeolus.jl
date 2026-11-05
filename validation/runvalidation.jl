# The validation runs of Aeolus.jl: simulated cases that confirm the physics
# works and stays stable: GABLS, channels and log laws, decaying turbulence,
# growth rates, long nest runs, observed campaigns and published benchmarks.
#
# They are not tests. A test checks a unit of code (analytic values,
# identities, conservation over a few steps, interfaces, short code-path
# runs) and lives in test/; the test suite is the PR gate. The validation
# suite is not a gate: it is run on demand, weekly in CI, and by anyone who
# changes the physics a case validates, who quotes the result in the PR.
#
#   julia --project=. validation/run.jl                       # all, in parallel
#   julia --project=. validation/run.jl --only era6_gabls1.jl
#   julia --project=. validation/runvalidation.jl             # all, serially
#
# The include list below is the single list of validation files; a new case
# needs only its include line (and a tag in validation_tags.toml if it is
# heavy or timing-sensitive). Shared fixtures stay in test/ (support/,
# fixtures/, data/); validation-only ones are in validation/support,
# validation/fixtures and validation/data. See docs/src/validation.md.

include(joinpath(@__DIR__, "..", "test", "support", "environment.jl"))

@testset "Validation" begin
    include("urban_checks.jl")       # EX-a (#438): end-to-end urban runs on the Berlin configs (smoke matrix, flagship, parent validations)
end

@testset "Era II" begin
    include("era2_anelastic.jl")              # Phase 27  warm bubble in Boussinesq / anelastic / pseudo-incompressible
    include("era2_anelastic_cbl.jl")          # #259/#320 Boussinesq vs anelastic: 5 h CBL to ~2 km
    include("era2_sgs_numerics.jl")           # #132  dynamic Smagorinsky C_s ≈ Lilly in developed turbulence
    include("era2_nest_boundary.jl")          # Phase 30  one-way nest in a CBL: overlap means within the spread (#261)
    include("era2_nest_twoway_longrun.jl")    # Phase 30 two-way long runs: CBL, neutral, vortex (#263)
    include("era2_nest_hierarchy_longrun.jl") # #328 two two-way nest levels, 50 turnovers
end

@testset "Era IV" begin
    include("era4_activation.jl")       # Phase 42  ARG vs explicit parcel
    include("era4_superdroplet.jl")     # Phase 41  Golovin analytic solution, convergence with N_sd
    include("era4_icenucleation.jl")    # Phase 43  ISDAC Arctic stratus column, 2-8 h at five INP scalings
    include("era4_broadening.jl")       # Phase 45  bulk vs superdroplet broadening, σ_r² ∝ t^½
    include("era4_radiation_fog.jl")    # Phase 44  ParisFog 2011: dt convergence, 6 h budgets, 14 h case
end

@testset "Era V" begin
    include("era5_land_surface_fluxnet.jl")   # Phase 47  land column vs FLUXNET/PLUMBER2 towers (#21)
    include("era5_canopy_profiles.jl")        # Phase 48  canopy wind/temperature vs wind tunnel, CHATS, Hyytiälä (#25)
    include("era5_building_energy.jl")        # Phase 50  resolved zone vs ISO 13790 5R1C, 150-day periodic regime (#31)
    include("era5_hvac.jl")                   # Phase 50  resolved zone vs ISO 13790 5R1C ideal cooling need (#32)
    include("era5_bestest.jl")                # Phase 50  ASHRAE 140-2020 BESTEST annual loads (#33)
    include("era5_anthropogenic_heat.jl")     # Phase 49  Toulouse winter building energy vs CAPITOUL inventory (#29)
    include("era5_snowpack.jl")               # Phase 51  bulk density vs Sturm et al. (2010) alpine climatology (#34)
    include("era5_snow_metamorphism.jl")      # Phase 51  30-day albedo decay vs reference-scheme curves (#35)
    include("era5_ice_thermodynamics.jl")     # Phase 51  MU71 vs BL99 reference code and Tables 1-2 (#37)
    include("era5_ocean_mixed_layer.jl")      # Phase 52  warm layers vs ship near-surface thermometers (#38)
    include("era5_sea_spray_coare.jl")        # Phase 52  diurnal warm layer vs TOGA COARE observations (#40)
    include("era5_urban_validation.jl")       # Phase 49  BUBBLE / CAPITOUL validation of the urban canopy (#30)
    include("era5_heterogeneous_surface.jl")  # Phase 53  resolved patchy-surface LES vs tiles (#43)
end

@testset "Era VI" begin
    include("era6_lagrangian_sgs.jl")    # P54-b: Lagrangian dynamic Smagorinsky in isotropic turbulence: C_s vs plane, positivity (#173)
    include("era6_neutral_abl.jl")       # P54-d: half channel (u*², τ(z)), two-wall channel at Re_τ = 1000 (#174)
    include("era6_stratified_sgs.jl")    # P55-e: StabilityCorrected on linear and breaking internal waves (#187)
    include("era6_amd_vreman.jl")        # P55-d: AMD and Vreman dissipation vs Smagorinsky 0.17 in developed turbulence (#176)
    include("era6_apriori_sgs.jl")       # Phase 56  96³ filtered-DNS fixture: a-priori Smagorinsky, Bardina, dynamic C_s (#178)
    include("era6_tke.jl")               # P56-b: TKE positivity over 500 steps, reduced CBL (#182)
    include("era6_wall_models_channel.jl") # P57-b: channel Re_τ = 1000 and rough half channel, matching height + filters (#183)
    include("era6_wall_models_tble.jl")  # P57-c: TBLE channel at Re_τ = 1000 vs Lee & Moser (#189)
    include("era6_wall_models_immersed.jl") # P57-d: rotated channel, IBM walls vs z-walls, Lee & Moser U_b⁺ (#179)
    include("era6_wall_models_dynamic.jl") # P57-e: dynamic roughness, grid invariance 32² vs 16² (#190)
    include("era6_sigma_sgs.jl")         # P59-b: σ-model KE decay vs Smagorinsky 0.17 (#184)
    include("era6_relaminarisation.jl")  # P59-e: decaying turbulence in stable stratification vs neutral (#195)
    include("era6_deep_top.jl")          # N-parent-b: gravity-wave absorption by the θ damping layer (Klemp & Lilly 1978) (#254)
    include("era6_lasd.jl")              # P54-e: scale-dependent Lagrangian dynamic model, log law (#186)
    include("era6_cbc_decay.jl")         # Phase 54  Comte-Bellot–Corrsin decay benchmark (#168)
    include("era6_gabls2.jl")            # N-parent-c: GABLS2 (reduced) with large-scale subsidence (#248)
    include("era6_transition_kh.jl")     # Phase 59  Kelvin–Helmholtz / Miles–Howard growth rates (#180)
    include("era6_sgs_validation.jl")    # P56-e: TKE and RSM against filtered DNS (#192)
    include("era6_flat_plate.jl")        # P59-d: laminar flat plate, bypass transition (#185)
    include("era6_gabls1.jl")            # P55-f, ONE-g: GABLS1 on the model, reduced (#191, #456)
    include("era6_gabls3.jl")            # P58-d: GABLS3 on the SBL driver: budgets, MOST, LLJ (#194)
    include("era6_outflow_conditions.jl") # N-open-c: outgoing vortex reflection, 2000-step stability (#260)
    include("era6_inflow_fetch.jl")      # N-inflow-d: development length behind a synthetic inflow (#266)
    include("era6_inflow_nest_longrun.jl") # N-inflow-e: inflow generator on a nest child, 2 h one-/two-way (#268)
end
