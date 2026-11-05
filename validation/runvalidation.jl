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
    include("era5_urban_validation.jl")       # Phase 49  BUBBLE / CAPITOUL validation of the urban canopy (#30)
    include("era5_heterogeneous_surface.jl")  # Phase 53  resolved patchy-surface LES vs tiles (#43)
end

@testset "Era VI" begin
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
