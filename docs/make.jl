using Documenter
using Aeolus

DocMeta.setdocmeta!(Aeolus, :DocTestSetup, :(using Aeolus); recursive=true)

# The example drivers (DRIVER_FILES: berlin, sbl, dns, ...) are loaded to
# document their options and `using FFTW` etc. at top level: give them the
# package's own environment, whose dependencies the docs environment does
# not list directly.
push!(LOAD_PATH, normpath(joinpath(@__DIR__, "..")))

# The option reference of every driver and the input-format specifications,
# generated from the schemas into docs/src/reference/ (CFG-i, #364; not committed).
include(joinpath(@__DIR__, "generate_reference.jl"))
generate_reference()
const OPTION_PAGES = ["`$d`" => "reference/options/$d.md" for d in reference_drivers()]

makedocs(
    modules  = [Aeolus,
                Aeolus.Grids, Aeolus.Fields, Aeolus.BoundaryConditions,
                Aeolus.Operators, Aeolus.TimeSteppers,
                Aeolus.PressureSolvers, Aeolus.Simulations,
                Aeolus.SimulationDriver, Aeolus.Config, Aeolus.InputDrivers],
    sitename = "Aeolus.jl",
    authors  = "Andrzej Dulny",
    format   = Documenter.HTML(
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical  = "https://badulion.github.io/Aeolus.jl",
        edit_link  = "master",
        repolink   = "https://github.com/badulion/Aeolus.jl",
        assets     = [asset("assets/favicon.svg"; class = :ico, islocal = true)],
        sidebar_sitename = true,
        # long single pages: the KA port log (223 KiB) and the generated Berlin
        # option reference (137 KiB, one table per coupling); Documenter's
        # 200 KiB limit otherwise fails the build
        size_threshold_ignore = ["design/ka_port.md", "reference/options/berlin.md"],
    ),
    pages = [
        "Home"           => "index.md",
        "Installation"   => "installation.md",
        "Quick start"    => "quickstart.md",
        "Running the tests" => "testing.md",
        "Theory" => [
            "Overview"                    => "theory/overview.md",
            "Boussinesq equations"        => "theory/boussinesq.md",
            "LES filtering & closure"     => "theory/les_filtering.md",
            "Staggered discretization"    => "theory/staggered_grid.md",
            "Fractional-step projection"  => "theory/fractional_step.md",
            "Smagorinsky-Lilly SGS"       => "theory/sgs.md",
            "Discrete Poisson & FFT"      => "theory/poisson.md",
            "Grid nesting: conservative staggered prolongation" => "theory/nesting.md",
            "Stretched vertical grid"     => "theory/stretched_grid.md",
            "Nested time coupling: subcycling and temporal interpolation" => "theory/nesting_time.md",
            "Nested boundary conditions: one-way parent → child data" => "theory/nesting_boundaries.md",
            "Two-way nesting: anterpolation and refluxing" => "theory/nesting_twoway.md",
            "Radiative transfer"          => "theory/radiation.md",
            "Era IV overview: clouds and microphysics" => "theory/era4_overview.md",
            "P3 microphysics"             => "theory/p3.md",
            "Ice nucleation"              => "theory/ice_nucleation.md",
            "Sub-grid cloud PDF"          => "theory/subgrid_cloud.md",
            "SGS activation and growth"   => "theory/sgs_activation_growth.md",
            "Spectral broadening"         => "theory/spectral_broadening.md",
            "Visibility (Koschmieder)"    => "theory/visibility.md",
            "Fog and dew"                 => "theory/fog.md",
            "Radiation fog"               => "theory/radiation_fog.md",
            "Urban chemistry and deposition" => "theory/urban_chemistry.md",
            "Era V overview: land, urban, surface" => "theory/era5_overview.md",
            "Surface exchange (MOST)"     => "theory/surface_exchange.md",
            "Soil water (Richards)"       => "theory/soil_water.md",
            "Soil heat and freeze–thaw"   => "theory/soil_heat.md",
            "Water table and aquifer"     => "theory/water_table.md",
            "Leaf photosynthesis"         => "theory/photosynthesis.md",
            "Stomatal conductance"        => "theory/stomatal_conductance.md",
            "Vegetated land column"       => "theory/land_surface.md",
            "Resolved plant canopy"       => "theory/plant_canopy.md",
            "Urban facet conduction"      => "theory/urban_conduction.md",
            "Glazing and window facets"   => "theory/urban_glazing.md",
            "Facet–radiosity coupling"    => "theory/urban_radiosity_coupling.md",
            "Building energy (zone)"      => "theory/building_energy.md",
            "HVAC and exhaust heat"       => "theory/hvac_exhaust.md",
            "Occupancy and BESTEST"       => "theory/occupancy_validation.md",
            "Anthropogenic heat (BEM)"    => "theory/anthropogenic_heat.md",
            "BUBBLE / CAPITOUL validation" => "theory/urban_validation.md",
            "Layered snowpack"            => "theory/snowpack.md",
            "Snow metamorphism and albedo" => "theory/snow_metamorphism.md",
            "Lake and sea ice"            => "theory/ice_thermodynamics.md",
            "Water surfaces (mixed layer)" => "theory/ocean_mixed_layer.md",
            "Air–sea fluxes and waves"    => "theory/air_sea_flux.md",
            "Sea spray and full COARE"     => "theory/sea_spray_coare.md",
            "Sub-grid surface tiles"      => "theory/surface_tiles.md",
            "Blending height"             => "theory/blending_height.md",
            "Heterogeneous-surface validation" => "theory/heterogeneous_surface.md",
            # Era VI — high-fidelity turbulence closure
            "Era VI overview: turbulence closure" => "theory/era6_overview.md",
            "Transition and wall-bounded flow" => "theory/transition.md",
            "Stability functions and Ri_g" => "theory/turbulence_stability.md",
            "Wall models"                 => "theory/wall_models.md",
            "Large-scale forcing"         => "theory/large_scale_forcing.md",
            "Prognostic SGS TKE and Reynolds stress" => "theory/sgs_tke.md",
            "Stable BLs: GABLS1-4, CASES-99, channel" => "theory/stable_boundary_layer.md",
        ],
        "Design" => [
            "Architecture"                => "design/architecture.md",
            "Multiple dispatch on Location" => "design/dispatch.md",
            "Pluggable pressure solvers"  => "design/pressure_solvers.md",
            "The Berlin case"             => [
                "Overview and architecture"      => "design/berlin_case.md",
                "Numerics and dynamical core"    => "design/berlin_numerics.md",
                "Surface exchange and radiation" => "design/berlin_surface_radiation.md",
                "Turbulence and SGS"             => "design/berlin_turbulence.md",
                "Ledgers and tests"              => "design/berlin_ledgers.md",
                "Nesting"                        => "design/berlin_nesting.md",
                "Air quality"                    => "design/berlin_air_quality.md",
            ],
            "The Berlin parent"           => "design/berlin_parent.md",
            "Run configuration"           => "design/run_configuration.md",
            "Checkpoint / restart and FFT planning" => "design/checkpoint_restart.md",
            "Kernel port (KernelAbstractions)" => "design/ka_port.md",
            "Roadmap"                     => "design/roadmap.md",
            "Long-range roadmap (Eras II–XI)" => "design/longterm_roadmap.md",
        ],
        "Reference" => [
            "Options"              => OPTION_PAGES,
            "Static input driver"  => "reference/input_static.md",
            "Dynamic input driver" => "reference/input_dynamic.md",
        ],
        "API reference" => [
            "Grids"                => "api/grids.md",
            "Fields"               => "api/fields.md",
            "BoundaryConditions"   => "api/boundary_conditions.md",
            "Operators"            => "api/operators.md",
            "TimeSteppers"         => "api/time_steppers.md",
            "PressureSolvers"      => "api/pressure_solvers.md",
            "Simulations"          => "api/simulations.md",
            "I/O and checkpoints"  => "api/io.md",
            "Run configuration"    => "api/config.md",
            "Input drivers"        => "api/input_drivers.md",
            "Logging"              => "api/logging.md",
            "Physics modules"      => "api/physics.md",
            "Era II — numerics" => [
                "Overview"                     => "api/era2.md",
                "Cut-cell IBM"                 => "api/era2/cutcell.md",
                "Stretched grids"              => "api/era2/stretched_grids.md",
                "Time integration"             => "api/era2/time_integration.md",
                "Advection schemes"            => "api/era2/advection.md",
                "Anelastic dynamics"           => "api/era2/anelastic.md",
                "Multigrid"                    => "api/era2/multigrid.md",
                "Nesting"                      => "api/era2/nesting.md",
                "Block-structured AMR"         => "api/era2/adaptive_mesh.md",
            ],
            "Era III — radiation" => [
                "Overview"                     => "api/era3.md",
                "RadiationCore"                => "api/era3/radiation_core.md",
                "Gas optics"                   => "api/era3/gas_optics.md",
                "Two-stream"                   => "api/era3/two_stream.md",
                "Monte-Carlo RT"               => "api/era3/monte_carlo_rt.md",
                "Radiosity"                    => "api/era3/radiosity.md",
                "Surface optics"               => "api/era3/surface_optics.md",
                "Aerosol optics"               => "api/era3/aerosol_optics.md",
                "Solar geometry"               => "api/era3/solar_geometry.md",
            ],
            "Era IV — microphysics" => [
                "Overview"                     => "api/era4.md",
                "MoistThermodynamics"          => "api/era4/moist_thermodynamics.md",
                "P3 microphysics"              => "api/era4/p3_microphysics.md",
                "P3 processes and secondary ice" => "api/era4/p3_processes.md",
                "P3 sedimentation and precipitation fraction" => "api/era4/p3_sedimentation.md",
                "Superdroplet"                 => "api/era4/superdroplet.md",
                "Aerosol-cloud interaction"    => "api/era4/aerosol_cloud.md",
                "Ice nucleation"               => "api/era4/ice_nucleation.md",
                "Visibility"                   => "api/era4/visibility.md",
                "Fog and dew"                  => "api/era4/radiation_fog.md",
                "Turbulent microphysics"       => "api/era4/turbulent_microphysics.md",
            ],
            "Era V — surface exchange" => [
                "Overview"                     => "api/era5.md",
                "SurfaceExchange"              => "api/era5/surface_exchange.md",
                "SoilColumn"                   => "api/era5/soil_column.md",
                "Photosynthesis"               => "api/era5/photosynthesis.md",
                "LandSurface"                  => "api/era5/land_surface.md",
                "PlantCanopy"                  => "api/era5/plant_canopy.md",
                "UrbanSurface"                 => "api/era5/urban_surface.md",
                "BuildingEnergy"               => "api/era5/building_energy.md",
                "Snowpack"                     => "api/era5/snowpack.md",
                "IceThermodynamics"            => "api/era5/ice_thermodynamics.md",
                "OceanMixedLayer"              => "api/era5/ocean_mixed_layer.md",
                "AirSeaFlux"                   => "api/era5/air_sea_flux.md",
                "SurfaceTiles"                 => "api/era5/surface_tiles.md",
            ],
            "Era VI — turbulence closure" => [
                "Overview"                     => "api/era6.md",
                "LargeScaleForcing"            => "api/era6/large_scale_forcing.md",
                "PrognosticSGS"                => "api/era6/prognostic_sgs.md",
                "TurbulenceStability"          => "api/era6/turbulence_stability.md",
                "Transition"                   => "api/era6/transition.md",
                "WallModels"                   => "api/era6/wall_models.md",
            ],
        ],
        "Examples" => [
            "Scalar advection 2D"           => "examples/scalar_advection.md",
            "Poisson manufactured (periodic)" => "examples/poisson.md",
            "Poisson manufactured (Neumann)"  => "examples/poisson_neumann.md",
            "Taylor-Green vortex"           => "examples/tgv.md",
            "Buoyant convection"            => "examples/convection.md",
            "Battery of examples (Tiers 1–3)" => "examples/battery.md",
        ],
        "Benchmarks"    => "benchmarks.md",
    ],
    warnonly = [:missing_docs, :cross_references],
    remotes  = nothing,   # source links disabled; docs build standalone
)

# Deploy to the gh-pages branch (#354). Only from GitHub Actions, so a local
# `julia --project=docs docs/make.jl` builds into docs/build and stops here.
# Pushes to master update `dev`, version tags update `stable` and `vX.Y.Z`;
# pull requests build without deploying (no previews). Authentication is the
# workflow's GITHUB_TOKEN (DOCUMENTER_KEY, if set, is used instead).
if get(ENV, "GITHUB_ACTIONS", "false") == "true"
    deploydocs(
        repo      = "github.com/badulion/Aeolus.jl.git",
        devbranch = "master",
        push_preview = false,
    )
end
