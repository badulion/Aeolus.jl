using Documenter
using Aeolus

DocMeta.setdocmeta!(Aeolus, :DocTestSetup, :(using Aeolus); recursive=true)

makedocs(
    modules  = [Aeolus,
                Aeolus.Grids, Aeolus.Fields, Aeolus.BoundaryConditions,
                Aeolus.Operators, Aeolus.TimeSteppers,
                Aeolus.PressureSolvers, Aeolus.Simulations],
    sitename = "Aeolus.jl",
    authors  = "Andrzej Dulny",
    format   = Documenter.HTML(
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical  = "https://andrzej-dulny.github.io/Aeolus.jl",
        edit_link  = "main",
        assets     = String[],
    ),
    pages = [
        "Home"           => "index.md",
        "Installation"   => "installation.md",
        "Quick start"    => "quickstart.md",
        "Theory" => [
            "Overview"                    => "theory/overview.md",
            "Boussinesq equations"        => "theory/boussinesq.md",
            "LES filtering & closure"     => "theory/les_filtering.md",
            "Staggered discretization"    => "theory/staggered_grid.md",
            "Fractional-step projection"  => "theory/fractional_step.md",
            "Smagorinsky-Lilly SGS"       => "theory/sgs.md",
            "Discrete Poisson & FFT"      => "theory/poisson.md",
        ],
        "Design" => [
            "Architecture"                => "design/architecture.md",
            "Multiple dispatch on Location" => "design/dispatch.md",
            "Pluggable pressure solvers"  => "design/pressure_solvers.md",
            "Roadmap"                     => "design/roadmap.md",
            "Long-range roadmap (Eras II–XI)" => "design/longterm_roadmap.md",
        ],
        "API reference" => [
            "Grids"                => "api/grids.md",
            "Fields"               => "api/fields.md",
            "BoundaryConditions"   => "api/boundary_conditions.md",
            "Operators"            => "api/operators.md",
            "TimeSteppers"         => "api/time_steppers.md",
            "PressureSolvers"      => "api/pressure_solvers.md",
            "Simulations"          => "api/simulations.md",
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
    remotes  = nothing,   # not currently a git repo
)
