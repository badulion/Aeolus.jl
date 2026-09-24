#!/usr/bin/env python3
"""Generate .github/roadmap-issues.toml from the long-range roadmap.

Tiered by horizon: the near term is decomposed into concrete issues by
hand (below), the mid term gets one issue per phase parsed from the
roadmap, and the far term gets one tracking epic per era. Decomposing
work eight years out is fiction, so we do not.

Run:  python3 .github/scripts/generate_roadmap_issues.py
"""
import re, pathlib, datetime

ROOT = pathlib.Path(__file__).resolve().parents[2]
ROADMAP = ROOT / "docs/src/design/longterm_roadmap.md"
OUT = ROOT / ".github/roadmap-issues.toml"

ERAS = [
    ("I",    "Era I — LES core and physics scaffolds",            1, 20,  "closed",
     "The dynamical core, pluggable pressure solvers, physics scaffolds and the first urban flagship."),
    ("II",   "Era II — Numerical foundations for fidelity",      21, 31,  "closed",
     "Cut-cell IBM, ghost-cell walls, adaptive dt, IMEX, WENO-Z/TENO, anelastic dynamics, multigrid, nesting, AMR. 2,094 tests."),
    ("III",  "Era III — High-fidelity radiation",                32, 38,  "closed",
     "Correlated-k gas optics, two-stream, 3-D Monte-Carlo, urban radiosity, spectral BRDF, Mie aerosol, solar geometry. 4,881 tests."),
    ("IV",   "Era IV — High-fidelity clouds and microphysics",   39, 45,  "open",
     "Exact moist thermodynamics, bulk and Lagrangian microphysics, aerosol-cloud interaction, ice nucleation, fog. Phases 39, 41, 42 shipped."),
    ("V",    "Era V — High-fidelity land, urban and surface exchange", 46, 53, "open",
     "Richards-equation soil, Farquhar photosynthesis, resolved canopy, production urban canopy, building energy, snow, water surfaces, heterogeneity."),
    ("VI",   "Era VI — High-fidelity turbulence closure",        54, 59,  "open",
     "Scale-dependent dynamic SGS, anisotropic closures, prognostic SGS stress, high-Re wall models, stable boundary layer, transition."),
    ("VII",  "Era VII — High-fidelity chemistry and aerosol",    60, 66,  "open",
     "KPP mechanism, online photolysis, aqueous and heterogeneous chemistry, modal/sectional aerosol, SOA, emissions, deposition."),
    ("VIII", "Era VIII — Real-world data, forcing and validation", 67, 73, "open",
     "ERA5 driver, mesoscale nesting, production geospatial ingestion, observation operators, benchmark library, provenance."),
    ("IX",   "Era IX — Data assimilation and inverse problems",  74, 79,  "open",
     "Adjoint via AD, 4D-Var, ensemble Kalman filtering, hybrid assimilation, parameter and source inversion, UQ."),
    ("X",    "Era X — Performance, scale and portability",       80, 87,  "open",
     "Production MPI, GPU port, multi-GPU, mixed precision, performance engineering, async I/O, spectral element, DG."),
    ("XI",   "Era XI — Machine learning, applications, ecosystem", 88, 100, "open",
     "Neural closures and emulators, differentiable physics, foundation models, wildfire, wind energy, urban services, air quality, ecosystem."),
]
# Technical debt is not a roadmap phase, so it gets its own milestone rather
# than being filed under an era — which would otherwise leave a CLOSED era
# milestone carrying open issues.
DEBT_MILESTONE = ("Technical debt",
    "Caveats of shipped code: known biases, uneven coverage, disabled "
    "features and interfaces worth generalising. Not roadmap phases.")
ERA_OF_PHASE = {}
for roman, title, lo, hi, state, desc in ERAS:
    for p in range(lo, hi + 1):
        ERA_OF_PHASE[p] = (roman, title)

# Phases already delivered — no issues are created for these.
DONE = set(range(1, 39)) | {39, 41, 42}

# ---------------------------------------------------------------- near term
# Hand-authored decomposition. (phase, [(slug, title, scope, labels)])
NEAR = {
40: [("gamma", "Gamma-function machinery and size-distribution moments",
      "Implement `lgamma` (Lanczos) and the incomplete gamma function (series + continued fraction), then the gamma-distribution moments P3 needs. `SpecialFunctions` does not load on every target, so these are written in-package. Validate against known values: Gamma(1)=1, Gamma(1/2)=sqrt(pi), Gamma(5)=24, and the P(a,x) limits. Everything downstream depends on these being right.", ["microphysics"]),
     ("massdim", "P3 mass-dimension relation and predicted rime density",
      "The heart of the scheme: four regimes as a function of D (small spherical ice, dense nonspherical, partially rimed, graupel) with thresholds D_th, D_gr, D_cr SOLVED FOR from rime fraction and predicted density rather than tabulated. Carry q_rim and B_rim prognostically so density emerges. Sweep rime fraction 0 to 1 and show mass, density and fall speed vary continuously - no category jump. That continuity is the whole point of P3.", ["microphysics"]),
     ("rates", "Process rates: condensation, autoconversion, riming, melting",
      "Deposition/sublimation with ventilation, autoconversion and accretion (state which formulation), rain self-collection and breakup, riming with the Cober & List rime-density parameterisation, melting and shedding. Use `MoistThermodynamics.saturation_adjustment` - do not re-derive thermodynamics.", ["microphysics"]),
     ("sedim", "Sedimentation, size sorting and radar reflectivity",
      "Per-moment terminal-velocity weighting (mass-weighted and number-weighted differ, and that difference IS size sorting), positivity-preserving with sub-stepping above Courant 1. Radar reflectivity Z in dBZ - the observable P3 is validated against. Check a ~5 mm/h rain rate gives 35-45 dBZ.", ["microphysics", "validation"])],
43: [("inp", "Prognostic ice-nucleating-particle species",
      "Dust and biological INP as prognostic tracers with their own size distributions, sources and sinks. Reuse the Phase 37 aerosol size-distribution types rather than defining new ones.", ["microphysics"]),
     ("freezing", "Immersion and deposition freezing",
      "Immersion freezing keyed on INP surface area and supercooling, deposition nucleation above ice saturation. Connect to the Phase 39 `liquid_fraction` mixed-phase ramp so the two do not disagree about what is frozen.", ["microphysics"]),
     ("secondary", "Secondary ice production and Arctic stratus case",
      "Hallett-Mossop rime splintering and its temperature window. Validate mixed-phase persistence in an Arctic stratus case - the classic test that a scheme does not glaciate too fast.", ["microphysics", "validation"])],
44: [("fallspeed", "Terminal-velocity spectrum and sub-grid precipitation fraction",
      "A proper size-dependent terminal velocity rather than a single fall speed, and a sub-grid precipitation fraction so partially precipitating cells behave correctly.", ["microphysics"]),
     ("fog", "Radiation fog with droplet-radiative feedback",
      "Fog formation and dissipation with the droplet-radiative feedback that actually controls it - fog cools itself by radiating, which thickens it. Requires coupling the droplet field to the Era III radiation.", ["microphysics", "radiation"]),
     ("visibility", "Visibility diagnostic",
      "Visibility from droplet number and size (Koschmieder), for transport and air-quality applications.", ["microphysics"]),
     ("dewcoupling", "Re-enable fog/dew condensation coupling",
      "`apply_fog_dew!` ships DISABLED in examples/urban because the warming term amplifies before proper coupling to atmospheric qv is wired in. Fix the coupling and turn it on. See the note in examples/urban/moisture_addons.jl.", ["microphysics", "tech-debt"])],
45: [("pdf", "Sub-grid supersaturation distribution",
      "A PDF of supersaturation within a grid cell rather than a single mean value, driven by the SGS scheme.", ["microphysics", "turbulence"]),
     ("growth", "Effect on activation and condensational growth",
      "Feed the sub-grid supersaturation distribution into activation (Phase 42) and droplet growth, so entrainment can broaden the spectrum.", ["microphysics"]),
     ("broadening", "Spectral broadening validation against superdroplet and DNS",
      "Compare the broadened spectrum against the Phase 41 superdroplet reference and published DNS. This is the mechanism by which entrainment widens droplet spectra, and the bulk scheme should reproduce it.", ["microphysics", "validation"])],
46: [("richards", "Richards-equation water transport",
      "Richards equation with temperature-dependent hydraulic conductivity, over 10+ layers. Implicit or adaptive, since the equation is stiff when the soil is near saturation.", ["land"]),
     ("thermal", "Multi-layer soil thermal column with freeze-thaw",
      "Heat conduction coupled to the moisture profile, with latent heat of freezing. This is what gives the surface a real nocturnal sink - see the step_soil! energy bug in the history for why that matters.", ["land"]),
     ("watertable", "Water table and drainage",
      "A prognostic water table with lateral drainage, so soil moisture is not simply relaxed to a prescribed profile.", ["land"]),
     ("replace", "Replace the three-layer example soil with the production column",
      "examples/urban/land_surface_addons.jl carries a minimal three-layer soil. Replace it with the production column and check the Berlin diurnal cycle still behaves.", ["land", "urban", "tech-debt"])],
47: [("farquhar", "Farquhar photosynthesis", "C3 and C4 photosynthesis with the usual Rubisco-, light- and export-limited rates.", ["land"]),
     ("stomatal", "Ball-Berry / Medlyn stomatal conductance",
      "Stomatal conductance responding to light, CO2, humidity and soil water, replacing the fixed moisture-availability factor currently used.", ["land"]),
     ("fluxnet", "Soil-water stress coupling and FLUXNET validation",
      "Couple to the Phase 46 soil column and validate latent and sensible heat against FLUXNET eddy-covariance towers across several biomes.", ["land", "validation"])],
48: [("drag", "Leaf-area density and implicit canopy drag",
      "Resolved canopy drag with a leaf-area-density profile, treated IMPLICITLY so it is unconditionally stable. `apply_vegetation_drag!` currently ships disabled because explicit Euler is CFL-unstable at dt = 2 s - this issue is what fixes that.", ["land", "tech-debt"]),
     ("radiation", "Radiation attenuation through the canopy",
      "Beer-Lambert or two-stream attenuation through the leaf-area profile, coupled to Era III radiation.", ["land", "radiation"]),
     ("leafenergy", "Leaf energy balance and leaf temperature",
      "Prognostic leaf temperature from the leaf-level energy balance, rather than assuming leaves are at air temperature.", ["land"]),
     ("profiles", "Within-canopy profile validation",
      "Validate within-canopy wind and temperature profiles against forest-tower observations and wind-tunnel canopy-flow data.", ["land", "validation"])],
49: [("wallcond", "Per-facet multi-layer wall conduction",
      "Multi-layer conduction through roof, wall and ground facets with real material layers, replacing the single-slab surface.", ["urban"]),
     ("glazing", "Window and glazing transmission",
      "Shortwave transmission through glazing into the building interior, and its effect on the facade energy balance. Supersedes the glass_material.jl example add-on.", ["urban", "radiation"]),
     ("facetcouple", "Couple facet temperatures to the Era III radiosity solver",
      "The radiosity solver already resolves every facet, but the surface energy balance collapses a whole facade to ONE temperature per column - a documented approximation in examples/urban. This issue removes it: each facet carries its own temperature and exchanges longwave with the others directly.", ["urban", "radiation", "tech-debt"]),
     ("anthro", "Anthropogenic heat from a building energy model",
      "Replace the prescribed diurnal anthropogenic heat profile with output from the Phase 50 building energy model.", ["urban"]),
     ("bubble", "BUBBLE / CAPITOUL validation",
      "Validate facade and surface temperatures against the BUBBLE and CAPITOUL urban field campaigns. This is the acceptance test for the whole urban canopy.", ["urban", "validation"])],
50: [("indoor", "Indoor air temperature and thermal mass", "Prognostic indoor air temperature with building thermal mass.", ["urban"]),
     ("hvac", "HVAC demand and exhaust heat",
      "HVAC load from the indoor energy balance, and the exhaust heat it dumps outside - the air-conditioning feedback that amplifies urban heat islands.", ["urban"]),
     ("occupancy", "Occupancy schedules and validation",
      "Occupancy and equipment schedules; validate simulated energy demand against an instrumented building.", ["urban", "validation"])],
51: [("snowpack", "Multi-layer snowpack with compaction", "Layered snowpack with densification and compaction.", ["land"]),
     ("metamorph", "Metamorphism and albedo ageing",
      "Grain growth and albedo ageing. `SurfaceOptics` already has a Wiscombe-Warren snow albedo with grain size and soot - use it rather than adding a second snow albedo.", ["land", "radiation"]),
     ("percolation", "Liquid water percolation and refreezing", "Meltwater percolation through the pack and refreezing.", ["land"]),
     ("lakeice", "Lake and sea-ice thermodynamics", "Ice growth and melt on lakes and coastal sea.", ["land"])],
52: [("mixedlayer", "Mixed-layer model and diurnal warm layer",
      "A mixed-layer water column with a diurnal warm-layer scheme, so water surface temperature is prognostic rather than fixed.", ["land"]),
     ("roughness", "Wave-dependent surface roughness",
      "Charnock and beyond - roughness that responds to the wave state rather than a constant.", ["land"]),
     ("coare", "Sea spray and COARE validation",
      "Sea-spray heat flux; validate the SST diurnal cycle against buoy observations and air-sea fluxes against COARE.", ["land", "validation"])],
53: [("tiles", "Sub-grid tile framework",
      "Multiple surface types per grid cell, each with its own energy balance. Note the urban model already has a five-class map - this generalises it.", ["land", "urban"]),
     ("blending", "Blending-height formulation",
      "Aggregate tile fluxes correctly through a blending height so the grid-cell flux is right and internal boundary layers are represented.", ["land"]),
     ("hetero", "Heterogeneous-surface validation",
      "Compare resolved heterogeneous LES against tiled aggregation, and against observations from patchy-terrain campaigns.", ["land", "validation"])],
}

# --------------------------------------------------------------- tech debt
TECH_DEBT = [
 ("superdroplet-bias", "Assert superdroplet convergence rather than a loose threshold",
  "IV",
  "test/era4_superdroplet.jl asserts a single-seed L1 error below 0.25 against Golovin at t=3600 s. That passes for the wrong reason: it tolerates Monte-Carlo noise rather than testing correctness.\n\nMeasured properties worth asserting instead:\n\n- The finite-N_sd BIAS converges: ~2.5% at N_sd=2^12, falling ~2.26x per 4x N_sd (about N^-0.6). It does NOT average away over seeds - holding N fixed and going from 10 to 160 seeds leaves it at ~0.03.\n- The collision RATE is correct: ensemble-mean number decay matches Golovin's analytic n0*exp(-b*L*t) to <=1.3%, improving 3.4x per 4x N_sd.\n\nReplace the threshold with these two.",
  ["microphysics", "tech-debt", "validation"]),
 ("coverage-41-42", "Raise test coverage of Phases 41 and 42",
  "IV",
  "Exported-surface coverage is 28/73 (38%) for `Superdroplet` and 12/56 (21%) for `AerosolCloud`. Whole subsystems are never called by the suite: `cell_from_state`, `advance_positions!`, `aerosol_from_vant_hoff`, `autoconversion_rate`, the Twomey/Albrecht susceptibility functions.\n\nThese modules were written by agents that were cut off mid-work; the tested paths work, but most of the surface is unexercised.",
  ["microphysics", "tech-debt"]),
 ("p41-unfound-defects", "Locate the two defects the Phase 41 author named but never fixed",
  "IV",
  "The agent that wrote `Superdroplet.jl` reported it was fixing four defects when it was cut off: \"recursive constructors, a bogus blend expression, the import form, and the substep criterion\".\n\nTwo were found and fixed - the import form, and a bracketing bug in the implicit growth step that collapsed growing pure-water droplets onto the floor. The **recursive constructors** and the **bogus blend expression** were never located. They may sit in untested code.\n\nAudit the module against that description.",
  ["microphysics", "tech-debt"]),
 ("rrtmgp-backend", "Optional RRTMGP backend behind AbstractGasOptics",
  "III",
  "Phase 32 gas optics is a band-model reconstruction (Malkmus -> Domoto), not a fit to line-by-line data, because RRTMGP.jl needs a NetCDF/HDF5 stack that does not build on every target. Broadband fluxes land within a few percent of observations.\n\nAdd an optional RRTMGP-backed implementation behind the same `AbstractGasOptics` interface, selected at run time where it loads. The interface already supports this - no core change needed.",
  ["radiation", "tech-debt"]),
 ("export-collision-scan", "Add an export-collision check to the test suite",
  "IV",
  "Name collisions across modules have been caught three times by hand, each time only when something broke: `material_albedo`/`material_emissivity` (Phases 35 vs 36), `band_wavelength` (four copies), `effective_radius` (Phases 37 vs 41). An ambiguous export makes the name silently undefined at the top level.\n\nAdd a test that scans every module's export list and fails on any name exported by two modules without a shared generic declared in a contract module.",
  ["infrastructure", "tech-debt"]),
]

def slugify(s):
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")[:48]

def parse_roadmap():
    """title, body and validation target for every phase heading."""
    text = ROADMAP.read_text()
    out = {}
    heads = list(re.finditer(r"^### Phase (\d+) — (.+?)\s*$", text, re.M))
    for i, m in enumerate(heads):
        n = int(m.group(1))
        title = re.sub(r"\s*(✅|🚧).*$", "", m.group(2)).strip()
        end = heads[i+1].start() if i + 1 < len(heads) else len(text)
        body = text[m.end():end].strip()
        body = re.split(r"^---\s*$|^# Era ", body, flags=re.M)[0].strip()
        val = ""
        vm = re.search(r"\*\*Validated by\*\*:\s*(.+?)(?:\n\n|\Z)", body, re.S)
        if vm:
            val = " ".join(vm.group(1).split())
            body = body[:vm.start()].strip()
        out[n] = (title, " ".join(body.split()), val)
    return out

def esc(s):
    return s.replace("\\", "\\\\").replace('"""', '\\"\\"\\"')

def main():
    phases = parse_roadmap()
    lines = [
        "# Aeolus.jl roadmap, as GitHub milestones and issues.",
        "#",
        "# GENERATED by .github/scripts/generate_roadmap_issues.py — edit that,",
        "# or edit this file directly and do not regenerate. Seed GitHub with",
        "#   julia --project=. .github/scripts/seed_github.jl",
        "#",
        f"# Generated {datetime.date.today().isoformat()}.",
        "",
    ]
    for roman, title, lo, hi, state, desc in ERAS:
        lines += ["[[milestone]]",
                  f'title = "{title}"',
                  f'state = "{state}"',
                  f'description = "Phases {lo}-{hi}. {desc}"', ""]
    lines += ["[[milestone]]", f'title = "{DEBT_MILESTONE[0]}"',
              'state = "open"', f'description = "{DEBT_MILESTONE[1]}"', ""]

    n_issues = 0
    def emit(title, milestone, labels, body):
        nonlocal n_issues
        n_issues += 1
        lines.extend(["[[issue]]",
                      f'title = "{esc(title)}"',
                      f'milestone = "{milestone}"',
                      "labels = [" + ", ".join(f'"{l}"' for l in labels) + "]",
                      'body = """',
                      body.strip(), '"""', ""])

    def accept(extra):
        return ("\n## Acceptance criteria\n\n"
                + "".join(f"- [ ] {e}\n" for e in extra)
                + "- [ ] Tests assert physics — conservation, analytic limits, "
                  "published benchmarks — not merely that the code runs\n"
                  "- [ ] Full suite green: `julia --project=. test/runtests.jl`\n"
                  "- [ ] Measured validation numbers quoted in the PR body\n"
                  "- [ ] No new export collisions\n")

    for p in sorted(set(list(NEAR) + list(phases))):
        if p in DONE or p not in ERA_OF_PHASE:
            continue
        roman, ms = ERA_OF_PHASE[p]
        ptitle, pbody, pval = phases.get(p, ("", "", ""))
        if p in NEAR:
            for slug, stitle, scope, labels in NEAR[p]:
                body = (f"Part of **Phase {p} — {ptitle}**.\n\n"
                        f"## Scope\n\n{scope}\n\n"
                        f"## Phase context\n\n{pbody}\n"
                        + (f"\n**Phase validation target:** {pval}\n" if pval else "")
                        + accept([]) +
                        f"\n## References\n\n- Roadmap: `docs/src/design/longterm_roadmap.md`, Phase {p}\n"
                        f"- Workflow: [`docs/CONTRIBUTING.md`](../blob/master/docs/CONTRIBUTING.md)\n")
                emit(f"Phase {p} — {stitle}", ms,
                     [f"era-{roman.lower()}", f"phase-{p}"] + labels, body)
        elif roman in ("VI", "VII", "VIII"):
            body = (f"## Phase {p} — {ptitle}\n\n{pbody}\n"
                    + (f"\n**Validated by:** {pval}\n" if pval else "")
                    + "\n## Note\n\nThis phase has not been decomposed yet. Break it into "
                      "concrete issues before starting — see the tiering rule in "
                      "`docs/CONTRIBUTING.md`.\n"
                    + accept([]) +
                    f"\n## References\n\n- Roadmap: `docs/src/design/longterm_roadmap.md`, Phase {p}\n")
            emit(f"Phase {p} — {ptitle}", ms, [f"era-{roman.lower()}", f"phase-{p}"], body)

    for roman, title, lo, hi, state, desc in ERAS:
        if roman not in ("IX", "X", "XI"):
            continue
        rows = "".join(
            f"- **Phase {p}** — {phases[p][0]}\n" for p in range(lo, hi + 1) if p in phases)
        body = (f"Tracking epic for **{title}**.\n\n{desc}\n\n"
                f"## Phases\n\n{rows}\n"
                "## Note\n\nThis era is far enough out that decomposing it now would be "
                "guesswork. Break it into per-phase issues when the preceding era is "
                "close to complete.\n\n"
                "## References\n\n- Roadmap: `docs/src/design/longterm_roadmap.md`\n")
        emit(f"{title} (epic)", title, [f"era-{roman.lower()}", "epic"], body)

    for slug, title, era, body, labels in TECH_DEBT:
        era_label = f"era-{era.lower()}"
        emit(title, DEBT_MILESTONE[0],
             sorted(set(labels + [era_label])), body + accept([]))

    OUT.write_text("\n".join(lines))
    print(f"wrote {OUT.relative_to(ROOT)}: {len(ERAS)} milestones, {n_issues} issues")

if __name__ == "__main__":
    main()
