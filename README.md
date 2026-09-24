<div align="center">

# Aeolus.jl

**A Julia-native large-eddy simulation framework**

*Resolved turbulence, high-fidelity radiation and exact moist thermodynamics —
with one non-standard-library dependency.*

<img src="assets/hero.png" alt="Aeolus.jl — Alexanderplatz, Berlin" width="760">

<sub><b>Alexanderplatz, Berlin, 13:30 local, sun 50° above the horizon.</b>
322 buildings from OpenStreetMap, 24 h diurnal cycle at Δt = 2 s.
Surface temperature · shortwave reaching the ground, with building shadows ·
building height · resolved vertical velocity at 30 m.</sub>

</div>

---

## At a glance

- **General-purpose LES core** — Boussinesq, anelastic and pseudo-incompressible
  dynamics on a staggered finite-volume grid, behind an `AbstractMesh` interface
  that a spectral-element backend can plug into without touching user code.
- **A radiation stack that is unusual for an LES** — correlated-*k* gas optics,
  two-stream *and* 3-D Monte-Carlo transfer, urban radiosity, spectral BRDF
  surface optics, Mie aerosol.
- **Validated against external references**, not just against itself — MODIS
  kernel constants, Wiscombe's Mie benchmark, Golovin's analytic coalescence
  solution, the DYCOMS-II RF01 intercomparison.
- **7,407 tests.** 31,000 lines of source, 10,000 lines of tests.
- **FFTW is the only non-stdlib dependency.** Everything else — NetCDF, PNG,
  Mie scattering, error and Bessel functions — is written in the package.
- Single-node CPU. MPI and GPU are planned, not present.

---

## What the stack can do

<table>
<tr><td width="34%"><b>Dynamical core</b></td><td>

Boussinesq · anelastic · pseudo-incompressible · staggered Arakawa-C finite
volume · fractional-step pressure projection · SSP-RK3, IMEX (ARK2,
Rosenbrock2) · adaptive Δt with CFL diagnostics

</td></tr>
<tr><td><b>Pressure solvers</b></td><td>

FFT (triply periodic) · cosine-FFT (Neumann in *z*) · conjugate gradient ·
geometric multigrid · density-weighted variants for anelastic flow

</td></tr>
<tr><td><b>Advection &amp; geometry</b></td><td>

WENO-5 / WENO-Z / TENO · skew-symmetric energy-conserving operators ·
cut-cell immersed boundaries with ghost-cell wall reconstruction ·
stretched and terrain-following (Gal-Chen) vertical coordinates

</td></tr>
<tr><td><b>Mesh refinement</b></td><td>

One- and two-way grid nesting with flux correction · block-structured AMR
(Berger-Rigoutsos clustering, Berger-Colella refluxing)

</td></tr>
<tr><td><b>Radiation</b></td><td>

Correlated-*k* gas optics over 16 longwave and 14 shortwave bands
(H₂O, CO₂, O₃, CH₄, N₂O, O₂) · two-stream δ-Eddington / PIFM / quadrature
with adding-doubling · 3-D Monte-Carlo transfer with next-event estimation ·
urban radiosity with occlusion-tested view factors · spectral surface optics
(16 materials, 4 BRDF models, Cox-Munk water, Wiscombe-Warren snow) ·
Mie aerosol with κ-Köhler growth and OPAC mixtures · Meeus solar position,
Bird clear-sky, Erbs/Reindl direct-diffuse partitioning

</td></tr>
<tr><td><b>Clouds &amp; microphysics</b></td><td>

Exact moist thermodynamics — temperature-dependent latent heat, saturation
vapour pressure as the analytic Clausius-Clapeyron integral, condensate
loading · robust bracketed saturation adjustment · Köhler activation ·
Abdul-Razzak &amp; Ghan parameterisation with an explicit parcel reference ·
superdroplet Lagrangian microphysics · Kessler and two-moment bulk schemes

</td></tr>
<tr><td><b>Surface &amp; urban</b></td><td>

OpenStreetMap ingestion → building geometry and land use · facet radiosity
resolving shadowing, inter-reflection and canyon longwave trapping ·
multi-layer soil conduction · latent heat flux · anthropogenic heat ·
material albedo and emissivity derived from the spectral library

</td></tr>
<tr><td><b>Output</b></td><td>

CF-1.8 NetCDF, readable by ncview, xarray, Paraview, CDO and Panoply ·
pure-Julia PNG, APNG, PPM and SVG rendering

</td></tr>
</table>

---

## Validated against

Every figure below is produced by the test suite, against a reference the
code does not contain.

| Check | Aeolus | Reference |
|:--|:--|:--|
| MODIS Ross-thick white-sky albedo | `0.189186` | `0.189184` (published) |
| MODIS Li-sparse-reciprocal WSA | `-1.377660` | `-1.377622` (published) |
| Mie extinction, *x*=10, *m*=1.5 | `2.881999` | `2.8820` (Wiscombe) |
| Mie extinction, *x*=0.101, *m*=0.75 | `8.033538e-6` | `8.033542e-6` (Wiscombe) |
| View factor vs closed form | `9.2e-15` | exact |
| Canyon sky-view factor | within `1.4%` | √((H/W)²+1) − H/W |
| DYCOMS-II RF01 cloud-top net LW | `70.3 W/m²` | `70.1` (intercomparison) |
| Clear-sky OLR | `240.2 W/m²` | ~240 (observed) |
| Greenhouse effect | `149.0 W/m²` | ~150 (observed) |
| Clausius-Clapeyron self-consistency | `5.2e-12` | exact |
| Liquid/ice saturation gap | `27.00 Pa @ 261.5 K` | ~27 Pa near 261 K |
| Golovin coalescence, *L*¹ at 1200 s | `0.037` | analytic |
| Monte-Carlo vs two-stream albedo | median `0.0059` | — |
| Cloud optical depth, LWP 100 g/m², *r*<sub>eff</sub> 10 μm | `15.0` | 15 (textbook) |

Plus the classical CFD cases — Taylor-Green vortex, decaying isotropic
turbulence, channel flow, buoyant convection — in `examples/`.

---

## Installation

```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
```

Julia ≥ 1.10. The only non-standard-library dependency is **FFTW**.

> **Why so few dependencies?** The usual scientific-Julia stack — NetCDF,
> HDF5, Makie, SpecialFunctions — needs binary libraries that do not build on
> every target platform. Rather than ship a package that fails to load,
> Aeolus writes what it needs: a CF-compliant NetCDF-3 writer, a PNG/APNG
> encoder, a Mie solver, and series expansions for the error and Bessel
> functions. See [`src/NetCDFIO.jl`](src/NetCDFIO.jl) for the clearest case.

---

## Quick start

A dry convective boundary layer — warm below, cool above, left to overturn:

```julia
using Aeolus, Printf
using Aeolus.Fields: set!

L, θ₀, Δθ = 1000.0, 300.0, 2.0
mesh = RegularGrid(; size = (48, 48, 48), extent = (L, L, L), halo = (2, 2, 2))
sim  = BoussinesqSimulation(mesh; θ₀ = θ₀, g_accel = 9.81,
                                  C_s = 0.17, Pr_t = 1/3)

# unstable stratification plus a small perturbation to break symmetry
set!(sim.θ, (x, y, z) -> θ₀ + Δθ * (0.5 - z/L) + 1e-3 * Δθ * (rand() - 0.5))

w_star = sqrt(9.81 * Δθ / θ₀ * L)          # free-fall velocity scale
dt     = 0.3 * mesh.dx / w_star            # CFL

for _ in 1:400
    step!(sim, dt)
end

@printf("KE = %.1f   max|div u| = %.2e   max nu_t = %.4f\n",
        kinetic_energy(sim), max_divergence(sim), max_νt(sim))
```

```
KE = 24701.0   max|div u| = 2.80e-17   max nu_t = 0.0337
```

Divergence stays at machine precision: the fractional-step projection is
doing its job. Run the full test suite with `Pkg.test()`, or a worked case
directly:

```bash
julia --project=. examples/taylor_green_vortex.jl
julia --project=. examples/buoyant_convection.jl
```

---

## Flagship application — urban climate

The most demanding thing in the repository is a full diurnal simulation of a
real city block, and it is where most of the radiation stack earns its place.

**Why it is hard.** A street canyon is not a flat surface with an albedo.
Sunlight that misses the road hits a wall and bounces back; longwave radiation
that would escape to a cold sky is intercepted by the building opposite. The
result is *radiative trapping*: the effective albedo of a canyon is
substantially lower than the albedo of the materials it is built from, so the
city absorbs more than the sum of its surfaces would suggest. That is a
first-order driver of the urban heat island, and a sky-view-factor
approximation cannot represent it.

**Measured on the real Alexanderplatz geometry** — 322 OpenStreetMap
buildings, 64 × 64 × 32 cells, 24 h at Δt = 2 s:

| | |
|:--|:--|
| Effective albedo of the scene | `0.126` |
| Area-weighted material albedo | `0.175` |
| **Radiative trapping** | **28.0 %** |
| Radiating facets | 5,810 |
| View-factor reciprocity / closure | `4.5e-17` / `2.4e-15` |
| Sky-view factor (min / mean / max) | `0.149` / `0.794` / `1.000` |
| Peak surface temperature contrast | `18.6 K` |
| Divergence after 43,200 steps | `1.6e-15` |
| Wall-clock, one CPU | **9.2 min** |
| Output | 228 MB NetCDF, 48 time levels |

A full day of a city block, with resolved turbulence and facet-level
radiation, costs about nine minutes on a single processor.

### Run your own city

```bash
# 1. Fetch a 1 km box from OpenStreetMap (edit the bounding box inside)
julia --project=. examples/urban/data/fetch_alexanderplatz.jl

# 2. Simulate 24 hours — writes CF-1.8 NetCDF
julia --project=. examples/urban/alexanderplatz_diurnal.jl

# 3. Render the diurnal cycle as an animation
julia --project=. examples/urban/alexanderplatz_animation.jl
```

Nothing in the simulation is tuned to Alexanderplatz — the same script runs on
any Overpass download. Output is a single `.nc` file carrying the full 3-D
state (`u`, `v`, `w`, `θ`, `p`, `qv`, `ql`, `qr`, tracers) plus surface fields
and static geometry, at every saved time level.

---

## Documentation

```bash
julia --project=docs docs/make.jl     # build locally
```

- **Theory** — Boussinesq equations, LES filtering and closure, staggered
  discretization, fractional-step projection, discrete Poisson,
  [radiative transfer](docs/src/theory/radiation.md)
- **Design** — architecture, dispatch on `Location`, pluggable pressure solvers
- **API reference** — generated for all 40 modules
- **Examples** — worked cases with validation output

---

## Roadmap

Development is organised into *eras*, each a coherent block of capability that
ends with a validation target. Full plan:
[`docs/src/design/longterm_roadmap.md`](docs/src/design/longterm_roadmap.md).

- [x] **Era I** — LES core, physics scaffolds, urban flagship *(Phases 1–20)*
- [x] **Era II** — numerical foundations: cut-cell IBM, IMEX, WENO-Z/TENO,
      anelastic, multigrid, nesting, AMR *(21–31)* · 2,094 tests
- [x] **Era III** — high-fidelity radiation: gas optics, two-stream,
      Monte-Carlo, radiosity, BRDF, aerosol, solar geometry *(32–38)* · 4,881 tests
- [ ] **Era IV** — clouds and microphysics *(39–45)*
  - [x] Phase 39 — exact moist thermodynamics
  - [ ] Phase 40 — P3 bulk microphysics
  - [x] Phase 41 — superdroplet Lagrangian microphysics
  - [x] Phase 42 — aerosol-cloud interaction
  - [ ] Phase 43 — ice nucleation
  - [ ] Phase 44 — precipitation, fog and visibility
  - [ ] Phase 45 — turbulence-microphysics interaction
- [ ] **Era V** — land surface and urban canopy *(46–53)*
- [ ] **Era VI** — turbulence closure *(54–59)*
- [ ] **Era VII** — chemistry and aerosol *(60–66)*
- [ ] **Era VIII** — real-world data, forcing and validation *(67–73)*
- [ ] **Era IX** — data assimilation and inverse problems *(74–79)*
- [ ] **Era X** — performance and scale: MPI, GPU *(80–87)*
- [ ] **Era XI** — machine learning, applications, ecosystem *(88–100)*

---

## Known limitations

Caveats of what currently ships, as distinct from what is not yet built.

- **Single-node CPU only.** Domain decomposition and GPU backends exist as
  interfaces, not implementations. Era X.
- **Gas optics is a band-model reconstruction**, not a fit to line-by-line
  data. `RRTMGP.jl` does not load on every platform, so the *k*-distribution
  is built from a Malkmus narrow-band model via Domoto inversion. Broadband
  fluxes land within a few percent of observations; a line-by-line backend can
  be added behind the same interface.
- **One surface temperature per grid column.** The radiosity solver resolves
  every facet individually, but the surface energy balance collapses a whole
  facade to a single temperature. Removed by the multi-layer canopy in Era V.
- **Superdroplet coalescence carries a finite-*N*<sub>sd</sub> bias** — about
  2.5 % at *N*<sub>sd</sub> = 2¹², shrinking as ≈ *N*<sup>-0.6</sup>. It does
  not average away over random seeds. The collision *rate* is correct: mean
  number decay matches Golovin's analytic solution to ≤ 1.3 %.
- **Two urban add-ons ship disabled**, with reasons documented in
  `examples/urban/`: vegetation drag is CFL-unstable under explicit Euler at
  Δt = 2 s and needs implicit treatment; fog/dew condensation needs tighter
  coupling to the atmospheric moisture field.
- **Test coverage is uneven.** The dynamical core and radiation are covered
  thoroughly; the newest modules (Phases 41–42) exercise 38 % and 21 % of
  their exported surface respectively.

---

## License and citation

MIT — see [`LICENSE`](LICENSE).

If you use Aeolus in published work, please cite it via
[`CITATION.cff`](CITATION.cff) (GitHub renders a *Cite this repository*
button from it).
