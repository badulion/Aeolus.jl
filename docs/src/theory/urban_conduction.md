# Urban facet conduction

A city stores heat in its fabric. By day, roofs, walls and roads take up
a large part of the absorbed sunshine; by night they give it back, which
is a principal cause of the nocturnal urban heat island. How much they
store and how late they release it depends on what they are made of and
in what order the materials are stacked. A brick wall and an insulated
curtain wall with the same outside temperature pass very different heat
to the street at night and to the room by day. The single-slab surface
of the urban example, one heat capacity per surface class, cannot tell
them apart. `src/UrbanSurface.jl` models each facet as a
multi-layer conduction column, following TEB (Masson 2000) and PALM-USM
(Resler et al. 2017).

## 1. The equation and the grid

Across a facet, lateral gradients are small compared with the gradient
through its thickness. The 1-D heat equation

```math
\rho c\,\frac{\partial T}{\partial t} = \frac{\partial}{\partial z}\Bigl(k\,\frac{\partial T}{\partial z}\Bigr)
```

holds in each homogeneous layer, with ``T`` and the flux ``-k\,\partial T/\partial z``
continuous at material interfaces. ``z`` runs from the outer face
(``z = 0``) inward.

Each `WallLayer` is split into equal finite-volume cells, and material
interfaces always lie on cell faces. Temperatures sit at cell centres.
The conductance between cells ``i`` and ``i+1`` is that of the two half
cells in series,

```math
K_{i+1/2} = \Bigl(\frac{\Delta z_i}{2k_i} + \frac{\Delta z_{i+1}}{2k_{i+1}}\Bigr)^{-1},
```

which is the exact flux across an interface for a piecewise-linear
profile. A steady composite wall is therefore reproduced exactly at any
resolution, even with one cell per layer.

## 2. Boundary conditions: one Robin form for everything

Both faces take the same boundary condition. The heat flux entering the
column is

```math
F = h\,(T_\text{env} - T_\text{face}) + q .
```

This single form covers all the cases a facet needs:

| case | ``h`` | ``T_\text{env}`` | ``q`` |
|:-----|:------|:-----------------|:------|
| prescribed face temperature (deep ground) | ``\infty`` | ``T`` | — |
| prescribed flux, adiabatic | 0 | — | ``q`` |
| interior face ↔ indoor air (ISO 6946) | ``1/R_{si}`` | ``T_\text{in}`` | 0 |
| outer face, linearised energy balance | ``-F'`` | ``T_s^n`` | ``F_0`` |

The face temperature is eliminated with the half-cell conductance
``K = 2k/\Delta z``. The boundary flux then becomes an affine function of
the adjacent cell temperature ``T_c``:

```math
F = \frac{K h}{K+h}\,(T_\text{env} - T_c) + \frac{K}{K+h}\,q ,
\qquad
T_\text{face} = \frac{K T_c + h T_\text{env} + q}{K + h}.
```

Because the column only ever sees a `ColumnBoundary`, it knows nothing
about the atmosphere. The building energy model (Phase 50) closes the
same column with its own indoor-air temperature.

## 3. Time stepping

The θ-scheme

```math
\rho_i c_i \Delta z_i\,\frac{T_i^{n+1} - T_i^n}{\Delta t}
 = \theta\,R_i(T^{n+1}) + (1-\theta)\,R_i(T^n),
```

with ``R_i`` the net flux into cell ``i`` (including boundary fluxes),
is a tridiagonal system, solved by the Thomas algorithm in ``O(n)``.
``\theta = 1/2`` (Crank–Nicolson) is second order and A-stable.
``\theta = 1`` (backward Euler) is first order but L-stable: it damps
the stiffest modes monotonically, which is preferable when ``\Delta t``
is far longer than ``\Delta z^2/\kappa`` of the thinnest cell, for
example a 1 cm waterproofing layer stepped hourly. ``\theta < 1/2`` is
only conditionally stable (explicit Euler on 2 cm brick cells at
``\Delta t = 600`` s diverges within 100 steps), so the column rejects it.

`step_conduction!` returns the θ-weighted boundary fluxes it applied.
Summing the cell equations telescopes the interior conductances away, so

```math
\sum_i \rho_i c_i \Delta z_i (T_i^{n+1} - T_i^n) = (F_\text{outer} + F_\text{inner})\,\Delta t
```

holds exactly. Energy is conserved to round-off, not to truncation error.

## 4. The outer face: surface energy balance

The skin temperature ``T_s`` is the temperature of the outer face. It
has no heat capacity of its own; the first cell provides that. Its
energy balance

```math
F(T_s) \equiv R_n(T_s) - H(T_s) - LE(T_s) = G
```

is nonlinear, through ``\varepsilon\sigma T_s^4`` and through the
stability dependence of ``H``. It is linearised about the current skin
temperature,

```math
F(T) \approx F_0 + F'\,(T - T_s^n),\qquad
F' = \frac{\partial R_n}{\partial T} - \frac{\partial H}{\partial T} - \frac{\partial LE}{\partial T} < 0,
```

where ``F_0`` and ``F'`` come from `SurfaceExchange.turbulent_fluxes`
(Monin–Obukhov, with ``F'`` by a one-sided difference so the stability
dependence of ``C_H`` is included). The linearised balance is exactly the
Robin condition ``h = -F'``, ``T_\text{env} = T_s^n``, ``q = F_0``: the
combined radiative and turbulent surface coefficient. The column step is
then implicit in the skin temperature, which stops a thin, highly
conductive outer cell from oscillating, and the new skin temperature is
the outer face temperature.

The fluxes returned by `step_surface!` are those actually applied,
evaluated on the same linearisation. So ``R_n - H - LE = G`` holds to
round-off, and ``G\,\Delta t`` minus the interior flux times
``\Delta t`` is the change in stored heat. Each step is one Newton
iteration on the skin balance. An inconsistent initial skin is pulled
onto the balance with a quadratically falling residual (measured:
360 → 0.45 → 5·10⁻⁶ → 6·10⁻¹¹ W m⁻²).

Vertical walls fall outside Monin–Obukhov similarity. For them a facet
may use a fixed convective coefficient, ``H = h_c (T_s - T_\text{air})``
(`h_convective`). Wind-dependent wall correlations belong with the
radiosity coupling of the street canyon.

## 5. Materials and optics: one source of truth

`thermal_material` holds conductivity, density and specific heat only,
each row cited: Bergman et al. (2011, Tables A.1, A.3) for masonry,
concrete, glass, steel, plasters, insulation and soils, and Goward (1981)
via Heisler & Brazel (2010, Table 2-3) for asphalt. Incropera's asphalt,
``k = 0.062`` W m⁻¹ K⁻¹, is an order of magnitude below every urban
compilation and would make a road an insulator. Heisler & Brazel also
tabulate the thermal admittance ``\mu = \sqrt{k\rho c}``: the asphalt row
reproduces their 1204 J m⁻² K⁻¹ s⁻¹ᐟ² to 0.003 %.

Optical properties are not restated. An exposed material carries the
name of its `SurfaceOptics` spectral-library entry. A facet's albedo is
the white-sky broadband albedo of that spectral curve, and its
emissivity is the library's broadband value. A steel roof therefore
inherits the metal-roof ``\varepsilon = 0.25``. `Radiosity.UrbanMaterials`
derives its class greys from the same library, so the two cannot drift
apart.

## 6. Validation

All numbers are from `test/era5_urban_surface.jl`.

| check | reference | measured |
|:------|:----------|:---------|
| Steady composite wall (render / brick / 8 cm EPS / plaster) | ISO 6946 ``U = 1/(R_{si} + \sum d/k + R_{se}) = 0.3847`` W m⁻² K⁻¹ | flux equals ``U\Delta T`` to 1·10⁻¹⁰ (rel.), profile piecewise-linear to 2·10⁻¹² K, same with one cell per layer |
| Periodic surface temperature, semi-infinite concrete | Carslaw & Jaeger: flux ``A\mu\sqrt\omega``, phase lead π/4; ``e^{-z/d}``, lag ``z/d`` | flux amplitude 6.5·10⁻⁵, phase 2.6·10⁻⁴ rad; at ``d`` and ``2d`` amplitude ≤ 1.3·10⁻⁴, phase ≤ 1.7·10⁻³ rad |
| Periodic forcing of three constructions, flux into the room | ISO 13786 heat-transfer matrix (analytic) | amplitude within 0.87 % / 1.15 % / 0.37 % at 2 cm cells, ≤ 0.1 % at 5 mm; time lag within 0.024 h (brick wall 7.74 h, insulated brick 9.60 h, 20 cm slab 4.42 h) |
| Decaying slab mode | ``e^{-\kappa\pi^2t/L^2}\sin(\pi z/L)`` | space order 1.98–2.00; time order CN 2.00, BE 0.99–1.00 |
| Column energy under random Robin / Dirichlet / flux conditions | exact bookkeeping | ~1·10⁻¹² relative |
| Facet steady state (fixed ``h_c`` and Monin–Obukhov) | bisection of ``R_n - H = (T_s - T_\text{in})/(R + R_{si})`` | skin temperature to 1·10⁻⁶ K |
| 3-day diurnal roof / wall / road | exact bookkeeping; Δt = 60 s vs 600 s | SEB closure ≤ 3·10⁻¹⁴ W m⁻², conservation to round-off, hourly skin within 0.16 K |

**Not yet validated against observations.** The phase target is the
BUBBLE and CAPITOUL facade and surface temperatures. That comparison
needs the facet radiosity forcing (canyon shading, sky view) and
belongs to the Phase 49 validation issue. What is validated here is the
numerical solution of the physics: steady, periodic, transient, and the
bookkeeping of energy.

## References

- Bergman, T. L., Lavine, A. S., Incropera, F. P. & DeWitt, D. P. (2011).
  *Fundamentals of Heat and Mass Transfer*, 7th ed. Wiley, Appendix A.
- Carslaw, H. S. & Jaeger, J. C. (1959). *Conduction of Heat in Solids*,
  2nd ed. Oxford, §2.6.
- Goward, S. N. (1981). Thermal behavior of urban landscapes and the urban
  heat island. *Phys. Geogr.*, 2, 19-33.
- Heisler, G. M. & Brazel, A. J. (2010). The urban physical environment:
  temperature and urban heat islands. *Urban Ecosystem Ecology*, Agron.
  Monogr. 55, 29-56.
- ISO 6946:2017. Building components and building elements — Thermal
  resistance and thermal transmittance — Calculation methods.
- ISO 13786:2017. Thermal performance of building components — Dynamic
  thermal characteristics — Calculation methods.
- Masson, V. (2000). A physically-based scheme for the urban energy budget
  in atmospheric models. *Boundary-Layer Meteorol.*, 94, 357-397.
- Resler, J. et al. (2017). PALM-USM v1.0: A new urban surface model
  integrated into the PALM large-eddy simulation model. *Geosci. Model
  Dev.*, 10, 3635-3659.
