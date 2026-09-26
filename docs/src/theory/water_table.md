# Water table and unconfined aquifer

Issue #17 gives the [soil column](soil_water.md) a prognostic water table.
Without one, the base of the column is a fixed condition (free drainage,
bedrock or a fixed head). With one, rain that percolates through the
column is stored in an aquifer and drains laterally at a rate set by how
high the water table stands, and a high water table in turn keeps the
soil wet from below. The model is the SIMTOP groundwater scheme of
Niu et al. (2005, 2007) as used in CLM4.0 (Oleson et al. 2010, §7.5),
implemented as `UnconfinedAquifer`, a stateful `AbstractSoilBottomBoundary`
of `SoilWaterColumn` in `src/SoilColumn.jl`.

```julia
using Aeolus, Aeolus.SoilColumn

col = SoilWaterColumn(clm5_soil_grid(), carsel_parrish_texture(:silt_loam);
                      bottom = UnconfinedAquifer())
set_water_table!(col, 5.0)            # hydrostatic start, table 5 m down
step_soil_water!(col, 86400.0; water_input = 5e-7)
water_table_depth(col), aquifer_baseflow_rate(col), total_water_storage(col)
```

## 1. Aquifer storage and the water table

The aquifer is a layer of thickness ``D`` (default 25 m) below the base of
the column ``z_b``, with depth ``z`` positive downward. It holds a water
depth ``W``. While the water table ``z_\nabla`` is below the column, the
aquifer drains by gravity with specific yield ``S_y`` (0.2):

```math
z_\nabla = z_b + D - \frac{W}{S_y}, \qquad W \le W_{max} = S_y D
```

(CLM4 eq. 7.165, where ``W_{max}`` = 5000 mm and the default
``W`` = 4800 mm places the table 1 m below the column.) When the aquifer
is full the water table is inside the soil column. The aquifer can then
store water only elastically, with the column's specific storage ``S_s``,
and its pressure head at the base is ``\psi_b = (W - W_{max})/(S_s D)``.
Hydrostatics then puts the table at ``z_\nabla = z_b - \psi_b``. As a
result ``W \mapsto z_\nabla`` is continuous and monotone across both
regimes, piecewise linear with capacity
``C = -dW/dz_\nabla = S_y`` or ``S_s D``.

CLM4 handles the table-in-column case differently. It keeps two stores
and freezes the aquifer, extracting baseflow directly from the saturated
soil layers and diagnosing the table from their air-filled porosity
(eq. 7.167). Here the Richards solver already represents saturated
layers, so a single store is enough. When the table is in the column,
baseflow leaves the aquifer, and the aquifer draws water from the
saturated column through the base.

## 2. Recharge

Recharge ``q`` is the flux across the column base, positive downward. It
is the steady Darcy flux between the bottom soil node
(``\psi_N`` at ``z_N``) and the water table (``\psi = 0`` at ``z_\nabla``).
When the table is in the column it is instead the flux to the saturated
base (``\psi_b`` at ``z_b``). CLM4 uses a virtual node halfway to the table
(eq. 7.147–7.148), and Niu et al. (2007) use a single conductivity. Both
replace the unresolved unsaturated zone by one mean conductivity. That
works for downward drainage but fails badly for capillary rise from a
deep table, because ``K`` varies by orders of magnitude along the path.

This implementation instead takes ``K`` exponential in ``\psi`` between
the two points, ``K \propto e^{\alpha\psi}`` with
``\alpha = \ln(K_2/K_1)/(\psi_2 - \psi_1)``, and integrates
``q = K(1 - d\psi/dz)`` exactly (Gardner 1958):

```math
q = \frac{K_1 - K_2 e^{-\alpha L}}{1 - e^{-\alpha L}}
```

(`gardner_two_point_flux`). This expression has three properties:

- It is exact for a Gardner soil.
- It reduces to Darcy's law when ``K`` is uniform.
- It vanishes for the hydrostatic pair ``\psi_1 = \psi_2 - L`` in any
  soil, so hydrostatic equilibrium is a steady state of the discrete
  system.

Take a Gardner soil with ``\alpha = 2`` m⁻¹, a node at ``\psi = -10`` m
and the table 5 m below it. The exact capillary rise is
``4.5\times10^{-10}`` m s⁻¹. A logarithmic-mean conductivity gives
``5.0\times10^{-7}``, 1100 times too much.

## 3. Baseflow

SIMTOP's topographically controlled lateral drainage (Niu et al. 2005;
CLM4 eq. 7.159) is

```math
q_{drai} = (1 - f_{imp})\, q_{max}\, e^{-f z_\nabla},
```

with ``q_{max} = 5.5\times10^{-3}`` kg m⁻² s⁻¹ (5.5e-6 m s⁻¹) and
``f = 2.5`` m⁻¹, the CLM4.0 global values (Oleson et al. 2010, below
eq. 7.160). CLM4.5 replaced ``q_{max}`` by ``10\sin\beta`` kg m⁻² s⁻¹
(``\beta`` the mean topographic slope) and ``f_{imp}`` by an ice impedance
factor; those variants are not implemented. The frozen-soil impermeable
fraction (CLM4 eq. 7.160, ``\alpha = 3``) is

```math
f_{imp} = \frac{e^{-\alpha(1-F)} - e^{-\alpha}}{1 - e^{-\alpha}},
```

where ``F`` is the thickness-weighted ice mass fraction of the layers
from the water table down to the base. ``F`` stays zero until the
freeze–thaw model (#16) fills `θ_ice`. Niu et al. (2005) derive
``q_{max}`` from the grid cell's mean topographic index. That derivation
is not implemented: ``q_{max}`` is a parameter.

## 4. Time integration

Each accepted sub-step of the Richards solver proceeds in two stages.

1. **Implicit coupling.** For the current bottom-node potential, the
   aquifer solves the backward-Euler balance
   ``W_1 = W_0 + \Delta t\,[q(\psi_N, W_1) - q_{drai}(W_1)]``. In this
   balance the target point is frozen at its start-of-step depth, and
   its potential is the hydrostatic ``z_t - z_\nabla(W_1)``. The balance
   is strictly increasing in ``W_1``, so a safeguarded Newton iteration
   on a bracket always converges. The aquifer then hands the solver the
   tangent ``q \approx a + b\,\psi_N``, including ``dW_1/d\psi_N``.
   Without that term the coupling is unstable when the aquifer is full,
   because the elastic capacity is only ``S_s D`` = 2.5e-4 m per m of
   head.
2. **Exact aquifer update.** After convergence, ``W`` is advanced by the
   exact solution of ``dW/dt = q - q_{drai}(W)`` at the converged
   recharge. Within a regime ``z_\nabla`` is linear in ``W``, so
   ``u = e^{f z_\nabla}`` obeys the linear equation
   ``du/dt = (f/C)(q_0 - q u)``. Its solution is followed up to the
   crossing of ``z_b``, if there is one, and then continued in the other
   regime. Both regimes relax towards the same ``u_\infty = q_0/q``, so
   a path crosses at most once.

Baseflow over the sub-step is ``q\Delta t - \Delta W``. Soil plus aquifer
water (`total_water_storage`) therefore changes by exactly
infiltration − extraction − baseflow. With no recharge, the scheme
reproduces the SIMTOP recession

```math
q_{drai}(t) = \frac{q_0}{1 + t/\tau}, \qquad \tau = \frac{S_y}{f q_0},
\qquad z_\nabla(t) = z_0 + \frac{1}{f}\ln\left(1 + \frac{t}{\tau}\right)
```

for any step length. Under steady recharge ``R`` the table settles at
``z_\nabla = \ln(q_{max}/R)/f``.

## 5. Validation

All numbers are measured by `test/era5_water_table.jl`.

| check | reference | result |
| :-- | :-- | :-- |
| two-point flux, Gardner soil, 15 flux/length pairs incl. 99 % of max. capillary rise | Gardner (1958) closed form | rtol 1e-10 |
| hydrostatic pair carries no flux (vG loam, CH clay, sand) | exact | < 1e-12 K |
| recession after 200 d, table 3 m (τ = 304.4 d), 1 step and 2000 steps | ``q_0/(1+t/τ)`` | z = 3.202019 m both; rtol 1e-12 |
| constant recharge, 400 d | RK4, 40 000 steps | rtol 1e-9 |
| table from inside the column, through the kink, 30 d | RK4, 2·10⁶ steps | rtol 1e-7 |
| closed column, table ends below it (CLM5 grid, loam over sandy loam, 6 yr) | ``\psi = z - z_\nabla`` | 7.0e-8 m; water drift 4.9e-14 m |
| same, table ends inside the column (2 yr) | ``\psi = z - z_\nabla`` | 1.9e-7 m; drift 1.8e-15 m |
| steady recharge P = 2e-8 m s⁻¹, Gardner soil, 2 m column | ``z_\nabla = \ln(q_{max}/P)/f`` = 2.246708 m | 2.246708 m |
| steady profile above the table, 20 / 40 / 80 layers | Gardner (1958) | 5.9e-5 / 1.5e-5 / 3.9e-6 m (2nd order) |
| same with the table inside a 4 m column | Gardner above, ``(1-r)(z-z_\nabla)`` below | < 1e-3 m |
| 1 yr of storms, evaporation and root uptake, CLM5 silt loam; table 9.00 → 1.92 m and back | soil + aquifer budget | residual 1.6e-14 m |
| impermeable fraction at F = 1/2 | CLM4 eq. 7.160: 0.18243 | rtol 1e-12 |

**Not validated against observations.** Water-table and baseflow records
(e.g. GRACE, as in Niu et al. 2007, or well data at FLUXNET sites) need
the complete land column of #21. So far the scheme is checked only
against the closed-form and reference-integration results above.

## References

- Gardner, W. R. (1958). Some steady-state solutions of the unsaturated moisture flow equation with application to evaporation from a water table. *Soil Sci.*, 85, 228–232.
- Niu, G.-Y., Yang, Z.-L., Dickinson, R. E. & Gulden, L. E. (2005). A simple TOPMODEL-based runoff parameterization (SIMTOP) for use in global climate models. *J. Geophys. Res.*, 110, D21106.
- Niu, G.-Y. & Yang, Z.-L. (2006). Effects of frozen soil on snowmelt runoff and soil water storage at a continental scale. *J. Hydrometeor.*, 7, 937–952.
- Niu, G.-Y., Yang, Z.-L., Dickinson, R. E., Gulden, L. E. & Su, H. (2007). Development of a simple groundwater model for use in climate models and evaluation with Gravity Recovery and Climate Experiment data. *J. Geophys. Res.*, 112, D07103.
- Oleson, K. W. et al. (2010). Technical description of version 4.0 of the Community Land Model (CLM). NCAR/TN-478+STR, §7.4.2.4, §7.5.
