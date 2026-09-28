# Resolved plant canopy: leaf area and implicit drag

A resolved canopy is vegetation that occupies several model layers, so
that the LES sees the leaves as a distributed momentum sink rather than
as a roughness length. This page covers the two pieces implemented in
`src/PlantCanopy.jl` for Phase 48: the vertical distribution of leaf
area, and the drag it exerts, treated implicitly so it is stable at any
time step. Radiation attenuation (#23) and the leaf energy balance (#24)
build on the same leaf-area coordinate.

## 1. Leaf-area density

The leaf-area density ``a(z)`` [m² m⁻³] is one-sided leaf area per unit
volume; its column integral is the leaf-area index,
``\mathrm{LAI} = \int_0^h a\,dz``. Three profiles are provided, each
normalised to its LAI by construction:

| Profile | ``a(z)`` | Use |
| :-- | :-- | :-- |
| `UniformLAD(LAI, h; z_base)` | ``\mathrm{LAI}/(h - z_{base})`` on the crown | analytic theory, idealised cases |
| `LalicMihailovicLAD(LAI, h; z_m)` | ``L_m r^n e^{n(1-r)}``, ``r = \frac{h-z_m}{h-z}``, ``n = 6`` below ``z_m``, ``1/2`` above | forests (Lalic & Mihailovic 2004) |
| `BetaLAD(LAI, h; p, q)` | ``\frac{\mathrm{LAI}}{h}\frac{x^{p-1}(1-x)^{q-1}}{B(p,q)}``, ``x = z/h`` | flexible crown shapes (Markkanen et al. 2003) |

For Lalic–Mihailovic the peak ``L_m = a(z_m)`` is solved for so that the
profile integrates to the given LAI.

On a model grid the LAD of a layer is its **layer mean**
``a_k = Δz_k^{-1}\int_{z_k}^{z_{k+1}} a\,dz`` (`layer_leaf_area_density`),
not a point sample. With the urban example's ``Δz = 6.25`` m and a
20 m canopy the top layer spans 18.75–25 m; centre sampling puts no
leaves there and loses 6 % of the LAI, the layer integral loses none.
The integrals use Gauss–Legendre quadrature graded towards the ends of
each smooth piece, which keeps the ``(1-x)^{q-1}`` end of a beta profile
with non-integer ``q`` at round-off accuracy.

The leaf area above a height, ``Λ(z) = \int_z^h a\,dz'``
(`cumulative_leaf_area`), is the optical-depth coordinate for canopy
radiative transfer.

## 2. Canopy drag, and why it must be implicit

Form drag on the leaves is (Shaw & Schumann 1992)

```math
\frac{\partial U_i}{\partial t} = -C_d\,a\,|U|\,U_i ,
```

with ``C_d ≈ 0.15``–``0.3``. Taken alone, the speed obeys
``d|U|/dt = -C_d a |U|^2`` and the direction is fixed, so the exact
solution over a step ``Δt`` from ``U^*`` is

```math
U^{n+1} = \frac{U^*}{1 + Δt\,C_d\,a\,|U^*|}.
```

This is also backward Euler with the drag coefficient ``C_d a |U|``
lagged, but here the lagging costs nothing: for the pure drag equation
the factor is exact at any ``Δt``. It lies in ``(0, 1]``, so the step is
unconditionally stable and can never reverse the flow.

Explicit Euler, ``U^{n+1} = U^n (1 - Δt\,C_d a |U^n|)``, reverses the flow
once ``Δt\,C_d a|U| > 1`` and amplifies it once that exceeds 2. For a
dense forest (``a ≈ 1`` m⁻¹, ``C_d = 0.2``) in a 5 m s⁻¹ wind the limit is
``Δt = 2`` s — the urban example's step. The old explicit
`apply_vegetation_drag!` shipped disabled after an instability at that
step was reported, but in the urban example's own configuration
``Δt\,C_d\,\mathrm{LAI}\,|U| ≤ 0.05`` and re-running it for #22 did not
reproduce the instability; its cause is undiagnosed. The implicit form
removes the stiffness limit either way.

### In the LES

`apply_canopy_drag!(sim, drag, Δt)` runs in the `post_predictor!` hook,
on the predicted velocity ``u^*`` before the pressure projection. The
three velocity components live on different faces of the Arakawa-C
grid. At each face the LAD is the mean of the two adjacent cells, the
speed ``|U|`` includes the other two components averaged from their
four neighbouring faces, and all factors are computed before any is
applied. The call returns the momentum removed,
``\sum (U^{n+1} - U^*)ΔV/Δt``, which closes the momentum budget exactly.

## 3. In-canopy wind: the exponential profile

In a deep uniform canopy with a constant mixing length ``l`` the steady
mean momentum balance

```math
\frac{d}{dz}\left(l^2 \left|\frac{dU}{dz}\right|\frac{dU}{dz}\right) = C_d\,a\,U^2
```

has the solution ``U = U_h e^{γ(z-h)}`` with ``γ = (C_d a / 2l^2)^{1/3}``
(Inoue 1963). Harman & Finnigan (2007) express the same profile through
``β = u_*/U_h`` and the canopy-drag length ``L_c = 1/(C_d a)``: the
mixing length is ``l = 2β^3 L_c`` and ``U = U_h e^{β(z-h)/l}``, so
``γ = β/l``. Substituting shows that the stress at the canopy top
``l^2 (dU/dz)^2 = β^2 U_h^2``, i.e. the friction velocity comes back as
``β U_h``.

`solve_canopy_wind_column` solves the balance above on a 1-D column,
with ``K = l^2|∂U/∂z| + ν``, the wind prescribed at the column top, a
free-slip or no-slip ground, and the implicit drag folded into a
tridiagonal pseudo-time step. At convergence the discrete stress at the
top equals the drag on the leaves plus the ground stress.

Measured against the analytic profile (``h = 20`` m, LAI = 10,
``C_d = 0.25``, ``β = 0.3``, so ``l = 0.432`` m and ``γh = 13.9``),
compared down to ``U = e^{-5} U_h``:

| Layers | max rel. error of ``U(z)`` | ``u_*/U_h`` |
| --: | --: | --: |
| 100 | ``5.5\times10^{-3}`` | 0.29904 |
| 200 | ``1.4\times10^{-3}`` | 0.29976 |
| 400 | ``3.5\times10^{-4}`` | 0.29994 |

Convergence is second order and ``u_*/U_h → β``. The comparison needs
``ν`` far below ``l^2|∂U/∂z|``: deep in a dense canopy that viscosity
decays with ``U``, and a molecular ``ν = 1.5\times10^{-5}`` m² s⁻¹
already shifts the profile by ≈ 0.15 % at depth ``5/γ``.

What this does **not** yet establish: agreement with forest-tower or
wind-tunnel observations. That is the job of the canopy-profile
validation issue (#25), once the radiation and leaf energy balance are
in place.

## References

- Harman, I. N. & Finnigan, J. J. (2007). A simple unified theory for
  flow in the canopy and roughness sublayer. *Boundary-Layer Meteorol.*,
  123, 339–363.
- Inoue, E. (1963). On the turbulent structure of airflow within crop
  canopies. *J. Meteorol. Soc. Japan*, 41, 317–326.
- Lalic, B. & Mihailovic, D. T. (2004). An empirical relation describing
  leaf-area density inside the forest for environmental modeling.
  *J. Appl. Meteorol.*, 43, 641–645.
- Markkanen, T., Rannik, Ü., Marcolla, B., Cescatti, A. & Vesala, T.
  (2003). Footprints and fetches for fluxes over forest canopies with
  varying structure and density. *Boundary-Layer Meteorol.*, 106, 437–459.
- Shaw, R. H. & Schumann, U. (1992). Large-eddy simulation of turbulent
  flow above and within a forest. *Boundary-Layer Meteorol.*, 61, 47–64.
