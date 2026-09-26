# Resolved plant canopy: leaf area, drag and radiation

A resolved canopy is vegetation that occupies several model layers, so
that the LES sees the leaves as a distributed momentum sink rather than
as a roughness length. This page covers what `src/PlantCanopy.jl`
implements for Phase 48: the vertical distribution of leaf area, the
drag it exerts, treated implicitly so it is stable at any time step, and
the attenuation of shortwave and longwave radiation through it
(section 4). The leaf energy balance (#24) builds on the same
leaf-area coordinate.

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

## 4. Radiation in the canopy

The optical depth of a canopy is its cumulative leaf area above a height,
``Λ(z) = \int_z^h a\,dz'`` (`cumulative_leaf_area`). How much of a beam
unit leaf area intercepts depends on leaf orientation through Ross's
projection function ``G(μ)``; `RossGoudriaanLeaves(χ_L)` uses Goudriaan's
fit ``G = φ_1 + φ_2 μ`` (valid for ``-0.4 ≤ χ_L ≤ 0.6``; ``χ_L = 0`` is the
spherical distribution with ``G = 1/2``), and `HorizontalLeaves()` has
``G = μ`` exactly.

**Beer–Lambert.** The unscattered beam from a sun at zenith cosine ``μ``
reaches depth ``Λ`` with ``e^{-KΛ}``, ``K = G(μ)/μ``, and that is also the
sunlit fraction of the leaves there. For black leaves the exact diffuse
transmittance is the hemispheric integral
``T_d = 2\int_0^1 e^{-GΛ/μ} μ\,dμ`` (`canopy_diffuse_transmittance`;
``2E_3(Λ/2)`` for spherical leaves).

**Two-stream (Sellers 1985).** With leaf reflectance ``α``, transmittance
``τ_l`` and ``ω = α + τ_l``,

```math
-\bar μ \frac{dI^\uparrow}{dΛ} + [1-(1-β)ω]I^\uparrow - ωβ I^\downarrow = ω\bar μ K β_0 e^{-KΛ},\qquad
\bar μ \frac{dI^\downarrow}{dΛ} + [1-(1-β)ω]I^\downarrow - ωβ I^\uparrow = ω\bar μ K (1-β_0) e^{-KΛ},
```

with ``\bar μ = \int_0^1 μ/G\,dμ`` and the up-scatter fractions ``β``,
``β_0`` of the CLM5.0 Technical Note (§2.3.1). Dividing by ``\bar μ``
puts this in exactly the Meador–Weaver form solved by the Era III
`TwoStream` module, with optical depth ``Λ``, beam cosine ``1/K`` and

```math
γ_1 = \frac{1-(1-β)ω}{\bar μ},\quad γ_2 = \frac{ωβ}{\bar μ},\quad γ_3 = β_0,\quad γ_4 = 1-β_0 .
```

`canopy_shortwave` therefore does not re-implement the two-stream: it
supplies these coefficients as a `TwoStreamVariant` and calls
`layer_reflectance_transmittance_sw` and `adding_solver!` (which gained
a `F_dif_top` keyword for the sky diffuse entering at the canopy top).
The ground enters as the lower boundary albedo. Because the equations
depend on height only through ``Λ``, the solution does not depend on how
the leaf area is split into layers.

**Sunlit and shaded leaves.** A layer between ``Λ_t`` and ``Λ_b`` holds
``(e^{-KΛ_t} - e^{-KΛ_b})/K`` of sunlit leaf area. Sunlit leaves absorb
the fraction ``1-ω`` of the beam they intercept; the diffuse absorption
of the layer (sky diffuse plus scattered beam) is shared in proportion to
leaf area (de Pury & Farquhar 1997).

**Longwave.** `canopy_longwave` treats each layer as a slab with leaf
reflectance ``1 - ε_{leaf}``, takes ``R`` and ``T`` from the same
two-stream layer formulae, emits ``(1-R-T)σT_{leaf}^4`` in each direction
(Kirchhoff) and solves the coupled layer–ground system exactly.

`canopy_shortwave_field!` applies the column solution to each grid
column of a resolved LES canopy (independent-column approximation) and
returns the absorbed shortwave per cell in W m⁻³.

### What the tests measure

| Check | Reference | Measured |
| :-- | :-- | --: |
| ``\bar μ`` closed form (incl. series near ``χ_L = 0``) | quadrature of ``\int μ/G`` | ``6.7\times10^{-16}`` rel. |
| ``β_0`` closed form for ``a_s`` | quadrature of its integral | ``< 10^{-8}`` rel. |
| ``β_0`` at the removable singularity ``μφ_2 + G = 0`` (``χ_L = -0.4``, ``μ ≈ 0.996``) | quadrature of its integral | ``8.9\times10^{-16}`` rel. |
| Beam flux at every face | ``e^{-KΛ(z)}`` | ``< 10^{-9}`` rel. |
| Black-leaf diffuse | ``2E_3(Λ/2)`` via series ``E_1`` | ``< 10^{-10}`` rel. |
| Energy closure (576 cases; a bookkeeping identity, since absorption is the flux divergence) | incident = reflected + canopy + ground | ``3.2\times10^{-16}`` |
| Face fluxes | RK4 integration of the Sellers ODEs | ``3.4\times10^{-14}`` of incident |
| Face fluxes at the resonance ``K = k`` (``μ ≈ 0.54`` for spherical PAR leaves) | RK4 integration of the Sellers ODEs | ``1.9\times10^{-9}`` of incident |
| Deep horizontal-leaf canopy albedo | Goudriaan ``ρ_h = \frac{1-\sqrt{1-σ}}{1+\sqrt{1-σ}}`` | ``< 10^{-9}`` abs. |
| Isothermal longwave | zero net exchange | ``< 10^{-9}`` W m⁻² |

Two limitations are worth stating with numbers.

*Angular closure.* The two-stream extinguishes diffuse light at
``1/\bar μ`` per unit leaf area, which for spherical leaves is 1. The
exact black-leaf diffuse transmittance is larger: at ``Λ = 1`` it is
0.443 against the two-stream's 0.368, and at ``Λ = 3`` 0.114 against
0.050. The two-stream therefore puts more diffuse absorption in the
upper crown than an exact angular treatment. For horizontal leaves the
two coincide, since every direction sees ``K = 1``.

*Spherical-leaf canopy albedo.* For spherical leaves with ``σ = 0.15``
(PAR) over a deep canopy the two-stream albedo is 0.0454, 0.0338, 0.0254
at ``μ = 0.3, 0.6, 1``; the widely used approximation
``ρ_{cb} = 1 - e^{-2ρ_h K/(1+K)}`` gives 0.0495, 0.0362, 0.0267, i.e. the
two differ by 5–8 %.

Not validated here: comparison with measured within-canopy radiation
profiles or with published tables of another canopy two-stream code.

## References

- CLM5.0 Technical Note (Lawrence et al., NCAR), §2.3.1, Canopy
  Radiative Transfer.
- de Pury, D. G. G. & Farquhar, G. D. (1997). Simple scaling of
  photosynthesis from leaves to canopies without the errors of big-leaf
  models. *Plant Cell Environ.*, 20, 537–557.
- Goudriaan, J. (1977). *Crop Micrometeorology: A Simulation Study.*
  Pudoc, Wageningen.
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
- Ross, J. (1981). *The Radiation Regime and Architecture of Plant
  Stands.* Junk, The Hague.
- Sellers, P. J. (1985). Canopy reflectance, photosynthesis and
  transpiration. *Int. J. Remote Sens.*, 6, 1335–1372.
- Shaw, R. H. & Schumann, U. (1992). Large-eddy simulation of turbulent
  flow above and within a forest. *Boundary-Layer Meteorol.*, 61, 47–64.
