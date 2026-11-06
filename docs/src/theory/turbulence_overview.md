# Turbulence closure: overview

The turbulence-closure work (Phases 54–59) rebuilt the subgrid-scale (SGS) model, the walls
and the stable boundary layer around one closure contract. This page
covers four things: that contract, how the closures, wall models and
stability functions fit together, what the Berlin flagship runs by
default, and the measured validation. The physics of each piece is on
its own page (list at the end). The numbers quoted here are taken from
the merged PRs (#203–#231, #240–#244, #271) and the
[roadmap](../design/roadmap.md).

## One contract for every closure

Every closure is an `AbstractSGSClosure` of `Operators`
([closure contract](sgs.md)). A closure computes ν_t and κ_t from the
current state with `compute_sgs!`. That call is pure, so substages, CFL
estimates and rejected tries can call it freely. `advance_sgs!` then
commits the closure's memory once per accepted step, and
`sgs_state` / `restore_sgs_state!` let `step_adaptive!` roll back a
rejected try bit for bit. All five cores take the closure through one
`sgs` keyword: Euler `step!`, `step_rk3!`, `step_imex!`, `step_adaptive!`
and the anelastic and pseudo-incompressible cores. The CFL limit uses
max(ν_t, κ_t). The contract also carries:

* **The momentum form** (`sgs_stress = :laplacian | :symmetric`). The
  stress form ∂_j(2ν_ij S_ij) takes separate horizontal and vertical
  viscosities. It gives exactly zero tendency for solid-body rotation
  and dissipates −Σ 2ν_ij S_ij S_ij to round-off. Its explicit limit is
  8νΣΔ⁻², twice the Laplacian's.
* **The scalar form**: a diagonal (κ_h, κ_v) operator, a general
  symmetric K_ij in triad form, or the rotated isentropic tensor. qv and
  every transported tracer are diffused with the closure's κ_t
  (`sgs_tracer_diffusivity!`).
* **Traits**: `sgs_explicit_stress`, so that `step_imex!` refuses a
  closure it cannot treat implicitly, and `laminar_consistent`.

Wrappers are closures too. `StabilityCorrected(base)` makes any eddy-
viscosity closure depend on Ri_g and anisotropic. `IsentropicDiffusion`
rotates its scalar tensor, and `SolidMaskedSGS` zeroes every coefficient
inside buildings. A wrapper with neutral settings reproduces its base
bit for bit.

## The closures

| closure | type | laminar flow | phase | Berlin |
| :-- | :-- | :-- | :-- | :-- |
| static Smagorinsky | algebraic | ν_t > 0 in shear | LES core | `:smagorinsky` option |
| plane / Lagrangian / scale-dependent (LASD) dynamic | dynamic Germano | decays to 0 | 54 | `:dynamic` option, LASD default |
| AMD (scalar and buoyancy terms), Vreman | algebraic | 0 for shear (AMD also rotation) | 55 | `:amd`, `:vreman` options |
| `StabilityCorrected` | wrapper, Ri_g | — | 55 | on for every eddy-viscosity provider |
| Deardorff / Moeng TKE | prognostic e | decays | 56 | **`:tke`, default** |
| differential Reynolds stress | prognostic τ_ij | decays | 56 | `:reynolds_stress` option, not with IMEX |
| σ-model | algebraic | 0 for shear, rotation, 2-D, z³ at walls | 59 | `:sigma` option |
| `MLSGS` | learned | — | LES core | `:mlsgs` option |

The stability functions f_m(Ri) and f_h(Ri) are defined once, in
`TurbulenceStability` ([theory](turbulence_stability.md)). The
`StabilityCorrected` wrapper, the fog column and the core N² profiles all
call it. Its `MOSTConsistent` family inverts Ri(ζ) = ζφ_h/φ_m², so a
closure and the surface layer agree on the same stability.

## Walls

`WallModels` ([theory](wall_models.md)) applies a wall law implicitly at
the end-of-step velocity, so it conserves momentum and stays stable far
above the explicit limit. The laws are Spalding, a rough log law, MOST,
and MOST with a prescribed flux. MOST takes κ and the stability
functions from `SurfaceExchange`. The input can be sampled at a matching
height and filtered in time and in space, which removes most of the
log-layer mismatch. The same operator acts on z-walls, on immersed
(mask) walls, with the stencil flux removed so nothing is counted twice,
and on cut-cell walls through an image-point probe. Two further models
are optional: a non-equilibrium TBLE column with a pressure gradient and
the Wang & Moin dynamic κ, and the Anderson & Meneveau dynamic z0 for
partially resolved roughness. A resolved `NoSlipWall` and moving walls
cover laminar and wall-resolved flow ([transition](transition.md)).

## Stable boundary layers and transition

`LargeScaleForcing` ([theory](large_scale_forcing.md)) provides the
geostrophic pressure gradient, nudging, advective tendencies, a surface-
temperature series and subsidence for every case driver. GABLS1–4 and CASES-99 are configurations of the model (its
`LargeScale`, `SurfaceLayer`, retargeted `Sponge` and `Towers` components;
the `sbl` run type until ONE-g, #456)
([theory](stable_boundary_layer.md)). The maximum sustainable heat flux
(Van de Wiel et al. 2012) in `SurfaceExchange` predicts when the
surface layer collapses. The `Transition` module measures the turbulent
fraction γ, intermittency, Ri_f, Re_b, growth rates and the
relaminarisation criteria ([theory](transition.md)).

## Measured validation

Against external references (the analytic and conservation checks of
each phase are on its page):

| phase | check | result | reference |
| :-- | :-- | :-- | :-- |
| 54 | CBC decay at 32³, dynamic: K98/K42, K171/K42 | +13.2 %, −3.6 % | Comte-Bellot & Corrsin (1971) |
| 54 | half channel, Φ_m over 0.05–0.3 H, LASD, no wall damping | 0.934–1.151 | 1 (log law); Smagorinsky 0.965–1.589 |
| 55 | GABLS1 at 64³, h / u* / w'θ'₀ | 196.4 m / 0.284 m s⁻¹ / −0.0134 K m s⁻¹ | ≈200 / 0.245–0.283 / −0.0148 to −0.0094 |
| 55 | stratified channel Re_τ 180, Ri_τ 18: U_b⁺, Nu ratio | within 1.4–2.7 %; 0.63–0.67 | 16.91; 0.667 (García-Villalba & del Álamo) |
| 55 | CBC K171/K42 with AMD | −4.7 % | first closure within ±10 % |
| 56 | TKE CBL at 64³, σ_w² peak | 0.459 w*² at 0.351 z_i | 0.466 at 0.313 (Lenschow et al. 1980) |
| 56 | a priori at 256³ (Re_λ 78.6), Smagorinsky ρ_Π(4h) | 0.832 | 0.4–0.85 (literature band) |
| 56 | rotating decay, resolved-KE error: RSM / Smagorinsky | +6.4 % / −11.9 % | filtered DNS |
| 56 | surface-mounted cube, x_R/H: Smagorinsky / TKE / RSM | 1.82 / 1.80 / 1.97 | 1.612 (Martinuzzi & Tropea 1993) |
| 57 | channel U_b⁺ error, Re_τ 182 / 544 / 1000 / 5186 | −1.24 / −1.19 / −1.40 / −1.45 % | Lee & Moser (2015) |
| 57 | non-equilibrium τ_w after dp/dx tripled, half-way time | 5.965 δ/u_τ (equilibrium 6.255) | 5.960 (resolved) |
| 58 | GABLS3 low-level jet at 6.25 m | 7.70 m s⁻¹ at 126.8 m | 7.8–9.3 m s⁻¹ at 100–130 m |
| 58 | GABLS4 exp. 3 at 2 m: u* / H / h | 0.092 / −11.0 / 30.7 m | 0.08–0.12 / −13.5 to −5.3 / 14.5–19.5 m (six 1 m LES) |
| 58 | GABLS2 at 30 h, corrected: H / u* / h | −23.0 W m⁻² / 0.192 / 100 m | −40 to −10 / 0.19–0.25 / 80–220 m (SCMs) |
| 58 | CASES-99 hockey stick at 30 m, V_s | 7.56 m s⁻¹ | 6.5 m s⁻¹ (Sun et al. 2012) |
| 58 | collapse LES, first-level wind | 1.253–1.257 U₀ | 2^{1/3} = 1.260 (Van de Wiel et al. 2012) |
| 59 | Kelvin–Helmholtz growth rate | 0.1864 U/δ | 0.1897 (Michalke) |
| 59 | T3A-like plate: onset Re_θ, downstream c_f | 198; 1.06–1.14 × correlation | 190–200 (Mayle; Abu-Ghannam & Shaw) |
| 59 | cooled open channel at Re_τ 360, critical h/L | ≈ 0.474 (0.45–0.50) | 0.492 (Nieuwstadt 2005, κ in L) |

**Misses, stated in each phase**:

* GABLS4 h, L and θ*.
* The CASES-99 shape at 10 m and 50 m.
* The stratified channel at Ri_τ = 60, which relaminarises.
* Collapse at Re_τ 180, where survival at h/L 0.3 fails.
* The k⁻⁵ᐟ³ range of E11.
* K171 within ±10 % for every closure but AMD.
* The cube's recirculation front x_F.

Most of these are resolution-limited; the roadmap gives the evidence
for each. The SBL runs are dry.

## In the 3-D flagship

The Berlin case runs the TKE closure by default, with the stress-form
divergence and κ_t diffusion of qv and every tracer. Every
eddy-viscosity provider is wrapped in `StabilityCorrected`. The ground
and facade wall models are on, and TKE production at the facades
matches the facade stress. The `:geostrophic` coupling uses the
geostrophic pressure gradient. `turbulence_diag` writes N², Ri_g, f_m,
γ and the relaminarised fraction, and the surface exchange writes
|H|/H_max for every column ([the Berlin case](../design/berlin_case.md)).

The Berlin night is not yet physical (#238). Its constant background
ν = 2 m²/s is about 100× the night-time SGS κ, so it hides most of the
closure differences measured above.

## Detailed pages

* [LES filtering and closure](les_filtering.md), including the a-priori
  method and the filtered-DNS fixture
* [SGS closures](sgs.md): contract, Smagorinsky, dynamic, LASD, AMD,
  Vreman, σ, stress form, tensor diffusion, stability correction
* [Prognostic SGS TKE and Reynolds stress](sgs_tke.md)
* [Stability functions and Ri_g](turbulence_stability.md)
* [Wall models](wall_models.md)
* [Large-scale forcing](large_scale_forcing.md)
* [Stable BLs: GABLS1–4, CASES-99](stable_boundary_layer.md)
* [Transition and wall-bounded flow](transition.md)
