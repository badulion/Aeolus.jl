# Two-way nesting: anterpolation, refluxing and the secondary circulation

In a one-way nest the parent never sees the child. In a two-way nest the
child's solution is fed back onto the parent once per parent step, after
the child has subcycled to the parent's time (N-nest-e, #263). This page
describes that feedback, [`TwoWayFeedback`](@ref), and what it costs in
conservation and stability. The pieces it is built from:

| piece | where | page |
| :-- | :-- | :-- |
| restriction (volume / mass measure, solid-aware) | `restrict_to_parent!` (N-nest-a #250, N-nest-b #256) | [nesting](nesting.md) |
| refluxing of a flux mismatch | `apply_flux_correction!` (N-nest-a) | [nesting](nesting.md) |
| the face flux of a scalar | `AdvectionSchemes.scalar_face_flux` (the kernel of `boundary_scalar_flux` too), the closures' `sgs_scalar_face_diffusivity` | [fractional step](fractional_step.md) |
| subcycling, stage times, retries | `NestedRun` (N-nest-c #257) | [nested time coupling](nesting_time.md) |
| the child's open faces | `NestBoundaryData` (N-nest-d #261) | [nested boundary conditions](nesting_boundaries.md) |

The maintainer's decision for the Berlin case is one-way nesting. Two-way
is implemented and tested on synthetic boxes (below); N-berlin-twoway
(#267) couples it to the Berlin run as an option, `options[:nesting].mode
= :two_way`, which is **unvalidated** in the urban setting until the
long-run stability campaign #331 (see the
[Berlin nesting page](../design/berlin_nesting.md#Two-way-option-(#267))).
Only after #331 validates it does two-way become the default for
self-nesting.

## The feedback, step by step

After the child substeps of a parent step, [`two_way_feedback!`](@ref):

1. **anterpolates** the child field set (`u, v, w, θ`, `qv`, tracers) onto
   the covered parent cells of the feedback mask;
2. replaces the parent's velocity on the interface planes of the child's
   **open** faces by the child's flux (the interface flux correction; a
   wall or periodic child face is not an interface);
3. **refluxes** `θ`, `qv` and the declared tracers from flux registers;
4. **re-projects** the parent, so ``∇·u = 0`` holds in the parent again.

At the start of the next parent step the child is projected onto its new
boundary data (the parent changed after the previous window's end data;
without this projection the child's first stage starts divergent near its
open faces, and SSP-RK3, which combines that start state into its result,
kept a third of it: ``|∇·u|`` grew to ``5\times10^{-9}`` in 200 steps of
the vortex harness).

## Anterpolation

Clark & Farley (1984) and PALM (Hellsten et al. 2021, §3.5, Eq. 15) set a
covered parent value to the mean of the child values inside it,

```math
φ_P \leftarrow ⟨φ_C⟩ = \frac{\sum_{c \in P} W_c\,φ_c\,ΔV_c}{\sum_{c \in P} W_c\,ΔV_c},
```

the conservative restriction of the [nesting page](nesting.md):
volume-weighted for the Boussinesq core (``W = 1``), reference-mass
weighted for a sound-proof pair (``W = ρ̄`` or ``ρ̄θ̄``), and solid-aware
(building cells and faces are left out; a parent cell with no fluid child
keeps its value). Velocities are area- (mass-flux-) weighted face means.
It is idempotent: a second application changes nothing.

**Relaxation** (our extension; PALM replaces directly). With a time
scale ``τ``,

```math
φ_P \leftarrow φ_P + \frac{Δt_p}{τ}\bigl(⟨φ_C⟩ - φ_P\bigr),
```

so a covered-parent deviation decays as ``(1 - Δt_p/τ)^n``, with an
e-folding time ``-Δt_p/\ln(1 - Δt_p/τ) = τ\,(1 - Δt_p/2τ + …)``.
``τ = Δt_p`` (and `τ = nothing`, the default) is direct replacement.

### Where the parent is not anterpolated

[`feedback_mask`](@ref) leaves out:

* the **buffer zone**: the outermost `buffer` parent cells next to every
  open face of the child (default 2, as in PALM). Next to its boundary the
  child's solution is still mostly its boundary data, i.e. the parent
  itself; feeding it back closes a loop through which noise at the
  interface can grow;
* the **canopy layer**: cells whose lower face is below `z_anterp`
  (Hellsten et al. 2021, §3.5). In an urban child the parent cannot
  represent the canopy the child resolves (its buildings are a few cells
  of a mostly solid layer), and the solid-aware means of nearly solid
  layers are noisy;
* the **inflow band**: `exclude_width` cells next to the child's inflow
  faces (`exclude_sides = :inflow` finds them from the sign of the child's
  face-mean normal velocity), so synthetic inflow turbulence (N-inflow-c)
  added at the child's inflow is never fed back.

Masked parent cells and faces are left bit for bit as the parent computed
them. A velocity face is anterpolated when it lies strictly inside the
covered region and both its parent cells are in the mask.

## Refluxing with flux registers

The parent cells just outside the nest were advanced with the coarse flux
through the interface, the child with the fine fluxes. The composite
integral (uncovered parent cells plus all child cells) drifts by the
difference unless it is put back (Berger & Colella 1989):

```math
δΦ = Φ_\text{coarse} - \frac{1}{r_t r_{t'}}\sum_\text{fine}Φ,
\qquad
φ_\text{outside} \mathrel{+}= \pm\frac{δΦ\,A_p}{W_p V_p}.
```

**The registers** ([`NestFluxRegisters`](@ref)) hold ``Φ = ∫F\,dt`` on the
interface planes only (both grids). Scalars the cores transport (`θ`, a
core `qv`, tracers updated inside `body_force!`) are booked by a hook that
[`NestedRun`](@ref) chains in front of each domain's `body_force!`: the
cores call it inside every stage, after the tendencies and before the
predictor, when the halos, the velocities and the SGS coefficients are
exactly those the stage used. The booked face flux is the one face kernel
`AdvectionSchemes.scalar_face_flux` (shared with `boundary_scalar_flux`,
#327), the sum of

* the advective flux ``u_n ϕ_f`` (the face reconstruction of the core's
  advection; `:centered` for `θ` and `qv`),
* the molecular flux ``-ν\,∂ϕ/∂n`` when the simulation is viscous,
* the SGS flux ``-s\,\tfrac12(K_L+K_R)\,∂ϕ/∂n`` with the coefficients of
  the closure's `sgs_scalar_face_diffusivity`, the face form of its
  `add_sgs_scalar!` (θ's diffusivity for `θ`, the tracer diffusivity for
  `qv`). A closure with a tensor scalar flux has no two-point face form
  and is refused,

times the stage weight of the core (1 for `step!`, ``\tfrac16, \tfrac16,
\tfrac23`` for the three Euler stages of `step_rk3!`, which is what their
Shu–Osher combination amounts to) and ``W`` for a sound-proof pair.
Rejected (rolled-back) steps discard their bookings. A tracer that is not
a simulation prognostic is refluxed when its owner declares its transport,
`tracer_fluxes = (:co2 => NestScalarFlux(scheme = :upwind),)`; `qv` is
declared the same way where it is a tracer of the field set (a moist
sound-proof nest) rather than a core prognostic.

**Tracers moved by a separate transport.** A transport that advects its
tracers outside the core's stages — after the dynamics, with its own
scheme, velocity and time step, like the Berlin case's forward-Euler
WENO-Z5 slot — is invisible to the hook: booking its scheme at the core's
stage state books the wrong flux. Such a tracer is declared with
`NestScalarFlux(applied_by = :transport)` and the transport books exactly
what it applies with [`book_nest_flux!`](@ref)`(sim, name, ϕ, dt; scheme,
diffusivity)`, called with the state it is about to update (a no-op for a
simulation outside a two-way run). The domain's core then wraps the
dynamics and the transport and declares the weights of its dynamics
([`nest_stage_weights`](@ref)). Measured (`test/era2_nest_twoway.jl`, 100
steps, forward-Euler WENO-Z5 slot after `step!`): composite ``∫``tracer
drift 1.2e-14 per step booked by the transport, 2.9e-7 booked at the core
stage (the pre-#327 way: right scheme, wrong velocity), 2.5e-6 not booked;
a `qv` tracer of an anelastic pair, W-weighted slot, 20 steps: ``∫ρ̄q_v``
drift 2.9e-15.

**Volume-consistent refluxing.** Each grid conserves volume, but their
time-integrated volume fluxes through one face patch differ: the child's
boundary velocity is interpolated in time over the parent step (and its
stages read it at their own times), the parent's is its stage velocity.
Refluxing the flux of ``θ`` itself books ``θ`` times that volume
mismatch: for ``θ ≈ 300`` K this heated and cooled the cells around the
nest by up to kelvins per step and drove a 4 m/s circulation within ten
minutes of the convective test below. The feedback therefore refluxes
the flux of ``ϕ - ϕ_\text{ref}``, ``ϕ_\text{ref}`` the parent's mean of
``ϕ`` over the covered cells, with the volume fluxes booked in one more
set of registers. Both velocity fields are divergence-free, so the volume
mismatches of all patches sum to zero over the closed interface: the
total correction, and the exact conservation of the composite integral,
do not depend on ``ϕ_\text{ref}``; only the spurious redistribution
``θ\,δV`` between patches is gone.

**What conservation needs.** A sponge (`NestSponge`) is a source and
breaks it. A child face that is a wall or periodic must coincide with the
parent's own wall or periodic seam: a child lid inside the parent blocks
the child but not the parent, the parent's flux through that plane has no
fine counterpart, and the run warns. The anelastic core's SGS flux of `θ`
is in the ``W``-weighted flux form (ONE-j, #460; see
[Sound-proof cores](anelastic.md)) and conserves ``∫ρ̄θ``; its molecular
term ``ν∇²θ`` is unweighted and does not. The registers book the
``W``-weighted diffusive flux, exact for the lateral planes.

## Secondary circulation

Hellsten et al. (2021, §4.1, Fig. 8) found that two-way nests of a
convective boundary layer develop a secondary circulation: a mean updraft
of 0.4–0.9 m/s over the child, subsidence around it. They exclude
differential surface heating (the surface flux is prescribed in both
domains) and attribute it to "a slightly different divergence of the
vertical heat flux between the region occupied by the child domain and the
remaining parent domain due to different grid spacing". Anterpolation is
exactly what makes the parent feel that difference, so this is a property
of two-way nesting, not a defect of one scheme.

Our two-way CBL nests show a **mean descent** over the child instead
(−0.36 m/s in the suite's single run, −0.42 m/s in the 4 h benchmark).
#327 (M5) investigated whether that is a bug or a sampling artefact, with
reduced runs (parent at 100 m, 2 h, the 2nd hour averaged, 4–6 seeds per
configuration, paired one-way/two-way runs from the same initial noise;
≈ 70 min of runs in total):

| configuration (child; parent periodic, 100 m) | two-way − one-way child-mean ``w`` at mid-depth | two-way covered − outside parent ``θ``, lower half |
| :-- | :-- | :-- |
| suite case: 1.2 km × 800 m at r = 3, top in the entrainment zone (6 seeds) | −0.163 ± 0.057 m/s (all 6 < 0) | −0.030 ± 0.007 K |
| larger child: 2.4 km × 800 m at r = 3 in a 6.4 km parent (4 seeds) | −0.091 ± 0.017 m/s | −0.038 ± 0.003 K |
| Hellsten-like: 2.4 km × 500 m (top mid-CBL), r = 2, lapse-rate start (4) | −0.084 ± 0.019 m/s | −0.023 ± 0.003 K |
| child top above the CBL: 1.2 km × 1200 m, r = 3 (4) | −0.189 ± 0.053 m/s | −0.026 ± 0.007 K |
| child top above the CBL: 2.4 km × 1200 m, r = 2 (4) | −0.058 ± 0.011 m/s | −0.031 ± 0.003 K |
| **r = 1**, suite geometry (4) | +0.034 ± 0.062 m/s | +0.009 ± 0.006 K |
| **r = 1**, Hellsten-like geometry (4) | −0.011 ± 0.011 m/s | +0.003 ± 0.002 K |
| **r = 1**, top above the CBL (4) | +0.007 ± 0.044 m/s | +0.003 ± 0.003 K |
| suite case, buffer 0 / 4 instead of 2 (4) | −0.219 ± 0.026 / −0.174 ± 0.051 m/s | −0.029 / −0.021 K |
| suite case, no refluxing (4) | −0.199 ± 0.073 m/s | −0.024 K |
| suite case, no interface flux correction (4) | identical to the default run | |

(Means ± standard errors over the seeds; the one-way child mean, which the
parent never sees, averages to zero within its error in every
configuration, e.g. −0.002 ± 0.061 m/s in the suite case.) Standalone
periodic CBLs of the suite case, which hold the same heat at every
resolution (column integrals equal to 2e-4), are **cooler below 800 m at
the child's spacing than at the parent's**: −0.080 K at 33 m against 100 m,
the mean of seeds 1 and 2 (−0.087 K and −0.072 K; the suite runs seed 1),
and −0.044 K below 500 m at 50 m against 100 m for the lapse-rate start;
the finer CBL carries more of the surface heat into its entrainment zone.

What this shows:

* the descent is **not a sampling artefact**: it is systematic across
  seeds, child sizes and child depths;
* it is **not produced by the coupling machinery**: with r = 1 (the same
  physics on both grids) there is no circulation in any geometry, and the
  buffer width (0, 2, 4), refluxing (on or off) and the interface flux
  correction (a no-op in this case: the prolongation is flux-conserving, so
  the child's interface flux already is the parent's) do not change its
  sign. The anterpolation weighting is the volume-weighted child mean of
  Clark & Farley (1984) and Hellsten et al. (2021, Eq. 15);
* it is **thermally direct** and has the sign of the resolution
  dependence: the anterpolated column takes the child's cooler lower CBL,
  is heavier than its surroundings, and sinks. A child top above the CBL
  does not remove it. This is the mechanism Hellsten et al. propose, with
  the sign set by how the model's CBL changes with grid spacing; in
  Aeolus's numerics (centred advection, Smagorinsky) a finer CBL is cooler
  below the inversion, and a mean descent over the child is the physical
  response of a two-way nest to that. The opposite sign in PALM implies its
  finer child was warmer; we have not run PALM's numerics to confirm this.

So the two-way coupling is conservative, stable and free of a
coupling-made circulation, but the secondary circulation of a convective
two-way nest is real and set-up dependent (≈ 0.05–0.2 m/s of descent here,
growing with the resolution contrast and shrinking with the child's
size). Two-way nesting is **not validated against** Hellsten et al.'s
circulation; it reproduces their mechanism, not their sign. This is one
reason the Berlin nest of the urban site stays one-way until its
validation (#331), although two-way is the self-nesting default (ONE-f,
#455). `test/era2_nest_twoway_longrun.jl`
asserts that the suite run's circulation is thermally direct and that the
standalone CBL is cooler at the child's spacing.

## Measured behaviour

From `test/era2_nest_twoway.jl` and `test/era2_nest_twoway_longrun.jl`.

| property | result | bound |
| :-- | :-- | :-- |
| composite ``∫θ``, ``∫q_v``, ``∫``tracer, 2000 parent steps (Smagorinsky, ``ν > 0``): drift per step | 1.3e-14, 1.9e-14, 1.5e-14 | ≤ 1e-12 |
| the same after 2000 steps | 5.4e-15, 3.6e-15, 6.4e-15 | ≤ 1e-10 |
| without refluxing, per step | 7.5e-5, 7.0e-5, 3.7e-6 | > 1e4 × the above |
| `step_rk3!` cores, 100 steps | 5.7e-16 (θ), 1.3e-14 (qv) | ≤ 1e-11 |
| anelastic pair, composite ``∫ρ̄θ``, 20 steps | 1.1e-15 | ≤ 1e-12 |
| relaxation ``τ = 100Δt``: ``\max|d_n - (1-Δt/τ)^n|`` | 4.4e-16 | < 1e-12 |
| fitted e-folding / ``τ`` | 0.995 | within 1 % |
| masked cells (buffer, ``z < z_\text{anterp}``, inflow band) | bit-identical to one-way | |
| feedback applied twice (buffer 2, no re-projection / buffer 0, re-projection) | 0 / 1.7e-16 | round-off |
| child-only 50 MW heat source: one-way parent vs standalone | bit-identical | |
| the same, two-way: parent θ excess vs ``Qt/ρc_p`` − child content | 6.4e-11 | < 1e-10 |

### Long runs (`test/era2_nest_twoway_longrun.jl`, ≈ 240 s on an idle core)

| case | measured |
| :-- | :-- |
| CBL (parent 3.2 × 3.2 × 1.6 km at 100 m, child 1.2 × 1.2 km × 800 m at r = 3, 0.1 K m/s, 2 h), two-way | 744 parent steps, 4598 child substeps, no retries, finite |
| two-way / one-way KE, quarter-hours of the 2nd hour | 0.871, 1.047, 0.951, 1.061 (bound ±20 %) |
| largest child divergence, every step | 8.7e-17 s⁻¹ |
| composite θ budget (after 30 min) | 9.8e-14 (one-way, which does not conserve it: 1.0e-2) |
| 2Δ θ mode on the parent rows crossing the nest | 4.7e-3 K → 4.2e-3 K over the 2nd hour, growth −0.064 ± 0.031 h⁻¹ |
| child-mean ``w`` at 400 m, 2nd hour | one-way −0.164 m/s, two-way −0.358 m/s (one realisation; see the ensemble above) |
| covered − outside parent ``θ`` below 400 m, 2nd hour | one-way −0.017 K, two-way −0.063 K (thermally direct: asserted) |
| standalone CBL (1.6 km box), mean ``θ`` below 800 m, 33 m − 100 m | −0.087 K at equal column heat, seed 1 (asserted < −0.03 K) |
| forced neutral flow (Kolmogorov, 16³ parent, r = 2), 61 eddy turnovers | KE of turnovers −40…−20 vs −20…0: ratio 1.011 |
| outgoing Lamb–Oseen vortex harness (N-open-c), two-way, 2000 steps | child kinetic energy within 1.2e-6 of a standalone parent |

### The 4 h benchmark at 50 m (`benchmarks/bench_twoway_cbl.jl`)

Parent 64 × 64 × 32 at 50 m, child 72 × 72 × 48 at 16.7 m, 4 h, one core:

| | one-way | two-way |
| :-- | :-- | :-- |
| parent steps / child substeps (substeps per step) | 4166 / 25583 (2–16) | 4153 / 25383 (2–10) |
| retries, non-finite values | 0, none | 0, none |
| wall time | 854 s | 989 s |
| largest child divergence | 4.5e-16 s⁻¹ | 4.5e-16 s⁻¹ |
| composite θ budget error | 4.7e-3 (not conserved one-way) | 3.1e-13 |
| KE mean over hour 4 / hour 3 | 1.217 | 1.116 |
| monotone KE growth over the last hour (12 five-minute means) | no | no |
| child-mean ``w`` at 400 m over hour 4 | −0.247 m/s | −0.420 m/s |
| 2Δ θ mode on the rows crossing the nest, growth over hours 3–4 | −0.123 ± 0.007 h⁻¹ | −0.019 ± 0.006 h⁻¹ |

The two-way run meets "last hour within 20 % of the hour before"; the
one-way run, whose CBL still deepens, does not (1.217).

The child-mean ``w`` of a single run is dominated by where the few
largest thermals happen to be (the child covers about one convective
cell): the one-way runs show −0.16 m/s (2nd hour, 100 m) and −0.25 m/s
(4th hour, 50 m) although nothing feeds back. The ensemble of the previous
section separates the two: the one-way mean is zero within its error, the
two-way descent is systematic and comes from the resolution dependence of
the CBL.

Two expectations of the issue did not hold as stated, for physical
reasons, and the tests say so: the kinetic energy of a CBL deepening under
its inversion grows by 30–50 % in its 2nd hour (``KE ∝ z_i^{5/3}``, one-way
as well), so the suite compares two-way with one-way per quarter-hour and
the benchmark applies "last hour within 20 % of the hour before" to hours
3 and 4; and 10-turnover means of the small neutral box differ by up to
25 % between windows (one-way as well), so the neutral criterion uses
20-turnover windows.

## Mixed cores

A Boussinesq domain and an anelastic one may be nested either way round,
one-way or two-way (NEST-p2, #482;
[`NestCoreConversion`](@ref Aeolus.Nesting.NestCoreConversion)). The two
cores conserve different things: the Boussinesq core the volume flux
(``\nabla\cdot\mathbf{u} = 0``, its mass ``ρ_0 V`` with the constant
``ρ_0 = p_{\rm ref}/(R_d θ_0)``), the anelastic core the mass flux
(``\nabla\cdot(\bar ρ\mathbf{u}) = 0``). Across the interface the
conserved quantity is the mass flux, and a scalar's flux is that mass flux
times the (intensive) scalar. The nest therefore measures each side with
its own density ``W`` (``ρ_0`` or ``\bar ρ(z)``), so that an intensive
scalar is prolongated and restricted as a mass-weighted mean and a
constant stays that constant, and converts the velocity. A covered parent
layer ``K`` of mass ``W_{p,K}\,Δz_K`` with child mass
``M_{c,K} = \sum W_c Δz_c`` sends the child

```math
\sum_{c \in K} W_c\, u_c\, Δz_c = W_{p,K}\, u_K\, Δz_K ,
```

the parent velocity scaled by ``r_K = W_{p,K} Δz_K / M_{c,K}`` before the
conservative prolongation (``r_f = W_{p,f}/W_{c,f}`` at a horizontal face
for ``w``), and takes back ``u_K = \sum W_c u_c Δz_c / (W_{p,K} Δz_K)`` in
the anterpolation and the interface flux correction (the inverse scaling
around the feedback; a face the feedback leaves alone keeps its value bit
for bit). The flux registers of both sides book mass fluxes (the
Boussinesq side's weighted by ``ρ_0``), so the refluxing conserves the
composite ``\sum W θ V``.

Measured (`test/model_nesting.jl`; a 3.2 km deep parent, 16³ at
Δ = 200 m, with a 1.6 km deep child at ``r = 2``, where ``\bar ρ/ρ_0``
differs from 1 by up to 12 % (anelastic parent) and 14 % (anelastic
child)): the mass flux the boundary data carry through the west plane
matches the parent's per face patch to ``7\times10^{-16}`` (relative);
two-way, the composite ``\sum Wθ V`` changes by ``1.0\times10^{-14}``
and ``8.0\times10^{-15}`` over six windows and the interface mass-flux
imbalance after the correction is 0. An anelastic child steps with the
Euler step (the anelastic SSP-RK3 step takes no open faces).

## Current limits

* Refluxing needs a core with known stage weights in both domains:
  `step!`, `step_rk3!`, or a user core that declares them
  ([`nest_stage_weights`](@ref)). The IMEX cores (implicit `θ` diffusion)
  are refused with an `ArgumentError` and run two-way with
  `TwoWayFeedback(reflux = false)`.
* Two-way feedback needs parent and child of one core family (two
  Boussinesq sims, or two sound-proof sims of one system sharing one
  reference profile). A mixed pair has no consistent measure and is
  refused by every two-way entry point (`NestedRun`, `anterpolate!`,
  `restrict_state_to_parent!`); it nests one-way.
* Two-way feedback needs a conservation box on every axis (#328): at
  least two parent faces inside the child must be child faces; a child
  without one nests one-way. In a [`NestHierarchy`](@ref) the children of
  a domain feed back after each of its steps (post order, the domain
  re-projected once after all of them); see
  [nest hierarchies, non-aligned and stretched children](nesting_hierarchy.md).
* With a buffer zone the re-projection is not idempotent: the anterpolated
  faces next to the buffer leave the parent divergent, and the projection
  moves them (and the interface faces: the ledger's
  `:mass_after_projection`). The next step's child boundary data take the
  projected parent.

## References

* Berger, M. J. & Colella, P. (1989). Local adaptive mesh refinement for
  shock hydrodynamics. *J. Comput. Phys.* 82, 64–84.
* Clark, T. L. & Farley, R. D. (1984). Severe downslope windstorm
  calculations in two and three spatial dimensions using anelastic
  interactive grid nesting. *J. Atmos. Sci.* 41, 329–350.
* Hellsten, A. et al. (2021). A nested multi-scale system implemented in
  the large-eddy simulation model PALM model system 6.0. *Geosci. Model
  Dev.* 14, 3185–3214.
