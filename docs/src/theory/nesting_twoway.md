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
| the face flux of a scalar | `AdvectionSchemes.advective_face_flux`, the closures' `sgs_scalar_face_diffusivity` | [fractional step](fractional_step.md) |
| subcycling, stage times, retries | `NestedRun` (N-nest-c #257) | [nested time coupling](nesting_time.md) |
| the child's open faces | `NestBoundaryData` (N-nest-d #261) | [nested boundary conditions](nesting_boundaries.md) |

The maintainer's decision for the Berlin case is one-way nesting; two-way
is fully implemented and tested for long-run stability, and N-berlin-twoway
(#267) couples it to the Berlin run as an option.

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
interface planes only (both grids). They are booked by a hook that
[`NestedRun`](@ref) chains in front of each domain's `body_force!`: the
cores call it inside every stage, after the tendencies and before the
predictor, when the halos, the velocities and the SGS coefficients are
exactly those the stage used. The booked face flux is the sum of

* the advective flux `advective_face_flux` (``u_n ϕ_f``, the face
  reconstruction of the core's advection; `:centered` for `θ` and `qv`),
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
`tracer_fluxes = (:co2 => NestScalarFlux(scheme = :upwind),)`.

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
fine counterpart, and the run warns. The anelastic core diffuses `θ` with
the unweighted Laplacian, which does not conserve ``∫ρ̄θ`` by itself; the
registers book the ``W``-weighted diffusive flux, exact for the lateral
planes.

## Secondary circulation

Hellsten et al. (2021, §4.1, Fig. 8) found that two-way nests of a
convective boundary layer develop a spurious secondary circulation: the
better-resolved child transports heat differently from the parent, the
anterpolated parent region becomes a heat island or sink for the parent,
and a mean updraft of 0.4–0.9 m/s forms over the child. Anterpolation is
exactly what makes the parent feel the child's different mixing, so this
is a property of two-way nesting, not a defect of one scheme; the buffer
zone and the canopy restriction weaken it. This is one reason the Berlin
default is one-way.

The suite measures the child-mean ``w`` at 400 m in the scaled CBL for
one-way and two-way (a measurement, not a target; see below).

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

### Long runs (`test/era2_nest_twoway_longrun.jl`, ≈ 270 s)

| case | measured |
| :-- | :-- |
| CBL (parent 3.2 × 3.2 × 1.6 km at 100 m, child 1.2 × 1.2 km × 800 m at r = 3, 0.1 K m/s, 2 h), two-way | 744 parent steps, 4598 child substeps, no retries, finite |
| two-way / one-way KE, quarter-hours of the 2nd hour | 0.871, 1.047, 0.951, 1.061 (bound ±20 %) |
| largest child divergence, every step | 8.7e-17 s⁻¹ |
| composite θ budget (after 30 min) | 9.8e-14 (one-way, which does not conserve it: 1.0e-2) |
| 2Δ θ mode on the parent rows crossing the nest | 4.7e-3 K → 4.2e-3 K over the 2nd hour, growth −0.064 ± 0.031 h⁻¹ |
| child-mean ``w`` at 400 m, 2nd hour | one-way −0.164 m/s, two-way −0.358 m/s |
| forced neutral flow (Kolmogorov, 16³ parent, r = 2), 61 eddy turnovers | KE of turnovers −40…−20 vs −20…0: ratio 1.011 |
| outgoing Lamb–Oseen vortex harness (N-open-c), two-way, 2000 steps | child kinetic energy within 1.2e-6 of a standalone parent |

The child-mean ``w`` of a single run is dominated by where the few
largest thermals happen to be: the child covers about one convective
cell (1.2 km against ``1.5 z_i ≈ 1.2`` km), and the one-way run, which has
no secondary circulation, already shows −0.16 m/s. A two-way mean of
−0.36 m/s therefore does not establish a circulation of either sign; the
4 h benchmark at 50 m (`benchmarks/bench_twoway_cbl.jl`) measures it over
the last hour.

Two expectations of the issue did not hold as stated, for physical
reasons, and the tests say so: the kinetic energy of a CBL deepening under
its inversion grows by 30–50 % in its 2nd hour (``KE ∝ z_i^{5/3}``, one-way
as well), so the suite compares two-way with one-way per quarter-hour and
the benchmark applies "last hour within 20 % of the hour before" to hours
3 and 4; and 10-turnover means of the small neutral box differ by up to
25 % between windows (one-way as well), so the neutral criterion uses
20-turnover windows.

## Current limits

* Refluxing needs a core with known stage weights (`step!`, `step_rk3!`)
  in both domains; the IMEX cores and user cores run two-way with
  `TwoWayFeedback(reflux = false)`.
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
