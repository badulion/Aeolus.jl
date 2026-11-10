# Prognostic SGS TKE (Deardorff–Moeng) and Reynolds stress

Phase 56 (P56-b, #182). The 1.5-order closure of Deardorff (1980)
carries the subgrid-scale turbulent kinetic energy ``e`` as a prognostic
variable; the eddy viscosity follows from it, not from the resolved strain
alone. Aeolus implements it with the coefficients of Moeng (1984) as
[`TKEClosure`](@ref Aeolus.PrognosticSGS.TKEClosure) in the module
`Aeolus.PrognosticSGS` ([API](../api/turbulence/prognostic_sgs.md)), an
`AbstractSGSClosure` of the [closure contract](sgs.md#Closure-contract).

## The model

```math
\frac{∂e}{∂t} + \mathbf u·∇e = P_s + P_b − ε + ∇·(2ν_t∇e),
```

```math
ν_t = c_k\, l\sqrt e, \qquad κ_t = \Big(1 + \frac{2l}{Δ}\Big)ν_t, \qquad
ε = c_ε\frac{e^{3/2}}{l}, \qquad c_ε = 0.19 + 0.51\,\frac{l}{Δ},
```

```math
l = \min\Big(Δ,\; 0.76\,\frac{\sqrt e}{N},\; κ\,d_{wall}\Big)
```

with ``c_k = 0.1``, ``Δ = (Δx\,Δy\,Δz)^{1/3}``, the stability limit only
for ``N^2 > 0``, the wall cap ``κ d_{wall}`` (``κ = 0.4``, ``d_{wall}`` from
[`IBM.wall_distance`](@ref Aeolus.IBM.wall_distance)) when a wall distance
is given, and ``c_ε`` multiplied by 3.9 at the first level above a surface
(Moeng 1984; [`IBM.first_level_cells`](@ref Aeolus.IBM.first_level_cells)).
``N^2`` is the one of the stability contract,
`Operators.buoyancy_frequency_squared!` (P58-a, #170), with the virtual
potential temperature when the case is moist. Given a condensate field
(`qc`, with the pressure `p`; #239) it is the ``N^2`` of cloudy air:
condensate loading, and in saturated cells the saturated ``N^2`` of a
moist-adiabatic parcel (Durran & Klemp 1982). That is Deardorff's (1980)
stratocumulus buoyancy, so a moist-neutral fog or cloud layer has
``l = Δ`` instead of the stable-layer ``0.76\sqrt e/N``
([Turbulence stability](turbulence_stability.md#Saturated-air)).

Two limits fix the constants' meaning:

* **Neutral equilibrium** (``P_s = ε``, ``l = Δ``):
  ``e = (c_k/c_ε)Δ^2 S^2``, so ``ν_t = (C_sΔ)^2 S`` with
  ``C_s = c_k^{3/4}c_ε^{-1/4} = 0.1944`` — a Smagorinsky closure with
  ``C_s = 0.194``, about 14 % above the 0.17 of the static default.
* **Critical Richardson number.** In stable shear, with ``l = 0.76\sqrt e/N``
  below ``Δ``, ``P_s + P_b = ε`` divided by ``c_k l\sqrt e`` reads
  ``S^2 − (1 + 2x)N^2 = (0.19 + 0.51x)N^2/(c_k 0.76^2)`` with ``x = l/Δ``: it
  is linear in ``x``, so the equilibrium is explicit, and it has a root
  ``x > 0`` only for ``Ri = N^2/S^2 < Ri_c = 1/(1 + 0.19/(c_k\,0.76^2)) = 0.233``
  ([`tke_critical_richardson`](@ref Aeolus.PrognosticSGS.tke_critical_richardson)).
  Above it the equilibrium TKE is zero: in equilibrium the closure
  switches itself off without a stability function. In a transient
  ``ν_t`` stays positive while ``e`` decays.

**No stability function on top.** The ``0.76\sqrt e/N`` length is the
closure's own stability response, so no `TurbulenceStability` factor
``f_m(Ri)`` is applied to it, and the `StabilityCorrected` wrapper of
P55-e (#187), which multiplies a Smagorinsky-type ``ν_t`` by such a
factor, must not wrap a `TKEClosure`.

**No horizontal–vertical anisotropy.** The closure has one filter width
``Δ = (ΔxΔyΔz)^{1/3}`` and one ``ν_t``, ``κ_t`` in every direction, as in
Deardorff (1980) and PALM; the ``κ_h ≠ κ_v`` length scales
``Δ_h = \sqrt{ΔxΔy}``, ``Δ_v = Δz`` of `StabilityCorrected` are by design
not part of it. On the Berlin grid (``Δx/Δz = 2.5``, ``ℓ_h = 1.842``,
``ℓ_v = 0.2947``) the neutral equilibrium of `TKEClosure` therefore mixes
``1/ℓ_v = 3.39×`` more vertically and ``ℓ_h = 1.84×`` less horizontally
than a `StabilityCorrected` Smagorinsky closure of the same ``C_s = 0.194``
(``(0.1944/C_s)^2`` more again for a smaller ``C_s``: 4.44× and 1.41× less
for 0.17, 12.8× and 2.05× *more* for the Berlin `cfg.C_s = 0.1`;
`tke_equilibrium`, ``S = 0.1`` s⁻¹, ``N = 0``). The rotated scalar tensor
of `IsentropicDiffusion` is not offered for a `TKEClosure` either; both
are possible follow-ups.

## Discretization

**Coefficients** (`sgs_coefficients!`, pure) are cell-local: ``l``, ``ν_t``
and ``κ_t`` from the committed ``e`` and the current ``N^2``.
[`tke_length_scale`](@ref Aeolus.PrognosticSGS.tke_length_scale) also
returns ``s = \sqrt e/l``, the rate every sink is written with; it stays
finite as ``e → 0`` in stable air (``s → N/0.76``), where ``l → 0``.

**Shear production — the dual of the momentum operator.** The SGS
momentum tendency of the contract is the Laplacian flux form
``∂_j(ν_t ∂_j u_i)`` of `add_eddy_viscosity_u!` (and `_v!`, `_w!`): normal
fluxes with the cell ``ν_t``, cross fluxes on the cell edges with the
average of the four surrounding cells (weights 0.25). Its kinetic-energy
sink ``Σ \mathbf u·\mathbf T_{sgs} V`` is rearranged, flux by flux, into
a sum over cells: every flux ``f_L = ν_L δu_L/Δ`` contributes
``−u_p(±f_L)/Δ`` at each velocity point ``p`` it enters, and that is
returned to the cells of ``ν_L`` with the same weights.
[`dual_strain!`](@ref Aeolus.PrognosticSGS.dual_strain!) builds the
velocity factor ``D``, so that ``P_s = ν_t D`` and

```math
\sum_c P_s V = −\sum \mathbf u·\mathbf T_{sgs}(\mathbf u, ν_t)\, V
```

exactly, for any ``ν_t``. In the interior each flux contributes
``(δu/Δ)^2``, so ``D ≥ 0`` (for ``u = Sz``, ``D = S^2``); the shares of
halo cells are folded onto the interior cells they copy (periodic image,
or mirror at a wall), which is exact because a halo ``ν_t`` equals its
image. Each cell computes its own ``D`` (KA-l, #304): it gathers the
shares destined for it, and those of the halo cells it is the image of,
from the velocity points around it, so the kernel runs on any number of
threads with the same bits. ``D`` is built from the face and edge differences of the
momentum kernels, not from `Operators._velocity_gradient_at`: that
tensor's off-diagonal entries are centred over two cells, and their
square is not the dual of the edge-flux operator.

**Shear production at wall-modelled walls (#232).** Where a wall model
replaces the stencil's flux through an immersed wall
(`WallModels.remove_stencil_wall_flux!` takes
``ν_{t,e}(q_N − q_F)/Δ_d^2`` out of the tendency of a wall face ``F``,
``N`` the IBM-zeroed neighbour, and `apply_immersed_wall_stress!`
applies ``τ = a Δ_d q_F``), the dual above would still book the removed
stencil share ``ν_{t,e} q_F(q_F − q_N)/Δ_d^2`` as production — energy
the resolved flow never lost — and none of the work of the wall stress.
With the walls attached
([`attach_immersed_walls!`](@ref Aeolus.PrognosticSGS.attach_immersed_walls!))
that share is taken out of ``D`` at the four cells around the edge, as
`dual_strain!` put it there, and the wall stress times the resolved
shear across the wall, ``τ\,q_F/Δ_d = a q_F^2`` per unit volume (``a``
from `WallModels.immersed_wall_drag!`, the one evaluation of the law), is
added, half to each cell the face separates:

```math
\sum_c P_s V = −\sum \mathbf u·\big(\mathbf T_{sgs} − \mathbf T_{sgs}^{wall}\big) V
             + \sum_{\text{wall faces}} a\, q_F^2\, ΔV ,
```

the kinetic energy the SGS operator the step applies and the wall stress
take out of the resolved flow. Booking all of the wall stress's work as
production is the constant-stress log layer: its shear production
integrated from ``z_0`` to the first face, ``∫ τ_w\,∂U/∂n\,dn = τ_w U``,
is the work of the wall stress at the face velocity. The test
(`test/tke_facades.jl` (removed)) checks the budget to 0 relative error, the
removed share to 4.6e-15, the wall work against the energy
`apply_immersed_wall_stress!` removes to 4.7e-8 (``δt = 10^{-6}``), and,
for uniform flow along a wall, ``Σ P_s V/A = u_*^3 \ln(h/z_0)/κ`` to
round-off.

**Buoyancy production — the dual of the κ_t operator.**
``P_b = −κ_t N^2``. On the uniform grid the ``N^2`` of
`buoyancy_frequency_squared!` is the average of the two face gradients
around the cell, which is how the three-argument scalar kernel averages
``κ_t`` to the faces, so ``Σ P_b V = (g/θ_0) Σ z\,T_θ V``: the potential
energy the SGS heat flux moves. A prescribed surface heat flux
(`θ_flux_bottom`, `θ_flux_top` of the Boussinesq core) sits on the
boundary face and contributes ``(g/θ_0)F/2`` to the adjacent cell, as the
interior faces do.

**Transport.** ``−∇·(\mathbf u e)`` with the flux-limited
`AdvectionSchemes.advect_positive!` (WENO-Z5 when the halo is ≥ 3, else
the limited centred flux) and ``∇·(2ν_t∇e)`` with the three-argument
`add_eddy_diffusion_scalar!` (solid-masked when a mask is given). The
Berlin case transports ``e`` in its `:transport` stage instead
(`transport = false`).

**Time integration** (`advance_sgs!`, once per accepted step after the
projection; once per accepted piece of an adaptive step). With the
coefficients of ``e^n`` and the new state's ``N^2``:

```math
e^* = e^n + Δt\,\big(T + P_s + \max(P_b, 0)\big), \qquad
e^{n+1} = \frac{e^*}{1 + Δt\,(c_ε s^n + r_b^n)},
```

with ``r_b = κ_t N^2/e = (1 + 2l/Δ)c_k N^2/s`` the buoyant destruction
rate (``N^2 > 0``). Both sinks are linear in ``e^{n+1}`` (a Patankar
linearisation), so ``e^{n+1} ≥ 0`` whenever ``e^* ≥ 0`` and its fixed point
is the exact equilibrium ``P = ε``. The issue proposed ``\sqrt{e^*}`` in
the denominator; that form's fixed point is biased by ``O(Δt)`` (the
equilibrium tests to 1e-6 could not pass with it at a usable step), so the
rate is lagged at ``e^n``, identical in pure decay. `dissipation = :exact`
integrates ``de/dt = −c_ε e^{3/2}/l`` exactly at frozen ``l``. If transport
makes ``e^*`` negative it is set to zero and booked
([`tke_statistics`](@ref Aeolus.PrognosticSGS.tke_statistics): `clips`,
`clipped_energy`); none of the tests books a clip. ``e = 0`` in solids.
Because ``e`` is the closure's memory, `sgs_state` / `restore_sgs_state!`
make a step `step_adaptive!` rejects exact.

**One cache-blocked pass** (#554). On the CPU the part of the commit that
reads ``e`` (``ν_t``, the flux-limited advection, the ``2ν_t`` diffusion,
the update and its budget sums) runs row by row on the blocked driver of
the fused sweep (`FiniteVolume.blocked_update!`): a pipeline two rows
ahead keeps ``ν_t``, the face fluxes and the limiter's ratio in small
rings, and the new ``e`` is written back once no later row reads the old
one. Every value is computed by the point body of the separate loop, so
``e`` and the budget sums agree with the separate loops to round-off
(measured identical). ``N^2`` and the dual strain ``D`` are computed in
the pass too, row by row by the same bodies (#556). With immersed walls
attached they, ``ν_t`` and the wall production run as separate loops
before the pass, because the wall production corrects ``D`` and reads
``ν_t`` of every cell. `[timestep] loop_fusion
= "never"` runs the separate loops; the run's `results.toml`
(`[loop_fusion]`, `tke_update_blocked`) says which ran.

``e = 0`` is a fixed point of the equation (``ν_t = 0`` produces nothing):
[`initialize_tke!`](@ref Aeolus.PrognosticSGS.initialize_tke!) seeds the
local equilibrium of the initial flow
([`tke_equilibrium`](@ref Aeolus.PrognosticSGS.tke_equilibrium)), with a
floor `e_min`.

**Surface value.** With per-column friction velocities the first air
cell holds ``e_s = u_*^2/\sqrt{c_k c_ε}``, the TKE of a constant-stress
layer in equilibrium (``ν_tS = u_*^2``, ``ν_tS^2 = ε``); the energy this
sets is booked (`surface_energy`). The Berlin case takes ``u_*`` from its
MOST surface exchange (`bus.u_star`).

## On a stretched vertical grid (#259)

The Berlin nesting parent runs this closure on a `StretchedGrid` (25 m
levels to 500 m, then growing by 1.08 per level to 100 m). Four things
change, each tested in `test/tke_stretched.jl` (removed):

* **Filter width.** ``Δ_k = (Δx\,Δy\,Δz_k)^{1/3}`` per level, ``Δz_k`` the
  cell thickness — the width the core's Smagorinsky uses there (Deardorff
  1970). ``l``, ``c_ε``, ``κ_t = (1 + 2l/Δ_k)ν_t`` all take the local value.
* **Shear production.** The stretched Laplacian momentum operator
  differences the vertical fluxes of ``u, v`` over the centre spacing
  ``Δz_f`` and divides by the cell thickness ``Δz_c`` (``w`` the other way
  round). The duality is volume-weighted, ``Σ_c ν_c D_c V_c = −Σ_p u_p T_p V_p``,
  so each flux's share is accumulated per unit horizontal area (multiplied
  by the control-volume height of its velocity point), folded onto the
  interior as on the uniform grid, and divided by ``Δz_c`` of the receiving
  cell. Measured: the identity holds to 2.2e-16 for random fields on a
  1.08-stretched grid. For ``u = Sz``, ``D = S^2 (z_{k+1} − z_{k−1})/(2Δz_k)``
  (to 2.7e-15): exactly ``S^2`` on the uniform levels, ``S^2(1 + 1.5\times10^{-3})``
  on the geometric part, +1.7 % in the last uniform cell under the first
  stretched one.
* **Buoyancy production.** ``N^2`` is the difference of the two face values
  over the cell thickness, ``(θ_{k+1} − θ_{k−1})/(2Δz_c)``, the per-cell share
  of the stretched κ_t operator's potential-energy change: ``Σ(−κN^2)V``
  equals ``(g/θ_0)Σ z\,T_θ V`` to 4.0e-15. For a linear θ it carries the
  same 1.5e-3 / 1.7 % factor as ``D``.
* **Transport.** `AdvectionSchemes.advect_positive!` now runs on a
  `StretchedGrid` with the centred reconstruction (the vertical flux
  difference over ``Δz_c``), flux-limited toward first-order upwind so that
  ``e`` stays positive; the WENO / TENO stencils assume uniform spacing and
  raise an `ArgumentError` there. A 30-step stretched CBL books no clip
  (the plain centred flux form needed 370).

A `:uniform` `StretchedGrid` reproduces the `RegularGrid` closure (``e`` to
8.5e-17 relative after 20 convective steps without transport). A solid
mask is not supported on a stretched grid.

## Validation (`test/tke.jl` and the 500-step run of `validation/tke.jl`, measured)

| test | result |
| :-- | :-- |
| free decay at ``l = Δ`` (``c_ε = 0.70``) vs ``e(t) = (e_0^{-1/2} + c_εt/(2l))^{-2}``, 600 s | exact update 1.7e-14; semi-implicit errors 5.1e-2, 2.5e-2, 1.2e-2, 6.2e-3 for 60…480 steps (order 1.00) |
| neutral shear ``u = Sz``, box at equilibrium | ``e`` to 4.7e-15 of ``(c_k/c_ε)Δ^2S^2``; effective ``C_s`` 0.194413 = ``c_k^{3/4}c_ε^{-1/4}``; ``ν_t`` = `compute_smagorinsky_νt!`(C_s = 0.194413) to < 1e-6 |
| stable shear, ``Ri`` = 0, 0.1, 0.2: box vs scalar root-find | 1.4e-15, 1.1e-12, 4.1e-12; and the explicit linear-in-``x`` solution to 1e-10 |
| ``Ri = 0.25 > Ri_c = 0.2331`` | ``e`` decays to 7.5e-28 m² s⁻² (from 0.05) |
| ``Σ P_s V + Σ\mathbf u·\mathbf T_{sgs}V`` (periodic / walls) | 2.0e-16 / 1.5e-15 relative |
| ``Σ P_b V − (g/θ_0)Σ z\,T_θ V`` (walls) | 2.0e-15 relative |
| ``c_ε = 0``, ``ν = 0``: ``d/dt(KE + Σ eV)`` | < 1e-10 relative; per Euler step 1.0e-2, 2.4e-3, 5.9e-4 at Δt = 0.02, 0.01, 0.005 (second order: the time truncation) |
| ``c_ε > 0``: ``d/dt(KE + Σ eV) + Σ εV`` | < 1e-10 relative |
| 500 RK3 steps of decaying turbulence (halo 2 / stratified with walls) | ``e ≥ 0``, ``ν_t ≥ 0``, ``κ_t ≥ ν_t`` throughout; 0 clips |
| rejected adaptive step (first try committed, then rejected) | ``e``, ``u``, ``θ`` bitwise a direct ``Δt/2`` step |

**Convective boundary layer** (`examples/cbl_tke/cbl_tke.toml`, the model with `[sgs] model = "tke"`,
ONE-l #475; the 24³ validation case of `validation/tke.jl` is the harness `tke_cbl` of `test/fixtures/tke_cbl.jl`, which the
configuration reproduces bit for bit, `test/config_cases.jl` (removed)): free convection
over a heated free-slip floor, ``Q_s = 0.1`` K m s⁻¹, ``Γ = 3`` K km⁻¹ above
``z_{i0}``, periodic laterally, sponge in the top quarter, SSP-RK3 at the
CFL step.

| case | ``z_i`` | min heat-flux ratio | max SGS share of TKE, ``0.2 < z/z_i < 0.8`` | ``e`` peak | ``σ_w^2`` max |
| :-- | --: | --: | --: | --: | --: |
| 24³, Δ = 50 m, 1 h (validation) | 900 m | −0.157 | 0.203 | 75 m (0.08 ``z_i``) | 0.50 ``w_*^2`` at 0.31 ``z_i`` |
| 24³ with the ``κ d_{wall}`` cap | 850 m | −0.083 | 0.308 | 125 m (0.15 ``z_i``) | 0.34 ``w_*^2`` at 0.38 ``z_i`` |
| 64³, 50 × 50 × 25 m, 2 h (`examples/cbl_tke/cbl_tke.toml`) | 1175 m | −0.115 | 0.162 | 38 m (0.03 ``z_i``) | 0.459 ``w_*^2`` at 0.35 ``z_i`` |

Lenschow, Wyngaard & Pennell (1980): ``σ_w^2/w_*^2 = 1.8(z/z_i)^{2/3}(1 − 0.8z/z_i)^2``,
maximum 0.466 at ``z/z_i = 0.31``; the 64³ run is 0.035 RMS from it over
``0.1 < z/z_i < 0.9`` (64³ took 395 s for 3577 steps).

**The wall cap and Moeng's 3.9 together over-damp the first levels.** Both
correct the same thing: Moeng's factor makes ``ε/e^{3/2}`` at the first
level 2.73/Δ with ``l = Δ``; the ``κ d`` cap alone gives 1.46/Δ there (at
``z = Δ/2``), and both together 5.7/Δ. With the cap the 24³ CBL misses all
three criteria of #182 (they come from Moeng-type LES without a cap), so
the CBL is run with Moeng's closure and `wall_distance = nothing` is the
closure's default; the cap is there for walls Moeng's floor factor cannot
reach — the Berlin facades — and the Berlin case uses it.

**Decaying grid turbulence** (Comte-Bellot & Corrsin, `validation/cbc_decay.jl`,
32³, three seeds, with the corrected face-spectrum set-up of #205):
``K_{98}/K_{42}`` −9.4 %, ``K_{171}/K_{42}`` −26.9 %, RMS ``\log_{10}E``
0.065 / 0.120, ``E(k_c)``/CBC 0.81 / 0.38: too dissipative (before the
#205 fix, which started every closure with 13–22 % too much energy, it
was −18.5 % / −34.4 %). Two reasons. Its equilibrium is Smagorinsky with
``C_s = 0.194`` (static Smagorinsky at 0.194 gives +3.2 % / −10.1 %,
against +13.2 % / −2.6 % at 0.17), and ``e`` lags the fast decay: its
relaxation time ``l/(c_ε\sqrt e) ≈ 0.2`` s is close to the 0.28 s between
stations; on the pre-fix set-up the mean ``e`` was 1.8–2.6 times its
equilibrium with the decaying strain from station 98 on, so ``ν_t`` was
1.3–1.6 times too large.

## Differential Reynolds-stress closure (P56-d, #188)

[`ReynoldsStressClosure`](@ref Aeolus.PrognosticSGS.ReynoldsStressClosure)
is the second closure in `PrognosticSGS`. It goes past eddy viscosity in the
way Deardorff (1973) did: the six subgrid stresses
``τ_{ij} = \overline{u_i'u_j'}`` (m² s⁻²) are prognostic, and the momentum
flux is the stress itself, not ``−2ν_tS_{ij}``:

```math
\frac{Dτ_{ij}}{Dt} = P_{ij} + G_{ij} + C_{ij} + Φ_{ij} − \tfrac23 ε δ_{ij}
  + ∇·(2ν_t ∇τ_{ij}),
```

* **production** ``P_{ij} = −τ_{ik}∂_k\bar u_j − τ_{jk}∂_k\bar u_i``, exact;
* **buoyancy** ``G_{ij} = (g/θ_0)(δ_{i3}h_j + δ_{j3}h_i)`` with the
  Daly & Harlow (1970) heat flux ``h_i = −c_θ τ_{ij}∂_jθ_v``;
* **Coriolis** ``C_{ij} = −2Ω_k(ε_{ikm}τ_{mj} + ε_{jkm}τ_{mi})``, which is
  trace-free; `Ω = :auto` takes the f-plane ``(0, 0, f/2)`` that the
  resolved momentum uses;
* **pressure–strain**, the LRR-IP model (Launder, Reece & Rodi 1975) with
  the buoyancy term of Gibson & Launder (1978):
  ``Φ_{ij} = −C_1\frac{ε}{e}(τ_{ij} − \tfrac23 eδ_{ij}) − C_2(P_{ij} − \tfrac13P_{kk}δ_{ij}) − C_3(G_{ij} − \tfrac13G_{kk}δ_{ij})``,
  with ``C_1 = 1.8``, ``C_2 = 0.6``, ``C_3 = 0.5``;
* **dissipation** ``ε = c_εe^{3/2}/l`` and the length ``l``, both **shared
  with the TKE closure by function call**:
  [`tke_length_scale`](@ref Aeolus.PrognosticSGS.tke_length_scale) (with
  ``N^2`` of θ_v and the wall cap) and
  [`tke_dissipation_coefficient`](@ref Aeolus.PrognosticSGS.tke_dissipation_coefficient)
  (Moeng's 3.9 at the first level). ``ν_t`` for the ``2ν_t`` transport comes
  from [`tke_eddy_coefficients`](@ref Aeolus.PrognosticSGS.tke_eddy_coefficients).

Half the trace is the TKE equation. ``Φ`` and ``C`` are trace-free, so
``\tfrac12 τ_{kk}`` obeys ``De/Dt = −τ_{ik}∂_k\bar u_i + (g/θ_0)h_3 − ε + ∇·(2ν_t∇e)``.
For an eddy-viscosity stress this is the P56-b equation term by term.

**The Daly–Harlow coefficient.** The scalar flux is the tensor diffusivity
``K_{ij} = c_θτ_{ij}`` ([`daly_harlow_coefficient`](@ref Aeolus.PrognosticSGS.daly_harlow_coefficient)).
The default, `C_θ = :tke`, sets ``c_θ = \tfrac32κ_t/e`` with the TKE
closure's ``κ_t = (1 + 2l/Δ)ν_t``. An isotropic ``τ = \tfrac23eI`` then
diffuses scalars with exactly the P56-b ``κ_t``. In Daly & Harlow's
notation this is ``C_θ = \tfrac32(1 + 2l/Δ)c_kc_ε``: 0.315 at ``l = Δ``
and 0.099 in the stable ``l → 0`` limit, inside the 0.1–0.3 range of the
published constants. The value is a calibration, not a published
constant. A number `C_θ` gives Daly & Harlow's ``c_θ = C_θe/ε``.

**Homogeneous-shear equilibrium.** Because ``ε/e = c_ε\sqrt e/l`` grows
with ``e``, the SGS closure has a true equilibrium, ``P = ε``. Unlike the
RANS asymptote, ``P/ε`` does not depend on an ε equation.
``P_{ij} + Φ_{ij} = \tfrac23εδ_{ij}`` with ``P = ε`` gives
([`reynolds_stress_shear_equilibrium`](@ref Aeolus.PrognosticSGS.reynolds_stress_shear_equilibrium)):

```math
\frac{τ_{22}}{e} = \frac{τ_{33}}{e} = \frac23 − \frac23\frac{1 − C_2}{C_1} = 0.518519,\quad
\frac{τ_{11}}{e} = \frac23 + \frac{2(1 − C_2) + \tfrac23C_2 − \tfrac23}{C_1} = 0.962963,\quad
\frac{−τ_{13}}{e} = \sqrt{\frac{1 − C_2}{C_1}\frac{τ_{22}}{e}} = 0.339450 .
```

These are the anisotropies ``b_{11} = 0.148``, ``b_{22} = b_{33} = −0.074``,
``b_{13} = −0.170``. Tavoularis & Corrsin (1981) measured a larger
anisotropy in which the two cross-stream components differ (roughly
``b_{11} ≈ 0.20``, ``b_{22} ≈ −0.14``, ``b_{33} ≈ −0.06``,
``b_{12} ≈ −0.15``). The IP model's ``τ_{22} = τ_{33}`` is a known
deficiency. The comparison is quoted here and not asserted.

### Discretization

**Momentum.** `add_sgs_momentum!` adds ``−∂_jτ^{dev}_{ij}`` through the one
stress operator, `Operators.add_stress_divergence!` (P55-b). The
deviatoric diagonals stay at the cell centres; the isotropic part is a
pressure. Each off-diagonal is the mean of the four cells around its edge
([`stress_edges!`](@ref Aeolus.PrognosticSGS.stress_edges!)). An edge on a
wall, or one that touches a solid cell, carries no stress, because the
wall stress belongs to the surface exchange. The `stress` keyword of the
cores is ignored.

**Production is the dual of the momentum operator.** The cell-centred
velocity gradient of ``P_{ij}``
([`dual_velocity_gradient`](@ref Aeolus.PrognosticSGS.dual_velocity_gradient))
is built as follows:

* the diagonal is the face-to-face difference through the cell;
* each off-diagonal is the mean of its edge differences over the cell's
  four edges in that plane, with closed edges counting zero.

Summation by parts then gives
``−Σ_{faces}\mathbf u·∇·τ^{dev}V = Σ_{cells}τ^{dev}_{ik}G_{ik}V`` exactly,
walls and solids included. The resolved kinetic energy that the stress
removes is the SGS energy that ``P`` produces; the isotropic part pairs
with the discrete divergence. For a linear velocity field ``G`` is the
exact gradient away from closed edges.

**Scalars.** `add_sgs_scalar!` applies ``K = c_θτ`` (built in
`sgs_coefficients!`) through the one tensor operator,
`Operators.add_tensor_diffusion_scalar!` (P55-c), for θ and q_v. There is
no second flux operator. `sim.νt` holds ``ν_t`` and `sim.κt` holds
``\operatorname{tr}K/3``, which the adaptive step's diffusion limit uses.

**The commit** (`advance_sgs!`, once per accepted step):

1. With `transport`, the transport is explicit. The diagonals are advected
   with the flux limiter (`advect_positive!`), the off-diagonals with the
   plain scheme, and every component is diffused with ``2ν_t``. Prefer
   WENO-Z5, which needs a halo of 3. With centred differences the
   off-diagonals overshoot, and in a 16³ decaying box (amplitude 0.3 m/s,
   60 s) the realizability projection fired 9140 times and added 2.8 % of
   the SGS energy. With WENO-Z5 it did not fire.
2. Coriolis is applied exactly, as the rotation ``τ ← RτR^T`` with
   ``R = \exp(−2Δt[Ω]_×)`` (Rodrigues;
   [`coriolis_rotation`](@ref Aeolus.PrognosticSGS.coriolis_rotation)). The
   trace, the determinant and realizability are kept.
3. The other sources are linear in ``τ`` once ``ε/e`` and ``c_θ`` are lagged
   at ``τ^n``. They are integrated by backward Euler on the 6×6 system
   ``(I − ΔtL)τ^{n+1} = τ^*``
   ([`reynolds_stress_step`](@ref Aeolus.PrognosticSGS.reynolds_stress_step)).
   A steady state of the update satisfies ``Lτ = 0`` with ``L`` evaluated
   at ``τ`` itself, so it is exactly a root of the stress equations.

   Backward Euler inverts a growing mode once ``Δtλ ≥ 1``; production
   faster than the sinks is such a mode. The step is therefore split into
   ``⌈2Δtμ_∞(L)⌉`` sub-steps whenever the logarithmic norm ``μ_∞(L)``,
   which bounds every mode's growth rate, is positive. **Bug caught by
   the Berlin smoke check:** before the split, a single 1–2 s step at a
   street corner (run without the wall cap) made the stresses so
   unrealizable that the projection added 3.96e4 m⁵ s⁻², more SGS energy
   than the whole domain held. After the split the same run needs 0
   corrections.
4. Realizability (Schumann 1977) is enforced next. Every cell is projected
   onto the positive semi-definite cone: the eigenvalues are clipped at
   zero ([`project_realizable`](@ref Aeolus.PrognosticSGS.project_realizable)),
   which gives the nearest realizable stress in the Frobenius norm. Every
   correction and the energy it adds are booked
   ([`reynolds_stress_statistics`](@ref Aeolus.PrognosticSGS.reynolds_stress_statistics)).
5. With `surface`, the first air cell of each column is set to the LRR-IP
   constant-stress layer:
   * ``τ_{13}, τ_{23} = −u_*^2(\cos α, \sin α)``, against the cell's wind
     direction ``α``;
   * ``e_s = u_*^2/0.3395``;
   * the normal stresses ``(0.963, 0.519, 0.519)e_s`` in the wind frame.

   This is the counterpart of the TKE closure's ``e_s``.

**One cache-blocked pass** (#560). With `transport`, on the CPU, steps 1
to 4 (``ν_t``, the flux-limited advection of the diagonals, the plain
advection of the off-diagonals, the ``2ν_t`` diffusion and the source
update with its budget sums) run row by row on the blocked driver
(`FiniteVolume.blocked_update!`) with the pipeline of the TKE pass: one
set of flux and ratio rings per diagonal, the six new stresses held in
the driver's ring until no later row reads the old ones. The ring holds
``h_r`` rows (3 with WENO-Z5) because the plain advection of the
off-diagonals reads the stresses ``h_r`` rows back; the slabs keep
``h_r + 1`` edge planes (`rsm_blocked_lags`). Every value is computed by
the point body of the separate loop: the stresses and the budget sums
agree with the separate loops to round-off (measured identical).
``N^2`` stays a whole-field loop before the pass (unlike the TKE pass,
#556): computing it in the pass would save about 1.5 ms of an update of
110–270 ms. The pass pays little here: the update is limited by its arithmetic (the
6×6 solve, about 350 ns of thread time per cell), not by memory traffic.
Measured on 200×200×60 it was faster only with centred advection on
several threads (12 threads: 110 against 155 ms), and slower at one
thread and with WENO-Z5 at 12 threads (60 levels allow 7 slabs of
``2(h_r+1)`` planes). So `loop_fusion = "if_possible"` (the default) takes
it only with centred advection on several threads, `"always"` wherever it
can run, and `"never"` runs the separate loops; `results.toml`
(`[loop_fusion]`, `reynolds_stress_update_blocked`) says which ran.
Without transport (the urban `:reynolds_stress` option, whose stresses
the `:transport` stage moves) the update is one pointwise pass already.

`sgs_state` / `restore_sgs_state!` make a rejected adaptive step exact.
`step_imex!` refuses the closure through the contract trait
`Operators.sgs_explicit_stress`, because its implicit operator can only
represent an eddy viscosity.

**Stabilisation.** The issue allowed an optional eddy-viscosity split
(``τ = τ^{RSM} − 2ν_tS + 2ν_tS``, with the last term implicit). It was
not needed and is not implemented, so the momentum flux is the
Reynolds stress alone, and the claim of going beyond eddy viscosity holds
without qualification. Three runs were stable without it: the decaying
box (200 RK3 steps), the Berlin smoke runs (12 adaptive steps, with and
without the wall cap), and the 30 min flagship runs.

### Validation (`test/reynolds_stress.jl` (removed), measured)

1. **Trace.** For 200 random states, ½ tr of the right-hand side equals
   ``−τ:∇\bar u + βh_3 − ε`` to 1.2e-16, and ``\operatorname{tr}Φ +
   \operatorname{tr}C`` is 6.9e-17. For the eddy-viscosity state of a
   stratified shear (48 cases with ``Ri ∈ [−0.5, 0.2]``, the wall cap and
   the 3.9 factor), ½ tr of the RSM right-hand side equals the TKE
   right-hand side ``ν_tS^2 − κ_tN^2 − ε`` to 2.6e-16.
2. **Rotta.** The identity ``db/dt = −(C_1−1)(ε/e)b`` holds on the
   right-hand side to 1.1e-15. The closed form
   ``b = b_0(e/e_0)^{C_1−1}`` with ``e = (e_0^{−1/2} + c_εt/2Δ)^{−2}`` is
   reproduced by RK4 of the right-hand side to 6.3e-14 (e) and 4.5e-14
   (b). The 3-D closure converges at first order: errors 4.7e-2, 2.3e-2,
   1.2e-2 and 5.8e-3 at 60–480 steps over 600 s, ratios 2.03, 2.01, 2.01.
3. **Homogeneous shear.** A 3-D box with ``u = Sz`` gives
   0.962963 / 0.518519 / 0.518519 / 0.339450 to 1e-10, with ``e`` matching
   ``(0.33945\,SΔ/c_ε)^2`` to 3e-15 and no clips. The issue asked for the
   rounded 0.963 / 0.519 / 0.339 within 1e-4, but 0.519 and 0.339 are
   4.8e-4 and 4.5e-4 from the closed form. That expectation was a
   rounding error; the test asserts the closed form.
4. **Rotation.** Over 10 periods, ``τ(t) = R(2Ωt)τ_0R^T`` holds to 5.1e-14,
   with ``R`` from the matrix exponential, independent of the Rodrigues
   code. ``\operatorname{tr}τ`` is invariant to 3.8e-14 and ``\det τ`` to
   1.0e-13. RK4 of the right-hand side's ``C_{ij}`` returns to ``τ_0``
   after one period of ``2Ω``.
5. **Stratification.** The equilibrium of the 3-D box matches the root of
   the hand-written steady LRR-IP + Daly–Harlow equations (Newton) to
   3.0e-14 at ``Ri = 0.02, 0.05, 0.08, 0.11``. ``b_{33} = −0.082, −0.095,
   −0.111, −0.116`` while ``Nl/\sqrt e = 0.30, 0.51, 0.70, 0.76``, so
   ``b_{33} < 0`` and ``|b_{33}|`` is monotone.
6. **Realizability and energy.** A 16³ periodic decaying box (WENO-Z5,
   200 RK3 steps of 0.1 s) is realizable at every step with 0 clips.
   ``Σ(\text{resolved KE} + e V)`` changes by ``−ΣεVΔt`` with a residual of
   at most 5.9e-7 per step (relative), which is the time truncation of the
   exchange: it falls by 3.9× per halving of Δt. The exchange itself is
   exact: with walls and a solid block, the stress divergence's
   resolved-KE tendency equals ``Σ τ^{dev}:G V`` to 6.8e-16.

Further tests:

* With an isotropic ``τ``, the θ tendency through the tensor operator
  equals the TKE closure's to 1e-12, and ``ν_t`` is the same function.
* The surface value is checked: ``|τ_{h3}| = u_*^2`` against the wind, and
  the wind-frame normal stresses are ``(a_{11}, a_{22})e_s``.
* A growing mode with ``Δt|G| = 4`` is handled. One backward-Euler step
  gives ``e = −3.5`` × 10⁻⁴, which is not realizable. The sub-stepped
  update gives 8.3 × 10⁻⁴; the frozen-coefficient exact value is
  7.1 × 10⁻⁴.
* Rollback is bit for bit, and `step_imex!` refuses the closure.

## Validation against filtered DNS (P56-e, #192)

The TKE closure and the Reynolds-stress closure are compared *a priori in
time* with filtered DNS, with Smagorinsky (``C_s = 0.17``) and dynamic
Smagorinsky as baselines (the Lagrangian scale-dependent model, #186, is
not in this branch's stack). Fixture: `validation/fixtures/sgs_validation.jl`
(module `SGSValidation`; it was the `sgs_decay` run type from EX-d #441
to ONE-l #475); validation run: `validation/sgs_validation.jl`. The 96³ example
and the wall-bounded surface-mounted cube (below) were run types of
`Aeolus.RunDrivers`; they are not configurations of the model and were
dropped with it (ONE-l, #475): the filtered-DNS study is the a-priori
study the single-model design drops (the a-posteriori comparison is
`examples/dns_vs_les/`), and the cube is a channel case (a periodic
channel held at its bulk velocity). Their numbers below are the record
of the runs made before.

**Method.** The P56-c DNS of decaying isotropic turbulence (2π box,
spun up from a fixed seed to ``t_0 = 2``; see
[LES filtering](les_filtering.md)) is continued from ``t_0`` for two
turnovers ``T = L/u'`` (``u'`` and the integral scale ``L`` measured at
``t_0``) in three ways: unchanged; with f-plane rotation
``f = u'/(\mathrm{Ro}\,L)``, Ro = 0.2; and with a uniform background
stratification ``N = u'/(\mathrm{Fr}\,L)``, Fr = 0.3. The DNS velocity is
filtered with the Gaussian of width ``Δ = 4h`` and resampled on the
four-times-coarser grid by Fourier truncation. The exact subfilter stress
``τ_{ij} = \overline{u_iu_j} − \bar u_i\bar u_j`` and the subfilter heat
flux are sampled at the coarse centres by the 2×2×2 mean, which keeps
``τ`` positive semi-definite (a Fourier resample of ``τ`` does not). Every
LES (``Δ = 4h``, the same ν, f and N) starts from the filtered DNS at
``t_0``. The TKE closure gets ``e = ½τ_{kk}`` (`initialize_tke!(c, τ)`) and
the Reynolds-stress closure gets ``τ_{ij}`` (`initialize_reynolds_stress!(c, τ)`;
no cell needed a realizability correction).

*Stratification in a periodic box.* The simulations carry
``θ = θ_0 + θ'`` with ``θ'`` periodic. The background ``Γz``
(``Γ = N^2θ_0/g``) enters as the body force ``−Γ\bar w`` on ``θ'``,
through the `body_force!` hook of `step_rk3!`. The face average of θ in
the w equation and the centre average of w in the θ equation are
adjoint, so ``K + PE`` falls at every step (asserted). The closures need
the full θ for ``N^2`` and for the flux ``−K∇θ``, so a fixture wrapper
hands them θ₀ + Γ(z − z_mid) + θ′, with halos continued across the
periodic boundary with the jump ``ΓL_z``.

**Measured** (48³ DNS, Re_λ = 32 at ``t_0``; LES 12³; 44 s in CI):

| resolved-KE error at ``t_0 + 2T`` | isotropic | rotating | stratified |
| :-- | --: | --: | --: |
| Smagorinsky | +2.2 % | −11.9 % | −0.8 % |
| dynamic Smagorinsky | +9.2 % | +7.5 % | +11.6 % |
| TKE | −11.6 % | −22.6 % | +28.4 % |
| Reynolds stress | −11.7 % | +6.4 % | +15.9 % |

| quantity | filtered DNS | TKE | Reynolds stress |
| :-- | --: | --: | --: |
| isotropic: ``e`` at ``t_0 + 2T`` | 0.0443 | 0.125 | 0.125 |
| stratified: ``⟨τ_{33}⟩/⟨τ_{11}⟩``, mean over ``(t_0, t_0+2T]`` | 0.70 | 1.00 | 0.76 |
| stratified: pointwise ``ρ(τ_{33}/τ_{11})`` at ``t_0 + T/8`` | — | 0.17 | 0.71 |
| stratified: Lumley II at ``t_0 + 2T`` | 0.270 | 0.0012 | 0.049 |
| stratified: mean ``⟨w'θ'⟩_{sgs}`` over ``(t_0, t_0+2T]`` | −0.232 | −0.315 | −0.347 |

* **Rotation.** Rotation slows the DNS decay, as Bardina, Ferziger &
  Rogallo (1985) found: ``K(t_0+2T)/K(t_0)`` is 0.389 against 0.295
  without rotation. The Reynolds-stress closure has the smallest KE error
  of the four closures (6.4 %, against 11.9 % for Smagorinsky and 22.6 %
  for TKE). It is the only closure that sees the rotation through its own
  equations. The Coriolis term leaves ``\operatorname{tr}τ`` unchanged in
  3-D: the same anisotropic stress field advanced at rest with and without
  Ω has traces that agree to 1e-15, while ``τ_{11}`` differs by 0.28.
* **Stratification.** The Reynolds-stress closure reproduces the
  suppressed vertical subfilter stress. The mean ``τ_{33}/τ_{11}`` is 0.76,
  against 0.70 for the DNS; an eddy viscosity cannot do this (TKE: 1.00).
  Its II is closer to the DNS's than TKE's, although both are well below
  the pointwise anisotropy of the exact stress (0.27).

**Expectations of the issue that failed at this Re_λ.** The tests keep
them as `@test_broken`, with the numbers.

* *Resolved KE within 10 % at ``t_0 + 2T``* holds for both Smagorinsky
  baselines but not for TKE (−11.6 %) or Reynolds stress (−11.7 %).
  *TKE: e within 25 % of ½⟨τ_kk⟩* fails too: ``e`` is 2.8 times the DNS
  value. Both failures have the same cause, which the test asserts. The
  Deardorff constants (``c_k = 0.1``, and ``c_ε = 0.70`` at ``l = Δ``;
  Moeng & Wyngaard 1988) are inertial-range values, but ``Δ = 4h ≈ 8η``
  sits in the near-dissipation range. With the exact ``e`` at ``t_0``:
  * the production ``c_kΔ\sqrt e\,2⟨\bar S\bar S⟩ = 0.184`` is 1.9 times
    the exact transfer ``Π = −⟨τ_{ij}\bar S_{ij}⟩ = 0.096``;
  * the dissipation ``c_εe^{3/2}/Δ = 0.073`` is 0.68 times the viscous
    dissipation of the subfilter scales, ``2ν(⟨SS⟩ − ⟨\bar S\bar S⟩) = 0.109``.

  So ``e`` grows for a turnover while the DNS's decays, and ``ν_t``
  follows it. The trace of the Reynolds-stress equation is the same TKE
  equation, so its ``e`` grows the same way (0.125 at ``2T``).
* *The τ₃₃/τ₁₁ of the Reynolds-stress closure correlates with the DNS,
  ρ > 0.5.* Pointwise, this holds while the LES still follows the DNS:
  0.71 at ``T/8`` and 0.56 at ``T/4``, against 0.17 and 0.20 for TKE. It
  then falls to 0.02 at ``2T``. Every LES decorrelates pointwise from the
  DNS within a turnover: the correlation of ``τ_{33}`` itself is 0.15 at
  ``2T``. The test therefore asserts it at ``T/8`` and ``T/4``, and
  asserts the mean ratio over the whole run.
* *The Reynolds-stress heat flux is closer to the DNS than TKE's.* Both
  are within 2×, but TKE is closer (1.35× against 1.49×). The DNS flux
  oscillates with the buoyancy period ``2π/N = 1.9T`` and is
  counter-gradient for about a third of the run, reaching +1.4. No
  down-gradient closure follows this. The Reynolds-stress ``e`` also
  decays more slowly (0.029 against 0.015 at ``2T``), which makes its flux
  larger.

**96³** (the example, Re_λ = 64, LES 24³, one run):

| 96³, resolved-KE error at ``t_0 + 2T`` | isotropic | rotating | stratified | rotating + stratified |
| :-- | --: | --: | --: | --: |
| Smagorinsky | −5.1 % | −7.4 % | −6.1 % | −10.2 % |
| dynamic Smagorinsky | +1.0 % | +2.6 % | +3.4 % | +3.6 % |
| TKE | −13.6 % | −14.0 % | +9.8 % | +5.5 % |
| Reynolds stress | −22.0 % | +1.4 % | −0.8 % | +5.7 % |
| DNS ``K(t_0+2T)/K(t_0)`` | 0.373 | 0.461 | 0.295 | 0.382 |

(Measured for #192. Re-run on 2026-10-09 as the config, EX-d #441: the old
script on today's code prints the same tables to every digit, and both
differ from the table above in the last digit, from changes to the code
since: DNS ``K`` ratios 0.375, 0.463, 0.296, 0.384; TKE −13.7 % (isotropic),
+5.6 % (rotating + stratified); Reynolds stress −21.9 %, +1.5 %, −0.7 %
(isotropic, rotating, stratified); ``u' = 0.866``, ``L = 1.205``,
``T = 1.392``; the stratified mean SGS heat flux −0.087 (DNS), −0.375
(TKE), −0.362 (Reynolds stress). Every other entry is unchanged.)

At 96³ (``u' = 0.864``, ``L = 1.208``, ``T = 1.398``; DNS 68 s spin-up
and 60–83 s per continuation, LES 0.4–5 s):

* Rotation again slows the DNS decay (0.461 against 0.373). The
  Reynolds-stress closure has the smallest rotating KE error (1.4 %).
* In the stratified case the mean ``τ_{33}/τ_{11}`` is 0.80 for the DNS,
  0.75 for the Reynolds-stress closure and 1.005 for TKE.
* The isotropic ``e`` at ``2T`` is still too large: DNS 0.050, TKE 0.108,
  Reynolds stress 0.116. The isotropic KE errors of TKE (−13.6 %) and
  Reynolds stress (−22.0 %) still miss 10 %. ``Δ/η`` is the same as at
  48³ (``ν ∝ N^{−4/3}``), so the near-dissipation-range argument above
  applies unchanged.
* The heat-flux expectation fails more clearly here. The mean DNS
  subfilter flux is −0.086, against −0.374 (TKE) and −0.361 (Reynolds
  stress), so both closures are outside 2×. At this Re the DNS flux
  spends more of the run counter-gradient.

The figure the example wrote (`rotating_stratified_decay_96.svg`)
has one row per case. The panels are the resolved KE, the SGS energy,
and the Lumley-triangle trajectory ``(ξ, η) = ((III/6)^{1/3}, (II/6)^{1/2})``
of the mean invariants. The exact stress sits close to the axisymmetric
(rod-like) boundary at ``η ≈ 0.21`` and hardly moves. The Reynolds-stress
closure starts there, because it is seeded with the exact ``τ``, and then
relaxes towards isotropy along that boundary. The TKE closure's implied
stress starts much closer to the isotropic corner.

### Surface-mounted cube (example)

The former `obstacle_channel` run type (EX-d #441 to ONE-l #475, a
sweep over the closures; dropped with the channel cases) compared the
three closures on
the wall-bounded case of Martinuzzi & Tropea (1993): a cube of height
``H`` in a channel of height ``2H`` at ``\mathrm{Re}_H = 40\,000``, where
the mean flow reattaches behind the cube at ``x_R/H = 1.612`` (from the
rear face, on the centre line) and the horseshoe vortex separates the
floor flow ``x_F/H = 1.04`` ahead of the front face.

*Set-up.* A periodic box of ``10H × 6H × 2H`` at 16 cells per ``H``
(160 × 96 × 32 fluid levels), with the cube at ``x = 3H``–``4H``. The cube
and the floor are the immersed boundary of `IBM`, applied by direct
forcing (`implicit_ibm_correction!`) after every RK3 predictor through the
`post_predictor!` hook of `step_rk3!`. The lid at ``z = 2H`` is free-slip:
the channel's upper wall is replaced by a symmetry plane. A PI-controlled
streamwise body force holds ``U_b = 1``. The closures run as in Berlin:
TKE and Reynolds stress with the wall cap ``κ d`` from
`IBM.wall_distance` and Moeng's 3.9 at the first level, Smagorinsky with
``C_s = 0.1``. The statistics are averaged over ``30\,H/U_b`` after a
``30\,H/U_b`` spin-up. ``x_R`` and ``x_F`` are the sign changes of the mean
``u`` at the first fluid level on the centre line.

| 16 cells per ``H``, one run each | ``x_R/H`` | ``x_F/H`` | steps | wall time | s/step |
| :-- | --: | --: | --: | --: | --: |
| Martinuzzi & Tropea (1993) | 1.612 | 1.04 | | | |
| Smagorinsky (``C_s = 0.1``) | 1.82 (+13 %) | 0.51 | 4469 | 1063 s | 0.24 |
| TKE | 1.80 (+11 %) | 0.21 | 4152 | 1323 s | 0.32 |
| Reynolds stress | 1.97 (+22 %) | 0.26 | 4536 | 4743 s | 1.05 |

The runs shared a heavily loaded machine, so the costs are rough; the
Reynolds-stress run was the slowest by 3–4×, more than its 2× in Berlin.

**Re-run as the config (EX-d, #441, 2026-10-09).** The table above is
not reproducible: FFTW's default MEASURE planner chooses its plans by
timing them, so two processes can take different plans, and the flow is
chaotic. Two MEASURE runs of the same code (the config and the old script)
gave ``x_R/H`` = 1.83 and 1.91 (Smagorinsky) and 1.64 and 1.54 (TKE),
and 1.90 for one Reynolds-stress run. The config therefore plans with
ESTIMATE (`[fftw] planner = "estimate"`), and a rerun is bitwise the run
(two Smagorinsky runs: identical floor profiles):

| ESTIMATE, one run each | ``x_R/H`` | ``x_F/H`` | steps | wall time | s/step |
| :-- | --: | --: | --: | --: | --: |
| Smagorinsky (``C_s = 0.1``) | 1.88 (+16 %) | 0.49 | 4429 | 575 s | 0.13 |
| TKE | 1.46 (−9 %) | 0.38 | 4242 | 602 s | 0.14 |
| Reynolds stress | 1.95 (+21 %) | 0.33 | 4559 | 1813 s | 0.40 |

(Four runs at once on 24 cores.) Over all runs ``x_R/H`` spans 1.82–1.91
for Smagorinsky, 1.46–1.80 for TKE and 1.90–1.97 for Reynolds stress.

* The 30 ``H/U_b`` average does not pin ``x_R`` down: runs that differ
  only in the last bits of the pressure solve spread by up to 0.34 ``H``
  (TKE). Smagorinsky and the Reynolds-stress closure put the reattachment
  13–22 % too far downstream in every run; TKE lands on either side of
  the measurement (−9 % to +12 %). The ranking of the closures quoted
  before ("TKE closest, the Reynolds-stress closure furthest") is within
  this spread for TKE. The differential stress model does not improve the
  recirculation length at this resolution.
* The front separation is far too short under every closure (0.17–0.52 H
  over all runs, against 1.04 H). At 16 cells per ``H`` the floor boundary layer
  approaching the cube is resolved by a few cells with no wall model, so
  the horseshoe vortex is too weak and too close to the cube.
* The P57-d facade wall model (#179) that the issue asks for is not in
  this branch's stack (PR #211 is open), so the cube walls are no-slip
  with the SGS stress as the only wall treatment. With that model the
  comparison should be repeated. The 30 ``H/U_b`` average is short;
  the curves in `surface_mounted_cube_16.svg` (in the first run folder's `output/`) are not
  yet smooth behind ``x = 6H``.

## Berlin coupling

`src/components/urban/sgs_tke.jl`, provider `:tke` of the `:sgs` slot
(see [the Berlin case](../design/berlin_numerics.md#Numerics-slots)), with ``e``
the advected tracer `:e`, the wall cap from the building mask, the
surface value from the MOST friction velocity, and the facade walls of
`wall_model_facades` attached, so the facade production is the wall
stress times the resolved shear (#232; on the summer flagship the
stencil production it replaces was 27 % of the canopy ``Σ P_s V``, and
near-facade ``ν_t`` rises by 5.3 %: see the facade section of
[the Berlin case](../design/berlin_surface_radiation.md#Facade-wall-model-(#179))). The Reynolds-stress closure
is the option `:reynolds_stress` (`src/components/urban/sgs_reynolds_stress.jl`):
the six stresses are the advected tracers `:tau11 … :tau23`, with the same
wall cap and an LRR-IP surface value. Both providers report a-priori
metrics at the test-filter scale (P56-e): the closure's subfilter energy
against the resolved energy between the grid and the 3-cell box test
filter, ``k_{test} = ½⟨L_{kk}⟩`` with ``L_{ij}`` the resolved (Germano)
stress, the correlation of the model stress with ``L_{ij}``, and the
median Lumley II of both. They also write ``e`` and ``II`` (`II_sgs`) to
the NetCDF file. Measured numbers are in the Berlin design page.

## References

  * Deardorff, J. W. (1980), *Stratocumulus-capped mixed layers derived
    from a three-dimensional model*, Boundary-Layer Meteorol. 18, 495–527.
  * Moeng, C.-H. (1984), *A large-eddy-simulation model for the study of
    planetary boundary-layer turbulence*, J. Atmos. Sci. 41, 2052–2062.
  * Moeng, C.-H. & Wyngaard, J. C. (1988), *Spectral analysis of
    large-eddy simulations of the convective boundary layer*, J. Atmos.
    Sci. 45, 3573–3587.
  * Lenschow, D. H., Wyngaard, J. C. & Pennell, W. T. (1980), *Mean-field
    and second-moment budgets in a baroclinic, convective boundary
    layer*, J. Atmos. Sci. 37, 1313–1326.
  * Comte-Bellot, G. & Corrsin, S. (1971), *Simple Eulerian time
    correlation of full- and narrow-band velocity signals in
    grid-generated, 'isotropic' turbulence*, J. Fluid Mech. 48, 273–337.
  * Patankar, S. V. (1980), *Numerical Heat Transfer and Fluid Flow*,
    Hemisphere, §7.2.
  * Danielsson, P.-E. (1980), *Euclidean distance mapping*, Comput.
    Graph. Image Process. 14, 227–248.
  * Deardorff, J. W. (1973), *The use of subgrid transport equations in a
    three-dimensional model of atmospheric turbulence*, J. Fluids Eng. 95,
    429–438.
  * Launder, B. E., Reece, G. J. & Rodi, W. (1975), *Progress in the
    development of a Reynolds-stress turbulence closure*, J. Fluid Mech.
    68, 537–566.
  * Gibson, M. M. & Launder, B. E. (1978), *Ground effects on pressure
    fluctuations in the atmospheric boundary layer*, J. Fluid Mech. 86,
    491–511.
  * Daly, B. J. & Harlow, F. H. (1970), *Transport equations in
    turbulence*, Phys. Fluids 13, 2634–2649.
  * Schumann, U. (1977), *Realizability of Reynolds-stress turbulence
    models*, Phys. Fluids 20, 721–725.
  * Lumley, J. L. (1978), *Computational modeling of turbulent flows*,
    Adv. Appl. Mech. 18, 123–176 (the anisotropy invariants).
  * Tavoularis, S. & Corrsin, S. (1981), *Experiments in nearly
    homogeneous turbulent shear flow with a uniform mean temperature
    gradient. Part 1*, J. Fluid Mech. 104, 311–347.
  * Bardina, J., Ferziger, J. H. & Rogallo, R. S. (1985), *Effect of
    rotation on isotropic turbulence: computation and modelling*, J.
    Fluid Mech. 154, 321–336.
  * Riley, J. J., Metcalfe, R. W. & Weissman, M. A. (1981), *Direct
    numerical simulations of homogeneous turbulence in density-stratified
    fluids*, AIP Conf. Proc. 76, 79–112.
  * Germano, M., Piomelli, U., Moin, P. & Cabot, W. H. (1991), *A dynamic
    subgrid-scale eddy viscosity model*, Phys. Fluids A 3, 1760–1765.
  * Liu, S., Meneveau, C. & Katz, J. (1994), *On the properties of
    similarity subgrid-scale models as deduced from measurements in a
    turbulent jet*, J. Fluid Mech. 275, 83–119.
  * Martinuzzi, R. & Tropea, C. (1993), *The flow around surface-mounted,
    prismatic obstacles placed in a fully developed channel flow*, J.
    Fluids Eng. 115, 85–92.
  * Maronga, B. et al. (2015), *The Parallelized Large-Eddy Simulation
    Model (PALM) version 4.0*, Geosci. Model Dev. 8, 2515–2551 (the TKE
    closure as the urban-LES default).
