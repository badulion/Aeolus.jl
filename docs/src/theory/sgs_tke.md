# Prognostic SGS TKE (Deardorff–Moeng)

Era VI, Phase 56 (P56-b, #182). The 1.5-order closure of Deardorff (1980)
carries the subgrid-scale turbulent kinetic energy ``e`` as a prognostic
variable; the eddy viscosity follows from it, not from the resolved strain
alone. Aeolus implements it with the coefficients of Moeng (1984) as
[`TKEClosure`](@ref Aeolus.PrognosticSGS.TKEClosure) in the module
`Aeolus.PrognosticSGS` ([API](../api/era6/prognostic_sgs.md)), an
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
potential temperature when the case is moist.

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
  Above it the TKE decays to zero: the closure switches itself off
  without a stability function.

**No stability function on top.** The ``0.76\sqrt e/N`` length is the
closure's own stability response, so no `TurbulenceStability` factor
``f_m(Ri)`` is applied to it, and the `StabilityCorrected` wrapper of
P55-e (#187), which multiplies a Smagorinsky-type ``ν_t`` by such a
factor, must not wrap a `TKEClosure`.

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
image. ``D`` is built from the face and edge differences of the
momentum kernels, not from `Operators._velocity_gradient_at`: that
tensor's off-diagonal entries are centred over two cells, and their
square is not the dual of the edge-flux operator.

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

## Validation (`test/era6_tke.jl`, measured)

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

**Convective boundary layer** (`test/tke_cbl_driver.jl`): free convection
over a heated free-slip floor, ``Q_s = 0.1`` K m s⁻¹, ``Γ = 3`` K km⁻¹ above
``z_{i0}``, periodic laterally, sponge in the top quarter, SSP-RK3 at the
CFL step.

| case | ``z_i`` | min heat-flux ratio | max SGS share of TKE, ``0.2 < z/z_i < 0.8`` | ``e`` peak | ``σ_w^2`` max |
| :-- | --: | --: | --: | --: | --: |
| 24³, Δ = 50 m, 1 h (CI) | 900 m | −0.157 | 0.203 | 75 m (0.08 ``z_i``) | 0.50 ``w_*^2`` at 0.31 ``z_i`` |
| 24³ with the ``κ d_{wall}`` cap | 850 m | −0.083 | 0.308 | 125 m (0.15 ``z_i``) | 0.34 ``w_*^2`` at 0.38 ``z_i`` |
| 64³, 50 × 50 × 25 m, 2 h (`examples/sgs/cbl_tke.jl`) | 1175 m | −0.115 | 0.162 | 38 m (0.03 ``z_i``) | 0.459 ``w_*^2`` at 0.35 ``z_i`` |

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

**Decaying grid turbulence** (Comte-Bellot & Corrsin, `test/era6_cbc_decay.jl`,
32³, three seeds): ``K_{98}/K_{42}`` −18.5 %, ``K_{171}/K_{42}`` −34.4 %,
RMS ``\log_{10}E`` 0.084 / 0.172, ``E(k_c)``/CBC 0.58 / 0.25: too
dissipative, for two measured reasons. Its equilibrium is Smagorinsky with
``C_s = 0.194`` (static Smagorinsky at 0.194 gives −6.3 % / −19.8 %), and
``e`` lags the fast decay: its relaxation time ``l/(c_ε\sqrt e) ≈ 0.2`` s
is close to the 0.28 s between stations, and from station 98 on the mean
``e`` is 1.8–2.6 times its equilibrium with the decaying strain, so
``ν_t`` is 1.3–1.6 times too large.

## Berlin coupling

`examples/urban/couplings/sgs_tke.jl`, provider `:tke` of the `:sgs` slot
(see [the Berlin case](../design/berlin_case.md#Numerics-slots)), with ``e``
the advected tracer `:e`, the wall cap from the building mask, and the
surface value from the MOST friction velocity. Measured numbers are in
the Berlin design page.

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
  * Maronga, B. et al. (2015), *The Parallelized Large-Eddy Simulation
    Model (PALM) version 4.0*, Geosci. Model Dev. 8, 2515–2551 (the TKE
    closure as the urban-LES default).
