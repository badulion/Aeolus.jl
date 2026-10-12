# The Berlin case: air quality

## Air quality: chemistry and deposition (#135)

The `:chemistry` slot's default provider `:nox_o3_voc` (`aq_chem.jl`)
runs the `Aeolus.Chemistry` mechanism ([theory](../theory/urban_chemistry.md))
on five tracers registered with `advect = true`: NO, NO2, O3, VOC and
HNO3, in ppb. The `:transport` stage moves them. The example's toy
mechanism, its deposition table and its dead `apply_wet_deposition!`
are gone. What stays in the coupling is case data: the emission
inventory, the class → surface-kind map and the roughness per class.

Every step in `:chemistry`, in this order:

1. Emissions go into the first air cell `k_air` of each column (above the
   roof on building columns). Traffic NOx is emitted on street and
   building-adjacent columns: 2 µg m⁻² s⁻¹ as NO2 at the morning peak,
   10 % primary NO2, with an assumed diurnal shape (`traffic_profile`).
   Isoprene is emitted on park columns: 0.3 µg m⁻² s⁻¹ in full sun,
   scaled by J(NO2). Both are order-of-magnitude estimates, not a
   Berlin inventory.
2. `apply_photochemistry!` with the clear-sky `j_NO2(μ₀)` of the current
   sun, OH prescribed as 5 × 10⁶ cm⁻³ × J/J_max, T = θ̄ Π̄ (θ̄ the
   level mean over air cells) and the reference p̄ per level (`case.ref`,
   #131).
3. Dry deposition of NO, NO2, O3 and HNO3 from `k_air` with
   ``v_d = 1/(r_a + r_b + r_c)``.
4. Rain scavenging of HNO3 by `qr`.

In `:surface_update`, on the surface cadence, `r_a = 1/(C_H U)` and `u*`
are recomputed per column. They come from
`SurfaceExchange.solve_surface_layer` between the skin `bus.Ts` and the
first air cell (cell-centre wind, θ, qv at `z = Δz/2`), with
`z0h = z0m/10` and the per-class `z0m` that #129 gives its surfaces.
Since #129 (PR #158) the bus carries u* (`bus.u_star`) and every
column's surface carries its roughness, but this coupling still runs
its own surface-layer solve; reading them from the bus is open.

`physics[:chemistry] = :off` (provider of `NoChem`) removes the species.
It is not the default. The smoke matrix also runs a `night titration`
variant (start 00:30, J = OH = 0).

### Measured (flagship grid 64 × 64 × 32, 20 June, 12:00–12:30)

Remeasured on master (`915bd49`: reference state, MOST / soil surfaces
of #129, adaptive step of #166), `CaseConfig(:alexanderplatz; nsteps =
900, start_h = 12)` with the default couplings, with and without
`Ts0 = 300.0`, run to 1800 s of simulated time (962 / 1079 adaptive
steps; at #135 and in the first remeasurement 900 steps of 2 s). Street / roof /
park are surface classes 1 / 2 / 3; r_a, u* and v_d are medians over
the class; fluxes and emissions are means over the ledger's elapsed time.

| quantity | default (`Ts0` = 285 K) | warm surfaces (`Ts0` = 300 K) |
| :-- | --: | --: |
| Leighton ratio φ, median over air cells with NO, O3 > 0.01 ppb, at 12:30 | 1.0013 | 1.0013 |
| φ / φ_ext (peroxy-radical steady state), median | 0.9964 | 0.9963 |
| r_a street / roof / park, median (s m⁻¹) | 89.5 / 5.6e6 / 5.76e5 | 89.6 / 88.4 / 8.48e5 |
| u* street / park, median (m s⁻¹) | 0.094 / 0.0016 | 0.093 / 0.0013 |
| v_d O3 park / street (mm s⁻¹) | 0.002 / 0.464 | 0.001 / 0.464 |
| v_d NO2, HNO3 street (mm s⁻¹) | 1.53, 5.78 | 1.52, 5.74 |
| mean dry deposition flux O3 / NO2 / HNO3 (nmol m⁻² s⁻¹) | 0.316 / 0.320 / 0.063 | 0.507 / 0.382 / 0.093 |
| mean emission NO / NO2 / isoprene (nmol m⁻² s⁻¹) | 17.84 / 1.98 / 0.504 | same |
| surface-layer NO / NO2 / O3 at 12:30 (ppb) | 7.74 / 10.66 / 35.55 | 7.50 / 10.45 / 35.55 |

The default flagship starts its surfaces (the soil columns and slabs of
#129) 8 K below the air (`Ts0` = 285 K). At step 0 every class is stably
stratified, r_a 6.2e3–7.9e3 s m⁻¹ and u* ≈ 0.015 m s⁻¹, short of the
clamp. The street soil skins heat within minutes, and by 12:30 the
streets are unstable (r_a 89 s m⁻¹, v_d O3 0.46 mm s⁻¹): the domain-mean
O3 deposition is 0.32 nmol m⁻² s⁻¹, 17 × its value with the pre-#129
slabs, when r_a reached 1e5–1e9 s m⁻¹ and dry deposition was
effectively off. Roofs and park stay decoupled (r_a 5.6e6 and 5.8e5
s m⁻¹; mean Ts at 12:30 291.2 K on roofs and 291.05 K in the park, under
first-cell air of about 294 K; streets 302.8 K). Warm surfaces do not
change that for the park: its skin cools from 300 to about 291 K within
the half hour and decouples (r_a 1.0e3–1.05e9 s m⁻¹), so the PR #162
expectation that the default case would look like the right-hand column
holds for the streets but not for the park. The periodic domain has no
inflow of background air, so emitted NOx accumulates: surface-layer NO
rises from 1.68 to 7.74 ppb in 30 min.

Wet deposition, measured in the smoke window with `qr = 1e-4` seeded in
levels 2–4: the HNO3 removed per step equals ``s[\mathrm{HNO_3}]\,(Σ q_r Δz)\,Δt``
to 0.1 %. The rain sediments out in about four steps.

**Cost.** On the flagship grid on master `915bd49` (600 s from 12:00,
with and without the chemistry (`aq_chem.enabled=false`,
`chem_off.enabled=true`), run one after the
other on a loaded 24-core machine, first-call compilation included; both
runs take the same 230 adaptive steps, mean dt 2.61 s, since the
chemistry does not feed back on the flow) the step goes from 233.7 to
289.1 ms (wall time over steps, i.e. 89.6 → 110.8 ms per simulated
second; 190.5 → 248.8 ms summed over the stage timings), so the
chemistry adds about 24–31 %. The WENO-Z5 transport of the five species
adds 50.0 ms (10.0 ms per tracer: `scalar_advection` 33.7 → 83.7 ms per
step); the `aq_chem` stage costs 12.1 ms (92 ns per cell) including
emissions and deposition; the r_a refresh 3.3 ms averaged over steps.
The absolute numbers depend on the load: a first pair run alongside
five other measurements gave 325.7 → 509.0 ms per step. (Remeasured at
`2895fed` with 300 steps of 2 s: 135.5 → 169.2 ms. At #135, before
#129, the step went from 43.1 to 101.1 ms.) The chemistry stays on by default because
the flagship is meant to exercise all the physics, and `:off` exists for
runs that do not need air quality.
