# Prognostic SGS TKE and Reynolds stress — PrognosticSGS

`PrognosticSGS` is the Deardorff (1980) 1.5-order closure with a
prognostic subgrid-scale turbulent kinetic energy ``e`` and the Moeng
(1984) coefficients (Phase 56, P56-b, #182). `TKEClosure` is an
`AbstractSGSClosure` of the P54-a contract, so every core and time
stepper takes it as `sgs = closure`: `sgs_coefficients!` reads ``e``,
`advance_sgs!` integrates the TKE equation once per accepted step, and a
step `step_adaptive!` rejects is rolled back exactly. The theory, the
discrete duality of the production terms and the measured validation are
on [Prognostic SGS TKE (Deardorff–Moeng)](../../theory/sgs_tke.md).

The mixing length is capped at ``κ d`` with the distance of
`IBM.wall_distance`, and Moeng's factor 3.9 on ``c_ε`` applies at
`IBM.first_level_cells` (both documented with `Aeolus.IBM` on
[Physics modules](../physics.md)).

```julia
using Aeolus
m   = RegularGrid(; size = (32, 32, 32), extent = (1600.0, 1600.0, 1600.0), halo = (3, 3, 3))
sim = BoussinesqSimulation(m; θ₀ = 300.0, θ_flux_bottom = 0.1,
                           bcs = FieldBoundaryConditions(:periodic, :periodic, :wall),
                           θ_bcs = FieldBoundaryConditions(:periodic, :periodic, :noflux))
tke = TKEClosure(m; first_level = first_level_cells(SolidMask(m)))
initialize_tke!(tke, sim; e_min = 1e-3)      # e = 0 is a fixed point: seed it
step_rk3!(sim, 2.0; sgs = tke)
tke_budget(tke, sim)                          # Σ P_s V, Σ P_b V, Σ ε V, transport, Σ e V
```

`ReynoldsStressClosure` (P56-d, #188) carries the six subgrid stresses
``τ_{ij}`` instead: exact production, Daly–Harlow buoyancy through the
P55-c tensor operator, exact Coriolis, LRR-IP pressure–strain, the
momentum flux ``−∂_jτ^{dev}_{ij}`` through the P55-b stress operator, and
Schumann realizability, with ``l``, ``ε`` and ``ν_t`` the TKE closure's
functions (theory: the Reynolds-stress section of the same page).

```julia
rsm = ReynoldsStressClosure(m; first_level = first_level_cells(SolidMask(m)))
initialize_reynolds_stress!(rsm, sim; e_min = 1e-3)   # isotropic seed
step_rk3!(sim, 2.0; sgs = rsm)                         # not step_imex!: sgs_explicit_stress
reynolds_stress_budget(rsm, sim)                       # Σ −τ:∇u V, Σ βh₃ V, Σ ε V, Σ e V
reynolds_stress_statistics(rsm)                        # realizability corrections, booked
```

```@autodocs
Modules = [Aeolus.PrognosticSGS]
```
