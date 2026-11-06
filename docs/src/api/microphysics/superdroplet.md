# Superdroplet microphysics (Phase 41)

`Superdroplet` is the Lagrangian probabilistic reference of the clouds-and-microphysics work
(Shima et al. 2009). The droplet population is carried as computational
particles, each with a multiplicity — the number of real droplets it
stands for — plus the wet radius and dissolved solute mass those
droplets share. Collision–coalescence is a Monte Carlo process over
``\lfloor N/2 \rfloor`` random non-overlapping pairs with the
multiple-coalescence rule, so it costs ``O(N)`` and conserves water mass
to round-off; condensation is the diffusional growth equation solved per
particle. The size distribution is never assumed: it is measured from
the ensemble, which is what makes this the scheme a bulk closure (P3,
the `TurbulentMicrophysics` moment closure) is judged against. The
Golovin-kernel validation in `validation/superdroplet.jl` checks the
multiplicity bookkeeping against the analytic solution.

```@autodocs
Modules = [Aeolus.Superdroplet]
```
