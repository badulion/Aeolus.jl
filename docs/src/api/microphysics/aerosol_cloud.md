# Aerosol-cloud interaction (Phase 42)

`AerosolCloud` diagnoses the cloud droplet number ``N_d`` from the
aerosol population instead of prescribing it, so the two aerosol
indirect effects are computed. Log-normal dry modes with
κ-Köhler hygroscopicity give each particle its critical supersaturation;
the updraught sets the supersaturation the parcel reaches. Two routes
from there to ``N_d`` and ``r_{\mathrm{eff}}`` are provided and compared
in the tests: the Abdul-Razzak & Ghan (2000) algebraic parameterisation
(the workhorse) and an explicit bin-resolved adiabatic parcel (the
reference). The resulting droplet number feeds the Twomey (optical
depth, albedo) and Albrecht (autoconversion) effects. The CCN spectrum
used by [`TurbulentMicrophysics`](turbulent_microphysics.md),
`ccn_fraction` / `ccn_spectrum`, is defined here.

```@autodocs
Modules = [Aeolus.AerosolCloud]
```
