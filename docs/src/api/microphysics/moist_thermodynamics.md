# Shared contract — MoistThermodynamics (Phase 39)

`MoistThermodynamics` is to Era IV what `RadiationCore` is to Era III:
every microphysics scheme, activation parameterisation and cloud
diagnostic in Phases 40–45 speaks it, so they agree on what temperature,
saturation and buoyancy mean. It is *exact* in three respects that LES
codes routinely approximate: the latent heats vary with temperature
through Kirchhoff's relation, the saturation vapour pressure is the
analytic integral of Clausius–Clapeyron with that same ``L(T)`` rather
than a Tetens/Magnus fit, and the moist heat capacity ``c_{pm}`` carries
the vapour and condensate loading. Shared quantities that several
schemes compute — for instance `effective_radius`, which
`TurbulentMicrophysics` extends — are declared here as generics that the
implementing modules extend.

```@autodocs
Modules = [Aeolus.MoistThermodynamics]
```
