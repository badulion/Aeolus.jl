# Clouds and microphysics: overview

The clouds-and-microphysics work (Phases 39–45) replaced the Kessler / two-moment scaffold with
schemes that represent cloud microstructure: a shared thermodynamic
contract, a P3 bulk scheme, a superdroplet reference, aerosol
activation, ice nucleation, fog and visibility, and sub-grid
supersaturation. This page shows how the pieces fit together, gives the
theory of the three phases that have no page of their own (39, 41, 42),
and collects the measured validation. The detailed pages are listed at
the end.

## The contract: one thermodynamics for every scheme

`MoistThermodynamics` plays the role for clouds and microphysics that
`RadiationCore` plays for radiation: every scheme takes temperature, saturation, latent
heat and buoyancy from it, so two schemes cannot disagree about whether
a parcel is saturated.

**Latent heat and saturation are consistent by construction.** The
latent heat follows Kirchhoff's relation,

```math
L_v(T) = L_{v0} + (c_{pv} - c_l)(T - T_0),
```

and the saturation vapour pressure is the analytic integral of the
Clausius–Clapeyron equation ``d\ln e_s/dT = L_v/(R_v T^2)`` with that
same ``L_v(T)``:

```math
\ln\frac{e_s(T)}{e_0} = \frac{L_{v0} - (c_{pv}-c_l)T_0}{R_v}
  \left(\frac{1}{T_0} - \frac{1}{T}\right)
  + \frac{c_{pv}-c_l}{R_v}\ln\frac{T}{T_0},
```

and the same over ice with ``L_s`` and ``c_i``. A fitted formula
(Tetens, Bolton) is accurate near 273 K and drifts in the cold upper
troposphere, which is where ice microphysics needs it. The residual of
Clausius–Clapeyron is 5.2e-12 (liquid) and 4.2e-12 (ice); the
liquid–ice gap peaks at 27.00 Pa at 261.5 K. Extrapolated 100 K to the
boiling point with nothing fitted, ``e_s`` reaches 99,754 Pa, 1.55 % low:
the constant-heat-capacity assumption, recorded rather than tuned.

**Two saturation adjustments, for two different questions** (#82):

* `saturation_adjustment` is a **diagnosis**: given the prognostic
  ``(θ_{li}, q_t)``, which partition into vapour and condensate is
  consistent with them? It conserves ``θ_{li}`` and is what a scheme
  that carries ``θ_{li}`` needs (`TurbulentMicrophysics`, CAPE / CIN).
* `isobaric_saturation_adjustment` is a **process**: condensation at
  constant pressure, which conserves the moist enthalpy referenced to
  liquid water (`moist_enthalpy`),
  ``h = (c_{pd} q_d + c_l q_t)\,T + L_v(T)\,q_v - L_f(T)\,q_i``, so that
  condensing ``Δq`` warms the parcel by exactly
  ``c_{pm}ΔT = L_v(T_0)\,Δq``. P3, the fog adjustment and the fog
  column use it.

Using the diagnosis as a process overstates the latent heating by 1.2 %
at 900 hPa and 6.4 % at 600 hPa; most of that gap comes from the Exner
exponent ``R_m/c_{pm}`` taken at the current partition. Moist static
energy is the invariant of neither: it misses 24 % of ``L\,Δq`` in an
isobaric condensation.

**Köhler theory lives here too.** Three modules (AerosolOptics,
Superdroplet, AerosolCloud) each carried their own; they had drifted far
enough (a surface tension of 0.0728 against 0.0720 N m⁻¹ at 298 K, 1.2 %
in critical supersaturation) to disagree about whether a particle
activates. The κ-Köhler form of Petters & Kreidenweis is now the one
implementation, and the three agree to 1.6e-12.

## Superdroplets (Phase 41)

`Superdroplet` follows Shima et al. (2009). Each superdroplet stands for
``ξ`` identical real droplets. In a cell with ``N_s`` superdroplets,
``\lfloor N_s/2 \rfloor`` random pairs are tested per step instead of all
``N_s(N_s-1)/2``; the pair probability is scaled up by the ratio of the
two counts. In a successful collision with ``ξ_j ≥ ξ_k`` each of the
``ξ_k`` droplets of ``k`` absorbs one droplet of ``j`` and ``ξ_j`` drops by
``ξ_k`` ("all or nothing"), so mass is conserved exactly. Diffusional growth is integrated implicitly in ``r^2``
with Kelvin and Raoult terms. Against Golovin's analytic solution for the
additive kernel the ``L^1`` error is 0.037 at 1200 s, 0.080 at 2400 s,
0.118 at 3600 s (2¹⁵ superdroplets), mass is conserved to 5e-15, and the
error falls with ``N_s`` — a wrong multiplicity rule would converge to a
wrong answer instead.

## Aerosol activation (Phase 42)

`AerosolCloud` uses Abdul-Razzak & Ghan (2000): the peak supersaturation
of a rising parcel from a closed-form balance between the supersaturation
source ``αw`` and the condensation sink of the activating modes, and the
activated fraction of each log-normal mode from its critical
supersaturation. An explicit adiabatic parcel model is the reference: a
particle activates when its critical supersaturation lies below the
parcel's ``S_{max}``, because crossing the Köhler barrier is irreversible
for an ascending parcel (testing whether the droplet has *grown* past
``r_c`` at the instant of ``S_{max}`` reported zero activation in every
case — the first bug this module's tests caught). Over w = 0.1–5 m s⁻¹
and N_a = 100–1000 cm⁻³ ARG under-predicts ``S_{max}`` by 8–24 % and the
activated fraction agrees with the parcel to 1–25 % (mean 7 %), the
accuracy the literature claims for it.

## How the schemes connect

| scheme | module | takes from | gives to |
| :-- | :-- | :-- | :-- |
| P3 ice and rain (40) | `P3Microphysics` | contract adjustment, Superdroplet fall speeds and growth resistances, AerosolCloud autoconversion | Kessler fall speeds and sedimentation (44) |
| superdroplets (41) | `Superdroplet` | contract Köhler | the reference for 45; Stokes settling for INP (43) |
| activation (42) | `AerosolCloud` | contract Köhler, AerosolOptics modes | CCN spectrum for 45 |
| ice nucleation (43) | `IceNucleation` | Phase 37 aerosol modes, contract liquid fraction | nucleated ice into P3 |
| precipitation fraction (44) | `P3Microphysics` | P3 process rates | — |
| fog and dew (44) | `RadiationFog` | contract process adjustment, the radiation stack's gas and cloud optics | the urban case's `fog_dew` coupling |
| visibility (44) | `Visibility` | AerosolOptics Mie, GasOptics Rayleigh | — |
| sub-grid supersaturation (45) | `TurbulentMicrophysics` | contract diagnosis, AerosolCloud CCN spectrum, Superdroplet growth | — |

## Measured validation

| phase | check | result | reference |
| :-- | :-- | :-- | :-- |
| 39 | Clausius–Clapeyron residual | 5.2e-12 | exact |
| 39 | process adjustment, 14,880 states | max \|Δh\|/h 4.4e-16 | exact |
| 40 | rain reflectivity at 5 mm h⁻¹ | 33.93 dBZ | 34.19 (Marshall–Palmer) |
| 40 | fall speed of a 1 mm drop | 3.983 m s⁻¹ | 4.03 (Gunn–Kinzer) |
| 41 | Golovin coalescence, ``L^1`` at 1200 s | 0.037 | analytic |
| 42 | ARG activated fraction vs parcel | mean 7 % | parcel model |
| 43 | settling of a 10 μm particle | 0.304 cm s⁻¹ | 0.31 (Seinfeld & Pandis) |
| 43 | Arctic stratus initial LWP | 14.38 g m⁻² | ~14 (Paukert & Hoose 2014) |
| 44 | fog visibility at LWC 0.1 / 0.2 / 0.3 g m⁻³ | 171 / 108 / 83 m | 205 / 111 / 78 (Kunkel 1984) |
| 44 | ParisFog 2011 maximum LWP | 92 g m⁻² | ~70 (observed) |
| 45 | spectral width growth exponent | 0.5155 (superdroplets) | 0.5142 (OU), ½ (DNS) |

What is **not** validated against observations, and why, is recorded
per phase in the [long-range roadmap](../design/longterm_roadmap.md):
no P3 reference-code comparison, no cloud-chamber or disdrometer data,
the ISDAC stratus ice water path not compared (the paper was not
accessible), and the DNS target met by the analytic Ornstein–Uhlenbeck
law rather than digitised DNS data.

## Detailed pages

* [P3 microphysics](p3.md)
* [Ice nucleation](ice_nucleation.md)
* [Sub-grid cloud PDF](subgrid_cloud.md)
* [SGS activation and growth](sgs_activation_growth.md)
* [Spectral broadening](spectral_broadening.md)
* [Visibility (Koschmieder)](visibility.md)
* [Fog and dew](fog.md)
* [Radiation fog](radiation_fog.md)
