# Era III — High-fidelity radiation

The theory behind these modules is developed in
[Radiative transfer](../theory/radiation.md); this page is the
generated API reference.

## Shared contract

`RadiationCore` fixes the interface every component in this era speaks:
the spectral grid and its g-points, the ``\tau``/``\omega``/``g``
optical-property triple with its correct combination rule, level-based
radiative fluxes, and the abstract solver interfaces.

```@autodocs
Modules = [Aeolus.RadiationCore]
```

## Correlated-k gas optics (Phase 32)

```@autodocs
Modules = [Aeolus.GasOptics]
```

## Two-stream solver and cloud optics (Phase 33)

```@autodocs
Modules = [Aeolus.TwoStream]
```

## 3-D Monte-Carlo radiative transfer (Phase 34)

```@autodocs
Modules = [Aeolus.MonteCarloRT]
```

## Urban radiosity (Phase 35)

```@autodocs
Modules = [Aeolus.Radiosity]
```

## Spectral albedo and BRDF (Phase 36)

```@autodocs
Modules = [Aeolus.SurfaceOptics]
```

## Aerosol optics (Phase 37)

```@autodocs
Modules = [Aeolus.AerosolOptics]
```

## Solar geometry (Phase 38)

```@autodocs
Modules = [Aeolus.SolarGeometry]
```
