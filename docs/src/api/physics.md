# Physics modules

Era I physics components layered on the dynamical core, plus the I/O
and parallel-decomposition machinery.

## I/O and checkpointing

```@autodocs
Modules = [Aeolus.IO]
```

## NetCDF reader / writer

```@autodocs
Modules = [Aeolus.NetCDFIO]
```

## Microphysics

```@autodocs
Modules = [Aeolus.Microphysics]
```

## Immersed boundaries

```@autodocs
Modules = [Aeolus.IBM]
```

## Domain decomposition

```@autodocs
Modules = [Aeolus.Parallel]
```

## Grid nesting

```@autodocs
Modules = [Aeolus.Nesting]
```

## Compute backends

```@autodocs
Modules = [Aeolus.Backends]
```

## Spectral element (future backend)

```@autodocs
Modules = [Aeolus.SpectralElement]
```

## Radiation (Era I scaffold)

Superseded for production use by the Era III stack; retained because it
is cheap and adequate for idealised cases. `SlabLand` is the first
implementer of the Era V surface interface
([SurfaceExchange](era5/surface_exchange.md)): `surface_temperature` is
the SurfaceExchange generic, and `step_surface!` computes its own
Monin–Obukhov fluxes.

```@autodocs
Modules = [Aeolus.Radiation]
```

## Chemistry

```@autodocs
Modules = [Aeolus.Chemistry]
```

## Lagrangian particles

```@autodocs
Modules = [Aeolus.Particles]
```

## Machine-learning closures

```@autodocs
Modules = [Aeolus.ML]
```

## Data assimilation

```@autodocs
Modules = [Aeolus.DataAssimilation]
```
