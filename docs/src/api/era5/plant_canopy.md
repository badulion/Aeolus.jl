# Plant canopy — PlantCanopy

`PlantCanopy` resolves vegetation that spans several model layers
(Phase 48). It owns the leaf-area-density profiles
(`UniformLAD`, `LalicMihailovicLAD`, `BetaLAD`) and the canopy drag
``-C_d a |U| U``, applied implicitly as ``U/(1 + Δt C_d a |U|)`` — the
exact solution of the drag equation over the step, stable at any `dt`.
The theory, including the comparison of the 1-D column with the
Inoue / Harman–Finnigan exponential wind profile, is in
[Resolved plant canopy](../../theory/plant_canopy.md).

## Adding a canopy to an LES

```julia
using Aeolus
mesh = RegularGrid(; size = (64, 64, 32), extent = (1000.0, 1000.0, 200.0))
sim  = BoussinesqSimulation(mesh; bcs = FieldBoundaryConditions(:periodic, :periodic, :wall))

forest = LalicMihailovicLAD(4.0, 20.0; z_m = 12.0)       # LAI 4, 20 m tall
mask   = [i <= 32 for i in 1:64, j in 1:64]               # western half
drag   = ResolvedCanopyDrag(mesh, canopy_lad_field(mesh, forest, mask); C_d = 0.2)

step!(sim, 2.0; post_predictor! = (s, Δt) -> apply_canopy_drag!(s, drag, Δt))
```

```@autodocs
Modules = [Aeolus.PlantCanopy]
```
