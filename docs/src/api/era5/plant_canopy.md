# Plant canopy — PlantCanopy

`PlantCanopy` resolves vegetation that spans several model layers
(Phase 48). It owns the leaf-area-density profiles
(`UniformLAD`, `LalicMihailovicLAD`, `BetaLAD`), the shortwave and
longwave radiation through them, and the canopy drag
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

## Radiation through the canopy

The shortwave is solved per band with the canopy two-stream of Sellers
(1985), using the Era III `TwoStream` layer formulae and adding solver;
the sun angle can come straight from `SolarGeometry`.

```julia
sp  = solar_position(solar_time(2024, 6, 21, 11, 0, 0; latitude = 52.52, longitude = 13.40))
irr = clear_sky_irradiance(sp, BirdClearSky())
zf  = collect(0.0:1.0:30.0)
par = canopy_shortwave(forest, zf; sun = sp, sw_direct = 0.5irr.dni * sp.μ₀,
                       sw_diffuse = 0.5irr.dhi, leaves = RossGoudriaanLeaves(0.25),
                       optics = LeafOptics(0.10, 0.05), albedo_direct = 0.1)
par.absorbed_sunlit ./ par.lai_sunlit      # W m⁻² of sunlit leaf, per layer
lw  = canopy_longwave(forest, zf; lw_down = 330.0, T_leaf = 298.0, T_ground = 293.0)
```

The 50 % PAR share of the broadband flux above is only for illustration.

```@autodocs
Modules = [Aeolus.PlantCanopy]
```
