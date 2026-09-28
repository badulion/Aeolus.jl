# Plant canopy — PlantCanopy

`PlantCanopy` resolves vegetation that spans several model layers
(Phase 48). It owns the leaf-area-density profiles
(`UniformLAD`, `LalicMihailovicLAD`, `BetaLAD`), the shortwave and
longwave radiation through them, the leaf energy balance and leaf
temperature, and the canopy drag
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

## Leaf temperature

The leaf energy balance turns the absorbed radiation into leaf
temperatures, sensible heat and transpiration. A single leaf:

```julia
p  = pft_photosynthesis_parameters(:bdt_temperate)
st = CoupledLeafPhysiology(p, pft_stomatal_model(:bdt_temperate); ca = 420.0)
leaf = leaf_energy_balance(st, CampbellNormanBoundaryLayer(leaf_width = 0.05);
                           R_abs = 1300.0, T_air = 298.15, e_air = 1200.0, U = 1.0,
                           Q = 1100.0)       # R_abs: SW + LW, both sides; Q: PAR photons
leaf.T_leaf - 298.15, leaf.H, leaf.LE
```

A resolved canopy, with the sunlit and shaded leaves of every layer and
their longwave exchange solved together:

```julia
nir = canopy_shortwave(forest, zf; sun = sp, sw_direct = 0.5irr.dni * sp.μ₀,
                       sw_diffuse = 0.5irr.dhi, leaves = RossGoudriaanLeaves(0.25),
                       optics = LeafOptics(0.45, 0.25), albedo_direct = 0.2)
sol = canopy_leaf_energy_balance(forest, zf; shortwave = (par, nir), par = 1,
                                 T_air = 298.15, e_air = 1500.0, U = 1.0,
                                 lw_down = 330.0, T_ground = 296.0,
                                 leaves = RossGoudriaanLeaves(0.25), stomata = st)
sol.T_sunlit, sol.T_shaded        # K, per layer
sol.heat_source, sol.vapour_source  # W m⁻³ and kg m⁻³ s⁻¹ for the air
```

Pass `T_prev = sol`, `heat_capacity` (J m⁻² K⁻¹) and `dt` to make the leaf
temperature prognostic (backward Euler). `leaf_penman_monteith` gives the
linearised solution. The theory and the comparison with measured leaf
temperatures are in
[Resolved plant canopy §5](../../theory/plant_canopy.md).

```@autodocs
Modules = [Aeolus.PlantCanopy]
```
