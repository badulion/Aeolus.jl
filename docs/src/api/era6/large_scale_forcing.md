# Large-scale forcing — LargeScaleForcing

`LargeScaleForcing` holds the prescribed forcing of a case whose larger
scales are not resolved: the geostrophic pressure gradient, nudging of
the plane mean of u, v, θ, qv (`ProfileNudging`; `GeostrophicNudging` is
its constructor), large-scale subsidence (`Subsidence`), prescribed
large-scale advective tendencies and a prescribed surface-temperature
series. Every quantity is a
number, a height profile or a height-time [`ForcingTable`](@ref
Aeolus.LargeScaleForcing.ForcingTable), and every forcing is applied
through one entry point, `apply_forcing!(sim, forcing, t, Δt)`, from the
`body_force!` hook of `step!`. Theory and the measured validation are in
[Large-scale forcing](../../theory/large_scale_forcing.md).

```julia
using Aeolus
mesh = RegularGrid(; size = (32, 32, 64), extent = (1000.0, 1000.0, 400.0))
sim  = BoussinesqSimulation(mesh; f_coriolis = 1.39e-4, C_s = 0.17, θ₀ = 263.5)

# GABLS1: U_g = 8 m/s, cooling surface (the T_s series feeds a wall law)
pg  = GeostrophicPressureGradient(8.0, 0.0)
Ts  = SurfaceTemperatureSeries([0.0, 9 * 3600.0], [265.0, 262.75])

# a height-time table: U_g backing with height and time
Ug  = ForcingTable([0.0, 1000.0], [0.0, 86400.0], [5.0 3.0; 8.0 6.0])
adv = AdvectiveTendencies(; θ = ForcingTable([0.0, 400.0], [-1e-5, -2e-5]))  # K/s

step!(sim, 1.0; body_force! = (s, Δt) -> apply_forcing!(s, (GeostrophicPressureGradient(Ug), adv), s.t, Δt))
surface_temperature(Ts, 3600.0)     # 264.75 K

# free-troposphere θ nudging above 1.2 z_i (τ = 6 h) and subsidence w_s = -D z
zi(s) = 800.0                                        # any boundary-layer-height diagnostic
nud = ProfileNudging((θ = ForcingTable([0.0, 3000.0], [290.0, 302.0]),), 6 * 3600.0;
                     weight = NudgingWeight(zi; factor = 1.2, depth = 200.0))
sub = Subsidence((z, t) -> -3e-6 * z; fields = (:θ,))
I = apply_forcing!(sim, sub, sim.t, 1.0)             # (θ = ∫ (∂θ/∂t)_s dV,) in K m³ s⁻¹
```

The Berlin case runs `GeostrophicPressureGradient` by default
(`:geostrophic` coupling, `mode = :pressure_gradient`) and
`GeostrophicNudging` as the option `mode = :nudging`
([The Berlin case](../../design/berlin_case.md)).

```@autodocs
Modules = [Aeolus.LargeScaleForcing]
```
