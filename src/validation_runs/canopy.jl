# Within-canopy wind and temperature profiles (Phase 48, #25): the 1-D
# PlantCanopy column the `plant_canopy` run type solves (examples/validation/configs/canopy_profiles.toml). Moved from
# examples/validation/canopy_profiles.jl (EX-c #440), where it ran in the
# script; the published profiles it is compared with are in
# test/data/canopy_profiles.jl.

using ..MoistThermodynamics: DEFAULT_CONSTANTS, R_d, saturation_vapor_pressure

"Linear interpolation of the profile `vs(zs)` at `z` (clamped to the end segments)."
function _canopy_interp(zs, vs, z)
    i = clamp(searchsortedlast(zs, z), 1, length(zs) - 1)
    t = (z - zs[i]) / (zs[i+1] - zs[i])
    return (1 - t) * vs[i] + t * vs[i+1]
end

"""
    canopy_wind_profile(p::AbstractLeafAreaDensity, h; C_d, ztop, N = 400, β = 0.3, u_top = 5.0)
        -> NamedTuple

The steady 1-D canopy wind of `solve_canopy_wind_column` on `N` layers
from the ground to `ztop` (m), for the leaf-area density `p` of a canopy
of height `h` (m): drag `C_d`, a no-slip ground, the wind `u_top` at the
top, and the Harman & Finnigan (2007) mixing length
``l_c = 2β^3/(C_d\\,\\mathrm{LAI}/h)`` (`harman_finnigan_mixing_length`,
wall-limited by `canopy_mixing_length`). Nothing is fitted. Returns
`sol` (the column solution: `z`, `u`, `u_star`, …), `lad` (per layer),
`zf` (the faces), `l_c`, `U_h` (the wind at `h`), `τ` (the kinematic
stress ``K\\,∂U/∂z`` at the faces: the ground stress at face 1, ``u_*^2``
at the top), `τ_h` (the stress at `h`) and `d` (the displacement height,
`canopy_displacement_height`).
"""
function canopy_wind_profile(p, h; C_d, ztop, N = 400, β = 0.3, u_top = 5.0)
    zf = collect(range(0, ztop; length = N + 1))
    lad = layer_leaf_area_density(p, zf)
    l_c = harman_finnigan_mixing_length(β, C_d, leaf_area_index(p) / h)
    sol = solve_canopy_wind_column(zf, lad; C_d = C_d, u_top = u_top, bottom = :no_slip,
                                   l_mix = z -> canopy_mixing_length(z; h = h, l_c = l_c))
    Nz = length(sol.u)
    τ = [f == 1 ? sol.tau_ground :
         f == Nz + 1 ? sol.u_star^2 :
         sol.K[f] * (sol.u[f] - sol.u[f-1]) / (sol.z[f] - sol.z[f-1]) for f in 1:Nz+1]
    return (; sol, lad, zf, l_c, U_h = _canopy_interp(sol.z, sol.u, h), τ, τ_h = _canopy_interp(zf, τ, h),
            d = canopy_displacement_height(sol, lad; C_d = C_d))
end

"""
    canopy_temperature_profile(p, h; C_d, u_star, T_ref, rh, μ, S_dir, S_dif, lw_down, gs,
                               ztop, N, β = 0.3, Sc = 0.5, u_top = 5.0, chi_l = 0.01,
                               leaf_width = 0.01, tol = 1e-7, maxiter = 100) -> NamedTuple

The in-canopy air temperature of the canopy `p` (height `h`): the wind of
[`canopy_wind_profile`](@ref) scaled to the friction velocity `u_star`,
the Harman & Finnigan (2008) scalar diffusivity (Schmidt number `Sc`),
and the leaf energy balance of every layer (`canopy_leaf_energy_balance`:
sunlit and shaded leaves of a Ross–Goudriaan distribution `chi_l`, the
two-stream shortwave of `canopy_shortwave` in the visible (45 %, leaf
optics 0.07 / 0.05, ground albedo 0.10) and the near infrared (55 %,
0.35 / 0.10, 0.20) for a sun at cos zenith `μ` (none at `μ = 0`), the
longwave `lw_down`, stomatal conductance `gs`, the Campbell & Norman
boundary layer of leaves `leaf_width` wide), solved with the steady
scalar column (`solve_canopy_scalar_column`, `T_ref` at the top, no
ground flux) and iterated until the air temperature the leaves see is
the one their heat produces (`tol` K, at most `maxiter` sweeps).
The air holds `rh` of saturation at `T_ref`. Returns `z` (layer
centres), `ΔT` (the temperature minus `T_ref`), `iterations`, and the
pieces of the last sweep: `Ta` (the air temperature), `leaf` (the leaf
energy balance), `col` (the scalar column), `K` (the diffusivity at the
faces) and `ρcp` (J m⁻³ K⁻¹).
"""
function canopy_temperature_profile(p, h; C_d, u_star, T_ref, rh, μ, S_dir, S_dif, lw_down, gs,
                                    ztop, N, β = 0.3, Sc = 0.5, u_top = 5.0, chi_l = 0.01,
                                    leaf_width = 0.01, tol = 1e-7, maxiter = 100)
    m = canopy_wind_profile(p, h; C_d = C_d, ztop = ztop, N = N, β = β, u_top = u_top)
    U = m.sol.u .* (u_star / m.sol.u_star)
    K = [harman_finnigan_scalar_diffusivity(z; h = h, u_star = u_star, l_mix = m.l_c,
                                            β = β, Sc = Sc) for z in m.zf]
    c = DEFAULT_CONSTANTS
    ρcp = 101325 / (R_d(c) * T_ref) * c.cp_dry
    ea = rh * saturation_vapor_pressure(T_ref)
    leaves = RossGoudriaanLeaves(chi_l)
    sw = μ > 0 ? (canopy_shortwave(m.zf, m.lad; sun = μ, sw_direct = 0.45S_dir, sw_diffuse = 0.45S_dif,
                                   optics = LeafOptics(0.07, 0.05), albedo_direct = 0.10, leaves = leaves),
                  canopy_shortwave(m.zf, m.lad; sun = μ, sw_direct = 0.55S_dir, sw_diffuse = 0.55S_dif,
                                   optics = LeafOptics(0.35, 0.10), albedo_direct = 0.20, leaves = leaves)) :
         nothing
    Ta = fill(Float64(T_ref), N); leaf = nothing; col = nothing; iters = 0
    for it in 1:maxiter
        iters = it
        leaf = canopy_leaf_energy_balance(m.zf, m.lad; shortwave = sw, par = sw === nothing ? nothing : 1,
                                          T_air = Ta, e_air = ea, U = max.(U, 0.05), leaves = leaves,
                                          lw_down = lw_down, T_ground = Ta[1], stomata = gs,
                                          boundary_layer = CampbellNormanBoundaryLayer(leaf_width = leaf_width))
        col = solve_canopy_scalar_column(m.zf, K, leaf.heat_source ./ ρcp; top_value = T_ref,
                                         bottom_flux = 0.0)
        δ = maximum(abs.(col.c .- Ta)); Ta .= col.c
        δ < tol && break
    end
    return (; z = col.z, ΔT = Ta .- T_ref, iterations = iters, Ta, leaf, col, K, ρcp)
end

"""
    canopy_lad(c::ConfigSection) -> AbstractLeafAreaDensity

The leaf-area density of the `[canopy]` section of the `plant_canopy`
case: `UniformLAD(ah, h)` (`lad = "uniform"`), or `TabulatedLAD` at the
heights `lad_z` (fractions of `h`) of the values `lad_values`, read as
a·h (`lad_units = "ah"`, divided by `h`) or as m² m⁻³ (`"m2_m3"`).
"""
function canopy_lad(c)
    c.lad === :uniform && return UniformLAD(c.ah, c.h)
    vals = c.lad_units === :ah ? c.lad_values ./ c.h : c.lad_values
    return TabulatedLAD(c.lad_z .* c.h, vals)
end

# ---------------------------------------------------------------------------
# the `plant_canopy` run type
# ---------------------------------------------------------------------------

"""
    PlantCanopyDriver()

The `plant_canopy` run type: `[run] driver = "plant_canopy"`, the steady
1-D canopy column of [`canopy_wind_profile`](@ref) (`[canopy] mode =
"wind"`) or [`canopy_temperature_profile`](@ref) (`"temperature"`) for a
canopy of uniform or tabulated leaf-area density ([`canopy_lad`](@ref)).
Its profiles go to `output/results.toml`. Config:
`examples/validation/configs/canopy_profiles.toml` (the three published
canopies of Phase 48, #25).
"""
struct PlantCanopyDriver <: AbstractRunDriver end

Config.driver_name(::PlantCanopyDriver) = "plant_canopy"
Config.config_dir(::PlantCanopyDriver) = VALIDATION_CONFIG_DIR

function Config.driver_sections(::PlantCanopyDriver)
    c = section("canopy"; doc = "The canopy and its 1-D column (PlantCanopy).") do s
        option!(s, "mode", Symbol; default = :wind, allowed = [:wind, :temperature],
                doc = "wind: the wind profile (canopy_wind_profile); temperature: the in-canopy air temperature of the coupled leaf energy balance (canopy_temperature_profile)")
        option!(s, "h", Float64; default = 1.0, range = (0, Inf), open_lower = true, units = "m", doc = "canopy height")
        option!(s, "lad", Symbol; default = :uniform, allowed = [:uniform, :table],
                doc = "uniform: a·h = ah below h; table: lad_values at the heights lad_z")
        option!(s, "ah", Float64; units = "1", default = 0.5, range = (0, Inf),
                doc = "a·h of the uniform canopy (frontal or leaf area per volume times h)")
        option!(s, "lad_z", Vector{Float64}; units = "1", default = Float64[], range = (0, Inf),
                doc = "heights of the table, as fractions of h, increasing")
        option!(s, "lad_values", Vector{Float64}; units = "1 or m^2 m^-3", default = Float64[], range = (0, Inf),
                doc = "the table's leaf-area density, in the units of lad_units")
        option!(s, "lad_units", Symbol; default = :ah, allowed = [:ah, :m2_m3],
                doc = "ah: a·h (divided by h); m2_m3: m² m⁻³")
        option!(s, "c_d", Float64; units = "1", default = 0.2, range = (0, Inf), open_lower = true,
                doc = "leaf drag coefficient")
        option!(s, "ztop_over_h", Float64; units = "1", default = 2.0, range = (1, Inf),
                doc = "top of the column, in h (when ztop is \"none\")")
        option!(s, "ztop", Optional{Float64}; units = "m", range = (0, Inf), open_lower = true,
                doc = "top of the column; \"none\": ztop_over_h h")
        option!(s, "layers", Int; default = 400, range = (4, 100_000), doc = "layers of the column")
        option!(s, "beta", Float64; units = "1", default = 0.3, range = (0, Inf), open_lower = true,
                doc = "β = u*/U_h of the Harman & Finnigan mixing length")
        option!(s, "u_top", Float64; units = "m s^-1", default = 5.0, range = (0, Inf), open_lower = true,
                doc = "wind at the column top (the profile is normalised; temperature mode rescales it to u_star)")
    end
    t = section("canopy_energy"; doc = "The leaf energy balance of the temperature mode (canopy_temperature_profile).") do s
        option!(s, "u_star", Float64; units = "m s^-1", default = 0.4, range = (0, Inf), open_lower = true,
                doc = "friction velocity above the canopy")
        option!(s, "t_ref", Float64; units = "K", default = 293.15, range = (0, Inf), open_lower = true,
                doc = "air temperature at the column top")
        option!(s, "rh", Float64; units = "1", default = 0.5, range = (0, 1), doc = "relative humidity (of saturation at t_ref)")
        option!(s, "cos_zenith", Float64; units = "1", default = 0.0, range = (0, 1),
                doc = "cosine of the solar zenith angle (0: night, no shortwave)")
        option!(s, "sw_direct", Float64; units = "W m^-2", default = 0.0, range = (0, Inf), doc = "direct shortwave above the canopy")
        option!(s, "sw_diffuse", Float64; units = "W m^-2", default = 0.0, range = (0, Inf), doc = "diffuse shortwave above the canopy")
        option!(s, "lw_down", Float64; units = "W m^-2", default = 330.0, range = (0, Inf), doc = "downwelling longwave above the canopy")
        option!(s, "g_s", Float64; units = "m s^-1", default = 0.1, range = (0, Inf), doc = "stomatal conductance")
        option!(s, "schmidt", Float64; units = "1", default = 0.5, range = (0, Inf), open_lower = true,
                doc = "turbulent Schmidt number of the scalar diffusivity")
        option!(s, "chi_l", Float64; units = "1", default = 0.01, range = (-0.4, 0.6),
                doc = "Ross–Goudriaan leaf-angle index χ_L (0.01: CLM needleleaf evergreen)")
        option!(s, "leaf_width", Float64; units = "m", default = 0.01, range = (0, Inf), open_lower = true,
                doc = "leaf width of the Campbell & Norman boundary layer")
    end
    return Section[c, t]
end

function _canopy_checks(cfg)
    e = no_checkpoints(cfg)
    c = cfg.canopy
    if c.lad === :table
        (length(c.lad_z) >= 2 && length(c.lad_z) == length(c.lad_values)) ||
            push!(e, ConfigError("canopy.lad_z", "a table needs lad_z and lad_values of the same length ≥ 2"))
        all(>(0), diff(c.lad_z)) || push!(e, ConfigError("canopy.lad_z", "lad_z must increase"))
    else
        isempty(c.lad_z) && isempty(c.lad_values) ||
            push!(e, ConfigError("canopy.lad_z", "lad_z and lad_values belong to lad = \"table\""))
    end
    if c.mode === :wind
        append!(e, [ConfigError("canopy_energy.$(o.key)", "[canopy_energy] belongs to mode = \"temperature\": it would be ignored")
                    for o in only(s for s in Config.driver_sections(PlantCanopyDriver()) if s.name == "canopy_energy").options
                    if !isequal(cfg.canopy_energy[o.key], o.default)])
    end
    return e
end
Config.driver_checks(::PlantCanopyDriver) = Any[
    """`[checkpoint]` stays off: the column is one steady solve
    `lad = "table"` has `lad_z` (increasing) and `lad_values` of one length ≥ 2; `lad = "uniform"` has neither
    `[canopy_energy]` keeps its defaults unless `mode = "temperature"`""" => _canopy_checks]

"What the `plant_canopy` run type builds and runs: `lad` (the leaf-area density), `profile` (the result) and `cfg`."
mutable struct PlantCanopyRun
    lad::Any
    profile::Any
    cfg::RunConfig
end

Config.build(::PlantCanopyDriver, cfg::RunConfig) = PlantCanopyRun(canopy_lad(cfg.canopy), nothing, cfg)

function Simulations.run!(::PlantCanopyDriver, r::PlantCanopyRun, rf::RunFolder, ::CheckpointSchedule;
                          callback = nothing)
    c = r.cfg.canopy
    ztop = something(c.ztop, c.ztop_over_h * c.h)
    if c.mode === :wind
        m = canopy_wind_profile(r.lad, c.h; C_d = c.c_d, ztop = ztop, N = c.layers, β = c.beta, u_top = c.u_top)
        r.profile = m
        @info(@sprintf("canopy wind: U_h = %.4g m/s, u* = %.4g m/s, d = %.3g m (d/h = %.3f)",
                       m.U_h, m.sol.u_star, m.d, m.d / c.h), subsystem = :run)
        write_results(rf, Dict("h" => c.h, "z" => m.sol.z, "u" => m.sol.u, "u_star" => m.sol.u_star,
                                                 "U_h" => m.U_h, "d" => m.d, "l_c" => m.l_c, "zf" => m.zf,
                                                 "lad" => m.lad))
    else
        t = r.cfg.canopy_energy
        p = canopy_temperature_profile(r.lad, c.h; C_d = c.c_d, u_star = t.u_star, T_ref = t.t_ref, rh = t.rh,
                                       μ = t.cos_zenith, S_dir = t.sw_direct, S_dif = t.sw_diffuse,
                                       lw_down = t.lw_down, gs = t.g_s, ztop = ztop, N = c.layers, β = c.beta,
                                       Sc = t.schmidt, u_top = c.u_top, chi_l = t.chi_l, leaf_width = t.leaf_width)
        r.profile = p
        @info(@sprintf("canopy temperature: ΔT in [%.3g, %.3g] K after %d sweeps", extrema(p.ΔT)..., p.iterations),
              subsystem = :run)
        write_results(rf, Dict("h" => c.h, "z" => p.z, "dT" => p.ΔT, "iterations" => p.iterations))
    end
    callback === nothing || callback(r, 1)
    return r
end

"The profile of the run: `u` (wind mode) or `dT` (temperature mode)."
Config.state_fields(r::PlantCanopyRun) =
    r.cfg.canopy.mode === :wind ? (u = r.profile.sol.u,) : (dT = r.profile.ΔT,)

register_driver!(PlantCanopyDriver())
