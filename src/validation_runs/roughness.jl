# The dynamic roughness of a staggered cube array (P57-e, #190): the
# a-priori evaluation the `dynamic_roughness` run type makes (examples/validation/configs/dynamic_roughness_cubes.toml). Moved
# from examples/validation/dynamic_roughness_cubes.jl (EX-c #440), where it
# ran in the script.

using ..SurfaceTiles: macdonald_roughness

"""
    staggered_cube_heights(ncube, npc; h = 1.0, offset = npc ÷ 4) -> Matrix

The height map of a staggered array of cubes of side `h` on a periodic
tile of `2ncube × 2ncube` cube sides, resolved by `npc` fine points per
side: `ncube` cubes per row on a `2h × 2h` lattice, alternate rows
shifted by `h` (plan-area density λ_p = 0.25), every cube moved by
`offset` fine points so that coarse cells cut the cubes.
"""
function staggered_cube_heights(ncube::Integer, npc::Integer; h = 1.0, offset::Integer = npc ÷ 4)
    Nf = 2ncube * npc
    hf = zeros(Nf, Nf)
    for row in 0:ncube-1, col in 0:ncube-1
        x0 = col * 2npc + (isodd(row) ? npc : 0) + offset
        y0 = row * 2npc + offset
        for b in 1:npc, a in 1:npc
            hf[mod1(x0 + a, Nf), mod1(y0 + b, Nf)] = h
        end
    end
    return hf
end

"""
    cube_array_roughness(hf, h, Lf; delta, C_d = 1.0, z_ref, U, z0_floor) -> NamedTuple

The dynamic roughness (Anderson & Meneveau 2011, `DynamicRoughness`) of
the fine height map `hf` of a periodic tile of side `Lf` (m) seen by an
LES grid of spacing `delta` (a whole number of fine points): the relief
the grid keeps exerts the surface-gradient drag, the sub-filter rest
enters ``z_0 = α σ_Δ`` with the dynamic α (drag coefficient `C_d`,
`z0_floor`), under a uniform wind `U` at `z_ref` (every drag term is ∝ U²,
so α does not depend on `U`). The effective roughness of the whole
surface follows from the total drag ``τ = u_*^2``:
``z_{0,\\mathrm{eff}} = (z_{ref} − d)\\,e^{−κU/u_*}``, with `d = 0` (heights
above the base) and with the Macdonald et al. (1998) displacement of the
array (`macdonald_roughness` at λ_p = λ_f, the staggered constants).

Returns `lambda_p` (measured from the map), `d_mac`, `z0_mac` (Macdonald,
in m), `alpha`, `converged`, `z0_mean` (the plane mean of the modelled
z0), `form_fraction` (resolved form drag / total), `ustar_over_u`,
`z0_eff_d0`, `z0_eff_dmac`, `ratio_mac` (z0_eff_dmac / z0_mac).
"""
function cube_array_roughness(hf::AbstractMatrix, h::Real, Lf::Real; delta::Real, C_d::Real = 1.0,
                              z_ref::Real, U::Real, z0_floor::Real)
    κ = Aeolus.SurfaceExchange.von_karman()
    λp = sum(hf .> 0) / length(hf)
    mac = macdonald_roughness(λp, λp, 1.0)          # d/h and z0/h
    dMac, z0Mac = mac.d * h, mac.z0 * h
    Nfine = size(hf, 1)
    r = round(Int, delta / (Lf / Nfine))
    N = Nfine ÷ r
    dx = Lf / N
    st = subfilter_height(hf, r; dx = dx, dy = dx)
    dr = DynamicRoughness(st; z0_floor = z0_floor, C_d = C_d)
    uc = fill(Float64(U), N, N); vc = zeros(N, N)
    update_dynamic_roughness!(dr, uc, vc, z_ref)
    D = roughness_surface_drag(dr, uc, vc, z_ref)
    us = sqrt(D.total)
    z0e0 = z_ref * exp(-κ * U / us)
    z0ed = (z_ref - dMac) * exp(-κ * U / us)        # the same u* at the wind height z_ref − d
    return (lambda_p = λp, d_mac = dMac, z0_mac = z0Mac, alpha = dr.α, converged = dr.converged,
            z0_mean = sum(dr.z0) / length(dr.z0), form_fraction = D.form / D.total, ustar_over_u = us / U,
            z0_eff_d0 = z0e0, z0_eff_dmac = z0ed, ratio_mac = z0ed / z0Mac)
end

# ---------------------------------------------------------------------------
# the `dynamic_roughness` run type
# ---------------------------------------------------------------------------

"""
    DynamicRoughnessDriver()

The `dynamic_roughness` run type: `[run] model = "dynamic_roughness"`,
the a-priori dynamic roughness of [`cube_array_roughness`](@ref) for the
staggered cube array of [`staggered_cube_heights`](@ref) (`[cubes]`) seen
by an LES grid of spacing `[roughness] delta_over_h` h. Its numbers go to
`output/results.toml`. Config:
`examples/validation/configs/dynamic_roughness_cubes.toml` (P57-e, #190).
"""
struct DynamicRoughnessDriver <: AbstractRunDriver end

Config.driver_name(::DynamicRoughnessDriver) = "dynamic_roughness"
Config.config_dir(::DynamicRoughnessDriver) = VALIDATION_CONFIG_DIR

function Config.driver_sections(::DynamicRoughnessDriver)
    c = section("cubes"; doc = "The staggered cube array (staggered_cube_heights), a periodic tile of 2 ncube × 2 ncube cube sides.") do s
        option!(s, "h", Float64; default = 1.0, range = (0, Inf), open_lower = true, units = "m", doc = "cube side")
        option!(s, "per_row", Int; default = 20, range = (1, 10_000), doc = "cubes per row (ncube)")
        option!(s, "points_per_side", Int; default = 16, range = (1, 1024), doc = "fine points per cube side")
        option!(s, "offset", Optional{Int}; range = (0, typemax(Int)),
                doc = "shift of every cube in fine points, so that LES cells cut the cubes; \"none\": points_per_side ÷ 4")
    end
    r = section("roughness"; doc = "The LES grid and the dynamic roughness (DynamicRoughness, Anderson & Meneveau 2011).") do s
        option!(s, "delta_over_h", Float64; units = "1", default = 0.5, range = (0, Inf), open_lower = true,
                doc = "LES grid spacing Δ in h (a whole number of fine points)")
        option!(s, "c_d", Float64; units = "1", default = 1.0, range = (0, Inf), open_lower = true,
                doc = "drag coefficient of the surface-gradient drag")
        option!(s, "z_ref_over_h", Float64; units = "1", default = 2.0, range = (0, Inf), open_lower = true,
                doc = "height of the wind above the base, in h")
        option!(s, "wind", Float64; units = "m s^-1", default = 5.0, range = (0, Inf), open_lower = true,
                doc = "the uniform wind at z_ref (α does not depend on it: every drag term is ∝ U²)")
        option!(s, "z0_floor_over_h", Float64; units = "1", default = 1e-4, range = (0, Inf), open_lower = true,
                doc = "the floor of the modelled z0, in h")
    end
    return Section[c, r]
end

function _roughness_checks(cfg)
    e = no_checkpoints(cfg)
    npc = cfg.cubes.points_per_side
    rr = cfg.roughness.delta_over_h * npc
    (isinteger(rr) && rr >= 1 && (2cfg.cubes.per_row * npc) % Int(rr) == 0) ||
        push!(e, ConfigError("roughness.delta_over_h",
                             "Δ must be a whole number of fine points that divides the tile: delta_over_h × points_per_side = $rr"))
    return e
end
Config.driver_checks(::DynamicRoughnessDriver) = Any[
    """`[checkpoint]` stays off: the evaluation is one call
    `delta_over_h × points_per_side` is a whole number of fine points that divides the tile""" => _roughness_checks]

"What the `dynamic_roughness` run type builds and runs: `heights` (the fine height map), `result` and `cfg`."
mutable struct DynamicRoughnessRun
    heights::Matrix{Float64}
    result::Any
    cfg::RunConfig
end

function Config.build(::DynamicRoughnessDriver, cfg::RunConfig)
    c = cfg.cubes
    hf = staggered_cube_heights(c.per_row, c.points_per_side; h = c.h,
                                offset = something(c.offset, c.points_per_side ÷ 4))
    return DynamicRoughnessRun(hf, nothing, cfg)
end

function Simulations.run!(::DynamicRoughnessDriver, r::DynamicRoughnessRun, rf::RunFolder, ::CheckpointSchedule;
                          callback = nothing)
    c = r.cfg.cubes; g = r.cfg.roughness; h = c.h
    x = cube_array_roughness(r.heights, h, 2c.per_row * h; delta = g.delta_over_h * h, C_d = g.c_d,
                             z_ref = g.z_ref_over_h * h, U = g.wind, z0_floor = g.z0_floor_over_h * h)
    r.result = x
    @info(@sprintf("dynamic roughness, Δ/h = %.2f, C_d = %.2f: α = %.4f (converged %s), z0_eff/h = %.4f (d = d_Mac), %.2f × Macdonald",
                   g.delta_over_h, g.c_d, x.alpha, x.converged, x.z0_eff_dmac / h, x.ratio_mac), subsystem = :run)
    write_results(rf, Dict(String(k) => getfield(x, k) for k in keys(x)))
    callback === nothing || callback(r, 1)
    return r
end

"The numbers of the evaluation, as one-element vectors (α, ⟨z0⟩, u*/U, z0_eff)."
Config.state_fields(r::DynamicRoughnessRun) =
    (alpha = [r.result.alpha], z0_mean = [r.result.z0_mean], ustar_over_u = [r.result.ustar_over_u],
     z0_eff_dmac = [r.result.z0_eff_dmac])

register_driver!(DynamicRoughnessDriver())
