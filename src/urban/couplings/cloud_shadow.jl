# Cloud shadow: the 3-D slant-path direct-beam transmittance of the
# resolved cloud and fog (SolarGeometry.cloud_shadow_transmittance!),
# applied to the direct-beam share of the radiosity shortwave.
#
# Replaces `apply_cloud_shadow!` (formerly in sw_modifiers.jl), which
# took the vertical column integral of ql with an ad-hoc extinction of
# 700 m²/kg, kept 15 % of the removed light, ignored the sun angle and
# compounded every step between radiation updates (#130).

"""
    CloudShadow

Additive coupling. At the `:radiation` cadence (and step 1), right after
the radiosity solve:

1. the per-cell cloud optical depth `τ = k·ρ·q·Δz` from `ql` (and `qi`,
   if a coupling registered it), with the `RadiationCore` mass
   extinction for effective radii `r_eff_liq`, `r_eff_ice` (the same as
   the column radiation's cloud optics) and the density profile of the
   column radiation; zero in solid cells;
2. delta-scaled (`TwoStream.delta_scale`, with the single-scattering
   albedo and asymmetry of the Mie liquid cloud in the visible), so the
   forward-scattered peak stays in the beam, as in the two-stream solve;
3. the direct-beam transmittance `T[i, j]` along the slant path toward
   the sun, from the top of each column's solid stack
   (`cloud_shadow_transmittance!` with `k_start`);
4. `bus.sw ← bus.sw · (1 − f_dir·(1 − T))`, where `f_dir` is the share of
   the column's absorbed shortwave that arrived as direct beam in the
   radiosity solution.

It runs at the radiation cadence, like the radiosity solve whose beam it
attenuates, so it cannot compound between updates. The optical-depth
field and the per-column rays are launches on the case's backend (KA-n,
#305): pointwise, so identical at every thread count.

Known limitations: interreflections of the shaded beam and the window
sun the buildings take from the radiosity solution are not attenuated;
the diffuse light the cloud scatters sideways is not added (the column
radiation's diffuse is the clear-LES one).
"""
mutable struct CloudShadow <: AbstractCoupling
    trans::Matrix{Float64}        # direct-beam cloud transmittance
    fdir::Matrix{Float64}         # direct share of the absorbed shortwave
    removed::Matrix{Float64}      # W/m² taken off bus.sw at the last update
    k_start::Matrix{Int}          # solid cells under each column
    r_eff_liq::Float64
    r_eff_ice::Float64
    fδ::Float64                   # delta-scaling factor τ′/τ of the liquid cloud (visible)
    sun_up::Bool
    cloud::Bool                   # any cloud optical depth in the air at the last update
end

coupling_name(::Type{CloudShadow}) = :cloud_shadow
modules_used(::Type{CloudShadow}) = (:SolarGeometry, :TwoStream, :RadiationCore)
default_options(::Type{CloudShadow}) = (r_eff_liq = 10.0, r_eff_ice = 30.0)
stages(::Type{CloudShadow}) = ((:radiation, 15, Cadence(:radiation; first = true)),)

function build(::Type{CloudShadow}, case, o)
    m = case.mesh
    Nx, Ny, Nz = m.Nx, m.Ny, m.Nz
    mask = case.geom.sm.mask
    kst = [count(k -> mask[m.Hx + i, m.Hy + j, m.Hz + k], 1:Nz) for i in 1:Nx, j in 1:Ny]
    # single-scattering albedo and asymmetry of the Mie liquid cloud in the
    # band containing 550 nm, from a one-layer column through TwoStream
    spec = shortwave_bands()
    b = findfirst(b -> 1e4 / band_limits(spec, b)[2] <= 0.55 <= 1e4 / band_limits(spec, b)[1],
                  1:nbands(spec))
    col = AtmosphericColumn(Float64, 1; ql = [1e-3], ρ = [1.0], dz = [10.0],
                            r_eff_liq = [o.r_eff_liq])
    op = allocate_optics(Float64, (1,), spec)
    compute_optics!(op, mie_cloud_optics(spec), col)
    g1 = first(gpoint_range(spec, b))
    τ, ω, g = op.τ[1, g1], op.ssa[1, g1], op.g[1, g1]
    fδ = delta_scale(τ, ω, g)[1] / τ
    return CloudShadow(ones(Nx, Ny), zeros(Nx, Ny), zeros(Nx, Ny), kst,
                       Float64(o.r_eff_liq), Float64(o.r_eff_ice), fδ, false, false)
end

"Air density per LES level: the column radiation's, else the reference ρ̄ (#131)."
function _air_density(case)
    cr = find_coupling(case, :column_radiation)
    Nz = case.mesh.Nz
    cr !== nothing && cr.col !== nothing && return cr.col.ρ[1:Nz]
    return ref_density(case)
end

function run_stage!(c::CloudShadow, case, ::Val{:radiation}, dt)
    bus, m = case.bus, case.mesh
    sw = bus.swsol
    c.sun_up = sw !== nothing
    if !c.sun_up
        fill!(c.trans, 1.0); fill!(c.fdir, 0.0); fill!(c.removed, 0.0)
        return nothing
    end
    ql = interior_view(tracer(case, :ql))
    qi = haskey(case.tracers.index, :qi) ? interior_view(tracer(case, :qi)) : zero(ql)
    ρ = _air_density(case)
    τ = cloud_optical_depth_column(ql, qi, ρ, m.dz;
                                   k_liq = liquid_mass_extinction(c.r_eff_liq),
                                   k_ice = ice_mass_extinction(c.r_eff_ice))
    mask = case.geom.sm.mask
    # zero in the solid cells, delta-scaled in the air: pointwise (KA-n, #305)
    launch!(Aeolus.Backends.backend(τ), _shadow_tau_cell!, (1:m.Nx, 1:m.Ny, 1:m.Nz),
            τ, mask, m.Hx, m.Hy, m.Hz, c.fδ)
    c.cloud = any(>(0), τ)
    sp = solar_position_at(case.cfg, case.clock.hour)
    cloud_shadow_transmittance!(c.trans, τ, m, sun_direction_enu(sp); k_start = c.k_start)
    _direct_share!(c.fdir, case)
    @inbounds for I in eachindex(bus.sw)
        r = bus.sw[I] * c.fdir[I] * (1 - c.trans[I])
        c.removed[I] = r
        bus.sw[I] -= r
    end
    return nothing
end

@inline function _shadow_tau_cell!(i, j, k, τ, mask, Hx, Hy, Hz, fδ)
    @inbounds τ[i, j, k] = mask[Hx + i, Hy + j, Hz + k] ? 0.0 : max(τ[i, j, k], 0.0) * fδ
    return nothing
end

# Share of each column's absorbed shortwave that arrived as direct beam:
# Σ A(1−α)E_dir / Σ A(1−α)Q over the column's facets.
function _direct_share!(fdir, case)
    scene, sw = case.geom.scene, case.bus.swsol
    α = facet_albedos(scene)
    A = facet_areas(scene)
    num = zeros(size(fdir)); den = zeros(size(fdir))
    @inbounds for (fi, (ci, cj)) in enumerate(case.geom.facet_col)
        num[ci, cj] += A[fi] * (1 - α[fi]) * sw.direct[fi]
        den[ci, cj] += A[fi] * sw.absorbed[fi]
    end
    @. fdir = ifelse(den > 0, num / den, 0.0)
    return fdir
end

output_fields(c::CloudShadow, case) = OutputField[
    OutputField("cloud_trans", "1", "direct-beam transmittance of the resolved cloud (slant path)",
                :xy, cs -> c.trans)]

"""
    diagnostics(c::CloudShadow, case)

The direct-beam transmittance of the resolved cloud `trans_min`,
`trans_mean` (1) and the shortwave it removed `sw_removed` (W m⁻²,
domain mean); `NaN` while the sun is down.
"""
function diagnostics(c::CloudShadow, case)
    up(f) = () -> c.sun_up ? f() : NaN
    return Diagnostic[
        Diagnostic("trans_min", "1", up(() -> minimum(c.trans))),
        Diagnostic("trans_mean", "1", up(() -> sum(c.trans) / length(c.trans))),
        Diagnostic("sw_removed", "W m-2", up(() -> sum(c.removed) / length(c.removed)))]
end

function smoke_check(c::CloudShadow, case)
    checks = ["0 ≤ direct share ≤ 1" => all(x -> -1e-12 <= x <= 1 + 1e-12, c.fdir),
              "0 < T ≤ 1" => all(t -> 0 < t <= 1, c.trans),
              "removed shortwave ≥ 0" => all(>=(0), c.removed),
              "delta-scaled liquid τ′/τ within (0, 1)" => 0 < c.fδ < 1]
    if c.sun_up
        push!(checks, "cloud in the air ⇒ some shadow; none ⇒ T ≡ 1" =>
              (c.cloud ? minimum(c.trans) < 1 : all(==(1.0), c.trans)))
    end
    return checks
end

register!(CloudShadow)
