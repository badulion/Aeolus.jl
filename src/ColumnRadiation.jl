"""
    Aeolus.ColumnRadiation

The radiation of a model's mean column (Era III): the model's mean
profiles of θ, q_v and q_l on its own layers, made hydrostatic and
stitched under a standard atmosphere to 30 km, through the Era III
radiation stack — correlated-k gas optics (`GasOptics`), Mie cloud optics
and the two-stream solvers (`TwoStream`) — at the sun of
`SolarGeometry`. It gives the downwelling longwave and the direct and
diffuse shortwave at the surface, and the heating rate of the model
layers from the net-flux divergence.

[`column_radiation_solver`](@ref) builds the optics and the solvers,
[`stitched_column`](@ref) the column, [`column_radiation!`](@ref) runs
one call. The urban case's `column_radiation` coupling (its domain-mean
LES column) and its nest parent (its plane-mean column on stretched
levels) both use it. Moved from the urban case's glue (ONE-e, #454);
see `docs/src/design/berlin_surface_radiation.md`.
"""
module ColumnRadiation

using ..RadiationCore: RadiativeFluxes, AtmosphericColumn, shortwave_bands, longwave_bands,
                       allocate_column_optics, allocate_fluxes, compute_optics!, combine_optics!,
                       heating_rate!, column_from_profiles, stitch_columns
using ..GasOptics: correlated_k_optics, us_standard_column
using ..TwoStream: mie_cloud_optics, two_stream_solver, solve_sw!, solve_lw!, TwoStreamWorkspace
using ..SolarGeometry: earth_sun_distance_factor

export ColumnRadiationModel, column_radiation_solver, stitched_column, column_radiation!,
       column_heating_kday

"""
    ColumnRadiationModel

The state of the radiation of one model column ([`column_radiation_solver`](@ref)):

  * the optics and solvers: `gas_sw`, `gas_lw` (correlated-k), `cloud_sw`,
    `cloud_lw` (Mie), `solver_sw`, `solver_lw` (two-stream) and their
    multi-column scratch `ws_sw`, `ws_lw` (KA-n, #305);
  * the settings: `nz_upper` (layers of the standard atmosphere above the
    model), `r_eff_liq` (µm, the droplets' effective radius), `p_surf` (Pa,
    the ground pressure of the column), `surface_sw` (`:twostream`: the
    surface shortwave is the column's; `:bird`: the caller's clear sky);
  * the last call ([`column_radiation!`](@ref)): `col` (the stitched
    column), `flw`, `fsw` (`nothing` at night) and `fsw_clear` (the
    shortwave of the column without the model's condensate), `heat`,
    `heat_lw`, `heat_sw` (K s⁻¹, dT/dt of the model layers), `μ₀`, `sun`
    and `ncalls`;
  * `buf` (scratch of the length of `heat`) and `dt_applied`, the step of
    the caller's last application of `heat`.
"""
mutable struct ColumnRadiationModel{GS, GL, CS, CL, SS, SL}
    gas_sw::GS
    gas_lw::GL
    cloud_sw::CS
    cloud_lw::CL
    solver_sw::SS
    solver_lw::SL
    nz_upper::Int
    r_eff_liq::Float64
    p_surf::Float64
    surface_sw::Symbol
    col::Any                      # last stitched column (with the model's condensate)
    fsw::Any                      # its fluxes (sw: nothing at night)
    flw::Any
    fsw_clear::Any                # shortwave fluxes without the model's condensate
    heat::Vector{Float64}         # K/s on the model levels, lw + sw
    heat_lw::Vector{Float64}
    heat_sw::Vector{Float64}
    buf::Vector{Float64}
    μ₀::Float64
    sun::Any                      # solar position of the last call
    dt_applied::Float64           # dt of the caller's last application of heat
    ncalls::Int
    ws_sw::TwoStreamWorkspace{Float64}   # scratch of the multi-column solves (KA-n, #305)
    ws_lw::TwoStreamWorkspace{Float64}
end

"""
    column_radiation_solver(Nz, p_surf; nz_upper = 60, r_eff_liq = 10, surface_sw = :twostream)
        -> ColumnRadiationModel

The column-radiation state for a model with `Nz` levels and ground
pressure `p_surf` (Pa): correlated-k gas optics, Mie cloud optics and the
two-stream solvers of the shortwave and longwave bands. The one
constructor: the urban case's `:column_radiation` coupling builds it for
its LES column, the nest parent for its own grid, and both then call
[`column_radiation!`](@ref).
"""
function column_radiation_solver(Nz::Integer, p_surf::Real; nz_upper::Integer = 60,
                                 r_eff_liq::Real = 10.0, surface_sw::Symbol = :twostream)
    surface_sw in (:twostream, :bird) ||
        error("column_radiation: surface_sw must be :twostream or :bird, got $surface_sw")
    sw, lw = shortwave_bands(), longwave_bands()
    return ColumnRadiationModel(correlated_k_optics(sw), correlated_k_optics(lw),
                                mie_cloud_optics(sw), mie_cloud_optics(lw),
                                two_stream_solver(sw), two_stream_solver(lw),
                                Int(nz_upper), Float64(r_eff_liq), Float64(p_surf), surface_sw,
                                nothing, nothing, nothing, nothing,
                                zeros(Nz), zeros(Nz), zeros(Nz), zeros(Nz),
                                0.0, nothing, 0.0, 0,
                                TwoStreamWorkspace{Float64}(), TwoStreamWorkspace{Float64}())
end

"""
    stitched_column(c, θ, qv, ql, dz; T_rad, humid = :midlatitude_summer) -> AtmosphericColumn

The radiation column of a model's mean profiles: `θ`, `qv`, `ql` (K,
kg kg⁻¹) on layers of thickness `dz` (m, a vector: uniform for the urban
child, stretched for its parent), hydrostatic from `c.p_surf`
(`column_from_profiles`), with the surface radiative temperature `T_rad`,
under `us_standard_column` (`c.nz_upper` layers to 30 km, its surface
temperature chosen so that its lapse rate, 6.5 K km⁻¹, continues from the
model top temperature; humidity profile `humid`).
"""
function stitched_column(c::ColumnRadiationModel, θ, qv, ql, dz::AbstractVector; T_rad::Real,
                         humid::Symbol = :midlatitude_summer)
    low = column_from_profiles(θ, dz; p_surf = c.p_surf, qv = max.(qv, 0.0), ql = ql,
                               T_sfc = T_rad, r_eff_liq = c.r_eff_liq)
    # the standard atmosphere above, with its lapse rate continued from the model top
    T_top = low.T_lev[end]
    up = us_standard_column(; nz = c.nz_upper, z_top = 30_000.0,
                            T_sfc = T_top + 6.5e-3 * sum(dz), humid = humid)
    return stitch_columns(low, up)
end

# The optics of the columns `cols` in the multi-column layout τ[k, g, c]
# (KA-n, #305): correlated-k gas optics, plus the Mie cloud optics when a
# column has condensate. One column (the domain or plane mean); its bits
# are those of the single-column optics.
function _optics(gas, cloud, cols)
    length(cols) == 1 || error("column_radiation: _optics combines the cloud optics of one column")
    spec = gas.spec
    n = length(cols[1].dz)
    o = allocate_column_optics(Float64, n, length(cols), spec)
    compute_optics!(o, gas, cols)
    col = cols[1]
    if any(>(0), col.ql) || any(>(0), col.qi)
        oc = allocate_column_optics(Float64, n, length(cols), spec)
        compute_optics!(oc, cloud, cols)
        combine_optics!(o, oc)
    end
    return o
end

# The single-column fluxes of column 1 of multi-column fluxes (no copy).
_column_fluxes(f) = RadiativeFluxes{Float64,1}(vec(f.flux_up), vec(f.flux_dn),
                                               vec(f.flux_dn_dir), vec(f.flux_net))

function _solve_sw(c, col, μ, α, fdist)
    cols = [col]
    f = allocate_fluxes(Float64, (1,), length(col.dz) + 1)
    solve_sw!(f, c.solver_sw, _optics(c.gas_sw, c.cloud_sw, cols); μ₀ = μ,
              sfc_albedo_direct = α, workspace = c.ws_sw)
    f1 = _column_fluxes(f)
    for a in (f1.flux_up, f1.flux_dn, f1.flux_dn_dir, f1.flux_net)
        a .*= fdist
    end
    return f1
end

# The fluxes on the model levels 1 … Nz+1 only (heating_rate! wants a
# RadiativeFluxes whose layers are the ones it fills).
function _model_levels(f, Nz)
    r = 1:Nz + 1
    return RadiativeFluxes{Float64,1}(f.flux_up[r], f.flux_dn[r], f.flux_dn_dir[r], f.flux_net[r])
end

"""
    column_radiation!(c, col, sp, ε, α; clear = () -> col) -> c

One radiation call on the stitched column `col` ([`stitched_column`](@ref))
at the solar position `sp` (`SolarGeometry.SolarPosition`), for a surface
of emissivity `ε` and shortwave albedo `α`: correlated-k optics (plus Mie
cloud optics where `col` has condensate), `solve_lw!` and, with the sun
up, `solve_sw!` scaled by the Earth–Sun distance factor. Fills
`c.heat_lw`, `c.heat_sw`, `c.heat` (K/s, dT/dt of the model layers, from
the net-flux divergence), `c.col`, `c.flw`, `c.fsw` (`nothing` at night),
`c.fsw_clear` — the shortwave of the column `clear()` returns (the column
without the model's condensate; only built when `col` has any) — `c.μ₀`
and `c.sun`.

The surface downwelling longwave is `c.flw.flux_dn[1]` and the direct /
diffuse shortwave are `c.fsw_clear.flux_dn_dir[1]` and
`c.fsw_clear.flux_dn[1] − dir`.
"""
function column_radiation!(c::ColumnRadiationModel, col, sp, ε::Real, α::Real; clear = () -> col)
    Nz = length(c.heat)
    μ = sp.elevation > 0 ? sp.μ₀ : 0.0
    cols = [col]
    flw2 = allocate_fluxes(Float64, (1,), length(col.dz) + 1)
    solve_lw!(flw2, c.solver_lw, _optics(c.gas_lw, c.cloud_lw, cols), cols; sfc_emissivity = ε,
              workspace = c.ws_lw)
    flw = _column_fluxes(flw2)
    heating_rate!(c.heat_lw, _model_levels(flw, Nz), col.dz[1:Nz], col.ρ[1:Nz])
    fdist = earth_sun_distance_factor(sp)
    if μ > 0
        fsw = _solve_sw(c, col, μ, α, fdist)
        heating_rate!(c.heat_sw, _model_levels(fsw, Nz), col.dz[1:Nz], col.ρ[1:Nz])
        # the surface beam without the model condensate (the 3-D shadow applies it)
        fcl = any(>(0), @view col.ql[1:Nz]) ? _solve_sw(c, clear(), μ, α, fdist) : fsw
    else
        fsw = nothing; fcl = nothing
        fill!(c.heat_sw, 0.0)
    end
    @. c.heat = c.heat_lw + c.heat_sw
    c.col, c.flw, c.fsw, c.fsw_clear, c.μ₀, c.sun = col, flw, fsw, fcl, μ, sp
    c.ncalls += 1
    return c
end

"""
    column_heating_kday(c) -> (total, longwave, shortwave)

The mean heating of the model layers of the last call, in K day⁻¹,
weighted by their mass ``ρ\\,Δz``.
"""
function column_heating_kday(c::ColumnRadiationModel)
    Nz = length(c.heat)
    w = [c.col.ρ[k] * c.col.dz[k] for k in 1:Nz]
    avg(h) = 86400 * sum(w .* h) / sum(w)
    return (avg(c.heat), avg(c.heat_lw), avg(c.heat_sw))
end

end # module ColumnRadiation
