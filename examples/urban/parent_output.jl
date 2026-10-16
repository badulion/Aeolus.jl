# The output of a run of the Berlin nest parent alone (berlin_parent.toml,
# berlin_parent_coarse.toml; NEST-p3 #483: a domain of the model's core),
# read for the analysis scripts (berlin_parent_lindenberg_check.jl,
# berlin_parent_figures.jl, plot_parent_surface.jl). It reads the run
# folder only, and runs nothing:
#
#   <run>/output/parent.nc   the plane-mean profiles of θ, qv and the heat
#                            flux every output, the energy and water budgets
#                            (cumulative terms, every step), the tiled
#                            surface's maps
#   <run>/diagnostics.csv    the step sizes, their limiters and retries
#   <run>/run.toml           the resolved config (its start hour)

using Aeolus

"""
    read_parent_output(run) -> NamedTuple

The parent's output of the run folder `run`: `t` (s from the start) and
`hour_lt` (local hour) of every output record, the level heights `z`
(m), the profiles `θ`, `qv`, `heat_flux` (level × record), the
boundary-layer heights of every record `zi_parcel` (`boundary_layer_height`
on θ_v, `:parcel`, as for the sounding) and `zi_min_flux` (the height of
the minimum of the total heat flux, the CBL definition), the budget series
`ts` (s) with `energy` and `water` (term name => cumulative J m⁻² /
kg m⁻²; `storage`, `closure` among them), the step sizes `dts`, their
`retries`, and the `grid` (cells).
"""
function read_parent_output(run::AbstractString)
    cfg = load_run_config(run)
    ds = NCDataset(joinpath(run, "output", "parent.nc"))
    t = Float64.(ds["time"][:]); z = Float64.(ds["z"][:])
    θ = Float64.(ds["theta_profile"][:, :]); qv = Float64.(ds["qv_profile"][:, :])
    hf = Float64.(ds["heat_flux_profile"][:, :])
    ts = Float64.(ds["time_series"][:])
    # (a term booked first after the start is the file's fill value before: 0 so far)
    nofill(v) = [x === missing || x > 1e36 ? 0.0 : Float64(x) for x in v]
    series(prefix) = Dict(replace(k, prefix => "") => nofill(ds[k][:]) for k in keys(ds) if startswith(k, prefix))
    energy, water = series("budget_energy__"), series("budget_water__")
    grid = [length(ds["x"]), length(ds["y"]), length(z)]
    close(ds)
    MT = Aeolus.MoistThermodynamics
    θv(n) = [MT.virtual_temperature(θ[k, n], MT.PhasePartition(qv[k, n])) for k in eachindex(z)]
    zi_parcel = [boundary_layer_height(z, θv(n); method = :parcel) for n in eachindex(t)]
    zi_min_flux = [boundary_layer_height(z, hf[:, n]; method = :min_flux) for n in eachindex(t)]
    lines = readlines(joinpath(run, "diagnostics.csv"))
    head = [first(split(h, " [")) for h in split(lines[1], ',')]
    col(name) = [parse(Float64, split(l, ',')[findfirst(==(name), head)]) for l in lines[2:end]]
    return (; t, hour_lt = mod.(cfg.radiation.start_hour .+ t ./ 3600, 24), z, θ, qv, heat_flux = hf,
            zi_parcel, zi_min_flux, ts, energy, water, dts = col("dt"), retries = col("retries"), grid, cfg)
end

"The mean rate (W m⁻²) of the cumulative budget `series` over the model times `[t0, t1]` (s)."
function budget_rate(p, series, t0, t1)
    i0, i1 = argmin(abs.(p.ts .- t0)), argmin(abs.(p.ts .- t1))
    return (series[i1] - series[i0]) / (p.ts[i1] - p.ts[i0])
end
