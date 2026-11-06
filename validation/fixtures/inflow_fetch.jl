# The development length behind a synthetic inflow (N-inflow-d, #266) on
# the periodic, neutral, wall-modelled half channel of P54-d: the harness
# of test/inflow_fetch.jl. It was the `inflow_fetch` run type of
# Aeolus.RunDrivers (EX-d #441; formerly test/inflow_fetch_harness.jl and
# examples/nesting/inflow_fetch.jl); since ONE-l (#475) a test fixture: the
# channel cases are not configurations of the model (dropped,
# docs/src/design/single_model.md). Include it once (it defines
# `module InflowFetch`; it needs the channel fixture).

isdefined(@__MODULE__, :ChannelCases) || include(joinpath(@__DIR__, "..", "..", "test", "fixtures", "channel_cases.jl"))

module InflowFetch

using Aeolus
using Aeolus.Grids: RegularGrid
using Aeolus.BoundaryConditions: ProfileBoundaryData
using ..ChannelCases: half_channel_case, run_channel!, channel_dt, advance_channel!
using Aeolus.Operators: Operators, total_stress_profile
using Aeolus.Simulations: Simulations
using Aeolus.Transition: ParentResolvedStatistics, update_inflow_statistics!, FetchStatistics, accumulate_fetch!,
                         fetch_profiles, InflowTurbulence, free_stream_generator, added_mass_flux, injected_ke_flux,
                         development_length

export fetch_precursor, fetch_child, pressure_gradient_bias, nearest_level

# Units: δ = H = 1, u* = 1, so times are in δ/u*.

"""
    fetch_precursor(; N, extent, t_spin, t_avg, seed) -> NamedTuple

Spin up a periodic half channel on `N` cells and `extent = (Lx, Ly)`,
then average for `t_avg` (δ/u*): the generator statistics `src`
(ParentResolvedStatistics, :full, τ_avg = Inf), the mean profile `U`
(cell centres), the x- and time-averaged `FetchStatistics` profiles
`tke`, `uw` (resolved), and the total-stress profile `stress`.
"""
function fetch_precursor(; N, extent, t_spin, t_avg, seed = 1, z0_over_H = 1e-4)
    case = half_channel_case(; N = N, extent = extent, seed = seed, z0_over_H = z0_over_H)
    t0 = time()
    run_channel!(case, t_spin)
    src = ParentResolvedStatistics(case.sim; closure = case.closure, face = :west,
                                   mode = :full, τ_avg = Inf)
    fs = FetchStatistics(case.sim)
    t_end = t_spin + t_avg
    nz = N[3]
    stress = zeros(nz + 1); W = 0.0
    while case.sim.t < t_end - 1e-12
        h = min(channel_dt(case), t_end - case.sim.t)
        advance_channel!(case, h)
        update_inflow_statistics!(src, h)
        accumulate_fetch!(fs, case.sim, h)
        stress .+= h .* total_stress_profile(case).total
        W += h
    end
    p = fetch_profiles(fs)
    nx = size(p.U, 1)
    xmean(A) = vec(sum(A; dims = 1)) ./ nx
    return (case = case, src = src, z = p.z, U = xmean(p.U), tke = xmean(p.tke),
            uw = xmean(p.uw), uu = xmean(p.uu), vv = xmean(p.vv), ww = xmean(p.ww),
            stress = stress ./ W, z_faces = [(k - 1) * case.sim.mesh.dz for k in 1:nz+1],
            wall_time = time() - t0)
end

# the precursor's mean profile on the storage levels of a child mesh with
# the same vertical grid (halo levels: the nearest interior level)
function _storage_profile(U::AbstractVector, m)
    nz = length(U)
    m.Nz == nz || throw(DimensionMismatch("child has $(m.Nz) levels, the profile $nz"))
    return [U[clamp(k - m.Hz, 1, nz)] for k in 1:nz+2m.Hz]
end

"""
    fetch_child(pre; generator = :df, N, extent, t_spin, t_avg, x_stress = 4.0,
                lateral = :open, y_margin = 0.5) -> NamedTuple

The inflow–outflow half channel fed by the precursor `pre`
(`generator` = `:off`, `:df` (DigitalFilterGenerator) or `:fsd`
(FreeStreamInflow)), spun up for `t_spin` and averaged for `t_avg`
(δ/u*). Returns the `fetch_profiles` (`p`), the time-mean bulk velocity
of every x-face column (`bulk`, m/s), the time- and plane-mean pressure
per column (`pmean`), the total stress at `x = x_stress` (three columns,
`stress`), the largest relative added mass flux of the generator
(`added`), the injected KE flux and the wall time. The bulk velocity,
pressure and stress are sampled every `sample_every` steps.
`lateral = :open` (default) opens the south and north faces
(`half_channel_case(lateral = :open)`); the fetch statistics, pressure
and stress are then averaged over the cells farther than `y_margin`
(δ) from them (`jrange`, returned). The bulk velocity is always the
volume flux of the whole face.
"""
function fetch_child(pre; generator::Symbol = :df, N, extent, t_spin, t_avg,
                     x_stress::Real = 4.0, nmodes::Integer = 200, seed::Integer = 1,
                     sample_every::Integer = 1, lateral::Symbol = :open,
                     lateral_outflow::Symbol = :dirichlet, y_margin::Real = 0.5)
    probe = RegularGrid(size = N, extent = (extent[1], extent[2], 1.0))
    data = ProfileBoundaryData(u = _storage_profile(pre.U, probe), v = 0.0, w = 0.0, θ = 0.0)
    case = half_channel_case(; N = N, extent = extent, inflow = data, noise = 0.0,
                             lateral = lateral, lateral_outflow = lateral_outflow)
    sim = case.sim; g = sim.mesh
    jr = lateral === :open ?
         (findfirst(j -> (j - 0.5) * g.dy >= y_margin, 1:g.Ny):
          findlast(j -> g.Ly - (j - 0.5) * g.dy >= y_margin, 1:g.Ny)) : (1:g.Ny)
    it = nothing
    if generator === :df
        it = InflowTurbulence(sim; statistics = pre.src, faces = (:west,), seed = seed)
    elseif generator === :fsd
        # the isotropic random-mode field of P59-d, one length (the face-mean
        # streamwise L_s of the statistics), swept at the bulk velocity
        Ub = sum(pre.U) / length(pre.U)
        gen = free_stream_generator(; U∞ = Ub, nmodes = nmodes)
        it = InflowTurbulence(sim; statistics = pre.src, faces = (:west,), generator = gen,
                              seed = seed)
    else
        generator === :off || throw(ArgumentError("generator must be :off, :df or :fsd"))
    end
    t0 = time()
    run_channel!(case, t_spin)
    fs = FetchStatistics(sim; jrange = jr)
    Nx, Ny, Nz, Hx, Hy, Hz = g.Nx, g.Ny, g.Nz, g.Hx, g.Hy, g.Hz
    bulk = zeros(Nx + 1); pm = zeros(Nx); stress = zeros(Nz + 1)
    is = round(Int, x_stress / g.dx)
    W = 0.0; Ws = 0.0; added = 0.0; ke = 0.0; n = 0
    t_end = t_spin + t_avg
    while sim.t < t_end - 1e-12
        h = min(channel_dt(case), t_end - sim.t)
        advance_channel!(case, h)
        accumulate_fetch!(fs, sim, h)
        if it !== nothing
            added = max(added, added_mass_flux(it).west.relative)
            ke += h * injected_ke_flux(it).total
        end
        W += h
        (n += 1) % sample_every == 0 || continue
        # bulk velocity, pressure and the stress band: every `sample_every` steps
        Ws += h
        @inbounds for i in 1:Nx+1
            s = 0.0
            for k in 1:Nz, j in 1:Ny
                s += sim.u.data[Hx+i, Hy+j, Hz+k]
            end
            bulk[i] += h * s / (Ny * Nz)
        end
        @inbounds for i in 1:Nx
            s = 0.0
            for k in 1:Nz, j in jr
                s += sim.p.data[Hx+i, Hy+j, Hz+k]
            end
            pm[i] += h * s / (length(jr) * Nz)
        end
        νt = Operators.compute_sgs!(case.closure, sim).νv
        stress .+= h .* total_stress_profile(sim.u, sim.v, sim.w, νt; irange = is-1:is+1,
                                              jrange = jr).total
    end
    return (case = case, inflow = it, p = fetch_profiles(fs), jrange = jr, bulk = bulk ./ Ws, pmean = pm ./ Ws,
            stress = stress ./ Ws, x_stress = (is - 0.5) * g.dx, added = added,
            ke_flux = ke / W, wall_time = time() - t0,
            maxdiv = Simulations.max_divergence(sim))
end

"""
    pressure_gradient_bias(child, G; x_range = (1.0, 4.0)) -> b

The least-squares slope of the time- and plane-mean pressure over
`x_range` (δ), divided by the driving gradient `G` (the body force of
the precursor, u*²/δ): the extra mean pressure gradient the developing
flow needs on top of the precursor's forcing.
"""
function pressure_gradient_bias(child, G; x_range = (1.0, 4.0))
    x = child.p.x
    sel = findall(xx -> x_range[1] <= xx <= x_range[2], x)
    xs = x[sel]; ps = child.pmean[sel]
    xm = sum(xs) / length(xs); pm = sum(ps) / length(ps)
    slope = sum((xs .- xm) .* (ps .- pm)) / sum((xs .- xm) .^ 2)
    return -slope / G            # dp/dx < 0 adds to the drive
end

"The index of the level of `z` nearest to the height `h`."
nearest_level(z::AbstractVector, h::Real) = findmin(abs.(z .- h))[2]

end # module
