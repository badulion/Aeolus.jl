# The convective boundary layer harness of the prognostic SGS TKE closure
# (P56-b, #182): `tke_cbl`, run by test/era6_tke.jl at 24³. It was the
# `cbl` run type of Aeolus.RunDrivers (EX-d #441; formerly
# test/tke_cbl_driver.jl); the CBL itself is the model's configuration
# examples/cbl_tke/cbl_tke.toml since ONE-l (#475), and
# test/config_validation_cases.jl asserts that it reproduces this harness
# bit for bit. Include it once (it defines `module TKECBL`).

module TKECBL

using Aeolus
using Aeolus.Grids: RegularGrid
using Aeolus.Fields: interior_view, set!
using Aeolus.BoundaryConditions: FieldBoundaryConditions
using Aeolus.Simulations: BoussinesqSimulation, step_rk3!
using Aeolus.TimeIntegration: CFLConstraints, adaptive_dt
using Aeolus.IBM: SolidMask, wall_distance, first_level_cells
using Aeolus.PrognosticSGS: TKEClosure, initialize_tke!, tke_statistics
using Aeolus.TurbulenceStability: boundary_layer_height
using Printf: @sprintf
using Random: Xoshiro, rand

export tke_cbl, tke_cbl_summary

"""
    tke_cbl(N; Δ = 50.0, Δz = Δ, Q = 0.1, T_end = 3600.0, avg = 900.0, z_i0 = 600.0,
            Γ = 0.003, θ₀ = 300.0, seed = 1, wall_cap = false, surface_factor = 3.9,
            C_advective = 0.4, progress = false, on_step = nothing) -> NamedTuple

Run the CBL on an `N × N × N` grid with horizontal spacing `Δ` and vertical
`Δz` and return the averaged profiles (`zc` cell centres, `zf` interior w
faces): `e` (SGS TKE), `kres` (resolved TKE), `w2` (resolved ``σ_w^2``),
`flux` (total heat flux / Q at the w faces: resolved ``⟨w'θ'⟩`` plus the SGS
``−⟨κ_t ∂θ/∂z⟩``), `sgs_flux`, `θ`; and the scalars `zi` (height of the
minimum heat flux, `boundary_layer_height(zf, flux; method = :min_flux)`), `min_flux_ratio`, `max_sgs_fraction` (of
``e/(e + k_{res})`` over ``0.2 < z/z_i < 0.8``), `z_epeak`, `w_star`
(``(g Q z_i/θ_0)^{1/3}``), `w2max_norm` (``\\max σ_w^2/w_*^2``),
`z_w2max_norm` (its ``z/z_i``), `clips`, `min_e`, `steps`, `wall` (s).

Set-up (after Moeng 1984 and Sullivan & Patton 2011, simplified): free
convection (no mean wind) over a free-slip floor heated with a constant
kinematic flux Q, periodic laterally, a lid with a Rayleigh sponge in the
top quarter. θ is uniform up to z_i0 and increases at Γ above, with a
small random perturbation in the lowest 200 m. SSP-RK3 (`step_rk3!`)
with the CFL step of `adaptive_dt` (advective number `C_advective`), the
closure passed as `sgs`. Statistics are horizontal means averaged over
every step of the last `avg` seconds.

`on_step(sim, closure, Δt)`, if given, is called after every step (e.g. to
fold the state into `Transition.ParentResolvedStatistics`, N-inflow-b).

`wall_cap = true` caps the mixing length at ``κ`` × `IBM.wall_distance`
(the floor); the default is Moeng's (1984) closure, ``l ≤ Δ`` with
``c_ε × 3.9`` (`surface_factor`) at the first level.
"""
function tke_cbl(N::Integer; Δ = 50.0, Δz = Δ, Q = 0.1, T_end = 3600.0, avg = 900.0,
                 z_i0 = 600.0, Γ = 0.003, θ₀ = 300.0, seed = 1, wall_cap::Bool = false,
                 surface_factor = 3.9, C_advective = 0.4, progress::Bool = false,
                 on_step = nothing)
    rng = Xoshiro(seed)
    L, H = N * Δ, N * Δz
    m = RegularGrid(; size = (N, N, N), extent = (L, L, H), halo = (3, 3, 3))
    sim = BoussinesqSimulation(m; θ₀ = θ₀, θ_flux_bottom = Q, sponge_α_max = 0.01,
                               sponge_z_start = 0.75H,
                               bcs = FieldBoundaryConditions(:periodic, :periodic, :wall),
                               θ_bcs = FieldBoundaryConditions(:periodic, :periodic, :noflux))
    set!(sim.θ, (x, y, z) -> (z < z_i0 ? θ₀ : θ₀ + Γ * (z - z_i0)) +
                             (z < 200 ? 0.1 * (rand(rng) - 0.5) : 0.0))
    sm = SolidMask(m)
    c = TKEClosure(m; wall_distance = wall_cap ? wall_distance(sm; bottom = true) : nothing,
                   first_level = first_level_cells(sm; bottom = true),
                   surface_factor = surface_factor)
    initialize_tke!(c, sim; e_min = 1e-3)          # e = 0 is a fixed point: seed it
    cfl = CFLConstraints(; C_advective = C_advective, dt_max = 20.0)
    acc = nothing; nacc = 0; dt = nothing; steps = 0; min_e = Inf
    wall = @elapsed while sim.t < T_end - 1e-9
        h = min(adaptive_dt(sim, cfl; dt_prev = dt, sgs = c), T_end - sim.t)
        step_rk3!(sim, h; sgs = c)
        dt = h; steps += 1
        on_step === nothing || on_step(sim, c, h)
        min_e = min(min_e, minimum(interior_view(c.e)))
        if sim.t > T_end - avg
            p = _cbl_profiles(sim, c, Q)
            acc = acc === nothing ? p : map(+, acc, p)
            nacc += 1
        end
        progress && steps % 200 == 0 &&
            @info "cbl" t = round(sim.t; digits = 1) dt = round(h; digits = 3) max_e = maximum(interior_view(c.e))
    end
    P = map(x -> x ./ nacc, acc)
    zc = [(k - 0.5) * m.dz for k in 1:m.Nz]
    zf = [k * m.dz for k in 1:m.Nz-1]
    # the one z_i diagnostic: the height of the minimum total heat flux
    zi = boundary_layer_height(zf, P.flux; method = :min_flux)
    frac = P.e ./ (P.e .+ P.kres)
    sel = (zc .> 0.2zi) .& (zc .< 0.8zi)
    w_star = cbrt(sim.g_accel / θ₀ * Q * zi)
    kw = argmax(P.w2)
    st = tke_statistics(c)
    return (; P..., zc, zf, zi, min_flux_ratio = minimum(P.flux),
              max_sgs_fraction = maximum(frac[sel]), sgs_fraction = frac,
              z_epeak = zc[argmax(P.e)], w_star, w2max_norm = P.w2[kw] / w_star^2,
              z_w2max_norm = zc[kw] / zi, clips = st.clips, min_e, steps, wall, N,
              Δ, Δz, Q, T_end)
end

# horizontal-mean profiles of one state
function _cbl_profiles(sim, c, Q)
    m = sim.mesh
    Hx, Hy, Hz, Nx, Ny, Nz = m.Hx, m.Hy, m.Hz, m.Nx, m.Ny, m.Nz
    u = interior_view(sim.u); v = interior_view(sim.v); θ = interior_view(sim.θ)
    e = interior_view(c.e); κ = interior_view(sim.κt)
    wd = sim.w.data[Hx+1:Hx+Nx, Hy+1:Hy+Ny, Hz+1:Hz+Nz+1]       # all w faces, walls included
    uc = 0.5 .* (u .+ circshift(u, (-1, 0, 0)))
    vc = 0.5 .* (v .+ circshift(v, (0, -1, 0)))
    wc = 0.5 .* (wd[:, :, 1:end-1] .+ wd[:, :, 2:end])
    # the plane mean, as Statistics.mean(a; dims = (1, 2)) computes it (the
    # sum times 1//n; Statistics is not a dependency of the package)
    pm(a) = vec(sum(a; dims = (1, 2)) .* (1 // (size(a, 1) * size(a, 2))))
    pv(a) = pm(a .^ 2) .- pm(a) .^ 2
    θf = 0.5 .* (θ[:, :, 1:end-1] .+ θ[:, :, 2:end])
    wf = wd[:, :, 2:end-1]
    κf = 0.5 .* (κ[:, :, 1:end-1] .+ κ[:, :, 2:end])
    res = pm(wf .* θf) .- pm(wf) .* pm(θf)
    sgs = pm(.-κf .* (θ[:, :, 2:end] .- θ[:, :, 1:end-1]) ./ m.dz)
    return (e = pm(e), kres = 0.5 .* (pv(uc) .+ pv(vc) .+ pv(wc)), w2 = pv(wc),
            flux = (res .+ sgs) ./ Q, sgs_flux = sgs ./ Q, θ = pm(θ))
end

"""
    tke_cbl_summary(r) -> String

One line of the numbers of a [`tke_cbl`](@ref) result.
"""
tke_cbl_summary(r) =
    @sprintf("CBL %d³ (Δ = %.0f m, Δz = %.0f m), %d steps, %.1f s: z_i = %.0f m, min flux ratio %.3f, max SGS TKE fraction (0.2–0.8 z_i) %.3f, e peak at z = %.0f m (%.2f z_i), σ_w² max %.2f w*² at z/z_i = %.2f, clips %d",
             r.N, r.Δ, r.Δz, r.steps, r.wall, r.zi, r.min_flux_ratio, r.max_sgs_fraction,
             r.z_epeak, r.z_epeak / r.zi, r.w2max_norm, r.z_w2max_norm, r.clips)

end # module
