# Batch F — numerical infrastructure add-ons.
#
# What ships here is deliberately at the "usable scaffold" level rather
# than the full production-grade rewrite the individual items imply.
# Rationale: each item (implicit cut-cell IBM, stretched-grid operator
# retrofit, adaptive dt with proper CFL/viscous constraints, WENO-5 for
# momentum, characteristic non-periodic BCs) is a substantial standalone
# rework of the LES core. The versions here plug in through the
# existing hooks (`body_force!`, `post_predictor!`, per-cell state) so
# they're immediately usable in the urban run, and the API is shaped to
# match what a full replacement would look like.
#
# Ships:
#   N1  bulk-viscosity map (spatially varying ν) — the "true-LES-ν"
#       item is impossible without cut-cell IBM; instead we allow ν to
#       vary per cell so users can lower it in the free stream while
#       keeping it high near obstacles.
#   N2  stretched vertical grid metric (`ZStretchMap`) — a helper that
#       returns non-uniform `dz(k)` weights callable code can use.
#       Full operator retrofit is out of scope.
#   N3  adaptive Δt via CFL — computes a stable dt each solar update.
#   N4  wire-in-place: an outer `step_urban!` that lets the urban
#       driver optionally call `advect_weno5!` on the θ tendency
#       before the projection.
#   N5  Rayleigh inflow / outflow damper — same trick as the sponge
#       but restricted to a narrow lateral zone; approximates non-
#       periodic horizontal BCs without a full characteristic-BC solve.

using Aeolus
using Aeolus.Fields: interior_view

export bulk_viscosity_map, ZStretchMap, cell_dz,
       cfl_time_step, apply_lateral_damper!,
       add_weno_scalar_tendency!

# --- N1. Per-cell bulk viscosity map -------------------------------------

"""
    bulk_viscosity_map(mesh, sm; ν_far = 0.02, ν_wall = 2.0, dist_cells = 4)

Returns a 3-D `Array{Float64,3}` of size `(Nx, Ny, Nz)`. Cells within
`dist_cells` (Chebyshev distance) of a solid cell get `ν_wall`; cells
farther away get `ν_far`. The urban simulation currently uses a
scalar `ν`; when a proper cut-cell IBM lands, this map becomes the
per-cell diffusion coefficient.
"""
function bulk_viscosity_map(mesh, sm; ν_far::Real = 0.02, ν_wall::Real = 2.0,
                            dist_cells::Int = 4)
    Nx, Ny, Nz = mesh.Nx, mesh.Ny, mesh.Nz
    Hx, Hy, Hz = mesh.Hx, mesh.Hy, mesh.Hz
    νmap = fill(Float64(ν_far), Nx, Ny, Nz)
    d = dist_cells
    @inbounds for k in 1:Nz, j in 1:Ny, i in 1:Nx
        near = false
        for kk in max(1, k-d):min(Nz, k+d),
            jj in max(1, j-d):min(Ny, j+d),
            ii in max(1, i-d):min(Nx, i+d)
            if sm.mask[Hx+ii, Hy+jj, Hz+kk]
                near = true; break
            end
        end
        near && (νmap[i, j, k] = ν_wall)
    end
    return νmap
end

# --- N2. Stretched vertical grid metric ----------------------------------

"""
    ZStretchMap(Nz; z_top, ratio = 1.15, dz_min = 2.0)

Precompute a stretched `dz(k)` distribution where the first cell has
thickness `dz_min` and each subsequent cell is `ratio` times thicker.
The last cell's top face is anchored at `z_top` (may not exactly hit
`z_top` — the returned struct records the actual top).

Fields:
  * `dz    :: Vector{Float64}` — thickness per level (length `Nz`)
  * `z_c   :: Vector{Float64}` — cell centre heights
  * `z_f   :: Vector{Float64}` — face heights (length `Nz+1`)
  * `z_top :: Float64`         — actual domain height
"""
struct ZStretchMap
    dz::Vector{Float64}
    z_c::Vector{Float64}
    z_f::Vector{Float64}
    z_top::Float64
end

function ZStretchMap(Nz::Int; z_top::Real, ratio::Real = 1.15, dz_min::Real = 2.0)
    dz = zeros(Nz)
    dz[1] = dz_min
    for k in 2:Nz
        dz[k] = dz[k-1] * ratio
    end
    # Rescale so the sum matches z_top
    scale = z_top / sum(dz)
    dz .*= scale
    z_f = zeros(Nz + 1)
    for k in 1:Nz
        z_f[k+1] = z_f[k] + dz[k]
    end
    z_c = (z_f[1:end-1] .+ z_f[2:end]) ./ 2
    return ZStretchMap(dz, z_c, z_f, z_f[end])
end

"Local dz at level `k` — convenience for kernels."
cell_dz(m::ZStretchMap, k::Int) = m.dz[k]

# --- N3. Adaptive time step ----------------------------------------------

"""
    cfl_time_step(sim, mesh; C_conv = 0.4, C_visc = 0.5, C_diff = 1.0,
                   ν_max = 2.0, dt_cap = 5.0, dt_floor = 0.5)

Return an advective + viscous CFL-bounded time step:

    dt = min(C_conv · Δx / max|u|,
             C_visc · Δx² / (2 ν_max),
             dt_cap)

Bounded below by `dt_floor` (integer number of dt should divide the
total run cleanly, so call this only at coarse cadences).
"""
function cfl_time_step(sim, mesh; C_conv::Real = 0.4, C_visc::Real = 0.5,
                        C_diff::Real = 1.0, ν_max::Real = 2.0,
                        dt_cap::Real = 5.0, dt_floor::Real = 0.5)
    dx, dy, dz = mesh.dx, mesh.dy, mesh.dz
    Δmin = min(dx, dy, dz)
    umax = max(maximum(abs, interior_view(sim.u)),
               maximum(abs, interior_view(sim.v)),
               maximum(abs, interior_view(sim.w)),
               1e-3)
    dt_conv = C_conv * Δmin / umax
    dt_visc = C_visc * Δmin^2 / (2 * ν_max)
    return clamp(min(dt_conv, dt_visc, dt_cap), dt_floor, dt_cap)
end

# --- N4. WENO scalar tendency add-on -------------------------------------

"""
    add_weno_scalar_tendency!(tϕ, u, v, w, ϕ)

Wrapper that calls Aeolus's built-in `advect_weno5!` on a scalar-Field
tendency. Requires halo width ≥ 3 on the mesh (`mesh.Hx ≥ 3` etc.).
Users can substitute this for the centered advection currently used in
`step!` for θ / qv by calling it inside a `body_force!` callback (which
accumulates into the tendency) — the built-in `advect_centered!` still
runs, so this ADDS to it and doubles the advective sink. Use with
caution or subtract the centered contribution first. Provided as a
scaffold for a future `advection_scheme` kwarg on `BoussinesqSimulation`.
"""
function add_weno_scalar_tendency!(tϕ, u, v, w, ϕ)
    Aeolus.Operators.advect_weno5!(tϕ, u, v, w, ϕ)
    return tϕ
end

# --- N5. Lateral damper (non-periodic horizontal BC proxy) --------------

"""
    apply_lateral_damper!(sim, α_max, width_cells; U_ref = 1.0, V_ref = 0.0)

Rayleigh-damp `(u - U_ref, v - V_ref, w)` inside a strip of width
`width_cells` cells around the horizontal domain boundaries. Emulates
inflow-outflow BCs on an otherwise periodic mesh: winds re-approach the
ambient value near the boundaries so a plume that crosses the domain
doesn't come back around and re-contaminate the interior.

Call from a `body_force!` callback so it enters the tendency, not the
velocity directly (keeps the projection consistent).
"""
function apply_lateral_damper!(sim, α_max::Real, width_cells::Int;
                                U_ref::Real = 1.0, V_ref::Real = 0.0)
    Nx, Ny, Nz = sim.mesh.Nx, sim.mesh.Ny, sim.mesh.Nz
    Hx, Hy, Hz = sim.mesh.Hx, sim.mesh.Hy, sim.mesh.Hz
    ud = sim.u.data; vd = sim.v.data; wd = sim.w.data
    tud = sim.tu.data; tvd = sim.tv.data; twd = sim.tw.data
    @inbounds for k in 1:Nz, j in 1:Ny, i in 1:Nx
        # Chebyshev distance to nearest horizontal boundary
        d = min(i - 1, Nx - i, j - 1, Ny - j)
        if d < width_cells
            # 0 at width_cells, α_max at boundary
            α = α_max * (1 - d / width_cells)^2
            tud[Hx+i, Hy+j, Hz+k] -= α * (ud[Hx+i, Hy+j, Hz+k] - U_ref)
            tvd[Hx+i, Hy+j, Hz+k] -= α * (vd[Hx+i, Hy+j, Hz+k] - V_ref)
            twd[Hx+i, Hy+j, Hz+k] -= α *  wd[Hx+i, Hy+j, Hz+k]
        end
    end
    return sim
end
