# Outgoing-vortex harness for open and nested boundaries (N-open-c, #260;
# kept for the nesting tests of N-nest-e, #263).
#
# A Lamb–Oseen vortex (core radius r_c = 8Δx, axis along y) is carried by
# a uniform stream U through a child box with open x faces, between
# free-slip walls at z = 0 and z = Lz (a 2-D x–z flow, Ny = 1). The walls
# make the far field decay exponentially, ∝ exp(−π|x|/Lz) (the image
# system is a row of alternating vortices): a periodic row would carry a
# vortex sheet's uniform jump of ±Γ/(2L) in the cross-stream velocity,
# which no uniform parent can match and which a doubly periodic truth
# cannot hold. The truth is the same
# flow in a periodic domain three times longer, on the same grid, with the
# child window as its middle third; it doubles as the "parent" that feeds
# the child's boundary data. The child and the truth use the same core
# (step_rk3!, centred momentum advection, inviscid) and the same Δt, so
# the difference between them is the error made at the open boundary.
#
# Parents (boundary data of the child):
#   :uniform — ProfileBoundaryData(u = U, v = w = 0): a parent that does
#              not contain the vortex (what a coarse parent sees);
#   :truth   — the truth itself, linearly interpolated in time between
#              its stored states (every `parent_every` child steps; 1 =
#              the perfect parent).
#
# R is the relative L2 error of the child velocity (u and w over the
# interior faces) against the truth at the same time, normalised by the
# L2 norm of the initial vortex perturbation in the child window.
#
# Reference: Lamb–Oseen vortex, u_θ(r) = Γ/(2πr) (1 − exp(−r²/r_c²))
# (e.g. Saffman 1992, Vortex Dynamics, §1.3); the test geometry follows
# the outgoing-vortex tests of non-reflecting boundary conditions
# (Colonius 2004, Annu. Rev. Fluid Mech. 36, 315).

using Aeolus
using Aeolus.Grids: interior, FaceX, FaceY, FaceZ, CellCenter
using Aeolus.Fields: Field
using Aeolus.BoundaryConditions: OpenPlane, AbstractBoundaryData
using Aeolus.Simulations: BoussinesqSimulation, step_rk3!
using Aeolus.Operators: divergence, ∂x, ∂y, ∂z
using Aeolus.PressureSolvers: solve!

Base.@kwdef struct VortexHarness
    Nx::Int = 192           # child cells along x (the truth has 3Nx)
    Nz::Int = 48            # between free-slip walls (6 r_c)
    Δ::Float64 = 1.0
    H::Int = 3              # halo (WENO-Z5 depth)
    U::Float64 = 1.0
    vmax::Float64 = 0.5     # peak swirl velocity
    rc_cells::Float64 = 8.0
    x0_cells::Float64 = 96.0  # initial vortex centre in the child (cells)
    dt::Float64 = 0.2
end

_ovh_rc(h::VortexHarness) = h.rc_cells * h.Δ
# Lamb–Oseen peak u_θ = 0.638 Γ/(2π r_c) at r = 1.1209 r_c
_ovh_Γ(h::VortexHarness) = h.vmax * 2π * _ovh_rc(h) /
                           ((1 - exp(-1.1209^2)) / 1.1209)

# Velocity (u, w) of a Lamb–Oseen vortex at (xc, zc = Lz/2) between
# free-slip walls at z = 0 and z = Lz: the vortex, its copies at
# zc + 2nLz and its mirror images of opposite sign at zc + (2n − 1)Lz,
# truncated symmetrically about the channel centre (an unpaired far image
# would add a uniform stream Γ/(2πD) through the channel).
function _ovh_vortex(h::VortexHarness, x, z, xc, zc)
    Γ = _ovh_Γ(h); rc = _ovh_rc(h); Lz = h.Nz * h.Δ
    u = 0.0; w = 0.0
    images = [(zc + 2n * Lz, 1.0) for n in -20:20]
    for n in 1:20
        push!(images, (zc + (2n - 1) * Lz, -1.0), (zc - (2n - 1) * Lz, -1.0))
    end
    for (zv, sgn) in images
        dx = x - xc; dz = z - zv
        r2 = dx^2 + dz^2
        r2 == 0 && continue
        f = sgn * Γ / (2π * r2) * (1 - exp(-r2 / rc^2))
        u += -f * dz; w += f * dx
    end
    return u, w
end

# Discrete projection of a velocity field (makes the truth's initial state
# exactly divergence-free on its grid).
function _ovh_project!(sim)
    fill_halos!(sim.u, sim.bcs); fill_halos!(sim.v, sim.bcs); fill_halos!(sim.w, sim.bcs)
    divergence(sim.div_buf, sim.u, sim.v, sim.w)
    solve!(sim.φ, sim.div_buf, sim.solver)
    fill_halos!(sim.φ, sim.bcs)
    ∂x(sim.dφdx, sim.φ); ∂y(sim.dφdy, sim.φ); ∂z(sim.dφdz, sim.φ)
    ii, jj, kk = interior(sim.mesh)
    sim.u.data[ii, jj, kk] .-= sim.dφdx.data[ii, jj, kk]
    sim.v.data[ii, jj, kk] .-= sim.dφdy.data[ii, jj, kk]
    sim.w.data[ii, jj, kk] .-= sim.dφdz.data[ii, jj, kk]
    fill_halos!(sim.u, sim.bcs); fill_halos!(sim.v, sim.bcs); fill_halos!(sim.w, sim.bcs)
    return sim
end

function _ovh_truth(h::VortexHarness)
    L = h.Nx * h.Δ
    m = RegularGrid(size = (3h.Nx, 1, h.Nz), extent = (3L, h.Δ, h.Nz * h.Δ),
                    origin = (-L, 0.0, 0.0), halo = (h.H, h.H, h.H))
    sim = BoussinesqSimulation(m; bcs = FieldBoundaryConditions(:periodic, :periodic, :wall))
    xc = h.x0_cells * h.Δ; zc = h.Nz * h.Δ / 2
    H = h.H
    for k in axes(sim.u.data, 3), j in axes(sim.u.data, 2), i in axes(sim.u.data, 1)
        x = -L + (i - H - 1) * h.Δ; z = (k - H - 0.5) * h.Δ
        sim.u.data[i, j, k] = h.U + _ovh_vortex(h, x, z, xc, zc)[1]
    end
    for k in axes(sim.w.data, 3), j in axes(sim.w.data, 2), i in axes(sim.w.data, 1)
        x = -L + (i - H - 0.5) * h.Δ; z = (k - H - 1) * h.Δ
        sim.w.data[i, j, k] = _ovh_vortex(h, x, z, xc, zc)[2]
    end
    _ovh_project!(sim)
    return sim
end

"""
Boundary data from the truth: linear interpolation in time between two
stored states `(ta, A)` and `(tb, B)` (extrapolation outside), copied at
the storage indices shifted by `shift` cells along x.
"""
mutable struct TruthBoundaryData <: AbstractBoundaryData
    shift::Int
    ta::Float64
    tb::Float64
    A::Dict{Symbol,Array{Float64,3}}
    B::Dict{Symbol,Array{Float64,3}}
end

_ovh_state(sim) = Dict(:u => copy(sim.u.data), :v => copy(sim.v.data),
                       :w => copy(sim.w.data), :θ => copy(sim.θ.data))

function Aeolus.BoundaryConditions.boundary_values!(planes::AbstractVector{<:OpenPlane},
                                                    d::TruthBoundaryData, t)
    s = d.tb == d.ta ? 1.0 : (t - d.ta) / (d.tb - d.ta)
    for p in planes
        p.mode === :mirror && continue
        A = d.A[p.name]; B = d.B[p.name]
        for (c, k) in enumerate(p.krange), (b, j) in enumerate(p.jrange),
            (a, i) in enumerate(p.irange)
            it = i + d.shift
            p.data[a, b, c] = (1 - s) * A[it, j, k] + s * B[it, j, k]
        end
    end
    return planes
end

"""
    run_vortex_harness(h; outflow, parent, parent_every = 1, nsteps,
                       correction = :uniform, record_every = 0)

Run the child and the truth side by side. Returns `(R, Rmax, child,
truth, ke_child, ke_truth)`: `R` at the end, `Rmax` over the run, and
(every `record_every` steps) the child and truth-window kinetic energies
`∫|u|²` over the child interior.
"""
function run_vortex_harness(h::VortexHarness; outflow::Symbol = :dirichlet,
                            parent::Symbol = :uniform, parent_every::Int = 1,
                            nsteps::Int = round(Int, ((h.Nx - h.x0_cells + 3h.rc_cells) * h.Δ) /
                                                      (h.U * h.dt)),
                            correction::Symbol = :uniform, record_every::Int = 0)
    truth = _ovh_truth(h)
    L = h.Nx * h.Δ; H = h.H; Nx = h.Nx
    mc = RegularGrid(size = (Nx, 1, h.Nz), extent = (L, h.Δ, h.Nz * h.Δ),
                     origin = (0.0, 0.0, 0.0), halo = (H, H, H))
    td = TruthBoundaryData(Nx, 0.0, 0.0, _ovh_state(truth), _ovh_state(truth))
    data = parent === :uniform ? ProfileBoundaryData(u = h.U, v = 0.0, w = 0.0, θ = 0.0) :
           parent === :truth   ? td :
           throw(ArgumentError("parent must be :uniform or :truth"))
    bcs = open_boundary_conditions(mc, data; x = :open, y = :periodic, top = :wall,
                                   outflow = outflow, correction = correction, halo = H)
    child = BoussinesqSimulation(mc; bcs = bcs)
    child.u.data .= truth.u.data[Nx+1:2Nx+2H, :, :]
    child.w.data .= truth.w.data[Nx+1:2Nx+2H, :, :]
    ii, jj, kk = interior(mc)
    ip = first(ii):last(ii)                      # interior x faces (west face pinned)
    win(i) = i .+ Nx
    kw = first(kk)+1:last(kk)                    # interior z faces (wall faces are 0)
    perturb(s) = sqrt(sum(abs2, view(s.u.data, win(ip), jj, kk) .- h.U) +
                      sum(abs2, view(s.w.data, win(ii), jj, kw)))
    norm0 = perturb(truth)
    err() = sqrt(sum(abs2, view(child.u.data, ip, jj, kk) .- view(truth.u.data, win(ip), jj, kk)) +
                 sum(abs2, view(child.w.data, ii, jj, kw) .- view(truth.w.data, win(ii), jj, kw))) / norm0
    ke(s, sh) = (sum(abs2, view(s.u.data, sh(ip), jj, kk)) +
                 sum(abs2, view(s.w.data, sh(ii), jj, kw))) * h.Δ^3
    kec = Float64[]; ket = Float64[]; Rs = Float64[]
    Rmax = 0.0
    dtp = parent_every * h.dt
    tb = 0.0
    for n in 1:nsteps
        # keep the parent one parent step ahead of the child's step end
        while tb < child.t + h.dt - 1e-12
            td.A = _ovh_state(truth); td.ta = truth.t
            step_rk3!(truth, dtp)
            td.B = _ovh_state(truth); td.tb = truth.t
            tb = truth.t
        end
        step_rk3!(child, h.dt)
        if parent_every == 1
            e_ = err(); Rmax = max(Rmax, e_); push!(Rs, e_)
        end
        if record_every > 0 && n % record_every == 0
            push!(kec, ke(child, identity))
            # the truth window at the child's time (interpolated)
            s = (child.t - td.ta) / (td.tb - td.ta)
            kt = (1 - s) * (sum(abs2, view(td.A[:u], win(ip), jj, kk)) +
                            sum(abs2, view(td.A[:w], win(ii), jj, kw))) +
                 s * (sum(abs2, view(td.B[:u], win(ip), jj, kk)) +
                      sum(abs2, view(td.B[:w], win(ii), jj, kw)))
            push!(ket, kt * h.Δ^3)
        end
    end
    R = parent_every == 1 ? err() : NaN
    return (R = R, Rmax = Rmax, Rs = Rs, child = child, truth = truth, ke_child = kec, ke_truth = ket)
end
