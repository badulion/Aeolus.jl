# Passive scalar advection in a prescribed 2D solid-body rotation flow.
#
# This is Phase 0's end-to-end check: it exercises the mesh, fields, halo
# fill, upwind advection kernel, and RK3 time stepper. A Gaussian blob is
# advected in a constant rotational flow; after one full revolution it
# should return (approximately) to its initial position.

using Aeolus
using Printf

# --- grid ---------------------------------------------------------------
Nx, Ny, Nz = 128, 128, 1
L = 1.0
mesh = RegularGrid(Float64; size=(Nx, Ny, Nz), extent=(L, L, L),
                            halo=(2, 2, 2), origin=(-L/2, -L/2, 0.0))

# --- prescribed rotational velocity field -------------------------------
# u = -ω y,  v = ω x, w = 0 → rigid rotation about the origin.
ω = 2π                                # one revolution per unit time
u = Field(CellCenter(), mesh)
v = Field(CellCenter(), mesh)
w = Field(CellCenter(), mesh)         # zero in 2D
Aeolus.Fields.set!(u, (x, y, z) -> -ω * y)
Aeolus.Fields.set!(v, (x, y, z) ->  ω * x)

# --- initial scalar: off-center Gaussian --------------------------------
ϕ = Field(CellCenter(), mesh)
x0, y0, σ = 0.25, 0.0, 0.08
Aeolus.Fields.set!(ϕ, (x, y, z) -> exp(-((x-x0)^2 + (y-y0)^2) / (2σ^2)))

# --- boundary conditions ------------------------------------------------
bcs = FieldBoundaryConditions(:periodic)
fill_halos!(u, bcs); fill_halos!(v, bcs); fill_halos!(w, bcs); fill_halos!(ϕ, bcs)

# --- time stepping ------------------------------------------------------
state = (ϕ = ϕ,)
stepper = RK3(state)

# Compute a stable dt from Courant condition (u_max ~ ω * L/2)
umax = ω * L / 2
CFL  = 0.4
dt   = CFL * mesh.dx / umax

T_final = 1.0                         # one full revolution
nsteps  = ceil(Int, T_final / dt)
dt      = T_final / nsteps

function tendency!(k, state, t)
    ϕ = state.ϕ
    kϕ = k.ϕ
    fill!(kϕ.data, 0)
    fill_halos!(ϕ, bcs)
    advect_upwind!(kϕ, u, v, w, ϕ)
    return k
end

# Diagnostics: initial mass and max
mass0 = sum(Aeolus.Fields.interior_view(state.ϕ)) * mesh.dx * mesh.dy * mesh.dz
max0  = maximum(Aeolus.Fields.interior_view(state.ϕ))

@printf("2D advection: N = %d × %d, dt = %.4e, nsteps = %d\n", Nx, Ny, dt, nsteps)
@printf("t = %.3f    mass = %.6e    max = %.6f\n", 0.0, mass0, max0)

t = 0.0
for step in 1:nsteps
    step!(stepper, state, tendency!, dt, t)
    global t += dt
    if step % max(1, nsteps ÷ 8) == 0
        mass = sum(Aeolus.Fields.interior_view(state.ϕ)) * mesh.dx * mesh.dy * mesh.dz
        mx   = maximum(Aeolus.Fields.interior_view(state.ϕ))
        @printf("t = %.3f    mass = %.6e    max = %.6f\n", t, mass, mx)
    end
end

# Final check: peak should be near the starting point (upwind is diffusive
# so amplitude drops, but the location should be right).
ϕf = Aeolus.Fields.interior_view(state.ϕ)
xs = collect(Aeolus.Grids.xnodes(mesh, CellCenter()))
ys = collect(Aeolus.Grids.ynodes(mesh, CellCenter()))
imax = argmax(ϕf[:, :, 1])
xpeak, ypeak = xs[imax[1]], ys[imax[2]]
@printf("peak location after 1 rev: (%.3f, %.3f)   expected: (%.3f, %.3f)\n",
        xpeak, ypeak, x0, y0)
