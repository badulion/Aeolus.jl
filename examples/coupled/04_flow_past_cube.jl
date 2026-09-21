# Coupled example 04 — flow past a cube (IBM) with tracer wake.
#
# Uniform wind approaches a solid cube (via IBM SolidMask); a passive
# tracer is emitted from the cube's leeward face; wake structure is
# revealed by the tracer field.

using Aeolus
using Aeolus.Fields: set!, interior_view
using Aeolus.Operators: advect_centered!
using Printf

include(joinpath(@__DIR__, "..", "viz", "plot_utils.jl"))

const OUT = joinpath(@__DIR__, "..", "urban", "output"); mkpath(OUT)

Lx = 200.0; Ly = 80.0; Lz = 80.0
Nx = 96; Ny = 48; Nz = 40
mesh = RegularGrid(; size=(Nx,Ny,Nz), extent=(Lx,Ly,Lz), halo=(2,2,2))

sim = BoussinesqSimulation(mesh;
    ν = 1e-1, C_s = 0.15,
    bcs   = FieldBoundaryConditions(:periodic, :periodic, :wall))

U_inf = 3.0
set!(sim.u, (x,y,z) -> U_inf)

# Cube: 20×20×20 m, centered laterally, floor-mounted, at x = 60 m
cube_x0, cube_x1 = 60.0, 80.0
cube_y0, cube_y1 = 30.0, 50.0
cube_z0, cube_z1 = 0.0,  20.0
sm = SolidMask(mesh)
mark_solid!(sm, (x,y,z) ->
    cube_x0 <= x <= cube_x1 && cube_y0 <= y <= cube_y1 && cube_z0 <= z <= cube_z1)

# Passive tracer field
c  = Field(CellCenter(), mesh)
tc = Field(CellCenter(), mesh)
set!(c, (x,y,z) -> 0.0)

# Leeward-face source: cells just downstream of the cube at cube's height
Hx, Hy, Hz = mesh.Hx, mesh.Hy, mesh.Hz
i_src_start = round(Int, cube_x1 / mesh.dx) + 1
i_src_end   = i_src_start + 1
j_src = (round(Int, cube_y0 / mesh.dy)+1) : (round(Int, cube_y1 / mesh.dy))
k_src = round(Int, (cube_z1 - 2) / mesh.dz)

dt = 0.15 * mesh.dx / U_inf
T_end = 40.0
nsteps = ceil(Int, T_end / dt); dt = T_end / nsteps

force! = (sim, Δt) -> add_ibm_penalty!(sim, sm; τ = 3*Δt)
for n in 1:nsteps
    step!(sim, dt; body_force! = force!)
    # advect the tracer
    fill_halos!(c, sim.bcs)
    fill!(tc.data, 0)
    advect_centered!(tc, sim.u, sim.v, sim.w, c)
    apply_linear_chemistry!(tc, c, 30.0)   # decay
    # inject at cube's leeward face
    for i in i_src_start:i_src_end, j in j_src
        tc.data[Hx+i, Hy+j, Hz+k_src] += 1.0
    end
    interior_view(c) .+= dt .* view(tc.data, Hx+1:Hx+Nx, Hy+1:Hy+Ny, Hz+1:Hz+Nz)
end

# Vertical center-plane slice: y = Ly/2
j_center = round(Int, (Ly/2) / mesh.dy)
slice_u = view(interior_view(sim.u), :, j_center, :)     # shape (Nx, Nz)
slice_c = view(interior_view(c),      :, j_center, :)

# Rotate to image (Nz rows, Nx cols) with z increasing upward
save_heatmap(joinpath(OUT, "coupled_04_u_slice.ppm"),
             reverse(Array(slice_u'); dims=1); upscale=6)
save_heatmap(joinpath(OUT, "coupled_04_tracer.ppm"),
             reverse(Array(slice_c'); dims=1); upscale=6)

@printf("Flow past cube complete. U range: [%.2f, %.2f], tracer max: %.2e\n",
        minimum(slice_u), maximum(slice_u), maximum(slice_c))
