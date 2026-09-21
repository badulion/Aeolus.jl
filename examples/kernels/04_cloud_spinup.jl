# Kernel demo 04 — Kessler cloud spinup.
#
# Cell-column model: initialize ql above the "cloud threshold" and let
# Kessler microphysics convert it to qr, then let rain fall out. Plot
# vertical profiles at multiple times.

using Aeolus
using Aeolus.Fields: set!, interior_view
using Printf

include(joinpath(@__DIR__, "..", "viz", "plot_utils.jl"))

const OUT = joinpath(@__DIR__, "..", "urban", "output"); mkpath(OUT)

Nx = Ny = 4; Nz = 32; H = 2000.0
mesh = RegularGrid(; size=(Nx,Ny,Nz), extent=(H,H,H), halo=(1,1,1))

qv = Field(CellCenter(), mesh); set!(qv, (x,y,z) -> 0.01)
ql = Field(CellCenter(), mesh); set!(ql, (x,y,z) -> 0.6H < z < 0.9H ? 3e-3 : 0.0)
qr = Field(CellCenter(), mesh); set!(qr, (x,y,z) -> 0.0)

zs = collect(Aeolus.Grids.znodes(mesh, Aeolus.Grids.CellCenter()))

function profile(f::Field)
    v = interior_view(f)
    return [sum(view(v, :, :, k)) / (size(v,1)*size(v,2)) for k in 1:size(v,3)]
end

# Snapshots
ql_snap = Vector{Vector{Float64}}()
qr_snap = Vector{Vector{Float64}}()
labels  = String[]
push!(ql_snap, profile(ql)); push!(qr_snap, profile(qr)); push!(labels, "t=0s")

for (i, dt) in enumerate((60.0, 60.0, 120.0, 120.0, 120.0))
    apply_kessler_step!(qv, ql, qr, dt; Vt=6.0)
    push!(ql_snap, profile(ql))
    push!(qr_snap, profile(qr))
    push!(labels, "t=$(round(Int, sum((60.0,60.0,120.0,120.0,120.0)[1:i])))s")
end

save_multiline(joinpath(OUT, "04_cloud_ql_profile.svg"),
               zs, ql_snap, labels;
               xlabel="z (m)", ylabel="ql (kg/kg)",
               title="Cloud water — Kessler spinup")

save_multiline(joinpath(OUT, "04_cloud_qr_profile.svg"),
               zs, qr_snap, labels;
               xlabel="z (m)", ylabel="qr (kg/kg)",
               title="Rain water — Kessler spinup")

@printf("Cloud spinup complete. Final ql max = %.2e, qr max = %.2e\n",
        maximum(profile(ql)), maximum(profile(qr)))
@printf("output → %s , %s\n",
        joinpath(OUT, "04_cloud_ql_profile.svg"),
        joinpath(OUT, "04_cloud_qr_profile.svg"))
