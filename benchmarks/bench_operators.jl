# Benchmarks for individual FV kernels — one grid size, all key stencils.
#
# Run: julia --project=benchmarks benchmarks/bench_operators.jl

using Aeolus
using Aeolus.Fields: set!
using BenchmarkTools
using Printf

const N = 64

function build_grid_and_fields()
    mesh = RegularGrid(; size=(N,N,N), extent=(2π, 2π, 2π), halo=(2,2,2))
    bcs  = FieldBoundaryConditions(:periodic)

    p    = Field(CellCenter(), mesh); set!(p, (x,y,z) -> sin(x)*cos(y)*sin(z))
    φ    = Field(CellCenter(), mesh); set!(φ, (x,y,z) -> cos(x)*sin(y)*cos(z))
    θ    = Field(CellCenter(), mesh); set!(θ, (x,y,z) -> sin(x + y))

    u    = Field(FaceX(), mesh); set!(u, (x,y,z) ->  sin(x)*cos(y)*cos(z))
    v    = Field(FaceY(), mesh); set!(v, (x,y,z) -> -cos(x)*sin(y)*cos(z))
    w    = Field(FaceZ(), mesh); set!(w, (x,y,z) ->  0.0)

    dpdx = Field(FaceX(), mesh)
    divu = Field(CellCenter(), mesh)
    tu   = Field(FaceX(), mesh); tv = Field(FaceY(), mesh); tw = Field(FaceZ(), mesh)
    tθ   = Field(CellCenter(), mesh)
    νt   = Field(CellCenter(), mesh)

    for f in (p, φ, θ, u, v, w); fill_halos!(f, bcs); end
    return (; mesh, bcs, p, φ, θ, u, v, w, dpdx, divu, tu, tv, tw, tθ, νt)
end

function print_row(label, bench)
    med   = median(bench.times) / 1e6         # ms
    minmt = minimum(bench.times) / 1e6
    npts  = N^3
    ns_per_pt = median(bench.times) / npts
    @printf("  %-38s   %8.3f ms   min %8.3f   %6.1f ns/point\n",
            label, med, minmt, ns_per_pt)
end

s = build_grid_and_fields()

println("Aeolus operator benchmarks — grid $(N)^3 = $(N^3) points")
println("─"^92)

print_row("∂x   (CellCenter → FaceX)",       @benchmark ∂x($s.dpdx, $s.p))
print_row("∂x   (CellCenter → CellCenter)",  @benchmark ∂x($s.φ,    $s.p))
print_row("divergence (FaceX,Y,Z → CC)",     @benchmark divergence($s.divu, $s.u, $s.v, $s.w))
print_row("laplacian  (CellCenter)",         @benchmark laplacian($s.divu, $s.p))
print_row("laplacian  (FaceX)",              @benchmark laplacian($s.dpdx, $s.u))
print_row("momentum_advect!",                @benchmark momentum_advect!($s.tu, $s.tv, $s.tw, $s.u, $s.v, $s.w))
print_row("advect_centered! (staggered θ)",  @benchmark advect_centered!($s.tθ, $s.u, $s.v, $s.w, $s.θ))
print_row("compute_smagorinsky_νt!",         @benchmark compute_smagorinsky_νt!($s.νt, $s.u, $s.v, $s.w, 0.17))
print_row("add_eddy_viscosity_u!",           @benchmark add_eddy_viscosity_u!($s.tu, $s.u, $s.νt))
print_row("add_buoyancy!",                   @benchmark add_buoyancy!($s.tw, $s.θ, 300.0, 9.81))
print_row("add_coriolis!",                   @benchmark add_coriolis!($s.tu, $s.tv, $s.u, $s.v, 1e-4))
print_row("fill_halos! (CellCenter)",        @benchmark fill_halos!($s.p, $s.bcs))

println("─"^92)
