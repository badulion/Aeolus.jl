# N-nest-d (#261, epic #246): one-way parent → child nesting in a
# convective boundary layer. A 1200² × 600 m parent (surface heat flux
# 0.1 K m/s, 2 m/s mean wind, Smagorinsky) carries a 600² × 400 m child at
# r = 2 for 2400 s; the overlap-mean θ(z), u(z) of child and parent over
# 1200–2400 s must agree within the run-to-run spread of the parent itself
# (4 seeds). The one-way boundary data stay tested in
# test/nest_boundary.jl (mass correction, advection through the nest,
# top boundary, solids, perfect parent, outflow, sponge, halos).
#
# References: Hellsten et al. (2021), GMD 14, 3185 (PALM nesting);
# Deardorff (1970), JAS 27, 1211 (w* = (g/θ₀ H zᵢ)^(1/3)).

using Test
using Printf
using Random
using Aeolus
using Aeolus.Nesting
using Aeolus.Fields: interior_view
using Aeolus.BoundaryConditions: FieldBoundaryConditions

isdefined(@__MODULE__, :_nb_init_child!) || include(joinpath(@__DIR__, "..", "test", "support", "nest_boundary.jl"))

_nb_zc(m, k) = m.z0 + (k - m.Hz - 0.5) * m.dz

@testset "one-way nest boundary data (N-nest-d, #261)" begin

# -------------------------------------------------------------------------
# A convective boundary layer (surface heat flux 0.1 K m/s, a 2 m/s mean
# wind, Smagorinsky) in a 1200² × 600 m parent with a 600² × 400 m child at
# r = 2 on the ground (open sides and top). After 1200 s of spin-up
# (~3 t* = zᵢ/w*, w* ≈ 1.2 m/s) the overlap-mean θ(z) and u(z), averaged
# over 1200–2400 s, of child and parent must agree within the run-to-run
# spread of the parent itself (4 seeds).
function _nb_cbl_parent(pm, seed)
    p = BoussinesqSimulation(pm; θ₀ = 300.0, C_s = 0.17, θ_flux_bottom = 0.1,
                             bcs = FieldBoundaryConditions(:periodic, :periodic, :wall))
    rng = Xoshiro(seed)
    for k in axes(p.θ.data, 3)
        z = _nb_zc(pm, k)
        p.θ.data[:, :, k] .= 300.0 + (z > 300 ? 0.005 * (z - 300) : 0.0)
    end
    θi = interior_view(p.θ)
    θi[:, :, 1:3] .+= 0.2 .* (rand(rng, size(θi, 1), size(θi, 2), 3) .- 0.5)
    fill!(p.u.data, 2.0)
    step!(p, 1e-3); p.t = 0.0
    return p
end

_nb_hmean(f, ci, cj, kk) = [sum(interior_view(f)[ci, cj, k]) / (length(ci) * length(cj)) for k in kk]

function _nb_cbl(seed; nested, T1 = 1200.0, T2 = 2400.0)
    pm = RegularGrid(Float64; size = (24, 24, 12), extent = (1200.0, 1200.0, 600.0), halo = (3, 3, 3))
    p = _nb_cbl_parent(pm, seed)
    nest = nested_grid(pm, ((300.0, 900.0), (300.0, 900.0), (0.0, 400.0)), 2)
    cm = nest_child_mesh(nest); cx, cy, cz = nest_covered_range(nest)
    cfl = CFLConstraints(C_advective = 0.5)
    c = nothing; run = nothing
    if nested
        c = BoussinesqSimulation(cm; θ₀ = 300.0, C_s = 0.17, θ_flux_bottom = 0.1,
                                 bcs = nest_bcs(cm, :nest, :nest; bottom = :wall, top = :nest))
        _nb_init_child!(c, p, nest)
        run = NestedRun(p, c, nest; feedback = :one_way, cfl_parent = cfl)
    end
    ap = zeros(length(cz), 2); ac = zeros(cm.Nz, 2); n = 0; dtp = nothing
    while p.t < T2
        if nested
            step_nested!(run)
        else
            dt = adaptive_dt(p, cfl; dt_prev = dtp); step!(p, dt); dtp = dt
        end
        p.t >= T1 || continue
        ap[:, 1] .+= _nb_hmean(p.θ, cx, cy, cz); ap[:, 2] .+= _nb_hmean(p.u, cx, cy, cz)
        if nested
            ac[:, 1] .+= _nb_hmean(c.θ, 1:cm.Nx, 1:cm.Ny, 1:cm.Nz)
            ac[:, 2] .+= _nb_hmean(c.u, 1:cm.Nx, 1:cm.Ny, 1:cm.Nz)
        end
        n += 1
    end
    ap ./= n
    # child levels averaged onto the parent levels (r_z = 2)
    pc = nested ? [(ac[2k-1, j] + ac[2k, j]) / (2n) for k in 1:length(cz), j in 1:2] : nothing
    return ap, pc, run
end

@testset "one-way consistency: convective boundary layer, overlap means" begin
    pp, pc, run = _nb_cbl(1; nested = true)
    runs = [pp, (first(_nb_cbl(s; nested = false)) for s in 2:4)...]
    spread = [maximum(a[k, j] for a in runs) - minimum(a[k, j] for a in runs)
              for k in axes(pp, 1), j in 1:2]
    d = abs.(pc .- pp)
    st = run.stats
    @printf("  CBL: %d parent steps, %d child substeps (n %d..%d), retries %d/%d\n",
            st.parent_steps, st.child_substeps, st.n_min, st.n_max, st.parent_retries,
            st.child_retries)
    for k in axes(pp, 1)
        @printf("    z = %3.0f m: θ parent %.4f child %.4f |Δ| %.4f spread %.4f K | u parent %.3f child %.3f |Δ| %.3f spread %.3f m/s\n",
                25.0 + 50.0 * (k - 1), pp[k, 1], pc[k, 1], d[k, 1], spread[k, 1],
                pp[k, 2], pc[k, 2], d[k, 2], spread[k, 2])
    end
    @printf("  max |Δθ| / spread = %.2f, max |Δu| / spread = %.2f\n",
            maximum(d[:, 1] ./ spread[:, 1]), maximum(d[:, 2] ./ spread[:, 2]))
    @test all(d[:, 1] .<= spread[:, 1])
    @test all(d[:, 2] .<= spread[:, 2])
    @test st.child_retries == 0 && st.parent_retries == 0
end

end
