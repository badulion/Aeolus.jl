# N-parent-e (#259): the Deardorff / Moeng prognostic SGS TKE closure
# (src/PrognosticSGS.jl) on a vertically stretched StretchedGrid — the SGS
# closure of the Berlin nesting parent, which is the child's default `:sgs`
# provider (`:tke`) run on the parent's 25 m → 100 m levels.
#
# What changes on the stretched grid, and what is asserted:
#   * the shear production is still the exact dual of the (stretched)
#     Laplacian SGS momentum operator: Σ ν D V = −Σ u·T V;
#   * N² is the dual of the stretched κ_t scalar operator: Σ(−κ N²)V is the
#     potential-energy change (g/θ₀)Σ z T_θ V of the SGS heat flux;
#   * the filter width is local, Δ_k = (Δx Δy Δz_k)^{1/3};
#   * a :uniform StretchedGrid reproduces the RegularGrid closure;
#   * a stretched convective run keeps e ≥ 0, ν_t ≥ 0 and finite.
#
# Numbers measured on this branch are quoted next to each assertion.

using Test
using Aeolus
using Aeolus.Fields: interior_view, set!
using Aeolus.StretchedGrids: StretchedGrid, surface_refined_faces
using Aeolus.PrognosticSGS: dual_strain!, TKEClosure, tke_budget, tke_statistics, initialize_tke!
using Aeolus.Operators: buoyancy_frequency_squared!, add_eddy_diffusion_scalar!
using Aeolus.Operators.FiniteVolume: _add_eddy_viscosity_u!, _add_eddy_viscosity_v!,
                                     _add_eddy_viscosity_w!
using Aeolus.AdvectionSchemes: advect_positive!
using Random: Xoshiro

@testset "Era VI — SGS TKE on a stretched grid (N-parent-e, #259)" begin

zf = surface_refined_faces(dz_surface = 25.0, z_uniform = 200.0, z_top = 1200.0,
                           ratio = 1.08, dz_max = 100.0)
Nz = length(zf) - 1
grid(; halo = (2, 2, 2)) = StretchedGrid(; size = (8, 6, Nz), extent = (250.0, 180.0, 1200.0),
                                          halo = halo, stretching = (:custom, zf))
bcs = FieldBoundaryConditions(:periodic, :periodic, :wall)
lvl(g, a, k) = a[g.Hz+k]

@testset "shear production: the exact dual of the stretched momentum operator" begin
    g = grid()
    rng = Xoshiro(1)
    u = Field(FaceX(), g); v = Field(FaceY(), g); w = Field(FaceZ(), g); ν = Field(CellCenter(), g)
    for f in (u, v, w)
        interior_view(f) .= randn(rng, size(interior_view(f)))
    end
    interior_view(ν) .= rand(rng, size(interior_view(ν)))
    for f in (u, v, w, ν)
        fill_halos!(f, bcs)
    end
    D = Field(CellCenter(), g)
    dual_strain!(D, u, v, w, bcs)
    tu = Field(FaceX(), g); tv = Field(FaceY(), g); tw = Field(FaceZ(), g)
    _add_eddy_viscosity_u!(tu, u, ν, g); _add_eddy_viscosity_v!(tv, v, ν, g)
    _add_eddy_viscosity_w!(tw, w, ν, g)
    A = g.dx * g.dy
    sk(a, b, k) = sum(view(interior_view(a), :, :, k) .* view(interior_view(b), :, :, k))
    lhs = A * sum(sk(ν, D, k) * lvl(g, g.dzc, k) for k in 1:Nz)
    rhs = -A * sum(sk(u, tu, k) * lvl(g, g.dzc, k) + sk(v, tv, k) * lvl(g, g.dzc, k) +
                   sk(w, tw, k) * lvl(g, g.dzf, k) for k in 1:Nz)
    @test abs(lhs / rhs - 1) < 1e-12            # measured 2.2e-16
    @test lhs > 0

    # u = S z: D = S² (z_{k+1} − z_{k−1})/(2Δz_k) inside, S² where uniform
    S = 0.01
    fill!(u.data, 0); fill!(v.data, 0); fill!(w.data, 0)
    for k in 1:Nz
        interior_view(u)[:, :, k] .= S * lvl(g, g.zc, k)
    end
    fill_halos!(u, bcs)
    dual_strain!(D, u, v, w, bcs)
    worst = 0.0; dev = zeros(Nz)
    for k in 2:Nz-1
        ex = S^2 * (lvl(g, g.zc, k + 1) - lvl(g, g.zc, k - 1)) / (2 * lvl(g, g.dzc, k))
        worst = max(worst, maximum(abs.(view(interior_view(D), :, :, k) .- ex)) / ex)
        dev[k] = abs(ex / S^2 - 1)
    end
    @test worst < 1e-12                         # measured 2.7e-15
    # the dual is a first-order estimate of S² where the stretching changes:
    # 1.7 % in the last uniform cell (25 m under 26.7 m), 0.15 % on the
    # geometric part (ratio 1.08: (1/(2r) + 1 + r/2)/2 − 1 = 1.48e-3)
    @test maximum(dev[10:Nz-1]) < 1.6e-3        # measured 1.481e-3
    @test maximum(dev) < 0.02                   # measured 1.70e-2 (level 8)
    @test all(isapprox.(view(interior_view(D), :, :, 2:7), S^2; rtol = 1e-12))   # uniform levels
end

@testset "N²: the dual of the stretched κ_t operator; linear profile" begin
    g = grid()
    θbcs = FieldBoundaryConditions(:periodic, :periodic; bottom = :noflux, top = :noflux)
    rng = Xoshiro(2)
    θ = Field(CellCenter(), g); κ = Field(CellCenter(), g); N2 = Field(CellCenter(), g)
    T = Field(CellCenter(), g)
    θ₀, gr = 300.0, 9.81
    interior_view(θ) .= 300 .+ randn(rng, size(interior_view(θ)))
    interior_view(κ) .= rand(rng, size(interior_view(κ)))
    fill_halos!(θ, θbcs); fill_halos!(κ, θbcs)
    buoyancy_frequency_squared!(N2, θ; θ₀ = θ₀, g = gr)
    add_eddy_diffusion_scalar!(T, θ, κ, κ)
    A = g.dx * g.dy
    sk(a, b, k) = sum(view(interior_view(a), :, :, k) .* view(interior_view(b), :, :, k))
    Pb = -A * sum(sk(κ, N2, k) * lvl(g, g.dzc, k) for k in 1:Nz)
    PE = gr / θ₀ * A * sum(lvl(g, g.zc, k) * sum(view(interior_view(T), :, :, k)) * lvl(g, g.dzc, k)
                           for k in 1:Nz)
    @test abs(Pb / PE - 1) < 1e-11              # measured 4.0e-15

    Γ = 3e-3
    for k in 1:Nz
        interior_view(θ)[:, :, k] .= 290 + Γ * lvl(g, g.zc, k)
    end
    fill_halos!(θ, θbcs)
    buoyancy_frequency_squared!(N2, θ; θ₀ = θ₀, g = gr)
    dev = [abs(interior_view(N2)[1, 1, k] / (gr / θ₀ * Γ) - 1) for k in 2:Nz-1]
    @test maximum(dev[9:end]) < 1.6e-3          # measured 1.481e-3 (geometric part)
    @test maximum(dev) < 0.02                   # measured 1.70e-2 (where the stretching starts)
    @test all(isapprox.(interior_view(N2)[:, :, 2:7], gr / θ₀ * Γ; rtol = 1e-12))
end

@testset "positive flux-limited advection on the stretched grid" begin
    g = grid()
    rng = Xoshiro(4)
    u = Field(FaceX(), g); v = Field(FaceY(), g); w = Field(FaceZ(), g)
    ϕ = Field(CellCenter(), g); t = Field(CellCenter(), g)
    for f in (u, v, w)
        interior_view(f) .= 2 .* randn(rng, size(interior_view(f)))
    end
    fill_halos!(u, bcs); fill_halos!(v, bcs); fill_halos!(w, bcs)
    # sparse positive field: most cells zero, as e in stable air
    interior_view(ϕ) .= max.(randn(rng, size(interior_view(ϕ))) .- 1.0, 0.0)
    sbcs = FieldBoundaryConditions(:periodic, :periodic; bottom = :noflux, top = :noflux)
    fill_halos!(ϕ, sbcs)
    dt = 1.0                                    # Σ|u|Δt/Δx ≤ 0.6 here
    advect_positive!(t, u, v, w, ϕ, dt; scheme = :centered, bcs = sbcs)
    ϕ1 = interior_view(ϕ) .+ dt .* interior_view(t)
    @test minimum(ϕ1) >= -1e-14                 # measured -6.9e-18 (round-off of a zero cell)
    # conservative: Σ tend V = 0 with periodic sides and the walls' w = 0
    V = [g.dx * g.dy * lvl(g, g.dzc, k) for k in 1:Nz]
    tot = sum(sum(view(interior_view(t), :, :, k)) * V[k] for k in 1:Nz)
    ref = sum(abs.(interior_view(t))[:, :, k] |> sum for k in 1:Nz) * maximum(V)
    @test abs(tot) < 1e-12 * ref                # measured 1.2e-18
    @test_throws ArgumentError advect_positive!(t, u, v, w, ϕ, dt; scheme = :weno_z5, bcs = sbcs)
end

@testset "local filter width and relative volumes" begin
    g = grid()
    c = TKEClosure(g)
    @test length(c.Δk) == Nz
    @test all(c.Δk[k] ≈ (g.dx * g.dy * lvl(g, g.dzc, k))^(1 / 3) for k in 1:Nz)
    @test c.Δ == c.Δk[1]
    @test c.vrel[1] == 1 && c.vrel[end] ≈ lvl(g, g.dzc, Nz) / lvl(g, g.dzc, 1)
    @test c.advection === :centered
    @test_throws ArgumentError TKEClosure(g; advection = :weno_z5)
    @test_throws ArgumentError TKEClosure(g; mask = falses(size(c.e.data)))
    # a RegularGrid closure is unchanged: Δ everywhere, unit relative volumes
    r = RegularGrid(; size = (8, 6, 10), extent = (250.0, 180.0, 250.0), halo = (3, 3, 3))
    cr = TKEClosure(r)
    @test all(==(cr.Δ), cr.Δk) && all(==(1.0), cr.vrel)
    @test cr.Δ == (r.dx * r.dy * r.dz)^(1 / 3) && cr.advection === :weno_z5
end

# A dry convective box: surface heat flux, linear stable θ above 300 m.
function cbl(mesh; transport = false, steps = 20, dt = 2.0)
    sim = BoussinesqSimulation(mesh; ν = 0.0, θ₀ = 300.0, f_coriolis = 1e-4,
                               θ_flux_bottom = 0.1, bcs = bcs,
                               θ_bcs = FieldBoundaryConditions(:periodic, :periodic; bottom = :noflux,
                                                               top = :noflux))
    rng = Xoshiro(3)
    set!(sim.θ, (x, y, z) -> 300 + max(z - 300, 0) * 3e-3)
    interior_view(sim.θ)[:, :, 1:4] .+= 0.1 .* rand(rng, mesh.Nx, mesh.Ny, 4)
    set!(sim.u, (x, y, z) -> 2.0)
    c = TKEClosure(mesh; transport = transport)
    initialize_tke!(c, sim; e_min = 1e-4)
    for _ in 1:steps
        step!(sim, dt; sgs = c)
    end
    return sim, c
end

@testset "uniform StretchedGrid = RegularGrid (closure without transport)" begin
    r = RegularGrid(; size = (8, 6, 16), extent = (250.0, 180.0, 400.0), halo = (2, 2, 2))
    s = StretchedGrid(; size = (8, 6, 16), extent = (250.0, 180.0, 400.0), halo = (2, 2, 2))
    sr, cr = cbl(r); ss, cs = cbl(s)
    de = maximum(abs.(interior_view(cs.e) .- interior_view(cr.e))) / maximum(interior_view(cr.e))
    du = maximum(abs.(interior_view(ss.u) .- interior_view(sr.u)))
    @test de < 1e-10                            # measured 8.5e-17
    @test du < 1e-10                            # measured 0.0
    br, bs = tke_budget(cr, sr), tke_budget(cs, ss)
    @test bs.shear ≈ br.shear rtol = 1e-10
    @test bs.buoyancy ≈ br.buoyancy rtol = 1e-10
end

@testset "stretched convective run: positive, finite, energy-consistent" begin
    g = grid()
    sim, c = cbl(g; transport = true, steps = 30)
    e = interior_view(c.e)
    @test all(isfinite, e) && minimum(e) >= 0
    @test minimum(interior_view(sim.νt)) >= 0
    st = tke_statistics(c)
    @test st.steps == 30
    # the flux-limited centred transport keeps e ≥ 0 without clipping
    # (the plain centred flux form needed 370 clips in these 30 steps)
    @test st.clips == 0                         # measured 0
    b = tke_budget(c, sim)
    @test b.shear >= 0 && b.dissipation > 0 && b.tke > 0
    # the surface heat flux makes the lowest levels unstable: buoyancy production
    @test b.buoyancy > 0
end

end
