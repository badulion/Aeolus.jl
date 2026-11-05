# Era VI, Phase 54 (P54-b, #173): validation of the Lagrangian-averaged
# dynamic Smagorinsky closure of Meneveau, Lund & Cabot (1996, JFM 319, 353)
# in simulated isotropic turbulence (32³):
#
#   4. the domain-mean C_s at t = 1 and 3 against the plane value (MLC96
#      fig. 6), the memory following a static Smagorinsky LES passively;
#   5. positivity of I_LM, I_MM and ν_t over 200 steps with the closure
#      driving the flow.
#
# The module stays tested in test/era6_lagrangian_sgs.jl (relaxation,
# pathline transport, laminar shear, the masked cube, the contract).

using Test
using Aeolus
using Aeolus.Fields: interior_view, set!, Field
using Random: Xoshiro, randn, rand
using Statistics: mean
using Printf

const OPS = Aeolus.Operators

# A solenoidal random field with E(k) ∝ k^(-5/3) up to k_c, u_rms = 1 (the
# initialiser of P54-c; shared with era2_sgs_numerics.jl, #317).
isdefined(@__MODULE__, :kolmogorov_field!) || include(joinpath(@__DIR__, "..", "test", "support", "kolmogorov.jl"))

function _lag_plane_cs(sim; mask = nothing)
    ws = DynamicSmagorinskyWorkspace(sim.mesh)
    νt = Field(CellCenter(), sim.mesh)
    for f in (sim.u, sim.v, sim.w)
        fill_halos!(f, sim.bcs)
    end
    compute_dynamic_smagorinsky_νt!(νt, sim.u, sim.v, sim.w; workspace = ws, mask)
    return mean(sqrt.(ws.cs2)), ws, νt
end

@testset "Era VI — Lagrangian dynamic Smagorinsky in isotropic turbulence (P54-b, #173)" begin


    mesh = RegularGrid(; size = (32, 32, 32), extent = (2π, 2π, 2π), halo = (3, 3, 3))

    @testset "4. isotropic turbulence: C_s against the plane value (MLC96 fig. 6)" begin
        # The flow of era2_sgs_numerics.jl (static Smagorinsky 0.17 LES of a
        # k^(-5/3) field, dt = 0.01); the Lagrangian memory follows it
        # passively (advance_sgs! after every step), so both averages see
        # the same resolved field.
        sim = BoussinesqSimulation(mesh; C_s = 0.17, ν = 1e-4)
        kolmogorov_field!(sim, Xoshiro(1))
        lag = LagrangianDynamicSmagorinsky(mesh)
        out = Dict{Int,NTuple{2,Float64}}()
        for n in 1:300
            step!(sim, 0.01)
            advance_sgs!(lag, sim, 0.01)
            if n in (100, 300)
                OPS.sgs_viscosity!(Field(CellCenter(), mesh), lag, sim)
                out[n] = (mean(sqrt.(lag.workspace.cs2)), first(_lag_plane_cs(sim)))
            end
        end
        for (n, t) in ((100, 1), (300, 3))
            cl, cp = out[n]
            @printf("  [isotropic] t = %d: Lagrangian mean C_s %.4f, plane %.4f (%+.1f %%)\n",
                    t, cl, cp, 100 * (cl / cp - 1))
            @test abs(cl / cp - 1) < 0.15
        end
        @test 0.10 < out[100][1] < 0.20 && 0.10 < out[300][1] < 0.20
    end

    @testset "5. positivity with the closure driving the flow (200 steps)" begin
        sim = BoussinesqSimulation(mesh; ν = 1e-4)              # C_s = 0: only the closure
        kolmogorov_field!(sim, Xoshiro(2))
        lag = LagrangianDynamicSmagorinsky(mesh)
        ok_LM = ok_MM = ok_ν = true
        clipmax = 0.0
        for n in 1:200
            step!(sim, 0.01; sgs = lag)
            ok_LM &= minimum(lag.I_LM) >= 0
            ok_MM &= minimum(lag.I_MM) > 0
            ok_ν &= minimum(interior_view(sim.νt)) >= 0
            clipmax = max(clipmax, mean(lag.workspace.cs2 .>= lag.cs_max^2))
        end
        @test ok_LM && ok_MM && ok_ν
        @test clipmax < 0.01
        @test all(isfinite, sim.u.data) && maximum(interior_view(sim.νt)) > 0
        @printf("  [positivity] 200 steps: min I_LM %.2e, min I_MM %.2e, max clip fraction %.2e, I_LM = 0 in %.2f %% of cells, mean C_s %.4f\n",
                minimum(lag.I_LM), minimum(lag.I_MM), clipmax, 100 * mean(lag.I_LM .== 0),
                mean(sqrt.(lag.workspace.cs2)))
    end
end
