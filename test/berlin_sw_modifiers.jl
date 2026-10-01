# Berlin case: the shortwave modifiers and the radiosity class map (#128).
#
# Before #128 the tree shade and the cloud shadow multiplied `bus.sw` in
# place every step while `bus.sw` is recomputed only at the radiation
# cadence, so the park shortwave fell by exp(−0.5·2.5) = 0.287 per step
# (0.29 → 0.082 → 0.024 → 0.0068 of the street value in the :smoke
# window between two radiation updates). And the radiosity scene was
# built without a class map, so park and water ground facets had street
# material. These tests pin both fixes.

using Test
using Aeolus
using Aeolus.Fields: interior_view
using Statistics: mean

isdefined(Main, :BerlinCase) ||
    include(joinpath(@__DIR__, "..", "examples", "urban", "BerlinCase.jl"))
const BC = Main.BerlinCase

@testset "Berlin: shortwave modifiers and radiosity class map (#128)" begin
    cfg = BC.CaseConfig(:smoke; verbose = false, output_path = nothing)
    case = BC.build_case(cfg; io = devnull)
    g = case.geom
    sw_mod = BC.find_coupling(case, :sw_modifiers)
    @test sw_mod !== nothing
    shade = exp(-0.5 * cfg.park_lai)              # Beer's law, k_leaf = 0.5
    park = findall(==(BC.SURFACE_PARK), g.sfc)
    street = findall(i -> g.sfc[i] == BC.SURFACE_STREET && g.heightmap[i] == 0,
                     CartesianIndices(g.sfc))
    @test length(park) == 50 && length(street) > 50

    @testset "scene class map: park and water keep their class, wall-tagged cells are street" begin
        sfc = UInt8[BC.SURFACE_STREET BC.SURFACE_PARK BC.SURFACE_WATER;
                    BC.SURFACE_WALL   BC.SURFACE_ROOF BC.SURFACE_GLASS]
        @test BC.scene_class_map(sfc) == [URBAN_STREET URBAN_PARK URBAN_WATER;
                                          URBAN_STREET URBAN_ROOF URBAN_STREET]
        ground = [f for f in g.scene.facets if f.kind === :ground]
        @test count(f -> f.material == URBAN_PARK, ground) == length(park)
        @test all(f -> (f.material == URBAN_PARK) == (g.sfc[f.col...] == BC.SURFACE_PARK), ground)
    end

    @testset "park ground facets absorb with the park albedo" begin
        # a radiation step at noon, then the facet shortwave balance
        BC.step_case!(case)
        sol = case.bus.swsol
        @test sol !== nothing
        α = case.geom.scene.materials.albedo
        @test α[URBAN_PARK] > α[URBAN_STREET] + 0.03  # grass/trees are brighter than asphalt
        for (i, f) in enumerate(g.scene.facets)
            f.kind === :ground || continue
            @test sol.absorbed[i] ≈ (1 - α[f.material]) * sol.incident[i] rtol = 1e-12
        end
    end

    @testset "the modifiers apply once per radiation field, not per step" begin
        # no cloud water, so the cloud shadow (#130) leaves the base unchanged
        fill!(interior_view(BC.tracer(case, :ql)), 0.0)
        ratios = Float64[]; parksw = Float64[]
        for n in 2:12                                 # radiation fires at steps 4, 8, 12
            BC.step_case!(case)
            sw = case.bus.sw
            # exactly exp(−k·LAI) of the unmodified field on the park, 1 elsewhere
            @test all(sw[park] .≈ shade .* sw_mod.sw_base[park])
            @test all(I -> sw[I] == sw_mod.sw_base[I], setdiff(CartesianIndices(sw), park))
            push!(ratios, mean(sw[park]) / mean(sw[street]))
            push!(parksw, mean(sw[park]))
        end
        # Park / open-street shortwave: the same on every step and equal to
        # the Beer's-law factor to 1 % (unshaded park and street effective
        # irradiance differ by the geometry; measured ratio 0.2875 against
        # exp(−1.25) = 0.2865). Before #128 it fell to 0.0068 by the
        # fourth step after a radiation update.
        @test maximum(ratios) - minimum(ratios) < 1e-3
        @test all(r -> isapprox(r, shade; rtol = 0.01), ratios)
        # the park keeps a quarter of the noon sun, not a vanishing fraction
        @test minimum(parksw) > 200
    end

    @testset "a new radiation field becomes the base; cloud water does not act here" begin
        # Since #130 the cloud shadow is cloud_shadow.jl, applied at the
        # radiation cadence before this stage; sw_modifiers only shades.
        sw = case.bus.sw
        R = fill(800.0, size(sw))
        ql = interior_view(BC.tracer(case, :ql))
        fill!(ql, 0.0)
        c0 = first(street)
        ql[c0, 1] = 1e-3 / case.mesh.dz               # 1 g/m² over one open street column
        copyto!(sw, R)                                # a "radiation update"
        for _ in 1:4
            BC.run_stage!(sw_mod, case, Val(:sw_modifiers), cfg.dt)
            @test all(sw[park] .≈ 800 * shade)
            @test sw[street] == R[street]              # unshaded columns untouched, ql or not
        end
        copyto!(sw, 2 .* R)
        BC.run_stage!(sw_mod, case, Val(:sw_modifiers), cfg.dt)
        @test sw_mod.sw_base == 2 .* R
        @test all(sw[park] .≈ 1600 * shade)
        fill!(ql, 0.0)
        # the coupling's own smoke checks agree
        @test all(last, BC.smoke_check(sw_mod, case))
    end
end
