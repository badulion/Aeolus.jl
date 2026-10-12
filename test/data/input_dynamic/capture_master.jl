# Captured on the code before CFG-g (#362) with (first on master + #380, commit
# c33d292a; re-captured on master 5b488e11 after KA-m's fused saturation vapour
# pressure (ff7073e7) moved summer_qv_* in the last bits)
#   julia --project=. test/data/input_dynamic/capture_master.jl <checkout> master_digests.toml
# The CSV-path initial state
# of the summer case and the 12-step smoke "summer sounding" state.
using TOML
using Aeolus
include(joinpath(ARGS[1], "examples", "urban", "BerlinCase.jl"))
const BC = Aeolus.UrbanCase
include(joinpath(@__DIR__, "..", "..", "support", "berlin_configs.jl"))
Aeolus.Operators.set_fftw_policy!(; planner = :estimate)   # the round trip's planner
include(joinpath(ARGS[1], "test", "support", "berlin_fixture.jl"))

out = Dict{String,Any}()
cfg = site_config(berlin_toml("summer"))
mesh = Aeolus.Grids.RegularGrid(; size = cfg.size, extent = cfg.extent, halo = cfg.halo)
zc = [mesh.z0 + (k - 0.5) * mesh.dz for k in 1:mesh.Nz]
θ0 = BC.initial_θ(cfg); q0 = BC.initial_qv(cfg)
out["summer_theta_zc"] = digest([θ0(0.0, 0.0, z) for z in zc])
out["summer_qv_zc"] = digest([q0(0.0, 0.0, z) for z in zc])
zf = collect(range(0.0, 4000.0; length = 4001))
out["summer_theta_fine"] = digest([θ0(0.0, 0.0, z) for z in zf])
out["summer_qv_fine"] = digest([q0(0.0, 0.0, z) for z in zf])
out["summer_p_surface"] = BC.case_surface_pressure(cfg)
ref = BC.build_reference(cfg, mesh)
out["summer_ref_p"] = digest(ref.p̄); out["summer_ref_rho"] = digest(ref.ρ̄)
out["summer_ref_T"] = digest(ref.T̄); out["summer_ref_pi"] = digest(ref.π̄)
out["summer_zi"] = BC.initial_inversion_height(cfg)
pc = BC.ParentConfig(cfg)
out["summer_zi_max"] = BC.sounding_zi_max(cfg)
Ug, Vg = BC.parent_geostrophic(pc)
out["summer_ug"] = digest([Ug(z, t) for z in zf[1:40:end], t in (0.0, 17.3, 3600.0, 40000.5)])
out["summer_vg"] = digest([Vg(z, t) for z in zf[1:40:end], t in (0.0, 17.3, 3600.0, 40000.5)])

# the smoke "summer sounding" run, 12 steps
c = site_config(berlin_toml("smoke"); sounding = site_config(berlin_toml("summer")).sounding)
case = BC.build_case(c)
BC.run_case!(case)
st = berlin_state(case)
out["smoke_sounding_state"] = Dict{String,Any}(String(k) => digest(v) for (k, v) in pairs(st))
open(ARGS[2], "w") do io
    TOML.print(io, out)
end
println("done ", ARGS[2])
