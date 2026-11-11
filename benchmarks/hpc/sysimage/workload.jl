# The precompile workload of a build's system image (#566): what a benchmark
# process runs, so its first step is compiled ahead. Berlin 100 m from
# ENV["SYSIMG_CFG"]: build, steps under each loop_fusion setting the build
# has, a few with the launch trace on, and the state mode's field copies.
using Aeolus, Serialization
cfg = Aeolus.load_run_config(ENV["SYSIMG_CFG"]; overrides = ["fftw.planner=\"estimate\""])
model = Aeolus.Config.build(Aeolus.SimulationDriver.SimulationDriverType(), cfg).model
S = Aeolus.Simulations
fus(s) = isdefined(S, :set_loop_fusion!) && foreach(d -> S.set_loop_fusion!(d.state, s), model.domains)
for s in (:if_possible, :never, :if_possible)
    fus(s); Aeolus.step!(model, 2.0)
end
B = Aeolus.Backends
if isdefined(B, :set_launch_trace!)
    B.set_launch_trace!(true); Aeolus.step!(model, 2.0); B.set_launch_trace!(false)
    B.launch_trace_summary(B.launch_trace()); B.reset_launch_trace!()
end
d = first(model.domains)
serialize(IOBuffer(), Array(Aeolus.Fields.interior_view(d.state.u)))
