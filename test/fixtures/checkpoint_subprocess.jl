# One process of the cross-process checkpoint test (#245, test/io_checkpoint.jl):
#
#   julia --project=. test/fixtures/checkpoint_subprocess.jl <role> <dir>
#
# role = "first":  an ESTIMATE run of 4 steps → <dir>/estimate_first.jls; then a
#                  PATIENT run through a CheckpointSchedule (wisdom saved at run
#                  start, checkpoint at step 2) to step 4 → <dir>/patient_continuous.jls
# role = "second": the same ESTIMATE run → <dir>/estimate_second.jls; then the
#                  PATIENT run resumed from the step-2 checkpoint with the saved
#                  wisdom, to step 4 → <dir>/patient_restart.jls
#
# Each output holds the fields with halos, the TKE closure memory and the
# FFTW policy the process ran with.

using Aeolus
using Aeolus.Fields: set!
using Aeolus.BoundaryConditions: FieldBoundaryConditions
using Random, Serialization

role, dir = ARGS[1], ARGS[2]

function sub_sim()
    # 32×32×16: the smallest grid measured where PATIENT and ESTIMATE plans of the
    # pressure solvers differ in the last bit (max |Δp| ≈ 1.3e-18 on a unit
    # source; 16×12×10 gives identical plans)
    m = RegularGrid(; size = (32, 32, 16), extent = (3200.0, 3200.0, 800.0), halo = (3, 3, 3))
    sim = BoussinesqSimulation(m; θ₀ = 300.0, f_coriolis = 1e-4, θ_flux_bottom = 0.05,
                               bcs = FieldBoundaryConditions(:periodic, :periodic, :wall))
    rng = Xoshiro(7)
    set!(sim.u, (x, y, z) -> 2.0 + 0.6 * randn(rng))
    set!(sim.v, (x, y, z) -> 0.6 * randn(rng))
    set!(sim.θ, (x, y, z) -> 300.0 + 0.004 * z + 0.2 * randn(rng))
    Aeolus.TimeIntegration._project!(sim, 1.0)
    c = TKEClosure(m); initialize_tke!(c, sim)
    return sim, c
end

record(sim, c) = (fields = map(f -> copy(f.data), checkpoint_fields(sim)),
                  closure = checkpoint_state(c), t = sim.t, fftw = fftw_policy())

# 1. ESTIMATE, fresh, 4 steps
set_fftw_policy!(; planner = :estimate)
sim, c = sub_sim()
for _ in 1:4; step_rk3!(sim, 2.0; sgs = c); end
serialize(joinpath(dir, "estimate_$(role).jls"), record(sim, c))

# 2. PATIENT through a CheckpointSchedule in the run folder dir/run
set_fftw_policy!(; planner = :patient)
ck = CheckpointSchedule(joinpath(dir, "run"); every_steps = 1000, keep = 1, verbose = false)
sim, c = sub_sim()
st = StepperState(:rk3)
d = resume!(ck, sim; sgs = c, stepper = st)
role == "first" ? (d === nothing || error("first process found a checkpoint")) :
                  (d === nothing && error("second process found no checkpoint"))
while st.step < 4
    step_rk3!(sim, 2.0; sgs = c); record_step!(st, 2.0)
    role == "first" && st.step == 2 && maybe_checkpoint!(ck, sim; step = 2, force = true, sgs = c, stepper = st)
end
serialize(joinpath(dir, role == "first" ? "patient_continuous.jls" : "patient_restart.jls"), record(sim, c))
