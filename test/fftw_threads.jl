# #397: FFTW's threaded planner under MEASURE crashed (signal 11 in
# `spawn_apply`) while its threads ran on Julia tasks (FFTW.jl's `spawnloop`
# callback; upstream JuliaMath/FFTW.jl#306). Aeolus's FFTW policy puts the
# threaded transforms on FFTW's own threads (`_fftw_own_threads!`).
#
# Measured before the fix on a Ryzen AI 9 HX 370 (12 cores): planning a
# 256³ irfft under MEASURE at 12 Julia and 12 FFTW threads crashed 4 of 4
# processes; with the fix 0 of 4 (and 0 of 8 for the Berlin r2r plus this
# irfft). On a machine with fewer cores the old code may not crash, so a pass
# there proves less; it never fails a correct build.

using Test

const _FT_SCRIPT = """
using Aeolus, FFTW
const O = Aeolus.Operators
O.set_fftw_policy!(; planner = :measure, threads = Threads.nthreads())
z = randn(256, 256, 256)
q = plan_rfft(z; flags = O.fftw_planner_flags()); h = q * z
r = plan_irfft(h, 256; flags = O.fftw_planner_flags())
y = r * h
print("max error ", maximum(abs, y .- z))
"""

@testset "FFTW: threaded MEASURE planning of a 256³ irfft does not crash (#397)" begin
    project = dirname(Base.active_project())
    cmd = `$(Base.julia_cmd()) --startup-file=no -t 12 --project=$project -e $_FT_SCRIPT`
    out = IOBuffer(); err = IOBuffer()
    p = run(pipeline(ignorestatus(cmd); stdout = out, stderr = err))
    msg = String(take!(out)); e = String(take!(err))
    @test success(p)
    occursin("signal", e) && @info "fftw_threads.jl: the subprocess crashed" e
    # the round trip: FFTW's unnormalised inverse is normalised by irfft, so
    # z is recovered to round-off (256³ points, a few ulp of the O(5) values)
    m = match(r"max error ([0-9.eE+-]+)", msg)
    @test m !== nothing && parse(Float64, m[1]) < 1e-12
end
