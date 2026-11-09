# The update of the Reynolds-stress closure (#560): its source update
# allocates nothing per cell, and (below) the update as one cache-blocked
# pass on the blocked driver of #533/#554 (src/prognostic_sgs/
# reynolds_stress_blocked.jl) against the separate loops it replaces (the
# reference, what loop_fusion = "never" runs).

using Test
using Aeolus
using Aeolus.Fields: interior_view
using Aeolus.Operators: advance_sgs!
using Aeolus.Backends: CPUBackend, with_backend

const RSB = Aeolus.PrognosticSGS

include(joinpath(@__DIR__, "support", "rsc_blocked_cases.jl"))

@testset "Reynolds-stress source update: no allocation per cell (#560)" begin
    # The 6 × 6 system of every cell is stack-allocated (MMatrix); it
    # stays on the stack only if the elimination is inlined into the cell's
    # body. Before #560 it was not, and every cell allocated 368 B on the
    # heap. The update allocates its per-plane sums only: the same bytes on
    # a grid four times larger in x-y.
    al = map(((8, 8, 6), (16, 16, 6))) do n
        g = RegularGrid(; size = n, extent = 10.0 .* n, halo = (3, 3, 3))
        sim, c = rsc_case(g; transport = false)
        with_backend(CPUBackend(serial = true)) do
            RSB._rsm_update!(c, sim, 0.4); RSB._rsm_update!(c, sim, 0.4)
            @allocated RSB._rsm_update!(c, sim, 0.4)
        end
    end
    @test al[1] == al[2]
    @test al[2] < 2048
end
