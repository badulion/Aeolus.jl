# Transcripts of the column-physics loops before KA-m (#301), kept as the
# bit-for-bit reference of test/ka_columns.jl. Each is the pre-port loop
# body as it was on the base of the port (KA-b #367 merged with master at
# c617bc1), with its serial loop order and its running accumulators.

module LegacyColumns

using Aeolus
using Aeolus.MoistThermodynamics: DEFAULT_CONSTANTS, PhasePartition, exner_function
using Aeolus.RadiationFog: fog_condensation

"RadiationFog.fog_adjust! before KA-m: one sweep, one running total."
function fog_adjust!(θ, qv, ql, p, dt; τ = 0.0, levels = axes(θ, 3), c = DEFAULT_CONSTANTS)
    Nx, Ny, _ = size(θ)
    total = 0.0
    @inbounds for k in levels
        pk = p isa AbstractVector ? float(p[k]) : float(p)
        Π  = exner_function(pk, PhasePartition(0.0), c)
        for j in 1:Ny, i in 1:Nx
            T = θ[i, j, k] * Π
            r = fog_condensation(T, pk, qv[i, j, k], ql[i, j, k], dt; τ = τ, c = c)
            r.Δq == 0 && continue
            θ[i, j, k]  = r.T / Π
            qv[i, j, k] = r.qv
            ql[i, j, k] = r.ql
            total += r.Δq
        end
    end
    return total
end

end # module
