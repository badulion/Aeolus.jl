module Chemistry

# Phase 20 (component) — Passive tracer with linear first-order chemistry.
#
# Represents any single reactive tracer with lifetime `τ` and optional
# spatially-varying source `S(x, y, z)`:
#
#     ∂c/∂t + u · ∇c = S - c / τ
#
# Advection is handled outside this module (use the existing
# `advect_centered!` or `advect_weno5!` on the tracer field). Here we
# just supply the local source/sink term.
#
# A full KPP-generated multi-species mechanism is the natural follow-up
# but requires an external code-generation step outside session scope.

using ..Grids: RegularGrid, interior, CellCenter
using ..Fields: Field

export apply_linear_chemistry!

"""
    apply_linear_chemistry!(tc, c, τ; source = nothing)

Add first-order chemical relaxation `-c/τ` (plus an optional source
`S`) to the tracer tendency `tc`. If `source` is a scalar, it's applied
uniformly; if it's a 3D array (interior-sized), the per-cell values are
added.

Positive `τ` implies decay (photolysis, dry deposition); pass `τ = Inf`
for an inert tracer with only the source term.
"""
function apply_linear_chemistry!(tc::Field{CellCenter},
                                 c::Field{CellCenter},
                                 τ::Real;
                                 source = nothing)
    invτ = τ == Inf ? zero(eltype(tc.data)) : 1 / eltype(tc.data)(τ)
    ii, jj, kk = interior(tc.mesh)
    tcd = tc.data; cd = c.data
    if source === nothing
        @inbounds for k in kk, j in jj, i in ii
            tcd[i,j,k] -= invτ * cd[i,j,k]
        end
    elseif source isa Real
        s = eltype(tcd)(source)
        @inbounds for k in kk, j in jj, i in ii
            tcd[i,j,k] += s - invτ * cd[i,j,k]
        end
    elseif source isa AbstractArray{<:Real,3}
        size(source) == (length(ii), length(jj), length(kk)) ||
            error("apply_linear_chemistry!: source array size mismatch")
        @inbounds for (kk_, kv) in enumerate(kk),
                       (jj_, jv) in enumerate(jj),
                       (ii_, iv) in enumerate(ii)
            tcd[iv,jv,kv] += source[ii_,jj_,kk_] - invτ * cd[iv,jv,kv]
        end
    else
        error("apply_linear_chemistry!: source must be Nothing, Real, or 3-D Array")
    end
    return tc
end

end # module

# ---------------------------------------------------------------------------
# Two additional Phase-20 modules live in their own files: Particles, ML,
# DataAssimilation. They share the pragmatic-scaffold spirit.
# ---------------------------------------------------------------------------
