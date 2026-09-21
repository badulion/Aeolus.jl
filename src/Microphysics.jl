module Microphysics

# Phase 12 — Kessler warm-rain microphysics.
#
# The Kessler (1969) parameterization is the simplest cloud-microphysics
# scheme: three water mixing-ratio categories (qv, ql, qr) exchange mass
# via
#
#   * autoconversion  : ql above a threshold converts to qr
#   * accretion       : qr grows by collecting ql
#   * rain fallout    : qr sediments downward at a terminal velocity
#
# Evaporation of falling rain and a full saturation adjustment (which
# couples to θ and pressure) are deferred to a later refinement.
#
# The kernel here is fully local — it operates cell-by-cell on the
# prognostic fields.

using ..Grids: RegularGrid, interior
using ..Fields: Field, interior_view

export apply_kessler_step!

"""
    apply_kessler_step!(qv, ql, qr, dt; k_auto=1e-3, ql_c=1e-3, k_acc=2.2,
                        Vt=5.0, mesh=qv.mesh)

Advance one microphysics substep in-place. Handles autoconversion,
accretion, and 1st-order upwind sedimentation of `qr` at fallout speed
`Vt` (m/s downward, ≥ 0).

  * `k_auto` [1/s]  — autoconversion rate.
  * `ql_c` [kg/kg]  — cloud-water threshold above which autoconversion
    activates.
  * `k_acc` [1/s]   — accretion rate.
  * `Vt` [m/s]      — rain terminal fall speed (≥ 0).

`qv` is untouched here — evaporation of falling rain is a follow-up.
"""
function apply_kessler_step!(qv::Field, ql::Field, qr::Field, dt::Real;
                             k_auto::Real=1e-3, ql_c::Real=1e-3,
                             k_acc::Real=2.2, Vt::Real=5.0,
                             mesh::RegularGrid = qv.mesh)
    dz  = mesh.dz
    Δt  = dt
    ii, jj, kk = interior(mesh)

    qld = ql.data; qrd = qr.data

    # 1. Autoconversion + accretion (source/sink between ql and qr).
    @inbounds for k in kk, j in jj, i in ii
        qle = qld[i,j,k]
        qre = qrd[i,j,k]
        auto = k_auto * max(qle - ql_c, 0.0)
        accr = k_acc * qle * (max(qre, 0.0) ^ (7/8))
        dqr = (auto + accr) * Δt
        # cap conversion to available ql
        dqr = min(dqr, max(qle, 0.0))
        qld[i,j,k] -= dqr
        qrd[i,j,k] += dqr
    end

    # 2. Rain fallout — 1st-order upwind (rain falls downward).
    # ∂qr/∂t += ∂(Vt qr)/∂z (positive Vt means fallout, so qr gained
    # from above by 1 cell and lost downward).
    if Vt > 0
        α = Vt * Δt / dz
        α = min(α, 0.5)   # keep CFL stable
        @inbounds for k in kk, j in jj, i in ii
            k_up   = k + 1
            k_down = k - 1
            in_from_above = k_up <= last(kk) ? qrd[i,j,k_up] : 0.0
            out_to_below  = qrd[i,j,k]
            qrd[i,j,k] += α * (in_from_above - out_to_below)
        end
    end

    return (qv, ql, qr)
end

end # module
