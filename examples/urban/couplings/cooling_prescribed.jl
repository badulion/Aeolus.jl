# Clear-air radiative cooling: a prescribed uniform −2.315e-5 K/s
# (−2 K/day) on θ (Aeolus.Radiation.apply_radiative_cooling!).
#
# The column-radiation issue replaces it with a GasOptics + TwoStream
# column (slot `:column_radiation`, provider `:twostream`).

"""
    CoolingPrescribed

Slot `:column_radiation`, provider `:prescribed` (default): every step
after the surface update, θ += `rate`·dt at every level (`rate` =
−2.315e-5 K/s).
"""
struct CoolingPrescribed <: AbstractCoupling
    rate::Vector{Float64}
    buf::Vector{Float64}
end

coupling_name(::Type{CoolingPrescribed}) = :cooling_prescribed
slot(::Type{CoolingPrescribed}) = :column_radiation
provider(::Type{CoolingPrescribed}) = :prescribed
modules_used(::Type{CoolingPrescribed}) = (:Radiation,)
default_options(::Type{CoolingPrescribed}) = (rate = -2.315e-5,)
stages(::Type{CoolingPrescribed}) = ((:column_tendency, 10, EVERY_STEP),)
build(::Type{CoolingPrescribed}, case, o) =
    CoolingPrescribed(fill(Float64(o.rate), case.mesh.Nz), zeros(case.mesh.Nz))

function run_stage!(c::CoolingPrescribed, case, ::Val{:column_tendency}, dt)
    c.buf .= c.rate .* dt
    apply_radiative_cooling!(case.sim.θ, c.buf)
    return nothing
end

register!(CoolingPrescribed)
