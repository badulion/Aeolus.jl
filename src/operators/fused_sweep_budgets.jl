# =========================================================================
# The budgets inside the fused sweep (#533 plan step 7). Included by
# FiniteVolume, after fused_sweep.jl.
# =========================================================================
#
# With the budgets on, each in-core component's contribution is
# attributed per level: `scale × w_k Σ_ij∈air Δt_ijk`, the change of the
# level content of a tendency field over the component's entry
# (components/budgets.jl). Measured on the whole tendency field, that is
# two passes over the field per component and quantity, and it needs the
# tendency fields, which the sweep does not write. Inside the sweep the
# same sums are accumulated while the row is in cache: a booked term
# (`BookedTerm`) sums the row of each booked field of the tendency buffer
# over i (the air cells only, where the budget leaves solid cells out)
# before and after it adds its contribution, and adds the difference to
# its per-level sum (`LevelSums`). The budget weights `w_k`, the scale and
# the booking happen once per substage, after the sweep
# (`finish_sweep!` of the domain's budget term, Components).
#
# The per-level sums need no reduction across threads: the sweep cuts the
# k-range into slabs, one task each, and computes the tendencies of every
# row (j, k) of a slab, edge planes included, in that slab's task, so
# each level's sum is written by one task only. Its order of summation (i
# within a row, rows in j order) does not depend on the number of slabs.

"""
    SWEEP_BOOKED

The fields of the sweep's tendency buffer whose per-level sums a
[`BookedTerm`](@ref) can accumulate: u, v, θ, qv (the core tendencies of
the momentum, energy and water budgets).
"""
const SWEEP_BOOKED = (SWEEP_U, SWEEP_V, SWEEP_Θ, SWEEP_QV)

"""
    LevelSums(nslots, Nz, on, masked, fluid, Hx, Hy, Hz)

The per-level sums of what the booked terms of a fused sweep add to the
tendencies (#533 plan step 7): `acc[f, slot, k]` is `Σ_ij` over the
interior of level `k` (`1:Nz`) of term `slot`'s contribution to field
`SWEEP_BOOKED[f]`, for the fields with `on[f]`. Where `masked[f]`, only the
cells where `fluid` (interior-sized: nonzero, or `true`, for air; the
budget meter's mask) is set are summed. `Hx, Hy, Hz`: the halo widths
(storage index − halo = interior index). Zeroed before each sweep by the
owner of the sums.
"""
struct LevelSums{M<:AbstractArray}
    acc::Array{Float64,3}
    on::NTuple{4,Bool}
    masked::NTuple{4,Bool}
    fluid::M
    Hx::Int
    Hy::Int
    Hz::Int
end
LevelSums(nslots::Integer, Nz::Integer, on::NTuple{4,Bool}, masked::NTuple{4,Bool}, fluid::AbstractArray,
          Hx::Integer, Hy::Integer, Hz::Integer) =
    LevelSums(zeros(4, nslots, Nz), on, masked, fluid, Int(Hx), Int(Hy), Int(Hz))

"""
    BookedTerm(term, sums::LevelSums, slot)

`term` inside the fused sweep with its contribution to the booked
tendencies summed per level into `sums.acc[:, slot, :]` (#533 plan step
7): the sweep term of an in-core component the budgets attribute. Adds
exactly what `term` adds; its footprint, inputs and `prepare!` are
`term`'s. Per row it sums each booked field of the tendency buffer over
the row before and after `term`'s [`add_row!`](@ref) (in cache, no pass
over a field) and adds the difference to the level's sum.
"""
struct BookedTerm{T<:SweepTerm,L<:LevelSums} <: SweepTerm
    term::T
    sums::L
    slot::Int
end
footprint(t::BookedTerm) = footprint(t.term)
sweep_inputs(t::BookedTerm) = sweep_inputs(t.term)
prepare!(t::BookedTerm, sim, Δt) = prepare!(t.term, sim, Δt)
finish_sweep!(t::BookedTerm, sim, Δt) = finish_sweep!(t.term, sim, Δt)

# The per-point form (#533 step 8) has no per-level sums: a level is
# shared by many work items. Its point kernel adds the term's tendency
# only; the drivers keep a term tuple that books out of the per-point form
# (`books`, `builtin.budgets_per_point`).
@inline add_tendency!(t::BookedTerm, tb, i, j, k, st) = add_tendency!(t.term, tb, i, j, k, st)

"""
    books(t::SweepTerm) -> Bool

Whether the sweep term `t` books per-level sums inside the sweep (a
[`BookedTerm`](@ref), the domain's budget term): such a tuple runs only in
the ring-buffer sweep, whose rows of a level are summed by one task, and
not in the per-point form (integration of #533 steps 7 and 8).
"""
books(::SweepTerm) = false
books(::BookedTerm) = true

@inline function add_row!(t::BookedTerm, buf, a, b, s, j, k, ii, st)
    L = t.sums
    jl, kl = j - L.Hy, k - L.Hz
    s0 = _booked_sums(L, buf, a, b, s, ii, jl, kl)
    add_row!(t.term, buf, a, b, s, j, k, ii, st)
    s1 = _booked_sums(L, buf, a, b, s, ii, jl, kl)
    acc = L.acc
    @inbounds for f in 1:4
        L.on[f] && (acc[f, t.slot, kl] += s1[f] - s0[f])
    end
    return nothing
end

# the row sums of the booked fields (0 for the others)
@inline _booked_sums(L::LevelSums, buf, a, b, s, ii, jl, kl) =
    ntuple(f -> !L.on[f] ? 0.0 :
                L.masked[f] ? _row_sum(buf, a, b, SWEEP_BOOKED[f], s, ii, L.fluid, L.Hx, jl, kl) :
                              _row_sum(buf, a, b, SWEEP_BOOKED[f], s, ii), Val(4))

@noinline function _row_sum(buf, a, b, f, s, ii)
    acc = 0.0
    @inbounds @simd for i in ii
        acc += Float64(buf[i, a, b, f, s])
    end
    return acc
end
@noinline function _row_sum(buf, a, b, f, s, ii, fl, Hx, jl, kl)
    acc = 0.0
    @inbounds @simd for i in ii
        acc += ifelse(_is_air(fl[i-Hx, jl, kl]), Float64(buf[i, a, b, f, s]), 0.0)
    end
    return acc
end
@inline _is_air(x::Bool) = x
@inline _is_air(x::Real) = !iszero(x)
