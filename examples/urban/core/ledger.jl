# Energy and water budget ledger — STUB.
#
# The ledger issue (smoke + coverage tests) fills this in: each coupling's
# `budget!(c, case, ledger)` adds its terms, and the ledger reports the
# domain surface energy residual Rn − H − LE − G − ΔS per class and the
# water closure. Here it is only the container, so couplings can already
# be written against it.

"""
    Ledger

Named budget terms, `terms[:name] => value` (W m⁻² or kg m⁻² s⁻¹ as the
coupling documents). A stub: see `core/ledger.jl`.
"""
struct Ledger
    terms::Dict{Symbol,Float64}
end
Ledger() = Ledger(Dict{Symbol,Float64}())

"Add `value` to ledger term `name`."
add_term!(l::Ledger, name::Symbol, value::Real) =
    (l.terms[name] = get(l.terms, name, 0.0) + Float64(value); l)

"Clear all terms and call every coupling's `budget!`."
function collect_budget!(case)
    empty!(case.ledger.terms)
    for c in case.couplings
        budget!(c, case, case.ledger)
    end
    return case.ledger
end
