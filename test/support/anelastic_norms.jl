# Field norms shared by the anelastic tests (test/era2_anelastic.jl) and the
# warm-bubble validation (validation/era2_anelastic.jl).

using Aeolus.Fields: interior_view

"L2 norm over a field's interior."
_l2(a) = sqrt(sum(abs2, a))

"Relative L2 difference between the interiors of two like-located fields."
_reldiff(a, b) = _l2(interior_view(a) .- interior_view(b)) / _l2(interior_view(b))
