# Fields

A [`Field`](@ref Aeolus.Fields.Field) is a scalar field on a mesh at a
given [`Location`](@ref Aeolus.Grids.Location). It carries its own
mesh reference and provides bounds-checked indexing plus a halo-aware
`interior_view`.

```@docs
Aeolus.Fields.Field
Aeolus.Fields.location
Aeolus.Fields.mesh
Aeolus.Fields.interior_view
Aeolus.Fields.set!
```
