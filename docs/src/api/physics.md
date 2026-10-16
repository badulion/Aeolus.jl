# Physics modules

Era I physics components layered on the dynamical core, plus the I/O
and parallel-decomposition machinery.

## I/O and checkpointing

Full-state checkpoints, restart and periodic checkpoints have
their own page: [I/O, checkpoints and restart](io.md).

## NetCDF output

NetCDF is read and written with NCDatasets (#310). `Aeolus.NCIO` holds
the output policy: NetCDF-4, deflate with shuffle, chunks of one
horizontal level and one record, streamed records, CF time units. Read
any file with `NCDataset(path)` and index it (`ds["theta"][:, :, k, r]`;
Julia index order, the reverse of `ncdump`).

**Migration from `NetCDFIO`** (removed in #310):

| before | now |
|:-- | :-- |
| `nc_create(path; attributes)` | same, returns an `NCDataset` (NetCDF-4) |
| `nc_defdim!(nc, "x", n)`, `nc_defdim!(nc, "time", nothing)` | `ds.dim["x"] = n`, `ds.dim["time"] = Inf` |
| `nc_defvar!(nc, name, T, dims; attributes)` | same (now compressed and chunked) |
| `nc_endef!(nc)` | not needed |
| `nc_write!(nc, name, A)` | `ds[name][:, :] = A` |
| `nc_append_record!(nc, data)` | same |
| `nc_close!(nc)` | `close(ds)` |
| `nc_write_fields(path, coords, fields)` | removed (unused) |
| `f = nc_open(path)`, `nc_read(f, name)` | `ds = NCDataset(path)`, `Array(ds[name])` |
| `nc_read_record(f, name, r)` | `ds[name][:, :, :, r]` |
| `nc_nrecords(f)`, `nc_varnames(f)`, `nc_attributes(f)` | `ds.dim["time"]`, `keys(ds)`, `ds.attrib` |

Files written before #310 are NetCDF-3 (CDF-2) and still read. Their
time units, `"seconds since simulation start"`, are not CF: NCDatasets
warns and returns the raw seconds. New files carry the case start in
UTC (`"seconds since 2019-06-29 04:00:00"`) and `ds["time"][:]` decodes to
`DateTime`; the raw seconds are `ds["time"].var[:]`.

```@autodocs
Modules = [Aeolus.NCIO]
```

## Microphysics

```@autodocs
Modules = [Aeolus.Microphysics]
```

## Immersed boundaries

```@autodocs
Modules = [Aeolus.IBM]
```

## Domain decomposition

```@autodocs
Modules = [Aeolus.Parallel]
```

## Compute backends

```@autodocs
Modules = [Aeolus.Backends]
```

## Spectral element (future backend)

```@autodocs
Modules = [Aeolus.SpectralElement]
```

## Radiation (Era I scaffold)

Superseded for production use by the Era III stack; retained because it
is cheap and adequate for idealised cases. `SlabLand` is the first
implementer of the Era V surface interface
([SurfaceExchange](era5/surface_exchange.md)): `surface_temperature` is
the SurfaceExchange generic, and `step_surface!` computes its own
Monin–Obukhov fluxes.

```@autodocs
Modules = [Aeolus.Radiation]
```

## Chemistry

```@autodocs
Modules = [Aeolus.Chemistry]
```

## Lagrangian particles

```@autodocs
Modules = [Aeolus.Particles]
```

## Machine-learning closures

```@autodocs
Modules = [Aeolus.ML]
```

## Data assimilation

```@autodocs
Modules = [Aeolus.DataAssimilation]
```
