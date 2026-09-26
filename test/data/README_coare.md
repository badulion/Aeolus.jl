# COARE reference data

Used by `test/era5_air_sea_flux.jl` (issue #39).

| file | what it is |
| :-- | :-- |
| `coare35_test_input.txt` | `test_35_data.txt` shipped with the NOAA-PSL COARE 3.5 Python reference: 116 hourly bulk records (wind, air T, RH, SST, radiation; heights 16 m). |
| `coare35_test_output.txt` | `test_35_output_py_082020.txt`, the reference's own output for that input (`jcool = 1`, cool skin on). |
| `coare35_generate.py` | script that runs the same reference (`coare35vn`, `jcool = 0`) on a **synthetic** input grid: 2–25 m/s, air–sea ΔT = −2 K (unstable) and +3 K (stable), with no waves, the wave-age option and the sea-state option (fully developed seas, Taylor & Yelland 2001). |
| `coare35_synthetic.txt` | output of that script. The inputs are a regular grid, not observations; the outputs are the reference model's. |

Source: <https://github.com/NOAA-PSL/COARE-algorithm>, commit
`bd1cac80d2dae454e699f9ef46204dfd88086a71`, directory `Python/COARE3.5`.
MIT licence, Copyright (c) 2022 NOAA/OAR/ESRL Physical Sciences Laboratory.

These are *reference-model* numbers, not flux observations: agreement
shows the implementation reproduces COARE 3.5, not that COARE 3.5 is
right. The comparisons with measurements are the published
C_D10N / C_E10N values quoted in the test (Fairall et al. 2003;
Edson et al. 2013).

## Issue #40 (`test/era5_sea_spray_coare.jl`)

| file | what it is |
| :-- | :-- |
| `coare36_met_subset.txt` | the meteorological inputs (wind, air T, RH, pressure, shortwave, rain) of COARE 3.6's `test_36_data.txt` for exactly the 653 records of `coare36_warm_cool_subset.txt` (issue #38), which holds the ocean-side inputs and the reference outputs (u\*, τ, H, LE, cool skin, warm layer, rain heat flux) of `coare36vnWarm_et.py` for them. Together they drive the full algorithm (bulk + cool skin + warm layer) and check it against the reference code. Reference-model numbers, not flux observations. |
