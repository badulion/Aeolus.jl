# Published within-canopy profiles used by test/era5_canopy_profiles.jl
# and examples/validation/canopy_profiles.jl (issue #25).
#
# None of the source papers tabulates its profiles; every number below was
# read from a published figure. How:
#
#   * "vector": the figure is vector graphics in the publisher's PDF. The
#     page was converted to SVG (poppler `pdftocairo -svg`) and the marker
#     centres (or polyline vertices) were read from the path coordinates;
#     the axes were calibrated from the tick marks. Reading error is far
#     below the marker size, ≲ 0.005 of the axis range. Each dataset lists
#     a CHECK: a value that the figure's own normalisation fixes (e.g.
#     U/U_h = 1 at z = h), recovered by the calibration.
#   * "raster": the figure is a bitmap. The centre of each marker was
#     located from its pixels (longest run of marker-coloured pixels along
#     the row at the marker's nominal height); error ≈ ±1 px ≈ ±0.02 in a·h.
#
# All heights are normalised by the canopy height h unless stated.

"""
    canopy_profile_observations()

Named tuple of the published canopy datasets (see the header of
`test/data/canopy_profiles.jl` for the digitisation method).
"""
function canopy_profile_observations()

    # ---------------------------------------------------------------------
    # 1. Wind tunnel, waving-wheat model canopy (Brunet, Finnigan & Raupach
    #    1994, BLM 70, 95-132; ensemble of runs). Profiles as re-plotted
    #    (circles "WT") in Huang, Cassiani & Albertson (2008), "The effects
    #    of vegetation density on turbulent structures within canopy
    #    sublayer", AMS symposium extended abstract, paper 2.2, Fig. 2a
    #    (<u>/<u>_h) and Fig. 2b (-<u'w'>/u*^2, u* at z = h),
    #    https://ams.confex.com/ams/pdfpapers/138880.pdf — vector.
    #    CHECK: U/U_h at z/h = 1.000 reads 0.999; -uw/u*^2 at z = h reads 1.000.
    #
    #    Canopy: nylon monofilament "wheat", h = 47 mm, frontal area per
    #    unit volume a = 10.6 m⁻¹ (a h = 0.50) — Patton (1997, PhD thesis,
    #    UC Davis, Table 4.1). Finnigan, Shaw & Patton (2009, JFM 637,
    #    387-424, §3.2 and Table 1) model the same canopy with h C_d a =
    #    0.236 and C_d = 0.4725, "70 % of the C_d quoted by Brunet et al.
    #    (1994) for wind-tunnel runs at a free-stream velocity of 10.2 m/s";
    #    hence a h = 0.4995 (agrees with Patton) and the tunnel's own
    #    C_d = 0.675. Patton (1997, Table 4.1) quotes C_d = 1.2 for the
    #    same canopy in the windbreak runs of Judd et al. (1996).
    # ---------------------------------------------------------------------
    wind_tunnel = (
        name = "Brunet et al. (1994) wind-tunnel waving wheat",
        h = 0.047,
        ah = 0.4995,                  # a·h (FSP 2009: 0.236 / 0.4725)
        C_d = 0.675,                  # FSP 2009: 0.4725 / 0.7
        C_d_alt = 1.2,                # Patton (1997) Table 4.1
        z = [0.167, 0.334, 0.500, 0.667, 0.834, 1.000, 1.200, 1.500, 2.000],
        U_over_Uh = [0.226, 0.311, 0.403, 0.520, 0.731, 1.000, 1.238, 1.511, 1.849],
        stress = [-0.017, 0.051, 0.163, 0.387, 0.709, 1.000, 1.024, 1.034, 1.035],
    )

    # ---------------------------------------------------------------------
    # 2. CHATS walnut orchard, Dixon CA, 2007 (Patton et al. 2011, BAMS 92,
    #    593-611). h = 10 m. Sonic anemometers at 1.5, 3, 4.5, 6, 7.5, 9,
    #    10, 11, 12.5, 14, 18, 23, 29 m (Dupont & Patton 2012a, AFM 157,
    #    11-29, Fig. 2).
    #
    #    Mean wind, near-neutral regime "NN": Dupont & Patton (2012a) Fig. 8
    #    (no leaves) and Fig. 9 (with leaves), panel <u>/u_h — vector (black
    #    squares). CHECK: <u>/u_h at z = 10 m reads 1.000 in both figures.
    #
    #    Plant-area density: Dupont & Patton (2012a) Fig. 1b (adapted from
    #    Patton et al. 2011), normalised a·h at z/h = 0.05, 0.15, …, 0.95,
    #    plus the plotted end point a = 0 at z/h = 1 — raster. The
    #    no-leaves square at z/h = 0.05 is hidden under the with-leaves
    #    circle (a·h = 0.28); that value is used for both. The digitised
    #    profiles integrate (midpoint rule) to 2.78 and 0.77, against the
    #    PAI "about 2.5" and "about 0.7" quoted in the text (Patton et al.
    #    2011, p. 597); the profiles are used as plotted.
    #
    #    Displacement height from the within-canopy momentum-flux profiles
    #    in near-neutral conditions: d = 7.8 m with leaves, 5.8 m without
    #    (Dupont & Patton 2022, BLM 184, 195-230, Table 1).
    #    Shear length L_s/h at the inflection point, NN: 0.32 with leaves,
    #    0.72 without (Dupont & Patton 2012a, Table 1).
    # ---------------------------------------------------------------------
    chats_z = [1.5, 3.0, 4.5, 6.0, 7.5, 9.0, 10.0, 11.0, 12.5, 14.0, 18.0, 23.0, 29.0] ./ 10
    pad_z = [0.0, 0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 1.0]
    chats_leaves = (
        name = "CHATS walnut orchard, with leaves",
        h = 10.0,
        pad_z = pad_z,
        pad_ah = [0.28, 0.28, 0.51, 2.44, 1.67, 3.19, 3.13, 3.23, 4.29, 5.46, 3.58, 0.0],
        z = chats_z,
        U_over_Uh = [0.271, 0.136, 0.180, 0.312, 0.461, 0.637, 1.000,
                     1.265, 1.495, 1.774, 2.098, 2.498, 2.834],
        d = 7.8,
        Ls_over_h = 0.32,
    )
    chats_noleaves = (
        name = "CHATS walnut orchard, no leaves",
        h = 10.0,
        pad_z = pad_z,
        pad_ah = [0.28, 0.28, 0.31, 0.35, 1.05, 1.51, 0.54, 0.72, 1.38, 0.55, 0.97, 0.0],
        z = chats_z,
        U_over_Uh = [0.509, 0.494, 0.526, 0.627, 0.787, 0.898, 1.000,
                     1.093, 1.215, 1.316, 1.579, 1.771, 2.021],
        d = 5.8,
        Ls_over_h = 0.72,
    )

    # ---------------------------------------------------------------------
    # 3. Scots pine forest, SMEAR II, Hyytiälä, Finland, summer 2005
    #    (Launiainen et al. 2007, Tellus 59B, 919-936). h = 15 m, one-sided
    #    LAI 3.5 and C_d = 0.15 (their Table 2, the values with which they
    #    fit the Massman & Weil (1999) model to the near-neutral data),
    #    d = 0.78 h (Rannik et al. 2003, quoted in §2.1).
    #
    #    LAD: Fig. 1, all-sided needle-area density [m² m⁻³] vs z/h — vector
    #    (14-vertex polyline; 0 below z/h = 0.40). CHECK: the polyline
    #    integrates to an all-sided LAI of 7.02 (text: 7), i.e. one-sided
    #    3.51 (Table 2: 3.5). Stored here one-sided (÷ 2).
    #
    #    Mean wind: Fig. 3a, median U/u* (u* above the canopy) per stability
    #    class, sonic heights z/h = 0.02, 0.06, 0.13, 0.27, 0.40, 0.78, 1.47
    #    (Table 1) — vector. The paper shifts the markers of each class
    #    vertically "for clarity"; the Table 1 heights are used.
    #
    #    Temperature: Fig. 3b, median Θ(z) − Θ(ref) [K] — vector. The panel
    #    has no y-axis title; its markers sit at z/h = 0.28, 0.56, 1.12,
    #    2.24, 3.36, 4.48, i.e. 4.2, 8.4, 16.8, 33.6, 50.4, 67.2 m, the
    #    SMEAR II mast temperature levels, and all four classes pass through
    #    0 at 16.8 m: the reference "h_c" of the axis label is the 16.8 m
    #    level. CHECK: the four classes read 0.000 at 16.8 m.
    # ---------------------------------------------------------------------
    lad_z = [0.0, 0.4002, 0.4067, 0.4135, 0.4203, 0.4268, 0.4336, 0.4401,
             0.4800, 0.5534, 0.7001, 0.8536, 0.9270, 1.0]
    lad_all = [0.0, 0.0, 0.0046, 0.0198, 0.0447, 0.0788, 0.1208, 0.1696,
               0.5789, 1.3990, 1.4967, 0.3968, 0.0788, 0.0]
    hyytiala = (
        name = "Hyytiälä Scots pine (SMEAR II)",
        h = 15.0,
        LAI = 3.5,
        C_d = 0.15,
        d_over_h = 0.78,
        lad_z = lad_z,                 # z/h
        lad = lad_all ./ 2,            # one-sided, m² m⁻³
        z = [0.02, 0.06, 0.13, 0.27, 0.40, 0.78, 1.47],
        U_over_ustar = (
            US = [0.480, 0.870, 1.125, 1.110, 1.035, 1.155, 4.260],
            NN = [0.484, 0.891, 1.198, 1.208, 1.162, 1.469, 4.964],
        ),
        T_z_m = [4.2, 8.4, 16.8, 33.6, 50.4, 67.2],
        T_minus_ref = (
            US = [0.292, 0.301, 0.0, -0.233, -0.338, -0.426],
            NN = [-0.052, -0.022, 0.0, -0.025, 0.030, -0.003],
            TS = [-0.563, -0.383, 0.0, 0.350, 0.681, 0.921],
            S  = [-3.156, -2.113, 0.0, 1.094, 1.882, 2.237],
        ),
    )

    return (; wind_tunnel, chats_leaves, chats_noleaves, hyytiala)
end
