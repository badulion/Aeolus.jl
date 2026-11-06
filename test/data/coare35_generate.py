"""Generate COARE 3.5 reference-model output for test/air_sea_flux.jl.

Runs the NOAA-PSL COARE 3.5 Python reference (github.com/NOAA-PSL/
COARE-algorithm, commit bd1cac80d2dae454e699f9ef46204dfd88086a71,
Python/COARE3.5/coare35vn.py, MIT licence) on a synthetic grid of bulk
inputs. The INPUTS are synthetic (a regular grid, not observations); the
OUTPUTS are the reference model's. jcool = 0, i.e. `ts` is the interface
temperature, so no cool-skin correction enters.

Usage (from Python/COARE3.5 of that repository, with numpy):
    python3 coare35_generate.py > coare35_synthetic.txt
"""
import numpy as np
from coare35vn import coare35vn

g = 9.8062  # COARE grv(45 deg), used only to build deep-water c_p below
rows = []
for regime, ts, t, rh in (("unstable", 28.0, 26.0, 80.0), ("stable", 15.0, 18.0, 85.0)):
    for U in (2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 15.0, 18.0, 22.0, 25.0):
        for wave in ("none", "age", "state"):
            # fully developed sea (Taylor & Yelland 2001): Tp = 0.729 U, Hs = 0.0248 U^2
            Tp, Hs = 0.729 * U, 0.0248 * U**2
            cp = g * Tp / (2 * np.pi)
            rows.append((regime, wave, U, t, rh, ts, cp, Hs))

print("# regime wave u t rh ts cp sigH | usr tau hsb hlb tsr qsr zot zoq Cd Ch Ce L zet Cdn_10 Chn_10 Cen_10")
for regime, wave, U, t, rh, ts, cp, Hs in rows:
    kw = dict(P=1010.0, Rs=0.0, Rl=400.0, zu=10.0, zt=10.0, zq=10.0, lat=45.0,
              zi=600.0, rain=None, jcool=0)
    if wave == "age":
        kw.update(cp=np.array([cp]))
    elif wave == "state":
        kw.update(cp=np.array([cp]), sigH=np.array([Hs]))
    A = coare35vn(np.array([U]), np.array([t]), np.array([rh]), np.array([ts]), **kw)[0]
    usr, tau, hsb, hlb, hlwebb, tsr, qsr, zot, zoq, Cd, Ch, Ce, L, zet = A[:14]
    Cdn, Chn, Cen = A[18:21]
    print(regime, wave, " ".join("%.10g" % v for v in
          (U, t, rh, ts, cp, Hs, usr, tau, hsb, hlb, tsr, qsr, zot, zoq, Cd, Ch, Ce, L, zet, Cdn, Chn, Cen)))
