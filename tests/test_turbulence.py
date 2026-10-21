#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cross-repository comparison of atmospheric turbulence (fitting error / uncompensated modes):
  Semi-analytic-Error-Budget (Zernike) ↔ P3 (Fourier Spatial Filters)

This test compares the variance of low-order atmospheric aberrations (Tip/Tilt and Focus)
calculated through two completely different mathematical approaches:

1. Semi-Analytic (SA) using `arte`:
   Calculates the exact temporal Power Spectral Density (PSD) for specific Zernike modes
   (mode 2=Tip, 3=Tilt, 4=Focus) and integrates them over temporal frequencies [Hz].

2. P3 (Fourier Model):
   Calculates the 2D spatial von Karman PSD (Wphi). It isolates the variance of specific
   modes by applying exact spatial Zernike variance filters (derived from Sasiela 1993).

Run with:
    python -m pytest tests/test_turbulence.py -v -s
"""

import os
import pathlib
import unittest
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

# ── Semi-analytic imports ────────────────────────────────────────────────────
from src.Functions import turbulence_psd

# ── P3 imports ───────────────────────────────────────────────────────────────
import p3.aoSystem as _p3_ao
from p3.aoSystem.fourierModel import fourierModel

# ── File paths ───────────────────────────────────────────────────────────────
_P3_PACKAGE_DIR = str(pathlib.Path(_p3_ao.__file__).parent.parent.absolute())
_P3_REPO_ROOT = str(pathlib.Path(_p3_ao.__file__).parent.parent.parent.absolute())
_P3_INI_8M = os.path.join(_P3_REPO_ROOT, "tests", "scao_test_wvl1100nm.ini")

# ── Shared Physics Parameters ────────────────────────────────────────────────
_VERBOSE = True
_DISPLAY = False

_D = 8.222                  # Telescope diameter [m]
_R0 = 0.15                  # Fried parameter [m] at 500 nm
_L0 = 25.0                  # Outer scale [m]
_WIND_SPEED = 15.0          # Wind speed [m/s]
_WIND_DIR = 0.0             # Wind direction [deg]
_LAYERS_ALT = 0.0           # Single ground layer for simplicity
_WVL_REF = 500e-9           # Evaluation wavelength [m]

# Temporal frequency grid for SA
_FRAME_RATE = 1000.0
_N_POINTS = 1000
_FREQS_HZ = np.logspace(-3, np.log10(_FRAME_RATE / 2.0), _N_POINTS)
_FREQS_1oM = np.logspace(-4, 4, 300)

class TestTurbulenceVariance(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """
        Initializes both the P3 Fourier model and the SA Zernike PSDs.
        """
        # =====================================================================
        # 1. P3 (Fourier Model) Setup
        # =====================================================================
        os.chdir(_P3_PACKAGE_DIR)

        cls.fao = fourierModel(
            _P3_INI_8M,
            path_root=_P3_PACKAGE_DIR,
            calcPSF=False, display=False, getErrorBreakDown=False,
            verbose=False, reduce_memory=False
        )

        cls.fao.ao.tel.D = _D
        cls.fao.ao.atm.wvl = _WVL_REF
        cls.fao.ao.atm.r0 = _R0
        cls.fao.ao.atm.L0 = _L0
        cls.fao.ao.atm.wSpeed = np.array([_WIND_SPEED])

        # Extract full spatial grid frequency elements directly
        cls.k_full = np.sqrt(cls.fao.freq.k2_)
        cls.pf_full = cls.fao.freq.pistonFilter_
        cls.dk = cls.fao.freq.PSDstep

        # Generate the continuous von Karman spectrum on the full grid
        cls.Wphi_full = cls.fao.ao.atm.spectrum(cls.k_full)

        # =====================================================================
        # 2. SA (arte) Setup
        # =====================================================================
        cls.psd_atmo_sa = turbulence_psd(
            rho=0.0, theta=0.0,
            aperture_radius=_D / 2.0, aperture_center=[0.0, 0.0, 0.0],
            r0=_R0, L0=_L0, layers_altitude=_LAYERS_ALT,
            wind_speed=_WIND_SPEED, wind_direction=_WIND_DIR,
            space_freqs=_FREQS_1oM, tempor_freqs=_FREQS_HZ, n_modes=3
        )

        # turbulence_psd returns a PSD per rad/s: integrate over omega = 2*pi*f.
        omega = 2.0 * np.pi * _FREQS_HZ
        cls.var_tip_sa_nm2   = np.real(integrate.simpson(cls.psd_atmo_sa[0, :], omega))
        cls.var_tilt_sa_nm2  = np.real(integrate.simpson(cls.psd_atmo_sa[1, :], omega))
        cls.var_focus_sa_nm2 = np.real(integrate.simpson(cls.psd_atmo_sa[2, :], omega))

    def test_tip_tilt_variance_comparison(self):
        """
        Compares the total Tip+Tilt variance.
        """
        # ── SA: Tip + Tilt Variance ──────────────────────────────────────────
        sa_tt_rad2 = self.var_tip_sa_nm2 + self.var_tilt_sa_nm2
        sa_tt_nm2 = sa_tt_rad2

        # ── P3: Tip + Tilt Variance ──────────────────────────────────────────
        # pf_full is the Piston filter. TiltFilter() isolates high frequencies.
        psd_tt_p3 = self.Wphi_full * (self.pf_full - self.fao.TiltFilter())

        p3_tt_rad2 = float(np.sum(psd_tt_p3) * self.dk**2)
        rad2nm = 500 / (2.0 * np.pi)
        p3_tt_nm2 = p3_tt_rad2 * rad2nm**2

        if _VERBOSE:
            print("\n" + "="*60)
            print(" ATMOSPHERIC TURBULENCE: TIP + TILT VARIANCE")
            print("="*60)
            print(f"SA (arte Zernike 2+3 temporal integral):  {sa_tt_nm2:8.1f} nm²")
            print(f"P3 (Fourier Spatial 2D integral):         {p3_tt_nm2:8.1f} nm²")
            print("-" * 60)
            print(f"Ratio SA / P3:                            {sa_tt_nm2 / p3_tt_nm2:.3f}")
            print("="*60 + "\n")

        np.testing.assert_allclose(
            sa_tt_nm2, p3_tt_nm2, rtol=0.15,
            err_msg="The Tip/Tilt variance differs too much between SA and P3."
        )

    def test_focus_variance_comparison(self):
        """
        Compares the Focus variance.
        """
        # ── SA: Focus Variance ───────────────────────────────────────────────
        sa_focus_nm2 = self.var_focus_sa_nm2

        # ── P3: Focus Variance ───────────────────────────────────────────────
        # FocusFilter() acts as a high-pass filter for focus.
        psd_focus_p3 = self.Wphi_full * (1.0 - self.fao.FocusFilter())

        p3_focus_rad2 = float(np.sum(psd_focus_p3) * self.dk**2)
        rad2nm = 500 / (2.0 * np.pi)
        p3_focus_nm2 = p3_focus_rad2 * rad2nm**2

        if _VERBOSE:
            print("\n" + "="*60)
            print(" ATMOSPHERIC TURBULENCE: FOCUS VARIANCE")
            print("="*60)
            print(f"SA (arte Zernike 4 temporal integral):    {sa_focus_nm2:8.1f} nm²")
            print(f"P3 (Fourier Spatial 2D integral):         {p3_focus_nm2:8.1f} nm²")
            print("-" * 60)
            print(f"Ratio SA / P3:                            {sa_focus_nm2 / p3_focus_nm2:.3f}")
            print("="*60 + "\n")

        np.testing.assert_allclose(
            sa_focus_nm2, p3_focus_nm2, rtol=0.15,
            err_msg="The Focus variance differs too much between SA and P3."
        )

    def test_plot_sa_temporal_psds(self):
        """
        Optional plot to visualize the temporal PSDs of Tip, Tilt, and Focus
        generated by the semi-analytic code.
        """
        if not _DISPLAY:
            self.skipTest("Plotting is disabled. Set _DISPLAY = True to see the plots.")

        plt.figure(figsize=(10, 6))

        # Mode 2: Tip, Mode 3: Tilt, Mode 4: Focus
        plt.loglog(_FREQS_HZ, np.real(self.psd_atmo_sa[0, :]), label="Mode 2 (Tip)", linewidth=2)
        plt.loglog(_FREQS_HZ, np.real(self.psd_atmo_sa[1, :]), '--', label="Mode 3 (Tilt)", linewidth=2)
        plt.loglog(_FREQS_HZ, np.real(self.psd_atmo_sa[2, :]), label="Mode 4 (Focus)", linewidth=2)

        # Add the theoretical f^(-17/3) slope for visual reference
        f_ref = _FREQS_HZ[_FREQS_HZ > 1.0]
        psd_ref = np.real(self.psd_atmo_sa[0, _FREQS_HZ > 1.0])
        if len(f_ref) > 0:
            asymptote = psd_ref[0] * (f_ref / f_ref[0])**(-17/3)
            plt.loglog(f_ref, asymptote, 'k:', label=r"$f^{-17/3}$ asymptote")

        plt.title(f"Temporal Atmospheric PSDs (Von Karman, v={_WIND_SPEED}m/s, $L_0$={_L0}m)")
        plt.xlabel("Temporal Frequency [Hz]")
        plt.ylabel("Phase PSD [$rad^2 / Hz$]")
        plt.grid(True, which="both", ls="-", alpha=0.3)
        plt.legend(loc="lower left")

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    unittest.main(verbosity=2)
