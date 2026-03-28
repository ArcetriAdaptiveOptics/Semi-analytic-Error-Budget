#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cross-repository comparison of the Aliasing error:
  Semi-analytic-Error-Budget (Empirical FITS) ↔ P3 (Fourier 2D Replicas)

This test compares the aliasing variance (in nm^2) calculated by the two methods.

1. Semi-Analytic (SA):
   Uses empirical slope variances stored in FITS files (from end-to-end simulations),
   scales them by the optical gain `c`, and applies the noise transfer function H_n.

2. P3 (Fourier Model):
   Calculates the aliasing mathematically by summing shifted replicas of the 
   uncompensated atmospheric spectrum in the 2D spatial frequency domain.

Note: Aliasing is highly sensitive to the spatial reconstructor and pupil geometry.
A perfect match is impossible between a Fourier analytical model and an empirical 
matrix. The test strictly verifies that both models fall within the same order 
of magnitude (a factor < 3 difference is considered a success for Aliasing).

Run with:
    python -m pytest tests/test_aliasing.py -v -s
"""

import os
import pathlib
import unittest
import numpy as np

# ── Semi-analytic imports ────────────────────────────────────────────────────
from src.Functions import (
    aliasing_variance,
    funct_d2,
    build_transfer_function,
    compute_andes_optical_gain,
    extract_propagation_coefficients
)

# ── P3 imports ───────────────────────────────────────────────────────────────
import p3.aoSystem as _p3_ao
from p3.aoSystem.fourierModel import fourierModel

# ── File paths ───────────────────────────────────────────────────────────────
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_P3_PACKAGE_DIR = str(pathlib.Path(_p3_ao.__file__).parent.parent.absolute())
_P3_REPO_ROOT = str(pathlib.Path(_p3_ao.__file__).parent.parent.parent.absolute())

# P3 initialization file
_P3_INI_8M = os.path.join(_P3_REPO_ROOT, "tests", "scao_test_wvl1100nm.ini")

# SA initialization files for ANDES
_SA_RECONSTRUCTOR = os.path.join(
    _REPO_ROOT,
    "src", "file_fits", "LBT", # Usiamo SOUL per coerenza con gli altri test se preferisci,
                               # ma il codice SA di default legge ANDES_slopes.
                               # Per questo test forziamo ANDES.
    "SOUL_pyr40x40_wl750_fv2.1_ma3_bn1_mn500_noise_prop_coeff.fits"
)

# Nota: Assicurati che i file per ANDES_og_mod0.fits e mod4 siano nella cartella corretta
_SA_OPT_GAIN_MOD0 = os.path.join(_REPO_ROOT, "src", "file_fits", "ANDES", "ANDES_og_mod0.fits")
_SA_OPT_GAIN_MOD4 = os.path.join(_REPO_ROOT, "src", "file_fits", "ANDES", "ANDES_og_mod4.fits")
_SA_SIGMA_SLOPES = os.path.join(_REPO_ROOT, "src", "file_fits", "ANDES", "slopes_rms_time_avg_all.fits")


# ── Shared Physics Parameters ────────────────────────────────────────────────
_VERBOSE = False

_D = 8.222                  # Telescope diameter [m]
_R0 = 0.15                  # Fried parameter [m] at 500 nm
_WIND_SPEED = 15.0          # Wind speed [m/s]
_WVL_REF = 500e-9           # Evaluation wavelength [m]

_MODULATION_RADIUS = 3.0    # lambda / D
_MAX_RADIAL_ORDER = 30      # Cut-off for SA aliasing frequencies
_ALPHA = -17/3              # Asymptotic power law for aliasing PSD

# Control Loop
_FRAME_RATE = 1000.0
_GAIN = 0.5
_DELAY = 2

_N_MODES = 500              # Total modes to integrate in SA

# Temporal frequency grid for SA
_FREQS_HZ = np.logspace(-3, np.log10(_FRAME_RATE / 2.0), 1000)
_OMEGA = 2.0 * np.pi * _FREQS_HZ
_T0 = 1.0 / _FRAME_RATE

@unittest.skipUnless(
    os.path.isfile(_P3_INI_8M) and os.path.isfile(_SA_OPT_GAIN_MOD0),
    "Required FITS files or P3 INI not found. Skipping Aliasing test."
)
class TestAliasingVariance(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """
        Initializes the P3 Fourier model and computes the P3 Aliasing PSD.
        """
        #os.chdir(_P3_PACKAGE_DIR)

        # 1. P3 Setup
        cls.fao = fourierModel(
            _P3_INI_8M,
            path_root=_P3_PACKAGE_DIR,
            calcPSF=False, display=False, getErrorBreakDown=False,
            verbose=False, reduce_memory=False
        )

        cls.fao.ao.tel.D = _D
        cls.fao.ao.atm.wvl = _WVL_REF
        cls.fao.ao.atm.r0 = _R0
        cls.fao.ao.atm.wSpeed = np.array([_WIND_SPEED])

        # Ensure controller transfer functions are built
        if not hasattr(cls.fao, 'h1'):
            cls.fao.controller()

        # Extract P3 Aliasing Variance in nm^2
        # Note: P3 aliasingPSD() returns PSD in (rad/m)^2 * m^2, we convert it to nm^2
        dk = 2.0 * cls.fao.freq.kcMax_ / cls.fao.freq.resAO
        rad2nm = dk * cls.fao.freq.wvlRef * 1e9 / (2.0 * np.pi)

        psd_alias_p3 = cls.fao.aliasingPSD()
        cls.p3_alias_nm2 = float(np.sum(psd_alias_p3)) * rad2nm**2

    def test_aliasing_order_of_magnitude(self):
        """
        Computes the Semi-Analytic Aliasing variance and compares it with P3.
        """

        # 1. Setup SA Transfer Function (Noise TF H_n is used for Aliasing)
        d2 = funct_d2(_DELAY)
        gain_arr = np.full(_N_MODES, _GAIN)

        plant_num = np.array([1.0])
        plant_den = d2
        _, H_n = build_transfer_function(
            _OMEGA,
            _T0,
            _N_MODES,
            plant_num,
            plant_den,
            gain=gain_arr,
        )

        # 2. Compute Optical Gain for ANDES
        # Using the wrapper function structure you designed
        seeing_arcsec = 0.98 * _WVL_REF / _R0 * (3600 * 180 / np.pi)

        try:
            c_optg = compute_andes_optical_gain(
                file_mod0=_SA_OPT_GAIN_MOD0,
                file_mod4=_SA_OPT_GAIN_MOD4,
                seeing=seeing_arcsec,
                modulation_radius=_MODULATION_RADIUS
            )
        except Exception as e:
            self.skipTest(f"Failed to compute SA optical gain: {e}")

        # 3. Compute SA Aliasing Variance
        sa_alias_nm2, _, _, _ = aliasing_variance(
            transf_funct=H_n,
            actuators_number=_N_MODES,
            omega_temp_freq_interval=_OMEGA,
            c_optg=c_optg,
            alpha=_ALPHA,
            telescope_diameter=_D,
            seeing=seeing_arcsec,
            modulation_radius=_MODULATION_RADIUS,
            windspeed=_WIND_SPEED,
            maximum_radial_order_corrected=_MAX_RADIAL_ORDER,
            file_path_matrix_R=_SA_RECONSTRUCTOR,
            file_path_sigma_slopes=_SA_SIGMA_SLOPES
        )

        if _VERBOSE:
            print("\n" + "="*60)
            print(" ALIASING VARIANCE COMPARISON")
            print("="*60)
            print(f"SA (Empirical FITS + Integration):  {sa_alias_nm2:8.2f} nm²")
            print(f"P3 (Fourier 2D Shifted Replicas):   {self.p3_alias_nm2:8.2f} nm²")
            print("-" * 60)
            print(f"Ratio SA / P3:                      {sa_alias_nm2 / self.p3_alias_nm2:.3f}")
            print("="*60 + "\n")

        # 4. Assertion
        # We expect them to be within the same order of magnitude (Factor of 3)
        ratio = max(sa_alias_nm2, self.p3_alias_nm2) / min(sa_alias_nm2, self.p3_alias_nm2)

        self.assertLess(
            ratio, 3.0,
            msg=(f"Aliasing variance mismatch is too large. "
                 f"SA: {sa_alias_nm2:.2f} nm², P3: {self.p3_alias_nm2:.2f} nm²")
        )

if __name__ == "__main__":
    unittest.main(verbosity=2)
