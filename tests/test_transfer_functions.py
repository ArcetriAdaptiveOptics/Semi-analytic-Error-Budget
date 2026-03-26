#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test suite for the Adaptive Optics Closed-Loop Transfer Functions (CLTF).

This test verifies that the analytical transfer functions implemented using 
NumPy polynomials in the Semi-Analytic (SA) code perfectly match the discrete-time 
transfer functions implemented in P3 (fourierModel.py).

The two main transfer functions are:
  - Rejection Transfer Function (H_r): Filters atmospheric turbulence.
  - Noise Transfer Function (H_n): Filters WFS measurement noise.

To see the plots, change `_DISPLAY = False` to `_DISPLAY = True` below, 
or run the tests interactively.

Run from the root directory with:
    python -m pytest tests/test_transfer_functions.py -v -s
"""

import unittest
import numpy as np
import matplotlib.pyplot as plt

# ── Semi-analytic imports ────────────────────────────────────────────────────
from src.Functions import build_transfer_function, funct_d2

# ── Shared test parameters ───────────────────────────────────────────────────
_DISPLAY    = False    # Set to True to plot the Transfer Functions
_GAIN       = 0.5      # Integrator loop gain
_DELAY      = 2        # Loop delay in frames
_FRAME_RATE = 1000.0   # Loop frequency [Hz]
_N_POINTS   = 1000     # Number of frequency points for the grid

class TestTransferFunctions(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """
        Initializes the frequency grid and computes the transfer functions
        for both SA and P3 models.
        """
        # 1. Setup the temporal frequency grid
        # We start from 0.1 Hz up to the Nyquist frequency (fs / 2)
        cls.freqs_hz = np.logspace(-1, np.log10(_FRAME_RATE / 2.0), _N_POINTS)
        cls.omega = 2.0 * np.pi * cls.freqs_hz
        cls.t0 = 1.0 / _FRAME_RATE

        # =====================================================================
        # SEMI-ANALYTIC (SA) COMPUTATION
        # =====================================================================
        d2 = funct_d2(_DELAY)
        gain_arr = np.array([_GAIN]) # Single mode test
        n_modes = 1

        plant_num = np.array([1.0])
        plant_den = d2

        H_r_sa, H_n_sa = build_transfer_function(
            cls.omega,
            cls.t0,
            n_modes,
            plant_num,
            plant_den,
            gain=gain_arr,
        )

        cls.H_r_sa = H_r_sa[0, :]
        cls.H_n_sa = H_n_sa[0, :]

        # =====================================================================
        # P3 COMPUTATION (from fourierModel.py -> controller)
        # =====================================================================
        Ts = cls.t0
        # P3 defines z on the unit circle
        z = np.exp(-2j * np.pi * cls.freqs_hz * Ts)
        eps = np.finfo(float).eps

        # Integrator: hInt = gain / (1 - z^-1)
        denom = 1.0 - z**(-1.0)
        denom = np.where(np.abs(denom) < eps, eps, denom)
        hInt = _GAIN / denom

        # Rejection: rtfInt = 1 / (1 + hInt * z^-delay)
        denom2 = 1.0 + hInt * z**(-_DELAY)
        denom2 = np.where(np.abs(denom2) < eps, eps, denom2)
        cls.H_r_p3 = 1.0 / denom2

        # Aliasing / Noise
        cls.H_alias_p3 = hInt * z**(-_DELAY) * cls.H_r_p3
        cls.H_n_p3 = cls.H_alias_p3 / z

    def test_rejection_transfer_function_squared_magnitude(self):
        """
        Verifies that the squared magnitude of the Rejection Transfer Function 
        |H_r|^2 matches between SA and P3.
        """
        mag_sq_sa = np.abs(self.H_r_sa)**2
        mag_sq_p3 = np.abs(self.H_r_p3)**2

        np.testing.assert_allclose(
            mag_sq_sa, mag_sq_p3, rtol=1e-10,
            err_msg="The Rejection Transfer Function |H_r|^2 differs between SA and P3."
        )

    def test_noise_transfer_function_squared_magnitude(self):
        """
        Verifies that the squared magnitude of the Noise Transfer Function 
        |H_n|^2 matches between SA and P3.
        
        Note: P3 defines the NTF with an extra 1/z phase delay compared to SA 
        (which mathematically corresponds to the 'Aliasing TF' in P3).
        However, since |1/z| = 1 on the unit circle, the squared magnitudes 
        (which are integrated to compute the variance) must be identical.
        """
        mag_sq_sa = np.abs(self.H_n_sa)**2
        mag_sq_p3 = np.abs(self.H_n_p3)**2

        np.testing.assert_allclose(
            mag_sq_sa, mag_sq_p3, rtol=1e-10,
            err_msg="The Noise Transfer Function |H_n|^2 differs between SA and P3."
        )

    def test_plot_transfer_functions(self):
        """
        Plots the squared magnitudes of the transfer functions if _DISPLAY is True.
        """
        if not _DISPLAY:
            self.skipTest("Plotting is disabled. Set _DISPLAY = True to see the plots.")

        plt.figure(figsize=(10, 6))

        # Plot Rejection TF
        plt.loglog(self.freqs_hz, np.abs(self.H_r_sa)**2, 
                   label="SA: $|H_r|^2$ (Rejection)", color='tab:blue', linewidth=4, alpha=0.5)
        plt.loglog(self.freqs_hz, np.abs(self.H_r_p3)**2, 
                   '--', label="P3: $|H_r|^2$", color='navy')

        # Plot Noise TF
        plt.loglog(self.freqs_hz, np.abs(self.H_n_sa)**2, 
                   label="SA: $|H_n|^2$ (Noise)", color='tab:orange', linewidth=4, alpha=0.5)
        plt.loglog(self.freqs_hz, np.abs(self.H_n_p3)**2, 
                   '--', label="P3: $|H_n|^2$", color='darkred')

        plt.title(f"Transfer Functions Squared Magnitude (Gain={_GAIN},"
                  f" Delay={_DELAY}, Fs={_FRAME_RATE}Hz)")
        plt.xlabel("Temporal Frequency [Hz]")
        plt.ylabel("Squared Magnitude $|H|^2$")
        plt.grid(True, which="both", ls="-", alpha=0.3)
        plt.legend(loc="best")

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    unittest.main(verbosity=2)
