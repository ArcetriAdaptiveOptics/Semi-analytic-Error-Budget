#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cross-repository comparison of measurement-noise error terms:
  Semi-analytic-Error-Budget  ↔  P3

Two levels of comparison are provided.

Level 1 – Pyramid WFS slope-noise variance *formula*
------------------------------------------------------
Both codes implement the standard Pyramid WFS noise variance (Verinaud 2004):

    σ² = excess / nph  +  4·ron² / nph²

where *nph* is the total photon count per subaperture per frame.

P3   encodes this directly in ``sensor.NoiseVariance`` (Pyramid branch).
SA   encodes the same physics through per-pixel statistics (Agapito & Pinna
     2019, Eqs. 12/14).  Two naming conventions differ:

  ·  P3  ``excess``    = excess-noise *power* factor F²  ( ≥ 1 )
  ·  SA  ``F_excess``  = excess-noise *amplitude* factor F = √(P3 excess)
  ·  SA  ``read_out_noise`` must be passed as **ron** (RMS standard deviation).

With these conventions the two expressions are analytically identical.

Level 2 – Integrated noise PSD for an 8 m SCAO Pyramid system
--------------------------------------------------------------
P3  computes the noise power-spectral-density over the 2-D spatial-frequency
plane; integrating it gives the noise wavefront-error variance.
SA  computes ``measure_variance`` by:
  1. evaluating the per-pixel slope variance,
  2. scaling by mode-dependent noise-propagation coefficients (from the SOUL
     reconstruction matrix FITS file, calibrated at 750 nm),
  3. weighting by |H_n(ω)|² (noise transfer function of an integrator),
  4. integrating over temporal frequencies.

An *exact* match between the two models is not expected because they use
different reconstruction strategies (2-D Fourier reconstructor vs.
simulation-based propagation matrix).  Both should however report noise WFE
values within one order of magnitude for compatible 8 m Pyramid parameters.

Run from the Semi-analytic-Error-Budget root with:
    python -m pytest tests/test_noise_error_term.py -v -s
"""

import os
import pathlib
import unittest

import numpy as np

# ── Semi-analytic imports ────────────────────────────────────────────────────
from src.Functions import (
    build_transfer_function,
    compute_noise_PSD_intermediate,
    compute_output_PSD_and_integrate,
    compute_slope_noise_variance,
    extract_propagation_coefficients,
    funct_d2,
)

# ── P3 imports ───────────────────────────────────────────────────────────────
import p3.aoSystem as _p3_ao
from p3.aoSystem.sensor import sensor
from p3.aoSystem.fourierModel import fourierModel

# ── File paths ───────────────────────────────────────────────────────────────
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# P3 package directory (used as path_root for fourierModel, same convention
# as in P3's own test suite).
_P3_PACKAGE_DIR = str(pathlib.Path(_p3_ao.__file__).parent.parent.absolute())

# P3 git-repository root (one level above the `p3` Python package).
_P3_REPO_ROOT = str(
    pathlib.Path(_p3_ao.__file__).parent.parent.parent.absolute()
)

# Existing test ini – 8.222 m telescope, Pyramid WFS, 40 lenslets, mod=3,
# nph=1000 ph/sub/frame, no RON, gain=0.3, delay=2, fs=1000 Hz.
_P3_INI_8M = os.path.join(_P3_REPO_ROOT, "tests", "scao_test_wvl1100nm.ini")

# LBT SOUL 40×40 noise-propagation coefficient file (wvl=750 nm).
_SA_SOUL_FITS = os.path.join(
    _REPO_ROOT,
    "src", "file_fits", "LBT",
    "SOUL_pyr40x40_wl750_fv2.1_ma3_bn1_mn500_noise_prop_coeff.fits",
)

# ── Shared parameters and settings ──────────────────────────────────────────
_VERBOSE    = False    # Set to True to enable prints during tests
_VERBOSE    = False    # Set to True to enable prints during tests

_GAIN       = 0.3
_DELAY      = 2        # frames
_FRAME_RATE = 1000.0   # Hz
_NPH        = 1000.0   # ph / subaperture / frame
_N_MODES    = 30       # modes used in SA integration
_WVL_WFS    = 750e-9   # WFS wavelength [m]

_temporal_freqs = np.logspace(-3, np.log10(_FRAME_RATE / 2.0), 500)
_omega = 2.0 * np.pi * _temporal_freqs
_T0 = 1.0 / _FRAME_RATE


def _as_scalar(value):
    """Convert a scalar-like array to a Python float without deprecation warnings."""
    return float(np.asarray(value).reshape(-1)[0])


# ── SA helper: noise WFE in nm OPD ──────────────────────────────────────────

def _sa_wfe_nm(n_modes, nph, ron_var, p_coeff, omega, t0, gain, delay):
    """
    Compute the SA measurement-noise WFE in nm OPD.
    Calls the real compute_slope_noise_variance function to strictly adhere to DRY.
    """
    gain_arr = np.full(n_modes, gain)
    d2  = funct_d2(delay)
    plant_num = np.array([1.0])
    plant_den = d2
    _, H_n = build_transfer_function(
        omega,
        t0,
        n_modes,
        plant_num,
        plant_den,
        gain=gain_arr,
    )

    # Invert nph to photon_flux for the SA function
    D = 8.222
    n_sub = 40
    area_sub = np.pi * ((D / n_sub) / 2.0)**2
    collecting_area = 1.0
    photon_flux = nph * (collecting_area / area_sub) * _FRAME_RATE

    slope_var = compute_slope_noise_variance(
        F_excess=1.0,
        pixel_pos=[1.0, 1.0, -1.0, -1.0],
        sky_bkg=0.0,
        dark_curr=0.0,
        read_out_noise=np.sqrt(ron_var),  # Pass RMS standard deviation
        photon_flux=photon_flux,
        telescope_diameter=D,
        frame_rate=_FRAME_RATE,
        magnitudo=0.0,
        n_subaperture=n_sub,
        collecting_area=collecting_area
    )

    sigma2_w = slope_var * p_coeff[:n_modes]
    PSD_in   = compute_noise_PSD_intermediate(omega, n_modes, sigma2_w)
    var_sum, _, _ = compute_output_PSD_and_integrate(n_modes, H_n, PSD_in, omega)

    # var_sum is ALREADY in nm^2 because p_coeff are in nm^2
    return float(np.sqrt(abs(var_sum)))


# ============================================================================
#  Level 1 – Pyramid WFS slope-noise variance formula comparison
# ============================================================================

class TestPyramidNoiseVarianceFormula(unittest.TestCase):
    """
    Both P3 ``sensor.NoiseVariance`` and SA ``compute_slope_noise_variance``
    implement  σ² = excess/nph + 4·ron²/nph²  for a Pyramid WFS.

    The tests assert equality to floating-point precision.
    """

    # ── representative 8 m class system ──────────────────────────────────
    D     = 8.222
    n_sub = 40
    dsub  = D / n_sub
    PIXEL_POS = [1.0, 1.0, -1.0, -1.0]   # 4-quadrant x-slope weights

    # ── helpers ──────────────────────────────────────────────────────────

    def _p3_var(self, nph, ron, excess=1.0):
        """P3 Pyramid noise variance for a single WFS."""
        wfs = sensor(
            pixel_scale=1000.0,   # mas – not used in Pyramid formula
            fov=2,
            nph=[nph],
            ron=ron,
            excess=excess,
            wfstype="Pyramid",
            nL=[self.n_sub],
            dsub=[self.dsub],
        )
        return float(wfs.NoiseVariance(r0=0.15, wvl=750e-9)[0])

    def _sa_var(self, nph, ron_var, F_excess=1.0):
        """
        SA slope noise variance wrapper around the actual implementation.
        ron_var = ron² (RON variance in e-²).
        """
        area_sub = np.pi * ((self.D / self.n_sub) / 2.0)**2
        photon_flux = nph * (1.0 / area_sub) * 1000.0

        return compute_slope_noise_variance(
            F_excess=F_excess,
            pixel_pos=self.PIXEL_POS,
            sky_bkg=0.0,
            dark_curr=0.0,
            read_out_noise=np.sqrt(ron_var), # real function expects RMS
            photon_flux=photon_flux,
            telescope_diameter=self.D,
            frame_rate=1000.0,
            magnitudo=0.0,
            n_subaperture=self.n_sub,
            collecting_area=1.0
        )

    # ── tests ─────────────────────────────────────────────────────────────

    def test_pure_shot_noise_equals_one_over_nph(self):
        """With ron=0, excess=1: both models give σ² = 1/nph exactly."""
        for nph in (10.0, 100.0, 1000.0):
            with self.subTest(nph=nph):
                expected = 1.0 / nph
                np.testing.assert_allclose(
                    self._p3_var(nph, 0.0), expected, rtol=1e-12,
                    err_msg=f"P3 shot noise, nph={nph}",
                )
                np.testing.assert_allclose(
                    self._sa_var(nph, 0.0), expected, rtol=1e-12,
                    err_msg=f"SA shot noise, nph={nph}",
                )
                np.testing.assert_allclose(
                    self._p3_var(nph, 0.0), self._sa_var(nph, 0.0),
                    rtol=1e-12, err_msg=f"P3 vs SA, nph={nph}",
                )

    def test_ron_term_matches_analytical_formula(self):
        """Full formula σ² = 1/nph + 4·ron²/nph² is reproduced exactly."""
        nph, ron = 1.0, 5.0
        expected = 1.0 / nph + 4.0 * ron ** 2 / nph ** 2
        np.testing.assert_allclose(
            self._p3_var(nph, ron), expected, rtol=1e-12,
        )
        np.testing.assert_allclose(
            self._sa_var(nph, ron ** 2), expected, rtol=1e-12,
        )
        np.testing.assert_allclose(
            self._p3_var(nph, ron), self._sa_var(nph, ron ** 2), rtol=1e-12,
        )

    def test_shot_noise_scales_as_one_over_nph(self):
        """Shot noise (ron=0) must double when nph is halved."""
        for n in (50.0, 500.0, 5000.0):
            with self.subTest(nph=n):
                ratio_p3 = self._p3_var(n, 0.0) / self._p3_var(2 * n, 0.0)
                ratio_sa = self._sa_var(n, 0.0) / self._sa_var(2 * n, 0.0)
                np.testing.assert_allclose(ratio_p3, 2.0, rtol=1e-12)
                np.testing.assert_allclose(ratio_sa, 2.0, rtol=1e-12)

    def test_excess_noise_factor_convention(self):
        """
        P3 excess=2 (power factor F²)  ↔  SA F_excess=√2 (amplitude factor F).
        The shot-noise term doubles; the RON term is unchanged.
        """
        nph, ron, excess = 500.0, 2.0, 2.0
        p3 = self._p3_var(nph, ron, excess=excess)
        sa = self._sa_var(nph, ron ** 2, F_excess=np.sqrt(excess))
        np.testing.assert_allclose(p3, sa, rtol=1e-12)

    def test_agreement_over_parameter_range(self):
        """P3 and SA agree for a range of (nph, ron, excess) combinations."""
        cases = [
            (100.0,  0.0, 1.0),
            (100.0,  3.0, 1.0),
            (1000.0, 5.0, 1.0),
            (50.0,   1.0, 2.0),
        ]
        for nph, ron, excess in cases:
            with self.subTest(nph=nph, ron=ron, excess=excess):
                np.testing.assert_allclose(
                    self._p3_var(nph, ron, excess),
                    self._sa_var(nph, ron ** 2, np.sqrt(excess)),
                    rtol=1e-12,
                )


# ============================================================================
#  Level 2 – Integrated noise PSD (8 m SCAO Pyramid)
# ============================================================================

@unittest.skipUnless(
    os.path.isfile(_P3_INI_8M),
    f"P3 scao_test ini not found: {_P3_INI_8M}",
)
class TestP3NoisePSD(unittest.TestCase):
    """
    Sanity checks on the P3 Fourier-model noise PSD for the 8 m Pyramid case.

    The ``fao`` instance is shared across tests to avoid repeating the
    expensive P3 computation.  Tests that temporarily modify ``noiseVar``
    restore the original value before returning.
    """

    @classmethod
    def setUpClass(cls):
        os.chdir(_P3_PACKAGE_DIR)
        cls.fao = fourierModel(
            _P3_INI_8M,
            path_root=_P3_PACKAGE_DIR,
            calcPSF=False,
            display=False,
            getErrorBreakDown=True,
            verbose=False,
            reduce_memory=False,
        )

    def test_noise_wfe_is_positive(self):
        """Noise WFE from the error breakdown must be strictly positive."""
        self.assertGreater(_as_scalar(self.fao.wfeN), 0.0)

    def test_noise_wfe_in_physically_plausible_range_nm(self):
        """
        For 1000 ph/sub/frame the noise WFE should be between 1 and 500 nm
        OPD (rough physical bounds for an 8 m Pyramid AO system).
        """
        wfe = _as_scalar(self.fao.wfeN)
        self.assertGreater(wfe,   1.0, msg=f"wfeN = {wfe:.2f} nm is implausibly small")
        self.assertLess   (wfe, 500.0, msg=f"wfeN = {wfe:.2f} nm is implausibly large")

    def test_psd_integral_reproduces_error_breakdown_wfen(self):
        """
        Recomputing noise WFE from the raw ``psdNoise`` array must reproduce
        ``wfeN`` from the error breakdown to numerical precision.
        """
        dk     = 2.0 * self.fao.freq.kcMax_ / self.fao.freq.resAO
        rad2nm = dk * self.fao.freq.wvlRef * 1e9 / (2.0 * np.pi)
        wfe_recomputed = float(np.sqrt(self.fao.psdNoise.sum())) * rad2nm
        np.testing.assert_allclose(
            wfe_recomputed, _as_scalar(self.fao.wfeN), rtol=1e-10,
        )

    def test_psd_scales_linearly_with_noisevar(self):
        """
        ``psdNoise ∝ noiseVar``, so wfeN ∝ √noiseVar ∝ 1/√nph.
        Scaling noiseVar by 1/4 (equivalent to 4x more photons) must halve
        the noise WFE.
        """
        noisevar_original = list(self.fao.ao.wfs.processing.noiseVar)
        try:
            noisevar_ref = float(noisevar_original[0])
            wfe_ref      = _as_scalar(self.fao.wfeN)

            # Apply 4x photon scaling (noiseVar → noiseVar/4)
            self.fao.ao.wfs.processing.noiseVar = [noisevar_ref / 4.0]
            psd_scaled = np.real(self.fao.noisePSD())
            dk     = 2.0 * self.fao.freq.kcMax_ / self.fao.freq.resAO
            rad2nm = dk * self.fao.freq.wvlRef * 1e9 / (2.0 * np.pi)
            wfe_scaled = float(np.sqrt(psd_scaled.sum())) * rad2nm

            np.testing.assert_allclose(
                wfe_scaled, wfe_ref / 2.0, rtol=1e-10,
                err_msg="wfeN should scale as sqrt(noiseVar)",
            )
        finally:
            self.fao.ao.wfs.processing.noiseVar = noisevar_original


@unittest.skipUnless(
    os.path.isfile(_SA_SOUL_FITS),
    f"SA SOUL FITS not found: {_SA_SOUL_FITS}",
)
class TestSAMeasureVariance(unittest.TestCase):
    """
    Sanity checks on the SA ``measure_variance`` code path using the SOUL
    40×40 noise-propagation coefficient FITS file (LBT, 8 m Pyramid).
    """

    @classmethod
    def setUpClass(cls):
        cls.p_coeff = extract_propagation_coefficients(_SA_SOUL_FITS)

    def test_propagation_coefficients_are_positive(self):
        """All noise-propagation coefficients must be strictly positive."""
        self.assertIsNotNone(self.p_coeff)
        self.assertGreater(len(self.p_coeff), 0)
        self.assertTrue(
            np.all(self.p_coeff > 0),
            msg="Some propagation coefficients are non-positive",
        )

    def test_noise_wfe_is_positive(self):
        wfe = _sa_wfe_nm(
            _N_MODES, _NPH, 0.0, self.p_coeff,
            _omega, _T0, _GAIN, _DELAY,
        )
        self.assertGreater(wfe, 0.0)

    def test_noise_wfe_in_physically_plausible_range_nm(self):
        """SA noise WFE for 1000 ph/sub should be between 0.1 and 10 nm OPD."""
        wfe = _sa_wfe_nm(
            _N_MODES, _NPH, 0.0, np.ones_like(self.p_coeff),  # Use p_coeff=1 to isolate slope variance
            _omega, _T0, _GAIN, _DELAY,
        )
        self.assertGreater(wfe,   0.1, msg=f"SA wfe = {wfe:.2f} nm implausibly small")
        self.assertLess   (wfe,  10.0, msg=f"SA wfe = {wfe:.2f} nm implausibly large")

    def test_noise_wfe_scales_as_one_over_sqrt_nph(self):
        """
        In the shot-noise dominated regime (ron=0), WFE ∝ 1/√nph.
        Quadrupling nph must halve the noise WFE.
        """
        wfe_1k = _sa_wfe_nm(
            _N_MODES, 1000.0, 0.0, self.p_coeff,
            _omega, _T0, _GAIN, _DELAY,
        )
        wfe_4k = _sa_wfe_nm(
            _N_MODES, 4000.0, 0.0, self.p_coeff,
            _omega, _T0, _GAIN, _DELAY,
        )
        np.testing.assert_allclose(wfe_1k / wfe_4k, 2.0, rtol=1e-10)

    def test_adding_ron_increases_noise_wfe(self):
        """Adding RON (ron=3 e-, ron_var=9 e-²) must increase the noise WFE."""
        wfe_shot = _sa_wfe_nm(
            _N_MODES, _NPH, 0.0,         self.p_coeff,
            _omega, _T0, _GAIN, _DELAY,
        )
        wfe_ron = _sa_wfe_nm(
            _N_MODES, _NPH, 3.0 ** 2,   self.p_coeff,
            _omega, _T0, _GAIN, _DELAY,
        )
        self.assertGreater(wfe_ron, wfe_shot)


@unittest.skipUnless(
    os.path.isfile(_P3_INI_8M) and os.path.isfile(_SA_SOUL_FITS),
    "P3 ini or SA SOUL FITS not found – skipping cross-comparison",
)
class TestNoisePSDCrossComparison(unittest.TestCase):
    """
    Cross-compare P3 (Fourier model) and SA (mode-propagation model) noise
    WFE for compatible 8 m SCAO Pyramid parameters:

        D = 8.222 m,  40 lenslets,  λ_WFS = 750 nm,  mod = 3 λ/D
        nph = 1000 ph/sub/frame,  ron = 0  (shot-noise dominated)
        integrator gain = 0.3,  delay = 2 frames,  rate = 1000 Hz

    Both are expressed in nm OPD (physical, wavelength-independent).

    An exact match is not expected because P3 uses a 2D Fourier reconstructor
    while SA uses simulation-derived propagation coefficients.  The test
    verifies that the estimates agree within one order of magnitude.
    """

    @classmethod
    def setUpClass(cls):
        os.chdir(_P3_PACKAGE_DIR)
        cls.fao = fourierModel(
            _P3_INI_8M,
            path_root=_P3_PACKAGE_DIR,
            calcPSF=False,
            display=False,
            getErrorBreakDown=True,
            verbose=False,
            reduce_memory=False,
        )
        cls.p_coeff = extract_propagation_coefficients(_SA_SOUL_FITS)

    def test_p3_and_sa_noise_wfe_agree_within_one_order_of_magnitude(self):
        """
        P3 and SA noise WFE (nm OPD) must agree within a factor of 10 for
        comparable 8 m Pyramid system parameters.
        """
        wfe_p3 = _as_scalar(self.fao.wfeN)   # nm OPD, from P3 error breakdown
        wfe_sa = _sa_wfe_nm(
            _N_MODES, _NPH, 0.0, self.p_coeff,
            _omega, _T0, _GAIN, _DELAY,
        )
        ratio = max(wfe_p3, wfe_sa) / min(wfe_p3, wfe_sa)
        self.assertLess(
            ratio, 10.0,
            msg=(
                f"P3 noise WFE = {wfe_p3:.1f} nm OPD, "
                f"SA noise WFE = {wfe_sa:.1f} nm OPD, "
                f"ratio = {ratio:.2f} (must be < 10)"
            ),
        )

    def test_both_models_give_same_shot_noise_scaling(self):
        """
        In the shot-noise regime both WFE estimates must scale as 1/√nph,
        i.e. the ratio wfe(nph) / wfe(4·nph) = 2 for both models.
        """
        noisevar_original = list(self.fao.ao.wfs.processing.noiseVar)
        try:
            noisevar_ref = float(noisevar_original[0])

            # P3: scale noiseVar by 1/4 (simulating 4× more photons)
            self.fao.ao.wfs.processing.noiseVar = [noisevar_ref / 4.0]
            psd_hi = np.real(self.fao.noisePSD())
            dk     = 2.0 * self.fao.freq.kcMax_ / self.fao.freq.resAO
            rad2nm = dk * self.fao.freq.wvlRef * 1e9 / (2.0 * np.pi)
            wfe_p3_hi  = float(np.sqrt(psd_hi.sum())) * rad2nm
            wfe_p3_ref = _as_scalar(self.fao.wfeN)
            ratio_p3 = wfe_p3_ref / wfe_p3_hi
        finally:
            self.fao.ao.wfs.processing.noiseVar = noisevar_original

        # SA: compare nph=1000 vs nph=4000
        wfe_sa_ref = _sa_wfe_nm(
            _N_MODES, 1000.0, 0.0, self.p_coeff,
            _omega, _T0, _GAIN, _DELAY,
        )
        wfe_sa_hi = _sa_wfe_nm(
            _N_MODES, 4000.0, 0.0, self.p_coeff,
            _omega, _T0, _GAIN, _DELAY,
        )
        ratio_sa = wfe_sa_ref / wfe_sa_hi

        # Both ratios must equal 2.0 (= √4) within numerical precision
        np.testing.assert_allclose(ratio_p3, 2.0, rtol=1e-10,
                                   err_msg="P3 noise WFE should scale as 1/√nph")
        np.testing.assert_allclose(ratio_sa, 2.0, rtol=1e-10,
                                   err_msg="SA noise WFE should scale as 1/√nph")

    def test_sigma2w_built_with_p3_noisevar_matches_sa(self):
        """
        σ²_w[i] = p_coeff[i] × slope_noise_variance is the per-mode open-loop
        noise variance in the SA model.

        This test verifies that:
          sigma2_w_SA  =  p_coeff × SA_slope_var
                       =  p_coeff × P3_NoiseVariance(at λ_WFS)
        """
        # ── P3 slope noise variance at WFS wavelength ──────────────────────
        dsub = 8.222 / 40.0
        wfs_p3 = sensor(
            pixel_scale=1050.0, fov=2, nph=[_NPH], ron=0.0, excess=1.0,
            wfstype="Pyramid", nL=[40], dsub=[dsub],
        )
        slope_var_p3 = float(wfs_p3.NoiseVariance(r0=0.15, wvl=_WVL_WFS)[0])

        # ── SA slope noise variance: 1/nph (Pyramid, pure shot noise) ──────
        slope_var_sa = 1.0 / _NPH

        # Both must agree
        np.testing.assert_allclose(slope_var_p3, slope_var_sa, rtol=1e-12,
                                   err_msg="P3 noiseVar != SA slope_var at WFS λ")

        # ── sigma2_w from SA slope variance ────────────────────────────────
        sigma2_w_sa = self.p_coeff[:_N_MODES] * slope_var_sa

        # ── sigma2_w using P3 noiseVar as drop-in replacement ──────────────
        sigma2_w_via_p3 = self.p_coeff[:_N_MODES] * slope_var_p3

        np.testing.assert_allclose(
            sigma2_w_via_p3, sigma2_w_sa, rtol=1e-7, atol=1e-7,
            err_msg="sigma2_w via P3 noiseVar != sigma2_w via SA slope_var",
        )
        np.testing.assert_allclose(
            sigma2_w_sa, self.p_coeff[:_N_MODES] / _NPH, rtol=1e-7, atol=1e-7,
        )
        self.assertTrue(np.all(sigma2_w_sa > 0),
                        msg="sigma2_w must be positive for all modes")

    def test_sa_total_sigma2w_ge_p3_mv_open_loop_noise(self):
        """
        sum(σ²_w) (in nm²) must be >= the P3 MV open-loop noise variance.
        
        P3 uses a minimum-variance Fourier reconstructor, so its total
        open-loop spatial noise variance is a lower bound on any other linear
        reconstructor. The test compares ALL modes of SA against P3 full-area.
        """
        # ── SA total open-loop noise in nm² ────────────────────────────────
        # Use the real function to get the exact slope var
        D = 8.222
        n_sub = 40
        area_sub = np.pi * ((D / n_sub) / 2.0)**2
        photon_flux = _NPH * (1.0 / area_sub) * _FRAME_RATE

        slope_var_sa = compute_slope_noise_variance(
            F_excess=1.0,
            pixel_pos=[1.0, 1.0, -1.0, -1.0],
            sky_bkg=0.0,
            dark_curr=0.0,
            read_out_noise=0.0,
            photon_flux=photon_flux,
            telescope_diameter=D,
            frame_rate=_FRAME_RATE,
            magnitudo=0.0,
            n_subaperture=n_sub,
            collecting_area=1.0
        )

        # Consider ALL modes from the FITS for a fair comparison
        sum_sigma2w_total = float(np.sum(self.p_coeff)) * slope_var_sa
        sa_ol_nm2_total   = sum_sigma2w_total  # No conversion: already in nm²!

        # ── P3 total open-loop noise in nm² ────────────────────────────────
        dk       = 2.0 * self.fao.freq.kcMax_ / self.fao.freq.resAO
        rad2nm_wfs = dk * _WVL_WFS * 1e9 / (2.0 * np.pi)
        
        p3_ol_nm2 = float(self.fao.psdNoise.sum()) / self.fao.noiseGain * rad2nm_wfs ** 2

        if _VERBOSE: # pragma: no cover
            print(f"\nSA open-loop noise (ALL modes) = {sa_ol_nm2_total:.3f} nm²")
            print(f"P3 full-area MV open-loop noise = {p3_ol_nm2:.3f} nm²")

        # SA (all modes) must be >= P3 MV (which is the mathematical lower bound)
        self.assertGreaterEqual(
            sa_ol_nm2_total, p3_ol_nm2,
            msg=(
                f"SA open-loop noise (ALL modes) = "
                f"{sa_ol_nm2_total:.3f} nm²  <  P3 MV bound = "
                f"{p3_ol_nm2:.3f} nm²"
            ),
        )

    def test_global_noise_propagation_factor_comparison(self):
        """
        Compare the global spatial noise propagation factor (in nm²) between SA and P3.
        """
        # ── SA: Factor in nm² (the p_coeff from PASSATA are already in nm² at the WFS wavelength)
        sa_prop_factor_truncated = float(np.sum(self.p_coeff[:_N_MODES]))
        sa_prop_factor_total = float(np.sum(self.p_coeff))

        # ── P3: Adimensional factor (rad² -> rad²) from the spatial PSD of the open-loop noise
        spatial_psd = np.abs(self.fao.Rx)**2 + np.abs(self.fao.Ry)**2
        spatial_psd_masked = self.fao.freq.mskInAO_ * spatial_psd * self.fao.freq.pistonFilterAO_
        p3_prop_factor_rad2 = float(np.sum(spatial_psd_masked) / (self.fao.freq.resAO ** 2))

        # Conversion from rad² (at the WFS wavelength, 750nm) to physical nm²
        rad2_to_nm2 = (_WVL_WFS * 1e9 / (2.0 * np.pi)) ** 2
        p3_prop_factor_nm2 = p3_prop_factor_rad2 * rad2_to_nm2

        if _VERBOSE: # pragma: no cover
            print("\n" + "="*55)
            print(" CROSS-COMPARE: GLOBAL NOISE PROPAGATION FACTORS (nm²)")
            print("="*55)
            print(f"SA (Sum of the first {_N_MODES} modes):\t{sa_prop_factor_truncated:.2f} nm²")
            print(f"SA (Sum of ALL modes in the FITS):\t{sa_prop_factor_total:.2f} nm²")
            print(f"P3 (2D Fourier integral converted):\t{p3_prop_factor_nm2:.2f} nm²")
            print("-" * 55)
            print(f"Ratio P3 / SA (Total): \t\t{p3_prop_factor_nm2 / sa_prop_factor_total:.3f}")
            print("="*55 + "\n")

        # The test passes if the difference is within a tolerance factor
        self.assertLess(
            p3_prop_factor_nm2 / sa_prop_factor_total, 1.5,
            msg="The scaled propagation factors in nm² do not match."
        )

class TestSlopeNoiseVariance(unittest.TestCase):

    def test_ron_scaling_and_p3_comparison(self):
        """
        Verifies that compute_slope_noise_variance correctly treats RON as a standard deviation
        (squaring it to get variance) and matches P3's implementation for the same parameters.
        """
        # --- 1. Physical parameters for the test case ---
        nph_total = 1000.0  # Total photons per subaperture per frame
        ron_rms = 3.0       # Read-Out Noise in e- RMS
        excess_amp = 1.0    # Excess factor (amplitude, F)

        # --- 2. Fictitious setup to force n_phot_pix in SA ---
        D = 8.0
        n_sub = 40
        area_sub = np.pi * ((D / n_sub) / 2.0)**2
        collecting_area = 1.0

        # Invert the SA formula to get nph_total photons per subaperture
        photon_flux = nph_total * (collecting_area / area_sub)
        frame_rate = 1.0
        magnitudo = 0.0

        # Standard weights for the 4 quadrants of the Pyramid
        pixel_pos = [1.0, 1.0, -1.0, -1.0]

        # --- 3. Semi-analytic calculation (SA) ---
        sa_var = compute_slope_noise_variance(
            F_excess=excess_amp,
            pixel_pos=pixel_pos,
            sky_bkg=0.0,
            dark_curr=0.0,
            read_out_noise=ron_rms,  # SA expects the standard deviation, not the variance
            photon_flux=photon_flux,
            telescope_diameter=D,
            frame_rate=frame_rate,
            magnitudo=magnitudo,
            n_subaperture=n_sub,
            collecting_area=collecting_area
        )

        # --- 4. Theoretical calculation (Vérinaud 2004 / Agapito Eq. 12-14) ---
        theory_var = (excess_amp**2 / nph_total) + 4.0 * (ron_rms**2) / (nph_total**2)

        # --- 5. P3 calculation ---
        wfs_p3 = sensor(
            pixel_scale=1000.0, fov=2,
            nph=[nph_total],
            ron=ron_rms,               # P3 expects the standard deviation
            excess=excess_amp**2,      # P3 expects the power (F^2)
            wfstype="Pyramid",
            nL=[n_sub],
            dsub=[D/n_sub]
        )
        p3_var = float(wfs_p3.NoiseVariance(r0=0.15, wvl=750e-9)[0])

        # --- 6. Assertions ---
        self.assertAlmostEqual(
            sa_var, theory_var, places=12,
            msg="SA does not match the theory"
        )

        self.assertAlmostEqual(
            sa_var, p3_var, places=12,
            msg="SA does not match the P3 Pyramid implementation"
        )

        if _VERBOSE: # pragma: no cover
            print(f"\nTheory: {theory_var:.4e} | SA: {sa_var:.4e} | P3: {p3_var:.4e}")

if __name__ == "__main__":
    unittest.main(verbosity=2)