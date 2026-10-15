#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for src/facs_functions.py (Full-Aperture Centroid Sensor tip/tilt
error-budget functions).

Run from the Semi-analytic-Error-Budget root with:
    python -m pytest tests/test_facs_functions.py -v
"""

import unittest

import numpy as np
from scipy.integrate import dblquad
from scipy.special import j1
from mastsel.mavisUtilities import meanVarPixelThr

from src.Functions import compute_noise_PSD_intermediate, compute_output_PSD_and_integrate
from src.facs_functions import (
    CRAD2AS,
    facs_long_exposure_psf,
    wcog_weight,
    wcog_gain_numerical,
    strehl_ratio_marechal,
    facs_pixel_noise_variance,
    facs_centroid_noise_variance_pix,
    facs_centroid_variance_pix_to_tiptilt_nm2,
    facs_slope_noise_variance_nm2,
    facs_measurement_variance,
    facs_total_variance,
    facs_gain_sweep,
)
from src.Functions import build_transfer_function, temporal_variance, funct_d2


# Cheap but realistic PSF set-up: 8 m telescope at 800 nm, ~300 controlled modes.
D_TEL = 8.0
WAVELENGTH = 800e-9
N_SAMP = 2
OUTER_SCALE = 25.0
F_CUT = np.sqrt(300 / np.pi) / D_TEL
N_PIX = 16
RAD2_TO_NM2 = (WAVELENGTH * 1e9 / (2 * np.pi)) ** 2


def _psf(s2_inband_rad2=0.0, s2_fit_rad2=0.0, n_pixels=N_PIX, x0=0.0):
    """Build a FACS PSF with residuals given in rad^2."""
    return facs_long_exposure_psf(
        n_pixels, N_SAMP, D_TEL, WAVELENGTH,
        s2_inband_rad2 * RAD2_TO_NM2, s2_fit_rad2 * RAD2_TO_NM2,
        F_CUT, OUTER_SCALE, x0=x0,
    )


class TestFacsLongExposurePsf(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.psf0 = _psf()

    def test_zero_residual_central_pixels_match_airy(self):
        """Central pixels equal the Airy pattern integrated over one pixel."""
        c = N_PIX // 2
        central = self.psf0[c - 1:c + 1, c - 1:c + 1]

        def airy(y, x):
            # Unit-flux Airy, coordinates in pixels (lambda/D = N_SAMP pixels).
            r = np.hypot(x, y) * np.pi / N_SAMP
            return (np.pi / 4) / N_SAMP ** 2 * (1.0 if r == 0 else (2 * j1(r) / r) ** 2)

        expected, _ = dblquad(airy, 0, 1, 0, 1, epsabs=1e-10)
        np.testing.assert_allclose(central, expected, rtol=2e-3)

    def test_zero_residual_symmetries(self):
        np.testing.assert_allclose(self.psf0, self.psf0.T, rtol=1e-6, atol=1e-12)
        np.testing.assert_allclose(self.psf0, self.psf0[::-1, :], rtol=1e-6, atol=1e-12)
        np.testing.assert_allclose(self.psf0, self.psf0[:, ::-1], rtol=1e-6, atol=1e-12)

    def test_zero_residual_window_flux(self):
        self.assertGreater(self.psf0.sum(), 0.9)
        self.assertLess(self.psf0.sum(), 1.0)

    def test_inband_residual_reduces_peak_as_exp_minus_s2(self):
        ratio = _psf(s2_inband_rad2=0.5).max() / self.psf0.max()
        self.assertGreaterEqual(ratio, np.exp(-0.5))
        self.assertAlmostEqual(ratio, np.exp(-0.5), delta=3e-2 * np.exp(-0.5))

    def test_fitting_residual_reduces_peak_as_exp_minus_s2(self):
        ratio = _psf(s2_fit_rad2=0.5).max() / self.psf0.max()
        self.assertAlmostEqual(ratio, np.exp(-0.5), delta=3e-2 * np.exp(-0.5))

    def test_window_flux_decreases_with_inband_variance(self):
        flux = [_psf(s2_inband_rad2=s2).sum() for s2 in (0.0, 0.2, 0.5, 1.0)]
        self.assertTrue(np.all(np.diff(flux) < 0))

    def test_window_flux_decreases_with_fitting_variance(self):
        flux = [_psf(s2_fit_rad2=s2).sum() for s2 in (0.0, 0.2, 0.5, 1.0)]
        self.assertTrue(np.all(np.diff(flux) < 0))

    def test_x0_shifts_map_by_one_column(self):
        """x0 = +/-1 translates the whole spot by exactly one column.

        The peak test via argmax is ill-posed here: for an even window the
        spot center falls between two pixels of equal value.
        """
        inner = slice(2, N_PIX - 2)
        shifted_right = _psf(x0=1.0)
        shifted_left = _psf(x0=-1.0)
        np.testing.assert_allclose(shifted_right[inner, 3:N_PIX - 1], self.psf0[inner, 2:N_PIX - 2],
                                   rtol=1e-6, atol=1e-12)
        np.testing.assert_allclose(shifted_left[inner, 1:N_PIX - 3], self.psf0[inner, 2:N_PIX - 2],
                                   rtol=1e-6, atol=1e-12)

    def test_rejects_undersampled_psf(self):
        with self.assertRaises(ValueError):
            facs_long_exposure_psf(N_PIX, 1, D_TEL, WAVELENGTH, 0.0, 0.0, F_CUT, OUTER_SCALE)


class TestWcogGainNumerical(unittest.TestCase):

    @staticmethod
    def _gaussian(n_pixels, fwhm, x0):
        xx, yy = np.meshgrid(*(2 * [np.arange(n_pixels) - (n_pixels - 1) / 2.0]))
        sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
        return np.exp(-((xx - x0) ** 2 + yy ** 2) / (2 * sigma ** 2))

    def test_gaussian_spot_matches_thomas_closed_form(self):
        n_t, n_w, shift = 6.0, 8.0, 0.01
        plus = self._gaussian(64, n_t, shift)
        minus = self._gaussian(64, n_t, -shift)
        gain = wcog_gain_numerical(plus, minus, wcog_weight(64, n_w), shift)
        self.assertAlmostEqual(gain, n_w ** 2 / (n_t ** 2 + n_w ** 2), delta=1e-3 * gain)

    def test_uniform_weight_gives_unit_gain(self):
        shift = 0.01
        plus = self._gaussian(64, 6.0, shift)
        minus = self._gaussian(64, 6.0, -shift)
        gain = wcog_gain_numerical(plus, minus, np.ones((64, 64)), shift)
        self.assertAlmostEqual(gain, 1.0, delta=1e-6)

    def test_uniform_weight_on_long_exposure_psf_is_close_to_unity(self):
        shift, n = 0.05, 32
        plus = _psf(n_pixels=n, x0=shift)
        minus = _psf(n_pixels=n, x0=-shift)
        gain = wcog_gain_numerical(plus, minus, np.ones((n, n)), shift)
        self.assertAlmostEqual(gain, 1.0, delta=0.05)


class TestWcogWeight(unittest.TestCase):

    def test_peak_value_is_one_at_center(self):
        n_pixels = 17
        weight = wcog_weight(n_pixels=n_pixels, fwhm_weight=2.0)
        center = n_pixels // 2
        self.assertAlmostEqual(weight[center, center], 1.0, places=12)

    def test_symmetric_under_x_y_swap(self):
        weight = wcog_weight(n_pixels=16, fwhm_weight=2.0)
        np.testing.assert_allclose(weight, weight.T)


class TestStrehlRatioMarechal(unittest.TestCase):

    def test_zero_residual_gives_unit_strehl(self):
        self.assertAlmostEqual(strehl_ratio_marechal(0.0, 800e-9), 1.0, places=12)

    def test_matches_closed_form(self):
        residual_nm, wavelength = 100.0, 800e-9
        sigma_rad = residual_nm * 1e-9 * 2 * np.pi / wavelength
        expected = np.exp(-sigma_rad ** 2)
        self.assertAlmostEqual(strehl_ratio_marechal(residual_nm, wavelength), expected, places=12)

    def test_decreases_with_larger_residual(self):
        s_small = strehl_ratio_marechal(50.0, 800e-9)
        s_large = strehl_ratio_marechal(200.0, 800e-9)
        self.assertGreater(s_small, s_large)

    def test_strehl_is_bounded_in_unit_interval(self):
        s = strehl_ratio_marechal(157.0, 800e-9)
        self.assertGreater(s, 0.0)
        self.assertLessEqual(s, 1.0)


class TestFacsPixelNoiseVariance(unittest.TestCase):

    def test_matches_direct_meanVarPixelThr_call(self):
        rng = np.random.default_rng(0)
        flux_map = rng.uniform(10.0, 1000.0, size=(8, 8))

        mean_ref, var_ref = meanVarPixelThr(flux_map, ron=3.0, bg=0.5, excess=1.5)
        mean_wrap, var_wrap = facs_pixel_noise_variance(
            flux_map, ron=3.0, sky_bkg=0.3, dark_curr=0.2, excess_noise_factor2=1.5,
        )

        np.testing.assert_allclose(mean_wrap, mean_ref, rtol=1e-10)
        np.testing.assert_allclose(var_wrap, var_ref, rtol=1e-10)


class TestFacsCentroidNoiseVariancePix(unittest.TestCase):

    def test_matches_hand_computed_formula_on_toy_grid(self):
        n_pixels = 4
        flux_map = np.full((n_pixels, n_pixels), 100.0)
        weight_map = np.ones((n_pixels, n_pixels))
        sigma2_map = np.full((n_pixels, n_pixels), 5.0)
        gamma = 1.3

        coords = np.arange(n_pixels) - (n_pixels - 1) / 2.0
        xx, _ = np.meshgrid(coords, coords)
        expected = gamma ** 2 * np.sum((xx ** 2) * (weight_map ** 2) * sigma2_map) \
            / np.sum(flux_map * weight_map) ** 2

        result = facs_centroid_noise_variance_pix(flux_map, weight_map, sigma2_map, gamma)
        self.assertAlmostEqual(result, expected, places=12)

    def test_scales_as_inverse_square_of_total_flux(self):
        n_pixels = 8
        weight_map = np.ones((n_pixels, n_pixels))
        sigma2_map = np.ones((n_pixels, n_pixels))

        var_low = facs_centroid_noise_variance_pix(
            np.full((n_pixels, n_pixels), 10.0), weight_map, sigma2_map, gamma=1.0,
        )
        var_high = facs_centroid_noise_variance_pix(
            np.full((n_pixels, n_pixels), 20.0), weight_map, sigma2_map, gamma=1.0,
        )
        # Doubling the (uniform) flux doubles the denominator's sqrt term,
        # so the variance drops by a factor of 4.
        self.assertAlmostEqual(var_low / var_high, 4.0, places=10)


class TestUnitConversion(unittest.TestCase):

    def test_scales_linearly_with_telescope_diameter(self):
        sigma2_pix = 1e-4
        pixel_scale_arcsec = 2e-3

        nm2_small_d = facs_centroid_variance_pix_to_tiptilt_nm2(sigma2_pix, pixel_scale_arcsec, telescope_diameter=8.0)
        nm2_large_d = facs_centroid_variance_pix_to_tiptilt_nm2(sigma2_pix, pixel_scale_arcsec, telescope_diameter=16.0)

        # sigma_nm ~ D, so sigma2_nm ~ D^2: doubling D quadruples the variance.
        self.assertAlmostEqual(nm2_large_d / nm2_small_d, 4.0, places=10)

    def test_matches_user_provided_formula(self):
        # theta[arcsec] = a[nm] * factor*1e-9/D * CRAD2AS  =>  invert for a[nm].
        telescope_diameter = 38.5
        a_nm = 100.0
        factor = 4.0
        theta_arcsec = a_nm * factor * 1e-9 / telescope_diameter * CRAD2AS

        sigma2_pix = (theta_arcsec) ** 2  # pixel_scale_arcsec = 1 arcsec/pixel
        sigma2_nm2 = facs_centroid_variance_pix_to_tiptilt_nm2(
            sigma2_pix, pixel_scale_arcsec=1.0, telescope_diameter=telescope_diameter,
            tilt_to_angle_factor=factor,
        )
        self.assertAlmostEqual(np.sqrt(sigma2_nm2), a_nm, places=6)


class TestFacsSlopeNoiseVarianceNm2(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.psf = _psf(s2_inband_rad2=0.1, s2_fit_rad2=0.1)
        cls.weight = wcog_weight(N_PIX, 6.0)

    def setUp(self):
        self.kwargs = dict(
            weight_map=self.weight, gamma=1.3, ron=0.5, sky_bkg=0.2, dark_curr=0.1,
            excess_noise_factor2=1.5, pixel_scale_arcsec=2.14e-3,
            telescope_diameter=D_TEL,
        )

    def test_matches_manual_chain(self):
        flux = 1e4 * self.psf
        result = facs_slope_noise_variance_nm2(flux, **self.kwargs)

        _, var_map = facs_pixel_noise_variance(
            flux, 0.5, 0.2, 0.1, 1.5,
        )
        sigma2_pix = facs_centroid_noise_variance_pix(flux, self.weight, var_map, 1.3)
        expected = facs_centroid_variance_pix_to_tiptilt_nm2(sigma2_pix, 2.14e-3, D_TEL)
        self.assertGreater(result, 0.0)
        np.testing.assert_allclose(result, expected, rtol=1e-12)

    def test_photon_limited_variance_halves_when_flux_doubles(self):
        kwargs = dict(self.kwargs, ron=0.0, sky_bkg=0.0, dark_curr=0.0,
                      excess_noise_factor2=1.0)
        var1 = facs_slope_noise_variance_nm2(1e6 * self.psf, **kwargs)
        var2 = facs_slope_noise_variance_nm2(2e6 * self.psf, **kwargs)
        np.testing.assert_allclose(var1 / var2, 2.0, rtol=1e-6)

    def test_scales_as_gamma_squared(self):
        flux = 1e4 * self.psf
        var1 = facs_slope_noise_variance_nm2(flux, **dict(self.kwargs, gamma=1.0))
        var2 = facs_slope_noise_variance_nm2(flux, **dict(self.kwargs, gamma=2.0))
        np.testing.assert_allclose(var2 / var1, 4.0, rtol=1e-10)

    def test_increases_with_readout_noise(self):
        flux = 1e3 * self.psf
        var_low = facs_slope_noise_variance_nm2(flux, **dict(self.kwargs, ron=0.1))
        var_high = facs_slope_noise_variance_nm2(flux, **dict(self.kwargs, ron=5.0))
        self.assertGreater(var_high, var_low)


class TestFacsMeasurementVariance(unittest.TestCase):

    def test_matches_manual_noise_psd_propagation(self):
        omega = 2 * np.pi * np.logspace(0, 2, 200)
        sigma2_tt_nm2 = 4.0
        transf_funct = np.ones((2, omega.size), dtype=complex)  # identity NTF

        var_OL, var_CL, PSD_out, PSD_in = facs_measurement_variance(transf_funct, omega, sigma2_tt_nm2)

        sigma2_w = np.full(2, sigma2_tt_nm2)
        PSD_in_expected = compute_noise_PSD_intermediate(omega, 2, sigma2_w)
        var_OL_expected, var_CL_expected, PSD_out_expected = compute_output_PSD_and_integrate(
            2, transf_funct, PSD_in_expected, omega,
        )

        np.testing.assert_allclose(PSD_in, PSD_in_expected)
        np.testing.assert_allclose(PSD_out, PSD_out_expected)
        self.assertAlmostEqual(var_OL, var_OL_expected, places=10)
        self.assertAlmostEqual(var_CL, var_CL_expected, places=10)

    def test_identity_ntf_leaves_variance_unchanged(self):
        omega = 2 * np.pi * np.logspace(0, 2, 200)
        sigma2_tt_nm2 = 9.0
        transf_funct = np.ones((2, omega.size), dtype=complex)

        var_OL, var_CL, _, _ = facs_measurement_variance(transf_funct, omega, sigma2_tt_nm2)
        self.assertAlmostEqual(var_OL, var_CL, places=8)


class TestFacsTotalVariance(unittest.TestCase):

    def test_sums_ho_and_tt_contributions(self):
        var_total, var_ho, var_tt = facs_total_variance(
            var_fit_ho=10.0, var_temp_ho=20.0, var_meas_ho=3.0, var_alias_ho=7.0,
            var_temp_tt=50.0, var_meas_tt=1.0,
        )
        self.assertAlmostEqual(var_ho, 40.0, places=12)
        self.assertAlmostEqual(var_tt, 51.0, places=12)
        self.assertAlmostEqual(var_total, 91.0, places=12)

    def test_matches_sum_of_the_two_branches(self):
        var_total, var_ho, var_tt = facs_total_variance(
            var_fit_ho=5.0, var_temp_ho=5.0, var_meas_ho=5.0, var_alias_ho=5.0,
            var_temp_tt=5.0, var_meas_tt=5.0,
        )
        self.assertAlmostEqual(var_total, var_ho + var_tt, places=12)


class TestFacsGainSweep(unittest.TestCase):

    def setUp(self):
        self.t_0 = 1e-3
        self.omega = 2 * np.pi * np.logspace(-2, np.log10(1 / (2 * self.t_0)), 500)
        self.plant_num = np.array([1.0])
        self.plant_den = funct_d2(2)
        self.gains = np.arange(0.05, 0.95, 0.05)
        self.zeros = np.zeros((2, self.omega.size))
        # Narrow bump at 0.5 Hz, well below the loop bandwidth
        f = self.omega / (2 * np.pi)
        bump = 1e4 * np.exp(-0.5 * ((f - 0.5) / 0.1) ** 2)
        self.psd_lf = np.vstack([bump, bump])

    def _sweep(self, psd_atmo, psd_vib, sigma2, gains=None):
        gains = self.gains if gains is None else gains
        return facs_gain_sweep(gains, self.omega, self.t_0, self.plant_num,
                               self.plant_den, psd_atmo, psd_vib, sigma2)

    def test_output_shape(self):
        best, var = self._sweep(self.psd_lf, self.zeros, 100.0)
        self.assertEqual(var.shape, (self.gains.size, 2))
        self.assertIn(best, self.gains)

    def test_noise_only_selects_smallest_gain(self):
        best, var = self._sweep(self.zeros, self.zeros, 100.0)
        self.assertEqual(best, self.gains[0])
        np.testing.assert_allclose(var[:, 0], 0.0, atol=1e-12)
        self.assertTrue(np.all(np.diff(var[:, 1]) > 0))

    def test_low_frequency_disturbance_selects_largest_gain(self):
        best, var = self._sweep(self.psd_lf, self.zeros, 0.0)
        self.assertEqual(best, self.gains[-1])
        self.assertTrue(np.all(np.diff(var[:, 0]) < 0))
        np.testing.assert_allclose(var[:, 1], 0.0, atol=1e-12)

    def test_matches_direct_computation(self):
        sigma2 = 50.0
        psd_vib = 0.1 * self.psd_lf
        _, var = self._sweep(self.psd_lf, psd_vib, sigma2)
        k = 7
        H_r, H_n = build_transfer_function(
            self.omega, self.t_0, 2, self.plant_num, self.plant_den,
            gain=np.full(2, self.gains[k]))
        _, v_t, _, _ = temporal_variance(self.psd_lf, psd_vib, H_r, 2, self.omega)
        _, v_m, _, _ = facs_measurement_variance(H_n, self.omega, sigma2)
        np.testing.assert_allclose(var[k], [np.real(v_t), np.real(v_m)], rtol=1e-10)

    def test_linear_in_noise_variance(self):
        _, var1 = self._sweep(self.psd_lf, self.zeros, 30.0)
        _, var2 = self._sweep(self.psd_lf, self.zeros, 60.0)
        np.testing.assert_allclose(var2[:, 1], 2 * var1[:, 1], rtol=1e-10)
        np.testing.assert_allclose(var2[:, 0], var1[:, 0], rtol=1e-12)


if __name__ == "__main__":
    unittest.main(verbosity=2)
