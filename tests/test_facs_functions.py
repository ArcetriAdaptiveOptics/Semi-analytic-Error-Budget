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
from mastsel.mavisUtilities import meanVarPixelThr

from src.Functions import compute_noise_PSD_intermediate, compute_output_PSD_and_integrate
from src.facs_functions import (
    CRAD2AS,
    diffraction_limited_spot,
    wcog_weight,
    wcog_gamma,
    strehl_ratio_marechal,
    broaden_spot_for_strehl,
    facs_pixel_noise_variance,
    facs_centroid_noise_variance_pix,
    facs_centroid_variance_pix_to_tiptilt_nm2,
    facs_slope_noise_variance_nm2,
    facs_measurement_variance,
    facs_total_variance,
)


class TestDiffractionLimitedSpot(unittest.TestCase):

    def test_normalized_to_unit_sum(self):
        spot = diffraction_limited_spot(n_pixels=16, n_samp=2.0)
        self.assertAlmostEqual(np.sum(spot), 1.0, places=12)

    def test_non_negative(self):
        spot = diffraction_limited_spot(n_pixels=16, n_samp=2.0)
        self.assertTrue(np.all(spot >= 0.0))

    def test_peak_at_window_center(self):
        n_pixels = 17  # odd -> exact center pixel
        spot = diffraction_limited_spot(n_pixels=n_pixels, n_samp=2.0)
        center = n_pixels // 2
        self.assertEqual(np.unravel_index(np.argmax(spot), spot.shape), (center, center))

    def test_offset_shifts_the_peak(self):
        n_pixels = 17
        spot = diffraction_limited_spot(n_pixels=n_pixels, n_samp=2.0, x0=2.0, y0=0.0)
        center = n_pixels // 2
        peak_row, peak_col = np.unravel_index(np.argmax(spot), spot.shape)
        self.assertEqual(peak_row, center)
        self.assertEqual(peak_col, center + 2)


class TestWcogWeight(unittest.TestCase):

    def test_peak_value_is_one_at_center(self):
        n_pixels = 17
        weight = wcog_weight(n_pixels=n_pixels, fwhm_weight=2.0)
        center = n_pixels // 2
        self.assertAlmostEqual(weight[center, center], 1.0, places=12)

    def test_symmetric_under_x_y_swap(self):
        weight = wcog_weight(n_pixels=16, fwhm_weight=2.0)
        np.testing.assert_allclose(weight, weight.T)


class TestWcogGamma(unittest.TestCase):

    def test_matches_closed_form(self):
        fwhm_spot, fwhm_weight = 1.772, 2.0
        expected = (fwhm_spot ** 2 + fwhm_weight ** 2) / fwhm_weight ** 2
        self.assertAlmostEqual(wcog_gamma(fwhm_spot, fwhm_weight), expected, places=12)

    def test_equals_one_in_the_narrow_weight_limit(self):
        # As N_w -> infinity (very broad, weight -> uniform) gamma -> 1.
        self.assertAlmostEqual(wcog_gamma(fwhm_spot=2.0, fwhm_weight=1e6), 1.0, places=6)


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


class TestBroadenSpotForStrehl(unittest.TestCase):

    def test_unit_strehl_leaves_inputs_unchanged(self):
        n_samp, fwhm = broaden_spot_for_strehl(2.0, 1.772, strehl=1.0)
        self.assertAlmostEqual(n_samp, 2.0, places=12)
        self.assertAlmostEqual(fwhm, 1.772, places=12)

    def test_matches_inverse_sqrt_strehl_scaling(self):
        strehl = 0.25
        n_samp, fwhm = broaden_spot_for_strehl(2.0, 1.772, strehl)
        self.assertAlmostEqual(n_samp, 2.0 / np.sqrt(strehl), places=12)
        self.assertAlmostEqual(fwhm, 1.772 / np.sqrt(strehl), places=12)

    def test_lower_strehl_broadens_more(self):
        _, fwhm_mild = broaden_spot_for_strehl(2.0, 1.772, strehl=0.8)
        _, fwhm_severe = broaden_spot_for_strehl(2.0, 1.772, strehl=0.2)
        self.assertGreater(fwhm_severe, fwhm_mild)

    def test_rejects_out_of_range_strehl(self):
        with self.assertRaises(ValueError):
            broaden_spot_for_strehl(2.0, 1.772, strehl=0.0)
        with self.assertRaises(ValueError):
            broaden_spot_for_strehl(2.0, 1.772, strehl=1.5)


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

    COMMON_KWARGS = dict(
        ron=0.5, sky_bkg=0.0, dark_curr=0.0, excess_noise_factor2=1.0,
        n_pixels=16, n_samp=2.0, fwhm_spot_pix=1.772, fwhm_weight_pix=2.0,
        pixel_scale_arcsec=2.14e-3, telescope_diameter=38.5,
    )

    def test_is_positive(self):
        sigma2 = facs_slope_noise_variance_nm2(photon_flux_total=1e5, **self.COMMON_KWARGS)
        self.assertGreater(sigma2, 0.0)

    def test_decreases_with_more_photons(self):
        sigma2_faint = facs_slope_noise_variance_nm2(photon_flux_total=1e4, **self.COMMON_KWARGS)
        sigma2_bright = facs_slope_noise_variance_nm2(photon_flux_total=1e8, **self.COMMON_KWARGS)
        self.assertGreater(sigma2_faint, sigma2_bright)

    def test_increases_with_readout_noise(self):
        kwargs_low_ron = dict(self.COMMON_KWARGS)
        kwargs_low_ron['ron'] = 0.1
        kwargs_high_ron = dict(self.COMMON_KWARGS)
        kwargs_high_ron['ron'] = 5.0

        sigma2_low = facs_slope_noise_variance_nm2(photon_flux_total=1e3, **kwargs_low_ron)
        sigma2_high = facs_slope_noise_variance_nm2(photon_flux_total=1e3, **kwargs_high_ron)
        self.assertGreater(sigma2_high, sigma2_low)


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


if __name__ == "__main__":
    unittest.main(verbosity=2)
