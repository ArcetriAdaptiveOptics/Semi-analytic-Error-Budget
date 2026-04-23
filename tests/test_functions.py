#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for src/Functions.py.

Run from the repository root with either:
    python -m pytest test/test_functions.py -v
or:
    python -m unittest discover -s test
"""

import os
import sys
import unittest
from unittest.mock import patch

import numpy as np
from scipy import integrate as scipy_integrate

from src.Functions import (
    DEFAULT_ALIASING_ALPHA,
    _load_andes_gain_grid,
    aliasing_psd_from_coeffs,
    build_integrator_controller_polynomials,
    compute_k_prime,
    compute_noise_PSD_intermediate,
    compute_slope_noise_variance,
    flux_for_frame_for_pixel,
    funct_d2,
    fitting_variance,
    func_out,
    integrate_function,
    interpolate_vector,
    load_parameters,
    load_PSD_windshake,
    omega_0,
    PSD_final_alias,
    read_sigma_slopes,
    resize_psd_like,
    seeing_to_r0,
    total_variance,
    transfer_funct,
)

# Absolute path to the repository root (parent of this test/ directory)
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


# ---------------------------------------------------------------------------
# funct_d2
# ---------------------------------------------------------------------------

class TestFunctD2(unittest.TestCase):
    """funct_d2 builds the delay-block denominator polynomial d2 = z^T_total."""

    def test_length_equals_T_total_plus_one(self):
        # The polynomial z^n has n+1 coefficients
        for n in range(1, 6):
            self.assertEqual(len(funct_d2(n)), n + 1)

    def test_leading_coefficient_is_one(self):
        for n in range(1, 5):
            self.assertEqual(funct_d2(n)[0], 1.0)

    def test_remaining_coefficients_are_zero(self):
        # All coefficients below the leading one must be zero (pure delay)
        result = funct_d2(4)
        np.testing.assert_array_equal(result[1:], np.zeros(4))

    def test_returns_ndarray(self):
        self.assertIsInstance(funct_d2(3), np.ndarray)


# ---------------------------------------------------------------------------
# fitting_variance
# ---------------------------------------------------------------------------

class TestFittingVariance(unittest.TestCase):
    """fitting_variance = coeff * N_act^(-0.9) * (D/r0)^(5/3)  [Eq. 7]."""

    @staticmethod
    def _formula(coeff, n_act, D, r0):
        return coeff * n_act ** (-0.9) * (D / r0) ** (5 / 3)

    def test_unit_inputs_return_coefficient(self):
        # With N=1, D=1, r0=1 the formula reduces to just 'coeff'
        for coeff in (0.1, 0.2778, 1.0):
            self.assertAlmostEqual(
                fitting_variance(coeff, 1, 1.0, 1.0), coeff, places=12
            )

    def test_matches_analytical_formula(self):
        cases = [
            (0.2778, 2, 38.5, 0.16),
            (0.2778, 1, 10.0, 0.20),
            (0.5,   4,  8.0, 0.12),
        ]
        for coeff, n_act, D, r0 in cases:
            with self.subTest(params=(coeff, n_act, D, r0)):
                self.assertAlmostEqual(
                    fitting_variance(coeff, n_act, D, r0),
                    self._formula(coeff, n_act, D, r0),
                    places=10,
                )

    def test_positive_for_physical_inputs(self):
        self.assertGreater(fitting_variance(0.2778, 2, 38.5, 0.16), 0)


# ---------------------------------------------------------------------------
# func_out
# ---------------------------------------------------------------------------

class TestFuncOut(unittest.TestCase):
    """func_out computes output PSD = |H|^2 * PSD_in  [integrand in Eqs. 8/10/15]."""

    def test_identity_transfer_function(self):
        # |H|^2 = 1 everywhere → output equals input
        H = np.ones(5, dtype=complex)
        PSD = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        np.testing.assert_array_almost_equal(func_out(H, PSD), PSD)

    def test_zero_transfer_function(self):
        H = np.zeros(5, dtype=complex)
        PSD = np.ones(5)
        np.testing.assert_array_almost_equal(func_out(H, PSD), np.zeros(5))

    def test_modulus_squared_applied_correctly(self):
        # |1+1j|^2 = 2, so output must be 2 * PSD_in
        H = np.array([1 + 1j, 1 + 1j])
        PSD = np.array([3.0, 4.0])
        np.testing.assert_array_almost_equal(func_out(H, PSD), 2 * PSD)

    def test_output_has_no_imaginary_part(self):
        # |H|^2 is real; multiplied by a real PSD the result must be real
        H = np.array([0.5 + 0.5j, 1.0 + 2.0j])
        PSD = np.array([1.0, 2.0])
        np.testing.assert_array_almost_equal(np.imag(func_out(H, PSD)), [0.0, 0.0])


# ---------------------------------------------------------------------------
# seeing_to_r0
# ---------------------------------------------------------------------------

class TestSeeingToR0(unittest.TestCase):
    """seeing_to_r0 converts seeing in arcsec to the Fried parameter r0 in meters."""

    def test_typical_seeing_value(self):
        # A seeing of 1.0 arcsec at 500 nm corresponds to about 0.1006 m
        # (using r0 = 0.976 * lambda / seeing_rad)
        r0 = seeing_to_r0(1.0, wavelength=500e-9)
        self.assertAlmostEqual(r0, 0.10064, places=4)

    def test_scales_inversely_with_seeing(self):
        # Doubling the seeing should halve r0
        r0_good_seeing = seeing_to_r0(0.5)
        r0_bad_seeing = seeing_to_r0(1.0)
        self.assertAlmostEqual(r0_good_seeing / r0_bad_seeing, 2.0, places=10)

    def test_scales_linearly_with_wavelength(self):
        # Doubling the wavelength should double r0
        r0_500nm = seeing_to_r0(1.0, wavelength=500e-9)
        r0_1000nm = seeing_to_r0(1.0, wavelength=1000e-9)
        self.assertAlmostEqual(r0_1000nm / r0_500nm, 2.0, places=10)
        
    def test_positive_value(self):
        self.assertGreater(seeing_to_r0(0.8), 0.0)


# ---------------------------------------------------------------------------
# integrate_function
# ---------------------------------------------------------------------------

class TestIntegrateFunction(unittest.TestCase):
    """integrate_function wraps scipy.integrate.simpson."""

    def test_constant_over_unit_interval(self):
        # ∫_0^1 1 dx = 1
        x = np.linspace(0, 1, 1001)
        self.assertAlmostEqual(integrate_function(np.ones_like(x), x), 1.0, places=6)

    def test_linear_function_over_unit_interval(self):
        # ∫_0^1 x dx = 0.5
        x = np.linspace(0, 1, 1001)
        self.assertAlmostEqual(integrate_function(x, x), 0.5, places=6)

    def test_quadratic_function(self):
        # ∫_0^1 x^2 dx = 1/3
        x = np.linspace(0, 1, 10001)
        self.assertAlmostEqual(integrate_function(x ** 2, x), 1 / 3, places=6)

    def test_result_equals_scipy_simpson_directly(self):
        x = np.linspace(0.1, 10.0, 5001)
        f = np.sin(x) ** 2
        self.assertAlmostEqual(
            integrate_function(f, x),
            scipy_integrate.simpson(f, x),
            places=12,
        )


# ---------------------------------------------------------------------------
# resize_psd_like
# ---------------------------------------------------------------------------

class TestResizePsdLike(unittest.TestCase):
    """resize_psd_like zero-pads a vibration PSD to match the atmospheric PSD shape."""

    def test_output_shape_matches_atmo(self):
        atmo = np.ones((5, 100))
        vib  = np.ones((2, 100))
        self.assertEqual(resize_psd_like(atmo, vib).shape, atmo.shape)

    def test_first_rows_are_copied(self):
        atmo = np.zeros((4, 10))
        vib  = np.arange(20).reshape(2, 10).astype(float)
        result = resize_psd_like(atmo, vib)
        np.testing.assert_array_equal(result[:2, :], vib)

    def test_extra_rows_are_zero(self):
        atmo = np.zeros((4, 10))
        vib  = np.ones((2, 10))
        result = resize_psd_like(atmo, vib)
        np.testing.assert_array_equal(result[2:, :], np.zeros((2, 10)))

    def test_same_number_of_rows_is_identity(self):
        atmo = np.zeros((3, 10))
        vib  = np.arange(30).reshape(3, 10).astype(float)
        np.testing.assert_array_equal(resize_psd_like(atmo, vib), vib)


# ---------------------------------------------------------------------------
# omega_0
# ---------------------------------------------------------------------------

class TestOmega0(unittest.TestCase):
    """omega_0 = 2*pi*(N_max+1)*v/D  (AO temporal bandwidth)."""

    @staticmethod
    def _formula(D, v, N):
        return 2 * np.pi * (N + 1) * v / D

    def test_matches_formula(self):
        cases = [(38.5, 8.0, 88), (8.0, 10.0, 20), (1.0, 1.0, 0)]
        for D, v, N in cases:
            with self.subTest(D=D, v=v, N=N):
                self.assertAlmostEqual(omega_0(D, v, N), self._formula(D, v, N), places=12)

    def test_positive(self):
        self.assertGreater(omega_0(38.5, 8.0, 88), 0)

    def test_scales_linearly_with_windspeed(self):
        # Doubling windspeed should double omega_0
        self.assertAlmostEqual(
            omega_0(38.5, 16.0, 88) / omega_0(38.5, 8.0, 88), 2.0, places=12
        )


# ---------------------------------------------------------------------------
# total_variance
# ---------------------------------------------------------------------------

class TestTotalVariance(unittest.TestCase):
    """total_variance sums the real parts of the four error contributions."""

    def test_integer_sum(self):
        self.assertAlmostEqual(total_variance(1, 2, 3, 4), 10.0, places=12)

    def test_float_sum(self):
        self.assertAlmostEqual(total_variance(0.1, 0.2, 0.3, 0.4), 1.0, places=12)

    def test_imaginary_parts_are_discarded(self):
        # The function must take np.real() of each term before summing
        result = total_variance(1 + 5j, 2 + 3j, 3 + 1j, 4 + 2j)
        self.assertAlmostEqual(result, 10.0, places=12)

    def test_all_zero(self):
        self.assertAlmostEqual(total_variance(0, 0, 0, 0), 0.0, places=12)


# ---------------------------------------------------------------------------
# build_integrator_controller_polynomials
# ---------------------------------------------------------------------------

class TestBuildIntegratorControllerPolynomials(unittest.TestCase):
    """
    build_integrator_controller_polynomials builds the integral controller
    C = g*Z/(Z-1) in polynomial form.
    """

    def test_numerator_polynomial_coefficients(self):
        # Numerator of g*Z/(Z-1) is g*Z: polynomial coefficients [g, 0]
        gain = 0.5
        num_int, _ = build_integrator_controller_polynomials(gain)
        np.testing.assert_array_almost_equal(np.real(num_int), [gain, 0.0], decimal=12)

    def test_denominator_polynomial_coefficients(self):
        # Denominator of g*Z/(Z-1) is (Z-1): polynomial coefficients [1, -1]
        _, den_int = build_integrator_controller_polynomials(1.0)
        np.testing.assert_array_almost_equal(np.real(den_int), [1.0, -1.0], decimal=12)

    def test_numerator_scales_with_gain(self):
        # Doubling the gain doubles the leading numerator coefficient
        num_int_1, _ = build_integrator_controller_polynomials(1.0)
        num_int_2, _ = build_integrator_controller_polynomials(2.0)
        self.assertAlmostEqual(
            np.real(num_int_2[0]) / np.real(num_int_1[0]), 2.0, places=12
        )


# ---------------------------------------------------------------------------
# transfer_funct
# ---------------------------------------------------------------------------

class TestTransferFunct(unittest.TestCase):
    """transfer_funct evaluates H_r and H_n from Eqs. (4)-(5) of Agapito & Pinna (2019).

    With integral control (C = g*Z/(Z-1)) and trivial W, R, M polynomials ([1]):
      - At DC (Z=1): H_n = 1  (noise fully propagated)
      - At DC (Z=1): H_r = 0  (perfect rejection via integral action)
    """

    @staticmethod
    def _dc_inputs(gain=0.5):
        """Return polynomial arrays evaluated at DC (omega=0)."""
        plant_num = np.array([1.0])   # n1*n2*n3 = 1
        plant_den = funct_d2(1)        # d1*d2*d3 = delay polynomial
        omega_dc = np.array([0.0])
        num_int, den_int = build_integrator_controller_polynomials(gain)
        Z = np.exp(1j * omega_dc * 0.001)
        return plant_num, num_int, plant_den, den_int, Z

    def test_H_n_equals_one_at_DC(self):
        # Integral control has unity DC gain on the noise path
        plant_num, num_int, plant_den, den_int, Z = self._dc_inputs(gain=0.5)
        _, H_n = transfer_funct(plant_num, num_int, plant_den, den_int, Z)
        self.assertAlmostEqual(abs(H_n[0]), 1.0, places=6)

    def test_H_r_equals_zero_at_DC(self):
        # Integral control perfectly rejects a DC disturbance
        plant_num, num_int, plant_den, den_int, Z = self._dc_inputs(gain=0.5)
        H_r, _ = transfer_funct(plant_num, num_int, plant_den, den_int, Z)
        self.assertAlmostEqual(abs(H_r[0]), 0.0, places=6)

    def test_output_length_matches_frequency_vector(self):
        n_freq = 50
        omega = np.linspace(0.01, 100.0, n_freq)
        num_int, den_int = build_integrator_controller_polynomials(0.3)
        Z = np.exp(1j * omega * 0.001)
        plant_num = np.array([1.0])
        plant_den = funct_d2(1)
        H_r, _ = transfer_funct(plant_num, num_int, plant_den, den_int, Z)
        self.assertEqual(len(H_r), n_freq)


# ---------------------------------------------------------------------------
# interpolate_vector
# ---------------------------------------------------------------------------

class TestInterpolateVector(unittest.TestCase):
    """interpolate_vector: linear interpolation, zeros outside the original range."""

    def test_identity_on_same_grid(self):
        x = np.linspace(0, 1, 10)
        y = x ** 2
        np.testing.assert_array_almost_equal(interpolate_vector(x, x, y), y, decimal=12)

    def test_out_of_range_returns_zero(self):
        x_orig = np.array([1.0, 2.0, 3.0])
        y_orig = np.array([1.0, 4.0, 9.0])
        x_new  = np.array([0.0, 4.0])   # both outside [1, 3]
        np.testing.assert_array_equal(interpolate_vector(x_new, x_orig, y_orig), [0.0, 0.0])

    def test_midpoint_interpolation(self):
        x_orig = np.array([0.0, 1.0])
        y_orig = np.array([0.0, 10.0])
        result = interpolate_vector(np.array([0.5]), x_orig, y_orig)
        self.assertAlmostEqual(result[0], 5.0, places=12)


# ---------------------------------------------------------------------------
# compute_noise_PSD
# ---------------------------------------------------------------------------

class TestComputeNoisePSD(unittest.TestCase):
    """compute_noise_PSD builds a flat (white) noise PSD for each mode."""

    def test_flat_value_equals_variance_over_bandwidth(self):
        omega     = np.linspace(1.0, 10.0, 200)
        bandwidth = 10.0 - 1.0
        sigma2_w  = np.array([1.0, 4.0])
        PSD_w     = compute_noise_PSD_intermediate(omega, 2, sigma2_w)
        for i, s2 in enumerate(sigma2_w):
            np.testing.assert_array_almost_equal(
                PSD_w[i, :], np.full(200, s2 / bandwidth)
            )

    def test_output_shape(self):
        omega    = np.linspace(1.0, 10.0, 50)
        PSD_w    = compute_noise_PSD_intermediate(omega, 3, np.ones(3))
        self.assertEqual(PSD_w.shape, (3, 50))

    def test_integral_recovers_variance(self):
        # ∫ PSD_w dω over the bandwidth must equal sigma2_w
        omega    = np.linspace(1.0, 10.0, 100001)
        sigma2_w = np.array([3.0])
        PSD_w    = compute_noise_PSD_intermediate(omega, 1, sigma2_w)
        recovered = scipy_integrate.simpson(PSD_w[0, :], omega)
        self.assertAlmostEqual(recovered, sigma2_w[0], places=4)


# ---------------------------------------------------------------------------
# flux_for_frame_for_pixel
# ---------------------------------------------------------------------------

class TestFluxForFrameForPixel(unittest.TestCase):
    """flux_for_frame_for_pixel computes the photon count per pixel per frame."""

    @staticmethod
    def _formula(flux, D, fr, mag, n_sub, area):
        sub_r    = (D / n_sub) / 2
        sub_area = np.pi * sub_r ** 2
        return (flux / fr) * (sub_area / area) * 10 ** (-mag / 2.5) / 4

    def test_matches_analytical_formula(self):
        cases = [
            (9e12, 38.5, 1000, 7,  100, 1000),
            (1e10,  8.0,  500, 10,  40,  500),
        ]
        for flux, D, fr, mag, n_sub, area in cases:
            with self.subTest(mag=mag):
                self.assertAlmostEqual(
                    flux_for_frame_for_pixel(flux, D, fr, mag, n_sub, area),
                    self._formula(flux, D, fr, mag, n_sub, area),
                    places=10,
                )

    def test_brighter_star_gives_higher_flux(self):
        kw = dict(photon_flux=9e12, telescope_diameter=38.5, frame_rate=1000,
                  n_subaperture=100, collecting_area=1000)
        self.assertGreater(
            flux_for_frame_for_pixel(**kw, magnitudo=5),
            flux_for_frame_for_pixel(**kw, magnitudo=10),
        )

    def test_result_is_positive(self):
        self.assertGreater(
            flux_for_frame_for_pixel(9e12, 38.5, 1000, 7, 100, 1000), 0
        )


# ---------------------------------------------------------------------------
# compute_slope_noise_variance
# ---------------------------------------------------------------------------

class TestComputeSlopeNoiseVariance(unittest.TestCase):
    """compute_slope_noise_variance computes sigma^2_slope  [Eqs. 12, 14]."""

    def test_pure_photon_noise_case(self):
        # With sky=dark=RON=0 and F=1:
        #   pixel_variance = n_phot
        #   slope_var = sum(x_i^2 * n_phot) / (4*n_phot)^2 = sum(x_i^2) / (16*n_phot)
        pixel_pos = np.array([1.0, 0.0, 0.0, 0.0])
        n_phot = flux_for_frame_for_pixel(9e12, 38.5, 1000, 7, 100, 1000)
        expected = np.sum(pixel_pos ** 2) / n_phot    
        result = compute_slope_noise_variance(
            1.0, pixel_pos, 0, 0, 0, 9e12, 38.5, 1000, 7, 100, 1000
        )
        self.assertAlmostEqual(result, expected, places=10)

    def test_result_is_positive(self):
        result = compute_slope_noise_variance(
            1.0, [1, 0, 0, 0], 0, 0, 5, 9e12, 38.5, 1000, 7, 100, 1000
        )
        self.assertGreater(result, 0)


# ---------------------------------------------------------------------------
# compute_k_prime
# ---------------------------------------------------------------------------

class TestComputeKPrime(unittest.TestCase):
    """compute_k_prime computes the aliasing normalisation coefficient k'."""

    @staticmethod
    def _formula(omega, alpha, sigma_slope, c, D, v, N):
        w0      = 2 * np.pi * (N + 1) * v / D
        om_min  = np.min(omega)
        om_max  = np.max(omega)
        integral = (
            w0 ** alpha * (w0 - om_min)
            + (om_max ** (alpha + 1) - w0 ** (alpha + 1)) / (alpha + 1)
        )
        return sigma_slope ** 2 / (c ** 2 * integral)


class TestAliasingDefaultsAndOpticalGain(unittest.TestCase):
    """Regression tests for aliasing defaults and optical-gain broadcasting."""

    @staticmethod
    def _formula(omega, alpha, sigma_slope, c, D, v, N):
        w0      = 2 * np.pi * (N + 1) * v / D
        om_min  = np.min(omega)
        om_max  = np.max(omega)
        integral = (
            w0 ** alpha * (w0 - om_min)
            + (om_max ** (alpha + 1) - w0 ** (alpha + 1)) / (alpha + 1)
        )
        return sigma_slope ** 2 / (c ** 2 * integral)

    @patch("src.Functions.k_coeff_aliasing", return_value=np.array([1.0]))
    def test_psd_final_alias_defaults_alpha_to_minus_17_over_3(self, mock_k):
        omega = np.array([1.0, 2.0, 4.0])

        PSD_final_alias(
            c_optg=2.0,
            actuators_number=1,
            omega_temp_freq_interval=omega,
            telescope_diameter=8.0,
            seeing=0.8,
            modulation_radius=3.0,
            windspeed=10.0,
            maximum_radial_order_corrected=5,
            file_path_matrix_R="dummy.fits",
        )

        self.assertEqual(mock_k.call_args.kwargs["alpha"], DEFAULT_ALIASING_ALPHA)

    @patch("src.Functions.k_coeff_aliasing", return_value=np.array([1.0, 4.0]))
    def test_psd_final_alias_accepts_row_vector_optical_gain(self, mock_k):
        omega = np.array([1.0, 2.0, 4.0])
        c_optg = np.array([[2.0, 4.0, 8.0, 16.0]])

        result = PSD_final_alias(
            c_optg=c_optg,
            actuators_number=2,
            omega_temp_freq_interval=omega,
            telescope_diameter=8.0,
            seeing=0.8,
            modulation_radius=3.0,
            windspeed=10.0,
            maximum_radial_order_corrected=5,
            file_path_matrix_R="dummy.fits",
            alpha=-1.0,
        )

        expected = aliasing_psd_from_coeffs(
            2,
            omega,
            np.array([1.0, 4.0]),
            alpha=-1.0,
            telescope_diameter=8.0,
            windspeed=10.0,
            maximum_radial_order_corrected=5,
        ) / np.array([[2.0], [4.0]]) ** 2

        np.testing.assert_allclose(result, expected)
        mock_k.assert_called_once()

    def test_matches_analytical_formula(self):
        omega = np.logspace(0, 3, 500)   # 1 … 1000 rad/s
        alpha, sigma, c = -17 / 3, 0.01, 0.8
        D, v, N = 38.5, 8.0, 88
        expected = self._formula(omega, alpha, sigma, c, D, v, N)
        self.assertAlmostEqual(
            compute_k_prime(omega, sigma, D, v, N, alpha=alpha) / c**2,
            expected,
            places=10,
        )

    def test_result_is_positive(self):
        omega = np.logspace(0, 3, 200)
        self.assertGreater(
            compute_k_prime(omega, 0.01, 38.5, 8.0, 88, alpha=-17 / 3),
            0,
        )


# ---------------------------------------------------------------------------
# Tests that require files already present in the repository
# ---------------------------------------------------------------------------

class TestLoadParameters(unittest.TestCase):
    """load_parameters reads the YAML configuration file."""

    def setUp(self):
        self.yaml_path = os.path.join(REPO_ROOT, "params_mod4.yaml")

    def test_returns_dict(self):
        self.assertIsInstance(load_parameters(self.yaml_path), dict)

    def test_expected_top_level_keys_present(self):
        params = load_parameters(self.yaml_path)
        for key in ("telescope", "atmosphere", "wavefront_sensor",
                    "data", "frequency_ranges", "control"):
            self.assertIn(key, params)

    def test_telescope_diameter_value(self):
        params = load_parameters(self.yaml_path)
        self.assertAlmostEqual(params["telescope"]["telescope_diam"], 38.5)

    def test_number_of_actuators(self):
        params = load_parameters(self.yaml_path)
        self.assertIsInstance(params["control"]["n_modes"], int)


class _ChdirMixin:
    """Mixin that changes cwd to REPO_ROOT before each test and restores it after."""

    def setUp(self):
        self._orig_dir = os.getcwd()
        os.chdir(REPO_ROOT)

    def tearDown(self):
        os.chdir(self._orig_dir)


class TestBuildOpticalGainGrid(_ChdirMixin, unittest.TestCase):
    """_load_andes_gain_grid assembles a 2×N array from ANDES FITS files."""

    def test_output_is_3d_ndarray(self):
        file_mod0 = "src/file_fits/ANDES/ANDES_og_mod0.fits"
        file_mod4 = "src/file_fits/ANDES/ANDES_og_mod4.fits"
        grid, _, _ = _load_andes_gain_grid(file_mod0, file_mod4)
        self.assertIsInstance(grid, np.ndarray)
        self.assertEqual(grid.ndim, 3)

    def test_two_rows_one_per_modulation_radius(self):
        # One row for mod0, one for mod4
        file_mod0 = "src/file_fits/ANDES/ANDES_og_mod0.fits"
        file_mod4 = "src/file_fits/ANDES/ANDES_og_mod4.fits"
        grid, _, _ = _load_andes_gain_grid(file_mod0, file_mod4)
        self.assertEqual(grid.shape[0], 2)

    def test_all_values_are_positive(self):
        file_mod0 = "src/file_fits/ANDES/ANDES_og_mod0.fits"
        file_mod4 = "src/file_fits/ANDES/ANDES_og_mod4.fits"
        grid, _, _ = _load_andes_gain_grid(file_mod0, file_mod4)
        self.assertTrue(np.all(grid > 0))


class TestReadSigmaSlopes(_ChdirMixin, unittest.TestCase):
    """read_sigma_slopes reads slope RMS data from FITS."""

    FILE = "src/file_fits/ANDES/slopes_rms_time_avg_all.fits"

    def test_returns_ndarray_and_axes(self):
        """Verifies that the function returns three ndarrays (data, seeing_axis, mod_radius_axis)."""
        data, seeing_vals, modal_radius_vals = read_sigma_slopes(self.FILE)
        self.assertIsInstance(data, np.ndarray)
        self.assertIsInstance(seeing_vals, np.ndarray)
        self.assertIsInstance(modal_radius_vals, np.ndarray)

    def test_non_negative_values(self):
        """Verifies that the extracted slope RMS data contains only non-negative values."""
        data, _, _ = read_sigma_slopes(self.FILE)
        self.assertTrue(np.all(data >= 0))


class TestLoadPSDWindshake(_ChdirMixin, unittest.TestCase):
    """load_PSD_windshake reads the windshake tip/tilt PSD from a FITS file."""

    FILE = "src/file_fits/ANDES/morfeo_windshake8ms_psd_2022_1k.fits"

    def test_returns_two_arrays(self):
        freq, psd = load_PSD_windshake(self.FILE)
        self.assertIsNotNone(freq)
        self.assertIsNotNone(psd)

    def test_frequencies_are_positive(self):
        freq, _ = load_PSD_windshake(self.FILE)
        self.assertTrue(np.all(freq > 0))

    def test_psd_has_two_modes_tip_and_tilt(self):
        freq, psd = load_PSD_windshake(self.FILE)
        self.assertEqual(psd.shape[0], 2)

    def test_psd_and_frequency_lengths_match(self):
        freq, psd = load_PSD_windshake(self.FILE)
        self.assertEqual(psd.shape[1], len(freq))
