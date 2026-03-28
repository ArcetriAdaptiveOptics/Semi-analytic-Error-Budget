#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest
from unittest.mock import patch

import numpy as np

from src.Functions import (
    aliasing_psd_from_coeffs,
    build_transfer_function,
    build_transfer_function_from_controller_polynomials,
    compute_noise_PSD_intermediate,
    funct_d2,
    total_variance,
)
from src.controller_optimization import (
    SingleModeControllerOptimizationContext,
    prepare_single_mode_control_optimization,
)


class TestControllerPolynomialTransferFunction(unittest.TestCase):

    def test_integrator_polynomials_match_gain_builder(self):
        omega = np.logspace(-2, 2, 64)
        t_0 = 1.0 / 1000.0
        d2 = funct_d2(2)
        gain = np.array([0.35])

        plant_num = np.array([1.0])
        plant_den = d2

        H_r_gain, H_n_gain = build_transfer_function(
            omega, t_0, 1, plant_num, plant_den, gain=gain
        )

        controller_num = np.array([0.35, 0.0])
        controller_den = np.array([1.0, -1.0])

        H_r_iir, H_n_iir = build_transfer_function_from_controller_polynomials(
            controller_num, controller_den, omega, t_0, 1, plant_num, plant_den
        )

        np.testing.assert_allclose(H_r_gain, H_r_iir, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(H_n_gain, H_n_iir, rtol=1e-12, atol=1e-12)


class TestPrepareSingleModeControlOptimization(unittest.TestCase):

    @patch("src.controller_optimization.extract_propagation_coefficients", return_value=np.array([10.0, 20.0, 30.0]))
    @patch("src.controller_optimization.compute_slope_noise_variance", return_value=0.5)
    @patch("src.controller_optimization.k_coeff_aliasing", return_value=np.array([2.0, 4.0, 6.0]))
    def test_prepare_precomputes_single_mode_inputs(self, mock_k, mock_slope, mock_prop):
        omega = np.array([1.0, 2.0, 4.0])

        context = prepare_single_mode_control_optimization(
            mode_index=1,
            omega_temp_freq_interval=omega,
            t_0=0.001,
            PSD_atmo_turb=np.array([
                [1.0, 1.0, 1.0],
                [2.0, 2.0, 2.0],
                [3.0, 3.0, 3.0],
            ]),
            PSD_vibration=np.array([
                [0.1, 0.1, 0.1],
            ]),
            alpha=-1.0,
            telescope_diameter=8.0,
            seeing=0.8,
            modulation_radius=3.0,
            windspeed=10.0,
            maximum_radial_order_corrected=5,
            c_optg=2.0,
            F_excess=1.0,
            pixel_pos=[1.0, -1.0],
            sky_bkg=0.0,
            dark_curr=0.0,
            read_out_noise=0.0,
            photon_flux=1.0,
            frame_rate=1.0,
            magnitudo=0.0,
            n_subaperture=1,
            collecting_area=1.0,
            file_path_matrix_R="dummy.fits"
        )

        expected_alias = aliasing_psd_from_coeffs(
            1, omega, np.array([4.0]), -1.0, 8.0, 10.0, 5
        ) / (2.0 ** 2)
        expected_measurement = compute_noise_PSD_intermediate(
            omega, 1, np.array([20.0 * 0.5])
        ) / (2.0 ** 2)

        np.testing.assert_array_equal(context.PSD_input_atmos, np.array([[2.0, 2.0, 2.0]]))
        np.testing.assert_array_equal(context.PSD_input_vibration, np.zeros((1, 3)))
        np.testing.assert_allclose(context.PSD_input_alias, expected_alias)
        np.testing.assert_allclose(context.PSD_input_measurement, expected_measurement)
        mock_k.assert_called_once()
        mock_slope.assert_called_once()
        mock_prop.assert_called_once()


class TestSingleModeControllerOptimizationContext(unittest.TestCase):

    def test_evaluate_returns_cost_psds_and_history(self):
        omega = np.array([1.0, 2.0, 4.0, 8.0])
        context = SingleModeControllerOptimizationContext(
            mode_index=0,
            omega_temp_freq_interval=omega,
            t_0=0.001,
            plant_num=np.array([1.0]),
            plant_den=funct_d2(1),
            PSD_input_atmos=np.array([[1.0, 1.0, 1.0, 1.0]]),
            PSD_input_vibration=np.array([[0.2, 0.2, 0.2, 0.2]]),
            PSD_input_alias=np.array([[0.1, 0.1, 0.1, 0.1]]),
            PSD_input_measurement=np.array([[0.05, 0.05, 0.05, 0.05]]),
        )

        result = context.evaluate(
            controller_num=np.array([0.4, 0.0]),
            controller_den=np.array([1.0, -1.0]),
            store_history=True,
        )

        self.assertEqual(len(context.history), 1)
        self.assertAlmostEqual(context.history[0].cost, result.cost, places=12)
        self.assertEqual(result.psd_output["total"].shape, (1, 4))

        np.testing.assert_allclose(
            result.psd_output["total"],
            result.psd_output["temporal"] + result.psd_output["aliasing"] + result.psd_output["measurement"],
        )

        expected_total = total_variance(
            0.0,
            result.variance_terms["temporal"],
            result.variance_terms["aliasing"],
            result.variance_terms["measurement"],
        )
        self.assertAlmostEqual(result.cost, expected_total, places=12)

        _ = context.evaluate(
            controller_num=np.array([0.3, 0.0]),
            controller_den=np.array([1.0, -1.0]),
            store_history=False,
        )
        self.assertEqual(len(context.history), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
