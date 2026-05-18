import importlib.util
import os
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
MAIN_SAEB_PATH = REPO_ROOT / "scripts" / "main_saeb.py"


def _load_main_saeb_module():
    spec = importlib.util.spec_from_file_location("main_saeb_test_module", MAIN_SAEB_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class TestMainSaebBlockOptimization(unittest.TestCase):

    def test_run_loads_psd_before_block_optimization(self):
        main_saeb = _load_main_saeb_module()

        fake_param = {
            "control": {
                "n_modes": 4,
                "sampling_time": 0.01,
                "gain_mode": "block_optimization",
                "gain_min": 0.1,
                "gain_block_sizes": [2, None],
                "slope_computer_weights": [1, 0, 0, 0],
                "bin": 1,
            },
            "telescope": {
                "telescope_diam": 8.0,
            },
            "atmosphere": {
                "outer_scale": 25.0,
                "wind_speed": 8.0,
                "seeing": 0.8,
            },
            "wavefront_sensor": {
                "value_for_F_excess_noise": 2.0,
                "sky_backgr": 0.0,
                "dark_curr": 0.0,
                "noise_readout": 0.0,
                "number_of_sub": 10,
                "modulation_radius": 3.0,
            },
            "guide_star": {
                "flux_photons": 1.0,
                "magn": 0.0,
            },
            "data": {
                "reconstruction_matrix": "dummy.fits",
                "windshake_psd": "dummy_wind.fits",
                "sigma_slopes": "dummy_sigma.fits",
                "optical_gain_models": ["og0.fits", "og4.fits"],
            },
            "plant": {
                "numerator": [1.0],
                "denominator": [1.0],
                "total_delay": 3,
            },
            "frequency_ranges": {
                "temporal_freqs_min": -3,
                "temporal_freqs_n": 4,
            },
            "display": {
                "enabled": False,
                "summary_modes_to_plot": None,
            },
        }

        call_order = []

        def fake_load_parameters(_):
            call_order.append("load_parameters")
            return fake_param

        def fake_load_psd_windshake(_):
            call_order.append("load_psd_windshake")
            freq = np.array([1.0, 2.0, 4.0, 8.0])
            psd = np.ones((4, 4))
            return freq, psd

        def fake_optimize_gain_blocks(*args, **kwargs):
            call_order.append("optimize_gain_blocks")
            self.assertIn("load_psd_windshake", call_order)
            return np.full(4, 0.2), [
                {"label": "Block 1", "gain_values": np.array([0.1, 0.2]), "variances": np.array([3.0, 2.0])},
                {"label": "Block 2", "gain_values": np.array([0.1, 0.2]), "variances": np.array([2.0, 1.0])},
            ]

        def fake_funct_d2(total_delay):
            self.assertEqual(total_delay, 3)
            return np.array([1.0])

        with patch.object(main_saeb, "load_parameters", side_effect=fake_load_parameters), \
             patch.object(main_saeb, "resolve_binning_config", side_effect=lambda param: param), \
             patch.object(main_saeb, "load_PSD_windshake", side_effect=fake_load_psd_windshake), \
             patch.object(main_saeb, "compute_optical_gain", return_value=np.ones(4)), \
             patch.object(main_saeb, "funct_d2", side_effect=fake_funct_d2), \
             patch.object(main_saeb, "turbulence_psd", return_value=np.ones((4, 4))), \
             patch.object(main_saeb, "build_transfer_function", return_value=(np.ones((4, 4)), np.ones((4, 4)))), \
             patch.object(main_saeb, "fitting_variance", return_value=1.0), \
             patch.object(main_saeb, "temporal_variance", return_value=(1.0, 1.0, np.ones((4, 4)), np.ones((4, 4)))), \
             patch.object(main_saeb, "vibration_variance", return_value=(1.0, 1.0, np.ones((4, 4)), np.ones((4, 4)))), \
             patch.object(main_saeb, "aliasing_variance", return_value=(1.0, 1.0, np.ones((4, 4)), np.ones((4, 4)))), \
             patch.object(main_saeb, "measure_variance", return_value=(1.0, 1.0, np.ones((4, 4)), np.ones((4, 4)))), \
             patch.object(main_saeb, "total_variance", return_value=1.0), \
             patch.object(main_saeb, "summary_display") as mock_summary, \
             patch.object(main_saeb, "plot_gain_optimization_sweep") as mock_plot_sweep, \
             patch.object(main_saeb, "optimize_gain_blocks", side_effect=fake_optimize_gain_blocks):
            main_saeb.run("params_ANDES.yaml")

        self.assertEqual(call_order[0], "load_parameters")
        self.assertIn("load_psd_windshake", call_order)
        self.assertIn("optimize_gain_blocks", call_order)
        mock_summary.assert_not_called()
        mock_plot_sweep.assert_not_called()
