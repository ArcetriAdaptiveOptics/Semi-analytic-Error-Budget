import unittest
from unittest.mock import patch

import numpy as np

from src.Functions import gain_maximum_from_total_delay
from src.Functions import normalize_gain_block_sizes
from src.Functions import optimize_gain_blocks
from src.Functions import resolve_gain_mode


class TestGainOptimizationHelpers(unittest.TestCase):

    def test_normalize_gain_block_sizes_validates_total(self):
        self.assertEqual(normalize_gain_block_sizes([2, 3], 5), [2, 3])
        self.assertEqual(normalize_gain_block_sizes([2, None], 5), [2, 3])
        self.assertEqual(normalize_gain_block_sizes([2, -1], 5), [2, 3])

        with self.assertRaises(ValueError):
            normalize_gain_block_sizes([2, 2], 5)

    def test_gain_maximum_from_total_delay(self):
        self.assertEqual(gain_maximum_from_total_delay(3), 0.6)

    def test_resolve_gain_mode_prefers_explicit_value(self):
        self.assertEqual(resolve_gain_mode({"gain_mode": "fixed"}), "fixed")
        self.assertEqual(resolve_gain_mode({"gain_block_sizes": [2, 3]}), "block_optimization")
        self.assertEqual(resolve_gain_mode({"gain_value": 0.01}), "fixed")

    @patch('src.Functions._compute_modal_variance_grid')
    def test_optimize_gain_blocks_runs_block_by_block(self, mock_modal_grid):
        # Mock the modal variance grid: shape (n_modes=5, n_gains=4)
        # Create simple variances that favor different gains for different mode groups
        gain_values = np.array([0.1, 0.15, 0.2, 0.25])
        modal_variances = np.array([
            [3.0, 2.0, 1.5, 2.0],  # mode 0: minimum at gain 0.2
            [3.0, 2.0, 1.5, 2.0],  # mode 1: minimum at gain 0.2 (same block)
            [1.0, 1.2, 3.0, 2.0],  # mode 2: minimum at gain 0.1
            [1.0, 1.2, 3.0, 2.0],  # mode 3: minimum at gain 0.1 (same block as 2)
            [1.0, 1.2, 3.0, 2.0],  # mode 4: minimum at gain 0.1 (same block as 2-3)
        ])
        mock_modal_grid.return_value = (gain_values, modal_variances)

        gain_vector, sweep_results = optimize_gain_blocks(
            gain_min=0.1,
            gain_max=0.3,
            omega_temp_freq_interval=np.array([1.0]),
            t_freqs=np.array([1.0]),
            f=np.array([1.0]),
            t_0=0.001,
            plant_num=np.array([1.0]),
            plant_den=np.array([1.0]),
            telescope_diameter=8.0,
            fried_parameter=0.15,
            excess_noise_factor=1.0,
            sky_background=0.0,
            dark_current=0.0,
            readout_noise=0.0,
            photon_flux=1000.0,
            frame_rate=1000.0,
            magnitude=0.0,
            n_subaperture=40,
            slope_computer_weights=np.array([1.0]),
            fitting_coeff=0.27,
            alpha=-17 / 3,
            seeing=0.8,
            modulation_radius=3.0,
            wind_speed=15.0,
            maximum_radial_order_corrected=10,
            reconstruction_matrix_path='dummy.fits',
            psd_turbulence=np.zeros((5, 1)),
            psd_windshake=np.zeros((5, 1)),
            sigma_slopes_path='dummy.fits',
            c_optg=np.array([1.0]),
            actuators_number=5,
            gain_block_sizes=[2, 3],
        )

        self.assertEqual(gain_vector.shape, (5,))
        self.assertEqual(len(sweep_results), 2)
        self.assertAlmostEqual(gain_vector[0], gain_vector[1])
        self.assertAlmostEqual(gain_vector[2], gain_vector[3])
        self.assertAlmostEqual(gain_vector[3], gain_vector[4])
        self.assertEqual(sweep_results[0]["label"], "Block 1 (modes 0-1)")
        self.assertEqual(sweep_results[1]["label"], "Block 2 (modes 2-4)")


class TestModalGainEdgeCases(unittest.TestCase):
    """
    Edge-case tests for modal gain optimization.

    The modal variance per mode is:
        V(g) = V_fit (constant) + V_temporal(g) + V_aliasing(g) + V_measurement(g)

    Physical intuition:
    - V_temporal decreases with gain (better correction) up to stability limit.
    - V_aliasing and V_measurement increase with gain (noise / aliasing propagation).

    Edge cases:
    - Zero noise+aliasing:  V(g) decreasing → best gain = gain_max (last element).
    - Zero temporal:        V(g) increasing → best gain = gain_min (first element).
    """

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    @staticmethod
    def _make_kwargs(n_modes, gain_block_sizes):
        """Minimal keyword arguments for optimize_gain_blocks."""
        return dict(
            gain_min=0.1,
            gain_max=0.5,
            omega_temp_freq_interval=np.array([1.0]),
            t_freqs=np.array([1.0]),
            f=np.array([1.0]),
            t_0=0.001,
            plant_num=np.array([1.0]),
            plant_den=np.array([1.0]),
            telescope_diameter=8.0,
            fried_parameter=0.15,
            excess_noise_factor=1.0,
            sky_background=0.0,
            dark_current=0.0,
            readout_noise=0.0,
            photon_flux=1000.0,
            frame_rate=1000.0,
            magnitude=0.0,
            n_subaperture=40,
            slope_computer_weights=np.array([1.0]),
            fitting_coeff=0.27,
            alpha=-17 / 3,
            seeing=0.8,
            modulation_radius=3.0,
            wind_speed=15.0,
            maximum_radial_order_corrected=10,
            reconstruction_matrix_path='dummy.fits',
            psd_turbulence=np.zeros((n_modes, 1)),
            psd_windshake=np.zeros((n_modes, 1)),
            sigma_slopes_path='dummy.fits',
            c_optg=np.array([1.0]),
            actuators_number=n_modes,
            gain_block_sizes=gain_block_sizes,
        )

    # -----------------------------------------------------------------------
    # Edge case 1 – noise/aliasing-free modes → maximum gain selected
    # -----------------------------------------------------------------------

    @patch('src.Functions._compute_modal_variance_grid')
    def test_noisefree_modes_select_maximum_gain(self, mock_grid):
        """
        Two modes whose variance decreases monotonically (no noise / aliasing).
        The optimizer should pick the highest gain value.
        """
        gain_values = np.array([0.1, 0.2, 0.3, 0.4])
        # Variance falls as gain rises — simulate pure temporal correction.
        modal_variances = np.array([
            [10.0, 7.0, 4.0, 1.0],  # mode 0: decreasing
            [12.0, 9.0, 5.0, 2.0],  # mode 1: decreasing
        ])
        mock_grid.return_value = (gain_values, modal_variances)

        gain_vector, sweep_results = optimize_gain_blocks(
            **self._make_kwargs(n_modes=2, gain_block_sizes=[2])
        )

        # Both modes in one block → gain should be gain_values[-1] = 0.4
        self.assertAlmostEqual(gain_vector[0], 0.4)
        self.assertAlmostEqual(gain_vector[1], 0.4)
        self.assertEqual(sweep_results[0]["best_gain"], 0.4)

    # -----------------------------------------------------------------------
    # Edge case 2 – temporal-free modes → minimum gain selected
    # -----------------------------------------------------------------------

    @patch('src.Functions._compute_modal_variance_grid')
    def test_temporalfree_modes_select_minimum_gain(self, mock_grid):
        """
        Two modes whose variance increases monotonically (no temporal component).
        The optimizer should pick the lowest gain value.
        """
        gain_values = np.array([0.1, 0.2, 0.3, 0.4])
        # Variance rises as gain rises — simulate pure noise/aliasing.
        modal_variances = np.array([
            [1.0, 3.0, 6.0, 10.0],  # mode 0: increasing
            [2.0, 4.0, 7.0, 11.0],  # mode 1: increasing
        ])
        mock_grid.return_value = (gain_values, modal_variances)

        gain_vector, sweep_results = optimize_gain_blocks(
            **self._make_kwargs(n_modes=2, gain_block_sizes=[2])
        )

        # Both modes in one block → gain should be gain_values[0] = 0.1
        self.assertAlmostEqual(gain_vector[0], 0.1)
        self.assertAlmostEqual(gain_vector[1], 0.1)
        self.assertEqual(sweep_results[0]["best_gain"], 0.1)

    # -----------------------------------------------------------------------
    # Edge case 3 – mixed blocks: first block noise-free, second temporal-free
    # -----------------------------------------------------------------------

    @patch('src.Functions._compute_modal_variance_grid')
    def test_mixed_blocks_select_opposite_gains(self, mock_grid):
        """
        Block 1 (modes 0-1): noise/aliasing-free → expects maximum gain.
        Block 2 (modes 2-3): temporal-free       → expects minimum gain.
        """
        gain_values = np.array([0.1, 0.2, 0.3, 0.4])
        modal_variances = np.array([
            [10.0, 6.0, 3.0, 1.0],   # mode 0: decreasing → best = 0.4
            [12.0, 8.0, 4.0, 2.0],   # mode 1: decreasing → best = 0.4
            [1.0,  3.0, 6.0, 10.0],  # mode 2: increasing → best = 0.1
            [2.0,  4.0, 8.0, 12.0],  # mode 3: increasing → best = 0.1
        ])
        mock_grid.return_value = (gain_values, modal_variances)

        gain_vector, sweep_results = optimize_gain_blocks(
            **self._make_kwargs(n_modes=4, gain_block_sizes=[2, 2])
        )

        self.assertAlmostEqual(sweep_results[0]["best_gain"], 0.4,
                               msg="Noise-free block should select max gain")
        self.assertAlmostEqual(sweep_results[1]["best_gain"], 0.1,
                               msg="Temporal-free block should select min gain")
        # Modes within each block share the same gain
        self.assertAlmostEqual(gain_vector[0], gain_vector[1])
        self.assertAlmostEqual(gain_vector[2], gain_vector[3])
        # The two blocks must select different gains
        self.assertNotAlmostEqual(gain_vector[0], gain_vector[2])

    # -----------------------------------------------------------------------
    # Edge case 4 – each mode is its own block
    # -----------------------------------------------------------------------

    @patch('src.Functions._compute_modal_variance_grid')
    def test_individual_mode_blocks_each_get_own_optimal_gain(self, mock_grid):
        """
        4 modes, each in its own block.  Each mode has a different optimal gain.
        """
        gain_values = np.array([0.1, 0.2, 0.3, 0.4])
        modal_variances = np.array([
            [5.0, 4.0, 3.0, 2.0],   # mode 0: best = 0.4
            [3.0, 2.0, 4.0, 5.0],   # mode 1: best = 0.2
            [4.0, 3.0, 2.0, 3.0],   # mode 2: best = 0.3
            [2.0, 4.0, 5.0, 6.0],   # mode 3: best = 0.1
        ])
        mock_grid.return_value = (gain_values, modal_variances)

        gain_vector, sweep_results = optimize_gain_blocks(
            **self._make_kwargs(n_modes=4, gain_block_sizes=[1, 1, 1, 1])
        )

        self.assertAlmostEqual(gain_vector[0], 0.4)
        self.assertAlmostEqual(gain_vector[1], 0.2)
        self.assertAlmostEqual(gain_vector[2], 0.3)
        self.assertAlmostEqual(gain_vector[3], 0.1)

    # -----------------------------------------------------------------------
    # Edge case 5 – single block over all modes
    # -----------------------------------------------------------------------

    @patch('src.Functions._compute_modal_variance_grid')
    def test_single_block_aggregates_all_modes(self, mock_grid):
        """
        One block containing all 4 modes.  The best gain minimises the SUM of
        all modal variances, which may differ from any individual modal optimum.
        """
        gain_values = np.array([0.1, 0.2, 0.3, 0.4])
        modal_variances = np.array([
            [5.0, 2.0, 3.0, 4.0],   # mode 0: individual best = 0.2
            [4.0, 3.0, 1.0, 5.0],   # mode 1: individual best = 0.3
            [6.0, 1.0, 4.0, 3.0],   # mode 2: individual best = 0.2
            [3.0, 4.0, 2.0, 5.0],   # mode 3: individual best = 0.3
        ])
        # Sum across modes: [18, 10, 10, 17] → tie at index 1 and 2 → argmin = 1 → gain 0.2
        mock_grid.return_value = (gain_values, modal_variances)

        gain_vector, sweep_results = optimize_gain_blocks(
            **self._make_kwargs(n_modes=4, gain_block_sizes=[4])
        )

        # All modes share the same gain
        self.assertTrue(np.all(gain_vector == gain_vector[0]))
        # Gain must be one of the two tied minima
        self.assertIn(gain_vector[0], [0.2, 0.3])
        self.assertEqual(len(sweep_results), 1)

    # -----------------------------------------------------------------------
    # Edge case 6 – find_best_gain returns correct subset aggregation
    # -----------------------------------------------------------------------

    @patch('src.Functions._compute_modal_variance_grid')
    def test_find_best_gain_aggregates_only_selected_modes(self, mock_grid):
        """
        find_best_gain with modes_to_optimize=[1, 2] should sum only rows 1 and 2
        of the modal variance matrix, ignoring rows 0 and 3.
        """
        from src.Functions import find_best_gain

        gain_values = np.array([0.1, 0.2, 0.3])
        # Row 0 and 3 have minimum at 0.1; rows 1-2 have minimum at 0.3.
        modal_variances = np.array([
            [1.0, 5.0, 8.0],   # mode 0: best = 0.1  (should be ignored)
            [8.0, 5.0, 1.0],   # mode 1: best = 0.3  ← selected
            [9.0, 6.0, 2.0],   # mode 2: best = 0.3  ← selected
            [1.0, 4.0, 7.0],   # mode 3: best = 0.1  (should be ignored)
        ])
        mock_grid.return_value = (gain_values, modal_variances)

        kw = dict(
            gain_min=0.1, gain_max=0.3,
            omega_temp_freq_interval=np.array([1.0]),
            t_freqs=np.array([1.0]), f=np.array([1.0]),
            t_0=0.001, plant_num=np.array([1.0]), plant_den=np.array([1.0]),
            telescope_diameter=8.0, fried_parameter=0.15,
            excess_noise_factor=1.0, sky_background=0.0, dark_current=0.0,
            readout_noise=0.0, photon_flux=1000.0, frame_rate=1000.0,
            magnitude=0.0, n_subaperture=40,
            slope_computer_weights=np.array([1.0]), fitting_coeff=0.27,
            alpha=-17/3, seeing=0.8, modulation_radius=3.0, wind_speed=15.0,
            maximum_radial_order_corrected=10, reconstruction_matrix_path='dummy.fits',
            psd_turbulence=np.zeros((4, 1)), psd_windshake=np.zeros((4, 1)),
            sigma_slopes_path='dummy.fits', c_optg=np.array([1.0]),
            actuators_number=4, modes_to_optimize=[1, 2],
        )

        best_gain, gain_vals, tot_var = find_best_gain(**kw)

        self.assertAlmostEqual(best_gain, 0.3)
        # tot_var must be the sum of rows 1 and 2 only
        expected = modal_variances[1] + modal_variances[2]
        np.testing.assert_array_almost_equal(tot_var, expected)
