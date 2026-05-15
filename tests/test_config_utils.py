#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for src/config_utils.py.

Run from the repository root with:
    python -m pytest tests/test_config_utils.py -v
"""

import unittest

from src.config_utils import resolve_binning_config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _base_param(bin_value=1):
    """Return a minimal param dict mimicking what load_parameters produces."""
    return {
        'control': {
            'bin': bin_value,
            'n_modes': 300,
        },
        'wavefront_sensor': {
            'number_of_sub': 40,
        },
        'data': {
            'reconstruction_matrix': 'rm_orig.fits',
            'sigma_slopes': 'sl_orig.fits',
            'modal_psd_alias': 'mpa_orig.fits',
            'windshake_psd': 'wind.fits',
            'optical_gain_models': ['og0_orig.fits', 'og3_orig.fits'],
        },
    }


_CONFIGS = {
    1: {
        'n_modes': 300,
        'n_subapert': 40,
        'reconstruction_matrix': 'rm_b1.fits',
        'sigma_slopes': 'sl_b1.fits',
        'modal_psd_alias': 'mpa_b1.fits',
        'optical_gain_models': ['og0_b1.fits', 'og3_b1.fits'],
    },
    2: {
        'n_modes': 209,
        'n_subapert': 20,
        'reconstruction_matrix': 'rm_b2.fits',
        'sigma_slopes': 'sl_b2.fits',
        'modal_psd_alias': 'mpa_b2.fits',
        'optical_gain_models': ['og0_b2.fits', 'og3_b2.fits'],
    },
    3: {
        'n_modes': 90,
        'n_subapert': 13,
        'reconstruction_matrix': 'rm_b3.fits',
        'sigma_slopes': 'sl_b3.fits',
        'modal_psd_alias': 'mpa_b3.fits',
        'optical_gain_models': ['og0_b3.fits', 'og3_b3.fits'],
    },
    4: {
        'n_modes': 54,
        'n_subapert': 10,
        'reconstruction_matrix': 'rm_b4.fits',
        'sigma_slopes': 'sl_b4.fits',
        'modal_psd_alias': 'mpa_b4.fits',
        'optical_gain_models': ['og0_b4.fits', 'og3_b4.fits'],
    },
}


def _param_with_configs(bin_value=1, configs=None):
    param = _base_param(bin_value)
    param['binning_configs'] = configs if configs is not None else dict(_CONFIGS)
    return param


# ---------------------------------------------------------------------------
# Backward compatibility: no binning_configs section
# ---------------------------------------------------------------------------

class TestNoBinningConfigsSection(unittest.TestCase):
    """When 'binning_configs' is absent the function must be a strict no-op."""

    def test_returns_same_object(self):
        param = _base_param()
        result = resolve_binning_config(param)
        self.assertIs(result, param)

    def test_n_modes_unchanged(self):
        param = _base_param()
        resolve_binning_config(param)
        self.assertEqual(param['control']['n_modes'], 300)

    def test_n_subapert_unchanged(self):
        param = _base_param()
        resolve_binning_config(param)
        self.assertEqual(param['wavefront_sensor']['number_of_sub'], 40)

    def test_reconstruction_matrix_unchanged(self):
        param = _base_param()
        resolve_binning_config(param)
        self.assertEqual(param['data']['reconstruction_matrix'], 'rm_orig.fits')


# ---------------------------------------------------------------------------
# Correct overwrite for each binning value
# ---------------------------------------------------------------------------

class TestBinningOverwrites(unittest.TestCase):

    def _check_bin(self, bin_value):
        param = _param_with_configs(bin_value)
        resolve_binning_config(param)
        expected = _CONFIGS[bin_value]
        self.assertEqual(param['control']['n_modes'], expected['n_modes'])
        self.assertEqual(param['wavefront_sensor']['number_of_sub'], expected['n_subapert'])
        self.assertEqual(param['data']['reconstruction_matrix'], expected['reconstruction_matrix'])
        self.assertEqual(param['data']['sigma_slopes'], expected['sigma_slopes'])
        self.assertEqual(param['data']['modal_psd_alias'], expected['modal_psd_alias'])
        self.assertEqual(param['data']['optical_gain_models'], expected['optical_gain_models'])

    def test_bin1(self):
        self._check_bin(1)

    def test_bin2(self):
        self._check_bin(2)

    def test_bin3(self):
        self._check_bin(3)

    def test_bin4(self):
        self._check_bin(4)

    def test_returns_same_object(self):
        param = _param_with_configs(2)
        result = resolve_binning_config(param)
        self.assertIs(result, param)


# ---------------------------------------------------------------------------
# Windshake PSD is NOT part of the binning config and must not be touched
# ---------------------------------------------------------------------------

class TestFieldsNotOverwritten(unittest.TestCase):

    def test_windshake_psd_preserved(self):
        param = _param_with_configs(2)
        resolve_binning_config(param)
        self.assertEqual(param['data']['windshake_psd'], 'wind.fits')


# ---------------------------------------------------------------------------
# Optional fields: optical_gain_cube
# ---------------------------------------------------------------------------

class TestOptionalCubeField(unittest.TestCase):

    def test_optical_gain_cube_set_when_present(self):
        configs = {1: {'n_modes': 300, 'n_subapert': 40, 'optical_gain_cube': 'cube_b1.fits'}}
        param = _param_with_configs(1, configs)
        resolve_binning_config(param)
        self.assertEqual(param['data']['optical_gain_cube'], 'cube_b1.fits')

    def test_optical_gain_models_absent_entry_leaves_flat_value(self):
        """If a bin entry has no optical_gain_models key, the flat value is kept."""
        configs = {2: {'n_modes': 209, 'n_subapert': 20,
                       'reconstruction_matrix': 'rm.fits',
                       'sigma_slopes': 'sl.fits',
                       'modal_psd_alias': 'mpa.fits'}}
        param = _param_with_configs(2, configs)
        resolve_binning_config(param)
        # flat value should survive
        self.assertEqual(param['data']['optical_gain_models'], ['og0_orig.fits', 'og3_orig.fits'])


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestErrorHandling(unittest.TestCase):

    def test_missing_bin_raises_key_error(self):
        param = _param_with_configs(5)  # bin 5 not defined
        with self.assertRaises(KeyError):
            resolve_binning_config(param)

    def test_missing_control_bin_raises_value_error(self):
        param = _base_param()
        del param['control']['bin']
        param['binning_configs'] = _CONFIGS
        with self.assertRaises(ValueError):
            resolve_binning_config(param)


# ---------------------------------------------------------------------------
# YAML key robustness: integer vs string keys
# ---------------------------------------------------------------------------

class TestStringIntKeyRobustness(unittest.TestCase):
    """YAML parsers may produce '2' or 2 as the dict key; both must work."""

    def test_string_keys_resolved(self):
        configs = {'1': {'n_modes': 300, 'n_subapert': 40},
                   '2': {'n_modes': 209, 'n_subapert': 20}}
        param = _base_param(bin_value=2)
        param['binning_configs'] = configs
        resolve_binning_config(param)
        self.assertEqual(param['control']['n_modes'], 209)
        self.assertEqual(param['wavefront_sensor']['number_of_sub'], 20)

    def test_int_keys_resolved(self):
        param = _param_with_configs(bin_value=3)
        resolve_binning_config(param)
        self.assertEqual(param['control']['n_modes'], 90)


if __name__ == '__main__':
    unittest.main()
