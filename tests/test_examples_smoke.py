#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import unittest
from contextlib import contextmanager
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")

from examples.plot_psds import plot_system_psds
from examples.verify_aliasing_energy import verify_aliasing_energy


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


@contextmanager
def repo_root_cwd():
    previous_cwd = os.getcwd()
    os.chdir(REPO_ROOT)
    try:
        yield
    finally:
        os.chdir(previous_cwd)


class TestExampleScriptsSmoke(unittest.TestCase):
    """Smoke tests for the example scripts."""

    def test_plot_system_psds_runs_without_display(self):
        with patch("matplotlib.pyplot.show") as mock_show:
            with repo_root_cwd():
                result = plot_system_psds(mode_index=0, plot_inputs=False, show_plot=False)

        mock_show.assert_not_called()
        self.assertIsInstance(result, dict)
        self.assertEqual(result["mode_index"], 0)
        self.assertIn("psd_total", result)
        self.assertGreater(result["var_alias_out"], 0.0)

    def test_verify_aliasing_energy_runs_without_display(self):
        with patch("matplotlib.pyplot.show") as mock_show:
            with repo_root_cwd():
                result = verify_aliasing_energy()

        mock_show.assert_not_called()
        self.assertIsInstance(result, dict)
        self.assertIn("var_from_psd_total", result)
        self.assertGreater(result["var_from_psd_total"], 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
