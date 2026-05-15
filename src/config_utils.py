#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parameter resolution utilities shared across run scripts.

This module contains helpers that operate on the parameter dictionary returned
by ``load_parameters`` (from ``src.Functions``) *before* the dictionary is
consumed by a run script.  Physics and signal-processing functions live in
``src.Functions``; orchestration logic belongs here.

Public API
----------
resolve_binning_config(param)
    Overwrite the flat YAML parameters for the selected detector binning when a
    ``binning_configs`` table is present in the YAML file.
"""

# ---------------------------------------------------------------------------
# Mandatory data-file fields that a binning config entry may override
# ---------------------------------------------------------------------------
_BINNING_DATA_MANDATORY_FIELDS = (
    'reconstruction_matrix',
    'sigma_slopes',
    'modal_psd_alias',
)

# Optional data-file fields: when present in the bin entry they replace the
# flat value; the sibling exclusive field is *not* touched (the existing
# mutual-exclusion check in the run scripts still applies).
_BINNING_DATA_OPTIONAL_FIELDS = (
    'optical_gain_models',
    'optical_gain_cube',
)


def resolve_binning_config(param):
    """Overwrite flat YAML parameters for the selected detector binning.

    If the YAML file contains a top-level ``binning_configs`` section, the
    function looks up the entry whose key matches ``control.bin`` and
    overwrites the following flat parameters:

    * ``control.n_modes``
    * ``wavefront_sensor.number_of_sub``
    * ``data.reconstruction_matrix``
    * ``data.sigma_slopes``
    * ``data.modal_psd_alias``
    * ``data.optical_gain_models``  (when present in the entry)
    * ``data.optical_gain_cube``    (when present in the entry)

    When ``binning_configs`` is **absent** the function is a no-op and returns
    *param* unchanged, preserving full backward compatibility with YAML files
    that do not use this feature.

    Parameters
    ----------
    param : dict
        Parameter dictionary as returned by ``load_parameters``.

    Returns
    -------
    dict
        The (possibly modified) parameter dictionary.  The same object is
        returned (modified in-place) to allow chaining.

    Raises
    ------
    KeyError
        If ``binning_configs`` is present but contains no entry for the
        selected bin value.
    ValueError
        If ``control.bin`` is missing when ``binning_configs`` is present.
    """
    if 'binning_configs' not in param:
        return param

    bin_value = param.get('control', {}).get('bin')
    if bin_value is None:
        raise ValueError(
            "'binning_configs' is present in the YAML but 'control.bin' is not set."
        )

    configs = param['binning_configs']

    # YAML parsers may read integer keys as int or as str depending on the
    # YAML dialect.  Try both to be robust.
    cfg = configs.get(bin_value)
    if cfg is None:
        cfg = configs.get(str(bin_value))
    if cfg is None:
        cfg = configs.get(int(bin_value)) if str(bin_value).isdigit() else None

    if cfg is None:
        available = list(configs.keys())
        raise KeyError(
            f"Binning value {bin_value!r} not found in 'binning_configs'. "
            f"Available keys: {available}"
        )

    # --- control ---
    if 'n_modes' in cfg:
        param['control']['n_modes'] = cfg['n_modes']

    # --- wavefront sensor ---
    if 'n_subapert' in cfg:
        param['wavefront_sensor']['number_of_sub'] = cfg['n_subapert']

    # --- data: mandatory fields ---
    for field in _BINNING_DATA_MANDATORY_FIELDS:
        if field in cfg:
            param['data'][field] = cfg[field]

    # --- data: optional / mutually-exclusive fields ---
    for field in _BINNING_DATA_OPTIONAL_FIELDS:
        if field in cfg:
            param['data'][field] = cfg[field]

    return param
