#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Single-mode IIR controller optimization scaffolding.

This module provides the infrastructure to efficiently optimize an AO loop
controller on a single Zernike mode.  All controller-independent quantities
(atmospheric PSD, vibration PSD, aliasing PSD, measurement-noise PSD) are
pre-computed once by :func:`prepare_single_mode_control_optimization`, which
returns a :class:`SingleModeControllerOptimizationContext`.  The context
exposes an :meth:`~SingleModeControllerOptimizationContext.evaluate` method
that accepts arbitrary IIR numerator/denominator polynomials, recomputes only
the transfer functions and the resulting cost, and optionally records the
trajectory in ``context.history``.
"""

# pylint: disable=C

from dataclasses import dataclass, field

import numpy as np

from src.Functions import (
    aliasing_psd_from_coeffs,
    build_transfer_function_from_controller_polynomials,
    compute_noise_PSD_intermediate,
    compute_slope_noise_variance,
    extract_propagation_coefficients,
    func_out,
    integrate_function,
    k_coeff_aliasing,
    total_variance,
)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


# @dataclass automatically generates the constructor (__init__) for classes that act as simple data containers.
@dataclass
class SingleModeControllerOptimizationRecord:
    """Lightweight history entry stored after each :meth:`SingleModeControllerOptimizationContext.evaluate` call."""
    controller_num: np.ndarray
    controller_den: np.ndarray
    cost: float
    variance_terms: dict


# @dataclass automatically generates the constructor (__init__) for classes that act as simple data containers.
@dataclass
class SingleModeControllerOptimizationResult:
    """Full result returned by :meth:`SingleModeControllerOptimizationContext.evaluate`."""
    controller_num: np.ndarray
    controller_den: np.ndarray
    cost: float
    variance_terms: dict
    psd_input: dict
    psd_output: dict
    transfer_function: dict


# ---------------------------------------------------------------------------
# Context
# ---------------------------------------------------------------------------

# @dataclass automatically generates the constructor (__init__) for classes that act as simple data containers.
@dataclass
class SingleModeControllerOptimizationContext:
    """Pre-computed context for single-mode IIR controller optimization.

    All static, controller-independent inputs are stored as attributes.
    Call :meth:`evaluate` repeatedly with different controller polynomials;
    only the transfer functions and the resulting variances are recomputed.

    Attributes
    ----------
    mode_index : int
        Zero-based index of the Zernike mode being optimized.
    omega_temp_freq_interval : np.ndarray
        Angular temporal frequency vector [rad s⁻¹].
    t_0 : float
        Sampling period [s].
    plant_num : np.ndarray
        Pre-multiplied plant numerator polynomial (e.g. ``n1 * n2 * n3``).
    plant_den : np.ndarray
        Pre-multiplied plant denominator polynomial (e.g. ``d1 * d2 * d3``).
    PSD_input_atmos : np.ndarray, shape (1, n_freq)
        Atmospheric turbulence PSD for the selected mode.
    PSD_input_vibration : np.ndarray, shape (1, n_freq)
        Vibration PSD for the selected mode (zeros if mode not available).
    PSD_input_alias : np.ndarray, shape (1, n_freq)
        Aliasing input PSD for the selected mode (already divided by c_optg²).
    PSD_input_measurement : np.ndarray, shape (1, n_freq)
        Measurement-noise input PSD (already divided by c_optg²).
    history : list of SingleModeControllerOptimizationRecord
        Evaluation history; grows with each ``evaluate(store_history=True)`` call.
    """
    mode_index: int
    omega_temp_freq_interval: np.ndarray
    t_0: float
    plant_num: np.ndarray
    plant_den: np.ndarray
    PSD_input_atmos: np.ndarray
    PSD_input_vibration: np.ndarray
    PSD_input_alias: np.ndarray
    PSD_input_measurement: np.ndarray
    history: list = field(default_factory=list)

    def evaluate(self, controller_num, controller_den, store_history=False):
        """Evaluate the total variance for a given IIR controller.

        Parameters
        ----------
        controller_num : array_like
            Numerator polynomial coefficients (descending powers of Z).
        controller_den : array_like
            Denominator polynomial coefficients (descending powers of Z).
        store_history : bool, optional
            If ``True``, append a record to ``self.history``.
            It is useful to keep track of the optimization trajectory,
            but it can grow indefinitely. 
            Use with ``store_history=False`` for a fixed memory footprint.

        Returns
        -------
        SingleModeControllerOptimizationResult
        """
        H_r, H_n = build_transfer_function_from_controller_polynomials(
            controller_num,
            controller_den,
            self.omega_temp_freq_interval,
            self.t_0,
            1,
            self.plant_num,
            self.plant_den,
        )

        PSD_output_atmos = func_out(H_r[0, :], self.PSD_input_atmos[0, :])[np.newaxis, :]
        PSD_output_vibration = func_out(H_r[0, :], self.PSD_input_vibration[0, :])[np.newaxis, :]
        PSD_output_temporal = PSD_output_atmos + PSD_output_vibration
        PSD_output_alias = func_out(H_n[0, :], self.PSD_input_alias[0, :])[np.newaxis, :]
        PSD_output_measurement = func_out(H_n[0, :], self.PSD_input_measurement[0, :])[np.newaxis, :]
        PSD_output_total = PSD_output_temporal + PSD_output_alias + PSD_output_measurement

        var_temp_atmos = integrate_function(PSD_output_atmos[0, :], self.omega_temp_freq_interval)
        var_temp_vibration = integrate_function(PSD_output_vibration[0, :], self.omega_temp_freq_interval)
        var_temp_total = var_temp_atmos + var_temp_vibration
        var_alias = integrate_function(PSD_output_alias[0, :], self.omega_temp_freq_interval)
        var_measurement = integrate_function(PSD_output_measurement[0, :], self.omega_temp_freq_interval)
        # Note: No static fitting variance term is included here, because it is independent of the controller
        #       and does not affect the optimization trajectory.
        cost = total_variance(0.0, var_temp_total, var_alias, var_measurement)

        variance_terms = {
            "temporal_atmosphere": float(np.real(var_temp_atmos)),
            "temporal_vibration": float(np.real(var_temp_vibration)),
            "temporal": float(np.real(var_temp_total)),
            "aliasing": float(np.real(var_alias)),
            "measurement": float(np.real(var_measurement)),
            "total": float(np.real(cost)),
        }

        result = SingleModeControllerOptimizationResult(
            controller_num=np.array(controller_num, dtype=complex, copy=True),
            controller_den=np.array(controller_den, dtype=complex, copy=True),
            cost=float(np.real(cost)),
            variance_terms=variance_terms,
            psd_input={
                "atmosphere": np.array(self.PSD_input_atmos, copy=True),
                "vibration": np.array(self.PSD_input_vibration, copy=True),
                "temporal": np.array(self.PSD_input_atmos + self.PSD_input_vibration, copy=True),
                "aliasing": np.array(self.PSD_input_alias, copy=True),
                "measurement": np.array(self.PSD_input_measurement, copy=True),
                "total": np.array(
                    self.PSD_input_atmos + self.PSD_input_vibration
                    + self.PSD_input_alias + self.PSD_input_measurement,
                    copy=True,
                ),
            },
            psd_output={
                "atmosphere": PSD_output_atmos,
                "vibration": PSD_output_vibration,
                "temporal": PSD_output_temporal,
                "aliasing": PSD_output_alias,
                "measurement": PSD_output_measurement,
                "total": PSD_output_total,
            },
            transfer_function={
                "H_r": np.array(H_r, copy=True),
                "H_n": np.array(H_n, copy=True),
            },
        )

        if store_history:
            self.history.append(
                SingleModeControllerOptimizationRecord(
                    controller_num=np.array(controller_num, dtype=complex, copy=True),
                    controller_den=np.array(controller_den, dtype=complex, copy=True),
                    cost=float(np.real(cost)),
                    variance_terms=dict(variance_terms),
                )
            )

        return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _select_single_mode_psd(PSD_in, mode_index, array_name, n_frequencies, allow_missing=False):
    """Extract row ``mode_index`` from a 2-D PSD array as shape ``(1, n_freq)``."""
    PSD_in = np.asarray(PSD_in, dtype=float)

    if PSD_in.ndim != 2:
        raise ValueError(f"{array_name} must be a 2D array")

    if PSD_in.shape[1] != n_frequencies:
        raise ValueError(
            f"{array_name} must have {n_frequencies} frequency samples; got {PSD_in.shape[1]}"
        )

    if mode_index < PSD_in.shape[0]:
        return PSD_in[mode_index: mode_index + 1, :]

    if allow_missing:
        return np.zeros((1, n_frequencies), dtype=float)

    raise ValueError(
        f"Requested mode_index={mode_index} for {array_name}, "
        f"but only {PSD_in.shape[0]} modes are available"
    )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def prepare_single_mode_control_optimization(
    mode_index,
    omega_temp_freq_interval,
    t_0,
    PSD_atmo_turb,
    PSD_vibration,
    alpha,
    telescope_diameter,
    seeing,
    modulation_radius,
    windspeed,
    maximum_radial_order_corrected,
    c_optg,
    F_excess,
    pixel_pos,
    sky_bkg,
    dark_curr,
    read_out_noise,
    photon_flux,
    frame_rate,
    magnitudo,
    n_subaperture,
    collecting_area,
    file_path_matrix_R,
    file_path_sigma_slopes=None,
    plant_num=None,
    plant_den=None,
):
    """Prepare a reusable single-mode optimization context for an IIR controller.

    All controller-independent inputs are precomputed once:

    * temporal input PSDs (atmosphere and vibration),
    * aliasing input PSD,
    * measurement input PSD.

    The returned context can then evaluate many controller candidates cheaply,
    updating only H_r, H_n and the resulting cost.

    Parameters
    ----------
    mode_index : int
        Zero-based Zernike mode index to optimize.
    omega_temp_freq_interval : array_like
        Angular temporal frequency vector [rad s⁻¹].
    t_0 : float
        Sampling period [s].
    PSD_atmo_turb : np.ndarray, shape (n_modes, n_freq)
        Atmospheric turbulence PSD.
    PSD_vibration : np.ndarray, shape (n_modes_vib, n_freq)
        Vibration PSD (can have fewer modes than ``PSD_atmo_turb``).
    alpha : float
        Spectral index for the aliasing model.
    telescope_diameter : float
        Telescope diameter [m].
    seeing : float
        Seeing value (for optical-gain interpolation and aliasing).
    modulation_radius : float
        Pyramid modulation radius [λ/D].
    windspeed : float
        Wind speed [m s⁻¹].
    maximum_radial_order_corrected : int
        Maximum corrected radial order.
    c_optg : float
        Optical gain (non-zero).
    F_excess : float
        Excess noise factor.
    pixel_pos : array_like
        Pixel positions for slope noise variance computation.
    sky_bkg, dark_curr, read_out_noise : float
        Detector noise parameters.
    photon_flux : float
        Guide-star photon flux.
    frame_rate : float
        Loop frame rate [Hz].
    magnitudo : float
        Guide-star magnitude.
    n_subaperture : int
        Number of WFS sub-apertures per axis.
    collecting_area : float
        Telescope collecting area [m²].
    file_path_matrix_R : str
        Path to reconstruction-matrix FITS file.
    file_path_sigma_slopes : str, optional
        Path to sigma-slopes FITS file (uses default if ``None``).
    plant_num : array_like, optional
        Pre-multiplied plant numerator (e.g. ``n1 * n2 * n3``).
        Defaults to ``[1.0]``.
    plant_den : array_like, optional
        Pre-multiplied plant denominator (e.g. ``d1 * d2 * d3``).
        Defaults to ``[1.0]``.

    Returns
    -------
    SingleModeControllerOptimizationContext
    """
    if mode_index < 0:
        raise ValueError("mode_index must be >= 0")

    omega_temp_freq_interval = np.asarray(omega_temp_freq_interval, dtype=float).ravel()

    if omega_temp_freq_interval.size == 0:
        raise ValueError("omega_temp_freq_interval must not be empty")

    if np.isclose(c_optg, 0.0):
        raise ValueError("c_optg must be non-zero to build aliasing and measurement PSDs")

    n_frequencies = omega_temp_freq_interval.size

    if plant_num is None:
        plant_num = np.array([1.0])
    if plant_den is None:
        plant_den = np.array([1.0])

    PSD_input_atmos = _select_single_mode_psd(
        PSD_atmo_turb, mode_index, "PSD_atmo_turb", n_frequencies, allow_missing=False
    )
    PSD_input_vibration = _select_single_mode_psd(
        PSD_vibration, mode_index, "PSD_vibration", n_frequencies, allow_missing=True
    )

    k_alias = np.asarray(
        k_coeff_aliasing(
            modulation_radius, seeing, alpha, telescope_diameter,
            omega_temp_freq_interval, file_path_matrix_R, windspeed,
            maximum_radial_order_corrected, file_path_sigma_slopes,
        ),
        dtype=float,
    )

    if mode_index >= k_alias.size:
        raise ValueError(
            f"Requested mode_index={mode_index}, but aliasing coefficients are "
            f"available only up to {k_alias.size - 1}"
        )

    PSD_input_alias = aliasing_psd_from_coeffs(
        1, omega_temp_freq_interval,
        np.array([k_alias[mode_index]], dtype=float),
        alpha, telescope_diameter, windspeed, maximum_radial_order_corrected,
    ) / (c_optg ** 2)

    slope_noise_variance = compute_slope_noise_variance(
        F_excess, pixel_pos, sky_bkg, dark_curr, read_out_noise,
        photon_flux, telescope_diameter, frame_rate, magnitudo,
        n_subaperture, collecting_area,
    )

    p_coefficient = np.asarray(
        extract_propagation_coefficients(file_path_matrix_R), dtype=float
    )

    if mode_index >= p_coefficient.size:
        raise ValueError(
            f"Requested mode_index={mode_index}, but propagation coefficients are "
            f"available only up to {p_coefficient.size - 1}"
        )

    sigma2_w = np.array([p_coefficient[mode_index] * slope_noise_variance], dtype=float)
    PSD_input_measurement = (
        compute_noise_PSD_intermediate(omega_temp_freq_interval, 1, sigma2_w) / (c_optg ** 2)
    )

    return SingleModeControllerOptimizationContext(
        mode_index=mode_index,
        omega_temp_freq_interval=omega_temp_freq_interval,
        t_0=float(t_0),
        plant_num=np.asarray(plant_num, dtype=float),
        plant_den=np.asarray(plant_den, dtype=float),
        PSD_input_atmos=PSD_input_atmos,
        PSD_input_vibration=PSD_input_vibration,
        PSD_input_alias=PSD_input_alias,
        PSD_input_measurement=PSD_input_measurement,
    )
