#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Main entrypoint for Semi-analytic Error Budget runs.

Usage:
    python scripts/main_saeb.py params_mod4_4000modes.yaml
    python scripts/main_saeb.py run params_mod4_4000modes.yaml
"""

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.Functions import seeing_to_r0
from src.Functions import aliasing_variance
from src.Functions import build_transfer_function
from src.Functions import compute_optical_gain
from src.Functions import final_soul_optical_gain
from src.Functions import fitting_variance
from src.Functions import funct_d2
from src.Functions import load_parameters
from src.Functions import load_PSD_windshake
from src.Functions import measure_variance
from src.Functions import radial_order_from_n_modes
from src.Functions import temporal_variance
from src.Functions import total_variance
from src.Functions import turbulence_psd
from src.Functions import vibration_variance
from src.Functions import integrate_modal_psd
from src.Functions import gain_maximum_from_total_delay
from src.Functions import optimize_gain_blocks
from src.Functions import resolve_gain_mode
from src.plots import summary_display
from src.plots import plot_gain_optimization_sweep
from src.config_utils import resolve_binning_config


def _resolve_yaml_path(yaml_file):
    yaml_path = Path(yaml_file)

    if not yaml_path.is_absolute():
        yaml_path = (PROJECT_ROOT / yaml_path).resolve()

    if not yaml_path.exists():
        raise FileNotFoundError(f"YAML file not found: {yaml_path}")

    return yaml_path


def _build_gain_vector(loop_params, n_actuators):
    gain_minimum = loop_params.get('gain_min', None)
    gain_number = loop_params.get('gain_n', None)
    gain_value = loop_params.get('gain_value', None)
    gain_vector = loop_params.get('gain_vector', None)

    if gain_value is not None and gain_vector is not None:
        raise ValueError("Cannot set both gain_value and gain_vector")

    if gain_vector is not None:
        gain_vector = np.asarray(gain_vector, dtype=float).ravel()

        if gain_vector.size == 1:
            return np.full(n_actuators, gain_vector.item())

        if gain_vector.size == n_actuators:
            return gain_vector

        raise ValueError("gain_vector must have length 1 or N_act")

    if gain_value is not None:
        if isinstance(gain_value, list):
            if gain_number is None:
                raise ValueError("When gain_value is a list, gain_n must be provided to specify"
                                 " how many actuators each value applies to")
            if not isinstance(gain_number, list):
                raise ValueError("When gain_value is a list, gain_n must also be a list of the"
                                 " same length")
            if len(gain_number) != len(gain_value):
                raise ValueError(f"When gain_n is a list, length of gain_n {len(gain_number)} must"
                                 f" match length of gain_value {len(gain_value)}")
            gain_value = [val for i, val in enumerate(gain_value) for _ in range(gain_number[i])]
            return np.asarray(gain_value, dtype=float).ravel()
        else:
            return np.full(n_actuators, float(gain_value))

    if gain_number == 1:
        return np.full(n_actuators, float(gain_minimum))

    if gain_number is not None and gain_number > 1:
        raise ValueError(
            f"gain_n={gain_number} > 1 implies a gain sweep (as in Total_Variance.py). "
            "main_saeb.py does not support gain optimisation. "
            "Use gain_n: 1 with gain_min for a uniform gain, "
            "or use gain_value / gain_vector for explicit per-mode assignment."
        )

    raise ValueError("Set gain_n: 1 with gain_min, or provide gain_value/gain_vector")


def run(yaml_file):
    yaml_path = _resolve_yaml_path(yaml_file)
    param = load_parameters(str(yaml_path))

    if param is None:
        raise RuntimeError("Parameters not loaded")

    param = resolve_binning_config(param)

    print("Parameters loaded successfully.")

    n_actuators = param['control']['n_modes']
    telescope_diameter = param['telescope']['telescope_diam']
    aperture_radius = telescope_diameter / 2
    aperture_center = [0, 0, 0]

    outer_scale = param['atmosphere']['outer_scale']
    layers_altitude = 0.0
    wind_direction = 0.0
    wind_speed = param['atmosphere']['wind_speed']
    seeing_ = param['atmosphere']['seeing']
    fried_param = seeing_to_r0(seeing_)

    rho = 0
    theta = 0

    value_F_excess_noise = param['wavefront_sensor']['value_for_F_excess_noise']
    F_excess_noise = np.sqrt(value_F_excess_noise)
    sky_background = param['wavefront_sensor']['sky_backgr']
    dark_current = param['wavefront_sensor']['dark_curr']
    readout_noise = param['wavefront_sensor']['noise_readout']

    file_path_R1 = param['data']['reconstruction_matrix']
    file_path_wind1 = param['data']['windshake_psd']
    file_optg = param['data'].get('optical_gain_models', None)
    file_sigma_slope = param['data']['sigma_slopes']
    file_optg_cube = param['data'].get('optical_gain_cube', None)

    plant = param['plant']
    plant_num = np.asarray(plant['numerator'])
    plant_den_base = np.asarray(plant['denominator'])

    control = param['control']
    t_0 = control['sampling_time']
    total_delay = plant['total_delay']
    gain_mode = resolve_gain_mode(control)
    modulation_radius = param['wavefront_sensor']['modulation_radius']
    maximum_rad_order_corr = radial_order_from_n_modes(n_actuators)

    spatial_freqs = np.logspace(-4, 4, 100)
    temporal_freqs_minimum = param['frequency_ranges']['temporal_freqs_min']
    temporal_freqs_maximum = np.log10(1.0 / (2.0 * t_0))
    temporal_freqs_number = param['frequency_ranges']['temporal_freqs_n']
    temporal_freqs = np.logspace(temporal_freqs_minimum,
                                 temporal_freqs_maximum,
                                 temporal_freqs_number)
    omega_temporal_freqs = 2 * np.pi * temporal_freqs

    fitting_coeff = 0.2778
    alpha_ = -17 / 3

    phot_flux = float(param['guide_star']['flux_photons'])
    frame_rate = 1.0 / t_0
    magnitudo = param['guide_star']['magn']
    n_subapert = param['wavefront_sensor']['number_of_sub']
    x_pixel = control['slope_computer_weights']

    display_cfg = param.get('display', {})
    display = bool(display_cfg.get('enabled', True))
    summary_modes_to_plot = display_cfg.get('summary_modes_to_plot', None)

    if summary_modes_to_plot is not None and not isinstance(summary_modes_to_plot, (list, tuple, np.ndarray)):
        summary_modes_to_plot = None

    gain_sweeps = None

    freq, PSD_wind_vib = load_PSD_windshake(file_path_wind1, target_frequencies=temporal_freqs)

    if (freq is None and PSD_wind_vib is None) or (freq is None or PSD_wind_vib is None):
        raise RuntimeError("PSD windshake or corresponding frequencies not loaded")

    print("PSD windshake and corresponding frequencies loaded successfully.")

    PSD_atmosf = turbulence_psd(rho, theta, aperture_radius,
                                aperture_center, fried_param, outer_scale,
                                layers_altitude, wind_speed, wind_direction,
                                spatial_freqs, temporal_freqs, n_modes=n_actuators)

    c_optg = 0
    if file_optg is None and file_optg_cube is not None:
        c_optg = final_soul_optical_gain(
            file_optg_cube,
            control['bin'],
            magnitudo,
            n_actuators,
        )
    else:
        c_optg = compute_optical_gain(
            file_optg[0],
            file_optg[1],
            seeing_,
            modulation_radius,
            actuators_number=n_actuators,
            modulation_radii=(0.0, 4.0),
        )

    plant_den = np.polymul(plant_den_base, funct_d2(total_delay))

    gain_block_sizes = control.get('gain_block_sizes', control.get('gain_blocks', None))

    if gain_mode == 'block_optimization':
        if control.get('gain_value') is not None or control.get('gain_vector') is not None:
            raise ValueError("gain_block_sizes cannot be combined with gain_value or gain_vector")

        gain_maximum = gain_maximum_from_total_delay(total_delay)
        gain_, gain_sweeps = optimize_gain_blocks(
            gain_min = control['gain_min'],
            gain_max = gain_maximum,
            omega_temp_freq_interval = omega_temporal_freqs,
            t_0 = t_0,
            plant_num = plant_num,
            plant_den = plant_den,
            telescope_diameter = telescope_diameter,
            fried_parameter = fried_param,
            excess_noise_factor = F_excess_noise,
            sky_background = sky_background,
            dark_current = dark_current,
            readout_noise = readout_noise,
            photon_flux = phot_flux,
            frame_rate = frame_rate,
            magnitude = magnitudo,
            n_subaperture = n_subapert,
            slope_computer_weights = x_pixel,
            fitting_coeff = fitting_coeff,
            alpha = alpha_,
            seeing = seeing_,
            modulation_radius = modulation_radius,
            wind_speed = wind_speed,
            maximum_radial_order_corrected = maximum_rad_order_corr,
            reconstruction_matrix_path = file_path_R1,
            psd_turbulence = PSD_atmosf,
            psd_windshake = PSD_wind_vib,
            sigma_slopes_path = file_sigma_slope,
            c_optg = c_optg,
            actuators_number = n_actuators,
            gain_block_sizes = gain_block_sizes,
            verbose=False,
        )
    elif gain_mode == 'fixed':
        gain_ = _build_gain_vector(control, n_actuators)
    else:
        raise ValueError(
            "main_saeb.py does not support legacy gain sweeps. Set gain_mode: fixed "
            "or gain_mode: block_optimization."
        )

    H_r_temp, H_n_meas = build_transfer_function(
        omega_temporal_freqs,
        t_0,
        n_actuators,
        plant_num,
        plant_den,
        gain=gain_,
    )
    H_n_alias = H_n_meas

    var_fit = fitting_variance(fitting_coeff, n_actuators, telescope_diameter, fried_param)

    PSD_vibr_zero = np.zeros_like(PSD_wind_vib)

    _, var_temp_atmo_CL, PSD_out_temp_atmo, PSD_in_temp_atmo = temporal_variance(
        PSD_atmosf,
        PSD_vibr_zero,
        H_r_temp,
        n_actuators,
        omega_temporal_freqs,
    )

    _, var_vibr_CL, PSD_out_vibr, PSD_in_vibr = vibration_variance(PSD_wind_vib, H_r_temp,
                                                                    n_actuators, omega_temporal_freqs)

    _, var_alias_CL, PSD_out_alias, PSD_in_alias = aliasing_variance(
        transf_funct=H_n_alias,
        actuators_number=n_actuators,
        omega_temp_freq_interval=omega_temporal_freqs,
        c_optg=c_optg,
        alpha=alpha_,
        telescope_diameter=telescope_diameter,
        seeing=seeing_,
        modulation_radius=modulation_radius,
        windspeed=wind_speed,
        maximum_radial_order_corrected=maximum_rad_order_corr,
        file_path_matrix_R=file_path_R1,
        file_path_sigma_slopes=file_sigma_slope,
    )

    _, var_meas_CL, PSD_out_meas, PSD_in_meas = measure_variance(
        F_excess_noise,
        x_pixel,
        sky_background,
        dark_current,
        readout_noise,
        phot_flux,
        telescope_diameter,
        frame_rate,
        magnitudo,
        n_subapert,
        file_path_R1,
        H_n_meas,
        n_actuators,
        omega_temporal_freqs,
        c_optg,
    )

    total_variance(var_fit, var_temp_atmo_CL + var_vibr_CL, var_alias_CL, var_meas_CL)

    result = {
        'var_fit':   float(np.real(var_fit)),
        'var_temp':  float(np.real(var_temp_atmo_CL)),
        'var_vibr':  float(np.real(var_vibr_CL)),
        'var_alias': float(np.real(var_alias_CL)),
        'var_meas':  float(np.real(var_meas_CL)),
        'var_total': float(np.real(
            var_fit + var_temp_atmo_CL + var_vibr_CL + var_alias_CL + var_meas_CL
        )),
    }

    if not display:
        return result

    var_temp_modes = integrate_modal_psd(PSD_out_temp_atmo, omega_temporal_freqs)
    var_vibr_modes = integrate_modal_psd(PSD_out_vibr, omega_temporal_freqs)
    var_alias_modes = integrate_modal_psd(PSD_out_alias, omega_temporal_freqs)
    var_meas_modes = integrate_modal_psd(PSD_out_meas, omega_temporal_freqs)
    n_modes_display = var_temp_modes.size
    var_fit_modes = np.full(n_modes_display, np.real(var_fit) / n_modes_display)
    

    summary_display(var_fit_modes, var_temp_modes, var_alias_modes, var_meas_modes,
                    PSD_out_temp_atmo, PSD_out_alias, PSD_out_meas,
                    omega_temporal_freqs, H_r_temp, H_n_meas,
                    PSD_input_atmos=PSD_in_temp_atmo, PSD_input_wind=PSD_in_vibr,
                    PSD_input_alias=PSD_in_alias, PSD_input_meas=PSD_in_meas,
                    var_vibr_modes=var_vibr_modes, PSD_out_vibr=PSD_out_vibr,
                    modes_to_plot=summary_modes_to_plot)

    if gain_sweeps is not None:
        plot_gain_optimization_sweep(gain_sweeps=gain_sweeps)

    return result


def main():
    parser = argparse.ArgumentParser(description="Semi-analytic Error Budget runner")
    parser.add_argument(
        "args",
        nargs="+",
        help="Use either: <yaml_file> or run <yaml_file>"
    )
    args = parser.parse_args()

    if len(args.args) == 1:
        run(args.args[0])
        return

    if len(args.args) == 2 and args.args[0] == "run":
        run(args.args[1])
        return

    parser.error("Usage: python scripts/main_sa.py <yaml_file> or"
                 " python scripts/main_sa.py run <yaml_file>")


if __name__ == "__main__":
    main()
