#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt

from src.Functions import (
    load_parameters, 
    turbulence_psd, 
    temporal_variance,
    aliasing_variance,
    measure_variance,
    build_transfer_function,
    compute_andes_optical_gain,
    funct_d2,
    radial_order_from_n_modes
)

def plot_system_psds(mode_index=0, plot_inputs=False):
    print(f"\nGenerating PSD plot for Mode {mode_index}...")

    # 1. Load parameters
    param = load_parameters('params_mod4.yaml')

    n_actuators = param['control']['n_modes']
    D = param['telescope']['telescope_diam']
    aperture_radius = D / 2.0
    aperture_center = [0, 0, 0]

    L0 = param['atmosphere']['outer_scale']
    layers_altitude = 0.0
    wind_direction = 0.0
    wind_speed = param['atmosphere']['wind_speed']
    seeing = param['atmosphere']['seeing']
    r0 = 0.98 * 500 / seeing

    F_excess_noise = np.sqrt(param['wavefront_sensor']['value_for_F_excess_noise'])
    sky_background = param['wavefront_sensor']['sky_backgr']
    dark_current = param['wavefront_sensor']['dark_curr']
    readout_noise = param['wavefront_sensor']['noise_readout']

    file_path_R1 = param['data']['reconstruction_matrix']
    file_mod0 = param['data']['optical_gain_models'][0]
    file_mod4 = param['data']['optical_gain_models'][1]
    sigma_slopes_path = param['data']['sigma_slopes']

    d1 = param['plant']['d_1']
    d3 = param['plant']['d_3']
    n1 = param['plant']['n_1']
    n2 = param['plant']['n_2']
    n3 = param['plant']['n_3']

    t_0 = param['control']['sampling_time']
    frame_rate = 1.0 / t_0
    temporal_freqs = np.logspace(-3, np.log10(frame_rate / 2.0), 1000)
    omega = 2 * np.pi * temporal_freqs

    spatial_freqs = np.logspace(-4, 4, 100)

    T_tot = param['control']['total_delay']
    modulation_radius = param['wavefront_sensor']['modulation_radius']
    maximum_radial_order = radial_order_from_n_modes(n_actuators)
    alpha_ = -17 / 3

    phot_flux = float(param['guide_star']['flux_photons'])
    magnitude = param['guide_star']['magn']
    n_subapert = param['wavefront_sensor']['number_of_sub']
    collecting_area = param['telescope']['collect_area']
    x_pixel = param['control']['slope_computer_weights']

    gain_value = param['control'].get('gain_value')
    if gain_value is not None:
        gain_array = np.full(n_actuators, float(np.asarray(gain_value).ravel()[0]))
    else:
        gain_array = np.full(n_actuators, float(param['control']['gain_min']))

    d2 = funct_d2(T_tot)

    # 2. Generate Atmospheric Input
    PSD_atmosf = turbulence_psd(0, 0, aperture_radius, aperture_center, r0, L0, layers_altitude,
                                wind_speed, wind_direction, spatial_freqs, temporal_freqs, n_modes=n_actuators)
    PSD_vibration_zeros = np.zeros_like(PSD_atmosf)

    # 3. Build Transfer Functions
    plant_num = np.polymul(np.polymul(np.asarray(n1), np.asarray(n2)), np.asarray(n3))
    plant_den = np.polymul(np.polymul(np.asarray(d1), np.asarray(d2)), np.asarray(d3))
    H_r, H_n = build_transfer_function(
        omega,
        t_0,
        n_actuators,
        plant_num,
        plant_den,
        gain=gain_array,
    )

    # 4. Compute Optical Gain (needed for aliasing)
    c_optg = compute_andes_optical_gain(file_mod0, file_mod4, seeing, modulation_radius)

    # 5. Extract PSDs from separate functions
    _, PSD_out_temp, PSD_in_temp = temporal_variance(
        PSD_atmosf, PSD_vibration_zeros, H_r, n_actuators, omega
    )

    _, PSD_out_alias, PSD_in_alias = aliasing_variance(
        H_n, n_actuators, omega, alpha_, D, seeing, modulation_radius, wind_speed,
        maximum_radial_order, file_path_R1, c_optg, sigma_slopes_path
    )

    _, PSD_out_meas, PSD_in_meas = measure_variance(
        F_excess_noise, x_pixel, sky_background, dark_current, readout_noise,
        phot_flux, D, frame_rate, magnitude, n_subapert, collecting_area,
        file_path_R1, omega, H_n, n_actuators
    )

    if mode_index < 0 or mode_index >= n_actuators:
        raise ValueError(f"mode_index must be between 0 and {n_actuators - 1}")

    PSD_in_temp = np.real(PSD_in_temp)
    PSD_in_alias = np.real(PSD_in_alias)
    PSD_in_meas = np.real(PSD_in_meas)
    PSD_out_temp = np.real(PSD_out_temp)
    PSD_out_alias = np.real(PSD_out_alias)
    PSD_out_meas = np.real(PSD_out_meas)

    # Variance calculation
    var_temp_in = integrate.simpson(PSD_in_temp[mode_index, :], temporal_freqs)
    var_alias_in = integrate.simpson(PSD_in_alias[mode_index, :], temporal_freqs)
    var_meas_in = integrate.simpson(PSD_in_meas[mode_index, :], temporal_freqs)
    var_temp_out = integrate.simpson(PSD_out_temp[mode_index, :], temporal_freqs)
    var_alias_out = integrate.simpson(PSD_out_alias[mode_index, :], temporal_freqs)
    var_meas_out = integrate.simpson(PSD_out_meas[mode_index, :], temporal_freqs)

    # Print variances for debug
    print(f"Turbulence Variance (Input):    {var_temp_in:.2e} nm²")
    print(f"Aliasing Variance (Input):      {var_alias_in:.2e} nm²")
    print(f"Noise Variance (Input):         {var_meas_in:.2e} nm²")
    print(f"Turbulence Variance (Output):   {var_temp_out:.2e} nm²")
    print(f"Aliasing Variance (Output):     {var_alias_out:.2e} nm²")
    print(f"Noise Variance (Output):        {var_meas_out:.2e} nm²")

    # 6. Create Plot
    plt.figure(figsize=(12, 7))
    freqs = temporal_freqs

    if plot_inputs:
        plt.loglog(freqs, PSD_in_temp[mode_index, :], label='Turbulence (Open Loop)',
                   color='tab:blue', linestyle='--', alpha=0.6)
        plt.loglog(freqs, PSD_in_alias[mode_index, :], label='Aliasing (Open Loop)',
                   color='tab:orange', linestyle='--', alpha=0.6)
        plt.loglog(freqs, PSD_in_meas[mode_index, :], label='Noise (Open Loop)',
                   color='tab:green', linestyle='--', alpha=0.6)

    plt.loglog(freqs, PSD_out_temp[mode_index, :], label='Turbulence Residual (Servo-lag)',
               color='tab:blue', linewidth=2.5)
    plt.loglog(freqs, PSD_out_alias[mode_index, :], label='Aliasing Residual',
               color='tab:orange', linewidth=2.5)
    plt.loglog(freqs, PSD_out_meas[mode_index, :], label='Noise Residual',
               color='tab:green', linewidth=2.5)

    psd_total = (PSD_out_temp[mode_index, :] +
                 PSD_out_alias[mode_index, :] +
                 PSD_out_meas[mode_index, :])

    plt.loglog(freqs, psd_total, label='Total PSD (Sum)',
               color='black', linewidth=3, linestyle='-.')

    plt.title(f"Spectral Analysis (PSD) in Closed Loop - Zernike Mode {mode_index}"
              f"\nLoop Gain = {gain_array[0]:.2f}", fontsize=14)
    plt.xlabel("Temporal Frequency [Hz]", fontsize=12)
    plt.ylabel("Power Spectral Density [nm² / Hz]", fontsize=12)
    plt.grid(True, which="both", linestyle=":", alpha=0.7)
    plt.legend(loc='lower left', fontsize=11)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_system_psds(mode_index=0, plot_inputs=True)
