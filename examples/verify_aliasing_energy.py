#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy import integrate
from astropy.io import fits

from src.Functions import (
    DEFAULT_ALIASING_ALPHA,
    PSD_final_alias,
    compute_andes_optical_gain,
    load_parameters,
    read_sigma_slopes,
    double_interpolation_sigma_slope,
    extract_propagation_coefficients,
    radial_order_from_n_modes,
)

def verify_aliasing_energy():
    print("\n" + "="*80)
    print(" VERIFY ALIASING ENERGY CONSERVATION")
    print(" (Variance from Slopes vs PSD Integral SA vs PSD Integral PASSATA)")
    print("="*80 + "\n")

    # 1. Load Parameters
    param = load_parameters('params_mod4.yaml')
    D = param['telescope']['telescope_diam']
    seeing = param['atmosphere']['seeing']
    modulation_radius = param['wavefront_sensor']['modulation_radius']
    alpha = DEFAULT_ALIASING_ALPHA
    n_actuators = param['control']['n_modes']
    wind_speed = param['atmosphere']['wind_speed']
    max_rad_ord = radial_order_from_n_modes(n_actuators)

    file_mod0 = param['data']['optical_gain_models'][0]
    file_mod4 = param['data']['optical_gain_models'][1]
    file_reconstructor = param['data']['reconstruction_matrix']
    file_sigma_slopes = param['data']['sigma_slopes']

    t_0 = param['control']['sampling_time']
    frame_rate = 1.0 / t_0
    temporal_freqs = np.logspace(-3, np.log10(1.0 / (2.0 * t_0)), 1000)
    omega_sa = 2 * np.pi * temporal_freqs

    # 2. Compute Optical Gain
    c_optg = compute_andes_optical_gain(
        file_mod0,
        file_mod4,
        seeing,
        modulation_radius,
        actuators_number=n_actuators,
    )

    # --------------------------------------------------------------------------------
    # METHOD 1: Direct target variance
    # ---------------------------------------------------------------------------
    data_slopes = read_sigma_slopes(file_sigma_slopes)
    seeing_vals = data_slopes[0,0,:]                                           
    modal_radius_vals = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 8.0])

    sigma_slope_alias = double_interpolation_sigma_slope(
        modal_radius_vals, seeing_vals, data_slopes,
        modulation_radius, seeing
    )

    p_coeff = np.asarray(extract_propagation_coefficients(file_reconstructor), dtype=float).ravel()
    n_modes = min(n_actuators, p_coeff.size, c_optg.shape[0])
    p_coeff = p_coeff[:n_modes]
    c_optg_modes = np.squeeze(c_optg[:n_modes])

    var_from_slopes = (sigma_slope_alias ** 2) * p_coeff / (c_optg_modes ** 2)

    # --------------------------------------------------------------------------------
    # METHOD 2: Numerical integral of SA PSD
    # --------------------------------------------------------------------------------
    psd_sa = PSD_final_alias(
        c_optg=c_optg,
        actuators_number=n_modes,
        omega_temp_freq_interval=omega_sa,
        alpha=alpha,
        telescope_diameter=D,
        seeing=seeing,
        modulation_radius=modulation_radius,
        windspeed=wind_speed,
        maximum_radial_order_corrected=max_rad_ord,
        file_path_matrix_R=file_reconstructor,
        file_path_sigma_slopes=file_sigma_slopes,
    )
    var_from_psd = integrate.simpson(psd_sa, omega_sa, axis=-1)

    # --------------------------------------------------------------------------------
    # METHOD 3: Integral of empirical PASSATA PSD
    # --------------------------------------------------------------------------------
    file_psd_passata = "src/file_fits/ANDES/modal_psd_aliasing.fits"
    try:
        with fits.open(file_psd_passata) as hdul:
            psd_passata = hdul[0].data # pylint: disable=no-member

        freq_passata = np.linspace(0, frame_rate / 2.0, psd_passata.shape[1])
        omega_passata = 2 * np.pi * freq_passata
        var_passata = integrate.simpson(psd_passata, omega_passata, axis=-1)
        passata_available = True
    except FileNotFoundError:
        print(f"[!] File {file_psd_passata} not found. Skipping End-to-End comparison.")
        passata_available = False

    # --------------------------------------------------------------------------------
    # PRINT RESULTS
    # --------------------------------------------------------------------------------
    n_modes_report = min(len(var_from_slopes), len(var_from_psd))
    if passata_available:
        n_modes_report = min(n_modes_report, len(var_passata))

    print(f"shape var_from_psd: {var_from_psd.shape}, shape omega_sa: {omega_sa.shape}")
    print(f"{'Mode':<6} | {'Var from Slopes (SA)':<22} | {'Var PSD Integral (SA)':<22}"
          f" | {'Var PASSATA Integral':<22}")
    print("-" * 80)

    for i in range(min(10, n_modes_report)):
        pass_val = f"{var_passata[i]:.4e}" if passata_available else "N/A"
        print(f"{i:<6} | {var_from_slopes[i]:<22.4e} | {var_from_psd[i]:<22.4e} | {pass_val:<22}")

    print("-" * 80)
    sum_slopes = float(np.sum(var_from_slopes))
    sum_psd = float(np.sum(var_from_psd))
    sum_passata = f"{np.sum(var_passata):.4e}" if passata_available else "N/A"
    print(f"{'TOTAL':<6} | {sum_slopes:<22.4e} | {sum_psd:<22.4e}"
          f" | {sum_passata:<22}\n")

    return {
        "n_modes": int(n_modes),
        "var_from_slopes_total": sum_slopes,
        "var_from_psd_total": sum_psd,
        "passata_available": bool(passata_available),
    }


if __name__ == "__main__":
    verify_aliasing_energy()
