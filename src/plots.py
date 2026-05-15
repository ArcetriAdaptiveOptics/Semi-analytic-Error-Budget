#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 12:11:30 2026

@author: greta
"""

# pylint: disable=C


import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from astropy.io import fits                                                    

from src.Functions import total_variance
from src.Functions import extract_propagation_coefficients
from src.Functions import PSD_final_alias
from src.Functions import double_interpolation_sigma_slope
from src.Functions import read_sigma_slopes
from src.Functions import load_PSD_windshake

from src.Functions import fitting_variance
from src.Functions import temporal_variance
from src.Functions import aliasing_variance
from src.Functions import measure_variance
from src.Functions import build_transfer_function
from src.Functions import interpolate_and_normalize_psd
from src.Functions import align_psd_modes
from src.Functions import final_soul_optical_gain
from src.Functions import compute_optical_gain
from src.Functions import _integrate_modal_psd


# Function to compute the total residual variance for a set of gain values,
# by combining fitting, temporal, aliasing and measurement error contributions 
  
def variance_total_for_test(actuators_number, gain_values, omega_temp_freq_interval, t_freqs, f,
                            t_0, plant_num, plant_den, telescope_diameter, fried_parameter,
                            excess_noise_factor, sky_background, dark_current, readout_noise,
                            photon_flux, frame_rate, magnitude, n_subaperture, collecting_area,
                            slope_computer_weights, fitting_coeff, alpha, seeing, modulation_radius,
                            wind_speed, maximum_radial_order_corrected, reconstruction_matrix_path,
                            psd_turbulence, psd_windshake, sigma_slopes_path):
    
                                                            
    tot_variance = np.zeros_like(gain_values, dtype=float)
    
    for i in range(len(gain_values)):
        
        g = gain_values[i]
        gain_val = np.array([g])                                               

###########################       
        
        H_r_temp, H_n_meas = build_transfer_function(
            omega_temp_freq_interval,
            t_0,
            actuators_number,
            plant_num,
            plant_den,
            gain=gain_val,
        )
        H_n_alias = H_n_meas
        
        
        variance_fit = fitting_variance(fitting_coeff, actuators_number, telescope_diameter, fried_parameter)
        
         
        if np.array_equal(t_freqs, f): 
            
            _, variance_temporal,_ , _ = temporal_variance(psd_turbulence, psd_windshake, H_r_temp, actuators_number,
                                                              omega_temp_freq_interval)

        else: 
            
            PSD_wind_vib_interp_norm = interpolate_and_normalize_psd(t_freqs, f, psd_windshake, actuators_number)
            _, variance_temporal,_ , _ = temporal_variance(psd_turbulence, PSD_wind_vib_interp_norm,
                                                           H_r_temp, actuators_number, omega_temp_freq_interval)

        
        
        
        _, variance_aliasing, _, _ = aliasing_variance(
            transf_funct=H_n_alias,
            actuators_number=actuators_number,
            omega_temp_freq_interval=omega_temp_freq_interval,
            c_optg=gain_val,
            alpha=alpha,
            telescope_diameter=telescope_diameter,
            seeing=seeing,
            modulation_radius=modulation_radius,
            windspeed=wind_speed,
            maximum_radial_order_corrected=maximum_radial_order_corrected,
            file_path_matrix_R=reconstruction_matrix_path,
            file_path_sigma_slopes=sigma_slopes_path,
        )

        _, variance_measurement, _, _ = measure_variance(
            excess_noise_factor,
            slope_computer_weights,
            sky_background,
            dark_current,
            readout_noise,
            photon_flux,
            telescope_diameter,
            frame_rate,
            magnitude,
            n_subaperture,
            collecting_area,
            reconstruction_matrix_path,
            H_n_meas,
            actuators_number,
            omega_temp_freq_interval,
            gain_val, verbose=False
        )
       
        tot_variance[i] = total_variance(np.real(variance_fit), np.real(variance_temporal), 
                                         np.real(variance_aliasing), np.real(variance_measurement))            
    
    return tot_variance

        
# Defines a function that allows, when needed, to plot PSD_in, PSD_out, and the transfer 
# function for the variances (temp, alias, meas)

def plot(f, H_r_t, H_n_m, H_n_a, PSD_in_v, PSD_out_v, PSD_in_t, PSD_out_t, PSD_in_m, PSD_out_m, PSD_in_a, PSD_out_a):
    
    PSD_in = [PSD_in_v, PSD_in_t, PSD_in_m, PSD_in_a]                  
    PSD_out = [PSD_out_v, PSD_out_t, PSD_out_m, PSD_out_a]              
    H = [H_r_t, H_r_t, H_n_m, H_n_a]
       
    labels_PSD = ["vibr", "temp", "meas","alias"]
    labels_H = ["r", "r", "n", "n"]
       
    for i in range(len(PSD_in)):                                                           
          
        plt.loglog(f, PSD_in[i][0, :], label=f"PSD_in_{labels_PSD[i]} (mode 0)")       
        plt.loglog(f, PSD_out[i][0, :], label=f"PSD_out_{labels_PSD[i]} (mode 0)")
        plt.xlabel("Frequency [rad/s]")
        plt.ylabel("PSD")
        plt.title(f"PSD {labels_PSD[i]} (modo 0)")
        plt.legend()
        plt.grid()
        plt.show()
           
        plt.loglog(f, np.abs(H[i][0, :])**2)     
        plt.xlabel("Frequency [rad/s]")
        plt.ylabel(f"|H_{labels_H[i]}|^2")
        plt.title(f"Transfert function H_{labels_H[i]} (mode 0)")
        plt.grid()
        plt.show()
       
        
# Function to plot the three output PSDs (temporal, measurement and aliasing)
# for mode 0 on the same graph.
       
def plot_all_PSD(f, PSD_out_t, PSD_out_m, PSD_out_a, PSD_out_v=None):

    PSD_out = [PSD_out_t, PSD_out_m, PSD_out_a]
    labels = ["temp", "meas", "alias"]

    if PSD_out_v is not None:
        PSD_out.append(PSD_out_v)
        labels.append("vibr")

    for i in range(len(PSD_out)):

        plt.loglog(f, PSD_out[i][0, :], label=f"PSD_out_{labels[i]} (mode 0)")

    plt.xlabel("Frequency [rad/s]")
    plt.ylabel("PSD")
    plt.title("PSDs – mode 0")
    plt.legend()
    plt.grid()

    plt.show()


# Function to print a compact closed-loop summary and generate a basic set of plots:
# (1) variance vs mode, (2) total CL PSD, (3) CL PSD for selected modes,
# (4) transfer functions for selected modes.

def summary_display(var_fit_modes, var_temp_modes, var_alias_modes, var_meas_modes,
                    PSD_out_temp, PSD_out_alias, PSD_out_meas, frequencies,
                    H_r_temp, H_n_meas, PSD_input_atmos=None, PSD_input_wind=None,
                    PSD_input_alias=None, PSD_input_meas=None,
                    modes_to_plot=None, var_vibr_modes=None, PSD_out_vibr=None):

    frequencies = np.asarray(frequencies, dtype=float).ravel()
    n_frequencies = frequencies.size

    if n_frequencies == 0:
        raise ValueError("Frequency vector must not be empty")

    def _as_mode_frequency_matrix(array_in, array_name):
        array_out = np.asarray(array_in)

        if array_out.ndim != 2:
            raise ValueError(f"{array_name} must be a 2D array")

        if array_out.shape[1] == n_frequencies:
            return array_out

        if array_out.shape[0] == n_frequencies:
            return array_out.T

        raise ValueError(
            f"{array_name} has shape {array_out.shape}; expected (N_modes, N_frequencies) "
            f"or (N_frequencies, N_modes) with N_frequencies={n_frequencies}"
        )

    PSD_out_temp = _as_mode_frequency_matrix(PSD_out_temp, "PSD_out_temp")
    PSD_out_alias = _as_mode_frequency_matrix(PSD_out_alias, "PSD_out_alias")
    PSD_out_meas = _as_mode_frequency_matrix(PSD_out_meas, "PSD_out_meas")
    H_r_temp = _as_mode_frequency_matrix(H_r_temp, "H_r_temp")
    H_n_meas = _as_mode_frequency_matrix(H_n_meas, "H_n_meas")

    if PSD_out_vibr is not None:
        PSD_out_vibr = _as_mode_frequency_matrix(PSD_out_vibr, "PSD_out_vibr")

    PSD_out_temp = np.real_if_close(PSD_out_temp, tol=1000)
    PSD_out_alias = np.real_if_close(PSD_out_alias, tol=1000)
    PSD_out_meas = np.real_if_close(PSD_out_meas, tol=1000)

    if np.iscomplexobj(PSD_out_temp):
        PSD_out_temp = np.real(PSD_out_temp)
    if np.iscomplexobj(PSD_out_alias):
        PSD_out_alias = np.real(PSD_out_alias)
    if np.iscomplexobj(PSD_out_meas):
        PSD_out_meas = np.real(PSD_out_meas)
    if PSD_out_vibr is not None:
        PSD_out_vibr = np.real_if_close(PSD_out_vibr, tol=1000)
        if np.iscomplexobj(PSD_out_vibr):
            PSD_out_vibr = np.real(PSD_out_vibr)

    n_modes = PSD_out_temp.shape[0]

    if n_modes == 0:
        raise ValueError("Empty modal vectors are not allowed")

    def _as_mode_vector(vector_in, vector_name):
        vector_out = np.real_if_close(np.asarray(vector_in), tol=1000).ravel()

        if np.iscomplexobj(vector_out):
            vector_out = np.real(vector_out)

        vector_out = vector_out.astype(float)

        if vector_out.size == 1:
            return np.full(n_modes, float(vector_out.item()))

        if vector_out.size == n_modes:
            return vector_out

        raise ValueError(
            f"{vector_name} length is {vector_out.size}, expected 1 or N_modes={n_modes}"
        )

    var_fit_modes = _as_mode_vector(var_fit_modes, "var_fit_modes")
    var_temp_modes = _as_mode_vector(var_temp_modes, "var_temp_modes")
    var_alias_modes = _as_mode_vector(var_alias_modes, "var_alias_modes")
    var_meas_modes = _as_mode_vector(var_meas_modes, "var_meas_modes")

    if var_vibr_modes is not None:
        var_vibr_modes = _as_mode_vector(var_vibr_modes, "var_vibr_modes")

    vectors = [var_fit_modes, var_temp_modes, var_alias_modes, var_meas_modes]

    for vec in vectors:
        if vec.size != n_modes:
            raise ValueError("All variance vectors must have length N_modes")

    expected_shape = (n_modes, frequencies.size)

    arrays_2d = [PSD_out_temp, PSD_out_alias, PSD_out_meas, H_r_temp, H_n_meas]

    if PSD_out_vibr is not None:
        arrays_2d.append(PSD_out_vibr)

    for arr in arrays_2d:
        if arr.shape != expected_shape:
            raise ValueError("PSD and transfer-function arrays must have shape (N_modes, N_frequencies)")

    H_n_alias = H_n_meas

    def _process_optional_input_psd(array_in, array_name):
        if array_in is None:
            return None

        try:
            array_out = np.asarray(array_in, dtype=float)
            array_out = _as_mode_frequency_matrix(array_out, array_name)
            array_out = align_psd_modes(array_out, n_modes)
            array_out = np.real_if_close(array_out, tol=1000)
            if np.iscomplexobj(array_out):
                array_out = np.real(array_out)
            return array_out
        except (TypeError, ValueError):
            return None

    PSD_input_atmos_proc = _process_optional_input_psd(PSD_input_atmos, "PSD_input_atmos")
    PSD_input_wind_proc = _process_optional_input_psd(PSD_input_wind, "PSD_input_wind")
    PSD_input_alias_proc = _process_optional_input_psd(PSD_input_alias, "PSD_input_alias")
    PSD_input_meas_proc = _process_optional_input_psd(PSD_input_meas, "PSD_input_meas")

    PSD_cl_total = PSD_out_temp + PSD_out_alias + PSD_out_meas
    if PSD_out_vibr is not None:
        PSD_cl_total = PSD_cl_total + PSD_out_vibr
    PSD_cl_total_all_modes = np.sum(PSD_cl_total, axis=0)

    if any(psd is not None for psd in (
        PSD_input_atmos_proc,
        PSD_input_wind_proc,
        PSD_input_alias_proc,
        PSD_input_meas_proc,
    )):
        input_PSD_available = True
        PSD_input_total = np.zeros_like(PSD_cl_total)
        if PSD_input_atmos_proc is not None:
            PSD_input_total += PSD_input_atmos_proc
        if PSD_input_wind_proc is not None:
            PSD_input_total += PSD_input_wind_proc
        if PSD_input_alias_proc is not None:
            PSD_input_total += PSD_input_alias_proc
        if PSD_input_meas_proc is not None:
            PSD_input_total += PSD_input_meas_proc
        PSD_input_total_all_modes = np.zeros(frequencies.size)
        if PSD_input_atmos_proc is not None:
            PSD_input_total_all_modes += np.sum(PSD_input_atmos_proc, axis=0)
        if PSD_input_wind_proc is not None:
            PSD_input_total_all_modes += np.sum(PSD_input_wind_proc, axis=0)
        if PSD_input_alias_proc is not None:
            PSD_input_total_all_modes += np.sum(PSD_input_alias_proc, axis=0)
        if PSD_input_meas_proc is not None:
            PSD_input_total_all_modes += np.sum(PSD_input_meas_proc, axis=0)
    else:
        input_PSD_available = False

    var_total_modes = var_fit_modes + var_temp_modes + var_alias_modes + var_meas_modes
    if var_vibr_modes is not None:
        var_total_modes = var_total_modes + var_vibr_modes

    fit_total = np.sum(var_fit_modes)
    temp_total = np.sum(var_temp_modes)
    alias_total = np.sum(var_alias_modes)
    meas_total = np.sum(var_meas_modes)
    vibr_total = np.sum(var_vibr_modes) if var_vibr_modes is not None else None
    total_variance_sum = np.sum(var_total_modes)

    print("\n===== CLOSED-LOOP SUMMARY =====")
    print(f"N_modes: {n_modes}")
    print(f"N_frequencies: {frequencies.size}")
    print(f"Fitting total variance [nm^2]:      {fit_total:.6e}")
    print(f"Temporal total variance [nm^2]:     {temp_total:.6e}")
    print(f"Aliasing total variance [nm^2]:     {alias_total:.6e}")
    print(f"Measurement total variance [nm^2]:  {meas_total:.6e}")
    if vibr_total is not None:
        print(f"Vibration total variance [nm^2]:   {vibr_total:.6e} (separate term)")
    print(f"Total output variance [nm^2]:       {total_variance_sum:.6e}")

    if modes_to_plot is None:
        modes_to_plot = [0, n_modes // 2, n_modes - 1]

    mode_indices = np.array(modes_to_plot, dtype=int)
    mode_indices = mode_indices[(mode_indices >= 0) & (mode_indices < n_modes)]
    mode_indices = np.unique(mode_indices)

    if mode_indices.size == 0:
        mode_indices = np.array([0], dtype=int)

    mode_axis = np.arange(n_modes)+1

    # 1. Variance vs mode
    plt.figure()
    plt.plot(mode_axis, var_total_modes, label="total")
    plt.plot(mode_axis, var_fit_modes, label="fitting")
    plt.plot(mode_axis, var_temp_modes, label="temporal")
    plt.plot(mode_axis, var_alias_modes, label="aliasing")
    plt.plot(mode_axis, var_meas_modes, label="measurement")
    if var_vibr_modes is not None:
        plt.plot(mode_axis, var_vibr_modes, '--', label="vibration")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Mode index")
    plt.ylabel("Variance [nm^2]")
    plt.title("Variance vs mode")
    plt.grid()
    plt.legend()

    # 2. Total CL and OL PSD
    plt.figure()
    plt.loglog(frequencies, PSD_cl_total_all_modes, label="Closed-loop total PSD")
    if input_PSD_available:
        plt.loglog(frequencies, PSD_input_total_all_modes, 'k--', linewidth=2, label="Total input PSD")
    plt.xlabel("Frequency [rad/s]")
    plt.ylabel("PSD [nm^2/Hz]")
    plt.title("Closed-loop total PSD")
    plt.legend()
    plt.grid()

    # 3. CL and OL PSD for selected modes
    plt.figure()
    for mode in mode_indices:
        plt.loglog(frequencies, PSD_cl_total[mode, :], label=f"mode {mode}")
        if input_PSD_available:
            plt.loglog(frequencies, PSD_input_total[mode, :], '--', label=f"input mode {mode}")
    plt.xlabel("Frequency [rad/s]")
    plt.ylabel("PSD [nm^2/Hz]")
    plt.title("Closed-loop PSD for selected modes")
    plt.grid()
    plt.legend()

    # 4. CL and OL PSD components for selected modes
    # one figure per mode, with temp, alias, meas, vibr (if available) in different colors and line styles
    for mode in mode_indices:
        plt.figure()
        plt.loglog(frequencies, PSD_out_temp[mode, :], 'k', label=f"temp mode {mode}")
        plt.loglog(frequencies, PSD_out_alias[mode, :], 'b', label=f"alias mode {mode}")
        plt.loglog(frequencies, PSD_out_meas[mode, :], 'g', label=f"meas mode {mode}")
        if var_vibr_modes is not None and PSD_out_vibr is not None:
            plt.loglog(frequencies, PSD_out_vibr[mode, :], 'r', label=f"vibr mode {mode}")
        if input_PSD_available:
            if PSD_input_atmos_proc is not None:
                plt.loglog(frequencies, PSD_input_atmos_proc[mode, :], 'k--', label=f"atmos input mode {mode}")
            if PSD_input_wind_proc is not None:
                plt.loglog(frequencies, PSD_input_wind_proc[mode, :], 'r--', label=f"wind input mode {mode}")
            if PSD_input_alias_proc is not None:
                plt.loglog(frequencies, PSD_input_alias_proc[mode, :], 'b--', label=f"alias input mode {mode}")
            if PSD_input_meas_proc is not None:
                plt.loglog(frequencies, PSD_input_meas_proc[mode, :], 'g--', label=f"meas input mode {mode}")
        plt.xlabel("Frequency [rad/s]")
        plt.ylabel("PSD [nm^2/Hz]")
        plt.title("Closed-loop PSD components for selected modes")
        plt.grid()
        plt.legend()

    # 5. Transfer functions for selected modes
    plt.figure()
    for mode in mode_indices:
        plt.loglog(frequencies, np.abs(H_r_temp[mode, :])**2, label=f"|H_r|² mode {mode}")
        plt.loglog(frequencies, np.abs(H_n_meas[mode, :])**2, '--', label=f"|H_n|² mode {mode}")
    plt.xlabel("Frequency [rad/s]")
    plt.ylabel("Magnitude squared")
    plt.title("Transfer functions for selected modes")
    plt.grid()
    plt.legend()

    plt.show()

    return {
        "var_total_modes": var_total_modes,
        "totals": {
            "fit": fit_total,
            "temp": temp_total,
            "alias": alias_total,
            "meas": meas_total,
            "vibr": vibr_total,
            "total": total_variance_sum
        },
        "modes_plotted": mode_indices,
        "H_n_alias": H_n_alias
    }


# Function to verify the consistency of the aliasing variance calculation.
# It computes the analytical aliasing variance (summing the selected modes)
# and compares it with the variance obtained by integrating the corresponding 
# aliasing PSD. 
# The function also print the variance of the first mode alone in both cases.
    
def check(reconstruction_matrix_path, telescope_diameter, seeing, target_modulation_radius,
          actuators_number, alpha, omega_temp_freq_interval, wind_speed,
          maximum_radial_order_corrected, magnitudo, c_optg, 
          file_path_sigma_slopes):

    print("\nALIASING VARIANCE CHECK (check):\n")
    
    p_coefficient = extract_propagation_coefficients(reconstruction_matrix_path)
    
    if p_coefficient is None:   
                                              
        raise RuntimeError("Propagation coefficients not loaded") 
   
    print("Propagation coefficients loaded successfully.")

    data_slopes, seeing_vals, modal_radius_vals = read_sigma_slopes(file_path_sigma_slopes)
    
    sigma_slope_alias = double_interpolation_sigma_slope(modal_radius_vals, seeing_vals, data_slopes, 
                                                         target_modulation_radius, seeing)
  
    sigma_alias_2_two_modes = 0.0
    
    for i in range (actuators_number):
        
        sigma_alias_2 = p_coefficient[i] * (sigma_slope_alias ** 2) / c_optg[i] ** 2
        
        sigma_alias_2_two_modes += sigma_alias_2
        
    print("\nALIASING VARIANCE (OPEN LOOP):", sigma_alias_2_two_modes)
    
    PSD_al = PSD_final_alias(
        c_optg,
        actuators_number,
        omega_temp_freq_interval,
        telescope_diameter,
        seeing,
        target_modulation_radius,
        wind_speed,
        maximum_radial_order_corrected,
        reconstruction_matrix_path,
        alpha=alpha,
        file_path_sigma_slopes=None,
    )
    
    integral_per_mode = integrate.simpson(PSD_al, omega_temp_freq_interval)
    sigma_alias_2_PSD_total = np.sum(integral_per_mode)
    
    print("ALIASING VARIANCE FROM PSD (OPEN LOOP):", sigma_alias_2_PSD_total)
    
    sigma_alias_2_one_mode = p_coefficient[0] * (sigma_slope_alias ** 2) / c_optg[0] ** 2
    
    print("ALIASING VARIANCE ONE MODE (OPEN LOOP):", sigma_alias_2_one_mode)
    
    print("ALIASING VARIANCE FROM PSD ONE MODE (OPEN LOOP):", integral_per_mode[0], "\n")


# Function to compare the mode 0 aliasing PSD from data files with the one computed 
    
def plot_PSD_alias_mode_0 (actuators_number, omega_temp_freq_interval, alpha, telescope_diameter,
                           seeing, target_modulation_radius, wind_speed, maximum_radial_order_corrected,
                           magnitudo,reconstruction_matrix_path, c_optg, sigma_slopes_path,
                           modal_psd_aliasing_path):

    
    with fits.open(modal_psd_aliasing_path) as hdul:
        data = hdul[0].data # pylint: disable=no-member
        
        freq_hz = data[:, 0]
        mode_0 = data[:, 1]
        
        freq_rad_s = 2 * np.pi * freq_hz
        
       
        PSD_aliasing_mode0_given = mode_0 / (c_optg[0] ** 2 * 2 * np.pi)             
        
    PSD_alising_mine = PSD_final_alias(
        c_optg,
        actuators_number,
        omega_temp_freq_interval,
        telescope_diameter,
        seeing,
        target_modulation_radius,
        wind_speed,
        maximum_radial_order_corrected,
        reconstruction_matrix_path,
        alpha=alpha,
        file_path_sigma_slopes=sigma_slopes_path,
    )
    
    
    PSD_alising_mine_mode0 = PSD_alising_mine[0,:]
    
    PSD_aliasing_mode0_given_interp = np.interp(omega_temp_freq_interval, freq_rad_s, 
                                                PSD_aliasing_mode0_given, left=0, right=0)
    
    # diff = PSD_alising_mine_mode0 - PSD_aliasing_mode0_given_interp
    ratio = PSD_alising_mine_mode0 / PSD_aliasing_mode0_given_interp

    plt.figure()
    plt.subplot(2,1,1)
    plt.loglog(omega_temp_freq_interval, PSD_alising_mine_mode0, label="PSD alias mine mode 0")
    plt.loglog(omega_temp_freq_interval, PSD_aliasing_mode0_given_interp, label="PSD alias data mode 0")
    plt.ylabel("PSD")
    plt.legend()
    plt.grid()
    
    plt.subplot(2,1,2)
    plt.semilogx(omega_temp_freq_interval, ratio)
    plt.xlabel("Frequency [rad/s]")
    plt.ylabel("Ratio (mine/data)")
    plt.grid()
    plt.show()
    
    
# Function to compute and plot the total open-loop and closed-loop PSD (mode 0) 
# by summing temporal, aliasing, and measurement contributions.

def plot_PSD_OL_CL_mode_0 (gain, omega_temp_freq_interval, t_0, actuators_number, num1, num2, num3, den1, den2, den3,
                           PSD_atmo_turb, PSD_vibration, alpha, telescope_diameter, seeing, modulation_radius, windspeed, 
                           maximum_radial_order_corrected, c_optg, F_excess, pixel_pos, sky_bkg, dark_curr, read_out_noise, 
                           photon_flux,frame_rate, magnitudo, n_subaperture, collecting_area, temporal_frequencies, frequencies, 
                           file_path_matrix_R, file_path_sigma_slopes):

    plant_num = np.polymul(np.polymul(np.asarray(num1), np.asarray(num2)), np.asarray(num3))
    plant_den = np.polymul(np.polymul(np.asarray(den1), np.asarray(den2)), np.asarray(den3))

    H_r, H_n = build_transfer_function(
        omega_temp_freq_interval,
        t_0,
        actuators_number,
        plant_num,
        plant_den,
        gain=gain,
    )
    
    if np.array_equal(temporal_frequencies, frequencies):
    
        _, _, PSD_output_temp, PSD_input_temp = temporal_variance (PSD_atmo_turb, PSD_vibration, H_r,  
                                                                   actuators_number, omega_temp_freq_interval)
        
    else:
        
        PSD_wind_vib_interp_normalized = interpolate_and_normalize_psd(temporal_frequencies, frequencies, PSD_vibration, actuators_number)
        _, _, PSD_output_temp, PSD_input_temp = temporal_variance (PSD_atmo_turb, PSD_wind_vib_interp_normalized, 
                                                                                 H_r, actuators_number, omega_temp_freq_interval)
        
        
    
    _, _, PSD_output_alias, PSD_input_alias = aliasing_variance(
        transf_funct=H_n,
        actuators_number=actuators_number,
        omega_temp_freq_interval=omega_temp_freq_interval,
        c_optg=c_optg,
        alpha=alpha,
        telescope_diameter=telescope_diameter,
        seeing=seeing,
        modulation_radius=modulation_radius,
        windspeed=windspeed,
        maximum_radial_order_corrected=maximum_radial_order_corrected,
        file_path_matrix_R=file_path_matrix_R,
        file_path_sigma_slopes=file_path_sigma_slopes,
    )

    _, _, PSD_output_meas, PSD_input_meas = measure_variance(
        F_excess,
        pixel_pos,
        sky_bkg,
        dark_curr,
        read_out_noise,
        photon_flux,
        telescope_diameter,
        frame_rate,
        magnitudo,
        n_subaperture,
        collecting_area,
        file_path_matrix_R,
        H_n,
        actuators_number,
        omega_temp_freq_interval,
        c_optg, verbose=False
    )
    
    
    PSD_total_input_mode0 = PSD_input_temp[0] + PSD_input_alias[0] + PSD_input_meas[0]
    
    PSD_total_output_mode0 = PSD_output_temp[0] + PSD_output_alias[0] + PSD_output_meas[0]

    plt.loglog(omega_temp_freq_interval, PSD_total_input_mode0, label = "PSD total mode 0 - OPEN LOOP")  
    plt.loglog(omega_temp_freq_interval, PSD_total_output_mode0, label = "PSD total mode 0 - CLOSED LOOP")    
    plt.xlabel('Frequency[rad/s]')
    plt.ylabel('Total PSD')
    plt.title('Total PSD (PSD temp + PSD alias + PSD meas) mode 0')
    plt.legend()
    plt.grid()
    plt.show()


# Plot to visualize the shape of the PSD vibration

def plot_psd_vibr_soul (file_path_wind):
    
    freq, PSD_vibr = load_PSD_windshake(file_path_wind)
    
    PSD_vibr = PSD_vibr/(2 * np.pi)
   
    freq = 2 * np.pi * freq
  
    #print ("PSD_shape:", PSD_vibr.shape)
    #print ("Freq_shape:", freq.shape)
    
    plt.loglog(freq, PSD_vibr[0,:], label = "psd vibrations mode 0")  
       
    plt.xlabel('Frequency[rad/s]')
    plt.ylabel('PSD')
    plt.title('PSD_vibr mode 0')
    plt.grid()
    plt.legend()
    plt.show()
    
 
# Function to plot the total residual variance of the system as a function 
# of the gain, showing separate curves for orthogonal blocks (e.g. Tip-Tilt vs Higher Orders)

def plot_gain_optimization_sweep(gain_vals_TT, var_TT, gain_vals_HO=None, var_HO=None):
    
    plt.figure(figsize=(9, 6))
    
    plt.plot(gain_vals_TT, var_TT, linewidth=2, label='Tip-Tilt Block (Modes 0-1)')
    
    # Se abbiamo calcolato anche gli alti ordini, li disegniamo
    if gain_vals_HO is not None and var_HO is not None:
        plt.plot(gain_vals_HO, var_HO, linewidth=2, label='Higher Orders Block')
        
    plt.yscale('log')
    plt.xlabel('Integrator Gain')
    plt.ylabel('Total Variance [nm^2]')
    plt.title('Gain Optimization Sweep')
    plt.grid(True, which='both', linestyle=':', alpha=0.7)
    plt.legend()
    plt.show()
 
 
# Function to compare the optical gain values obtained from the two different 
# functions to compute optg in SOUL case.

def optg_soul_comparison (file_soul_optical_gain_cube, target_binning, 
                          target_magnitude, actuators_number, file_mod0, 
                          file_mod1, target_seeing, target_modulation_radius):   
    
    
    optg_cube = final_soul_optical_gain(file_soul_optical_gain_cube, target_binning, 
                                     target_magnitude, actuators_number)
    
    optg_grid = compute_optical_gain(file_mod0, file_mod1, target_seeing, target_modulation_radius, 
                                  actuators_number)
    
    n_modes = np.arange(actuators_number)

    plt.plot(n_modes, optg_cube, label = "Optical gain with cube")
    plt.plot(n_modes, optg_grid, label = "Optical gain with grid")
    plt.xlabel('Modes')
    plt.ylabel('Gain')
    plt.title('OPTG Version 1 vs 2')
    plt.grid()
    plt.legend()
    plt.show()
    
   
# Function to compute and plot modal variances by integrating PSD components over 
# frequency and combining them into total variance per mode.

    
def plot_variance_vs_modes(PSD_out_temp, PSD_out_vibr, PSD_out_alias, PSD_out_meas, 
                           var_fit, omega_temporal_freqs, actuators_number): 

    var_temp_modes = _integrate_modal_psd(PSD_out_temp, omega_temporal_freqs)
    var_vibr_modes = _integrate_modal_psd(PSD_out_vibr, omega_temporal_freqs)
    var_alias_modes = _integrate_modal_psd(PSD_out_alias, omega_temporal_freqs)
    var_meas_modes = _integrate_modal_psd(PSD_out_meas, omega_temporal_freqs)
    n_modes = actuators_number
    var_fit_modes = np.full(n_modes, np.real(var_fit) / n_modes)
    
    var_total_modes = var_temp_modes + var_alias_modes + var_meas_modes + var_fit_modes
  
    # total_temp_variance = np.sum(var_temp_modes)
    # total_vibr_variance = np.sum(var_vibr_modes)
    # total_alias_variance = np.sum(var_alias_modes)
    # total_meas_variance = np.sum(var_meas_modes)
    # total_fit_variance = np.sum(var_fit_modes)
    # total_variance = np.sum(var_total_modes)
    # print ("\nTEMP SUM:", total_temp_variance)
    # print ("VIBR SUM:", total_vibr_variance)
    # print ("ALIASING SUM:", total_alias_variance)
    # print ("MEAS SUM:", total_meas_variance)
    # print ("FIT SUM:", total_fit_variance)
    # print ("TOTAL SUM:", total_variance)
    
    mode_axis = np.arange(n_modes) + 1
  
    plt.figure()
    plt.plot(mode_axis, var_total_modes, label="total")
    plt.plot(mode_axis, var_temp_modes, label="temporal")
    plt.plot(mode_axis, var_alias_modes, label="aliasing")
    plt.plot(mode_axis, var_meas_modes, label="measurement")
    plt.plot(mode_axis, var_vibr_modes, '--', label="vibration only")
    plt.plot(mode_axis, var_fit_modes, label="fitting")


    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Mode index")
    plt.ylabel("Variance [nm^2]")
    plt.title("Variance vs mode")
    plt.grid()
    plt.legend()
  
    plt.show()  
    
      


























