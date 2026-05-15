#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 14 15:17:28 2025

@author: greta
"""

# pylint: disable=C

import numpy as np
import matplotlib.pyplot as plt

from src.Functions import seeing_to_r0
from src.Functions import turbulence_psd
from src.Functions import funct_d2
from src.Functions import total_variance
from src.Functions import interpolate_and_normalize_psd
from src.Functions import load_parameters
from src.Functions import load_PSD_windshake
from src.Functions import radial_order_from_n_modes

from src.Functions import fitting_variance
from src.Functions import build_transfer_function
from src.Functions import temporal_variance
from src.Functions import aliasing_variance
from src.Functions import measure_variance
from src.Functions import vibration_variance
from src.Functions import PSD_conversion
from src.Functions import find_best_gain
from src.Functions import compute_optical_gain
from src.Functions import final_soul_optical_gain

from src.plots import plot_gain_optimization_sweep
from src.plots import plot_all_PSD
from src.plots import check
from src.plots import plot_PSD_alias_mode_0
from src.plots import plot
from src.plots import plot_PSD_OL_CL_mode_0
from src.plots import plot_psd_vibr_soul
from src.plots import optg_soul_comparison
from src.plots import plot_variance_vs_modes

system = "SOUL"

if system == "ANDES":

    param = load_parameters('params_ANDES.yaml')

elif system =="SOUL":
    
    param = load_parameters('params_SOUL.yaml')
      
else:
    
    raise RuntimeError("system must be 'ANDES' or 'SOUL'") 


print("Parameters loaded successfully.")
  
n_actuators = param['control']['n_modes']
telescope_diameter = param['telescope']['telescope_diam']
aperture_radius = telescope_diameter / 2
aperture_center = [0, 0, 0]
  
outer_scale = param['atmosphere']['outer_scale']
layers_altitude = 0.0
wind_direction = 0.0
wind_speed = param['atmosphere']['wind_speed']
seeing = param['atmosphere']['seeing']
fried_param = seeing_to_r0(seeing)


rho = 0
theta = 0


value_F_excess_noise = param['wavefront_sensor']['value_for_F_excess_noise']
F_excess_noise = np.sqrt(value_F_excess_noise)
sky_background = param['wavefront_sensor']['sky_backgr']
dark_current = param['wavefront_sensor']['dark_curr']
readout_noise = param['wavefront_sensor']['noise_readout']
mod_radii_andes = param['wavefront_sensor'].get('modal_radius_andes', None)
mod_radii_soul = param['wavefront_sensor'].get('modal_radius_soul', None)

 
file_path_R1 = param['data']['reconstruction_matrix'] 
file_sigma_slope = param['data']['sigma_slopes']
file_path_wind = param['data']['windshake_psd']
file_modal_psd_alias_path = param['data']['modal_psd_alias']
file_optg = param['data'].get('optical_gain_models', None)
file_optg_cube = param['data'].get('optical_gain_cube', None)
if file_optg is not None and file_optg_cube is not None:
    raise RuntimeError("Both optical gain models and optical gain cube provided."
                       " Please provide only one of them.")


d1 = param['plant']['d_1']
d3 = param['plant']['d_3']
n1 = param['plant']['n_1']
n2 = param['plant']['n_2']
n3 = param['plant']['n_3']

t_0 = param['control']['sampling_time']
total_delay = param['control']['total_delay']
gain_minimum = param['control']['gain_min']

spatial_freqs = np.logspace(-4, 4, 100)
temporal_freqs_minimum = param['frequency_ranges']['temporal_freqs_min']
temporal_freqs_maximum = np.log10(1.0 / (2.0 * t_0))
temporal_freqs_number = param['frequency_ranges']['temporal_freqs_n']
temporal_freqs = np.logspace(temporal_freqs_minimum, temporal_freqs_maximum, temporal_freqs_number)
omega_temporal_freqs = 2 * np.pi * temporal_freqs
g_maximum_mapping = {                                                          
    1: 2.0,                                                                    
    2: 1.0,
    3: 0.6,
    4: 0.4
}
gain_maximum = g_maximum_mapping.get(total_delay)
gain_number = param['control']['gain_n']
gain_value = param['control'].get('gain_value', None)
bin_value = param['control']['bin']
 
modulation_radius = param['wavefront_sensor']['modulation_radius']
# here we do not use n_actuators because it can be reduced to analyse the error on a small number of modes.
if system == "ANDES":
    maximum_radial_order = 88
elif system == "SOUL":
    maximum_radial_order = 35
else:
    maximum_radial_order = radial_order_from_n_modes(n_actuators)
 
fitting_coeff = 0.2778
alpha_ = -17/3
 
phot_flux = float(param['guide_star']['flux_photons'])
frame_rate = 1.0 / t_0
magnitude = param['guide_star']['magn']
n_subapert = param['wavefront_sensor']['number_of_sub']
collecting_area = param['telescope']['collect_area']
x_pixel = param['control']['slope_computer_weights']

display = param['display']['enabled']

if file_optg is not None and file_optg_cube is None:

    c_optg = compute_optical_gain(file_optg[0], file_optg[1], seeing, 
                                  modulation_radius, n_actuators,
                                  modulation_radii=(0.0, 4.0))
elif file_optg is None and file_optg_cube is not None:
    
    c_optg = final_soul_optical_gain(file_optg_cube, bin_value,
                                       magnitude, n_actuators)
else:
    
    raise RuntimeError("Either 'file_optg_cube' or 'file_optg' must be provided") 


freq, PSD_wind_vib = load_PSD_windshake(file_path_wind)

if (freq is None and PSD_wind_vib is None) or (freq is None or PSD_wind_vib is None):                                     
    
    raise RuntimeError("PSD windshake or corresponding frequencies not loaded") 

print("PSD windshake and corresponding frequencies loaded successfully.")


PSD_wind_vib = PSD_conversion(PSD_wind_vib)

PSD_atmosf = turbulence_psd(rho, theta, aperture_radius, aperture_center, fried_param, outer_scale,
                            layers_altitude, wind_speed, wind_direction, spatial_freqs, temporal_freqs,
                            n_modes=n_actuators)

d2 = funct_d2(total_delay)
plant_num = np.polymul(np.polymul(np.asarray(n1), np.asarray(n2)), np.asarray(n3))
plant_den = np.polymul(np.polymul(np.asarray(d1), d2), np.asarray(d3))

# -----------------------------------------------------------------------------
# GAIN CONFIGURATION AND OPTIMIZATION
# -----------------------------------------------------------------------------

if gain_value is not None:
    # Use explicitly provided gain values from the YAML file
    gain_value_array = np.asarray(gain_value, dtype=float).ravel()
    gain_number_array = np.asarray(gain_number, dtype=int).ravel()

    if gain_value_array.size == 1 and gain_number_array.size == 1:
        gain_ = np.full(n_actuators, gain_value_array.item())
    elif gain_value_array.size == gain_number_array.size:
        gain_ = np.concatenate([
            np.full(gain_number_array[index], gain_value_array[index])
            for index in range(gain_value_array.size)
        ])
    else:
        raise ValueError("gain_value and gain_n must have the same length")
else:
    if gain_number == 1:
        
        # ---------------------------------------------------------------------
        # BLOCK OPTIMIZATION
        # ---------------------------------------------------------------------
        final_gain_vector = np.zeros(n_actuators)
        
        # STEP 1: Tip and Tilt Optimization (Modes 0 and 1)
        modes_TT = [0, 1] if n_actuators > 1 else [0]
        
        print("\n--- Tip-Tilt Gain Optimization ---")
        # Catturiamo i 3 output
        best_gain_TT, gain_vals_TT, var_TT = find_best_gain(
            gain_minimum, gain_maximum, omega_temporal_freqs, temporal_freqs, freq,
            t_0, plant_num, plant_den, telescope_diameter, fried_param,
            F_excess_noise, sky_background, dark_current, readout_noise,
            phot_flux, frame_rate, magnitude, n_subapert, collecting_area,
            x_pixel, fitting_coeff, alpha_, seeing, modulation_radius,
            wind_speed, maximum_radial_order, file_path_R1,
            PSD_atmosf, PSD_wind_vib, file_sigma_slope, c_optg,
            actuators_number=n_actuators, modes_to_optimize=modes_TT, 
            base_gain_vector=final_gain_vector, verbose=False
            )
        
        final_gain_vector[modes_TT] = best_gain_TT
        print(f"\nBest Tip-Tilt gain found: {best_gain_TT:.2f}")

        # STEP 2: Higher Orders Optimization (Mode 2 onwards)
        if n_actuators > 2:
            modes_HO = list(range(2, n_actuators))
            
            print("\n--- Higher Orders Gain Optimization ---")
            # Catturiamo i 3 output
            best_gain_HO, gain_vals_HO, var_HO = find_best_gain(
                gain_minimum, gain_maximum, omega_temporal_freqs, temporal_freqs, freq,
                t_0, plant_num, plant_den, telescope_diameter, fried_param,
                F_excess_noise, sky_background, dark_current, readout_noise,
                phot_flux, frame_rate, magnitude, n_subapert, collecting_area,
                x_pixel, fitting_coeff, alpha_, seeing, modulation_radius,
                wind_speed, maximum_radial_order, file_path_R1,
                PSD_atmosf, PSD_wind_vib, file_sigma_slope, c_optg,
                actuators_number=n_actuators, modes_to_optimize=modes_HO, 
                base_gain_vector=final_gain_vector, verbose=False
                )
            
            final_gain_vector[modes_HO] = best_gain_HO
            print(f"\nBest Higher Orders gain found: {best_gain_HO:.2f}")

        gain_ = final_gain_vector

    elif gain_number == n_actuators:
        # Fallback for sweep configuration
        gain_ = np.linspace(gain_minimum, gain_maximum, gain_number)
    else:
        raise ValueError("Set gain_n to 1 or n_modes, or provide gain_value")

if gain_.size != n_actuators:
    raise ValueError(f"Gain vector length {gain_.size} does not match n_modes={n_actuators}")
    

H_r_temp, H_n_meas = build_transfer_function(
    omega_temporal_freqs,
    t_0,
    n_actuators,
    plant_num,
    plant_den,
    gain=gain_,
)
H_n_alias = H_n_meas

#################
# FIT  ---->  Variance 
#################

var_fit = fitting_variance(fitting_coeff, n_actuators, telescope_diameter, fried_param)

#################
# VIBRATIONS  ---->  Variance OL, Variance CL, PSD CL, PSD OL
#################


if np.array_equal(temporal_freqs, freq): 
    
    
    var_vibr_OL, var_vibr_CL, PSD_out_vibr, PSD_in_vibr = vibration_variance (PSD_wind_vib, H_r_temp, n_actuators, omega_temporal_freqs)

else: 
    
    PSD_wind_vib_interp_norm = interpolate_and_normalize_psd(temporal_freqs, freq, PSD_wind_vib, n_actuators)
    var_vibr_OL, var_vibr_CL, PSD_out_vibr, PSD_in_vibr = vibration_variance (PSD_wind_vib_interp_norm, H_r_temp, 
                                                                              n_actuators, omega_temporal_freqs)
    

#################
# TEMPORAL  ---->  Variance OL, Variance CL, PSD CL, PSD OL
#################


if np.array_equal(temporal_freqs, freq): 
    
    var_temp_OL, var_temp_CL, PSD_out_temp, PSD_in_temp = temporal_variance (PSD_atmosf, PSD_wind_vib, H_r_temp, n_actuators, 
                                                                             omega_temporal_freqs)

else: 
    
    PSD_wind_vib_interp_norm = interpolate_and_normalize_psd(temporal_freqs, freq, PSD_wind_vib, n_actuators)
    var_temp_OL, var_temp_CL, PSD_out_temp, PSD_in_temp = temporal_variance (PSD_atmosf, PSD_wind_vib_interp_norm, 
                                                                             H_r_temp, n_actuators, omega_temporal_freqs)


#################
# ALIASING  ---->  Variance OL, Variance CL, PSD CL, PSD OL
#################

var_alias_OL, var_alias_CL, PSD_out_alias, PSD_in_alias = aliasing_variance(
    transf_funct=H_n_alias,
    actuators_number=n_actuators,
    omega_temp_freq_interval=omega_temporal_freqs,
    c_optg=c_optg,
    alpha=alpha_,
    telescope_diameter=telescope_diameter,
    seeing=seeing,
    modulation_radius=modulation_radius,
    windspeed=wind_speed,
    maximum_radial_order_corrected=maximum_radial_order,
    file_path_matrix_R=file_path_R1,
    file_path_sigma_slopes=file_sigma_slope,
)

#################
# MEAS  ---->  Variance OL, Variance CL, PSD CL, PSD OL
#################


var_meas_OL, var_meas_CL, PSD_out_meas, PSD_in_meas = measure_variance(
    F_excess_noise,
    x_pixel,
    sky_background,
    dark_current,
    readout_noise,
    phot_flux,
    telescope_diameter,
    frame_rate,
    magnitude,
    n_subapert,
    collecting_area,
    file_path_R1,
    H_n_meas,
    n_actuators,
    omega_temporal_freqs,
    c_optg, verbose=True
)

print ("\nTOTAL VARIANCE USING THE BEST GAIN:")
print ("\nOPEN LOOP TOTAL VARIANCE:")
var_total_OL = total_variance(var_fit, var_temp_OL, var_alias_OL, var_meas_OL, verbose=True)
print ("CLOSED LOOP TOTAL VARIANCE:")
var_total_CL = total_variance(var_fit, var_temp_CL, var_alias_CL, var_meas_CL, verbose=True)

##### PLOTS AND CHECKS

if display:
    
    plot_variance_vs_modes(PSD_out_temp, PSD_out_vibr, PSD_out_alias, PSD_out_meas,
                           var_fit, omega_temporal_freqs, n_actuators)

    # 1. Plot Gain Sweep Optimization
    if gain_number == 1:
        if n_actuators <= 2:
            plot_gain_optimization_sweep(gain_vals_TT, var_TT)
        else:
            plot_gain_optimization_sweep(gain_vals_TT, var_TT, gain_vals_HO, var_HO)

    plot_psd_vibr_soul (file_path_wind)

    plot(omega_temporal_freqs, H_r_temp, H_n_meas, H_n_alias, PSD_in_vibr, PSD_out_vibr, PSD_in_temp, PSD_out_temp,
         PSD_in_meas, PSD_out_meas, PSD_in_alias, PSD_out_alias)
                               

    plot_all_PSD(omega_temporal_freqs, PSD_out_temp, PSD_out_meas, PSD_out_alias)


    # check(file_path_R1, telescope_diameter, seeing, modulation_radius,
    #       n_actuators, alpha_, omega_temporal_freqs, wind_speed, maximum_radial_order,
    #       magnitude, bin_value, c_optg, file_sigma_slope)
    #       ####### con bin_value (usando cubo per soul)
    
    check(file_path_R1, telescope_diameter, seeing, modulation_radius,
          n_actuators, alpha_, omega_temporal_freqs, wind_speed, maximum_radial_order,
          magnitude, c_optg, file_sigma_slope)
    
    if system == "ANDES":
        # plot_PSD_alias_mode_0(n_actuators, omega_temporal_freqs, alpha_, telescope_diameter,
        #                       seeing, modulation_radius, wind_speed, maximum_radial_order,
        #                       bin_value, magnitude, file_path_R1, c_optg, file_sigma_slope, 
        #                       file_modal_psd_alias_path) 
        #                       ####### con bin_value (usando cubo per soul)
        
        plot_PSD_alias_mode_0(n_actuators, omega_temporal_freqs, alpha_, telescope_diameter,
                              seeing, modulation_radius, wind_speed, maximum_radial_order,
                              magnitude, file_path_R1, c_optg, file_sigma_slope, 
                              file_modal_psd_alias_path)


    plot_PSD_OL_CL_mode_0(gain_, omega_temporal_freqs, t_0, n_actuators, n1, n2, n3, d1, d2, d3,
                          PSD_atmosf, PSD_wind_vib, alpha_, telescope_diameter, seeing, modulation_radius, wind_speed, 
                          maximum_radial_order, c_optg, F_excess_noise, x_pixel, sky_background, dark_current, readout_noise, 
                          phot_flux, frame_rate, magnitude, n_subapert, collecting_area, temporal_freqs, freq, 
                          file_path_R1, file_sigma_slope)
    
    if system == "SOUL":

        if file_optg_cube is None:
            file_optg_cube = "src/file_fits/LBT/SOUL_OPTG.fits"
    
        optg_soul_comparison (file_optg_cube, bin_value, magnitude, n_actuators, 
                              file_optg[0], file_optg[1], seeing, modulation_radius)
        
        
        
      
    
    
    
    
    
    
    
    
    
    
    
    
