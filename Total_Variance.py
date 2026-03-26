#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 14 15:17:28 2025

@author: greta
"""

# pylint: disable=C

import numpy as np 

from src.Functions import turbulence_psd
from src.Functions import funct_d2
from src.Functions import total_variance
from src.Functions import interpolate_and_normalize_psd
from src.Functions import load_parameters
from src.Functions import load_PSD_windshake
from src.Functions import radial_order_from_n_modes

from src.Functions import fitting_variance
from src.Functions import compute_andes_optical_gain
from src.Functions import build_transfer_function
from src.Functions import temporal_variance
from src.Functions import aliasing_variance
from src.Functions import measure_variance
from src.Functions import vibration_variance

from src.plots import plot_total_variance_mode_0
from src.plots import plot_all_PSD
from src.plots import check
from src.plots import plot_PSD_alias_mode_0
from src.plots import plot
from src.plots import plot_PSD_OL_CL_mode_0


param = load_parameters('params_mod4.yaml')
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
fried_param = 0.98 * 500 / seeing

rho = 0
theta = 0

value_F_excess_noise = param['wavefront_sensor']['value_for_F_excess_noise']
F_excess_noise = np.sqrt(value_F_excess_noise)
sky_background = param['wavefront_sensor']['sky_backgr']
dark_current = param['wavefront_sensor']['dark_curr']
readout_noise = param['wavefront_sensor']['noise_readout']
  
file_path_R1 = param['data']['reconstruction_matrix']
file_path_wind1 = param['data']['windshake_psd']
file_optg = param['data']['optical_gain_models']
file_sigma_slope = param['data']['sigma_slopes']

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

if gain_value is not None:
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
        gain_ = np.full(n_actuators, float(gain_minimum))
    elif gain_number == n_actuators:
        gain_ = np.linspace(gain_minimum, gain_maximum, gain_number)
    else:
        raise ValueError("Set gain_n to 1 or n_modes, or provide gain_value")

if gain_.size != n_actuators:
    raise ValueError(f"Gain vector length {gain_.size} does not match n_modes={n_actuators}")
modulation_radius = param['wavefront_sensor']['modulation_radius']
# here we do not use n_actuators because it can be reduced to analyse the error on a small number of modes.
if param['system']['name'] == "ANDES":
    maximum_radial_order = 88
elif param['system']['name'] == "SOUL":
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

system = param['system']['name']


display = True

freq, PSD_wind_vib = load_PSD_windshake(file_path_wind1)

if (freq is None and PSD_wind_vib is None) or (freq is None or PSD_wind_vib is None):                                     
    
    raise RuntimeError("PSD windshake or corresponding frequencies not loaded") 

print("PSD windshake and corresponding frequencies loaded successfully.")

PSD_atmosf = turbulence_psd(rho, theta, aperture_radius, aperture_center, fried_param, outer_scale,
                            layers_altitude, wind_speed, wind_direction, spatial_freqs, temporal_freqs)

d2 = funct_d2(total_delay)
plant_num = np.polymul(np.polymul(np.asarray(n1), np.asarray(n2)), np.asarray(n3))
plant_den = np.polymul(np.polymul(np.asarray(d1), d2), np.asarray(d3))


c_optg = 0

if system == "ANDES":
    
    c_optg = compute_andes_optical_gain(file_optg[0], file_optg[1], seeing, modulation_radius)


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

var_alias_OL, var_alias_CL, PSD_out_alias, PSD_in_alias = aliasing_variance(H_n_alias, n_actuators, omega_temporal_freqs,
                                                                            c_optg, alpha_, telescope_diameter, seeing, modulation_radius,
                                                                            wind_speed, maximum_radial_order, file_path_R1,  
                                                                            file_sigma_slope)

#################
# MEAS  ---->  Variance OL, Variance CL, PSD CL, PSD OL
#################


var_meas_OL, var_meas_CL, PSD_out_meas, PSD_in_meas  = measure_variance (F_excess_noise, x_pixel, sky_background, dark_current, 
                                                                         readout_noise, phot_flux, telescope_diameter, frame_rate, 
                                                                         magnitude, n_subapert, collecting_area, file_path_R1, 
                                                                         omega_temporal_freqs, H_n_meas, n_actuators, 
                                                                         c_optg, alpha_, maximum_radial_order,
                                                                         seeing, modulation_radius, wind_speed, 
                                                                         file_path_sigma_slopes=None)

print ("OPEN LOOP:")
var_total_OL = total_variance(var_fit, var_temp_OL, var_alias_OL, var_meas_OL)
print ("CLOSED LOOP:")
var_total_CL = total_variance(var_fit, var_temp_CL, var_alias_CL, var_meas_CL)


##### PLOTS AND CHECKS


if display:

    plot_total_variance_mode_0(gain_minimum, gain_maximum, omega_temporal_freqs, temporal_freqs, freq,
                               t_0, plant_num, plant_den, telescope_diameter, fried_param, F_excess_noise,
                               sky_background, dark_current, readout_noise, phot_flux, frame_rate, magnitude,
                               n_subapert, collecting_area, x_pixel, fitting_coeff, alpha_, seeing,
                               modulation_radius, wind_speed, maximum_radial_order, file_path_R1, file_optg,
                               PSD_atmosf, PSD_wind_vib, file_sigma_slope)


    plot(omega_temporal_freqs, H_r_temp, H_n_meas, H_n_alias, PSD_in_temp, PSD_out_temp,
         PSD_in_meas, PSD_out_meas, PSD_in_alias, PSD_out_alias)
                               

    plot_all_PSD(omega_temporal_freqs, PSD_out_temp, PSD_out_meas, PSD_out_alias)


    check(file_path_R1, telescope_diameter, seeing, modulation_radius,
          n_actuators, alpha_, omega_temporal_freqs, wind_speed, maximum_radial_order,
          file_optg, file_sigma_slope, system="ANDES")


    plot_PSD_alias_mode_0(n_actuators, omega_temporal_freqs, alpha_, telescope_diameter,
                          seeing, modulation_radius, wind_speed, maximum_radial_order,
                          file_path_R1, file_optg, file_sigma_slope, system="ANDES")


    plot_PSD_OL_CL_mode_0(gain_, omega_temporal_freqs, t_0, n_actuators, n1, n2, n3, d1, d2, d3,
                          PSD_atmosf, PSD_wind_vib, alpha_, telescope_diameter, seeing, modulation_radius, wind_speed, 
                          maximum_radial_order, c_optg, F_excess_noise, x_pixel, sky_background, dark_current, readout_noise, 
                          phot_flux, frame_rate, magnitude, n_subapert, collecting_area, temporal_freqs, freq, 
                          file_path_R1, file_sigma_slope)




