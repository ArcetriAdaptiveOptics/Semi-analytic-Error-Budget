#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Error budget for a hybrid AO system: pyramid WFS for higher-order (HO)
modes, Full-Aperture Centroid Sensor (FACS) for tip/tilt.

This mirrors Total_Variance.py, but the tip/tilt modes (indices 0-1 in the
modal arrays) are removed from the pyramid's own fitting/temporal/
measurement/aliasing budget and replaced by an independent 2-mode temporal +
measurement-noise budget for a FACS-based tip/tilt loop (see
src/facs_functions.py). FACS has no aliasing term (neglected for a
full-aperture sensor) and no fitting term of its own (fitting only depends
on the DM/ASM resolution, parameterized here by the number of HO modes the
pyramid corrects).

Run with:
    python Total_Variance_FACS.py
"""

# pylint: disable=C

import numpy as np

from src.Functions import (
    seeing_to_r0,
    turbulence_psd,
    funct_d2,
    load_parameters,
    load_PSD_windshake,
    fitting_variance,
    build_transfer_function,
    temporal_variance,
    aliasing_variance,
    measure_variance,
    compute_optical_gain,
    integrate_modal_psd,
    gain_maximum_from_total_delay,
    optimize_gain_blocks,
    resolve_gain_mode,
    flux_for_frame_for_pixel,
)
from src.facs_functions import (
    strehl_ratio_marechal,
    broaden_spot_for_strehl,
    facs_slope_noise_variance_nm2,
    facs_measurement_variance,
    facs_total_variance,
)
from src.config_utils import resolve_binning_config

param = load_parameters('params_ANDES_FACS.yaml')
param = resolve_binning_config(param)

print("Parameters loaded successfully.")

# =============================================================================
# SHARED TELESCOPE / ATMOSPHERE / PLANT SETUP
# =============================================================================

n_total_modes = param['control']['n_modes']
n_modes_ho = n_total_modes - 2  # HO modes corrected by the pyramid (TT excluded)

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

fitting_coeff = 0.2778
alpha_ = -17 / 3
maximum_radial_order = 88  # ANDES-scale system, as in Total_Variance.py

plant = param['plant']
plant_num = np.asarray(plant['numerator'])
plant_den_base = np.asarray(plant['denominator'])
total_delay = plant['total_delay']
t_0 = param['control']['sampling_time']
plant_den = np.polymul(plant_den_base, funct_d2(total_delay))

spatial_freqs = np.logspace(-4, 4, 100)
temporal_freqs_minimum = param['frequency_ranges']['temporal_freqs_min']
temporal_freqs_maximum = np.log10(1.0 / (2.0 * t_0))
temporal_freqs_number = param['frequency_ranges']['temporal_freqs_n']
temporal_freqs = np.logspace(temporal_freqs_minimum, temporal_freqs_maximum, temporal_freqs_number)
omega_temporal_freqs = 2 * np.pi * temporal_freqs

PSD_atmosf = turbulence_psd(rho, theta, aperture_radius, aperture_center, fried_param, outer_scale,
                            layers_altitude, wind_speed, wind_direction, spatial_freqs, temporal_freqs,
                            n_modes=n_total_modes)

freq, PSD_wind_vib = load_PSD_windshake(param['data']['windshake_psd'], target_frequencies=temporal_freqs)
if (freq is None and PSD_wind_vib is None) or (freq is None or PSD_wind_vib is None):
    raise RuntimeError("PSD windshake or corresponding frequencies not loaded")
print("PSD windshake and corresponding frequencies loaded successfully.")

# =============================================================================
# HO BRANCH (PYRAMID) — unchanged pipeline, full n_total_modes system.
# Only modes[2:] (HO) are kept downstream; modes[0:2] (tip/tilt) are
# discarded in favor of the FACS branch below.
# =============================================================================

value_F_excess_noise = param['wavefront_sensor']['value_for_F_excess_noise']
F_excess_noise = np.sqrt(value_F_excess_noise)
sky_background = param['wavefront_sensor']['sky_backgr']
dark_current = param['wavefront_sensor']['dark_curr']
readout_noise = param['wavefront_sensor']['noise_readout']

file_path_R1 = param['data']['reconstruction_matrix']
file_sigma_slope = param['data']['sigma_slopes']
file_optg = param['data']['optical_gain_models']

frame_rate = 1.0 / t_0
n_subapert = param['wavefront_sensor']['number_of_sub']
x_pixel = param['control']['slope_computer_weights']
modulation_radius = param['wavefront_sensor']['modulation_radius']

# -----------------------------------------------------------------------------
# Photon budget split between the pyramid and FACS.
#
# 'shared': FACS and the pyramid sense the same light through a plain
# beamsplitter, so the pyramid's own flux must be reduced by (1-flux_split)
# to avoid double-counting photons on both branches.
# 'independent': FACS senses a different band (dichroic) with its own
# photometry ('guide_star_facs'); the pyramid keeps its full flux.
# -----------------------------------------------------------------------------
facs = param['wavefront_sensor_facs']
flux_mode = facs.get('flux_mode', 'shared')

if flux_mode == 'shared':
    flux_split = float(facs['flux_split'])
    if not 0.0 < flux_split < 1.0:
        raise ValueError("wavefront_sensor_facs.flux_split must be in (0, 1)")
    guide_star_flux = float(param['guide_star']['flux_photons'])
    phot_flux = guide_star_flux * (1.0 - flux_split)
    phot_flux_facs = guide_star_flux * flux_split
    magnitude = param['guide_star']['magn']
    magnitude_facs = magnitude
elif flux_mode == 'independent':
    phot_flux = float(param['guide_star']['flux_photons'])
    magnitude = param['guide_star']['magn']
    guide_star_facs = param['guide_star_facs']
    phot_flux_facs = float(guide_star_facs['flux_photons'])
    magnitude_facs = guide_star_facs['magn']
else:
    raise ValueError(
        f"Unsupported wavefront_sensor_facs.flux_mode={flux_mode!r}. "
        "Expected 'shared' or 'independent'."
    )

c_optg = compute_optical_gain(file_optg[0], file_optg[1], seeing,
                              modulation_radius, n_total_modes,
                              modulation_radii=(0.0, 4.0))

gain_minimum = param['control']['gain_min']
gain_block_sizes = param['control'].get('gain_block_sizes', param['control'].get('gain_blocks', None))
gain_mode = resolve_gain_mode(param['control'])

if gain_mode != 'block_optimization':
    raise ValueError("Total_Variance_FACS.py currently only supports control.gain_mode: block_optimization")

gain_maximum = gain_maximum_from_total_delay(total_delay)
gain_ho, gain_sweeps = optimize_gain_blocks(
    gain_min=gain_minimum,
    gain_max=gain_maximum,
    omega_temp_freq_interval=omega_temporal_freqs,
    t_0=t_0,
    plant_num=plant_num,
    plant_den=plant_den,
    telescope_diameter=telescope_diameter,
    fried_parameter=fried_param,
    excess_noise_factor=F_excess_noise,
    sky_background=sky_background,
    dark_current=dark_current,
    readout_noise=readout_noise,
    photon_flux=phot_flux,
    frame_rate=frame_rate,
    magnitude=magnitude,
    n_subaperture=n_subapert,
    slope_computer_weights=x_pixel,
    fitting_coeff=fitting_coeff,
    alpha=alpha_,
    seeing=seeing,
    modulation_radius=modulation_radius,
    wind_speed=wind_speed,
    maximum_radial_order_corrected=maximum_radial_order,
    reconstruction_matrix_path=file_path_R1,
    psd_turbulence=PSD_atmosf,
    psd_windshake=PSD_wind_vib,
    sigma_slopes_path=file_sigma_slope,
    c_optg=c_optg,
    actuators_number=n_total_modes,
    gain_block_sizes=gain_block_sizes,
    verbose=False,
    verbose_flux=False,
    verbose_gain=False,
)
# Note: gain_ho[0:2] is the pyramid's own tip/tilt gain, computed here for
# free by the block optimizer but unused: tip/tilt is corrected by the FACS
# loop instead (see below).

H_r_ho, H_n_ho = build_transfer_function(
    omega_temporal_freqs, t_0, n_total_modes, plant_num, plant_den, gain=gain_ho,
)

_, _, PSD_out_temp_full, _ = temporal_variance(PSD_atmosf, PSD_wind_vib, H_r_ho, n_total_modes, omega_temporal_freqs)

_, _, PSD_out_alias_full, _ = aliasing_variance(
    transf_funct=H_n_ho,
    actuators_number=n_total_modes,
    omega_temp_freq_interval=omega_temporal_freqs,
    c_optg=c_optg,
    telescope_diameter=telescope_diameter,
    seeing=seeing,
    modulation_radius=modulation_radius,
    windspeed=wind_speed,
    maximum_radial_order_corrected=maximum_radial_order,
    file_path_matrix_R=file_path_R1,
    alpha=alpha_,
    file_path_sigma_slopes=file_sigma_slope,
)

_, _, PSD_out_meas_full, _ = measure_variance(
    F_excess_noise, x_pixel, sky_background, dark_current, readout_noise,
    phot_flux, telescope_diameter, frame_rate, magnitude, n_subapert,
    file_path_R1, H_n_ho, n_total_modes, omega_temporal_freqs, c_optg,
)

var_temp_modal = integrate_modal_psd(PSD_out_temp_full, omega_temporal_freqs)
var_alias_modal = integrate_modal_psd(PSD_out_alias_full, omega_temporal_freqs)
var_meas_modal = integrate_modal_psd(PSD_out_meas_full, omega_temporal_freqs)

var_temp_ho = np.sum(var_temp_modal[2:])
var_alias_ho = np.sum(var_alias_modal[2:])
var_meas_ho = np.sum(var_meas_modal[2:])
var_fit_ho = fitting_variance(fitting_coeff, n_modes_ho, telescope_diameter, fried_param)

# =============================================================================
# TT BRANCH (FACS) — independent 2-mode closed loop.
# =============================================================================

control_facs = param['control_facs']

t_0_facs = control_facs.get('sampling_time', t_0)
if t_0_facs != t_0:
    plant_den_facs = np.polymul(plant_den_base, funct_d2(total_delay))
else:
    plant_den_facs = plant_den

gain_facs = np.full(2, float(control_facs['gain_value']))
H_r_facs, H_n_facs = build_transfer_function(
    omega_temporal_freqs, t_0_facs, 2, plant_num, plant_den_facs, gain=gain_facs,
)

var_temp_tt_OL, var_temp_tt_CL, PSD_out_temp_tt, PSD_in_temp_tt = temporal_variance(
    PSD_atmosf[0:2, :], PSD_wind_vib[0:2, :], H_r_facs, 2, omega_temporal_freqs,
)

flux_components = flux_for_frame_for_pixel(
    phot_flux_facs, telescope_diameter, 1.0 / t_0_facs, magnitude_facs,
    n_subaperture=1, return_components=True,
)
facs_flux_total_per_frame = flux_components['flux_total_per_frame']

# The FACS PSF is not diffraction-limited: it sees the same focal-plane
# image left by the pyramid's own (imperfect) HO correction. Broaden the
# spot accordingly, via a Maréchal-approximation Strehl ratio estimated
# from the HO residual already computed above.
ho_residual_rms_nm = np.sqrt(var_fit_ho + var_temp_ho + var_meas_ho + var_alias_ho)
wavelength_facs = float(facs['wavelength']) * 1e-9  # wavefront_sensor_facs.wavelength is in [nm]
strehl_facs = strehl_ratio_marechal(ho_residual_rms_nm, wavelength_facs)
if strehl_facs < 0.2:
    print(f"\nWarning: FACS Strehl estimate ({strehl_facs:.2f}) is low; the "
          "Maréchal/Gaussian-core broadening approximation is only "
          "qualitative in this regime.")
n_samp_facs, fwhm_spot_pix_facs = broaden_spot_for_strehl(
    facs['n_samp'], facs['fwhm_spot_pix'], strehl_facs,
)

sigma2_tt_nm2 = facs_slope_noise_variance_nm2(
    photon_flux_total=facs_flux_total_per_frame,
    ron=facs['noise_readout'],
    sky_bkg=facs['sky_backgr'],
    dark_curr=facs['dark_curr'],
    excess_noise_factor2=facs['value_for_F_excess_noise'],
    n_pixels=facs['n_pixels'],
    n_samp=n_samp_facs,
    fwhm_spot_pix=fwhm_spot_pix_facs,
    fwhm_weight_pix=facs['fwhm_weight_pix'],
    pixel_scale_arcsec=facs['pixel_scale_arcsec'],
    telescope_diameter=telescope_diameter,
    threshold=facs['threshold'],
    order=facs['order'],
    npoints=facs['npoints'],
)

var_meas_tt_OL, var_meas_tt_CL, PSD_out_meas_tt, PSD_in_meas_tt = facs_measurement_variance(
    H_n_facs, omega_temporal_freqs, sigma2_tt_nm2,
)

# =============================================================================
# COMBINED SYSTEM TOTAL: HO (pyramid) + TT (FACS)
# =============================================================================

print("\nTOTAL VARIANCE (HO via pyramid + TT via FACS), CLOSED LOOP:")
var_total, var_ho, var_tt = facs_total_variance(
    var_fit_ho, var_temp_ho, var_meas_ho, var_alias_ho,
    var_temp_tt_CL, var_meas_tt_CL, verbose=True,
)

print(f"\nFACS tip/tilt measurement-noise variance: {sigma2_tt_nm2:.4f} nm^2 "
      f"({np.sqrt(sigma2_tt_nm2):.4f} nm RMS, open loop, per axis)")
print(f"FACS Strehl estimate at {wavelength_facs * 1e9:.0f} nm (from HO residual "
      f"{ho_residual_rms_nm:.1f} nm RMS): {strehl_facs:.3f} "
      f"(spot FWHM {facs['fwhm_spot_pix']:.2f} -> {fwhm_spot_pix_facs:.2f} pix)")
