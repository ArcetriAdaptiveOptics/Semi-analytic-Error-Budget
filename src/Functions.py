#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 12 15:43:33 2025

@author: greta
"""
# pylint: disable=C

from dataclasses import dataclass, field

import yaml
import os
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from scipy import integrate
from astropy.io import fits                                                    
from functools import reduce                                                   
import sympy as sp                                                             
from scipy.interpolate import RegularGridInterpolator                        

from arte.types.guide_source import GuideSource 
from arte.types.aperture import CircularOpticalAperture 
from arte.atmo.von_karman_covariance_calculator import VonKarmanSpatioTemporalCovariance 
from arte.atmo.cn2_profile import Cn2Profile                                                                                     


DEFAULT_SIGMA_SLOPES_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    'src',
    'file_fits',
    'ANDES',
    'slopes_rms_time_avg_all.fits'
)


# Reads the YAML file (where parameters are listed) and returns a dictionary.

def load_parameters(yaml_file):
    
    with open(yaml_file, 'r', encoding='utf-8') as stream:                     
                                                                               
        try: 
            
            parameters = yaml.safe_load(stream)                                
            return parameters
        
        except yaml.YAMLError as exc:
            raise ValueError("Error in parameters YAML file") from exc
     
        except FileNotFoundError as exc:
            raise FileNotFoundError from exc


# Function to compute the maximum radial order from the total number of corrected modes 

def radial_order_from_n_modes(n_modes):
    n_modes = int(n_modes)

    if n_modes < 1:
        raise ValueError("n_modes must be >= 1")

    radial_order = int(np.floor((np.sqrt(1 + 8 * n_modes) - 3) / 2))

    if radial_order < 0:
        raise ValueError("Computed radial order is negative; check n_modes")

    return radial_order


# Function to define the d2 array, whose length depends on the value of T_total.

def funct_d2 (T_total):
    
    d2 = np.zeros(T_total + 1)
    d2[0] = 1
    
    return d2


# Function that returns the numerator and denominator of the integrator controller C = g*Z/(Z-1)
# expressed as polynomials in Z (descending powers, np.polyval convention).
# Note: Formerly named funct_C, renamed to build_integrator_controller_polynomials for clarity.

def build_integrator_controller_polynomials(gain):

    Z_symbolic = sp.symbols('Z')
    C = (Z_symbolic * gain) / (Z_symbolic - 1)

    num, den = sp.fraction(C)

    num_int = np.array(sp.Poly(num, Z_symbolic).all_coeffs(), dtype=complex)
    den_int = np.array(sp.Poly(den, Z_symbolic).all_coeffs(), dtype=complex)

    return num_int, den_int
  
  
# Function to compute the numerator and denominator polynomials of the transfer functions H_r and H_n
# using np.polymul, np.polyadd and polyval (to evaluate them at Z).
# The expressions written below were derived from Equations (4) and (5) (in "Semianalytical error budget 
# for adaptive optics systems with pyramid wavefront sensors", Agapito and Pinna, 2019).
# The plant (WFS * Reconstructor+Delay * DM) is represented by its pre-multiplied polynomials
# plant_num and plant_den.  The controller for a single mode is controller_num / controller_den.

def transfer_funct(plant_num, controller_num, plant_den, controller_den, Z):
    """Evaluate both H_r and H_n for a single mode.

    Parameters
    ----------
    plant_num : array_like
        Pre-multiplied plant numerator polynomial (np.polyval convention).
    controller_num : array_like
        Controller numerator polynomial for this mode.
    plant_den : array_like
        Pre-multiplied plant denominator polynomial.
    controller_den : array_like
        Controller denominator polynomial for this mode.
    Z : array_like
        Complex frequency vector.
    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(H_r, H_n)`` evaluated over ``Z``.
    """
    H_r_coeff_num = np.polymul(plant_den, controller_den)
    H_r_coeff_den = np.polyadd(
        np.polymul(plant_den, controller_den),
        np.polymul(plant_num, controller_num),
    )
    H_n_coeff_num = np.polymul(plant_num, controller_num)
    H_n_coeff_den = H_r_coeff_den

    H_r_num = np.polyval(H_r_coeff_num, Z)
    H_r_den = np.polyval(H_r_coeff_den, Z)
    H_n_num = np.polyval(H_n_coeff_num, Z)
    H_n_den = np.polyval(H_n_coeff_den, Z)

    H_r = H_r_num / H_r_den
    H_n = H_n_num / H_n_den

    return H_r, H_n
        
        
# Function to compute the controller (num_int, den_int) polynomial arrays for all modes using an integrator.

def compute_int_coeff(gain, omega_temp_freq_interval, t_0, actuators_number):

    num_int_example, den_int_example = build_integrator_controller_polynomials(gain[0])

    num_int_array = np.zeros((actuators_number, len(num_int_example)), dtype=complex)
    den_int_array = np.zeros((actuators_number, len(den_int_example)), dtype=complex)

    for i in range(actuators_number):
        num_int_array[i, :], den_int_array[i, :] = build_integrator_controller_polynomials(gain[i])

    return num_int_array, den_int_array
  

# Function to compute both transfer functions H_r and H_n for all modes.

def compute_H(actuators_number, omega_temp_freq_interval, plant_num, controller_num_matrix, plant_den, controller_den_matrix, Z):

    H_r = np.zeros((actuators_number, len(omega_temp_freq_interval)), dtype=complex)
    H_n = np.zeros((actuators_number, len(omega_temp_freq_interval)), dtype=complex)

    for i in range(actuators_number):
        H_r[i, :], H_n[i, :] = transfer_funct(
            plant_num,
            controller_num_matrix[i, :],
            plant_den,
            controller_den_matrix[i, :],
            Z,
        )

    return H_r, H_n


def _as_controller_coefficient_matrix(coefficients, actuators_number, coefficient_name):

    coefficients = np.asarray(coefficients, dtype=complex)

    if coefficients.ndim == 1:
        if coefficients.size == 0:
            raise ValueError(f"{coefficient_name} must not be empty")
        return np.tile(coefficients[np.newaxis, :], (actuators_number, 1))

    if coefficients.ndim == 2:
        if coefficients.shape[1] == 0:
            raise ValueError(f"{coefficient_name} must not be empty")

        if coefficients.shape[0] == 1 and actuators_number > 1:
            return np.repeat(coefficients, actuators_number, axis=0)

        if coefficients.shape[0] != actuators_number:
            raise ValueError(
                f"{coefficient_name} must have shape (N_coeff,) or (N_modes, N_coeff); "
                f"received {coefficients.shape} for N_modes={actuators_number}"
            )

        return coefficients

    raise ValueError(f"{coefficient_name} must be a 1D or 2D array")


def build_transfer_function_from_controller_polynomials(controller_num, controller_den,
                                                        omega_temp_freq_interval, t_0,
                                                        actuators_number, plant_num, plant_den):
    """
    Build the closed-loop transfer function using explicit controller polynomials.

    Parameters
    ----------
    controller_num : array_like
        Controller numerator polynomial (descending powers of Z).
        Shape ``(N_coeff,)`` (broadcast to all modes) or ``(N_modes, N_coeff)``.
        An integrator C(Z) = g Z / (Z-1) → ``[g, 0]``.
    controller_den : array_like
        Controller denominator polynomial.  Same shape rules as above.
    omega_temp_freq_interval : array_like
        Angular temporal frequency vector [rad s⁻¹].
    t_0 : float
        Sampling period [s].
    actuators_number : int
        Number of modes / actuators.
    plant_num : array_like
        Pre-multiplied plant numerator polynomial (e.g. ``n1 * n2 * n3``).
    plant_den : array_like
        Pre-multiplied plant denominator polynomial (e.g. ``d1 * d2 * d3``).
    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(H_r, H_n)`` matrices of shape ``(N_modes, N_freq)``.
    """
    controller_num_matrix = _as_controller_coefficient_matrix(
        controller_num, actuators_number, "controller_num"
    )
    controller_den_matrix = _as_controller_coefficient_matrix(
        controller_den, actuators_number, "controller_den"
    )

    if np.any(np.isclose(controller_den_matrix[:, 0], 0.0)):
        raise ValueError("The leading controller denominator coefficient must be non-zero")

    Z = np.exp(1j * np.asarray(omega_temp_freq_interval) * t_0)

    return compute_H(
        actuators_number,
        omega_temp_freq_interval,
        plant_num,
        controller_num_matrix,
        plant_den,
        controller_den_matrix,
        Z,
    )


# Function to compute the transfer function H.
# The plant is given as pre-multiplied polynomials plant_num and plant_den.
# The controller is either an integrator (pass gain=) or explicit IIR polynomials
# (pass controller_num= and controller_den=).

def build_transfer_function(omega_temp_freq_interval, t_0, actuators_number,
                            plant_num, plant_den,
                            *, gain=None, controller_num=None, controller_den=None):
    """Build both closed-loop transfer functions H_r and H_n.

    Parameters
    ----------
    omega_temp_freq_interval : array_like
        Angular temporal frequency vector [rad s⁻¹].
    t_0 : float
        Sampling period [s].
    actuators_number : int
        Number of modes / actuators.
    plant_num : array_like
        Pre-multiplied plant numerator polynomial (e.g. ``n1 * n2 * n3``).
    plant_den : array_like
        Pre-multiplied plant denominator polynomial (e.g. ``d1 * d2 * d3``).
    gain : array_like, optional
        Per-mode integrator gain vector.  Mutually exclusive with
        ``controller_num``/``controller_den``.
    controller_num : array_like, optional
        Explicit controller numerator (descending powers of Z).
    controller_den : array_like, optional
        Explicit controller denominator (descending powers of Z).
    """
    if gain is not None:
        ctrl_num, ctrl_den = compute_int_coeff(gain, omega_temp_freq_interval, t_0, actuators_number)
    elif controller_num is not None and controller_den is not None:
        ctrl_num = controller_num
        ctrl_den = controller_den
    else:
        raise ValueError(
            "Provide either 'gain' (integrator) or both "
            "'controller_num' and 'controller_den'"
        )

    return build_transfer_function_from_controller_polynomials(
        ctrl_num,
        ctrl_den,
        omega_temp_freq_interval,
        t_0,
        actuators_number,
        plant_num,
        plant_den,
    )
    

# Function to obtain the atmospheric PSD for n_modes Zernike modes starting from tip (j=2).
# Returns a 2D array of shape (n_modes, len(tempor_freqs)).
# Default n_modes=2 reproduces the original tip-and-tilt behaviour.

def turbulence_psd(rho, theta, aperture_radius, aperture_center, r0, L0, layers_altitude,
                   wind_speed, wind_direction, space_freqs, tempor_freqs, n_modes=2):
    
    source = GuideSource((rho, theta), np.inf)
    aperture = CircularOpticalAperture(aperture_radius, aperture_center)
    cn2_profile = Cn2Profile.from_r0s([r0],[L0], [layers_altitude], [wind_speed], [wind_direction])

    vk = VonKarmanSpatioTemporalCovariance(
        source1=source, source2=source, aperture1=aperture, aperture2=aperture,
        cn2_profile=cn2_profile, spat_freqs=space_freqs)

    modes = list(range(2, 2 + n_modes))

    # worker function to compute a single mode
    def compute_single_mode(m):
        psd = vk.getGeneralZernikeCPSD(j=[m], k=[m], temp_freqs=tempor_freqs)
        return np.real(psd[0, 0, :])

    PSD_atmo = []
    
    # -------------------------------------------------------------------------
    # Multiprocessing approach.
    # -------------------------------------------------------------------------
    print(f" -> Parallel computation started on {os.cpu_count()} cores for {n_modes} modes...")
    
    with ThreadPoolExecutor() as executor:
        for m, result in zip(modes, executor.map(compute_single_mode, modes)):
            PSD_atmo.append(result)
            if (m - 1) % 500 == 0:
                print(f"    -> Completed {m - 1}/{n_modes} modes...")

    PSD_atmo = np.array(PSD_atmo)

    # Convert from rad^2 to nm^2
    wvl_ref = 500e-9  
    rad2_to_nm2 = (wvl_ref * 1e9 / (2 * np.pi))**2
    PSD_atmo *= rad2_to_nm2

    return PSD_atmo


# Function to calculate the Fitting Error, see Equation (7) (in "Semianalytical error budget 
# for adaptive optics systems with pyramid wavefront sensors", Agapito and Pinna, 2019)

def fitting_variance(fitting_coeff, actuators_number, telescope_diameter, r0):
    
    var_fitting = fitting_coeff * actuators_number ** (-0.9) * (telescope_diameter/r0) ** (5/3)
    print("Fitting:", var_fitting)
    return var_fitting


# Funtion to compute the output PSD by multiplying the squared modulus of the transfert function with the
# input PSD.
# This term appears as the integrand in Equations (8), (10), and (15) in "Semianalytical error budget 
# for adaptive optics systems with pyramid wavefront sensors", Agapito and Pinna (2019).

def func_out(function, function_input):
    
    funtion_output = (np.abs(function)) ** 2 * (function_input)   
    
    return funtion_output
    

# Function to compute the numerical integral of the output PSD over the interval of 
# the considered frequencies using Simpson's function.

def integrate_function(integrand_function, integr_interval): 
    
    result = integrate.simpson(integrand_function, integr_interval)
    
    return result
    

# Function that allows us to load the windshake PSD and their corresponding frequencies

def load_PSD_windshake(file_path_wind): 
    
    with fits.open(file_path_wind) as hdul: 
        
        try: 
            
            data = hdul[0].data                                                # pylint: disable=E1101 
            frequencies = data[0, :]                                           # first row: frequencies 
            PSD_windshake = data[1:, :]                                        # second and third rows: PSD tip and tilt

            return frequencies, PSD_windshake

        except Exception as exc: 
        
            
            print(exc)
            return None, None
 
    
# Function to resize PSD so the number of modes matches the number of actuators

def align_psd_modes(PSD_in, actuators_number):

    PSD_in = np.asarray(PSD_in)  

    if PSD_in.ndim != 2:
        raise ValueError("PSD input must be a 2D array")

    n_modes_in, n_freq = PSD_in.shape

    PSD_out = np.zeros((actuators_number, n_freq), dtype=PSD_in.dtype)
    n_copy = min(n_modes_in, actuators_number)
    PSD_out[:n_copy, :] = PSD_in[:n_copy, :]

    return PSD_out 
 
    
# Function to resize the vibration PSD to match the size of the atmospheric turbulence PSD 
# by filling with zeros, if needed.

def resize_psd_like(PSD_atmo_turb, PSD_vibration):
    
    PSD_vib_1= np.zeros_like(PSD_atmo_turb)
    m = min(PSD_vibration.shape[0], PSD_vib_1.shape[0])
    PSD_vib_1[:m, :] = PSD_vibration[:m, :]
    
    return PSD_vib_1  


# Computes the output PSD by applying the corresponding transfer function to the input PSD 
# (using the function "func_out"). Than, the function integrates each modal output PSD over 
# the given frequency range, as described in Equation (8), (10) and (15) in "Semianalytical error budget 
# for adaptive optics systems with pyramid wavefront sensors", Agapito and Pinna (2019). 
# The function also allows the computation of the variance in the open loop case.

def compute_output_PSD_and_integrate(actuators_number, transf_funct, PSD_input, omega_temp_freq_interval):
    
    variance_OL = 0 
    variance_CL = 0 
    
    PSD_out = np.zeros_like(PSD_input)
    
    for i in range (actuators_number):                                  
         
        integral_OL = integrate_function(PSD_input[i, :], omega_temp_freq_interval)
        variance_OL += integral_OL
        
        PSD_out[i, :] = func_out(transf_funct[i, :], PSD_input[i, :])
        integral_CL = integrate_function(PSD_out[i, :], omega_temp_freq_interval)
        variance_CL += integral_CL
         
    return variance_OL, variance_CL, PSD_out


# Function to compute vibration variance in open loop and closed loop, returning also open loop and closed loop PSDs

def vibration_variance(PSD_vibration, transf_funct, actuators_number, omega_temp_freq_interval):
 
    PSD_vib = align_psd_modes(PSD_vibration, actuators_number)
    PSD_input = PSD_vib
    
    variance_vibr_OL, variance_vibr_CL, PSD_output = compute_output_PSD_and_integrate (actuators_number, transf_funct, 
                                                                                       PSD_input, omega_temp_freq_interval)

    print("Vibration_OL:", variance_vibr_OL)  
    print("Vibration_CL:", variance_vibr_CL)  
    
    return variance_vibr_OL, variance_vibr_CL, PSD_output, PSD_input 


# Computes the temporal variance by resizing the vibration PSD, summing atmospheric turbulence PSD 
# and vibration PSD, applying the transfer function, and integrating over the frequency interval.
# See Equation (8) (in "Semianalytical error budget for adaptive optics systems with pyramid wavefront sensors", 
# Agapito and Pinna, 2019)

def temporal_variance (PSD_atmo_turb, PSD_vibration, transf_funct, actuators_number, 
                       omega_temp_freq_interval): 

    PSD_atmo = align_psd_modes(PSD_atmo_turb, actuators_number)
    PSD_vib_aligned = align_psd_modes(PSD_vibration, actuators_number)
    PSD_vib = resize_psd_like(PSD_atmo, PSD_vib_aligned)

    PSD_input = PSD_atmo + PSD_vib

    variance_temp_OL, variance_temp_CL, PSD_output = compute_output_PSD_and_integrate(actuators_number, transf_funct, 
                                                                                      PSD_input, omega_temp_freq_interval)
    
    print("Temporal_OL:", variance_temp_OL) 
    print("Temporal_CL:", variance_temp_CL) 
    
    return variance_temp_OL, variance_temp_CL, PSD_output, PSD_input 


# Function to extract the noise propagation coefficients from the reconstruction matrix.

def extract_propagation_coefficients(file_path_matrix_R): 
    
    with fits.open(file_path_matrix_R) as hdul: 
        
        try: 

            if len(hdul) > 1:
                h = hdul[1]
            else:
                h = hdul[0]
            R = np.array(h.data.copy())                # pylint: disable=E1101
            if R.ndim == 2:    
                return np.diag(R @ R.T)
            else:
                # If the data is already a 1D array, we assume it's the diagonal of R @ R.T
                # and return it directly
                return R

        except Exception as exc:
            
            print(exc) 
            return None


# =============================================================================
# OPTICAL GAIN MODULE
# =============================================================================

# Function to build the optical gain grid from FITS files.
# It reads the last-mode optical gain values for two modulation radii and arranges them 
# into a 2D grid indexed by modulation radius and seeing.

def _load_andes_gain_grid(file_mod0, file_mod4):
    """
    Loads and stacks the ANDES optical gain data from two separate FITS files.
    """
    with fits.open(file_mod0) as hdul:
        gain_mod0 = hdul[0].data[:, -1]                # pylint: disable=E1101 
        
    with fits.open(file_mod4) as hdul:
        gain_mod4 = hdul[0].data[:, -1]                 # pylint: disable=E1101 
        
    return np.vstack([gain_mod0, gain_mod4])


# Function to compute the optical gain for a given modulation radius and seeing.
# Uses an optical gain grid from ANDES_og_mod0.fits and ANDES_og_mod4.fits and performs
# a 2D interpolation to estimate the gain for the given modulation radius and seeing

def compute_andes_optical_gain(file_mod0, file_mod4,
                               seeing, modulation_radius):
    """
    Computes the optical gain for the ANDES system using 2D interpolation.
    Axes: modulation radius, seeing.
    """
    gain_grid = _load_andes_gain_grid(file_mod0, file_mod4)
    
    # Define the grid axes for ANDES
    modal_radius_vals = np.array([0.0, 4.0])
    seeing_vals = np.array([0.4, 0.6, 0.8, 1.0, 1.2, 1.4])
    
    # 2D Interpolation
    interp_optical_gain = RegularGridInterpolator(
        (modal_radius_vals, seeing_vals), 
        gain_grid, 
        bounds_error=False, 
        fill_value=None
    )
    
    interpolated_gain = float(interp_optical_gain((modulation_radius, seeing)))
    return interpolated_gain


# Function to perform a 2D interpolation of the optical gain over modulation 
# radius and seeing using RegularGridInterpolator

def double_interpolation_optical_gain(modal_radius_val, seeing_val, optical_gain_grid, modulation_radius, seeing):
   
    interp_optical_gain = RegularGridInterpolator((modal_radius_val, seeing_val), optical_gain_grid, bounds_error=False, fill_value=None)  
     
    last_opt_gain = float(interp_optical_gain((modulation_radius, seeing)))
    return last_opt_gain


# SOUL optical gain data is stored in a single FITS file as a 3D data cube,
# with axes for modulated modes, binning, and magnitudes.

def _load_soul_gain_cube(filepath):
    """
    Loads the SOUL optical gain data cube and its axes from a single FITS file.
    """
    with fits.open(filepath) as hdul:
        magnitudes = hdul[1].data     # Shape: (11,)               # pylint: disable=E1101 
        binning = hdul[2].data        # Shape: (4,)                # pylint: disable=E1101 
        mod_modes = hdul[3].data      # Shape: (6,)                # pylint: disable=E1101 
        gain_cube = hdul[4].data      # Shape: (6, 4, 11)          # pylint: disable=E1101 
        
    return mod_modes, binning, magnitudes, gain_cube


# SOUL optical gain is computed by performing a 3D interpolation over the modulated modes,
# binning, and magnitudes.

def compute_soul_optical_gain(filepath, target_mod_modes, target_binning, target_magnitude):
    """
    Computes the optical gain for the SOUL system using 3D interpolation.
    Axes order must match the data cube shape: (mod_modes, binning, magnitudes).
    """
    mod_modes, binning, magnitudes, gain_cube = _load_soul_gain_cube(filepath)
    
    # 3D Interpolation
    interp_optical_gain = RegularGridInterpolator(
        (mod_modes, binning, magnitudes), 
        gain_cube, 
        bounds_error=False, 
        fill_value=None
    )
    
    interpolated_gain = float(interp_optical_gain((target_mod_modes, target_binning, target_magnitude)))
    return interpolated_gain


# Function to read and return the sigma slopes data from the FITS file.

def read_sigma_slopes(file_path_sigma_slopes=None):
    if file_path_sigma_slopes is None:
        file_path_sigma_slopes = DEFAULT_SIGMA_SLOPES_PATH
    
    
    with fits.open (file_path_sigma_slopes) as hdul:
    
        data = hdul[0].data                                              # pylint: disable=E1101   
        
        return data


# Function to perform 2D interpolation of the sigma slopes data for a given modulation
# radius and seeing.

def double_interpolation_sigma_slope(modal_radius_vals, seeing_vals, data_slopes, 
                                     modulation_radius, seeing):
    
    interp_sigma = RegularGridInterpolator((modal_radius_vals, seeing_vals), data_slopes[1,:,:], 
                                           bounds_error=False, fill_value=None) 
    sigma_slope_aliasing = float(interp_sigma((modulation_radius, seeing)))       
    
    return sigma_slope_aliasing


# Function to compute omega_0 (w_0)

def omega_0(telescope_diameter, windspeed, maximum_radial_order_corrected):

    omega0 = 2 * np.pi * (maximum_radial_order_corrected + 1) * windspeed/telescope_diameter
    
    return omega0


# Function to compute the preliminary aliasing coefficient k' based on sigma slope, 
# temporal frequency interval and system parameters. 

def compute_k_prime(omega_temp_freq_interval, alpha, sigma_slope_alias, telescope_diameter, 
                    windspeed, maximum_radial_order_corrected):
    
    w_0 = omega_0(telescope_diameter, windspeed, maximum_radial_order_corrected)
    
    omega_min = np.min(omega_temp_freq_interval) 
    omega_max = np.max(omega_temp_freq_interval) 
    integral_result = w_0 ** alpha * (w_0 - omega_min) + (omega_max ** (alpha + 1)
                                                                  - w_0 ** (alpha + 1)) / (alpha + 1)
    k_prime = (sigma_slope_alias ** 2) / integral_result  
    
    return k_prime
    
    
# Function to compute the aliasing modal coefficients k based on a given modulation
# radius and seeing.
# It uses sigma slope data from FITS files and applies 2D interpolation over 
# modulation radius and seeing.

def k_coeff_aliasing(modulation_radius, seeing, alpha, telescope_diameter, 
                     omega_temp_freq_interval, file_path_matrix_R, windspeed,
                     maximum_radial_order_corrected, file_path_sigma_slopes=None):
    
    data_slopes = read_sigma_slopes(file_path_sigma_slopes)  
    
    seeing_vals = data_slopes[0,0,:]                                           
    modal_radius_vals = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 8.0]) 
    
    sigma_slope_alias = double_interpolation_sigma_slope(modal_radius_vals, seeing_vals, data_slopes, 
                                                         modulation_radius, seeing)
    
    k_pr = compute_k_prime(omega_temp_freq_interval, alpha, sigma_slope_alias, telescope_diameter, 
                           windspeed, maximum_radial_order_corrected)
    
    p_coefficient = extract_propagation_coefficients(file_path_matrix_R)
    
    if p_coefficient is None:   
                                              
        raise RuntimeError("Propagation coefficients not loaded") 
   
    print("Propagation coefficients loaded successfully.")
   
    k_coeff = p_coefficient * k_pr
    
    return k_coeff


# Function to compute the aliasing PSD matrix from the modal coefficients k.

def aliasing_psd_from_coeffs(actuators_number, omega_temp_freq_interval, k, 
                             alpha, telescope_diameter, windspeed, maximum_radial_order_corrected):
    
    w_0 = omega_0(telescope_diameter, windspeed, maximum_radial_order_corrected)
    
    PSDaliasing = np.zeros((actuators_number, len(omega_temp_freq_interval)))
    
    for j in range(len(omega_temp_freq_interval)): 
        
        if omega_temp_freq_interval[j] <= w_0:
            
            for i in range (actuators_number): 
                
                PSDaliasing[i, j] = k[i] * (w_0 ** alpha)  
            
        else:  
            
            for i in range (actuators_number):   
            
                PSDaliasing[i, j] = k[i] * (omega_temp_freq_interval[j] ** alpha)
    
    return PSDaliasing  


# Computes the final aliasing PSD by scaling the intermediate aliasing PSD
# with the optical gain factor (1/c_optg^2).

def PSD_final_alias(c_optg, actuators_number, omega_temp_freq_interval, alpha,
                    telescope_diameter, seeing, modulation_radius, windspeed,
                    maximum_radial_order_corrected, file_path_matrix_R,
                    file_path_sigma_slopes=None):

    # optical gain effects are applied in the next step, so this is an "intermediate" results
    k = k_coeff_aliasing(modulation_radius, seeing, alpha, telescope_diameter,
                         omega_temp_freq_interval, file_path_matrix_R, windspeed,
                         maximum_radial_order_corrected, file_path_sigma_slopes)
    
    PSD_intermed = aliasing_psd_from_coeffs(actuators_number, omega_temp_freq_interval, 
                                            k, alpha, telescope_diameter, windspeed, 
                                            maximum_radial_order_corrected)

    return PSD_intermed / (c_optg**2)


# Computes the final measurement-noise PSD by scaling the intermediate noise PSD
# with the optical gain factor (1/c_optg^2).

def PSD_final_meas(c_optg, sigma2_w, actuators_number, omega_temp_freq_interval):

    # optical gain effects are applied in the next step, so this is an "intermediate" results
    PSD_intermed = compute_noise_PSD_intermediate(
        omega_temp_freq_interval,
        actuators_number,
        sigma2_w,
    )

    return PSD_intermed / (c_optg**2)
  

# Computes the aliasing variance by applying the transfer function to the aliasing PSD 
# and integrating the output over the specified frequency interval.
# See Equation (15) (in "Semianalytical error budget for adaptive optics systems
# with pyramid wavefront sensors", Agapito and Pinna, 2019).  

def aliasing_variance (transf_funct, actuators_number, omega_temp_freq_interval, 
                       c_optg, alpha, telescope_diameter, seeing, modulation_radius, 
                       windspeed, maximum_radial_order_corrected, file_path_matrix_R,
                       file_path_sigma_slopes=None):
    
    PSD_input = PSD_final_alias(
        c_optg,
        actuators_number,
        omega_temp_freq_interval,
        alpha,
        telescope_diameter,
        seeing,
        modulation_radius,
        windspeed,
        maximum_radial_order_corrected,
        file_path_matrix_R,
        file_path_sigma_slopes,
    )
    
    variance_alias_OL, variance_alias_CL, PSD_output = compute_output_PSD_and_integrate(actuators_number, transf_funct, 
                                                                                        PSD_input, omega_temp_freq_interval)
    
    print("Aliasing_OL:", variance_alias_OL)  
    print("Aliasing_CL:", variance_alias_CL)  
    
    return variance_alias_OL, variance_alias_CL, PSD_output, PSD_input 
    

# Function to compute the photon flux per frame per pixel, accounting for frame rate, 
# sub-aperture geometry, and source magnitude.

def flux_for_frame_for_pixel(photon_flux, telescope_diameter, frame_rate, magnitudo,
                             n_subaperture, collecting_area):
    
    flux_per_frame = photon_flux / frame_rate
    
    sub_aperture_radius = (telescope_diameter / n_subaperture)/2
    
    sub_aperture_area = np.pi * (sub_aperture_radius) ** 2

    flux_per_frame_per_sub_aperture = flux_per_frame * (sub_aperture_area / collecting_area)
    
    flux_per_frame_per_sub_aperture_magnitudo = flux_per_frame_per_sub_aperture * 10 ** (- magnitudo / 2.5)
    
    flux_per_frame_per_sub_aperture_magnitudo_per_pixel = flux_per_frame_per_sub_aperture_magnitudo / 4

    return flux_per_frame_per_sub_aperture_magnitudo_per_pixel


# Computes the pixel variance, as described in Equation (14) in "Semianalytical error budget 
# for adaptive optics systems with pyramid wavefront sensors", Agapito and Pinna (2019). 
# Than computes the slope noise variance, using the pixel variance and weighting 
# pixel noise by their positions, as described in Equation (12) in "Semianalytical error budget 
# for adaptive optics systems with pyramid wavefront sensors", Agapito and Pinna (2019).

def compute_slope_noise_variance(F_excess, pixel_pos, sky_bkg, dark_curr, read_out_noise,
                                 photon_flux, telescope_diameter,frame_rate, magnitudo, 
                                 n_subaperture, collecting_area):
    
    n_phot_pix = flux_for_frame_for_pixel(photon_flux, telescope_diameter, frame_rate, magnitudo,
                                          n_subaperture, collecting_area)
    
    pixel_pos = np.array(pixel_pos)                                            
    pixel_variance = F_excess ** 2 * (n_phot_pix + sky_bkg + dark_curr) + read_out_noise**2
    pix_intensity = n_phot_pix
    slope_variance = np.sum((pixel_pos ** 2) * pixel_variance) / (4 * (pix_intensity)) ** 2   
   
    return slope_variance
 

# Function which returns the noise PSD (PSD_w), assuming that the noise w has a flat temporal
# PSD over the entire frequency range, as stated in "Semianalytical error budget 
# for adaptive optics systems with pyramid wavefront sensors", Agapito and Pinna (2019).

def compute_noise_PSD_intermediate (omega_temp_freq_interval, actuators_number, sigma2_w):
    
    PSD_w_intermed = np.zeros((actuators_number, len(omega_temp_freq_interval)))
  
    omega_interval_min = np.min(omega_temp_freq_interval)                 
    omega_interval_max = np.max(omega_temp_freq_interval)
   
    for i in range(actuators_number):

        PSD_w_intermed[i, :] = sigma2_w[i] /(omega_interval_max -  omega_interval_min) 

    return PSD_w_intermed


# Calculates the slope noise variance, extracts the propagation coefficients from the FITS
# file, and computes the measurement noise variance for each mode. 
# Then, it calculates the corresponding measurement noise PSD, to obtain the total measurement 
# variance, as described in Equation (10) (in "Semianalytical error budget 
# for adaptive optics systems with pyramid wavefront sensors", Agapito and Pinna (2019). 

def measure_variance (F_excess, pixel_pos, sky_bkg, dark_curr, read_out_noise,
                      photon_flux, telescope_diameter,frame_rate, magnitudo, n_subaperture, 
                      collecting_area, file_path_matrix_R, omega_temp_freq_interval, 
                      transf_funct, actuators_number, c_optg, alpha, maximum_radial_order_corrected,
                      seeing, modulation_radius, windspeed, file_path_sigma_slopes=None):
    
  
    slope_noise_variance = compute_slope_noise_variance(F_excess, pixel_pos, sky_bkg, dark_curr, 
                                                        read_out_noise, photon_flux, telescope_diameter,
                                                        frame_rate, magnitudo, n_subaperture, collecting_area)
   
    p_coefficient = extract_propagation_coefficients(file_path_matrix_R)
    
    if p_coefficient is None:                                                
        
        raise RuntimeError("Propagation coefficients not loaded") 
   
    print("Propagation coefficients loaded successfully.")
    
    if len(p_coefficient) < actuators_number:
        raise ValueError("Not enough propagation coefficients for the selected number of modes")

    p_coefficient = p_coefficient[:actuators_number]
    sigma2_w = p_coefficient * slope_noise_variance
    
    
    PSD_input = PSD_final_meas(
        c_optg,
        sigma2_w,
        actuators_number,
        omega_temp_freq_interval,
    )
    
    variance_meas_OL, variance_meas_CL, PSD_output = compute_output_PSD_and_integrate(actuators_number, transf_funct, 
                                                                                      PSD_input, omega_temp_freq_interval)
    print("Measure_OL:", variance_meas_OL)  
    print("Measure_CL:", variance_meas_CL)  
    
    return variance_meas_OL, variance_meas_CL, PSD_output, PSD_input 
 

# Function to interpolate a 1D vector to a new set of points, setting values outside the original range to 0.

def interpolate_vector(x_interpolation, x_original, vector_original):
    
    vector_interpolated = np.interp (x_interpolation, x_original, vector_original, left = 0, right = 0) 
    
    return vector_interpolated


# Function to interpolate and normalize PSD to a new frequency interval.

def interpolate_and_normalize_psd(freqs_interpolation, freqs_original, PSD_original, actuators_number):

    PSD_original = np.asarray(PSD_original)
    PSD_interpolated = np.zeros((actuators_number, len(freqs_interpolation)))
    PSD_interpolated_normalized = np.zeros_like(PSD_interpolated)
    sigma2 = np.zeros(actuators_number)
    sigma2_interp = np.zeros(actuators_number)
    
    Omega_freqs_interpolation = 2 * np.pi * freqs_interpolation  
    Omega_freqs_original = 2 * np.pi * freqs_original
    
    n_modes_available = min(actuators_number, PSD_original.shape[0])

    for i in range(n_modes_available):

        PSD_interpolated[i, :] = interpolate_vector(freqs_interpolation, freqs_original, PSD_original[i, :])

        sigma2[i] = integrate.simpson(PSD_original[i, :], Omega_freqs_original)
        sigma2_interp[i] = integrate.simpson(PSD_interpolated[i, :], Omega_freqs_interpolation)

        if sigma2_interp[i] > 0:
            PSD_interpolated_normalized[i, :] = PSD_interpolated[i, :] * sigma2[i] / sigma2_interp[i]

    return PSD_interpolated_normalized


# Function to obtain the PSDs (temp, alias, meas) OL and CL 

def compute_PSD_OL_CL (PSD_atmo_turb, PSD_vibration, omega_temp_freq_interval, actuators_number, 
                       alpha, telescope_diameter, seeing, modulation_radius, windspeed, 
                       maximum_radial_order_corrected, c_optg, F_excess, pixel_pos, sky_bkg, 
                       dark_curr, read_out_noise, photon_flux, frame_rate, magnitudo, 
                       n_subaperture, collecting_area, temporal_frequencies, frequencies,
                       H_r, H_n, file_path_matrix_R, file_path_sigma_slopes):
    
    if np.array_equal(temporal_frequencies, frequencies):
    
        _, _, PSD_output_temp, PSD_input_temp = temporal_variance (PSD_atmo_turb, PSD_vibration, H_r,  
                                                                   actuators_number, omega_temp_freq_interval)
        
    else:
        
        PSD_wind_vib_interp_normalized = interpolate_and_normalize_psd(temporal_frequencies, frequencies, PSD_vibration, actuators_number)
        _, _, PSD_output_temp, PSD_input_temp = temporal_variance (PSD_atmo_turb, PSD_wind_vib_interp_normalized, 
                                                                   H_r, actuators_number, omega_temp_freq_interval)
        
        
    
    _, _, PSD_output_alias, PSD_input_alias = aliasing_variance (H_n, actuators_number, omega_temp_freq_interval, c_optg,
                                                                 alpha, telescope_diameter, seeing, modulation_radius, windspeed, 
                                                                 maximum_radial_order_corrected, file_path_matrix_R,  
                                                                 file_path_sigma_slopes)  

    
    _, _, PSD_output_meas, PSD_input_meas = measure_variance (F_excess, pixel_pos, sky_bkg, dark_curr, read_out_noise,
                                                              photon_flux, telescope_diameter,frame_rate, magnitudo, 
                                                              n_subaperture, collecting_area, file_path_matrix_R, 
                                                              omega_temp_freq_interval, H_n, actuators_number,
                                                              c_optg, alpha, maximum_radial_order_corrected,
                                                              seeing, modulation_radius, windspeed, 
                                                              file_path_sigma_slopes)
    
   
    return PSD_output_temp, PSD_input_temp, PSD_output_alias, PSD_input_alias, PSD_output_meas, PSD_input_meas


# Function to compute the total PSDs (temp + alias + meas) OL and CL 

def total_PSD_OL_CL(omega_temp_freq_interval, t_0, actuators_number, plant_num, plant_den,
                    PSD_atmo_turb, PSD_vibration, alpha, telescope_diameter, seeing, modulation_radius, windspeed,
                    maximum_radial_order_corrected, c_optg, F_excess, pixel_pos, sky_bkg, dark_curr, read_out_noise,
                    photon_flux, frame_rate, magnitudo, n_subaperture, collecting_area, temporal_frequencies, frequencies,
                    file_path_matrix_R, file_path_sigma_slopes,
                    *, gain=None, controller_num=None, controller_den=None):

    H_r, H_n = build_transfer_function(
        omega_temp_freq_interval,
        t_0,
        actuators_number,
        plant_num,
        plant_den,
        gain=gain,
        controller_num=controller_num,
        controller_den=controller_den,
    )
    
   
    PSD_out_temp, PSD_in_temp, PSD_out_alias, PSD_in_alias, PSD_out_meas, PSD_in_meas = compute_PSD_OL_CL(PSD_atmo_turb, PSD_vibration, 
                                                                                                          omega_temp_freq_interval, actuators_number, 
                                                                                                          alpha, telescope_diameter, seeing, 
                                                                                                          modulation_radius, windspeed, 
                                                                                                          maximum_radial_order_corrected, 
                                                                                                          c_optg, F_excess, pixel_pos, 
                                                                                                          sky_bkg, dark_curr, read_out_noise,
                                                                                                          photon_flux, frame_rate, magnitudo, 
                                                                                                          n_subaperture, collecting_area, 
                                                                                                          temporal_frequencies, frequencies, 
                                                                                                          H_r, H_n, file_path_matrix_R,  
                                                                                                          file_path_sigma_slopes)
    
    
    PSD_total_input = PSD_in_temp + PSD_in_alias + PSD_in_meas
    
    PSD_total_output = PSD_out_temp + PSD_out_alias + PSD_out_meas
    
    return PSD_total_input, PSD_total_output


# Function to compute the total variance by summing fitting variance, temporal variance, and meas variance contributions.

def total_variance(fit_err, temp_err, alias_err, meas_err):
    var_tot = np.real(fit_err) + np.real(temp_err) + np.real(meas_err) + np.real(alias_err)
    print ("Total variance:", var_tot)
    return var_tot 


@dataclass
class SingleModeControllerOptimizationRecord:
    controller_num: np.ndarray
    controller_den: np.ndarray
    cost: float
    variance_terms: dict


@dataclass
class SingleModeControllerOptimizationResult:
    controller_num: np.ndarray
    controller_den: np.ndarray
    cost: float
    variance_terms: dict
    psd_input: dict
    psd_output: dict
    transfer_function: dict


@dataclass
class SingleModeControllerOptimizationContext:
    mode_index: int
    omega_temp_freq_interval: np.ndarray
    t_0: float
    num1: np.ndarray
    num2: np.ndarray
    num3: np.ndarray
    den1: np.ndarray
    den2: np.ndarray
    den3: np.ndarray
    static_fit_variance: float
    PSD_input_atmos: np.ndarray
    PSD_input_vibration: np.ndarray
    PSD_input_alias: np.ndarray
    PSD_input_measurement: np.ndarray
    history: list = field(default_factory=list)

    def evaluate(self, controller_num, controller_den, store_history=True):
        plant_num = np.polymul(np.polymul(np.asarray(self.num1), np.asarray(self.num2)), np.asarray(self.num3))
        plant_den = np.polymul(np.polymul(np.asarray(self.den1), np.asarray(self.den2)), np.asarray(self.den3))
        H_r, H_n = build_transfer_function_from_controller_polynomials(
            controller_num,
            controller_den,
            self.omega_temp_freq_interval,
            self.t_0,
            1,
            plant_num,
            plant_den,
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
        cost = total_variance(self.static_fit_variance, var_temp_total, var_alias, var_measurement)

        variance_terms = {
            "fitting": float(np.real(self.static_fit_variance)),
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
                    self.PSD_input_atmos + self.PSD_input_vibration + self.PSD_input_alias + self.PSD_input_measurement,
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


def _select_single_mode_psd(PSD_in, mode_index, array_name, n_frequencies, allow_missing=False):

    PSD_in = np.asarray(PSD_in, dtype=float)

    if PSD_in.ndim != 2:
        raise ValueError(f"{array_name} must be a 2D array")

    if PSD_in.shape[1] != n_frequencies:
        raise ValueError(
            f"{array_name} must have {n_frequencies} frequency samples; got {PSD_in.shape[1]}"
        )

    if mode_index < PSD_in.shape[0]:
        return PSD_in[mode_index:mode_index + 1, :]

    if allow_missing:
        return np.zeros((1, n_frequencies), dtype=float)

    raise ValueError(
        f"Requested mode_index={mode_index} for {array_name}, but only {PSD_in.shape[0]} modes are available"
    )


def prepare_single_mode_control_optimization(mode_index, omega_temp_freq_interval, t_0,
                                             PSD_atmo_turb, PSD_vibration,
                                             alpha, telescope_diameter, seeing,
                                             modulation_radius, windspeed,
                                             maximum_radial_order_corrected, c_optg,
                                             F_excess, pixel_pos, sky_bkg, dark_curr,
                                             read_out_noise, photon_flux, frame_rate,
                                             magnitudo, n_subaperture, collecting_area,
                                             file_path_matrix_R,
                                             file_path_sigma_slopes=None,
                                             static_fit_variance=0.0,
                                             num1=None, num2=None, num3=None,
                                             den1=None, den2=None, den3=None):
    """
    Prepare a reusable single-mode optimization context for an IIR controller.

    This function precomputes all controller-independent inputs:
      - temporal input PSDs (atmosphere and vibration),
      - aliasing input PSD,
      - measurement input PSD,
      - static cost contribution (e.g. fitting variance).

    The returned context can then evaluate many controller candidates cheaply,
    updating only H_r, H_n and the resulting cost.
    """

    if mode_index < 0:
        raise ValueError("mode_index must be >= 0")

    omega_temp_freq_interval = np.asarray(omega_temp_freq_interval, dtype=float).ravel()

    if omega_temp_freq_interval.size == 0:
        raise ValueError("omega_temp_freq_interval must not be empty")

    if np.isclose(c_optg, 0.0):
        raise ValueError("c_optg must be non-zero to build aliasing and measurement PSDs")

    n_frequencies = omega_temp_freq_interval.size

    if num1 is None:
        num1 = np.array([1.0])
    if num2 is None:
        num2 = np.array([1.0])
    if num3 is None:
        num3 = np.array([1.0])
    if den1 is None:
        den1 = np.array([1.0])
    if den2 is None:
        den2 = np.array([1.0])
    if den3 is None:
        den3 = np.array([1.0])

    PSD_input_atmos = _select_single_mode_psd(PSD_atmo_turb, mode_index, "PSD_atmo_turb",
                                              n_frequencies, allow_missing=False)
    PSD_input_vibration = _select_single_mode_psd(PSD_vibration, mode_index, "PSD_vibration",
                                                  n_frequencies, allow_missing=True)

    k_alias = np.asarray(
        k_coeff_aliasing(modulation_radius, seeing, alpha, telescope_diameter,
                         omega_temp_freq_interval, file_path_matrix_R, windspeed,
                         maximum_radial_order_corrected, file_path_sigma_slopes),
        dtype=float,
    )

    if mode_index >= k_alias.size:
        raise ValueError(
            f"Requested mode_index={mode_index}, but aliasing coefficients are available only up to {k_alias.size - 1}"
        )

    PSD_input_alias = aliasing_psd_from_coeffs(
        1, omega_temp_freq_interval, np.array([k_alias[mode_index]], dtype=float),
        alpha, telescope_diameter, windspeed, maximum_radial_order_corrected
    ) / (c_optg ** 2)

    slope_noise_variance = compute_slope_noise_variance(
        F_excess, pixel_pos, sky_bkg, dark_curr, read_out_noise,
        photon_flux, telescope_diameter, frame_rate, magnitudo,
        n_subaperture, collecting_area
    )

    p_coefficient = np.asarray(extract_propagation_coefficients(file_path_matrix_R), dtype=float)

    if mode_index >= p_coefficient.size:
        raise ValueError(
            f"Requested mode_index={mode_index}, but propagation coefficients are available only up to {p_coefficient.size - 1}"
        )

    sigma2_w = np.array([p_coefficient[mode_index] * slope_noise_variance], dtype=float)
    PSD_input_measurement = compute_noise_PSD_intermediate(omega_temp_freq_interval, 1, sigma2_w) / (c_optg ** 2)

    return SingleModeControllerOptimizationContext(
        mode_index=mode_index,
        omega_temp_freq_interval=omega_temp_freq_interval,
        t_0=float(t_0),
        num1=np.asarray(num1, dtype=float),
        num2=np.asarray(num2, dtype=float),
        num3=np.asarray(num3, dtype=float),
        den1=np.asarray(den1, dtype=float),
        den2=np.asarray(den2, dtype=float),
        den3=np.asarray(den3, dtype=float),
        static_fit_variance=float(np.real(static_fit_variance)),
        PSD_input_atmos=PSD_input_atmos,
        PSD_input_vibration=PSD_input_vibration,
        PSD_input_alias=PSD_input_alias,
        PSD_input_measurement=PSD_input_measurement,
    )


         
























