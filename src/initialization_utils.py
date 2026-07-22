from types import SimpleNamespace

import numpy as np
import control as ct
from scipy import integrate
from src.Functions import (
    load_parameters,
    load_PSD_windshake,
    seeing_to_r0,
    funct_d2,
    radial_order_from_n_modes,
    turbulence_psd,
    compute_optical_gain,
    fitting_variance,
    DEFAULT_ALIASING_ALPHA)

from src.controller_optimization import prepare_single_mode_control_optimization

DEFAULT_FITTING_COEFF = 0.28

# initialize the parameters

def init_parameters(param_dir,alpha_=DEFAULT_ALIASING_ALPHA):
    
    param = load_parameters(param_dir)
    
    # load parameters from yaml file
    n_actuators = param['control']['n_modes']
    D = param['telescope']['telescope_diam']
    aperture_radius = D / 2.0
    aperture_center = [0, 0, 0]

    L0 = param['atmosphere']['outer_scale']
    layers_altitude = 0.0
    wind_direction = 0.0
    wind_speed = param['atmosphere']['wind_speed']
    seeing = param['atmosphere']['seeing']
    r0 = seeing_to_r0(seeing)

    F_excess_noise = np.sqrt(param['wavefront_sensor']['value_for_F_excess_noise'])
    sky_background = param['wavefront_sensor']['sky_backgr']
    dark_current = param['wavefront_sensor']['dark_curr']
    readout_noise = param['wavefront_sensor']['noise_readout']

    file_path_R1 = param['data']['reconstruction_matrix']
    sigma_slopes_path = param['data']['sigma_slopes']
    file_path_wind = param['data']['windshake_psd']
    file_optg = param['data']['optical_gain_models']

    t_0 = param['control']['sampling_time']
    frame_rate = 1.0 / t_0
    temporal_freqs_min = param['frequency_ranges']['temporal_freqs_min']
    temporal_freqs_n = param['frequency_ranges']['temporal_freqs_n']
    
    temporal_freqs = np.logspace(temporal_freqs_min, np.log10(frame_rate / 2.0), temporal_freqs_n)
    omega = 2 * np.pi * temporal_freqs

    spatial_freqs = np.logspace(-4, 4, 300)

    total_delay = param['plant']['total_delay']  
    plant_num = np.asarray(param['plant']['numerator'])
    plant_den_base = np.asarray(param['plant']['denominator'])
    plant_den = np.polymul(plant_den_base, funct_d2(total_delay)) 
    plant_tf = ct.tf(plant_num, plant_den, t_0)
    
    modulation_radius = param['wavefront_sensor']['modulation_radius']
    maximum_radial_order = radial_order_from_n_modes(n_actuators)

    phot_flux = float(param['guide_star']['flux_photons'])
    magnitude = param['guide_star']['magn']
    n_subapert = param['wavefront_sensor']['number_of_sub']
    x_pixel = param['control']['slope_computer_weights']
    
    controller_type = param['optimization'].get('ctrl_type')
    
    if controller_type is not None:
        if controller_type == 1:  # integral controller
            gain_value = param['control'].get('gain_value')
            if gain_value is not None:
                gain_array = np.full(n_actuators, float(np.asarray(gain_value).ravel()[0]))
                ctrl_num_array = None
                ctrl_den_array = None
                gain_min = float(param['control']['gain_min'])
                gain_max = 1.0
                if gain_min >= gain_max:
                    raise ValueError(f"gain_min ({gain_min}) must be less than gain_max ({gain_max})")
                control_dirt = {
                                'gain_array': gain_array,
                                'gain_min': gain_min,
                                'gain_max': gain_max                                    
                                }
            else:
                gain_array = np.full(n_actuators, float(param['control']['gain_min']))
                ctrl_num_array = None
                ctrl_den_array = None
                gain_min = float(param['control']['gain_min'])
                gain_max = 1.0
                control_dirt = {
                                'gain_array': gain_array,
                                'gain_min': gain_min,
                                'gain_max': gain_max
                                }
        elif controller_type == 2: # polynomial controller
            ctrl_order = param['optimization'].get('order')
            if ctrl_order is not None:
                gain_array = None
                n_num_poly = ctrl_order[0] + 1
                n_den_poly = ctrl_order[1] + 1
                ctrl_num_array = np.zeros(n_num_poly, dtype=float)
                ctrl_num_array[0] = 1.0
                ctrl_den_array = np.zeros(n_den_poly, dtype=float)
                ctrl_den_array[0] = 1.0
                control_dirt = {
                                'ctrl_num_array': ctrl_num_array,
                                'ctrl_den_array': ctrl_den_array,
                                }
            else:
                raise ValueError("Provide controller's order (polynomial controller)")    
        elif controller_type == 3:
            gain_value = param['control'].get('gain_value')
            forgetting_factor = param['control'].get('forgetting_factor')
            gain_max = 1.0
            if gain_value is not None:
                gain_array = np.full(n_actuators, float(np.asarray(gain_value).ravel()[0]))
                gain_min = float(param['control']['gain_min'])
                if gain_min >= gain_max:
                    raise ValueError(f"gain_min ({gain_min}) must be less than gain_max ({gain_max})")
            else:
                gain_array = np.full(n_actuators, float(param['control']['gain_min']))
                gain_min = float(param['control']['gain_min'])
                if gain_min >= gain_max:
                    raise ValueError(f"gain_min ({gain_min}) must be less than gain_max ({gain_max})")
            if forgetting_factor is not None:
                forgetting_factor_array = np.full(n_actuators, float(np.asarray(forgetting_factor).ravel()[0]))
            else:
                forgetting_factor_array = np.full(n_actuators, 0.8)
            control_dirt = {
                            'gain_array': gain_array,
                            'gain_min': gain_min,
                            'gain_max': gain_max,
                            'ff_array': forgetting_factor_array,
                            'ff_min': 0.7,
                            'ff_max': 1.0,
                            }                
        else:
            raise ValueError("Provide wrong 'ctrl_type'. Please check. ")
    else:
        raise ValueError("Provide 'ctrl_type' ") 
    
    # 2. Generate Atmospheric Input
    PSD_atmosf = turbulence_psd(0, 0, aperture_radius, aperture_center,
                                r0, L0, layers_altitude,
                                wind_speed, wind_direction,
                                spatial_freqs, temporal_freqs,
                                n_modes=n_actuators)
    
    freq, PSD_wind_vib = load_PSD_windshake(file_path_wind, temporal_freqs)

    if (freq is None and PSD_wind_vib is None) or (freq is None or PSD_wind_vib is None):                                     
    
        raise RuntimeError("PSD windshake or corresponding frequencies not loaded") 

    print("PSD windshake and corresponding frequencies loaded successfully.") 

    # PSD_vibration = np.zeros_like(PSD_atmosf)

    
    # 4. Compute Optical Gain (needed for aliasing)
    # c_optg = compute_optical_gain(file_mod0=file_mod0,
    #                             file_mod1=file_mod4,
    #                             seeing=seeing,
    #                             modulation_radius=modulation_radius,
    #                             actuators_number=n_actuators)
    
    c_optg = compute_optical_gain(file_optg[0], file_optg[1], seeing, 
                                    modulation_radius, n_actuators,
                                    modulation_radii=(0.0, 4.0))
    
    # 5. initialize the optimization context for single mode control optimization
    
    # fitting variance
    static_fit_variance = fitting_variance(
        fitting_coeff=DEFAULT_FITTING_COEFF, 
        actuators_number=n_actuators, 
        telescope_diameter=D, 
        r0=r0)
    
    main_dict = {
        'omega': omega,
        't_0': t_0,
        'PSD_atmosf': PSD_atmosf,
        'PSD_vibration': PSD_wind_vib,
        'D': D,
        'seeing': seeing,
        'modulation_radius': modulation_radius,
        'wind_speed': wind_speed,
        'maximum_radial_order': maximum_radial_order,
        'c_optg': c_optg,
        'F_excess_noise': F_excess_noise,
        'sky_background': sky_background,
        'dark_current': dark_current,
        'readout_noise': readout_noise,
        'phot_flux': phot_flux,
        'frame_rate': frame_rate,
        'magnitude': magnitude,
        'n_subapert': n_subapert,
        'file_path_R1': file_path_R1,
        'alpha_': alpha_,
        'file_path_sigma_slopes': sigma_slopes_path,
        'static_fit_variance': static_fit_variance,
        'x_pixel': x_pixel,
        'temporal_freqs': temporal_freqs,
        'plant_tf': plant_tf,
        'plant_num': plant_num,
        'plant_den': plant_den,
        'controller_type': controller_type
    }
    
    merged_dict = {**main_dict, **control_dirt}
    
    return merged_dict

def init_optimization_context(init_params, mode_index):
    init_params=SimpleNamespace(**init_params)
    obj_to_optimize = prepare_single_mode_control_optimization(
        mode_index=mode_index,
        omega_temp_freq_interval=init_params.omega,
        t_0=init_params.t_0,
        PSD_atmo_turb=init_params.PSD_atmosf,
        PSD_vibration=init_params.PSD_vibration,
        telescope_diameter=init_params.D,
        seeing=init_params.seeing,
        modulation_radius=init_params.modulation_radius,
        windspeed=init_params.wind_speed,
        maximum_radial_order_corrected=init_params.maximum_radial_order,
        c_optg=init_params.c_optg[mode_index],
        F_excess=init_params.F_excess_noise,
        pixel_pos=init_params.x_pixel,
        sky_bkg=init_params.sky_background,
        dark_curr=init_params.dark_current,
        read_out_noise=init_params.readout_noise,
        photon_flux=init_params.phot_flux,
        frame_rate=init_params.frame_rate,
        magnitudo=init_params.magnitude,
        n_subaperture=init_params.n_subapert,
        file_path_matrix_R=init_params.file_path_R1,
        alpha=init_params.alpha_,
        file_path_sigma_slopes=init_params.file_path_sigma_slopes,
        # static_fit_variance=init_params.static_fit_variance,
        plant_num=init_params.plant_num,
        plant_den=init_params.plant_den
    )
    return obj_to_optimize

def input_variance_singlemode(evaluate_input, temporal_freqs, mode_index=0):
    # note: the PSDs in evaluate_input are based on nm/(rad*s)
    PSD_in_temp = evaluate_input.psd_input["temporal"][mode_index, :]
    PSD_in_atmos = evaluate_input.psd_input["atmosphere"][mode_index, :]
    PSD_in_vibra = evaluate_input.psd_input["vibration"][mode_index, :]
    PSD_in_alias = evaluate_input.psd_input["aliasing"][mode_index, :]
    PSD_in_meas  = evaluate_input.psd_input["measurement"][mode_index, :]
    PSD_in_total = evaluate_input.psd_input["total"][mode_index, :]
    
    PSD_in_temp = np.real(PSD_in_temp)
    PSD_in_atmos = np.real(PSD_in_atmos)
    PSD_in_vibra = np.real(PSD_in_vibra)
    PSD_in_alias = np.real(PSD_in_alias)
    PSD_in_meas = np.real(PSD_in_meas)
    PSD_in_total = np.real(PSD_in_total)
    
    # Variance calculation
    var_temp_in = integrate.simpson(PSD_in_temp, temporal_freqs)
    var_atmos_in = integrate.simpson(PSD_in_atmos, temporal_freqs)
    var_vibra_in = integrate.simpson(PSD_in_vibra, temporal_freqs)
    var_alias_in = integrate.simpson(PSD_in_alias, temporal_freqs)
    var_meas_in = integrate.simpson(PSD_in_meas, temporal_freqs)
    var_total_in = integrate.simpson(PSD_in_total, temporal_freqs)
    
    return {
        'var_temp_in': var_temp_in,
        'var_atmos_in': var_atmos_in,
        'var_vibra_in': var_vibra_in,
        'var_alias_in': var_alias_in,
        'var_meas_in': var_meas_in,
        'var_total_in': var_total_in
    }

def output_variance_singlemode(evaluate_output, temporal_freqs, mode_index=0):
    PSD_out_temp = evaluate_output.psd_output["temporal"][mode_index, :]
    PSD_out_atmos = evaluate_output.psd_output["atmosphere"][mode_index, :]
    PSD_out_vibra = evaluate_output.psd_output["vibration"][mode_index, :]
    PSD_out_alias = evaluate_output.psd_output["aliasing"][mode_index, :]
    PSD_out_meas  = evaluate_output.psd_output["measurement"][mode_index, :]
    PSD_out_total = evaluate_output.psd_output["total"][mode_index, :]
    
    PSD_out_temp = np.real(PSD_out_temp)
    PSD_out_atmos = np.real(PSD_out_atmos)
    PSD_out_vibra = np.real(PSD_out_vibra)
    PSD_out_alias = np.real(PSD_out_alias)
    PSD_out_meas = np.real(PSD_out_meas)
    PSD_out_total = np.real(PSD_out_total)
    
    # Variance calculation
    var_temp_out = integrate.simpson(PSD_out_temp, temporal_freqs)
    var_atmos_out = integrate.simpson(PSD_out_atmos, temporal_freqs)
    var_vibra_out = integrate.simpson(PSD_out_vibra, temporal_freqs)
    var_alias_out = integrate.simpson(PSD_out_alias, temporal_freqs)
    var_meas_out = integrate.simpson(PSD_out_meas, temporal_freqs)
    var_total_out = integrate.simpson(PSD_out_total, temporal_freqs)
    
    return {
        'var_temp_out': var_temp_out,
        'var_atmos_out': var_atmos_out,
        'var_vibra_out': var_vibra_out,
        'var_alias_out': var_alias_out,
        'var_meas_out': var_meas_out,
        'var_total_out': var_total_out
    }

def print_psd_variance_terms(evaluate, temporal_freqs, mode_index=0, mode="in"):
    if mode == "in":
        variance_in = input_variance_singlemode(evaluate, temporal_freqs, mode_index)
        print(f" - Atmospheric variance: {format_3digits(variance_in['var_atmos_in'])}")
        print(f" - Vibration variance: {format_3digits(variance_in['var_vibra_in'])}")
        print(f" - Aliasing variance: {format_3digits(variance_in['var_alias_in'])}")
        print(f" - Measurement variance: {format_3digits(variance_in['var_meas_in'])}")
        print(f" - Total variance: {format_3digits(variance_in['var_total_in'])}")
        print(f" - Temporal variance: {format_3digits(variance_in['var_temp_in'])}")
        print("")
    elif mode == "out":
        variance_out = output_variance_singlemode(evaluate, temporal_freqs, mode_index)
        print(f" - Atmospheric variance: {format_3digits(variance_out['var_atmos_out'])}")
        print(f" - Vibration variance: {format_3digits(variance_out['var_vibra_out'])}")
        print(f" - Aliasing variance: {format_3digits(variance_out['var_alias_out'])}")
        print(f" - Measurement variance: {format_3digits(variance_out['var_meas_out'])}")
        print(f" - Total variance: {format_3digits(variance_out['var_total_out'])}")
        print(f" - Temporal variance: {format_3digits(variance_out['var_temp_out'])}")
        print("")
    else:
        raise ValueError("Invalid mode. Use 'in' for input variance or 'out' for output variance.")

def input_RMS_singlemode(evaluate_input, omega, mode_index=0):
    # note: the PSDs in evaluate_input are based on nm/(rad*s)
    PSD_in_temp = evaluate_input.psd_input["temporal"][mode_index, :]
    PSD_in_atmos = evaluate_input.psd_input["atmosphere"][mode_index, :]
    PSD_in_vibra = evaluate_input.psd_input["vibration"][mode_index, :]
    PSD_in_alias = evaluate_input.psd_input["aliasing"][mode_index, :]
    PSD_in_meas  = evaluate_input.psd_input["measurement"][mode_index, :]
    PSD_in_total = evaluate_input.psd_input["total"][mode_index, :]
    
    PSD_in_temp = np.real(PSD_in_temp)
    PSD_in_atmos = np.real(PSD_in_atmos)
    PSD_in_vibra = np.real(PSD_in_vibra)
    PSD_in_alias = np.real(PSD_in_alias)
    PSD_in_meas = np.real(PSD_in_meas)
    PSD_in_total = np.real(PSD_in_total)
    
    # Variance calculation
    var_temp_in = integrate.simpson(PSD_in_temp, omega)
    var_atmos_in = integrate.simpson(PSD_in_atmos, omega)
    var_vibra_in = integrate.simpson(PSD_in_vibra, omega)
    var_alias_in = integrate.simpson(PSD_in_alias, omega)
    var_meas_in = integrate.simpson(PSD_in_meas, omega)
    var_total_in = integrate.simpson(PSD_in_total, omega)
    
    return {
        'RMS_temp_in': np.sqrt(var_temp_in),
        'RMS_atmos_in': np.sqrt(var_atmos_in),
        'RMS_vibra_in': np.sqrt(var_vibra_in),
        'RMS_alias_in': np.sqrt(var_alias_in),
        'RMS_meas_in': np.sqrt(var_meas_in),
        'RMS_total_in': np.sqrt(var_total_in)
    }

def output_RMS_singlemode(evaluate_output, omega, mode_index=0):
    PSD_out_temp = evaluate_output.psd_output["temporal"][mode_index, :]
    PSD_out_atmos = evaluate_output.psd_output["atmosphere"][mode_index, :]
    PSD_out_vibra = evaluate_output.psd_output["vibration"][mode_index, :]
    PSD_out_alias = evaluate_output.psd_output["aliasing"][mode_index, :]
    PSD_out_meas  = evaluate_output.psd_output["measurement"][mode_index, :]
    PSD_out_total = evaluate_output.psd_output["total"][mode_index, :]
    
    PSD_out_temp = np.real(PSD_out_temp)
    PSD_out_atmos = np.real(PSD_out_atmos)
    PSD_out_vibra = np.real(PSD_out_vibra)
    PSD_out_alias = np.real(PSD_out_alias)
    PSD_out_meas = np.real(PSD_out_meas)
    PSD_out_total = np.real(PSD_out_total)
    
    # Variance calculation
    var_temp_out = integrate.simpson(PSD_out_temp, omega)
    var_atmos_out = integrate.simpson(PSD_out_atmos, omega)
    var_vibra_out = integrate.simpson(PSD_out_vibra, omega)
    var_alias_out = integrate.simpson(PSD_out_alias, omega)
    var_meas_out = integrate.simpson(PSD_out_meas, omega)
    var_total_out = integrate.simpson(PSD_out_total, omega)
    
    return {
        'RMS_temp_out': np.sqrt(var_temp_out),
        'RMS_atmos_out': np.sqrt(var_atmos_out),
        'RMS_vibra_out': np.sqrt(var_vibra_out),
        'RMS_alias_out': np.sqrt(var_alias_out),
        'RMS_meas_out': np.sqrt(var_meas_out),
        'RMS_total_out': np.sqrt(var_total_out)
    }
    
def print_psd_RMS_terms(evaluate, omega, mode_index=0, mode="in"):
    if mode == "in":
        rms_in = input_RMS_singlemode(evaluate, omega, mode_index)
        print(f" - Atmospheric RMS: {format_3digits(rms_in['RMS_atmos_in'])}")
        print(f" - Vibration RMS: {format_3digits(rms_in['RMS_vibra_in'])}")
        print(f" - Aliasing RMS: {format_3digits(rms_in['RMS_alias_in'])}")
        print(f" - Measurement RMS: {format_3digits(rms_in['RMS_meas_in'])}")
        print(f" - Total RMS: {format_3digits(rms_in['RMS_total_in'])}")
        print(f" - Temporal RMS: {format_3digits(rms_in['RMS_temp_in'])}")
        print("")
        
        return {
            "RMS_atmos_in": rms_in['RMS_atmos_in'],
            "RMS_vibra_in": rms_in['RMS_vibra_in'],
            "RMS_alias_in": rms_in['RMS_alias_in'],
            "RMS_meas_in": rms_in['RMS_meas_in'],
            "RMS_total_in": rms_in['RMS_total_in'],
            "RMS_temp_in": rms_in['RMS_temp_in']
        }
    elif mode == "out":
        rms_out = output_RMS_singlemode(evaluate, omega, mode_index)
        print(f" - Atmospheric RMS: {format_3digits(rms_out['RMS_atmos_out'])}")
        print(f" - Vibration RMS: {format_3digits(rms_out['RMS_vibra_out'])}")
        print(f" - Aliasing RMS: {format_3digits(rms_out['RMS_alias_out'])}")
        print(f" - Measurement RMS: {format_3digits(rms_out['RMS_meas_out'])}")
        print(f" - Total RMS: {format_3digits(rms_out['RMS_total_out'])}")
        print(f" - Temporal RMS: {format_3digits(rms_out['RMS_temp_out'])}")
        print("")
        
        return {
            "RMS_atmos_out": rms_out['RMS_atmos_out'],
            "RMS_vibra_out": rms_out['RMS_vibra_out'],
            "RMS_alias_out": rms_out['RMS_alias_out'],
            "RMS_meas_out": rms_out['RMS_meas_out'],
            "RMS_total_out": rms_out['RMS_total_out'],
            "RMS_temp_out": rms_out['RMS_temp_out']
        }
        
    else:
        raise ValueError("Invalid mode. Use 'in' for input variance or 'out' for output variance.")
   
def format_3digits(x):
    if x >= 0.001:
        return f"{x:.3f}"
    else:
        return f"{x:.3e}"
    
def format_6digits(x):
    if x >= 1E-6:
        return f"{x:.6f}"
    else:
        return f"{x:.3e}"