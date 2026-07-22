#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2026-04-24 18:12

@author: Jueqi Lin

======= control plot =======

"""

import os

import numpy as np
import control as ct
import matplotlib.pyplot as plt
from src.Functions import build_transfer_function


def bodeplot_Hz(
    transfer_functions_ct=None,
    omega_limits=None,
    omega_num=1000,
    labels=None,
    styles=None,
    title=None,
    subtitle=None,
    fig=None,
    ax1=None,
    ax2=None):
    
    # after using this function, "plt.show()" must be used to display the figure
    
    if transfer_functions_ct is None:
        raise ValueError("Transfer function(s) must not be empty")
    if omega_limits is None:
        raise ValueError("omega_limits must not be empty")
    
    # Determine if it's a single transfer function or collection
    try:
        n_systems = len(transfer_functions_ct)
        first_elem = transfer_functions_ct[0]
        if hasattr(first_elem, 'num') or hasattr(first_elem, 'poles'):
            pass
        else:
            n_systems = 1
            transfer_functions_ct = [transfer_functions_ct]
    except (TypeError, AttributeError):
        n_systems = 1
        transfer_functions_ct = [transfer_functions_ct]
    
    # Handle omega_limits
    omega_limits = np.asarray(omega_limits, dtype=float)
    if omega_limits.ndim == 1:
        if len(omega_limits) != 2:
            raise ValueError("omega_limits must be a 2-element array or Nx2 array for N systems")
        omega_limits_list = [omega_limits] * n_systems
    elif omega_limits.ndim == 2:
        if omega_limits.shape[0] != n_systems:
            raise ValueError(f"omega_limits has {omega_limits.shape[0]} rows but there are {n_systems} transfer functions")
        if omega_limits.shape[1] != 2:
            raise ValueError("omega_limits must have 2 columns (min and max frequency)")
        omega_limits_list = [omega_limits[i, :] for i in range(n_systems)]
    else:
        raise ValueError("omega_limits must be a 1D or 2D array")
    
    # Handle labels
    if labels is None:
        if n_systems == 1:
            labels = [None]
        else:
            labels = [f'System {i+1}' for i in range(n_systems)]
    elif isinstance(labels, str):
        if n_systems == 1:
            labels = [labels]
        else:
            labels = [f'{labels} [{i+1}]' for i in range(n_systems)]
    elif isinstance(labels, list):
        if len(labels) == 1 and n_systems > 1:
            base_label = labels[0] if labels[0] is not None else 'System'
            labels = [f'{base_label} [{i+1}]' for i in range(n_systems)]
        elif len(labels) != n_systems:
            raise ValueError(f"Number of labels ({len(labels)}) must match number of transfer functions ({n_systems})")
    else:
        if n_systems == 1:
            labels = [str(labels)]
        else:
            labels = [f'{labels} [{i+1}]' for i in range(n_systems)]
    
    # Handle styles
    if styles is None:
        # default style table
        default_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        styles = [{'color': default_colors[i % len(default_colors)], 
                   'linestyle': '-', 
                   'linewidth': 2} for i in range(n_systems)]
    elif isinstance(styles, dict):
        styles = [styles] * n_systems
    elif isinstance(styles, list):
        if len(styles) == 1 and n_systems > 1:
            styles = styles * n_systems
        elif len(styles) != n_systems:
            raise ValueError(f"Number of styles ({len(styles)}) must match number of transfer functions ({n_systems})")
    else:
        raise ValueError("styles must be a dict, list of dicts, or None")
    
    # Create figure and axes if not provided
    if fig is None or ax1 is None or ax2 is None:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 5))
    
    # Plot each transfer function
    for i in range(n_systems):
        tf_ct = transfer_functions_ct[i]
        label = labels[i]
        omega_lim = omega_limits_list[i]
        style = styles[i]
        
        omega_limits_rad = omega_lim * 2 * np.pi
        
        mag, phase, omega = ct.frequency_response(
            tf_ct, 
            omega_limits=omega_limits_rad,
            omega_num=omega_num)    
        
        phase_unwrap = np.unwrap(phase) 
        phase_deg = np.rad2deg(phase_unwrap)  
        
        omega_Hz = omega / (2*np.pi)
        
        ax1.semilogx(omega_Hz, 20 * np.log10(mag), 
                     color=style.get('color'),
                     linestyle=style.get('linestyle', '-'),
                     linewidth=style.get('linewidth', 2),
                     marker=style.get('marker'),
                     markersize=style.get('markersize', 4),
                     markevery=style.get('markevery', None),
                     alpha=style.get('alpha', 1.0),
                     label=label)
        
        ax2.semilogx(omega_Hz, phase_deg, 
                     color=style.get('color'),
                     linestyle=style.get('linestyle', '-'),
                     linewidth=style.get('linewidth', 2),
                     marker=style.get('marker'),
                     markersize=style.get('markersize', 4),
                     markevery=style.get('markevery', None),
                     alpha=style.get('alpha', 1.0),
                     label=label)
    
    # Configure axes
    ax1.set_ylabel('Magnitude [dB]')
    if title or subtitle:
        if title and subtitle:
            ax1.set_title(f"{title}\n{subtitle}")
        elif title:
            ax1.set_title(title)
        else:  # only subtitle
            ax1.set_title(subtitle)
    if any(label is not None for label in labels) or n_systems > 1:
        ax1.legend()
    ax1.grid(True, linewidth=0.5, alpha=0.6)
    
    ax2.set_xlabel('Frequency [Hz]')
    ax2.set_ylabel('Phase [deg]')
    ax2.grid(True, linewidth=0.5, alpha=0.6)
    
    fig.set_tight_layout(True)
    
    return fig, (ax1, ax2)
        
def set_psd_plot_title_text(controller_type, mode_index, **title_text_params):
    
    """_summary_

    'gain': gain_optimized if controller_type == 1 else None,
    'ctrl_num': ctrl_num_optimized if controller_type == 2 else None,
    'ctrl_den': ctrl_den_optimized if controller_type == 2 else None
    'gain_leaky': gain_leaky_optimized if controller_type == 3 else None,
    'ff_leaky': ff_leaky_optimized if controller_type == 3 else None

    Returns:
        title_text: psd plot title text based on the controller type and parameters
    """
    base_text = f"Spectral Analysis (PSD) in Closed Loop - Zernike Mode {mode_index}"
    
    if controller_type == 1:
        gain = title_text_params.get('gain')
        if gain is None:
            raise ValueError("gain is required for integral controller")
        title_text = (f"{base_text}\nLoop Gain = {gain:.2f}")
        
    elif controller_type == 2:
        ctrl_num = title_text_params.get('ctrl_num')
        ctrl_den = title_text_params.get('ctrl_den')
        if ctrl_num is None or ctrl_den is None:
            raise ValueError("ctrl_num and ctrl_den are required for standard controller mode")
        num_text = "[" + ",".join(f"{x:.2f}" for x in ctrl_num) + "]"
        den_text = "[" + ",".join(f"{x:.2f}" for x in ctrl_den) + "]"
        title_text = (f"{base_text}\nController: num = {num_text}, den = {den_text}")
        
    elif controller_type == 3:
        gain_leaky = title_text_params.get('gain_leaky')
        ff_leaky = title_text_params.get('ff_leaky')
        if gain_leaky is None or ff_leaky is None:
            raise ValueError("gain_leaky and ff_leaky are required for leaky integral controller mode")
        title_text = (f"{base_text}\nLeaky Integral Controller: Gain = {gain_leaky:.2f}, Forgetting Factor = {ff_leaky:.2f}")
    else:
        title_text = base_text
        
    return title_text

def plot_psds_single_mode(mode_index,
              freqs,
              evaluate_optimized=None,
              plot_inputs=True,
              title_text=None,
              save_fig_param=None):
    
    if evaluate_optimized is None:
        raise ValueError("evaluate_optimized must be provided to access PSD data for plotting")
    
    PSD_in_temp = evaluate_optimized.psd_input["temporal"][0, :]
    PSD_in_alias = evaluate_optimized.psd_input["aliasing"][0, :]
    PSD_in_meas  = evaluate_optimized.psd_input["measurement"][0, :]
    PSD_in_total = evaluate_optimized.psd_input["total"][0, :]

    PSD_out_temp = evaluate_optimized.psd_output["temporal"][0, :]
    PSD_out_alias = evaluate_optimized.psd_output["aliasing"][0, :]
    PSD_out_meas  = evaluate_optimized.psd_output["measurement"][0, :]
    PSD_out_total = evaluate_optimized.psd_output["total"][0, :]
    
    
    plt.figure(figsize=(12, 7))

    if plot_inputs:
        plt.loglog(freqs, PSD_in_temp, label='Temporal = atmosphere + vibration (Open Loop)',
                   color='tab:blue', linestyle='--', alpha=0.6)
        plt.loglog(freqs, PSD_in_alias, label='Aliasing (Open Loop)',
                   color='tab:orange', linestyle='--', alpha=0.6)
        plt.loglog(freqs, PSD_in_meas, label='Noise (Open Loop)',
                   color='tab:green', linestyle='--', alpha=0.6)
        plt.loglog(freqs, PSD_in_total, label='Total (Open Loop)',
                   color='black', linestyle='--', alpha=0.6)
        
    plt.loglog(freqs, PSD_out_temp, label='Temporal Residual (Servo-lag)',
               color='tab:blue', linewidth=2.5)
    plt.loglog(freqs, PSD_out_alias, label='Aliasing Residual',
               color='tab:orange', linewidth=2.5)
    plt.loglog(freqs, PSD_out_meas, label='Noise Residual',
               color='tab:green', linewidth=2.5)

    plt.loglog(freqs, PSD_out_total, label='Total PSD (Sum)',
               color='black', linewidth=3, linestyle='-.')

    # title text was defined above if-else based on the controller type and parameters
    if title_text is None:
        title_text = f"Spectral Analysis (PSD) in Closed Loop - Zernike Mode {mode_index}"
    plt.title(title_text, fontsize=14)
    plt.xlabel("Temporal Frequency [Hz]", fontsize=12)
    plt.ylabel("Power Spectral Density [nm² / Hz]", fontsize=12)
    plt.grid(True, which="both", linestyle=":", alpha=0.7)
    plt.legend(loc='lower left', fontsize=11)

    plt.tight_layout()
    
    if save_fig_param is not None:
        save_fig = save_fig_param.get('save_fig', False)
        fig_name = save_fig_param.get('fig_name', 'psd_plot.png')
        fig_folder = save_fig_param.get('fig_folder', '.')
        dpi = save_fig_param.get('dpi', 300)
        
        if save_fig:
            full_path = os.path.join(fig_folder, fig_name)
            plt.savefig(full_path, dpi=dpi)
            print(f"Figure saved to: {full_path}")

    return()

def plot_nyquist(G_ol_tf,
                 freqs_Hz=None):
    
    if freqs_Hz is None:
        freqs = np.logspace(0, 2, 1000)* 2* np.pi
    else:
        freqs = freqs_Hz * 2 * np.pi
    
    plt.figure(figsize=(4, 4))
    
    tf_nyquist_response = ct.nyquist_response(G_ol_tf)
    cplt = ct.nyquist_plot(tf_nyquist_response, freqs, title='', linewidth=2, max_curve_magnitude=25)
    plt.title('Nyquist plot', fontsize=10)
    
    return(tf_nyquist_response.count)

