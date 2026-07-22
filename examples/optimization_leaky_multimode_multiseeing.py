#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

== Start Script for Semi-analytic Error Budget Optimization ==

(leaky integral controller as an example)       

NOTE: 1. ONLY FOR SINGLE MODE
      2. The controller type and parameters will be determined by the YAML file automatically:
         - controller type==1: integral controller, only optimize the gain value
         - controller type==2: polynomial controller, optimize the coefficients of the numerator and denominator polynomials
         - controller type==3: leaky integral controller, optimize the gain and forgetting factor
     
=========== content ===========

    1. Load parameters and initialize the parameters
    2. Initialize the optimization context for single mode
    3. Optimization
    4. Plotting
        4.1 bode figure for the transfer functions
        4.2 Compare PSDs and plot
        4.3 Nyquist plot for the open-loop transfer function
        
===============================
        
Created on 2026-05-03 14:59

@author: Jueqi Lin
"""

from datetime import datetime
import os
import numpy as np
from scipy.optimize import dual_annealing
import matplotlib.pyplot as plt
from src.initialization_utils import (
    init_parameters,
    init_optimization_context, 
    print_psd_RMS_terms)
from src.control_plot import (
    bodeplot_Hz,
    set_psd_plot_title_text,
    plot_psds_single_mode)
from src.table_utilis import rms_data_singlemode, save_rms_table_vertical
from src.control_utils import cost

# DEFAULT_SRC_PATH = os.path.dirname(__file__)
DEFAULT_FIG_PATH =  'ANDES/sensor_fusion/EBO'
# DEFAULT_FITTING_COEFF = 0.28
DEFAULT_TABLE_PATH = 'ANDES/sensor_fusion/EBO'

def optimization_leaky_multimode_multiseeing(
    param_dir='params_mod4_leaky_seeing_0.4.yaml',
    mode_index_set=[0],
    seeing_set=None,
    save_fig=False,
    save_table=False,
    show_fig=False):
    
    # 1. Load parameters and initialize the parameters
    init_params=init_parameters(param_dir)
    all_rms_data = []

    if seeing_set is None:
        seeing_set = [init_params['seeing']]
    
    for seeing in seeing_set:
        init_params['seeing'] = seeing
        
        for mode_index in mode_index_set:
            # 2. Initialize the optimization context for single mode
            # obj_to_optimize is an instance of SingleModeControllerOptimizationContext
            obj_to_optimize = init_optimization_context(
                                init_params, 
                                mode_index=mode_index)  
            
            # 3. Optimization
            # evaluate function just for one mode!!!  
            n_iter_optimization = 200
            seed_optimization = 50
            
            # weight for different terms in cost functions: 
            # cost = cost_variance_without_fitting * weight[0]
            #      + penalty for stability * weight[1]
            #      + stability margin * weight[2]
            #      + penalty for H_n's peak * weight[3]
            #      + penalty for H_r's peak * weight[4]
            #      + gain margin * weight[5]
            weight_cost = np.array([10, 1, 1e2, 1e2, 1e4, 1e2], dtype=float)
            
            # build up the cost function to be optimized with dual_annealing method
            # for leaky integral controller
            res_cost_initial =  cost(obj_to_optimize, 
                                    weight_cost=weight_cost,
                                    gain_leaky=init_params['gain_array'][mode_index], 
                                    ff_leaky=init_params['ff_array'][mode_index]) 
            
            cost_function_value_no_opti = res_cost_initial['cost_function_value']
            evaluate_no_opti = res_cost_initial['evaluate_result']
            
            x0_leaky = np.r_[init_params['gain_array'][mode_index], init_params['ff_array'][mode_index]]
            opti_cost_func = lambda x: cost(obj_to_optimize,
                                            weight_cost=weight_cost,
                                            gain_leaky = x[0], 
                                            ff_leaky = x[1])['cost_function_value']
            opti_bounds = [(init_params['gain_min'], init_params['gain_max']), (init_params['ff_min'], init_params['ff_max'])]
            
            res_opti_dual_annealing = dual_annealing(opti_cost_func, 
                                                    opti_bounds, 
                                                    x0=x0_leaky, 
                                                    maxiter=n_iter_optimization, 
                                                    seed=seed_optimization)
            
            plant_num = init_params['plant_tf'].num[0][0]
            plant_den = init_params['plant_tf'].den[0][0]
            
            print()  
            print("=========before optimization============")
            print()
            
            print("Gain and Forgetting Factor:", x0_leaky[0], x0_leaky[1])
            print("plant num and den:", plant_num, plant_den)
            print("Cost function value:", cost_function_value_no_opti)
            print()
            # print("Evaluation result (variance terms):")
            # print_psd_variance_terms(evaluate_no_opti, init_params['omega'], mode="in")
            print("Evaluation result (RMS):")
            RMS_no_opti = print_psd_RMS_terms(evaluate_no_opti, init_params['omega'], mode="in")
            print("Initial Gain:", x0_leaky[0])
            print("Initial Forgetting Factor:", x0_leaky[1])
            print("Optical gain:", init_params['c_optg'])
            print("opti_bounds:",opti_bounds)
            
            print() 
            print("=========Optimization============")
            print()
            print("Optimal leaky integral controller Gain & Forgetting Factor found:", res_opti_dual_annealing.x)
            
            gain_leaky_optimized = res_opti_dual_annealing.x[0]
            ff_leaky_optimized = res_opti_dual_annealing.x[1]
                                        
            res_cost_optimized = cost(obj_to_optimize, 
                                    weight_cost=weight_cost,
                                    gain_leaky=gain_leaky_optimized, 
                                    ff_leaky=ff_leaky_optimized)

            title_text = set_psd_plot_title_text(init_params['controller_type'], 
                                                mode_index=mode_index,
                                                gain_leaky=gain_leaky_optimized,
                                                ff_leaky=ff_leaky_optimized
                                                )    
        
            cost_function_value_optimized = res_cost_optimized['cost_function_value']
            evaluate_optimized = res_cost_optimized['evaluate_result']
            stability_penalty, sm_penalty, H_n_tf_peak_penalty, H_r_tf_peak_penalty, gm_penalty = res_cost_optimized['penalty']
            H_n_tf_optimized = res_cost_optimized['H_n_tf']
            H_r_tf_optimized = res_cost_optimized['H_r_tf']
            # H_ol_tf_optimized = res_cost_optimized['H_ol_tf']
            H_ol_margins_optimized = res_cost_optimized['H_ol_margins']

            # H_n_bandwidth_optimized_Hz = res_cost_optimized['bandwidth_H_n'] * 180 / np.pi
            
            print()  
            print("============ After Optimization ===========") 
            print()  
            print("Cost function value:", cost_function_value_optimized )
            # print("Evaluation result:")
            # print_psd_variance_terms(evaluate_optimized, init_params['omega'], mode="out")
            print("Evaluation result (RMS):")
            RMS_optimized = print_psd_RMS_terms(evaluate_optimized, init_params['omega'], mode="out")

            print("Optimized stability penalty:", stability_penalty)
            print("Optimized stability margin penalty:", sm_penalty)
            print("Optimized gain margin penalty:", gm_penalty)
            print('H_n peak penalty:', H_n_tf_peak_penalty)
            print('H_r peak penalty:', H_r_tf_peak_penalty)
            
            print("Optimized controller num and den:", 
                evaluate_optimized.controller_num, 
                evaluate_optimized.controller_den)
            print()
            # print('H_r_tf:',H_r_tf_optimized)
            # print('H_n_tf:',H_n_tf_optimized)
            
            # 4. Plotting
            
            # 4.1 bode figure for the transfer functions
            
            gm = H_ol_margins_optimized[0]
            pm = H_ol_margins_optimized[1]
            
            frame_rate = init_params['frame_rate']
            
            fig1, ax3 = bodeplot_Hz(
            transfer_functions_ct=H_n_tf_optimized,
            omega_limits=[1e-5, frame_rate/2],
            omega_num=1000,
            labels="H_n ",
            title=f"Transfer functions H_n & H_r - Mode {mode_index}",
            subtitle=f"[GM: {gm:.2f} dB, PM: {pm:.2f} deg]")
            
            fig_bode, _ = bodeplot_Hz(
            transfer_functions_ct=H_r_tf_optimized,
            omega_limits=[1e-5,frame_rate/2],
            omega_num=1000,
            labels="H_r ",
            styles={'linestyle':'--'},
            fig=fig1,
            ax1=ax3[0],
            ax2=ax3[1])
            
            save_fig_param = {
                'save_fig': save_fig,
                'fig_name': f"TF_ModRad_{init_params['modulation_radius']}_seeing_{init_params['seeing']}_mag_{init_params['magnitude']}_mode_{mode_index}_leaky.png",
                'fig_folder': os.path.join(DEFAULT_FIG_PATH, 'figures'),
                'dpi': 300
            }
            
            if save_fig_param is not None:
                save_fig = save_fig_param.get('save_fig', False)
                fig_name = save_fig_param.get('fig_name', 'tf_bode.png')
                fig_folder = save_fig_param.get('fig_folder', '.')
                dpi = save_fig_param.get('dpi', 300)
                
                if save_fig:
                    full_path = os.path.join(fig_folder, fig_name)
                    fig_bode.savefig(full_path, dpi=dpi)
                    print(f"Figure saved to: {full_path}")
            
            
            # 4.2 Compare PSDs and plot
        
            save_fig_param = {
                'save_fig': save_fig,
                'fig_name': f"PSD_ModRad_{init_params['modulation_radius']}_seeing_{init_params['seeing']}_mag_{init_params['magnitude']}_mode_{mode_index}_leaky.png",
                'fig_folder': os.path.join(DEFAULT_FIG_PATH, 'figures'),
                'dpi': 300
            }
            plot_psds_single_mode(mode_index,
                    init_params['temporal_freqs'],
                    evaluate_optimized,
                    plot_inputs=True,
                    title_text=title_text,
                    save_fig_param=save_fig_param)
            
            # 4.3 Nyquist plot for the open-loop transfer function
            
            # freqs_nyquist = np.logspace(0, np.log10(frame_rate/ 2.0), 2000)    
            # nyquist_count = plot_nyquist(H_ol_tf_optimized,
            #                              freqs_Hz=freqs_nyquist)    
            
            if save_table is True:
                # data_to_table_singlemode = {**data_to_table_singlemode, **RMS_no_opti, **RMS_optimized}
                data_to_table_singlemode = rms_data_singlemode(
                                                mode_index, 
                                                RMS_no_opti, 
                                                RMS_optimized, 
                                                res_opti_dual_annealing.x[0], 
                                                res_opti_dual_annealing.x[1],
                                                gm=gm,
                                                pm=pm)
                all_rms_data.append(data_to_table_singlemode)
                
            print("Optimization for mode index", mode_index, "completed.")
        
        if save_table is True: 
            timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
            table_name =f"Result_ModRad_{init_params['modulation_radius']}_seeing_{init_params['seeing']}_mag_{init_params['magnitude']}_leaky_{timestamp}.xlsx"
            table_name_full_path = os.path.join(DEFAULT_TABLE_PATH, 'tables', table_name)
            save_table_param = {
                            'table_name': table_name_full_path,
                            'table_folder': os.path.join(DEFAULT_TABLE_PATH, 'tables'),
                            'row_parameters': ['Gain (leaky)', 'FF (leaky)', 'GM', 'PM', 'Atmos (input)', 'Atmos (output)', 'Vibra (input)', 'Vibra (output)', 
                                            'Alias (input)', 'Alias (output)', 'Meas (input)', 'Meas (output)', 
                                            'Total (input)', 'Total (output)', 'Temporal (input)', 'Temporal (output)']
                                }     
            save_rms_table_vertical(save_table_param, all_rms_data)
            
        if show_fig:
            plt.show()
    
    return obj_to_optimize

if __name__ == "__main__":
    obj_to_optimize = optimization_leaky_multimode_multiseeing(
        param_dir = 'config_yaml/Multi_mag_ModRad3/params_4000modes_ModRad_3_mag_16.yaml',
        mode_index_set=[0, 2, 100, 200, 400, 700, 1000, 2000, 3000, 3999], 
        seeing_set=[0.4, 0.6, 0.8, 1.0, 1.2, 1.4],
        save_fig=True,
        save_table=True,
        show_fig=False)
    