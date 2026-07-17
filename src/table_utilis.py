#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2026-05-26 16:33

@author: Jueqi Lin
"""

import os
import pandas as pd

from src.initialization_utils import format_3digits

def rms_data_singlemode(mode_index, RMS_no_opti, RMS_optimized, gain_leaky, ff_leaky, gm=None, pm=None):
    
    return {
        "Mode Index": mode_index,
        "Atmos (input)": format_3digits(RMS_no_opti['RMS_atmos_in']),
        "Atmos (output)": format_3digits(RMS_optimized['RMS_atmos_out']),
        "Vibra (input)": format_3digits(RMS_no_opti['RMS_vibra_in']),
        "Vibra (output)": format_3digits(RMS_optimized['RMS_vibra_out']),
        "Alias (input)": format_3digits(RMS_no_opti['RMS_alias_in']),
        "Alias (output)": format_3digits(RMS_optimized['RMS_alias_out']),
        "Meas (input)": format_3digits(RMS_no_opti['RMS_meas_in']),
        "Meas (output)": format_3digits(RMS_optimized['RMS_meas_out']),
        "Total (input)": format_3digits(RMS_no_opti['RMS_total_in']),
        "Total (output)": format_3digits(RMS_optimized['RMS_total_out']),
        "Temporal (input)": format_3digits(RMS_no_opti['RMS_temp_in']),
        "Temporal (output)": format_3digits(RMS_optimized['RMS_temp_out']),
        "Gain (leaky)": format_3digits(gain_leaky),
        "FF (leaky)": format_3digits(ff_leaky),
        "GM": format_3digits(gm),
        "PM": format_3digits(pm)
    }
    
def save_rms_table_vertical(save_table_param, all_rms_data):
    
    row_parameters = save_table_param['row_parameters']
    table_content = {'Parameter': row_parameters}
    for idx, data_row in enumerate(all_rms_data):
        mode_idx = data_row['Mode Index']
        table_content[f'Mode_{mode_idx}'] = []
        for param in row_parameters:
            table_content[f'Mode_{mode_idx}'].append(data_row[param])
        
    df = pd.DataFrame(table_content)
    df.to_excel(save_table_param['table_name'], index=False)
