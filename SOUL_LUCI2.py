#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 14:28:10 2026

@author: greta
"""

from scipy.io import readsav
from astropy.table import Table
import pandas as pd
import numpy as np
import yaml

from src.Functions import load_parameters
from scripts.main_saeb import run

# Show all DataFrame columns when printing
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)


analysis_mode = "single"

# =============================================================================
# FUNCTIONS
# =============================================================================

# Converts a NumPy array into the machine's native endian format. This is needed 
# because some data read from IDL .sav files can be big-endian, while Pandas on 
# little-endian machines may raise errors during operations such as groupby.
    
def to_native_endian(array):

    array = np.asarray(array)

    if array.dtype.byteorder not in ("=", "|"):
        return array.astype(array.dtype.newbyteorder("="))

    return array


# Converts a byte array into standard Python strings.

def decode_bytes_array(array):


    result = []

    for x in array:

        if isinstance(x, bytes):
            x = x.decode("utf-8")

        x = x.strip()

        result.append(x)

    return result


#  Loads the base YAML file, updates the parameters, runs SAEB, returns 
#  the output dictionary.

def run_saeb(seeing, magnitude, binning, ao_framerate, output_info):
   
    param = load_parameters("params_Total_variance_SOUL_LUCI2.yaml")
    param["atmosphere"]["seeing"] = float(seeing)
    param["guide_star"]["magn"] = float(magnitude)
    param["control"]["bin"] = int(binning)
    sampling_time = 1.0 / float(ao_framerate)
    param["control"]["sampling_time"] = sampling_time

    # delay
    if sampling_time >= 0.00167:
        param["plant"]["total_delay"] = 1
    else:
        param["plant"]["total_delay"] = 2

    # save modified YAML
    with open("params_Total_variance_SOUL_LUCI2_modified.yaml", "w") as file:
        yaml.dump(param, file, sort_keys=False)

    # run SAEB
    result = run("params_Total_variance_SOUL_LUCI2_modified.yaml")

    # create output dictionary
    output = dict(output_info)
    output.update(result)

    return output


# =============================================================================
# DATA LOADING AND DATAFRAME CREATION
# =============================================================================

data = readsav( "src/file_fits/LBT/Table_CAT_LUCI_DX_SKY.sav", python_dict=True)

# keys check
print("data keys:",data.keys())

table = data["table"]

df = pd.DataFrame({
    "TN": decode_bytes_array(table["TN"][0]),
    "WFS_MAG": to_native_endian(table["WFS_MAG"][0]),
    "STAR_ID": decode_bytes_array(table["STAR_ID"][0]),
    "AO_FRAMERATE": to_native_endian(table["AO_FRAMERATE"][0]),
    "BINNING": to_native_endian(table["BINNING"][0]),
    "DIMM": to_native_endian(table["DIMM"][0]),
})


print("\nFirst rows of the DataFrame:")
print(df.head())

print("\nDimensions of the DataFrame:")
print(df.shape)

# print(df.to_string())


# =============================================================================
# SINGLE TN OR GROUPED SAEB ANALYSIS
# =============================================================================

if analysis_mode == "single":

    # TN ultrafaint
    tn_target = "20230427_111423"
    
    mask = df["TN"] == tn_target
    df_single = df[mask]
    
    print(df_single)
    print(df_single.shape)
    
    row = df_single.iloc[0]
    
    output_info = {
        "TN": row["TN"],
        "WFS_MAG": row["WFS_MAG"],
        "STAR_ID": row["STAR_ID"],
        "AO_FRAMERATE": row["AO_FRAMERATE"],
        "BINNING": row["BINNING"],
        "DIMM": row["DIMM"],
    }
    
    output = run_saeb(seeing=row["DIMM"], magnitude=row["WFS_MAG"], binning=row["BINNING"],
                      ao_framerate=row["AO_FRAMERATE"], output_info=output_info)
    
    print("\nOUTPUT:", output)
    
    table = Table([output])

    # Save FITS
    table.write("src/file_fits/LBT/SAEB_results_single_TN.fits", overwrite=True)
    
    # Save CSV
    df_output = table.to_pandas()
    df_output.to_csv("src/file_fits/LBT/SAEB_results_single_TN.csv", index=False)
    
    print("\nSingle TN results:")
    print(df_output)
    
    print("\nFiles saved:")
    print("- SAEB_results_single_TN.fits")
    print("- SAEB_results_single_TN.csv")
    

elif analysis_mode == "all":

    grouped = df.groupby(["STAR_ID", "BINNING", "AO_FRAMERATE"])
    
    groups_list = list(grouped)
    
    print("\nNumber of groups:")
    print(len(groups_list))
    
    
    # Print first groups for checking
    first_groups = groups_list[:5]
    
    for i, group in enumerate(first_groups):
    
        print("\n" + "=" * 80)
    
        group_name = group[0]
        group_df = group[1]
    
        print("Group number:", i)
        print("Group name:", group_name)
    
        star_id = group_name[0]
        binning = group_name[1]
        ao_framerate = group_name[2]
    
        print("STAR_ID:", star_id)
        print("BINNING:", binning)
        print("AO_FRAMERATE:", ao_framerate)
    
        print("\nGroup data:")
        print(group_df)
    
    
    summary = []
    
    for group in groups_list:
    
        print("\n" + "=" * 80)
    
        group_name = group[0]
        group_df = group[1]
    
        star_id = group_name[0]
        binning = group_name[1]
        ao_framerate = group_name[2]
    
        dimm_median = group_df["DIMM"].median()
        wfs_mag_median = group_df["WFS_MAG"].median()
    
        print("STAR_ID:", star_id)
        print("BINNING:", binning)
        print("AO_FRAMERATE:", ao_framerate)
        print("DIMM median:", dimm_median)
        print("WFS_MAG median:", wfs_mag_median)
    
        output_info = {
            "STAR_ID": star_id,
            "BINNING": binning,
            "AO_FRAMERATE": ao_framerate,
            "DIMM_MEDIAN": dimm_median,
            "WFS_MAG_MEDIAN": wfs_mag_median,
        }
    
        output = run_saeb(seeing=dimm_median, magnitude=wfs_mag_median, binning=binning, 
                          ao_framerate=ao_framerate, output_info=output_info)
    
        summary.append(output)
    
        print("\nOUTPUT:")
        print(output)
    
    
    table = Table(rows=summary)

    # Save FITS
    table.write("src/file_fits/LBT/SAEB_results_all_groups.fits", overwrite=True)
    
    # Save CSV
    df_summary = table.to_pandas()
    df_summary.to_csv("src/file_fits/LBT/SAEB_results_all_groups.csv", index=False)
    
    print("\nSummary table:")
    print(df_summary)
    
    print("\nFiles saved:")
    print("- SAEB_results_all_groups.fits")
    print("- SAEB_results_all_groups.csv")
    print("Number of groups:", len(summary))
        
        



    
    
    
    
    
    
    
    