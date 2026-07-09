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


def to_native_endian(array):
    """
    Converts a NumPy array into the machine's native endian format.
    This is needed because some data read from IDL .sav files can be
    big-endian, while Pandas on little-endian machines may raise errors
    during operations such as groupby.
    """
    array = np.asarray(array)

    if array.dtype.byteorder not in ("=", "|"):
        return array.astype(array.dtype.newbyteorder("="))

    return array


def decode_bytes_array(array):
    """
    Converts a byte array into standard Python strings.
    """

    result = []

    for x in array:

        if isinstance(x, bytes):
            x = x.decode("utf-8")

        x = x.strip()

        result.append(x)

    return result

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
# SINGLE TN
# =============================================================================

# # TN ultrafaint 

# tn_target = "20230427_111423"

# mask = df["TN"] == tn_target
# df_single = df[mask]

# print(df_single)
# print(df_single.shape)

# row = df_single.iloc[0]


# param = load_parameters('params_Total_variance_SOUL+LUCI2.yaml')

# # seeing
# param["atmosphere"]["seeing"] = float(row["DIMM"])

# # magnitude 
# param["guide_star"]["magn"] = float(row["WFS_MAG"])

# # binning
# param["control"]["bin"] = int(row["BINNING"])

# # sampling time
# param["control"]["sampling_time"] = 1.0 / float(row["AO_FRAMERATE"])

# if param["control"]["sampling_time"] >= 0.00167:
#     param["plant"]["total_delay"] = 1
# else:
#     param["plant"]["total_delay"] = 2

# #print(type(param["control"]["bin"]))

# with open("params_Total_variance_SOUL+LUCI2_modified_single_TN.yaml", "w") as file:
#     yaml.dump(param, file, sort_keys=False)

# print("Modified YAML file saved.")

# result = run("params_Total_variance_SOUL+LUCI2_modified_single_TN.yaml")

# # print(result)

# output = {
#     "TN": row["TN"],
#     "WFS_MAG": row["WFS_MAG"],
#     "STAR_ID": row["STAR_ID"],
#     "AO_FRAMERATE": row["AO_FRAMERATE"],
#     "BINNING": row["BINNING"],
#     "DIMM": row["DIMM"],
# }

# output.update(result)

# print ("\nOUTPUT:",output)

# table = Table([output])

# table.write("src/file_fits/LBT/SAEB_results_single_TN.fits", overwrite=True)

# #print ("TABLE:", table)


# =============================================================================
# GENERAL CASE - ALL TN
# =============================================================================

grouped = df.groupby(["STAR_ID", "BINNING", "AO_FRAMERATE"])

groups_list = list(grouped)

print("\nNumber of groups:")
print(len(groups_list))

 
first_groups = groups_list[:5]

i = 0

for group in first_groups:

    print("\n" + "=" * 80)

    group_name = group[0]
    group_df = group[1]

    print("Gruppo numero:", i) 

    print("Nome gruppo:", group_name)

    star_id = group_name[0]
    binning = group_name[1]
    ao_framerate = group_name[2]

    print("STAR_ID:", star_id)
    print("BINNING:", binning)
    print("AO_FRAMERATE:", ao_framerate)

    print("\nDati del gruppo:")
    print(group_df)

    i = i + 1

    
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
    print("Mediana DIMM:", dimm_median)
    print("Mediana WFS_MAG:", wfs_mag_median)
    
    param = load_parameters("params_Total_variance_SOUL+LUCI2.yaml")

    # seeing
    param["atmosphere"]["seeing"] = float(dimm_median)

    # magnitude
    param["guide_star"]["magn"] = float(wfs_mag_median)

    # binning
    param["control"]["bin"] = int(binning)

    # sampling time
    param["control"]["sampling_time"] = 1.0 / float(ao_framerate)

    # delay
    if param["control"]["sampling_time"] >= 0.00167:
        param["plant"]["total_delay"] = 1
    else:
        param["plant"]["total_delay"] = 2
   
    with open("params_Total_variance_SOUL+LUCI2_modified_all_TN.yaml", "w") as file:
        yaml.dump(param, file, sort_keys=False)

    result = run("params_Total_variance_SOUL+LUCI2_modified_all_TN.yaml")

    output = {
        "STAR_ID": star_id,
        "BINNING": binning,
        "AO_FRAMERATE": ao_framerate,
        "DIMM_MEDIAN": dimm_median,
        "WFS_MAG_MEDIAN": wfs_mag_median,
    }

    output.update(result)

    summary.append(output)

    print("\nOUTPUT:")
    print(output)  


table = Table(rows=summary)
table.write("src/file_fits/LBT/SAEB_results_all_groups.fits", overwrite=True)

print ("TABLE:", table)
print("\nFile FITS saved with", len(summary), "gruppi.")












    
    
    
    
    
    
    
    