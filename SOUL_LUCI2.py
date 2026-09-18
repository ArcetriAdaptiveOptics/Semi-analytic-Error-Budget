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
import matplotlib.pyplot as plt
import yaml
from pathlib import Path

from src.Functions import load_parameters
from scripts.main_saeb import run

# Show all DataFrame columns when printing
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)

analysis_mode = "all"
TT_REFERENCE_WAVELENGTH_NM = 1650.0
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "results" / "LBT"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TMP_DIR = BASE_DIR / "tmp" / "LBT"
TMP_DIR.mkdir(parents=True, exist_ok=True)

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


def tt_residual_nm_from_sr(sr_with_tt, sr_ttfree, wavelength_nm):

    if not np.isfinite(sr_with_tt) or not np.isfinite(sr_ttfree):
        return np.nan

    if sr_with_tt <= 0 or sr_ttfree <= 0:
        return np.nan

    if sr_with_tt > sr_ttfree:
        return np.nan

    ratio = sr_with_tt / sr_ttfree
    sigma_tt_rad_sq = -np.log(ratio)

    if sigma_tt_rad_sq < 0:
        return np.nan

    sigma_tt_rad = np.sqrt(sigma_tt_rad_sq)
    sigma_tt_nm = sigma_tt_rad * wavelength_nm / (2.0 * np.pi)

    return float(sigma_tt_nm)


def tt_comparison_metrics(tt_from_slopes_nm, tt_from_budget_nm):

    if not np.isfinite(tt_from_slopes_nm) or not np.isfinite(tt_from_budget_nm):
        return np.nan, np.nan

    delta_nm = float(tt_from_slopes_nm - tt_from_budget_nm)

    if tt_from_budget_nm > 0:
        ratio = float(tt_from_slopes_nm / tt_from_budget_nm)
    else:
        ratio = np.nan

    return delta_nm, ratio


def plot_sr_and_tt_comparisons(df_summary):

    marker_map = {
        1: "o",
        2: "s",
        3: "^",
        4: "D",
    }

    # SR comparison: measured vs SAEB estimate.
    sr_valid = df_summary[
        (df_summary["SR_LUCI"] != -1)
        & np.isfinite(df_summary["SR_LUCI"])
        & np.isfinite(df_summary["SR_ESTIMATION"])
    ]

    plt.figure(figsize=(8, 6))
    for binning in sorted(sr_valid["BINNING"].dropna().unique()):
        bin_data = sr_valid[sr_valid["BINNING"] == binning]
        if len(bin_data) == 0:
            continue
        plt.scatter(
            bin_data["SR_LUCI"],
            bin_data["SR_ESTIMATION"],
            marker=marker_map.get(int(binning), "x"),
            s=60,
            alpha=0.85,
            label=f"BINNING={int(binning)}",
        )

    if len(sr_valid) > 0:
        x_min = min(sr_valid["SR_LUCI"].min(), sr_valid["SR_ESTIMATION"].min())
        x_max = max(sr_valid["SR_LUCI"].max(), sr_valid["SR_ESTIMATION"].max())
        plt.plot([x_min, x_max], [x_min, x_max], "k--", linewidth=1.2, label="y = x")

    plt.xlabel("SR_LUCI")
    plt.ylabel("SR_ESTIMATION")
    plt.title("SR comparison by binning")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    sr_plot_path = OUTPUT_DIR / "SR_LUCI_vs_SR_ESTIMATION_by_binning.png"
    plt.savefig(sr_plot_path, dpi=200)
    plt.close()

    # TT comparison: residual from slopes vs error budget TT residual.
    tt_valid = df_summary[
        np.isfinite(df_summary["TT_RESIDUAL_NM"])
        & np.isfinite(df_summary["TT_RESIDUAL_EB_NM"])
    ]

    plt.figure(figsize=(8, 6))
    for binning in sorted(tt_valid["BINNING"].dropna().unique()):
        bin_data = tt_valid[tt_valid["BINNING"] == binning]
        if len(bin_data) == 0:
            continue
        plt.scatter(
            bin_data["TT_RESIDUAL_NM"],
            bin_data["TT_RESIDUAL_EB_NM"],
            marker=marker_map.get(int(binning), "x"),
            s=60,
            alpha=0.85,
            label=f"BINNING={int(binning)}",
        )

    if len(tt_valid) > 0:
        x_min = min(tt_valid["TT_RESIDUAL_NM"].min(), tt_valid["TT_RESIDUAL_EB_NM"].min())
        x_max = max(tt_valid["TT_RESIDUAL_NM"].max(), tt_valid["TT_RESIDUAL_EB_NM"].max())
        plt.plot([x_min, x_max], [x_min, x_max], "k--", linewidth=1.2, label="y = x")

    plt.xlabel("TT_RESIDUAL_NM from SR slopes [nm]")
    plt.ylabel("TT_RESIDUAL_EB_NM from error budget [nm]")
    plt.title("TT residual comparison by binning")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    tt_plot_path = OUTPUT_DIR / "TT_residual_comparison_by_binning.png"
    plt.savefig(tt_plot_path, dpi=200)
    plt.close()

    print("\nPlots saved:")
    print("-", sr_plot_path)
    print("-", tt_plot_path)


#  Loads the base YAML file, updates the parameters, runs SAEB, computes the
#  Strehl ratio and its difference from the measured LUCI value. Returns 
#  the output dictionary.


def run_saeb(seeing, magnitude, binning, ao_framerate, output_info):
   
    param = load_parameters("params_Total_variance_SOUL_LUCI2.yaml")
    param["atmosphere"]["seeing"] = float(seeing)
    param["guide_star"]["magn"] = float(magnitude)
    param["control"]["bin"] = int(binning)
    sampling_time = 1.0 / float(ao_framerate)
    param["control"]["sampling_time"] = sampling_time

    if sampling_time >= 0.00167:
        param["plant"]["total_delay"] = 1
    else:
        param["plant"]["total_delay"] = 2

    # save modified YAML
    modified_yaml_path = TMP_DIR / "params_Total_variance_SOUL_LUCI2_modified.yaml"
    with open(modified_yaml_path, "w") as file:
        yaml.dump(param, file, sort_keys=False)

    result = run(str(modified_yaml_path), return_std=True)

    # create output dictionary
    output = dict(output_info)
    output.update(result)

    if "TT_RESIDUAL_NM" in output and "std_tt" in output:
        tt_delta_nm, tt_ratio = tt_comparison_metrics(output["TT_RESIDUAL_NM"], output["std_tt"])
        output["TT_RESIDUAL_EB_NM"] = output["std_tt"]
        output["TT_RESIDUAL_DIFF_NM"] = tt_delta_nm
        output["TT_RESIDUAL_RATIO"] = tt_ratio
    
    # Compute Strehl Ratio
    
    if "var_total" in output:
        variance_tot = output["var_total"]
    else:
        variance_tot = output["std_total"]**2
    Lambda_Luci = output["LAMBDA_LUCI"]
    
    if Lambda_Luci != -1:
        output["SR_SAEB"] = np.exp(-variance_tot * (2*np.pi/Lambda_Luci)**2)
    else:
         output["SR_SAEB"] = np.nan
    
    # Difference with measured Strehl
    
    if output["SR_LUCI"] != -1 and not np.isnan(output["SR_SAEB"]):
        output["SR_DIFF"] = output["SR_SAEB"] - output["SR_LUCI"]
    else:
        output["SR_DIFF"] = np.nan 

    return output


# =============================================================================
# DATA LOADING AND DATAFRAME CREATION
# =============================================================================

data = readsav( "data/LBT/Table_CAT_LUCI_DX_SKY.sav", python_dict=True)

# keys check
print("data keys:",data.keys())

table = data["table"]

df_columns = {
    "TN": decode_bytes_array(table["TN"][0]),
    "WFS_MAG": to_native_endian(table["WFS_MAG"][0]),
    "STAR_ID": decode_bytes_array(table["STAR_ID"][0]),
    "AO_FRAMERATE": to_native_endian(table["AO_FRAMERATE"][0]),
    "BINNING": to_native_endian(table["BINNING"][0]),
    "DIMM": to_native_endian(table["DIMM"][0]),
    "LAMBDA_LUCI": to_native_endian(table["LAMBDA_LUCI"][0]),
    "SR_LUCI": to_native_endian(table["SR_LUCI"][0])
}

if "SR_SLOPES" in table.dtype.names:
    df_columns["SR_SLOPES"] = to_native_endian(table["SR_SLOPES"][0])

if "SR_SLOPES_TTFREE" in table.dtype.names:
    df_columns["SR_SLOPES_TTFREE"] = to_native_endian(table["SR_SLOPES_TTFREE"][0])

df = pd.DataFrame(df_columns)

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
        "LAMBDA_LUCI": row["LAMBDA_LUCI"],
        "SR_LUCI": row["SR_LUCI"]
    }

    if "SR_SLOPES" in row.index:
        output_info["SR_SLOPES"] = row["SR_SLOPES"]

    if "SR_SLOPES_TTFREE" in row.index:
        output_info["SR_SLOPES_TTFREE"] = row["SR_SLOPES_TTFREE"]

    if "SR_SLOPES" in row.index and "SR_SLOPES_TTFREE" in row.index:
        tt_residual_nm = tt_residual_nm_from_sr(
            row["SR_SLOPES"],
            row["SR_SLOPES_TTFREE"],
            TT_REFERENCE_WAVELENGTH_NM,
        )
        output_info["TT_RESIDUAL_NM"] = tt_residual_nm

        print("SR_SLOPES:", row["SR_SLOPES"])
        print("SR_SLOPES_TTFREE:", row["SR_SLOPES_TTFREE"])
        print("TT residual [nm]:", tt_residual_nm)
        
    output = run_saeb(seeing=row["DIMM"], magnitude=row["WFS_MAG"], binning=row["BINNING"],
                      ao_framerate=row["AO_FRAMERATE"], output_info=output_info)
    
    print("\nOUTPUT:", output)
    
    table = Table([output])

    # Save FITS
    single_fits_path = OUTPUT_DIR / "SAEB_results_single_TN.fits"
    table.write(single_fits_path, overwrite=True)
    
    # Save CSV
    df_output = table.to_pandas()
    single_csv_path = OUTPUT_DIR / "SAEB_results_single_TN.csv"
    df_output.to_csv(single_csv_path, index=False)
    
    print("\nSingle TN results:")
    print(df_output)
    
    print("\nFiles saved:")
    print("-", single_fits_path)
    print("-", single_csv_path)
    

elif analysis_mode == "all":

    grouped = df.groupby(["STAR_ID", "BINNING", "AO_FRAMERATE"])
    
    groups_list = list(grouped)
    
    print("\nNumber of groups:")
    print(len(groups_list))
    
    
    summary = []
    
    for group in groups_list:
    
        print("\n" + "=" * 80)
    
        group_name = group[0]
        group_df = group[1]
    
        star_id = group_name[0]
        binning = group_name[1]
        ao_framerate = group_name[2]
        tn_case = group_df["TN"].iloc[0]
    
        dimm_median = group_df["DIMM"].median()
        wfs_mag_median = group_df["WFS_MAG"].median()

        if "SR_SLOPES" in group_df.columns and "SR_SLOPES_TTFREE" in group_df.columns:
            tn_case_rows = group_df[
                (group_df["TN"] == tn_case)
                & (group_df["SR_SLOPES"] != -1)
                & np.isfinite(group_df["SR_SLOPES"])
                & (group_df["SR_SLOPES_TTFREE"] != -1)
                & np.isfinite(group_df["SR_SLOPES_TTFREE"])
            ]

            if len(tn_case_rows) > 0:
                case_row = tn_case_rows.iloc[0]
                sr_slopes_case = float(case_row["SR_SLOPES"])
                sr_slopes_ttfree_case = float(case_row["SR_SLOPES_TTFREE"])
            else:
                sr_slopes_case = np.nan
                sr_slopes_ttfree_case = np.nan
        else:
            sr_slopes_case = np.nan
            sr_slopes_ttfree_case = np.nan

        tt_residual_nm = tt_residual_nm_from_sr(
            sr_slopes_case,
            sr_slopes_ttfree_case,
            TT_REFERENCE_WAVELENGTH_NM,
        )
        
        # Valid wavelength values
        valid_lambda = group_df[group_df["LAMBDA_LUCI"] != -1]
        
        if len(valid_lambda) > 0:
            lambda_luci = valid_lambda["LAMBDA_LUCI"].iloc[0]
        else:
            lambda_luci = -1
     
        # Valid SR values
        valid_sr = group_df[group_df["SR_LUCI"] != -1]

        if len(valid_sr) > 0:
            sr_luci = valid_sr["SR_LUCI"].median()
        else:
            sr_luci = -1

        print("STAR_ID:", star_id)
        print("BINNING:", binning)
        print("AO_FRAMERATE:", ao_framerate)
        print("DIMM median:", dimm_median)
        print("WFS_MAG median:", wfs_mag_median)
        print("LAMBDA_LUCI:", lambda_luci)
        print("SR_LUCI:", sr_luci)
        print("TN:", tn_case)
        print("SR_SLOPES case:", sr_slopes_case)
        print("SR_SLOPES_TTFREE case:", sr_slopes_ttfree_case)
        print("TT residual [nm]:", tt_residual_nm)

    
        output_info = {
            "TN": tn_case,
            "STAR_ID": star_id,
            "BINNING": binning,
            "AO_FRAMERATE": ao_framerate,
            "DIMM_MEDIAN": dimm_median,
            "WFS_MAG_MEDIAN": wfs_mag_median,
            "LAMBDA_LUCI":lambda_luci,
            "SR_LUCI": sr_luci,
            "SR_SLOPES_CASE": sr_slopes_case,
            "SR_SLOPES_TTFREE_CASE": sr_slopes_ttfree_case,
            "TT_RESIDUAL_NM": tt_residual_nm
         }
    
        output = run_saeb(seeing=dimm_median, magnitude=wfs_mag_median, binning=binning, 
                          ao_framerate=ao_framerate, output_info=output_info)
    
        summary.append(output)
    
        print("\nOUTPUT:")
        print(output)
    
    
    table = Table(rows=summary)

    # Save FITS
    all_fits_path = OUTPUT_DIR / "SAEB_results_all_groups.fits"
    table.write(all_fits_path, overwrite=True)
    
    # Save CSV
    df_summary = table.to_pandas()

    if "sr_estimation" in df_summary.columns:
        df_summary["SR_ESTIMATION"] = df_summary["sr_estimation"]

    all_csv_path = OUTPUT_DIR / "SAEB_results_all_groups.csv"
    df_summary.to_csv(all_csv_path, index=False)

    plot_sr_and_tt_comparisons(df_summary)
    
    print("\nSummary table:")
    print(df_summary)
    
    print("\nFiles saved:")
    print("-", all_fits_path)
    print("-", all_csv_path)
    print("Processed groups:", len(summary))
        
        



    
    
    
    
    
    
    
    
    
    
    
    
    