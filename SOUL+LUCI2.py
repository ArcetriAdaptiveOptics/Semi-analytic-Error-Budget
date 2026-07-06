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


# Funzioni utili

def to_native_endian(array):
    """
    Converte un array NumPy nel formato endian nativo della macchina.
     Serve perché alcuni dati letti da file IDL .sav possono essere
    big-endian, mentre Pandas su macchine little-endian può dare errore
    durante operazioni come groupby.
    """
    array = np.asarray(array)

    if array.dtype.byteorder not in ("=", "|"):
        return array.astype(array.dtype.newbyteorder("="))

    return array


def decode_bytes_array(array):
    """
    Converte un array di byte in normali stringhe Python.
    """

    result = []

    for x in array:

        if isinstance(x, bytes):
            x = x.decode("utf-8")

        x = x.strip()

        result.append(x)

    return result


# Caricamento dati

# Leggo il file .sav e lo converto in un dizionario Python
data = readsav( "src/file_fits/LBT/Table_CAT_LUCI_DX_SKY.sav", python_dict=True)

# Controllo le chiavi disponibili nel dizionario
print("data keys:",data.keys())

# Estraggo la tabella principale
table = data["table"]


# Creazione del DataFrame Pandas

df = pd.DataFrame({
    "TN": decode_bytes_array(table["TN"][0]),
    "WFS_MAG": to_native_endian(table["WFS_MAG"][0]),
    "STAR_ID": decode_bytes_array(table["STAR_ID"][0]),
    "AO_FRAMERATE": to_native_endian(table["AO_FRAMERATE"][0]),
    "BINNING": to_native_endian(table["BINNING"][0]),
    "DIMM": to_native_endian(table["DIMM"][0]),
})


# Controlli preliminari

print("\nPrime righe del DataFrame:")
print(df.head())

print("\nDimensioni del DataFrame:")
print(df.shape)


# Stampo tutta la tabella (solo se necessario).
# print(df.to_string())


# =============================================================================
# CASO SINGOLO TN - Prova 
# =============================================================================

# TN stella ultrafaint 

tn_target = "20230427_111423"

mask = df["TN"] == tn_target
df_single = df[mask]

print(df_single)
print(df_single.shape)


row = df_single.iloc[0]

# Carico il file.yaml

param = load_parameters('params_Total_variance_SOUL+LUCI2.yaml')


# Sovrascivo i parametri

# seeing
param["atmosphere"]["seeing"] = float(row["DIMM"])

# magnitudine 
param["guide_star"]["magn"] = float(row["WFS_MAG"])

# binning
param["control"]["bin"] = int(row["BINNING"])

# sampling time
param["control"]["sampling_time"] = 1.0 / float(row["AO_FRAMERATE"])

# Se sampling_time >= 0.00167 s, delay è 1; altrimenti 2 

if param["control"]["sampling_time"] >= 0.00167:
    param["plant"]["total_delay"] = 1
else:
    param["plant"]["total_delay"] = 2

print(type(param["control"]["bin"]))


# Salvo lo YAML modificato

with open("params_Total_variance_SOUL+LUCI2_modified_single_TN.yaml", "w") as file:
    yaml.dump(param, file, sort_keys=False)

print("File YAML modificato salvato.")


# Giro da main_saeb.py il run sullo yaml modificato

result = run("params_Total_variance_SOUL+LUCI2_modified_single_TN.yaml")


# print(result)

# Metto insieme telemetria e risultati

output = {
    "TN": row["TN"],
    "WFS_MAG": row["WFS_MAG"],
    "STAR_ID": row["STAR_ID"],
    "AO_FRAMERATE": row["AO_FRAMERATE"],
    "BINNING": row["BINNING"],
    "DIMM": row["DIMM"],
}

# aggiungo var_fit, var_temp ecc a output
output.update(result)

print ("\nOUTPUT:",output)

# Costrusco il fits con telemetria + risultati

table = Table([output])

table.write("src/file_fits/LBT/SAEB_results_single_TN.fits", overwrite=True)

#print ("TABLE:", table)




# =============================================================================
# CASO GENERALE CON TUTTI I TN
# =============================================================================

# Creazione dei gruppi

# Raggruppo per stella, binning e frame rate 
grouped = df.groupby(["STAR_ID", "BINNING", "AO_FRAMERATE"])


# Conversione dei gruppi in lista
groups_list = list(grouped)


print("\nNumero di gruppi:")
print(len(groups_list))
 

# Stampa dei gruppi (prendo solo i primi 5 gruppi per controllare)
primi_gruppi = groups_list[:5]

i = 0

for gruppo in primi_gruppi:

    print("\n" + "=" * 60)

    # gruppo è una coppia: (nome_gruppo, dataframe)
    group_name = gruppo[0]
    group_df = gruppo[1]

    print("Gruppo numero:", i) 

    print("Nome gruppo:", group_name)

    # group_name è una tupla: (STAR_ID, BINNING, AO_FRAMERATE)
    star_id = group_name[0]
    binning = group_name[1]
    ao_framerate = group_name[2]

    print("STAR_ID:", star_id)
    print("BINNING:", binning)
    print("AO_FRAMERATE:", ao_framerate)

    print("\nDati del gruppo:")
    print(group_df)

    i = i + 1
    
 
# Ciclo for per calcolo mediana, sovrascrivo telemetria in yaml, 
# faccio girare run  sullo yaml, salvo telemetria e risultati in un fits 
# (che metto in src-->file_fits-->LBT)
    
 
for gruppo in groups_list:

    print("\n" + "=" * 60)

    group_name = gruppo[0]
    group_df = gruppo[1]

    # Calcolo della mediana del seeing DIMM
    dimm_mediana = group_df["DIMM"].median()

    # Calcolo della mediana della magnitudine WFS
    wfs_mag_mediana = group_df["WFS_MAG"].median()
    
    print("Mediana DIMM:", dimm_mediana)
    print("Mediana WFS_MAG:", wfs_mag_mediana)


 
    
    

    




















    
    
    
    
    
    
    
    