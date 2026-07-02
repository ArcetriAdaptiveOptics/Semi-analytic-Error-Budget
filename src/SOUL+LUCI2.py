#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 14:28:10 2026

@author: greta
"""

from scipy.io import readsav
import pandas as pd
import numpy as np


# ============================================================
# Funzioni di utilità
# ============================================================

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
    Converte un array di byte string in stringhe Python normali.

    Esempio:
    b'Wolf 1409'  ->  'Wolf 1409'
    """
    return [
        x.decode("utf-8").strip() if isinstance(x, bytes) else str(x).strip()
        for x in array
    ]


# ============================================================
# Caricamento dati
# ============================================================

# Leggo il file .sav e lo converto in un dizionario Python
data = readsav( "file_fits/LBT/Table_CAT_LUCI_DX_SKY.sav", python_dict=True)

# Controllo le chiavi disponibili nel dizionario
print(data.keys())

# Estraggo la tabella principale
table = data["table"]


# ============================================================
# Creazione del DataFrame Pandas
# ============================================================

df = pd.DataFrame({
    "TN": decode_bytes_array(table["TN"][0]),
    "WFS_MAG": to_native_endian(table["WFS_MAG"][0]),
    "STAR_ID": decode_bytes_array(table["STAR_ID"][0]),
    "AO_FRAMERATE": to_native_endian(table["AO_FRAMERATE"][0]),
    "BINNING": to_native_endian(table["BINNING"][0]),
    "DIMM": to_native_endian(table["DIMM"][0]),
})


# ============================================================
# Controlli preliminari
# ============================================================

print("\nPrime righe del DataFrame:")
print(df.head())

print("\nDimensioni del DataFrame:")
print(df.shape)

print("\nTipi delle colonne:")
print(df.dtypes)

print("\nControllo byte order delle colonne numeriche:")
print("WFS_MAG:", df["WFS_MAG"].values.dtype)
print("AO_FRAMERATE:", df["AO_FRAMERATE"].values.dtype)
print("BINNING:", df["BINNING"].values.dtype)
print("DIMM:", df["DIMM"].values.dtype)


# Stampo tutta la tabella.

# print(df.to_string())


# ============================================================
# Creazione dei gruppi
# ============================================================

# Raggruppo per stella, binning e frame rate AO
grouped = df.groupby(["STAR_ID", "BINNING", "AO_FRAMERATE"])


# ============================================================
# Conversione dei gruppi in lista
# ============================================================

groups_list = list(grouped)

print("\nNumero di gruppi:")
print(len(groups_list))


# ============================================================
# Stampa dei gruppi
# ============================================================

# Stampo solo i primi 5 gruppi per controllo
for i, (group_name, group_df) in enumerate(groups_list[:5]):

    print("\n" + "=" * 60)
    print("Gruppo numero:", i)
    print("Nome gruppo:", group_name)

    star_id, binning, ao_framerate = group_name

    print("STAR_ID:", star_id)
    print("BINNING:", binning)
    print("AO_FRAMERATE:", ao_framerate)

    print("\nDati del gruppo:")
    print(group_df)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    