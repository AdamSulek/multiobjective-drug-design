import pandas as pd
import numpy as np

# === pliki wejściowe / wyjściowe ===
INPUT_PARQUET = "savi_merged_all_3.parquet"
OUTPUT_PARQUET = "savi_3D_wo_X.parquet"

IDS_NPY = "ids.npy"
X_NPY = "X_uint8.npy"

ID_COLUMN = "ID"          # <- zmień jeśli masz inną nazwę
X_COLUMN = "X_ecfp_2"

print("Wczytywanie parquet...")
df = pd.read_parquet(INPUT_PARQUET)

print("Liczba wierszy:", len(df))

# --- zapis ID w tej samej kolejności ---
print("Zapisywanie ID...")
ids = df[ID_COLUMN].to_numpy()
np.save(IDS_NPY, ids)

# --- konwersja X ---
print("Konwersja X_ecfp_2 do macierzy numpy...")

# upewnij się że każdy element to array
X_list = df[X_COLUMN].apply(np.array)

# sprawdź długości
lengths = X_list.apply(len)
if lengths.nunique() != 1:
    raise ValueError("Wektory X_ecfp_2 mają różne długości!")

X = np.vstack(X_list.values).astype(np.uint8)

print("Shape X:", X.shape)

np.save(X_NPY, X)

# --- usuń kolumnę X ---
print("Usuwanie kolumny X_ecfp_2...")
df_wo_X = df.drop(columns=[X_COLUMN])

print("Zapisywanie nowego parquet...")
df_wo_X.to_parquet(OUTPUT_PARQUET, index=False)

print("Gotowe.")