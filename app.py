# app.py ────────────────────────────────────────────────────────────────
"""
ENAHO 2022 – Limpieza básica, outliers y visualización didáctica
Autor: YorchisYorch | Fecha: 2025-07-01
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# 1. Configuración inicial ------------------------------------------------
DATA_FILE = "Enaho01-2022-100.csv"   # nombre del CSV
OUTPUT_DIR = Path("fig")             # usar carpeta fig/ para LaTeX
OUTPUT_DIR.mkdir(exist_ok=True)

print("Cargando datos…")
df_raw = pd.read_csv(DATA_FILE, encoding="latin-1")

# 2. Limpieza rápida ------------------------------------------------------
df = (
    df_raw.dropna(axis=1, how="all")      # quita columnas vacías
           .drop_duplicates()             # quita filas duplicadas
)

# Convierte textos que son números a numérico
for col in df.select_dtypes(include="object").columns:
    if df[col].str.match(r"^-?\d+(\.\d+)?$", na=False).all():
        df[col] = pd.to_numeric(df[col])

num_cols = df.select_dtypes(include=np.number).columns

# 3. Funciones para outliers (método IQR) --------------------------------
def flag_outliers(series, k=1.5):
    q1, q3 = series.quantile([0.25, 0.75])
    iqr = q3 - q1
    lo, hi = q1 - k * iqr, q3 + k * iqr
    return (series < lo) | (series > hi)

def winsorize(series, k=1.5):
    q1, q3 = series.quantile([0.25, 0.75])
    iqr = q3 - q1
    lo, hi = q1 - k * iqr, q3 + k * iqr
    return series.clip(lower=lo, upper=hi)

# 4. Detecta y cuenta outliers -------------------------------------------
outlier_flags = df[num_cols].apply(flag_outliers)
outlier_counts = outlier_flags.sum().sort_values(ascending=False)
outlier_counts.to_csv(OUTPUT_DIR / "outlier_counts.csv", header=["n_outliers"])

print(f"Variables con más outliers:\n{outlier_counts.head(10)}\n")

# 5. Winsorización (opcional) --------------------------------------------
df_winsor = df.copy()
df_winsor[num_cols] = df_winsor[num_cols].apply(winsorize)

df_winsor.to_csv(OUTPUT_DIR / "enaho_winsor.csv", index=False)
print("Base winsorizada guardada en output/enaho_winsor.csv")

# 6. Estadística descriptiva ---------------------------------------------
desc_raw    = df[num_cols].describe().T
desc_winsor = df_winsor[num_cols].describe().T

desc_raw.to_csv(OUTPUT_DIR / "stats_raw.csv")
desc_winsor.to_csv(OUTPUT_DIR / "stats_winsor.csv")

# 7. Gráficos didácticos --------------------------------------------------
plt.rcParams.update({"figure.autolayout": True})  # evita cortes

## 7.1 Barra – top 20 variables con más outliers
top_n = 20
plt.figure(figsize=(9, 6))
outlier_counts.head(top_n).plot(kind="bar")
plt.title(f"Top {top_n} variables con más outliers")
plt.ylabel("Número de outliers")
plt.xticks(rotation=45, ha="right", fontsize=8)
plt.savefig(OUTPUT_DIR / "outliers_top20.png", dpi=300)
plt.close()

## 7.2 Histograma + boxplot de una variable ejemplo
demo_var = "FACTOR07" if "FACTOR07" in num_cols else num_cols[0]

# Histograma antes y después
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.hist(df[demo_var].dropna(), bins=30)
plt.title(f"{demo_var} (crudo)")

plt.subplot(1, 2, 2)
plt.hist(df_winsor[demo_var].dropna(), bins=30)
plt.title(f"{demo_var} (winsor)")

plt.savefig(OUTPUT_DIR / "hist_ingreso.png", dpi=300)
plt.close()

# Boxplot crudo (separado)
plt.figure(figsize=(4, 5))
plt.boxplot(df[demo_var].dropna())
plt.title(f'Distribución de {demo_var} (Crudo)')
plt.ylabel(demo_var)
plt.savefig(OUTPUT_DIR / "box_FACTOR07.png", dpi=300)
plt.close()

# Boxplot winsorizado (separado)
plt.figure(figsize=(4, 5))
plt.boxplot(df_winsor[demo_var].dropna())
plt.title(f'Distribución de {demo_var} (Winsorizado)')
plt.ylabel(demo_var)
plt.savefig(OUTPUT_DIR / "box_FACTOR07_winsor.png", dpi=300)
plt.close()

# Gráfico de barras pobre vs no pobre
# Crear variable de pobreza sintética basada en percentiles de FACTOR07
threshold = df[demo_var].quantile(0.3)  # 30% más pobres
pobre = (df[demo_var] < threshold).astype(int)

plt.figure(figsize=(10, 6))
categories = ['No Pobre', 'Pobre']
means = [df[pobre == 0][demo_var].mean(), df[pobre == 1][demo_var].mean()]
plt.bar(categories, means, color=['skyblue', 'lightcoral'])
plt.title('Comparación Media de Ingreso: Pobre vs No Pobre')
plt.ylabel('Media de Ingreso (FACTOR07)')
for i, v in enumerate(means):
    plt.text(i, v + max(means)*0.01, f'{v:.2f}', ha='center', va='bottom')
plt.savefig(OUTPUT_DIR / "bar_pobre_vs_nopobre.png", dpi=300)
plt.close()

print("Gráficos guardados en la carpeta 'fig'.")

# 8. Mensaje final --------------------------------------------------------
print("\nProceso completado ✔")
print("► Estadísticos     → fig/stats_raw.csv y fig/stats_winsor.csv")
print("► Outliers         → fig/outlier_counts.csv")
print("► Gráficos         → fig/*.png")
