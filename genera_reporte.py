# Generador de Reporte PDF para ENAHO 2022
# Autor: YorchisYorch | Fecha: 2025-07-15

import pandas as pd
from fpdf import FPDF
from pathlib import Path
import glob

# Paths
OUTPUT_DIR = Path("output")
STATS_FILE = OUTPUT_DIR / "stats_winsor.csv"
PDF_FILE = OUTPUT_DIR / "Reporte_ENAHO2022.pdf"

# Leer estadísticos
df_stats = pd.read_csv(STATS_FILE)

# Inicializar PDF
pdf = FPDF()
pdf.set_auto_page_break(auto=True, margin=15)
pdf.add_page()
pdf.set_font("helvetica", 'B', 16)
pdf.cell(0, 10, "ENAHO 2022 - Reporte de Evidencia", ln=True, align="C")
pdf.ln(5)
pdf.set_font("helvetica", '', 12)
pdf.multi_cell(0, 10, """
Este reporte contiene la evidencia del proceso de limpieza, winsorización y análisis descriptivo de la base ENAHO 2022. Incluye estadísticos y gráficos generados automáticamente.

Autor: YorchisYorch
Fecha: 2025-07-15
""")
pdf.ln(5)

# Tabla de estadísticos (primeras 10 filas)
pdf.set_font("helvetica", 'B', 12)
pdf.cell(0, 10, "Estadísticos descriptivos (primeras 10 variables):", ln=True)
pdf.set_font("helvetica", '', 9)

cols = list(df_stats.columns)
col_widths = [25] + [18]*(len(cols)-1)

# Encabezado
for i, col in enumerate(cols):
    pdf.cell(col_widths[i], 8, str(col), border=1)
pdf.ln()

# Primeras 10 filas
for idx, row in df_stats.head(10).iterrows():
    for i, col in enumerate(cols):
        pdf.cell(col_widths[i], 8, str(row[col])[:15], border=1)
    pdf.ln()
pdf.ln(5)

# Incluir gráficos
pdf.set_font("helvetica", 'B', 12)
pdf.cell(0, 10, "Gráficos generados:", ln=True)
pdf.ln(2)

img_files = sorted(glob.glob(str(OUTPUT_DIR / "*.png")))
for img in img_files:
    pdf.image(img, w=120)
    pdf.ln(5)

# Guardar PDF
pdf.output(str(PDF_FILE))
print(f"Reporte PDF generado: {PDF_FILE}")
