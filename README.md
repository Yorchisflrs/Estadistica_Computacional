# Análisis ENAHO 2022 - Dataset de Prueba

Este directorio contiene el análisis completo de limpieza de datos, detección de outliers y generación de reportes para un subconjunto de 100 registros de la ENAHO 2022.

## Archivos principales

### Scripts Python
- `app.py` - Script principal de análisis y limpieza de datos
- `genera_reporte.py` - Generador de reporte PDF
- `gen_enaho_report_assets.py` - Script completo para generar assets LaTeX
- `gen_latex_images.py` - Generador específico de imágenes para LaTeX

### Datos
- `Enaho01-2022-100.csv` - Dataset de 100 registros ENAHO 2022
- `dataset.R` - Script R (si aplica)

### Resultados generados

#### Carpeta `fig/` (para LaTeX)
- `outliers_top20.png` - Top 20 variables con más outliers
- `box_FACTOR07.png` - Boxplot crudo de FACTOR07
- `box_FACTOR07_winsor.png` - Boxplot winsorizado de FACTOR07
- `hist_ingreso.png` - Histograma de distribución de ingreso
- `bar_pobre_vs_nopobre.png` - Comparación por grupos de pobreza
- `stats_winsor.csv` - Estadísticos descriptivos winsor izados
- `outlier_counts.csv` - Conteo de outliers por variable
- `enaho_winsor.csv` - Dataset winsorizado completo

#### Carpeta `output/` (archivos originales)
- `Reporte_ENAHO2022.pdf` - Reporte PDF completo
- Otros archivos CSV y PNG

## Uso

### Para ejecutar el análisis completo:
```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
pip install pandas numpy matplotlib fpdf2
python app.py
```

### Para generar reporte PDF:
```bash
python genera_reporte.py
```

### Para generar imágenes específicas para LaTeX:
```bash
python gen_latex_images.py
```

## Metodología

1. **Limpieza de datos**: Eliminación de columnas vacías y filas duplicadas
2. **Detección de outliers**: Método IQR con k=1.5
3. **Winsorización**: Truncamiento de valores extremos a límites IQR
4. **Análisis descriptivo**: Estadísticos antes y después del tratamiento
5. **Visualización**: Gráficos comparativos y de distribución

## Autor
Jorge L. Flores Turpo - Estadística Computacional - FINESI - UNA

## Fecha
Julio 2025
