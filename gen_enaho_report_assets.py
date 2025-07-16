#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generador de assets para informe LaTeX - ENAHO 2022
Autor: YorchisYorch | Fecha: 2025-07-16

Script reproducible que procesa ENAHO, detecta outliers, genera figuras y tablas
para un informe LaTeX completo con modelo predictivo de pobreza.
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, roc_curve
from sklearn.pipeline import Pipeline
import re

# Intentar importar SHAP (opcional)
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("AVISO: SHAP no está instalado. Se omitirán gráficos SHAP.")

warnings.filterwarnings('ignore')

# =============================================================================
# 1. CONFIGURACIÓN EDITABLE
# =============================================================================
CONFIG = {
    'DATA_FILE': "Enaho01-2022-100.csv",
    'COL_INGRESO': "FACTOR07",           # Cambiar por columna de ingreso real
    'COL_POBRE': None,                   # Si None, se calcula con POVERTY_LINE
    'COL_FACTOR': "FACTOR07",            # Peso muestral
    'COL_TAMHOG': "T111A",               # Tamaño del hogar
    'POVERTY_LINE': 450,                 # Soles para línea de pobreza
    'MAX_CAT_LEVELS': 20,                # Máximo niveles para categorías
    'RANDOM_STATE': 42,
    'TOP_OUTLIERS_PLOT': 20
}

# =============================================================================
# 2. FUNCIONES AUXILIARES
# =============================================================================

def create_dirs():
    """Crear directorios fig/ y tab/ si no existen."""
    Path("fig").mkdir(exist_ok=True)
    Path("tab").mkdir(exist_ok=True)
    print("✓ Directorios fig/ y tab/ creados")

def load_data(filepath):
    """
    Cargar datos desde CSV con encoding latin-1.
    
    Returns:
        pd.DataFrame: Datos cargados
    """
    try:
        df = pd.read_csv(filepath, encoding="latin-1")
        print(f"✓ Datos cargados: {df.shape[0]} filas, {df.shape[1]} columnas")
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"No se encontró el archivo: {filepath}")
    except Exception as e:
        raise Exception(f"Error al cargar datos: {e}")

def clean_data(df):
    """
    Limpieza básica de datos.
    
    Args:
        df: DataFrame original
        
    Returns:
        pd.DataFrame: Datos limpios
    """
    print("\n--- LIMPIEZA DE DATOS ---")
    
    # Eliminar columnas totalmente vacías
    cols_before = df.shape[1]
    df = df.dropna(axis=1, how='all')
    cols_after = df.shape[1]
    print(f"• Columnas eliminadas (vacías): {cols_before - cols_after}")
    
    # Eliminar filas duplicadas
    rows_before = df.shape[0]
    df = df.drop_duplicates()
    rows_after = df.shape[0]
    print(f"• Filas eliminadas (duplicadas): {rows_before - rows_after}")
    
    # Convertir columnas numéricas en texto a numérico
    text_cols = df.select_dtypes(include=['object']).columns
    converted = 0
    for col in text_cols:
        if df[col].notna().sum() > 0:  # Si tiene datos no nulos
            # Verificar si todos los valores no nulos son numéricos
            sample = df[col].dropna().astype(str)
            if len(sample) > 0 and sample.str.match(r'^-?\d+(\.\d+)?$').all():
                df[col] = pd.to_numeric(df[col], errors='coerce')
                converted += 1
    
    print(f"• Columnas convertidas a numéricas: {converted}")
    
    # Identificar tipos finales
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    print(f"• Columnas numéricas: {len(num_cols)}")
    print(f"• Columnas categóricas: {len(cat_cols)}")
    
    return df, num_cols, cat_cols

def validate_config(df, config):
    """
    Validar que las columnas configuradas existan.
    
    Args:
        df: DataFrame
        config: Diccionario de configuración
        
    Returns:
        dict: Configuración validada
    """
    print("\n--- VALIDACIÓN DE CONFIGURACIÓN ---")
    validated = config.copy()
    
    # Validar columna de ingreso
    if config['COL_INGRESO'] not in df.columns:
        print(f"ADVERTENCIA: Columna {config['COL_INGRESO']} no existe")
        # Buscar columnas que contengan 'ingreso' o similar
        candidates = [col for col in df.columns if 'ingreso' in col.lower() or 'income' in col.lower()]
        if candidates:
            validated['COL_INGRESO'] = candidates[0]
            print(f"• Usando {candidates[0]} como ingreso")
        else:
            print("• No se encontró columna de ingreso. Algunos gráficos se omitirán")
            validated['COL_INGRESO'] = None
    
    # Validar otras columnas
    for key in ['COL_FACTOR', 'COL_TAMHOG']:
        if config[key] and config[key] not in df.columns:
            print(f"ADVERTENCIA: Columna {config[key]} no existe")
            validated[key] = None
    
    # Validar columna de pobreza
    if config['COL_POBRE'] and config['COL_POBRE'] not in df.columns:
        print(f"ADVERTENCIA: Columna {config['COL_POBRE']} no existe")
        validated['COL_POBRE'] = None
    
    return validated

def detect_outliers(df, num_cols, k=1.5):
    """
    Detectar outliers usando método IQR.
    
    Args:
        df: DataFrame
        num_cols: Lista de columnas numéricas
        k: Factor para IQR (default 1.5)
        
    Returns:
        tuple: (outlier_flags, outlier_counts, limits_dict)
    """
    print("\n--- DETECCIÓN DE OUTLIERS ---")
    
    outlier_flags = pd.DataFrame(index=df.index)
    limits = {}
    
    for col in num_cols:
        if col in df.columns and df[col].notna().sum() > 0:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            LI = Q1 - k * IQR
            LS = Q3 + k * IQR
            
            outlier_flags[col] = (df[col] < LI) | (df[col] > LS)
            limits[col] = {'LI': LI, 'LS': LS, 'Q1': Q1, 'Q3': Q3}
    
    outlier_counts = outlier_flags.sum().sort_values(ascending=False)
    
    print(f"• Variables analizadas: {len(outlier_counts)}")
    print(f"• Total outliers detectados: {outlier_flags.sum().sum()}")
    print(f"• Top 5 variables con outliers:")
    for var, count in outlier_counts.head().items():
        print(f"  - {var}: {count}")
    
    return outlier_flags, outlier_counts, limits

def winsorize_data(df, limits, num_cols):
    """
    Aplicar winsorización usando límites IQR.
    
    Args:
        df: DataFrame original
        limits: Diccionario con límites por variable
        num_cols: Lista de columnas numéricas
        
    Returns:
        pd.DataFrame: Datos winsor izados
    """
    df_winsor = df.copy()
    
    for col in num_cols:
        if col in limits:
            LI, LS = limits[col]['LI'], limits[col]['LS']
            df_winsor[col] = df_winsor[col].clip(lower=LI, upper=LS)
    
    print(f"✓ Winsorización aplicada a {len(limits)} variables")
    return df_winsor

def create_poverty_indicator(df, config):
    """
    Crear indicador de pobreza si no existe.
    
    Args:
        df: DataFrame
        config: Configuración validada
        
    Returns:
        pd.Series: Indicador de pobreza (1=pobre, 0=no pobre)
    """
    if config['COL_POBRE'] and config['COL_POBRE'] in df.columns:
        print(f"✓ Usando columna existente de pobreza: {config['COL_POBRE']}")
        return df[config['COL_POBRE']]
    
    elif config['COL_INGRESO'] and config['COL_INGRESO'] in df.columns:
        pobre = (df[config['COL_INGRESO']] < config['POVERTY_LINE']).astype(int)
        print(f"✓ Indicador de pobreza creado: {pobre.sum()} pobres de {len(pobre)} observaciones")
        print(f"• Línea de pobreza: {config['POVERTY_LINE']} soles")
        return pobre
    
    else:
        print("ADVERTENCIA: No se puede crear indicador de pobreza")
        return None

# =============================================================================
# 3. GENERACIÓN DE TABLAS
# =============================================================================

def make_tables(df, df_winsor, outlier_counts, config, num_cols):
    """
    Generar tablas LaTeX para el informe.
    """
    print("\n--- GENERANDO TABLAS ---")
    
    # Estadísticos descriptivos
    desc_raw = df[num_cols].describe().T
    desc_winsor = df_winsor[num_cols].describe().T
    
    # Tabla de rangos crudo vs winsor
    ranges_table = pd.DataFrame({
        'min_raw': desc_raw['min'],
        'max_raw': desc_raw['max'],
        'min_winsor': desc_winsor['min'],
        'max_winsor': desc_winsor['max'],
        'n_outliers': outlier_counts.reindex(desc_raw.index, fill_value=0)
    })
    
    # Guardar outlier counts
    outlier_counts.to_csv("tab/outlier_counts.csv", header=['n_outliers'])
    print("✓ tab/outlier_counts.csv")
    
    # Exportar tablas LaTeX
    ranges_table.to_latex("tab/rangos_raw_vs_winsor.tex", float_format="%.2f", escape=False)
    print("✓ tab/rangos_raw_vs_winsor.tex")
    
    desc_winsor.to_latex("tab/stats_winsor_full.tex", float_format="%.2f", escape=False)
    print("✓ tab/stats_winsor_full.tex")
    
    # Tabla resumen con columnas clave
    key_cols = []
    for col in [config['COL_INGRESO'], config['COL_TAMHOG'], config['COL_FACTOR']]:
        if col and col in desc_winsor.index:
            key_cols.append(col)
    
    if key_cols:
        stats_short = desc_winsor.loc[key_cols, ['mean', '50%', 'std']]
        stats_short.columns = ['Media', 'Mediana', 'Desv_Std']
        stats_short.to_latex("tab/stats_winsor_short.tex", float_format="%.2f", escape=False)
        print("✓ tab/stats_winsor_short.tex")
    
    return desc_raw, desc_winsor

# =============================================================================
# 4. GENERACIÓN DE FIGURAS
# =============================================================================

def make_plots(df, df_winsor, outlier_counts, config, pobre=None):
    """
    Generar todas las figuras estáticas.
    """
    print("\n--- GENERANDO FIGURAS ---")
    
    plt.style.use('default')
    plt.rcParams.update({'figure.autolayout': True})
    
    # 4.1 Barra outliers
    plt.figure(figsize=(10, 6))
    top_outliers = outlier_counts.head(config['TOP_OUTLIERS_PLOT'])
    top_outliers.plot(kind='barh')
    plt.title(f'Top {config["TOP_OUTLIERS_PLOT"]} Variables con Más Outliers')
    plt.xlabel('Número de Outliers')
    plt.tight_layout()
    plt.savefig('fig/outliers_top20.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ fig/outliers_top20.png")
    
    # 4.2 Boxplots crudo/winsor
    demo_var = config['COL_FACTOR'] if config['COL_FACTOR'] in df.columns else df.select_dtypes(include=[np.number]).columns[0]
    
    # Boxplot crudo
    plt.figure(figsize=(6, 8))
    plt.boxplot(df[demo_var].dropna())
    plt.title(f'Distribución {demo_var} (Crudo)')
    plt.ylabel(demo_var)
    plt.savefig('fig/box_FACTOR07.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ fig/box_FACTOR07.png")
    
    # Boxplot winsor
    plt.figure(figsize=(6, 8))
    plt.boxplot(df_winsor[demo_var].dropna())
    plt.title(f'Distribución {demo_var} (Winsorizado)')
    plt.ylabel(demo_var)
    plt.savefig('fig/box_FACTOR07_winsor.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ fig/box_FACTOR07_winsor.png")
    
    # 4.3 Histograma ingreso
    if config['COL_INGRESO'] and config['COL_INGRESO'] in df.columns:
        plt.figure(figsize=(10, 6))
        plt.hist(df[config['COL_INGRESO']].dropna(), bins=50, alpha=0.7, edgecolor='black')
        if config['COL_POBRE'] is None:  # Si usamos línea de pobreza
            plt.axvline(config['POVERTY_LINE'], color='red', linestyle='--', 
                       label=f'Línea de Pobreza ({config["POVERTY_LINE"]})')
            plt.legend()
        plt.title('Distribución de Ingreso')
        plt.xlabel('Ingreso')
        plt.ylabel('Frecuencia')
        plt.savefig('fig/hist_ingreso.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ fig/hist_ingreso.png")
    
    # 4.4 Barras pobre vs no pobre
    if pobre is not None and config['COL_INGRESO']:
        try:
            df_analysis = df.copy()
            df_analysis['pobre'] = pobre
            
            means = []
            labels = []
            categories = ['No Pobre', 'Pobre']
            
            for cat, val in zip(categories, [0, 1]):
                subset = df_analysis[df_analysis['pobre'] == val]
                if len(subset) > 0:
                    mean_vals = []
                    if config['COL_INGRESO'] in subset.columns:
                        mean_vals.append(subset[config['COL_INGRESO']].mean())
                    if config['COL_TAMHOG'] and config['COL_TAMHOG'] in subset.columns:
                        mean_vals.append(subset[config['COL_TAMHOG']].mean())
                    means.append(mean_vals)
                    labels.append(cat)
            
            if means:
                fig, ax = plt.subplots(figsize=(10, 6))
                x = np.arange(len(means[0]))
                width = 0.35
                
                for i, (mean_list, label) in enumerate(zip(means, labels)):
                    ax.bar(x + i*width, mean_list, width, label=label)
                
                var_names = [config['COL_INGRESO']]
                if config['COL_TAMHOG']:
                    var_names.append(config['COL_TAMHOG'])
                
                ax.set_xlabel('Variables')
                ax.set_ylabel('Media')
                ax.set_title('Comparación Pobre vs No Pobre')
                ax.set_xticks(x + width/2)
                ax.set_xticklabels(var_names)
                ax.legend()
                
                plt.savefig('fig/bar_pobre_vs_nopobre.png', dpi=300, bbox_inches='tight')
                plt.close()
                print("✓ fig/bar_pobre_vs_nopobre.png")
        except Exception as e:
            print(f"• Error en gráfico pobreza: {e}")

# =============================================================================
# 5. MODELO PREDICTIVO
# =============================================================================

def build_model(df, pobre, config):
    """
    Construir modelo Random Forest para predecir pobreza.
    
    Returns:
        tuple: (model, metrics, feature_names, X_test, y_test, y_pred_proba)
    """
    print("\n--- MODELO PREDICTIVO ---")
    
    if pobre is None:
        print("• No se puede construir modelo sin variable objetivo")
        return None, None, None, None, None, None
    
    # Seleccionar variables predictoras
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    # Filtrar categóricas con pocos niveles
    cat_filtered = []
    for col in cat_cols:
        if df[col].nunique() <= config['MAX_CAT_LEVELS']:
            cat_filtered.append(col)
    
    print(f"• Variables numéricas: {len(num_cols)}")
    print(f"• Variables categóricas (filtradas): {len(cat_filtered)}")
    
    # Preparar datos
    X_cols = num_cols + cat_filtered
    X = df[X_cols].copy()
    y = pobre.copy()
    
    # Eliminar filas con target faltante
    mask = y.notna()
    X = X[mask]
    y = y[mask]
    
    print(f"• Observaciones para modelo: {len(X)}")
    print(f"• Distribución objetivo - Pobres: {y.sum()}, No pobres: {(1-y).sum()}")
    
    # Pipeline de preprocesamiento
    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(drop='first', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, num_cols),
        ('cat', categorical_transformer, cat_filtered)
    ])
    
    # Modelo completo
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(
            n_estimators=300,
            random_state=config['RANDOM_STATE'],
            n_jobs=-1
        ))
    ])
    
    # División train/test estratificada
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=config['RANDOM_STATE']
    )
    
    # Entrenar modelo
    print("• Entrenando Random Forest...")
    model.fit(X_train, y_train)
    
    # Predicciones
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Métricas
    metrics = {
        'roc_auc': roc_auc_score(y_test, y_pred_proba),
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred)
    }
    
    print(f"• ROC-AUC: {metrics['roc_auc']:.3f}")
    print(f"• Accuracy: {metrics['accuracy']:.3f}")
    print(f"• Precision: {metrics['precision']:.3f}")
    print(f"• Recall: {metrics['recall']:.3f}")
    
    # Guardar métricas
    pd.DataFrame([metrics]).to_csv('tab/metrics.csv', index=False)
    print("✓ tab/metrics.csv")
    
    # Obtener nombres de features
    feature_names = (num_cols + 
                    list(model.named_steps['preprocessor']
                         .named_transformers_['cat']
                         .named_steps['onehot']
                         .get_feature_names_out(cat_filtered)))
    
    return model, metrics, feature_names, X_test, y_test, y_pred_proba

def make_model_plots(model, metrics, feature_names, X_test, y_test, y_pred_proba, config):
    """
    Generar gráficos del modelo.
    """
    print("\n--- GRÁFICOS DEL MODELO ---")
    
    if model is None:
        return
    
    # 5.1 Curva ROC
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {metrics["roc_auc"]:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Curva ROC - Predicción de Pobreza')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('fig/roc_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ fig/roc_curve.png")
    
    # 5.2 Importancia de features
    importances = model.named_steps['classifier'].feature_importances_
    feat_imp = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    plt.figure(figsize=(10, 8))
    top_15 = feat_imp.head(15)
    plt.barh(range(len(top_15)), top_15['importance'])
    plt.yticks(range(len(top_15)), top_15['feature'])
    plt.xlabel('Importancia (Gini)')
    plt.title('Top 15 Features - Importancia Random Forest')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('fig/feat_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ fig/feat_importance.png")
    
    # 5.3 SHAP plots (si disponible)
    if SHAP_AVAILABLE:
        try:
            # Preparar datos para SHAP
            X_test_transformed = model.named_steps['preprocessor'].transform(X_test)
            
            # Convertir a dense si es sparse
            if hasattr(X_test_transformed, 'toarray'):
                X_test_transformed = X_test_transformed.toarray()
            
            # Tomar muestra para SHAP (máximo 100 observaciones)
            n_shap = min(100, len(X_test_transformed))
            X_shap = X_test_transformed[:n_shap]
            
            explainer = shap.TreeExplainer(model.named_steps['classifier'])
            shap_values = explainer.shap_values(X_shap)
            
            # Si es clasificación binaria, tomar clase positiva
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
            
            # Summary plot
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values, X_shap, feature_names=feature_names, show=False)
            plt.savefig('fig/shap_summary.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("✓ fig/shap_summary.png")
            
            # Force plot (una observación)
            try:
                plt.figure(figsize=(12, 6))
                shap.force_plot(
                    explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value,
                    shap_values[0],
                    X_shap[0],
                    feature_names=feature_names,
                    matplotlib=True,
                    show=False
                )
                plt.savefig('fig/force_example1.png', dpi=300, bbox_inches='tight')
                plt.close()
                print("✓ fig/force_example1.png")
            except:
                # Crear figura vacía con mensaje
                plt.figure(figsize=(10, 4))
                plt.text(0.5, 0.5, 'SHAP Force Plot no disponible\npara esta configuración',
                        ha='center', va='center', fontsize=14)
                plt.axis('off')
                plt.savefig('fig/force_example1.png', dpi=300, bbox_inches='tight')
                plt.close()
                print("• fig/force_example1.png (placeholder)")
                
        except Exception as e:
            print(f"• Error en SHAP: {e}")
            # Crear figuras placeholder
            for filename in ['fig/shap_summary.png', 'fig/force_example1.png']:
                plt.figure(figsize=(10, 6))
                plt.text(0.5, 0.5, f'SHAP no disponible\nError: {str(e)[:50]}...',
                        ha='center', va='center', fontsize=12)
                plt.axis('off')
                plt.savefig(filename, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"• {filename} (placeholder)")
    else:
        # Crear figuras placeholder si SHAP no está disponible
        for filename in ['fig/shap_summary.png', 'fig/force_example1.png']:
            plt.figure(figsize=(10, 6))
            plt.text(0.5, 0.5, 'SHAP no está instalado\npip install shap',
                    ha='center', va='center', fontsize=14)
            plt.axis('off')
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"• {filename} (placeholder)")

# =============================================================================
# 6. FUNCIÓN PRINCIPAL
# =============================================================================

def main():
    """
    Función principal que ejecuta todo el pipeline.
    """
    print("=" * 60)
    print("GENERADOR DE ASSETS PARA INFORME ENAHO 2022")
    print("=" * 60)
    
    # Crear directorios
    create_dirs()
    
    # Cargar datos
    df_raw = load_data(CONFIG['DATA_FILE'])
    
    # Limpiar datos
    df, num_cols, cat_cols = clean_data(df_raw)
    
    # Validar configuración
    config = validate_config(df, CONFIG)
    
    # Detectar outliers
    outlier_flags, outlier_counts, limits = detect_outliers(df, num_cols)
    
    # Winsorizar
    df_winsor = winsorize_data(df, limits, num_cols)
    
    # Crear indicador de pobreza
    pobre = create_poverty_indicator(df, config)
    
    # Generar tablas
    desc_raw, desc_winsor = make_tables(df, df_winsor, outlier_counts, config, num_cols)
    
    # Generar figuras básicas
    make_plots(df, df_winsor, outlier_counts, config, pobre)
    
    # Modelo predictivo
    model, metrics, feature_names, X_test, y_test, y_pred_proba = build_model(df, pobre, config)
    
    # Gráficos del modelo
    make_model_plots(model, metrics, feature_names, X_test, y_test, y_pred_proba, config)
    
    # Guardar datasets
    print("\n--- GUARDANDO DATASETS ---")
    df_winsor.to_csv('enaho_winsor.csv', index=False)
    print("✓ enaho_winsor.csv")
    
    # Dataset con flags de outliers
    df_flags = df.copy()
    for col in outlier_flags.columns:
        df_flags[f'{col}_outlier'] = outlier_flags[col].astype(int)
    df_flags.to_csv('enaho_flags.csv', index=False)
    print("✓ enaho_flags.csv")
    
    # Resumen final
    print("\n" + "=" * 60)
    print("RESUMEN PARA INFORME LaTeX:")
    print("=" * 60)
    print(f"• Total observaciones: {len(df):,}")
    print(f"• Variables numéricas: {len(num_cols)}")
    print(f"• Variables categóricas: {len(cat_cols)}")
    print(f"• Total outliers detectados: {outlier_flags.sum().sum():,}")
    print(f"• Variable con más outliers: {outlier_counts.index[0]} ({outlier_counts.iloc[0]} outliers)")
    
    if pobre is not None:
        print(f"• Incidencia de pobreza: {pobre.mean():.1%} ({pobre.sum():,} de {len(pobre):,})")
        
        if config['COL_INGRESO'] and config['COL_INGRESO'] in df.columns:
            ing_pobre = df[pobre == 1][config['COL_INGRESO']].mean()
            ing_nopobre = df[pobre == 0][config['COL_INGRESO']].mean()
            print(f"• Ingreso promedio pobres: {ing_pobre:.2f}")
            print(f"• Ingreso promedio no pobres: {ing_nopobre:.2f}")
    
    if metrics:
        print(f"• Modelo Random Forest - AUC: {metrics['roc_auc']:.3f}")
        print(f"• Modelo Random Forest - Accuracy: {metrics['accuracy']:.3f}")
    
    print("\n✓ Todos los archivos generados en fig/ y tab/")
    print("✓ Proceso completado exitosamente")

if __name__ == "__main__":
    main()
