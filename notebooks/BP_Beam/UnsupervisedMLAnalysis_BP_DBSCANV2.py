import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

## Leer directorio del script
script_dir = os.path.dirname(os.path.abspath(__file__))
excelpath = os.path.join(script_dir, 'BP - V33_EA_ch1.xlsx')
BP_ch1 = pd.read_excel(excelpath)
data = BP_ch1

# Filtrar filas donde 'load_level' contiene 'RISE'
data_rise = data[data['load level'].str.contains('RISE', case=False, na=False)]

# Seleccionar solo columnas relevantes para clustering
columns_to_exclude = ['CH', 'THR']
data_rise_filtered = data_rise.drop(columns=columns_to_exclude, errors='ignore').loc[:, 'RISE':'ABS-ENERGY']
data_rise_filtered = data_rise_filtered.select_dtypes(include=[np.number])

# Estandarizar los datos
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data_rise_filtered)

# Generar gráfico de codo para elegir eps
def plot_knee_curve(data):
    neighbors = NearestNeighbors(n_neighbors=5)
    neighbors_fit = neighbors.fit(data)
    distances, indices = neighbors_fit.kneighbors(data)

    # Ordenar las distancias de menor a mayor
    distances = np.sort(distances[:, 4])  # Usamos el 5º vecino más cercano

    # Graficar las distancias ordenadas
    plt.figure(figsize=(8, 5))
    plt.plot(distances, linewidth=2)
    plt.title('Gráfico de Codo para Determinar eps')
    plt.xlabel('Puntos ordenados')
    plt.ylim(0, 20)
    plt.ylabel('Distancia al 5º vecino más cercano')
    plt.grid()
    plt.show()

# Llamar a la función para graficar el codo
plot_knee_curve(scaled_data)

# Función para realizar clustering por nivel de carga y ciclo
def perform_clustering(data, eps=3):
    clustering_results = {}
    cluster_analysis = {}
    cluster_summary = []  # Para el Excel adicional

    for load_level in data['load level'].unique():
        subset = data[data['load level'] == load_level].copy()

        # Mostrar la cantidad de filas en el nivel actual
        print(f"{load_level}: {len(subset)} filas")
        
        # Seleccionar solo columnas relevantes para clustering
        relevant_columns = subset.drop(columns=['load level'], errors='ignore').select_dtypes(include=[np.number])

        if relevant_columns.empty:
            print(f"No numeric data found for {load_level}")
            continue

        # Estandarizar los datos
        scaled_data = scaler.fit_transform(relevant_columns)

        # Definir un valor dinámico para min_samples basado en el tamaño del nivel
        min_samples = max(2, min(5, len(subset)))

        # Aplicar DBSCAN
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(scaled_data)
        
        # Agregar los resultados de clustering
        subset.loc[:, 'cluster'] = db.labels_
        clustering_results[load_level] = subset

        # Mostrar el número de clusters
        n_clusters = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
        print(f"{load_level}: {n_clusters} cluster(s) (min_samples={min_samples})")

        # Calcular el porcentaje de cada cluster
        cluster_counts = subset['cluster'].value_counts(normalize=True) * 100
        print(f"Porcentaje de puntos por cluster en {load_level}:")
        print(cluster_counts)

        # Análisis de parámetros por cluster
        numeric_columns = subset.drop(columns=['load level'], errors='ignore').select_dtypes(include=[np.number])
        numeric_columns.fillna(numeric_columns.mean(), inplace=True)

        cluster_stats = numeric_columns.groupby(subset['cluster']).agg(['mean', 'std', 'median', 'min', 'max'])
        cluster_analysis[load_level] = cluster_stats

        # Agregar al resumen para Excel adicional
        for cluster_id, stats in cluster_stats.iterrows():
            row = {
                'load level': load_level,
                'cluster': cluster_id
            }
            for col in stats.index:
                row[col] = stats[col]
            cluster_summary.append(row)

        # Mostrar análisis para cada cluster
        print(f"Análisis de parámetros por cluster en {load_level}:")
        print(cluster_stats)

    # Crear DataFrame con el resumen
    cluster_summary_df = pd.DataFrame(cluster_summary)
    return clustering_results, cluster_analysis, cluster_summary_df

# Ejecutar el clustering
clustering_results, cluster_analysis, cluster_summary_df = perform_clustering(data_rise)

# Guardar los resultados por nivel de carga
for load_level, df in clustering_results.items():
    output_file = os.path.join(script_dir, f"clustering_results_{load_level}.xlsx")
    df.to_excel(output_file, index=False)
    print(f"Resultados para {load_level} guardados en {output_file}")

# Guardar análisis comparativo en un archivo Excel
comparative_analysis_path = os.path.join(script_dir, "comparative_cluster_analysis.xlsx")
with pd.ExcelWriter(comparative_analysis_path) as writer:
    for load_level, stats in cluster_analysis.items():
        stats.to_excel(writer, sheet_name=load_level)

# Guardar el resumen en un Excel adicional
summary_path = os.path.join(script_dir, "cluster_summary.xlsx")
cluster_summary_df.to_excel(summary_path, index=False)
print(f"Resumen de clusters guardado en {summary_path}")

# Verificar si los parámetros medios coinciden entre clusters
def compare_cluster_means(cluster_analysis):
    mean_comparisons = {}
    for load_level, stats in cluster_analysis.items():
        means = stats.xs('mean', axis=1, level=1)
        for col in means.columns:
            mean_comparisons[(load_level, col)] = means[col].unique()

    # Mostrar parámetros con coincidencias entre clusters
    for (load_level, col), unique_means in mean_comparisons.items():
        print(f"{load_level} - {col}: {len(unique_means)} valores medios únicos")

compare_cluster_means(cluster_analysis)
