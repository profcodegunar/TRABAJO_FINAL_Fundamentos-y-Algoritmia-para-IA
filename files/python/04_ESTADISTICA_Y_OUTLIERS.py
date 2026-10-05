

# ------------------------------------------------------------------------

import pandas as pd
import numpy as np

# Leemos nuevamente el dataset.
df = pd.read_csv("../dataset/rendimiento_estudiantil_senati.csv")

# Seleccionamos la variable que queremos analizar.
serie = df["promedio_final"]

serie.head()

media = serie.mean()
mediana = serie.median()

print(f"Media del promedio final: {media:.2f}")
print(f"Mediana del promedio final: {mediana:.2f}")

varianza_numpy = np.var(serie, ddof=1)
desviacion_numpy = np.std(serie, ddof=1)

print(f"Varianza con NumPy: {varianza_numpy:.2f}")
print(f"Desviación estándar con NumPy: {desviacion_numpy:.2f}")

# Calculamos la media manualmente.
n = len(serie)
media_manual = sum(serie) / n

# Calculamos la varianza muestral manualmente.
varianza_manual = sum((valor - media_manual) ** 2 for valor in serie) / (n - 1)

# La desviación estándar es la raíz cuadrada de la varianza.
desviacion_manual = varianza_manual ** 0.5

print(f"Media manual: {media_manual:.2f}")
print(f"Varianza manual: {varianza_manual:.2f}")
print(f"Desviación estándar manual: {desviacion_manual:.2f}")

limite_inferior = media - 2 * desviacion_numpy
limite_superior = media + 2 * desviacion_numpy

outliers = df[
    (df["promedio_final"] < limite_inferior) |
    (df["promedio_final"] > limite_superior)
]

print(f"Límite inferior: {limite_inferior:.2f}")
print(f"Límite superior: {limite_superior:.2f}")
print(f"Cantidad de valores atípicos detectados: {len(outliers)}")

outliers[["id_estudiante", "promedio_final", "rendimiento_categoria"]]

# Guardamos los registros atípicos para utilizarlos como evidencia del proyecto.
outliers.to_csv("../resultados/outliers_promedio_final.csv", index=False, encoding="utf-8-sig")

print("Archivo guardado en: resultados/outliers_promedio_final.csv")
