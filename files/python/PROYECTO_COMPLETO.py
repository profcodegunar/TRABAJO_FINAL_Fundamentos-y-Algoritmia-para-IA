# ===== 01_LEEMOS_Y_CONOCEMOS_EL_DATASET =====

# Importamos Pandas porque nos permite trabajar con datos en forma de tabla.
import pandas as pd

# Indicamos la ruta del archivo CSV.
ruta = "../dataset/rendimiento_estudiantil_senati.csv"

# Leemos el archivo y lo guardamos en una variable llamada df.
df = pd.read_csv(ruta)

# Mostramos las primeras 10 filas para conocer cómo están organizados los datos.
df.head(10)

# Ahora revisamos cuántas filas y columnas tiene el dataset.
filas, columnas = df.shape
print(f"El dataset contiene {filas} registros y {columnas} columnas.")

# También mostramos los nombres de las columnas.
print("\nColumnas disponibles:")
for columna in df.columns:
    print("-", columna)

# info() nos muestra el tipo de cada columna y cuántos datos no nulos existen.
df.info()

# isnull() identifica los valores vacíos y sum() los cuenta por columna.
nulos = df.isnull().sum()

# Mostramos solamente las columnas que realmente tienen datos faltantes.
nulos[nulos > 0]

# Resumen estadístico de las variables numéricas.
df.describe().round(2)

# ===== 02_LIMPIEZA_Y_PREPROCESAMIENTO =====

import pandas as pd

# Volvemos a leer el dataset para que este cuaderno pueda ejecutarse de manera independiente.
df = pd.read_csv("../dataset/rendimiento_estudiantil_senati.csv")

# Revisamos nuevamente los valores faltantes antes de preparar los datos.
df.isnull().sum()[df.isnull().sum() > 0]

# X contiene las variables de entrada.
X = df.drop(columns=["promedio_final", "rendimiento_categoria"])

# y contiene la variable objetivo.
y = df["promedio_final"]

print("Dimensión de X:", X.shape)
print("Dimensión de y:", y.shape)

columnas_numericas = [
    "edad", "horas_estudio_semana", "asistencia_pct",
    "tareas_completadas_pct", "participacion_pct",
    "uso_plataforma_horas", "tutorias_mes",
    "promedio_prev", "examen_parcial", "proyecto_final"
]

columnas_categoricas = ["genero", "internet_calidad"]

print("Variables numéricas:", len(columnas_numericas))
print("Variables categóricas:", len(columnas_categoricas))

# Importamos las herramientas de Scikit-learn que vamos a utilizar.
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Pipeline para las variables numéricas.
proceso_numerico = Pipeline(steps=[
    ("imputar_nulos", SimpleImputer(strategy="median")),
    ("normalizar", StandardScaler())
])

# Pipeline para las variables categóricas.
proceso_categorico = Pipeline(steps=[
    ("imputar_nulos", SimpleImputer(strategy="most_frequent")),
    ("codificar", OneHotEncoder(handle_unknown="ignore"))
])

# Unimos ambos procesos en un solo transformador.
preprocesador = ColumnTransformer(transformers=[
    ("numericas", proceso_numerico, columnas_numericas),
    ("categoricas", proceso_categorico, columnas_categoricas)
])

print("Preprocesador construido correctamente.")

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42
)

print("Registros para entrenamiento:", len(X_train))
print("Registros para prueba:", len(X_test))

# Aplicamos el preprocesamiento a los datos de entrenamiento.
X_train_procesado = preprocesador.fit_transform(X_train)

# En los datos de prueba solamente transformamos; no volvemos a aprender parámetros.
X_test_procesado = preprocesador.transform(X_test)

print("Forma del conjunto de entrenamiento procesado:", X_train_procesado.shape)
print("Forma del conjunto de prueba procesado:", X_test_procesado.shape)

# ===== 03_ALGEBRA_LINEAL_CON_NUMPY =====

import numpy as np

# Creamos dos vectores sencillos para practicar operaciones básicas.
vector_a = np.array([12, 15, 18, 20])
vector_b = np.array([2, 4, 6, 8])

print("Vector A:", vector_a)
print("Vector B:", vector_b)

print("Suma:", vector_a + vector_b)
print("Resta:", vector_a - vector_b)

producto_punto = np.dot(vector_a, vector_b)
print("Producto punto:", producto_punto)

norma_a = np.linalg.norm(vector_a)
print("Norma del vector A:", round(norma_a, 4))

matriz_ejemplo = np.array([
    [12, 85, 14],
    [9,  78, 12],
    [15, 92, 16]
])

print("Matriz de ejemplo:")
print(matriz_ejemplo)
print("\nFilas y columnas:", matriz_ejemplo.shape)

# Matriz de coeficientes.
A = np.array([
    [2, 1, -1],
    [-3, -1, 2],
    [-2, 1, 2]
])

# Vector de resultados.
b = np.array([8, -11, -3])

# Resolvemos el sistema Ax = b.
solucion = np.linalg.solve(A, b)

print("Solución [x, y, z]:", solucion)

# ===== 04_ESTADISTICA_Y_OUTLIERS =====

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

# ===== 05_GRAFICOS_E_INTERPRETACION =====

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Leemos el dataset.
df = pd.read_csv("../dataset/rendimiento_estudiantil_senati.csv")

# Nos aseguramos de que la carpeta de resultados exista.
os.makedirs("../resultados", exist_ok=True)

sns.set_theme(style="whitegrid")

plt.figure(figsize=(9, 5))
sns.histplot(df["promedio_final"], kde=True, color="royalblue")
plt.title("Distribución del promedio final")
plt.xlabel("Promedio final")
plt.ylabel("Frecuencia")
plt.tight_layout()
plt.savefig("../resultados/histograma_promedio_final.png", dpi=150)
plt.show()

plt.figure(figsize=(9, 5))
sns.scatterplot(
    data=df,
    x="horas_estudio_semana",
    y="promedio_final",
    hue="rendimiento_categoria",
    palette="deep"
)
plt.title("Horas de estudio por semana vs. promedio final")
plt.xlabel("Horas de estudio por semana")
plt.ylabel("Promedio final")
plt.tight_layout()
plt.savefig("../resultados/dispersion_estudio_vs_promedio.png", dpi=150)
plt.show()

# ===== 06_MODELO_MACHINE_LEARNING =====

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Leemos el dataset.
df = pd.read_csv("../dataset/rendimiento_estudiantil_senati.csv")

# Separamos variables de entrada y variable objetivo.
X = df.drop(columns=["promedio_final", "rendimiento_categoria"])
y = df["promedio_final"]

columnas_numericas = [
    "edad", "horas_estudio_semana", "asistencia_pct",
    "tareas_completadas_pct", "participacion_pct",
    "uso_plataforma_horas", "tutorias_mes",
    "promedio_prev", "examen_parcial", "proyecto_final"
]

columnas_categoricas = ["genero", "internet_calidad"]

proceso_numerico = Pipeline(steps=[
    ("imputar", SimpleImputer(strategy="median")),
    ("normalizar", StandardScaler())
])

proceso_categorico = Pipeline(steps=[
    ("imputar", SimpleImputer(strategy="most_frequent")),
    ("codificar", OneHotEncoder(handle_unknown="ignore"))
])

preprocesador = ColumnTransformer(transformers=[
    ("num", proceso_numerico, columnas_numericas),
    ("cat", proceso_categorico, columnas_categoricas)
])

modelo = Ridge(alpha=1.0)

pipeline = Pipeline(steps=[
    ("preprocesamiento", preprocesador),
    ("modelo", modelo)
])

print("Pipeline listo para entrenar.")

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42
)

pipeline.fit(X_train, y_train)
predicciones = pipeline.predict(X_test)

print("Modelo entrenado correctamente.")

mae = mean_absolute_error(y_test, predicciones)
rmse = np.sqrt(mean_squared_error(y_test, predicciones))
r2 = r2_score(y_test, predicciones)

print(f"MAE : {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²  : {r2:.4f}")

comparacion = pd.DataFrame({
    "Promedio real": y_test.values,
    "Promedio estimado": np.round(predicciones, 2)
})

comparacion.head(15)

# Guardamos las métricas y predicciones como evidencia del proyecto.
metricas = pd.DataFrame({
    "metrica": ["MAE", "RMSE", "R2"],
    "valor": [round(mae, 4), round(rmse, 4), round(r2, 4)]
})

metricas.to_csv("../resultados/metricas_modelo.csv", index=False, encoding="utf-8-sig")
comparacion.to_csv("../resultados/predicciones_prueba.csv", index=False, encoding="utf-8-sig")

print("Resultados guardados correctamente.")
