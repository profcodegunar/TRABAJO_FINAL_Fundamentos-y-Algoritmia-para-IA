

# ------------------------------------------------------------------------

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
