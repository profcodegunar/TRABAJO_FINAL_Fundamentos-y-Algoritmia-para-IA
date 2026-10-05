

# ------------------------------------------------------------------------

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
