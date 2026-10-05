

# ------------------------------------------------------------------------

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
