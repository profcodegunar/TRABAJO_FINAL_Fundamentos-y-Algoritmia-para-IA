

# ------------------------------------------------------------------------

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
