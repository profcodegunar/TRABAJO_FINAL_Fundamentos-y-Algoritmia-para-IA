

# ------------------------------------------------------------------------

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
