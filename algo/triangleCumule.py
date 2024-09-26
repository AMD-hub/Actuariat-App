import numpy as np
import pandas as pd

def mettre_nan_sous_deuxieme_diagonale(triangle):
    n = triangle.shape[1]
    for i in range(n):
        for j in range(n):
            if j > n - i - 1:
                triangle.iat[i, j] = np.nan

def execute(table):
    triangle = table.copy()

    # Garder toutes les colonnes sauf la première
    triangle = triangle.iloc[:, 1:]

    # Transformer les valeurs sous la deuxième diagonale en NaN
    mettre_nan_sous_deuxieme_diagonale(triangle)

    # Convertir le triangle en tableau NumPy
    triangle_numpy = triangle.values

    # Calculer la somme cumulée en ignorant les valeurs NaN
    for i in range(triangle_numpy.shape[0]):
        cum_sum = 0
        for j in range(triangle_numpy.shape[1]):
            if not np.isnan(triangle_numpy[i, j]):
                cum_sum += triangle_numpy[i, j]
                triangle_numpy[i, j] = cum_sum
            else:
                cum_sum = np.nan

    # Créer un DataFrame à partir du nouveau triangle cumulé
    df_nouveau_triangle = pd.DataFrame(triangle_numpy)

    # Rajouter la première colonne du DataFrame initial
    df_nouveau_triangle.insert(0, "Années", table.iloc[:, 0])

    return df_nouveau_triangle
