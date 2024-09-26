import numpy as np
import pandas as pd
import numpy as np

def count_lower_triangle_elements(df):
    """
    Compte les éléments dans le triangle inférieur d'un DataFrame carré, incluant la diagonale.
    
    Args:
    df (pd.DataFrame): Un DataFrame carré de pandas.
    
    Returns:
    int: Nombre d'éléments dans le triangle inférieur du DataFrame.
    """
    if df.shape[0] != df.shape[1]:
        raise ValueError("Le DataFrame n'est pas carré.")
    
    count = 0
    # Convertir le DataFrame en matrice NumPy pour un traitement plus facile
    matrix = df.to_numpy()
    n = matrix.shape[0]  # Obtenir le nombre de lignes (ou colonnes)
    
    for i in range(n):
        for j in range(i + 1):
            count += 1
    
    return count

def construire_triangle_residus(residus, taille):
    # Créer un tableau de zéros de taille spécifiée
    triangle_residus = np.zeros((taille, taille))
    
    # Calculer les indices de départ pour chaque ligne dans le triangle
    index_debut = 0
    for i in range(taille):
        # Calculer le nombre d'éléments à prendre pour la ligne actuelle
        nombre_elements = taille - i
        
        # Calculer l'index de fin pour les résidus simulés
        index_fin = index_debut + nombre_elements
        
        # Remplir la ligne avec les résidus appropriés et les zéros nécessaires
        triangle_residus[i, :nombre_elements] = residus[index_debut:index_fin]
        
        # Mettre à jour l'index de début pour la prochaine ligne
        index_debut = index_fin

    return triangle_residus

def execute(triangle_cumule, triangle_charge, taux):
    triangle = triangle_cumule.copy()

    # Utiliser la première colonne comme index sans spécifier son nom
    triangle.set_index(triangle.columns[0], inplace=True)
    # Définir l'index directement sans utiliser la colonne comme référence après

    # Calculer les facteurs de développement
    n = triangle.shape[1]  # Nombre de colonnes pour les facteurs de développement
    factors = []

    for i in range(1, n):
        numerator = triangle.iloc[:n-i, i].sum()
        denominator = triangle.iloc[:n-i, i-1].sum()
        if denominator == 0:
            factor = 0  # Gérer si le dénominateur est zéro
        else:
            factor = numerator / denominator
        factors.append(factor)


    increments_triangle = triangle.diff(axis=1)
    increments_triangle.iloc[:, 0] = triangle.iloc[:, 0]  # Remplacer NaN avec les valeurs cumulées originales

    increments_triangle

    # Conversion du triangle de règlement cumulé en numpy array pour le traitement
    triangle_numpy = triangle.to_numpy()

    # Initialisation du triangle estimé avec des NaN pour préparation
    estimated_triangle = np.full(triangle_numpy.shape, np.nan)
    # Remplissage de la deuxième diagonale du triangle estimé avec les valeurs de la deuxième diagonale du triangle réel
    n = triangle.shape[1]
    for i in range(n):
        estimated_triangle[i, n-i-1] = triangle_numpy[i,n-i-1]



    # Calcul des estimations \hat{C}_{i,j}
    for i in range(n):
        for j in range(n-i-1, 0, -1):  # Commencer par la fin de la ligne et remonter.
            # Pas besoin de vérifier si j == 0 car notre boucle commence à partir de n-i-1.
            estimated_triangle[i, j-1] = estimated_triangle[i, j] / factors[j-1]


    # Convertir le tableau numpy estimé en DataFrame pour utiliser la méthode diff()
    estimated_triangle_df = pd.DataFrame(estimated_triangle)

    # Calculer les différences pour obtenir les incréments
    increments_estimated_triangle_df = estimated_triangle_df.diff(axis=1)

    # Puisque la différenciation place des NaN dans la première colonne, remplaçons-les
    # par les valeurs cumulées originales (ce qui correspond aux valeurs de la première colonne du triangle estimé)
    increments_estimated_triangle_df.iloc[:, 0] = estimated_triangle_df.iloc[:, 0]

    # Initialisation d'un nouveau DataFrame pour les différences
    difference_triangle = np.zeros_like(increments_triangle)

    # Boucle pour calculer la différence entre les éléments des deux triangles
    for i in range (increments_triangle.shape[0]):
        for j in range (increments_triangle.shape[1]):
            
                # Accès par position et calcul de la différence
                difference_triangle[i, j] = (increments_triangle.iloc[i, j] - increments_estimated_triangle_df.iloc[i, j])/np.sqrt(increments_estimated_triangle_df.iloc[i, j])
    difference_triangle
    # Afficher les résidus
    résidus= pd.DataFrame(difference_triangle)
    # Calculer la somme des carrés des éléments de chaque colonne du DataFrame résidus
    somme_carres_colonnes = (résidus ** 2).sum(axis=0)
    # Calculer N, P, et h
    N = résidus.shape[0] * (résidus.shape[0] + 1) / 2
    P = 2 * résidus.shape[0] - 1
    h = résidus.shape[1]

    # Calculer l'ajustement de l'écart-type pour chaque colonne de la somme des carrés des éléments
    ajustements = [(N / (N - P)) * (1 / (h - j + 1)) for j in range(1, somme_carres_colonnes.shape[0]+1)]

    # Ajuster l'écart-type en multipliant chaque élément de la somme des carrés des colonnes par l'ajustement correspondant
    somme_carres_colonnes_adj = somme_carres_colonnes * ajustements


    # Initialiser un DataFrame pour stocker les résidus ajustés
    résidus_ajustés = pd.DataFrame()
    # Calculer les résidus ajustés
    for j in range(résidus.shape[1]):
        if somme_carres_colonnes_adj[j] == 0:
            # Si la somme des carrés ajustée est zéro, éviter la division par zéro en mettant les résidus ajustés à zéro
            résidus_ajustés[ j] = 0
        else: 
            ajustement = np.sqrt(N / (N - P)) / np.sqrt(somme_carres_colonnes_adj[j])
            résidus_ajustés[j] = résidus[j] * ajustement
    # Calculer les résidus ajustés centrés
    résidus_ajustés_centrés = résidus_ajustés - résidus_ajustés.mean()

    df = pd.DataFrame(résidus_ajustés_centrés)

    # Convertir en NumPy array et retirer les NaN
    # Initialisation du meilleur estimateur avec des zéros

    # Initialiser une liste pour stocker les valeurs du triangle inférieur
    lower_triangle_values = []

    # Parcourir le DataFrame pour extraire le triangle inférieur
    for i in range(résidus_ajustés_centrés.shape[0]):  # Parcours des lignes
        for j in range(résidus_ajustés_centrés.shape[1]):  # Parcours des colonnes
            if i+j <= len(résidus_ajustés_centrés)-1:  # Condition pour le triangle inférieur, incluant la diagonale
                lower_triangle_values.append(résidus_ajustés_centrés.iloc[i, j])

    # Convertir la liste en un tableau NumPy unidimensionnel
    Residus_de_Pearson = np.array(lower_triangle_values)

    # Initialize the best estimate
    Best_Estimate = np.zeros(200)

    best = []
    bestA=[]
    # Run the simulation 20,000 times
    for k in range( 200):
        
        # Sample 66 residuals with replacement
        Residus_simules = np.random.choice(Residus_de_Pearson, count_lower_triangle_elements(df), replace=True)
        
        # Create an len(résidus_ajustés_centrés)xlen(résidus_ajustés_centrés) matrix of zeros
        Triangle_residus = np.zeros((len(résidus_ajustés_centrés), len(résidus_ajustés_centrés)))
        
        # Fill each row in the triangle, appending the necessary number of zeros
        Triangle_residus = construire_triangle_residus(Residus_simules, len(résidus_ajustés_centrés))
        # Convert to DataFrame and optionally print
        df_triangle_residus = pd.DataFrame(Triangle_residus)
        

        # Création d'une matrice len(résidus_ajustés_centrés)xlen(résidus_ajustés_centrés) initialisée à zéro
        Triangle_D_prime = np.zeros((len(résidus_ajustés_centrés), len(résidus_ajustés_centrés)))
        Phi =somme_carres_colonnes_adj
        increments = pd.DataFrame(increments_estimated_triangle_df)
        Triangle_Dth= increments

        # Convertir Triangle_Dth en DataFrame si ce n'est pas déjà le cas
        Triangle_Dth = pd.DataFrame(Triangle_Dth)

        # Initialiser Triangle_D_prime comme un DataFrame de zéros de même taille que Triangle_Dth
        Triangle_D_prime = pd.DataFrame(np.zeros_like(Triangle_Dth), columns=Triangle_Dth.columns, index=Triangle_Dth.index)

        # Conversion de Phi en array numpy pour faciliter l'accès par index
        Phi = np.array(Phi)

        # Reconstruction du triangle de paiement décumulé
        for i in range(len(résidus_ajustés_centrés)):
            for j in range(len(résidus_ajustés_centrés)):
            # Utilisation de iloc pour accéder aux valeurs
                Triangle_D_prime.iloc[i, j] = Triangle_Dth.iloc[i, j] + np.sqrt(Triangle_Dth.iloc[i, j]) * np.sqrt(Phi[j]) * Triangle_residus[i, j]


        # Vérifiez d'abord le type de Triangle_D_prime et convertissez-le en numpy array si nécessaire
        if isinstance(Triangle_D_prime, pd.DataFrame):
            Triangle_D_prime = Triangle_D_prime.values

        # Initialiser un triangle de paiement cumulé de taille len(résidus_ajustés_centrés)xlen(résidus_ajustés_centrés) avec des zéros
        Triangle_C_prime = np.zeros((len(résidus_ajustés_centrés), len(résidus_ajustés_centrés)))

        # Construction du triangle de paiement cumulé

        for i in range(len(résidus_ajustés_centrés)):  # Les indices en Python commencent à 0, mais nous utiliserons i+1 pour aligner avec votre logique R (1 à len(résidus_ajustés_centrés))
            for j in range(len(résidus_ajustés_centrés) - i):  # Ajuster la limite de la boucle interne pour correspondre à la structure triangulaire
                if j == 0:
                    Triangle_C_prime[i, j] = Triangle_D_prime[i, j]
                else:
                    Triangle_C_prime[i, j] = Triangle_C_prime[i, j-1] + Triangle_D_prime[i, j]

        # Si vous souhaitez convertir le numpy array en DataFrame pour une meilleure manipulation ou visualisation :
        Triangle_C_prime_df = pd.DataFrame(Triangle_C_prime, columns=[f' {i+1}' for i in range(len(résidus_ajustés_centrés))], index=[f'{i+1}' for i in range(len(résidus_ajustés_centrés))])

    
    
        facteur_prime = []

        for i in range(1, len(résidus_ajustés_centrés)):
            numerateur = Triangle_C_prime[:len(résidus_ajustés_centrés)-i, i].sum()
            denominateur = Triangle_C_prime[:len(résidus_ajustés_centrés)-i, i-1].sum()
            if denominateur == 0:
                facteur = 0  # Gérer si le dénominateur est zéro
            else:
                facteur = numerateur / denominateur
            facteur_prime.append(facteur)
        
        # Remplir le triangle
        for i in range(1, len(résidus_ajustés_centrés)):
            for j in range(0, len(résidus_ajustés_centrés)):
                if j >= i:
                    Triangle_C_prime[-i, j] = Triangle_C_prime[-i, j-1] * facteur_prime[j-1]
        Triangle_C_prime_rempli=pd.DataFrame(Triangle_C_prime)
    
        

        # Supposons que Triangle_C_prime est déjà défini comme un DataFrame ou une matrice numpy

        # Vérifiez d'abord le type de Triangle_C_prime et convertissez-le en numpy array si nécessaire
        if isinstance(Triangle_C_prime_rempli, pd.DataFrame):
            Triangle_C_prime_rempli = Triangle_C_prime_rempli.values

        # Initialiser le pseudo triangle décumulé de taille len(résidus_ajustés_centrés)xlen(résidus_ajustés_centrés) avec des zéros
        Pseudo_Triangle_D_prime = np.zeros((len(résidus_ajustés_centrés), len(résidus_ajustés_centrés)))

        # Construction du pseudo triangle décumulé
        for i in range(len(résidus_ajustés_centrés)):  # Les indices en Python commencent à 0, ajuster selon besoin
            for j in range(len(résidus_ajustés_centrés)):  # Parcourir toutes les colonnes
                if j == 0:
                    Pseudo_Triangle_D_prime[i, j] = Triangle_C_prime_rempli[i, j]
                else:
                    Pseudo_Triangle_D_prime[i, j] = Triangle_C_prime_rempli[i, j] - Triangle_C_prime_rempli[i, j-1]

        # Si vous souhaitez convertir le numpy array en DataFrame pour une meilleure manipulation ou visualisation :
        Pseudo_Triangle_D_prime_df = pd.DataFrame(Pseudo_Triangle_D_prime, columns=[f' {j+1}' for j in range(len(résidus_ajustés_centrés))], index=[f' {i+1}' for i in range(len(résidus_ajustés_centrés))])
        Triangle_C_prime_rempli_df= pd.DataFrame(Triangle_C_prime_rempli, columns=[f' {j+1}' for j in range(len(résidus_ajustés_centrés))], index=[f' {i+1}' for i in range(len(résidus_ajustés_centrés))])
        triangle_reglements = Triangle_C_prime_rempli_df

        # Sélectionner la première colonne
        premiere_colonne = triangle_charge.iloc[:, 0]

        # Créer un nouveau DataFrame en insérant cette colonne au début
        # Utilisation de 'insert' pour ajouter au début (position 0)
        triangle_reglements.insert(0, 'nouvelle_colonne', premiere_colonne)
        # Affiche les premières lignes pour vérification

        n_charge = len(triangle_charge)
        n_reglements = len(triangle_reglements)

        triangle_values_charge = []
        triangle_values_reglements = []

        # Extraire la seconde diagonale des triangles
        for i in range(n_charge):
            for j in range(n_charge + 1):
                if i + j == n_charge:
                    triangle_values_charge.append(triangle_charge.iloc[i, j])
        for i in range(n_reglements):
            for j in range(n_reglements + 1):
                if i + j == n_reglements:
                    triangle_values_reglements.append(triangle_reglements.iloc[i, j])

        # Dernières colonnes pour les deux triangles
        last_col_charge = triangle_charge.iloc[:, -1].tolist()
        last_col_reglements = triangle_reglements.iloc[:, -1].tolist()

        # Calculer les colonnes pour BE_charge
        ibnr_charge = [last_col_charge[i] - triangle_values_charge[i] for i in range(len(triangle_values_charge))]
        psap_dd_charge = [last_col_charge[i] - triangle_values_reglements[i] - ibnr_charge[i] for i in range(len(last_col_charge))]
        be_charge_col = [psap_dd_charge[i] + ibnr_charge[i] for i in range(len(psap_dd_charge))]
        # Calculer les colonnes pour BE_reglement
        ibnr_reglements = [last_col_reglements[i] - triangle_values_reglements[i] - psap_dd_charge[i] for i in range(len(triangle_values_reglements))]
        be_reglements_col = [psap_dd_charge[i] + ibnr_reglements[i] for i in range(len(psap_dd_charge))]

        # Création des tableaux de sortie
        be_charge = pd.DataFrame({
        'Année_de_survenance': triangle_charge.iloc[:, 0],
        'PSAP DD': psap_dd_charge,
        'Charge ultime': last_col_charge,
        'Dernière charge': triangle_values_charge,
        'IBNR': ibnr_charge,
        'BE': be_charge_col
        })

        be_reglement = pd.DataFrame({
        'Année_de_survenance': triangle_charge.iloc[:, 0],
        'PSAP DD': psap_dd_charge,
        'Paiements à l\'ultime': last_col_reglements,
        'Paiements derniers': triangle_values_reglements,
        'IBNR': ibnr_reglements,
        'BE': be_reglements_col
        })
        # Ajouter une ligne pour les sommes des colonnes
        be_charge.loc['Total'] = be_charge.sum()
        be_reglement.loc['Total'] = be_reglement.sum()

        # Charger les données depuis Excel
        df_BE = be_reglement
        matrice_A = triangle_reglements
        df_charge = be_charge

        # Extraire la colonne 'BE' du DataFrame
        B = df_BE['BE'].values
        D = df_charge['BE'].values
        taux = pd.DataFrame(taux, columns=['taux'])['taux'].values

        # Initialiser C, E, X avec les mêmes dimensions que matrice_A
        C = np.zeros_like(matrice_A)
        E = np.zeros_like(matrice_A)
        X = np.zeros_like(matrice_A)

        matrice_A_modifie = matrice_A.iloc[:, 1:]

        # Calculer les valeurs de Cij pour i > 0 et j > 0
        for i in range(matrice_A_modifie.shape[0]):
            for j in range(matrice_A_modifie.shape[1]):
                if B[i] == 0 or i + j <= matrice_A_modifie.shape[0] - 1:
                    C[i, j] = np.nan
                else:
                    C[i, j] = (matrice_A_modifie.iloc[i, j] - matrice_A_modifie.iloc[i, j-1]) / B[i]

        # Vérification de la somme des lignes de C et ajustement si nécessaire
        for i in range(C.shape[0]):
            row_without_first_last = C[i, 1:-1]
            if np.all(np.isnan(row_without_first_last)):
                C[i, matrice_A_modifie.shape[0] - i] = 1

        # Calculer les valeurs de Eij
        for i in range(matrice_A_modifie.shape[0]):
            for j in range(matrice_A_modifie.shape[1]+1):
                if i + j <= matrice_A_modifie.shape[0] - 1:
                    E[i, j] = np.nan
                else:
                    E[i, j] = C[i, j] * D[i]

        for i in range(matrice_A_modifie.shape[0]):
            for j in range(matrice_A_modifie.shape[1]+1):
                if i < j:
                    X[i, j] = np.nan
                else:
                    X[i, j] = E[i, matrice_A_modifie.shape[0] + j - i]

        X = X[:, :-1]
        
        # Calculer la somme de chaque ligne de X pour la colonne 'BE'
        BE = np.nansum(X, axis=1)
        

        # Répéter les valeurs de taux pour correspondre au nombre de lignes dans BE
        taux_repeated = np.repeat(taux, len(BE) // len(taux) + 1)[:len(BE)]

        # Créer le tableau pandas avec les colonnes 'BE', 'Taux_zc', et 'Déflateur'
        tableau = pd.DataFrame({
        'Année de survenance': triangle_charge.iloc[:, 0],
        'BE': BE,
        'Taux_zc': taux_repeated,
        'Déflateur': [1 / (1 + taux_repeated[i]) ** (i + 1) for i in range(len(BE))]
        })

        # Transformer la colonne 'déflateur' en une ligne pour utiliser somme prod
        ligne_deflateur = tableau['Déflateur'].values
        sommeprod = np.nansum(X * ligne_deflateur, axis=1)
        tableau['BE actualisé'] = sommeprod


        # Convertir en DataFrame et sauvegarder en Excel
        cadences = pd.DataFrame(C)
        liquidation = pd.DataFrame(E)
        BE = pd.DataFrame(X)
        BestE = tableau['BE'].to_numpy()
        BestEA = tableau['BE actualisé'].to_numpy()
    
        
        best.append(np.array(BestE))
        bestA.append(np.array(BestEA))
        Best_Estimate[k] = np.sum(sommeprod)
    
    # Optionally break after the first iteration for testing
    Best_estimate = np.mean(best, axis=0)
    Best_estimate_actualisé = np.mean(bestA, axis=0)

        # Calcul de la somme des éléments correspondants de chaque tableau
    # Créer le tableau pandas avec les colonnes 'BE', 'Taux_zc', et 'Déflateur'
    Best_estimate_boot = pd.DataFrame({
        'Année de survenance': triangle_charge.iloc[:, 0],
        'mean BE': Best_estimate,
        'mean BE actualisé': Best_estimate_actualisé
    
        })
    # Calculate the mean of all Best Estimates
    mean_best_estimate = np.mean(Best_Estimate)

    return Best_estimate_boot, mean_best_estimate
