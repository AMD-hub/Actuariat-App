import pandas as pd
import numpy as np 

def extract_second_diagonal(df):
    n = len(df)
    triangle_values = []

    for i in range(n):
        for j in range(n+1):
            if i + j == n :  # Condition pour que i + j = n
                triangle_values.append(df.iloc[i, j])

    return triangle_values

def calculate_be_tables(triangle_charge, triangle_reglements):

    # Extract second diagonals for both triangles
    last_diag_charge     = extract_second_diagonal(triangle_charge)
    last_diag_reglements = extract_second_diagonal(triangle_reglements)

    # Last columns for both triangles
    last_col_charge = triangle_charge.iloc[:, -1].tolist()
    last_col_reglements = triangle_reglements.iloc[:, -1].tolist()

    # Calculate columns for BE_charge
    ibnr_charge = [last_col_charge[i] - last_diag_charge[i] for i in range(len(last_diag_charge))]
    psap_dd_charge = [last_col_charge[i] - last_diag_reglements[i] - ibnr_charge[i] for i in range(len(last_col_charge))]
    be_charge_col = [psap_dd_charge[i] + ibnr_charge[i] for i in range(len(psap_dd_charge))]

    # Calculate columns for BE_reglement
    ibnr_reglements = [last_col_reglements[i] - last_diag_reglements[i] - psap_dd_charge[i] for i in range(len(last_diag_reglements))]
    be_reglements_col = [psap_dd_charge[i] + ibnr_reglements[i] for i in range(len(psap_dd_charge))]

    # Creating the output tables
    be_charge = pd.DataFrame({
        'Année_de_survenance':triangle_charge.iloc[:,0],
        'PSAP DD': psap_dd_charge,
        'Charge ultime': last_col_charge,
        'Dernière charge': last_diag_charge,
        'IBNR': ibnr_charge,
        'BE': be_charge_col
    })
    
    be_reglement = pd.DataFrame({
        'Année_de_survenance':triangle_charge.iloc[:,0],
        'PSAP DD': psap_dd_charge,
        'Paiements à l\'ultime': last_col_reglements,
        'Paiements derniers': last_diag_reglements,
        'IBNR': ibnr_reglements,
        'BE': be_reglements_col
    })

    # Add a row for column sums
    be_charge.loc['Total'] = be_charge.sum()
    be_reglement.loc['Total'] = be_reglement.sum()
    

    # Renommage de la dernière valeur de 'Année_de_survenance' en 'Total'
    be_reglement.at['Total', 'Année_de_survenance'] = 'Total'
    be_charge.at['Total', 'Année_de_survenance'] = 'Total'
    return be_charge, be_reglement



def calculate_be_actualise(triangle_charge, triangle_reglements, taux_zc):
    triangle_reg = triangle_reglements.values
    # Extract second diagonals for both triangles
    last_diag_charge = extract_second_diagonal(triangle_charge)
    last_diag_reglements = extract_second_diagonal(triangle_reglements)

    # Last columns for both triangles
    last_col_charge = triangle_charge.iloc[:, -1].tolist()
    last_col_reglements = triangle_reglements.iloc[:, -1].tolist()

    # Calculate columns for BE_charge
    ibnr_charge = [last_col_charge[i] - last_diag_charge[i] for i in range(len(last_diag_charge))]
    psap_dd_charge = [last_col_charge[i] - last_diag_reglements[i] - ibnr_charge[i] for i in range(len(last_col_charge))]
    be_charge_col = [psap_dd_charge[i] + ibnr_charge[i] for i in range(len(psap_dd_charge))]

    # Calculate columns for BE_reglement
    ibnr_reglements = [last_col_reglements[i] - last_diag_reglements[i] - psap_dd_charge[i] for i in range(len(last_diag_reglements))]
    be_reglements_col = [psap_dd_charge[i] + ibnr_reglements[i] for i in range(len(psap_dd_charge))]

    # Creating the output tables
    be_charge = pd.DataFrame({
        'Année_de_survenance':triangle_charge.iloc[:,0],
        'PSAP DD': psap_dd_charge,
        'Charge ultime': last_col_charge,
        'Dernière charge': last_diag_charge,
        'IBNR': ibnr_charge,
        'BE': be_charge_col
    })
    
    be_reglement = pd.DataFrame({
        'Année_de_survenance':triangle_charge.iloc[:,0],
        'PSAP DD': psap_dd_charge,
        'Paiements à l\'ultime': last_col_reglements,
        'Paiements derniers': last_diag_reglements,
        'IBNR': ibnr_reglements,
        'BE': be_reglements_col
    })

    # Add a row for column sums
    be_charge.loc['Total'] = be_charge.sum()
    be_reglement.loc['Total'] = be_reglement.sum()
    
   

    # Renommage de la dernière valeur de 'Année_de_survenance' en 'Total'
    be_reglement.at['Total', 'Année_de_survenance'] = 'Total'
    be_charge.at['Total', 'Année_de_survenance'] = 'Total'
    # Extraire la colonne 'BE' du DataFrame
    B = be_reglement['BE'].values
    D = be_charge['BE'].values
    taux= taux_zc['taux']
    # Initialiser C avec les mêmes dimensions que A
    C = np.zeros_like(triangle_reglements)
    E = np.zeros_like(triangle_reglements)
    X= np.zeros_like(triangle_reglements)
    
    régl_modifie = triangle_reg[:, 1:]
    
    
    # Calculer les valeurs de Cij pour i > 0 et j > 0
    for i in range(0, régl_modifie.shape[0]):  # On commence à 1 pour ignorer la première ligne
        for j in range(0, régl_modifie.shape[1]):
            if B[i] == 0 or i + j <= régl_modifie.shape[0] - 1:
                C[i, j] = np.nan
            else:
                C[i, j] = (régl_modifie[i, j] - régl_modifie[i, j-1]) / B[i]  # j-1 pour aligner les indices de B avec ceux de A

      # Vérification de la somme des lignes de C et ajustement si nécessaire
    for i in range(0, C.shape[0]):
        row_without_first_last = C[i, 1:-1]  # Exclure la première et la dernière colonne
        if np.all(np.isnan(row_without_first_last)):
        # Votre code pour traiter les cas où toutes les valeurs sont NaN

           
            C[i, régl_modifie.shape[0] - i ] = 1
       

    
    # Calculer les valeurs de Eij 
    for i in range(0, régl_modifie.shape[0]):
        for j in range(0, régl_modifie.shape[1]+1):
            if  i + j <= régl_modifie.shape[0] - 1:
                E[i, j] = np.nan
            else:
                E[i, j] = C[i,j] * D[i] 
    

    for i in range(0, régl_modifie.shape[0]):  
        for j in range(0, régl_modifie.shape[1]+1):
            if i<j:
                X[i, j] = np.nan
            else:
                X[i, j] = E[i,régl_modifie.shape[0] +j-i]  

    X = X[:, :-1]
    # Calculer la somme de chaque ligne de X pour la colonne 'BE'
    BE = np.nansum(X, axis=1)
   # Répéter les valeurs de taux_zc pour correspondre au nombre de lignes dans BE
    taux_zc_repeated = np.repeat(taux, len(BE) // len(taux) + 1)[:len(BE)]

    # Créer le tableau pandas avec les colonnes 'BE', 'taux_zc', et 'déflateur'
    tableau = pd.DataFrame({
        'Année de survenance':triangle_reg[:,0] ,
    'BE': BE,
    'Taux_zc': taux_zc_repeated,
    'Déflateur': [1 / (1 + taux_zc_repeated[i]) ** (i+1) for i in range(len(BE))]
    
})
    # Transformer la colonne 'déflateur' en une ligne pour utiliser somme prod
    ligne_deflateur = tableau['Déflateur'].values

    # Calculer l'opération SOMMEPROD de chaque ligne de X avec la ligne 'déflateur'
    sommeprod = np.nansum(X *ligne_deflateur , axis=1)



    # Ajouter la sommeprod au tableau pandas
    tableau['BE actualisé'] = sommeprod

    premiere_colonne = triangle_reg[:, 0].reshape(-1, 1) 
    # Concaténer la première colonne au début du tableau original
    C = np.hstack((premiere_colonne, C))
    E = np.hstack((premiere_colonne, E))
    X= np.hstack((premiere_colonne, X))
    # Add a row for column sums
    tableau.loc['Total'] = tableau.sum()
    # Renommage de la dernière valeur de 'Année_de_survenance' en 'Total'
    tableau.at['Total', 'Année de survenance'] = 'Total'
    return tableau
