import numpy as np
import pandas as pd



def metrique_rmse(y_pred, arc_id : str):
     
    """
    Mettez votre prediction sous la forme d'un DataFrame de dimensions (120, 2), attention à remettre vos données à la bonne échelle
    """


    arc_list = {
        'champs' : {'noeud_amont' : 'Av_Champs_Elysees-Washington', 'noeud_aval' : 'Av_Champs_Elysees-Berri'},
     'convention' : {'noeud_amont' : 'Convention-Blomet', 'noeud_aval' : 'Lecourbe-Convention'},
      'sts' : {'noeud_amont' : 'Sts_Peres-Voltaire', 'noeud_aval' : 'Sts_Peres-Universite'},
      }

    assert arc_id in arc_list, f'arc_id must be in {arc_list.keys()}'

    assert y_pred.shape==(120, 2), 'Len(y) must be 120 (5 days) and two columns'

    y_true=traiter_donnees(pd.read_csv(fr'C:\Users\louis\OneDrive\Documents\CS\BCG Datathon\Datathon_BCG_colissiMONSTRE\datathon_bcg\data\{arc_id}_2023.csv', delimiter=';'), arc=arc_id).drop(columns=['Libelle','etat_arc'])[-120:]
    
    assert y_pred.index.equals(y_true.index), 'Mauvaises dates'

    df1=y_true['debit_horaire']
    df2=y_pred['debit_horaire']

    df1, df2 = df1.align(df2, axis=0, join='inner')  # 'axis=0' pour aligner selon les index (lignes), 'join=inner' pour garder uniquement les index en commun

    # Convertir les valeurs en tableaux NumPy pour calculer la RMSE
    values_df1 = df1.values
    values_df2 = df2.values

    # Calculer la différence au carré entre les valeurs des deux DataFrames
    squared_diff = (values_df1 - values_df2) ** 2

    # Calculer la moyenne des différences au carré
    mse = np.mean(squared_diff)

    # Calculer la racine carrée de l'erreur quadratique moyenne (RMSE)
    rmse_debit_horaire = np.sqrt(mse)

    df1=y_true['taux_occupation']
    df2=y_pred['taux_occupation']

    df1, df2 = df1.align(df2, axis=0, join='inner')  # 'axis=0' pour aligner selon les index (lignes), 'join=inner' pour garder uniquement les index en commun

    # Convertir les valeurs en tableaux NumPy pour calculer la RMSE
    values_df1 = df1.values
    values_df2 = df2.values

    # Calculer la différence au carré entre les valeurs des deux DataFrames
    squared_diff = (values_df1 - values_df2) ** 2

    # Calculer la moyenne des différences au carré
    mse = np.mean(squared_diff)

    # Calculer la racine carrée de l'erreur quadratique moyenne (RMSE)
    rmse_taux_occupation = np.sqrt(mse)

    print(f"RMSE for taux_occupation = {rmse_taux_occupation}", 
          f"RMSE for debit_horaire = {rmse_debit_horaire}")
    


def conserver_cinq_derniers_jours(df):
    nombre_de_lignes_a_garder = 120
    return df.tail(nombre_de_lignes_a_garder)