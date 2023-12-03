import pandas as pd
import matplotlib as plt


def node_filter(df, arc_id : str):

    """To take only the node of the scope"""

    arc_list = {
        'champs' : {'noeud_amont' : 'Av_Champs_Elysees-Washington', 'noeud_aval' : 'Av_Champs_Elysees-Berri'},
     'convention' : {'noeud_amont' : 'Convention-Blomet', 'noeud_aval' : 'Lecourbe-Convention'},
      'sts' : {'noeud_amont' : 'Sts_Peres-Voltaire', 'noeud_aval' : 'Sts_Peres-Universite'},
      }

    assert arc_id in arc_list, f'arc_id must be in {arc_list.keys()}'


    msk_node = (df['Libelle noeud amont'] == arc_list[arc_id]['noeud_amont'])&(df['Libelle noeud aval'] == arc_list[arc_id]['noeud_aval'])


    return df.loc[msk_node]


def traiter_donnees(df, arc : str): #bilan de ce qu'on a fait au dessus, sans prendre en compte l'état de l'arc pour l'instant

    """
    prend le dataframe tout sale que tu prends en csv et renvoie le dataframe avec les dates en index pour bien tracer et sans utc et garde que les colonnes utiles
    """
    #On commence par garder uniquement les noeuds qui nous intéressent
    df = node_filter(df, arc_id=arc)
    #tri par date
    df = df.sort_values(by=['Date et heure de comptage'], ascending=True)

    #on garde que les colonnes qui nous intéressent
    colonnes_a_garder = ['Libelle', 'Date et heure de comptage','Taux d\'occupation', 'Etat arc', 'Débit horaire']
    df = df.loc[:,colonnes_a_garder]

    #renommer les colonnes pour que ce soit pratique
    df = df.rename(columns={'Date et heure de comptage': 'timestamp', 'Taux d\'occupation': 'taux_occupation', 'Etat arc': 'etat_arc', 'Débit horaire': 'debit_horaire'})

    #jsplus pk j'ai fait ça
    df.reset_index(drop=True, inplace=True)

    df['timestamp'] = pd.to_datetime(df['timestamp'], utc = True)
    df['timestamp'] = df['timestamp'].dt.tz_convert('Europe/Paris')
    df['timestamp'] = df['timestamp'].dt.tz_localize(None)
    df.set_index('timestamp', inplace=True)

    df_final = completer_heures_manquantes(df)
    

    return df_final


def plot_daily_mean(df, column, title):
    daily_mean = df.loc[:,[column]].resample('D').mean()
    fig, ax = plt.subplots()
    ax.plot(daily_mean.index, daily_mean[column], label=column)
    ax.set_xlabel("Date")
    ax.set_title(title)
    ax.legend()
    plt.xticks(rotation=45)
    plt.show()

def completer_heures_manquantes(df):
    min_time = df.index.min()
    max_time = df.index.max()
    full_range = pd.date_range(start=min_time, end=max_time, freq='H')  # 'H' pour une fréquence horaire

    # Réindexer votre DataFrame pour avoir une série temporelle continue
    continuous_df = df.reindex(full_range)
    return continuous_df

def remplacer_valeurs_manquantes_par_decalage(df):
    # Décalez les colonnes nécessaires d'une semaine
    df['taux_occupation_shifted'] = df['taux_occupation'].shift(periods=7, freq='D')
    df['debit_horaire_shifted'] = df['debit_horaire'].shift(periods=7, freq='D')

    # Remplacez les valeurs manquantes par les valeurs décalées
    df['taux_occupation'].fillna(df['taux_occupation_shifted'], inplace=True)
    df['debit_horaire'].fillna(df['debit_horaire_shifted'], inplace=True)

    # Supprimez les colonnes de décalage si vous ne les voulez plus dans le DataFrame final
    df.drop(['taux_occupation_shifted', 'debit_horaire_shifted'], axis=1, inplace=True)

    return df

def clean_meteo(df):


    # Réduire le DataFrame aux colonnes spécifiées
    nouveau_data = df.loc[:, ['Date', 'Précipitations dans la dernière heure', 'Température (°C)']]
    nouveau_data.dropna(subset=['Date'], inplace=True)

    for i in range (len(nouveau_data)):
        # Convertir la chaîne en objet datetime
        date_time_obj = datetime.fromisoformat(nouveau_data['Date'][i])

        # Formatter la date en chaîne au nouveau format AAAA-MM-JJ HH:MM:SS
        nouveau_format_date = date_time_obj.strftime('%Y-%m-%d %H:%M:%S')

        nouveau_data['Date'][i]=nouveau_format_date
    nouveau_data['Date'] = pd.to_datetime(nouveau_data['Date'])
    nouveau_data = nouveau_data.sort_values(by='Date')
    
        #renommer les colonnes pour que ce soit pratique
    nouveau_data = nouveau_data.rename(columns={'Date': 'timestamp'})


    nouveau_data['timestamp'] = pd.to_datetime(nouveau_data['timestamp'], utc = True)
    nouveau_data['timestamp'] = nouveau_data['timestamp'].dt.tz_convert('Europe/Paris')
    nouveau_data['timestamp'] = nouveau_data['timestamp'].dt.tz_localize(None)
    nouveau_data.set_index('timestamp', inplace=True)
    nouveau_data=completer_heures_manquantes(nouveau_data)

    nouveau_data['precipitations'] = nouveau_data['Précipitations dans la dernière heure']
    nouveau_data['temperature'] = nouveau_data['Température (°C)']


    # Interpoler les valeurs manquantes en utilisant les méthodes de remplissage existantes (pad/ffill et backfill/bfill)
    nouveau_data['precipitations'].fillna(method='bfill', inplace=True)
    nouveau_data['temperature'].fillna(method='bfill', inplace=True)
    nouveau_data.drop(columns=['Précipitations dans la dernière heure','Température (°C)'], inplace=True)

    return nouveau_data
