import pandas as pd
import matplotlib as plt
import requests
from io import StringIO

def node_filter(df, arc_id : str):

    """To take only the node of the scope"""

    arc_list = {
        'champs' : {'noeud_amont' : 'Av_Champs_Elysees-Washington', 'noeud_aval' : 'Av_Champs_Elysees-Berri'},
     'convention' : {'noeud_amont' : 'Convention-Blomet', 'noeud_aval' : 'Lecourbe-Convention'},
      'sts' : {'noeud_amont' : 'Sts_Peres-Voltaire', 'noeud_aval' : 'Sts_Peres-Universite'},
      }

    assert arc_id in arc_list, f'arc_id must be in {arc_list.keys()}'


    msk_node = (df['libelle_nd_amont'] == arc_list[arc_id]['noeud_amont'])&(df['libelle_nd_aval'] == arc_list[arc_id]['noeud_aval'])


    return df.loc[msk_node]

def load_traffic_data(arc : str, year : int=None):
    """download raw traffic data from open data soft API, be aware that it can take a while to download convention

    Args:
        arc (str): has to be either champs, sts or convention

    Returns:
        DataFrame: raw dataframe
    """
    arc_id = {
        'champs':'AV_Champs_Elysees',
        'sts':'Sts_Peres',
        'convention':'Convention'
    }

    assert arc in arc_id.keys(), 'arc name is not valid, it has to be convention sts or champs'
    

    
    # file too heavy if we don't select upstream node for convention
    if arc == 'convention':
        url = 'https://parisdata.opendatasoft.com/api/explore/v2.1/catalog/datasets/comptages-routiers-permanents/exports/csv?refine=libelle%3A%22Convention%22&refine=libelle_nd_amont%3A%22Convention-Blomet%22'

    else : url = f'https://parisdata.opendatasoft.com/api/explore/v2.1/catalog/datasets/comptages-routiers-permanents/exports/csv?refine=libelle%3A%22{arc_id[arc]}%22'

    print(f'loading data for {arc} [...]')
    response = requests.get(url)
    response.raise_for_status()

    df = pd.read_csv(StringIO(response.text), delimiter=';')

    df.rename(columns={'t_1h': 'Date et heure de comptage', 'q': 'Débit horaire', 'k':"Taux d'occupation", 'etat_barre':'etat_arc'}, inplace=True)

    return df 

def load_worksites():
    """download raw worskite obstructing traffic from open data soft API

    Returns:
        DataFrame: raw dataframe
    """

    url ='https://opendata.paris.fr/api/explore/v2.1/catalog/datasets/chantiers-perturbants/exports/csv'

    print(f'loading data for worksites [...]')
    response = requests.get(url)
    response.raise_for_status()

    df = pd.read_csv(StringIO(response.text), delimiter=';')

    return df

def clean_worksites_data(df):
    """cleaning worksites data from API

    Args:
        df (DataFrame): raw dataframe from load_worksites function

    Returns:
        DataFrame: dataframe cleaned
    """

    msk_arc = df['voie'].str.contains('pères|convention|elysées', case=False, na=False)

    colonnes_a_garder = ['date_debut', 'date_fin', 'niveau_perturbation', 'voie', 'impact_circulation']
    df = df.loc[msk_arc, colonnes_a_garder]

    return df


def traiter_donnees(df, arc : str): #bilan de ce qu'on a fait au dessus, sans prendre en compte l'état de l'arc pour l'instant

    """
    prend le dataframe tout sale que tu prends en csv et renvoie le dataframe avec les dates en index pour bien tracer et sans utc et garde que les colonnes utiles
    """
    #On commence par garder uniquement les noeuds qui nous intéressent
    df = node_filter(df, arc_id=arc)
    #tri par date
    df = df.sort_values(by=['Date et heure de comptage'], ascending=True)

    #on garde que les colonnes qui nous intéressent
    colonnes_a_garder = ['libelle', 'Date et heure de comptage','Taux d\'occupation', 'etat_arc', 'Débit horaire']
    df = df.loc[:,colonnes_a_garder]

    #renommer les colonnes pour que ce soit pratique
    df = df.rename(columns={'Date et heure de comptage': 'timestamp', 'Taux d\'occupation': 'taux_occupation', 'Débit horaire': 'debit_horaire'})

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

    """
    Prendre en argument le dataset meteo, rend un dataset clean sans trou avec le même format que la fonction traiter_donner()
    """


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

