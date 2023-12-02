def traiter_donnees(df): #bilan de ce qu'on a fait au dessus, sans prendre en compte l'état de l'arc pour l'instant

    """
    prend le dataframe tout sale que tu prends en csv et renvoie le dataframe avec les dates en index pour bien tracer et sans utc et garde que les colonnes utiles
    """

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
    return df


def plot_daily_mean(df, column, title):
    daily_mean = df.loc[:,[column]].resample('D').mean()
    fig, ax = plt.subplots()
    ax.plot(daily_mean.index, daily_mean[column], label=column)
    ax.set_xlabel("Date")
    ax.set_title(title)
    ax.legend()
    plt.xticks(rotation=45)
    plt.show()