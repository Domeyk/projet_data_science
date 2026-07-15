import pandas as pd
import numpy as np


def injecter_manquants(df, colonne, mecanisme,
                       variable_liee=None, proba=0.2,
                       centre=0, pente=0.1, seed=42):
    # je travaille sur une copie : l'original reste ma "verite"
    df_troue = df.copy()

    # fixation du hasard pour la reproductibilite
    np.random.seed(seed)

    # calcul de la proba de manquer p,differement selon le mecanisme
    if mecanisme == "MCAR":
        # proba constante : le hasard pur
        p = proba
    elif mecanisme == "MAR":
        # proba dependant d'une AUTRE variable, via la sigmoide
        score = pente * (df[variable_liee] - centre)
        p = 1 / (1 + np.exp(-score))
    elif mecanisme == "MNAR":
        # proba dependant de la valeur CACHEE elle-meme, via la sigmoide
        score = pente * (df[colonne] - centre)
        p = 1 / (1 + np.exp(-score))

    # tirage de Bernoulli, ligne par ligne
    tirage = np.random.rand(len(df))
    manquant = tirage < p

    # je cache les valeurs choisies
    df_troue.loc[manquant, colonne] = np.nan
    return df_troue
if __name__ == "__main__":
    df = pd.read_csv("data/processed/freMTPL2freq_pretraite.csv")

    # MCAR sur DrivAge
    d1 = injecter_manquants(df, "DrivAge", "MCAR", proba=0.2)
    print("MCAR  -> trous dans DrivAge    :", d1["DrivAge"].isnull().sum())

    # MAR sur BonusMalus, pilote par l'age
    d2 = injecter_manquants(df, "BonusMalus", "MAR",
                            variable_liee="DrivAge", centre=55, pente=0.12)
    print("MAR   -> trous dans BonusMalus :", d2["BonusMalus"].isnull().sum())

    # MNAR sur BonusMalus, pilote par lui-meme
    d3 = injecter_manquants(df, "BonusMalus", "MNAR", centre=90, pente=0.05)
    print("MNAR  -> trous dans BonusMalus :", d3["BonusMalus"].isnull().sum())