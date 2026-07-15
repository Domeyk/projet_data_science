import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from injection import injecter_manquants

# je lis la verite complete
df = pd.read_csv("data/processed/freMTPL2freq_pretraite.csv")

# j'injecte les MEMES trous MCAR dans DrivAge (20%) pour comparer a armes egales
df_troue = injecter_manquants(df, "DrivAge", "MCAR", proba=0.2)
lignes_trouees = df_troue["DrivAge"].isnull()

# je ne garde que les colonnes numeriques (k-NN calcule des distances -> besoin de nombres)
colonnes_num = ["ClaimNb", "Exposure", "VehPower", "VehAge",
                "DrivAge", "BonusMalus", "Density"]
df_num = df_troue[colonnes_num]

# --- IMPUTATION PAR k-NN ---
# k=5 : pour chaque trou, on regarde les 5 lignes les plus proches
imputeur = KNNImputer(n_neighbors=5)
tableau_impute = imputeur.fit_transform(df_num)

# fit_transform renvoie un tableau brut : je le remets en DataFrame avec les bons noms
df_impute = pd.DataFrame(tableau_impute, columns=colonnes_num, index=df_num.index)

# --- EVALUATION : devine vs verite ---
vraies_valeurs = df.loc[lignes_trouees, "DrivAge"]
valeurs_devinees = df_impute.loc[lignes_trouees, "DrivAge"]

erreur = np.sqrt(((vraies_valeurs - valeurs_devinees) ** 2).mean())
print("Erreur moyenne (RMSE) de l'imputation k-NN :", round(erreur, 2), "ans")
print("Pour rappel, la moyenne faisait : 14.25 ans")