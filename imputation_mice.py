import pandas as pd
import numpy as np
# ces deux lignes activent MICE (fonctionnalite "experimentale" de scikit-learn)
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from injection import injecter_manquants

# je lis la verite complete
df = pd.read_csv("data/processed/freMTPL2freq_pretraite.csv")

# j'injecte les MEMES trous MCAR dans DrivAge (20%) pour comparer a armes egales
df_troue = injecter_manquants(df, "DrivAge", "MCAR", proba=0.2)
lignes_trouees = df_troue["DrivAge"].isnull()

# je ne garde que les colonnes numeriques (MICE regresse sur des nombres)
colonnes_num = ["ClaimNb", "Exposure", "VehPower", "VehAge",
                "DrivAge", "BonusMalus", "Density"]
df_num = df_troue[colonnes_num]

# --- IMPUTATION PAR MICE ---
# max_iter=10 : nombre de tours de la boucle ; random_state=42 pour la reproductibilite
imputeur = IterativeImputer(max_iter=10, random_state=42)
tableau_impute = imputeur.fit_transform(df_num)

# je remets le resultat en DataFrame avec les bons noms de colonnes
df_impute = pd.DataFrame(tableau_impute, columns=colonnes_num, index=df_num.index)

# --- EVALUATION : devine vs verite ---
vraies_valeurs = df.loc[lignes_trouees, "DrivAge"]
valeurs_devinees = df_impute.loc[lignes_trouees, "DrivAge"]

erreur = np.sqrt(((vraies_valeurs - valeurs_devinees) ** 2).mean())
print("Erreur moyenne (RMSE) de l'imputation MICE :", round(erreur, 2), "ans")
print("Rappels -> moyenne : 14.25 ans  |  k-NN : 12.55 ans")