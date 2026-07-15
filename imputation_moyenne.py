import pandas as pd
import numpy as np
from injection import injecter_manquants

# je lis la verite complete
df = pd.read_csv("data/processed/freMTPL2freq_pretraite.csv")

# j'injecte des trous MCAR dans DrivAge (20%)
df_troue = injecter_manquants(df, "DrivAge", "MCAR", proba=0.2)

# je repere les lignes qui ont un trou (pour pouvoir comparer apres)
lignes_trouees = df_troue["DrivAge"].isnull()

# --- IMPUTATION PAR LA MOYENNE ---
# je calcule la moyenne des valeurs ENCORE visibles
moyenne = df_troue["DrivAge"].mean()
print("Moyenne utilisee pour combler les trous :", round(moyenne, 1))

# je remplace tous les trous par cette moyenne
df_impute = df_troue.copy()
df_impute["DrivAge"] = df_impute["DrivAge"].fillna(moyenne)

# --- COMPARAISON : devine vs verite ---
# vraies valeurs (dans df) vs valeurs devinees (dans df_impute), sur les lignes trouees
vraies_valeurs = df.loc[lignes_trouees, "DrivAge"]
valeurs_devinees = df_impute.loc[lignes_trouees, "DrivAge"]

# erreur moyenne (RMSE) : en moyenne, de combien d'annees on se trompe ?
erreur = np.sqrt(((vraies_valeurs - valeurs_devinees) ** 2).mean())
print("Erreur moyenne (RMSE) de l'imputation par la moyenne :", round(erreur, 2), "ans")