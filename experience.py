import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer
from injection import injecter_manquants
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

# je lis la verite complete
df = pd.read_csv("data/processed/freMTPL2freq_pretraite.csv")

# les colonnes numeriques utilisees par k-NN et MICE
colonnes_num = ["ClaimNb", "Exposure", "VehPower", "VehAge",
                "DrivAge", "BonusMalus", "Density"]

# la variable qu'on troue et qu'on cherche a deviner (la meme pour les 3 mecanismes)
cible = "BonusMalus"

# mes trois mecanismes, avec les reglages deja valides
mecanismes = {
    "MCAR": {"mecanisme": "MCAR", "proba": 0.2},
    "MAR":  {"mecanisme": "MAR", "variable_liee": "DrivAge", "centre": 55, "pente": 0.12},
    "MNAR": {"mecanisme": "MNAR", "centre": 90, "pente": 0.05},
}

# petite fonction pour l'erreur (racine de l'erreur quadratique moyenne)
def rmse(vrai, devine):
    return np.sqrt(((vrai - devine) ** 2).mean())

# entete du tableau
print(f"{'Mecanisme':<9} {'Moyenne':>9} {'k-NN':>9} {'MICE':>9}")
print("-" * 40)

for nom, params in mecanismes.items():
    # 1) j'injecte les trous sur la cible selon le mecanisme
    df_troue = injecter_manquants(df, cible, **params)
    trous = df_troue[cible].isnull()
    vrai = df.loc[trous, cible]
    df_num = df_troue[colonnes_num]

    # 2a) imputation par la moyenne
    moy = df_num.fillna(df_num.mean())
    e_moy = rmse(vrai, moy.loc[trous, cible])

    # 2b) imputation k-NN
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df_num)                       # mise a l'echelle
    knn_scaled = KNNImputer(n_neighbors=5).fit_transform(df_scaled)
    knn = pd.DataFrame(scaler.inverse_transform(knn_scaled),        # retour a l'echelle d'origine
                       columns=colonnes_num, index=df_num.index)
    e_knn = rmse(vrai, knn.loc[trous, cible])

    # 2c) imputation MICE
    foret = RandomForestRegressor(n_estimators=20, max_depth=10, n_jobs=-1, random_state=42)
    mice = pd.DataFrame(IterativeImputer(estimator=foret, max_iter=10, random_state=42).fit_transform(df_num),
                        columns=colonnes_num, index=df_num.index)
    e_mice = rmse(vrai, mice.loc[trous, cible])

    # une ligne du tableau
    print(f"{nom:<9} {e_moy:>9.2f} {e_knn:>9.2f} {e_mice:>9.2f}")