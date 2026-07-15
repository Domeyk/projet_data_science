import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer
from injection import injecter_manquants

# je lis la verite complete
df = pd.read_csv("data/processed/freMTPL2freq_pretraite.csv")

# predicteurs numeriques du GLM ; Exposure servira d'offset ; ClaimNb est la cible
predicteurs = ["VehPower", "VehAge", "DrivAge", "BonusMalus", "Density"]
colonnes_num = ["Exposure", "VehPower", "VehAge", "DrivAge", "BonusMalus", "Density"]
cible = "BonusMalus"

# ajuste un GLM Poisson (Cours 2 : lien log, Exposure en offset)
def ajuster_glm(data):
    X = sm.add_constant(data[predicteurs])
    y = data["ClaimNb"]
    offset = np.log(data["Exposure"])
    return sm.GLM(y, X, family=sm.families.Poisson(), offset=offset).fit()

# impute selon la methode demandee
def imputer(df_num, methode):
    if methode == "Moyenne":
        return df_num.fillna(df_num.mean())
    if methode == "k-NN":
        sc = StandardScaler()
        s = sc.fit_transform(df_num)
        r = KNNImputer(n_neighbors=5).fit_transform(s)
        return pd.DataFrame(sc.inverse_transform(r), columns=colonnes_num, index=df_num.index)
    if methode == "MICE":
        foret = RandomForestRegressor(n_estimators=20, max_depth=10, n_jobs=-1, random_state=42)
        r = IterativeImputer(estimator=foret, max_iter=10, random_state=42).fit_transform(df_num)
        return pd.DataFrame(r, columns=colonnes_num, index=df_num.index)

# --- REFERENCE : GLM sur la base COMPLETE (la verite ideale) ---
ref = ajuster_glm(df)
coef_ref = ref.params["BonusMalus"]
print(f"REFERENCE (base complete) : coef BonusMalus = {coef_ref:.5f} | deviance = {ref.deviance:.0f}\n")

# --- pour chaque mecanisme et chaque methode : imputer -> ajuster -> comparer ---
mecanismes = {
    "MCAR": {"mecanisme": "MCAR", "proba": 0.2},
    "MAR":  {"mecanisme": "MAR", "variable_liee": "DrivAge", "centre": 55, "pente": 0.12},
    "MNAR": {"mecanisme": "MNAR", "centre": 90, "pente": 0.05},
}
print(f"{'Mecanisme':<9}{'Methode':<9}{'coef BM':>10}{'ecart/ref':>11}{'deviance':>10}")
print("-" * 49)
for nom, params in mecanismes.items():
    df_troue = injecter_manquants(df, cible, **params)
    for methode in ["Moyenne", "k-NN", "MICE"]:
        imp = imputer(df_troue[colonnes_num], methode)
        data = df.copy()
        data["BonusMalus"] = imp["BonusMalus"].values   # on ne remplace QUE le BonusMalus impute
        g = ajuster_glm(data)
        coef = g.params["BonusMalus"]
        print(f"{nom:<9}{methode:<9}{coef:>10.5f}{coef - coef_ref:>+11.5f}{g.deviance:>10.0f}")