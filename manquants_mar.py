import pandas as pd
import numpy as np

# je lis les donnees nettoyees (ma "verite" complete)
df = pd.read_csv("data/processed/freMTPL2freq_pretraite.csv")
df_troue = df.copy()

# je fige le hasard
np.random.seed(42)

# MAR : la proba de manquer depend de l'age du conducteur ---
# je centre l'age autour de 55 ans et je le met a l'echelle (pour la sigmoide)
score = 0.12 * (df["DrivAge"] - 55)

# fonction logistique (sigmoide) : transforme le score en proba entre 0 et 1
proba = 1 / (1 + np.exp(-score))

# tirage de Bernoulli, ligne par ligne, avec une proba differente a chaque ligne
tirage = np.random.rand(len(df))
manquant = tirage < proba

# je cache le BonusMalus sur les lignes tirees
df_troue.loc[manquant, "BonusMalus"] = np.nan

# verifications
print("Valeurs manquantes dans BonusMalus :", df_troue["BonusMalus"].isnull().sum())
print("Soit environ", round(100 * manquant.mean()), "% des lignes")

#  verification du MAR : l'age moyen differe selon que la valeur manque ou non
print("Age moyen quand BonusMalus est CACHE   :", round(df.loc[manquant, "DrivAge"].mean(), 1))
print("Age moyen quand BonusMalus est PRESENT :", round(df.loc[~manquant, "DrivAge"].mean(), 1))