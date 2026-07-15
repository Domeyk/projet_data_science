import pandas as pd
import numpy as np

# je lis les donnees nettoyees (ma "verite" complete)
df = pd.read_csv("data/processed/freMTPL2freq_pretraite.csv")
df_troue = df.copy()

# je fige le hasard
np.random.seed(42)

# MNAR : la proba de manquer depend du BonusMalus LUI-MEME
# je centre le BonusMalus autour de 90 et je le met a l'echelle
score = 0.05 * (df["BonusMalus"] - 90)

# meme sigmoide (Cours 2) : score -> proba entre 0 et 1
proba = 1 / (1 + np.exp(-score))

# tirage de Bernoulli, ligne par ligne
tirage = np.random.rand(len(df))
manquant = tirage < proba

# je cache le BonusMalus sur les lignes tirees
df_troue.loc[manquant, "BonusMalus"] = np.nan

# verifications
print("Valeurs manquantes dans BonusMalus :", df_troue["BonusMalus"].isnull().sum())
print("Soit environ", round(100 * manquant.mean()), "% des lignes")

# vérification du MNAR : le BonusMalus moyen des valeurs CACHEES doit etre plus eleve
print("BonusMalus moyen des valeurs CACHEES  :", round(df.loc[manquant, "BonusMalus"].mean(), 1))
print("BonusMalus moyen des valeurs PRESENTES:", round(df.loc[~manquant, "BonusMalus"].mean(), 1))