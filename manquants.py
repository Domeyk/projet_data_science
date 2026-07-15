import pandas as pd
import numpy as np

# je lis les donnees nettoyees (c'est ma "verite" complete)
df = pd.read_csv("data/processed/freMTPL2freq_pretraite.csv")

# je fais une copie sur laquelle je vais creer les trous
df_troue = df.copy()

# je fige le hasard pour avoir toujours le meme resultat (reproductibilite)
np.random.seed(42)

# MCAR : chaque ligne a la MEME probabilite d'etre manquante
proba = 0.2

# pour chaque ligne, je tire un nombre au hasard entre 0 et 1
tirage = np.random.rand(len(df))

# la valeur est cachee si le tirage tombe sous 0.2 (loi de Bernoulli de parametre 0.2)
manquant = tirage < proba

# je cache l'age du conducteur (DrivAge) sur les lignes tirees
df_troue.loc[manquant, "DrivAge"] = np.nan

# je verifie combien de trous ont ete crees
print("Valeurs manquantes dans DrivAge :", df_troue["DrivAge"].isnull().sum())
print("Soit environ", round(100 * manquant.mean()), "% des lignes")