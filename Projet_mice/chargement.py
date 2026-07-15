# On sort l'outil des tableaux
import pandas as pd

# Lire les données brutes rangées dans le dépôt
df = pd.read_csv("data/raw/freMTPL2freq_brut.csv")

# Afficher la taille du tableau et les colonnes
print("Taille du tableau :", df.shape)
print("Colonnes :", list(df.columns))

# Afficher les 5 premières lignes
print(df.head())