import pandas as pd
import os

# je lis les donnees brutes
df = pd.read_csv("data/raw/freMTPL2freq_brut.csv")
print("Avant nettoyage :", df.shape)

# je corrige les expositions superieures a 1 (un contrat ne peut pas etre actif plus d'un an)
df.loc[df["Exposure"] > 1, "Exposure"] = 1

# je plafonne le nombre de sinistres a 4 (au-dela c'est une valeur aberrante)
df.loc[df["ClaimNb"] > 4, "ClaimNb"] = 4

# je garde 50000 lignes au hasard pour que les calculs soient plus rapides
df = df.sample(n=100000, random_state=42)
print("Apres nettoyage :", df.shape)

# je cree le dossier processed et j'enregistre les donnees nettoyees
os.makedirs("data/processed", exist_ok=True)
df.to_csv("data/processed/freMTPL2freq_pretraite.csv", index=False)
print("Fichier enregistre dans data/processed")
