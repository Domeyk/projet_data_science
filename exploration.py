import pandas as pd

# Lire les données brutes
df = pd.read_csv("data/raw/freMTPL2freq_brut.csv")

# 1) Vue d'ensemble : type de chaque colonne + nombre de valeurs présentes
print("===== INFOS GÉNÉRALES =====")
print(df.info())

# 2) LE point clé pour notre projet : y a-t-il des valeurs manquantes ?
print("\n===== VALEURS MANQUANTES PAR COLONNE =====")
print(df.isnull().sum())

# 3) Statistiques de base sur les colonnes numériques (moyenne, min, max...)
print("\n===== STATISTIQUES DES COLONNES NUMÉRIQUES =====")
print(df.describe())