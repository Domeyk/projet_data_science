# Imputation des données manquantes : comparaison Moyenne / k-NN / MICE

**Projet de Data Science et Tarification — ENSEA 2025-2026**

## Contexte

En assurance, les données comportent souvent des valeurs manquantes, que les modèles de
tarification ne tolèrent pas : il faut donc les **imputer**. Ce projet compare trois
méthodes d'imputation — **moyenne**, **k-NN** et **MICE** — et mesure leur impact sur un
**GLM Poisson** prédisant la fréquence de sinistres, à partir du portefeuille automobile
français `freMTPL2freq`. La méthode de référence est l'article de van Buuren &
Groothuis-Oudshoorn (2011).

La base étant complète, elle sert de **vérité-terrain** : on crée les trous nous-mêmes
(selon trois mécanismes : MCAR, MAR, MNAR), puis on évalue les méthodes sur deux axes —
la **reconstruction** (erreur RMSE) et l'**impact sur le GLM** (coefficient et déviance).

## Structure du dépôt
Projet_mice/
data/
raw/                 -> donnees brutes (freMTPL2freq_brut.csv)
processed/           -> donnees nettoyees (freMTPL2freq_pretraite.csv)
results/figures/         -> graphiques generes
chargement.py            -> lecture des donnees brutes
pretraitement.py         -> nettoyage + sous-echantillonnage
injection.py             -> injection des manquants (MCAR / MAR / MNAR)
imputation_moyenne.py    -> imputation par la moyenne
imputation_knn.py        -> imputation par k-NN
imputation_mice.py       -> imputation par MICE
experience.py            -> tableau RMSE (3 methodes x 3 mecanismes)
evaluation_glm.py        -> impact sur le GLM Poisson
figures.py               -> genere les deux graphiques
requirements.txt
README.md
## Installation

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1        # Windows (PowerShell)
pip install -r requirements.txt
```

## Exécution

```bash
python pretraitement.py     # génère les données nettoyées
python experience.py        # tableau RMSE (axe 1)
python evaluation_glm.py    # impact sur le GLM Poisson (axe 2)
python figures.py           # génère les graphiques
```

Une graine fixe (`random_state=42`) garantit la reproductibilité. k-NN et MICE peuvent
prendre quelques minutes.

## Résultats principaux

- Sous **MCAR** et **MAR**, MICE reconstruit le mieux les données et préserve le
  coefficient du GLM, là où la moyenne l'atténue.
- Sous **MNAR**, l'hypothèse de MICE est violée : toutes les méthodes se dégradent.

## Auteurs

DEDOU OKOMA YANN EVAN MAUREL — OKA KOUAME JEAN-MARIE VIANNEY — ONANA ABENAN VALERY AUGUSTE