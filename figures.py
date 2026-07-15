import matplotlib.pyplot as plt
import numpy as np
import os

# je cree le dossier ou ranger les figures
os.makedirs("results/figures", exist_ok=True)


couleurs = ["#8AA5BE", "#4F6982", "#283750"]   # Moyenne, k-NN, MICE
methodes = ["Moyenne", "k-NN", "MICE"]
mecanismes = ["MCAR", "MAR", "MNAR"]

# position des groupes de barres sur l'axe horizontal
x = np.arange(len(mecanismes))
largeur = 0.1   # largeur d'une barre

# ============ FIGURE 1 : RMSE (reconstruction) ============
rmse = {
    "Moyenne": [15.67, 13.39, 25.18],
    "k-NN":    [12.71, 10.82, 19.58],
    "MICE":    [11.82, 10.03, 18.82],
}

plt.figure(figsize=(6, 5))
plt.rcParams.update({"font.size": 13})
for i, methode in enumerate(methodes):
    # je decale chaque methode pour que les barres soient cote a cote
    plt.bar(x + (i - 1) * largeur, rmse[methode], largeur,
            label=methode, color=couleurs[i])

plt.xticks(x, mecanismes)
plt.ylabel("Erreur de reconstruction (RMSE)")
plt.title("Qualite de reconstruction du BonusMalus")
plt.legend()
plt.tight_layout()
plt.savefig("results/figures/figure_rmse.png", dpi=200)

# ============ FIGURE 2 : ecart au coefficient de reference ============
ecart = {
    "Moyenne": [-0.00113, -0.00127, -0.00072],
    "k-NN":    [-0.00035, -0.00028,  0.00326],
    "MICE":    [-0.00008,  0.00011,  0.00353],
}

plt.figure(figsize=(6, 5))
plt.rcParams.update({"font.size": 13})
for i, methode in enumerate(methodes):
    plt.bar(x + (i - 1) * largeur, ecart[methode], largeur,
            label=methode, color=couleurs[i])

plt.axhline(0, color="black", linewidth=0.8)   # ligne de reference (ecart = 0)
plt.xticks(x, mecanismes)
plt.ylabel("Ecart au coefficient de reference")
plt.title("Impact de l'imputation sur le GLM Poisson")
plt.legend()
plt.tight_layout()
plt.savefig("results/figures/figure_coef.png", dpi=200)

# j'affiche les deux figures a l'ecran
plt.show()
print("Figures enregistrees dans results/figures/")