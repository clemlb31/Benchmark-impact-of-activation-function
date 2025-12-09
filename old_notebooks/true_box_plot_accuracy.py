import json
from collections import defaultdict

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# === 1) Charger l'historique d'entraînement ===
with open("mnist_histories.json", "r") as f:
    runs = json.load(f)



# Choix de la métrique à afficher dans le boxplot
metric = "accuracy"   # ou "mae_mean" si tu veux comparer les MAE

# === 2) Regrouper les MSE par (batch_size, activation) ===
# data[batch_size][activation] = [liste des MSE]
data = defaultdict(lambda: defaultdict(list))

for r in runs:
    act = r["activation_function"]                    # "relu" ou "gelu"
    batch = r["training_parameters"]["batch_size"]    # 32, 1024, etc.
    acc = r["final_test_loss"][metric]
    data[batch][act].append(acc)



# === 3) Préparer les données pour le boxplot ===
batches = sorted(data.keys())        # ex : [32, 1024]
width = 0.35                         # largeur de chaque boîte



positions = []       # positions x de chaque boîte
box_data = []        # listes de MSE par boîte
box_colors = []      # couleurs associées (bleu/relu, rouge/gelu)



for i, batch in enumerate(batches):
    center = i  # centre du groupe correspondant à ce batch



    # position ReLU vs GELU autour du centre
    pos_relu = center - width / 2
    pos_gelu = center + width / 2



    positions.extend([pos_relu, pos_gelu])
    box_data.append(data[batch]["relu"])
    box_data.append(data[batch]["gelu"])
    box_colors.extend(["cornflowerblue", "lightcoral"])



# === 4) Tracer le boxplot ===

fig, ax = plt.subplots(figsize=(10, 5))
bp = ax.boxplot(
    box_data,
    positions=positions,
    widths=width,
    patch_artist=True,  # pour pouvoir colorer l'intérieur des boîtes
)

# Colorer les boîtes : ReLU = bleu, GELU = rouge
for box, color in zip(bp["boxes"], box_colors):
    box.set(facecolor=color)


# Grille horizontale en pointillés gris clair
ax.yaxis.grid(True, linestyle="--", linewidth=0.5, color="lightgray", alpha=0.7)



# Labels des ticks en x au centre des groupes de batch
ax.set_xticks(range(len(batches)))
ax.set_xticklabels([f"batch {b}" for b in batches])

ax.set_ylabel("Accuracy")
ax.set_title("Accuracy - ReLU vs GELU - jeu de test")



# === 5) Légende en bas à droite ===
legend_handles = [
    Patch(facecolor="cornflowerblue", label="ReLU"),
    Patch(facecolor="lightcoral", label="GELU"),
]
ax.legend(handles=legend_handles, loc="lower right")

plt.tight_layout()
plt.show()