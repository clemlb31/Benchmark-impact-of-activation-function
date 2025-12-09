import json

import numpy as np

import matplotlib.pyplot as plt

from matplotlib.patches import Patch



# =========================

# 1) Chargement des données

# =========================

with open("mnist_histories.json", "r") as f:

    runs = json.load(f)



# couleurs pastel cohérentes

COLOR_RELU = "#7FA6D5"   # bleu pastel

COLOR_GELU = "#E99690"   # rouge pastel



# ================================

# 2) Calcul des métriques d'overfitting

# ================================

generalization_gap = {"relu": [], "gelu": []}

overfit_score = {"relu": [], "gelu": []}



for r in runs:

    act = r["activation_function"].lower()   # "relu" ou "gelu"

    if act not in generalization_gap:

        continue



    train = np.array(r["train_loss"])

    val = np.array(r["val_loss"])



    # sécurité

    if len(train) == 0 or len(val) == 0:

        continue



    # --- 1) Generalization gap à la fin ---

    train_final = train[-1]

    val_final = val[-1]

    gap = float(val_final - train_final)

    generalization_gap[act].append(gap)



    # --- 2) Overfitting score (remontée de la val_loss) ---

    best_val = float(val.min())

    overfit = float(val_final - best_val)  # > 0 => la val a remonté

    overfit_score[act].append(overfit)



print("Generalization gap (ReLU) :", generalization_gap["relu"])

print("Generalization gap (GELU) :", generalization_gap["gelu"])

print("Overfitting score (ReLU)  :", overfit_score["relu"])

print("Overfitting score (GELU)  :", overfit_score["gelu"])



# =======================================

# 3) Boxplot du generalization gap (final)

# =======================================

data_gap = [generalization_gap["relu"], generalization_gap["gelu"]]

labels = ["ReLU", "GELU"]

colors = [COLOR_RELU, COLOR_GELU]



fig1, ax1 = plt.subplots(figsize=(6, 4))



bp1 = ax1.boxplot(

    data_gap,

    patch_artist=True,

)



for i, box in enumerate(bp1["boxes"]):

    box.set(facecolor=colors[i], edgecolor="black",

            linewidth=1.3, alpha=0.8)



for median in bp1["medians"]:

    median.set(color="black", linewidth=1.3)

for whisker in bp1["whiskers"]:

    whisker.set(color="black", linewidth=1.0)

for cap in bp1["caps"]:

    cap.set(color="black", linewidth=1.0)

for flier in bp1["fliers"]:

    flier.set(markeredgecolor="black", markerfacecolor="white",

              markersize=4)



ax1.set_xticklabels(labels)

ax1.set_ylabel("Generalization gap\n(val_final - train_final)")

ax1.set_title("Comparaison de l'overfitting (gap final)")

ax1.yaxis.grid(True, linestyle="--", linewidth=0.4,

               color="lightgray", alpha=0.6)



legend_handles = [

    Patch(facecolor=COLOR_RELU, label="ReLU"),

    Patch(facecolor=COLOR_GELU, label="GELU"),

]

ax1.legend(handles=legend_handles, loc="lower right",

           )



plt.tight_layout()

plt.show()



# =======================================

# 4) Boxplot de la remontée de val_loss

#    (overfitting score)

# =======================================

data_overfit = [overfit_score["relu"], overfit_score["gelu"]]



fig2, ax2 = plt.subplots(figsize=(6, 4))



bp2 = ax2.boxplot(

    data_overfit,

    patch_artist=True,

)



for i, box in enumerate(bp2["boxes"]):

    box.set(facecolor=colors[i], edgecolor="black",

            linewidth=1.3, alpha=0.8)



for median in bp2["medians"]:

    median.set(color="black", linewidth=1.3)

for whisker in bp2["whiskers"]:

    whisker.set(color="black", linewidth=1.0)

for cap in bp2["caps"]:

    cap.set(color="black", linewidth=1.0)

for flier in bp2["fliers"]:

    flier.set(markeredgecolor="black", markerfacecolor="white",

              markersize=4)



ax2.set_xticklabels(labels)

ax2.set_ylabel("Overfitting score\n(val_final - min(val))")

ax2.set_title("Remontée de la validation loss (surapprentissage)")

ax2.yaxis.grid(True, linestyle="--", linewidth=0.4,

               color="lightgray", alpha=0.6)



ax2.legend(handles=legend_handles, loc="lower right",

           )



plt.tight_layout()

plt.show()