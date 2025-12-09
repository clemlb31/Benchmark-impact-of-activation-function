import json

import numpy as np

import matplotlib.pyplot as plt

from matplotlib.patches import Patch



# =========================

# 1) Chargement des données

# =========================

with open("mnist_histories.json", "r") as f:

    runs = json.load(f)



# tolérance pour définir la "convergence" (5% au-dessus du minimum)

TOL = 0.05



# couleurs pastel cohérentes partout

COLOR_RELU = "#7FA6D5"   # bleu pastel

COLOR_GELU = "#E99690"   # rouge pastel



# ==========================================

# 2) Calcul des époques de convergence par run

#    + séparation des courbes val_loss

# ==========================================

conv_epochs = {"relu": [], "gelu": []}

relu_curves = []

gelu_curves = []



for r in runs:

    act = r["activation_function"].lower()   # "relu" ou "gelu"

    val_loss = np.array(r["val_loss"])



    # stocker la courbe pour la partie "moyenne"

    if act == "relu":

        relu_curves.append(val_loss)

    elif act == "gelu":

        gelu_curves.append(val_loss)



    # calcul de l'époque de convergence (si on a une courbe)

    if len(val_loss) == 0 or act not in conv_epochs:

        continue



    best = val_loss.min()

    thresh = best * (1 + TOL)



    # première époque où val_loss <= seuil

    idx_candidates = np.where(val_loss <= thresh)[0]

    if len(idx_candidates) == 0:

        # si jamais ça n'atteint jamais le seuil, on prend la dernière époque

        epoch = len(val_loss)

    else:

        epoch = int(idx_candidates[0]) + 1  # epochs en 1-based



    conv_epochs[act].append(epoch)



print("Époques de convergence (ReLU) :", conv_epochs["relu"])

print("Époques de convergence (GELU) :", conv_epochs["gelu"])



# ================================

# 3) Boxplot vitesse de convergence

# ================================

data_box = [conv_epochs["relu"], conv_epochs["gelu"]]

labels_box = ["ReLU", "GELU"]

colors_box = [COLOR_RELU, COLOR_GELU]



fig1, ax1 = plt.subplots(figsize=(6, 4))



bp = ax1.boxplot(

    data_box,

    patch_artist=True,

)



# couleurs + bordures noires

for i, box in enumerate(bp["boxes"]):

    box.set(facecolor=colors_box[i], edgecolor="black",

            linewidth=1.3, alpha=0.8)



for median in bp["medians"]:

    median.set(color="black", linewidth=1.3)

for whisker in bp["whiskers"]:

    whisker.set(color="black", linewidth=1.0)

for cap in bp["caps"]:

    cap.set(color="black", linewidth=1.0)

for flier in bp["fliers"]:

    flier.set(markeredgecolor="black", markerfacecolor="white",

              markersize=4)



ax1.set_xticklabels(labels_box)

ax1.set_ylabel("Époque de convergence (val_loss)")

ax1.set_title(f"Vitesse de convergence (tolérance {int(TOL*100)}%)")



ax1.yaxis.grid(True, linestyle="--", linewidth=0.4,

               color="lightgray", alpha=0.6)



legend_handles = [

    Patch(facecolor=COLOR_RELU, label="ReLU"),

    Patch(facecolor=COLOR_GELU, label="GELU"),

]

ax1.legend(handles=legend_handles, loc="lower right",

           title="Activation")



plt.tight_layout()

plt.show()



# ======================================

# 4) Courbes moyennes de validation loss

# ======================================

relu_curves = [np.array(c) for c in relu_curves]

gelu_curves = [np.array(c) for c in gelu_curves]



# sécurité : si une des listes est vide, on évite le crash

if len(relu_curves) > 0:

    relu_mean = np.mean(relu_curves, axis=0)

    relu_std = np.std(relu_curves, axis=0)

else:

    relu_mean = relu_std = None



if len(gelu_curves) > 0:

    gelu_mean = np.mean(gelu_curves, axis=0)

    gelu_std = np.std(gelu_curves, axis=0)

else:

    gelu_mean = gelu_std = None



# on suppose que toutes les courbes ont la même longueur

n_epochs = len(relu_curves[0]) if len(relu_curves) > 0 else len(gelu_curves[0])

epochs = np.arange(1, n_epochs + 1)



fig2, ax2 = plt.subplots(figsize=(8, 4))



if relu_mean is not None:

    ax2.plot(epochs, relu_mean, label="ReLU", color=COLOR_RELU)

    ax2.fill_between(epochs, relu_mean - relu_std, relu_mean + relu_std,

                     color=COLOR_RELU, alpha=0.2)



if gelu_mean is not None:

    ax2.plot(epochs, gelu_mean, label="GELU", color=COLOR_GELU)

    ax2.fill_between(epochs, gelu_mean - gelu_std, gelu_mean + gelu_std,

                     color=COLOR_GELU, alpha=0.2)



ax2.set_xlabel("Époques")

ax2.set_ylabel("Validation loss")

ax2.set_title("Courbes moyennes de validation loss")

ax2.yaxis.grid(True, linestyle="--", linewidth=0.4,

               color="lightgray", alpha=0.6)

ax2.legend(loc="upper right")



plt.tight_layout()

plt.show()