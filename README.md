# Influence de la fonction d'activation sur les performances d'un réseau de neurones

On s'est intéressé dans ce projet à la question de l'incidence du choix de la fonction d'activation d'un réseau de neurones sur ses performances. 

Parmi les nombreuses manières de caractériser la performance d'un modèle d'apprentissage, nous nous sommes penchés sur trois critères:

* La tendance à l'*overfitting*, ici mesurée d'une part par la différence entre la valeur finale de la fonction de perte sur les jeux de données d'entraînement et de validation, et d'autre part par la remontée éventuelle de la valeur de la fonction de perte au cours de la validation.
* L'erreur moyenne de prédiction sur le jeu de données de test.
* La vitesse de convergence du réseau, mesurée ici par le nombre d'époques moyen nécessaire pour que la fonction de perte atteigne son minimum sur le jeu de données de validation.

## 1. Méthodologie

Afin de réduire la complexité du problème et le nombre de tests, nous nous sommes limités à deux fonctions d'activation d'usage courant, à savoir la fonction $$\mathrm{ReLU}: x \mapsto \max(0, x)$$ et la fonction $$\mathrm{GELU}: x \mapsto \frac{x}{\sqrt{2\pi}}\int_{-\infty}^x e^{-t^2/2}dt.$$

Par ailleurs, afin de diminuer le risque de mesurer des effets liés aux modèles, nous avons comparé ces fonctions d'activation sur plusieurs architectures de réseaux de neurones que nous détaillerons plus loin, destinées à résoudre des problèmes de natures différentes.

Pour chaque architecture, nous avons fait varier les tailles de *batches* (entre 64 et 2048), c'est-à-dire le nombre d'échantillons testés avant la mise à jour des poids du réseau, la valeur de la graine du générateur aléatoire (entre 1, 2, et 3), ainsi que la présence ou non de *batch-normalization*.

Chaque architecture a été entraînée sur un nombre fixé (généralement 100) d'*epochs* pour chaque combinaison des paramètres (taille de *batches*, graine, fonction d'activation, batch-normalization), pour un total de 24 entraînements par modèle. 

## 2. Description des problèmes et architectures de test

On a comparé nos fonctions d'activation sur cinq réseaux différents:
* Deux réseaux denses effectuant des tâches de régression
* Deux réseaux convolutionnels effectuant des tâches de classification d'images
* Un réseau récurrent LSTM effectuant une tâche de régression.

### 2.1 Prédiction du prix de commandes de boissons

Dans cette tâche de régression, on tente de prédire le prix d'une commande de boissons en fonction d'un certain nombre de ses attributs (prix unitaire, réduction, nombre d'unités de la commande) en se basant sur la base de données suivante: https://www.kaggle.com/datasets/sebastianwillmann/beverage-sales. L'exemple est quelque peu artificiel puisque la quantité à prédire est une fonction explicite (quoique non linéaire) des attributs en question: prix total = (nombre d'unités) * (prix unitaire) * (1-réduction), mais on peut l'interpréter comme une tentative d'approcher cette fonction par un réseau de neurones.

On utilise ici un réseau dense à deux couches cachées (avec en sortie de chaque couche la fonction d'activation choisie):
* Une première couche à 64 neurones
* Une seconde couche à 32 neurones
* Une couche de sortie à un unique neurone.


### 2.2 Prédiction des notes d'examens d'une cohorte d'étudiants

Dans cette tâche de régression, on tente de prédire la note finale d'étudiants à un examen en fonction d'une collection de données qualitatives (genre, qualité d'enseignement, accompagnement parental, handicap, etc.) et quantitatives (notes précédentes, nombre d'heures d'études durant le semestre, nombre d'heures de sommeil, etc.) en se basant sur la base de données suivante: https://www.kaggle.com/datasets/minahilfatima12328/performance-trends-in-education/data.

On utilise pour ce faire un réseau dense à trois couches cachées:
* Une première couche à 128 neurones
* Une seconde couche à 64 neurones
* Une troisième couche à 32 neurones
* Une couche de sortie à un unique neurone.


### 2.3 Classification des images du dataset *MNIST*

Dans cette tâche de classification, on tente de reconnaître les chiffres écrits à la main dans les images du célèbre jeu de données *MNIST* (https://www.kaggle.com/datasets/hojjatk/mnist-dataset).

On utilise pour ce faire une architecture constituée d'un réseau convolutionnel comportant:
* Deux couches de convolution 2d comportant 64 filtres de taille 3x3 avec un *zero-padding* de 1
* Une couche de *max pooling* 2x2 avec un *stride* de 2
* Une couche de convolution 2d comportant 128 filtres de taille 3x3 avec un *zero-padding* de 1
* Une couche de *max pooling* 2x2 avec un *stride* de 2

suivi d'un réseau dense à deux couches cachées:
* Une couche *Flatten*
* Une couche cachée à 256 neurones
* Une couche cachée à 128 neurones
* Une couche de sortie à 10 neurones.

On applique une régularisation par *dropout*.

### 2.4 Classification d'images d'animaux

Dans cette tâche, on tente d'assigner à des photos d'animaux réparties en 10 classes la classe à laquelle elles appartiennent, en s'appuyant sur la base de données suivante https://www.kaggle.com/datasets/alessiocorrado99/animals10.

On utilise pour ce faire une architecture constituée d'un réseau convolutionnel comportant:
* Trois blocs composés d'une couche de convolution 2d comportant 32, 64, puis 128 filtres de taille 3x3 avec un *zero-padding* de 1, suivie d'une couche de *max-pooling* 2x2 avec un *stride* de 2
* Une couche d'*adaptive average pooling* de taille de sortie (1, 1)

suivi d'un réseau dense à une couche cachée:
* Une couche *Flatten*
* Une couche cachée à 256 neurones
* Une couche de sortie à 10 neurones.

On applique une régularisation par *dropout*.


### 2.5 Prédiction de l'évolution des prix de l'alimentation

Dans cette tâche de régression, on tente de prédire les valeurs de l'inflation en fonction du pays et des valeurs passées en se basant sur la base de données suivante: https://www.kaggle.com/datasets/umitka/food-price-inflation/data

Contrairement aux tâches précédentes, les données d'entrée ont subi un pré-traitement important pour enrichir l'information temporelle (feature engineering). Pour chaque pas de temps, le modèle reçoit non seulement la valeur brute, mais également :

* Des variables calendaires encodées de manière cyclique (sinus et cosinus du jour de l'année, du mois et du jour de la semaine) pour capturer la saisonnalité.

* Des valeurs retardées (lags) à différents intervalles (1 jour, 1 semaine, 1 mois).

* Des statistiques glissantes (rolling statistics) sur différentes fenêtres (moyenne, écart-type, min/max sur 7, 14 et 30 jours).

Les données sont structurées en fenêtres glissantes (sliding windows) d'une taille fixée (par défaut 20 pas de temps) pour prédire la valeur suivante.

On utilise pour ce faire une architecture récurrente de type LSTM (Long Short-Term Memory), configurée comme suit :

* Un bloc récurrent composé de 2 couches LSTM empilées (taille cachée de 64), permettant de capturer les dépendances temporelles à long terme. On ne conserve que l'état caché du dernier pas de temps de la séquence.

* Un réseau dense (tête de régression) composé de trois couches linéaires :

* Une couche transformant la sortie du LSTM (64 vers 64 neurones), suivie d'une normalisation (Batch Normalization optionnelle) et de la fonction d'activation.

* Une couche intermédiaire réduisant la dimension (64 vers 32 neurones), suivie d'une normalisation et de la fonction d'activation.

* Une couche de sortie linéaire à un unique neurone pour la prédiction finale.

Afin de lutter contre le surapprentissage, une régularisation par dropout (avec un taux de 0.2) est appliquée entre les couches LSTM ainsi qu'après les fonctions d'activation des couches denses. L'initialisation des poids a été réalisée via les méthodes de Xavier (pour les couches linéaires) et orthogonale (pour les poids récurrents) afin de favoriser la convergence.

## 3. Description des résultats obtenus

On décrit dans la suite, pour chaque critère, les performances obtenues sur chacun des problèmes de test suivant la fonction d'activation. Du fait d'un temps d'entraînement prohibitif, nous n'avons pas pu obtenir les résultats du problème de classification d'images d'animaux.

### 3.1 Vitesse de convergence

On trace ci-dessous les courbes de la moyenne de la *validation loss* en fonction des époques.  Hormis sur le problème de prédiction de l'inflation (où le réseau n'apprend tout simplement pas, à en juger par les courbes de *validation loss*), les réseaux entraînés avec l'une et l'autre des fonctions d'activation ne présentent pas de différences notables de vitesse de convergence, définie encore une fois comme le nombre d'époques nécessaire pour que la *validation loss* atteigne son minimum.

#### Prédiction du prix des commandes de boissons
![Boissons](./plots/brevage_val_loss_curves.png)
![Boissons](./plots/brevage_convergence_speed.png)
#### Prédiction des notes d'examens
![Notes](./plots/Student_val_loss_curves.png)
![Notes](./plots/Student_convergence_speed.png)
#### Classification des images MNIST
![MNIST](./plots/MNIST_val_loss_curves.png)
![MNIST](./plots/MNIST_convergence_speed.png)
#### Prédiction de l'inflation
![Inflation](./plots/Food_Price_val_loss_curves.png)
![Inflation](./plots/Food_Price_convergence_speed.png)

### 3.2 *Overfitting*

On trace ci-dessous les box plots des scores d'overfitting, à savoir la différence entre la fonction perte finale sur le jeu d'apprentissage et le jeu de validation, et la remontée de la fonction de perte au cours de la validation. Bien que les deux fonctions soient relativement proches, on semble déceler une tendance plus grande à l'overfitting chez les réseaux munis de la fonction GELU.


#### Prédiction du prix des commandes de boissons
![Boissons](./plots/brevage_overfitting_score.png)
![Boissons](./plots/brevage_generalization_gap.png)

#### Prédiction des notes d'examens
![Notes](./plots/Student_overfitting_score.png)
![Notes](./plots/Student_generalization_gap.png)

#### Classification des images MNIST
![MNIST](./plots/MNIST_overfitting_score.png)
![MNIST](./plots/MNIST_generalization_gap.png)

#### Prédiction de l'inflation
![Inflation](./plots/Food_Price_overfitting_score.png)
![Inflation](./plots/Food_Price_generalization_gap.png)


### 3.3 Erreur de prédiction sur le test

On trace ci-dessous les box plots des moyennes des erreurs finales de prédiction sur les jeux de données de test, mesurée pour le problème de classification par l'entropie croisée et l'*accuracy*, et pour les tâches de régression par la *mean square error* et la *mean average error*. Il est encore une fois difficile de déceler une tendance.

#### Prédiction du prix des commandes de boissons
##### Erreur L1
![Boissons](./plots/brevage_mae_mean_.png)
![Boissons](./plots/brevage_mae_mean_by%20batch%20norm.png)
![Boissons](./plots/brevage_mae_mean_by%20batch%20size.png)

##### Erreur L2
![Boissons](./plots/brevage_mse_mean_.png)
![Boissons](./plots/brevage_mse_mean_by%20batch%20norm.png)
![Boissons](./plots/brevage_mse_mean_by%20batch%20size.png)

#### Prédiction des notes d'examens
##### Erreur L1
![Boissons](./plots/Student_mae_mean_.png)
![Boissons](./plots/Student_mae_mean_by%20batch%20norm.png)
![Boissons](./plots/Student_mae_mean_by%20batch%20size.png)

##### Erreur L2
![Boissons](./plots/Student_mse_mean_.png)
![Boissons](./plots/Student_mse_mean_by%20batch%20norm.png)
![Boissons](./plots/Student_mse_mean_by%20batch%20size.png)

#### Classification des images MNIST
##### Accuracy
![Boissons](./plots/MNIST_accuracy_.png)
![Boissons](./plots/MNIST_accuracy_by%20batch%20norm.png)
![Boissons](./plots/MNIST_accuracy_by%20batch%20size.png)

##### Cross-entropy
![Boissons](./plots/MNIST_cross_entropy_.png)
![Boissons](./plots/MNIST_cross_entropy_by%20batch%20norm.png)
![Boissons](./plots/MNIST_cross_entropy_by%20batch%20size.png)

#### Prédiction de l'inflation
##### Erreur L1
![Boissons](./plots/Food_Price_mae_mean_.png)
![Boissons](./plots/Food_Price_mae_mean_by%20batch%20norm.png)
![Boissons](./plots/Food_Price_mae_mean_by%20batch%20size.png)

##### Erreur L2
![Boissons](./plots/Food_Price_mse_mean_.png)
![Boissons](./plots/Food_Price_mse_mean_by%20batch%20norm.png)
![Boissons](./plots/Food_Price_mse_mean_by%20batch%20size.png)


## 4. Conclusion

Nos tests ne démontrent une tendance claire en termes de performance entre les réseaux munis de la fonction d'activation ReLU ou de la fonction GELU, les différences entre les résultats obtenus étant minimes et différant selon les problèmes traités. Il ne semble pas que le choix de fonction d'activation, au moins parmi ces deux exemples, soit un paramètre particulièrement important.