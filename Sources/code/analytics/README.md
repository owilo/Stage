# Analytics - Analyse des résultats

Ce dossier contient plusieurs scripts Python dédiés à l'analyse des caractéristiques latentes et des transformations appliquées aux données.

## Fichiers et descriptions

### `characteristic_distribution.py`
Ce script analyse la distribution des dimensions latentes générées par l'autoencodeur. Il calcule des statistiques comme les scores KS (Kolmogorov-Smirnov) et les valeurs p pour évaluer si les dimensions suivent une distribution normale.

### `characteristics_interdependancy.py`
Ce script explore les relations entre les dimensions latentes. Il examine comment la variation d'une dimension (par exemple, `z0`) influence les moyennes conditionnelles des autres dimensions.

### `centroid_boxplot.py`
Ce script génère des boxplots pour visualiser la distribution des dimensions latentes par classe. Il permet de comparer les caractéristiques latentes des différentes classes du dataset MNIST.

### `centroid_distance.py`
Ce script calcule les distances euclidiennes et cosinus entre les centroïdes des classes latentes et les points latents transformés et les affiche sous forme de matrices de distances et des heatmaps.

### `covariance_plots.py`
Ce script calcule et visualise la matrice de covariance des dimensions latentes. Il permet d'identifier les corrélations entre les dimensions latentes.

### `latent_ANOVA.py`
Ce script utilise une analyse ANOVA pour évaluer le pouvoir discriminant des dimensions latentes. Il génère des scores F pour chaque dimension et produit un graphique pour visualiser les résultats.

### `trace_confusion.py`
Ce script analyse les performances des classifieurs et détecteurs de traces en générant des matrices de confusion. Il évalue également la classification des données transformées et reconstruites.

### `translation_distributions.py`
Ce script examine les distributions des dimensions latentes avant et après la transformation entre classes. Il génère des boxplots pour comparer les distributions des dimensions transformées.

### `translation_plots.py`
Ce script visualise les effets des transformations latentes sur les données. Il montre les images sources, transformées et reconstruites, ainsi que les translations appliquées dans l'espace latent.