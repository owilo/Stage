import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

def compute_confusion_matrix(cm, certainties, labels, filename, title_prefix=""):
    accuracy = np.trace(cm) / np.sum(cm)
    avg_certainty = np.mean(certainties)

    percentages = (cm / np.sum(cm, axis=1, keepdims=True)) * 100

    annot = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f"{percentages[i, j]:.1f}%"

    plt.figure(figsize=(10, 8))
    heatmap = sns.heatmap(percentages, annot=annot, fmt="", cmap="BuPu", xticklabels=labels, yticklabels=labels, vmin=0.0, vmax=100.0)

    cbar = heatmap.collections[0].colorbar
    cbar.ax.yaxis.set_major_formatter(mticker.PercentFormatter())

    plt.xlabel("Classe prédite", fontsize=12)
    plt.ylabel("Classe cible", fontsize=12)
    plt.suptitle(title_prefix, fontsize=18)
    plt.title(f"Précision : {accuracy:.2%} - Certitude moyenne : {avg_certainty:.2%}", fontsize=14)
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    plt.close()
    print(f"Matrice de confusion '{filename}' sauvegardée")