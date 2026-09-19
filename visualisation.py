import numpy as np
import matplotlib.pyplot as plt
from kmeans import NueesDynamiques
from evaluation import methode_coude

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3


def graphique_clusters():
    rng = np.random.default_rng(42)
    c1 = rng.normal([0, 0], 1, (100, 2))
    c2 = rng.normal([5, 5], 1, (100, 2))
    c3 = rng.normal([0, 6], 1, (100, 2))
    X = np.vstack([c1, c2, c3])

    model = NueesDynamiques(k=3, random_state=42, init='kmeans++')
    model.fit(X)

    plt.figure(figsize=(7, 5))
    couleurs = ['#e74c3c', '#3498db', '#2ecc71']
    for j in range(3):
        pts = X[model.labels == j]
        plt.scatter(pts[:, 0], pts[:, 1], c=couleurs[j],
                    alpha=0.6, s=30, label=f'Cluster {j+1}')
    plt.scatter(model.centres[:, 0], model.centres[:, 1],
                c='black', marker='X', s=250, edgecolors='white',
                linewidths=2, label='Centres', zorder=10)
    plt.title("Clusters obtenus (K = 3)", fontsize=13, fontweight='bold')
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig('figures/clusters.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("figures/clusters.png")


def graphique_convergence():
    rng = np.random.default_rng(42)
    X = np.vstack([
        rng.normal([0, 0], 1, (100, 2)),
        rng.normal([5, 5], 1, (100, 2)),
        rng.normal([0, 6], 1, (100, 2)),
    ])

    model = NueesDynamiques(k=3, random_state=42, init='kmeans++')
    model.fit(X)

    plt.figure(figsize=(7, 4.5))
    plt.plot(range(1, len(model.historique_inertie) + 1),
             model.historique_inertie,
             'o-', color='#2980b9', lw=2, markersize=8)
    plt.xlabel("Itération")
    plt.ylabel("Inertie intra-classe")
    plt.title("Convergence de l'inertie", fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('figures/convergence.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("figures/convergence.png")


def graphique_coude():
    from sklearn.datasets import load_iris
    X = load_iris().data

    ks, inerties = methode_coude(X, range(1, 9), seed=42)

    plt.figure(figsize=(7, 4.5))
    plt.plot(ks, inerties, 'o-', color='#e67e22', lw=2, markersize=8)
    plt.axvline(x=3, color='red', linestyle='--',
                alpha=0.7, label='K optimal = 3')
    plt.xlabel("Nombre de clusters K")
    plt.ylabel("Inertie")
    plt.title("Méthode du coude — Dataset Iris",
              fontsize=13, fontweight='bold')
    plt.legend()
    plt.tight_layout()
    plt.savefig('figures/coude.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("figures/coude.png")


if __name__ == "__main__":
    import os
    os.makedirs('figures', exist_ok=True)
    graphique_clusters()
    graphique_convergence()
    graphique_coude()