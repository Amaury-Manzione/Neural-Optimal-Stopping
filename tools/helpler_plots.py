import os

import matplotlib.pyplot as plt
import torch


def save_model(model, dir_model, filename):
    repertoire_ = "modèle" + "\\" + dir_model
    if not os.path.exists(repertoire_):
        os.makedirs(repertoire_)

    model_path = os.path.join(repertoire_, filename)
    torch.save(model.state_dict(), model_path)


def save_fig(dir_model, filename):
    repertoire = "graphes" + "\\" + dir_model
    if not os.path.exists(repertoire):
        os.makedirs(repertoire)

    # Chemin complet pour enregistrer le graphique
    chemin_fichier = os.path.join(repertoire, filename)

    # Sauvegarde du graphique
    plt.savefig(chemin_fichier)
    plt.close()
