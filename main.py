import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from tqdm import tqdm

import black_scholes as bs
import nos_class_multi_dimensional as nos_class_multi_dimensional

#######################################################################################

#                     2-DIMENSIONAL MAXCALL OPTION

#######################################################################################
d = 2
n_path = 500000
batch_size = 1000
n_simulation = 50
epochs = 100
spot = 100
sigma = 0.2
rate = 0.05
strike = 100
dividend = 0.1
maturity = 3
asset = [bs.BlackScholes(spot, rate, sigma, dividend) for _ in range(2)]


def payoff(x):
    intrinsic_value = torch.max(x) - strike
    return torch.maximum(intrinsic_value, torch.tensor(0))


def change_coordinates(x):
    return torch.max(x)


epsilon = 5
l_girsanov = (rate - dividend + 0.01 * np.log(d)) / 0.2


learning_rate = 1e-2
mlp = nos_class_multi_dimensional.NeuralNetworkNos(d + 1, strike, nn.ReLU())

nos = nos_class_multi_dimensional.NosMultiDimensional(
    asset,
    maturity,
    rate,
    payoff,
    change_coordinates,
    mlp,
)

sharp_region, history, time_taken = nos.find_optimal_region(
    n_simulation,
    n_path,
    epsilon,
    l_girsanov,
    batch_size,
    learning_rate,
    epochs,
    verbose=False,
)

repertoire_ = "modèle"
if not os.path.exists(repertoire_):
    os.makedirs(repertoire_)

model_filename = "sharp_region.pth"
model_path = os.path.join(repertoire_, model_filename)
torch.save(sharp_region.state_dict(), model_path)


plt.plot(history, label=f"learning rate : {learning_rate}")
plt.ylabel("loss function")
plt.xlabel("epochs")
plt.legend()

repertoire = "graphes"
if not os.path.exists(repertoire):
    os.makedirs(repertoire)

# Chemin complet pour enregistrer le graphique
chemin_fichier = os.path.join(repertoire, "graph_history_loss.png")

# Sauvegarde du graphique
plt.savefig(chemin_fichier)
plt.close()


def plot_region(t):

    t = 0
    x = np.linspace(50, 200, 100)
    y = np.linspace(50, 200, 100)
    X, Y = np.meshgrid(x, y)

    def f(x, y):
        alpha_X = max(x, y)
        input_to_tensor = torch.tensor([t, x / alpha_X, y / alpha_X]).unsqueeze(0)
        return sharp_region(input_to_tensor).detach().numpy()[0]

    # Evaluate the function on the grid
    Z = np.zeros_like(X)
    for i in tqdm(range(X.shape[0])):
        for j in range(X.shape[1]):
            Z[i, j] = f(X[i, j], Y[i, j])

    # Determine the regions based on the condition f(x, y) <= max(x, y)
    condition = Z <= np.maximum(X, Y)

    # Plot the results
    plt.figure(figsize=(8, 6))
    plt.contourf(X, Y, condition, levels=[-0.5, 0.5, 1.5], colors=["blue", "red"])
    plt.colorbar()
    plt.title(f"t = {t}")
    plt.xlabel("x")
    plt.ylabel("y")

    name_file = f"graph_t{t}.png"
    chemin_fichier = os.path.join(repertoire, name_file)
    plt.savefig(chemin_fichier)
    plt.close()


plot_region(0)
plot_region(2)
plot_region(3)
