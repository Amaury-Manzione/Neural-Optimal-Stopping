import os
import sys

import matplotlib.pyplot as plt
import nos_class_multi_dimensional as nos_class_multi_dimensional
import numpy as np
import torch
from torch import nn
from tqdm import tqdm

sys.path.append("..\..")
import src.diffusion.black_scholes as bs
import tools.helpler_plots as helper_plots

#######################################################################################

#                     2-DIMENSIONAL MAXCALL OPTION

#######################################################################################
d = 2
n_path = 10000
batch_size = 500
n_simulation = 50
epochs = 200
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

helper_plots.save_model(sharp_region, "nos", "sharp_region.pth")
