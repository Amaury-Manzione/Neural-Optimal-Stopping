import sys
import time
from typing import Callable, List

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

import ito_process as ito

torch.autograd.set_detect_anomaly(True)

########################################################################################

#                           CREATING THE NEURAL NETWORK

########################################################################################


class NeuralNetworkNos(nn.Module):
    """Multi-layer perceptron

    Parameters
    ----------
    nn : _type_
        neural network
    """

    def __init__(self, d: int, last_bias: int, activation_function, num_layers=1):
        """creating the NN for NOS algorithm

        Parameters
        ----------
        d : int
            dimension of the input
        batch_size : int
            size of the batches
        epochs : int
            number of epoch for training the NN
        learning_rate : float
            learning rate for gradient descent
        """
        super().__init__()
        self.last_bias = last_bias
        self.activation_function = activation_function
        self.num_layers = num_layers

        layers = []

        # Add first hidden layer
        layers.append(nn.Linear(d, 20 + d))
        layers.append(activation_function)

        # Add additional hidden layers
        for _ in range(num_layers):
            layers.append(nn.Linear(20 + d, 20 + d))
            layers.append(activation_function)
            # layers.append(nn.BatchNorm1d(20))

        layers.append(nn.Linear(20 + d, 1))

        self.linear_relu_stack = nn.Sequential(*layers)

        self.double()
        self.init_weights(last_bias)

    def init_weights(self, last_bias: int):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Initialize linear layer weights using Xavier initialization
                torch.nn.init.xavier_uniform_(m.weight)
                if isinstance(self.linear_relu_stack[-1], nn.Linear):
                    torch.nn.init.constant_(self.linear_relu_stack[-1].bias, last_bias)

    def forward(self, x):
        output = self.linear_relu_stack(x)
        return output


########################################################################################

#                           NOS CLASS

########################################################################################


class NosMultiDimensional:
    """class for Neural Optimal Stopping algorithm"""

    def __init__(
        self,
        asset: List[ito.ItoProcess],
        maturity: float,
        rate: float,
        payoff: Callable,
        change_coordinates: Callable,
        mlp: NeuralNetworkNos,
    ):
        """creating object of type NOS

        Parameters
        ----------
        asset : List[ito.ItoProcess]
            underlying for the option
        maturity : float
            maturity of the option
        rate : float
            interest rate
        payoff : Callable
            payoff of the option
        change_coordinates : Callable
            change of coordinates to apply for multi-dimensional asset.
        n_simulation : int
            number of points for discretizing the process
        """
        self.asset = asset
        self.maturity = maturity
        self.rate = rate
        self.payoff = payoff
        self.mlp = mlp
        self.change_coordinates = change_coordinates

    def find_optimal_region(
        self,
        n_simulation: int,
        n_path: int,
        epsilon: float,
        l_girsanov: float,
        batch_size: int,
        learning_rate: float,
        epochs: int,
        verbose=True,
    ):
        """_summary_

        Returns
        -------
        _type_
            _description_
        """
        t_init = time.time()

        # dimension
        d = len(self.asset)

        # timestep
        dt = float(self.maturity / n_simulation)
        timestep = torch.tensor([i * dt for i in range(n_simulation)])

        # history of losses per epoch
        history = torch.zeros(epochs)

        # optimizer and scheduler
        optimizer = torch.optim.Adam(
            self.mlp.parameters(), lr=learning_rate, maximize=True
        )

        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.02)
        paths, Z = self.asset[0].get_path_importance_sampling_test(
            d, n_simulation, dt, n_path, l_girsanov, d
        )
        paths, Z = torch.tensor(paths, requires_grad=False, dtype=float), torch.tensor(
            Z, requires_grad=False, dtype=float
        )

        list_indexs = np.array([i for i in range(n_path)])

        for epoch in tqdm(range(epochs), disable=verbose):

            index = np.random.choice(list_indexs, size=batch_size)
            paths_batch = paths[index, :, :]

            loss = torch.tensor(0.0)

            stopping_budgets = torch.ones(batch_size)

            for idx in range(len(timestep) - 1):
                tensors = []

                time_tensor = timestep[idx] * torch.ones(batch_size, dtype=torch.float)

                tensors.append(time_tensor)

                X = torch.squeeze(paths_batch[:, idx, :])
                alpha_X = torch.tensor(
                    [self.change_coordinates(row) for row in X], dtype=torch.float
                )

                for i in range(d):
                    tensors.append(X[:, i] / alpha_X)

                input_ = torch.stack(tensors, dim=1)

                output = self.mlp(
                    input_,
                ).squeeze()

                stopping_probabilities = torch.minimum(
                    torch.maximum(
                        (-output + alpha_X + epsilon) / (2 * epsilon),
                        torch.tensor(0),
                    ),
                    torch.tensor(1),
                )

                new_stop_budg = stopping_budgets * (1 - stopping_probabilities)

                payoff = torch.tensor(
                    [self.payoff(row) for row in X], dtype=torch.float
                )
                payoff = payoff * torch.exp(-self.rate * timestep[idx])

                loss = loss + torch.sum(
                    stopping_probabilities * stopping_budgets * payoff * Z[index, idx]
                )

                stopping_budgets = new_stop_budg

            payoff = torch.tensor(
                [self.payoff(row) for row in torch.squeeze(paths_batch[:, -1, :])],
                dtype=torch.float,
            )
            payoff = payoff * torch.exp(torch.tensor(-self.rate * self.maturity))

            stopping_probabilities = torch.ones(batch_size)

            loss = loss + torch.sum(
                stopping_probabilities * stopping_budgets * payoff * Z[index, -1]
            )

            loss = loss / batch_size

            history[epoch] = loss.item()

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

        t = time.time() - t_init

        return self.mlp, history, t

    def get_price(self, sharp_region, n_simulation: int, n_mc: int, verbose=True):
        """Get price of bermudan option given the trained region.

        Parameters
        ----------
        sharp_region : _type_
            trained stop region.
        n_simulation : int
            number of points for discretizing the process.
        n_mc : int
            number of monte-carlo scenarios.
        """
        # timestep
        dt = float(self.maturity / n_simulation)
        timesteps = torch.tensor([i * dt for i in range(n_simulation)])

        sharp_regions = sharp_region(torch.unsqueeze(timesteps, 1))

        paths = self.asset[0].get_path(n_simulation, dt, n_mc)
        paths = torch.tensor(paths)

        price = torch.zeros(n_mc)

        for n in tqdm(range(n_mc), disable=verbose):
            for idx in range(n_simulation):
                if paths[n, idx] <= sharp_regions[idx]:
                    price[n] = self.payoff(paths[n, idx]) * np.exp(
                        -self.rate * dt * idx
                    )
                    break
            price[n] = self.payoff(paths[n, idx]) * np.exp(-self.rate * dt * idx)

        return torch.mean(price)
