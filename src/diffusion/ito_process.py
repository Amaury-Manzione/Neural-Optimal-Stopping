from typing import Callable

import numpy as np
from scipy import linalg


class ItoProcess:
    """Object represents an Ito process:

    dX_t = b(X_t)dt + s(X_t)dW_t
    """

    def __init__(self, x0: float, drift: Callable, vol: Callable):
        """_summary_

        Parameters
        ----------
        x0 : float
            initial value
        drift : Callable
            drift_part
        vol : Callable
            volatility_part
        """
        self.x0 = x0
        self.drift = drift
        self.vol = vol

    def get_path(self, n: int, dt: float, N: int, seed=None) -> np.array:
        """
        Parameters
        ----------
        n : int
            number of simulation
        dt : float
            time step
        N : int
            number of paths simulated
        seed : None,optional
            fix randomness, by default None
        Returns
        -------
        np.array
            return an Euler-Maruyama simulation of X_t :
        X_i+1 = X_i + b(X_i)dt + sigma(X_i)(BM[i],BM[i-1])
        """
        if seed != None:
            np.random.seed(seed)

        process = np.zeros((N, n + 1))
        BM = np.random.normal(0, np.sqrt(dt), (N, n + 1))

        process[:, 0] = self.x0

        for i in range(1, n + 1):
            process[:, i] = (
                process[:, i - 1]
                + self.drift(process[:, i - 1]) * dt
                + self.vol(process[:, i - 1]) * BM[:, i - 1]
            )

        return process

    def get_path_tensor(self, n: int, dt: float, N: int, seed=None):
        pass

    def get_path_correlated(
        self, n: int, dt: float, N: int, corr_matrix: np.array, seed=None
    ) -> np.array:
        """
        Parameters
        ----------
        n : int
            number of simulation
        dt : float
            time step
        N : int
            number of paths simulated
        corr_matrix : np.array
            correlation matrix
        seed : None,optional
            fix randomness, by default None
        Returns
        -------
        np.array
            return an Euler-Maruyama simulation of X_t :
        X_i+1 = X_i + b(X_i)dt + sigma(X_i)(BM[i],BM[i-1])
        """
        if seed != None:
            np.random.seed(seed)

        process = np.zeros((N, n + 1))
        L = np.linalg.cholesky(corr_matrix)
        BM = np.random.normal(0, np.sqrt(dt), (N, n + 1))
        for i in range(n + 1):
            BM[:, i] = np.matmul(L, BM[:, i])

        process[:, 0] = self.x0

        for i in range(1, n + 1):
            process[:, i] = (
                process[:, i - 1]
                + self.drift(process[:, i - 1]) * dt
                + self.vol(process[:, i - 1]) * BM[:, i - 1]
            )

        return process

    def get_path_importance_sampling(
        self, n: int, dt: float, N: float, l: float, seed=None
    ):
        return

    def get_path_importance_sampling_multi_dim(self):
        pass
