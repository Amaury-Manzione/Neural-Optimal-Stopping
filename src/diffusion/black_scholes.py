import numpy as np
import torch

import src.diffusion.ito_process as ito_process


class BlackScholes(ito_process.ItoProcess):
    """Object represents a Black-Scholes process
    dX_t = X_t * r * dt + X_t * sigma * dW_t

    Parameters
    ----------
    ItoProcess : _type_
        mother class
    """

    def __init__(self, x0: float, r: float, sigma: float, dividend=0):
        """Construct object representing a Black-Scholes process namely:
        dX_t = X_t * r * dt + X_t * sigma * dW_t

        Parameters
        ----------
        x0 : float
            inital value
        r : float
            interest rate
        sigma : float
            volatility
        dividend : _type_, optional
            dividend
        """
        self.r = r
        self.sigma = sigma
        self.dividend = dividend
        super().__init__(
            x0, drift=lambda x: (r - dividend) * x, vol=lambda x: sigma * x
        )

    def get_path(self, n: int, dt: float, N: float, seed=None) -> np.array:
        """_summary_

        Parameters
        ----------
        n : int
            number of simulation
        dt : float
            time step
        N : float
            number of paths simulated
        seed : None
            for fixing randomness
        Returns
        -------
        np.array
        return the explicit simulation of X_t :
        X_i+1 = X_i*(b(X_i) - (sigma(X_i)^2)/2dt + sigma(X_i)(BM[i],BM[i-1]))
        """

        np.random.seed(seed)

        process = np.zeros((N, n))
        BM = np.random.normal(0, np.sqrt(dt), (N, n))

        process[:, 0] = self.x0

        for i in range(1, n):
            process[:, i] = process[:, i - 1] * np.exp(
                (self.r - self.dividend - 0.5 * self.sigma**2) * dt
                + self.sigma * BM[:, i - 1]
            )

        return process

    def get_path_tensor(self, n: int, dt: float, N: int, seed=None) -> torch.Tensor:
        """
        Simulate paths using explicit simulation of X_t :
        X_i+1 = X_i*(b(X_i) - (sigma(X_i)^2)/2*dt + sigma(X_i)*(BM[i],BM[i-1]))

        Parameters
        ----------
        n : int
            Number of time steps
        dt : float
            Time step size
        N : int
            Number of paths
        seed : None or int
            Seed for fixing randomness

        Returns
        -------
        torch.Tensor
            Tensor containing the simulated paths
        """

        if seed is not None:
            torch.manual_seed(seed)

        # Initialize the process tensor
        process = torch.zeros((N, n))

        # Brownian Motion increments
        BM = torch.normal(mean=0.0, std=torch.sqrt(torch.tensor(dt)), size=(N, n))

        # Set the initial value
        process[:, 0] = self.x0

        for i in range(1, n):
            process[:, i] = process[:, i - 1] * torch.exp(
                (self.r - self.dividend - 0.5 * self.sigma**2) * dt
                + self.sigma * BM[:, i - 1]
            )

        return process

    def get_path_correlated(
        self, n: int, dt: float, N: int, corr_matrix: np.array, seed=None
    ) -> np.array:
        """Return correlated paths with exact scheme

        Parameters
        ----------
        n : int
           number of simulation
        dt : float
            timestep
        N : int
            number of paths simulated
        corr_matrix : np.array
            correlation matrix
        seed : _type_, optional
            for fixing randomness, by default None

        Returns
        -------
        np.array
            _description_
        """

        np.random.seed(seed)

        process = np.zeros((N, n))
        L = np.linalg.cholesky(corr_matrix)

        BM = np.random.normal(0, np.sqrt(dt), (N, n))
        for i in range(n):
            BM[:, i] = np.matmul(L, BM[:, i])

        process[:, 0] = self.x0

        for i in range(1, n):
            process[:, i] = process[:, i - 1] * np.exp(
                (self.r - self.dividend - 0.5 * self.sigma**2) * dt
                + self.sigma * BM[:, i - 1]
            )

        return process

    def get_path_importance_sampling(
        self, n: int, dt: float, N: float, l: float, seed=None
    ) -> np.array:
        """_summary_

        Parameters
        ----------
        n : int
            number of points for discretizing the process.
        dt : float
            timestep.
        N : float
            number of monte-carlo scenarios.
        l : float
            girsanov parameter for change of measure.
        seed : _type_, optional
            _description_, by default None

        Returns
        -------
        np.array
        return the explicit simulation of X_t :
        X_i+1 = X_i*(b(X_i) - (sigma(X_i)^2)/2dt + sigma(X_i)(BM[i],BM[i-1]))
        """

        np.random.seed(seed)

        process = np.zeros((N, n))
        Z = np.ones((N, n))
        BM = np.random.normal(0, np.sqrt(dt), (N, n))

        process[:, 0] = self.x0

        for i in range(1, n):
            process[:, i] = process[:, i - 1] * np.exp(
                (self.r - (0.5 * self.sigma**2) - self.dividend - (self.sigma * l)) * dt
                + self.sigma * BM[:, i - 1]
            )
            Z[:, i] = (
                Z[:, i - 1] * np.exp(-0.5 * (l**2) * dt) * np.exp(BM[:, i - 1] * l)
            )

        return process, Z

    def get_path_importance_sampling_multi_dim(
        self, d: int, n: int, dt: float, N: float, l: float, seed=None
    ) -> np.array:
        """_summary_

        Parameters
        ----------
        d : int
            dimension of the underlying.
        n : int
            number of points for discretizing the process.
        dt : float
            timestep.
        N : float
            number of monte-carlo scenarios.
        l : float
            girsanov parameter for change of measure.
        seed : _type_, optional
            _description_, by default None

        Returns
        -------
        np.array
        return the explicit simulation of X_t :
        X_i+1 = X_i*(b(X_i) - (sigma(X_i)^2)/2dt + sigma(X_i)(BM[i],BM[i-1]))
        """

        np.random.seed(seed)

        process = np.zeros((N, n, d))
        Z = np.ones((N, n))
        BM = np.random.normal(0, np.sqrt(dt), (N, n, d))

        process[:, 0] = self.x0

        for i in range(1, n):
            list_brownians = np.zeros((N, d))
            for j in range(d):
                process[:, i, j] = process[:, i - 1, j] * np.exp(
                    (self.r - (0.5 * self.sigma**2) - self.dividend - (self.sigma * l))
                    * dt
                    + self.sigma * BM[:, i - 1, j]
                )
                list_brownians[:, j] = np.exp(BM[:, i - 1, j] * l)

            list_brownians_prod = np.prod(list_brownians, axis=1)
            Z[:, i] = Z[:, i - 1] * np.exp(-0.5 * (l**2) * dt) * list_brownians_prod

        return process, Z
