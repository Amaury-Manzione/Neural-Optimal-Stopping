import sys

sys.path.append("..")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch

import src.diffusion.black_scholes as black_scholes
import src.dqn.agent as agent
import src.dqn.multi_dim_bermudan as bermudan

strike = 1
r = 0.06
vol = 0.2
spot = 36 / 40
maturity = 1
n_simulation = 50
gamma = 1
epsilon = 0.01
epsilon_decay = 0.0008
epsilon_max = 0.99
learning_rate = 1e-3
d = 2
batch_size = 500
replay_memory_init = 5000
replay_memory_capacity = 100000
N_update = 150


myagent = agent.Agent(
    epsilon,
    epsilon_decay,
    epsilon_max,
    replay_memory_init,
    replay_memory_capacity,
    N_update,
    learning_rate,
    batch_size,
    d,
)

process = [black_scholes.BlackScholes(spot, r, vol) for i in range(d)]

put_option_payoff = lambda x: torch.maximum(
    torch.max(x) - strike, torch.tensor(0)
).item()


option = bermudan.MultiDimensionalBermudan(
    process, put_option_payoff, maturity, r, myagent
)

n_episodes = 5000
rewars, rewards_batch, losses, pos, epsilon, exploit_explore, myagent = option.train(
    n_simulation, n_episodes, disable=False, seed=2024
)
