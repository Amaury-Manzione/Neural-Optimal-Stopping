import sys
from multiprocessing import cpu_count

import agent
import matplotlib.pyplot as plt
import numpy as np
import optuna
import torch

sys.path.append("..\..")

import src.diffusion.black_scholes as black_scholes
import src.dqn.one_dim_bermudan as one_dim_bermudan

# defining the parameters

# parameters
strike = 40
r = 0.06
vol = 0.2
spot = 36 / strike
maturity = 1
n_simulation = 50
d = 1
replay_memory_init = 10000
replay_memory_capacity = 100000
num_episodes = 200

# hyperparameters
epsilon = 0.01
epsilon_decay = 0.0008
epsilon_max = 0.99
learning_rate = 1e-3
batch_size = 500
N_update = 20000

process = black_scholes.BlackScholes(spot, r, vol)
put_option_payoff = lambda x: torch.max(torch.tensor(1) - x, torch.tensor(0))


# defining objective function
def objective(trial):
    list_batch = np.arange(start=100, stop=20000, step=50, dtype=int).tolist()
    list_N_update = np.arange(start=10, stop=300, step=10, dtype=int).tolist()
    batch_size_list = trial.suggest_categorical("batch_size", list_batch)
    learning_rate_list = trial.suggest_float("learning_rate", 1e-6, 1e-1)
    N_update_list = trial.suggest_categorical("N_update", list_N_update)
    myagent = agent.Agent(
        epsilon,
        epsilon_decay,
        epsilon_max,
        batch_size_list,
        replay_memory_capacity,
        N_update_list,
        learning_rate_list,
        batch_size_list,
        d,
    )
    option = one_dim_bermudan.OneDimensionalBermudan(
        process, put_option_payoff, maturity, r, myagent
    )
    rewards, _, losses, _, _, _, _ = option.train(
        n_simulation, num_episodes, disable=True
    )
    return torch.mean(rewards[-200:]) * strike, losses[-1]


# creating study
study = optuna.create_study(
    study_name="example_study_computer_univ",
    storage="sqlite:///study_univ_computer.db",
    directions=["maximize", "minimize"],
    load_if_exists=True,
)

study.trials.clear()


number_cores = cpu_count()

study.optimize(objective, n_trials=300, n_jobs=number_cores)

print("Best trials:")
for i, trial in enumerate(study.best_trials):
    print(f"Trial {i}:")
    print("  Values: ", trial.values)
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
