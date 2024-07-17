import sys
import time as time
from typing import Callable, List

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

sys.path.append("..\..")

import src.diffusion.ito_process as ito_process
import src.dqn.agent as agent


class MultiDimensionalBermudan:
    """Class representing a bermudan option priced with deep q-learning paradigm."""

    def __init__(
        self,
        process: List[ito_process.ItoProcess],
        payoff: Callable,
        maturity: float,
        rate: float,
        myagent: agent.Agent,
    ):
        """Construct an object of type bermudan.

        Parameters
        ----------
        process : ito_process.ito_process
            underlying of the option.
        payoff : Callable
            payoff of the option, for example f : x -> (K -x)_{+}
        maturity : float
            maturity of the option
        rate : float
            interest rate
        myagent : agent.Agent
            agent for dqn.
        """
        self.process = process
        self.payoff = payoff
        self.maturity = maturity
        self.rate = rate
        self.myagent = myagent

    def explore(self, episode, list_times, paths, dt):
        stopping_time_index = torch.multinomial(
            torch.ones(list_times.size()), 1, replacement=True
        )
        stopping_time = list_times[stopping_time_index].item()
        reward = torch.exp(torch.tensor(-self.rate * dt * stopping_time)) * self.payoff(
            paths[episode, stopping_time, :]
        )
        input_to_tensor = torch.cat(
            (paths[episode, stopping_time, :], torch.tensor([stopping_time * dt]))
        )
        self.myagent.add_to_replay_buffer(
            input_to_tensor.to(torch.double),
            input_to_tensor.to(torch.double),
            reward,
            torch.tensor(0),
            torch.tensor(1),
            torch.tensor(stopping_time, dtype=torch.float),
        )
        for i in range(stopping_time - 1):
            state_to_tensor = input_to_tensor = torch.cat(
                (paths[episode, i, :], torch.tensor([i * dt]))
            )
            new_state_to_tensor = input_to_tensor = torch.cat(
                (paths[episode, i + 1, :], torch.tensor([(i + 1) * dt]))
            )
            self.myagent.add_to_replay_buffer(
                state_to_tensor.to(torch.double),
                new_state_to_tensor.to(torch.double),
                torch.tensor(0),
                torch.tensor(1),
                torch.tensor(0),
                torch.tensor(i, dtype=torch.float),
            )

        return stopping_time

    def exploit(self, episode, paths, dt):
        done = False
        current_time = 1
        state = paths[episode, current_time, :]
        n = paths.shape[1]
        reward = torch.tensor(0)
        while not done:
            state_time = torch.cat((state, torch.tensor([current_time * dt])))
            actions = self.myagent.get_Q(state_time.to(torch.double))
            action = torch.argmax(actions).item()
            if action == 0 or current_time == n - 1:
                input_to_tensor = torch.cat((state, torch.tensor([current_time * dt])))
                reward = torch.exp(
                    torch.tensor(-self.rate * dt * current_time)
                ) * self.payoff(state)
                done = True
                action = 0
                self.myagent.add_to_replay_buffer(
                    input_to_tensor.to(torch.double),
                    input_to_tensor.to(torch.double),
                    reward,
                    torch.tensor(action),
                    torch.tensor(int(done)),
                    torch.tensor(current_time, dtype=torch.float),
                )
            else:
                new_state = paths[episode, current_time + 1]
                state_to_tensor = input_to_tensor = torch.cat(
                    (state, torch.tensor([current_time * dt]))
                )
                new_state_to_tensor = input_to_tensor = torch.cat(
                    (new_state, torch.tensor([(current_time + 1) * dt]))
                )
                self.myagent.add_to_replay_buffer(
                    state_to_tensor.to(torch.double),
                    new_state_to_tensor.to(torch.double),
                    reward,
                    torch.tensor(action),
                    torch.tensor(int(done)),
                    torch.tensor(current_time, dtype=torch.float),
                )
                current_time += 1
                state = new_state
        return current_time, reward

    def train(
        self, n_simulation: int, n_episodes: int, verbose=False, disable=True, seed=None
    ):
        """Train the agent to find optimal Q function

        Parameters
        ----------
        n_simulation : int
            number of steps to discretize the process
        n_episodes : int
            number of episodes for training
        replay_memory_capacity : int
            total capacity for the replay buffer
        batch_size : float
            size of the batches
        """
        # dimension
        d = len(self.process)

        # timestep
        dt = float(self.maturity / n_simulation)
        list_times = torch.tensor([i for i in range(n_simulation)])

        # loading the environment
        paths = torch.zeros(n_episodes, n_simulation, d)
        for i in range(d):
            paths[:, :, i] = self.process[i].get_path_tensor(
                n_simulation, dt, n_episodes, seed
            )

        # history of rewards per episode
        history_batch_reward = torch.zeros(n_episodes)
        history_reward = torch.zeros(n_episodes)
        history_losses = torch.zeros(n_episodes)
        history_pos = torch.zeros(n_episodes)
        history_epsilon = torch.zeros(n_episodes)
        history_explore_exploit = torch.zeros(n_episodes)

        count_target_update = 0

        # training loop
        for episode in tqdm(range(n_episodes), disable=disable):
            # norm_euclid = np.mean(paths[episode, :] ** 2)
            u = np.random.uniform()
            if u > self.myagent.epsilon:
                history_pos[episode] = self.explore(episode, list_times, paths, dt)
                history_explore_exploit[episode] = 0
            else:
                history_pos[episode], history_reward[episode] = self.exploit(
                    episode, paths, dt
                )
                history_explore_exploit[episode] = 1
            history_losses[episode], history_batch_reward[episode] = (
                self.myagent.train_online_network()
            )
            history_epsilon[episode] = self.myagent.epsilon
            self.myagent.epsilon = min(
                self.myagent.epsilon + self.myagent.epsilon_decay,
                self.myagent.epsilon_max,
            )
            count_target_update += 1
            if count_target_update % self.myagent.n_update == 0:
                self.myagent.update_target_network()
            if verbose:
                print(
                    f"Episode {episode}, Reward: {history_batch_reward[episode]}, pos : {history_pos[episode]}"
                )

        return (
            history_reward,
            history_batch_reward,
            history_losses,
            history_pos,
            history_epsilon,
            history_explore_exploit,
            self.myagent,
        )

    def get_price(self, trained_agent: agent.Agent, n_simulation, n_mc):
        """Return price of the option.

        Parameters
        ----------
        trained_agent : agent.Agent
            Trained agent.
        n_simulation : int
            Number of steps to discretize the process.
        n_mc : int
            Number of Monte Carlo scenarios.
        """
        dt = self.maturity / n_simulation
        mc_paths = self.process.get_path(n_simulation, dt, n_mc)
        discount_factors = np.exp(-self.rate * dt * np.arange(n_simulation))
        payoffs = np.zeros(n_mc)

        for n in tqdm(range(n_mc)):
            current_time = 1
            done = False
            while not done:
                state = torch.tensor([mc_paths[n, current_time], current_time * dt])
                action = trained_agent.get_policy(state)

                if current_time == n_simulation - 1 or action == 0:
                    payoffs[n] = discount_factors[current_time] * self.payoff(
                        mc_paths[n, current_time]
                    )
                    done = True
                else:
                    current_time += 1

        return np.mean(payoffs)
