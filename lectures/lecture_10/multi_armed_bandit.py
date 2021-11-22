"""Helper function for multi-armed bandit problem."""
from abc import ABC
from abc import abstractmethod
from typing import Dict
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm


class TestBed:
    """The environment."""

    def __init__(self, num_arms: int, random_seed: int = 42, **kwargs):
        """Initialize the environment as k normal distributions."""
        self.random_seed = random_seed
        np.random.seed(self.random_seed)
        self.num_arms = num_arms
        self.mus = np.random.normal(loc=0., scale=1., size=self.num_arms)
        self.stds = [1. for _ in range(self.num_arms)]
        self.best_arm = np.argmax(self.mus)
        self.distributions = pd.DataFrame()

    def emit(self, arm_idx: int) -> float:
        """Emit a sample from a given arm."""
        assert 0 <= arm_idx < self.num_arms
        return np.random.normal(
            loc=self.mus[arm_idx],
            scale=self.stds[arm_idx])

    def describe(self):
        """Describe basic information about the testbed."""
        tqdm.write("=====\nDetails about the testbed:")
        tqdm.write(f"Number of arms: {self.num_arms}")
        tqdm.write(f"Means of the arms: {self.mus}")
        tqdm.write(f"Stds of the arms: {self.stds}")
        tqdm.write(f"The best arm is {self.best_arm}")

    def visualize(self, n_samples: int = 1000):
        """Visualize the distribution of the arms, with help of pandas."""
        n_samples = int(n_samples)
        values = []
        idx = []
        if len(self.distributions) == 0:
            for i in range(self.num_arms):
                idx.extend([i] * n_samples)
                values.extend(np.random.normal(
                    loc=self.mus[i],
                    scale=self.stds[i],
                    size=n_samples))
            # make the dataframe for plotting
            self.distributions = pd.DataFrame(
                data={
                    "arm_index": idx,
                    "value": values},
            )

        # plot
        sns.set(font_scale=1.2)
        sns.set_style("whitegrid", {"grid.linestyle": "--"})
        fig = plt.Figure()
        sns.violinplot(
            x="arm_index", y="value", data=self.distributions,
            inner=None, ax=fig.gca())
        plt.tight_layout()
        plt.show()

        return fig


class Agent(ABC):
    """Abstract class for an agent."""

    @abstractmethod
    def __init__(self, env: TestBed, verbose: bool = False, **kwargs):
        """Define the environment the agent will be in, and values."""
        self.env = env
        self.verbose: bool = verbose
        self.arm_values: List[float] = []  # values for each arm, same shape as arm counts
        self.arm_counts: List[int] = []  # number of pulls for each arm
        self.arms_history: List[int] = []  # history of arms pulled for each step
        self.rewards_history: List[float] = []

        self.simulation_finished: bool = False
        self.current_step = 0  # note this is 0-index

        self.current_arm = None
        self.current_value = None
        self.random_seed: int = kwargs.get("random_seed", 42)

    def init_values(self, value: float = None):
        """Initialize the value estimates."""
        self.simulation_finished = False
        self.current_step = 0
        self.arm_counts = np.zeros(self.env.num_arms)

        if value is not None:  # uniform initial values
            self.arm_values = np.array(
                [value for _ in range(self.env.num_arms)])
        else:  # it is None, then random values from N(0, 1)
            self.arm_values = np.random.normal(
                size=self.env.num_arms)

    @abstractmethod
    def pick_arm(self) -> int:
        """Pick an arm according to the agent"s policy."""
        raise NotImplementedError

    @abstractmethod
    def update_values(self) -> None:
        """Update the values estimates."""
        raise NotImplementedError

    def take_single_step(self) -> None:
        """Pick an arm, pull to take the reward.
        Here we do NOT updating the arms values and counts.
        """
        self.current_arm = self.pick_arm()
        self.current_value = self.env.emit(self.current_arm)

    def update_logs(self) -> None:
        """Update the logs at step s."""
        self.rewards_history[self.current_step] = self.current_value
        self.arms_history[self.current_step] = self.current_arm

    def run(self, steps: int):
        """Run the simulation."""
        if self.arm_values is None:
            raise "Please initialize the agent."
        if self.simulation_finished:
            return

        self.arms_history = -1 * np.ones(steps)
        self.rewards_history = np.zeros(steps)

        for _ in tqdm(range(steps),
                      desc="Agent running",
                      disable=~self.verbose):
            self.take_single_step()

            self.arm_counts[self.current_arm] += 1
            self.update_values()
            self.update_logs()
            self.current_step += 1

        self.simulation_finished = True
        return

    def describe(self):
        """Describe basic information about the agent"s end state."""
        if not self.simulation_finished:
            tqdm.write("Agent has not run yet.")
        else:
            tqdm.write("\n======")
            tqdm.write("reward history: ", self.rewards_history)
            tqdm.write("arm history: ", self.arms_history)
            tqdm.write("arm counts: ", self.arm_counts)
            tqdm.write("arm values: ", self.arm_values)


class EpsilonGreedyAgent(Agent):
    """epsilon-greedy policy."""

    def __init__(
            self,
            env: TestBed,
            epsilon: float,
            verbose: bool = False,
            **kwargs):
        """Put the agent in an environment."""
        super().__init__(env=env, verbose=verbose, **kwargs)
        self.epsilon = epsilon
        assert 0. <= self.epsilon < 1.

    def pick_arm(self):
        """Pick the arm that has largest value. Breaks tie randomly."""
        idx = np.random.choice(
            np.flatnonzero(self.arm_values == np.max(self.arm_values)))

        if self.epsilon == 0:  # greedy
            return idx
        else:  # explore with probability of epsilon
            if np.random.uniform() < self.epsilon:
                if np.min(self.arm_values) == np.max(self.arm_values):
                    return idx
                idx = np.random.choice(np.flatnonzero(
                    self.arm_values != np.max(self.arm_values)))
            return idx

    def update_values(self):
        """Update the values estimates."""
        # sample average
        self.arm_values[self.current_arm] += \
            ((self.current_value - self.arm_values[self.current_arm]) /
             self.arm_counts[self.current_arm])


class GreedyAgent(EpsilonGreedyAgent):
    """Greedy policy as a special case of epsilon greedy policy."""

    def __init__(
            self,
            env: TestBed,
            verbose: bool = False, **kwargs):
        """Enforce epsilon to zero."""
        super().__init__(env=env, epsilon=0.0, verbose=verbose)


class UCBAgent(Agent):
    """Upper confidence bound policy."""

    def __init__(
            self,
            env: TestBed,
            c: float,
            verbose: bool = False,
            **kwargs):
        """Put the agent in an environment."""
        super().__init__(env=env, verbose=verbose, **kwargs)
        self.c = c
        self.arm_ucbs = None

    def pick_arm(self):
        """Find the arm to pull, according to UCB policy."""
        # if there is any arm not pulled, pick random one from them
        if np.min(self.arm_counts) == 0:
            idx = np.random.choice(
                np.flatnonzero(self.arm_counts == np.min(self.arm_counts)))

        # otherwise find the ucb
        else:
            self.arm_ucbs = []
            for i in range(self.env.num_arms):
                value = self.arm_values[i] + self.c * np.sqrt(
                    (np.log(self.current_step + 1) /  # the step is 0-index
                     self.arm_counts[i]))
                self.arm_ucbs.append(value)
            idx = np.random.choice(
                np.flatnonzero(self.arm_ucbs == np.max(self.arm_ucbs)))

        return idx

    def update_values(self):
        """Update the values estimates."""
        # sample average
        self.arm_values[self.current_arm] += \
            ((self.current_value - self.arm_values[self.current_arm]) /
             self.arm_counts[self.current_arm])


class Simulation:
    """Put an (new) agents in an (new) environment, run. Then repeat."""

    def __init__(
            self,
            env_type: TestBed,  # the Callable
            agent_type: Agent,  # the Callable
            num_agents: int,
            init_value: int,
            step: int,
            random_seed: int = 42,
            env_kwargs: Dict = {},
            agent_kwargs: Dict = {},
            **kwargs):
        """Combine the agent and the environment."""
        self.env_type = env_type
        self.env_kwargs = env_kwargs
        self.num_arms = self.env_kwargs.get("num_arms", 10)

        self.agent_type = agent_type
        self.agent_kwargs = agent_kwargs
        self.epsilon = self.agent_kwargs.get("epsilon", 0.0)
        self.num_agents = num_agents
        self.init_value = init_value

        self.step = step
        self.random_seed = random_seed

        self.agent_rewards_histories = []  # list of lists
        self.avg_rewards_history = None
        self.agent_arms_histories = []  # list of lists
        self.avg_arms_history = None

    def run_all_agents(self):
        """Run simulation."""
        for i in tqdm(range(self.num_agents), desc="Simulation running"):
            self.env = self.env_type(
                num_arms=self.num_arms,
                random_seed=self.random_seed + i)
            self.agent = self.agent_type(env=self.env, **self.agent_kwargs)
            self.agent.init_values(self.init_value, )
            self.agent.run(self.step)

            # collect results
            self.agent_rewards_histories.append(
                self.agent.rewards_history)
            self.agent_arms_histories.append(
                self.agent.arms_history)

    def aggregate_rewards(self, make_plot: bool = True):
        """Collect aggregated rewards from all the agent/env pairs."""
        self.avg_rewards_history = np.mean(
            self.agent_rewards_histories, axis=0)
        steps = list(range(len(self.avg_rewards_history)))
        if make_plot:
            sns.set(font_scale=1.2)
            sns.set_style("whitegrid", {"grid.linestyle": "--"})
            sns.lineplot(x=steps,
                         y=self.avg_rewards_history)
            plt.xlabel("Step")
            plt.ylabel("Average reward")
            plt.title(f"Result from {self.num_agents} simulations.")
            plt.tight_layout()
            plt.show()

        return steps, self.avg_rewards_history


if __name__ == "__main__":

    simulation = Simulation(
        env_type=TestBed,
        agent_type=UCBAgent,
        num_agents=10,
        init_value=None,
        step=1000,
        env_kwargs={"num_arms": 10},
        agent_kwargs={"c": 2}
    )

    simulation.run_all_agents()