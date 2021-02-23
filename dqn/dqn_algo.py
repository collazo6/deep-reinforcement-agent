import matplotlib.pyplot as plt
from collections import deque
import pandas as pd
import numpy as np
import torch

plt.style.use('dark_background')


class DQNTrainer:
    """
    A class to implement and utilize the training process steps of
    Deep Q Learning.

    Attributes:
        agent: An Agent object used for training.
        env: A UnityEnvironment used for Agent evaluation and training.
        n_episodes: An integer for maximum number of training episodes.
        max_t: An integer for maximum number of timesteps per episode.
        eps_start: A float for the starting value of epsilon, for
            epsilon-greedy action selection.
        eps_end: A float to set the minimum value of epsilon.
        eps_decay: A float multiplicative factor (per episode) for decreasing
            epsilon.
        save_dir: String designating directory to save files.
    """

    def __init__(self, agent, env, max_t, eps_start, eps_end,
                 eps_decay, save_dir):
        """Initializes DQNTrainer object."""

        # Initialize variables for trainer.
        self.agent = agent
        self.env = env
        self.brain_name = env.brain_names[0]
        self.max_t = max_t
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.eps = self.eps_start
        self.save_dir = save_dir
        self.i_episode = 1
        self.scores = []
        self.scores_window = deque(maxlen=100)

    def process_observation(self, observation):
        """Adds leading dimension for model utilization."""
        return np.expand_dims(observation, 0)

    def run_episode(self):
        """Runs a single episode and returns the corresponding score."""

        # Reset initial state and score.
        env_info = self.env.reset(train_mode=True)[self.brain_name]
        observation = env_info.vector_observations[0].copy()
        state = self.process_observation(observation)

        score = 0

        # For each time step, select action and evaluate next state params.
        for t in range(self.max_t):
            action = self.agent.act(state, self.eps)
            action = int(action)
            env_info = self.env.step(action)[self.brain_name]
            observation = env_info.vector_observations[0].copy()
            next_state = self.process_observation(observation)
            reward = env_info.rewards[0]
            done = env_info.local_done[0]
            self.agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break

        return score

    def step_train(self) -> None:
        """Executes one step of the training process."""

        # Get corresponding score from most recent run.
        score = self.run_episode()

        # Save most recent score to score variables and decrease epsilon.
        self.scores_window.append(score)
        self.scores.append(score)
        self.eps = max(self.eps_end, self.eps_decay * self.eps)

    def train(self, n_episodes):
        """Run training for n_episodes."""

        # For each episode, evaluate scores and save when environment solved.
        for i in range(1, n_episodes + 1):
            self.step_train()

            # Temporarily print score for each episode.
            print(
                '\rEpisode {0}\tAverage Score: {1:.2f}'.format(
                    self.i_episode, np.mean(self.scores_window)
                ),
                end=""
            )

            # For each 100 episodes, print relevant statistics.
            if self.i_episode % 100 == 0:
                self.plt_rolling_avgs()
                print(f'\rEpisode {self.i_episode}'
                      f'\tAverage Score: {np.mean(self.scores_window):.2f}'
                      f'\tEps: {self.eps:.3f}'
                      f'\tNumber of memories: {len(self.agent.memory)}')

            # If env solved, plot learning curve and save model params.
            if np.mean(self.scores_window) >= 13:

                self.plt_rolling_avgs()
                self.save()

                print(
                    f'\nEnvironment solved in {self.i_episode - 100} episodes!'
                    f'\tAverage Score: {np.mean(self.scores_window):.2f}'
                )
                break

            self.i_episode += 1

        return self.scores

    def save(self):
        """Saves parameters for successful network."""
        torch.save(
            self.agent.qnetwork_local.state_dict(),
            rf'{self.save_dir}/checkpoint_{self.i_episode}.pth'
        )

    def restore(self, i_episode):
        """Loads parameters for successful network."""
        self.agent.qnetwork_local.load_state_dict(
            torch.load(rf'{self.save_dir}/checkpoint_{i_episode}.pth')
        )

    def plt_rolling_avgs(self):
        """Plots learning curve for successful network."""

        # Set title of plot based on Q Network algorithm used.
        model_type = 'Dueling Q Network' if self.agent.hparams.duel else \
            'Q Network'

        # Calculate rolling averages based on the last 25 episodes.
        rolling_avgs = pd.DataFrame(self.scores).rolling(25).mean()

        # Force index to start at 1 for 1st episode.
        rolling_avgs.index += 1

        # Plot rolling averages and save resulting plot
        plt.plot(rolling_avgs, marker='o', markersize=6, markerfacecolor='w')
        plt.title(f'Learning Curve: {model_type}', fontsize=30)
        plt.grid(color='w', linewidth=0.2)
        plt.savefig(rf'{self.save_dir}/scores_mavg_{self.i_episode}')
        plt.show()
