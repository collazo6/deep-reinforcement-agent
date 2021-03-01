import matplotlib.pyplot as plt
from collections import deque
import pandas as pd
import numpy as np
import torch
import math
import os

plt.style.use('dark_background')


class DQNTrainer:
    """
    A class to implement and utilize the training process steps of
    Deep Q Learning.

    Attributes:
        agent: An Agent object used for training.
        env: A UnityEnvironment used for Agent evaluation and training.
        end_score: The integer score (averaged over the past 100 episodes)
            in which the environment is considered solved by the agent.
        max_t: An integer for maximum number of timesteps per episode.
        eps_start: A float for the starting value of epsilon, for
            epsilon-greedy action selection.
        eps_end: A float to set the minimum value of epsilon.
        eps_decay: A float multiplicative factor (per episode) for decreasing
            epsilon.
        save_dir: String designating directory to save files.
    """

    def __init__(self, agent, env, end_score, max_t, eps_start, eps_end,
                 eps_decay, save_dir):
        """Initializes DQNTrainer object."""

        # Initialize variables for trainer.
        self.agent = agent
        self.env = env
        self.brain_name = env.brain_names[0]
        self.end_score = end_score
        self.max_t = max_t
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.eps = self.eps_start
        self.save_dir = save_dir
        self.i_episode = 1
        self.scores = []
        self.scores_window = deque(maxlen=100)

    def run_episode(self):
        """Runs a single episode and returns the corresponding score."""

        # Reset initial state and score.
        env_info = self.env.reset(train_mode=True)[self.brain_name]
        observation = env_info.vector_observations[0].copy()
        state = np.expand_dims(observation, 0)
        score = 0

        # For each time step, select action and evaluate next state params.
        for t in range(self.max_t):
            action = self.agent.act(state, self.eps)
            action = int(action)
            env_info = self.env.step(action)[self.brain_name]
            observation = env_info.vector_observations[0].copy()
            next_state = np.expand_dims(observation, 0)
            reward = env_info.rewards[0]
            done = env_info.local_done[0]
            self.agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break

        return score

    def step_train(self):
        """Executes one step of the training process."""

        # Get corresponding score from most recent run.
        score = self.run_episode()

        # Save most recent score to score variables and decrease epsilon.
        self.scores_window.append(score)
        self.scores.append(score)
        self.eps = max(self.eps_end, self.eps_decay * self.eps)

    def train(self, n_episodes):
        """
        Run training on agent for n_episodes.

        Attributes:
            n_episodes: An integer for maximum number of training episodes.
        """

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
                print(f'\rEpisode {self.i_episode}'
                      f'\tAverage Score: {np.mean(self.scores_window):.2f}'
                      f'\tEps: {self.eps:.3f}'
                      f'\tNumber of memories: {len(self.agent.memory)}')

            # If training resolved, save files and notify of statistics.
            if np.mean(self.scores_window) >= self.end_score or \
                    self.i_episode == n_episodes:

                # Create and save learning curve plot.
                self.plt_rolling_avgs()

                # Save successful agent model params and notify of statistics.
                if self.i_episode < n_episodes:
                    self.save()
                    s_ep = self.i_episode - 100
                    print(
                        f'\nEnvironment solved in {s_ep} episodes!'
                        f'\tAverage Score: {np.mean(self.scores_window):.2f}'
                    )
                break

            self.i_episode += 1

        return self.scores

    def save(self):
        """Saves parameters for successful network."""
        model = 'dqn' if self.agent.hparams.duel else 'qn'
        torch.save(
            self.agent.qnetwork_local.state_dict(),
            rf'{self.save_dir}/checkpoint_{model}.pth'
        )

    def restore(self, filename):
        """Loads parameters for successful network."""
        self.agent.qnetwork_local.load_state_dict(
            torch.load(os.path.join(self.save_dir, filename))
        )

    def plt_rolling_avgs(self):
        """Plots learning curve for successful network."""

        # Set model type strings for plot title and filename.
        model_type, model = ('Dueling Q Network', 'dqn') if \
            self.agent.hparams.duel else ('Q Network', 'qn')

        # Calculate rolling averages based on the last 25 episodes.
        rolling_avgs = pd.DataFrame(self.scores).rolling(100).mean()

        # Force index to start at 1 for 1st episode.
        rolling_avgs.index += 1

        # Set coordinates (episode, score) when agent solved env.
        x = self.i_episode
        y = rolling_avgs[0].iloc[-1]

        # Set x, y ticks for graph axes and color for line.
        x_end = int(100 * math.ceil(x/100)) + 1
        x_max = x_end if x_end - x >= 50 else x_end + 100
        x_ticks = np.arange(0, x_max, 50)
        y_ticks = np.arange(0, int(y)+2, 1)
        line_color = 'c' if model == 'qn' else 'y'

        # Plot rolling averages and save resulting plot
        fig, ax = plt.subplots(figsize=(12, 9))
        ax.plot(rolling_avgs, color=line_color)
        ax.grid(color='w', linewidth=0.2)
        ax.set_xticks(x_ticks)
        ax.set_yticks(y_ticks)
        ax.set_title(f'Learning Curve: {model_type}', fontsize=30)
        ax.set_xlabel('Episode', fontsize=20)
        ax.set_ylabel('Score', fontsize=20)
        ax.annotate(
            f'Episode: {x}\nScore: {y}',
            fontsize=10,
            xy=(x, y),
            xytext=(x-22, y+0.15),
            horizontalalignment='left'
        )
        plt.savefig(rf'{self.save_dir}/scores_mavg_{model}_{x}')
        plt.show()
