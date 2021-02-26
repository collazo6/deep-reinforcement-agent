from dqn.model import QNetwork, DuelingQNetwork
from collections import namedtuple
import torch.nn.functional as F
from collections import deque
import torch.optim as optim
import numpy as np
import random
import torch

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Initialize namedtuple to store Agent hyperparams.
AgentHyperparams = namedtuple(
    'AgentHyperparams',
    [
        'buffer_size',
        'batch_size',
        'gamma',
        'tau',
        'lr',
        'update_frequency',
        'duel'
    ]
)


class Agent:
    """
    A class to implement an Agent which interacts with and learns from the
    environment!

    Attributes:
        state_size: An integer count of dimensions for each state.
        action_size: An integer count of dimensions for each action.
        seed: An integer random seed.

        agent_hparams:
            buffer_size: An integer for replay buffer size.
            batch_size: An integer for minibatch size.
            gamma: A float designating the discount factor.
            tau: A float designating multiplication factor for soft update of
                target parameters.
            lr: A float designating the learning rate of the optimizer.
            update_frequency: An integer designating the step frequency of
                updating target network parameters.
            duel: A boolean which specifies the use of either Dueling Q
                Networks or traditional Q Networks for training.
    """

    def __init__(self, state_size, action_size, seed, agent_hparams):
        """Initializes an Agent object."""

        # Initialize variables for Agent training.
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.hparams = agent_hparams

        # Create local and target network for stabilized learning.
        if self.hparams.duel:
            self.qnetwork_local = DuelingQNetwork(state_size, action_size,
                                                  seed).to(device)
            self.qnetwork_target = DuelingQNetwork(state_size, action_size,
                                                   seed).to(device)
        else:
            self.qnetwork_local = QNetwork(state_size, action_size, seed)\
                .to(device)
            self.qnetwork_target = QNetwork(state_size, action_size, seed)\
                .to(device)

        # Initialize Optimizer.
        self.optimizer = optim.Adam(
            self.qnetwork_local.parameters(),
            lr=self.hparams.lr
        )

        # Initilize replay memory.
        self.memory = ReplayBuffer(
            action_size,
            self.hparams.buffer_size,
            self.hparams.batch_size,
            seed
        )

        # Initialize time step (for updating every update_frequency time steps)
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        """
        Adds experiences to memory and updates target Q network every
        update_frequency time steps.
        """

        # Add experience to replay memory.
        self.memory.add(state, action, reward, next_state, done)

        # Soft update target Q network every update_frequency time steps.
        self.t_step = (self.t_step + 1) % self.hparams.update_frequency
        if self.t_step == 0:

            # If enough memories available, get random subset and learn.
            if len(self.memory) > self.hparams.batch_size:
                experiences = self.memory.sample()
                self.learn(experiences, self.hparams.gamma)

    def act(self, state, eps=0.):
        """
        Returns actions for given state as per current policy.

        Parameters:
            state: Numpy array with info on current state.
            eps: A float epsilon, for epsilon-greedy action selection.
        """

        # Evaluate action values at current state.
        state = torch.from_numpy(state).float().to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection, explore vs. exploit.
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        """
        Update value parameters using given batch of experience tuples.

        Parameters:
            experiences: A tuple of (s, a, r, s', done) tuples.
            gamma: A float designating the discount factor.
        """

        # Extract relevant variables from experiences.
        states, actions, rewards, next_states, dones = experiences

        # Get max predicted Q values (for next states) from target model.
        Q_targets_next = self.qnetwork_target(next_states).detach()\
            .max(1)[0].unsqueeze(1)

        # Compute Q targets for current states.
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model.
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute and minimize the loss
        loss = F.mse_loss(Q_expected, Q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update the target network!
        self.soft_update(
            self.qnetwork_local,
            self.qnetwork_target,
            self.hparams.tau
        )

    def soft_update(self, local_model, target_model, tau):
        """
        Executes a soft update on the target model parameters using the
        following equation:
            θ_target = τ * θ_local + (1 - τ) * θ_target

        Parameters:
            local_model: PyTorch model from which weights will be copied.
            target_model: PyTorch model which weights will be copied to.
            tau: A float designating the interpolation parameter.
        """

        # Soft update target model parameters.
        for target_param, local_param in \
                zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(
                tau * local_param.data + (1.0 - tau) * target_param.data
            )


class ReplayBuffer:
    """
    A class to implement a fixed-size buffer to store experience tuples.

    Attributes:
        action_size: An integer count of dimensions for each action.
        buffer_size: An integer for replay buffer size.
        batch_size: An integer for minibatch size.
        seed: An integer random seed.
    """

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initializes a ReplayBuffer object."""

        # Initialize variables for ReplayBuffer.
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple(
            'Experience',
            field_names=['state', 'action', 'reward', 'next_state', 'done']
        )
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Adds a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly samples a batch of experiences from memory."""

        # Creates list with random batch of experiences.
        experiences = random.sample(self.memory, k=self.batch_size)

        # Extracts relevant information from experiences.
        states = torch.from_numpy(np.vstack(
            [e.state for e in experiences if e is not None]
        )).float().to(device)
        actions = torch.from_numpy(np.vstack(
            [e.action for e in experiences if e is not None]
        )).long().to(device)
        rewards = torch.from_numpy(np.vstack(
            [e.reward for e in experiences if e is not None]
        )).float().to(device)
        next_states = torch.from_numpy(np.vstack(
            [e.next_state for e in experiences if e is not None]
        )).float().to(device)
        dones = torch.from_numpy(np.vstack(
            [e.done for e in experiences if e is not None]
        ).astype(np.uint8)).float().to(device)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
