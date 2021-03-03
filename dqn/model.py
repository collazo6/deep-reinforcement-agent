import torch.nn.functional as F
import torch.nn as nn
import torch


class QNetwork(nn.Module):
    """
    A class to build and create a Deep Q Network through PyTorch to learn
    optimal actions given environmental variables.

    Attributes:
        state_size: An integer count of dimensions for each state.
        action_size: An integer count of dimensions for each action.
        seed: An integer random seed.
        fc1_units: An integer number of nodes in first hidden layer.
        fc2_units: An integer number of nodes in second hidden layer.
    """

    def __init__(self, state_size, action_size, seed,
                 fc1_units=64, fc2_units=64):
        """Initializes parameters and builds model."""

        # Initialize inheritance and variables.
        super().__init__()
        self.seed = torch.manual_seed(seed)
        _, input_dim = state_size

        # Create neural network layers in PyTorch.
        self.fc1 = nn.Linear(input_dim, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)

    def forward(self, state):
        """Builds neural network that maps state to action values."""

        # Build neural network architecture.
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class DuelingQNetwork(nn.Module):
    """
    A class to build and create a Dueling Deep Q Network algorithm through
    PyTorch to learn optimal actions given environmental variables.

    Attributes:
        state_size: An integer count of dimensions for each state.
        action_size: An integer count of dimensions for each action.
        seed: An integer random seed.
        fc1_units: An integer number of nodes in first hidden layer.
        fc2_units: An integer number of nodes in second hidden layer.
    """

    def __init__(self, state_size, action_size, seed,
                 fc1_units=64, fc2_units=64):
        """Initializes parameters and builds model."""

        super().__init__()
        self.seed = torch.manual_seed(seed)
        _, input_dim = state_size

        # Create neural network layers in PyTorch.
        self.feature_layer = nn.Sequential(
            nn.Linear(input_dim, fc1_units),
            nn.ReLU(),
            nn.Linear(fc1_units, fc2_units),
            nn.ReLU()
        )
        self.value_stream = nn.Sequential(
            nn.Linear(fc2_units, fc2_units),
            nn.ReLU(),
            nn.Linear(fc2_units, 1)
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(fc2_units, fc2_units),
            nn.ReLU(),
            nn.Linear(fc2_units, action_size)
        )

    def forward(self, state):
        """Builds neural network that maps state to action values."""

        # Build neural network architecture.
        features = self.feature_layer(state)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        q_values = values + (advantages - advantages.max())

        return q_values
