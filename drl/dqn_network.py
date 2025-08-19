import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DQNNetwork(nn.Module):
    """Deep Q-Network for the quantum state control task."""
    
    def __init__(self, state_dim, action_dim, hidden_dims=[256, 256]):
        """
        Initialize the DQN network.
        
        Args:
            state_dim (int): Dimension of the state space (2 * chain_length)
            action_dim (int): Number of possible actions
            hidden_dims (list): List of hidden layer dimensions
        """
        super(DQNNetwork, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Create the network layers
        layers = []
        input_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))  # Add dropout for regularization
            input_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(input_dim, action_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier uniform initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0.0)
    
    def forward(self, state):
        """
        Forward pass through the network.
        
        Args:
            state (torch.Tensor): Input state tensor
            
        Returns:
            torch.Tensor: Q-values for each action
        """
        return self.network(state)
    
    def get_action(self, state, epsilon=0.0):
        """
        Get action using epsilon-greedy policy.
        
        Args:
            state (np.ndarray or torch.Tensor): Current state
            epsilon (float): Exploration probability
            
        Returns:
            int: Selected action
        """
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state).unsqueeze(0)
        
        if np.random.random() < epsilon:
            return np.random.randint(0, self.action_dim)
        
        with torch.no_grad():
            q_values = self.forward(state)
            return q_values.argmax().item()


class DuelingDQNNetwork(nn.Module):
    """Dueling DQN architecture for improved value estimation."""
    
    def __init__(self, state_dim, action_dim, hidden_dims=[256, 256]):
        """
        Initialize the Dueling DQN network.
        
        Args:
            state_dim (int): Dimension of the state space
            action_dim (int): Number of possible actions
            hidden_dims (list): List of hidden layer dimensions
        """
        super(DuelingDQNNetwork, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Shared feature extraction layers
        self.feature_layers = nn.Sequential(
            nn.Linear(state_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dims[1], hidden_dims[1] // 2),
            nn.ReLU(),
            nn.Linear(hidden_dims[1] // 2, 1)
        )
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dims[1], hidden_dims[1] // 2),
            nn.ReLU(),
            nn.Linear(hidden_dims[1] // 2, action_dim)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier uniform initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0.0)
    
    def forward(self, state):
        """
        Forward pass through the dueling network.
        
        Args:
            state (torch.Tensor): Input state tensor
        
        Returns:
            torch.Tensor: Q-values for each action
        """
        # Ensure state is on the same device as the model
        state = state.to(next(self.parameters()).device)
        features = self.feature_layers(state)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        # Combine value and advantage streams
        # Q(s,a) = V(s) + A(s,a) - mean(A(s,a'))
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q_values
    
    def get_action(self, state, epsilon=0.0):
        """
        Get action using epsilon-greedy policy.
        
        Args:
            state (np.ndarray or torch.Tensor): Current state
            epsilon (float): Exploration probability
            
        Returns:
            int: Selected action
        """
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state).unsqueeze(0)
        # Ensure state is on the same device as the model
        state = state.to(next(self.parameters()).device)
        if np.random.random() < epsilon:
            return np.random.randint(0, self.action_dim)
        with torch.no_grad():
            q_values = self.forward(state)
            return q_values.argmax().item()
