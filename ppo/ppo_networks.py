import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical


class ActorNetwork(nn.Module):
    """Actor network for PPO - outputs action probabilities."""
    
    def __init__(self, state_dim, action_dim, hidden_dims=[256, 256]):
        """
        Initialize the actor network.
        
        Args:
            state_dim (int): Dimension of the state space
            action_dim (int): Number of possible actions
            hidden_dims (list): List of hidden layer dimensions
        """
        super(ActorNetwork, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Build the network layers
        layers = []
        input_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            input_dim = hidden_dim
        
        # Output layer (no activation, will use softmax in forward)
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
        Forward pass through the actor network.
        
        Args:
            state (torch.Tensor): Input state tensor
            
        Returns:
            torch.Tensor: Action logits
        """
        return self.network(state)
    
    def get_action_and_log_prob(self, state):
        """
        Get action and its log probability.
        
        Args:
            state (torch.Tensor): Input state
            
        Returns:
            tuple: (action, log_prob, entropy)
        """
        logits = self.forward(state)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        
        return action, log_prob, entropy
    
    def get_log_prob(self, state, action):
        """
        Get log probability of a specific action.
        
        Args:
            state (torch.Tensor): Input state
            action (torch.Tensor): Action taken
            
        Returns:
            tuple: (log_prob, entropy)
        """
        logits = self.forward(state)
        dist = Categorical(logits=logits)
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        
        return log_prob, entropy


class CriticNetwork(nn.Module):
    """Critic network for PPO - estimates state values."""
    
    def __init__(self, state_dim, hidden_dims=[256, 256]):
        """
        Initialize the critic network.
        
        Args:
            state_dim (int): Dimension of the state space
            hidden_dims (list): List of hidden layer dimensions
        """
        super(CriticNetwork, self).__init__()
        
        self.state_dim = state_dim
        
        # Build the network layers
        layers = []
        input_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            input_dim = hidden_dim
        
        # Output layer (single value)
        layers.append(nn.Linear(input_dim, 1))
        
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
        Forward pass through the critic network.
        
        Args:
            state (torch.Tensor): Input state tensor
            
        Returns:
            torch.Tensor: State value estimate
        """
        return self.network(state).squeeze(-1)


class ActorCriticNetwork(nn.Module):
    """Combined Actor-Critic network with shared features."""
    
    def __init__(self, state_dim, action_dim, hidden_dims=[256, 256]):
        """
        Initialize the actor-critic network.
        
        Args:
            state_dim (int): Dimension of the state space
            action_dim (int): Number of possible actions
            hidden_dims (list): List of hidden layer dimensions
        """
        super(ActorCriticNetwork, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Shared feature extraction layers
        self.shared_layers = nn.Sequential(
            nn.Linear(state_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Actor head (policy)
        self.actor_head = nn.Sequential(
            nn.Linear(hidden_dims[1], hidden_dims[1] // 2),
            nn.ReLU(),
            nn.Linear(hidden_dims[1] // 2, action_dim)
        )
        
        # Critic head (value function)
        self.critic_head = nn.Sequential(
            nn.Linear(hidden_dims[1], hidden_dims[1] // 2),
            nn.ReLU(),
            nn.Linear(hidden_dims[1] // 2, 1)
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
        Forward pass through the actor-critic network.
        
        Args:
            state (torch.Tensor): Input state tensor
            
        Returns:
            tuple: (action_logits, state_value)
        """
        features = self.shared_layers(state)
        action_logits = self.actor_head(features)
        state_value = self.critic_head(features).squeeze(-1)
        
        return action_logits, state_value
    
    def get_action_and_value(self, state):
        """
        Get action, log probability, and state value.
        
        Args:
            state (torch.Tensor): Input state
            
        Returns:
            tuple: (action, log_prob, entropy, value)
        """
        action_logits, state_value = self.forward(state)
        dist = Categorical(logits=action_logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        
        return action, log_prob, entropy, state_value
    
    def get_value(self, state):
        """
        Get state value estimate.
        
        Args:
            state (torch.Tensor): Input state
            
        Returns:
            torch.Tensor: State value
        """
        _, state_value = self.forward(state)
        return state_value
    
    def evaluate_actions(self, state, action):
        """
        Evaluate actions for PPO updates.
        
        Args:
            state (torch.Tensor): Input state
            action (torch.Tensor): Actions taken
            
        Returns:
            tuple: (log_prob, entropy, value)
        """
        action_logits, state_value = self.forward(state)
        dist = Categorical(logits=action_logits)
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        
        return log_prob, entropy, state_value
