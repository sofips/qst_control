import torch
import numpy as np
from collections import namedtuple

# Define trajectory tuple
Trajectory = namedtuple('Trajectory', [
    'states', 'actions', 'log_probs', 'rewards', 
    'values', 'dones', 'advantages', 'returns'
])


class PPOMemory:
    """Memory buffer for storing PPO trajectories and computing advantages."""
    
    def __init__(self, gamma=0.99, gae_lambda=0.95):
        """
        Initialize the PPO memory buffer.
        
        Args:
            gamma (float): Discount factor
            gae_lambda (float): GAE lambda parameter
        """
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        
        # Storage for trajectory data
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
        
        # Computed advantages and returns
        self.advantages = []
        self.returns = []
        
    def store_transition(self, state, action, log_prob, reward, value, done):
        """
        Store a single transition.
        
        Args:
            state (np.ndarray): Current state
            action (int): Action taken
            log_prob (float): Log probability of action
            reward (float): Reward received
            value (float): Value estimate
            done (bool): Whether episode ended
        """
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)
    
    def compute_advantages_and_returns(self, next_value=0):
        """
        Compute advantages using Generalized Advantage Estimation (GAE).
        
        Args:
            next_value (float): Value of next state (for bootstrapping)
        """
        # Convert to numpy arrays for easier computation
        rewards = np.array(self.rewards)
        values = np.array(self.values + [next_value])
        dones = np.array(self.dones + [True])
        
        # Compute advantages using GAE
        advantages = np.zeros_like(rewards)
        last_advantage = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[t + 1]
                next_value_est = values[t + 1]
            else:
                next_non_terminal = 1.0 - dones[t + 1]
                next_value_est = values[t + 1]
            
            delta = (rewards[t] + 
                    self.gamma * next_value_est * next_non_terminal - 
                    values[t])
            
            advantages[t] = (delta + 
                           self.gamma * self.gae_lambda * 
                           next_non_terminal * last_advantage)
            
            last_advantage = advantages[t]
        
        # Compute returns (advantages + values)
        returns = advantages + values[:-1]
        
        self.advantages = advantages.tolist()
        self.returns = returns.tolist()
    
    def get_batches(self, batch_size):
        """
        Get mini-batches for training.
        
        Args:
            batch_size (int): Size of each mini-batch
            
        Yields:
            dict: Mini-batch of data
        """
        # Convert to tensors (fix: list to numpy array first)
        states = torch.FloatTensor(np.array(self.states))
        actions = torch.LongTensor(self.actions)
        log_probs = torch.FloatTensor(self.log_probs)
        advantages = torch.FloatTensor(self.advantages)
        returns = torch.FloatTensor(self.returns)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Get total number of samples
        total_samples = len(self.states)
        
        # Generate random indices
        indices = torch.randperm(total_samples)
        
        # Yield mini-batches
        for start in range(0, total_samples, batch_size):
            end = min(start + batch_size, total_samples)
            batch_indices = indices[start:end]
            
            yield {
                'states': states[batch_indices],
                'actions': actions[batch_indices],
                'old_log_probs': log_probs[batch_indices],
                'advantages': advantages[batch_indices],
                'returns': returns[batch_indices]
            }
    
    def clear(self):
        """Clear all stored data."""
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.values.clear()
        self.dones.clear()
        self.advantages.clear()
        self.returns.clear()
    
    def __len__(self):
        """Return the number of stored transitions."""
        return len(self.states)
    
    def is_ready(self, min_samples=1):
        """Check if buffer has enough samples for training."""
        return len(self.states) >= min_samples


class RolloutBuffer:
    """Alternative rollout buffer implementation for PPO."""
    
    def __init__(self, buffer_size, state_dim, gamma=0.99, gae_lambda=0.95):
        """
        Initialize the rollout buffer.
        
        Args:
            buffer_size (int): Maximum number of transitions to store
            state_dim (int): Dimension of state space
            gamma (float): Discount factor
            gae_lambda (float): GAE lambda parameter
        """
        self.buffer_size = buffer_size
        self.state_dim = state_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        
        # Pre-allocate arrays for efficiency
        self.states = np.zeros((buffer_size, state_dim), dtype=np.float32)
        self.actions = np.zeros(buffer_size, dtype=np.int32)
        self.log_probs = np.zeros(buffer_size, dtype=np.float32)
        self.rewards = np.zeros(buffer_size, dtype=np.float32)
        self.values = np.zeros(buffer_size, dtype=np.float32)
        self.dones = np.zeros(buffer_size, dtype=bool)
        self.advantages = np.zeros(buffer_size, dtype=np.float32)
        self.returns = np.zeros(buffer_size, dtype=np.float32)
        
        self.ptr = 0
        self.path_start_idx = 0
        self.max_size = buffer_size
    
    def store(self, state, action, log_prob, reward, value, done):
        """
        Store a single transition.
        
        Args:
            state (np.ndarray): Current state
            action (int): Action taken
            log_prob (float): Log probability of action
            reward (float): Reward received
            value (float): Value estimate
            done (bool): Whether episode ended
        """
        assert self.ptr < self.max_size
        
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.log_probs[self.ptr] = log_prob
        self.rewards[self.ptr] = reward
        self.values[self.ptr] = value
        self.dones[self.ptr] = done
        
        self.ptr += 1
    
    def finish_path(self, last_value=0):
        """
        Finish a trajectory and compute advantages.
        
        Args:
            last_value (float): Value of the last state (for bootstrapping)
        """
        path_slice = slice(self.path_start_idx, self.ptr)
        rewards = np.append(self.rewards[path_slice], last_value)
        values = np.append(self.values[path_slice], last_value)
        
        # Compute advantages using GAE
        deltas = rewards[:-1] + self.gamma * values[1:] - values[:-1]
        self.advantages[path_slice] = self._discount_cumsum(
            deltas, self.gamma * self.gae_lambda
        )
        
        # Compute returns
        self.returns[path_slice] = self._discount_cumsum(
            rewards, self.gamma
        )[:-1]
        
        self.path_start_idx = self.ptr
    
    def get(self):
        """
        Get all data and reset buffer.
        
        Returns:
            dict: All stored data
        """
        assert self.ptr == self.max_size
        self.ptr, self.path_start_idx = 0, 0
        
        # Normalize advantages
        adv_mean, adv_std = np.mean(self.advantages), np.std(self.advantages)
        self.advantages = (self.advantages - adv_mean) / (adv_std + 1e-8)
        
        data = {
            'states': torch.FloatTensor(self.states),
            'actions': torch.LongTensor(self.actions),
            'old_log_probs': torch.FloatTensor(self.log_probs),
            'advantages': torch.FloatTensor(self.advantages),
            'returns': torch.FloatTensor(self.returns)
        }
        
        return data
    
    def _discount_cumsum(self, x, discount):
        """Compute discounted cumulative sums."""
        return np.array([
            np.sum(discount ** np.arange(len(x[i:])) * x[i:])
            for i in range(len(x))
        ])
