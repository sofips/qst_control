import numpy as np
import torch
from collections import deque, namedtuple
import random


# Define experience tuple
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])


class ReplayBuffer:
    """Experience replay buffer for DQN training."""
    
    def __init__(self, capacity=100000, batch_size=32):
        """
        Initialize the replay buffer.
        
        Args:
            capacity (int): Maximum number of experiences to store
            batch_size (int): Size of batches to sample
        """
        self.buffer = deque(maxlen=capacity)
        self.batch_size = batch_size
        self.capacity = capacity
    
    def push(self, state, action, reward, next_state, done):
        """
        Add an experience to the buffer.
        
        Args:
            state (np.ndarray): Current state
            action (int): Action taken
            reward (float): Reward received
            next_state (np.ndarray): Next state
            done (bool): Whether the episode ended
        """
        experience = Experience(state, action, reward, next_state, done)
        self.buffer.append(experience)
    
    def sample(self):
        """
        Sample a batch of experiences from the buffer.
        
        Returns:
            tuple: Batch of (states, actions, rewards, next_states, dones)
        """
        experiences = random.sample(self.buffer, self.batch_size)
        
        states = torch.FloatTensor([e.state for e in experiences])
        actions = torch.LongTensor([e.action for e in experiences])
        rewards = torch.FloatTensor([e.reward for e in experiences])
        next_states = torch.FloatTensor([e.next_state for e in experiences])
        dones = torch.BoolTensor([e.done for e in experiences])
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        """Return the current size of the buffer."""
        return len(self.buffer)
    
    def is_ready(self):
        """Check if buffer has enough experiences for sampling."""
        return len(self.buffer) >= self.batch_size


class PrioritizedReplayBuffer:
    """Prioritized Experience Replay buffer for improved learning."""
    
    def __init__(self, capacity=100000, batch_size=32, alpha=0.6, beta=0.4, beta_increment=0.001):
        """
        Initialize the prioritized replay buffer.
        
        Args:
            capacity (int): Maximum number of experiences to store
            batch_size (int): Size of batches to sample
            alpha (float): Prioritization strength (0 = uniform, 1 = full prioritization)
            beta (float): Importance sampling strength (0 = no correction, 1 = full correction)
            beta_increment (float): Beta increment per sampling step
        """
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.batch_size = batch_size
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.max_priority = 1.0
        self.position = 0
    
    def push(self, state, action, reward, next_state, done):
        """
        Add an experience to the buffer with maximum priority.
        
        Args:
            state (np.ndarray): Current state
            action (int): Action taken
            reward (float): Reward received
            next_state (np.ndarray): Next state
            done (bool): Whether the episode ended
        """
        experience = Experience(state, action, reward, next_state, done)
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
            self.priorities.append(self.max_priority)
        else:
            self.buffer[self.position] = experience
            self.priorities[self.position] = self.max_priority
        
        self.position = (self.position + 1) % self.capacity
    
    def sample(self):
        """
        Sample a batch of experiences using prioritized sampling.
        
        Returns:
            tuple: Batch of (states, actions, rewards, next_states, dones, weights, indices)
        """
        if len(self.buffer) == self.capacity:
            priorities = np.array(self.priorities)
        else:
            priorities = np.array(list(self.priorities)[:len(self.buffer)])
        
        # Convert priorities to probabilities
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        # Sample indices
        indices = np.random.choice(len(self.buffer), self.batch_size, p=probs)
        experiences = [self.buffer[idx] for idx in indices]
        
        # Calculate importance sampling weights
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        
        # Update beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        states = torch.FloatTensor([e.state for e in experiences])
        actions = torch.LongTensor([e.action for e in experiences])
        rewards = torch.FloatTensor([e.reward for e in experiences])
        next_states = torch.FloatTensor([e.next_state for e in experiences])
        dones = torch.BoolTensor([e.done for e in experiences])
        weights = torch.FloatTensor(weights)
        
        return states, actions, rewards, next_states, dones, weights, indices
    
    def update_priorities(self, indices, priorities):
        """
        Update priorities for specific experiences.
        
        Args:
            indices (list): Indices of experiences to update
            priorities (np.ndarray): New priorities
        """
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)
    
    def __len__(self):
        """Return the current size of the buffer."""
        return len(self.buffer)
    
    def is_ready(self):
        """Check if buffer has enough experiences for sampling."""
        return len(self.buffer) >= self.batch_size
