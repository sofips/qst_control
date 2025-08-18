import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import os
import pickle

from dqn_network import DQNNetwork, DuelingDQNNetwork
from replay_buffer import ReplayBuffer, PrioritizedReplayBuffer


class DQNAgent:
    """Deep Q-Network agent for quantum state control."""
    
    def __init__(self, state_dim, action_dim, config=None):
        """
        Initialize the DQN agent.
        
        Args:
            state_dim (int): Dimension of the state space
            action_dim (int): Number of possible actions
            config (dict): Configuration parameters
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Default configuration
        default_config = {
            'learning_rate': 1e-4,
            'gamma': 0.99,
            'epsilon_start': 1.0,
            'epsilon_end': 0.01,
            'epsilon_decay': 0.995,
            'target_update_freq': 1000,
            'batch_size': 32,
            'buffer_size': 100000,
            'hidden_dims': [256, 256],
            'use_dueling': False,
            'use_prioritized_replay': False,
            'use_double_dqn': True,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu'
        }
        
        self.config = {**default_config, **(config or {})}
        self.device = torch.device(self.config['device'])
        
        # Initialize networks
        network_class = DuelingDQNNetwork if self.config['use_dueling'] else DQNNetwork
        
        self.q_network = network_class(
            state_dim, action_dim, self.config['hidden_dims']
        ).to(self.device)
        
        self.target_network = network_class(
            state_dim, action_dim, self.config['hidden_dims']
        ).to(self.device)
        
        # Copy weights to target network
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        # Initialize optimizer
        self.optimizer = optim.Adam(
            self.q_network.parameters(), 
            lr=self.config['learning_rate']
        )
        
        # Initialize replay buffer
        if self.config['use_prioritized_replay']:
            self.replay_buffer = PrioritizedReplayBuffer(
                capacity=self.config['buffer_size'],
                batch_size=self.config['batch_size']
            )
        else:
            self.replay_buffer = ReplayBuffer(
                capacity=self.config['buffer_size'],
                batch_size=self.config['batch_size']
            )
        
        # Training parameters
        self.epsilon = self.config['epsilon_start']
        self.steps_done = 0
        self.episode_rewards = []
        self.episode_losses = []
        self.recent_rewards = deque(maxlen=100)
        
    def get_action(self, state):
        """
        Select an action using epsilon-greedy policy.
        
        Args:
            state (np.ndarray): Current state
            
        Returns:
            int: Selected action
        """
        return self.q_network.get_action(state, self.epsilon)
    
    def store_experience(self, state, action, reward, next_state, done):
        """
        Store experience in replay buffer.
        
        Args:
            state (np.ndarray): Current state
            action (int): Action taken
            reward (float): Reward received
            next_state (np.ndarray): Next state
            done (bool): Whether episode ended
        """
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def update_epsilon(self):
        """Update epsilon for exploration."""
        self.epsilon = max(
            self.config['epsilon_end'],
            self.epsilon * self.config['epsilon_decay']
        )
    
    def update_target_network(self):
        """Update target network weights."""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def compute_loss(self, batch):
        """
        Compute the DQN loss.
        
        Args:
            batch: Batch of experiences
            
        Returns:
            torch.Tensor: Computed loss
        """
        if self.config['use_prioritized_replay']:
            states, actions, rewards, next_states, dones, weights, indices = batch
        else:
            states, actions, rewards, next_states, dones = batch
            weights = torch.ones(len(states)).to(self.device)
        
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        weights = weights.to(self.device)
        
        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Next Q values
        with torch.no_grad():
            if self.config['use_double_dqn']:
                # Double DQN: use main network to select actions, target network to evaluate
                next_actions = self.q_network(next_states).argmax(1, keepdim=True)
                next_q_values = self.target_network(next_states).gather(1, next_actions)
            else:
                # Standard DQN
                next_q_values = self.target_network(next_states).max(1)[0].unsqueeze(1)
            
            target_q_values = rewards.unsqueeze(1) + (
                self.config['gamma'] * next_q_values * (~dones).unsqueeze(1)
            )
        
        # Compute loss
        td_errors = current_q_values - target_q_values
        loss = (weights.unsqueeze(1) * td_errors.pow(2)).mean()
        
        # Update priorities if using prioritized replay
        if self.config['use_prioritized_replay']:
            priorities = td_errors.abs().detach().cpu().numpy().flatten()
            self.replay_buffer.update_priorities(indices, priorities)
        
        return loss
    
    def train_step(self):
        """
        Perform one training step.
        
        Returns:
            float: Loss value or None if not ready to train
        """
        if not self.replay_buffer.is_ready():
            return None
        
        # Sample batch
        batch = self.replay_buffer.sample()
        
        # Compute loss
        loss = self.compute_loss(batch)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        
        self.optimizer.step()
        
        # Update target network
        self.steps_done += 1
        if self.steps_done % self.config['target_update_freq'] == 0:
            self.update_target_network()
        
        return loss.item()
    
    def train_episode(self, env):
        """
        Train for one episode.
        
        Args:
            env: Environment to train on
            
        Returns:
            dict: Episode statistics
        """
        state, _ = env.reset()
        total_reward = 0
        episode_loss = 0
        step_count = 0
        
        while True:
            # Select action
            action = self.get_action(state)
            
            # Take step
            next_state, reward, done, truncated, info = env.step(action)
            
            # Store experience
            self.store_experience(state, action, reward, next_state, done or truncated)
            
            # Train
            loss = self.train_step()
            if loss is not None:
                episode_loss += loss
                step_count += 1
            
            total_reward += reward
            state = next_state
            
            if done or truncated:
                break
        
        # Update epsilon
        self.update_epsilon()
        
        # Store episode statistics
        self.episode_rewards.append(total_reward)
        self.recent_rewards.append(total_reward)
        
        if step_count > 0:
            self.episode_losses.append(episode_loss / step_count)
        
        return {
            'episode_reward': total_reward,
            'episode_loss': episode_loss / max(1, step_count),
            'epsilon': self.epsilon,
            'max_fidelity': info.get('max_fidelity', 0),
            'final_fidelity': info.get('state_fidelity', 0),
            'steps': step_count
        }
    
    def evaluate(self, env, num_episodes=10):
        """
        Evaluate the agent's performance.
        
        Args:
            env: Environment to evaluate on
            num_episodes (int): Number of episodes to evaluate
            
        Returns:
            dict: Evaluation statistics
        """
        old_epsilon = self.epsilon
        self.epsilon = 0  # No exploration during evaluation
        
        episode_rewards = []
        max_fidelities = []
        final_fidelities = []
        
        for _ in range(num_episodes):
            state, _ = env.reset()
            total_reward = 0
            
            while True:
                action = self.get_action(state)
                state, reward, done, truncated, info = env.step(action)
                total_reward += reward
                
                if done or truncated:
                    break
            
            episode_rewards.append(total_reward)
            max_fidelities.append(info.get('max_fidelity', 0))
            final_fidelities.append(info.get('state_fidelity', 0))
        
        self.epsilon = old_epsilon  # Restore epsilon
        
        return {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'mean_max_fidelity': np.mean(max_fidelities),
            'std_max_fidelity': np.std(max_fidelities),
            'mean_final_fidelity': np.mean(final_fidelities),
            'std_final_fidelity': np.std(final_fidelities)
        }
    
    def save(self, filepath):
        """
        Save the agent's state.
        
        Args:
            filepath (str): Path to save the agent
        """
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps_done': self.steps_done,
            'episode_rewards': self.episode_rewards,
            'episode_losses': self.episode_losses,
            'config': self.config
        }, filepath)
    
    def load(self, filepath):
        """
        Load the agent's state.
        
        Args:
            filepath (str): Path to load the agent from
        """
        checkpoint = torch.load(filepath, map_location=self.device, 
                               weights_only=False)
        
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.steps_done = checkpoint['steps_done']
        self.episode_rewards = checkpoint['episode_rewards']
        self.episode_losses = checkpoint['episode_losses']
    
    def plot_training_progress(self, save_path=None):
        """
        Plot training progress.
        
        Args:
            save_path (str): Path to save the plot
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot rewards
        if self.episode_rewards:
            ax1.plot(self.episode_rewards, alpha=0.6, label='Episode Reward')
            
            # Moving average
            if len(self.episode_rewards) > 10:
                window = min(100, len(self.episode_rewards) // 10)
                moving_avg = np.convolve(
                    self.episode_rewards, 
                    np.ones(window)/window, 
                    mode='valid'
                )
                ax1.plot(range(window-1, len(self.episode_rewards)), 
                        moving_avg, 'r-', label=f'Moving Average ({window})')
            
            ax1.set_xlabel('Episode')
            ax1.set_ylabel('Reward')
            ax1.set_title('Training Rewards')
            ax1.legend()
            ax1.grid(True)
        
        # Plot losses
        if self.episode_losses:
            ax2.plot(self.episode_losses, alpha=0.6, label='Episode Loss')
            ax2.set_xlabel('Episode')
            ax2.set_ylabel('Loss')
            ax2.set_title('Training Losses')
            ax2.legend()
            ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
