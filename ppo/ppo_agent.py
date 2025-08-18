import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import os

from ppo_networks import ActorNetwork, CriticNetwork, ActorCriticNetwork
from ppo_memory import PPOMemory, RolloutBuffer


class PPOAgent:
    """Proximal Policy Optimization agent for quantum state control."""
    
    def __init__(self, state_dim, action_dim, config=None):
        """
        Initialize the PPO agent.
        
        Args:
            state_dim (int): Dimension of the state space
            action_dim (int): Number of possible actions
            config (dict): Configuration parameters
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Default configuration
        default_config = {
            'learning_rate': 3e-4,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'clip_ratio': 0.2,
            'vf_coef': 0.5,
            'ent_coef': 0.01,
            'max_grad_norm': 0.5,
            'update_epochs': 10,
            'batch_size': 64,
            'buffer_size': 2048,
            'hidden_dims': [256, 256],
            'use_shared_network': True,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu'
        }
        
        self.config = {**default_config, **(config or {})}
        self.device = torch.device(self.config['device'])
        
        # Initialize networks
        if self.config['use_shared_network']:
            self.network = ActorCriticNetwork(
                state_dim, action_dim, self.config['hidden_dims']
            ).to(self.device)
            self.actor = None
            self.critic = None
        else:
            self.actor = ActorNetwork(
                state_dim, action_dim, self.config['hidden_dims']
            ).to(self.device)
            self.critic = CriticNetwork(
                state_dim, self.config['hidden_dims']
            ).to(self.device)
            self.network = None
        
        # Initialize optimizer
        if self.network:
            self.optimizer = optim.Adam(
                self.network.parameters(),
                lr=self.config['learning_rate']
            )
        else:
            # Separate optimizers for actor and critic
            self.actor_optimizer = optim.Adam(
                self.actor.parameters(),
                lr=self.config['learning_rate']
            )
            self.critic_optimizer = optim.Adam(
                self.critic.parameters(),
                lr=self.config['learning_rate']
            )
        
        # Initialize memory buffer
        self.memory = PPOMemory(
            gamma=self.config['gamma'],
            gae_lambda=self.config['gae_lambda']
        )
        
        # Training statistics
        self.episode_rewards = []
        self.episode_lengths = []
        self.policy_losses = []
        self.value_losses = []
        self.entropy_losses = []
        self.recent_rewards = deque(maxlen=100)
        
    def get_action_and_value(self, state):
        """
        Get action and value estimate for a given state.
        
        Args:
            state (np.ndarray): Current state
            
        Returns:
            tuple: (action, log_prob, value)
        """
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            if self.network:
                action, log_prob, entropy, value = self.network.get_action_and_value(state)
            else:
                action, log_prob, entropy = self.actor.get_action_and_log_prob(state)
                value = self.critic(state)
        
        return action.item(), log_prob.item(), value.item()
    
    def get_action(self, state, deterministic=False):
        """
        Get action only (for evaluation).
        
        Args:
            state (np.ndarray): Current state
            deterministic (bool): If True, return highest probability action
            
        Returns:
            int: Selected action
        """
        if deterministic:
            return self.get_deterministic_action(state)
        else:
            action, _, _ = self.get_action_and_value(state)
            return action
    
    def get_deterministic_action(self, state):
        """
        Get the action with highest probability (deterministic evaluation).
        
        Args:
            state (np.ndarray): Current state
            
        Returns:
            int: Action with highest probability
        """
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            if self.network:
                action_logits, _ = self.network(state)
            else:
                action_logits = self.actor(state)
            
            # Get action with highest probability (argmax)
            action = torch.argmax(action_logits, dim=-1)
        
        return action.item()
    
    def store_transition(self, state, action, log_prob, reward, value, done):
        """
        Store a transition in memory.
        
        Args:
            state (np.ndarray): Current state
            action (int): Action taken
            log_prob (float): Log probability of action
            reward (float): Reward received
            value (float): Value estimate
            done (bool): Whether episode ended
        """
        self.memory.store_transition(state, action, log_prob, reward, value, done)
    
    def update(self):
        """
        Update the policy using PPO.
        
        Returns:
            dict: Training statistics
        """
        if len(self.memory) == 0:
            return None
        
        # Compute advantages and returns
        self.memory.compute_advantages_and_returns()
        
        # Training statistics
        policy_losses = []
        value_losses = []
        entropy_losses = []
        
        # Multiple epochs of updates
        for epoch in range(self.config['update_epochs']):
            # Get mini-batches
            for batch in self.memory.get_batches(self.config['batch_size']):
                # Move to device
                states = batch['states'].to(self.device)
                actions = batch['actions'].to(self.device)
                old_log_probs = batch['old_log_probs'].to(self.device)
                advantages = batch['advantages'].to(self.device)
                returns = batch['returns'].to(self.device)
                
                # Forward pass
                if self.network:
                    new_log_probs, entropy, values = self.network.evaluate_actions(states, actions)
                else:
                    new_log_probs, entropy = self.actor.get_log_prob(states, actions)
                    values = self.critic(states)
                
                # Compute PPO loss
                ratio = torch.exp(new_log_probs - old_log_probs)
                
                # Policy loss (clipped)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - self.config['clip_ratio'], 
                                  1 + self.config['clip_ratio']) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = nn.MSELoss()(values, returns)
                
                # Entropy loss (for exploration)
                entropy_loss = -entropy.mean()
                
                # Total loss
                total_loss = (policy_loss + 
                            self.config['vf_coef'] * value_loss + 
                            self.config['ent_coef'] * entropy_loss)
                
                # Backward pass
                if self.network:
                    self.optimizer.zero_grad()
                    total_loss.backward()
                    nn.utils.clip_grad_norm_(self.network.parameters(), 
                                           self.config['max_grad_norm'])
                    self.optimizer.step()
                else:
                    # Separate updates for actor and critic
                    actor_loss = policy_loss + self.config['ent_coef'] * entropy_loss
                    
                    self.actor_optimizer.zero_grad()
                    actor_loss.backward(retain_graph=True)
                    nn.utils.clip_grad_norm_(self.actor.parameters(), 
                                           self.config['max_grad_norm'])
                    self.actor_optimizer.step()
                    
                    self.critic_optimizer.zero_grad()
                    value_loss.backward()
                    nn.utils.clip_grad_norm_(self.critic.parameters(), 
                                           self.config['max_grad_norm'])
                    self.critic_optimizer.step()
                
                # Store losses
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(entropy_loss.item())
        
        # Clear memory
        self.memory.clear()
        
        # Store training statistics
        stats = {
            'policy_loss': np.mean(policy_losses),
            'value_loss': np.mean(value_losses),
            'entropy_loss': np.mean(entropy_losses)
        }
        
        self.policy_losses.append(stats['policy_loss'])
        self.value_losses.append(stats['value_loss'])
        self.entropy_losses.append(stats['entropy_loss'])
        
        return stats
    
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
        episode_length = 0
        
        while True:
            # Get action and value
            action, log_prob, value = self.get_action_and_value(state)
            
            # Take step
            next_state, reward, done, truncated, info = env.step(action)
            
            # Store transition
            self.store_transition(state, action, log_prob, reward, value, done or truncated)
            
            total_reward += reward
            episode_length += 1
            state = next_state
            
            if done or truncated:
                break
        
        # Store episode statistics
        self.episode_rewards.append(total_reward)
        self.episode_lengths.append(episode_length)
        self.recent_rewards.append(total_reward)
        
        return {
            'episode_reward': total_reward,
            'episode_length': episode_length,
            'max_fidelity': info.get('max_fidelity', 0),
            'final_fidelity': info.get('state_fidelity', 0)
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
        episode_rewards = []
        episode_lengths = []
        max_fidelities = []
        final_fidelities = []
        
        for _ in range(num_episodes):
            state, _ = env.reset()
            total_reward = 0
            episode_length = 0
            
            while True:
                action = self.get_action(state)
                state, reward, done, truncated, info = env.step(action)
                total_reward += reward
                episode_length += 1
                
                if done or truncated:
                    break
            
            episode_rewards.append(total_reward)
            episode_lengths.append(episode_length)
            max_fidelities.append(info.get('max_fidelity', 0))
            final_fidelities.append(info.get('state_fidelity', 0))
        
        return {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'mean_length': np.mean(episode_lengths),
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
        checkpoint = {
            'config': self.config,
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'policy_losses': self.policy_losses,
            'value_losses': self.value_losses,
            'entropy_losses': self.entropy_losses
        }
        
        if self.network:
            checkpoint['network_state_dict'] = self.network.state_dict()
            checkpoint['optimizer_state_dict'] = self.optimizer.state_dict()
        else:
            checkpoint['actor_state_dict'] = self.actor.state_dict()
            checkpoint['critic_state_dict'] = self.critic.state_dict()
            checkpoint['actor_optimizer_state_dict'] = self.actor_optimizer.state_dict()
            checkpoint['critic_optimizer_state_dict'] = self.critic_optimizer.state_dict()
        
        torch.save(checkpoint, filepath)
    
    def load(self, filepath):
        """
        Load the agent's state.
        
        Args:
            filepath (str): Path to load the agent from
        """
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        
        if 'network_state_dict' in checkpoint:
            self.network.load_state_dict(checkpoint['network_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        else:
            self.actor.load_state_dict(checkpoint['actor_state_dict'])
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
            self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
            self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        
        self.episode_rewards = checkpoint['episode_rewards']
        self.episode_lengths = checkpoint['episode_lengths']
        self.policy_losses = checkpoint['policy_losses']
        self.value_losses = checkpoint['value_losses']
        self.entropy_losses = checkpoint['entropy_losses']
    
    def plot_training_progress(self, save_path=None):
        """
        Plot training progress.
        
        Args:
            save_path (str): Path to save the plot
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
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
        
        # Plot episode lengths
        if self.episode_lengths:
            ax2.plot(self.episode_lengths, alpha=0.6, label='Episode Length')
            ax2.set_xlabel('Episode')
            ax2.set_ylabel('Length')
            ax2.set_title('Episode Lengths')
            ax2.legend()
            ax2.grid(True)
        
        # Plot policy loss
        if self.policy_losses:
            ax3.plot(self.policy_losses, alpha=0.6, label='Policy Loss')
            ax3.set_xlabel('Update')
            ax3.set_ylabel('Loss')
            ax3.set_title('Policy Loss')
            ax3.legend()
            ax3.grid(True)
        
        # Plot value loss
        if self.value_losses:
            ax4.plot(self.value_losses, alpha=0.6, label='Value Loss')
            ax4.set_xlabel('Update')
            ax4.set_ylabel('Loss')
            ax4.set_title('Value Loss')
            ax4.legend()
            ax4.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
