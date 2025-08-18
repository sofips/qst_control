"""
Demo script to test the DQN implementation with the quantum environment.
This script will run a short training session to verify everything works.
"""

import sys
import os
import configparser
import numpy as np
import torch

# Add the shared module to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'shared'))

from quantum_environment import QuantumStateEnv
from dqn_agent import DQNAgent


def create_demo_config():
    """Create a simple configuration for demo purposes."""
    config = configparser.ConfigParser()
    
    # System parameters (smaller for faster demo)
    config['system_parameters'] = {
        'chain_length': '4',  # Smaller chain for faster training
        'action_set': 'zhang',
        'n_actions': '16',
        'field_strength': '1.0',
        'coupling': '1.0',
        'tstep_length': '0.1',
        'max_t_steps': '20',  # Shorter episodes
        'tolerance': '0.01'
    }
    
    # Learning parameters
    config['learning_parameters'] = {
        'reward_function': 'original',
        'gamma': '0.99'
    }
    
    return config


def test_environment():
    """Test that the environment works correctly."""
    print("Testing quantum environment...")
    
    config = create_demo_config()
    env = QuantumStateEnv(config)
    
    print(f"State space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print(f"Chain length: {env.chain_length}")
    print(f"Number of actions: {len(env.propagators)}")
    
    # Test environment reset and step
    state, info = env.reset()
    print(f"Initial state shape: {state.shape}")
    print(f"Initial fidelity: {info['state_fidelity']:.4f}")
    
    # Take a few random actions
    for i in range(5):
        action = env.action_space.sample()
        next_state, reward, done, truncated, info = env.step(action)
        print(f"Step {i+1}: action={action}, reward={reward:.2f}, fidelity={info['state_fidelity']:.4f}")
        
        if done or truncated:
            break
    
    print("Environment test completed successfully!\n")
    return env


def test_dqn_agent(env):
    """Test that the DQN agent works correctly."""
    print("Testing DQN agent...")
    
    # Simple configuration for testing
    dqn_config = {
        'learning_rate': 1e-3,
        'gamma': 0.99,
        'epsilon_start': 0.5,  # Lower epsilon for faster testing
        'epsilon_end': 0.01,
        'epsilon_decay': 0.99,
        'target_update_freq': 50,  # More frequent updates for demo
        'batch_size': 16,  # Smaller batch for demo
        'buffer_size': 1000,  # Smaller buffer for demo
        'hidden_dims': [64, 64],  # Smaller network for demo
        'use_dueling': False,
        'use_prioritized_replay': False,
        'use_double_dqn': True,
        'device': 'cpu'  # Use CPU for demo
    }
    
    agent = DQNAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        config=dqn_config
    )
    
    print(f"Agent created with device: {agent.device}")
    print(f"Q-network: {agent.q_network}")
    
    # Test action selection
    state, _ = env.reset()
    action = agent.get_action(state)
    print(f"Selected action: {action}")
    
    # Test experience storage and training
    next_state, reward, done, truncated, info = env.step(action)
    agent.store_experience(state, action, reward, next_state, done or truncated)
    
    print(f"Experience stored. Buffer size: {len(agent.replay_buffer)}")
    
    # Fill buffer with some experiences
    for _ in range(20):
        state, _ = env.reset()
        for _ in range(10):
            action = agent.get_action(state)
            next_state, reward, done, truncated, info = env.step(action)
            agent.store_experience(state, action, reward, next_state, done or truncated)
            state = next_state
            if done or truncated:
                break
    
    print(f"Buffer filled. Size: {len(agent.replay_buffer)}")
    
    # Test training step
    if agent.replay_buffer.is_ready():
        loss = agent.train_step()
        print(f"Training step completed. Loss: {loss:.4f}")
    
    print("DQN agent test completed successfully!\n")
    return agent


def run_mini_training(env, agent, num_episodes=10):
    """Run a mini training session to verify everything works."""
    print(f"Running mini training session ({num_episodes} episodes)...")
    
    episode_rewards = []
    
    for episode in range(num_episodes):
        stats = agent.train_episode(env)
        episode_rewards.append(stats['episode_reward'])
        
        print(f"Episode {episode+1}: "
              f"reward={stats['episode_reward']:.2f}, "
              f"max_fidelity={stats['max_fidelity']:.4f}, "
              f"epsilon={stats['epsilon']:.4f}")
    
    mean_reward = np.mean(episode_rewards)
    print(f"\nMini training completed!")
    print(f"Mean episode reward: {mean_reward:.2f}")
    print(f"Final epsilon: {agent.epsilon:.4f}")
    
    # Test evaluation
    eval_stats = agent.evaluate(env, num_episodes=3)
    print(f"Evaluation - Mean reward: {eval_stats['mean_reward']:.2f}, "
          f"Mean max fidelity: {eval_stats['mean_max_fidelity']:.4f}")


def main():
    """Main demo function."""
    print("=== DQN Quantum Control Demo ===\n")
    
    # Test 1: Environment
    env = test_environment()
    
    # Test 2: DQN Agent
    agent = test_dqn_agent(env)
    
    # Test 3: Mini training
    run_mini_training(env, agent, num_episodes=20)
    
    print("\n=== Demo completed successfully! ===")
    print("You can now run the full training with train_dqn.py")


if __name__ == "__main__":
    main()
