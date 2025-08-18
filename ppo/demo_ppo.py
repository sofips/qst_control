"""
Demo script to test the PPO implementation with the quantum environment.
This script will run a short training session to verify everything works.
"""

import sys
import os
import configparser
import numpy as np

# Add the shared module to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'shared'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'drl'))

from quantum_environment import QuantumStateEnv
from ppo_agent import PPOAgent


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
        print(f"Step {i+1}: action={action}, reward={reward:.2f}, "
              f"fidelity={info['state_fidelity']:.4f}")
        
        if done or truncated:
            break
    
    print("Environment test completed successfully!\n")
    return env


def test_ppo_agent(env):
    """Test that the PPO agent works correctly."""
    print("Testing PPO agent...")
    
    # Simple configuration for testing
    ppo_config = {
        'learning_rate': 1e-3,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_ratio': 0.2,
        'vf_coef': 0.5,
        'ent_coef': 0.01,
        'max_grad_norm': 0.5,
        'update_epochs': 4,  # Fewer epochs for demo
        'batch_size': 16,  # Smaller batch for demo
        'buffer_size': 1000,  # Not used with simple memory
        'hidden_dims': [64, 64],  # Smaller network for demo
        'use_shared_network': True,
        'device': 'cpu'  # Use CPU for demo
    }
    
    agent = PPOAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        config=ppo_config
    )
    
    print(f"Agent created with device: {agent.device}")
    print(f"Using shared network: {agent.network is not None}")
    
    # Test action selection
    state, _ = env.reset()
    action, log_prob, value = agent.get_action_and_value(state)
    print(f"Selected action: {action}, log_prob: {log_prob:.4f}, "
          f"value: {value:.4f}")
    
    # Test transition storage
    next_state, reward, done, truncated, info = env.step(action)
    agent.store_transition(state, action, log_prob, reward, value, done or truncated)
    
    print(f"Transition stored. Memory size: {len(agent.memory)}")
    
    # Fill memory with some experiences
    for _ in range(10):
        state, _ = env.reset()
        for _ in range(5):
            action, log_prob, value = agent.get_action_and_value(state)
            next_state, reward, done, truncated, info = env.step(action)
            agent.store_transition(state, action, log_prob, reward, value, done or truncated)
            state = next_state
            if done or truncated:
                break
    
    print(f"Memory filled. Size: {len(agent.memory)}")
    
    # Test update
    if len(agent.memory) > 0:
        update_stats = agent.update()
        if update_stats:
            print(f"Update completed. Policy loss: {update_stats['policy_loss']:.4f}, "
                  f"Value loss: {update_stats['value_loss']:.4f}")
    
    print("PPO agent test completed successfully!\n")
    return agent


def run_mini_training(env, agent, num_episodes=20):
    """Run a mini training session to verify everything works."""
    print(f"Running mini training session ({num_episodes} episodes)...")
    
    episode_rewards = []
    
    for episode in range(num_episodes):
        episode_stats = agent.train_episode(env)
        episode_rewards.append(episode_stats['episode_reward'])
        
        # Update every 5 episodes
        if (episode + 1) % 5 == 0:
            update_stats = agent.update()
            if update_stats:
                print(f"Episode {episode+1}: "
                      f"reward={episode_stats['episode_reward']:.2f}, "
                      f"max_fidelity={episode_stats['max_fidelity']:.4f}, "
                      f"policy_loss={update_stats['policy_loss']:.4f}")
            else:
                print(f"Episode {episode+1}: "
                      f"reward={episode_stats['episode_reward']:.2f}, "
                      f"max_fidelity={episode_stats['max_fidelity']:.4f}")
    
    mean_reward = np.mean(episode_rewards)
    print(f"\nMini training completed!")
    print(f"Mean episode reward: {mean_reward:.2f}")
    
    # Test evaluation
    eval_stats = agent.evaluate(env, num_episodes=3)
    print(f"Evaluation - Mean reward: {eval_stats['mean_reward']:.2f}, "
          f"Mean max fidelity: {eval_stats['mean_max_fidelity']:.4f}")


def compare_with_random_policy(env, agent):
    """Compare trained agent with random policy."""
    print("\nComparing with random policy...")
    
    # Evaluate trained agent
    agent_stats = agent.evaluate(env, num_episodes=10)
    
    # Evaluate random policy
    random_rewards = []
    random_fidelities = []
    
    for _ in range(10):
        state, _ = env.reset()
        total_reward = 0
        
        while True:
            action = env.action_space.sample()  # Random action
            state, reward, done, truncated, info = env.step(action)
            total_reward += reward
            
            if done or truncated:
                break
        
        random_rewards.append(total_reward)
        random_fidelities.append(info.get('max_fidelity', 0))
    
    print(f"Trained PPO Agent:")
    print(f"  Mean reward: {agent_stats['mean_reward']:.2f}")
    print(f"  Mean max fidelity: {agent_stats['mean_max_fidelity']:.4f}")
    
    print(f"Random Policy:")
    print(f"  Mean reward: {np.mean(random_rewards):.2f}")
    print(f"  Mean max fidelity: {np.mean(random_fidelities):.4f}")


def main():
    """Main demo function."""
    print("=== PPO Quantum Control Demo ===\n")
    
    # Test 1: Environment
    env = test_environment()
    
    # Test 2: PPO Agent
    agent = test_ppo_agent(env)
    
    # Test 3: Mini training
    run_mini_training(env, agent, num_episodes=30)
    
    # Test 4: Compare with random policy
    compare_with_random_policy(env, agent)
    
    print("\n=== Demo completed successfully! ===")
    print("You can now run the full training with train_ppo.py")


if __name__ == "__main__":
    main()
