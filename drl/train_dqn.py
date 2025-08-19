import sys
import os
import configparser
import numpy as np
import torch
import matplotlib.pyplot as plt
from datetime import datetime

# Add the shared module to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'shared'))

from quantum_environment import QuantumStateEnv
from dqn_agent import DQNAgent


def create_config():
    """Create a default configuration for the quantum environment."""
    config = configparser.ConfigParser()
    
    # System parameters
    config['system_parameters'] = {
        'chain_length': '16',
        'action_set': 'zhang',  # or 'oaps'
        'n_actions': '16',
        'field_strength': '100',
        'coupling': '1.0',
        'tstep_length': '0.15',
        'max_t_steps': '80',
        'tolerance': '0.05'
    }

    config["noise_parameters"] = {
    "noise": 'True',
    "noise_probability": '0.1',
    "noise_amplitude": '0.1',
}
    
    # Learning parameters
    config['learning_parameters'] = {
        'reward_function': 'original',  # or 'site evolution'
        'gamma': '1'
    }
    
    return config


def train_dqn(num_episodes=1000, config_path=None, save_dir='./dqn_results'):
    """
    Train a DQN agent on the quantum environment.
    
    Args:
        num_episodes (int): Number of training episodes
        config_path (str): Path to configuration file (optional)
        save_dir (str): Directory to save results
    """
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Load or create configuration
    if config_path and os.path.exists(config_path):
        config = configparser.ConfigParser()
        config.read(config_path)
    else:
        config = create_config()
        # Save default config
        with open(os.path.join(save_dir, 'config.ini'), 'w') as f:
            config.write(f)
    
    # Create environment
    env = QuantumStateEnv(config)
    
    # DQN configuration
    dqn_config = {
        'learning_rate': 1e-3,
        'gamma': 0.99,
        'epsilon_start': 1.0,
        'epsilon_end': 0.01,
        'epsilon_decay': 0.995,
        'target_update_freq': 200,
        'batch_size': 32,
        'buffer_size': 100000,
        'hidden_dims': [1024, 1024],
        'use_dueling': True,  # Use dueling architecture
        'use_prioritized_replay': True,  # Can enable for better performance
        'use_double_dqn': True,  # Use double DQN
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    # Create agent
    agent = DQNAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        config=dqn_config
    )
    
    print(f"Training DQN on {dqn_config['device']}")
    print(f"State dimension: {env.observation_space.shape[0]}")
    print(f"Action dimension: {env.action_space.n}")
    print(f"Training for {num_episodes} episodes")
    
    # Training loop
    best_reward = float('-inf')
    episode_stats = []
    
    for episode in range(num_episodes):
        # Train for one episode
        stats = agent.train_episode(env)
        episode_stats.append(stats)
        
        # Print progress
        if (episode + 1) % 100 == 0:
            recent_avg = np.mean([s['episode_reward'] for s in episode_stats[-100:]])
            recent_fidelity = np.mean([s['max_fidelity'] for s in episode_stats[-100:]])
            
            print(f"Episode {episode + 1}/{num_episodes}")
            print(f"  Recent avg reward: {recent_avg:.2f}")
            print(f"  Recent avg max fidelity: {recent_fidelity:.4f}")
            print(f"  Epsilon: {agent.epsilon:.4f}")
            print(f"  Buffer size: {len(agent.replay_buffer)}")
            
            # Save best model
            if recent_avg > best_reward:
                best_reward = recent_avg
                agent.save(os.path.join(save_dir, 'best_model.pth'))
        
        # Evaluate periodically
        if (episode + 1) % 500 == 0:
            eval_stats = agent.evaluate(env, num_episodes=10)
            print(f"  Evaluation at episode {episode + 1}:")
            print(f"    Mean reward: {eval_stats['mean_reward']:.2f} ± {eval_stats['std_reward']:.2f}")
            print(f"    Mean max fidelity: {eval_stats['mean_max_fidelity']:.4f} ± {eval_stats['std_max_fidelity']:.4f}")
    
    # Final evaluation
    print("\nFinal evaluation...")
    final_eval = agent.evaluate(env, num_episodes=100)
    print(f"Final mean reward: {final_eval['mean_reward']:.2f} ± {final_eval['std_reward']:.2f}")
    print(f"Final mean max fidelity: {final_eval['mean_max_fidelity']:.4f} ± {final_eval['std_max_fidelity']:.4f}")
    
    # Save final model and results
    agent.save(os.path.join(save_dir, 'final_model.pth'))
    
    # Plot training progress
    agent.plot_training_progress(os.path.join(save_dir, 'training_progress.png'))
    
    # Save training statistics
    np.save(os.path.join(save_dir, 'episode_stats.npy'), episode_stats)
    
    print(f"\nTraining completed! Results saved to {save_dir}")
    
    return agent, episode_stats


def evaluate_trained_agent(model_path, config_path=None, num_episodes=100):
    """
    Evaluate a trained DQN agent.
    
    Args:
        model_path (str): Path to the trained model
        config_path (str): Path to configuration file (optional)
        num_episodes (int): Number of episodes to evaluate
    """
    # Load configuration
    if config_path and os.path.exists(config_path):
        config = configparser.ConfigParser()
        config.read(config_path)
    else:
        config = create_config()
    
    # Create environment
    env = QuantumStateEnv(config)
    
    # Create agent (configuration will be loaded from checkpoint)
    agent = DQNAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n
    )
    
    # Load trained model
    agent.load(model_path)
    agent.epsilon = 0  # No exploration for evaluation
    
    print(f"Evaluating trained agent for {num_episodes} episodes...")
    
    # Evaluate
    eval_stats = agent.evaluate(env, num_episodes=num_episodes)
    
    print(f"Evaluation results:")
    print(f"  Mean reward: {eval_stats['mean_reward']:.2f} ± {eval_stats['std_reward']:.2f}")
    print(f"  Mean max fidelity: {eval_stats['mean_max_fidelity']:.4f} ± {eval_stats['std_max_fidelity']:.4f}")
    print(f"  Mean final fidelity: {eval_stats['mean_final_fidelity']:.4f} ± {eval_stats['std_final_fidelity']:.4f}")
    
    return eval_stats


if __name__ == "__main__":
    # Example usage
    print("Starting DQN training for quantum state control...")
    
    # Train the agent
    agent, stats = train_dqn(
        num_episodes=10000,
        save_dir='./dqn_quantum_results'
    )
    
    # Evaluate the trained agent
    final_eval = evaluate_trained_agent(
        './dqn_quantum_results/best_model.pth',
        num_episodes=100
    )
