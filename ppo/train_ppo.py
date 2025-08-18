import sys
import os
import configparser
import numpy as np
import torch
import matplotlib.pyplot as plt
from datetime import datetime

# Add the shared module to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'shared'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'drl'))

from quantum_environment import QuantumStateEnv
from ppo_agent import PPOAgent


def create_config():
    """Create a default configuration for the quantum environment."""
    config = configparser.ConfigParser()

    # System parameters
    config['system_parameters'] = {
        'chain_length': '8',
        'action_set': 'zhang',  # or 'oaps'
        'n_actions': '16',
        'field_strength': '100',
        'coupling': '1.0',
        'tstep_length': '0.15',
        'max_t_steps': '40',
        'tolerance': '0.05'
    }

    # Learning parameters
    config['learning_parameters'] = {
        'reward_function': 'original',  # or 'site evolution'
        'gamma': '0.95'
    }

    return config


def train_ppo(num_episodes=2000, config_path=None, save_dir='./ppo_results'):
    """
    Train a PPO agent on the quantum environment.

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

    # PPO configuration
    ppo_config = {
        'learning_rate': 3e-4,          # Learning rate
        'gamma': 0.99,                  # Discount factor
        'gae_lambda': 0.95,             # GAE lambda
        'clip_ratio': 0.2,              # PPO clipping ratio
        'vf_coef': 0.5,                 # Value function coefficient
        'ent_coef': 0.01,               # Entropy coefficient
        'max_grad_norm': 0.5,           # Gradient clipping
        'update_epochs': 10,            # PPO update epochs
        'batch_size': 64,               # Mini-batch size
        'buffer_size': 2048,            # Not used with simple memory
        'hidden_dims': [256, 256],      # Network architecture
        'use_shared_network': True,     # Use shared actor-critic network
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }

    # Create agent
    agent = PPOAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        config=ppo_config
    )

    print(f"Training PPO on {ppo_config['device']}")
    print(f"State dimension: {env.observation_space.shape[0]}")
    print(f"Action dimension: {env.action_space.n}")
    print(f"Training for {num_episodes} episodes")
    print(f"Using {'shared' if ppo_config['use_shared_network'] else 'separate'} networks")

    # Training loop
    best_reward = float('-inf')
    episode_stats = []
    update_count = 0

    for episode in range(num_episodes):
        # Train for one episode
        episode_stat = agent.train_episode(env)
        episode_stats.append(episode_stat)

        # Update policy every few episodes (collect trajectories)
        if (episode + 1) % 20 == 0:  # Update every 20 episodes
            update_stats = agent.update()
            update_count += 1

            if update_stats:
                print(f"Update {update_count}: "
                      f"Policy Loss: {update_stats['policy_loss']:.4f}, "
                      f"Value Loss: {update_stats['value_loss']:.4f}, "
                      f"Entropy Loss: {update_stats['entropy_loss']:.4f}")

        # Print progress
        if (episode + 1) % 100 == 0:
            recent_rewards = [s['episode_reward'] for s in episode_stats[-100:]]
            recent_fidelities = [s['max_fidelity'] for s in episode_stats[-100:]]
            recent_lengths = [s['episode_length'] for s in episode_stats[-100:]]
            recent_entropies = [s['entropy'] for s in episode_stats[-100:]]
            recent_max_probs = [s['max_prob'] for s in episode_stats[-100:]]
            avg_reward = np.mean(recent_rewards)
            avg_fidelity = np.mean(recent_fidelities)
            avg_length = np.mean(recent_lengths)
            avg_entropy = np.mean(recent_entropies)
            avg_max_prob = np.mean(recent_max_probs)
            print(f"Episode {episode + 1}/{num_episodes}")
            print(f"  Recent avg reward: {avg_reward:.2f}")
            print(f"  Recent avg max fidelity: {avg_fidelity:.4f}")
            print(f"  Recent avg episode length: {avg_length:.1f}")
            print(f"  Recent avg entropy: {avg_entropy:.4f}")
            print(f"  Recent avg max prob: {avg_max_prob:.4f}")
            print(f"  Memory size: {len(agent.memory)}")
            
            # Save best model
            if avg_reward > best_reward:
                best_reward = avg_reward
                agent.save(os.path.join(save_dir, 'best_model.pth'))
                print(f"  New best model saved! Reward: {best_reward:.2f}")
        
        # Evaluate periodically
        if (episode + 1) % 500 == 0:
            eval_stats = agent.evaluate(env, num_episodes=10)
            print(f"  Evaluation at episode {episode + 1}:")
            print(f"    Mean reward: {eval_stats['mean_reward']:.2f} ± {eval_stats['std_reward']:.2f}")
            print(f"    Mean max fidelity: {eval_stats['mean_max_fidelity']:.4f} ± {eval_stats['std_max_fidelity']:.4f}")
            print(f"    Mean episode length: {eval_stats['mean_length']:.1f}")
    
    # Final evaluation
    print("\nFinal evaluation...")
    final_eval = agent.evaluate(env, num_episodes=100)
    print(f"Final mean reward: {final_eval['mean_reward']:.2f} ± {final_eval['std_reward']:.2f}")
    print(f"Final mean max fidelity: {final_eval['mean_max_fidelity']:.4f} ± {final_eval['std_max_fidelity']:.4f}")
    print(f"Final mean episode length: {final_eval['mean_length']:.1f}")
    
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
    Evaluate a trained PPO agent.
    
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
    agent = PPOAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n
    )
    
    # Load trained model
    agent.load(model_path)
    
    print(f"Evaluating trained PPO agent for {num_episodes} episodes...")
    
    # Evaluate
    eval_stats = agent.evaluate(env, num_episodes=num_episodes)
    
    print(f"Evaluation results:")
    print(f"  Mean reward: {eval_stats['mean_reward']:.2f} ± {eval_stats['std_reward']:.2f}")
    print(f"  Mean max fidelity: {eval_stats['mean_max_fidelity']:.4f} ± {eval_stats['std_max_fidelity']:.4f}")
    print(f"  Mean final fidelity: {eval_stats['mean_final_fidelity']:.4f} ± {eval_stats['std_final_fidelity']:.4f}")
    print(f"  Mean episode length: {eval_stats['mean_length']:.1f}")
    
    return eval_stats


def compare_episode_trajectories(agent, env, num_episodes=5):
    """
    Show detailed trajectories of trained agent.
    
    Args:
        agent: Trained PPO agent
        env: Environment
        num_episodes (int): Number of episodes to show
    """
    print(f"\nShowing {num_episodes} detailed episode trajectories:")
    
    for ep in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0
        step = 0
        fidelities = []
        
        print(f"\nEpisode {ep + 1}:")
        print(f"  Step {step}: Fidelity = {env._get_info()['state_fidelity']:.4f}")
        
        while True:
            action = agent.get_action(state)
            next_state, reward, done, truncated, info = env.step(action)
            
            total_reward += reward
            step += 1
            fidelity = info['state_fidelity']
            fidelities.append(fidelity)
            
            if step % 10 == 0 or done or truncated:
                print(f"  Step {step}: Action = {action}, Reward = {reward:.2f}, Fidelity = {fidelity:.4f}")
            
            state = next_state
            
            if done or truncated:
                break
        
        max_fidelity = max(fidelities)
        final_fidelity = fidelities[-1]
        
        print(f"  Total reward: {total_reward:.2f}")
        print(f"  Max fidelity: {max_fidelity:.4f}")
        print(f"  Final fidelity: {final_fidelity:.4f}")
        print(f"  Episode length: {step}")


if __name__ == "__main__":
    # Example usage
    print("Starting PPO training for quantum state control...")
    
    # Train the agent
    agent, stats = train_ppo(
        num_episodes=3000,
        save_dir='./ppo_quantum_results'
    )
    
    # Evaluate the trained agent
    final_eval = evaluate_trained_agent(
        './ppo_quantum_results/best_model.pth',
        num_episodes=100
    )
    
    # Show some detailed trajectories
    config = create_config()
    env = QuantumStateEnv(config)
    compare_episode_trajectories(agent, env, num_episodes=3)
