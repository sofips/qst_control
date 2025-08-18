"""
Simple evaluation script for quick analysis of trained PPO agents.
Run this script to get essential performance metrics and visualizations.
"""

import sys
import os
import configparser
import numpy as np
import matplotlib.pyplot as plt

# Add the shared module to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'shared'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'drl'))

from quantum_environment import QuantumStateEnv
from ppo_agent import PPOAgent


def quick_evaluate(model_path, num_episodes=50):
    """Quick evaluation of a trained PPO agent."""
    
    print("="*50)
    print("QUICK PPO EVALUATION")
    print("="*50)
    
    # Load default config
    config = configparser.ConfigParser()
    config['system_parameters'] = {
        'chain_length': '8',
        'action_set': 'zhang',
        'n_actions': '16',
        'field_strength': '100',
        'coupling': '1.0',
        'tstep_length': '0.15',
        'max_t_steps': '80',
        'tolerance': '0.05'
    }
    config['learning_parameters'] = {
        'reward_function': 'original',
        'gamma': '0.95'
    }
    
    # Create environment and agent
    env = QuantumStateEnv(config)
    agent = PPOAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n
    )
    agent.load(model_path)
    
    print(f"Loaded model from: {model_path}")
    print(f"Environment: {env.chain_length} qubits, {env.action_space.n} actions")
    
    # Evaluate
    rewards = []
    max_fidelities = []
    final_fidelities = []
    episode_lengths = []
    action_counts = {}
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0
        length = 0
        
        while True:
            action = agent.get_action(state)
            action_counts[action] = action_counts.get(action, 0) + 1
            
            state, reward, done, truncated, info = env.step(action)
            total_reward += reward
            length += 1
            
            if done or truncated:
                break
        
        rewards.append(total_reward)
        max_fidelities.append(info.get('max_fidelity', 0))
        final_fidelities.append(info.get('state_fidelity', 0))
        episode_lengths.append(length)
        
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1:2d}: Reward={total_reward:6.2f}, "
                  f"Max Fidelity={info.get('max_fidelity', 0):.4f}")
    
    # Calculate statistics
    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    mean_max_fidelity = np.mean(max_fidelities)
    std_max_fidelity = np.std(max_fidelities)
    mean_final_fidelity = np.mean(final_fidelities)
    std_final_fidelity = np.std(final_fidelities)
    mean_length = np.mean(episode_lengths)
    
    # Print results
    print("\nRESULTS:")
    print("-" * 40)
    print(f"Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}")
    print(f"Mean Max Fidelity: {mean_max_fidelity:.4f} ± {std_max_fidelity:.4f}")
    print(f"Mean Final Fidelity: {mean_final_fidelity:.4f} ± {std_final_fidelity:.4f}")
    print(f"Mean Episode Length: {mean_length:.1f}")
    print(f"Fidelity > 0.90: {sum(1 for f in max_fidelities if f > 0.9)/len(max_fidelities):.1%}")
    print(f"Fidelity > 0.95: {sum(1 for f in max_fidelities if f > 0.95)/len(max_fidelities):.1%}")
    
    # Action analysis
    print(f"\nTop 5 Most Used Actions:")
    sorted_actions = sorted(action_counts.items(), key=lambda x: x[1], reverse=True)
    for action, count in sorted_actions[:5]:
        percentage = count / sum(action_counts.values()) * 100
        print(f"  Action {action:2d}: {count:3d} times ({percentage:4.1f}%)")
    
    # Quick plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
    
    # Reward distribution
    ax1.hist(rewards, bins=15, alpha=0.7, edgecolor='black')
    ax1.axvline(mean_reward, color='red', linestyle='--', label=f'Mean: {mean_reward:.2f}')
    ax1.set_xlabel('Episode Reward')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Reward Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Max fidelity distribution
    ax2.hist(max_fidelities, bins=15, alpha=0.7, edgecolor='black')
    ax2.axvline(mean_max_fidelity, color='red', linestyle='--', 
                label=f'Mean: {mean_max_fidelity:.4f}')
    ax2.set_xlabel('Max Fidelity')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Max Fidelity Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Action frequency
    actions = list(action_counts.keys())
    counts = list(action_counts.values())
    ax3.bar(actions, counts, alpha=0.7, edgecolor='black')
    ax3.set_xlabel('Action')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Action Usage')
    ax3.grid(True, alpha=0.3)
    
    # Performance over episodes
    ax4.plot(rewards, 'b-', alpha=0.7, label='Episode Rewards')
    ax4.axhline(mean_reward, color='red', linestyle='--', label=f'Mean: {mean_reward:.2f}')
    ax4.set_xlabel('Episode')
    ax4.set_ylabel('Reward')
    ax4.set_title('Performance Over Episodes')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('quick_evaluation.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nPlot saved as: quick_evaluation.png")
    
    return {
        'mean_reward': mean_reward,
        'mean_max_fidelity': mean_max_fidelity,
        'mean_final_fidelity': mean_final_fidelity,
        'rewards': rewards,
        'max_fidelities': max_fidelities,
        'action_counts': action_counts
    }


def show_single_trajectory(model_path):
    """Show a detailed single episode trajectory."""
    
    print("\n" + "="*50)
    print("SINGLE TRAJECTORY ANALYSIS")
    print("="*50)
    
    # Load config and environment
    config = configparser.ConfigParser()
    config['system_parameters'] = {
        'chain_length': '8',
        'action_set': 'zhang',
        'n_actions': '16',
        'field_strength': '100',
        'coupling': '1.0',
        'tstep_length': '0.15',
        'max_t_steps': '80',
        'tolerance': '0.05'
    }
    config['learning_parameters'] = {
        'reward_function': 'original',
        'gamma': '0.95'
    }
    
    env = QuantumStateEnv(config)
    agent = PPOAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n
    )
    agent.load(model_path)
    
    # Run one episode with detailed logging
    state, info = env.reset()
    print(f"Initial state fidelity: {info['state_fidelity']:.4f}")
    print(f"Target: Maximize fidelity of quantum state transfer\n")
    
    total_reward = 0
    step = 0
    fidelities = [info['state_fidelity']]
    actions = []
    rewards = []
    
    print(f"{'Step':<4} {'Action':<6} {'Reward':<8} {'Fidelity':<8} {'Cumulative':<10}")
    print("-" * 45)
    
    while True:
        action = agent.get_action(state)
        state, reward, done, truncated, info = env.step(action)
        
        total_reward += reward
        step += 1
        fidelities.append(info['state_fidelity'])
        actions.append(action)
        rewards.append(reward)
        
        # Print every few steps or important events
        if step <= 10 or step % 10 == 0 or done or truncated:
            print(f"{step:<4} {action:<6} {reward:<8.2f} {info['state_fidelity']:<8.4f} {total_reward:<10.2f}")
        
        if done or truncated:
            break
    
    max_fidelity = max(fidelities)
    max_step = fidelities.index(max_fidelity)
    final_fidelity = fidelities[-1]
    
    print(f"\nTrajectory Summary:")
    print(f"  Total Steps: {step}")
    print(f"  Total Reward: {total_reward:.2f}")
    print(f"  Max Fidelity: {max_fidelity:.4f} (reached at step {max_step})")
    print(f"  Final Fidelity: {final_fidelity:.4f}")
    print(f"  Actions Used: {len(set(actions))} unique actions")
    
    # Plot trajectory
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Fidelity evolution
    ax1.plot(fidelities, 'b-', linewidth=2, marker='o', markersize=3)
    ax1.axhline(max_fidelity, color='red', linestyle='--', alpha=0.7,
                label=f'Max: {max_fidelity:.4f}')
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Fidelity')
    ax1.set_title('Quantum State Fidelity Evolution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Actions taken
    ax2.step(range(len(actions)), actions, 'g-', linewidth=2, where='mid')
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Action')
    ax2.set_title('Control Actions Sequence')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('trajectory_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Trajectory plot saved as: trajectory_analysis.png")


def compare_performance(model_path, num_episodes=20):
    """Compare trained agent with random policy."""
    
    print("\n" + "="*50)
    print("PERFORMANCE COMPARISON")
    print("="*50)
    
    # Setup
    config = configparser.ConfigParser()
    config['system_parameters'] = {
        'chain_length': '8',
        'action_set': 'zhang',
        'n_actions': '16',
        'field_strength': '100',
        'coupling': '1.0',
        'tstep_length': '0.15',
        'max_t_steps': '80',
        'tolerance': '0.05'
    }
    config['learning_parameters'] = {
        'reward_function': 'original',
        'gamma': '0.95'
    }
    
    env = QuantumStateEnv(config)
    agent = PPOAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n
    )
    agent.load(model_path)
    
    # Test PPO agent
    print("Testing PPO agent...")
    ppo_rewards = []
    ppo_fidelities = []
    
    for _ in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0
        
        while True:
            action = agent.get_action(state)
            state, reward, done, truncated, info = env.step(action)
            total_reward += reward
            if done or truncated:
                break
        
        ppo_rewards.append(total_reward)
        ppo_fidelities.append(info.get('max_fidelity', 0))
    
    # Test random policy
    print("Testing random policy...")
    random_rewards = []
    random_fidelities = []
    
    for _ in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0
        
        while True:
            action = env.action_space.sample()
            state, reward, done, truncated, info = env.step(action)
            total_reward += reward
            if done or truncated:
                break
        
        random_rewards.append(total_reward)
        random_fidelities.append(info.get('max_fidelity', 0))
    
    # Compare results
    print(f"\nComparison Results ({num_episodes} episodes each):")
    print("-" * 50)
    print(f"{'Metric':<20} {'PPO Agent':<12} {'Random':<12} {'Improvement'}")
    print("-" * 50)
    
    ppo_mean_reward = np.mean(ppo_rewards)
    random_mean_reward = np.mean(random_rewards)
    reward_improvement = ppo_mean_reward / random_mean_reward if random_mean_reward > 0 else float('inf')
    
    ppo_mean_fidelity = np.mean(ppo_fidelities)
    random_mean_fidelity = np.mean(random_fidelities)
    fidelity_improvement = ppo_mean_fidelity / random_mean_fidelity if random_mean_fidelity > 0 else float('inf')
    
    print(f"{'Mean Reward':<20} {ppo_mean_reward:<12.2f} {random_mean_reward:<12.2f} {reward_improvement:.2f}x")
    print(f"{'Mean Max Fidelity':<20} {ppo_mean_fidelity:<12.4f} {random_mean_fidelity:<12.4f} {fidelity_improvement:.2f}x")
    print(f"{'Std Reward':<20} {np.std(ppo_rewards):<12.2f} {np.std(random_rewards):<12.2f}")
    print(f"{'Std Max Fidelity':<20} {np.std(ppo_fidelities):<12.4f} {np.std(random_fidelities):<12.4f}")


if __name__ == "__main__":
    model_path = "./ppo_quantum_results/best_model.pth"
    
    if os.path.exists(model_path):
        print("Running comprehensive evaluation...")
        
        # Quick evaluation
        results = quick_evaluate(model_path, num_episodes=50)
        
        # Single trajectory
        show_single_trajectory(model_path)
        
        # Comparison
        compare_performance(model_path, num_episodes=30)
        
        print("\nEvaluation complete!")
        
    else:
        print(f"Model not found at: {model_path}")
        print("Please train a model first using: python train_ppo.py")
