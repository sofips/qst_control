"""
Comprehensive evaluation script for trained PPO agents in quantum control.
This script provides detailed analysis of the learned control strategies.
"""

import sys
import os
import configparser
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

# Add the shared module to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'shared'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'drl'))

from quantum_environment import QuantumStateEnv
from ppo_agent import PPOAgent


def load_trained_agent(model_path, env):
    """Load a trained PPO agent from file."""
    agent = PPOAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n
    )
    agent.load(model_path)
    print(f"Loaded trained agent from: {model_path}")
    return agent


def basic_evaluation(agent, env, num_episodes=100, verbose=True, deterministic=False):
    """
    Basic evaluation of agent performance.
    
    Args:
        agent: Trained PPO agent
        env: Quantum environment
        num_episodes (int): Number of episodes to evaluate
        verbose (bool): Print detailed results
        deterministic (bool): Use deterministic policy (argmax) instead of sampling
        
    Returns:
        dict: Evaluation statistics
    """
    episode_rewards = []
    episode_lengths = []
    max_fidelities = []
    final_fidelities = []
    
    if verbose:
        print(f"\n{'='*50}")
        print(f"BASIC EVALUATION ({num_episodes} episodes)")
        print(f"{'='*50}")
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0
        episode_length = 0
        
        while True:
            action = agent.get_action(state, deterministic=deterministic)
            state, reward, done, truncated, info = env.step(action)
            total_reward += reward
            episode_length += 1
            
            if done or truncated:
                break
        
        episode_rewards.append(total_reward)
        episode_lengths.append(episode_length)
        max_fidelities.append(info.get('max_fidelity', 0))
        final_fidelities.append(info.get('state_fidelity', 0))
        
        if verbose and (episode + 1) % 20 == 0:
            print(f"Episode {episode + 1:3d}: Reward={total_reward:6.2f}, "
                  f"Max Fidelity={info.get('max_fidelity', 0):.4f}, "
                  f"Final Fidelity={info.get('state_fidelity', 0):.4f}")
    
    stats = {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'min_reward': np.min(episode_rewards),
        'max_reward': np.max(episode_rewards),
        'mean_length': np.mean(episode_lengths),
        'std_length': np.std(episode_lengths),
        'mean_max_fidelity': np.mean(max_fidelities),
        'std_max_fidelity': np.std(max_fidelities),
        'min_max_fidelity': np.min(max_fidelities),
        'max_max_fidelity': np.max(max_fidelities),
        'mean_final_fidelity': np.mean(final_fidelities),
        'std_final_fidelity': np.std(final_fidelities),
        'episodes': episode_rewards,
        'max_fidelities': max_fidelities,
        'final_fidelities': final_fidelities,
        'lengths': episode_lengths
    }
    
    if verbose:
        print(f"\nResults Summary:")
        print(f"  Mean Reward: {stats['mean_reward']:.2f} ± {stats['std_reward']:.2f}")
        print(f"  Reward Range: [{stats['min_reward']:.2f}, {stats['max_reward']:.2f}]")
        print(f"  Mean Max Fidelity: {stats['mean_max_fidelity']:.4f} ± {stats['std_max_fidelity']:.4f}")
        print(f"  Max Fidelity Range: [{stats['min_max_fidelity']:.4f}, {stats['max_max_fidelity']:.4f}]")
        print(f"  Mean Final Fidelity: {stats['mean_final_fidelity']:.4f} ± {stats['std_final_fidelity']:.4f}")
        print(f"  Mean Episode Length: {stats['mean_length']:.1f} ± {stats['std_length']:.1f}")
    
    return stats


def detailed_trajectory_analysis(agent, env, num_episodes=5):
    """
    Detailed analysis of individual trajectories.
    
    Args:
        agent: Trained PPO agent
        env: Quantum environment
        num_episodes (int): Number of trajectories to analyze
        
    Returns:
        list: Detailed trajectory data
    """
    print(f"\n{'='*50}")
    print(f"DETAILED TRAJECTORY ANALYSIS")
    print(f"{'='*50}")
    
    trajectories = []
    
    for episode in range(num_episodes):
        print(f"\n--- Episode {episode + 1} ---")
        
        state, info = env.reset()
        trajectory = {
            'states': [state.copy()],
            'actions': [],
            'rewards': [],
            'fidelities': [info['state_fidelity']],
            'mean_sites': [info.get('mean_site', 0)],
            'step_info': []
        }
        
        total_reward = 0
        step = 0
        
        print(f"Step {step:2d}: Fidelity={info['state_fidelity']:.4f}, "
              f"Mean Site={info.get('mean_site', 0):.2f}")
        
        while True:
            # Get action probabilities for analysis
            action, log_prob, value = agent.get_action_and_value(state)
            
            # Take step
            next_state, reward, done, truncated, info = env.step(action)
            
            # Store trajectory data
            trajectory['actions'].append(action)
            trajectory['rewards'].append(reward)
            trajectory['states'].append(next_state.copy())
            trajectory['fidelities'].append(info['state_fidelity'])
            trajectory['mean_sites'].append(info.get('mean_site', 0))
            trajectory['step_info'].append({
                'log_prob': log_prob,
                'value': value,
                'reward': reward
            })
            
            total_reward += reward
            step += 1
            
            # Print every few steps or important changes
            if step % 10 == 0 or done or truncated or abs(reward) > 50:
                print(f"Step {step:2d}: Action={action:2d}, Reward={reward:6.2f}, "
                      f"Fidelity={info['state_fidelity']:.4f}, "
                      f"Value={value:.2f}")
            
            state = next_state
            
            if done or truncated:
                break
        
        trajectory['total_reward'] = total_reward
        trajectory['episode_length'] = step
        trajectory['max_fidelity'] = max(trajectory['fidelities'])
        trajectory['final_fidelity'] = trajectory['fidelities'][-1]
        
        print(f"Episode Summary:")
        print(f"  Total Reward: {total_reward:.2f}")
        print(f"  Max Fidelity: {trajectory['max_fidelity']:.4f}")
        print(f"  Final Fidelity: {trajectory['final_fidelity']:.4f}")
        print(f"  Episode Length: {step}")
        
        trajectories.append(trajectory)
    
    return trajectories


def action_frequency_analysis(agent, env, num_episodes=50):
    """
    Analyze the frequency and patterns of actions taken by the agent.
    
    Args:
        agent: Trained PPO agent
        env: Quantum environment
        num_episodes (int): Number of episodes to analyze
        
    Returns:
        dict: Action analysis results
    """
    print(f"\n{'='*50}")
    print(f"ACTION FREQUENCY ANALYSIS")
    print(f"{'='*50}")
    
    action_counts = defaultdict(int)
    action_sequences = []
    action_transitions = defaultdict(lambda: defaultdict(int))
    step_action_patterns = defaultdict(list)
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_actions = []
        step = 0
        
        while True:
            action = agent.get_action(state)
            episode_actions.append(action)
            action_counts[action] += 1
            step_action_patterns[step].append(action)
            
            # Track action transitions
            if len(episode_actions) > 1:
                prev_action = episode_actions[-2]
                action_transitions[prev_action][action] += 1
            
            state, reward, done, truncated, info = env.step(action)
            step += 1
            
            if done or truncated:
                break
        
        action_sequences.append(episode_actions)
    
    # Calculate statistics
    total_actions = sum(action_counts.values())
    action_probs = {action: count/total_actions for action, count in action_counts.items()}
    
    print(f"Action Frequency Distribution:")
    for action in sorted(action_counts.keys()):
        count = action_counts[action]
        prob = action_probs[action]
        print(f"  Action {action:2d}: {count:4d} times ({prob:.3f})")
    
    # Find most common action sequences
    sequence_counts = defaultdict(int)
    for seq in action_sequences:
        if len(seq) >= 3:
            for i in range(len(seq) - 2):
                triple = tuple(seq[i:i+3])
                sequence_counts[triple] += 1
    
    print(f"\nMost Common Action Sequences (3-step):")
    top_sequences = sorted(sequence_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    for seq, count in top_sequences:
        print(f"  {seq}: {count} times")
    
    # Action consistency across time steps
    print(f"\nAction Consistency Across Episodes:")
    for step in sorted(step_action_patterns.keys())[:10]:  # First 10 steps
        actions_at_step = step_action_patterns[step]
        if len(actions_at_step) > 0:
            most_common = max(set(actions_at_step), key=actions_at_step.count)
            consistency = actions_at_step.count(most_common) / len(actions_at_step)
            print(f"  Step {step:2d}: Action {most_common:2d} ({consistency:.2%} consistency)")
    
    return {
        'action_counts': dict(action_counts),
        'action_probs': action_probs,
        'action_sequences': action_sequences,
        'action_transitions': dict(action_transitions),
        'step_patterns': dict(step_action_patterns),
        'top_sequences': top_sequences
    }


def fidelity_evolution_analysis(agent, env, num_episodes=10):
    """
    Analyze how fidelity evolves during episodes.
    
    Args:
        agent: Trained PPO agent
        env: Quantum environment
        num_episodes (int): Number of episodes to analyze
        
    Returns:
        dict: Fidelity evolution data
    """
    print(f"\n{'='*50}")
    print(f"FIDELITY EVOLUTION ANALYSIS")
    print(f"{'='*50}")
    
    all_fidelity_curves = []
    all_reward_curves = []
    
    for episode in range(num_episodes):
        state, info = env.reset()
        fidelity_curve = [info['state_fidelity']]
        reward_curve = []
        step = 0
        
        while True:
            action = agent.get_action(state)
            state, reward, done, truncated, info = env.step(action)
            
            fidelity_curve.append(info['state_fidelity'])
            reward_curve.append(reward)
            step += 1
            
            if done or truncated:
                break
        
        all_fidelity_curves.append(fidelity_curve)
        all_reward_curves.append(reward_curve)
        
        max_fidelity = max(fidelity_curve)
        final_fidelity = fidelity_curve[-1]
        max_step = np.argmax(fidelity_curve)
        
        print(f"Episode {episode + 1}: Max fidelity {max_fidelity:.4f} at step {max_step}, "
              f"Final fidelity {final_fidelity:.4f}")
    
    # Calculate average fidelity evolution
    max_length = max(len(curve) for curve in all_fidelity_curves)
    fidelity_matrix = np.full((num_episodes, max_length), np.nan)
    
    for i, curve in enumerate(all_fidelity_curves):
        fidelity_matrix[i, :len(curve)] = curve
    
    mean_fidelity_curve = np.nanmean(fidelity_matrix, axis=0)
    std_fidelity_curve = np.nanstd(fidelity_matrix, axis=0)
    
    print(f"\nAverage Fidelity Evolution:")
    for step in range(0, len(mean_fidelity_curve), 10):
        if not np.isnan(mean_fidelity_curve[step]):
            print(f"  Step {step:2d}: {mean_fidelity_curve[step]:.4f} ± {std_fidelity_curve[step]:.4f}")
    
    return {
        'fidelity_curves': all_fidelity_curves,
        'reward_curves': all_reward_curves,
        'mean_fidelity_curve': mean_fidelity_curve,
        'std_fidelity_curve': std_fidelity_curve,
        'fidelity_matrix': fidelity_matrix
    }


def compare_with_random_baseline(agent, env, num_episodes=50):
    """
    Compare trained agent performance with random baseline.
    
    Args:
        agent: Trained PPO agent
        env: Quantum environment
        num_episodes (int): Number of episodes for comparison
        
    Returns:
        dict: Comparison results
    """
    print(f"\n{'='*50}")
    print(f"COMPARISON WITH RANDOM BASELINE")
    print(f"{'='*50}")
    
    # Evaluate trained agent
    print("Evaluating trained PPO agent...")
    ppo_stats = basic_evaluation(agent, env, num_episodes, verbose=False)
    
    # Evaluate random policy
    print("Evaluating random policy...")
    random_rewards = []
    random_max_fidelities = []
    random_final_fidelities = []
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0
        
        while True:
            action = env.action_space.sample()  # Random action
            state, reward, done, truncated, info = env.step(action)
            total_reward += reward
            
            if done or truncated:
                break
        
        random_rewards.append(total_reward)
        random_max_fidelities.append(info.get('max_fidelity', 0))
        random_final_fidelities.append(info.get('state_fidelity', 0))
    
    random_stats = {
        'mean_reward': np.mean(random_rewards),
        'std_reward': np.std(random_rewards),
        'mean_max_fidelity': np.mean(random_max_fidelities),
        'std_max_fidelity': np.std(random_max_fidelities),
        'mean_final_fidelity': np.mean(random_final_fidelities),
        'std_final_fidelity': np.std(random_final_fidelities),
    }
    
    # Calculate improvement
    reward_improvement = ppo_stats['mean_reward'] / random_stats['mean_reward']
    fidelity_improvement = ppo_stats['mean_max_fidelity'] / random_stats['mean_max_fidelity']
    
    print(f"\nComparison Results:")
    print(f"{'Metric':<20} {'PPO Agent':<15} {'Random Policy':<15} {'Improvement':<12}")
    print(f"{'-'*65}")
    print(f"{'Mean Reward':<20} {ppo_stats['mean_reward']:<15.2f} {random_stats['mean_reward']:<15.2f} {reward_improvement:<12.2f}x")
    print(f"{'Mean Max Fidelity':<20} {ppo_stats['mean_max_fidelity']:<15.4f} {random_stats['mean_max_fidelity']:<15.4f} {fidelity_improvement:<12.2f}x")
    print(f"{'Mean Final Fidelity':<20} {ppo_stats['mean_final_fidelity']:<15.4f} {random_stats['mean_final_fidelity']:<15.4f} {ppo_stats['mean_final_fidelity']/random_stats['mean_final_fidelity']:<12.2f}x")
    
    return {
        'ppo_stats': ppo_stats,
        'random_stats': random_stats,
        'reward_improvement': reward_improvement,
        'fidelity_improvement': fidelity_improvement
    }


def plot_evaluation_results(basic_stats, fidelity_data, action_data, save_dir='./evaluation_plots'):
    """
    Create comprehensive plots of evaluation results.
    
    Args:
        basic_stats: Results from basic_evaluation
        fidelity_data: Results from fidelity_evolution_analysis
        action_data: Results from action_frequency_analysis
        save_dir: Directory to save plots
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create a large figure with subplots
    fig = plt.figure(figsize=(20, 15))
    
    # 1. Reward Distribution
    ax1 = plt.subplot(3, 4, 1)
    plt.hist(basic_stats['episodes'], bins=20, alpha=0.7, edgecolor='black')
    plt.axvline(basic_stats['mean_reward'], color='red', linestyle='--', 
                label=f'Mean: {basic_stats["mean_reward"]:.2f}')
    plt.xlabel('Episode Reward')
    plt.ylabel('Frequency')
    plt.title('Reward Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. Max Fidelity Distribution
    ax2 = plt.subplot(3, 4, 2)
    plt.hist(basic_stats['max_fidelities'], bins=20, alpha=0.7, edgecolor='black')
    plt.axvline(basic_stats['mean_max_fidelity'], color='red', linestyle='--',
                label=f'Mean: {basic_stats["mean_max_fidelity"]:.4f}')
    plt.xlabel('Max Fidelity')
    plt.ylabel('Frequency')
    plt.title('Max Fidelity Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. Action Frequency
    ax3 = plt.subplot(3, 4, 3)
    actions = list(action_data['action_counts'].keys())
    counts = list(action_data['action_counts'].values())
    plt.bar(actions, counts, alpha=0.7, edgecolor='black')
    plt.xlabel('Action')
    plt.ylabel('Frequency')
    plt.title('Action Frequency Distribution')
    plt.grid(True, alpha=0.3)
    
    # 4. Fidelity Evolution (mean with std)
    ax4 = plt.subplot(3, 4, 4)
    mean_curve = fidelity_data['mean_fidelity_curve']
    std_curve = fidelity_data['std_fidelity_curve']
    steps = np.arange(len(mean_curve))
    
    plt.plot(steps, mean_curve, 'b-', linewidth=2, label='Mean Fidelity')
    plt.fill_between(steps, mean_curve - std_curve, mean_curve + std_curve, 
                     alpha=0.3, label='±1 std')
    plt.xlabel('Time Step')
    plt.ylabel('Fidelity')
    plt.title('Average Fidelity Evolution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 5. Individual Fidelity Curves
    ax5 = plt.subplot(3, 4, 5)
    for i, curve in enumerate(fidelity_data['fidelity_curves'][:10]):  # Show first 10
        plt.plot(curve, alpha=0.6, linewidth=1)
    plt.xlabel('Time Step')
    plt.ylabel('Fidelity')
    plt.title('Individual Fidelity Trajectories')
    plt.grid(True, alpha=0.3)
    
    # 6. Episode Length Distribution
    ax6 = plt.subplot(3, 4, 6)
    plt.hist(basic_stats['lengths'], bins=10, alpha=0.7, edgecolor='black')
    plt.axvline(basic_stats['mean_length'], color='red', linestyle='--',
                label=f'Mean: {basic_stats["mean_length"]:.1f}')
    plt.xlabel('Episode Length')
    plt.ylabel('Frequency')
    plt.title('Episode Length Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 7. Final vs Max Fidelity Scatter
    ax7 = plt.subplot(3, 4, 7)
    plt.scatter(basic_stats['max_fidelities'], basic_stats['final_fidelities'], 
                alpha=0.6, s=30)
    plt.plot([0, 1], [0, 1], 'r--', alpha=0.5, label='y=x')
    plt.xlabel('Max Fidelity')
    plt.ylabel('Final Fidelity')
    plt.title('Final vs Max Fidelity')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 8. Action Transitions Heatmap (if available)
    ax8 = plt.subplot(3, 4, 8)
    if action_data['action_transitions']:
        n_actions = max(max(action_data['action_transitions'].keys(), default=0),
                       max([max(v.keys(), default=0) for v in action_data['action_transitions'].values()], default=0)) + 1
        transition_matrix = np.zeros((n_actions, n_actions))
        
        for from_action, transitions in action_data['action_transitions'].items():
            for to_action, count in transitions.items():
                transition_matrix[from_action, to_action] = count
        
        # Normalize rows
        row_sums = transition_matrix.sum(axis=1, keepdims=True)
        transition_matrix = np.divide(transition_matrix, row_sums, 
                                    out=np.zeros_like(transition_matrix), 
                                    where=row_sums!=0)
        
        sns.heatmap(transition_matrix, annot=True, fmt='.2f', cmap='Blues',
                   xticklabels=range(n_actions), yticklabels=range(n_actions))
        plt.xlabel('To Action')
        plt.ylabel('From Action')
        plt.title('Action Transition Probabilities')
    
    # 9-12. Performance metrics summary
    ax9 = plt.subplot(3, 4, (9, 12))
    ax9.axis('off')
    
    # Create summary text
    summary_text = f"""
PERFORMANCE SUMMARY

Basic Statistics:
• Mean Reward: {basic_stats['mean_reward']:.2f} ± {basic_stats['std_reward']:.2f}
• Reward Range: [{basic_stats['min_reward']:.2f}, {basic_stats['max_reward']:.2f}]
• Mean Max Fidelity: {basic_stats['mean_max_fidelity']:.4f} ± {basic_stats['std_max_fidelity']:.4f}
• Max Fidelity Range: [{basic_stats['min_max_fidelity']:.4f}, {basic_stats['max_max_fidelity']:.4f}]
• Mean Final Fidelity: {basic_stats['mean_final_fidelity']:.4f} ± {basic_stats['std_final_fidelity']:.4f}
• Mean Episode Length: {basic_stats['mean_length']:.1f} ± {basic_stats['std_length']:.1f}

Action Analysis:
• Total Unique Actions Used: {len(action_data['action_counts'])}
• Most Frequent Action: {max(action_data['action_counts'], key=action_data['action_counts'].get)}
• Action Entropy: {-sum(p * np.log2(p) for p in action_data['action_probs'].values() if p > 0):.2f} bits

Quantum Control Quality:
• Fidelity Consistency: {1 - basic_stats['std_max_fidelity']:.4f} (higher is better)
• Success Rate (>0.9 fidelity): {sum(1 for f in basic_stats['max_fidelities'] if f > 0.9) / len(basic_stats['max_fidelities']):.1%}
• High Performance (>0.95 fidelity): {sum(1 for f in basic_stats['max_fidelities'] if f > 0.95) / len(basic_stats['max_fidelities']):.1%}
"""
    
    ax9.text(0.05, 0.95, summary_text, transform=ax9.transAxes, fontsize=12,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'comprehensive_evaluation.png'), 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nPlots saved to: {save_dir}")


def main_evaluation(model_path, config_path=None):
    """
    Run comprehensive evaluation of a trained PPO agent.
    
    Args:
        model_path (str): Path to trained model
        config_path (str): Path to environment config (optional)
    """
    print("="*60)
    print("COMPREHENSIVE PPO AGENT EVALUATION")
    print("="*60)
    
    # Load configuration
    if config_path and os.path.exists(config_path):
        config = configparser.ConfigParser()
        config.read(config_path)
        print(f"Loaded config from: {config_path}")
    else:
        # Create default config (modify as needed)
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
        print("Using default configuration")
    
    # Create environment and load agent
    env = QuantumStateEnv(config)
    agent = load_trained_agent(model_path, env)
    
    print(f"Environment: {env.chain_length} qubits, {env.action_space.n} actions")
    print(f"Max episode length: {env.max_time_steps}")
    
    # Run all evaluations
    basic_stats = basic_evaluation(agent, env, num_episodes=100)
    trajectories = detailed_trajectory_analysis(agent, env, num_episodes=5)
    action_analysis = action_frequency_analysis(agent, env, num_episodes=100)
    fidelity_analysis = fidelity_evolution_analysis(agent, env, num_episodes=20)
    comparison = compare_with_random_baseline(agent, env, num_episodes=50)
    
    # Create plots
    plot_evaluation_results(basic_stats, fidelity_analysis, action_analysis)
    
    print(f"\n{'='*60}")
    print("EVALUATION COMPLETE")
    print(f"{'='*60}")
    
    return {
        'basic_stats': basic_stats,
        'trajectories': trajectories,
        'action_analysis': action_analysis,
        'fidelity_analysis': fidelity_analysis,
        'comparison': comparison
    }


if __name__ == "__main__":
    # Example usage
    model_path = "./ppo_quantum_results/best_model.pth"
    config_path = "./ppo_quantum_results/config.ini"
    
    if os.path.exists(model_path):
        results = main_evaluation(model_path, config_path)
    else:
        print(f"Model not found at: {model_path}")
        print("Please train a model first using train_ppo.py")
