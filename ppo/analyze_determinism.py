"""
Diagnostic script to analyze policy determinism and sources of variability.
"""

import sys
import os
import configparser
import numpy as np
import torch

# Add the shared module to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'shared'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'drl'))

from quantum_environment import QuantumStateEnv
from ppo_agent import PPOAgent


def analyze_policy_determinism(model_path, num_episodes=10):
    """Analyze how deterministic the trained policy is."""
    
    print("="*60)
    print("POLICY DETERMINISM ANALYSIS")
    print("="*60)
    
    # Load environment and agent
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
    
    print(f"Analyzing policy for {num_episodes} episodes...")
    
    # Track action sequences and probabilities
    action_sequences = []
    action_probabilities = []
    episode_rewards = []
    episode_fidelities = []
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_actions = []
        episode_probs = []
        total_reward = 0
        
        while True:
            # Get full action distribution
            if isinstance(state, np.ndarray):
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
            
            with torch.no_grad():
                if agent.network:
                    action_logits, _ = agent.network(state_tensor)
                else:
                    action_logits = agent.actor(state_tensor)
                
                # Get probabilities
                action_probs = torch.softmax(action_logits, dim=-1)
                probs_numpy = action_probs.cpu().numpy().flatten()
                
                # Sample action
                action = torch.multinomial(action_probs, 1).item()
            
            episode_actions.append(action)
            episode_probs.append(probs_numpy.copy())
            
            # Take step
            state, reward, done, truncated, info = env.step(action)
            total_reward += reward
            
            if done or truncated:
                break
        
        action_sequences.append(episode_actions)
        action_probabilities.append(episode_probs)
        episode_rewards.append(total_reward)
        episode_fidelities.append(info.get('max_fidelity', 0))
        
        print(f"Episode {episode + 1}: Reward={total_reward:.2f}, "
              f"Fidelity={info.get('max_fidelity', 0):.4f}, "
              f"Actions={len(episode_actions)}")
    
    # Analysis
    print(f"\nDETERMINISM ANALYSIS:")
    print("-" * 40)
    
    # Check if all sequences are identical
    first_sequence = action_sequences[0]
    all_identical = all(seq == first_sequence for seq in action_sequences)
    print(f"All action sequences identical: {all_identical}")
    
    if all_identical:
        print(f"Repeated action sequence: {first_sequence[:10]}..." if len(first_sequence) > 10 else f"Action sequence: {first_sequence}")
    else:
        print("Action sequences vary between episodes")
        # Show differences
        for i, seq in enumerate(action_sequences):
            if seq != first_sequence:
                diff_positions = [j for j, (a, b) in enumerate(zip(seq, first_sequence)) if a != b]
                print(f"  Episode {i+1} differs at positions: {diff_positions[:5]}...")
    
    # Check probability concentration
    print(f"\nPROBABILITY ANALYSIS:")
    print("-" * 40)
    
    # Average entropy across all steps
    entropies = []
    max_probs = []
    
    for episode_probs in action_probabilities:
        for step_probs in episode_probs:
            # Calculate entropy
            entropy = -np.sum(step_probs * np.log(step_probs + 1e-8))
            entropies.append(entropy)
            max_probs.append(np.max(step_probs))
    
    mean_entropy = np.mean(entropies)
    mean_max_prob = np.mean(max_probs)
    
    print(f"Mean policy entropy: {mean_entropy:.4f} (lower = more deterministic)")
    print(f"Mean max action probability: {mean_max_prob:.4f} (higher = more deterministic)")
    print(f"Entropy range: [{np.min(entropies):.4f}, {np.max(entropies):.4f}]")
    
    # Check performance consistency
    print(f"\nPERFORMANCE CONSISTENCY:")
    print("-" * 40)
    reward_std = np.std(episode_rewards)
    fidelity_std = np.std(episode_fidelities)
    
    print(f"Reward std: {reward_std:.4f} (lower = more consistent)")
    print(f"Fidelity std: {fidelity_std:.6f} (lower = more consistent)")
    print(f"Reward range: [{np.min(episode_rewards):.2f}, {np.max(episode_rewards):.2f}]")
    print(f"Fidelity range: [{np.min(episode_fidelities):.6f}, {np.max(episode_fidelities):.6f}]")
    
    # Determinism score
    determinism_score = (1 - mean_entropy / np.log(env.action_space.n)) * 100
    print(f"\nDeterminism Score: {determinism_score:.1f}% (100% = fully deterministic)")
    
    return {
        'all_identical': all_identical,
        'action_sequences': action_sequences,
        'mean_entropy': mean_entropy,
        'mean_max_prob': mean_max_prob,
        'reward_std': reward_std,
        'fidelity_std': fidelity_std,
        'determinism_score': determinism_score
    }


def test_environment_determinism():
    """Test if the environment itself is deterministic."""
    
    print("\n" + "="*60)
    print("ENVIRONMENT DETERMINISM TEST")
    print("="*60)
    
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
    
    # Test same action sequence multiple times
    test_actions = [0, 3, 7, 1, 5, 2, 8, 4]  # Arbitrary sequence
    results = []
    
    print(f"Testing environment with fixed action sequence: {test_actions}")
    
    for run in range(5):
        state, _ = env.reset()
        trajectory = [state.copy()]
        fidelities = []
        rewards = []
        
        for action in test_actions:
            state, reward, done, truncated, info = env.step(action)
            trajectory.append(state.copy())
            fidelities.append(info['state_fidelity'])
            rewards.append(reward)
            
            if done or truncated:
                break
        
        results.append({
            'trajectory': trajectory,
            'fidelities': fidelities,
            'rewards': rewards,
            'final_fidelity': fidelities[-1] if fidelities else 0
        })
        
        print(f"Run {run + 1}: Final fidelity = {fidelities[-1] if fidelities else 0:.6f}")
    
    # Check if results are identical
    first_result = results[0]
    all_env_identical = True
    
    for i, result in enumerate(results[1:], 1):
        if not np.allclose(result['final_fidelity'], first_result['final_fidelity'], atol=1e-10):
            all_env_identical = False
            print(f"Run {i+1} differs from Run 1!")
            break
    
    if all_env_identical:
        print("✓ Environment is deterministic - same actions produce identical results")
    else:
        print("✗ Environment has randomness - same actions produce different results")
    
    return all_env_identical


def main():
    """Main analysis function."""
    model_path = "./ppo_quantum_results/best_model.pth"
    
    if not os.path.exists(model_path):
        print(f"Model not found at: {model_path}")
        print("Please train a model first using: python train_ppo.py")
        return
    
    # Test environment determinism
    env_deterministic = test_environment_determinism()
    
    # Analyze policy determinism
    policy_analysis = analyze_policy_determinism(model_path, num_episodes=10)
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    print(f"Environment is deterministic: {env_deterministic}")
    print(f"Policy produces identical sequences: {policy_analysis['all_identical']}")
    print(f"Policy determinism score: {policy_analysis['determinism_score']:.1f}%")
    
    if policy_analysis['all_identical'] and env_deterministic:
        print("\n✓ EXPLANATION: Your agent has converged to a fully deterministic policy")
        print("  in a deterministic environment, which explains the identical results.")
    elif policy_analysis['determinism_score'] > 95:
        print("\n✓ EXPLANATION: Your agent has converged to a nearly deterministic policy")
        print("  with very high action probabilities, leading to consistent behavior.")
    else:
        print("\n? EXPLANATION: There might be other sources of randomness affecting results.")


if __name__ == "__main__":
    main()
