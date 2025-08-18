"""
Test script to demonstrate the difference between stochastic and deterministic evaluation.
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


def compare_evaluation_modes():
    """Compare stochastic vs deterministic evaluation."""
    
    print("="*60)
    print("STOCHASTIC vs DETERMINISTIC EVALUATION")
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
    
    model_path = "./ppo_quantum_results/best_model.pth"
    if not os.path.exists(model_path):
        print(f"Model not found at: {model_path}")
        print("Please train a model first!")
        return
    
    agent.load(model_path)
    
    print("Testing 5 episodes with each mode...\n")
    
    # Test stochastic evaluation (default)
    print("STOCHASTIC EVALUATION (sampling from policy):")
    print("-" * 50)
    stochastic_results = []
    
    for episode in range(5):
        state, _ = env.reset()
        total_reward = 0
        actions_taken = []
        
        while True:
            action = agent.get_action(state, deterministic=False)  # Stochastic
            actions_taken.append(action)
            state, reward, done, truncated, info = env.step(action)
            total_reward += reward
            
            if done or truncated:
                break
        
        fidelity = info.get('max_fidelity', 0)
        stochastic_results.append((total_reward, fidelity, actions_taken))
        print(f"Episode {episode+1}: Reward={total_reward:.2f}, "
              f"Fidelity={fidelity:.4f}, Actions={actions_taken[:5]}...")
    
    # Test deterministic evaluation
    print(f"\nDETERMINISTC EVALUATION (argmax of policy):")
    print("-" * 50)
    deterministic_results = []
    
    for episode in range(5):
        state, _ = env.reset()
        total_reward = 0
        actions_taken = []
        
        while True:
            action = agent.get_action(state, deterministic=True)  # Deterministic
            actions_taken.append(action)
            state, reward, done, truncated, info = env.step(action)
            total_reward += reward
            
            if done or truncated:
                break
        
        fidelity = info.get('max_fidelity', 0)
        deterministic_results.append((total_reward, fidelity, actions_taken))
        print(f"Episode {episode+1}: Reward={total_reward:.2f}, "
              f"Fidelity={fidelity:.4f}, Actions={actions_taken[:5]}...")
    
    # Analysis
    print(f"\nANALYSIS:")
    print("="*60)
    
    # Stochastic analysis
    stoch_rewards = [r[0] for r in stochastic_results]
    stoch_fidelities = [r[1] for r in stochastic_results]
    stoch_sequences = [r[2] for r in stochastic_results]
    
    print(f"STOCHASTIC RESULTS:")
    print(f"  Reward std: {np.std(stoch_rewards):.4f}")
    print(f"  Fidelity std: {np.std(stoch_fidelities):.6f}")
    print(f"  Identical sequences: {all(seq == stoch_sequences[0] for seq in stoch_sequences)}")
    
    # Deterministic analysis
    det_rewards = [r[0] for r in deterministic_results]
    det_fidelities = [r[1] for r in deterministic_results]
    det_sequences = [r[2] for r in deterministic_results]
    
    print(f"DETERMINISTIC RESULTS:")
    print(f"  Reward std: {np.std(det_rewards):.4f}")
    print(f"  Fidelity std: {np.std(det_fidelities):.6f}")
    print(f"  Identical sequences: {all(seq == det_sequences[0] for seq in det_sequences)}")
    
    # Explanation
    print(f"\nEXPLANATION:")
    print("-" * 40)
    if np.std(stoch_fidelities) > 1e-6:
        print("✓ Stochastic evaluation shows variability due to policy sampling")
    else:
        print("! Stochastic evaluation is nearly deterministic (converged policy)")
    
    if np.std(det_fidelities) < 1e-10:
        print("✓ Deterministic evaluation is perfectly consistent")
    else:
        print("? Deterministic evaluation still shows variability (check environment)")
    
    print(f"\nTo get consistent evaluation results, use:")
    print(f"  action = agent.get_action(state, deterministic=True)")


if __name__ == "__main__":
    compare_evaluation_modes()
