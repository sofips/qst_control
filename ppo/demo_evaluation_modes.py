"""
Simple demonstration of stochastic vs deterministic evaluation.
"""

import sys
import os
import configparser

# Add the shared module to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'shared'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'drl'))

from quantum_environment import QuantumStateEnv
from ppo_agent import PPOAgent


def quick_demo():
    """Quick demo showing evaluation differences."""
    
    # Setup environment
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
    
    try:
        agent.load("./ppo_quantum_results/best_model.pth")
    except FileNotFoundError:
        print("No trained model found. Please train first!")
        return
    
    print("Running 3 episodes each way...\n")
    
    # Stochastic (your current evaluation)
    print("STOCHASTIC (current method):")
    for i in range(3):
        state, _ = env.reset()
        total_reward = 0
        while True:
            action = agent.get_action(state, deterministic=False)
            state, reward, done, truncated, info = env.step(action)
            total_reward += reward
            if done or truncated:
                break
        print(f"  Episode {i+1}: Fidelity = {info.get('max_fidelity', 0):.6f}")
    
    print("\nDETERMINISTIC (consistent method):")
    for i in range(3):
        state, _ = env.reset()
        total_reward = 0
        while True:
            action = agent.get_action(state, deterministic=True)
            state, reward, done, truncated, info = env.step(action)
            total_reward += reward
            if done or truncated:
                break
        print(f"  Episode {i+1}: Fidelity = {info.get('max_fidelity', 0):.6f}")
    
    print("\nFor consistent evaluation, use: deterministic=True")


if __name__ == "__main__":
    quick_demo()
