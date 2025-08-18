# Deep Q-Network for Quantum State Control

This implementation provides a modern PyTorch-based Deep Q-Network (DQN) for controlling quantum states in your quantum environment. The implementation includes several advanced features and follows current best practices in deep reinforcement learning.

## Features

### Core DQN Implementation
- **Standard DQN**: Basic Deep Q-Network with experience replay
- **Double DQN**: Reduces overestimation bias by using separate networks for action selection and evaluation
- **Dueling DQN**: Separates value and advantage estimation for more stable learning
- **Prioritized Experience Replay**: Samples important transitions more frequently

### Modern RL Best Practices
- ✅ Gymnasium compatibility (latest standard)
- ✅ Target network with periodic updates
- ✅ Experience replay buffer
- ✅ Epsilon-greedy exploration with decay
- ✅ Gradient clipping for stability
- ✅ Proper network initialization
- ✅ Configurable hyperparameters
- ✅ Model saving/loading
- ✅ Training progress visualization

## File Structure

```
drl/
├── dqn_network.py          # Neural network architectures
├── replay_buffer.py        # Experience replay implementations
├── dqn_agent.py           # Main DQN agent
├── train_dqn.py           # Training script
├── demo.py                # Quick demo/test script
├── quantum_environment.py  # Your existing quantum environment
└── testing_environment.py  # Your existing test utilities
```

## Installation

### Using Conda (Recommended)

```bash
# Create a new conda environment
conda create -n quantum_dqn python=3.9
conda activate quantum_dqn

# Install PyTorch (choose the appropriate version for your system)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install other dependencies
conda install numpy scipy matplotlib
pip install gymnasium configparser
```

### Using pip

```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Test the Implementation

First, run the demo to make sure everything works:

```bash
cd drl
python demo.py
```

This will:
- Test the quantum environment
- Test the DQN agent
- Run a mini training session (20 episodes)

### 2. Full Training

Run the full training script:

```bash
python train_dqn.py
```

This will:
- Train for 2000 episodes by default
- Save the best model and training progress
- Create visualizations of the training process
- Evaluate the final agent performance

### 3. Custom Training

You can customize the training by modifying the configuration in `train_dqn.py`:

```python
# DQN configuration
dqn_config = {
    'learning_rate': 1e-4,        # Learning rate
    'gamma': 0.99,                # Discount factor
    'epsilon_start': 1.0,         # Initial exploration
    'epsilon_end': 0.01,          # Final exploration
    'epsilon_decay': 0.995,       # Exploration decay rate
    'target_update_freq': 1000,   # Target network update frequency
    'batch_size': 32,             # Batch size for training
    'buffer_size': 100000,        # Replay buffer size
    'hidden_dims': [256, 256],    # Network architecture
    'use_dueling': True,          # Use dueling architecture
    'use_prioritized_replay': False,  # Use prioritized replay
    'use_double_dqn': True,       # Use double DQN
}
```

## Configuration

### Environment Configuration

The quantum environment can be configured via a configuration file or programmatically:

```python
config = configparser.ConfigParser()

config['system_parameters'] = {
    'chain_length': '6',      # Number of qubits
    'action_set': 'zhang',    # Action set type ('zhang' or 'oaps')
    'n_actions': '16',        # Number of available actions
    'field_strength': '1.0',  # Magnetic field strength
    'coupling': '1.0',        # Coupling strength
    'tstep_length': '0.1',    # Time step length
    'max_t_steps': '50',      # Maximum episode length
    'tolerance': '0.01'       # Fidelity tolerance
}

config['learning_parameters'] = {
    'reward_function': 'original',  # Reward function type
    'gamma': '0.99'                 # Discount factor
}
```

### DQN Hyperparameters

Key hyperparameters to tune:

- **Learning Rate (1e-4)**: Controls how fast the network learns
- **Epsilon Decay (0.995)**: Controls exploration-exploitation trade-off
- **Target Update Frequency (1000)**: How often to update the target network
- **Batch Size (32)**: Number of experiences used per training step
- **Hidden Dimensions ([256, 256])**: Size and depth of the neural network

## Advanced Features

### 1. Dueling DQN

Enable dueling architecture for better value estimation:

```python
dqn_config['use_dueling'] = True
```

### 2. Prioritized Experience Replay

Enable prioritized sampling of important experiences:

```python
dqn_config['use_prioritized_replay'] = True
```

### 3. Double DQN

Reduce overestimation bias (enabled by default):

```python
dqn_config['use_double_dqn'] = True
```

## Monitoring Training

The training script automatically:
- Prints progress every 100 episodes
- Evaluates the agent every 500 episodes
- Saves the best performing model
- Creates training progress plots

Example output:
```
Episode 100/2000
  Recent avg reward: 45.23
  Recent avg max fidelity: 0.8234
  Epsilon: 0.6050
  Buffer size: 3200

Episode 500/2000
  Evaluation at episode 500:
    Mean reward: 78.45 ± 12.34
    Mean max fidelity: 0.9123 ± 0.0456
```

## Evaluation

Evaluate a trained model:

```python
from train_dqn import evaluate_trained_agent

eval_stats = evaluate_trained_agent(
    model_path='./dqn_quantum_results/best_model.pth',
    num_episodes=100
)
```

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce batch size or use CPU
2. **Slow training**: Enable GPU acceleration or reduce network size
3. **Poor performance**: Tune hyperparameters or increase training time
4. **Import errors**: Make sure all dependencies are installed

### Performance Tips

1. **Use GPU**: Set `device: 'cuda'` if available
2. **Tune hyperparameters**: Start with default values and adjust based on performance
3. **Monitor epsilon**: Ensure exploration decreases over time
4. **Check reward function**: Make sure rewards are properly scaled

## Citation

If you use this implementation in your research, please consider citing:

```bibtex
@article{your_paper,
    title={Deep Q-Network for Quantum State Control},
    author={Your Name},
    journal={Your Journal},
    year={2025}
}
```

## License

This implementation is provided as-is for research and educational purposes.
