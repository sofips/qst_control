# Proximal Policy Optimization (PPO) for Quantum State Control

This directory contains a complete PyTorch-based implementation of Proximal Policy Optimization (PPO) for controlling quantum states. PPO is a state-of-the-art policy gradient method that offers stable training and excellent performance on continuous control tasks.

## Features

### Core PPO Implementation
- **Actor-Critic Architecture**: Policy and value function networks
- **Shared Networks**: Option for shared feature extraction
- **Clipped Surrogate Objective**: Prevents large policy updates
- **Generalized Advantage Estimation (GAE)**: Better advantage computation
- **Multiple Update Epochs**: Efficient sample utilization
- **Gradient Clipping**: Training stability

### Modern RL Best Practices
- ✅ Gymnasium compatibility
- ✅ Configurable hyperparameters
- ✅ Automatic advantage normalization
- ✅ Entropy regularization for exploration
- ✅ Value function loss weighting
- ✅ Model saving/loading
- ✅ Training progress visualization
- ✅ Comprehensive evaluation metrics

## File Structure

```
ppo/
├── ppo_networks.py      # Actor, Critic, and ActorCritic networks
├── ppo_memory.py        # Memory buffer and GAE computation
├── ppo_agent.py         # Main PPO agent implementation
├── train_ppo.py         # Training script
├── demo_ppo.py          # Quick demo/test script
└── README.md            # This file
```

## Why PPO for Quantum Control?

**Advantages over DQN:**
- 🎯 **Policy-based**: Directly optimizes the policy (better for discrete actions)
- 🔄 **On-policy**: Uses fresh data for each update
- 📈 **Stable**: Clipped objective prevents destructive updates
- 🚀 **Sample efficient**: Multiple epochs per data collection
- 🎲 **Better exploration**: Stochastic policies with entropy bonus

**When to use PPO vs DQN:**
- **PPO**: When you want stable training and good exploration
- **DQN**: When you want off-policy learning and experience replay

## Installation

Same requirements as the DQN implementation:

```bash
conda create -n quantum_ppo python=3.9
conda activate quantum_ppo
conda install pytorch numpy scipy matplotlib
pip install gymnasium
```

## Quick Start

### 1. Test the Implementation

```bash
cd ppo
python demo_ppo.py
```

This will:
- Test the quantum environment
- Test the PPO agent
- Run a mini training session (30 episodes)
- Compare with random policy

### 2. Full Training

```bash
python train_ppo.py
```

This will:
- Train for 3000 episodes by default
- Update policy every 20 episodes
- Save the best model and training progress
- Create comprehensive visualizations

### 3. Custom Training

Modify the configuration in `train_ppo.py`:

```python
ppo_config = {
    'learning_rate': 3e-4,          # How fast to learn
    'gamma': 0.99,                  # Discount factor
    'gae_lambda': 0.95,             # GAE parameter
    'clip_ratio': 0.2,              # PPO clipping
    'vf_coef': 0.5,                 # Value function weight
    'ent_coef': 0.01,               # Entropy bonus
    'max_grad_norm': 0.5,           # Gradient clipping
    'update_epochs': 10,            # PPO update epochs
    'batch_size': 64,               # Mini-batch size
    'hidden_dims': [256, 256],      # Network size
    'use_shared_network': True,     # Shared features
}
```

## Key PPO Concepts

### 1. **Clipped Surrogate Objective**

PPO prevents large policy updates using clipping:

```
L^CLIP(θ) = E[min(r_t(θ)A_t, clip(r_t(θ), 1-ε, 1+ε)A_t)]
```

Where:
- `r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)` (probability ratio)
- `A_t` is the advantage estimate
- `ε` is the clip ratio (typically 0.2)

### 2. **Generalized Advantage Estimation (GAE)**

Better advantage computation that balances bias and variance:

```
A_t = δ_t + (γλ)δ_{t+1} + (γλ)²δ_{t+2} + ...
```

Where `δ_t = r_t + γV(s_{t+1}) - V(s_t)`

### 3. **Actor-Critic Architecture**

- **Actor**: Outputs action probabilities π(a|s)
- **Critic**: Estimates state values V(s)
- **Shared features**: Optional shared layers for efficiency

## Configuration Guide

### PPO Hyperparameters

**Core Parameters:**
- `learning_rate` (3e-4): How fast networks learn
- `clip_ratio` (0.2): How much policy can change per update
- `update_epochs` (10): How many times to use each batch of data

**GAE Parameters:**
- `gamma` (0.99): Future reward discount
- `gae_lambda` (0.95): GAE trade-off parameter

**Loss Weights:**
- `vf_coef` (0.5): Value function loss weight
- `ent_coef` (0.01): Entropy bonus weight

**Training Parameters:**
- `batch_size` (64): Mini-batch size for updates
- `max_grad_norm` (0.5): Gradient clipping threshold

### Environment Parameters

Same as DQN - modify in `create_config()`:

```python
config['system_parameters'] = {
    'chain_length': '8',        # System size
    'field_strength': '100',    # Control strength
    'max_t_steps': '80',        # Episode length
    'action_set': 'zhang',      # Control strategy
}
```

## Training Process

### 1. **Data Collection Phase**

```python
# Collect trajectories
for episode in range(collection_episodes):
    state, _ = env.reset()
    while not done:
        action, log_prob, value = agent.get_action_and_value(state)
        next_state, reward, done, _, info = env.step(action)
        agent.store_transition(state, action, log_prob, reward, value, done)
        state = next_state
```

### 2. **Advantage Computation**

```python
# Compute advantages using GAE
agent.memory.compute_advantages_and_returns()
```

### 3. **Policy Update Phase**

```python
# Multiple epochs of mini-batch updates
for epoch in range(update_epochs):
    for batch in agent.memory.get_batches(batch_size):
        # Compute PPO loss and update networks
        agent.update_networks(batch)
```

## Monitoring Training

The training script provides detailed monitoring:

```
Episode 100/3000
  Recent avg reward: 156.23
  Recent avg max fidelity: 0.8945
  Recent avg episode length: 67.3
  Memory size: 423

Update 15: Policy Loss: 0.0234, Value Loss: 0.1567, Entropy Loss: -2.3456

Episode 500/3000
  Evaluation at episode 500:
    Mean reward: 178.45 ± 23.12
    Mean max fidelity: 0.9234 ± 0.0567
    Mean episode length: 72.1
```

## Performance Comparison

**Expected behavior:**
- **Early training**: High entropy, random-like behavior
- **Mid training**: Policy starts to learn, entropy decreases
- **Late training**: Stable policy, consistent high performance

**Typical convergence:**
- Episodes 0-500: Learning basic control
- Episodes 500-1500: Refining strategy
- Episodes 1500+: Fine-tuning and stability

## Advanced Usage

### 1. **Separate Actor-Critic Networks**

```python
ppo_config['use_shared_network'] = False
```

Pros: More flexibility, separate learning rates possible
Cons: More parameters, potentially slower training

### 2. **Custom Network Architecture**

```python
ppo_config['hidden_dims'] = [512, 512, 256]  # Deeper network
```

### 3. **Hyperparameter Tuning**

Key parameters to tune:
- `learning_rate`: Start with 3e-4, adjust based on training stability
- `clip_ratio`: 0.1-0.3 range, smaller for more conservative updates
- `ent_coef`: 0.001-0.1, higher for more exploration

### 4. **Custom Evaluation**

```python
# Show detailed episode trajectories
compare_episode_trajectories(agent, env, num_episodes=5)
```

## Troubleshooting

### Common Issues

1. **Policy collapse**: Reduce learning rate or clip ratio
2. **Slow learning**: Increase learning rate or entropy coefficient
3. **Unstable training**: Enable gradient clipping, reduce batch size
4. **Poor exploration**: Increase entropy coefficient

### Performance Tips

1. **Use GPU**: Set `device: 'cuda'` if available
2. **Tune update frequency**: Balance between 10-50 episodes
3. **Monitor entropy**: Should decrease gradually during training
4. **Check advantage normalization**: Should be enabled by default

## Comparison with DQN

| Aspect | PPO | DQN |
|--------|-----|-----|
| **Learning Type** | On-policy | Off-policy |
| **Data Efficiency** | Moderate | High |
| **Training Stability** | High | Moderate |
| **Exploration** | Natural (stochastic policy) | ε-greedy |
| **Memory Usage** | Low | High (replay buffer) |
| **Convergence** | Smooth | Can be choppy |

## Citation

If you use this PPO implementation in your research:

```bibtex
@article{schulman2017proximal,
    title={Proximal policy optimization algorithms},
    author={Schulman, John and Wolski, Filip and Dhariwal, Prafulla and Radford, Alec and Klimov, Oleg},
    journal={arXiv preprint arXiv:1707.06347},
    year={2017}
}
```

## Next Steps

1. **Run the demo**: `python demo_ppo.py`
2. **Full training**: `python train_ppo.py`
3. **Compare with DQN**: Train both and compare results
4. **Experiment**: Try different hyperparameters and network architectures

The PPO implementation provides a stable, modern approach to quantum control that often outperforms value-based methods like DQN on policy optimization tasks!
