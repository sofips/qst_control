from typing import Optional
import numpy as np
import gymnasium as gym
import sys
import os

# Add the shared module to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'shared'))
from actions_module import zhang_actions, one_field_actions, gen_props
import itertools


def to_complex_mat(real_vector):
    """Convert a real vector to a complex vector."""
    complex_vector = [
        complex(real_vector[2 * i], real_vector[2 * i + 1])
        for i in range(real_vector.shape[0] // 2)
    ]
    mat_state = np.transpose(np.asmatrix(complex_vector))

    return mat_state


def to_real_vector(complex_vector):
    """Convert a complex vector to a real vector."""
    real_vector = np.array(
        list(itertools.chain(*[(i.real, i.imag) for i in complex_vector]))
    )
    return real_vector


def apply_propagator(state, propagator, real_input_state=True, random_phases=None):

    if real_input_state:
        state = to_complex_mat(state)

    next_state = propagator * state

    if random_phases is not None:
        for i in range(len(random_phases)):
            next_state[i] *= np.exp(1j * random_phases[i])
            
    next_state = to_real_vector(next_state)
    return next_state


def state_fidelity(state, real_input_state=True):

    if real_input_state:
        state = to_complex_mat(state)
    return np.asarray((abs(state[-1]) ** 2)[0, 0])  # calculate fidelity


def mean_site(state, real_input_state=True):

    if real_input_state:
        state = to_complex_mat(state)

    chain_length = np.shape(state)[0]

    ms = np.sum(
        np.asarray(
            [
                np.real(state[j] * np.conjugate(state[j])) * (j + 1)
                for j in range(0, chain_length - 1)
            ]
        )
    )
    return ms


def original_reward(state, time_step, config_instance):

    tol = config_instance.getfloat("system_parameters", "tolerance")
    gamma = config_instance.getfloat("learning_parameters", "gamma")

    fidelity = state_fidelity(state)

    if fidelity < 0.8:
        reward = fidelity * 10  # +1/diferencia entre localizacion y N
    elif 0.8 <= fidelity <= 1 - tol:
        reward = 100 / (1 + np.exp(10 * (1 - tol - fidelity)))
    elif fidelity > 1 - tol:
        reward = 2500

    reward = reward * (gamma**time_step)

    return reward


def site_evolution_reward(state, time_step, config_instance):
    gamma = config_instance.getfloat("learning_parameters", "gamma")

    fidelity = state_fidelity(state)
    mean_site_val = mean_site(state)

    reward = (1 + fidelity) * mean_site_val
    reward = reward * (gamma**time_step)

    return reward


class QuantumStateEnv(gym.Env):
    def __init__(self, config_instance):

        self.config_instance = config_instance

        # Define the state space
        self.chain_length = config_instance.getint("system_parameters",
                                                   "chain_length")
        self.n_features = 2 * self.chain_length
        self.observation_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(self.n_features,), dtype=np.float64
        )

        # Initialize the state
        self.initial_state = np.zeros(self.n_features, dtype=np.float64)
        self.initial_state[0] = (
            1.0  # Set the first coefficient to 1.0 (up down ... down state)
        )
        self.state = self.initial_state.copy()

        # Define the action space

        # Type of actions
        action_set = config_instance.get("system_parameters",
                                         "action_set")
        self.n_actions = config_instance.getint("system_parameters",
                                                "n_actions")

        # Magnetic field parameters
        self.field_strength = config_instance.getfloat(
            "system_parameters", "field_strength"
        )
        self.coupling = config_instance.getfloat("system_parameters",
                                                 "coupling")

        self.time_step = config_instance.getfloat("system_parameters",
                                                  "tstep_length")

        if action_set == "zhang":
            action_hamiltonians = zhang_actions(
                self.field_strength, self.chain_length, self.coupling
            )
        elif action_set == "oaps":
            action_hamiltonians = one_field_actions(
                self.field_strength, self.chain_length, self.coupling
            )
        else:
            raise ValueError(
                f"Unknown action set: {action_set}. Choose 'zhang' or 'oaps'."
            )

        # Generate propagators for the actions
        self.action_matrices = action_hamiltonians
        self.propagators = gen_props(action_hamiltonians, self.time_step)

        # Final action space using gym
        self.action_space = gym.spaces.Discrete(len(self.propagators))

        # Set time step and time limit
        self.max_time_steps = config_instance.getint("system_parameters",
                                                     "max_t_steps")
        self.current_time_step = 0

        self.noise = config_instance.getboolean("noise_parameters", "noise")
        
        if self.noise:
            self.noise_probability = config_instance.getfloat("noise_parameters", "noise_probability")
            self.noise_amplitude = config_instance.getfloat("noise_parameters", "noise_amplitude")
        else:
            self.noise_probability = 0.0
            self.noise_amplitude = 0.0

        # Access reward function
        reward_function = config_instance.get("learning_parameters",
                                              "reward_function")

        if reward_function == "original":
            self.reward_function = original_reward
        elif reward_function == "site evolution":
            self.reward_function = site_evolution_reward
        else:
            raise ValueError(
                f"Unknown reward function: {reward_function}."
                "Choose 'original' or 'site evolution'."
            )
        self.max_fidelity = 0.0

    def _get_info(self):
        """We access the relevant information about the state."""

        fidelity = state_fidelity(self.state, real_input_state=True)
        ms = mean_site(self.state, real_input_state=True)

        info = {
            "current_time_step": self.current_time_step,
            "state_fidelity": fidelity,
            "max_fidelity": self.max_fidelity,
            "mean_site": ms,
        }

        return info

    def reset(self, seed: Optional[int] = None):
        """Reset the environment to an initial state. We set the 1st qubit to
        up and the rest to down. We also reset the time step counter."""

        self.state = self.initial_state.copy()
        self.current_time_step = 0

        return self.state, self._get_info()

    def step(self, action):
        """Apply action and compute next state, reward, done, and info."""

        # get corresponding propagator
        propagator = self.propagators[action]

        # Apply the action to the current state
        if self.noise:
            if np.random.rand() < self.noise_probability:
                random_phases = np.random.uniform(-1, 1, size=self.chain_length)
                random_phases = random_phases * self.noise_amplitude
            else:
                random_phases = None
        else:
            random_phases = None

        self.state = apply_propagator(self.state, propagator, real_input_state=True, random_phases=random_phases)

        # Compute reward (for simplicity, we use fidelity as a reward)
        reward = self.reward_function(
            self.state,
            time_step=self.current_time_step,
            config_instance=self.config_instance,
        )

        self.state = self.state.flatten()  # Flatten the state for observation

        # Check if the episode is done (fixed time limit)
        self.current_time_step += 1
        done = self.current_time_step >= self.max_time_steps
        truncated = False  # No truncation in this environment
        info = self._get_info()  # fidelity and mean site
        fidelity = info["state_fidelity"]
        
        if fidelity > self.max_fidelity:
            self.max_fidelity = fidelity
        if done:
            
            print(f"Max fidelity reached: {self.max_fidelity:.4f} ")
        
        return self.state, reward, done, truncated, info


