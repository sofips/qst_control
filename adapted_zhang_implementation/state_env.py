# -*- coding: utf-8 -*-
import sys
import numpy as np
import itertools
import os
# Add repo root to sys.path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.insert(0, repo_root)

from shared import action_selector, gen_props, state_fidelity, mean_site


def to_complex_vector(real_vector):
    """Convert a real vector to a complex vector."""

    complex_vector = [complex(real_vector[2 * i], real_vector[2 * i + 1]) 
                      for i in range(real_vector.shape[0] // 2)]

    return complex_vector


def to_real_vector(complex_vector):
    """Convert a complex vector to a real vector."""
    real_vector = np.array(
            list(itertools.chain(*[(i.real, i.imag) for i in complex_vector]))
        )
    return real_vector


def original_reward(next_state, time_step, config_instance):

    tol = config_instance.getfloat("system_parameters", "tolerance")
    gamma = config_instance.getfloat("learning_parameters", "gamma")

    fidelity = state_fidelity(next_state)

    if fidelity < 0.8:
        reward = fidelity * 10  # +1/diferencia entre localizacion y N
    elif 0.8 <= fidelity <= 1-tol:
        reward = 100 / (1 + np.exp(10 * (1 - tol - fidelity)))
    elif fidelity > 1 - tol:
        reward = 2500

    reward = reward * (gamma ** time_step)

    return reward


def site_evolution_reward(next_state, time_step, config_instance):

    gamma = config_instance.getfloat("learning_parameters", "gamma")
    time_step = config_instance.getfloat("system_parameters", "tstep_length")

    fidelity = state_fidelity(next_state)

    mean_site_val = mean_site(next_state)

    reward = (1+fidelity)*mean_site_val
    reward = reward * (gamma ** time_step)

    return reward


class State(object):

    def __init__(self, config_instance):

        self.config_instance = config_instance
        reward_function = config_instance.get("learning_parameters",
                                              "reward_function")

        if reward_function == "original":
            self.reward_function = original_reward
        elif reward_function == "site evolution":
            self.reward_function = site_evolution_reward

        action_set = config_instance.get("system_parameters", "action_set")
        field_strength = config_instance.getfloat("system_parameters",
                                                  "field_strength")

        self.chain_length = config_instance.getint("system_parameters",
                                                   "chain_length")
        coupling = config_instance.getfloat("system_parameters",
                                            "coupling")

        action_hamiltonians = action_selector(action_set,
                                              self.chain_length,
                                              field_strength,
                                              coupling)

        DT = config_instance.getfloat("system_parameters", "tstep_length")
        self.propagators = gen_props(action_hamiltonians, DT)

    def reset(self):
        psi = [0 for _ in range(self.chain_length)]
        psi[0] = 1

        self.state = np.array(
            list(itertools.chain(*[(i.real, i.imag) for i in psi]))
        )
        self.stp = 0
        self.maxfid = 0

        return self.state

    def step(self, actionnum):
        self.stp += 1

        prop = self.propagators[actionnum]

        statess = [
            complex(self.state[2 * i], self.state[2 * i + 1])
            for i in range(self.chain_length)
        ]  # transfer real vector to complex vector

        statelist = np.transpose(np.mat(statess))  # to 'matrix'
        next_state = prop * statelist  # do operation

        reward = self.reward_function(
            next_state,
            time_step=self.stp,
            config_instance=self.config_instance
        )
        fidelity = state_fidelity(next_state)

        # apply noise in case of noisy environment
        noise = self.config_instance.getboolean("noise_parameters", "noise")
        noise_amplitude = self.config_instance.getfloat("noise_parameters",
                                                        "noise_amplitude")
        noise_probability = self.config_instance.getfloat("noise_parameters",
                                                          "noise_probability")

        if noise:
            if np.random.rand() < noise_probability:
                random_phases = np.random.uniform(-1, 1, size=self.chain_length) * noise_amplitude
                for i in range(self.chain_length):
                    next_state[i] = next_state[i] * np.exp(1j * random_phases[i])

        norm = np.linalg.norm(next_state)
        if 1 - norm > 1e-10:
            raise ValueError("State is not normalized after noise addition")

        if fidelity > self.maxfid:
            self.maxfid = state_fidelity(next_state)

        next_states = [next_state[i, 0] for i in range(self.chain_length)]
        next_states = np.array(
            list(itertools.chain(*[(i.real, i.imag) for i in next_states]))
        )  # complex to real vector

        self.state = next_states  # this vector is input to the network
        return self.state, reward, fidelity
