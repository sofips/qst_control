# -*- coding: utf-8 -*-

import numpy as np
import itertools
from actions import *

def to_complex_vector(real_vector):
    """Convert a real vector to a complex vector."""
    
    complex_vector = [complex(real_vector[2 * i], real_vector[2 * i + 1]) for i in range(real_vector.shape[0] // 2)]
     
    return complex_vector

def to_real_vector(complex_vector):
    """Convert a complex vector to a real vector."""
    real_vector = np.array(
            list(itertools.chain(*[(i.real, i.imag) for i in complex_vector]))
        ) 
    return real_vector

# state fidelity and mean site take complex state as input
# and return a real number

def state_fidelity(state):

    return np.asarray((abs(state[-1]) ** 2)[0, 0])  # calculate fidelity


def mean_site(state):

    chain_length = np.shape(state)[0]

    ms = np.sum(
        np.asarray([
            np.real(state[j] * np.conjugate(state[j])) * (j + 1)
            for j in range(0, chain_length)
        ])
    )
    return ms


def original_reward(next_state, natural_state, time_step, config_instance):
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


def site_evolution_reward(next_state, natural_state, time_step, config_instance):
    tol = config_instance.getfloat("system_parameters", "tolerance")
    gamma = config_instance.getfloat("learning_parameters", "gamma")
    chain_length = config_instance.getint("system_parameters", "chain_length")
    DT = config_instance.getfloat("system_parameters", "tstep_length")

    fidelity = state_fidelity(next_state)
    
    mean_site_val = mean_site(next_state)
    
    reward = (1+fidelity)*mean_site_val

    #nat_mean_site = mean_site(natural_state)
    
    #print(f"Fidelity: {fidelity}, Mean Site Value: {mean_site_val}, Natural Mean Site: {nat_mean_site}")

    #if time_step < chain_length:
    #reward =   100 * fidelity + 1000*(-nat_mean_site+mean_site_val)/chain_length
    # elif time_step < 3 * chain_length:
    #     reward = 100 * (mean_site_val / (time_step * DT)) ** 2
    # else:
    #     if fidelity > 1 - tol:
    #         reward = 10000 * fidelity
    #     else:
    #         reward = 1000 * (mean_site_val / chain_length) * fidelity ** 2

    reward = reward * (gamma ** time_step)

    return reward



class State(object):

    def __init__(self, config_instance):
        
        self.config_instance = config_instance
        reward_function = config_instance.get("learning_parameters", "reward_function")

        if reward_function == "original":
            self.reward_function = original_reward
        elif reward_function == "site evolution":
            self.reward_function = site_evolution_reward

        action_set = config_instance.get("system_parameters", "action_set")
        field_strength = config_instance.getfloat("system_parameters", "field_strength")

        if action_set not in ["zhang", "oaps"]:
            raise ValueError(f"Unknown action set: {action_set}. Choose 'zhang' or 'oaps'.")
        
        self.chain_length = config_instance.getint("system_parameters", "chain_length")
        coupling = config_instance.getfloat("system_parameters", "coupling")

        if action_set == "zhang":
            action_hamiltonians = zhang_actions(field_strength, self.chain_length, coupling)
        elif action_set == "oaps":
            action_hamiltonians = one_field_actions(field_strength, self.chain_length, coupling)

        DT = config_instance.getfloat("system_parameters", "tstep_length")
        self.propagators = gen_props(action_hamiltonians, DT)

    def reset(self):
        psi = [0 for _ in range(self.chain_length)]  # initial state is [1;0;0;0;0...]
        psi[0] = 1

        self.state = np.array(
            list(itertools.chain(*[(i.real, i.imag) for i in psi]))
        )
        self.stp = 0
        self.maxfid = 0

        self.natstate = self.state.copy()  # natural state is the initial state
        
        # the input of network is a real vector, so we need to transfer complex vector to real vector
        return self.state

    def step(self, actionnum):
        self.stp += 1

        prop = self.propagators[actionnum]
        natural_prop = self.propagators[0]  # natural state propagator

        statess = [
            complex(self.state[2 * i], self.state[2 * i + 1])
            for i in range(self.chain_length)
        ]  # transfer real vector to complex vector

        statelist = np.transpose(np.mat(statess))  # to 'matrix'
        next_state = prop * statelist  # do operation

        natstatess = [complex(self.natstate[2 * i], self.natstate[2 * i + 1])
                      for i in range(self.chain_length)]
        natstatelist = np.transpose(np.mat(natstatess))
        next_nat_state = natural_prop * natstatelist  # do operation on natural state
        
        
        reward = self.reward_function(
            next_state, next_nat_state, time_step=self.stp, config_instance=self.config_instance
        )
        fidelity = state_fidelity(next_state)

        # apply noise in case of noisy environment
        noise = self.config_instance.getboolean("noise_parameters", "noise")
        noise_amplitude = self.config_instance.getfloat("noise_parameters", "noise_amplitude")
        noise_probability = self.config_instance.getfloat("noise_parameters", "noise_probability")

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

        next_states = [next_state[i, 0] for i in range(self.chain_length)]  # 'matrix' to list
        next_states = np.array(
            list(itertools.chain(*[(i.real, i.imag) for i in next_states]))
        )  # complex to real vector

        next_nat_state = [next_nat_state[i, 0] for i in range(self.chain_length)]
        next_nat_state = np.array(
            list(itertools.chain(*[(i.real, i.imag) for i in next_nat_state]))
        )

        self.state = next_states  # this vector is input to the network
        self.natstate = next_nat_state
        return self.state, reward, fidelity
