# qst_control

This repository contains implementations of two optimization approaches: a Genetic Algorithm (GA) and a Deep Reinforcement Learning (DRL) agent to solve the Quantum State Transfer problem in spin chains or other equivalent qubit arrays.

## Physical Problem
Quantum State Transfer (QST) [1] is a fundamental task in quantum information processing, involving the faithful transmission of a quantum state across a network or chain of qubits. Efficient protocols to maximize transfer fidelity while minimizing errors and time are crucial.

This repo provides two algorithmic frameworks for optimizing the QST:

Genetic Algorithm: An evolutionary approach that evolves a population of candidate solutions to optimize the transfer protocol parameters based on PyGAD library [2].

Deep Reinforcement Learning: An RL agent trained to learn optimal control policies for the QST task by interacting with a quantum environment. We based our approach on the work of Zhang et al [3] whose source code was kindly shared by the authors under an MIT license.


[1] D.S. Acosta Coden, S.S. Gómez, A. Ferrón, and O. Osenda. Controlled quantum
state transfer in xx spin chains at the quantum speed limit. Physics Letters A,
387:127009, 2021.

[2] Ahmed Fawzy Gad. Pygad: An intuitive genetic algorithm python library, 2023.

[3] Xiao-Ming Zhang, Zi-Wei Cui, Xin Wang, and Man-Hong Yung. Automatic spinchain learning to explore the quantum speed limit. Phys. Rev. A, 97:052333, 2018.


