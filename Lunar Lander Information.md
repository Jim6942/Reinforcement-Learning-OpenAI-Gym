Lunar Lander Agent
Environment Overview
The Lunar Lander environment is a classic rocket trajectory optimization problem, where the goal is to land a spacecraft on a designated landing pad. The environment provides two versions: discrete and continuous action spaces.

Action Space:

Four discrete actions: do nothing, fire left orientation engine, fire main engine, fire right orientation engine.
Observation Space:

An 8-dimensional vector including the lander's coordinates, linear velocities, angle, angular velocity, and the state of the lander's legs (in contact with the ground or not).
Rewards:

Rewards are given for successful landing and penalties for crashing or moving away from the landing pad. Firing engines also incurs a small penalty.
Starting State:

The lander starts at the top center of the viewport with a random initial force applied to its center of mass.
Episode End:

The episode ends if the lander crashes, moves outside the viewport, or comes to rest.
Model and Agent Settings
The agent uses a Deep Q-Network (DQN) with a neural network model consisting of two hidden layers with 64 units each and ReLU activation functions. The output layer uses a linear activation function to estimate action values.

Agent Configuration:

Policy: EpsGreedyQPolicy for balancing exploration and exploitation.
Memory: SequentialMemory with a limit of 100,000 experiences.
Training: The agent is trained using the Adam optimizer with a learning rate of 0.001, and focuses on minimizing mean squared error (MSE).
Training and Performance:

Training the Lunar Lander agent is challenging and requires numerous attempts to achieve optimal performance.
The agent is trained with 100,000 warm-up steps, employs double DQN, and uses a dueling network architecture to improve stability and performance.
Testing:

After training, the agent's performance is tested over 20 episodes, and the results are visualized to ensure the agent's effectiveness in landing the spacecraft.
This section provides an overview of the Lunar Lander environment and the settings used to train the agent. Similar sections will be included for the other environments in this project.
