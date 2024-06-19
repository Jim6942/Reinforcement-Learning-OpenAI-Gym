Mountain Car Agent
Environment Overview
The Mountain Car environment is a classic control problem where a car must navigate through a valley to reach the goal at the top of a hill. The challenge is to strategically accelerate the car using discrete actions to climb the hill.

Action Space:

Three discrete actions: accelerate left, do nothing, or accelerate right.
Observation Space:

A 2-dimensional vector representing the car's position and velocity along the x-axis.
Rewards:

The car receives a reward of -1 for each timestep. The goal is to reach the flag on top of the right hill as quickly as possible.
Starting State:

The car starts with a position uniformly random between -0.6 and -0.4, and a velocity of 0.
Episode End:

The episode ends if the car reaches or exceeds the goal position (x >= 0.5) or the episode length exceeds 200 timesteps.
Model and Agent Settings
The agent uses a Deep Q-Network (DQN) with a neural network model consisting of two hidden layers with 64 units each, and ReLU activation functions. The output layer uses a linear activation function to estimate action values.

Agent Configuration:

Policy: EpsGreedyQPolicy for balancing exploration and exploitation.
Memory: SequentialMemory with a limit of 200,000 experiences.
Training: The agent is trained with the Adam optimizer at a learning rate of 0.001, focusing on minimizing mean absolute error (MAE).
Features: Double DQN is enabled to enhance stability and performance.
Training and Performance
The agent was trained for 100,000 warm-up steps to gather initial experience.
Training took approximately 30 minutes on a laptop with an Intel Core i5 CPU and Intel iRIS Xe graphics.
After training, the agent's performance was tested over 20 episodes, with the results visualized to ensure it can successfully navigate the course.
This section provides an overview of the Mountain Car environment and the settings used to train the agent. Similar sections will be included for the other environments in this project.
