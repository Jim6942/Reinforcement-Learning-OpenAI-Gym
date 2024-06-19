Acrobot Agent
Environment Overview
The Acrobot environment in OpenAI's Gym features a two-link pendulum with only the second joint actuated. The objective is to apply torques to swing the free end above a specified height, starting from a hanging downward position.

Action Space:

Three discrete actions representing torques: -1, 0, and 1.
Observation Space:

A 6-dimensional vector including the cosine and sine of two angles and their angular velocities.
Rewards:

Each step without reaching the goal incurs a reward of -1.
Reaching the goal gives a reward of 0.
The reward threshold for success is -100.
Starting State:

Initialized with small random values around a downward hanging position.
Episode End:

The episode terminates when the free end reaches the target height or exceeds 500 steps.
Model and Agent Settings
The agent uses a Deep Q-Network (DQN) with a neural network model consisting of two hidden layers with 64 units each and ReLU activation functions. The output layer uses a linear activation function to estimate action values.

Agent Configuration:

The agent follows an epsilon-greedy policy to balance exploration and exploitation.
It uses a memory buffer to store past experiences, with a limit of 100,000.
Double DQN and dueling network architectures enhance learning stability and performance.
The agent is trained with an Adam optimizer at a learning rate of 0.001, focusing on minimizing the mean absolute error (MAE).
Training
The agent was trained on a laptop with an Intel Core i5 CPU and Intel iRIS Xe graphics, taking approximately 30 minutes to an hour to achieve benchmark performance. The main challenge was optimizing hyperparameters to handle the sparse reward structure effectively.
