Cart Pole Agent
-

Environment Overview
-

The Cart Pole environment in OpenAI's Gym involves a pole attached to a cart moving along a frictionless track. The objective is to balance the pole by applying forces to push the cart left or right.

Action Space:
-

Two discrete actions: push the cart left (0) or right (1).

Observation Space:
-

A 4-dimensional vector including cart position, cart velocity, pole angle, and pole angular velocity.
Rewards:

A reward of +1 for each step the pole remains upright. The reward threshold for success is 475.

Starting State:
-

All observations are initialized with random values between -0.05 and 0.05.

Episode End:
-

The episode terminates if the pole angle exceeds ±12°, the cart position exceeds ±2.4, or the episode length exceeds 500 steps.

Model and Agent Settings
-

The agent uses a Deep Q-Network (DQN) with a neural network model consisting of two hidden layers with 64 units each and ReLU activation functions. The output layer uses a linear activation function to estimate action values.

Agent Configuration:
-

The agent employs a Boltzmann policy to balance exploration and exploitation.

It uses a memory buffer to store past experiences, with a limit of 50,000.

The agent is trained with an Adam optimizer at a learning rate of 0.001, focusing on minimizing the mean absolute error (MAE).

Training
-

The Cart Pole agent was trained to achieve the benchmark set by the OpenAI Gym. Training on a laptop with an Intel Core i5 CPU and Intel iRIS Xe graphics took approximately 30 minutes to an hour. The agent was able to successfully balance the pole by applying optimal left and right forces to the cart.








