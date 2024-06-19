import tensorflow as tf
from keras.layers import Dense, Flatten

from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

import random
import gym
import numpy

env = gym.make("CartPole-v1")

states = env.observation_space.shape[0]
actions = env.action_space.n
print(states)

model = tf.keras.Sequential()
model.add(Flatten(input_shape = (1, states)))
model.add(Dense(64, activation = "relu"))
model.add(Dense(64, activation = 'relu'))
model.add(Dense(actions, activation = 'linear'))



agent = DQNAgent(
    model = model,
    memory = SequentialMemory(limit = 50000, window_length=1),
    policy = BoltzmannQPolicy(),
    nb_actions = actions,
    nb_steps_warmup= 18,
    target_model_update= 0.01
)
agent.compile(tf.keras.optimizers.legacy.Adam(lr = 0.001), metrics = ['mae'])
agent.load_weights("Cart_pole_weights.h5f")
agent.test(env, nb_episodes = 10, visualize = True)


