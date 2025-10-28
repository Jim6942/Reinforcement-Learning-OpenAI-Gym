import random

import tensorflow as tf

from keras.layers import Dense, Flatten

from rl.agents import ContinuousDQNAgent
from rl.memory import SequentialMemory
from rl.policy import EpsGreedyQPolicy

import gym

env = gym.make("CarRacing-v0")

states = env.observation_space.shape[0]
actions = env.action_space.shape[0]

print(states, actions)

model = tf.keras.models.Sequential()
model.add(Flatten(input_shape = (1,states)))
model.add(Dense(128, activation = 'relu'))
model.add(Dense(128, activation = 'relu'))
model.add(Dense(64, activation = 'relu'))
model.add(Dense(actions, activation = 'linear'))

agent = ContinuousDQNAgent(
    model = model,
    nb_actions= actions,
    policy = EpsGreedyQPolicy,
    memory = SequentialMemory(limit = 160000, window_length=1),
    nb_steps_warmup=100000,
    enable_double_dqn = True,
    enable_dueling_network = True,

)


