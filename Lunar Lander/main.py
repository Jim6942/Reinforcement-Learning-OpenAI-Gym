import tensorflow as tf
import gym
import numpy
from keras.layers import Dense, Flatten
from rl.agents import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory
import random
import pickle


env = gym.make("LunarLander-v2")

states = env.observation_space.shape[0]
actions = env.action_space.n

model = tf.keras.Sequential()
model.add(Flatten(input_shape= (1,states)))
model.add(Dense(64, activation = 'relu'))
model.add(Dense(64, activation = 'relu'))
model.add(Dense(actions, activation = 'linear'))



agent = DQNAgent(
    model = model,
    policy = EpsGreedyQPolicy(),
    memory = SequentialMemory(limit = 100000, window_length = 1),
    nb_actions = actions,
    target_model_update = 0.01,
    nb_steps_warmup=100000,
    enable_double_dqn= True,
    enable_dueling_network=True
)

agent.compile(optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.001), metrics = ['mse'])
agent.load_weights("Lunar_lander_weights.h5f")
file_name = "lunar_lander.pkl"
with open(file_name, 'wb') as file:
    pickle.dump(agent, file)

agent.test(env, nb_episodes= 20, visualize = True)
