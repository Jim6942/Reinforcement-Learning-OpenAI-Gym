import tensorflow as tf
from keras.layers import Dense, Flatten

from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.agents import DQNAgent

import gym

env = gym.make("MountainCar-v0")

states = env.observation_space.shape[0]
actions = env.action_space.n

model = tf.keras.Sequential()
model.add(Flatten(input_shape=(1,states)))
model.add(Dense(64, activation = 'relu'))
model.add(Dense(64, activation = 'relu'))
model.add(Dense(actions, activation = 'linear'))

agent = DQNAgent(
    model = model,
    nb_actions= actions,
    nb_steps_warmup= 100000,
    policy = EpsGreedyQPolicy(),
    target_model_update= 0.01,
    memory = SequentialMemory(limit = 200000, window_length=1),
    enable_double_dqn=True

)

agent.compile(tf.keras.optimizers.legacy.Adam(lr=0.001), metrics=["mae"])

agent.load_weights("Mountain_Car_weights.h5f")

agent.test(env, nb_episodes=20, visualize=True)
