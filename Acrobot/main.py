import tensorflow as tf
import gym

from keras.layers import Dense, Flatten
from rl.agents import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory

env = gym.make("Acrobot-v1")

states = env.observation_space.shape[0]
actions = env.action_space.n

model = tf.keras.Sequential()
model.add(Flatten(input_shape=(1,states)))
model.add(Dense(64, activation = 'relu'))
model.add(Dense(64, activation = 'relu'))
model.add(Dense(actions, activation = 'linear'))

agent = DQNAgent(
    model = model,
    policy = EpsGreedyQPolicy(),
    memory = SequentialMemory(limit = 100000, window_length=1),
    nb_steps_warmup=100000,
    nb_actions= actions,
    enable_double_dqn=True,
    target_model_update=0.01,
    enable_dueling_network=True
)

agent.compile(tf.keras.optimizers.legacy.Adam(lr=0.001), metrics=['mae'])


agent.load_weights("Acrobot_weights.h5f")



agent.test(env, nb_episodes=20, visualize = True)
