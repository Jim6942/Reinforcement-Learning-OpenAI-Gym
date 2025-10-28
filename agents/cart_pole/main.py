import os, numpy as np, gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor

os.makedirs("agents/cartpole_dqn", exist_ok=True)
env = Monitor(gym.make("CartPole-v1"))
model = DQN("MlpPolicy", env, learning_rate=1e-3, 
            buffer_size=50_000,
            learning_starts=1000, 
            batch_size=64, gamma=0.99, 
            train_freq=4,
            target_update_interval=500, 
            exploration_fraction=0.2,
            exploration_final_eps=0.02, 
            seed=0, verbose=1)
            
model.learn(total_timesteps=150_000, log_interval=10)
model.save("agents/cartpole_dqn/best_model")
print("saved to agents/cartpole_dqn/best_model.zip")
