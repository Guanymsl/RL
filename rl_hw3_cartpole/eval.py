import gymnasium as gym
from gymnasium.wrappers import RecordVideo

from stable_baselines3 import DQN

import numpy as np

env_id = "CartPole-v1"
eval_env = gym.make(env_id)

model = DQN.load("models/dqn_cartpole", env=eval_env)

video_env = RecordVideo(
    gym.make(env_id, render_mode="rgb_array"),
    video_folder="videos",
    episode_trigger=lambda x: True,
)

obs, info = video_env.reset()

done = False
while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = video_env.step(action)
    done = terminated or truncated

video_env.close()
