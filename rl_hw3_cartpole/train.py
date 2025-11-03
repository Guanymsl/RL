import gymnasium as gym

from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy

import matplotlib.pyplot as plt
import numpy as np

env_id = "CartPole-v1"
env = gym.make(env_id)

model = DQN(
    "MlpPolicy",
    env,
    batch_size=64,
    exploration_fraction=0.3,
    exploration_final_eps=0.05,
    verbose=1,
)

timesteps_per_iter = 200000
iters = 5
mean_rewards = []

for i in range(iters):
    model.learn(total_timesteps=timesteps_per_iter, reset_num_timesteps=False)
    mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=10)
    mean_rewards.append(mean_reward)

model.save("models/dqn_cartpole")

plt.figure(figsize=(6,4))
plt.plot(np.arange(1, iters + 1), mean_rewards, color="green")
plt.title("Learning Curve")
plt.xlabel("Iteration")
plt.ylabel("Average Reward")
plt.tight_layout()
plt.savefig("images/4-1.png")

env.close()
