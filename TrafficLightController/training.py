import gymnasium as gym
from stable_baselines3 import PPO
import sumo_rl

env = gym.make(
    "sumo-rl-v0",
    use_gui=True,
    num_seconds=100000,
    net_file="sumo_rl/nets/2way-single-intersection/single-intersection.net.xml",
    route_file="sumo_rl/nets/2way-single-intersection/single-intersection-gen.rou.xml",
)

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100000)

model.save("ppo_sumo_model")