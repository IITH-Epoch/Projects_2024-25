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

model = PPO.load("ppo_sumo_model")
obs, information = env.reset()
done = False
total_reward = 0

while not done:
    action, _ = model.predict(obs)
    obs, reward, terminated , truncated, info = env.step(action)
    total_reward+=float(reward)
    done = truncated or terminated
    env.render()

print(total_reward)
env.reset()
