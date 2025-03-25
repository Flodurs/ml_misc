import gymnasium as gym
import REINFORCE 
import matplotlib.pyplot as plt

class sar:
    def __init__(self, s, a, r):
        self.state = s
        self.action = a
        self.reward = r


fig, ax = plt.subplots()
plot_data = []
plt.ion()
plt.show()


# Initialise the environment
env = gym.make("LunarLander-v2", render_mode="human")

# stuff
latest_trajectory = []
trajectorys = []
trajectory_sample_num = 5
agent = REINFORCE.REINFORCE(8,4, 0.0001, 0.0001, baseline_mode=2)
# Reset the environment to generate the first observation
old_observation, info = env.reset()
for i in range(10000000):
    # this is where you would insert your policy

    action = agent.ask(old_observation)
    # step (transition) through the environment with the action
    # receiving the next observation, reward and if the episode has terminated or truncated
    observation, reward, terminated, truncated, info = env.step(action)
    s = sar(old_observation, action, reward)
    latest_trajectory.append(s)
    old_observation = observation

    # If the episode has ended then we can reset to start a new episode
    if terminated or truncated:
        old_observation, info = env.reset()
        trajectorys.append(latest_trajectory)
        latest_trajectory = []
    if len(trajectorys) == trajectory_sample_num:
        agent.train_on_trajectorys(trajectorys)
        print("training ... ")
        avg_reward = 0.0
        for t in trajectorys:
            avg_reward += sum([p.reward for p in t])
        plot_data.append(avg_reward/trajectory_sample_num)
        ax.clear()
        ax.plot(plot_data)
        fig.canvas.draw()
        trajectorys = []

env.close()