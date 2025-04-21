import gymnasium as gym
import matplotlib.pyplot as plt
import munchhausen_dql as mdql

fig, ax = plt.subplots()
plot_data = []
rewards = []
plt.ion()
plt.show()

# Initialise the environment
env = gym.make("LunarLander-v3", render_mode="human")
# stuff
agent = mdql.munchhausen_dql(input_dim=8, output_dim=4, gamma = 0.99, epsilon=0.0001, buffer_size=300000, lr=0.0001, copy_interval=2000, tau=0.1, alpha=0.9)
# Reset the environment to generate the first observation
old_observation, info = env.reset()

for i in range(10000000):
    action = agent.ask(old_observation)
    # receiving the next observation, reward and if the episode has terminated or truncated
    observation, reward, terminated, truncated, info = env.step(action)
    agent.replay_buffer.append(old_observation, observation, action, reward, terminated or truncated)
    agent.train_step()
    old_observation = observation

    rewards.append(reward)

    # If the episode has ended then we can reset to start a new episode
    if terminated or truncated:
        #print(len(agent.replay_buffer.transitions))
        plot_data.append(sum(rewards))
        rewards = []
        ax.clear()
        ax.plot(plot_data)
        fig.canvas.draw()
        old_observation, info = env.reset()

env.close()