import gym

env = gym.make("Humanoid-v4", render_mode="human")
env.reset()

while 1:
    env.render()