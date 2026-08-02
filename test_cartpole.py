import gymnasium as gym
import torch
from train_cartpole import (
    DQN,
)  # Import the network architecture from your training script

env = gym.make("CartPole-v1", render_mode="human", max_episode_steps=1000)
env.unwrapped.force_mag = 40.0
env.unwrapped.gravity = 50.0
env.unwrapped.masspole = 1
env.unwrapped.length = 1.0
state, info = env.reset()
n_observations = len(state)
n_actions = env.action_space.n

#  Build the exact same neural network
policy_net = DQN(n_observations, n_actions)

policy_net.load_state_dict(
    torch.load("cartpole_dqn.pth", map_location=torch.device("cpu"))
)
policy_net.eval()  # Set the network to evaluation mode

#  Watch it play for 15 episodes
print("Starting Evaluation...")
for episode in range(15):
    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    total_reward = 0

    while True:
        with torch.no_grad():
            action = policy_net(state).max(1).indices.view(1, 1)

        observation, reward, terminated, truncated, _ = env.step(action.item())
        total_reward += reward

        if terminated or truncated:
            print(f"Episode {episode + 1} finished with a reward of {total_reward}")
            break

        state = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)

env.close()
