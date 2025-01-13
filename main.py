from env import DataCenterEnv
import numpy as np
import argparse
from agent import TD3, PPO, VPG

args = argparse.ArgumentParser()
args.add_argument('--path', type=str, default='train.xlsx')
args.add_argument('--algorithm', type=str, default='TD3')
args = args.parse_args()

np.set_printoptions(suppress=True, precision=2)
path_to_dataset = args.path

environment = DataCenterEnv(path_to_dataset)
aggregate_reward = 0
terminated = False
state = environment.observation()

state_dim = 4
action_dim = environment.continuous_action_space.shape[0]
max_action = float(environment.continuous_action_space.high[0])

if args.algorithm == "TD3":
    agent = TD3(state_dim, action_dim, max_action)
elif args.algorithm == "PPO":
    agent = PPO(state_dim, action_dim, 3e-4, 3e-4, 0.99, 10, 0.2, True)
    
agent.load('td3_model')

while not terminated:
    # agent is your own imported agent class
    # action = agent.act(state)
    action = np.random.uniform(-1, 1)
    # next_state is given as: [storage_level, price, hour, day]
    next_state, reward, terminated = environment.step(action)
    state = next_state
    aggregate_reward += reward

print("Action:", action)
print("Next state:", next_state)
print("Reward:", reward)
print('Total reward:', aggregate_reward)
