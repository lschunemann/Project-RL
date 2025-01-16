from env import DataCenterEnv
import numpy as np
import argparse
from agent import VPG, ActorCritic#TD3, PPO
import numpy as np
# from utils import run_episodes, sample_episode
from utils import train_agent, train_ppo

args = argparse.ArgumentParser()
args.add_argument('--path', type=str, default='train.xlsx')
args.add_argument('--algorithm', type=str, default='PPO')
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

# if args.algorithm == "TD3":
#     agent = TD3(environment)#state_dim, action_dim, max_action)
#     agent.load('models/td3_model')
# elif args.algorithm == "PPO":
#     agent = PPO(environment)#state_dim, action_dim, 3e-4, 3e-4, 0.99, 10, 0.2, True) 
#     agent.load('models/ppo_model')
if args.algorithm == "VPG":
    agent = VPG(state_dim, 128, action_dim)
    try:
        agent.load('models/vpg_model')
    except FileNotFoundError:
        # run_episodes(environment, agent, 1000, 0.99, 1e-3, sample_episode)
        train_agent(environment, agent, 1000, 0.99, 1e-3)
        agent.save('models/vpg_model')
elif args.algorithm == "AC":
    agent = ActorCritic(state_dim, 128, action_dim)
    try:
        agent.load('models/ac_model')
    except FileNotFoundError:
        # run_episodes(environment, agent, 1000, 0.99, 1e-3, sample_episode)
        train_agent(environment, agent, 1000, 0.99, 1e-3)
        agent.save('models/ac_model')
elif args.algorithm == "PPO":
    agent = ActorCritic(state_dim, 128, action_dim)
    try:
        agent.load('models/ppo_model')
    except FileNotFoundError:
        # run_episodes(environment, agent, 1000, 0.99, 1e-3, sample_episode)
        train_ppo(environment, agent, 1000, 2048, 64)
        agent.save('models/ac_model')
else:
    raise ValueError("Invalid algorithm")

while not terminated:
    # agent is your own imported agent class
    action = agent.act(state)
    # action = np.random.uniform(-1, 1)
    # next_state is given as: [storage_level, price, hour, day]
    next_state, reward, terminated = environment.step(action)
    state = next_state
    aggregate_reward += reward

    print("Action:", action)
    print("Next state:", next_state)
    print("Reward:", reward)
    print('Total reward:', aggregate_reward)
