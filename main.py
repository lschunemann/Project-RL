from env import DataCenterEnv
import numpy as np
import argparse
from agent import VPG, ActorCritic, PPO, Random, Sell_Max, Buy_Max, Do_nothing, Daily_Requirement, ImprovedQAgent, Double_Q
import numpy as np
# from utils import run_episodes, sample_episode
from utils import train_agent, train_ppo
from pathlib import Path
from QAgent import QAgent, ImprovedQAgent

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

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
    if (Path.cwd() / 'models' / 'vpg_model.pth').exists():
        agent.load('models/vpg_model.pth')
    else:
        train_agent(environment, agent, 1000, 0.99, 1e-3)
        agent.save('models/vpg_model.pth')
elif args.algorithm == "AC":
    agent = ActorCritic(state_dim, 128, action_dim)
    if (Path.cwd() / 'models' / 'ac_model.pth').exists():
        agent.load('models/ac_model.pth')
    else:
        train_agent(environment, agent, 1000, 0.99, 1e-3)
        agent.save('models/ac_model.pth')
elif args.algorithm == "PPO":
    agent = PPO(state_dim, 128, action_dim)
    # if (Path.cwd() / 'models' / 'ppo_model.pth').exists():
    #     agent.load('models/ppo_model.pth')
    # else:
    train_ppo(environment, agent, 1000, 2048, 64)
    agent.save('models/ppo_model.pth')
elif args.algorithm == 'random':
    agent = Random()
elif args.algorithm == 'do_nothing':
    agent = Do_nothing()
elif args.algorithm == 'sell_max':
    agent = Sell_Max()
elif args.algorithm == 'buy_max':
    agent = Buy_Max()
elif args.algorithm == 'daily_req':
    agent = Daily_Requirement()
elif args.algorithm == 'baseline_q':
    agent = QAgent(env=environment)
    agent.train(episodes=1000, bias_correction=False)
    agent.save_q_table('results/baseline_q_table.npy')
elif args.algorithm == 'improved_baseline_q':
    agent = ImprovedQAgent(env=environment)
    agent.train(episodes=1000, bias_correction=False)
    agent.save_q_table('results/improved_baseline_q_table.npy')
elif args.algorithm == 'tabular_double_q':
    agent = Double_Q(environment,
                      target_update_freq=100000, 
                      alpha=0.05,
                      gamma=0.99,
                      epsilon=1.0,
                      epsilon_decay=0.998,
                      epsilon_min=0.05)
    agent.train(episodes=5000)
    # agent.save_q_table('models/double')
else:
    raise ValueError("Invalid algorithm")

if args.path == 'train.xlsx':
    print(f"Training policy: {args.algorithm}")
else:
    print(f"Evaluating policy: {args.algorithm}")

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
