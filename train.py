import argparse
import numpy as np
import torch
from env import DataCenterEnv
from agent import TD3, VPG, PPO#, SAC
from utils import ReplayBuffer  # Ensure you have a ReplayBuffer class implemented

# Training function
def train(path_to_dataset, model_save_path, max_timesteps=1000000, batch_size=256, eval_freq=5000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    np.random.seed(0)
    torch.manual_seed(0)

    # Initialize environment and agent
    env = DataCenterEnv(path_to_dataset)
    state_dim = 4  # [storage_level, price, hour, day]
    action_dim = env.continuous_action_space.shape[0]
    max_action = float(env.continuous_action_space.high[0])

    # Initialize policy
    if args.policy == "VPG":
        agent = VPG(state_dim, 64, action_dim).to(device)
    elif args.policy == "TD3":
        agent = TD3(state_dim, action_dim, max_action).to(device)
    elif args.policy == "PPO":
        agent = PPO(state_dim, action_dim).to(device)
    else:
        raise ValueError(f"Unknown policy type: {args.policy}")
    
    replay_buffer = ReplayBuffer(state_dim, action_dim)
    
    state = env.reset()
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0

    for t in range(max_timesteps):
        episode_timesteps += 1

        # Select action with exploration noise
        if t < 25000:
            action = np.random.uniform(low=-1, high=1, size=action_dim)
        else:
            action = agent.select_action(np.array(state))
            action += np.random.normal(0, max_action * 0.1, size=action_dim)
            action = action.clip(-max_action, max_action)

        # Step the environment
        next_state, reward, done = env.step(action)
        done_bool = float(done) if episode_timesteps < 1000 else 0

        replay_buffer.add(state, action, next_state, reward, done_bool)
        state = next_state
        episode_reward += reward

        # Train agent
        if t >= batch_size:
            if args.policy == "TR3":
                agent.train(replay_buffer, batch_size)
            else:
                agent.train(True)

        if done:
            print(f"Episode {episode_num} ended with reward: {episode_reward}")
            state = env.reset()
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1

        # Save model
        if t % eval_freq == 0:
            print(f"Saving model at timestep {t}")
            agent.save(model_save_path + str(args.policy) +"_"+ str(t))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="train.xlsx", help="Path to training dataset")
    parser.add_argument("--policy", type=str, default="TD3", help="Policy type (VPG, TD3, PPO)")
    parser.add_argument("--save_path", type=str, default=f"models/", help="Path to save the model")
    parser.add_argument("--max_timesteps", type=int, default=1000000, help="Max training timesteps")
    args = parser.parse_args()

    train(args.path, args.save_path, max_timesteps=args.max_timesteps)
