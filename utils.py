import numpy as np
import torch

class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))

    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(torch.device("cuda" if torch.cuda.is_available() else "cpu")),
            torch.FloatTensor(self.action[ind]).to(torch.device("cuda" if torch.cuda.is_available() else "cpu")),
            torch.FloatTensor(self.next_state[ind]).to(torch.device("cuda" if torch.cuda.is_available() else "cpu")),
            torch.FloatTensor(self.reward[ind]).to(torch.device("cuda" if torch.cuda.is_available() else "cpu")),
            torch.FloatTensor(self.not_done[ind]).to(torch.device("cuda" if torch.cuda.is_available() else "cpu")),
        )


def normalize_tensor(tensor):
    if len(tensor) > 1:
        return (tensor - tensor.mean()) / (tensor.std() + 1e-8)
    return tensor

def compute_returns(rewards, discount_factor=0.99):
    returns = np.zeros_like(rewards.numpy())
    G = 0
    for i in reversed(range(len(rewards))):
        G = rewards[i].item() + discount_factor * G
        returns[i] = G
    return torch.from_numpy(returns)

def sample_episode(env, policy):
    states_list, actions_list, rewards_list, dones_list = [], [], [], []
    
    state = env.reset()
    done = False
    
    while not done:
        state_tensor = torch.FloatTensor(state)
        action = policy.act(state_tensor)
        
        next_state, reward, done = env.step(action)
        
        states_list.append(state)
        actions_list.append(action)
        rewards_list.append(reward)
        dones_list.append(done)
        
        state = next_state
    
    # Convert to tensors
    states = torch.FloatTensor(np.array(states_list))
    actions = torch.FloatTensor(np.array(actions_list)).unsqueeze(-1)
    rewards = torch.FloatTensor(np.array(rewards_list))
    dones = torch.FloatTensor(np.array(dones_list))
    
    return states, actions, rewards, dones

def train_agent(env, agent, num_episodes, lr, discount_factor):
    optimizer = torch.optim.Adam(agent.parameters(), lr=lr)
    episode_rewards = []
    running_reward = 0
    best_reward = float('-inf')
    patience = 100  # Increased patience
    no_improvement = 0
    best_state_dict = None
    
    for i in range(num_episodes):
        # Add exploration noise
        with torch.no_grad():
            exploration_std = max(1.0 * (1 - i/num_episodes), 0.3)  # Linear decay
            agent.log_std.data = torch.log(torch.ones_like(agent.log_std) * exploration_std)
        
        episode = sample_episode(env, agent)
        _, _, rewards, _ = episode
        total_reward = rewards.sum().item()
        
        running_reward = 0.05 * total_reward + (1 - 0.05) * running_reward
        
        loss, metrics = agent.compute_loss(episode, discount_factor)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=0.5)
        optimizer.step()
        
        episode_rewards.append(total_reward)
        
        # Save best model
        if total_reward > best_reward:
            best_reward = total_reward
            best_state_dict = {k: v.cpu().clone() for k, v in agent.state_dict().items()}
            no_improvement = 0
        else:
            no_improvement += 1
        
        if no_improvement >= patience:
            print(f"Early stopping at episode {i}")
            break
        
        if i % 10 == 0:
            metrics_str = ', '.join([f"{k}: {v:.4f}" for k, v in metrics.items()])
            print(f"Episode {i}, Reward: {total_reward:.2f}, Running Reward: {running_reward:.2f}, {metrics_str}")
    
    # Load best model
    if best_state_dict is not None:
        agent.load_state_dict(best_state_dict)
    
    return episode_rewards