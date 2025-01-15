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


def compute_reinforce_loss(policy, episode, discount_factor=0.99):
    states, actions, rewards, _ = episode
    
    # Calculate discounted returns
    returns = []
    G = 0
    for r in reversed(rewards.numpy()):
        G = r + discount_factor * G
        returns.insert(0, G)
    returns = torch.tensor(returns, dtype=torch.float32)
    
    # Normalize returns
    if len(returns) > 1:  # Only normalize if we have more than one return
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    
    # Get action log probabilities
    dist = policy.get_probs(states)
    log_probs = dist.log_prob(actions.squeeze())
    
    # Calculate policy loss
    policy_loss = -(log_probs * returns).mean()
    
    # Add entropy term for exploration
    entropy = dist.entropy().mean()
    loss = policy_loss - 0.01 * entropy  # Small entropy coefficient
    
    return loss


def sample_episode(env, policy):
    states = []
    actions = []
    rewards = []
    dones = []
    
    state = env.reset()
    done = False
    episode_reward = 0
    
    while not done:
        state_tensor = torch.FloatTensor(state)
        action = policy.act(state_tensor)
        
        next_state, reward, done = env.step(action)
        
        # Store experience
        states.append(state_tensor)
        actions.append(torch.FloatTensor([action]))
        rewards.append(reward)
        dones.append(done)
        
        state = next_state
        episode_reward += reward
    
    # Convert to tensors
    states = torch.stack(states)
    actions = torch.stack(actions)
    rewards = torch.tensor(rewards, dtype=torch.float32)
    dones = torch.tensor(dones, dtype=torch.float32)
    
    return states, actions, rewards, dones


def run_episodes(env, policy, num_episodes, discount_factor, lr, sampling_func=sample_episode):
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    episode_rewards = []
    
    for i in range(num_episodes):
        # Collect episode data
        episode = sampling_func(env, policy)
        _, _, rewards, _ = episode
        total_reward = rewards.sum().item()
        
        # Compute loss and update policy
        loss = compute_reinforce_loss(policy, episode, discount_factor)
        
        optimizer.zero_grad()
        loss.backward()
        # Clip gradients for stability
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.5)
        optimizer.step()
        
        episode_rewards.append(total_reward)
        
        if i % 10 == 0:
            print(f"Episode {i}, Total Reward: {total_reward:.2f}, Loss: {loss.item():.4f}")
    
    return episode_rewards
