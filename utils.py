import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    returns = np.zeros_like(rewards.cpu().numpy())
    G = 0
    for i in reversed(range(len(rewards))):
        G = rewards[i].cpu().item() + discount_factor * G
        returns[i] = G
    return torch.from_numpy(returns).to(device)

def sample_episode(env, policy):
    states_list, actions_list, rewards_list, dones_list = [], [], [], []
    
    state = env.reset()
    done = False
    
    while not done:
        state_tensor = torch.FloatTensor(state).to(device)
        action = policy.act(state_tensor)
        
        next_state, reward, done = env.step(action)
        
        states_list.append(state)
        actions_list.append(action)
        rewards_list.append(reward)
        dones_list.append(done)
        
        state = next_state
    
    # Convert to numpy arrays first
    states_np = np.array(states_list, dtype=np.float32)
    actions_np = np.array(actions_list, dtype=np.float32)
    rewards_np = np.array(rewards_list, dtype=np.float32)
    dones_np = np.array(dones_list, dtype=np.float32)
    
    # Convert to tensors on device
    states = torch.from_numpy(states_np).to(device)
    actions = torch.from_numpy(actions_np).unsqueeze(-1).to(device)
    rewards = torch.from_numpy(rewards_np).to(device)
    dones = torch.from_numpy(dones_np).to(device)
    
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


def collect_rollouts(env, policy, num_steps):
    states = []
    actions = []
    rewards = []
    dones = []
    values = []
    log_probs = []
    
    state = env.reset()
    episode_rewards = []
    
    for _ in range(num_steps):
        state_tensor = torch.FloatTensor(state).to(device)
        
        with torch.no_grad():
            dist = policy.get_probs(state_tensor)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            _, value = policy.forward(state_tensor)
        
        next_state, reward, done = env.step(action.cpu().numpy())
        
        states.append(state_tensor)
        actions.append(action)
        rewards.append(reward)
        dones.append(done)
        values.append(value.squeeze())
        log_probs.append(log_prob)
        
        state = next_state
        episode_rewards.append(reward)
        
        if done:
            state = env.reset()
            episode_rewards = []
    
    # Convert lists to tensors
    states = torch.stack(states)
    actions = torch.stack(actions)
    rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
    dones = torch.tensor(dones, dtype=torch.float32).to(device)
    values = torch.stack(values)
    log_probs = torch.stack(log_probs)
    
    return states, actions, rewards, dones, values, log_probs

def compute_gae(rewards, values, dones, next_value, gamma, lam):
    advantages = torch.zeros_like(rewards).to(device)
    last_gae = 0
    
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_non_terminal = 1.0 - dones[t]
            next_values = next_value
        else:
            next_non_terminal = 1.0 - dones[t]
            next_values = values[t + 1]
        
        delta = rewards[t] + gamma * next_values * next_non_terminal - values[t]
        advantages[t] = last_gae = delta + gamma * lam * next_non_terminal * last_gae
    
    returns = advantages + values
    return returns, advantages


def train_ppo(env, policy, num_epochs, steps_per_epoch, batch_size, clip_ratio=0.2):
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4, eps=1e-5)
    
    # PPO specific parameters
    gamma = 0.99
    lam = 0.95
    target_kl = 0.01
    value_loss_coef = 0.5
    entropy_coef = 0.01
    max_grad_norm = 0.5
    
    for epoch in range(num_epochs):
        # Collect rollouts
        with torch.no_grad():
            states, actions, rewards, dones, values, old_log_probs = collect_rollouts(env, policy, steps_per_epoch)
            
            # Scale rewards
            rewards = rewards / 1e6
            
            # Compute last value for GAE
            last_state = torch.FloatTensor(env.reset()).to(device)
            _, last_value = policy.forward(last_state)
            last_value = last_value.squeeze()
            
            # Compute returns and advantages
            returns, advantages = compute_gae(rewards, values, dones, last_value, gamma, lam)
            
            # Normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update
        for _ in range(4):  # Number of optimization epochs
            # Generate random permutation of indices
            indices = torch.randperm(steps_per_epoch)
            
            # Mini-batch updates
            for start in range(0, steps_per_epoch, batch_size):
                end = start + batch_size
                batch_indices = indices[start:end]
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                
                # Compute current policy distributions
                new_log_probs, entropy, values = policy.evaluate_actions(batch_states, batch_actions)
                
                # Compute ratio and clipped ratio
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                clipped_ratio = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio)
                
                # Compute policy loss
                policy_loss = -torch.min(
                    ratio * batch_advantages,
                    clipped_ratio * batch_advantages
                ).mean()
                
                # Compute value loss
                value_loss = torch.nn.functional.mse_loss(values, batch_returns)
                
                # Compute entropy loss
                entropy_loss = -entropy.mean()
                
                # Total loss
                total_loss = (
                    policy_loss +
                    value_loss_coef * value_loss +
                    entropy_coef * entropy_loss
                )
                
                # Optimization step
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
                optimizer.step()
                
                # Early stopping based on KL divergence
                with torch.no_grad():
                    kl = (batch_old_log_probs - new_log_probs).mean()
                    if kl > 1.5 * target_kl:
                        break
        
        # Logging
        if epoch % 10 == 0:
            print(f"Epoch {epoch}")
            print(f"Average Episode Reward: {rewards.sum().item():.2f}")
            print(f"Policy Loss: {policy_loss.item():.4f}")
            print(f"Value Loss: {value_loss.item():.4f}")
            print(f"Entropy: {entropy.mean().item():.4f}")
            print(f"KL Divergence: {kl.item():.4f}")
            print("----------------------------------------")
