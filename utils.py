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


def collect_rollouts(env, policy, num_steps, monitor):
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

        # Add to monitor
        monitor.add_step(state, action.cpu().numpy(), reward, value.cpu().numpy())
        
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
            monitor.reset_episode()
    
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

    monitor = PPOMonitor()
    
    for epoch in range(num_epochs):
        # Collect rollouts
        with torch.no_grad():
            states, actions, rewards, dones, values, old_log_probs = collect_rollouts(env, policy, steps_per_epoch, monitor)
            
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
        # Print monitoring information
        if epoch % 10 == 0:
            print_episode_summary(monitor, epoch)
            visualize_episode(monitor)
            
            # Additional policy checks
            test_states = torch.FloatTensor([
                [0, 50, 1, 1],    # Low storage, medium price, start of day
                [100, 50, 12, 1],  # High storage, medium price, middle of day
                [60, 100, 23, 1],  # Medium storage, high price, end of day
                [20, 10, 12, 1],   # Low storage, low price, middle of day
            ]).to(device)
            
            with torch.no_grad():
                dist = policy.get_probs(test_states)
                actions = dist.sample()
                print("\nPolicy Behavior Check:")
                print("State (Storage, Price, Hour, Day) -> Action")
                for state, action in zip(test_states, actions):
                    print(f"[{state[0]:3.0f}, {state[1]:3.0f}, {state[2]:2.0f}, {state[3]:2.0f}] -> {action.item():6.3f}")


def print_episode_summary(monitor, epoch):
    stats = monitor.get_statistics()
    if not stats:
        return
    
    print(f"\nEpoch {epoch} Summary:")
    print("="*50)
    print(f"Episode Reward: {stats['episode_reward']:,.2f}")
    print(f"Episode Length: {stats['episode_length']}")
    print("\nAction Statistics:")
    print(f"  Mean Action: {stats['mean_action']:.3f}")
    print(f"  Action Std: {stats['std_action']:.3f}")
    print(f"  Buy Percentage: {stats['buy_percentage']:.1f}%")
    print(f"  Sell Percentage: {stats['sell_percentage']:.1f}%")
    print("\nStorage Statistics:")
    print(f"  Mean Storage: {stats['mean_storage']:.2f}")
    print(f"  Storage Range: [{stats['min_storage']:.2f}, {stats['max_storage']:.2f}]")
    print("\nPrice Statistics:")
    print(f"  Mean Price: {stats['mean_price']:.2f}")
    print(f"  Price-Action Correlation: {stats['price_action_correlation']:.3f}")
    
    # Print action distribution histogram
    actions = np.array(monitor.current_episode['actions'])
    print("\nAction Distribution:")
    hist, bins = np.histogram(actions, bins=10, range=(-1, 1))
    max_height = max(hist)
    for i in range(len(hist)):
        bar_height = int((hist[i] / max_height) * 20)
        print(f"{bins[i]:6.2f} | {'#' * bar_height}")

def visualize_episode(monitor):
    """Create and save plots for the episode"""
    import matplotlib.pyplot as plt
    
    # Create figure with subplots
    fig, axes = plt.subplots(3, 1, figsize=(12, 12))
    
    # Plot actions and prices
    ax1 = axes[0]
    actions = monitor.current_episode['actions']
    prices = monitor.current_episode['prices']
    timesteps = range(len(actions))
    
    ax1.plot(timesteps, actions, label='Actions', color='blue')
    ax1_twin = ax1.twinx()
    ax1_twin.plot(timesteps, prices, label='Prices', color='red', alpha=0.5)
    ax1.set_ylabel('Action Value')
    ax1_twin.set_ylabel('Price')
    ax1.set_title('Actions and Prices over Time')
    
    # Plot storage levels
    ax2 = axes[1]
    storage = monitor.current_episode['storage']
    ax2.plot(timesteps, storage, label='Storage Level', color='green')
    ax2.axhline(y=120, color='r', linestyle='--', label='Daily Requirement')
    ax2.set_ylabel('Storage Level')
    ax2.set_title('Storage Level over Time')
    ax2.legend()
    
    # Plot cumulative rewards
    ax3 = axes[2]
    rewards = np.cumsum(monitor.current_episode['rewards'])
    ax3.plot(timesteps, rewards, label='Cumulative Reward', color='purple')
    ax3.set_ylabel('Cumulative Reward')
    ax3.set_xlabel('Timestep')
    ax3.set_title('Cumulative Reward over Time')
    
    plt.tight_layout()
    plt.savefig(f'episode_summary_{len(monitor.episode_rewards)}.png')
    plt.close()

class PPOMonitor:
    def __init__(self):
        self.episode_rewards = []
        self.episode_lengths = []
        self.action_history = []
        self.state_history = []
        self.value_estimates = []
        self.storage_levels = []
        self.prices = []
        self.current_episode = {
            'rewards': [],
            'actions': [],
            'states': [],
            'values': [],
            'storage': [],
            'prices': []
        }
    
    def reset_episode(self):
        if len(self.current_episode['rewards']) > 0:
            self.episode_rewards.append(sum(self.current_episode['rewards']))
            self.episode_lengths.append(len(self.current_episode['rewards']))
            
        self.current_episode = {
            'rewards': [],
            'actions': [],
            'states': [],
            'values': [],
            'storage': [],
            'prices': []
        }
    
    def add_step(self, state, action, reward, value):
        self.current_episode['rewards'].append(reward)
        self.current_episode['actions'].append(action)
        self.current_episode['states'].append(state)
        self.current_episode['values'].append(value)
        self.current_episode['storage'].append(state[0])  # Assuming storage is first state component
        self.current_episode['prices'].append(state[1])   # Assuming price is second state component
    
    def get_statistics(self):
        if len(self.current_episode['actions']) == 0:
            return {}
        
        actions = np.array(self.current_episode['actions'])
        storage = np.array(self.current_episode['storage'])
        prices = np.array(self.current_episode['prices'])
        
        stats = {
            'mean_action': actions.mean(),
            'std_action': actions.std(),
            'max_action': actions.max(),
            'min_action': actions.min(),
            'buy_percentage': (actions > 0).mean() * 100,
            'sell_percentage': (actions < 0).mean() * 100,
            'mean_storage': storage.mean(),
            'max_storage': storage.max(),
            'min_storage': storage.min(),
            'mean_price': prices.mean(),
            'price_action_correlation': np.corrcoef(prices, actions)[0,1],
            'episode_reward': sum(self.current_episode['rewards']),
            'episode_length': len(self.current_episode['rewards'])
        }
        return stats