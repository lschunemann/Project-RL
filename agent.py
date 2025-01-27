import torch
import torch.nn as nn
import torch.distributions as dist
import torch.nn.functional as F
import numpy as np
from utils import normalize_tensor, compute_returns
from QAgent import QAgent
import time
import os
import json
import matplotlib.pyplot as plt

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")


class BasePolicy(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(BasePolicy, self).__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            module.bias.data.zero_()
    
    @torch.no_grad()
    def act(self, state):
        dist = self.get_probs(state)
        action = dist.sample()
        return action.cpu().numpy()
    
    def compute_loss(self, episode, discount_factor=0.99):
        raise NotImplementedError
        
    def get_probs(self, state):
        raise NotImplementedError
    
    def load(self, path):
        self.load_state_dict(torch.load(path))
        print("Model loaded from {}".format(path))

    def save(self, path):
        torch.save(self.state_dict(), path)
        print("Model saved to {}".format(path))


class VPG(BasePolicy):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(VPG, self).__init__(state_dim, hidden_dim, action_dim)
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        
        self.apply(self._init_weights)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = torch.tanh(self.fc3(x))
        return mean
    
    def get_probs(self, state):
        action_mean = self.forward(state)
        std = torch.exp(self.log_std).clamp(min=1e-3, max=1)
        dist = torch.distributions.Normal(action_mean, std)
        return dist
    
    def compute_loss(self, episode, discount_factor=0.99):
        states, actions, rewards, _ = episode
        
        # Calculate returns
        returns = compute_returns(rewards, discount_factor)
        returns = normalize_tensor(returns)
        
        # Get action log probabilities
        dist = self.get_probs(states)
        log_probs = dist.log_prob(actions.squeeze())
        
        # Calculate policy loss
        policy_loss = -(log_probs * returns).mean()
        
        # Add entropy term
        entropy = dist.entropy().mean()
        loss = policy_loss - 0.01 * entropy
        
        return loss, {'policy_loss': policy_loss.item(), 'entropy': entropy.item()}

class ActorCritic(BasePolicy):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(ActorCritic, self).__init__(state_dim, hidden_dim, action_dim)
        
        self.state_normalizer = RunningNormalizer(state_dim).to(device)
        self.return_normalizer = RunningNormalizer(1).to(device)
        
        # Shared features
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        # Actor network (mean)
        self.actor_mean = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim)
        )
        
        # Fixed minimum std to prevent collapse
        self.min_std = 0.3
        self.max_std = 1.0
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        
        # Critic network
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self.apply(self._init_weights)
        
        # Initialize log_std
        with torch.no_grad():
            self.log_std.fill_(np.log(0.6))

        self.to(device)
    
    def forward(self, x):
        x = self.state_normalizer(x.to(device))
        shared_features = self.shared(x)
        action_mean = torch.tanh(self.actor_mean(shared_features))
        value = self.critic(shared_features)
        return action_mean, value
    
    def get_probs(self, state):
        action_mean, _ = self.forward(state)
        # Ensure std stays within bounds
        std = torch.exp(self.log_std).clamp(min=self.min_std, max=self.max_std)
        dist = torch.distributions.Normal(action_mean, std)
        return dist
    
    def compute_loss(self, episode, discount_factor=0.99):
        states, actions, rewards, _ = episode
        
        # Scale rewards
        rewards = rewards / 1e6
        
        # Calculate returns and values
        returns = compute_returns(rewards, discount_factor)
        self.return_normalizer.update(returns)
        normalized_returns = self.return_normalizer(returns)
        
        _, values = self.forward(states)
        values = values.squeeze()
        advantages = normalized_returns - values.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Get action distribution
        dist = self.get_probs(states)
        log_probs = dist.log_prob(actions.squeeze())
        
        # Policy loss
        policy_loss = -(log_probs * advantages).mean()
        
        # Value loss
        value_loss = F.smooth_l1_loss(values, normalized_returns)
        
        # Entropy loss with minimum threshold
        entropy = dist.entropy().mean()
        entropy_loss = -0.1 * entropy  # Increased entropy coefficient
        
        # Std loss to prevent collapse
        std = torch.exp(self.log_std)
        std_loss = 0.01 * F.mse_loss(std, torch.ones_like(std) * 0.6)
        
        # Combined loss
        total_loss = policy_loss + 0.5 * value_loss + entropy_loss + std_loss
        
        return total_loss, {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.item(),
            'mean_value': values.mean().item(),
            'mean_return': returns.mean().item(),
            'std': std.mean().item(),
            'advantage': advantages.mean().item()
        }

class RunningNormalizer:
    def __init__(self, shape):
        self.shape = shape
        self.mean = torch.zeros(shape).to(device)
        self.std = torch.ones(shape).to(device)
        self.count = 0
        
    def update(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float().to(device)
        
        batch_mean = x.mean(0)
        batch_std = x.std(0)
        batch_count = x.shape[0]
        
        delta = batch_mean - self.mean
        self.mean = self.mean + delta * batch_count / (self.count + batch_count)
        
        self.std = torch.sqrt(
            (self.std ** 2 * self.count + batch_std ** 2 * batch_count + 
             delta ** 2 * self.count * batch_count / (self.count + batch_count)) / 
            (self.count + batch_count)
        )
        
        self.count += batch_count
        
    def __call__(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float().to(device)
        return (x - self.mean) / (self.std + 1e-8)

class PPO(BasePolicy):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PPO, self).__init__(state_dim, hidden_dim, action_dim)
        
        self.state_normalizer = RunningNormalizer(state_dim)
        self.return_normalizer = RunningNormalizer(1)
        
        # Shared features
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        # Actor network (mean)
        self.actor_mean = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim),
            nn.Tanh()  # Force output to [-1, 1]
        )
        
        # Fixed log_std parameter
        initial_std = 0.5
        self.log_std = nn.Parameter(torch.ones(action_dim) * np.log(initial_std))
        
        # Critic network
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self.apply(self._init_weights)
        self.to(device)

    def forward(self, x):
        x = self.state_normalizer(x)
        shared_features = self.shared(x)
        action_mean = self.actor_mean(shared_features)  # Already in [-1, 1]
        value = self.critic(shared_features)
        return action_mean, value
    
    def get_probs(self, state):
        action_mean, _ = self.forward(state)
        std = torch.exp(self.log_std).clamp(min=0.1, max=0.5)  # Tighter std bounds
        dist = torch.distributions.Normal(action_mean, std)
        return dist
    
    @torch.no_grad()
    def act(self, state):
        dist = self.get_probs(state)
        action = dist.sample()
        # Ensure actions are in [-1, 1]
        return torch.clamp(action, -1, 1).cpu().numpy()
    
class Random(BasePolicy):
    def __init__(self):
        return
    #     # super().__init__()
    #     self.to(device)

    def act(self, state):
        return np.random.uniform(-1,1)
    
class Do_nothing(BasePolicy):
    def __init__(self):
        return
    #     # super().__init__()
    #     self.to(device)

    def act(self, state):
        return 0
    
class Sell_Max(BasePolicy):
    def __init__(self):
        return
    #     # super().__init__()
    #     self.to(device)

    def act(self, state):
        return -1
    
class Buy_Max(BasePolicy):
    def __init__(self):
        return
    #     # super().__init__()
    #     self.to(device)

    def act(self, state):
        return 1
    
class Daily_Requirement(BasePolicy):
    def __init__(self):
        return
    #     # super().__init__()
    #     self.to(device)

    def act(self, state):
        if state[0] < 120:
            return 1
        elif state[1] > 120:
            return -((state[1] - 120) / 10 ) % 10
        else: return 0
    

class ImprovedQAgent(QAgent):
    def __init__(self, env, **kwargs):
        super().__init__(env, **kwargs)
        
        # Improved state discretization
        # self.storage_bins = np.array([0, 30, 60, 90, 120, 170, 220, 270])  # More granular storage bins
        # self.price_bins = np.array([0, 20, 40, 60, 80, 100, 150, 200])     # Add price discretization
        # self.hour_bins = np.arange(1, 25)  # Keep hourly granularity
        
        # Initialize two Q-tables for Double Q-learning
        self.q1 = np.zeros((
            len(self.storage_bins) - 1,    # Storage bins
            # len(self.price_bins) - 1,      # Price bins
            len(self.hour_bins),           # Hour bins
            len(self.action_space)         # Actions
        ))
        
        self.q2 = np.zeros((
            len(self.storage_bins) - 1,
            # len(self.price_bins) - 1,
            len(self.hour_bins),
            len(self.action_space)
        ))
        
        # Price statistics for better reward shaping
        self.price_history = []
        self.price_ma = None
        self.price_std = None
        
        # Time-based exploration
        self.time_exploration = np.ones((24,)) * self.epsilon  # Exploration rate per hour
        
    # def discretize(self, observation):
    #     """Enhanced discretization including price."""
    #     storage, price, hour, _ = observation
        
    #     # Discretize all relevant components
    #     storage_idx = np.digitize(storage, self.storage_bins) - 1
    #     price_idx = np.digitize(price, self.price_bins) - 1
    #     hour_idx = int(hour) - 1
        
    #     # Clip indices to avoid out-of-bound errors
    #     storage_idx = np.clip(storage_idx, 0, len(self.storage_bins) - 2)
    #     price_idx = np.clip(price_idx, 0, len(self.price_bins) - 2)
        
    #     return (storage_idx, price_idx, hour_idx)
    
    def update_price_statistics(self, price):
        """Update price statistics for adaptive reward shaping."""
        self.price_history.append(price)
        if len(self.price_history) > 24:  # Use last 24 hours
            self.price_history.pop(0)
        
        self.price_ma = np.mean(self.price_history)
        self.price_std = np.std(self.price_history) if len(self.price_history) > 1 else 1.0
    
    def reward_shaping(self, reward, action, storage_level, price, hour):
        """Enhanced reward shaping considering time and price dynamics."""
        shaped_reward = reward
        
        # Storage management component
        storage_factor = 1.0 - (storage_level / self.env.daily_energy_demand)
        
        # Price component using z-score
        if self.price_ma is not None:
            price_zscore = (price - self.price_ma) / (self.price_std + 1e-6)
        else:
            price_zscore = 0
            
        # Time pressure component
        hours_left = 24 - hour
        time_pressure = 1.0 / (hours_left + 1)  # Increased urgency as day ends
        
        if action > 0:  # Buying
            # Discourage buying at high prices, encourage at low prices
            shaped_reward += -10 * price_zscore * action
            # Encourage buying when storage is low
            shaped_reward += 5 * storage_factor * action
            # Increase urgency when time is running out
            shaped_reward += 15 * time_pressure * storage_factor
            
        elif action < 0:  # Selling
            # Encourage selling at high prices
            shaped_reward += 10 * price_zscore * abs(action)
            # Discourage selling when storage is low
            shaped_reward -= 5 * storage_factor * abs(action)
            
        return shaped_reward
    
    def choose_action(self, state, greedy=False):
        """Enhanced action selection using average of both Q-tables."""
        hour = state[1]
        
        if not greedy and np.random.random() < self.time_exploration[hour]:
            return np.random.choice(len(self.action_space))
        else:
            # Use average of both Q-tables for action selection
            q_values = (self.q1[state] + self.q2[state]) / 2
            return np.argmax(q_values)
    
    def update_exploration(self, hour, reward):
        """Update hour-specific exploration rates based on performance."""
        if reward > 0:
            # Reduce exploration more for successful hours
            self.time_exploration[hour] *= self.epsilon_decay
        else:
            # Reduce exploration less for unsuccessful hours
            self.time_exploration[hour] *= np.sqrt(self.epsilon_decay)
            
        self.time_exploration[hour] = max(self.time_exploration[hour], self.epsilon_min)
    

    def train(self, episodes=1000, verbose=True):
        """Enhanced training with proper Double Q-learning."""
        print(f"Training for {episodes} episodes...")
        start = time.time()
        
        for episode in range(1, episodes+1):
            state = self.discretize(self.env.reset())
            total_reward = 0
            step = 0
            done = False
            
            while not done:
                step += 1
                action_idx = self.choose_action(state)
                action = self.action_space[action_idx]
                
                next_obs, reward, done = self.env.step([action])
                next_state = self.discretize(next_obs)
                
                # Update price statistics
                self.update_price_statistics(next_obs[1])
                
                # Enhanced reward shaping
                shaped_reward = self.reward_shaping(
                    reward, action, next_obs[0], next_obs[1], next_obs[2])
                
                # Randomly choose which Q-table to update
                if np.random.random() < 0.5:
                    # Update Q1
                    best_action = np.argmax(self.q1[next_state])
                    td_target = shaped_reward + self.gamma * self.q2[next_state + (best_action,)]
                    td_error = td_target - self.q1[state + (action_idx,)]
                    effective_alpha = self.alpha / (1.0 + abs(td_error))
                    self.q1[state + (action_idx,)] += effective_alpha * td_error
                else:
                    # Update Q2
                    best_action = np.argmax(self.q2[next_state])
                    td_target = shaped_reward + self.gamma * self.q1[next_state + (best_action,)]
                    td_error = td_target - self.q2[state + (action_idx,)]
                    effective_alpha = self.alpha / (1.0 + abs(td_error))
                    self.q2[state + (action_idx,)] += effective_alpha * td_error
                
                # Update exploration rate for the current hour
                self.update_exploration(state[1], reward)
                
                state = next_state
                total_reward += reward
                
                # Track metrics
                self.storage_levels.append(next_obs[0])
                self.actions.append(action)
                self.prices.append(next_obs[1])
            
            if verbose and episode % 10 == 0:
                # Calculate average Q-value difference for monitoring convergence
                q_diff = np.mean(np.abs(self.q1 - self.q2))
                print(f"Episode {episode}/{episodes}, "
                      f"Total Reward: {total_reward:.2f}, "
                      f"Avg Exploration: {np.mean(self.time_exploration):.3f}, "
                      f"Q-diff: {q_diff:.3f}")
                
        print(f"Training completed in {time.time()-start:.2f} seconds.")

    def save_q_table(self, path, verbose=False):
        """Save both Q-tables."""
        if not os.path.exists(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
        
        np.save(path + "_q1.npy", self.q1)
        np.save(path + "_q2.npy", self.q2)
        
        # Save hyperparameters
        hyperparameters_path = path + "_hyperparameters.json"
        with open(hyperparameters_path, "w") as f:
            json.dump(self.hyperparameters, f)
            
        if verbose:
            print(f"Q-tables saved to {path}_q1.npy and {path}_q2.npy")
            print(f"Hyperparameters saved to {hyperparameters_path}")
    
    def load_q_table(self, path, verbose=False):
        """Load both Q-tables."""
        self.q1 = np.load(path + "_q1.npy")
        self.q2 = np.load(path + "_q2.npy")
        
        if verbose:
            print(f"Q-tables loaded from {path}_q1.npy and {path}_q2.npy")
    
    def plot_q_differences(self):
        """Plot the differences between Q1 and Q2 to visualize convergence."""
        differences = np.abs(self.q1 - self.q2)
        
        plt.figure(figsize=(10, 6))
        plt.imshow(differences.mean(axis=(2,3)), cmap='hot', aspect='auto')
        plt.colorbar(label='Average Q-value difference')
        plt.title('Q1-Q2 Differences across State Space')
        plt.xlabel('Price bins')
        plt.ylabel('Storage bins')
        plt.show()


class Double_Q(QAgent):
    def __init__(self, environment, target_update_freq=100000, **kwargs):
        super().__init__(environment, **kwargs)

        # State discretization
        self.storage_bins = np.linspace(0, self.max_storage, 8)  # More bins for storage
        self.price_bins = np.percentile(environment.price_values.flatten(), 
                                      [0, 25, 50, 75, 90, 95, 97.5, 100])  # Price bins based on distribution
        self.hour_bins = np.arange(1, 25)

        self.q = np.zeros((
            len(self.storage_bins) - 1,    # Storage bins
            len(self.price_bins) - 1,      # Price bins
            len(self.hour_bins),           # Hour bins
            len(self.action_space)         # Actions
        ))
        
        # Target Q-table (updated periodically)
        self.q_target = np.zeros_like(self.q)

        # Target network update frequency (in steps)
        self.target_update_freq = target_update_freq
        self.steps_since_target_update = 0

        # Add save frequency
        self.save_freq = 50
        
        # Add reward tracking
        self.episode_rewards = []

        # Exploration parameters
        self.initial_epsilon = kwargs.get('epsilon', 1.0)
        self.epsilon = self.initial_epsilon
        self.epsilon_decay = kwargs.get('epsilon_decay', 0.995)
        self.epsilon_min = kwargs.get('epsilon_min', 0.01)

        # Price statistics
        all_prices = environment.price_values.flatten()
        self.min_price = np.min(all_prices)
        self.max_price = np.max(all_prices)
        self.price_mean = np.mean(all_prices)
        self.price_std = np.std(all_prices)
        
        # Learning parameters
        self.alpha = kwargs.get('alpha', 0.1)
        self.gamma = kwargs.get('gamma', 0.99)

        # Track best episode for early stopping
        self.best_reward = float('-inf')
        self.episodes_without_improvement = 0
        
        # Add visit counts for exploration
        self.state_visits = np.zeros_like(self.q)

    def plot_rewards(self):
        """Plot the reward history with moving average"""
        plt.figure(figsize=(12, 6))
        plt.plot(self.episode_rewards, alpha=0.6, label='Episode Reward')
        
        # Calculate and plot moving average
        window_size = 50
        if len(self.episode_rewards) > window_size:
            moving_avg = np.convolve(self.episode_rewards, 
                                   np.ones(window_size)/window_size, 
                                   mode='valid')
            plt.plot(range(window_size-1, len(self.episode_rewards)), 
                    moving_avg, 
                    'r', 
                    label=f'{window_size}-Episode Moving Average')
        
        plt.title('Training Rewards Over Time')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save plot
        plt.savefig('plots/reward_history.png')
        plt.close()

    def discretize(self, observation):
        """Enhanced discretization including price."""
        storage, price, hour, _ = observation
        
        # Discretize all relevant components
        storage_idx = np.digitize(storage, self.storage_bins) - 1
        price_idx = np.digitize(price, self.price_bins) - 1
        hour_idx = int(hour) - 1
        
        # Clip indices to avoid out-of-bound errors
        storage_idx = np.clip(storage_idx, 0, len(self.storage_bins) - 2)
        price_idx = np.clip(price_idx, 0, len(self.price_bins) - 2)
        
        return (storage_idx, price_idx, hour_idx)

    def update_target_network(self):
        """Update target Q-table with behavior Q-table values."""
        """Soft update of target network"""
        tau = 0.01  # Soft update parameter
        self.q_target = (1 - tau) * self.q_target + tau * self.q
        
    def get_exploration_bonus(self, state, action_idx):
        """UCB-style exploration bonus"""
        N = self.state_visits[state + (action_idx,)] + 1
        return np.sqrt(2 * np.log(self.total_steps + 1) / N)
    
    def choose_action(self, state, greedy=False):
        """Enhanced action selection with UCB exploration"""
        if greedy:
            return np.argmax(self.q[state])
        
        if np.random.random() < self.epsilon:
            # Epsilon-greedy exploration
            return np.random.choice(len(self.action_space))
        else:
            # UCB-based action selection
            ucb_values = self.q[state] + self.get_exploration_bonus(state, np.arange(len(self.action_space)))
            return np.argmax(ucb_values)
        
    def reward_shaping(self, reward, action, storage, price, hour):
        """Improved reward shaping focusing on cost minimization"""
        # Normalize storage relative to daily demand
        storage_ratio = storage / self.env.daily_energy_demand
        hours_left = 24 - hour
        
        # Normalize price
        price_normalized = (price - self.min_price) / (self.max_price - self.min_price)
        
        shaped_reward = reward  # Start with original reward
        
        # Storage management component
        if storage < self.env.daily_energy_demand and hours_left > 0:
            storage_urgency = (1 - storage_ratio) * (1 / (hours_left + 1))
            shaped_reward += storage_urgency * 1000  # Encourage maintaining sufficient storage
        
        # Price-based component
        if action > 0:  # Buying
            # Reward buying at low prices
            price_factor = 1 - price_normalized
            shaped_reward += price_factor * 500
        elif action < 0:  # Sellingß
            # Reward selling at high prices
            price_factor = price_normalized
            shaped_reward += price_factor * 500
        
        return shaped_reward / 1000  # Scale down the final reward
        
    def train(self, episodes=1000, verbose=True):
        print(f"Training for {episodes} episodes...")
        start = time.time()
        self.total_steps = 0
        best_reward = float('-inf')

        # Early stopping parameters
        patience = 50
        min_improvement = 1000
        
        # Initialize price statistics
        all_prices = self.env.price_values.flatten()
        self.price_mean = np.mean(all_prices)
        self.price_std = np.std(all_prices)
        
        for episode in range(1, episodes+1):
            state = self.discretize(self.env.reset())
            total_reward = 0
            episode_steps = 0
            done = False
            
            # Track actions for this episode
            buy_count = 0
            sell_count = 0
            hold_count = 0
            
            while not done:
                episode_steps += 1
                self.total_steps += 1
                
                # Action selection
                action_idx = self.choose_action(state)
                action = self.action_space[action_idx]
                
                # Count actions
                if action > 0:
                    buy_count += 1
                elif action < 0:
                    sell_count += 1
                else:
                    hold_count += 1
                
                # Take step
                next_obs, reward, done = self.env.step([action])
                next_state = self.discretize(next_obs)
                
                # Shaped reward
                shaped_reward = self.reward_shaping(
                    reward,  # Scale down rewards
                    action,
                    next_obs[0],
                    next_obs[1],
                    next_obs[2]
                )
                
                # Update visit counts
                self.state_visits[state + (action_idx,)] += 1
                
                # Q-learning update
                next_best_action = np.argmax(self.q[next_state])
                td_target = shaped_reward + self.gamma * self.q_target[next_state + (next_best_action,)]
                td_error = td_target - self.q[state + (action_idx,)]
                
                # Adaptive learning rate
                visit_count = self.state_visits[state + (action_idx,)]
                effective_alpha = self.alpha / (1 + 0.1 * np.sqrt(visit_count))
                
                # Update Q-value
                self.q[state + (action_idx,)] += effective_alpha * td_error
                
                # Periodic target update
                if self.total_steps % self.target_update_freq == 0:
                    self.update_target_network()
                
                state = next_state
                total_reward += reward
            
            # Early stopping check
            if total_reward > self.best_reward + min_improvement:
                self.best_reward = total_reward
                self.episodes_without_improvement = 0
                self.save_q_table("models/double_best_model")
                print(f"New best reward: {self.best_reward:.2f}")
            else:
                self.episodes_without_improvement += 1

            # Decay epsilon
            self.epsilon = max(
                self.epsilon_min,
                self.epsilon * self.epsilon_decay
            )
            
            # Calculate action distribution
            total_actions = episode_steps
            action_dist = {
                'buy': buy_count / total_actions,
                'hold': hold_count / total_actions,
                'sell': sell_count / total_actions
            }
            
            # Logging
            if verbose and episode % 10 == 0:
                print(f"\nEpisode {episode}/{episodes}")
                print(f"Total Reward: {total_reward:.2f}")
                print(f"Steps: {episode_steps}")
                print(f"Epsilon: {self.epsilon:.3f}")
                print(f"Action Distribution:")
                print(f"  Buy:  {action_dist['buy']*100:5.1f}%")
                print(f"  Hold: {action_dist['hold']*100:5.1f}%")
                print(f"  Sell: {action_dist['sell']*100:5.1f}%")
                print(f"Q-diff: {np.mean(np.abs(self.q - self.q_target)):.3f}")
                
                # Additional diagnostics
                print("\nState-Action Values:")
                print(f"Max Q-value: {np.max(self.q):.3f}")
                print(f"Min Q-value: {np.min(self.q):.3f}")
                print(f"Mean Q-value: {np.mean(self.q):.3f}")
                
                # Sample state analysis
                sample_state = (0, 0, 12)  # Low storage, low price, mid-day
                q_values = self.q[sample_state]
                print("\nSample State Q-values (low storage, low price, mid-day):")
                print(f"  Sell: {q_values[0]:.3f}")
                print(f"  Hold: {q_values[1]:.3f}")
                print(f"  Buy:  {q_values[2]:.3f}")

                 # Plot current reward history
                self.plot_rewards()

                print("-" * 50)
        
        print("\nTraining Summary:")
        print(f"Best reward achieved: {best_reward:.2f}")
        print(f"Training time: {time.time()-start:.2f} seconds")
        print(f"Final epsilon: {self.epsilon:.3f}")
        
        # Plot final Q-value distribution
        self.plot_q_distribution()
        # Final plots
        self.plot_rewards()
    
    def plot_q_distribution(self):
        """Plot the distribution of Q-values"""
        plt.figure(figsize=(10, 6))
        plt.hist(self.q.flatten(), bins=50, alpha=0.7)
        plt.title("Distribution of Q-values")
        plt.xlabel("Q-value")
        plt.ylabel("Frequency")
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def analyze_state(self, storage, price, hour):
        """Analyze a specific state"""
        state = self.discretize(np.array([storage, price, hour, 1]))
        q_values = self.q[state]
        
        print(f"\nState Analysis:")
        print(f"Storage: {storage:.1f}")
        print(f"Price: {price:.1f}")
        print(f"Hour: {hour}")
        print("\nQ-values:")
        print(f"Sell: {q_values[0]:.3f}")
        print(f"Hold: {q_values[1]:.3f}")
        print(f"Buy:  {q_values[2]:.3f}")
        print(f"\nRecommended action: {self.action_space[np.argmax(q_values)]}")
    
    # def reward_shaping(self, reward, action, storage, price, hour):
    #     """Enhanced reward shaping"""
    #     shaped_reward = reward
        
    #     # Storage management
    #     storage_target = self.env.daily_energy_demand
    #     storage_deficit = max(0, storage_target - storage)
    #     hours_left = 24 - hour
        
    #     # Price consideration
    #     price_zscore = (price - self.price_mean) / (self.price_std + 1e-8)
        
    #     # Basic shaping
    #     if action > 0:  # Buying
    #         # Penalize buying at high prices
    #         shaped_reward -= 0.1 * price_zscore * action
    #         # Reward buying when storage is low and time is running out
    #         if storage_deficit > 0 and hours_left < 12:
    #             shaped_reward += 0.2 * action
    #     elif action < 0:  # Selling
    #         # Reward selling at high prices
    #         shaped_reward += 0.1 * price_zscore * abs(action)
    #         # Penalize selling when storage is low
    #         if storage_deficit > 0:
    #             shaped_reward -= 0.2 * abs(action)
        
    #     return shaped_reward
    
    def save_q_table(self, path, verbose=False):
        """Save both behavior and target Q-tables."""
        if not os.path.exists(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
        
        np.save(path + "_q.npy", self.q)
        np.save(path + "_q_target.npy", self.q_target)
        
        # Save hyperparameters
        hyperparameters_path = path + "_hyperparameters.json"
        with open(hyperparameters_path, "w") as f:
            json.dump({**self.hyperparameters, 
                      "target_update_freq": self.target_update_freq}, f)
            
        if verbose:
            print(f"Q-tables saved to {path}_q.npy and {path}_q_target.npy")
            print(f"Hyperparameters saved to {hyperparameters_path}")
    
    def load_q_table(self, path, verbose=False):
        """Load both Q-tables."""
        self.q = np.load(path + "_q.npy")
        self.q_target = np.load(path + "_q_target.npy")
        
        if verbose:
            print(f"Q-tables loaded from {path}_q.npy and {path}_q_target.npy")
    
    def plot_q_stability(self):
        """Plot the differences between behavior and target Q-tables."""
        differences = np.abs(self.q - self.q_target)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Q-value Stability Analysis')
        
        # Plot average differences across different dimensions
        im1 = axes[0,0].imshow(differences.mean(axis=(2,3)), cmap='hot')
        axes[0,0].set_title('Storage vs Price')
        axes[0,0].set_xlabel('Price bins')
        axes[0,0].set_ylabel('Storage bins')
        plt.colorbar(im1, ax=axes[0,0])
        
        im2 = axes[0,1].imshow(differences.mean(axis=(1,3)), cmap='hot')
        axes[0,1].set_title('Storage vs Hour')
        axes[0,1].set_xlabel('Hour')
        axes[0,1].set_ylabel('Storage bins')
        plt.colorbar(im2, ax=axes[0,1])
        
        # Plot Q-value distributions
        axes[1,0].hist(self.q.flatten(), bins=50, alpha=0.5, label='Behavior Q')
        axes[1,0].hist(self.q_target.flatten(), bins=50, alpha=0.5, label='Target Q')
        axes[1,0].set_title('Q-value Distributions')
        axes[1,0].legend()
        
        # Plot Q-values for specific states
        test_state = (0, 0, 12)  # example state
        axes[1,1].bar(range(len(self.action_space)), 
                     self.q[test_state], alpha=0.5, label='Behavior Q')
        axes[1,1].bar(range(len(self.action_space)), 
                     self.q_target[test_state], alpha=0.5, label='Target Q')
        axes[1,1].set_title('Q-values for Test State')
        axes[1,1].set_xlabel('Action')
        axes[1,1].legend()
        
        plt.tight_layout()
        plt.show()
