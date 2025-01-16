import torch
import torch.nn as nn
import torch.distributions as dist
import torch.nn.functional as F
import numpy as np
from utils import normalize_tensor, compute_returns

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
        return action.numpy()
    
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
    
    def forward(self, x):
        x = self.state_normalizer(x)
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
        self.mean = torch.zeros(shape)
        self.std = torch.ones(shape)
        self.count = 0
        
    def update(self, x):
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
        return (x - self.mean) / (self.std + 1e-8)