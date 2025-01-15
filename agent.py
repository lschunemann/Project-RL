import torch
import torch.nn as nn
import torch.distributions as dist

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class VPG(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(VPG, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)  # Added another hidden layer
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        mean = torch.tanh(self.fc3(x))  # Use tanh to bound actions to [-1, 1]
        return mean
    
    def get_probs(self, state):
        action_mean = self.forward(state)
        std = torch.exp(self.log_std).clamp(min=1e-3, max=1)  # Clamp std for stability
        dist = torch.distributions.Normal(action_mean, std)
        return dist
    
    def act(self, state):
        with torch.no_grad():
            dist = self.get_probs(state)
            action = dist.sample()
            return action.numpy()

    def load(self, path):
        self.load_state_dict(torch.load(path))
        print("Model loaded from {}".format(path))

    def save(self, path):
        torch.save(self.state_dict(), path)
        print("Model saved to {}".format(path))
