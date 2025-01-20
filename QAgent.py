import os
import time
import json
import numpy as np
import matplotlib.pyplot as plt
from env import DataCenterEnv



class QAgent:
    
    def __init__(self, env, alpha=0.1, beta=0.983, tao=0.15, gamma=0.99, gamma_decay=0.999, gamma_min=0.03, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01, random_seed=1,
                 random_init=False, adaptive_lr=True, moving_average=True):
        self.env = env         
        self.alpha = alpha                                          # Learning rate
        self.tao = tao                                              # Reward shaping parameter
        self.beta = beta                                            # Moving average rate
        self.gamma = gamma                                          # Discount factor
        self.gamma_decay = gamma_decay                              # Decay rate for gamma
        self.gamma_min = gamma_min                                  # Minimum discount factor
        self.epsilon = epsilon                                      # Exploration rate
        self.epsilon_decay = epsilon_decay                          # Decay rate for epsilon
        self.epsilon_min = epsilon_min                              # Minimum exploration rate
        self.adaptive_lr = adaptive_lr                              # Adaptive learning rate
        self.moving_average = moving_average                        # Moving average for reward shaping
        self.max_storage = 290                                      # Maximum storage level

        np.random.seed(random_seed)                                 # Set random seed for reproducibility
        
        self.hyperparameters = {
            "alpha": alpha,
            "beta": beta,  
            "tao": tao,
            "gamma": gamma,
            "gamma_decay": gamma_decay,
            "gamma_min": gamma_min,
            "epsilon": epsilon,
            "epsilon_decay": epsilon_decay,
            "epsilon_min": epsilon_min,
            "random_seed": random_seed,
            "random_init": random_init,
            "adaptive_lr": adaptive_lr,
            "moving_average": moving_average
        }

        # Define bins for discretization
        self.storage_bins = np.linspace(0, self.max_storage, 5)         # Storage level: 4 bins
        self.hour_bins =  np.linspace(1, 24, 24)                        # Hour: 24 bins
        self.action_space = [-1, 0, 1]                                  # Actions: sell, hold, buy    
        
        self.state_action_number = (len(self.storage_bins)-1) * len(self.hour_bins) * len(self.action_space)
        
        self.average_price = 50.60                                  # Average price for reward shaping
        self.best_result = 0                                        # Best cumulative reward

        # Initialize Q-table
        if random_init == "Uniform":
            self.q_table = np.random.uniform(low=-1, high=1, size=(
                len(self.storage_bins) - 1,                     # Storage bins
                len(self.hour_bins),                            # Hour bins
                len(self.action_space)                          # Actions
            ))  
        elif random_init == "Normal":
            self.q_table = np.random.normal(loc=0, scale=1, size=(
                len(self.storage_bins) - 1, 
                len(self.hour_bins),   
                len(self.action_space)     
            ))
        else:
            self.q_table = np.zeros((
                len(self.storage_bins) - 1,
                len(self.hour_bins),
                len(self.action_space) 
            ))

        # Tracking attributes
        self.cumulative_rewards = []
        self.average_episode_rewards = []
        self.rewards_per_step = []
        self.storage_levels = []
        self.actions = []
        self.prices = []
        
        self.evaluate_cumulative_rewards = []
        self.evaluate_average_episode_rewards = []
        self.evaluate_storage_levels = []
        self.evaluate_actions = []
        self.evaluate_prices = []


    def discretize(self, observation):
        """Discretize the observation."""

        storage, _, hour, _, = observation      # Unpack just the storage level and hour
        
        # Discretize storage, price, and hour
        storage_idx = np.digitize(storage, self.storage_bins) - 1

        # Clip indices to avoid out-of-bound errors
        storage_idx = np.clip(storage_idx, 0, len(self.storage_bins) - 2)
        discrete_state_index = (storage_idx, int(hour)-1)
        
        return discrete_state_index
    
    
    def reward_shaping(self, reward, action, storage_level):
        """Apply reward shaping to the reward."""
        if action == 0:
            return reward + self.tao * storage_level
        elif action > 0:
            return reward + self.tao * storage_level + 10 * self.average_price
        elif action < 0:
            return reward + self.tao * storage_level - 10 * self.average_price
        else:
            raise ValueError('Action not recognized')


    def choose_action(self, state, greedy=False):
        """Choose an action using epsilon-greedy policy."""
        epsilon = self.epsilon if not greedy else 0
        if np.random.random() < epsilon:
            return np.random.choice(len(self.action_space))     # Explore
        else:
            return np.argmax(self.q_table[state])               # Exploit


    def train(self, episodes=1000, verbose=True, bias_correction=False):
        """Train the agent over a number of episodes."""
        print(f"Training for {episodes} episodes...")
        start = time.time()
        
        for episode in range(1, episodes+1):
            initial_state = self.env.reset()
            state = self.discretize(initial_state)
            total_reward = 0
            step = 0
            done = False
            
            while not done:
                step += 1

                action_idx = self.choose_action(state)
                action = self.action_space[action_idx]  # Convert action index to discrete action
                next_obs, reward, done = self.env.step([action])
                

                # Track metrics
                self.rewards_per_step.append(reward)
                self.storage_levels.append(next_obs[0])     # Storage level
                self.actions.append(action)                 # Action
                self.prices.append(next_obs[1])             # Price
                next_state = self.discretize(next_obs)
                
                
                if self.moving_average:
                    if step == 1:
                        self.average_price = initial_state[1]
                    else:
                        self.average_price = self.beta * self.average_price + (1 - self.beta) * next_obs[1]         # Exponentially weighted moving average (EWMA)
                        if bias_correction:
                            self.average_price = self.average_price / (1 - self.beta ** step)                       # Bias correction
                

                # Update Q-value using the Q-learning formula
                best_next_action = np.max(self.q_table[next_state])
                reward_shaped = self.reward_shaping(reward, action, next_obs[0])
                td_target = reward_shaped + self.gamma * best_next_action
                td_error = td_target - self.q_table[state + (action_idx,)]
                self.q_table[state + (action_idx,)] += self.alpha * td_error

                state = next_state
                total_reward += reward
  
            
                
            if self.adaptive_lr:
                self.alpha = self.alpha/np.sqrt(episode)                           # Decay learning rate
                
            self.cumulative_rewards.append(total_reward)                           # Track cumulative rewards per episode
            average_episode_reward = total_reward / step                           # Calculate average reward per episode
            self.average_episode_rewards.append(average_episode_reward)            # Track average rewards per episode
            
            self.gamma = max(self.gamma * self.gamma_decay, self.gamma_min)         # Decay gamma
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min) # Decay epsilon

            if verbose:
                print(f"Episode {episode}/{episodes}, Total Reward: {total_reward:.2f}, Average Reward: {average_episode_reward:.2f}, Epsilon: {self.epsilon:.2f}, Gamma: {self.gamma:.2f}")
            
        self.best_result = round(max(self.cumulative_rewards), 2)  
        end = time.time()
        
        print(f"Training completed in {end-start:.2f} seconds.")
        print(f"Best Cumulative Reward: {max(self.cumulative_rewards)}")


    def act(self, state):
        """Act greedy based on the current state and the learned Q-values."""
        state = self.discretize(state)
        action_idx = np.argmax(self.q_table[state])
        return self.action_space[action_idx]
    
    
    def evaluate(self, years=1, average=False, greedy=True):
        """Evaluate the agent over a number of years without exploration."""
        print(f"Evaluating for {years} years...")
        total_rewards = []
        for year in range(1, years+1):
            initial_state = self.env.reset()
            state = self.discretize(initial_state)
            year_actions = []
            year_prices = []
            total_reward = 0
            done = False

            while not done:
                
                action_idx = self.choose_action(state, greedy=greedy)
                action = self.action_space[action_idx]
                next_obs, reward, done = self.env.step([action])
                
                year_actions.append(action)
                year_prices.append(next_obs[1])

                state = self.discretize(next_obs)
                total_reward += reward
                
            total_rewards.append(total_reward)
            print(f"Year {year}, Total Reward: {total_reward}")
            
        avg_reward = np.mean(total_rewards)
        print(f"Average reward over {years} years: {avg_reward}")
        last_reward = total_rewards[-1]
        print(f"Last year reward: {last_reward}")
        
        if years > 1:
            plt.figure(figsize=(8, 6))
            plt.plot(total_rewards, color="blue")
            plt.title("Total Rewards per Year", fontsize=12)
            plt.xlabel("Year", fontsize=10)
            plt.ylabel("Total Reward", fontsize=10)
            plt.tick_params(axis='both', which='major', labelsize=10)
            plt.show()
        else:
            # Plot prices and actions in the same figure
            fig, ax1 = plt.subplots()
            color = 'tab:red'
            ax1.set_xlabel('Time')
            ax1.set_ylabel('Price', color=color)
            ax1.plot(year_prices, color=color)
            ax1.tick_params(axis='y', labelcolor=color)
            
            ax2 = ax1.twinx()
            color = 'tab:blue'
            ax2.set_ylabel('Action', color=color)
            ax2.plot(year_actions, color=color)
            ax2.tick_params(axis='y', labelcolor=color)
            
            fig.tight_layout()
            plt.show()
            
            fig.savefig('figures/prices_actions.png')
        
        if average:           
            return avg_reward
        else:    
            return last_reward


    def plot_metrics(self):
        """Plot cumulative rewards, rewards per step, storage levels, actions, and prices."""
        _, axes = plt.subplots(5, 1, figsize=(8, 12), constrained_layout=True)

        axes[0].plot(self.cumulative_rewards, color="blue")
        axes[0].set_title("Cumulative Rewards per Episode", fontsize=10)
        axes[0].set_xlabel("Episode", fontsize=8)
        axes[0].set_ylabel("Cumulative Reward", fontsize=8)
        axes[0].tick_params(axis='both', which='major', labelsize=8)

        axes[1].plot(self.rewards_per_step, color="green")
        axes[1].set_title("Rewards per Step", fontsize=10)
        axes[1].set_xlabel("Step", fontsize=8)
        axes[1].set_ylabel("Reward", fontsize=8)
        axes[1].tick_params(axis='both', which='major', labelsize=8)

        axes[2].plot(self.storage_levels, color="orange")
        axes[2].set_title("Storage Levels over Time", fontsize=10)
        axes[2].set_xlabel("Step", fontsize=8)
        axes[2].set_ylabel("Storage Level (MWh)", fontsize=8)
        axes[2].tick_params(axis='both', which='major', labelsize=8)

        axes[3].hist(self.actions, bins=3, color="purple", alpha=0.7)
        axes[3].set_title("Action Distribution", fontsize=10)
        axes[3].set_xlabel("Action", fontsize=8)
        axes[3].set_ylabel("Frequency", fontsize=8)
        axes[3].tick_params(axis='both', which='major', labelsize=8)

        axes[4].plot(self.prices, color="red")
        axes[4].set_title("Prices over Time", fontsize=10)
        axes[4].set_xlabel("Step", fontsize=8)
        axes[4].set_ylabel("Price", fontsize=8)
        axes[4].tick_params(axis='both', which='major', labelsize=8)

        plt.tight_layout()
        plt.show()
        
        
    def show_rewards(self, average=False):
        """Plot rewards per episode with confidence intervals."""
        if average:
            plt.figure(figsize=(8, 6))
            plt.plot(self.average_episode_rewards, color="green")
            plt.title("Average Rewards per Episode", fontsize=12)
            plt.xlabel("Episode", fontsize=10)
            plt.ylabel("Average Reward", fontsize=10)
            plt.tick_params(axis='both', which='major', labelsize=10)
            plt.show()
            
        else:
            plt.figure(figsize=(8, 6))
            plt.plot(self.cumulative_rewards, color="blue")
            plt.title("Cumulative Rewards per Episode", fontsize=12)
            plt.xlabel("Episode", fontsize=10)
            plt.ylabel("Cumulative Reward", fontsize=10)
            plt.tick_params(axis='both', which='major', labelsize=10)
            plt.show()
        
        path = "figures/average_rewards.png"
        if not os.path.exists(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)


    def save_q_table(self, path, verbose=False):
        """Save the Q-table and hyperparameters to a file."""
        if not os.path.exists(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
        if not path.endswith(".npy"):
            path += ".npy"
        
        # Save Q-table in numpy format  
        np.save(path, self.q_table)
        
        # Save hyperparameters in json file
        hyperparameters_path = path.replace(".npy", "_hyperparameters.json")
        with open(hyperparameters_path, "w") as f:
            json.dump(self.hyperparameters, f)
            
        if verbose:
            print(f"Q-table saved to {path}.")
            print(f"Hyperparameters saved to {hyperparameters_path}.")
    
    
    def load_q_table(self, path, verbose=False):
        """Load the Q-table from a file."""
        self.q_table = np.load(path)
        if verbose:
            print(f"Q-table loaded from {path}.")













def main():
    os.chdir(os.path.dirname(__file__))
    file_path = "data/train.xlsx"
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    environment = DataCenterEnv(path_to_test_data=file_path)
    agent = QAgent(environment)
    
    agent.train(episodes=1)
    agent.save_q_table("results/q_table.npy")
    #agent.load_q_table("results/q_table.npy")
    #agent.plot_metrics()
    agent.show_rewards()
    #agent.evaluate()


if __name__ == "__main__":
    main()


