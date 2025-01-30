import os
import time
import json
import numpy as np
import matplotlib.pyplot as plt
from env import DataCenterEnv
import statsmodels.api as sm
import pandas as pd



class QAgent:
    
    def __init__(self, env, alpha=0.1, beta=0.983, tau=0.15, gamma=0.99, gamma_decay=0.999, gamma_min=0.03, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01, random_seed=1,
                 random_init=False, adaptive_lr=True, moving_average="daily"):
        self.env = env         
        self.alpha = alpha                                          # Learning rate
        self.beta = beta                                            # Moving average rate
        self.tau = tau                                              # Reward shaping parameter
        self.gamma = gamma                                          # Discount factor
        self.gamma_decay = gamma_decay                              # Decay rate for gamma
        self.gamma_min = gamma_min                                  # Minimum discount factor
        self.epsilon = epsilon                                      # Exploration rate
        self.epsilon_decay = epsilon_decay                          # Decay rate for epsilon
        self.epsilon_min = epsilon_min                              # Minimum exploration rate
        self.adaptive_lr = adaptive_lr                              # Adaptive learning rate
        self.moving_average = moving_average                        # Moving average for reward shaping
        self.max_storage = 290                                      # Maximum storage level
        self.max_steps = len(self.env.price_values) * 24            # Maximum number of steps in the environment

        np.random.seed(random_seed)                                 # Set random seed for reproducibility
        
        self.hyperparameters = {
            "alpha": alpha,
            "beta": beta,  
            "tau": tau,
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
        self.storage_bins = np.array([0, 72.5, 290])               # Storage: 3 bins 145
        self.hour_bins =  np.linspace(1, 24, 24)                        # Hour: 24 bins
        self.action_space = [-1, 0, 1]                                  # Actions: sell, hold, buy    
        
        self.state_action_number = (len(self.storage_bins)-1) * len(self.hour_bins) * len(self.action_space) # 288 state-action pairs
        
        self.average_price = 50.60                                  # Average price for reward shaping
        self.best_result = 0                                        # Best cumulative reward

        # Initialize Q-table
        if random_init == "Uniform":
            self.q_table = np.random.uniform(low=-1, high=1, size=(
                len(self.storage_bins),                         # Storage bins
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
            return reward + self.tau * storage_level
        elif action == 1:
            return reward + self.tau * storage_level + 10 * self.average_price
        elif action == -1:
            return reward + self.tau * storage_level - 10 * self.average_price
        else:
            raise ValueError('Action not recognized')


    def choose_action(self, state, greedy=False):
        """Choose an action using epsilon-greedy policy and return the action index."""
        epsilon = self.epsilon if not greedy else 0.0
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
            hour = 1.0
            day = 1
            week = 1
            month = 1
            year = 1
            done = False
            
            while not done:
                step += 1
                hour += 1
                
                # If we reach 24 hours, reset hour and increment day
                if hour > 24:
                    hour = 1
                    day += 1

                # Increment week or month depending on moving average setting
                if self.moving_average == "weekly":
                    if day > 7:  # If we pass day 7, reset and increment week
                        week += 1
                        day = 1
                else:
                    # Approximate month length to 30 days (ignores 31-day months and February)
                    if day > 30:
                        month += 1
                        day = 1

                # Year transition
                if month > 12:
                    year += 1
                    month = 1
                    

                action_idx = self.choose_action(state)
                action = self.action_space[action_idx]  # Convert action index to discrete action
                next_obs, reward, done = self.env.step(action)
                next_state = self.discretize(next_obs)



                # Track metrics
                self.rewards_per_step.append(reward)
                self.storage_levels.append(next_obs[0])     # Storage level
                self.actions.append(action)                 # Action
                self.prices.append(next_obs[1])             # Price
                next_state = self.discretize(next_obs)
                
                
                # Moving average for reward shaping
                if self.moving_average == "episodely":
                    if step == 1:
                        self.average_price = next_obs[1]                                                            # Initialize average price at the first step of the episode
                    else:
                        self.average_price = self.beta * self.average_price + (1 - self.beta) * next_obs[1]         # Exponentially weighted moving average (EWMA)

                            
                if self.moving_average == "yearly":
                    if month == 1 and day == 1 and hour == 1:
                        self.average_price = next_obs[1]                                                            # Initialize average price at the first hour of the first day of the first month of the year
                    else:
                        self.average_price = self.beta * self.average_price + (1 - self.beta) * next_obs[1]

                            
                if self.moving_average == "monthly" or self.moving_average == "weekly":
                    if day == 1 and hour == 1:
                        self.average_price = next_obs[1]                                                            # Initialize average price at the first hour of the first day of the week/month
                    else:
                        self.average_price = self.beta * self.average_price + (1 - self.beta) * next_obs[1]

                            
                if self.moving_average == "daily":
                    if hour == 1:
                        self.average_price = next_obs[1]                                                            # Initialize average price at the first hour of the day
                    else:
                        self.average_price = self.beta * self.average_price + (1 - self.beta) * next_obs[1]         

                if bias_correction:
                    self.average_price = self.average_price / (1 - self.beta ** step)
            
                
                

                

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

            
            self.gamma = max(self.gamma * self.gamma_decay, self.gamma_min)         # Decay gamma
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min) # Decay epsilon

            self.cumulative_rewards.append(total_reward)                           # Track cumulative rewards per episode
            average_episode_reward = total_reward / step                           # Calculate average reward per episode
            self.average_episode_rewards.append(average_episode_reward)            # Track average rewards per episode

            if verbose:
                print(f"Episode {episode}/{episodes}, Total Reward: {total_reward:.2f}, Average Reward: {average_episode_reward:.2f}, Epsilon: {self.epsilon:.2f}, Gamma: {self.gamma:.2f}")
            
        self.best_result = round(max(self.cumulative_rewards), 2)  
        end = time.time()
        
        print(f"Training completed in {end-start:.2f}s ({(end-start)/episodes:.2f}s per episode)")
        print(f"Best Cumulative Reward: {max(self.cumulative_rewards)}")


    def act(self, state):
        """Act greedy based on the current state and the learned Q-values."""
        state = self.discretize(state)
        action_idx = np.argmax(self.q_table[state])
        return self.action_space[action_idx]
    
    
    def evaluate(self, path="", years=1, day_to_plot=7,
                 average=False, greedy=True, line_plot=False, x_bins=12):
        """Evaluate the agent over a number of years without exploration."""
        
        if self.max_steps == 0:
            raise ValueError("Max steps not set. Please train the agent first.")
        
        print(f"Evaluating for {years} years...")
        
        if path:
            self.env = DataCenterEnv(path_to_test_data=path)
            # Load the dataset and compute the length
            pd_data = pd.read_excel(path)
            self.max_steps = len(pd_data) * 24
            
        print(f"Total steps: {self.max_steps}")
        
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
        
        path = "figures/"
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        
        if years > 1:
            plt.figure(figsize=(8, 6))
            plt.plot(total_rewards, color="blue")
            plt.title("Total Rewards per Year", fontsize=12)
            plt.xlabel("Year", fontsize=10)
            plt.ylabel("Total Reward", fontsize=10)
            plt.tick_params(axis='both', which='major', labelsize=10)
            plt.savefig('figures/total_rewards.png')
            plt.show()
        else:
            # Plot prices and actions taken over the horizon
            days = [i for i in range(0, self.max_steps, 24)]
            start_idx = np.random.choice(days)
            end_idx = start_idx + day_to_plot * 24

            year_prices = year_prices[start_idx:end_idx]
            year_actions = year_actions[start_idx:end_idx]
            
            if line_plot:
                # Create a figure and primary axis
                fig, ax1 = plt.subplots(figsize=(12, 6))

                # Plot the price line (black) on primary y-axis
                ax1.plot(year_prices, color="black", label="Price")
                ax1.set_xlabel("Time (hours)")
                ax1.set_ylabel("Price", color="black")
                ax1.tick_params(axis="y", labelcolor="black")
                ax1.set_xticks([i for i in range(0, len(year_prices), x_bins)]) 
                ax1.set_title("Price and Actions Over Time")
                ax1.grid(visible=True, which="both", linestyle="--", alpha=0.5)


                # Create a secondary y-axis for actions
                ax2 = ax1.twinx()

                # Plot the actions as a stepped line (for discrete values)
                ax2.step(range(len(year_actions)), year_actions, color="red", label="Action")
                ax2.set_ylabel("Action", color="red")
                ax2.set_yticks([-1, 0, 1])  # Ensure y-ticks match discrete actions
                ax2.set_yticklabels(["Sell", "Hold", "Buy"])
                ax2.tick_params(axis="y", labelcolor="red")

                # Add legends
                ax1.legend(loc="upper left")
                ax2.legend(loc="upper right")
            
            else:
                fig, ax = plt.subplots(figsize=(12, 6))

                # Plot the time series (prices)
                ax.plot(year_prices, color="black", label="Price")
                ax.set_xlabel("Time (hours)")
                ax.set_ylabel("Price", color="black")
                ax.set_title("Prices and Actions Over Time")
                ax.set_xticks([i for i in range(0, len(year_prices), x_bins)])  # Customize ticks
                ax.tick_params(axis="y", labelcolor="black")
                ax.grid(visible=True, which="both", linestyle="--", alpha=0.5)

                # Add vertical lines to indicate days
                for day in days:
                    ax.axvline(day, color="gray", linestyle="--", alpha=0.5)

                
                ax2 = ax.twinx()
                ax2.imshow([year_actions], cmap="coolwarm", aspect="auto", alpha=0.5, extent=[0, len(year_prices), 0, 1])
                ax2.set_yticks([])  # Remove y-axis ticks for the imshow
                ax2.set_ylabel("Action", color="red")
                ax2.tick_params(axis="y", labelcolor="blue")
                

                lines, labels = ax.get_legend_handles_labels()
                ax.legend(lines, labels, loc="upper left")
            
            
            fig.tight_layout()
            fig.savefig('figures/prices_actions.png')
            plt.show()
            
        
        if average:           
            return avg_reward
        else:    
            return last_reward
                
        
    def show_rewards(self, average=False):
        """Plot rewards per episode with confidence intervals."""
        path = "figures/"
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        
        if average:
            plt.figure(figsize=(8, 6))
            plt.plot(self.average_episode_rewards, color="green")
            plt.title("Average Rewards per Episode", fontsize=12)
            plt.xlabel("Episode", fontsize=10)
            plt.ylabel("Average Reward", fontsize=10)
            plt.tick_params(axis='both', which='major', labelsize=10)
            plt.savefig('figures/average_rewards.png')
            plt.show()
            
        else:
            # Plot cumulative rewards per episode and the trend decomposition on the same plot
            decomposition = sm.tsa.seasonal_decompose(self.cumulative_rewards, model="additive", period=10)
            trend = decomposition.trend
            plt.figure(figsize=(8, 6))
            plt.plot(self.cumulative_rewards, color="blue", label="Cumulative Rewards")
            plt.plot(trend, color="red", label="Trend")
            plt.title("Cumulative Rewards per Episode", fontsize=12)
            plt.xlabel("Episode", fontsize=10)
            plt.ylabel("Cumulative Reward", fontsize=10)
            plt.tick_params(axis='both', which='major', labelsize=10)
            plt.legend()
            plt.savefig('figures/cumulative_rewards.png')
            plt.show()
            
            
        
            
    def save_q_table(self, path="results/q_table.npy", verbose=False):
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
             


class ImprovedQAgent(QAgent):
    def __init__(self, env, **kwargs):
        super().__init__(env, **kwargs)
        
        self.price_bins = [0, 29.9, 43, 65, float('inf')]           # Price: quartile-based bins

        
        self.q_table = np.zeros((
                len(self.storage_bins) - 1,
                len(self.hour_bins),
                len(self.action_space) 
            ))

        
        
    def discretize(self, observation):
        """Discretize the observation."""

        storage, price, hour, _, = observation                      # Unpack just the storage level, price, and hour
        
        # Discretize storage, price, and hour
        storage_idx = np.digitize(storage, self.storage_bins) - 1
        price_idx = np.digitize(price, self.price_bins) - 1


        # Clip indices to avoid out-of-bound errors
        storage_idx = np.clip(storage_idx, 0, len(self.storage_bins) - 2)
        price_idx = np.clip(price_idx, 0, len(self.price_bins) - 2)
        discrete_state_index = (storage_idx, price_idx, int(hour) - 1)
        
        return discrete_state_index








def main():
    os.chdir(os.path.dirname(__file__))
    file_path = "data/train.xlsx"
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    environment = DataCenterEnv(path_to_test_data=file_path)
    agent = ImprovedQAgent(environment, random_seed=3)
    
    #agent.train(episodes=1000)
    #agent.save_q_table()
    #agent.show_rewards()

    agent.load_q_table("results/improved_q_table.npy")
    agent.evaluate(path="data/validate.xlsx", day_to_plot=7)
    #agent.evaluate(path="data/validate.xlsx", line_plot=True, day_to_plot=1)





if __name__ == "__main__":
    main()


