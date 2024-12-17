import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import random
import matplotlib.pyplot as plt
from tqdm import tqdm

class VRPEnvironment:
    def __init__(self, num_customers=10, grid_size=50):  # Reduced default size
        self.num_customers = num_customers
        self.grid_size = grid_size
        self.reset()
        
    def reset(self):
        self.depot = np.array([self.grid_size/2, self.grid_size/2])
        self.customers = np.random.randint(0, self.grid_size, size=(self.num_customers, 2))
        self.visited = np.zeros(self.num_customers, dtype=bool)
        self.current_pos = self.depot.copy()
        return self.get_state()
    
    def step(self, action):
        if action >= self.num_customers or self.visited[action]:
            return self.get_state(), -50, True  # Smaller penalty, faster termination
            
        self.visited[action] = True
        prev_pos = self.current_pos.copy()
        self.current_pos = self.customers[action]
        distance = np.linalg.norm(self.current_pos - prev_pos)
        reward = -distance
        
        done = np.all(self.visited)
        if done:
            reward -= np.linalg.norm(self.current_pos - self.depot)
            
        return self.get_state(), reward, done
    
    def get_state(self):
        return np.concatenate([
            self.current_pos,
            self.customers.flatten(),
            self.visited
        ])

# Baseline DQN Implementation
class DQNNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
    def forward(self, x):
        return self.net(x)

class DQNAgent:
    def __init__(self, state_dim, action_dim, hidden_dim=64, lr=1e-3, gamma=0.95):
        self.device = torch.device("cpu")
        self.gamma = gamma
        self.action_dim = action_dim
        
        self.network = DQNNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_network = DQNNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_network.load_state_dict(self.network.state_dict())
        
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        self.memory = deque(maxlen=10000)
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        
    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device)
            q_values = self.network(state)
            return q_values.argmax().item()
            
    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        
    def update(self, batch_size=32):
        if len(self.memory) < batch_size:
            return
            
        batch = random.sample(self.memory, batch_size)
        state_batch = torch.FloatTensor([t[0] for t in batch]).to(self.device)
        action_batch = torch.LongTensor([t[1] for t in batch]).to(self.device)
        reward_batch = torch.FloatTensor([t[2] for t in batch]).to(self.device)
        next_state_batch = torch.FloatTensor([t[3] for t in batch]).to(self.device)
        done_batch = torch.FloatTensor([t[4] for t in batch]).to(self.device)
        
        current_q = self.network(state_batch).gather(1, action_batch.unsqueeze(1))
        next_q = self.target_network(next_state_batch).max(1)[0].detach()
        target_q = reward_batch + (1 - done_batch) * self.gamma * next_q
        
        loss = F.mse_loss(current_q.squeeze(), target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        return loss.item()

class MaxEntNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):  # Smaller network
        super().__init__()
        
        # Simplified architecture
        self.policy_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        self.value_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def policy(self, state):
        logits = self.policy_net(state)
        dist = F.softmax(logits, dim=-1)
        log_prob = F.log_softmax(logits, dim=-1)
        entropy = -(dist * log_prob).sum(dim=-1)
        return dist, entropy
        
    def value(self, state):
        return self.value_net(state)

class MaxEntRL:
    def __init__(self, state_dim, action_dim, hidden_dim=64, lr=1e-3, gamma=0.95, tau=0.1, alpha=0.1):
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        
        # Force CPU usage
        self.device = torch.device("cpu")
        
        self.network = MaxEntNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_network = MaxEntNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_network.load_state_dict(self.network.state_dict())
        
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        
        self.memory = deque(maxlen=10000)  # Smaller memory
        
    def select_action(self, state):
        with torch.no_grad():  # Prevent gradient computation during action selection
            state = torch.FloatTensor(state).to(self.device)
            dist, _ = self.network.policy(state)
            action = torch.multinomial(dist, 1).item()
        return action
        
    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        
    def update(self, batch_size=32):  # Smaller batch size
        if len(self.memory) < batch_size:
            return
            
        batch = random.sample(self.memory, batch_size)
        state_batch = torch.FloatTensor([t[0] for t in batch]).to(self.device)
        action_batch = torch.LongTensor([t[1] for t in batch]).to(self.device)
        reward_batch = torch.FloatTensor([t[2] for t in batch]).to(self.device)
        next_state_batch = torch.FloatTensor([t[3] for t in batch]).to(self.device)
        done_batch = torch.FloatTensor([t[4] for t in batch]).to(self.device)
        
        with torch.no_grad():
            next_value = self.target_network.value(next_state_batch).squeeze()
            expected_value = reward_batch + self.gamma * next_value * (1 - done_batch)
        
        curr_value = self.network.value(state_batch).squeeze()
        value_loss = F.mse_loss(curr_value, expected_value)
        
        dist, entropy = self.network.policy(state_batch)
        log_prob = torch.log(dist + 1e-10)
        advantage = (expected_value - curr_value).detach()
        
        policy_loss = -(log_prob[range(batch_size), action_batch] * advantage).mean()
        entropy_loss = -entropy.mean()
        
        total_loss = value_loss + policy_loss + self.alpha * entropy_loss
        
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        # Soft update target network
        for target_param, param in zip(self.target_network.parameters(), self.network.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
        return total_loss.item()
        
# SAC Implementation (simplified version)
class SACNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super().__init__()
        
        # Actor network (policy)
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim * 2)  # Mean and log_std
        )
        
        # Twin Q networks
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def actor_forward(self, state):
        output = self.actor(state)
        mean, log_std = output.chunk(2, dim=-1)
        log_std = torch.clamp(log_std, -20, 2)
        return mean, log_std
        
    def q_forward(self, state, action):
        sa = torch.cat([state, action], dim=1)
        return self.q1(sa), self.q2(sa)

class SACAgent:
    def __init__(self, state_dim, action_dim, hidden_dim=64, lr=1e-3, gamma=0.95, tau=0.01, alpha=0.2):
        self.device = torch.device("cpu")
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.action_dim = action_dim
        
        self.network = SACNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_network = SACNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_network.load_state_dict(self.network.state_dict())
        
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        self.memory = deque(maxlen=10000)
        
    def select_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device)
            mean, log_std = self.network.actor_forward(state)
            std = log_std.exp()
            dist = torch.distributions.Normal(mean, std)
            action = dist.sample()
            action = torch.softmax(action, dim=-1)
            return action.argmax().item()
            
    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        
    def update(self, batch_size=32):
        if len(self.memory) < batch_size:
            return
            
        batch = random.sample(self.memory, batch_size)
        state_batch = torch.FloatTensor([t[0] for t in batch]).to(self.device)
        action_batch = torch.FloatTensor([t[1] for t in batch]).to(self.device)
        reward_batch = torch.FloatTensor([t[2] for t in batch]).to(self.device)
        next_state_batch = torch.FloatTensor([t[3] for t in batch]).to(self.device)
        done_batch = torch.FloatTensor([t[4] for t in batch]).to(self.device)
        
        # Update critic
        with torch.no_grad():
            next_mean, next_log_std = self.target_network.actor_forward(next_state_batch)
            next_std = next_log_std.exp()
            next_dist = torch.distributions.Normal(next_mean, next_std)
            next_action = next_dist.sample()
            next_action_probs = torch.softmax(next_action, dim=-1)
            
            next_q1, next_q2 = self.target_network.q_forward(next_state_batch, next_action_probs)
            next_q = torch.min(next_q1, next_q2)
            target_q = reward_batch.unsqueeze(-1) + (1 - done_batch.unsqueeze(-1)) * self.gamma * next_q
            
        action_probs = F.one_hot(action_batch.long(), num_classes=self.action_dim).float()
        current_q1, current_q2 = self.network.q_forward(state_batch, action_probs)
        
        q1_loss = F.mse_loss(current_q1, target_q)
        q2_loss = F.mse_loss(current_q2, target_q)
        
        # Update actor
        mean, log_std = self.network.actor_forward(state_batch)
        std = log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        action = dist.rsample()
        action_probs = torch.softmax(action, dim=-1)
        
        q1, q2 = self.network.q_forward(state_batch, action_probs)
        min_q = torch.min(q1, q2)
        
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        actor_loss = (self.alpha * log_prob - min_q).mean()
        
        # Total loss
        loss = q1_loss + q2_loss + actor_loss
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network
        for target_param, param in zip(self.target_network.parameters(), self.network.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
        return loss.item()

def train_and_compare(episodes=200, env_size=10):
    env = VRPEnvironment(num_customers=env_size)
    state_dim = 2 + env_size*2 + env_size
    action_dim = env_size
    
    # Initialize agents
    maxent_agent = MaxEntRL(state_dim, action_dim)
    sac_agent = SACAgent(state_dim, action_dim)
    dqn_agent = DQNAgent(state_dim, action_dim)
    
    # History tracking
    maxent_rewards = []
    sac_rewards = []
    dqn_rewards = []
    
    # Training loop
    for algorithm, agent, rewards in [
        ("MaxEnt", maxent_agent, maxent_rewards),
        ("SAC", sac_agent, sac_rewards),
        ("DQN", dqn_agent, dqn_rewards)
    ]:
        print(f"\nTraining {algorithm}...")
        
        for episode in tqdm(range(episodes)):
            state = env.reset()
            episode_reward = 0
            steps = 0
            done = False
            
            while not done and steps < env_size * 2:
                steps += 1
                action = agent.select_action(state)
                next_state, reward, done = env.step(action)
                
                agent.store_transition(state, action, reward, next_state, done)
                agent.update()
                
                episode_reward += reward
                state = next_state
                
            rewards.append(episode_reward)
            
            if (episode + 1) % 20 == 0:
                mean_reward = np.mean(rewards[-20:])
                print(f"\n{algorithm} Episode {episode+1}, Average Reward: {mean_reward:.2f}")
    
    return maxent_rewards, sac_rewards, dqn_rewards

# Run comparison
print("Starting comparison training...")
maxent_rewards, sac_rewards, dqn_rewards = train_and_compare()

# Plotting
plt.figure(figsize=(15, 5))

# Raw rewards
plt.subplot(1, 2, 1)
plt.plot(maxent_rewards, label='MaxEnt', alpha=0.6)
plt.plot(sac_rewards, label='SAC', alpha=0.6)
plt.plot(dqn_rewards, label='DQN', alpha=0.6)
plt.title('Training Rewards')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.legend()

# Moving average
window = 20
plt.subplot(1, 2, 2)
plt.plot(np.convolve(maxent_rewards, np.ones(window)/window, mode='valid'), 
         label='MaxEnt', alpha=0.8)
plt.plot(np.convolve(sac_rewards, np.ones(window)/window, mode='valid'), 
         label='SAC', alpha=0.8)
plt.plot(np.convolve(dqn_rewards, np.ones(window)/window, mode='valid'), 
         label='DQN', alpha=0.8)
plt.title(f'Moving Average ({window} episodes)')
plt.xlabel('Episode')
plt.ylabel('Average Reward')
plt.legend()

plt.tight_layout()
plt.show()

# Print final statistics
def print_stats(rewards, name):
    print(f"\n{name} Statistics:")
    print(f"Final Average (last 20 episodes): {np.mean(rewards[-20:]):.2f}")
    print(f"Best Episode: {max(rewards):.2f}")
    print(f"Worst Episode: {min(rewards):.2f}")
    print(f"Overall Average: {np.mean(rewards):.2f}")
# Keep all previous code the same until the plotting section, then replace with:

def plot_comparison(maxent_rewards, sac_rewards, dqn_rewards, window=10):
    plt.figure(figsize=(15, 5))
    
    # Get number of episodes
    episodes = len(maxent_rewards)
    x_axis = np.arange(episodes)
    
    # Raw rewards
    plt.subplot(1, 2, 1)
    plt.plot(x_axis, maxent_rewards, label='MaxEnt', alpha=0.6)
    plt.plot(x_axis, sac_rewards, label='SAC', alpha=0.6)
    plt.plot(x_axis, dqn_rewards, label='DQN', alpha=0.6)
    plt.title('Training Rewards (Raw)')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Moving average
    def get_moving_average(rewards, window):
        weights = np.ones(window) / window
        return np.convolve(rewards, weights, mode='valid')
    
    # Calculate moving averages
    ma_maxent = get_moving_average(maxent_rewards, window)
    ma_sac = get_moving_average(sac_rewards, window)
    ma_dqn = get_moving_average(dqn_rewards, window)
    
    # Adjust x-axis for moving average plot to align with original episodes
    ma_x = np.arange(window-1, episodes)
    
    plt.subplot(1, 2, 2)
    plt.plot(ma_x, ma_maxent, label='MaxEnt', alpha=0.8)
    plt.plot(ma_x, ma_sac, label='SAC', alpha=0.8)
    plt.plot(ma_x, ma_dqn, label='DQN', alpha=0.8)
    plt.title(f'Training Rewards ({window}-Episode Moving Average)')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# After training, call the plotting function with a smaller window
print("Plotting results...")
plot_comparison(maxent_rewards, sac_rewards, dqn_rewards, window=10)  # Changed to 10-episode window

# Print statistics with both raw and smoothed metrics
def print_stats(rewards, name, window=10):
    print(f"\n{name} Statistics:")
    print(f"Raw Metrics:")
    print(f"Final Average (last {window} episodes): {np.mean(rewards[-window:]):.2f}")
    print(f"Best Episode: {max(rewards):.2f}")
    print(f"Worst Episode: {min(rewards):.2f}")
    print(f"Overall Average: {np.mean(rewards):.2f}")
    def get_moving_average(rewards, window):
        weights = np.ones(window) / window
        return np.convolve(rewards, weights, mode='valid')
    # Smoothed metrics
    smoothed = get_moving_average(rewards, window)
    print(f"\nSmoothed Metrics ({window}-episode window):")
    print(f"Best Smoothed Performance: {max(smoothed):.2f}")
    print(f"Final Smoothed Performance: {smoothed[-1]:.2f}")

print("\nPerformance Statistics:")
print_stats(maxent_rewards, "MaxEnt")
print_stats(sac_rewards, "SAC")
print_stats(dqn_rewards, "DQN")
