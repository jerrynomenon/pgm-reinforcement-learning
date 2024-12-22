import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import random
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
import gym
from gym import spaces

def parse_solomon_dataset(file_path):
    """
    Parse Solomon VRPTW C101 dataset
    Returns: Dictionary with problem instance data
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # Parse header information
    vehicle_capacity = int(lines[4].strip().split()[1])

    # Parse customer data (including depot as first customer)
    customers = []
    for line in lines[9:]:  # Data starts at line 9
        if line.strip():
            parts = [float(x) for x in line.strip().split()]
            customer = {
                'id': int(parts[0]),
                'x': parts[1],
                'y': parts[2],
                'demand': parts[3],
                'ready_time': parts[4],
                'due_time': parts[5],
                'service_time': parts[6]
            }
            customers.append(customer)

    return {
        'vehicle_capacity': vehicle_capacity,
        'customers': customers
    }

# Define VRP Environment
class VRPEnvironment:
    def __init__(self, num_customers=10, grid_size=50, solomon_data=None):
        self.num_customers = num_customers
        self.grid_size = grid_size

        if solomon_data:
            # Initialize from Solomon dataset
            self.init_from_solomon(solomon_data)
        else:
            # Random initialization
            self.vehicle_capacity = 200  # Increased for more realistic scenarios
            self.depot = np.array([grid_size/2, grid_size/2])
            self.customers = np.random.randint(0, grid_size, size=(num_customers, 2))
            self.demands = np.random.randint(1, 30, size=num_customers)
            self.ready_times = np.zeros(num_customers)
            self.due_times = np.ones(num_customers) * 1000  # More realistic time windows
            self.service_times = np.ones(num_customers) * 10  # Standard service time

        # Initialize state variables that will be reset
        self.reset()

    def reset(self):
        """Reset the environment to initial state"""
        self.visited = np.zeros(self.num_customers, dtype=bool)
        self.current_pos = self.depot.copy()
        self.current_time = 0
        self.current_load = 0
        return self.get_state()

    def init_from_solomon(self, solomon_data):
        """Initialize environment from Solomon dataset with proper scaling"""
        customers = solomon_data['customers']
        self.num_customers = len(customers) - 1  # Exclude depot
        self.vehicle_capacity = solomon_data['vehicle_capacity']

        # Set depot (first customer in Solomon format)
        depot = customers[0]
        self.depot = np.array([depot['x'], depot['y']])

        # Initialize arrays for customers (excluding depot)
        self.customers = np.array([[c['x'], c['y']] for c in customers[1:]])
        self.demands = np.array([c['demand'] for c in customers[1:]])
        self.ready_times = np.array([c['ready_time'] for c in customers[1:]])
        self.due_times = np.array([c['due_time'] for c in customers[1:]])
        self.service_times = np.array([c['service_time'] for c in customers[1:]])

        # Scale coordinates to reasonable range if needed
        if np.max(self.customers) > 1000:
            scale_factor = 1000 / np.max(self.customers)
            self.customers *= scale_factor
            self.depot *= scale_factor

    def get_state(self):
        """Return current state with proper normalization"""
        # Normalize coordinates
        max_coord = max(np.max(self.customers), np.max(self.depot))
        normalized_pos = self.current_pos / max_coord
        normalized_customers = self.customers / max_coord

        # Normalize time windows
        max_time = max(np.max(self.due_times), self.current_time)
        normalized_time = self.current_time / max_time if max_time > 0 else 0

        # Normalize capacity
        normalized_load = self.current_load / self.vehicle_capacity if self.vehicle_capacity > 0 else 0

        return np.concatenate([
            normalized_pos,                    # Current position (2,)
            normalized_customers.flatten(),    # Customer coordinates (num_customers * 2,)
            self.visited.astype(float),       # Visit status (num_customers,)
            [normalized_time],                # Current time (1,)
            [normalized_load],                # Current load (1,)
            self.demands / np.max(self.demands) if np.max(self.demands) > 0 else self.demands,  # Normalized demands
            self.ready_times / max_time if max_time > 0 else self.ready_times,  # Normalized ready times
            self.due_times / max_time if max_time > 0 else self.due_times      # Normalized due times
        ])

    def step(self, action):
        """Enhanced step function with better rewards and constraints"""
        if action >= self.num_customers or self.visited[action]:
            return self.get_state(), -100, True  # Increased penalty for invalid actions

        # Calculate travel time/distance
        next_pos = self.customers[action]
        travel_time = np.linalg.norm(self.current_pos - next_pos)

        # Update current time
        arrival_time = self.current_time + travel_time

        # Time window violations
        time_window_penalty = 0
        if arrival_time > self.due_times[action]:
            time_window_penalty = -50 * (arrival_time - self.due_times[action])
        elif arrival_time < self.ready_times[action]:
            arrival_time = self.ready_times[action]

        # Capacity violations
        if self.current_load + self.demands[action] > self.vehicle_capacity:
            return self.get_state(), -200, True  # Severe penalty for capacity violations

        # Update state
        self.current_pos = next_pos
        self.current_time = arrival_time + self.service_times[action]
        self.current_load += self.demands[action]
        self.visited[action] = True

        # Calculate reward
        base_reward = -travel_time  # Distance-based cost
        service_reward = 10  # Reward for serving a customer
        urgency_bonus = max(0, (self.due_times[action] - arrival_time) / self.due_times[action]) * 5

        total_reward = base_reward + service_reward + urgency_bonus + time_window_penalty

        # Check completion
        done = np.all(self.visited)
        if done:
            return_distance = np.linalg.norm(self.current_pos - self.depot)
            if self.current_time + return_distance > max(self.due_times):
                total_reward -= 100  # Penalty for late return
            total_reward -= return_distance

        return self.get_state(), total_reward, done

class GymVRPEnvironment(gym.Env):
    def __init__(self, num_customers=10, grid_size=50, solomon_data=None):
        super(GymVRPEnvironment, self).__init__()
        self.env = VRPEnvironment(num_customers, grid_size, solomon_data)

        # Dynamic state space based on problem size
        state_size = (2 +                     # Current position
                     self.env.num_customers * 2 +  # Customer coordinates
                     self.env.num_customers +      # Visit status
                     1 +                          # Current time
                     1 +                          # Current load
                     self.env.num_customers * 3)  # Demands and time windows

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(state_size,),
            dtype=np.float32
        )

        self.action_space = spaces.Discrete(self.env.num_customers)

    def reset(self):
        return self.env.reset().astype(np.float32)

    def step(self, action):
        next_state, reward, done = self.env.step(action)
        return next_state.astype(np.float32), reward, done, {}
      
# Define RL and OR Methods
# OR-Tools Solver
class ORToolsSolver:
    def solve(self, env):
        state = env.reset()
        depot = 0
        num_customers = env.num_customers

        # Create distance matrix
        distance_matrix = np.zeros((num_customers + 1, num_customers + 1))
        all_points = np.vstack([env.depot, env.customers])
        for i in range(num_customers + 1):
            for j in range(num_customers + 1):
                distance_matrix[i][j] = np.linalg.norm(all_points[i] - all_points[j])

        # Set up routing
        manager = pywrapcp.RoutingIndexManager(num_customers + 1, 1, depot)
        routing = pywrapcp.RoutingModel(manager)

        def distance_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return int(distance_matrix[from_node][to_node])

        transit_callback_index = routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        # Solve
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)

        solution = routing.SolveWithParameters(search_parameters)
        route = []

        if solution:
            index = routing.Start(0)
            while not routing.IsEnd(index):
                route.append(manager.IndexToNode(index) - 1)  # Adjust for depot
                index = solution.Value(routing.NextVar(index))
            route = [node for node in route if node >= 0]  # Remove depot if included

        return route  # Return the visiting order

# PPO Agent
class PPOAgent:
    def __init__(self, env, policy='MlpPolicy', **kwargs):
        self.env = DummyVecEnv([lambda: env])
        # Pass any additional kwargs to PPO constructor
        self.model = PPO(policy, self.env, **kwargs)

    def train(self, timesteps):
        self.model.learn(total_timesteps=timesteps)

    def evaluate(self, episodes=10):
        rewards = []
        for _ in range(episodes):
            state = self.env.reset()
            episode_reward = 0
            done = False

            while not done:
                action, _ = self.model.predict(state)
                state, reward, done, _ = self.env.step(action)
                episode_reward += reward[0]

            rewards.append(episode_reward)

        return rewards

# Baseline DQN Implementation
class DQNNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):  # Increased hidden dim
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),  # Added another layer
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
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()

        self.policy_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

        self.value_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
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
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()

        # Actor network outputs mean and log_std for each action
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim * 2)  # Mean and log_std for each action
        )

        # Two Q-networks for double Q-learning
        self.q1 = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)  # Q-value for each action
        )

        self.q2 = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)  # Q-value for each action
        )

        # Target Q-networks
        self.target_q1 = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

        self.target_q2 = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

class SACAgent:
    def __init__(self, state_dim, action_dim, hidden_dim=256, lr=3e-4, gamma=0.99, tau=0.005, alpha=0.2, buffer_size=100000):
        self.device = torch.device("cpu")
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.action_dim = action_dim

        self.network = SACNetwork(state_dim, action_dim, hidden_dim).to(self.device)

        # Separate optimizers for actor and critics
        self.actor_optimizer = optim.Adam(self.network.actor.parameters(), lr=lr)
        self.q1_optimizer = optim.Adam(self.network.q1.parameters(), lr=lr)
        self.q2_optimizer = optim.Adam(self.network.q2.parameters(), lr=lr)

        # Initialize target networks
        for target_param, param in zip(self.network.target_q1.parameters(), self.network.q1.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(self.network.target_q2.parameters(), self.network.q2.parameters()):
            target_param.data.copy_(param.data)

        self.memory = deque(maxlen=buffer_size)

        # Temperature parameter
        self.target_entropy = -action_dim  # Target entropy is -|A|
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)

    def select_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device)
            output = self.network.actor(state)
            mean, log_std = output.chunk(2, dim=-1)
            log_std = torch.clamp(log_std, -20, 2)
            std = log_std.exp()

            # Use reparameterization trick
            normal = torch.distributions.Normal(mean, std)
            action = normal.rsample()

            # Apply softmax to get probabilities
            action_probs = F.softmax(action, dim=-1)

            # During evaluation, take the most likely action
            return action_probs.argmax().item()

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def update(self, batch_size=256):
        if len(self.memory) < batch_size:
            return

        # Sample batch
        batch = random.sample(self.memory, batch_size)
        state_batch = torch.FloatTensor([t[0] for t in batch]).to(self.device)
        action_batch = torch.LongTensor([t[1] for t in batch]).to(self.device)
        reward_batch = torch.FloatTensor([t[2] for t in batch]).to(self.device)
        next_state_batch = torch.FloatTensor([t[3] for t in batch]).to(self.device)
        done_batch = torch.FloatTensor([t[4] for t in batch]).to(self.device)

        # Update temperature parameter
        actor_output = self.network.actor(state_batch)
        mean, log_std = actor_output.chunk(2, dim=-1)
        log_std = torch.clamp(log_std, -20, 2)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        sampled_actions = normal.rsample()
        log_probs = normal.log_prob(sampled_actions).sum(dim=-1)

        alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        self.alpha = self.log_alpha.exp().item()

        # Update critics
        with torch.no_grad():
            next_actor_output = self.network.actor(next_state_batch)
            next_mean, next_log_std = next_actor_output.chunk(2, dim=-1)
            next_log_std = torch.clamp(next_log_std, -20, 2)
            next_std = next_log_std.exp()
            next_normal = torch.distributions.Normal(next_mean, next_std)
            next_actions = next_normal.rsample()
            next_log_probs = next_normal.log_prob(next_actions).sum(dim=-1)

            next_action_probs = F.softmax(next_actions, dim=-1)
            next_q1 = self.network.target_q1(next_state_batch)
            next_q2 = self.network.target_q2(next_state_batch)
            next_q = torch.min(next_q1, next_q2)
            next_q = (next_action_probs * next_q).sum(dim=1)
            target_q = reward_batch + (1 - done_batch) * self.gamma * (next_q - self.alpha * next_log_probs)

        # Get current Q estimates
        current_q1 = self.network.q1(state_batch)
        current_q2 = self.network.q2(state_batch)
        current_q1 = current_q1.gather(1, action_batch.unsqueeze(1)).squeeze()
        current_q2 = current_q2.gather(1, action_batch.unsqueeze(1)).squeeze()

        # Compute critic losses
        q1_loss = F.mse_loss(current_q1, target_q)
        q2_loss = F.mse_loss(current_q2, target_q)

        # Update critics
        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        self.q1_optimizer.step()

        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        self.q2_optimizer.step()

        # Update actor
        actor_output = self.network.actor(state_batch)
        mean, log_std = actor_output.chunk(2, dim=-1)
        log_std = torch.clamp(log_std, -20, 2)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        actions = normal.rsample()
        log_probs = normal.log_prob(actions).sum(dim=-1)

        action_probs = F.softmax(actions, dim=-1)
        q1 = self.network.q1(state_batch)
        q2 = self.network.q2(state_batch)
        q = torch.min(q1, q2)
        q = (action_probs * q).sum(dim=1)

        actor_loss = (self.alpha * log_probs - q).mean()

        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft update target networks
        for target_param, param in zip(self.network.target_q1.parameters(), self.network.q1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for target_param, param in zip(self.network.target_q2.parameters(), self.network.q2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return actor_loss.item()

class NearestNeighbor:
    def solve(self, env):
        state = env.reset()
        current_pos = env.depot.copy()
        unvisited = set(range(env.num_customers))
        route = []  # To track the visiting order
        total_distance = 0

        while unvisited:
            nearest = min(unvisited, key=lambda i: np.linalg.norm(current_pos - env.customers[i]))
            total_distance += np.linalg.norm(current_pos - env.customers[nearest])
            route.append(nearest)
            current_pos = env.customers[nearest]
            unvisited.remove(nearest)

        total_distance += np.linalg.norm(current_pos - env.depot)
        return route  # Return the visiting order

# Experiment Setup
def run_experiments(episodes=1000, env_size=100):
    # Initialize environment
    env = GymVRPEnvironment(num_customers=env_size)

    # Get actual state dimension from environment
    state = env.reset()
    state_dim = len(state)
    action_dim = env.action_space.n

    print(f"Environment initialized with state_dim={state_dim}, action_dim={action_dim}")

    # Initialize agents with correct dimensions
    maxent_agent = MaxEntRL(state_dim, action_dim, hidden_dim=256)
    sac_agent = SACAgent(state_dim, action_dim, hidden_dim=256)
    dqn_agent = DQNAgent(state_dim, action_dim, hidden_dim=256)
    nn_solver = NearestNeighbor()
    ppo_env = GymVRPEnvironment(num_customers=env_size)
    ppo_agent = PPOAgent(
        ppo_env,
        verbose=0,
        policy_kwargs={'net_arch': [256, 256]}
    )
    or_solver = ORToolsSolver()

    # History tracking
    or_rewards = []
    ppo_rewards = []
    maxent_rewards = []
    sac_rewards = []
    dqn_rewards = []
    nn_rewards = []

    # Train PPO Agent
    print("\nTraining PPO...")
    # ppo_agent.model.learn(total_timesteps=episodes * 100)
    ppo_agent.train(timesteps=episodes * 100)
    # Training loop
    for algorithm, agent, rewards in [
        ("MaxEnt", maxent_agent, maxent_rewards),
        ("SAC", sac_agent, sac_rewards),
        ("DQN", dqn_agent, dqn_rewards),
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
                next_state, reward, done, _ = env.step(action)

                agent.store_transition(state, action, reward, next_state, done)
                agent.update()

                episode_reward += reward
                state = next_state

            rewards.append(episode_reward)

            if (episode + 1) % 20 == 0:
                mean_reward = np.mean(rewards[-20:])
                print(f"\n{algorithm} Episode {episode+1}, Average Reward: {mean_reward:.2f}")

    # Run Nearest Neighbor
    print("\nRunning Nearest Neighbor...")
    for _ in tqdm(range(episodes)):
        reward = nn_solver.solve(env.env)
        nn_rewards.append(reward)

    # Run OR-Tools Solver
    print("\nRunning OR-Tools Solver...")
    for _ in tqdm(range(episodes)):
        reward = or_solver.solve(env.env)
        or_rewards.append(reward)

    # Evaluate PPO
    print("\nEvaluating PPO...")
    ppo_rewards = ppo_agent.evaluate(episodes=episodes)

    return dqn_rewards, nn_rewards, or_rewards, ppo_rewards, maxent_rewards, sac_rewards, dqn_agent, maxent_agent, sac_agent, ppo_agent, nn_solver, or_solver

# Run and plot results
results = run_experiments()
dqn_rewards, nn_rewards, or_rewards, ppo_rewards, maxent_rewards, sac_rewards, dqn_agent, maxent_agent, sac_agent, ppo_agent, nn_solver, or_solver = results

# Define methods to evaluate
methods = {
    "DQN": lambda state: dqn_agent.select_action(state.numpy() if isinstance(state, torch.Tensor) else state),
    "MaxEnt": lambda state: maxent_agent.select_action(state.numpy() if isinstance(state, torch.Tensor) else state),
    "SAC": lambda state: sac_agent.select_action(state.numpy() if isinstance(state, torch.Tensor) else state),
    "PPO": ppo_agent,
    "Nearest Neighbor": lambda env: nn_solver.solve(env.env),
    "OR-Tools": lambda env: or_solver.solve(env.env)
}
# Compute total distance and evaluate methods
def get_route_from_method(env, method_name, method_fn):
    """Helper function to get route from different methods"""
    env.reset()
    current_pos = env.env.depot.copy()
    route = []

    try:
        if method_name in ["DQN", "MaxEnt", "SAC"]:
            # For RL agents
            state = env.reset()
            done = False
            while not done and len(route) < env.env.num_customers:
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0)
                    action = method_fn(state_tensor)
                    if isinstance(action, torch.Tensor):
                        action = action.item()
                next_state, _, done, _ = env.step(action)
                route.append(action)
                state = next_state
        elif method_name == "PPO":
            # For PPO
            state = env.reset()
            done = False
            while not done and len(route) < env.env.num_customers:
                action, _ = method_fn.model.predict(state)
                next_state, _, done, _ = env.step(action)
                route.append(int(action))
                state = next_state
        else:
            # For OR-Tools and Nearest Neighbor
            route = method_fn(env)

    except Exception as e:
        print(f"Error in {method_name}: {str(e)}")
        return None

    return route

def compute_route_distance(env, route):
    """Compute total distance for a given route"""
    if route is None:
        return float('inf')

    env.reset()
    total_distance = 0
    current_pos = env.env.depot.copy()

    for action in route:
        try:
            if action < 0 or action >= env.env.num_customers or env.env.visited[action]:
                continue

            next_pos = env.env.customers[action]
            total_distance += np.linalg.norm(current_pos - next_pos)
            current_pos = next_pos
            env.env.visited[action] = True

        except Exception as e:
            print(f"Error computing distance for action {action}: {str(e)}")
            return float('inf')

    # Add return to depot
    total_distance += np.linalg.norm(current_pos - env.env.depot)
    return total_distance

def evaluate_performance(env, methods, episodes=10):
    """Evaluate both distances and execution times"""
    all_distances = {method: [] for method in methods.keys()}
    all_times = {method: [] for method in methods.keys()}

    for episode in range(episodes):
        for method_name, method_fn in methods.items():
            try:
                start_time = time.time()
                route = get_route_from_method(env, method_name, method_fn)
                end_time = time.time()

                distance = compute_route_distance(env, route)
                execution_time = end_time - start_time

                all_distances[method_name].append(distance)
                all_times[method_name].append(execution_time)

            except Exception as e:
                print(f"Error in {method_name}: {str(e)}")
                all_distances[method_name].append(float('inf'))
                all_times[method_name].append(float('inf'))

    # Compute averages
    avg_distances = {method: np.mean(distances) for method, distances in all_distances.items()}
    avg_times = {method: np.mean(times) for method, times in all_times.items()}

    return avg_distances, avg_times  # Make sure we return both values

# Evaluate distances and times
print("\nEvaluating Performance...")
avg_distances, avg_times = evaluate_performance(GymVRPEnvironment(num_customers=100), methods)

# Generate a table of results
results_table = pd.DataFrame({
    "Method": list(avg_distances.keys()),
    "Average Distance": list(avg_distances.values()),
    "Average Time (s)": list(avg_times.values())
})

print("\nResults Table:")
print(results_table)

# Save results to CSV for analysis
results_table.to_csv("vrp_distances.csv", index=False)
print("Distance results saved to 'vrp_distances.csv'.")

# Plotting
plt.figure(figsize=(15, 5))

# Raw rewards
plt.subplot(1, 2, 1)
plt.plot(maxent_rewards, label='MaxEnt', alpha=0.6)
plt.plot(sac_rewards, label='SAC', alpha=0.6)
plt.plot(dqn_rewards, label='DQN', alpha=0.6)
plt.plot(ppo_rewards, label='PPO', alpha=0.6)
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
plt.plot(np.convolve(ppo_rewards, np.ones(window)/window, mode='valid'),
         label='PPO', alpha=0.8)
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

def plot_comparison(maxent_rewards, sac_rewards, dqn_rewards, ppo_rewards, window=10):
    plt.figure(figsize=(15, 5))

    # Get number of episodes
    episodes = len(maxent_rewards)
    x_axis = np.arange(episodes)

    # Raw rewards
    plt.subplot(1, 2, 1)
    plt.plot(x_axis, maxent_rewards, label='MaxEnt', alpha=0.6)
    plt.plot(x_axis, sac_rewards, label='SAC', alpha=0.6)
    plt.plot(x_axis, dqn_rewards, label='DQN', alpha=0.6)
    plt.plot(x_axis, ppo_rewards, label='PPO', alpha=0.6)
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
    ma_ppo = get_moving_average(ppo_rewards, window)

    # Adjust x-axis for moving average plot to align with original episodes
    ma_x = np.arange(window-1, episodes)

    plt.subplot(1, 2, 2)
    plt.plot(ma_x, ma_maxent, label='MaxEnt', alpha=0.8)
    plt.plot(ma_x, ma_sac, label='SAC', alpha=0.8)
    plt.plot(ma_x, ma_dqn, label='DQN', alpha=0.8)
    plt.plot(ma_x, ma_ppo, label='PPO', alpha=0.8)
    plt.title(f'Training Rewards ({window}-Episode Moving Average)')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

# After training, call the plotting function with a smaller window
print("Plotting results...")
plot_comparison(maxent_rewards, sac_rewards, dqn_rewards, ppo_rewards, window=10)  # Changed to 10-episode window

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

# Bar chart for solution quality comparison
methods = ["DQN", "Nearest Neighbor", "OR-Tools", "PPO"]
average_rewards = [np.mean(dqn_rewards), np.mean(nn_rewards), np.mean(or_rewards), np.mean(ppo_rewards)]

plt.figure(figsize=(8, 5))
plt.bar(methods, average_rewards, color=['blue', 'orange', 'green', 'purple'])
plt.xlabel("Method")
plt.ylabel("Average Reward (Solution Quality)")
plt.title("Solution Quality Comparison")
plt.show()

def visualize_routes(env, methods):
    plt.figure(figsize=(12, 8))
    depot = env.env.depot
    customers = env.env.customers

    for method, action_fn in methods.items():
        env.reset()
        actions = action_fn(env)
        route = [depot] + [customers[a] for a in actions if isinstance(a, int) and 0 <= a < len(customers)] + [depot]
        route = np.array(route)

        plt.plot(route[:, 0], route[:, 1], marker='o', label=method)
        plt.scatter(depot[0], depot[1], c='red', s=100, label='Depot')

    plt.title("Routes Generated by Different Methods")
    plt.xlabel("X-coordinate")
    plt.ylabel("Y-coordinate")
    plt.legend()
    plt.grid(True)
    plt.show()

visualize_routes(GymVRPEnvironment(num_customers=100), methods)

# Save results to CSV
data = {
    "DQN Rewards": dqn_rewards,
    "Nearest Neighbor Rewards": nn_rewards,
    "OR-Tools Rewards": or_rewards,
    "PPO Rewards": ppo_rewards
}
df = pd.DataFrame(data)
df.to_csv("vrp_results.csv", index=False)
print("Results saved to 'vrp_results.csv'.")
