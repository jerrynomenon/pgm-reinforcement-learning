
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import random
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
