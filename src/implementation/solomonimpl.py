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
            # Initialize with random data as before
            self.vehicle_capacity = 100
            self.depot = np.array([grid_size/2, grid_size/2])
            self.customers = np.random.randint(0, grid_size, size=(num_customers, 2))
            self.demands = np.random.randint(1, 20, size=num_customers)
            self.ready_times = np.zeros(num_customers)
            self.due_times = np.ones(num_customers) * float('inf')
            self.service_times = np.zeros(num_customers)

        self.reset()

    def init_from_solomon(self, solomon_data):
        """Initialize environment from Solomon dataset"""
        customers = solomon_data['customers']
        self.num_customers = len(customers) - 1  # Subtract depot
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

    def get_state(self):
        """
        Return current state including time window information
        """
        # Check expected dimensions for consistency
        return np.concatenate([
            self.current_pos,                     # Shape: (2,)
            self.customers.flatten(),            # Shape: (num_customers * 2,)
            self.visited.astype(float),          # Shape: (num_customers,)
            [self.current_time],                 # Shape: (1,)
            [self.current_load]                  # Shape: (1,)
        ])

    def reset(self):
        self.visited = np.zeros(self.num_customers, dtype=bool)
        self.current_pos = self.depot.copy()
        self.current_time = 0
        self.current_load = 0
        return self.get_state()

    def step(self, action):
        """
        Take action and return next state, reward, and done flag
        Includes time window and capacity constraints
        """
        if action >= self.num_customers or self.visited[action]:
            return self.get_state(), -50, True

        # Calculate travel time/distance
        next_pos = self.customers[action]
        travel_time = np.linalg.norm(self.current_pos - next_pos)

        # Update current time
        arrival_time = self.current_time + travel_time

        # Check time window constraints
        if arrival_time > self.due_times[action]:
            return self.get_state(), -100, True  # Late arrival penalty

        # Wait if arrived before ready time
        if arrival_time < self.ready_times[action]:
            arrival_time = self.ready_times[action]

        # Check capacity constraint
        if self.current_load + self.demands[action] > self.vehicle_capacity:
            return self.get_state(), -100, True  # Capacity violation penalty

        # Update state
        self.current_pos = next_pos
        self.current_time = arrival_time + self.service_times[action]
        self.current_load += self.demands[action]
        self.visited[action] = True

        # Calculate reward
        reward = -travel_time  # Basic distance-based reward

        # Check if all customers are visited
        done = np.all(self.visited)
        if done:
            # Add return to depot
            final_distance = np.linalg.norm(self.current_pos - self.depot)
            reward -= final_distance

            # Check if route ends within time windows
            if self.current_time + final_distance > max(self.due_times):
                reward -= 100  # Penalty for late completion

        return self.get_state(), reward, done

# Gym Wrapper for VRPEnvironment
class GymVRPEnvironment(gym.Env):
    def __init__(self, num_customers=10, grid_size=50, solomon_data=None):
        super(GymVRPEnvironment, self).__init__()
        self.env = VRPEnvironment(num_customers, grid_size, solomon_data)

        # Define action space
        self.action_space = spaces.Discrete(self.env.num_customers)

        # Update observation space based on actual state
        state_dim = len(self.env.get_state())
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(state_dim,),
            dtype=np.float32
        )

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
    def __init__(self, env):
        self.env = DummyVecEnv([lambda: env])
        self.model = PPO("MlpPolicy", self.env, verbose=0)

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
    maxent_agent = MaxEntRL(state_dim, action_dim)
    sac_agent = SACAgent(state_dim, action_dim)
    dqn_agent = DQNAgent(state_dim, action_dim)
    nn_solver = NearestNeighbor()
    ppo_env = GymVRPEnvironment(num_customers=env_size)  # Create separate env for PPO
    ppo_agent = PPOAgent(ppo_env)
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
    ppo_agent.model.learn(total_timesteps=episodes * 100)

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
# methods = {
#     "DQN": lambda env: [int(dqn_agent.select_action(env.reset())) for _ in range(env.env.num_customers)],
#     "MaxEnt": lambda env: [int(maxent_agent.select_action(env.reset())) for _ in range(env.env.num_customers)],
#     "SAC": lambda env: [int(sac_agent.select_action(env.reset())) for _ in range(env.env.num_customers)],
#     "PPO": lambda env: [int(ppo_agent.model.predict(env.reset())[0]) for _ in range(env.env.num_customers)],
#     "Nearest Neighbor": lambda env: nn_solver.solve(env.env),  # Already returns a list of actions
#     "OR-Tools": lambda env: or_solver.solve(env.env),          # Already returns a list of actions
# }
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

import time

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

def evaluate_distances(env, methods, episodes=10):
    """Evaluate methods over multiple episodes"""
    all_distances = {method: [] for method in methods.keys()}

    for episode in range(episodes):
        for method_name, method_fn in methods.items():
            route = get_route_from_method(env, method_name, method_fn)
            distance = compute_route_distance(env, route)
            all_distances[method_name].append(distance)

    # Compute averages
    avg_distances = {method: np.mean(distances) for method, distances in all_distances.items()}
    return avg_distances

def evaluate_performance(env, methods, episodes=10):
    """Evaluate both distances and execution times"""
    all_distances = {method: [] for method in methods.keys()}
    all_times = {method: [] for method in methods.keys()}

    for episode in range(episodes):
        for method_name, method_fn in methods.items():
            start_time = time.time()
            route = get_route_from_method(env, method_name, method_fn)
            end_time = time.time()

            distance = compute_route_distance(env, route)
            execution_time = end_time - start_time

            all_distances[method_name].append(distance)
            all_times[method_name].append(execution_time)

    # Compute averages
    avg_distances = {method: np.mean(distances) for method, distances in all_distances.items()}
    avg_times = {method: np.mean(times) for method, times in all_times.items()}


# def evaluate_distances(env, methods, episodes=10):
#     """
#     Evaluate the total distances for each method over a number of episodes.
#     """
#     distances = {method: [] for method in methods.keys()}

#     for _ in range(episodes):
#         env.reset()

#         for method, action_fn in methods.items():
#             actions = action_fn(env)
#             total_distance = compute_total_distance(env, actions)
#             distances[method].append(total_distance)

#     # Compute average distances for each method
#     avg_distances = {method: np.mean(dist) for method, dist in distances.items()}
#     return avg_distances

# Evaluate distances
print("\nEvaluating Total Distances...")
avg_distances = evaluate_distances(GymVRPEnvironment(num_customers=10), methods)

# Generate a table of results
results_table = pd.DataFrame({
    "Method": list(avg_distances.keys()),
    "Average Distance": list(avg_distances.values())
})
print("\nResults Table:")
print(results_table)

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
results_table.to_csv("vrp_performance.csv", index=False)
print("Performance results saved to 'vrp_performance.csv'.")

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

test_env = GymVRPEnvironment(num_customers=10)

print("\nEvaluating Total Distances...")
avg_distances = evaluate_distances(test_env, methods)

print("\nDistances Table:")
distances_df = pd.DataFrame({
    "Method": list(avg_distances.keys()),
    "Average Distance": list(avg_distances.values())
})
print(distances_df)

print("\nEvaluating Performance...")
avg_distances, avg_times = evaluate_performance(test_env, methods)

print("\nPerformance Table:")
performance_df = pd.DataFrame({
    "Method": list(avg_distances.keys()),
    "Average Distance": list(avg_distances.values()),
    "Average Time (s)": list(avg_times.values())
})
print(performance_df)

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
