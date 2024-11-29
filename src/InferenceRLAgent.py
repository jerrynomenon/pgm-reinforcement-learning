import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import List, Tuple, Dict
from collections import deque
from matplotlib.patches import Circle
import pandas as pd

class ExperimentTracker:
    def __init__(self):
        self.metrics = {
            "training_rewards": [],
            "routes": [],
            "vehicle_loads": [],
            "action_probs": [],
            "parameters": {}
        }

    def save_plot(self, fig, name):
        fig.savefig(f"{name}.png")
        plt.close(fig)

    def save_metrics(self):
        pd.DataFrame(self.metrics).to_csv("metrics.csv", index=False)

class VehicleRoutingVisualizer:
    def __init__(self, env):
        self.env = env
        self.training_rewards = []
        self.route_histories = []
        self.visited_order = []
        plt.style.use('seaborn-v0_8-colorblind')
        
    def plot_locations(self, route: List[int] = None, title: str = "Depot and Delivery Locations"):
        """Plot all locations and optionally show a route."""
        fig = plt.figure(figsize=(12, 8))
        
        # Plot all locations
        locations = np.array(self.env.locations)
        plt.scatter(locations[1:, 0], locations[1:, 1], c='blue', s=100, label='Delivery Locations')
        plt.scatter(locations[0, 0], locations[0, 1], c='red', s=150, marker='*', label='Depot')
        
        # Add demand labels
        for i, (x, y) in enumerate(locations):
            demand = self.env.demands[i]
            plt.annotate(f'D:{demand}', (x, y), xytext=(5, 5), textcoords='offset points')
            
        # Plot route if provided
        if route:
            route_coords = locations[route]
            plt.plot(route_coords[:, 0], route_coords[:, 1], 'g--', linewidth=2, label='Route')
            
            # Add arrows to show direction
            for i in range(len(route)-1):
                start = route_coords[i]
                end = route_coords[i+1]
                dx = end[0] - start[0]
                dy = end[1] - start[1]
                plt.arrow(start[0], start[1], dx*0.2, dy*0.2, 
                         head_width=2, head_length=3, fc='g', ec='g')
        
        plt.title(title)
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.legend()
        plt.grid(True)
        return fig  # Return the figure instead of showing it

    def plot_training_progress(self):
        """Plot training rewards over time."""
        fig = plt.figure(figsize=(12, 6))
        
        # Plot episode rewards
        plt.plot(self.training_rewards, label='Episode Reward', alpha=0.5)
        
        # Add rolling average
        window_size = min(50, len(self.training_rewards))
        if window_size > 0:
            rolling_mean = pd.Series(self.training_rewards).rolling(window=window_size).mean()
            plt.plot(rolling_mean, label=f'{window_size}-Episode Moving Average', 
                    color='red', linewidth=2)
        
        plt.title('Training Progress')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.legend()
        plt.grid(True)
        return fig

    def plot_vehicle_loads(self, route_loads: List[List[float]]):
        """Plot vehicle load progression through routes."""
        fig = plt.figure(figsize=(12, 6))
        
        for vehicle_idx, loads in enumerate(route_loads):
            plt.plot(loads, marker='o', label=f'Vehicle {vehicle_idx + 1}')
            plt.axhline(y=self.env.vehicle_capacity, color='r', linestyle='--', 
                       label='Capacity' if vehicle_idx == 0 else '')
            
        plt.title('Vehicle Load Progression')
        plt.xlabel('Stop Number')
        plt.ylabel('Current Load')
        plt.legend()
        plt.grid(True)
        return fig

    def plot_action_probabilities(self, state_history: List[np.ndarray], 
                                  action_prob_history: List[np.ndarray]):
        """Plot action probability distributions over time."""
        fig = plt.figure(figsize=(15, 6))
        
        # Create heatmap of action probabilities
        action_probs = np.array(action_prob_history)
        sns.heatmap(action_probs.T, cmap='YlOrRd', 
                   xticklabels=range(1, len(action_probs) + 1),
                   yticklabels=range(self.env.num_locations))
        
        plt.title('Action Probability Distribution Over Time')
        plt.xlabel('Time Step')
        plt.ylabel('Location')
        return fig

    def update_stats(self, episode_reward: float, route: List[int]):
        """Update tracking statistics."""
        self.training_rewards.append(episode_reward)
        self.route_histories.append(route)

class VehicleRoutingEnv:
    def __init__(self, num_locations: int, num_vehicles: int):
        """Initialize the vehicle routing environment"""
        self.num_locations = num_locations
        self.num_vehicles = num_vehicles
        
        # Random coordinates for locations between 0 and 100
        self.locations = [(random.uniform(0, 100), random.uniform(0, 100)) 
                         for _ in range(num_locations)]
        self.demands = [random.randint(1, 10) for _ in range(num_locations)]
        self.vehicle_capacity = 30
        self.reset()

    def reset(self):
        """Reset the environment to initial state"""
        self.current_vehicle = 0
        self.visited = [False] * self.num_locations
        self.visited[0] = True  # Depot is always visited
        self.vehicle_loads = [0] * self.num_vehicles
        self.current_location = 0  # Start at depot
        return self._get_state()

    def _get_distance(self, loc1: int, loc2: int) -> float:
        """Calculate Euclidean distance between two locations"""
        x1, y1 = self.locations[loc1]
        x2, y2 = self.locations[loc2]
        return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    def _get_state(self) -> np.ndarray:
        """
        Get current state representation.
        Returns:
            state: numpy array containing:
                - normalized current vehicle load
                - visited status for each location
                - normalized demands for each location
        """
        state = np.zeros(self.num_locations * 2 + 1)
        
        # Safely handle vehicle index
        if self.current_vehicle < self.num_vehicles:
            state[0] = self.vehicle_loads[self.current_vehicle] / self.vehicle_capacity
        else:
            state[0] = 1.0  # Full load if we've exceeded vehicle count
            
        state[1:self.num_locations+1] = self.visited
        state[self.num_locations+1:] = [self.demands[i] / self.vehicle_capacity 
                                      for i in range(self.num_locations)]
        return state

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Take a step in the environment.
        Args:
            action: location index to visit
        Returns:
            state: new state
            reward: cost (negative distance)
            done: whether episode is finished
            info: additional information
        """
        # Check if we've exceeded vehicle count
        if self.current_vehicle >= self.num_vehicles:
            return self._get_state(), -100, True, {}
    
        # Can't visit already visited locations (except depot)
        if self.visited[action] and action != 0:
            return self._get_state(), -100, True, {}
    
        # Calculate distance-based cost
        distance = self._get_distance(self.current_location, action)
        cost = -distance
    
        # Update vehicle load
        if action != 0:  # If not returning to depot
            new_load = self.vehicle_loads[self.current_vehicle] + self.demands[action]
            if new_load > self.vehicle_capacity:
                return self._get_state(), -100, True, {}
            self.vehicle_loads[self.current_vehicle] = new_load
        else:  # Returning to depot
            self.vehicle_loads[self.current_vehicle] = 0
            if self.current_vehicle < self.num_vehicles - 1:
                self.current_vehicle += 1
    
        # Update visited and current location
        self.visited[action] = True
        self.current_location = action
    
        # Check if done - all locations visited or no more vehicles
        done = all(self.visited) or self.current_vehicle >= self.num_vehicles
    
        return self._get_state(), cost, done, {}
    
    def render(self) -> None:
        """Print current state for debugging"""
        print(f"\nVehicle {self.current_vehicle + 1}/{self.num_vehicles}")
        print(f"Current location: {self.current_location}")
        print(f"Visited locations: {[i for i, v in enumerate(self.visited) if v]}")
        print(f"Vehicle loads: {self.vehicle_loads}")

class InferenceRLAgent:
    def __init__(self, env: VehicleRoutingEnv, learning_rate: float = 0.001, gamma: float = 0.99, entropy_weight: float = 0.01):
        self.env = env
        self.learning_rate = learning_rate
        self.gamma = gamma  # Discount factor
        self.entropy_weight = entropy_weight  # Weight for entropy regularization

        # Define the policy network (Q-network)
        self.policy_net = nn.Sequential(
            nn.Linear(env.num_locations * 2 + 1, 128),
            nn.ReLU(),
            nn.Linear(128, env.num_locations),
            nn.Softmax(dim=-1)  # Ensures outputs are valid probabilities
        )

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.MSELoss()

    def get_action_probabilities(self, state: np.ndarray) -> np.ndarray:
        """
        Computes the policy's action probabilities for a given state.
        """
        state_tensor = torch.from_numpy(state).float().unsqueeze(0)  # Add batch dimension
        with torch.no_grad():
            probs = self.policy_net(state_tensor).squeeze(0).numpy()
        return probs

    def select_action(self, state: np.ndarray) -> int:
        """
        Selects an action based on the current policy's probabilities.
        """
        probs = self.get_action_probabilities(state)
        action = np.random.choice(self.env.num_locations, p=probs)
        return action

    def compute_elbo_loss(self, states: List[np.ndarray], actions: List[int], rewards: List[float], next_states: List[np.ndarray], dones: List[bool]):
        """
        Computes the Evidence Lower BOund (ELBO) for variational inference.
        """
        states_tensor = torch.tensor(states, dtype=torch.float32)
        actions_tensor = torch.tensor(actions, dtype=torch.long).unsqueeze(1)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
        next_states_tensor = torch.tensor(next_states, dtype=torch.float32)
        dones_tensor = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)

        # Forward pass to get current Q-values and action probabilities
        action_probs = self.policy_net(states_tensor)
        log_probs = torch.log(action_probs + 1e-10)  # Log probabilities for KL divergence

        current_q = action_probs.gather(1, actions_tensor)

        # Compute target Q-values using the next states
        with torch.no_grad():
            next_action_probs = self.policy_net(next_states_tensor)
            next_log_probs = torch.log(next_action_probs + 1e-10)
            # Soft Bellman update with entropy
            target_q = rewards_tensor + (1 - dones_tensor) * self.gamma * (next_action_probs * next_log_probs).sum(dim=1, keepdim=True)

        # Compute ELBO components
        # Expected log likelihood
        expected_log_q = (action_probs * log_probs).sum(dim=1, keepdim=True)
        # KL divergence between current policy and target
        kl_divergence = F.kl_div(log_probs, action_probs, reduction='batchmean')

        # ELBO: Expected log likelihood - KL divergence
        elbo = expected_log_q.mean() - self.entropy_weight * kl_divergence

        # Since we maximize ELBO, minimize -ELBO
        loss = -elbo
        return loss

    def update_policy(self, loss: torch.Tensor):
        """
        Performs a gradient descent step to update the policy network.
        """
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train_step(self, states: List[np.ndarray], actions: List[int], rewards: List[float], next_states: List[np.ndarray], dones: List[bool]):
        """
        Performs a full training step: computes loss and updates policy.
        """
        loss = self.compute_elbo_loss(states, actions, rewards, next_states, dones)
        self.update_policy(loss)
        return loss.item()

class ReplayBuffer:
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

def train_inference_rl(env: VehicleRoutingEnv, agent: InferenceRLAgent, episodes: int = 1000, batch_size: int = 64):
    replay_buffer = ReplayBuffer(capacity=10000)
    visualizer = VehicleRoutingVisualizer(env)
    tracker = ExperimentTracker()
    tracker.metrics["parameters"] = {
        "num_locations": env.num_locations,
        "num_vehicles": env.num_vehicles,
        "vehicle_capacity": env.vehicle_capacity,
        "learning_rate": agent.learning_rate,
        "episodes": episodes
    }
    best_reward = float('-inf')
    best_route = None

    for episode in range(1, episodes + 1):
        state = env.reset()
        done = False
        total_reward = 0
        episode_route = []
        episode_actions = []
        episode_rewards = []
        episode_next_states = []
        episode_dones = []

        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)

            # Store experience
            replay_buffer.push(state, action, reward, next_state, done)

            # Update tracking
            episode_route.append(action)
            episode_actions.append(action)
            episode_rewards.append(reward)
            episode_next_states.append(next_state)
            episode_dones.append(done)

            state = next_state
            total_reward += reward

        # Update best reward and route
        if total_reward > best_reward:
            best_reward = total_reward
            best_route = episode_route.copy()

        # Sample a batch and train the agent
        if len(replay_buffer) >= batch_size:
            batch = replay_buffer.sample(batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)
            loss = agent.train_step(states, actions, rewards, next_states, dones)
            tracker.metrics["training_rewards"].append(float(total_reward))
            tracker.metrics["routes"].append(episode_route)
            # Vehicle loads and action_probs can be tracked similarly
        else:
            tracker.metrics["training_rewards"].append(float(total_reward))
            tracker.metrics["routes"].append(episode_route)

        # Update visualizer stats
        visualizer.update_stats(float(total_reward), episode_route)

        # Periodic logging and visualization
        if episode % 100 == 0:
            print(f"Episode {episode}, Total Reward: {total_reward:.2f}, Best Reward: {best_reward:.2f}")
            try:
                route_fig = visualizer.plot_locations(episode_route, f"Route at Episode {episode}")
                tracker.save_plot(route_fig, f"route_episode_{episode}")
                
                progress_fig = visualizer.plot_training_progress()
                tracker.save_plot(progress_fig, f"training_progress_{episode}")
                
                # Assuming vehicle_loads tracking is handled within visualizer
                # loads_fig = visualizer.plot_vehicle_loads([current_loads])
                # tracker.save_plot(loads_fig, f"vehicle_loads_{episode}")
            except Exception as e:
                print(f"Warning: Error generating plots at episode {episode}: {e}")

    # Save final metrics
    tracker.save_metrics()

    # Generate and save final visualization
    if best_route:
        try:
            final_fig = visualizer.plot_locations(best_route, "Best Route Found")
            tracker.save_plot(final_fig, "final_best_route")
        except Exception as e:
            print(f"Warning: Error generating final plot: {e}")

    print(f"Training completed. Best Reward: {best_reward:.2f}")
    return best_route, best_reward

if __name__ == "__main__":
    env = VehicleRoutingEnv(num_locations=5, num_vehicles=2)
    agent = InferenceRLAgent(env, learning_rate=0.001, gamma=0.99, entropy_weight=0.01)
    best_route, best_reward = train_inference_rl(env, agent, episodes=1000, batch_size=64)
