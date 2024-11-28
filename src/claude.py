import seaborn as sns
import numpy as np
from typing import List, Dict, Tuple
from matplotlib.patches import Circle
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8-colorblind')
import pandas as pd
import random
from utils import ExperimentTracker

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
    
    # New methods to expose transition and reward probabilities
    def get_transition_prob(self, state: np.ndarray, action: int, next_state: np.ndarray) -> float:
        """
        Returns the probability of transitioning to next_state given state and action.
        For deterministic environments, this is either 1 or 0.
        """
        # Since the environment is deterministic, check if next_state is achievable
        # Implement logic to compare the environment's step to verify
        # This requires replicating the step logic
        # For simplicity, return 1.0 if the action is valid and leads to next_state, else 0.0
        temp_env = VehicleRoutingEnv(self.num_locations, self.num_vehicles)
        temp_env.locations = self.locations.copy()
        temp_env.demands = self.demands.copy()
        temp_env.vehicle_capacity = self.vehicle_capacity
        temp_env.current_vehicle = self.current_vehicle
        temp_env.visited = self.visited.copy()
        temp_env.vehicle_loads = self.vehicle_loads.copy()
        temp_env.current_location = self.current_location
        
        _, _, done, _ = temp_env.step(action)
        generated_next_state = temp_env._get_state()
        if np.array_equal(generated_next_state, next_state):
            return 1.0
        else:
            return 0.0

    def get_reward_prob(self, state: np.ndarray, action: int, reward: float) -> float:
        """
        Returns the probability of receiving a reward given state and action.
        For deterministic environments, this is either 1 or 0.
        """
        # Similar to get_transition_prob
        temp_env = VehicleRoutingEnv(self.num_locations, self.num_vehicles)
        temp_env.locations = self.locations.copy()
        temp_env.demands = self.demands.copy()
        temp_env.vehicle_capacity = self.vehicle_capacity
        temp_env.current_vehicle = self.current_vehicle
        temp_env.visited = self.visited.copy()
        temp_env.vehicle_loads = self.vehicle_loads.copy()
        temp_env.current_location = self.current_location
        
        _, generated_reward, _, _ = temp_env.step(action)
        if generated_reward == reward:
            return 1.0
        else:
            return 0.0

class MaxEntropyRL:
    """
    Implementation of maximum entropy RL as described in the paper.
    Key equations:
    p(τ|O1:T) ∝ p(τ)exp(Σ r(st,at)) (Equation 4)
    Q(st,at) = r(st,at) + Est+1[V(st+1)] (Equation 15)
    π(at|st) = exp(Q(st,at) - V(st)) (Equation 16)
    """
    def __init__(self, env: VehicleRoutingEnv, learning_rate: float = 0.001):
        self.env = env
        self.learning_rate = learning_rate
        self.state_dim = env.num_locations * 2 + 1
        self.action_dim = env.num_locations
        # Initialize policy weights for linear approximation of Q(s,a)
        self.policy_weights = np.random.randn(self.state_dim, self.action_dim) * 0.1

    def get_action_probabilities(self, state: np.ndarray) -> np.ndarray:
        """
        Implements the policy π(at|st) = exp(Q(st,at) - V(st)) (Equation 16)
        Where Q(s,a) is approximated by linear function approximation
        """
        logits = state @ self.policy_weights  # Q(s,a) approximation
        probs = np.exp(logits / 0.5)  # Temperature parameter for entropy weight
        normalized_probs = probs / np.sum(probs)  # Normalized probabilities
        # Handle any numerical instabilities
        normalized_probs = np.nan_to_num(normalized_probs, 0)
        normalized_probs = normalized_probs / np.sum(normalized_probs)
        return normalized_probs

    def select_action(self, state: np.ndarray) -> int:
        """
        Selects an action according to the policy distribution
        Uses the policy from Equation 16 in the paper
        """
        probs = self.get_action_probabilities(state)
        try:
            return np.random.choice(self.action_dim, p=probs)
        except ValueError as e:
            print(f"Warning: Issue with probability distribution: {probs}")
            print(f"Selecting random action due to: {e}")
            return np.random.randint(self.action_dim)

    def update_policy(self, state: np.ndarray, action: int, reward: float):
        """
        Implements policy gradient with entropy regularization:
        ∇J(θ) = E[∇log π(at|st)(r(st,at) + αH(π))] (Section 4.1)
        """
        probs = self.get_action_probabilities(state)
        grad = np.zeros_like(self.policy_weights)
        grad[:, action] = state  # ∇log π(at|st)
        entropy_grad = -np.log(probs + 1e-10)  # Entropy gradient with numerical stability
        
        # Combined gradient with entropy regularization weight α=0.1
        total_grad = reward * grad + 0.1 * entropy_grad
        self.policy_weights += self.learning_rate * total_grad

    def compute_value(self, state: np.ndarray) -> float:
        """
        Computes V(st) = log ∫ exp(Q(st,at))dat (Equation 21)
        Using the soft-maximum over actions
        """
        logits = state @ self.policy_weights
        return np.log(np.sum(np.exp(logits / 0.5))) * 0.5  # Temperature-scaled value function

def train_with_tracking(episodes: int = 1000):
    """Training with comprehensive tracking and visualization"""
    tracker = ExperimentTracker()
    env = VehicleRoutingEnv(num_locations=5, num_vehicles=2)
    agent = MaxEntropyRL(env, learning_rate=0.01)  # Increased learning rate
    visualizer = VehicleRoutingVisualizer(env)

    # Save experiment parameters
    tracker.metrics["parameters"] = {
        "num_locations": env.num_locations,
        "num_vehicles": env.num_vehicles,
        "vehicle_capacity": env.vehicle_capacity,
        "learning_rate": agent.learning_rate,
        "episodes": episodes
    }

    best_reward = float('-inf')
    best_route = None
    running_reward = 0
    reward_history = []

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        current_route = []
        current_loads = []
        episode_probs = []
        step_count = 0
        
        while not done and step_count < 100:  # Add step limit
            step_count += 1
            probs = agent.get_action_probabilities(state)
            action = agent.select_action(state)
            
            # Safe tracking of vehicle loads
            if env.current_vehicle < env.num_vehicles:
                current_loads.append(float(env.vehicle_loads[env.current_vehicle]))
            else:
                current_loads.append(float(env.vehicle_capacity))
            
            next_state, reward, done, _ = env.step(action)
            
            # Track everything
            current_route.append(int(action))
            episode_probs.append(probs.tolist())  # Convert to list for JSON
            
            agent.update_policy(state, action, reward)
            state = next_state
            total_reward += reward

        # Update running reward
        running_reward = 0.95 * running_reward + 0.05 * total_reward
        reward_history.append(running_reward)

        # Update tracking with proper type conversion
        tracker.metrics["training_rewards"].append(float(total_reward))
        tracker.metrics["routes"].append(current_route)
        tracker.metrics["vehicle_loads"].append(current_loads)
        tracker.metrics["action_probs"].append(episode_probs)
        
        # Update visualizer stats
        visualizer.update_stats(float(total_reward), current_route)
        
        if total_reward > best_reward:
            best_reward = total_reward
            best_route = current_route.copy()  # Make a copy to be safe

        # Generate and save plots periodically
        if episode % 100 == 0:
            print(f"Episode {episode}, Total Reward: {total_reward:.2f}, Running Reward: {running_reward:.2f}")
            
            try:
                route_fig = visualizer.plot_locations(current_route, f"Route at Episode {episode}")
                tracker.save_plot(route_fig, f"route_episode_{episode}")
                
                progress_fig = visualizer.plot_training_progress()
                tracker.save_plot(progress_fig, f"training_progress_{episode}")
                
                loads_fig = visualizer.plot_vehicle_loads([current_loads])
                tracker.save_plot(loads_fig, f"vehicle_loads_{episode}")
            except Exception as e:
                print(f"Warning: Error generating plots at episode {episode}: {e}")

    # Save final metrics
    tracker.save_metrics()
    
    # Generate and save final visualization
    try:
        final_fig = visualizer.plot_locations(best_route, "Best Route Found")
        tracker.save_plot(final_fig, "final_best_route")
    except Exception as e:
        print(f"Warning: Error generating final plot: {e}")

    return tracker, best_route, best_reward

if __name__ == "__main__":
    tracker, best_route, best_reward = train_with_tracking()


# def train_with_visualization(episodes: int = 1000):
#     env = VehicleRoutingEnv(num_locations=5, num_vehicles=2)
#     agent = MaxEntropyRL(env)
#     visualizer = VehicleRoutingVisualizer(env)
    
#     best_reward = float('-inf')
#     best_route = None
#     route_loads = []
    
#     # Plot initial state
#     visualizer.plot_locations(title="Initial Problem Setup")
#     visualizer.plot_distance_matrix()
    
#     for episode in range(episodes):
#         state = env.reset()
#         total_reward = 0
#         done = False
#         current_route = []
#         current_loads = []
        
#         while not done:
#             action = agent.select_action(state)
#             next_state, reward, done, _ = env.step(action)
#             agent.update_policy(state, action, reward)
            
#             current_route.append(action)
#             current_loads.append(env.vehicle_loads[env.current_vehicle])
            
#             state = next_state
#             total_reward += reward
        
#         visualizer.update_stats(total_reward, current_route)
#         route_loads.append(current_loads)
        
#         if total_reward > best_reward:
#             best_reward = total_reward
#             best_route = current_route
        
#         if episode % 100 == 0:
#             print(f"Episode {episode}, Total Reward: {total_reward:.2f}")
    
#     # Generate final visualizations
#     print("\nTraining completed. Generating final visualizations...")
    
#     visualizer.plot_training_progress()
#     visualizer.plot_locations(best_route, "Best Route Found")
#     visualizer.plot_vehicle_loads(route_loads[-5:])  # Show last 5 episodes
    
#     # Calculate total distance for best route
#     total_distance = sum(env._get_distance(best_route[i], best_route[i+1]) 
#                         for i in range(len(best_route)-1))
    
#     visualizer.create_summary_dashboard(best_route, best_reward, total_distance)
    
#     return visualizer, best_route, best_reward

# if __name__ == "__main__":
#     visualizer, best_route, best_reward = train_with_visualization()