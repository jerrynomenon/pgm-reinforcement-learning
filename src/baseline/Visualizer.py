import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple

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