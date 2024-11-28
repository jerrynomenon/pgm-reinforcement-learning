import numpy as np
import random
from typing import List, Tuple
from utils import ExperimentTracker

class VehicleRoutingEnv:
    def __init__(self, num_locations: int, num_vehicles: int, temperature: float = 1.0):
        """Initialize the vehicle routing environment with temperature for probabilistic inference."""
        self.num_locations = num_locations
        self.num_vehicles = num_vehicles
        self.temperature = temperature  # Controls the influence of rewards in inference

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
        self.trajectory = []  # To store (state, action, reward) tuples
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
            else:
                # All vehicles have returned to depot
                pass
    
        # Update visited and current location
        self.visited[action] = True
        self.current_location = action
    
        # Append to trajectory
        self.trajectory.append((self._get_state(), action, cost))
    
        # Check if done - all locations visited or no more vehicles
        done = all(self.visited) or self.current_vehicle >= self.num_vehicles
    
        return self._get_state(), cost, done, {}