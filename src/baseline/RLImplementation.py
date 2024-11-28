import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import List, Tuple
from baseline.VehicleEnv import VehicleRoutingEnv
from baseline.Visualizer import VehicleRoutingVisualizer
from baseline.DataTracking import ExperimentTracker

class MaxEntropyRL:
    """
    Maximum Entropy Reinforcement Learning Agent using Probabilistic Inference.
    Implements the policy as a neural network with variational inference.
    """
    def __init__(self, env: VehicleRoutingEnv, learning_rate: float = 0.001):
        self.env = env
        self.learning_rate = learning_rate
        self.state_dim = env.num_locations * 2 + 1
        self.action_dim = env.num_locations

        # Define the policy network
        self.policy_network = nn.Sequential(
            nn.Linear(self.state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, self.action_dim)
        )

        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=self.learning_rate)

    def get_action_probabilities(self, state: np.ndarray) -> np.ndarray:
        """
        Computes the action probabilities using the policy network.
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0)  # Shape: [1, state_dim]
        logits = self.policy_network(state_tensor) / self.env.temperature  # Apply temperature
        probs = F.softmax(logits, dim=-1).detach().numpy().flatten()
        return probs

    def select_action(self, state: np.ndarray) -> int:
        """
        Selects an action based on the current policy.
        """
        probs = self.get_action_probabilities(state)
        action = np.random.choice(self.action_dim, p=probs)
        return action

    def update_policy(self, trajectory: List[Tuple[np.ndarray, int, float]]):
        """
        Updates the policy network using the collected trajectory.
        Implements the ELBO optimization.
        """
        # Convert trajectory to tensors
        states = torch.FloatTensor([step[0] for step in trajectory])  # Shape: [T, state_dim]
        actions = torch.LongTensor([step[1] for step in trajectory])  # Shape: [T]
        rewards = torch.FloatTensor([step[2] for step in trajectory])  # Shape: [T]

        # Compute log probabilities
        logits = self.policy_network(states) / self.env.temperature  # Shape: [T, action_dim]
        log_probs = F.log_softmax(logits, dim=-1)  # Shape: [T, action_dim]
        selected_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)  # Shape: [T]

        # Compute entropy
        probs = F.softmax(logits, dim=-1)  # Shape: [T, action_dim]
        entropy = -(log_probs * probs).sum(dim=1)  # Shape: [T]

        # Compute the objective (ELBO)
        # Here, we maximize rewards and entropy
        objective = (selected_log_probs * rewards + 0.1 * entropy).mean()

        # Perform gradient ascent (minimize -objective)
        self.optimizer.zero_grad()
        loss = -objective
        loss.backward()
        self.optimizer.step()

def train_with_probabilistic_inference(episodes: int = 1000):
    """Training with Probabilistic Inference Framework and Comprehensive Tracking"""
    tracker = ExperimentTracker()
    env = VehicleRoutingEnv(num_locations=5, num_vehicles=2, temperature=1.0)
    agent = MaxEntropyRL(env, learning_rate=0.001)  # Adjusted learning rate
    visualizer = VehicleRoutingVisualizer(env)

    # Save experiment parameters
    tracker.metrics["parameters"] = {
        "num_locations": env.num_locations,
        "num_vehicles": env.num_vehicles,
        "vehicle_capacity": env.vehicle_capacity,
        "learning_rate": agent.learning_rate,
        "episodes": episodes,
        "temperature": env.temperature
    }

    best_reward = float('-inf')
    best_route = None
    running_reward = 0
    reward_history = []

    for episode in range(1, episodes + 1):
        state = env.reset()
        total_reward = 0
        done = False
        trajectory = []

        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            trajectory.append((state, action, reward))
            state = next_state
            total_reward += reward

        # Update policy after episode
        agent.update_policy(trajectory)

        # Update running reward
        running_reward = 0.95 * running_reward + 0.05 * total_reward
        reward_history.append(running_reward)

        # Update tracking with proper type conversion
        tracker.metrics["training_rewards"].append(float(total_reward))
        tracker.metrics["routes"].append([step[1] for step in trajectory])
        # Additional tracking can be added as needed

        # Update visualizer stats
        visualizer.update_stats(float(total_reward), [step[1] for step in trajectory])

        if total_reward > best_reward:
            best_reward = total_reward
            best_route = [step[1] for step in trajectory].copy()  # Make a copy to be safe

        # Generate and save plots periodically
        if episode % 100 == 0 or episode == 1:
            print(f"Episode {episode}, Total Reward: {total_reward:.2f}, Running Reward: {running_reward:.2f}")

            try:
                route_fig = visualizer.plot_locations(best_route, f"Best Route at Episode {episode}")
                tracker.save_plot(route_fig, f"best_route_episode_{episode}")
                
                progress_fig = visualizer.plot_training_progress()
                tracker.save_plot(progress_fig, f"training_progress_{episode}")
                
                # Add more plots as needed
            except Exception as e:
                print(f"Warning: Error generating plots at episode {episode}: {e}")

    # Save final metrics
    tracker.save_metrics()
    
    # Generate and save final visualization
    try:
        final_fig = visualizer.plot_locations(best_route, "Final Best Route Found")
        tracker.save_plot(final_fig, "final_best_route")
    except Exception as e:
        print(f"Warning: Error generating final plot: {e}")

    return tracker, best_route, best_reward

if __name__ == "__main__":
    tracker, best_route, best_reward = train_with_probabilistic_inference(episodes=1000)
