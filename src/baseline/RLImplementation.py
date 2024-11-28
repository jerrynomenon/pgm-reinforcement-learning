import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import List, Tuple

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
