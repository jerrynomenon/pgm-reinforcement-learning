import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import random

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
        
