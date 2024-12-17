import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import random

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
