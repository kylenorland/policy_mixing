import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gym

# Hyperparameters
EPISODES = 1000
BATCH_SIZE = 32
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.001
TARGET_UPDATE = 10

# Define the DQN model
class DQN(nn.Module):
    def __init__(self, num_inputs, num_actions):
        super(DQN, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(num_inputs, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, num_actions)
        )
        
    def forward(self, x):
        return self.layers(x)

# Define the replay buffer
class ReplayBuffer():
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        
    def push(self, state, action, reward, next_state, done):
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size):

        batch = random.sample(self.buffer, batch_size)
        
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)

# Define the DQN agent
class DQNAgent():
    def __init__(self, env):
        self.env = env
        self.num_inputs = env.observation_space.shape[0]
        self.num_actions = env.action_space.n
        self.policy_net = DQN(self.num_inputs, self.num_actions)
        self.target_net = DQN(self.num_inputs, self.num_actions)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters())
        self.replay_buffer = ReplayBuffer(capacity=10000)
        self.steps = 0
        
    def act(self, state, epsilon):
        if np.random.rand() < epsilon:
            return self.env.action_space.sample()
        state = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.policy_net(state)
        _, action = torch.max(q_values, dim=1)
        return action.item()
    
    def optimize_model(self):
        if len(self.replay_buffer) < BATCH_SIZE:
            return
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(BATCH_SIZE)
        
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones).unsqueeze(1)
        
        q_values = self.policy_net(states).gather(1, actions)
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(dim=1, keepdim=True)[0]
        expected_q_values = rewards + GAMMA * next_q_values * (1 - dones)
        
        loss = nn.MSELoss()(q_values, expected_q_values.detach())
        
        self.optimizer.zero_grad()
        loss.backward()
        
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(),100)
        self.optimizer.step()
        
