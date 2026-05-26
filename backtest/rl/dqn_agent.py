import numpy as np
from collections import deque
import random

import torch
import torch.nn as nn
import torch.optim as optim


class QNetwork(nn.Module):
    def __init__(self, state_dim: int, hidden: int = 128, n_actions: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.FloatTensor(np.array(states)),
            torch.LongTensor(np.array(actions)).unsqueeze(1),
            torch.FloatTensor(np.array(rewards)).unsqueeze(1),
            torch.FloatTensor(np.array(next_states)),
            torch.FloatTensor(np.array(dones)).unsqueeze(1),
        )

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    def __init__(
        self,
        state_dim: int,
        n_actions: int = 3,
        hidden: int = 128,
        lr: float = 1e-5,
        gamma: float = 0.98,
        epsilon_start: float = 0.9,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 500,
        buffer_capacity: int = 10000,
        batch_size: int = 200,
        target_update: int = 50,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update
        self.step_count = 0
        self.losses = []

        self.q_net = QNetwork(state_dim, hidden, n_actions).to(self.device)
        self.target_net = QNetwork(state_dim, hidden, n_actions).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        self.memory = ReplayBuffer(buffer_capacity)

    def act(self, state: np.ndarray, eval_mode: bool = False) -> int:
        eps = self.epsilon_end if eval_mode else self.epsilon
        if np.random.random() < eps:
            return np.random.randint(self.n_actions)
        with torch.no_grad():
            s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q = self.q_net(s)
            return int(q.argmax().item())

    def learn(self):
        if len(self.memory) < self.batch_size:
            return
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        q_values = self.q_net(states).gather(1, actions)
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1, keepdim=True)[0]
            targets = rewards + self.gamma * next_q * (1 - dones)

        loss = self.criterion(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.losses.append(loss.item())

        self.step_count += 1
        self.epsilon = max(
            self.epsilon_end,
            self.epsilon - (0.9 - self.epsilon_end) / self.epsilon_decay
            if self.step_count < self.epsilon_decay
            else self.epsilon_end,
        )

        if self.step_count % self.target_update == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())
