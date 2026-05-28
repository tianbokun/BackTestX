import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class PPONetwork(nn.Module):
    def __init__(self, state_dim: int, hidden: int = 64):
        super().__init__()
        self.actor_base = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.mean = nn.Linear(hidden, 1)
        self.log_std = nn.Parameter(torch.zeros(1))

        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, state):
        features = self.actor_base(state)
        mean = self.mean(features)
        log_std = self.log_std.expand_as(mean)
        value = self.critic(state)
        return mean, log_std, value


class PPOMemory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []

    def push(self, state, action, log_prob, reward, done, value):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.dones.clear()
        self.values.clear()

    def __len__(self):
        return len(self.states)


class PPOAgent:
    def __init__(
        self,
        state_dim: int,
        hidden: int = 64,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        entropy_beta: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        ppo_epochs: int = 4,
        batch_size: int = 64,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_beta = entropy_beta
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size

        self.network = PPONetwork(state_dim, hidden).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        self.memory = PPOMemory()
        self.losses = []

    def act(self, state: np.ndarray, eval_mode: bool = False) -> tuple:
        with torch.no_grad():
            s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            mean, log_std, value = self.network(s)
            if eval_mode:
                position_ratio = float(torch.sigmoid(mean).cpu().numpy().squeeze())
                return position_ratio, 0.0, float(value.cpu().numpy().squeeze())
            std = log_std.exp()
            dist = torch.distributions.Normal(mean, std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum()
            position_ratio = float(torch.sigmoid(action).cpu().numpy().squeeze())
            log_prob_val = float(log_prob.cpu().numpy())
            value_val = float(value.cpu().numpy().squeeze())
        return position_ratio, log_prob_val, value_val

    def evaluate(self, states, actions):
        mean, log_std, values = self.network(states)
        std = log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        action_log_probs = dist.log_prob(actions).sum(dim=-1, keepdim=True)
        entropy = dist.entropy().sum(dim=-1, keepdim=True)
        return action_log_probs, values, entropy

    def _compute_gae(self, rewards, values, dones):
        advantages = []
        gae = 0
        values = values + [0]
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        returns = [adv + v for adv, v in zip(advantages, values[:-1])]
        return advantages, returns

    def learn(self):
        n = len(self.memory)
        if n < self.batch_size:
            return

        states = torch.FloatTensor(np.array(self.memory.states)).to(self.device)
        actions = torch.FloatTensor(np.array(self.memory.actions)).unsqueeze(1).to(self.device)
        old_log_probs = torch.FloatTensor(np.array(self.memory.log_probs)).unsqueeze(1).to(self.device)

        advantages, returns = self._compute_gae(
            self.memory.rewards, self.memory.values, self.memory.dones
        )
        advantages = torch.FloatTensor(np.array(advantages)).unsqueeze(1).to(self.device)
        returns = torch.FloatTensor(np.array(returns)).unsqueeze(1).to(self.device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        dataset_size = n
        for _ in range(self.ppo_epochs):
            indices = np.random.permutation(dataset_size)
            for start in range(0, dataset_size, self.batch_size):
                batch_idx = indices[start:start + self.batch_size]
                batch_states = states[batch_idx]
                batch_actions = actions[batch_idx]
                batch_old_log_probs = old_log_probs[batch_idx]
                batch_advantages = advantages[batch_idx]
                batch_returns = returns[batch_idx]

                new_log_probs, values, entropy = self.evaluate(batch_states, batch_actions)
                ratio = (new_log_probs - batch_old_log_probs).exp()
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = nn.MSELoss()(values, batch_returns)
                entropy_loss = -entropy.mean()
                total_loss = actor_loss + self.value_coef * critic_loss + self.entropy_beta * entropy_loss

                self.optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()

                self.losses.append(total_loss.item())

        self.memory.clear()

    def save(self, path: str):
        data = {
            "network_state": self.network.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "losses": self.losses,
        }
        import json
        from pathlib import Path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(data, path)

    @classmethod
    def load(cls, path: str, state_dim: int, hidden: int = 64):
        agent = cls(state_dim=state_dim, hidden=hidden)
        data = torch.load(path, map_location="cpu")
        agent.network.load_state_dict(data["network_state"])
        agent.optimizer.load_state_dict(data["optimizer_state"])
        agent.losses = data.get("losses", [])
        return agent
