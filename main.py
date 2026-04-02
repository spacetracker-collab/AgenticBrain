# =========================
# FINAL: ACTION-DRIVEN EMERGENT MULTI-AGENT SYSTEM
# (Adds real environment + causality + RL loop)
# =========================

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# =========================
# CONFIG
# =========================

class Config:
    state_dim = 64
    hidden_dim = 128
    action_dim = 8
    num_agents = 5
    lr = 5e-4
    goal_scale = 2.0
    comm_dim = 32
    grad_clip = 1.0
    noise_scale = 0.01
    step_scale = 0.2

cfg = Config()

# =========================
# CORE
# =========================

class ReasoningCore(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.state_dim + cfg.comm_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim)
        )

    def forward(self, x, comm):
        return self.net(torch.cat([x, comm], dim=-1))

class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Linear(cfg.hidden_dim, cfg.action_dim)

    def forward(self, x):
        return torch.tanh(self.net(x))

class Value(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Linear(cfg.hidden_dim, 1)

    def forward(self, x):
        return self.net(x)

# =========================
# COMMUNICATION
# =========================

class CommunicationModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(cfg.hidden_dim, cfg.comm_dim)

    def forward(self, H):
        msgs = F.normalize(self.encoder(H), dim=1)
        global_msg = msgs.mean(dim=0, keepdim=True)
        return global_msg.repeat(cfg.num_agents, 1)

# =========================
# GNN
# =========================

class GNNLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(cfg.hidden_dim, cfg.hidden_dim)

    def forward(self, h, adj):
        return F.relu(self.linear(torch.matmul(adj, h)))

# =========================
# SYSTEM
# =========================

class MultiAgentSystem(nn.Module):
    def __init__(self):
        super().__init__()

        self.core = ReasoningCore()
        self.policy = Policy()
        self.value = Value()
        self.comm = CommunicationModule()
        self.gnn = GNNLayer()

        self.action_map = nn.Linear(cfg.action_dim, cfg.state_dim)

        self.goal = F.normalize(torch.randn(cfg.state_dim), dim=0) * cfg.goal_scale

        self.optimizer = torch.optim.Adam(self.parameters(), lr=cfg.lr)

        self.reward_history = []
        self.coop_history = []
        self.loss_history = []

    def forward(self, states):
        comm_init = torch.zeros(cfg.num_agents, cfg.comm_dim)
        H = self.core(states, comm_init)

        comm = self.comm(H)
        H = self.core(states, comm)

        adj = torch.ones(cfg.num_agents, cfg.num_agents) / cfg.num_agents
        H = self.gnn(H, adj)

        H = H + torch.randn_like(H) * cfg.noise_scale
        H = F.normalize(H, dim=1)

        actions = self.policy(H)
        values = self.value(H)

        return H, actions, values

    def environment_step(self, states, actions):
        movement = self.action_map(actions)
        next_states = states + cfg.step_scale * movement
        return next_states

    def compute_loss(self, states, next_states, H, actions, values):
        dist = torch.norm(next_states - self.goal, dim=1)
        reward = -dist

        # cooperation
        sim = torch.matmul(H, H.T).mean()

        rewards = reward + 0.05 * sim

        value_loss = F.mse_loss(values.squeeze(), rewards.detach())

        policy_loss = -rewards.mean()

        loss = value_loss + policy_loss

        return loss, rewards.mean().item(), sim.item()

    def step(self):
        states = torch.randn(cfg.num_agents, cfg.state_dim)

        H, actions, values = self.forward(states)

        next_states = self.environment_step(states, actions)

        loss, reward, coop = self.compute_loss(states, next_states, H, actions, values)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), cfg.grad_clip)
        self.optimizer.step()

        self.reward_history.append(reward)
        self.coop_history.append(coop)
        self.loss_history.append(loss.item())

        return reward, coop, loss.item()

# =========================
# TRAIN
# =========================

if __name__ == "__main__":
    system = MultiAgentSystem()

    for step in range(1000):
        reward, coop, loss = system.step()

        if step % 100 == 0:
            print(f"Step {step} | Reward: {reward:.4f} | Coop: {coop:.4f} | Loss: {loss:.4f}")

    plt.figure()
    plt.plot(system.reward_history, label="Reward")
    plt.plot(system.coop_history, label="Cooperation")
    plt.plot(system.loss_history, label="Loss")
    plt.legend()
    plt.title("Action-Driven Emergent Intelligence")
    plt.xlabel("Steps")
    plt.ylabel("Metrics")
    plt.show()

# =========================
# README.md
# =========================

