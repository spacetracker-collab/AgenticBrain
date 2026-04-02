        
    # =========================
# FULLY DIFFERENTIABLE COOPERATIVE AGENTIC AI SYSTEM
# Single-file repo: main.py
# =========================

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# =========================
# CONFIG
# =========================

class Config:
    state_dim = 128
    hidden_dim = 256
    action_dim = 10
    num_agents = 5
    lr = 1e-3
    gamma = 0.95
    goal_scale = 5.0
    comm_dim = 64

cfg = Config()

# =========================
# SHARED MODULES (GLOBAL BRAIN)
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
        x = torch.cat([x, comm], dim=-1)
        return self.net(x)

class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Linear(cfg.hidden_dim, cfg.action_dim)

    def forward(self, x):
        return F.softmax(self.net(x), dim=-1)

class Value(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Linear(cfg.hidden_dim, 1)

    def forward(self, x):
        return self.net(x)

# =========================
# COMMUNICATION PROTOCOL (EMERGENT LANGUAGE)
# =========================

class CommunicationModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(cfg.hidden_dim, cfg.comm_dim)
        self.decoder = nn.Linear(cfg.comm_dim, cfg.hidden_dim)

    def forward(self, H):
        # encode messages
        messages = self.encoder(H)
        # global shared message (mean pooling)
        global_msg = messages.mean(dim=0, keepdim=True)
        # broadcast back
        comm = global_msg.repeat(cfg.num_agents, 1)
        return comm

# =========================
# GNN (DIFFERENTIABLE INTERACTION)
# =========================

class GNNLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(cfg.hidden_dim, cfg.hidden_dim)

    def forward(self, h, adj):
        return F.relu(self.linear(torch.matmul(adj, h)))

# =========================
# FULL SYSTEM (JOINT LEARNING)
# =========================

class MultiAgentSystem(nn.Module):
    def __init__(self):
        super().__init__()

        self.core = ReasoningCore()
        self.policy = Policy()
        self.value = Value()
        self.comm = CommunicationModule()
        self.gnn = GNNLayer()

        self.goal = torch.randn(cfg.state_dim) * cfg.goal_scale

        self.optimizer = torch.optim.Adam(self.parameters(), lr=cfg.lr)

        self.reward_history = []
        self.coop_history = []
        self.loss_history = []

    def forward(self, states):
        # initial hidden states
        comm_init = torch.zeros(cfg.num_agents, cfg.comm_dim)
        H = self.core(states, comm_init)

        # communication phase
        comm = self.comm(H)

        # reasoning with communication
        H = self.core(states, comm)

        # interaction phase
        adj = torch.ones(cfg.num_agents, cfg.num_agents) / cfg.num_agents
        H = self.gnn(H, adj)

        # policy + value
        probs = self.policy(H)
        values = self.value(H)

        return H, probs, values

    def compute_loss(self, states, H, probs, values):
        # cooperative reward
        dist = torch.norm(states - self.goal, dim=1)
        coop = torch.mean(torch.matmul(H, H.T))
        rewards = -dist + 0.1 * coop

        # value loss
        value_loss = F.mse_loss(values.squeeze(), rewards.detach())

        # policy loss (encourage actions aligned with reward)
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1).mean()
        policy_loss = -rewards.mean() - 0.01 * entropy

        total_loss = value_loss + policy_loss

        return total_loss, rewards.mean().item(), coop.item()

    def step(self):
        states = torch.randn(cfg.num_agents, cfg.state_dim)

        H, probs, values = self.forward(states)

        loss, avg_reward, coop = self.compute_loss(states, H, probs, values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.reward_history.append(avg_reward)
        self.coop_history.append(coop)
        self.loss_history.append(loss.item())

        return avg_reward, coop, loss.item()

# =========================
# TRAIN + VISUALIZE
# =========================

if __name__ == "__main__":
    system = MultiAgentSystem()

    for step in range(1000):
        reward, coop, loss = system.step()

        if step % 100 == 0:
            print(f"Step {step} | Reward: {reward:.3f} | Coop: {coop:.3f} | Loss: {loss:.3f}")

    # Visualization
    plt.figure()
    plt.plot(system.reward_history, label="Reward")
    plt.plot(system.coop_history, label="Cooperation")
    plt.plot(system.loss_history, label="Loss")
    plt.legend()
    plt.title("Emergent Cooperative Intelligence")
    plt.xlabel("Steps")
    plt.ylabel("Metrics")
    plt.show()

# =========================
# README.md
# =========================

