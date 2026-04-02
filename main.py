# =========================
# TASK-DRIVEN EMERGENT COOPERATIVE MULTI-AGENT SYSTEM
# (Breaks symmetry + adds structure + enables real emergence)
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
    lr = 5e-4
    goal_scale = 3.0
    comm_dim = 64
    grad_clip = 1.0
    reg_weight = 1e-3
    noise_scale = 0.01

cfg = Config()

# =========================
# CORE
# =========================

class ReasoningCore(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.state_dim + cfg.comm_dim, cfg.hidden_dim),
            nn.LayerNorm(cfg.hidden_dim),
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
        return F.softmax(self.net(x), dim=-1)

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
        messages = F.normalize(self.encoder(H), dim=1)
        global_msg = messages.mean(dim=0, keepdim=True)
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

        self.goal = F.normalize(torch.randn(cfg.state_dim), dim=0) * cfg.goal_scale

        self.optimizer = torch.optim.Adam(self.parameters(), lr=cfg.lr)

        self.reward_history = []
        self.coop_history = []
        self.loss_history = []
        self.div_history = []

    def generate_structured_states(self):
        base = torch.randn(cfg.num_agents, cfg.state_dim)

        # Shared task direction
        task_dir = F.normalize(self.goal, dim=0)
        base += task_dir * 0.5

        # Agent-specific bias (break symmetry)
        bias = torch.randn(cfg.num_agents, cfg.state_dim) * 0.1
        base += bias

        return base

    def forward(self, states):
        comm_init = torch.zeros(cfg.num_agents, cfg.comm_dim)
        H = self.core(states, comm_init)

        comm = self.comm(H)
        H = self.core(states, comm)

        adj = torch.ones(cfg.num_agents, cfg.num_agents) / cfg.num_agents
        H = self.gnn(H, adj)

        # Symmetry breaking noise
        H = H + torch.randn_like(H) * cfg.noise_scale

        H = F.normalize(H, dim=1)

        probs = self.policy(H)
        values = self.value(H)

        return H, probs, values

    def compute_loss(self, states, H, probs, values):
        # Goal reward (stronger)
        dist = torch.norm(states - self.goal, dim=1)
        dist_reward = -dist / (1.0 + dist)

        # Cooperation
        sim_matrix = torch.matmul(H, H.T)
        coop = torch.mean(sim_matrix)

        # Diversity
        identity = torch.eye(cfg.num_agents)
        diversity = torch.mean((sim_matrix - identity)**2)

        # Specialization
        variance = torch.var(H, dim=0).mean()

        # Balanced reward
        rewards = 2.0 * dist_reward + 0.02 * coop - 0.05 * diversity + 0.05 * variance

        value_loss = F.mse_loss(values.squeeze(), rewards.detach())

        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1).mean()
        policy_loss = -rewards.mean() - 0.01 * entropy

        reg = cfg.reg_weight * torch.mean(H**2)

        total_loss = value_loss + policy_loss + reg

        return total_loss, rewards.mean().item(), coop.item(), diversity.item()

    def step(self):
        states = self.generate_structured_states()

        H, probs, values = self.forward(states)

        loss, reward, coop, div = self.compute_loss(states, H, probs, values)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), cfg.grad_clip)
        self.optimizer.step()

        self.reward_history.append(reward)
        self.coop_history.append(coop)
        self.loss_history.append(loss.item())
        self.div_history.append(div)

        return reward, coop, div, loss.item()

# =========================
# TRAIN
# =========================

if __name__ == "__main__":
    system = MultiAgentSystem()

    for step in range(1000):
        reward, coop, div, loss = system.step()

        if step % 100 == 0:
            print(f"Step {step} | Reward: {reward:.4f} | Coop: {coop:.4f} | Div: {div:.4f} | Loss: {loss:.4f}")

    plt.figure()
    plt.plot(system.reward_history, label="Reward")
    plt.plot(system.coop_history, label="Cooperation")
    plt.plot(system.div_history, label="Diversity")
    plt.plot(system.loss_history, label="Loss")
    plt.legend()
    plt.title("True Emergent Cooperative Intelligence")
    plt.xlabel("Steps")
    plt.ylabel("Metrics")
    plt.show()

# =========================
# README.md
# =========================

