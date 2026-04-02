# =========================
# FINAL: ROLE-EMERGENT, HIERARCHICAL, ACTION-DRIVEN
# MULTI-AGENT INTELLIGENCE SYSTEM
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
    num_agents = 6
    lr = 5e-4
    goal_scale = 2.0
    comm_dim = 32
    grad_clip = 1.0
    noise_scale = 0.01
    step_scale = 0.2

cfg = Config()

# =========================
# CORE NETWORKS
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
# COMMUNICATION (LEARNED PROTOCOL)
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
# GNN SOCIAL LAYER
# =========================

class GNNLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(cfg.hidden_dim, cfg.hidden_dim)

    def forward(self, h, adj):
        return F.relu(self.linear(torch.matmul(adj, h)))

# =========================
# ROLE ASSIGNMENT (EMERGENT)
# =========================

class RoleHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Linear(cfg.hidden_dim, 3)  # 3 roles: explorer, exploiter, coordinator

    def forward(self, h):
        return F.softmax(self.net(h), dim=-1)

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
        self.roles = RoleHead()

        self.action_map = nn.Linear(cfg.action_dim, cfg.state_dim)

        # global + individual goals
        self.global_goal = F.normalize(torch.randn(cfg.state_dim), dim=0) * cfg.goal_scale
        self.local_goals = F.normalize(torch.randn(cfg.num_agents, cfg.state_dim), dim=1) * cfg.goal_scale

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
        role_probs = self.roles(H)

        return H, actions, values, role_probs

    def environment_step(self, states, actions):
        movement = self.action_map(actions)
        return states + cfg.step_scale * movement

    def compute_loss(self, states, next_states, H, actions, values, role_probs):
        # local goal reward (specialization)
        local_dist = torch.norm(next_states - self.local_goals, dim=1)
        local_reward = -local_dist

        # global coordination
        global_dist = torch.norm(next_states.mean(dim=0) - self.global_goal)
        global_reward = -global_dist

        # cooperation
        sim = torch.matmul(H, H.T).mean()

        # role entropy (encourage differentiation)
        role_entropy = -(role_probs * torch.log(role_probs + 1e-8)).sum(dim=1).mean()

        rewards = local_reward + 0.5 * global_reward + 0.05 * sim + 0.1 * role_entropy

        value_loss = F.mse_loss(values.squeeze(), rewards.detach())
        policy_loss = -rewards.mean()

        loss = value_loss + policy_loss

        return loss, rewards.mean().item(), sim.item()

    def step(self):
        states = torch.randn(cfg.num_agents, cfg.state_dim)

        H, actions, values, roles = self.forward(states)

        next_states = self.environment_step(states, actions)

        loss, reward, coop = self.compute_loss(states, next_states, H, actions, values, roles)

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
    plt.title("Role-Emergent Multi-Agent Intelligence")
    plt.xlabel("Steps")
    plt.ylabel("Metrics")
    plt.show()

# =========================
# README.md
# =========================

