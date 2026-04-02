# =========================
# FINAL AGENTIC AI (WITH DIVERSITY METRICS + ANTI-COLLAPSE)
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
    lr = 3e-4
    goal_scale = 2.0
    comm_dim = 32
    grad_clip = 1.0
    step_scale = 0.2

cfg = Config()

# =========================
# CORE
# =========================

class Core(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.state_dim + cfg.comm_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim)
        )

    def forward(self, s, c):
        return self.net(torch.cat([s, c], dim=-1))

class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Linear(cfg.hidden_dim, cfg.action_dim)

    def forward(self, h):
        return torch.tanh(self.net(h))

class Value(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Linear(cfg.hidden_dim, 1)

    def forward(self, h):
        return self.net(h)

# =========================
# COMM
# =========================

class Comm(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Linear(cfg.hidden_dim, cfg.comm_dim)

    def forward(self, H):
        m = F.normalize(self.enc(H), dim=1)
        g = m.mean(dim=0, keepdim=True)
        return g.repeat(cfg.num_agents, 1)

# =========================
# ROLE POLICIES
# =========================

class RolePolicies(nn.Module):
    def __init__(self):
        super().__init__()
        self.policies = nn.ModuleList([Policy(), Policy(), Policy()])
        self.role_head = nn.Linear(cfg.hidden_dim, 3)

    def forward(self, H):
        role_probs = F.softmax(self.role_head(H), dim=-1)
        actions = 0
        for i in range(3):
            actions += role_probs[:, i:i+1] * self.policies[i](H)
        return actions, role_probs

# =========================
# SYSTEM
# =========================

class AgenticSystem(nn.Module):
    def __init__(self):
        super().__init__()

        self.core = Core()
        self.comm = Comm()
        self.roles = RolePolicies()
        self.value = Value()

        self.action_map = nn.Linear(cfg.action_dim, cfg.state_dim)

        self.global_goal = F.normalize(torch.randn(cfg.state_dim), dim=0) * cfg.goal_scale
        self.local_goals = F.normalize(torch.randn(cfg.num_agents, cfg.state_dim), dim=1) * cfg.goal_scale

        self.opt = torch.optim.Adam(self.parameters(), lr=cfg.lr)

        # histories
        self.reward_hist = []
        self.coop_hist = []
        self.loss_hist = []
        self.rep_div_hist = []
        self.action_div_hist = []
        self.role_div_hist = []

    def forward(self, states):
        c0 = torch.zeros(cfg.num_agents, cfg.comm_dim)
        H = self.core(states, c0)
        c = self.comm(H)
        H = self.core(states, c)
        H = F.normalize(H, dim=1)

        actions, role_probs = self.roles(H)
        values = self.value(H)

        return H, actions, values, role_probs

    def env_step(self, states, actions):
        return states + cfg.step_scale * self.action_map(actions)

    def compute_loss(self, states, next_states, H, actions, values, role_probs):
        # rewards
        local_dist = torch.norm(next_states - self.local_goals, dim=1)
        global_dist = torch.norm(next_states.mean(dim=0) - self.global_goal)

        reward = -local_dist - 0.5 * global_dist

        # cooperation
        sim_matrix = torch.matmul(H, H.T)
        coop = sim_matrix.mean()

        # diversity metrics
        identity = torch.eye(cfg.num_agents)
        rep_div = torch.mean((sim_matrix - identity)**2)

        action_div = torch.var(actions, dim=0).mean()
        role_div = torch.var(role_probs, dim=0).mean()

        # anti-collapse reward shaping
        reward = reward + 0.05 * coop + 0.1 * action_div + 0.05 * role_div

        # losses
        value_loss = F.mse_loss(values.squeeze(), reward.detach())
        policy_loss = -reward.mean()

        loss = value_loss + policy_loss

        return loss, reward.mean().item(), coop.item(), rep_div.item(), action_div.item(), role_div.item()

    def step(self):
        states = torch.randn(cfg.num_agents, cfg.state_dim)

        H, actions, values, roles = self.forward(states)
        next_states = self.env_step(states, actions)

        loss, reward, coop, rep_div, act_div, role_div = self.compute_loss(states, next_states, H, actions, values, roles)

        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), cfg.grad_clip)
        self.opt.step()

        # log
        self.reward_hist.append(reward)
        self.coop_hist.append(coop)
        self.loss_hist.append(loss.item())
        self.rep_div_hist.append(rep_div)
        self.action_div_hist.append(act_div)
        self.role_div_hist.append(role_div)

        return reward, coop, act_div, role_div, loss.item()

# =========================
# TRAIN
# =========================

if __name__ == "__main__":
    sys = AgenticSystem()

    for i in range(1000):
        r, c, ad, rd, l = sys.step()
        if i % 100 == 0:
            print(f"Step {i} | Reward {r:.3f} | Coop {c:.3f} | ActDiv {ad:.3f} | RoleDiv {rd:.3f} | Loss {l:.3f}")

    plt.figure()
    plt.plot(sys.reward_hist, label="Reward")
    plt.plot(sys.coop_hist, label="Coop")
    plt.plot(sys.loss_hist, label="Loss")
    plt.plot(sys.rep_div_hist, label="RepDiv")
    plt.plot(sys.action_div_hist, label="ActionDiv")
    plt.plot(sys.role_div_hist, label="RoleDiv")

    plt.legend()
    plt.title("Agentic AI Emergence (Full Metrics)")
    plt.xlabel("Steps")
    plt.ylabel("Values")
    plt.show()

# =========================
# README.md
# =========================

"""
# Agentic AI (Final with Diversity Metrics)

## What This Shows
A complete agentic loop with:
- Action-driven learning
- Role-conditioned policies
- Communication
- Multi-agent coordination

## NEW: Diversity Metrics
- Representation Diversity (RepDiv)
- Action Diversity (ActDiv) ← most important
- Role Diversity (RoleDiv)

## Expected Behavior
- Reward improves
- Cooperation < 1.0
- Action diversity > 0
- Role diversity > 0

## Key Insight
True Agentic AI requires:
- Acting differently
- Not just thinking together

## Run
python main.py
"""
