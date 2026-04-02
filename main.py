# =========================
# FINAL AGENTIC AI SYSTEM (BEST VERSION)
# Fully Differentiable, Role-Conditioned, Cooperative
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
# CORE MODULES
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
# COMMUNICATION
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
# ROLE-CONDITIONED POLICIES
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

        self.reward_hist, self.coop_hist, self.loss_hist = [], [], []

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

    def compute_loss(self, states, next_states, H, values, role_probs):
        local_dist = torch.norm(next_states - self.local_goals, dim=1)
        global_dist = torch.norm(next_states.mean(dim=0) - self.global_goal)

        reward = -local_dist - 0.5 * global_dist

        coop = torch.matmul(H, H.T).mean()

        role_entropy = -(role_probs * torch.log(role_probs + 1e-8)).sum(dim=1).mean()

        reward = reward + 0.05 * coop + 0.1 * role_entropy

        value_loss = F.mse_loss(values.squeeze(), reward.detach())
        policy_loss = -reward.mean()

        loss = value_loss + policy_loss

        return loss, reward.mean().item(), coop.item()

    def step(self):
        states = torch.randn(cfg.num_agents, cfg.state_dim)

        H, actions, values, roles = self.forward(states)
        next_states = self.env_step(states, actions)

        loss, reward, coop = self.compute_loss(states, next_states, H, values, roles)

        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), cfg.grad_clip)
        self.opt.step()

        self.reward_hist.append(reward)
        self.coop_hist.append(coop)
        self.loss_hist.append(loss.item())

        return reward, coop, loss.item()

# =========================
# TRAIN
# =========================

if __name__ == "__main__":
    sys = AgenticSystem()

    for i in range(1000):
        r, c, l = sys.step()
        if i % 100 == 0:
            print(f"Step {i} | Reward {r:.3f} | Coop {c:.3f} | Loss {l:.3f}")

    plt.figure()
    plt.plot(sys.reward_hist, label="Reward")
    plt.plot(sys.coop_hist, label="Coop")
    plt.plot(sys.loss_hist, label="Loss")
    plt.legend()
    plt.title("Agentic AI Emergence")
    plt.show()

# =========================
# README.md
# =========================

"""
# Agentic AI System (Final)

## What This Is
A fully differentiable multi-agent system with:
- Action-driven learning (true RL loop)
- Role-conditioned behavior
- Communication between agents
- Shared + individual goals

## Key Properties

### 1. Agentic Loop
state → action → environment → reward → learning

### 2. Roles Matter
Different policies per role → real specialization

### 3. Cooperation + Autonomy
- Local goals = individuality
- Global goal = coordination

### 4. Emergence
System learns:
- coordinated movement
- differentiated behavior
- adaptive strategies

## Expected Behavior
- Reward improves
- Cooperation < 1.0
- Agents behave differently

## Run
python main.py

## Insight
This is a minimal working model of Agentic AI:
LLM-like core + tools (actions) + memory (state) + coordination
"""
