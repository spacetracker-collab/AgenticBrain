# =========================
# AGENTIC AI FULL SYSTEM (GOAL-DRIVEN + COOPERATIVE + VISUALIZATION)
# Single-file repo
# =========================

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import matplotlib.pyplot as plt

# =========================
# CONFIG
# =========================

class Config:
    state_dim = 128
    hidden_dim = 256
    action_dim = 10
    num_agents = 5
    memory_size = 200
    lr = 1e-3
    gamma = 0.95
    goal_scale = 5.0

cfg = Config()

# =========================
# MEMORY SYSTEM
# =========================

class Memory:
    def __init__(self):
        self.episodic = []

    def store(self, transition):
        self.episodic.append(transition)
        if len(self.episodic) > cfg.memory_size:
            self.episodic.pop(0)

    def sample(self, batch_size=16):
        return random.sample(self.episodic, min(len(self.episodic), batch_size))

# =========================
# REASONING CORE
# =========================

class ReasoningCore(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.state_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim)
        )

    def forward(self, x):
        return self.net(x)

# =========================
# POLICY + VALUE
# =========================

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
# GNN LAYER (COOPERATION)
# =========================

class GNNLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(cfg.hidden_dim, cfg.hidden_dim)

    def forward(self, h, adj):
        h_agg = torch.matmul(adj, h)
        return F.relu(self.linear(h_agg))

# =========================
# META-LEARNING
# =========================

class MetaLearner:
    def __init__(self, model):
        self.model = model

    def adapt(self, loss):
        for p in self.model.parameters():
            if p.grad is not None:
                p.data -= cfg.lr * p.grad

# =========================
# AGENT
# =========================

class Agent:
    def __init__(self):
        self.memory = Memory()
        self.core = ReasoningCore()
        self.policy = Policy()
        self.value = Value()
        self.meta = MetaLearner(self.core)

        self.optimizer = torch.optim.Adam(
            list(self.core.parameters()) +
            list(self.policy.parameters()) +
            list(self.value.parameters()), lr=cfg.lr)

    def act(self, state):
        h = self.core(state)
        probs = self.policy(h)
        action = torch.multinomial(probs, 1)
        return action, probs, h

    def learn(self):
        if len(self.memory.episodic) < 5:
            return 0

        batch = self.memory.sample()
        loss = 0

        for (s, a, r, s2) in batch:
            h = self.core(s)
            value = self.value(h)
            target = r
            loss += (value - target).pow(2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.meta.adapt(loss)

        return loss.item()

# =========================
# MULTI-AGENT SYSTEM
# =========================

class MultiAgentSystem:
    def __init__(self):
        self.agents = [Agent() for _ in range(cfg.num_agents)]
        self.gnn = GNNLayer()

        # Shared goal (cooperative)
        self.goal = torch.randn(cfg.state_dim) * cfg.goal_scale

        # Metrics
        self.avg_rewards = []
        self.avg_losses = []
        self.cooperation_scores = []

    def compute_reward(self, state, h_new):
        # distance to shared goal
        dist = torch.norm(state - self.goal)
        coop = torch.norm(h_new.mean(dim=0))

        # reward encourages goal + cooperation
        return -dist + 0.1 * coop

    def step(self):
        states = torch.randn(cfg.num_agents, cfg.state_dim)

        hidden_states = []
        actions = []

        for i, agent in enumerate(self.agents):
            action, _, h = agent.act(states[i])
            hidden_states.append(h)
            actions.append(action)

        H = torch.stack(hidden_states)

        # fully cooperative graph
        adj = torch.ones(cfg.num_agents, cfg.num_agents) / cfg.num_agents

        H_new = self.gnn(H, adj)

        rewards = []
        losses = []

        for i, agent in enumerate(self.agents):
            s = states[i]
            r = self.compute_reward(s, H_new)
            s2 = H_new[i].detach()

            agent.memory.store((s, actions[i], r, s2))
            loss = agent.learn()

            rewards.append(r.item())
            losses.append(loss)

        # metrics
        self.avg_rewards.append(sum(rewards) / len(rewards))
        self.avg_losses.append(sum(losses) / len(losses))

        # cooperation metric (alignment)
        sim = torch.mean(torch.matmul(H_new, H_new.T)).item()
        self.cooperation_scores.append(sim)

# =========================
# TRAIN + VISUALIZE
# =========================

if __name__ == "__main__":
    system = MultiAgentSystem()

    for step in range(1000):
        system.step()

        if step % 100 == 0:
            print(f"Step {step} | Reward: {system.avg_rewards[-1]:.3f} | Coop: {system.cooperation_scores[-1]:.3f}")

    # Visualization
    plt.figure()
    plt.plot(system.avg_rewards, label="Reward")
    plt.plot(system.cooperation_scores, label="Cooperation")
    plt.plot(system.avg_losses, label="Loss")
    plt.legend()
    plt.title("Agent Intelligence Over Time")
    plt.xlabel("Steps")
    plt.ylabel("Metrics")
    plt.show()

