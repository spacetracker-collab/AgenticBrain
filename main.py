import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# ======================
# CONFIG
# ======================
NUM_AGENTS = 4
STATE_DIM = 8
ACTION_DIM = 4
STEPS = 1000
LR = 1e-3
ENTROPY_BETA = 0.01

# ======================
# AGENT
# ======================
class Agent(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(STATE_DIM, 64),
            nn.ReLU(),
            nn.Linear(64, ACTION_DIM)
        )

    def forward(self, x):
        return self.net(x)


# ======================
# ENV (toy cooperative env)
# ======================
def get_state():
    return torch.randn(NUM_AGENTS, STATE_DIM)


def compute_reward(actions):
    # Encourage similarity but not identical (balance)
    mean_action = actions.mean(dim=0)
    similarity = -((actions - mean_action) ** 2).mean()

    # Add small penalty to avoid trivial collapse
    diversity_penalty = -torch.var(actions)

    return similarity + 0.1 * diversity_penalty


# ======================
# METRICS
# ======================
def cooperation(actions):
    mean_action = actions.mean(dim=0)
    return 1 - ((actions - mean_action) ** 2).mean().item()


def action_diversity(actions):
    return torch.var(actions).item()


def role_diversity(policies):
    stacked = torch.stack(policies)
    return torch.var(stacked).item()


def entropy(logits):
    probs = torch.softmax(logits, dim=-1)
    log_probs = torch.log(probs + 1e-8)
    return -(probs * log_probs).sum(dim=-1).mean()


# ======================
# INIT
# ======================
agents = [Agent() for _ in range(NUM_AGENTS)]
optims = [optim.Adam(agent.parameters(), lr=LR) for agent in agents]

# logs
rewards_log = []
coop_log = []
actdiv_log = []
rolediv_log = []
loss_log = []
entropy_log = []

# ======================
# TRAIN LOOP
# ======================
for step in range(STEPS):
    state = get_state()

    logits_list = []
    actions = []

    for i, agent in enumerate(agents):
        logits = agent(state[i])
        logits_list.append(logits)

        probs = torch.softmax(logits, dim=-1)
        action = torch.multinomial(probs, 1).float()
        actions.append(action)

    actions = torch.stack(actions).squeeze(-1)

    reward = compute_reward(actions)

    total_loss = 0
    ent = 0

    for i in range(NUM_AGENTS):
        logits = logits_list[i]
        probs = torch.softmax(logits, dim=-1)

        log_prob = torch.log(probs[actions[i].long()] + 1e-8)

        policy_loss = -reward * log_prob
        ent_i = entropy(logits)

        loss = policy_loss - ENTROPY_BETA * ent_i

        optims[i].zero_grad()
        loss.backward(retain_graph=True)
        optims[i].step()

        total_loss += loss.item()
        ent += ent_i.item()

    # metrics
    coop = cooperation(actions)
    act_div = action_diversity(actions)
    role_div = role_diversity(logits_list)

    rewards_log.append(reward.item())
    coop_log.append(coop)
    actdiv_log.append(act_div)
    rolediv_log.append(role_div)
    loss_log.append(total_loss)
    entropy_log.append(ent / NUM_AGENTS)

    if step % 100 == 0:
        print(f"Step {step} | Reward {reward:.3f} | Coop {coop:.3f} | "
              f"ActDiv {act_div:.3f} | RoleDiv {role_div:.3f} | "
              f"Loss {total_loss:.3f}")


# ======================
# PLOTTING (6-IN-1)
# ======================
fig, axs = plt.subplots(3, 2, figsize=(12, 10))

axs[0, 0].plot(rewards_log)
axs[0, 0].set_title("Reward")

axs[0, 1].plot(coop_log)
axs[0, 1].set_title("Cooperation")

axs[1, 0].plot(actdiv_log)
axs[1, 0].set_title("Action Diversity")

axs[1, 1].plot(rolediv_log)
axs[1, 1].set_title("Role Diversity")

axs[2, 0].plot(loss_log)
axs[2, 0].set_title("Loss")

axs[2, 1].plot(entropy_log)
axs[2, 1].set_title("Entropy")

plt.tight_layout()
plt.show()
