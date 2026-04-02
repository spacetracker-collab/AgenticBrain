import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# ======================
# CONFIG
# ======================
NUM_AGENTS = 4
STATE_DIM = 8
ACTION_DIM = 4
STEPS = 1000
LR = 1e-3

BASE_ENTROPY_BETA = 0.01
ROLE_BETA = 0.05
SMOOTHING = 0.97
GRAD_CLIP = 1.0

# ======================
# AGENT (ROLE-AWARE)
# ======================
class Agent(nn.Module):
    def __init__(self, agent_id):
        super().__init__()
        self.role_embed = nn.Embedding(NUM_AGENTS, 4)
        self.agent_id = torch.tensor(agent_id)

        self.net = nn.Sequential(
            nn.Linear(STATE_DIM + 4, 64),
            nn.ReLU(),
            nn.Linear(64, ACTION_DIM)
        )

    def forward(self, x):
        role = self.role_embed(self.agent_id)
        x = torch.cat([x, role], dim=-1)
        return self.net(x)

# ======================
# ENV
# ======================
def get_state():
    return torch.randn(NUM_AGENTS, STATE_DIM)

def compute_reward(actions):
    mean_action = actions.mean(dim=0)
    similarity = -((actions - mean_action) ** 2).mean()
    diversity = torch.var(actions)

    # Strong cooperation anchor
    return 5.0 * similarity + 0.05 * diversity

# ======================
# METRICS
# ======================
def cooperation(actions):
    mean_action = actions.mean(dim=0)
    return 1 - ((actions - mean_action) ** 2).mean().item()

def action_diversity(actions):
    return torch.var(actions).item()

def role_diversity(logits_list):
    return torch.var(torch.stack(logits_list)).item()

def entropy(logits):
    probs = torch.softmax(logits, dim=-1)
    log_probs = torch.log(probs + 1e-8)
    return -(probs * log_probs).sum()

# ======================
# INIT
# ======================
agents = [Agent(i) for i in range(NUM_AGENTS)]
optims = [optim.Adam(agent.parameters(), lr=LR) for agent in agents]

running_reward = 0

# logs
rewards_log = []
coop_log = []
actdiv_log = []
rolediv_log = []
loss_log = []
entropy_log = []

# ======================
# TRAIN
# ======================
for step in range(STEPS):

    temperature = max(0.3, 1.0 - 2 * step / STEPS)
    entropy_beta = max(0.001, BASE_ENTROPY_BETA * (1 - step / STEPS))

    state = get_state()

    logits_list = []
    actions = []

    for i, agent in enumerate(agents):
        logits = agent(state[i])
        logits_list.append(logits)

        probs = torch.softmax(logits / temperature, dim=-1)
        action = torch.multinomial(probs, 1)
        actions.append(action)

    actions = torch.stack(actions).squeeze(-1).float()

    reward = compute_reward(actions)

    # ======================
    # BASELINE (EMA)
    # ======================
    running_reward = SMOOTHING * running_reward + (1 - SMOOTHING) * reward

    advantage = reward - running_reward.detach()

    # NORMALIZE (CRITICAL FIX)
    advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

    # CLIP (prevents explosions)
    advantage = torch.clamp(advantage, -5, 5)

    coop = cooperation(actions)
    act_div = action_diversity(actions)

    # Encourage diversity only if cooperation is poor
    if coop < 0.3:
        advantage += 0.05 * act_div

    # ======================
    # ROLE SEPARATION
    # ======================
    role_loss = 0
    for i in range(NUM_AGENTS):
        for j in range(i + 1, NUM_AGENTS):
            role_loss += torch.mean((logits_list[i] - logits_list[j]) ** 2)

    # ======================
    # TOTAL LOSS
    # ======================
    total_loss = 0
    ent_total = 0

    for i in range(NUM_AGENTS):
        logits = logits_list[i]
        probs = torch.softmax(logits, dim=-1)

        log_prob = torch.log(probs[int(actions[i])] + 1e-8)
        ent = entropy(logits)

        loss = -advantage * log_prob - entropy_beta * ent
        total_loss += loss
        ent_total += ent.item()

    total_loss += ROLE_BETA * role_loss

    # ======================
    # BACKWARD
    # ======================
    for opt in optims:
        opt.zero_grad()

    total_loss.backward()

    # GRADIENT CLIPPING (CRITICAL)
    for agent in agents:
        torch.nn.utils.clip_grad_norm_(agent.parameters(), GRAD_CLIP)

    for opt in optims:
        opt.step()

    role_div = role_diversity(logits_list)

    # logging
    rewards_log.append(reward.item())
    coop_log.append(coop)
    actdiv_log.append(act_div)
    rolediv_log.append(role_div)
    loss_log.append(total_loss.item())
    entropy_log.append(ent_total / NUM_AGENTS)

    if step % 100 == 0:
        print(f"Step {step} | Reward {reward:.3f} | Coop {coop:.3f} | "
              f"ActDiv {act_div:.3f} | RoleDiv {role_div:.3f} | "
              f"Loss {total_loss.item():.3f}")

# ======================
# PLOT
# ======================
fig, axs = plt.subplots(3, 2, figsize=(12, 10))

axs[0, 0].plot(rewards_log); axs[0, 0].set_title("Reward")
axs[0, 1].plot(coop_log); axs[0, 1].set_title("Cooperation")
axs[1, 0].plot(actdiv_log); axs[1, 0].set_title("Action Diversity")
axs[1, 1].plot(rolediv_log); axs[1, 1].set_title("Role Diversity")
axs[2, 0].plot(loss_log); axs[2, 0].set_title("Loss")
axs[2, 1].plot(entropy_log); axs[2, 1].set_title("Entropy")

plt.tight_layout()
plt.show()
