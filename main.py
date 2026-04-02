import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

torch.autograd.set_detect_anomaly(False)

# ======================
# CONFIG
# ======================
NUM_AGENTS = 4
STATE_DIM = 8
ACTION_DIM = 4
COMM_DIM = 8

STEPS = 1000
LR = 1e-3

BASE_ENTROPY_BETA = 0.01
ROLE_BETA = 0.05
COMM_BETA = 0.01
SMOOTHING = 0.97
GRAD_CLIP = 1.0

# ======================
# SAFE SOFTMAX
# ======================
def safe_softmax(logits, temperature):
    logits = logits / temperature
    logits = torch.clamp(logits, -20, 20)
    logits = logits - logits.max()
    probs = torch.softmax(logits, dim=-1)
    probs = probs + 1e-8
    probs = probs / probs.sum()
    return probs

# ======================
# AGENT WITH COMMUNICATION
# ======================
class Agent(nn.Module):
    def __init__(self, agent_id):
        super().__init__()
        self.agent_id = torch.tensor(agent_id)

        self.role_embed = nn.Embedding(NUM_AGENTS, 4)

        self.encoder = nn.Sequential(
            nn.Linear(STATE_DIM + 4 + COMM_DIM, 64),
            nn.ReLU()
        )

        self.policy_head = nn.Linear(64, ACTION_DIM)
        self.comm_head = nn.Linear(64, COMM_DIM)

    def forward(self, state, comm):
        role = self.role_embed(self.agent_id)
        x = torch.cat([state, role, comm], dim=-1)
        hidden = self.encoder(x)

        logits = self.policy_head(hidden)
        message = self.comm_head(hidden)

        return logits, message

# ======================
# ENV
# ======================
def get_state():
    return torch.randn(NUM_AGENTS, STATE_DIM)

def compute_reward(actions):
    mean = actions.mean(dim=0)
    similarity = -((actions - mean) ** 2).mean()
    diversity = torch.var(actions)
    return 5.0 * similarity + 0.05 * diversity

# ======================
# METRICS
# ======================
def cooperation(actions):
    mean = actions.mean(dim=0)
    return 1 - ((actions - mean) ** 2).mean().item()

def entropy(probs):
    return -(probs * torch.log(torch.clamp(probs, min=1e-8))).sum()

# ======================
# INIT
# ======================
agents = [Agent(i) for i in range(NUM_AGENTS)]
optims = [optim.Adam(agent.parameters(), lr=LR) for agent in agents]

running_reward = 0

# logs
rewards_log, coop_log, loss_log = [], [], []
actdiv_log, rolediv_log, entropy_log = [], [], []

# ======================
# TRAIN
# ======================
for step in range(STEPS):

    temperature = max(0.4, 1.0 - 2 * step / STEPS)
    entropy_beta = max(0.001, BASE_ENTROPY_BETA * (1 - step / STEPS))

    state = get_state()

    # initial communication = zeros
    comms = torch.zeros(NUM_AGENTS, COMM_DIM)

    logits_list = []
    messages = []

    # PASS 1: generate messages
    for i, agent in enumerate(agents):
        logits, msg = agent(state[i], comms[i])
        logits_list.append(logits)
        messages.append(msg)

    messages = torch.stack(messages)

    # AGGREGATE COMMUNICATION (mean field)
    comms = messages.mean(dim=0).repeat(NUM_AGENTS, 1)

    actions = []
    probs_list = []

    # PASS 2: action selection with comm
    for i, agent in enumerate(agents):
        logits, _ = agent(state[i], comms[i])
        probs = safe_softmax(logits, temperature)

        action = torch.multinomial(probs, 1)
        actions.append(action)
        probs_list.append(probs)

    actions = torch.stack(actions).squeeze(-1).float()

    reward = compute_reward(actions)

    # ======================
    # BASELINE + ADVANTAGE
    # ======================
    running_reward = SMOOTHING * running_reward + (1 - SMOOTHING) * reward
    advantage = reward - running_reward.detach()

    # normalize
    advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
    advantage = torch.clamp(advantage, -5, 5)

    # ======================
    # LOSSES
    # ======================
    total_loss = 0
    ent_total = 0

    # role separation
    role_loss = 0
    for i in range(NUM_AGENTS):
        for j in range(i+1, NUM_AGENTS):
            role_loss += torch.mean((logits_list[i] - logits_list[j])**2)

    # communication regularization
    comm_loss = torch.var(messages)

    for i in range(NUM_AGENTS):
        probs = probs_list[i]

        log_prob = torch.log(torch.clamp(probs[int(actions[i])], min=1e-8))
        ent = entropy(probs)

        loss = -advantage * log_prob - entropy_beta * ent
        total_loss += loss
        ent_total += ent.item()

    total_loss += ROLE_BETA * role_loss + COMM_BETA * comm_loss

    # ======================
    # BACKPROP
    # ======================
    for opt in optims:
        opt.zero_grad()

    total_loss.backward()

    for agent in agents:
        torch.nn.utils.clip_grad_norm_(agent.parameters(), GRAD_CLIP)

    for opt in optims:
        opt.step()

    # ======================
    # LOGGING
    # ======================
    rewards_log.append(reward.item())
    coop_log.append(cooperation(actions))
    loss_log.append(total_loss.item())
    actdiv_log.append(torch.var(actions).item())
    rolediv_log.append(torch.var(torch.stack(logits_list)).item())
    entropy_log.append(ent_total / NUM_AGENTS)

    if step % 100 == 0:
        print(f"Step {step} | Reward {reward:.3f} | Coop {coop_log[-1]:.3f} | "
              f"ActDiv {actdiv_log[-1]:.3f} | RoleDiv {rolediv_log[-1]:.3f} | "
              f"Loss {total_loss.item():.3f}")

# ======================
# PLOT
# ======================
fig, axs = plt.subplots(3,2, figsize=(12,10))

axs[0,0].plot(rewards_log); axs[0,0].set_title("Reward")
axs[0,1].plot(coop_log); axs[0,1].set_title("Cooperation")
axs[1,0].plot(actdiv_log); axs[1,0].set_title("Action Diversity")
axs[1,1].plot(rolediv_log); axs[1,1].set_title("Role Diversity")
axs[2,0].plot(loss_log); axs[2,0].set_title("Loss")
axs[2,1].plot(entropy_log); axs[2,1].set_title("Entropy")

plt.tight_layout()
plt.show()
