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
COMM_DIM = 8
HIDDEN = 64

STEPS = 1000
LR = 1e-3

BASE_ENTROPY_BETA = 0.01
ROLE_BETA = 0.05
COMM_BETA = 0.01
SMOOTHING = 0.97
GRAD_CLIP = 1.0

# ======================
# SAFETY
# ======================
def sanitize(x):
    return torch.nan_to_num(x, nan=0.0, posinf=10.0, neginf=-10.0)

def safe_softmax(logits, temperature):
    logits = sanitize(logits)
    logits = logits / max(temperature, 0.3)
    logits = torch.clamp(logits, -10, 10)
    logits = logits - logits.max()

    probs = torch.softmax(logits, dim=-1)
    probs = sanitize(probs)
    probs = torch.clamp(probs, min=1e-6)
    probs = probs / probs.sum()
    return probs

# ======================
# GNN COMMUNICATION
# ======================
class GNNComm(nn.Module):
    def __init__(self):
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(COMM_DIM * 2, HIDDEN),
            nn.ReLU(),
            nn.Linear(HIDDEN, COMM_DIM)
        )

    def forward(self, messages):
        N = messages.shape[0]
        new_msgs = []

        for i in range(N):
            agg = 0
            for j in range(N):
                if i != j:
                    edge_input = torch.cat([messages[i], messages[j]])
                    agg = agg + self.edge_mlp(edge_input)
            new_msgs.append(agg / (N - 1))

        return torch.stack(new_msgs)

# ======================
# AGENT
# ======================
class Agent(nn.Module):
    def __init__(self, agent_id):
        super().__init__()
        self.agent_id = torch.tensor(agent_id)
        self.role_embed = nn.Embedding(NUM_AGENTS, 4)

        self.encoder = nn.Sequential(
            nn.Linear(STATE_DIM + 4 + COMM_DIM, HIDDEN),
            nn.ReLU()
        )

        self.policy = nn.Linear(HIDDEN, ACTION_DIM)
        self.comm_head = nn.Linear(HIDDEN, COMM_DIM)

    def forward(self, state, comm):
        role = self.role_embed(self.agent_id)
        x = torch.cat([state, role, comm], dim=-1)
        h = self.encoder(x)

        logits = self.policy(h)
        message = self.comm_head(h)

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

def cooperation(actions):
    mean = actions.mean(dim=0)
    return 1 - ((actions - mean) ** 2).mean().item()

def entropy(probs):
    return -(probs * torch.log(torch.clamp(probs, min=1e-6))).sum()

# ======================
# INIT
# ======================
agents = [Agent(i) for i in range(NUM_AGENTS)]
gnn = GNNComm()

params = []
for a in agents:
    params += list(a.parameters())
params += list(gnn.parameters())

optimizer = optim.Adam(params, lr=LR)

running_reward = 0

# logs
rewards_log, coop_log = [], []
actdiv_log, rolediv_log = [], []
loss_log, entropy_log = [], []

# ======================
# TRAIN
# ======================
for step in range(STEPS):

    temperature = max(0.4, 1.0 - 2 * step / STEPS)
    entropy_beta = max(0.001, BASE_ENTROPY_BETA * (1 - step / STEPS))

    state = get_state()
    comms = torch.zeros(NUM_AGENTS, COMM_DIM)

    logits_list = []
    messages = []

    # PASS 1: generate messages
    for i, agent in enumerate(agents):
        logits, msg = agent(state[i], comms[i])
        logits_list.append(logits)
        messages.append(msg)

    messages = torch.stack(messages)
    messages = sanitize(messages)
    messages = torch.clamp(messages, -5, 5)

    # GNN MESSAGE PASSING
    comms = gnn(messages)

    actions = []
    probs_list = []

    # PASS 2: action with GNN comm
    for i, agent in enumerate(agents):
        logits, _ = agent(state[i], comms[i])
        probs = safe_softmax(logits, temperature)

        if torch.isnan(probs).any() or torch.isinf(probs).any():
            probs = torch.ones_like(probs) / len(probs)

        action = torch.multinomial(probs, 1)
        actions.append(action)
        probs_list.append(probs)

    actions = torch.stack(actions).squeeze(-1).float()
    reward = compute_reward(actions)

    # BASELINE + ADVANTAGE
    running_reward = SMOOTHING * running_reward + (1 - SMOOTHING) * reward
    advantage = reward - running_reward.detach()
    advantage = sanitize(advantage)

    advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-6)
    advantage = torch.clamp(advantage, -5, 5)

    total_loss = 0
    ent_total = 0

    # ROLE LOSS
    role_loss = 0
    for i in range(NUM_AGENTS):
        for j in range(i + 1, NUM_AGENTS):
            role_loss += torch.mean((logits_list[i] - logits_list[j]) ** 2)

    # COMM LOSS
    comm_loss = torch.var(messages)

    for i in range(NUM_AGENTS):
        probs = probs_list[i]
        log_prob = torch.log(torch.clamp(probs[int(actions[i])], min=1e-6))
        ent = entropy(probs)

        loss = -advantage * log_prob - entropy_beta * ent
        total_loss += loss
        ent_total += ent.item()

    total_loss += ROLE_BETA * role_loss + COMM_BETA * comm_loss

    optimizer.zero_grad()
    total_loss.backward()

    torch.nn.utils.clip_grad_norm_(params, GRAD_CLIP)
    optimizer.step()

    # LOGS
    rewards_log.append(reward.item())
    coop_log.append(cooperation(actions))
    actdiv_log.append(torch.var(actions).item())
    rolediv_log.append(torch.var(torch.stack(logits_list)).item())
    loss_log.append(total_loss.item())
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
