import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

class AgentSubstrate(nn.Module):
    def __init__(self, id):
        super().__init__()
        self.id = id
        self.brain = nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, x): return self.brain(x)

class GodBrainOrchestrator(nn.Module):
    def __init__(self, num_agents):
        super().__init__()
        # Hierarchy weights: How much the God-Brain trusts each tool-agent
        self.weights = nn.Parameter(torch.ones(num_agents) / num_agents)
    
    def forward(self, agent_outputs):
        # Cooperation: Weighted consensus
        return torch.sum(agent_outputs * torch.softmax(self.weights, dim=0))

# --- Metrics ---
def get_metrics(outputs, goal, supervisor_output):
    reward = 1.0 / (1.0 + torch.abs(supervisor_output - goal)) # Normalized [0,1]
    diversity = torch.var(outputs) + 0.1 # Ensure non-zero
    # Cooperation: Inverse of variance weighted by closeness to goal
    cooperation = 1.0 / (1.0 + torch.std(outputs)) 
    iq_score = (reward * diversity * cooperation) * 100
    return reward, diversity, cooperation, iq_score

# --- Training Loop ---
def train():
    num_agents = 4
    goal = torch.tensor([10.0])
    agents = nn.ModuleList([AgentSubstrate(i) for i in range(num_agents)])
    god_brain = GodBrainOrchestrator(num_agents)
    optimizer = optim.Adam(list(agents.parameters()) + list(god_brain.parameters()), lr=0.01)

    stats = {"R": [], "D": [], "C": [], "IQ": []}

    print(f"{'Step':<6} | {'Reward':<8} | {'Diversity':<10} | {'Coop':<8} | {'IQ Score'}")
    print("-" * 60)

    for step in range(101):
        optimizer.zero_grad()
        tool_input = torch.randn(1, 10) # API/File System data simulation
        
        outputs = torch.cat([a(tool_input) for a in agents])
        final_decision = god_brain(outputs)
        
        R, D, C, IQ = get_metrics(outputs, goal, final_decision)
        
        # Loss: Maximize Reward, Diversity, and Cooperation
        loss = - (torch.log(R) + 0.2 * torch.log(D) + 0.5 * torch.log(C))
        
        loss.backward()
        optimizer.step()

        if step % 20 == 0:
            print(f"{step:<6} | {R.item():.4f} | {D.item():.4f} | {C.item():.4f} | {IQ.item():.2f}")
            stats["R"].append(R.item()); stats["D"].append(D.item())
            stats["C"].append(C.item()); stats["IQ"].append(IQ.item())

    print("\nTraining Complete. Intelligence successfully scaled.")

if __name__ == "__main__":
    train()
