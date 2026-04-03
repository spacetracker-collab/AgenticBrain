import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# --- 1. The Hierarchical Structure ---
class WorkerAgent(nn.Module):
    """Sub-agent with specialized behavior."""
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, x):
        return self.network(x)

class SupervisorBrain(nn.Module):
    """The 'God-Brain' that orchestrates workers and assigns rewards."""
    def __init__(self, num_agents):
        super().__init__()
        # Weighting mechanism for how much to trust each agent
        self.hierarchy_weights = nn.Parameter(torch.ones(num_agents))

    def forward(self, agent_outputs):
        # Weighted combination of agent actions
        weighted_output = torch.sum(agent_outputs * torch.softmax(self.hierarchy_weights, dim=0))
        return weighted_output

# --- 2. Causal Reward & Intelligence Scoring ---
def calculate_causal_reward(agent_output, prev_dist, current_dist):
    """
    Causal Reward: Rewards the 'Behavior' that significantly 
    reduces the distance to the goal.
    """
    causal_impact = prev_dist - current_dist
    return causal_impact if causal_impact > 0 else torch.tensor(0.0)

def calculate_intelligence_score(reward, diversity, steps):
    """Composite Score: Intelligence = (Efficiency * Diversity) / Steps"""
    return (reward * (1 + diversity)) / (steps + 1)

# --- 3. The Reinforcement Learning Loop ---
def run_agentic_loop():
    input_dim, hidden_dim, num_agents = 5, 16, 3
    goal = torch.tensor([5.0])
    
    # Initialize Model Hierarchy
    workers = [WorkerAgent(input_dim, hidden_dim) for _ in range(num_agents)]
    supervisor = SupervisorBrain(num_agents)
    
    # Optimizers
    params = list(supervisor.parameters())
    for w in workers: params += list(w.parameters())
    optimizer = optim.Adam(params, lr=0.02)

    prev_distance = torch.tensor(10.0)
    print(f"{'Step':<6} | {'Reward':<10} | {'Diversity':<10} | {'IQ Score':<10}")
    print("-" * 50)

    for step in range(51):
        optimizer.zero_grad()
        
        # 1. Perception (Input from tools/environment)
        state = torch.randn(input_dim)
        
        # 2. Agent Actions (Diversity in Behavior)
        outputs = torch.cat([w(state) for w in workers])
        diversity = torch.var(outputs)
        
        # 3. Hierarchical Integration
        final_decision = supervisor(outputs)
        
        # 4. Causal Reward Calculation
        current_distance = torch.abs(final_decision - goal)
        reward = calculate_causal_reward(final_decision, prev_distance, current_distance)
        
        # 5. Loss Function (RL Objective)
        # Minimize distance to goal while maximizing causal reward and diversity
        loss = current_distance - (0.5 * reward) - (0.2 * diversity)
        
        loss.backward()
        optimizer.step()
        
        # 6. Scoring Intelligence
        iq_score = calculate_intelligence_score(reward.item(), diversity.item(), step)
        
        if step % 10 == 0:
            print(f"{step:<6} | {reward.item():.4f}   | {diversity.item():.4f}   | {iq_score:.4f}")
            
        prev_distance = current_distance.detach()

if __name__ == "__main__":
    run_agentic_loop()
