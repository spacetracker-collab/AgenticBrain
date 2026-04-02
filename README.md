
""
# Role-Emergent Multi-Agent Intelligence System

## Final System Capabilities

This system implements:
- Action-driven learning (true RL loop)
- Multi-agent cooperation
- Agent specialization via local goals
- Global coordination
- Emergent role assignment (explorer / exploiter / coordinator)
- Learned communication protocol

## Architecture

state → reasoning → communication → GNN → roles → action → environment → reward → learning

## Key Innovations

1. Dual objective:
   - Local goals → specialization
   - Global goal → cooperation

2. Emergent roles:
   - Agents learn functional differentiation

3. Fully differentiable system:
   - End-to-end gradient flow

## Expected Behavior

- Cooperation stabilizes below 1.0
- Reward improves steadily
- Agents take different roles
- System exhibits coordinated intelligence

## Run

python main.py

## Insight

This system approximates a "digital organism":
- Agents = cells
- Communication = signaling
- Roles = specialization
- GNN = nervous system

Emergence arises from:
constraint + interaction + adaptation
"""
