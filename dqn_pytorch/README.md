# AI Flappy Bird with Deep Q-Learning (DQN) - Workshop Project

Welcome to the AI Flappy Bird workshop! This project demonstrates how to train an AI agent to play Flappy Bird using Deep Q-Network (DQN) reinforcement learning algorithm implemented in PyTorch.

## 🎯 Workshop Overview

In this workshop, you'll learn:
- **Reinforcement Learning fundamentals** - Understanding agents, environments, states, actions, and rewards
- **Deep Q-Networks (DQN)** - How neural networks can learn optimal game strategies
- **Experience Replay** - Collecting and learning from past experiences
- **Epsilon-Greedy Strategy** - Balancing exploration vs exploitation
- **Target Networks** - Stabilizing training with dual network architecture
- **PyTorch Implementation** - Building neural networks for RL

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- PyTorch
- Gymnasium (OpenAI Gym)
- Flappy Bird Gymnasium environment

### Installation
```bash
# Clone the repository
git clone <your-repo-url>
cd dqn_pytorch

# Install required packages
pip install torch gymnasium flappy-bird-gymnasium pyyaml matplotlib tensorboard
```

### Run a Pre-trained Model
```bash
# Test the trained Flappy Bird agent
python quick_start.py
```

### Train Your Own Agent
```bash
# Train on CartPole (quick test - 2-3 minutes)
python agent.py --env CartPole-v1 --episodes 1000

# Train on Flappy Bird (longer training - several hours)
python agent.py --env FlappyBird-v0 --episodes 50000
```

## 📁 Project Structure

```
dqn_pytorch/
├── agent.py                    # Main DQN agent implementation
├── dqn.py                     # Deep Q-Network neural network
├── experience_replay.py       # Experience replay buffer
├── hyperparameters.yml        # Training configuration
├── quick_start.py             # Demo script for pre-trained models
├── run_comparison.py          # Compare different training strategies
├── dual_agent_comparison.py   # Compare two agents side-by-side
└── runs/                      # Training logs and saved models
    ├── *.pt                   # Trained model weights
    ├── *.log                  # Training progress logs
    └── *.png                  # Performance graphs
```

## 🧠 How It Works

### 1. Deep Q-Network (DQN)
The neural network takes the game state as input and outputs Q-values for each possible action:
- **Input**: Bird position, pipe positions, velocities
- **Output**: Q-values for "flap" vs "do nothing"
- **Architecture**: Fully connected layers with ReLU activation

### 2. Experience Replay
The agent stores experiences (state, action, reward, next_state) and learns from random batches:
- Prevents overfitting to recent experiences
- Enables more efficient learning
- Configurable memory size (default: 100,000 experiences)

### 3. Epsilon-Greedy Exploration
Balances random exploration with learned behavior:
- Start with 100% random actions (ε = 1.0)
- Gradually decrease to 1% random actions (ε = 0.01)
- Ensures the agent discovers new strategies

### 4. Target Network
Uses two identical networks for stable training:
- **Policy Network**: Updated every step, makes decisions
- **Target Network**: Updated periodically, provides stable targets
- Prevents the "moving target" problem

## 🎮 Environments

### CartPole-v1 (Beginner)
- **Goal**: Balance a pole on a cart
- **Training time**: 2-3 minutes
- **Perfect for**: Testing your DQN implementation

### FlappyBird-v0 (Advanced)
- **Goal**: Navigate through pipes
- **Training time**: Several hours to days
- **Challenging because**: Sparse rewards, precise timing required

## 📊 Monitoring Training

### View Real-time Progress
```bash
# Start TensorBoard (if using)
tensorboard --logdir=runs

# Or check log files
tail -f runs/flappybird1.log
```

### Compare Agent Performance
```bash
# Compare two training strategies
python run_comparison.py

# Watch two agents play side-by-side
python dual_agent_comparison.py
```

## ⚙️ Configuration

Edit `hyperparameters.yml` to experiment with different settings:

```yaml
# Learning parameters
learning_rate: 0.001
discount_factor: 0.99
batch_size: 32

# Exploration
epsilon_start: 1.0
epsilon_end: 0.01
epsilon_decay: 0.995

# Network updates
target_update_frequency: 1000
replay_memory_size: 100000
```

## 🎯 Workshop Activities

### Beginner Level
1. **Run the demo** - See a trained agent play Flappy Bird
2. **Train CartPole** - Quick success to understand the process
3. **Modify hyperparameters** - See how changes affect learning

### Intermediate Level
4. **Analyze training curves** - Understand learning progress
5. **Compare strategies** - Safe vs risky exploration
6. **Implement improvements** - Double DQN, Dueling DQN

### Advanced Level
7. **Train Flappy Bird** - Full training session
8. **Optimize network architecture** - Experiment with layer sizes
9. **Create custom environments** - Apply DQN to new games

## 📈 Expected Results

### CartPole Training
- **Episodes 0-100**: Random performance (~20 points)
- **Episodes 100-300**: Rapid improvement
- **Episodes 300+**: Consistent high scores (500 points)

### Flappy Bird Training
- **Hours 1-4**: Learning basic flight
- **Hours 4-12**: Occasional pipe navigation
- **Hours 12+**: Consistent multi-pipe performance

## 🔧 Troubleshooting

### Training Not Improving?
- Check epsilon decay - too fast = no exploration
- Increase replay memory size
- Adjust learning rate (try 0.0001 or 0.01)
- Verify environment rewards

### Agent Playing Poorly?
- Ensure model is loaded correctly
- Check if training completed successfully
- Try different saved models in `runs/` folder

## 🤝 Contributing

This is a workshop project! Feel free to:
- Experiment with different network architectures
- Try new environments
- Implement DQN variants (Double DQN, Dueling DQN, etc.)
- Share your training results

## 📚 Further Learning

### Next Steps
- **Double DQN**: Reduces overestimation bias
- **Dueling DQN**: Separates value and advantage functions
- **Prioritized Experience Replay**: Learn from important experiences first
- **Rainbow DQN**: Combines multiple improvements

### Resources
- [Deep Q-Learning Paper](https://arxiv.org/abs/1312.5602)
- [OpenAI Gymnasium Documentation](https://gymnasium.farama.org/)
- [PyTorch RL Tutorial](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

**Happy Learning! 🤖🎮**

*Remember: The goal isn't just to train an AI that plays Flappy Bird, but to understand the principles of reinforcement learning that can be applied to countless real-world problems!*