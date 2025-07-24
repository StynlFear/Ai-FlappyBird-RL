import gymnasium as gym
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import random
import torch
from torch import nn
import yaml
import threading
import time
from datetime import datetime, timedelta
import os
import argparse

from experience_replay import ReplayMemory
from dqn import DQN
import flappy_bird_gymnasium

# For printing date and time
DATE_FORMAT = "%m-%d %H:%M:%S"

# Directory for saving run info
RUNS_DIR = "runs"
os.makedirs(RUNS_DIR, exist_ok=True)

# 'Agg': used to generate plots as images and save them to a file instead of rendering to screen
matplotlib.use('Agg')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu' # force cpu

class RiskyAgent():
    """Agent with higher exploration and risk-taking behavior"""
    
    def __init__(self, hyperparameter_set, agent_name="risky"):
        with open('hyperparameters.yml', 'r') as file:
            all_hyperparameter_sets = yaml.safe_load(file)
            hyperparameters = all_hyperparameter_sets[hyperparameter_set]

        self.hyperparameter_set = hyperparameter_set
        self.agent_name = agent_name
        
        # Use hyperparameters from config file
        self.env_id             = hyperparameters['env_id']
        self.learning_rate_a    = hyperparameters['learning_rate_a']
        self.discount_factor_g  = hyperparameters['discount_factor_g']
        self.network_sync_rate  = hyperparameters['network_sync_rate']
        self.replay_memory_size = hyperparameters['replay_memory_size']
        self.mini_batch_size    = hyperparameters['mini_batch_size']
        self.stop_on_reward     = hyperparameters['stop_on_reward']
        self.fc1_nodes          = hyperparameters['fc1_nodes']
        self.env_make_params    = hyperparameters.get('env_make_params',{})
        self.enable_double_dqn  = hyperparameters['enable_double_dqn']
        self.enable_dueling_dqn = hyperparameters['enable_dueling_dqn']
        
        # RISKY AGENT - FORCE 100% EXPLORATION OVERRIDE
        self.epsilon_init       = 1.0          # Always start at 100% exploration
        self.epsilon_decay      = 1.0          # NO DECAY - stays at 100% risk forever!
        self.epsilon_min        = 1.0          # Always 100% risky - pure exploration
        
        # Neural Network
        self.loss_fn = nn.MSELoss()
        self.optimizer = None

        # Path to Run info
        self.LOG_FILE   = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}_{self.agent_name}.log')
        self.MODEL_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}_{self.agent_name}.pt')
        
        # Performance tracking
        self.rewards_per_episode = []
        self.epsilon_history = []
        self.training_active = True

    def train(self, max_episodes=None):
        """Train the risky agent continuously"""
        start_time = datetime.now()
        
        log_message = f"{start_time.strftime(DATE_FORMAT)}: RISKY AGENT training starting (CONTINUOUS MODE)..."
        print(f"🎲 RISKY AGENT: {log_message}")
        with open(self.LOG_FILE, 'a', encoding='utf-8') as file:
            file.write(log_message + '\n')

        # Create environment
        env = gym.make(self.env_id, render_mode=None, **self.env_make_params)
        num_actions = env.action_space.n
        num_states = env.observation_space.shape[0]

        # Create networks
        policy_dqn = DQN(num_states, num_actions, self.fc1_nodes, self.enable_dueling_dqn).to(device)
        target_dqn = DQN(num_states, num_actions, self.fc1_nodes, self.enable_dueling_dqn).to(device)
        
        # Load existing model if it exists
        if os.path.exists(self.MODEL_FILE):
            print(f"🎲 RISKY AGENT: Loading existing model from {self.MODEL_FILE}")
            policy_dqn.load_state_dict(torch.load(self.MODEL_FILE))
            target_dqn.load_state_dict(torch.load(self.MODEL_FILE))
        else:
            print(f"🎲 RISKY AGENT: Starting fresh training")
            target_dqn.load_state_dict(policy_dqn.state_dict())

        # Initialize training variables
        epsilon = self.epsilon_init
        memory = ReplayMemory(self.replay_memory_size)
        self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=self.learning_rate_a)
        step_count = 0
        best_reward = -999999

        # Training loop - CONTINUOUS
        episode = 0
        while self.training_active:
                
            state, _ = env.reset()
            state = torch.tensor(state, dtype=torch.float, device=device)
            terminated = False
            episode_reward = 0.0

            while not terminated and episode_reward < self.stop_on_reward:
                # Epsilon-greedy action selection (risky behavior)
                if random.random() < epsilon:
                    action = env.action_space.sample()
                    action = torch.tensor(action, dtype=torch.int64, device=device)
                else:
                    with torch.no_grad():
                        action = policy_dqn(state.unsqueeze(dim=0)).squeeze().argmax()

                # Execute action
                new_state, reward, terminated, truncated, info = env.step(action.item())
                episode_reward += reward

                # Convert to tensors
                new_state = torch.tensor(new_state, dtype=torch.float, device=device)
                reward = torch.tensor(reward, dtype=torch.float, device=device)

                # Store experience
                memory.append((state, action, new_state, reward, terminated))
                step_count += 1
                state = new_state

            # Track performance
            self.rewards_per_episode.append(episode_reward)
            
            # Save model EVERY TIME reward improves
            if episode_reward > best_reward:
                torch.save(policy_dqn.state_dict(), self.MODEL_FILE)
                best_reward = episode_reward
                log_message = f"{datetime.now().strftime(DATE_FORMAT)}: RISKY AGENT new best reward {episode_reward:0.1f} at episode {episode}"
                print(f"🎲 RISKY AGENT: NEW BEST! Reward {episode_reward:0.1f} (episode {episode}) - MODEL SAVED")
                with open(self.LOG_FILE, 'a', encoding='utf-8') as file:
                    file.write(log_message + '\n')

            # Training
            if len(memory) > self.mini_batch_size:
                mini_batch = memory.sample(self.mini_batch_size)
                self.optimize(mini_batch, policy_dqn, target_dqn)
                
                # Decay epsilon
                epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)
                self.epsilon_history.append(epsilon)

                # Sync networks
                if step_count > self.network_sync_rate:
                    target_dqn.load_state_dict(policy_dqn.state_dict())
                    step_count = 0

            if episode % 50 == 0:
                print(f"🎲 RISKY AGENT - Episode {episode}: Reward = {episode_reward:.1f}, Epsilon = {epsilon:.4f} (risk level: {epsilon*100:.1f}%)")
            
            episode += 1

        env.close()

    def optimize(self, mini_batch, policy_dqn, target_dqn):
        """Optimize the neural network"""
        states, actions, new_states, rewards, terminations = zip(*mini_batch)
        
        states = torch.stack(states)
        actions = torch.stack(actions)
        new_states = torch.stack(new_states)
        rewards = torch.stack(rewards)
        terminations = torch.tensor(terminations).float().to(device)

        with torch.no_grad():
            if self.enable_double_dqn:
                best_actions_from_policy = policy_dqn(new_states).argmax(dim=1)
                target_q = rewards + (1-terminations) * self.discount_factor_g * \
                           target_dqn(new_states).gather(dim=1, index=best_actions_from_policy.unsqueeze(dim=1)).squeeze()
            else:
                target_q = rewards + (1-terminations) * self.discount_factor_g * target_dqn(new_states).max(dim=1)[0]

        current_q = policy_dqn(states).gather(dim=1, index=actions.unsqueeze(dim=1)).squeeze()
        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class SafeAgent():
    """Agent with lower exploration and conservative behavior"""
    
    def __init__(self, hyperparameter_set, agent_name="safe"):
        with open('hyperparameters.yml', 'r') as file:
            all_hyperparameter_sets = yaml.safe_load(file)
            hyperparameters = all_hyperparameter_sets[hyperparameter_set]

        self.hyperparameter_set = hyperparameter_set
        self.agent_name = agent_name
        
        # Use hyperparameters from config file
        self.env_id             = hyperparameters['env_id']
        self.learning_rate_a    = hyperparameters['learning_rate_a']
        self.discount_factor_g  = hyperparameters['discount_factor_g']
        self.network_sync_rate  = hyperparameters['network_sync_rate']
        self.replay_memory_size = hyperparameters['replay_memory_size']
        self.mini_batch_size    = hyperparameters['mini_batch_size']
        self.stop_on_reward     = hyperparameters['stop_on_reward']
        self.fc1_nodes          = hyperparameters['fc1_nodes']
        self.env_make_params    = hyperparameters.get('env_make_params',{})
        self.enable_double_dqn  = hyperparameters['enable_double_dqn']
        self.enable_dueling_dqn = hyperparameters['enable_dueling_dqn']
        
        # Use epsilon values from config file (conservative behavior)
        self.epsilon_init       = hyperparameters['epsilon_init']
        self.epsilon_decay      = hyperparameters['epsilon_decay']
        self.epsilon_min        = hyperparameters['epsilon_min']
        
        # Neural Network
        self.loss_fn = nn.MSELoss()
        self.optimizer = None

        # Path to Run info
        self.LOG_FILE   = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}_{self.agent_name}.log')
        self.MODEL_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}_{self.agent_name}.pt')
        
        # Performance tracking
        self.rewards_per_episode = []
        self.epsilon_history = []
        self.training_active = True

    def train(self, max_episodes=None):
        """Train the safe agent continuously"""
        start_time = datetime.now()
        
        log_message = f"{start_time.strftime(DATE_FORMAT)}: SAFE AGENT training starting (CONTINUOUS MODE)..."
        print(f"🛡️  SAFE AGENT: {log_message}")
        with open(self.LOG_FILE, 'a', encoding='utf-8') as file:
            file.write(log_message + '\n')

        # Create environment
        env = gym.make(self.env_id, render_mode=None, **self.env_make_params)
        num_actions = env.action_space.n
        num_states = env.observation_space.shape[0]

        # Create networks
        policy_dqn = DQN(num_states, num_actions, self.fc1_nodes, self.enable_dueling_dqn).to(device)
        target_dqn = DQN(num_states, num_actions, self.fc1_nodes, self.enable_dueling_dqn).to(device)
        
        # Load existing model if it exists
        if os.path.exists(self.MODEL_FILE):
            print(f"🛡️  SAFE AGENT: Loading existing model from {self.MODEL_FILE}")
            policy_dqn.load_state_dict(torch.load(self.MODEL_FILE))
            target_dqn.load_state_dict(torch.load(self.MODEL_FILE))
        else:
            print(f"🛡️  SAFE AGENT: Starting fresh training")
            target_dqn.load_state_dict(policy_dqn.state_dict())

        # Initialize training variables
        epsilon = self.epsilon_init
        memory = ReplayMemory(self.replay_memory_size)
        self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=self.learning_rate_a)
        step_count = 0
        best_reward = -999999

        # Training loop - CONTINUOUS
        episode = 0
        while self.training_active:
                
            state, _ = env.reset()
            state = torch.tensor(state, dtype=torch.float, device=device)
            terminated = False
            episode_reward = 0.0

            while not terminated and episode_reward < self.stop_on_reward:
                # Epsilon-greedy action selection (conservative behavior)
                if random.random() < epsilon:
                    action = env.action_space.sample()
                    action = torch.tensor(action, dtype=torch.int64, device=device)
                else:
                    with torch.no_grad():
                        action = policy_dqn(state.unsqueeze(dim=0)).squeeze().argmax()

                # Execute action
                new_state, reward, terminated, truncated, info = env.step(action.item())
                episode_reward += reward

                # Convert to tensors
                new_state = torch.tensor(new_state, dtype=torch.float, device=device)
                reward = torch.tensor(reward, dtype=torch.float, device=device)

                # Store experience
                memory.append((state, action, new_state, reward, terminated))
                step_count += 1
                state = new_state

            # Track performance
            self.rewards_per_episode.append(episode_reward)
            
            # Save model EVERY TIME reward improves
            if episode_reward > best_reward:
                torch.save(policy_dqn.state_dict(), self.MODEL_FILE)
                best_reward = episode_reward
                log_message = f"{datetime.now().strftime(DATE_FORMAT)}: SAFE AGENT new best reward {episode_reward:0.1f} at episode {episode}"
                print(f"🛡️  SAFE AGENT: NEW BEST! Reward {episode_reward:0.1f} (episode {episode}) - MODEL SAVED")
                with open(self.LOG_FILE, 'a', encoding='utf-8') as file:
                    file.write(log_message + '\n')

            # Training
            if len(memory) > self.mini_batch_size:
                mini_batch = memory.sample(self.mini_batch_size)
                self.optimize(mini_batch, policy_dqn, target_dqn)
                
                # Decay epsilon
                epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)
                self.epsilon_history.append(epsilon)

                # Sync networks
                if step_count > self.network_sync_rate:
                    target_dqn.load_state_dict(policy_dqn.state_dict())
                    step_count = 0

            if episode % 50 == 0:
                print(f"🛡️  SAFE AGENT - Episode {episode}: Reward = {episode_reward:.1f}, Epsilon = {epsilon:.4f} (risk level: {epsilon*100:.1f}%)")
            
            episode += 1

        env.close()

    def optimize(self, mini_batch, policy_dqn, target_dqn):
        """Optimize the neural network"""
        states, actions, new_states, rewards, terminations = zip(*mini_batch)
        
        states = torch.stack(states)
        actions = torch.stack(actions)
        new_states = torch.stack(new_states)
        rewards = torch.stack(rewards)
        terminations = torch.tensor(terminations).float().to(device)

        with torch.no_grad():
            if self.enable_double_dqn:
                best_actions_from_policy = policy_dqn(new_states).argmax(dim=1)
                target_q = rewards + (1-terminations) * self.discount_factor_g * \
                           target_dqn(new_states).gather(dim=1, index=best_actions_from_policy.unsqueeze(dim=1)).squeeze()
            else:
                target_q = rewards + (1-terminations) * self.discount_factor_g * target_dqn(new_states).max(dim=1)[0]

        current_q = policy_dqn(states).gather(dim=1, index=actions.unsqueeze(dim=1)).squeeze()
        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


def save_comparison_graph(risky_agent, safe_agent, hyperparameter_set):
    """Save comparison graph of both agents"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Calculate moving averages for smoother visualization
    def moving_average(data, window=100):
        if len(data) < window:
            return data
        return np.convolve(data, np.ones(window)/window, mode='valid')
    
    # Reward comparison
    risky_rewards = risky_agent.rewards_per_episode
    safe_rewards = safe_agent.rewards_per_episode
    
    if risky_rewards:
        risky_avg = moving_average(risky_rewards)
        ax1.plot(risky_avg, 'r-', label='Risky Agent', alpha=0.8, linewidth=2)
    
    if safe_rewards:
        safe_avg = moving_average(safe_rewards)
        ax1.plot(safe_avg, 'b-', label='Safe Agent', alpha=0.8, linewidth=2)
    
    ax1.set_title('Reward Comparison (Moving Average)')
    ax1.set_xlabel('Episodes')
    ax1.set_ylabel('Average Reward')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Individual risky agent performance
    if risky_rewards:
        ax2.plot(risky_rewards, 'r-', alpha=0.6, linewidth=1)
        ax2.plot(moving_average(risky_rewards), 'r-', linewidth=2)
    ax2.set_title('Risky Agent Performance')
    ax2.set_xlabel('Episodes')
    ax2.set_ylabel('Reward')
    ax2.grid(True, alpha=0.3)
    
    # Individual safe agent performance
    if safe_rewards:
        ax3.plot(safe_rewards, 'b-', alpha=0.6, linewidth=1)
        ax3.plot(moving_average(safe_rewards), 'b-', linewidth=2)
    ax3.set_title('Safe Agent Performance')
    ax3.set_xlabel('Episodes')
    ax3.set_ylabel('Reward')
    ax3.grid(True, alpha=0.3)
    
    # Epsilon comparison
    if risky_agent.epsilon_history:
        ax4.plot(risky_agent.epsilon_history, 'r-', label='Risky Agent', alpha=0.8)
    if safe_agent.epsilon_history:
        ax4.plot(safe_agent.epsilon_history, 'b-', label='Safe Agent', alpha=0.8)
    ax4.set_title('Exploration Rate (Epsilon) Comparison')
    ax4.set_xlabel('Training Steps')
    ax4.set_ylabel('Epsilon')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save graph
    comparison_file = os.path.join(RUNS_DIR, f'{hyperparameter_set}_comparison.png')
    fig.savefig(comparison_file, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"📊 Comparison graph saved to {comparison_file}")


def run_dual_training(risky_hyperparameter_set, safe_hyperparameter_set):
    """Run both agents simultaneously with different hyperparameter sets"""
    print("=" * 70)
    print(f"🚀 STARTING DUAL AGENT CONTINUOUS TRAINING COMPARISON")
    print("=" * 70)
    print(f"📋 Risky Agent Config: {risky_hyperparameter_set}")
    print(f"📋 Safe Agent Config:  {safe_hyperparameter_set}")
    print(f"🎯 Training Mode: CONTINUOUS (no episode limit)")
    print(f"💾 Auto-save: Models saved on every reward improvement")
    print()
    print("🎲 RISKY AGENT PROFILE:")
    print("   • Maximum exploration: ε stays at 1.0 (100% random)")
    print("   • No decay: ε never decreases - pure exploration!")
    print("   • Strategy: Pure risk-taker, always explores, maximum variance")
    print()
    print("🛡️  SAFE AGENT PROFILE:")
    print("   • Uses hyperparameter settings for controlled exploration")
    print("   • Epsilon decay as configured in hyperparameters")
    print("   • Strategy: Conservative, exploits knowledge, lower variance")
    print()
    print("📊 Real-time comparison graph will update every 60 seconds")
    print("⏹️  Press Ctrl+C to stop training and see final comparison")
    print("=" * 70)
    
    # Create agents with different hyperparameter sets
    risky_agent = RiskyAgent(risky_hyperparameter_set)
    safe_agent = SafeAgent(safe_hyperparameter_set)
    
    # Create threads for parallel training
    risky_thread = threading.Thread(target=risky_agent.train)
    safe_thread = threading.Thread(target=safe_agent.train)
    
    # Start training
    start_time = time.time()
    risky_thread.start()
    safe_thread.start()
    
    # Monitor progress and save graphs periodically
    graph_update_interval = 60  # Update graph every minute
    last_graph_update = time.time()
    
    try:
        while risky_thread.is_alive() or safe_thread.is_alive():
            time.sleep(10)  # Check every 10 seconds
            
            current_time = time.time()
            if current_time - last_graph_update > graph_update_interval:
                save_comparison_graph(risky_agent, safe_agent, f"{risky_hyperparameter_set}_vs_{safe_hyperparameter_set}")
                last_graph_update = current_time
                
                # Print status
                risky_episodes = len(risky_agent.rewards_per_episode)
                safe_episodes = len(safe_agent.rewards_per_episode)
                risky_best = max(risky_agent.rewards_per_episode) if risky_agent.rewards_per_episode else 0
                safe_best = max(safe_agent.rewards_per_episode) if safe_agent.rewards_per_episode else 0
                
                print(f"📊 PROGRESS UPDATE - {datetime.now().strftime('%H:%M:%S')}")
                print(f"   🎲 RISKY AGENT:  {risky_episodes:3d} episodes, best reward: {risky_best:6.1f}")
                print(f"   🛡️  SAFE AGENT:   {safe_episodes:3d} episodes, best reward: {safe_best:6.1f}")
                
                # Show current exploration levels if available
                risky_epsilon = risky_agent.epsilon_history[-1] if risky_agent.epsilon_history else 1.0
                safe_epsilon = safe_agent.epsilon_history[-1] if safe_agent.epsilon_history else 0.3
                print(f"   🎲 Risk level: {risky_epsilon*100:4.1f}% | 🛡️  Risk level: {safe_epsilon*100:4.1f}%")
                print("-" * 60)
    
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
        risky_agent.training_active = False
        safe_agent.training_active = False
    
    # Wait for threads to finish
    risky_thread.join()
    safe_thread.join()
    
    # Final comparison
    save_comparison_graph(risky_agent, safe_agent, f"{risky_hyperparameter_set}_vs_{safe_hyperparameter_set}")
    
    # Print final results
    training_time = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"✅ TRAINING COMPLETED in {training_time/60:.1f} minutes")
    print(f"{'='*70}")
    print(f"🎲 RISKY AGENT: {len(risky_agent.rewards_per_episode)} episodes completed")
    print(f"🛡️  SAFE AGENT:  {len(safe_agent.rewards_per_episode)} episodes completed")
    
    if risky_agent.rewards_per_episode and safe_agent.rewards_per_episode:
        risky_best = max(risky_agent.rewards_per_episode)
        safe_best = max(safe_agent.rewards_per_episode)
        risky_avg = np.mean(risky_agent.rewards_per_episode[-100:])  # Last 100 episodes
        safe_avg = np.mean(safe_agent.rewards_per_episode[-100:])   # Last 100 episodes
        
        print(f"\n🏆 FINAL PERFORMANCE COMPARISON:")
        print(f"   📈 Best Single Reward:")
        print(f"      🎲 Risky Agent: {risky_best:6.1f}")
        print(f"      🛡️  Safe Agent:  {safe_best:6.1f}")
        print(f"   📊 Average Last 100 Episodes:")
        print(f"      🎲 Risky Agent: {risky_avg:6.1f}")
        print(f"      🛡️  Safe Agent:  {safe_avg:6.1f}")
        
        if risky_avg > safe_avg:
            winner = "🎲 RISKY AGENT"
            margin = risky_avg - safe_avg
        else:
            winner = "🛡️  SAFE AGENT"
            margin = safe_avg - risky_avg
            
        print(f"\n🥇 WINNER (by average performance): {winner}")
        print(f"   Winning margin: {margin:+.1f} reward points")
        
        # Performance analysis
        print(f"\n📈 PERFORMANCE ANALYSIS:")
        risky_std = np.std(risky_agent.rewards_per_episode[-100:])
        safe_std = np.std(safe_agent.rewards_per_episode[-100:])
        print(f"   🎲 Risky Agent - Consistency (std dev): {risky_std:.1f}")
        print(f"   🛡️  Safe Agent  - Consistency (std dev): {safe_std:.1f}")
        
        if risky_std > safe_std:
            print(f"   → Safe Agent is more consistent (lower variance)")
        else:
            print(f"   → Risky Agent is more consistent (lower variance)")
    
    print(f"\n📊 Comparison graph saved to: runs/{risky_hyperparameter_set}_vs_{safe_hyperparameter_set}_comparison.png")
    print(f"{'='*70}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compare Risky vs Safe DQN agents with different hyperparameter sets.')
    parser.add_argument('risky_hyperparameters', help='Hyperparameter set name for risky agent (e.g., flappybird4_risky)')
    parser.add_argument('safe_hyperparameters', help='Hyperparameter set name for safe agent (e.g., flappybird4_safe)')
    args = parser.parse_args()

    run_dual_training(args.risky_hyperparameters, args.safe_hyperparameters)
