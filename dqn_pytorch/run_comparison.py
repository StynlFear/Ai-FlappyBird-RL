#!/usr/bin/env python3
"""
Quick script to run dual agent comparison
"""

import subprocess
import sys

def main():
    print("🚀 DQN Dual Agent Comparison")
    print("Available hyperparameter sets:")
    print("  1. flappybird1 (basic)")
    print("  2. flappybird4 (improved)")
    print("  3. flappybird5 (advanced)")
    print("  4. flappybird_comparison (optimized for comparison)")
    print("  5. cartpole_comparison (quick test)")
    print("  6. cartpole1 (basic cartpole)")
    
    choice = input("\nEnter number (1-6) or hyperparameter name (default: 4): ").strip()
    
    # Map numbers to hyperparameter sets
    hyperparameter_map = {
        "1": "flappybird1",
        "2": "flappybird4", 
        "3": "flappybird5",
        "4": "flappybird_comparison",
        "5": "cartpole_comparison",
        "6": "cartpole1"
    }
    
    if choice in hyperparameter_map:
        hyperparameter_set = hyperparameter_map[choice]
    elif choice == "":
        hyperparameter_set = "flappybird_comparison"  # Default
    else:
        hyperparameter_set = choice  # Allow direct input
    
    print(f"\n🎯 Starting CONTINUOUS comparison with {hyperparameter_set}")
    print("🔄 Training will run indefinitely until you stop it")
    print("💾 Models auto-save on every reward improvement")
    print("📊 Graphs update every 60 seconds")
    print("⏹️  Press Ctrl+C to stop training and see final comparison\n")
    
    try:
        subprocess.run([
            sys.executable, 
            "dual_agent_comparison.py", 
            hyperparameter_set
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running training: {e}")
    except KeyboardInterrupt:
        print("\n⚠️  Training stopped by user")

if __name__ == "__main__":
    main()
