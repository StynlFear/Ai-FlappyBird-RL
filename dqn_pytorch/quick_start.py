#!/usr/bin/env python3
"""
QUICK START GUIDE - How to run both agents simultaneously
"""

print("""
🚀 HOW TO RUN BOTH AGENTS AT THE SAME TIME
=============================================

There are 3 ways to run the dual agent comparison:

METHOD 1: Interactive Menu (Recommended)
---------------------------------------
python run_comparison.py

This will show you a menu and guide you through the process.


METHOD 2: Direct Command Line
-----------------------------
python dual_agent_comparison.py [config_name] --episodes [number]

Examples:
  python dual_agent_comparison.py flappybird_comparison --episodes 100
  python dual_agent_comparison.py cartpole_comparison --episodes 50


METHOD 3: Quick Test (CartPole - Fast)
--------------------------------------
python dual_agent_comparison.py cartpole_comparison --episodes 50

This runs a quick 2-3 minute test to see how it works.


WHAT YOU'LL SEE DURING TRAINING:
================================
🎲 RISKY AGENT: Messages with dice emoji
   • Starts with 100% random actions (high exploration)
   • Slowly becomes less random (but stays risky)
   • Takes more chances, higher variance in performance

🛡️  SAFE AGENT: Messages with shield emoji  
   • Starts with 30% random actions (moderate exploration)
   • Quickly becomes very conservative (1% random)
   • Plays it safe, more consistent performance

📊 Progress updates every 60 seconds showing:
   • Episodes completed by each agent
   • Best rewards achieved
   • Current risk levels (exploration %)

📈 Real-time graph updates showing performance comparison


OUTPUT FILES:
=============
runs/[config]_risky.pt        - Risky agent's trained model
runs/[config]_safe.pt         - Safe agent's trained model
runs/[config]_risky.log       - Risky agent's training log
runs/[config]_safe.log        - Safe agent's training log
runs/[config]_comparison.png  - Performance comparison graph


STOP TRAINING:
==============
Press Ctrl+C at any time to stop both agents and see final results.


TRY IT NOW:
===========
""")

import subprocess
import sys

choice = input("Want to run a quick demo? (y/n): ").strip().lower()

if choice == 'y':
    print("\n🚀 Starting quick CartPole demo (2-3 minutes)...")
    print("You'll see both agents training simultaneously!\n")
    
    try:
        subprocess.run([
            sys.executable, 
            "dual_agent_comparison.py", 
            "cartpole_comparison", 
            "--episodes", 
            "100"
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
    except KeyboardInterrupt:
        print("\n⚠️  Demo stopped by user")
else:
    print("\n👍 Use one of the methods above when you're ready!")
    print("Recommended: python run_comparison.py")
