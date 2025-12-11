import os
import sys
import json
import numpy as np
import gymnasium as gym
from monitor import ConvergenceMonitor

# --- CONFIGURATION (Injected by Framework) ---
ENV_ID = "CartPole-v1"
GOAL = 475.0
WINDOW = 100
MAX_STEPS = 10000
NUM_RUNS = 5
# ---------------------------------------------

def run():
    print("--- Starting Phase 2: Convergence ---")
    
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    program_dir = sys.argv[3]
    submission_dir = sys.argv[4]

    sys.path.append(submission_dir)
    try:
        from agent import Agent
    except ImportError:
        print("CRITICAL: Could not import Agent.")
        sys.exit(1)

    step_counts = []
    runs = int(NUM_RUNS)
    goal = float(GOAL)
    window = int(WINDOW)
    max_s = int(MAX_STEPS)

    for i in range(runs):
        print(f"Run {i+1}/{runs}")
        try:
            raw_env = gym.make(ENV_ID)
        except:
            sys.path.append(program_dir)
            import importlib
            module = importlib.import_module("custom_env")
            raw_env = module.make_env()
            
        env = ConvergenceMonitor(raw_env, goal, window, max_s)
        agent = Agent() # Fresh instance
        
        try:
            agent.train(env)
        except Exception as e:
            print(f"Training Error: {e}")
            
        if env.converged:
            score = env.total_steps
        else:
            score = max_s * 1.5
            
        step_counts.append(score)

    final_score = np.mean(step_counts)
    
    with open(os.path.join(output_dir, "scores.json"), "w") as f:
        json.dump({"score": final_score}, f)

if __name__ == "__main__":
    run()