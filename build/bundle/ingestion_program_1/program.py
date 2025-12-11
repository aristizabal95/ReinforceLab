import os
import sys
import json
import numpy as np
import gymnasium as gym

# --- CONFIGURATION (Injected by Framework) ---
ENV_ID = "CartPole-v1"
NUM_EPISODES = 100
# ---------------------------------------------

def run():
    print("--- Starting Phase 1: Evaluation ---")
    
    # Codabench directory structure parsing
    # Default args: program.py input output program submission
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    program_dir = sys.argv[3]
    submission_dir = sys.argv[4]

    print(f"Submission dir: {submission_dir}")
    
    # Add submission to path
    sys.path.append(submission_dir)
    
    try:
        from agent import Agent
    except ImportError as e:
        print(f"CRITICAL: Could not import Agent from submission. {e}")
        # List files for debugging
        print("Files in submission:", os.listdir(submission_dir))
        sys.exit(1)

    # Setup Env
    try:
        env = gym.make(ENV_ID)
    except:
        import importlib
        sys.path.append(program_dir) # Check program dir for custom envs
        module = importlib.import_module("custom_env")
        env = module.make_env()

    # Load Agent
    agent = Agent()
    # Convention: model.pt is at the root of the submission
    model_path = os.path.join(submission_dir, "model.pt")
    
    if os.path.exists(model_path):
        print(f"Loading model from {model_path}")
        agent.load(model_path)
    else:
        print("Warning: No model.pt found in submission root")

    scores = []
    for ep in range(int(NUM_EPISODES)):
        obs, _ = env.reset()
        done = False
        ep_score = 0
        while not done:
            action = agent.act(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_score += reward
        scores.append(ep_score)

    final_score = np.mean(scores)
    print(f"Final Score: {final_score}")

    # Write Output to 'scores.json' in output_dir
    with open(os.path.join(output_dir, "scores.json"), "w") as f:
        json.dump({"score": final_score}, f)

if __name__ == "__main__":
    run()