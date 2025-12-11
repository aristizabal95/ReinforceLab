import gymnasium as gym
from agent import Agent
from monitor import ConvergenceMonitor

# --- CONFIG (Default for local test) ---
ENV_ID = "___ENV_ID___"
# ---------------------------------------

def test_manual():
    print(f"--- Running Local Test on {ENV_ID} ---")
    env = gym.make(ENV_ID)
    agent = Agent()
    
    # Try to load model if exists
    try:
        agent.load("model.pt")
        print("Model loaded.")
    except:
        print("Running without trained model (random init).")
    
    obs, _ = env.reset()
    done = False
    score = 0
    while not done:
        action = agent.act(obs)
        obs, r, term, trunc, _ = env.step(action)
        done = term or trunc
        score += r
        
    print(f"Episode done. Score: {score}")

if __name__ == "__main__":
    test_manual()