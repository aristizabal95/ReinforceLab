# ReinforceLab: CartPole Competition

Train and evaluate reinforcement learning agents on the CartPole-v1 environment. Phase 1 evaluates pre-trained agents on average return over multiple episodes. Phase 2 measures training efficiency by tracking convergence time to target performance.

## Competition Overview

This reinforcement learning competition evaluates your agent's performance across two distinct phases:

### Phase 1: Evaluation
In the first phase, your trained agent will be evaluated on its **average return** over multiple episodes. The agent will run for **100** episodes in the `CartPole-v1` environment, and the average return across all episodes will be used as the score. Higher average returns indicate better performance.

**Key Details:**
- Your agent should be trained and saved as `model.pt` in your submission
- The agent will be loaded and evaluated without further training
- Performance is measured by the mean return across all evaluation episodes
- Higher scores are better (descending order on leaderboard)

### Phase 2: Training Convergence
In the second phase, your agent will be trained from scratch, and we measure how quickly it converges to a target performance level. The score is the **number of training steps** required to reach and maintain the goal reward of **475.0** for **100** consecutive episodes.

**Key Details:**
- Your agent will be trained from scratch (no pre-trained model)
- Training will run for up to **10000** steps
- The goal is to reach a reward of **475.0** and maintain it for **100** episodes
- The score is the number of steps taken to achieve convergence
- Lower scores (fewer steps) are better (ascending order on leaderboard)
- The evaluation is repeated **5** times, and the average convergence time is reported

## Submission Format

Your submission should include:
- `agent.py`: A file containing an `Agent` class with:
  - `act(observation)`: Method that returns an action given an observation
  - `train()` (optional): Method for training the agent
  - `load(model_path)`: Method to load a saved model
  - `save(model_path)`: Method to save the current model
- `model.pt` (for Phase 1): Pre-trained model file that will be loaded during evaluation

## Getting Started

1. Download the starting kit from the competition page
2. Implement your `Agent` class in `agent.py`
3. Train your agent and save it as `model.pt` for Phase 1 evaluation
4. Submit your code and model to Phase 1
5. For Phase 2, ensure your agent can train effectively from scratch

Good luck!

