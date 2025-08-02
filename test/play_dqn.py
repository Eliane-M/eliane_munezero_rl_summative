import time
import warnings
warnings.filterwarnings("ignore")

import gymnasium as gym
from stable_baselines3 import DQN
from pathlib import Path
import sys

# Include environment path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from custom_environment import TeenEducationEnvironment

def play_model(model_path="trained_teen_dqn_model.zip", episodes=3):
    print("Loading trained DQN model...")

    try:
        model = DQN.load(model_path)
        print("[SUCCESS] Model loaded successfully!")
    except FileNotFoundError:
        print(f"[ERROR] {model_path} not found. Please train the model first.")
        return
    except Exception as e:
        print(f"[ERROR] Error loading model: {e}")
        return

    env = TeenEducationEnvironment(render_mode="human")

    total_reward_all_episodes = 0

    print(f"\nEvaluating model on {episodes} episodes using Greedy Policy...\n")
    for episode in range(episodes):
        obs, info = env.reset()
        done = False
        truncated = False
        episode_reward = 0
        step_count = 0

        print(f"\n=== Episode {episode + 1}/{episodes} ===")
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            step_count += 1

            print(f"Step {step_count}:")
            print(f"  Action: {info['action_name']}")
            print(f"  State: {info['previous_state_name']} → {info['current_state_name']}")
            print(f"  Reward: {reward:.2f}")
            print(f"  Trust: {info['trust_level']:.1f}%, Engagement: {info['engagement_level']:.1f}%, Knowledge: {info['knowledge_retention']:.1f}%")

            env.render()
            time.sleep(1)

            if done:
                print(f"[TERMINATED] Final state: {info['current_state_name']}")
                break

        print(f"Episode Reward: {episode_reward:.2f}")
        total_reward_all_episodes += episode_reward

    avg_reward = total_reward_all_episodes / episodes
    print("\n" + "="*50)
    print("MODEL PERFORMANCE SUMMARY")
    print("="*50)
    print(f"Episodes played      : {episodes}")
    print(f"Average reward       : {avg_reward:.2f}")
    print("Policy used          : Deterministic (Greedy)")
    print("="*50)

    env.close()

if __name__ == "__main__":
    play_model()
