import time
import gymnasium as gym
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy
import imageio
import sys
from pathlib import Path
import torch

# Adding parent directory to Python path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from static_simulation import TeenEducationEnvironment



def train_dqn():
    # Initialize environment
    env = TeenEducationEnvironment()
    
    # Check environment compatibility
    check_env(env, warn=True)
    
    # Hyperparameters
    hyperparameters = {
        "learning_rate": 0.0002,            # Increased for faster learning
        "gamma": 0.99,                     # Discount factor (unchanged)
        "batch_size": 64,                  # Batch size (unchanged)
        "total_timesteps": 200000,         # Increased for more training
        "exploration_fraction": 0.3,       # Extended for more exploration
        "exploration_initial_eps": 1.0,    # Initial epsilon (unchanged)
        "exploration_final_eps": 0.15,      # Higher for late-stage exploration
        "buffer_size": 50000,              # Buffer size (unchanged)
        "target_update_interval": 500,     # Reduced for frequent target updates
        "learning_starts": 1000,          # Increased for more initial exploration
        "gradient_steps": 1,               # Gradient updates per step (unchanged)
        "policy_kwargs": dict(net_arch=[64, 64])  # Network architecture (unchanged)
    }
    
    # Initialize DQN model
    model = DQN(
        policy="MlpPolicy",
        env=env,
        learning_rate=hyperparameters["learning_rate"],
        gamma=hyperparameters["gamma"],
        batch_size=hyperparameters["batch_size"],
        buffer_size=hyperparameters["buffer_size"],
        exploration_fraction=hyperparameters["exploration_fraction"],
        exploration_initial_eps=hyperparameters["exploration_initial_eps"],
        exploration_final_eps=hyperparameters["exploration_final_eps"],
        target_update_interval=hyperparameters["target_update_interval"],
        learning_starts=hyperparameters["learning_starts"],
        gradient_steps=hyperparameters["gradient_steps"],
        policy_kwargs=hyperparameters["policy_kwargs"],
        verbose=1,
        # prioritized_replay=True
    )
    
    # Train model
    print("Training DQN model...")
    model.learn(total_timesteps=hyperparameters["total_timesteps"], progress_bar=True)
    
    # Save model
    model.save("dqn_teen_education")
    print("Model saved as 'dqn_teen_education.zip'")
    
    # Evaluate model
    eval_env = TeenEducationEnvironment()
    episode_rewards = []
    for _ in range(20):
        obs, _ = eval_env.reset()
        done = False
        total_reward = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = eval_env.step(int(action))
            total_reward += reward
            done = terminated or truncated
        episode_rewards.append(total_reward)

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    print(f"Mean reward over 20 episodes: {mean_reward:.2f} +/- {std_reward:.2f}")

    return model

def track_q_values(model, env, n_episodes=5):
    print("\nTracking estimated Q-values (optimal value approximation):")
    q_values_per_episode = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        total_q = 0
        steps = 0
        done = False
        while not done:
            obs_tensor = torch.tensor([obs], dtype=torch.float32).to(model.device)
            q_vals = model.q_net(obs_tensor).detach().cpu().numpy()[0]
            max_q = np.max(q_vals)
            total_q += max_q
            steps += 1

            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(int(action))
            done = terminated or truncated

        avg_q = total_q / steps
        q_values_per_episode.append(avg_q)
        print(f"Episode {ep + 1}: Average Max Q-value = {avg_q:.2f}")

    print(f"\nMean Optimal Value across {n_episodes} episodes: {np.mean(q_values_per_episode):.2f}")

def demo_trained_model(model):
    env = TeenEducationEnvironment(render_mode="human")
    for episode in range(3):
        print(f"\n=== Episode {episode + 1} ===")
        observation, info = env.reset()
        total_reward = 0
        for step in range(50):
            action, _ = model.predict(observation, deterministic=True)
            action = int(action)
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            print(f"Step {step + 1}:")
            print(f"  Action: {info['action_name']}")
            print(f"  Mentor Room: {info['mentor_room']}")
            for i, teen in enumerate(info['teens']):
                print(f"  Teen {i+1}: State={teen['state_name']}, Trust={teen['trust']:.1f}, Engagement={teen['engagement']:.1f}, Knowledge={teen['knowledge']:.1f}, Room={teen['room']}")
            print(f"  Reward: {reward:.2f}")
            env.render()
            if terminated or truncated:
                print(f"Episode terminated. Total reward: {total_reward:.2f}")
                break
    env.close()

def create_trained_demo_gif(model):
    env = TeenEducationEnvironment(render_mode="rgb_array")
    frames = []
    observation, info = env.reset()
    for _ in range(150):
        action, _ = model.predict(observation, deterministic=True)
        action = int(action)
        observation, reward, terminated, truncated, info = env.step(action)
        frame = env.render()
        frames.append(frame)
        if terminated or truncated:
            observation, info = env.reset()
    env.close()
    imageio.mimsave("trained_dqn_demo.gif", frames, duration=2)
    print("Trained demo GIF saved as 'trained_dqn_demo.gif'")

if __name__ == "__main__":
    # Train the model
    model = train_dqn()

    # Track Q-values
    track_q_values(model, TeenEducationEnvironment(), n_episodes=3)
    
    # Demonstrate the trained model
    demo_trained_model(model)
    
    # Generate GIF of trained model
    create_trained_demo_gif(model)