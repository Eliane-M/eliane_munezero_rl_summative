import os
import time
import gymnasium as gym
from matplotlib import pyplot as plt
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy
import imageio
import sys
from pathlib import Path
import torch
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

# Adding parent directory to Python path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

# Import the updated spatial environment
from environment.custom_environment import TeenEducationEnvironment


class RewardTrackerCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []

    def _on_step(self) -> bool:
        if len(self.locals["infos"]) > 0 and "episode" in self.locals["infos"][0]:
            ep_info = self.locals["infos"][0]["episode"]
            self.episode_rewards.append(ep_info["r"])
        return True

    def plot_rewards(self):
        plt.figure(figsize=(10, 6))
        plt.plot(np.cumsum(self.episode_rewards), label="Cumulative Reward")
        plt.title("Cumulative Reward Over Episodes")
        plt.xlabel("Episode")
        plt.ylabel("Cumulative Reward")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

def train_dqn():
    # Initialize environment
    env = TeenEducationEnvironment()

    log_dir = "..models.logs.dqn"
    os.makedirs(log_dir, exist_ok=True)

    # Wrap environment with Monitor to enable logging
    env = TeenEducationEnvironment()
    env = Monitor(env, log_dir)
    
    # Check environment compatibility
    check_env(env, warn=True)
    
    # Hyperparameters (adjusted for spatial environment)
    hyperparameters = {
        "learning_rate": 0.0001,            # Increased for faster learning
        "gamma": 0.99,                     # Discount factor (unchanged)
        "batch_size": 32,                  # Batch size (unchanged)
        "total_timesteps": 5000,         # Increased for spatial complexity
        "exploration_fraction": 0.15,      # Extended for more spatial exploration
        "exploration_initial_eps": 1.0,    # Initial epsilon (unchanged)
        "exploration_final_eps": 0.02,     # Higher for late-stage exploration
        "buffer_size": 500,              # Increased buffer for spatial patterns
        "target_update_interval": 100,     # Reduced for frequent target updates
        "learning_starts": 100,           # Increased for more initial exploration
        "gradient_steps": 1,               # Gradient updates per step (unchanged)
        "policy_kwargs": dict(net_arch=[64, 64, 64])  # Larger network for spatial complexity
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
    print("Training DQN model on Spatial Teen Education Environment...")
    model.learn(total_timesteps=hyperparameters["total_timesteps"], progress_bar=True)
    
    # Save model
    model.save("dqn_spatial_teen_education")
    print("Model saved as 'dqn_spatial_teen_education.zip'")
    
    # Evaluate model
    eval_env = TeenEducationEnvironment()
    episode_rewards = []
    success_episodes = 0
    
    for i in range(20):
        obs, _ = eval_env.reset()
        done = False
        total_reward = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(int(action))
            total_reward += reward
            done = terminated or truncated
            
            # Check if this was a successful episode
            if done and info['girls_in_room'] == eval_env.num_girls:
                success_episodes += 1
        
        episode_rewards.append(total_reward)
        print(f"Eval Episode {i+1}: Reward={total_reward:.1f}, Girls in Room={info['girls_in_room']}/{eval_env.num_girls}")

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    success_rate = success_episodes / 20 * 100
    
    print(f"\nEvaluation Results:")
    print(f"Mean reward over 20 episodes: {mean_reward:.2f} +/- {std_reward:.2f}")
    print(f"Success rate (all girls in empowered room): {success_rate:.1f}%")

    return model

def track_q_values(model, env, n_episodes=10):
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
    
    for episode in range(10):
        print(f"\n=== Episode {episode + 1} ===")
        observation, info = env.reset()
        total_reward = 0
        
        for step in range(100):  # Increased steps for spatial environment
            action, _ = model.predict(observation, deterministic=True)
            action = int(action)
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            print(f"Step {step + 1}:")
            print(f"  Action: {info['action_name']}")
            print(f"  Agent Position: ({info['agent_pos'][0]}, {info['agent_pos'][1]})")
            print(f"  Girls in Empowered Room: {info['girls_in_room']}/{env.num_girls}")
            print(f"  Girls Ready for Room: {info['girls_ready_for_room']}")
            print(f"  Average Trust: {info['average_trust']:.1f}%")
            print(f"  Nearby Girls: {info['nearby_girls']}")
            
            # Show individual girl states
            for i, girl in enumerate(env.girls):
                state_name = env.girl_states[girl['state']]
                ready_status = "READY" if girl['ready_for_room'] else "Not Ready"
                in_room_status = "IN ROOM" if girl['in_empowered_room'] else "Outside"
                print(f"    Girl {i+1}: {state_name}, Trust={girl['trust']:.1f}, {ready_status}, {in_room_status}")
            
            print(f"  Reward: {reward:.2f}, Total: {total_reward:.2f}")
            print("-" * 50)
            
            env.render()
            time.sleep(0.1)  # Small delay to see the action
            
            if terminated or truncated:
                print(f"Episode terminated. Total reward: {total_reward:.2f}")
                print(f"Final result: {info['girls_in_room']}/{env.num_girls} girls reached empowered room")
                break
    
    env.close()

def create_trained_demo_gif(model):
    env = TeenEducationEnvironment(render_mode="rgb_array")
    frames = []
    observation, info = env.reset()
    
    for step in range(900):  # Longer episodes for spatial environment
        action, _ = model.predict(observation, deterministic=True)
        action = int(action)
        observation, reward, terminated, truncated, info = env.step(action)
        frame = env.render()
        frames.append(frame)
        
        if terminated or truncated:
            # Add a few frames at the end to show final state
            for _ in range(10):
                frames.append(frame)
            observation, info = env.reset()
    
    env.close()
    imageio.mimsave("trained_spatial_dqn_demo.gif", frames, duration=0.2)
    print("Trained spatial demo GIF saved as 'trained_spatial_dqn_demo.gif'")

def analyze_training_performance(model):
    """Analyze how well the trained model performs on key metrics."""
    env = TeenEducationEnvironment()
    
    print("\n=== Training Performance Analysis ===")
    
    # Metrics tracking
    total_episodes = 3
    successful_episodes = 0
    total_girls_empowered = 0
    avg_steps_to_success = []
    
    for episode in range(total_episodes):
        obs, _ = env.reset()
        done = False
        steps = 0
        
        while not done and steps < 200:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            steps += 1
            done = terminated or truncated
        
        girls_in_room = info['girls_in_room']
        total_girls_empowered += girls_in_room
        
        if girls_in_room == env.num_girls:
            successful_episodes += 1
            avg_steps_to_success.append(steps)
        
        print(f"Episode {episode+1}: {girls_in_room}/{env.num_girls} girls empowered in {steps} steps")
    
    success_rate = (successful_episodes / total_episodes) * 100
    avg_girls_per_episode = total_girls_empowered / total_episodes
    avg_success_steps = np.mean(avg_steps_to_success) if avg_steps_to_success else 0
    
    print(f"\nPerformance Summary:")
    print(f"Success Rate: {success_rate:.1f}% ({successful_episodes}/{total_episodes} episodes)")
    print(f"Average Girls Empowered per Episode: {avg_girls_per_episode:.1f}/{env.num_girls}")
    print(f"Average Steps to Success: {avg_success_steps:.1f}")
    
    env.close()

if __name__ == "__main__":
    # Train the model
    model = train_dqn()

    # Analyze training performance
    analyze_training_performance(model)

    # Track Q-values
    track_q_values(model, TeenEducationEnvironment(), n_episodes=3)
    
    # Demonstrate the trained model
    demo_trained_model(model)
    
    # Generate GIF of trained model
    create_trained_demo_gif(model)

    # Plot reward tracking
    reward_tracker = RewardTrackerCallback()
    reward_tracker.plot_rewards()