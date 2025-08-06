import time
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.monitor import Monitor
import imageio
import sys
from pathlib import Path
import torch
import matplotlib.pyplot as plt
import pandas as pd

# Adding parent directory to Python path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

# Import the spatial environment
from project_root.environment.custom_environment import TeenEducationEnvironment

# Custom REINFORCE implementation
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

class REINFORCEPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dims=[256, 128]):
        super(REINFORCEPolicy, self).__init__()
        
        layers = []
        prev_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, action_dim))
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return F.softmax(self.network(x), dim=-1)

class REINFORCEAgent:
    def __init__(self, state_dim, action_dim, lr=0.001, gamma=0.99):
        self.policy = REINFORCEPolicy(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma
        self.reset_episode()
        
    def reset_episode(self):
        self.log_probs = []
        self.rewards = []
        
    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        probs = self.policy(state)
        m = Categorical(probs)
        action = m.sample()
        self.log_probs.append(m.log_prob(action))
        return action.item()
    
    def update(self):
        # Calculate discounted rewards
        discounted_rewards = []
        R = 0
        for r in reversed(self.rewards):
            R = r + self.gamma * R
            discounted_rewards.insert(0, R)
        
        # Normalize rewards
        discounted_rewards = torch.FloatTensor(discounted_rewards)
        if len(discounted_rewards) > 1:
            discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-8)
        
        # Calculate policy loss
        policy_loss = []
        for log_prob, reward in zip(self.log_probs, discounted_rewards):
            policy_loss.append(-log_prob * reward)
        
        policy_loss = torch.stack(policy_loss).sum()
        
        # Update policy
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()
        
        self.reset_episode()
        return policy_loss.item()

def train_reinforce():
    """Train REINFORCE algorithm"""
    print("=== Training REINFORCE ===")
    env = TeenEducationEnvironment()
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    agent = REINFORCEAgent(state_dim, action_dim, lr=0.002, gamma=0.99)
    
    episode_rewards = []
    episode_lengths = []
    success_rates = []
    
    num_episodes = 500
    eval_freq = 200
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        agent.reset_episode()
        total_reward = 0
        steps = 0
        
        for step in range(300):  # Max steps per episode
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            
            agent.rewards.append(reward)
            total_reward += reward
            steps += 1
            
            if terminated or truncated:
                break
                
            state = next_state
        
        # Update policy at end of episode
        loss = agent.update()
        
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        
        # Evaluation
        if episode % eval_freq == 0:
            eval_rewards, success_rate = evaluate_agent(agent, env, n_episodes=10)
            success_rates.append(success_rate)
            print(f"Episode {episode}: Avg Reward={np.mean(episode_rewards[-eval_freq:]):.2f}, "
                  f"Success Rate={success_rate:.1f}%, Loss={loss:.3f}")
    
    env.close()
    
    results = {
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'success_rates': success_rates,
        'final_success_rate': success_rates[-1] if success_rates else 0.0,
        'mean_reward': np.mean(episode_rewards[-200:])  # Last 200 episodes
    }
    
    return agent, results

def train_ppo():
    """Train PPO algorithm"""
    print("=== Training PPO ===")
    env = TeenEducationEnvironment()
    
    # PPO Hyperparameters optimized for spatial environment
    hyperparameters = {
        "learning_rate": 0.0003,
        "n_steps": 2048,
        "batch_size": 64,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.01,  # Encourage exploration
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "target_kl": 0.02,
        "policy_kwargs": dict(
            net_arch=[dict(pi=[256, 256], vf=[256, 256])],
            activation_fn=torch.nn.ReLU
        )
    }
    
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=hyperparameters["learning_rate"],
        n_steps=hyperparameters["n_steps"],
        batch_size=hyperparameters["batch_size"],
        n_epochs=hyperparameters["n_epochs"],
        gamma=hyperparameters["gamma"],
        gae_lambda=hyperparameters["gae_lambda"],
        clip_range=hyperparameters["clip_range"],
        ent_coef=hyperparameters["ent_coef"],
        vf_coef=hyperparameters["vf_coef"],
        max_grad_norm=hyperparameters["max_grad_norm"],
        target_kl=hyperparameters["target_kl"],
        policy_kwargs=hyperparameters["policy_kwargs"],
        verbose=1
    )
    
    # Train model
    total_timesteps = 100000
    model.learn(total_timesteps=total_timesteps, progress_bar=True)
    
    # Save model
    model.save("ppo_spatial_teen_education")
    print("PPO model saved as 'ppo_spatial_teen_education.zip'")
    
    # Evaluate
    results = evaluate_sb3_model(model, "PPO")
    
    return model, results

def train_a2c():
    """Train A2C (Actor-Critic) algorithm"""
    print("=== Training A2C (Actor-Critic) ===")
    env = TeenEducationEnvironment()
    
    # A2C Hyperparameters
    hyperparameters = {
        "learning_rate": 0.0007,
        "n_steps": 8,  # Shorter rollouts for A2C
        "gamma": 0.99,
        "gae_lambda": 1.0,
        "ent_coef": 0.01,
        "vf_coef": 0.25,
        "max_grad_norm": 0.5,
        "rms_prop_eps": 1e-5,
        "use_rms_prop": True,
        "policy_kwargs": dict(
            net_arch=[dict(pi=[256, 128], vf=[256, 128])],
            activation_fn=torch.nn.ReLU
        )
    }
    
    model = A2C(
        policy="MlpPolicy",
        env=env,
        learning_rate=hyperparameters["learning_rate"],
        n_steps=hyperparameters["n_steps"],
        gamma=hyperparameters["gamma"],
        gae_lambda=hyperparameters["gae_lambda"],
        ent_coef=hyperparameters["ent_coef"],
        vf_coef=hyperparameters["vf_coef"],
        max_grad_norm=hyperparameters["max_grad_norm"],
        rms_prop_eps=hyperparameters["rms_prop_eps"],
        use_rms_prop=hyperparameters["use_rms_prop"],
        policy_kwargs=hyperparameters["policy_kwargs"],
        verbose=1
    )
    
    # Train model
    total_timesteps = 100000
    model.learn(total_timesteps=total_timesteps, progress_bar=True)
    
    # Save model
    model.save("a2c_spatial_teen_education")
    print("A2C model saved as 'a2c_spatial_teen_education.zip'")
    
    # Evaluate
    results = evaluate_sb3_model(model, "A2C")
    
    return model, results

def evaluate_agent(agent, env, n_episodes=20):
    """Evaluate REINFORCE agent"""
    episode_rewards = []
    successes = 0
    
    for _ in range(n_episodes):
        state, _ = env.reset()
        total_reward = 0
        
        for _ in range(300):
            action = agent.select_action(state)
            state, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            if terminated or truncated:
                if info['girls_in_room'] == env.num_girls:
                    successes += 1
                break
        
        episode_rewards.append(total_reward)
    
    success_rate = (successes / n_episodes) * 100
    return episode_rewards, success_rate

def evaluate_sb3_model(model, model_name):
    """Evaluate Stable-Baselines3 models (PPO, A2C)"""
    env = TeenEducationEnvironment()
    
    episode_rewards = []
    successes = 0
    total_girls_empowered = 0
    steps_to_success = []
    
    print(f"\nEvaluating {model_name} model...")
    
    for episode in range(20):
        obs, _ = env.reset()
        total_reward = 0
        steps = 0
        
        for step in range(300):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            
            if terminated or truncated:
                girls_in_room = info['girls_in_room']
                total_girls_empowered += girls_in_room
                
                if girls_in_room == env.num_girls:
                    successes += 1
                    steps_to_success.append(steps)
                
                break
        
        episode_rewards.append(total_reward)
        print(f"Episode {episode+1}: Reward={total_reward:.1f}, Girls in Room={info['girls_in_room']}/{env.num_girls}")
    
    env.close()
    
    results = {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'success_rate': (successes / 20) * 100,
        'avg_girls_empowered': total_girls_empowered / 20,
        'avg_steps_to_success': np.mean(steps_to_success) if steps_to_success else 0,
        'episode_rewards': episode_rewards
    }
    
    print(f"\n{model_name} Evaluation Results:")
    print(f"Mean Reward: {results['mean_reward']:.2f} +/- {results['std_reward']:.2f}")
    print(f"Success Rate: {results['success_rate']:.1f}%")
    print(f"Average Girls Empowered: {results['avg_girls_empowered']:.1f}/{env.num_girls}")
    print(f"Average Steps to Success: {results['avg_steps_to_success']:.1f}")
    
    return results

def compare_algorithms(reinforce_results, ppo_results, a2c_results):
    """Compare all three algorithms"""
    print("\n" + "="*60)
    print("ALGORITHM COMPARISON SUMMARY")
    print("="*60)
    
    algorithms = ['REINFORCE', 'PPO', 'A2C']
    results = [reinforce_results, ppo_results, a2c_results]
    
    # Create comparison table
    comparison_data = []
    for alg, result in zip(algorithms, results):
        success_rate = result.get('success_rate', result.get('final_success_rate', 0))
        avg_girls_empowered = result.get('avg_girls_empowered', 'N/A')
        avg_steps_to_success = result.get('avg_steps_to_success', 0)

        comparison_data.append({
            'Algorithm': alg,
            'Mean Reward': f"{result['mean_reward']:.2f}",
            'Success Rate (%)': f"{success_rate:.1f}",
            'Avg Girls Empowered': f"{avg_girls_empowered:.1f}" if avg_girls_empowered != 'N/A' else 'N/A',
            'Avg Steps to Success': f"{avg_steps_to_success:.1f}"
        })

    
    df = pd.DataFrame(comparison_data)
    print(df.to_string(index=False))
    
    # Find best performer - handle different key names
    success_rates = []
    for result in results:
        success_rates.append(result.get('success_rate', result.get('final_success_rate', 0)))
    
    best_success_idx = np.argmax(success_rates)
    best_reward_idx = np.argmax([r['mean_reward'] for r in results])
    
    best_success_alg = algorithms[best_success_idx]
    best_reward_alg = algorithms[best_reward_idx]
    
    print(f"\nBest Success Rate: {best_success_alg} ({success_rates[best_success_idx]:.1f}%)")
    print(f"Best Mean Reward: {best_reward_alg} ({results[best_reward_idx]['mean_reward']:.2f})")
    
    # Plot comparison
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Success rates
    axes[0].bar(algorithms, success_rates, color=['blue', 'green', 'red'], alpha=0.7)
    axes[0].set_title('Success Rate Comparison')
    axes[0].set_ylabel('Success Rate (%)')
    axes[0].set_ylim(0, 100)
    
    # Mean rewards
    mean_rewards = [r['mean_reward'] for r in results]
    axes[1].bar(algorithms, mean_rewards, color=['blue', 'green', 'red'], alpha=0.7)
    axes[1].set_title('Mean Reward Comparison')
    axes[1].set_ylabel('Mean Reward')
    
    plt.tight_layout()
    plt.savefig('algorithm_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return df

def demo_best_model(models, results):
    """Demonstrate the best performing model"""
    # Find best model based on success rate - handle different key names
    success_rates = []
    for result in results:
        success_rates.append(result.get('success_rate', result.get('final_success_rate', 0)))
    
    best_idx = np.argmax(success_rates)
    best_model = models[best_idx]
    best_name = ['REINFORCE', 'PPO', 'A2C'][best_idx]
    
    print(f"\n=== Demonstrating Best Model: {best_name} ===")
    
    env = TeenEducationEnvironment(render_mode="human")
    
    for episode in range(2):
        print(f"\nEpisode {episode + 1}")
        obs, _ = env.reset()
        total_reward = 0
        
        for step in range(200):
            if best_name == 'REINFORCE':
                action = best_model.select_action(obs)
            else:
                action, _ = best_model.predict(obs, deterministic=True)
            
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            if step % 20 == 0:  # Print every 20 steps
                print(f"Step {step}: Girls in room: {info['girls_in_room']}/{env.num_girls}, "
                      f"Trust: {info['average_trust']:.1f}%, Reward: {total_reward:.1f}")
            
            env.render()
            time.sleep(0.05)
            
            if terminated or truncated:
                print(f"Episode ended! Final: {info['girls_in_room']}/{env.num_girls} girls empowered")
                break
    
    env.close()

def create_training_gifs(models):
    """Create GIFs for each trained model"""
    model_names = ['REINFORCE', 'PPO', 'A2C']
    
    for model, name in zip(models, model_names):
        print(f"Creating GIF for {name}...")
        env = TeenEducationEnvironment(render_mode="rgb_array")
        frames = []
        
        obs, _ = env.reset()
        for step in range(150):
            if name == 'REINFORCE':
                action = model.select_action(obs)
            else:
                action, _ = model.predict(obs, deterministic=True)
            
            obs, reward, terminated, truncated, info = env.step(action)
            frame = env.render()
            frames.append(frame)
            
            if terminated or truncated:
                # Add final frames
                for _ in range(10):
                    frames.append(frame)
                break
        
        env.close()
        imageio.mimsave(f"{name.lower()}_demo.gif", frames, duration=0.1)
        print(f"{name} demo GIF saved as '{name.lower()}_demo.gif'")

if __name__ == "__main__":
    print("Training Policy Gradient Algorithms for Spatial Teen Education Environment")
    print("="*80)
    
    # Train all three algorithms
    reinforce_agent, reinforce_results = train_reinforce()
    ppo_model, ppo_results = train_ppo()
    a2c_model, a2c_results = train_a2c()
    
    # Compare results
    comparison_df = compare_algorithms(reinforce_results, ppo_results, a2c_results)
    
    # Demo best model
    models = [reinforce_agent, ppo_model, a2c_model]
    results = [reinforce_results, ppo_results, a2c_results]
    demo_best_model(models, results)
    
    # Create GIFs
    create_training_gifs(models)
    
    print("\nTraining and evaluation complete!")
    print("Check the generated plots and GIFs for visual results.")