import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from stable_baselines3 import DQN
import gymnasium as gym
from pathlib import Path
import sys

parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))
from static_simulation import TeenEducationEnvironment

def train_reinforce(env, hyperparameters=None):
    returns_history = []
    if hyperparameters is None:
        hyperparameters = {
            "learning_rate": 0.001,
            "gamma": 0.99,
            "total_episodes": 2000,
            "hidden_dim": 128,
            "batch_size": 1,  # REINFORCE typically uses full episodes
            "baseline": True,  # Use baseline for variance reduction
            "entropy_coeff": 0.01,  # Encourage exploration
            "grad_clip": 0.5,  # Gradient clipping
            "log_interval": 100
        }
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    # Policy network
    policy = nn.Sequential(
        nn.Linear(state_dim, hyperparameters["hidden_dim"]),
        nn.ReLU(),
        nn.Linear(hyperparameters["hidden_dim"], hyperparameters["hidden_dim"]),
        nn.ReLU(),
        nn.Linear(hyperparameters["hidden_dim"], action_dim),
        nn.Softmax(dim=-1)
    )
    
    # Baseline network (optional)
    baseline = nn.Sequential(
        nn.Linear(state_dim, hyperparameters["hidden_dim"]),
        nn.ReLU(),
        nn.Linear(hyperparameters["hidden_dim"], 1)
    ) if hyperparameters["baseline"] else None
    
    optimizer = optim.Adam(policy.parameters(), lr=hyperparameters["learning_rate"])
    baseline_optimizer = optim.Adam(baseline.parameters(), lr=hyperparameters["learning_rate"]) if baseline else None
    
    for episode in range(hyperparameters["total_episodes"]):
        states, actions, rewards, log_probs = [], [], [], []
        state, _ = env.reset()
        done = False
        
        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            probs = policy(state_tensor)
            action = torch.multinomial(probs, 1).item()
            log_prob = torch.log(probs[0][action])
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            log_probs.append(log_prob)
            state = next_state
        
        # Calculate returns
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + hyperparameters["gamma"] * R
            returns.insert(0, R)
        returns = torch.FloatTensor(returns)
        
        # Normalize returns
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Calculate baseline if used
        if baseline:
            states_tensor = torch.tensor(np.array(states), dtype=torch.float32)
            baselines = baseline(states_tensor).squeeze()
            advantages = (returns - baselines).detach()
            
            # Update baseline
            baseline_loss = nn.MSELoss()(baselines, returns.detach())
            baseline_optimizer.zero_grad()
            baseline_loss.backward()
            baseline_optimizer.step()
        else:
            advantages = returns
        
        # Policy loss with entropy bonus
        policy_loss = []
        entropy_loss = 0
        for i, log_prob in enumerate(log_probs):
            policy_loss.append(-log_prob * advantages[i])
            # Add entropy for exploration
            probs = policy(torch.FloatTensor(states[i]).unsqueeze(0))
            entropy_loss += -(probs * torch.log(probs + 1e-8)).sum()
        
        total_loss = torch.stack(policy_loss).sum() - hyperparameters["entropy_coeff"] * entropy_loss
        
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), hyperparameters["grad_clip"])
        optimizer.step()
        
        if episode % hyperparameters["log_interval"] == 0:
            returns_history.append(sum(rewards))
            if len(returns_history) > 100:
                returns_history.pop(0)
            if episode % hyperparameters["log_interval"] == 0:
                avg_return = np.mean(returns_history)
                print(f"Episode {episode}, Return: {sum(rewards):.2f}, Avg(100): {avg_return:.2f}")

    return policy

def train_ppo(env, hyperparameters=None):
    if hyperparameters is None:
        hyperparameters = {
            "learning_rate": 0.0003,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_param": 0.2,
            "value_coeff": 0.5,
            "entropy_coeff": 0.01,
            "max_grad_norm": 0.5,
            "ppo_epochs": 10,
            "batch_size": 64,
            "total_timesteps": 100000,
            "n_steps": 2048,
            "hidden_dim": 128,
            "target_kl": 0.01,
            "log_interval": 10
        }

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    actor = nn.Sequential(
        nn.Linear(state_dim, hyperparameters["hidden_dim"]),
        nn.ReLU(),
        nn.Linear(hyperparameters["hidden_dim"], hyperparameters["hidden_dim"]),
        nn.ReLU(),
        nn.Linear(hyperparameters["hidden_dim"], action_dim),
        nn.Softmax(dim=-1)
    )

    critic = nn.Sequential(
        nn.Linear(state_dim, hyperparameters["hidden_dim"]),
        nn.ReLU(),
        nn.Linear(hyperparameters["hidden_dim"], hyperparameters["hidden_dim"]),
        nn.ReLU(),
        nn.Linear(hyperparameters["hidden_dim"], 1)
    )

    optimizer = optim.Adam(list(actor.parameters()) + list(critic.parameters()),
                           lr=hyperparameters["learning_rate"])

    obs_buffer, act_buffer, adv_buffer, ret_buffer, logp_buffer = [], [], [], [], []
    best_reward = float("-inf")

    total_steps = 0
    episode_rewards = []

    while total_steps < hyperparameters["total_timesteps"]:
        obs, _ = env.reset()
        done = False
        ep_rewards = []
        transitions = []

        for _ in range(hyperparameters["n_steps"]):
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            probs = actor(obs_tensor)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)

            next_obs, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated

            value = critic(obs_tensor).item()

            transitions.append((obs, action.item(), reward, log_prob.item(), value))
            obs = next_obs
            ep_rewards.append(reward)

            if done:
                obs, _ = env.reset()
                episode_rewards.append(sum(ep_rewards))
                ep_rewards = []

        # Unpack transitions
        obs_list, act_list, rew_list, logp_list, val_list = zip(*transitions)
        obs_tensor = torch.tensor(obs_list, dtype=torch.float32)
        act_tensor = torch.tensor(act_list)
        old_logp_tensor = torch.tensor(logp_list)
        val_tensor = torch.tensor(val_list)

        # Compute returns and advantages using GAE
        returns = []
        advs = []
        gae = 0
        next_value = 0
        for t in reversed(range(len(rew_list))):
            delta = rew_list[t] + hyperparameters["gamma"] * next_value - val_list[t]
            gae = delta + hyperparameters["gamma"] * hyperparameters["gae_lambda"] * gae
            advs.insert(0, gae)
            next_value = val_list[t]
            returns.insert(0, gae + val_list[t])

        adv_tensor = torch.tensor(advs, dtype=torch.float32)
        ret_tensor = torch.tensor(returns, dtype=torch.float32)

        # Normalize advantages
        adv_tensor = (adv_tensor - adv_tensor.mean()) / (adv_tensor.std() + 1e-8)

        # PPO Update
        for _ in range(hyperparameters["ppo_epochs"]):
            for i in range(0, len(obs_tensor), hyperparameters["batch_size"]):
                end = i + hyperparameters["batch_size"]
                batch_obs = obs_tensor[i:end]
                batch_act = act_tensor[i:end]
                batch_adv = adv_tensor[i:end]
                batch_ret = ret_tensor[i:end]
                batch_old_logp = old_logp_tensor[i:end]

                probs = actor(batch_obs)
                dist = torch.distributions.Categorical(probs)
                entropy = dist.entropy().mean()
                new_logp = dist.log_prob(batch_act)

                ratio = torch.exp(new_logp - batch_old_logp)
                surr1 = ratio * batch_adv
                surr2 = torch.clamp(ratio, 1.0 - hyperparameters["clip_param"],
                                             1.0 + hyperparameters["clip_param"]) * batch_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                value_preds = critic(batch_obs).squeeze()
                value_loss = nn.MSELoss()(value_preds, batch_ret)

                total_loss = policy_loss + hyperparameters["value_coeff"] * value_loss - \
                             hyperparameters["entropy_coeff"] * entropy

                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(list(actor.parameters()) + list(critic.parameters()),
                                               hyperparameters["max_grad_norm"])
                optimizer.step()

                # Optional: early stopping for KL
                approx_kl = (batch_old_logp - new_logp).mean().item()
                if approx_kl > hyperparameters["target_kl"]:
                    break

        total_steps += hyperparameters["n_steps"]

        if len(episode_rewards) > 0 and total_steps % (hyperparameters["log_interval"] * hyperparameters["n_steps"]) == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            print(f"Step {total_steps}, Average Reward (last 10): {avg_reward:.2f}")
            if avg_reward > best_reward:
                best_reward = avg_reward
                torch.save(actor.state_dict(), "best_ppo_actor.pth")
                torch.save(critic.state_dict(), "best_ppo_critic.pth")
                print("Saved best PPO model")

    return actor, critic

def train_a2c(env, hyperparameters=None):
    if hyperparameters is None:
        hyperparameters = {
            "learning_rate": 0.0007,
            "gamma": 0.99,
            "gae_lambda": 1.0,
            "value_coeff": 0.25,
            "entropy_coeff": 0.01,
            "max_grad_norm": 0.5,
            "n_steps": 5,
            "total_timesteps": 100000,
            "hidden_dim": 128,
            "rms_prop_eps": 1e-5,
            "alpha": 0.99,
            "log_interval": 100
        }

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    shared_net = nn.Sequential(
        nn.Linear(state_dim, hyperparameters["hidden_dim"]),
        nn.ReLU(),
        nn.Linear(hyperparameters["hidden_dim"], hyperparameters["hidden_dim"]),
        nn.ReLU()
    )

    actor_head = nn.Sequential(
        nn.Linear(hyperparameters["hidden_dim"], action_dim),
        nn.Softmax(dim=-1)
    )

    critic_head = nn.Linear(hyperparameters["hidden_dim"], 1)

    optimizer = optim.RMSprop(
        list(shared_net.parameters()) + list(actor_head.parameters()) + list(critic_head.parameters()),
        lr=hyperparameters["learning_rate"],
        alpha=hyperparameters["alpha"],
        eps=hyperparameters["rms_prop_eps"]
    )

    obs, _ = env.reset()
    total_steps = 0
    best_reward = float("-inf")
    episode_rewards = []

    while total_steps < hyperparameters["total_timesteps"]:
        states, actions, rewards, values, log_probs = [], [], [], [], []

        for _ in range(hyperparameters["n_steps"]):
            state_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            features = shared_net(state_tensor)
            probs = actor_head(features)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)

            value = critic_head(features).squeeze()

            next_obs, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated

            states.append(state_tensor)
            actions.append(action)
            rewards.append(reward)
            values.append(value)
            log_probs.append(log_prob)

            obs = next_obs
            total_steps += 1

            if done:
                episode_rewards.append(sum(rewards))
                obs, _ = env.reset()
                break

        # Bootstrap next value
        with torch.no_grad():
            next_state_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            next_value = critic_head(shared_net(next_state_tensor)).squeeze().item()

        returns = []
        R = next_value
        for r in reversed(rewards):
            R = r + hyperparameters["gamma"] * R
            returns.insert(0, R)
        returns = torch.tensor(returns)

        values_tensor = torch.stack(values)
        log_probs_tensor = torch.stack(log_probs)
        advantages = returns - values_tensor

        policy_loss = -(log_probs_tensor * advantages.detach()).mean()
        value_loss = advantages.pow(2).mean()
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1).mean()

        total_loss = policy_loss + hyperparameters["value_coeff"] * value_loss - \
                     hyperparameters["entropy_coeff"] * entropy

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(shared_net.parameters()) + list(actor_head.parameters()) + list(critic_head.parameters()),
            hyperparameters["max_grad_norm"]
        )
        optimizer.step()

        if total_steps % (hyperparameters["log_interval"] * hyperparameters["n_steps"]) == 0:
            avg_reward = np.mean(episode_rewards[-10:]) if episode_rewards else 0
            print(f"Step {total_steps}, Average Reward (last 10): {avg_reward:.2f}")
            if avg_reward > best_reward:
                best_reward = avg_reward
                torch.save(shared_net.state_dict(), "best_a2c_shared.pth")
                torch.save(actor_head.state_dict(), "best_a2c_actor.pth")
                torch.save(critic_head.state_dict(), "best_a2c_critic.pth")
                print("Saved best A2C model")

    return shared_net, actor_head, critic_head

# Usage example:
def train_all_models():
    env = TeenEducationEnvironment()
    
    # Train REINFORCE
    reinforce_policy = train_reinforce(env)
    
    # Train PPO
    ppo_actor, ppo_critic = train_ppo(env)
    
    # Train A2C
    a2c_shared, a2c_actor, a2c_critic = train_a2c(env)
    
    return reinforce_policy, (ppo_actor, ppo_critic), (a2c_shared, a2c_actor, a2c_critic)

if __name__ == "__main__":
    train_all_models()