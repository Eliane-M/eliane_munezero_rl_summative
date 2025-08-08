# Reinforcement Learning for Teenage Girl Empowerment

## Project Overview
This project implements and evaluates **Reinforcement Learning (RL)** algorithms to train an **AI mentor agent** in a simulated spatial-social environment.  
The environment represents the internet space (website and mobile app) where teenage girls interact with a chatbot mentor to gain more knowledge and learn best practices as their bodies undergo puberty.
The goal is to prevent teenage pregnancies as most of them arise from the teens being uninformed, and leaving them empowered to make informed decisions.

The AI agent learns **optimal strategies** for guiding teenagers from states of low awareness to states of full empowerment, considering **trust building, engagement, and knowledge retention**.

---

## Objectives
1. Model educational interactions as a **Markov Decision Process (MDP)**.
2. Train agents to **navigate a grid-based environment** and deliver targeted interventions.
3. Evaluate the effectiveness of different RL algorithms:
   - **Deep Q-Network (DQN)**
   - **Proximal Policy Optimization (PPO)**
   - **Advantage Actor-Critic (A2C)**
   - **REINFORCE**
4. Compare **sample efficiency, success rate, and stability** across algorithms.

---

## Environment Description

The 

### State Space
- **Type:** Continuous Box space `(shape=(30,))`
- **Agent State (2 values)**  
  - X coordinate: `0-19`  
  - Y coordinate: `0-14`
- **Girl States (5 per girl × 5 girls = 25 values)**  
  For each girl: position, emotional/knowledge state, trust level, empowered room status.
- **Global Metrics (3 values)**  
  - Number of girls in empowered room (0-5)  
  - Average trust (0-100)  
  - Steps remaining (0-300)

### Action Space
- **Discrete Actions** (Movement + Educational Interventions):
  1. Move in 8 directions or stay still
  2. Provide educational talks
  3. Share resources
  4. Offer emotional support
  5. Guide to empowered room

### Rewards
- Positive rewards for:
  - Increasing trust
  - Successfully empowering girls
  - Guiding girls into the empowered room
- Negative rewards for:
  - Ineffective interactions
  - Moving without purpose
  - Attempting actions in the wrong context

---

## Algorithms Implemented

### 1. Deep Q-Network (DQN)
- **Framework:** Stable-Baselines3 DQN  
- **Architecture:** MLP `[64, 64, 64]`  
- **Key Features:**  
  - Experience Replay Buffer  
  - Epsilon-Greedy Exploration  
  - Double Q-Learning  

### 2. Proximal Policy Optimization (PPO)
- **Framework:** Stable-Baselines3 PPO  
- **Architecture:** Actor-Critic with `[256, 256]` layers  
- **Special Features:**  
  - Clipped Surrogate Objective  
  - Generalized Advantage Estimation (GAE)  

### 3. Advantage Actor-Critic (A2C)
- **Framework:** Stable-Baselines3 A2C  
- **Architecture:** `[256, 128]`  
- **Special Features:**  
  - Synchronous updates  
  - RMSProp optimizer  

### 4. REINFORCE
- **Custom Implementation**  
- Monte Carlo policy gradient method  
- Softmax policy representation

---

## 📊 Results & Analysis

| Algorithm  | Success Rate (%) | Sample Efficiency | Stability | Training Time |
|------------|------------------|-------------------|-----------|---------------|
| **PPO**    | 84.2             | 8/10              | 9/10      | 19 mins       |
| **A2C**    | 72.1             | **Fastest**       | 8/10      | 10 mins       |
| **REINFORCE** | 71.3          | 6/10              | 7/10      | 20 mins       |
| **DQN**    | 66.1 (best config) | Low              | 5/10      | 53 mins       |

**Key Insights:**
- Policy gradient methods outperform DQN in spatial-social environments.
- DQN struggles with exploration and sample efficiency.
- PPO offers the best stability for deployment.
- A2C is the fastest to iterate for research.

---

## 📈 Graphs & Visualizations
Generated during training:
- Reward per Episode
  <img width="633" height="562" alt="image" src="https://github.com/user-attachments/assets/bce22091-8cd9-4db2-bfd5-44d5b2570b74" />

- Cumulative Rewards
  
- Success Rate over Time
  <img width="690" height="547" alt="image" src="https://github.com/user-attachments/assets/579c2e47-de71-4cb8-9d78-b23523c62cfa" />

- Action Distribution
- Average Q-Values (DQN)

---

## ⚙️ Installation & Setup

### Prerequisites
- Python 3.8+
- [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3)
- PyTorch
- NumPy
- Matplotlib
- Gymnasium

### Installation
```bash
pip install stable-baselines3[extra] torch numpy matplotlib gymnasium
