from stable_baselines3 import A2C
from pathlib import Path
import sys

# Include environment path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from static_simulation import TeenEducationEnvironment

from static_simulation import TeenEducationEnvironment

def play_a2c_model():
    env = TeenEducationEnvironment(render_mode="human")
    model = A2C.load("a2c_teen_edu")
    
    episodes = 3
    for episode in range(episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        step = 0
        print(f"\n=== Episode {episode + 1}/{episodes} ===")
        while not done:
            action, _ = model.predict(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            env.render()
            total_reward += reward
            step += 1
            done = terminated or truncated
        print(f"Episode {episode + 1} finished with reward {total_reward:.2f} in {step} steps")
    env.close()

if __name__ == "__main__":
    play_a2c_model()
