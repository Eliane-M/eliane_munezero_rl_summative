import argparse
import os
from static_simulation import TeenEducationEnvironment
from project_root.training.train_dqn import train_dqn
# from training.pg_training import train_ppo, train_a2c
import imageio
import numpy as np

def create_demo_gif():
    """Create a GIF showing random actions in the environment."""
    env = TeenEducationEnvironment(render_mode="rgb_array")
    frames = []
    
    observation, info = env.reset()
    
    for _ in range(30):  # 30 frames for gif
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        
        frame = env.render()
        frames.append(frame)
        
        if terminated or truncated:
            observation, info = env.reset()
    
    env.close()
    
    # Save as GIF
    imageio.mimsave("teen_education_demo.gif", frames, duration=0.5)
    print("Demo GIF saved as 'teen_education_demo.gif'")

def main():
    parser = argparse.ArgumentParser(description='Train RL models for Teen Education Environment')
    parser.add_argument('--mode', choices=['demo', 'train', 'evaluate'], 
                       default='demo', help='Mode to run')
    parser.add_argument('--algorithm', choices=['dqn', 'ppo', 'a2c'], 
                       default='dqn', help='Algorithm to train/evaluate')
    parser.add_argument('--timesteps', type=int, default=50000, 
                       help='Number of timesteps to train')
    
    args = parser.parse_args()
    
    if args.mode == 'demo':
        print("Creating demonstration GIF...")
        create_demo_gif()
        
        print("Running live demonstration...")
        from static_simulation import demo_random_actions
        demo_random_actions()
        
    elif args.mode == 'train':
        print(f"Training {args.algorithm.upper()} for {args.timesteps} timesteps...")
        
        # Create directories
        os.makedirs(f"models/{args.algorithm}", exist_ok=True)
        
        if args.algorithm == 'dqn':
            train_dqn(args.timesteps)
        elif args.algorithm == 'ppo':
            train_ppo(args.timesteps)
        elif args.algorithm == 'a2c':
            train_a2c(args.timesteps)
            
    elif args.mode == 'evaluate':
        print(f"Evaluating trained {args.algorithm.upper()} model...")
        # Implementation for evaluation would go here
        pass

if __name__ == "__main__":
    main()