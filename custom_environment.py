import gymnasium as gym
import numpy as np
from gymnasium import spaces
import pygame
import imageio
import asyncio
import platform

class MentorChatbotEnv(gym.Env):
    def __init__(self):
        super(MentorChatbotEnv, self).__init__()
        # Define states
        self.states = {
            "Unaware": 0,
            "Informed": 1,
            "Empowered": 2,
            "Confused": 3,
            "Embarrassed": 4
        }
        self.state_descriptions = {v: k for k, v in self.states.items()}
        self.actions = {
            0: "Explain",
            1: "Share Story",
            2: "Ask Question",
            3: "Reassure",
            4: "Ignore"
        }
        # Define action space: 0=Explain, 1=Share Story, 2=Ask Question, 3=Reassure, 4=Ignore
        self.action_space = spaces.Discrete(5)
        # Define observation space (state as integer)
        self.observation_space = spaces.Discrete(len(self.states))
        # Initialize state
        self.state = None
        self.step_count = 0
        self.max_steps = 20  # Max steps per episode

        # Pygame setup
        pygame.init()
        self.screen_width = 800
        self.screen_height = 600
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Mentor Chatbot Environment")
        self.font = pygame.font.SysFont("arial", 24)
        self.state_colors = {
            0: (200, 200, 200),  # Unaware: Gray
            1: (100, 100, 255),  # Informed: Blue
            2: (100, 255, 100),  # Empowered: Green
            3: (255, 100, 100),  # Confused: Red
            4: (255, 182, 193)   # Embarrassed: Pink
        }
        # env.frames = []  # Store frames for GIF

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Set initial state randomly to Unaware, Confused, or Embarrassed
        self.state = np.random.choice([self.states["Unaware"], self.states["Confused"], self.states["Embarrassed"]])
        self.step_count = 0
        info = {}
        return self.state, info

    def step(self, action):
        self.step_count += 1
        reward = 0
        done = False
        truncated = False

        # Define state transition probabilities and rewards
        if self.state == self.states["Unaware"]:
            if action == 0:  # Explain
                self.state = self.states["Informed"]
                reward = 10  # Improvement in knowledge
            elif action == 3:  # Reassure
                self.state = self.states["Confused"]
                reward = -5  # Not addressing ignorance
            elif action == 4:  # Ignore
                reward = -10  # Wrong action
            else:  # Share Story or Ask Question
                reward = -2  # Ineffective

        elif self.state == self.states["Informed"]:
            if action == 1:  # Share Story
                self.state = self.states["Empowered"]
                reward = 10  # Improvement in knowledge
            elif action == 2:  # Ask Question
                self.state = self.states["Empowered"]
                reward = 5   # Encourages reflection
            elif action == 4:  # Ignore
                reward = -10  # Wrong action
            else:  # Explain or Reassure
                reward = -2  # Redundant or ineffective

        elif self.state == self.states["Empowered"]:
            if action in [1, 2, 3]:  # Share Story, Ask Question, Reassure
                reward = 5  # Maintains positive state
            elif action == 4:  # Ignore
                reward = -10  # Wrong action
            else:  # Explain
                reward = -2  # Redundant

        elif self.state == self.states["Confused"]:
            if action == 0:  # Explain
                self.state = self.states["Informed"]
                reward = 10  # Improvement in knowledge
            elif action == 3:  # Reassure
                self.state = self.states["Embarrassed"]
                reward = 5   # Emotional uplift
            elif action == 4:  # Ignore
                reward = -10  # Wrong action
            else:  # Share Story or Ask Question
                reward = -2  # Ineffective

        elif self.state == self.states["Embarrassed"]:
            if action == 3:  # Reassure
                self.state = self.states["Confused"]
                reward = 5   # Emotional uplift
            elif action == 0:  # Explain
                self.state = self.states["Informed"]
                reward = 10  # Improvement in knowledge
            elif action == 4:  # Ignore
                reward = -10  # Wrong action
            else:  # Share Story or Ask Question
                reward = -2  # Ineffective

        # Check if episode is done
        if self.state == self.states["Empowered"]:
            done = True
            reward += 20  # Bonus for reaching Empowered
        if self.step_count >= self.max_steps:
            truncated = True

        info = {}
        return self.state, reward, done, truncated, info

    def render(self):
        # Clear screen with state-based background color
        self.screen.fill(self.state_colors[self.state])

        # Draw mentor (left) and girl (right) as simple shapes
        pygame.draw.circle(self.screen, (0, 128, 255), (150, 300), 50)  # Mentor: Blue circle
        pygame.draw.circle(self.screen, (255, 105, 180), (650, 300), 50)  # Girl: Pink circle

        # Draw labels
        mentor_label = self.font.render("Mentor", True, (0, 0, 0))
        girl_label = self.font.render("Girl", True, (0, 0, 0))
        self.screen.blit(mentor_label, (130, 360))
        self.screen.blit(girl_label, (630, 360))

        # Draw state label
        state_label = self.font.render(f"State: {self.state_descriptions[self.state]}", True, (0, 0, 0))
        self.screen.blit(state_label, (20, 20))

        # Capture frame
        frame = pygame.surfarray.array3d(self.screen)
        self.frames.append(np.transpose(frame, (1, 0, 2)))  # Transpose for imageio

        pygame.display.flip()

    def draw_speech_bubble(self, action):
        # Draw speech bubble above mentor
        pygame.draw.rect(self.screen, (255, 255, 255), (100, 100, 200, 80))  # White bubble
        pygame.draw.polygon(self.screen, (255, 255, 255), [(150, 180), (170, 180), (160, 200)])  # Tail
        action_text = self.font.render(self.actions[action], True, (0, 0, 0))
        self.screen.blit(action_text, (120, 140))

    def close(self):
        pygame.quit()

# Example Q-learning agent to train the environment
# def train_q_learning(env, episodes=1000, alpha=0.1, gamma=0.99, epsilon=0.1):
#     q_table = np.zeros((env.observation_space.n, env.action_space.n))
#     for episode in range(episodes):
#         state, _ = env.reset()
#         done = False
#         truncated = False
#         total_reward = 0
#         while not (done or truncated):
#             # Epsilon-greedy action selection
#             if np.random.random() < epsilon:
#                 action = env.action_space.sample()
#             else:
#                 action = np.argmax(q_table[state])
#             next_state, reward, done, truncated, _ = env.step(action)
#             # Q-update
#             q_table[state, action] += alpha * (reward + gamma * np.max(q_table[next_state]) - q_table[state, action])
#             state = next_state
#             total_reward += reward
#         if episode % 100 == 0:
#             print(f"Episode {episode}, Total Reward: {total_reward}")
#     return q_table

# if __name__ == "__main__":
#     env = MentorChatbotEnv()
#     q_table = train_q_learning(env)
#     # Test the trained agent
#     state, _ = env.reset()
#     env.render()
#     done = False
#     truncated = False
#     actions = {0: "Explain", 1: "Share Story", 2: "Ask Question", 3: "Reassure", 4: "Ignore"}
#     while not (done or truncated):
#         action = np.argmax(q_table[state])
#         print(f"Action: {actions[action]}")
#         state, reward, done, truncated, _ = env.step(action)
#         env.render()
#         print(f"Reward: {reward}\n")

async def main():
    env = MentorChatbotEnv()
    FPS = 10
    clock = pygame.time.Clock()
    state, _ = env.reset()
    env.frames = []  # Clear old frames

    for _ in range(10):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        action = env.action_space.sample()
        state, reward, done, truncated, _ = env.step(action)
        env.render()
        env.draw_speech_bubble(action)

        # Capture and store frame
        frame = pygame.surfarray.array3d(env.screen)
        env.frames.append(np.transpose(frame, (1, 0, 2)))

        if done or truncated:
            state, _ = env.reset()
        await asyncio.sleep(1.0 / FPS)
        clock.tick(FPS)

    # Save to GIF
    imageio.mimsave("mentor_chatbot.gif", env.frames, duration=1.0/FPS, loop=0)
    env.close()
