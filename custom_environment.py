import gymnasium as gym
from gymnasium import spaces
import imageio
import numpy as np
import pygame
import random
from typing import Dict, Tuple, Optional, Any

class TeenEducationEnvironment(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 3}

    def __init__(self, render_mode: Optional[str] = None):
        super().__init__()

        self.max_steps = 50
        self.current_step = 0

        # Room setup: 6 rooms as blocks, not in a grid
        self.num_rooms = 6
        self.room_size = 100
        self.screen_width = 600
        self.screen_height = 600

        # Room positions (x, y) in a circular layout
        self.room_positions = {
            0: (300, 100),  # Unaware (top)
            1: (450, 200),  # Confused (top-right)
            2: (450, 400),  # Embarrassed (bottom-right)
            3: (300, 500),  # Curious (bottom)
            4: (150, 400),  # Informed (bottom-left)
            5: (150, 200)   # Empowered (top-left)
        }

        # States
        self.states = {
            0: "Unaware",
            1: "Confused",
            2: "Embarrassed",
            3: "Curious",
            4: "Informed",
            5: "Empowered"
        }

        # Actions: Move to specific rooms + physical actions
        self.actions = {
            0: "Move_To_Unaware",
            1: "Move_To_Confused",
            2: "Move_To_Embarrassed",
            3: "Move_To_Curious",
            4: "Move_To_Informed",
            5: "Move_To_Empowered",
            6: "Distribute_Materials",
            7: "Lead_Group_Activity",
            8: "Demonstrate_Skills",
            9: "Role_Play_Scenario",
            10: "Visit_Health_Clinic",
            11: "Organize_Peer_Session",
            12: "Provide_Visual_Aid",
            13: "Encourage_Journaling"
        }

        # Observation space: [mentor_room, teen1_state, teen1_trust, teen1_engagement, teen1_knowledge, teen1_room, ...]
        self.num_teens = 3
        self.observation_space = spaces.Box(
            low=np.array([0] + [0, 0, 0, 0, 0] * self.num_teens),
            high=np.array([self.num_rooms-1] + [5, 100, 100, 100, self.num_rooms-1] * self.num_teens),
            dtype=np.float32
        )

        self.action_space = spaces.Discrete(len(self.actions))

        # State variables
        self.mentor_room = 0  # Start in Unaware room
        self.teens = [
            {"state": 0, "trust": 50.0, "engagement": 50.0, "knowledge": 1.0, "room": 0},
            {"state": 0, "trust": 50.0, "engagement": 50.0, "knowledge": 1.0, "room": 0},
            {"state": 0, "trust": 50.0, "engagement": 50.0, "knowledge": 1.0, "room": 0}
        ]

        # Rendering setup
        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        self.font = None

        if self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            pygame.display.set_caption("Teen Education Environment")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont("arial", 14)

    def _get_observation(self) -> np.ndarray:
        obs = [self.mentor_room]
        for teen in self.teens:
            obs.extend([teen["state"], teen["trust"], teen["engagement"], teen["knowledge"], teen["room"]])
        return np.array(obs, dtype=np.float32)

    def _calculate_reward(self, action: int) -> float:
        reward = 0.0

        # Count teens in mentor's room and those in their state-specific room
        teens_in_room = sum(1 for teen in self.teens if teen["room"] == self.mentor_room)
        teens_in_state_room = sum(1 for teen in self.teens if teen["room"] == teen["state"])

        # State progression reward (average across teens)
        state_progression_reward = {0: -10, 1: -5, 2: -2, 3: 5, 4: 15, 5: 25}
        for teen in self.teens:
            reward += state_progression_reward[teen["state"]] / self.num_teens

        # Trust and engagement bonuses
        avg_trust = sum(teen["trust"] for teen in self.teens) / self.num_teens
        avg_engagement = sum(teen["engagement"] for teen in self.teens) / self.num_teens
        if avg_trust > 70:
            reward += 5
        elif avg_trust < 30:
            reward -= 10
        if avg_engagement > 70:
            reward += 5
        elif avg_engagement < 30:
            reward -= 5

        # Knowledge bonus
        avg_knowledge = sum(teen["knowledge"] for teen in self.teens) / self.num_teens
        reward += avg_knowledge * 0.1

        # Action-specific rewards
        action_rewards = {i: -1 for i in range(6)}

        # Dynamic action rewards (only applied if mentor and teen are in the same room)
        if action >= 6:
            teens_in_room = [teen for teen in self.teens if teen["room"] == self.mentor_room]

            if action == 6:
                reward += sum((5 if teen["state"] >= 3 else -5) for teen in teens_in_room)
            elif action == 7:
                reward += sum((10 if teen["engagement"] > 50 else -5) for teen in teens_in_room)
            elif action == 8:
                reward += sum((8 if teen["state"] >= 4 else -3) for teen in teens_in_room)
            elif action == 9:
                reward += sum((7 if teen["trust"] > 50 else -2) for teen in teens_in_room)
            elif action == 10:
                reward += sum((10 if teen["state"] >= 4 else -10) for teen in teens_in_room)
            elif action == 11:
                reward += sum((8 if teen["engagement"] > 60 else -5) for teen in teens_in_room)
            elif action == 12:
                reward += sum((6 if teen["knowledge"] < 80 else -2) for teen in teens_in_room)
            elif action == 13:
                reward += sum((5 if teen["state"] >= 3 else 0) for teen in teens_in_room)
        else:
            reward += action_rewards.get(action, 0)

        # Proximity bonus
        teens_in_same_room = [teen for teen in self.teens if teen["room"] == self.mentor_room]
        reward += 3 * len(teens_in_same_room)

        # State-room alignment bonus
        reward += 5 * teens_in_state_room

        # Empowerment bonus
        empowered_teens = sum(1 for teen in self.teens if teen["state"] == 5)
        reward += 50 * empowered_teens

        if action >= 3 and teens_in_room == 0:
            reward -= 5

        # # Penalize mentor for staying in the same room (not moving)
        # if hasattr(self, 'last_action'):
        #     if action < self.num_rooms and self.last_action < self.num_rooms:
        #         if action == self.last_action:
        #             reward -= 3

        return reward

    def _update_state(self, action: int) -> None:
        # Handle mentor movement
        if action < self.num_rooms:
            self.mentor_room = action

        # Update teens
        for teen in self.teens:
            in_same_room = (teen["room"] == self.mentor_room)
            # Apply action to teens in the same room
            if action >= self.num_rooms and in_same_room:
                transitions = self._get_transition_probabilities(action, teen["state"])
                states_list = list(transitions.keys())
                probabilities = list(transitions.values())
                teen["state"] = np.random.choice(states_list, p=probabilities)

            # Update trust, engagement, knowledge
            trust_changes = {
                0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0,
                6: 2 if in_same_room else -2,
                7: 5 if in_same_room else -2,
                8: 3 if in_same_room and teen["state"] >= 4 else -2,
                9: 4 if in_same_room else -2,
                10: 6 if in_same_room and teen["state"] >= 4 else -5,
                11: 5 if in_same_room else -2,
                12: 3 if in_same_room else -2,
                13: 2 if in_same_room else 0
            }
            trust_change = trust_changes.get(action, 0)
            if teen["state"] <= 1:
                trust_change *= 0.7
            elif teen["state"] >= 4:
                trust_change *= 1.3
            teen["trust"] = np.clip(teen["trust"] + trust_change + np.random.normal(0, 1), 0, 100)

            engagement_changes = {
                0: -1, 1: -1, 2: -1, 3: -1, 4: -1, 5: -1,
                6: 2 if in_same_room else -2,
                7: 8 if in_same_room else -2,
                8: 4 if in_same_room else -2,
                9: 6 if in_same_room else -2,
                10: 3 if in_same_room else -3,
                11: 7 if in_same_room else -2,
                12: 5 if in_same_room else -2,
                13: 4 if in_same_room else 0
            }
            engagement_change = engagement_changes.get(action, 0)
            teen["engagement"] = np.clip(teen["engagement"] + engagement_change + np.random.normal(0, 1.5), 0, 100)

            knowledge_gains = {
                0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0,
                6: 6 if in_same_room else 0,
                7: 4 if in_same_room else 0,
                8: 8 if in_same_room else 0,
                9: 5 if in_same_room else 0,
                10: 7 if in_same_room else 0,
                11: 4 if in_same_room else 0,
                12: 7 if in_same_room else 0,
                13: 3 if in_same_room else 0
            }
            knowledge_gain = knowledge_gains.get(action, 0)
            effectiveness_multiplier = (teen["trust"] + teen["engagement"]) / 150
            effective_gain = knowledge_gain * effectiveness_multiplier
            decay = 0.2
            teen["knowledge"] = np.clip(teen["knowledge"] - decay + effective_gain, 0, 100)

            # Teen movement: Move to their state’s room with probability
            move_prob = 0.6  # Base probability
            if in_same_room and action >= self.num_rooms:
                move_prob += 0.3  # Boost if mentor is present and performs action
            if random.random() < move_prob:
                teen["room"] = teen["state"]
                self.movement_flash = {"teen": teen, "step": self.current_step}  # For rendering flash


    def _get_transition_probabilities(self, action: int, current_state: int) -> Dict[int, float]:
        base_transitions = {
            0: {
                6: {0: 0.3, 1: 0.6, 3: 0.1},
                7: {0: 0.4, 1: 0.4, 3: 0.2},
                8: {0: 0.5, 1: 0.3, 2: 0.2},
                9: {0: 0.4, 1: 0.3, 3: 0.3},
                10: {0: 0.6, 1: 0.3, 2: 0.1},
                11: {0: 0.5, 1: 0.3, 3: 0.2},
                12: {0: 0.3, 1: 0.5, 3: 0.2},
                13: {0: 0.5, 1: 0.3, 3: 0.2}
            },
            1: {
                6: {1: 0.2, 3: 0.6, 4: 0.2},
                7: {1: 0.3, 3: 0.5, 4: 0.2},
                8: {1: 0.4, 2: 0.3, 4: 0.3},
                9: {1: 0.3, 3: 0.5, 4: 0.2},
                10: {1: 0.4, 2: 0.3, 4: 0.3},
                11: {1: 0.3, 3: 0.5, 4: 0.2},
                12: {1: 0.1, 3: 0.6, 4: 0.3},
                13: {1: 0.3, 3: 0.6, 4: 0.1}
            },
            2: {
                6: {2: 0.3, 3: 0.5, 4: 0.2},
                7: {2: 0.2, 3: 0.6, 4: 0.2},
                8: {2: 0.4, 3: 0.4, 4: 0.2},
                9: {2: 0.1, 3: 0.7, 4: 0.2},
                10: {2: 0.5, 3: 0.3, 4: 0.2},
                11: {2: 0.2, 3: 0.6, 4: 0.2},
                12: {2: 0.3, 3: 0.5, 4: 0.2},
                13: {2: 0.3, 3: 0.6, 4: 0.1}
            },
            3: {
                6: {3: 0.2, 4: 0.6, 5: 0.2},
                7: {3: 0.2, 4: 0.6, 5: 0.2},
                8: {3: 0.3, 4: 0.5, 5: 0.2},
                9: {3: 0.2, 4: 0.6, 5: 0.2},
                10: {3: 0.2, 4: 0.5, 5: 0.3},
                11: {3: 0.2, 4: 0.6, 5: 0.2},
                12: {3: 0.1, 4: 0.7, 5: 0.2},
                13: {3: 0.2, 4: 0.6, 5: 0.2}
            },
            4: {
                6: {4: 0.6, 5: 0.4},
                7: {4: 0.5, 5: 0.5},
                8: {4: 0.4, 5: 0.6},
                9: {4: 0.3, 5: 0.7},
                10: {4: 0.3, 5: 0.7},
                11: {4: 0.5, 5: 0.5},
                12: {4: 0.5, 5: 0.5},
                13: {4: 0.4, 5: 0.6}
            },
            5: {
                6: {5: 0.9, 4: 0.1},
                7: {5: 1.0},
                8: {5: 1.0},
                9: {5: 1.0},
                10: {5: 1.0},
                11: {5: 1.0},
                12: {5: 1.0},
                13: {5: 1.0}
            }
        }
        return base_transitions.get(current_state, {action: {current_state: 1.0}}).get(action, {current_state: 1.0})

    def _render_frame(self):
        if self.screen is None:
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            self.font = pygame.font.SysFont("arial", 14)

        self.screen.fill((200, 200, 200))

        # Draw rooms
        state_colors = {
            0: (200, 200, 200), 1: (255, 100, 100), 2: (255, 182, 193),
            3: (255, 255, 100), 4: (100, 100, 255), 5: (100, 255, 100)
        }
        for room_id, pos in self.room_positions.items():
            rect = pygame.Rect(pos[0] - self.room_size // 2, pos[1] - self.room_size // 2, self.room_size, self.room_size)
            pygame.draw.rect(self.screen, state_colors[room_id], rect)
            pygame.draw.rect(self.screen, (50, 50, 50), rect, 1)
            state_label = self.font.render(self.states[room_id], True, (0, 0, 0))
            self.screen.blit(state_label, (pos[0] - 40, pos[1] - 50))

        # Draw mentor
        mentor_pos = self.room_positions[self.mentor_room]
        pygame.draw.circle(self.screen, (0, 128, 255), mentor_pos, 15)
        mentor_label = self.font.render("Mentor", True, (255, 255, 255))
        self.screen.blit(mentor_label, (mentor_pos[0] - 20, mentor_pos[1] + 20))

        # Draw teens
        teen_colors = [(255, 105, 180), (128, 0, 128), (255, 165, 0)]  # Pink, Purple, Orange
        for i, teen in enumerate(self.teens):
            teen_pos = self.room_positions[teen["room"]]
            offset_x = -10 + 10 * i
            pygame.draw.circle(self.screen, teen_colors[i], (teen_pos[0] + offset_x, teen_pos[1]), 15)
            # Flash effect for movement
            if hasattr(self, 'movement_flash') and self.movement_flash["teen"] is teen and self.movement_flash["step"] == self.current_step:
                pygame.draw.circle(self.screen, (255, 255, 255), (teen_pos[0] + offset_x, teen_pos[1]), 20, 2)
            teen_label = self.font.render(f"Teen {i+1}: {self.states[teen['state']]}", True, (0, 0, 0))
            self.screen.blit(teen_label, (teen_pos[0] - 40, teen_pos[1] + 30))
            metrics = f"T:{teen['trust']:.1f} E:{teen['engagement']:.1f} K:{teen['knowledge']:.1f}"
            self.screen.blit(self.font.render(metrics, True, (0, 0, 0)), (teen_pos[0] - 40, teen_pos[1] + 45))

        # Display step and last action
        step_text = self.font.render(f"Step: {self.current_step}/{self.max_steps}", True, (0, 0, 0))
        self.screen.blit(step_text, (10, self.screen_height - 20))
        if hasattr(self, 'last_action'):
            action_text = self.font.render(f"Action: {self.actions[self.last_action]}", True, (0, 0, 0))
            self.screen.blit(action_text, (10, self.screen_height - 40))
            if self.last_action >= self.num_rooms and any(teen["room"] == self.mentor_room for teen in self.teens):
                action_visuals = {
                    6: lambda: pygame.draw.rect(self.screen, (0, 255, 0), (mentor_pos[0] - 20, mentor_pos[1] - 40, 40, 40)),
                    7: lambda: [pygame.draw.circle(self.screen, (255, 255, 0), (mentor_pos[0] + pos, mentor_pos[1] - 40), 10) for pos in [-20, 0, 20]],
                    8: lambda: pygame.draw.polygon(self.screen, (0, 0, 255), [(mentor_pos[0] - 20, mentor_pos[1] - 40), (mentor_pos[0] + 20, mentor_pos[1] - 40), (mentor_pos[0], mentor_pos[1] - 20)]),
                    9: lambda: pygame.draw.line(self.screen, (255, 0, 0), (mentor_pos[0] - 20, mentor_pos[1] - 40), (mentor_pos[0] + 20, mentor_pos[1] - 40), 5),
                    10: lambda: pygame.draw.rect(self.screen, (0, 255, 255), (mentor_pos[0] - 30, mentor_pos[1] - 50, 60, 60)),
                    11: lambda: [pygame.draw.circle(self.screen, (255, 0, 255), (mentor_pos[0] + pos, mentor_pos[1] - 40), 10) for pos in [-20, 0, 20]],
                    12: lambda: pygame.draw.rect(self.screen, (255, 255, 0), (mentor_pos[0] - 30, mentor_pos[1] - 50, 60, 40)),
                    13: lambda: pygame.draw.rect(self.screen, (128, 128, 128), (mentor_pos[0] - 20, mentor_pos[1] - 50, 40, 50))
                }
                action_visuals.get(self.last_action, lambda: None)()

        if self.render_mode == "rgb_array":
            return np.transpose(pygame.surfarray.array3d(self.screen), axes=(1, 0, 2))

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}")
        action = int(action)
        self.last_action = action
        reward = self._calculate_reward(action)
        self._update_state(action)
        self.current_step += 1
        terminated = (
            all(teen["state"] == 5 for teen in self.teens) or
            any(teen["trust"] <= 10 for teen in self.teens) or
            self.current_step >= self.max_steps
        )
        truncated = self.current_step >= self.max_steps
        observation = self._get_observation()
        info = {
            "mentor_room": self.mentor_room,
            "teens": [
                {
                    "state_name": self.states[teen["state"]],
                    "trust": teen["trust"],
                    "engagement": teen["engagement"],
                    "knowledge": teen["knowledge"],
                    "room": teen["room"]
                } for teen in self.teens
            ],
            "action_name": self.actions[action],
            "step": self.current_step
        }
        return observation, reward, terminated, truncated, info

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        self.current_step = 0
        self.mentor_room = 0
        self.teens = [
            {"state": 0, "trust": 50.0, "engagement": 50.0, "knowledge": 0.0, "room": 0},
            {"state": 0, "trust": 50.0, "engagement": 50.0, "knowledge": 0.0, "room": 0},
            {"state": 0, "trust": 50.0, "engagement": 50.0, "knowledge": 0.0, "room": 0}
        ]
        if hasattr(self, 'last_action'):
            del self.last_action
        if hasattr(self, 'movement_flash'):
            del self.movement_flash
        observation = self._get_observation()
        info = {
            "mentor_room": self.mentor_room,
            "teens": [
                {
                    "state_name": self.states[teen["state"]],
                    "trust": teen["trust"],
                    "engagement": teen["engagement"],
                    "knowledge": teen["knowledge"],
                    "room": teen["room"]
                } for teen in self.teens
            ],
            "step": self.current_step
        }
        return observation, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()
        elif self.render_mode == "human":
            self._render_frame()
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()

def demo_random_actions():
    env = TeenEducationEnvironment(render_mode="human")
    for episode in range(3):
        print(f"\n=== Episode {episode + 1} ===")
        observation, info = env.reset()
        total_reward = 0
        for step in range(50):
            action = env.action_space.sample()
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

def create_demo_gif():
    env = TeenEducationEnvironment(render_mode="rgb_array")
    frames = []
    observation, info = env.reset()
    for _ in range(30):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        frame = env.render()
        frames.append(frame)
        if terminated or truncated:
            observation, info = env.reset()
    env.close()
    imageio.mimsave("teen_education_demo.gif", frames, duration=2)
    print("Demo GIF saved as 'teen_education_demo.gif'")

if __name__ == "__main__":
    demo_random_actions()