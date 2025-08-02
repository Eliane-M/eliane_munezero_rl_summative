import gymnasium as gym
from gymnasium import spaces
import imageio
import numpy as np
import pygame
import random
from typing import Dict, Tuple, Optional, Any

class TeenEducationEnvironment(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 5}

    def __init__(self, render_mode: Optional[str] = None):
        super().__init__()

        self.max_steps = 100
        self.current_step = 0

        # Grid setup
        self.grid_size = 10
        self.cell_size = 60
        self.screen_width = self.grid_size * self.cell_size
        self.screen_height = self.grid_size * self.cell_size + 150  # Extra space for info

        # Define room areas in the grid (each room is a 2x2 area)
        self.room_areas = {
            0: [(1, 1), (1, 2), (2, 1), (2, 2)],  # Unaware (top-left)
            1: [(1, 7), (1, 8), (2, 7), (2, 8)],  # Confused (top-right)
            2: [(7, 7), (7, 8), (8, 7), (8, 8)],  # Embarrassed (bottom-right)
            3: [(7, 1), (7, 2), (8, 1), (8, 2)],  # Curious (bottom-left)
            4: [(4, 1), (4, 2), (5, 1), (5, 2)],  # Informed (center-left)
            5: [(4, 7), (4, 8), (5, 7), (5, 8)]   # Empowered (center-right)
        }

        # Room centers for navigation
        self.room_centers = {
            0: (1.5, 1.5),  # Unaware
            1: (1.5, 7.5),  # Confused  
            2: (7.5, 7.5),  # Embarrassed
            3: (7.5, 1.5),  # Curious
            4: (4.5, 1.5),  # Informed
            5: (4.5, 7.5)   # Empowered
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

        # Actions: 4 movement directions + 8 educational activities
        self.actions = {
            0: "Move_Up",
            1: "Move_Down", 
            2: "Move_Left",
            3: "Move_Right",
            4: "Distribute_Materials",
            5: "Lead_Group_Activity",
            6: "Demonstrate_Skills",
            7: "Role_Play_Scenario",
            8: "Visit_Health_Clinic",
            9: "Organize_Peer_Session",
            10: "Provide_Visual_Aid", 
            11: "Encourage_Journaling"
        }

        # Movement vectors
        self.move_vectors = {
            0: (-1, 0),  # Up
            1: (1, 0),   # Down
            2: (0, -1),  # Left  
            3: (0, 1)    # Right
        }

        self.num_teens = 3
        # Observation: [mentor_x, mentor_y, teen1_x, teen1_y, teen1_state, teen1_trust, teen1_engagement, teen1_knowledge, ...]
        self.observation_space = spaces.Box(
            low=np.array([0, 0] + [0, 0, 0, 0, 0, 0] * self.num_teens),
            high=np.array([1, 1] + [1, 1, 1, 1, 1, 1] * self.num_teens),
            dtype=np.float32
        )

        self.action_space = spaces.Discrete(len(self.actions))

        # Initialize positions
        self.mentor_pos = [1, 1]  # Start in Unaware room
        self.teens = [
            {
                "pos": [2, 2],  # Start in Unaware room
                "state": 0,
                "trust": 50.0,
                "engagement": 50.0, 
                "knowledge": 1.0
            }
            for _ in range(self.num_teens)
        ]

        # Initialize tracking variables
        self.last_pos = None
        self.last_room = None
        self.last_action = None

        # Rendering setup
        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        self.font = None
        self.small_font = None

        if self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            pygame.display.set_caption("Teen Education Grid Environment")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont("arial", 14, bold=True)
            self.small_font = pygame.font.SysFont("arial", 10)

    def _get_observation(self) -> np.ndarray:
        obs = [self.mentor_pos[0] / (self.grid_size-1), self.mentor_pos[1] / (self.grid_size-1)]
        for teen in self.teens:
            obs.extend([
                teen["pos"][0] / (self.grid_size-1),
                teen["pos"][1] / (self.grid_size-1),
                teen["state"] / 5.0,  # States range from 0 to 5
                teen["trust"] / 100.0,
                teen["engagement"] / 100.0,
                teen["knowledge"] / 100.0
            ])
        return np.array(obs, dtype=np.float32)

    def _get_room_from_pos(self, pos: Tuple[int, int]) -> int:
        """Get which room a position belongs to"""
        for room_id, cells in self.room_areas.items():
            if tuple(pos) in cells:
                return room_id
        return -1  # Not in any room

    def _is_valid_pos(self, pos: Tuple[int, int]) -> bool:
        """Check if position is valid (within grid bounds)"""
        return 0 <= pos[0] < self.grid_size and 0 <= pos[1] < self.grid_size

    def _calculate_reward(self, action: int) -> float:
        reward = 0.0

        mentor_room = self._get_room_from_pos(self.mentor_pos)
        
        # Small penalty for each step to encourage efficiency
        reward -= 0.1

        # Calculate which teens are in the same room as mentor
        teens_in_mentor_room = []
        teens_in_state_rooms = 0
        
        for teen in self.teens:
            teen_room = self._get_room_from_pos(teen["pos"])
            if teen_room == mentor_room and mentor_room != -1:
                teens_in_mentor_room.append(teen)
            if teen_room == teen["state"]:
                teens_in_state_rooms += 1

        # State progression rewards
        state_rewards = {0: -5, 1: -2, 2: 0, 3: 5, 4: 10, 5: 25}
        for teen in self.teens:
            reward += state_rewards[teen["state"]] / self.num_teens

        # Trust and engagement bonuses
        avg_trust = sum(teen["trust"] for teen in self.teens) / self.num_teens
        avg_engagement = sum(teen["engagement"] for teen in self.teens) / self.num_teens
        
        if avg_trust > 70:
            reward += 3
        elif avg_trust < 30:
            reward -= 5
            
        if avg_engagement > 70:
            reward += 3
        elif avg_engagement < 30:
            reward -= 3

        # Knowledge bonus
        avg_knowledge = sum(teen["knowledge"] for teen in self.teens) / self.num_teens
        reward += avg_knowledge * 0.1

        # Action-specific rewards
        if action < 4:  # Movement actions
            # Track movement rewards properly
            if self.last_pos is not None:
                if self.mentor_pos == self.last_pos:
                    reward -= 1.0  # Penalize staying in the same position
                else:
                    reward += 0.3  # Reward for moving
            
            # Bonus for moving towards teens who need help
            struggling_teens = [teen for teen in self.teens if teen["state"] <= 2]
            if struggling_teens:
                for teen in struggling_teens:
                    old_distance = abs(self.mentor_pos[0] - teen["pos"][0]) + abs(self.mentor_pos[1] - teen["pos"][1])
                    new_pos = [
                        self.mentor_pos[0] + self.move_vectors[action][0],
                        self.mentor_pos[1] + self.move_vectors[action][1]
                    ]
                    if self._is_valid_pos(new_pos):
                        new_distance = abs(new_pos[0] - teen["pos"][0]) + abs(new_pos[1] - teen["pos"][1])
                        if new_distance < old_distance:
                            reward += 0.5  # Bonus for moving towards struggling teens
        
        else:  # Educational actions
            if len(teens_in_mentor_room) == 0:
                reward -= 5  # Penalty for performing actions without teens present
            else:
                # Action effectiveness based on teen states and room context
                action_rewards = {
                    4: 5 if mentor_room >= 3 else -2,  # Distribute Materials (better in advanced rooms)
                    5: 8 if len(teens_in_mentor_room) >= 2 else 3,  # Group Activity (better with more teens)
                    6: 10 if mentor_room >= 4 else -1,  # Demonstrate Skills (advanced rooms)
                    7: 7 if any(teen["trust"] > 50 for teen in teens_in_mentor_room) else -1,  # Role Play
                    8: 12 if mentor_room >= 4 else -5,  # Health Clinic (advanced rooms only)
                    9: 6 if any(teen["engagement"] > 60 for teen in teens_in_mentor_room) else -2,  # Peer Session
                    10: 4 if any(teen["knowledge"] < 70 for teen in teens_in_mentor_room) else -1,  # Visual Aid
                    11: 3 if mentor_room >= 3 else 1  # Journaling (decent anywhere, better in advanced rooms)
                }
                reward += action_rewards.get(action, 0) * len(teens_in_mentor_room)

        # Proximity bonus - being in same room as teens
        reward += 2 * len(teens_in_mentor_room)

        # Room alignment bonus - teens being in their appropriate state rooms
        empowered_in_right_room = sum(1 for teen in self.teens 
                             if teen["state"] == 5 and 
                             self._get_room_from_pos(teen["pos"]) == 5)
        reward += 5 * empowered_in_right_room

        # Penalty for empowered teens not in empowered room
        empowered_wrong_room = sum(1 for teen in self.teens 
                                if teen["state"] == 5 and 
                                self._get_room_from_pos(teen["pos"]) != 5)
        reward -= 1 * empowered_wrong_room

        # Major success bonus
        empowered_teens = sum(1 for teen in self.teens if teen["state"] == 5)
        reward += 50 * empowered_teens

        # Completion efficiency bonus
        if empowered_teens == self.num_teens:
            efficiency_bonus = (self.max_steps - self.current_step) * 1.0
            reward += efficiency_bonus

        # Room exploration bonus
        current_room = self._get_room_from_pos(self.mentor_pos)
        if self.last_room is not None and current_room != -1 and current_room != self.last_room:
            reward += 1.0  # Bonus for entering a new room

        # Proximity rewards - closer to teens is better
        for teen in self.teens:
            distance = abs(self.mentor_pos[0] - teen["pos"][0]) + abs(self.mentor_pos[1] - teen["pos"][1])
            reward += max(0, 3 - distance) * 0.1  # Closer = better

        return reward

    def _update_state(self, action: int) -> None:
        # Handle mentor movement
        if action < 4:
            move_vector = self.move_vectors[action]
            new_pos = [
                self.mentor_pos[0] + move_vector[0], 
                self.mentor_pos[1] + move_vector[1]
            ]

            if self._is_valid_pos(new_pos):
                self.mentor_pos = new_pos

        # Update last position and room tracking
        self.last_pos = self.mentor_pos.copy()
        self.last_room = self._get_room_from_pos(self.mentor_pos)

        # Get current mentor room
        mentor_room = self._get_room_from_pos(self.mentor_pos)

        # Update teens
        for teen in self.teens:
            teen_room = self._get_room_from_pos(teen["pos"])
            in_same_room = (teen_room == mentor_room and mentor_room != -1)
            
            # Apply educational actions to teens in same room
            if action >= 4 and in_same_room:
                transitions = self._get_transition_probabilities(action, teen["state"])
                if transitions:
                    states_list = list(transitions.keys())
                    probabilities = list(transitions.values())
                    teen["state"] = np.random.choice(states_list, p=probabilities)

            # Update trust, engagement, knowledge
            trust_changes = {
                0: 0, 1: 0, 2: 0, 3: 0,  # Movement actions
                4: 2 if in_same_room else -1,
                5: 4 if in_same_room else -1, 
                6: 3 if in_same_room and teen["state"] >= 4 else -1,
                7: 5 if in_same_room else -1,
                8: 6 if in_same_room and teen["state"] >= 4 else -2,
                9: 4 if in_same_room else -1,
                10: 3 if in_same_room else -1,
                11: 2 if in_same_room else 0
            }

            trust_change = trust_changes.get(action, 0)
            
            # State-dependent multipliers
            if teen["state"] <= 1:
                trust_change *= 0.7
            elif teen["state"] >= 4:
                trust_change *= 1.2
                
            teen["trust"] = np.clip(teen["trust"] + trust_change + np.random.normal(0, 0.5), 0, 100)

            engagement_changes = {
                0: -0.2, 1: -0.2, 2: -0.2, 3: -0.2,  # Slight decay for movement
                4: 2 if in_same_room else -1,
                5: 8 if in_same_room else -1,
                6: 4 if in_same_room else -1,
                7: 6 if in_same_room else -1,
                8: 3 if in_same_room else -2,
                9: 7 if in_same_room else -1,
                10: 5 if in_same_room else -1,
                11: 4 if in_same_room else 0
            }
            engagement_change = engagement_changes.get(action, 0)
            teen["engagement"] = np.clip(teen["engagement"] + engagement_change + np.random.normal(0, 1), 0, 100)

            knowledge_gains = {
                0: 0, 1: 0, 2: 0, 3: 0,  # No knowledge gain from movement
                4: 4 if in_same_room else 0,
                5: 2 if in_same_room else 0,
                6: 6 if in_same_room else 0,
                7: 3 if in_same_room else 0,
                8: 8 if in_same_room else 0,
                9: 3 if in_same_room else 0,
                10: 5 if in_same_room else 0,
                11: 2 if in_same_room else 0
            }
            knowledge_gain = knowledge_gains.get(action, 0)
            effectiveness = (teen["trust"] + teen["engagement"]) / 200
            effective_gain = knowledge_gain * max(0.3, effectiveness)
            teen["knowledge"] = np.clip(teen["knowledge"] - 0.1 + effective_gain, 0, 100)

            # Teen movement - they try to move towards their state room or follow mentor
            self._move_teen(teen)

    def _move_teen(self, teen: Dict) -> None:
        """Handle teen movement logic"""
        current_room = self._get_room_from_pos(teen["pos"])
        target_room = teen["state"]
        mentor_room = self._get_room_from_pos(self.mentor_pos)
        
        # Decision making for teen movement
        move_to_state_room = current_room != target_room
        move_to_mentor = current_room != mentor_room and teen["trust"] > 40
        
        target_pos = None
        
        if move_to_state_room and random.random() < 0.4:
            # Move towards state room
            target_center = self.room_centers[target_room]
            target_pos = [int(target_center[0]), int(target_center[1])]
        elif move_to_mentor and random.random() < 0.3:
            # Move towards mentor
            target_pos = self.mentor_pos.copy()
        elif random.random() < 0.2:
            # Random movement
            directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
            direction = random.choice(directions)
            target_pos = [teen["pos"][0] + direction[0], teen["pos"][1] + direction[1]]

        if target_pos and self._is_valid_pos(target_pos):
            # Move one step towards target
            dx = target_pos[0] - teen["pos"][0]
            dy = target_pos[1] - teen["pos"][1]
            
            if dx != 0:
                teen["pos"][0] += 1 if dx > 0 else -1
            elif dy != 0:
                teen["pos"][1] += 1 if dy > 0 else -1

        # If teen is empowered, stay in empowered room
        if teen["state"] == 5:
            empowered_center = self.room_centers[5]
            target_pos = [int(empowered_center[0]), int(empowered_center[1])]
            if teen["pos"] != target_pos:
                # Move towards empowered room center
                dx = target_pos[0] - teen["pos"][0]
                dy = target_pos[1] - teen["pos"][1]
                if dx != 0:
                    teen["pos"][0] += 1 if dx > 0 else -1
                elif dy != 0:
                    teen["pos"][1] += 1 if dy > 0 else -1
            return
        
        # Check if teen is in same room as mentor
        teen_room = self._get_room_from_pos(teen["pos"])
        mentor_room = self._get_room_from_pos(self.mentor_pos)
        
        # Only move if with mentor (same room) and high enough trust
        if teen_room == mentor_room and mentor_room != -1 and teen["trust"] > 40:
            # Follow mentor with some probability
            if random.random() < 0.6:
                # Move towards mentor position
                dx = self.mentor_pos[0] - teen["pos"][0]
                dy = self.mentor_pos[1] - teen["pos"][1]
                
                if abs(dx) > 1 or abs(dy) > 1:  # Only move if not adjacent
                    if dx != 0:
                        new_pos = [teen["pos"][0] + (1 if dx > 0 else -1), teen["pos"][1]]
                    elif dy != 0:
                        new_pos = [teen["pos"][0], teen["pos"][1] + (1 if dy > 0 else -1)]
                    
                    if self._is_valid_pos(new_pos):
                        teen["pos"] = new_pos

    def _get_transition_probabilities(self, action: int, current_state: int) -> Dict[int, float]:
        """State transition probabilities based on action and current state"""
        base_transitions = {
            0: {  # Unaware
                4: {0: 0.2, 1: 0.7, 3: 0.1},
                5: {0: 0.3, 1: 0.5, 3: 0.2},
                6: {0: 0.4, 1: 0.4, 2: 0.2},
                7: {0: 0.3, 1: 0.4, 3: 0.3},
                8: {0: 0.5, 1: 0.4, 2: 0.1},
                9: {0: 0.4, 1: 0.4, 3: 0.2},
                10: {0: 0.2, 1: 0.6, 3: 0.2},
                11: {0: 0.4, 1: 0.4, 3: 0.2}
            },
            1: {  # Confused
                4: {1: 0.1, 3: 0.7, 4: 0.2},
                5: {1: 0.2, 3: 0.6, 4: 0.2},
                6: {1: 0.3, 2: 0.2, 4: 0.5},
                7: {1: 0.2, 3: 0.6, 4: 0.2},
                8: {1: 0.3, 2: 0.2, 4: 0.5},
                9: {1: 0.2, 3: 0.6, 4: 0.2},
                10: {1: 0.1, 3: 0.7, 4: 0.2},
                11: {1: 0.2, 3: 0.7, 4: 0.1}
            },
            2: {  # Embarrassed
                4: {2: 0.2, 3: 0.6, 4: 0.2},
                5: {2: 0.1, 3: 0.7, 4: 0.2},
                6: {2: 0.3, 3: 0.5, 4: 0.2},
                7: {2: 0.05, 3: 0.8, 4: 0.15},
                8: {2: 0.4, 3: 0.4, 4: 0.2},
                9: {2: 0.1, 3: 0.7, 4: 0.2},
                10: {2: 0.2, 3: 0.6, 4: 0.2},
                11: {2: 0.2, 3: 0.7, 4: 0.1}
            },
            3: {  # Curious
                4: {3: 0.1, 4: 0.7, 5: 0.2},
                5: {3: 0.1, 4: 0.7, 5: 0.2},
                6: {3: 0.2, 4: 0.6, 5: 0.2},
                7: {3: 0.1, 4: 0.7, 5: 0.2},
                8: {3: 0.1, 4: 0.6, 5: 0.3},
                9: {3: 0.1, 4: 0.7, 5: 0.2},
                10: {3: 0.05, 4: 0.8, 5: 0.15},
                11: {3: 0.1, 4: 0.7, 5: 0.2}
            },
            4: {  # Informed
                4: {4: 0.4, 5: 0.6},
                5: {4: 0.3, 5: 0.7},
                6: {4: 0.2, 5: 0.8},
                7: {4: 0.1, 5: 0.9},
                8: {4: 0.1, 5: 0.9},
                9: {4: 0.3, 5: 0.7},
                10: {4: 0.4, 5: 0.6},
                11: {4: 0.2, 5: 0.8}
            },
            5: {  # Empowered (terminal)
                4: {5: 1.0}, 5: {5: 1.0}, 6: {5: 1.0}, 7: {5: 1.0},
                8: {5: 1.0}, 9: {5: 1.0}, 10: {5: 1.0}, 11: {5: 1.0}
            }
        }
        return base_transitions.get(current_state, {action: {current_state: 1.0}}).get(action, {current_state: 1.0})

    def _render_frame(self):
        if self.screen is None:
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            self.font = pygame.font.SysFont("arial", 14, bold=True)
            self.small_font = pygame.font.SysFont("arial", 10)

        # Background
        self.screen.fill((250, 250, 250))

        # Define room colors
        room_colors = {
            0: (220, 220, 220),  # Unaware - Gray
            1: (255, 200, 200),  # Confused - Light Red  
            2: (255, 220, 220),  # Embarrassed - Pink
            3: (255, 255, 200),  # Curious - Yellow
            4: (200, 200, 255),  # Informed - Light Blue
            5: (200, 255, 200),  # Empowered - Light Green
            -1: (240, 240, 240)  # Empty space - Very Light Gray
        }

        # Draw grid and rooms
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                x = col * self.cell_size
                y = row * self.cell_size
                
                # Determine room for this cell
                room_id = self._get_room_from_pos([row, col])
                color = room_colors.get(room_id, room_colors[-1])
                
                # Draw cell
                rect = pygame.Rect(x, y, self.cell_size, self.cell_size)
                pygame.draw.rect(self.screen, color, rect)
                pygame.draw.rect(self.screen, (100, 100, 100), rect, 1)

        # Draw room labels at the center of each 2x2 room
        for room_id, cells in self.room_areas.items():
            # Calculate center of the 2x2 room
            center_row = sum(cell[0] for cell in cells) / 4
            center_col = sum(cell[1] for cell in cells) / 4
            center_x = center_col * self.cell_size + self.cell_size / 2
            center_y = center_row * self.cell_size + self.cell_size / 2
            
            # Render room name
            room_text = self.font.render(self.states[room_id], True, (0, 0, 0))
            text_rect = room_text.get_rect(center=(center_x, center_y))
            
            # Optional: Draw a semi-transparent background for better readability
            bg_rect = text_rect.inflate(10, 5)  # Add padding
            pygame.draw.rect(self.screen, (255, 255, 255, 180), bg_rect, border_radius=5)
            
            self.screen.blit(room_text, text_rect)

        # Draw mentor
        mentor_x = self.mentor_pos[1] * self.cell_size + self.cell_size // 2
        mentor_y = self.mentor_pos[0] * self.cell_size + self.cell_size // 2
        pygame.draw.circle(self.screen, (0, 100, 200), (mentor_x, mentor_y), 15)
        pygame.draw.circle(self.screen, (255, 255, 255), (mentor_x, mentor_y), 15, 2)
        
        # Mentor label
        mentor_text = self.font.render("M", True, (255, 255, 255))
        mentor_rect = mentor_text.get_rect(center=(mentor_x, mentor_y))
        self.screen.blit(mentor_text, mentor_rect)

        # Draw teens
        teen_colors = [(255, 100, 150), (150, 100, 255), (255, 150, 50)]
        teen_labels = ["T1", "T2", "T3"]
        
        for i, teen in enumerate(self.teens):
            teen_x = teen["pos"][1] * self.cell_size + self.cell_size // 2
            teen_y = teen["pos"][0] * self.cell_size + self.cell_size // 2
            
            # Offset teens slightly if they're in the same cell
            offset = (i - 1) * 8
            teen_x += offset
            teen_y += offset
            
            # Draw teen
            pygame.draw.circle(self.screen, teen_colors[i], (teen_x, teen_y), 12)
            pygame.draw.circle(self.screen, (255, 255, 255), (teen_x, teen_y), 12, 2)
            
            # Teen label
            teen_text = self.small_font.render(teen_labels[i], True, (255, 255, 255))
            teen_rect = teen_text.get_rect(center=(teen_x, teen_y))
            self.screen.blit(teen_text, teen_rect)

        # Draw information panel at bottom
        info_y = self.grid_size * self.cell_size + 10
        
        # Step and action info
        step_text = self.font.render(f"Step: {self.current_step}/{self.max_steps}", True, (0, 0, 0))
        self.screen.blit(step_text, (10, info_y))
        
        if self.last_action is not None:
            action_text = self.font.render(f"Action: {self.actions[self.last_action]}", True, (0, 0, 0))
            self.screen.blit(action_text, (200, info_y))

        # Teen information
        for i, teen in enumerate(self.teens):
            teen_info = f"T{i+1}: {self.states[teen['state']]} | Trust:{teen['trust']:.0f} Eng:{teen['engagement']:.0f} Know:{teen['knowledge']:.0f}"
            teen_text = self.small_font.render(teen_info, True, teen_colors[i])
            self.screen.blit(teen_text, (10, info_y + 25 + i * 15))

        # Success indicator
        empowered_count = sum(1 for teen in self.teens if teen["state"] == 5)
        success_text = self.font.render(f"Empowered: {empowered_count}/{self.num_teens}", 
                                      True, (0, 150, 0) if empowered_count == self.num_teens else (0, 0, 0))
        self.screen.blit(success_text, (10, info_y + 90))

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
        
        # Termination conditions
        terminated = (
            all(teen["state"] == 5 for teen in self.teens) or  # All empowered
            any(teen["trust"] <= 5 for teen in self.teens) or   # Trust too low
            self.current_step >= self.max_steps  # Time limit
        )
        
        truncated = self.current_step >= self.max_steps
        observation = self._get_observation()
        
        info = {
            "mentor_pos": self.mentor_pos.copy(),
            "mentor_room": self._get_room_from_pos(self.mentor_pos),
            "teens": [
                {
                    "pos": teen["pos"].copy(),
                    "room": self._get_room_from_pos(teen["pos"]),
                    "state_name": self.states[teen["state"]],
                    "trust": teen["trust"],
                    "engagement": teen["engagement"],
                    "knowledge": teen["knowledge"]
                } for teen in self.teens
            ],
            "action_name": self.actions[action],
            "step": self.current_step,
            "success": all(teen["state"] == 5 for teen in self.teens),
            "empowered_count": sum(1 for teen in self.teens if teen["state"] == 5)
        }
        
        return observation, reward, terminated, truncated, info

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        self.current_step = 0
        
        # Reset mentor to Unaware room
        self.mentor_pos = [1, 1]
        
        # Reset teens with slight position variations
        start_positions = [[1, 2], [2, 1], [2, 2]]
        for i, teen in enumerate(self.teens):
            teen.update({
                "pos": start_positions[i].copy(),
                "state": 0,
                "trust": 45.0 + random.uniform(-5, 10),
                "engagement": 45.0 + random.uniform(-5, 10),
                "knowledge": 0.0
            })
        
        # Reset tracking variables
        self.last_action = None
        self.last_pos = None
        self.last_room = None
            
        observation = self._get_observation()
        info = {
            "mentor_pos": self.mentor_pos.copy(),
            "mentor_room": self._get_room_from_pos(self.mentor_pos),
            "teens": [
                {
                    "pos": teen["pos"].copy(),
                    "room": self._get_room_from_pos(teen["pos"]),
                    "state_name": self.states[teen["state"]],
                    "trust": teen["trust"],
                    "engagement": teen["engagement"],
                    "knowledge": teen["knowledge"]
                } for teen in self.teens
            ],
            "step": self.current_step,
            "success": False,
            "empowered_count": 0
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

def demo_grid_movement():
    """Demo function showing step-by-step movement in the grid"""
    env = TeenEducationEnvironment(render_mode="human")
    
    print("=== Teen Education Grid Environment Demo ===")
    print("Mentor moves step by step through the grid.")
    print("Teens also move around based on their states and trust levels.")
    print("Press any key to continue between episodes...")
    
    for episode in range(3):
        print(f"\n=== Episode {episode + 1} ===")
        observation, info = env.reset()
        total_reward = 0
        
        # Scripted sequence to demonstrate movement
        action_sequence = [
            # First, move right to explore
            3, 3, 3,  # Move right
            1, 1,     # Move down
            5,        # Lead group activity
            2, 2,     # Move left
            1, 1,     # Move down  
            6,        # Demonstrate skills
            0, 0,     # Move up
            3, 3,     # Move right
            7,        # Role play
            # Then some educational actions
            4, 5, 6, 7, 8, 9, 10, 11,
            # Move to different rooms
            1, 1, 3, 3, 0, 0, 2, 2,
            # More activities
            4, 5, 6, 7
        ]
        
        for step in range(min(len(action_sequence), 40)):
            if step < len(action_sequence):
                action = action_sequence[step]
            else:
                action = env.action_space.sample()
                
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            print(f"Step {step + 1}: {info['action_name']}")
            print(f"  Mentor: Room {info['mentor_room']} at {info['mentor_pos']}")
            print(f"  Empowered: {info['empowered_count']}/{len(info['teens'])}")
            print(f"  Reward: {reward:.2f}")
            
            # Show teen positions every 5 steps
            if step % 5 == 0:
                for i, teen in enumerate(info['teens']):
                    print(f"    Teen {i+1}: {teen['state_name']} in Room {teen['room']} at {teen['pos']}")
            
            env.render()
            
            if terminated or truncated:
                success_msg = "🎉 SUCCESS! All teens empowered!" if info['success'] else "Episode ended"
                print(f"{success_msg} Total reward: {total_reward:.2f}")
                break
                
        if episode == 0:
            input("\nPress Enter to continue to next episode...")
    
    env.close()

def plot_teen_metrics():
    """Plot teen metrics over time"""
    env = TeenEducationEnvironment(render_mode=None)
    trust_data, engagement_data, knowledge_data = [], [], []
    observation, info = env.reset()
    
    for action in [3, 3, 3, 1, 1, 5, 2, 2, 1, 1, 6, 0, 0, 3, 3, 7]:  # Subset of demo actions
        observation, reward, terminated, truncated, info = env.step(action)
        avg_trust = sum(teen["trust"] for teen in env.teens) / env.num_teens
        avg_engagement = sum(teen["engagement"] for teen in env.teens) / env.num_teens
        avg_knowledge = sum(teen["knowledge"] for teen in env.teens) / env.num_teens
        trust_data.append(avg_trust)
        engagement_data.append(avg_engagement)
        knowledge_data.append(avg_knowledge)
        if terminated or truncated:
            break
    
    env.close()
    
    return {
        "type": "line",
        "data": {
            "labels": list(range(1, len(trust_data) + 1)),
            "datasets": [
                {
                    "label": "Average Trust",
                    "data": trust_data,
                    "borderColor": "rgba(75, 192, 192, 1)",
                    "fill": False
                },
                {
                    "label": "Average Engagement",
                    "data": engagement_data,
                    "borderColor": "rgba(255, 99, 132, 1)",
                    "fill": False
                },
                {
                    "label": "Average Knowledge",
                    "data": knowledge_data,
                    "borderColor": "rgba(54, 162, 235, 1)",
                    "fill": False
                }
            ]
        },
        "options": {
            "scales": {
                "y": {
                    "beginAtZero": True,
                    "max": 100
                }
            },
            "plugins": {
                "title": {
                    "display": True,
                    "text": "Teen Metrics Over Steps"
                }
            }
        }
    }

def create_grid_demo_gif():
    """Create a demo GIF showing grid-based movement"""
    try:
        env = TeenEducationEnvironment(render_mode="rgb_array")
        frames = []
        observation, info = env.reset()
        
        print("Creating demo GIF with grid movement...")
        
        # Scripted sequence for interesting movement patterns
        action_sequence = [
            # Tour of all rooms
            3, 3, 3, 3, 3, 3,  # Move right across top
            1, 1, 1, 1, 1, 1,  # Move down right side
            2, 2, 2, 2, 2, 2,  # Move left across bottom  
            0, 0, 0, 0, 0, 0,  # Move up left side
            # Educational activities in different rooms
            5, 5,  # Group activity
            3, 3, 1, 1,  # Move to center
            6, 6,  # Demonstrate skills
            3, 3, 3,  # Move right
            7, 7,  # Role play
            1, 1, 1,  # Move down
            8, 8,  # Health clinic
            2, 2, 2,  # Move left
            9, 9,  # Peer session
        ]
        
        for i, action in enumerate(action_sequence):
            observation, reward, terminated, truncated, info = env.step(action)
            frame = env.render()
            if frame is not None:
                frames.append(frame)
                
                # Add extra frames to slow down the GIF
                if i % 2 == 0:
                    frames.append(frame)
            
            if terminated or truncated:
                if info['success']:
                    # Add celebration frames
                    for _ in range(10):
                        if frame is not None:
                            frames.append(frame)
                    break
                else:
                    # Reset and continue
                    observation, info = env.reset()
                    frame = env.render()
                    if frame is not None:
                        frames.append(frame)
        
        env.close()
        
        if frames:
            imageio.mimsave("teen_education_grid_demo.gif", frames, duration=0.4)
            print("Grid movement demo GIF saved as 'teen_education_grid_demo.gif'")
        else:
            print("No frames captured for GIF")
            
    except Exception as e:
        print(f"Error creating GIF: {e}")
        print("Make sure imageio is installed: pip install imageio")

def test_grid_environment():
    """Test the grid-based environment"""
    print("Testing Teen Education Grid Environment...")
    
    try:
        env = TeenEducationEnvironment(render_mode=None)
        
        # Test basic functionality
        obs, info = env.reset()
        print(f"✓ Environment reset successfully")
        print(f"✓ Observation shape: {obs.shape}")
        print(f"✓ Action space: {env.action_space}")
        print(f"✓ Observation space: {env.observation_space}")
        
        # Test movement actions
        print(f"✓ Initial mentor position: {info['mentor_pos']}")
        
        # Test moving right
        obs, reward, terminated, truncated, info = env.step(3)  # Move right
        print(f"✓ After moving right: {info['mentor_pos']}")
        
        # Test moving down
        obs, reward, terminated, truncated, info = env.step(1)  # Move down
        print(f"✓ After moving down: {info['mentor_pos']}")
        
        # Test educational action
        obs, reward, terminated, truncated, info = env.step(5)  # Lead group activity
        print(f"✓ Educational action performed: {info['action_name']}")
        
        # Test multiple steps
        total_reward = 0
        for i in range(20):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            if terminated:
                print(f"✓ Episode terminated after {i+4} steps")
                break
        
        print(f"✓ Total reward after testing: {total_reward:.2f}")
        print(f"✓ Final empowered count: {info['empowered_count']}")
        
        # Test room detection
        mentor_room = env._get_room_from_pos(info['mentor_pos'])
        print(f"✓ Mentor in room: {mentor_room} ({env.states.get(mentor_room, 'None')})")
        
        env.close()
        print("✅ All tests passed! Grid environment is working correctly.")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

def visualize_room_layout():
    """Print a visual representation of the room layout"""
    print("=== Room Layout ===")
    print("Grid size: 10x10")
    print("Room areas (each room is 2x2):")
    
    room_layout = [
        [".", ".", ".", ".", ".", ".", ".", ".", ".", "."],
        [".", "0", "0", ".", ".", ".", ".", "1", "1", "."],
        [".", "0", "0", ".", ".", ".", ".", "1", "1", "."],
        [".", ".", ".", ".", ".", ".", ".", ".", ".", "."],
        [".", "4", "4", ".", ".", ".", ".", "5", "5", "."],
        [".", "4", "4", ".", ".", ".", ".", "5", "5", "."],
        [".", ".", ".", ".", ".", ".", ".", ".", ".", "."],
        [".", "3", "3", ".", ".", ".", ".", "2", "2", "."],
        [".", "3", "3", ".", ".", ".", ".", "2", "2", "."],
        [".", ".", ".", ".", ".", ".", ".", ".", ".", "."]
    ]
    
    room_names = {
        "0": "Unaware", "1": "Confused", "2": "Embarrassed",
        "3": "Curious", "4": "Informed", "5": "Empowered"
    }
    
    print("\nGrid layout:")
    for i, row in enumerate(room_layout):
        print(f"{i}: {' '.join(row)}")
    
    print("\nRoom legend:")
    for num, name in room_names.items():
        print(f"{num}: {name}")
    print(".: Empty space")

if __name__ == "__main__":
    # Show room layout
    visualize_room_layout()
    
    # Test basic functionality
    print("\n" + "="*50)
    test_grid_environment()
    
    # Uncomment for visual demo (requires display)
    print("\n" + "="*50)
    print("Visual demo")
    demo_grid_movement()
    
    # Uncomment to create GIF (requires imageio)
    print("\n" + "="*50)
    print("Create a demo GIF showing grid movement:")
    create_grid_demo_gif()
    
    print("\n" + "="*50)
    print("READY FOR RL TRAINING!")