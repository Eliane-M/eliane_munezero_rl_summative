import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import random
import math
from typing import Dict, Tuple, Optional, Any, List
from PIL import Image
import os

class TeenEducationEnvironment(gym.Env):
    """
    Spatial environment where an AI mentor agent moves around a school/community center
    to interact with teenage girls in different emotional states, providing education
    to prevent teenage pregnancies and guide them to the empowered room.
    
    The agent must physically navigate to girls, assess their state, choose
    appropriate educational interventions while building trust and guiding them
    to the empowered room when they're ready.
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 8}
    
    def __init__(self, render_mode: Optional[str] = None, record_gif: bool = False, gif_path: str = "environment_recording.gif"):
        super().__init__()
        
        # GIF recording setup
        self.record_gif = record_gif
        self.gif_path = gif_path
        self.gif_frames = []
        self.gif_frame_skip = 1  # Record every frame (set to 2 to record every other frame, etc.)
        self.gif_frame_counter = 0
        
        # Environment dimensions
        self.width = 800
        self.height = 600
        self.grid_size = 40  # Size of each grid cell
        self.grid_width = self.width // self.grid_size  # 20 cells
        self.grid_height = self.height // self.grid_size  # 15 cells
        
        # Environment parameters
        self.max_steps = 300
        self.current_step = 0
        self.num_girls = 5  # Number of teenage girls in the environment
        
        # Agent properties
        self.agent_pos = [0, 0]  # [x, y] grid coordinates
        self.agent_speed = 1  # Moves 1 grid cell per action
        self.interaction_range = 1  # Can interact within 1 grid cell distance
        
        # Girl states and properties
        self.girl_states = {
            0: "Unaware",      # Red - needs basic education
            1: "Confused",     # Orange - needs clarification  
            2: "Embarrassed",  # Pink - needs reassurance
            3: "Curious",      # Yellow - ready for information
            4: "Informed",     # Light Blue - needs reinforcement
            5: "Empowered"     # Green - success state
        }
        
        # Educational actions (when near a girl)
        self.educational_actions = {
            0: "Provide_Basic_Info",
            1: "Share_Personal_Story", 
            2: "Ask_Open_Question",
            3: "Offer_Reassurance",
            4: "Give_Practical_Advice",
            5: "Address_Myths"
        }
        
        # Movement + educational actions + guidance action
        # Actions 0-7: Movement (8-directional + stay)
        # Actions 8-13: Educational interventions (when near girl)
        # Action 14: Guide to empowered room (when girl is ready)
        self.action_meanings = {
            0: "Move_Up",
            1: "Move_Down", 
            2: "Move_Left",
            3: "Move_Right",
            4: "Move_Up_Left",
            5: "Move_Up_Right",
            6: "Move_Down_Left", 
            7: "Move_Down_Right",
            8: "Stay_Still",
            9: "Provide_Basic_Info",
            10: "Share_Personal_Story",
            11: "Ask_Open_Question", 
            12: "Offer_Reassurance",
            13: "Give_Practical_Advice",
            14: "Address_Myths",
            15: "Guide_To_Empowered_Room"
        }
        
        # Action space: 16 total actions (8 movement + 1 stay + 6 educational + 1 guidance)
        self.action_space = spaces.Discrete(16)
        
        # Single empowered room in the environment (define before initializing girls)
        self.empowered_room = {
            "name": "empowered_room",
            "bounds": [(15, 2), (19, 6)]  # top-left, bottom-right coordinates
        }
        
        # Observation space: [agent_x, agent_y, girl1_x, girl1_y, girl1_state, girl1_trust, girl1_in_room,
        #                     girl2_x, girl2_y, girl2_state, girl2_trust, girl2_in_room, ...]
        # Plus global metrics: [girls_in_room, average_trust, steps_remaining]
        obs_size = 2 + (self.num_girls * 5) + 3  # 2 + 25 + 3 = 30
        self.observation_space = spaces.Box(
            low=0,
            high=max(self.grid_width, self.grid_height, 5, 100, self.max_steps),
            shape=(obs_size,),
            dtype=np.float32
        )
        
        # Initialize girls (after empowered_room is defined)
        self.girls = []
        self.reset_girls()
        
        # Rendering
        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        
        if self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption("Spatial Teen Education Environment - Guide to Empowerment")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont("arial", 12)
    
    def start_gif_recording(self, gif_path: str = None, frame_skip: int = 1):
        """Start recording GIF frames."""
        if gif_path:
            self.gif_path = gif_path
        self.record_gif = True
        self.gif_frames = []
        self.gif_frame_skip = frame_skip
        self.gif_frame_counter = 0
        print(f"Started GIF recording. Will save to: {self.gif_path}")
    
    def stop_gif_recording(self):
        """Stop recording and save GIF."""
        if not self.record_gif or not self.gif_frames:
            print("No GIF frames to save.")
            return
        
        self.record_gif = False
        print(f"Saving GIF with {len(self.gif_frames)} frames to {self.gif_path}...")
        
        try:
            # Convert frames to PIL Images
            pil_frames = []
            for frame in self.gif_frames:
                # Convert numpy array to PIL Image
                pil_image = Image.fromarray(frame)
                pil_frames.append(pil_image)
            
            # Save as GIF
            if pil_frames:
                # Calculate duration per frame (in milliseconds)
                duration = int(1000 / self.metadata["render_fps"])
                
                pil_frames[0].save(
                    self.gif_path,
                    save_all=True,
                    append_images=pil_frames[1:],
                    duration=duration,
                    loop=0,  # 0 means infinite loop
                    optimize=True
                )
                print(f"GIF saved successfully to {self.gif_path}")
            else:
                print("No frames to save.")
                
        except Exception as e:
            print(f"Error saving GIF: {e}")
        
        # Clear frames to free memory
        self.gif_frames = []
    
    def _capture_frame_for_gif(self):
        """Capture current frame for GIF if recording is enabled."""
        if not self.record_gif:
            return
        
        self.gif_frame_counter += 1
        if self.gif_frame_counter % self.gif_frame_skip == 0:
            # Always get RGB array for GIF, regardless of render mode
            if self.screen is not None:
                # Capture the current screen as RGB array
                frame = np.transpose(pygame.surfarray.array3d(self.screen), axes=(1, 0, 2))
                self.gif_frames.append(frame)
    
    def reset_girls(self):
        """Initialize or reset girl positions and states."""
        self.girls = []
        for i in range(self.num_girls):
            # Start girls at random positions, but not in the empowered room
            pos = self._get_random_position_outside_room()
            
            girl = {
                'id': i,
                'pos': pos,
                'state': random.randint(0, 2),  # Start in lower states
                'trust': 50.0,
                'engagement': 50.0,
                'knowledge': 0.0,
                'last_interaction_step': -10,  # When last interacted with
                'interactions_count': 0,
                'in_empowered_room': False,
                'ready_for_room': False  # True when state >= 4 and trust >= 70
            }
            self.girls.append(girl)
    
    def _get_random_position_outside_room(self) -> List[int]:
        """Get a random position that's not inside the empowered room."""
        room_bounds = self.empowered_room["bounds"]
        
        while True:
            x = random.randint(1, self.grid_width - 2)
            y = random.randint(1, self.grid_height - 2)
            
            # Check if position is outside empowered room
            if not (room_bounds[0][0] <= x <= room_bounds[1][0] and 
                    room_bounds[0][1] <= y <= room_bounds[1][1]):
                return [x, y]
    
    def _is_in_empowered_room(self, pos: List[int]) -> bool:
        """Check if a position is inside the empowered room."""
        room_bounds = self.empowered_room["bounds"]
        return (room_bounds[0][0] <= pos[0] <= room_bounds[1][0] and 
                room_bounds[0][1] <= pos[1] <= room_bounds[1][1])
    
    def _update_room_status(self):
        """Update which girls are in the empowered room."""
        for girl in self.girls:
            girl['in_empowered_room'] = self._is_in_empowered_room(girl['pos'])
            
            # Check if girl is ready for the empowered room
            girl['ready_for_room'] = (girl['state'] >= 4 and girl['trust'] >= 65)
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation state."""
        obs = []
        
        # Agent position
        obs.extend([self.agent_pos[0], self.agent_pos[1]])
        
        # Girl information (position, state, trust, in_room status for each girl)
        for girl in self.girls:
            obs.extend([
                girl['pos'][0], 
                girl['pos'][1],
                girl['state'],
                girl['trust'],
                1.0 if girl['in_empowered_room'] else 0.0
            ])
        
        # Global metrics
        girls_in_room = sum(1 for girl in self.girls if girl['in_empowered_room'])
        avg_trust = np.mean([girl['trust'] for girl in self.girls])
        steps_remaining = self.max_steps - self.current_step
        
        obs.extend([girls_in_room, avg_trust, steps_remaining])
        
        return np.array(obs, dtype=np.float32)
    
    def _get_nearby_girls(self) -> List[Dict]:
        """Get girls within interaction range of agent."""
        nearby = []
        for girl in self.girls:
            distance = abs(self.agent_pos[0] - girl['pos'][0]) + abs(self.agent_pos[1] - girl['pos'][1])
            if distance <= self.interaction_range:
                nearby.append(girl)
        return nearby
    
    def _is_valid_position(self, x: int, y: int) -> bool:
        """Check if position is within bounds."""
        return 0 <= x < self.grid_width and 0 <= y < self.grid_height
    
    def _move_agent(self, action: int) -> bool:
        """Move agent based on action. Returns True if move was successful."""
        if hasattr(action, 'item'):
            action = action.item()
        elif isinstance(action, np.ndarray):
            action = int(action)

        new_pos = self.agent_pos.copy()
        
        # Movement mapping
        moves = {
            0: [0, -1],   # Move_Up
            1: [0, 1],    # Move_Down
            2: [-1, 0],   # Move_Left  
            3: [1, 0],    # Move_Right
            4: [-1, -1],  # Move_Up_Left
            5: [1, -1],   # Move_Up_Right
            6: [-1, 1],   # Move_Down_Left
            7: [1, 1],    # Move_Down_Right
            8: [0, 0]     # Stay_Still
        }
        
        if action in moves:
            dx, dy = moves[action]
            new_pos[0] += dx
            new_pos[1] += dy
            
            if self._is_valid_position(new_pos[0], new_pos[1]):
                self.agent_pos = new_pos
                return True
        
        return False
    
    def _educational_interaction(self, action: int, girl: Dict) -> float:
        """Perform educational interaction with a girl. Returns reward."""
        if action < 9 or action > 14:  # Not an educational action
            return 0.0
        
        educational_action = action - 9  # Convert to 0-5 range
        
        # Update interaction tracking
        girl['last_interaction_step'] = self.current_step
        girl['interactions_count'] += 1
        
        # Calculate effectiveness based on current state and action appropriateness
        effectiveness = self._get_action_effectiveness(educational_action, girl)
        
        # Update girl's state based on effectiveness
        previous_state = girl['state']
        self._update_girl_state(girl, educational_action, effectiveness)
        
        # Update trust and engagement
        self._update_girl_metrics(girl, educational_action, effectiveness)
        
        # Calculate reward
        reward = self._calculate_interaction_reward(girl, previous_state, effectiveness)
        
        return reward
    
    def _guide_to_empowered_room(self, girl: Dict) -> float:
        """Guide a girl toward the empowered room. Returns reward."""
        reward = 0.0
        
        # Check if girl is ready to be guided
        if not girl['ready_for_room']:
            return -5.0  # Penalty for trying to guide unprepared girl
        
        # If girl is already in room, give small reward
        if girl['in_empowered_room']:
            return 2.0
        
        # Move girl one step closer to empowered room
        room_center = [
            (self.empowered_room["bounds"][0][0] + self.empowered_room["bounds"][1][0]) // 2,
            (self.empowered_room["bounds"][0][1] + self.empowered_room["bounds"][1][1]) // 2
        ]
        
        # Calculate direction to room
        dx = 0 if girl['pos'][0] == room_center[0] else (1 if room_center[0] > girl['pos'][0] else -1)
        dy = 0 if girl['pos'][1] == room_center[1] else (1 if room_center[1] > girl['pos'][1] else -1)
        
        # Move girl (with some probability of success based on trust)
        success_prob = min(0.9, girl['trust'] / 100.0 + 0.3)
        if random.random() < success_prob:
            new_x = np.clip(girl['pos'][0] + dx, 0, self.grid_width - 1)
            new_y = np.clip(girl['pos'][1] + dy, 0, self.grid_height - 1)
            girl['pos'] = [new_x, new_y]
            reward += 10.0  # Reward for successful guidance
            
            # Bonus if girl enters the room
            if self._is_in_empowered_room(girl['pos']):
                reward += 25.0
        else:
            reward -= 2.0  # Small penalty for failed guidance attempt
        
        return reward
    
    def _get_action_effectiveness(self, action: int, girl: Dict) -> float:
        """Calculate how effective an educational action is for a specific girl."""
        state = girl['state']
        trust = girl['trust']
        
        # Base effectiveness matrix (state x action)
        effectiveness_matrix = {
            0: [0.7, 0.4, 0.3, 0.2, 0.3, 0.5],  # Unaware: basic info most effective
            1: [0.8, 0.6, 0.7, 0.4, 0.6, 0.9],  # Confused: myth-busting very effective
            2: [0.4, 0.8, 0.5, 0.9, 0.3, 0.2],  # Embarrassed: stories and reassurance work
            3: [0.6, 0.7, 0.9, 0.5, 0.8, 0.7],  # Curious: questions and advice effective
            4: [0.5, 0.6, 0.7, 0.4, 0.9, 0.6],  # Informed: practical advice best
            5: [0.3, 0.4, 0.5, 0.3, 0.7, 0.3]   # Empowered: mainly reinforcement
        }
        
        base_effectiveness = effectiveness_matrix.get(state, [0.5] * 6)[action]
        
        # Adjust based on trust level
        trust_multiplier = 0.5 + (trust / 100.0)  # 0.5 to 1.5 range
        
        # Add some randomness for realism
        random_factor = random.uniform(0.8, 1.2)
        
        return base_effectiveness * trust_multiplier * random_factor
    
    def _update_girl_metrics(self, girl: Dict, action: int, effectiveness: float):
        """Update girl's trust, engagement, and knowledge based on interaction."""
        previous_trust = girl['trust']
        
        # Trust changes
        trust_change = (effectiveness - 0.5) * 10  # -5 to +5 range
        trust_change += random.uniform(-2, 2)  # Add noise
        girl['trust'] = np.clip(girl['trust'] + trust_change, 0, 100)
        
        # If trust increased significantly, move toward room
        if girl['trust'] > previous_trust + 5:
            self._move_girl_toward_empowered_room(girl, effectiveness)
        
        # Engagement changes  
        engagement_change = (effectiveness - 0.4) * 8
        engagement_change += random.uniform(-3, 3)
        girl['engagement'] = np.clip(girl['engagement'] + engagement_change, 0, 100)
        
        # Knowledge accumulation
        knowledge_gain = effectiveness * 5
        girl['knowledge'] = np.clip(girl['knowledge'] + knowledge_gain, 0, 100)

    def _update_girl_state(self, girl: Dict, action: int, effectiveness: float):
        """Update girl's emotional/knowledge state based on interaction."""
        previous_state = girl['state']
        
        if effectiveness > 0.7 and random.random() < 0.4:  # High effectiveness, chance to advance
            if girl['state'] < 5:
                girl['state'] += 2
        elif effectiveness < 0.3 and random.random() < 0.2:  # Low effectiveness, small chance to regress
            if girl['state'] > 0:
                girl['state'] -= 1
        
        # If girl's state improved, move her closer to empowered room
        if girl['state'] > previous_state:
            self._move_girl_toward_empowered_room(girl, effectiveness)
        # If girl's state regressed, move her slightly away from empowered room
        elif girl['state'] < previous_state:
            self._move_girl_away_from_empowered_room(girl, effectiveness)
    
    def _move_girl_toward_empowered_room(self, girl: Dict, effectiveness: float = 1.0):
        """Move a girl 1-2 steps closer to the empowered room when she advances in state."""
        room_center = [
            (self.empowered_room["bounds"][0][0] + self.empowered_room["bounds"][1][0]) // 2,
            (self.empowered_room["bounds"][0][1] + self.empowered_room["bounds"][1][1]) // 2
        ]
        
        # Calculate direction to room center
        dx = 0 if girl['pos'][0] == room_center[0] else (1 if room_center[0] > girl['pos'][0] else -1)
        dy = 0 if girl['pos'][1] == room_center[1] else (1 if room_center[1] > girl['pos'][1] else -1)
        
        # Move 1-2 steps closer (more steps for higher states)
        steps = min(2, girl['state'])  # State 0-1: 1 step, State 2+: 2 steps
        
        for _ in range(steps):
            if dx != 0 or dy != 0:  # Still has distance to cover
                new_x = np.clip(girl['pos'][0] + dx, 0, self.grid_width - 1)
                new_y = np.clip(girl['pos'][1] + dy, 0, self.grid_height - 1)
                girl['pos'] = [new_x, new_y]
                
                # Recalculate direction for next step
                dx = 0 if girl['pos'][0] == room_center[0] else (1 if room_center[0] > girl['pos'][0] else -1)
                dy = 0 if girl['pos'][1] == room_center[1] else (1 if room_center[1] > girl['pos'][1] else -1)
    
    def _move_girl_away_from_empowered_room(self, girl: Dict, effectiveness: float):
        """Move a girl 1 step away from the empowered room when she regresses in state."""
        room_center = [
            (self.empowered_room["bounds"][0][0] + self.empowered_room["bounds"][1][0]) // 2,
            (self.empowered_room["bounds"][0][1] + self.empowered_room["bounds"][1][1]) // 2
        ]
        
        # Calculate direction away from room center
        dx = 0 if girl['pos'][0] == room_center[0] else (-1 if room_center[0] > girl['pos'][0] else 1)
        dy = 0 if girl['pos'][1] == room_center[1] else (-1 if room_center[1] > girl['pos'][1] else 1)
        
        # If already at room center, move in a random direction
        if dx == 0 and dy == 0:
            dx = random.choice([-1, 0, 1])
            dy = random.choice([-1, 0, 1])
        
        # Move 1 step away
        new_x = np.clip(girl['pos'][0] + dx, 0, self.grid_width - 1)
        new_y = np.clip(girl['pos'][1] + dy, 0, self.grid_height - 1)
        girl['pos'] = [new_x, new_y]
    
    def _calculate_interaction_reward(self, girl: Dict, previous_state: int, effectiveness: float) -> float:
        """Calculate reward for an educational interaction."""
        reward = 0.0
        
        # INCREASED state progression rewards
        state_rewards = {0: 0, 1: 5, 2: 10, 3: 15, 4: 25, 5: 40}  # Much higher rewards
        reward += state_rewards.get(girl['state'], 0)
        
        # BIGGER state transition bonuses
        if girl['state'] > previous_state:
            reward += 30 * (girl['state'] - previous_state)  # Doubled from 15
        elif girl['state'] < previous_state:
            reward -= 15 * (previous_state - girl['state'])  # Increased penalty
        
        # Effectiveness bonus (unchanged)
        reward += effectiveness * 5
        
        # Trust and engagement bonuses (INCREASED)
        if girl['trust'] > 70:
            reward += 8  # Increased from 3
        elif girl['trust'] < 30:
            reward -= 3  # Reduced penalty from 8
            
        # MEGA bonus for reaching empowered state
        if girl['state'] == 5:
            reward += 60  # Doubled from 30
            
        return reward
    
    def _move_girls_randomly(self):
        """Occasionally move girls, but prefer movement toward empowered room for higher states."""
        if random.random() < 0.03:  # 3% chance per step
            girl = random.choice(self.girls)
            
            # Don't move girls that are in the empowered room unless they're not ready
            if girl['in_empowered_room'] and girl['ready_for_room']:
                return
            
            # Higher state girls have higher chance to move toward empowered room
            toward_room_probability = girl['state'] * 0.15  # 0% for state 0, up to 75% for state 5
            
            if random.random() < toward_room_probability:
                # Move toward empowered room
                self._move_girl_toward_empowered_room(girl)
            else:
                # Random movement (but constrained)
                new_x = np.clip(girl['pos'][0] + random.randint(-1, 1), 0, self.grid_width - 1)
                new_y = np.clip(girl['pos'][1] + random.randint(-1, 1), 0, self.grid_height - 1)
                
                # Don't let unprepared girls randomly enter the empowered room
                if not girl['ready_for_room'] and self._is_in_empowered_room([new_x, new_y]):
                    return
                
                girl['pos'] = [new_x, new_y]
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        
        self.current_step = 0
        self.agent_pos = [0, 0]  # Start in top-left corner
        self.reset_girls()
        self._update_room_status()
        
        observation = self._get_observation()
        info = {
            "agent_pos": self.agent_pos,
            "girls_states": [girl['state'] for girl in self.girls],
            "girls_in_room": 0,
            "girls_ready_for_room": sum(1 for girl in self.girls if girl['ready_for_room'])
        }
        
        return observation, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one step in the environment."""
        if hasattr(action, 'item'):
            action = action.item()
        elif isinstance(action, np.ndarray):
            action = int(action)

        
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}")
        
        reward = 0.0
        self.current_step += 1
        
        if action <= 8:  # Movement action
            moved = self._move_agent(action)
            if moved:
                # REDUCE movement reward - make interaction more attractive
                reward += 0.5 
            else:
                reward -= 1.0  # Penalty for trying to move out of bounds
                
            # PENALTY for moving when there are nearby girls who need help
            nearby_girls = self._get_nearby_girls()
            if nearby_girls:
                # Check if any nearby girl needs help
                needs_help = any(not girl['in_empowered_room'] for girl in nearby_girls)
                if needs_help and action != 8:  # Moving away when help is needed (except staying still)
                    reward -= 0.5  # Penalty for abandoning girls who need help
        
        elif action <= 14:  # Educational action (9-14)
            nearby_girls = self._get_nearby_girls()
            if nearby_girls:
                # Interact with the first nearby girl (could be extended to choose)
                girl = nearby_girls[0]
                interaction_reward = self._educational_interaction(action, girl)
                reward += interaction_reward
                
                # BONUS for taking educational action when near girls
                reward += 1.0  # Always reward for trying to educate
                
            else:
                reward -= 1.0  # Reduced penalty (was -2.0)
        
        elif action == 15:  # Guide to empowered room action
            nearby_girls = self._get_nearby_girls()
            if nearby_girls:
                girl = nearby_girls[0]
                guidance_reward = self._guide_to_empowered_room(girl)
                reward += guidance_reward
                
                # BONUS for taking guidance action
                reward += 0.5  # Small bonus for attempting guidance
                
            else:
                reward -= 1.0  # Reduced penalty (was -2.0)
        
        # Move girls occasionally
        self._move_girls_randomly()
        
        # Update room status
        self._update_room_status()
        
        # Calculate termination conditions
        girls_in_room = sum(1 for girl in self.girls if girl['in_empowered_room'])
        avg_trust = np.mean([girl['trust'] for girl in self.girls])
        
        terminated = (
            girls_in_room >= self.num_girls or  # All girls in empowered room
            avg_trust <= 20 or  # Lost trust globally
            self.current_step >= self.max_steps
        )
        
        truncated = self.current_step >= self.max_steps
        
        # Reward for having girls in the empowered room (HIGHEST REWARDS!)
        girls_in_room_reward = girls_in_room * 5.0  # 5 points per girl in room
        reward += girls_in_room_reward
        
        # Exponential bonus for getting more girls in the room
        if girls_in_room >= 2:
            reward += girls_in_room ** 2 * 2  # Exponential scaling: 8, 18, 32, 50 for 2,3,4,5 girls
        
        # Mega bonus for getting ALL girls in the room
        if girls_in_room == self.num_girls:
            reward += 100  # Huge bonus for complete success
        
        # Distance-based reward (encourage moving toward girls who need help)
        min_distance_to_needy = float('inf')
        for girl in self.girls:
            if not girl['in_empowered_room']:  # Not yet in room
                distance = abs(self.agent_pos[0] - girl['pos'][0]) + abs(self.agent_pos[1] - girl['pos'][1])
                min_distance_to_needy = min(min_distance_to_needy, distance)
        
        if min_distance_to_needy != float('inf'):
            reward += max(0, 5 - min_distance_to_needy) * 0.1  # Reward for being close to girls who need help
        
        observation = self._get_observation()
        info = {
            "agent_pos": self.agent_pos,
            "girls_states": [girl['state'] for girl in self.girls],
            "girls_in_room": girls_in_room,
            "girls_ready_for_room": sum(1 for girl in self.girls if girl['ready_for_room']),
            "average_trust": avg_trust,
            "nearby_girls": len(self._get_nearby_girls()),
            "action_name": self.action_meanings[action]
        }
        
        return observation, reward, terminated, truncated, info
    
    def render(self):
        """Render the environment."""
        if self.render_mode == "rgb_array":
            frame = self._render_frame()
            self._capture_frame_for_gif()
            return frame
        elif self.render_mode == "human":
            self._render_frame()
            # Capture frame for GIF AFTER rendering to screen
            self._capture_frame_for_gif()
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])
    
    def _render_frame(self):
        """Render a single frame."""
        if self.screen is None:
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((self.width, self.height))
            self.font = pygame.font.SysFont("arial", 12)
        
        # Colors
        colors = {
            'background': (240, 240, 240),
            'grid': (200, 200, 200),
            'agent': (0, 0, 255),
            'empowered_room': (144, 238, 144),  # Light green for empowered room
            'girl_states': {
                0: (255, 0, 0),      # Unaware - Red
                1: (255, 165, 0),    # Confused - Orange
                2: (255, 192, 203),  # Embarrassed - Pink
                3: (255, 255, 0),    # Curious - Yellow
                4: (173, 216, 230),  # Informed - Light Blue
                5: (0, 255, 0)       # Empowered - Green
            }
        }
        
        # Fill background
        self.screen.fill(colors['background'])
        
        # Draw empowered room
        room_bounds = self.empowered_room["bounds"]
        room_rect = (
            room_bounds[0][0] * self.grid_size,
            room_bounds[0][1] * self.grid_size,
            (room_bounds[1][0] - room_bounds[0][0] + 1) * self.grid_size,
            (room_bounds[1][1] - room_bounds[0][1] + 1) * self.grid_size
        )
        pygame.draw.rect(self.screen, colors['empowered_room'], room_rect, 0)
        pygame.draw.rect(self.screen, (50, 150, 50), room_rect, 3)
        
        # Room label
        label = self.font.render("EMPOWERED ROOM", True, (0, 100, 0))
        self.screen.blit(label, (room_rect[0] + 5, room_rect[1] + 5))
        
        # Draw grid
        for x in range(0, self.width, self.grid_size):
            pygame.draw.line(self.screen, colors['grid'], (x, 0), (x, self.height), 1)
        for y in range(0, self.height, self.grid_size):
            pygame.draw.line(self.screen, colors['grid'], (0, y), (self.width, y), 1)
        
        # Draw girls
        for girl in self.girls:
            x = girl['pos'][0] * self.grid_size + self.grid_size // 2
            y = girl['pos'][1] * self.grid_size + self.grid_size // 2
            color = colors['girl_states'][girl['state']]
            
            # Special highlight if girl is ready for room but not in it
            if girl['ready_for_room'] and not girl['in_empowered_room']:
                pygame.draw.circle(self.screen, (255, 255, 255), (x, y), 20)
            
            # Draw movement trail for girls moving toward empowered room
            if girl['state'] >= 3:  # Show trail for curious+ girls
                room_center = [
                    (self.empowered_room["bounds"][0][0] + self.empowered_room["bounds"][1][0]) // 2,
                    (self.empowered_room["bounds"][0][1] + self.empowered_room["bounds"][1][1]) // 2
                ]
                room_center_px = [
                    room_center[0] * self.grid_size + self.grid_size // 2,
                    room_center[1] * self.grid_size + self.grid_size // 2
                ]
                # Draw a faint line from girl to room center
                trail_color = (200, 200, 200) if girl['state'] < 5 else (100, 255, 100)
                pygame.draw.line(self.screen, trail_color, (x, y), room_center_px, 1)
            
            pygame.draw.circle(self.screen, color, (x, y), 15)
            pygame.draw.circle(self.screen, (0, 0, 0), (x, y), 15, 2)
            
            # State label
            state_label = self.font.render(f"G{girl['id']}", True, (0, 0, 0))
            self.screen.blit(state_label, (x - 8, y - 5))
            
            # Trust level as a small bar
            trust_width = int(girl['trust'] / 100 * 20)
            pygame.draw.rect(self.screen, (255, 0, 0), (x - 10, y + 20, 20, 3))
            pygame.draw.rect(self.screen, (0, 255, 0), (x - 10, y + 20, trust_width, 3))
            
            # Ready indicator
            if girl['ready_for_room']:
                ready_label = self.font.render("READY", True, (0, 150, 0))
                self.screen.blit(ready_label, (x - 15, y + 25))
        
        # Draw agent
        agent_x = self.agent_pos[0] * self.grid_size + self.grid_size // 2  
        agent_y = self.agent_pos[1] * self.grid_size + self.grid_size // 2
        pygame.draw.rect(self.screen, colors['agent'], 
                        (agent_x - 12, agent_y - 12, 24, 24))
        pygame.draw.rect(self.screen, (0, 0, 0), 
                        (agent_x - 12, agent_y - 12, 24, 24), 2)
        
        # Agent label
        agent_label = self.font.render("AI", True, (255, 255, 255))
        self.screen.blit(agent_label, (agent_x - 8, agent_y - 5))
        
        # Draw interaction range
        nearby_girls = self._get_nearby_girls()
        if nearby_girls:
            pygame.draw.circle(self.screen, (0, 255, 0), (agent_x, agent_y), 
                             (self.interaction_range + 1) * self.grid_size, 2)
        
        # Display info
        girls_in_room = sum(1 for girl in self.girls if girl['in_empowered_room'])
        girls_ready = sum(1 for girl in self.girls if girl['ready_for_room'])
        avg_trust = np.mean([girl['trust'] for girl in self.girls])
        
        info_texts = [
            f"Step: {self.current_step}/{self.max_steps}",
            f"Agent: ({self.agent_pos[0]}, {self.agent_pos[1]})",
            f"In Room: {girls_in_room}/{self.num_girls}",
            f"Ready for Room: {girls_ready}",
            f"Avg Trust: {avg_trust:.1f}%",
            f"Nearby Girls: {len(nearby_girls)}"
        ]
        
        # Add GIF recording indicator
        if self.record_gif:
            info_texts.append(f"🔴 Recording GIF ({len(self.gif_frames)} frames)")
        
        for i, text in enumerate(info_texts):
            text_surface = self.font.render(text, True, (0, 0, 0))
            self.screen.blit(text_surface, (10, 10 + i * 20))
        
        # Legend
        legend_y = self.height - 170
        legend_text = self.font.render("Girl States:", True, (0, 0, 0))
        self.screen.blit(legend_text, (10, legend_y))
        
        for i, (state_id, state_name) in enumerate(self.girl_states.items()):
            color = colors['girl_states'][state_id]
            y_pos = legend_y + 20 + i * 15
            pygame.draw.circle(self.screen, color, (15, y_pos + 5), 5)
            state_text = self.font.render(state_name, True, (0, 0, 0))
            self.screen.blit(state_text, (25, y_pos))
        
        # Action legend
        action_legend_y = legend_y + 110
        action_text = self.font.render("Actions: Move(0-8), Educate(9-14), Guide(15)", True, (0, 0, 0))
        self.screen.blit(action_text, (10, action_legend_y))
        
        # Return RGB array for both render modes when needed
        if self.render_mode == "rgb_array" or self.record_gif:
            return np.transpose(pygame.surfarray.array3d(self.screen), axes=(1, 0, 2))
        return None
    
    def close(self):
        """Clean up resources."""
        # Save any remaining GIF frames before closing
        if self.record_gif and self.gif_frames:
            self.stop_gif_recording()
        
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()


# Demo script with GIF recording functionality
def demo_spatial_environment_with_gif():
    """Demonstrate the spatial environment with random actions and GIF recording."""
    # Create environment with GIF recording enabled
    env = TeenEducationEnvironment(render_mode="human", record_gif=False)
    
    print("=== Spatial Teen Education Environment Demo with GIF Recording ===")
    print("Agent (blue square) must move around and interact with girls (colored circles)")
    print("Actions 0-8: Movement, Actions 9-14: Educational interactions, Action 15: Guide to room")
    print("Goal: Guide all girls to the green 'EMPOWERED ROOM' after educating them")
    print("Girls must be in state 4+ with 65+ trust to be ready for the room")
    print()
    
    for episode in range(3):
        print(f"\n--- Episode {episode + 1} ---")
        
        # Start GIF recording for this episode
        gif_filename = f"teen_education_episode_{episode + 1}.gif"
        env.start_gif_recording(gif_filename, frame_skip=2)  # Record every 2nd frame to reduce file size
        
        observation, info = env.reset()
        total_reward = 0
        
        for step in range(100):  # Reduced steps for shorter GIF
            # Intelligent action selection for demo
            nearby_girls = env._get_nearby_girls()
            
            if nearby_girls:
                girl = nearby_girls[0]
                if girl['ready_for_room'] and not girl['in_empowered_room'] and random.random() < 0.8:
                    action = 15  # Guide to empowered room
                    action_type = "Guidance"
                elif random.random() < 0.7:
                    action = random.randint(9, 14)  # Educational action
                    action_type = "Educational"
                else:
                    action = random.randint(0, 8)   # Movement action  
                    action_type = "Movement"
            else:
                action = random.randint(0, 8)   # Movement action  
                action_type = "Movement"
            
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            print(f"Step {step + 1}: {action_type} action {action} "
                  f"({info['action_name']}) -> Reward: {reward:.2f}")
            print(f"  Agent at ({info['agent_pos'][0]}, {info['agent_pos'][1]}), "
                  f"In Room: {info['girls_in_room']}/{env.num_girls}, "
                  f"Ready: {info['girls_ready_for_room']}, "
                  f"Avg Trust: {info['average_trust']:.1f}%")
            
            env.render()
            
            if terminated or truncated:
                print(f"Episode ended! Total reward: {total_reward:.2f}")
                print(f"Final girls in room: {info['girls_in_room']}/{env.num_girls}")
                break
        
        # Stop GIF recording and save
        env.stop_gif_recording()
        print(f"GIF saved as: {gif_filename}")
    
    env.close()

# Alternative function for manual GIF control
def demo_with_manual_gif_control():
    """Demo with manual GIF recording control."""
    env = TeenEducationEnvironment(render_mode="human")
    
    print("=== Manual GIF Recording Demo ===")
    print("You can start/stop GIF recording programmatically")
    print()
    
    observation, info = env.reset()
    
    # Run for a bit without recording
    print("Running 20 steps without recording...")
    for step in range(20):
        action = random.randint(0, 15)
        observation, reward, terminated, truncated, info = env.step(action)
        env.render()
        if terminated or truncated:
            break
    
    # Start recording
    print("Starting GIF recording...")
    env.start_gif_recording("manual_recording.gif", frame_skip=1)
    
    # Run with recording
    for step in range(50):
        nearby_girls = env._get_nearby_girls()
        if nearby_girls and random.random() < 0.6:
            action = random.randint(9, 15)  # Prefer interactions
        else:
            action = random.randint(0, 8)   # Movement
            
        observation, reward, terminated, truncated, info = env.step(action)
        env.render()
        
        if terminated or truncated:
            break
    
    # Stop recording
    print("Stopping GIF recording...")
    env.stop_gif_recording()
    
    env.close()

if __name__ == "__main__":
    # Run the demo with GIF recording
    demo_spatial_environment_with_gif()
