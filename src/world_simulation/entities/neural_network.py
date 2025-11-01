"""Neural network implementation for NPC decision making."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple, List


class NPCDecisionNetwork(nn.Module):
    """
    Neural network for NPC decision making.
    
    Input features:
    - Internal state: health, hunger, stamina, age (normalized)
    - Environmental: is_night, distance_to_nearest_food, distance_to_nearest_shelter
    - Capabilities: can_reproduce, age_stage (encoded)
    - Recent actions: last_action (encoded)
    
    Output actions:
    - Action type: wander, seek_food, eat, rest, seek_shelter, stay_in_shelter
    - Movement: target_x, target_z (relative to current position)
    """
    
    def __init__(self, input_size: int = 18, hidden_size: int = 64, output_size: int = 8):
        """
        Initialize the neural network.
        
        Args:
            input_size: Number of input features
            hidden_size: Size of hidden layers
            output_size: Number of output actions (6 action types + 2 movement coords)
        """
        super(NPCDecisionNetwork, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Neural network layers
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc_action = nn.Linear(hidden_size // 2, 6)  # 6 action types
        self.fc_movement = nn.Linear(hidden_size // 2, 2)  # target_x, target_z
        
        # Activation functions
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)
        self.tanh = nn.Tanh()
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, input_size)
            
        Returns:
            Tuple of (action_probs, movement_target) where:
            - action_probs: Probability distribution over actions (batch_size, 6)
            - movement_target: Relative movement target (batch_size, 2)
        """
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        
        action_logits = self.fc_action(x)
        action_probs = self.softmax(action_logits)
        
        movement = self.tanh(self.fc_movement(x))  # Normalized to [-1, 1]
        
        return action_probs, movement
    
    def get_weights(self) -> np.ndarray:
        """Get all network weights as a numpy array for genetic algorithm."""
        weights = []
        for param in self.parameters():
            weights.append(param.data.cpu().numpy().flatten())
        return np.concatenate(weights)
    
    def set_weights(self, weights: np.ndarray):
        """Set network weights from a numpy array."""
        idx = 0
        for param in self.parameters():
            param_size = param.numel()
            param_data = weights[idx:idx + param_size].reshape(param.shape)
            param.data = torch.from_numpy(param_data).float()
            idx += param_size
    
    def mutate(self, mutation_rate: float = 0.1, mutation_strength: float = 0.1):
        """Apply random mutations to the network weights."""
        with torch.no_grad():
            for param in self.parameters():
                if np.random.random() < mutation_rate:
                    noise = torch.randn_like(param) * mutation_strength
                    param.add_(noise)
    
    @staticmethod
    def crossover(parent1: 'NPCDecisionNetwork', parent2: 'NPCDecisionNetwork') -> 'NPCDecisionNetwork':
        """Create a child network by crossing over two parent networks."""
        child = NPCDecisionNetwork(parent1.input_size, parent1.hidden_size, parent1.output_size)
        
        weights1 = parent1.get_weights()
        weights2 = parent2.get_weights()
        
        # Uniform crossover: randomly choose weights from each parent
        mask = np.random.random(len(weights1)) < 0.5
        child_weights = np.where(mask, weights1, weights2)
        
        child.set_weights(child_weights)
        return child


class FeatureExtractor:
    """Extract features from NPC state and world context for neural network input."""
    
    # Action encoding map
    ACTION_TO_ID = {
        'wandering': 0,
        'seeking_food': 1,
        'eating': 2,
        'hunting': 2,  # Map hunting to same as eating (can be refined later)
        'resting': 3,
        'seeking_shelter': 4,
        'in_shelter': 5,
    }
    
    ID_TO_ACTION = {v: k for k, v in ACTION_TO_ID.items()}
    
    @staticmethod
    def extract_features(npc, world) -> np.ndarray:
        """
        Extract features for neural network input.
        
        Args:
            npc: NPC instance
            world: World instance
            
        Returns:
            Feature vector of shape (18,)
        """
        features = []
        
        # Internal state (normalized to [0, 1])
        features.append(npc.health / npc.max_health if npc.max_health > 0 else 0.0)
        features.append(npc.hunger / npc.max_hunger if npc.max_hunger > 0 else 0.0)
        features.append(npc.stamina / npc.max_stamina if npc.max_stamina > 0 else 0.0)
        features.append(min(1.0, npc.age / npc.lifespan) if npc.lifespan > 0 else 0.0)
        
        # Age stage encoding (one-hot: child=0, adult=1, elder=2)
        age_stage_vec = [0.0, 0.0, 0.0]
        if npc.age_stage == "child":
            age_stage_vec[0] = 1.0
        elif npc.age_stage == "adult":
            age_stage_vec[1] = 1.0
        elif npc.age_stage == "elder":
            age_stage_vec[2] = 1.0
        features.extend(age_stage_vec)
        
        # Environmental features
        is_night = 1.0 if world.is_night() else 0.0
        features.append(is_night)
        
        # Find nearest food source (trees or animals)
        vision_range = npc.genome.get('vision_range', 10.0)
        nearest_food_dist = vision_range  # Default to max vision
        
        # Check trees
        for tree in world.trees:
            if tree.is_alive and tree.get_ripe_fruit_count() > 0:
                dx = tree.x - npc.x
                dz = tree.z - npc.z
                dist = np.sqrt(dx**2 + dz**2)
                if dist < nearest_food_dist:
                    nearest_food_dist = dist
        
        # Check animals (prefer animals if very hungry)
        prefer_animal = npc.hunger < 40.0
        nearest_animal_dist = vision_range
        for animal in world.animals:
            if animal.is_alive:
                dx = animal.x - npc.x
                dz = animal.z - npc.z
                dist = np.sqrt(dx**2 + dz**2)
                if dist < nearest_animal_dist:
                    nearest_animal_dist = dist
        
        # Use closer food source
        if prefer_animal and nearest_animal_dist < vision_range:
            nearest_food_dist = min(nearest_food_dist, nearest_animal_dist)
        else:
            nearest_food_dist = min(nearest_food_dist, nearest_animal_dist)
        
        # Normalize distance (0 = nearby, 1 = far)
        features.append(min(1.0, nearest_food_dist / vision_range))
        
        # Find nearest shelter
        nearest_shelter_dist = 50.0  # Default to large distance
        for house in world.houses:
            if house.is_built and house.can_shelter_adult():
                dx = house.x - npc.x
                dz = house.z - npc.z
                dist = np.sqrt(dx**2 + dz**2)
                if dist < nearest_shelter_dist:
                    nearest_shelter_dist = dist
        
        # Normalize distance (0 = nearby, 1 = far)
        features.append(min(1.0, nearest_shelter_dist / 50.0))
        
        # Capabilities
        features.append(1.0 if npc.can_reproduce else 0.0)
        features.append(1.0 if npc.current_house is not None else 0.0)
        
        # Recent action (one-hot encoded)
        last_action_vec = [0.0] * 6
        if npc.state in FeatureExtractor.ACTION_TO_ID:
            action_id = FeatureExtractor.ACTION_TO_ID[npc.state]
            last_action_vec[action_id] = 1.0
        features.extend(last_action_vec)
        
        return np.array(features, dtype=np.float32)
    
    @staticmethod
    def decode_action(action_probs: np.ndarray) -> str:
        """Decode action probabilities to action name."""
        action_id = np.argmax(action_probs)
        return FeatureExtractor.ID_TO_ACTION.get(action_id, 'wandering')
    
    @staticmethod
    def decode_movement(movement_vec: np.ndarray, npc, max_distance: float = 15.0) -> Tuple[float, float]:
        """
        Decode movement vector to target coordinates.
        
        Args:
            movement_vec: Normalized movement vector [-1, 1]
            npc: NPC instance
            max_distance: Maximum distance to move
            
        Returns:
            Tuple of (target_x, target_z)
        """
        # Movement vector is relative to current position
        angle = (movement_vec[0] + 1.0) * np.pi  # Map [-1, 1] to [0, 2π]
        distance = (movement_vec[1] + 1.0) * 0.5 * max_distance  # Map [-1, 1] to [0, max_distance]
        
        target_x = npc.x + np.cos(angle) * distance
        target_z = npc.z + np.sin(angle) * distance
        
        return target_x, target_z

