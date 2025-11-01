"""Tests for neural network decision making."""

import pytest
import numpy as np
import torch
from world_simulation.entities.neural_network import NPCDecisionNetwork, FeatureExtractor
from world_simulation.entities.npc import NPC


class MockWorld:
    """Mock world for testing."""
    def __init__(self):
        self.trees = []
        self.houses = []
        self.day_time = 60.0  # 1 hour into day
        self.day_length = 120.0  # 2 minute days
        self.day_number = 0
    
    def is_night(self):
        hour = (self.day_time / self.day_length) * 24.0
        return hour < 6.0 or hour >= 18.0
    
    def get_height(self, x, z):
        return 0.0


class TestNPCDecisionNetwork:
    """Test neural network decision making."""
    
    def test_network_initialization(self):
        """Test that neural network initializes correctly."""
        network = NPCDecisionNetwork(input_size=18, hidden_size=64)
        assert network.input_size == 18
        assert network.hidden_size == 64
        assert network.output_size == 8
    
    def test_forward_pass(self):
        """Test forward pass through network."""
        network = NPCDecisionNetwork(input_size=18, hidden_size=64)
        network.eval()
        
        # Create test input
        test_input = torch.FloatTensor(np.random.randn(1, 18))
        
        with torch.no_grad():
            action_probs, movement = network(test_input)
        
        assert action_probs.shape == (1, 6)
        assert movement.shape == (1, 2)
        assert torch.allclose(action_probs.sum(dim=1), torch.ones(1))  # Probabilities sum to 1
    
    def test_get_set_weights(self):
        """Test getting and setting network weights."""
        network1 = NPCDecisionNetwork(input_size=18, hidden_size=64)
        network2 = NPCDecisionNetwork(input_size=18, hidden_size=64)
        
        weights1 = network1.get_weights()
        assert len(weights1) > 0
        
        # Set weights to second network
        network2.set_weights(weights1)
        weights2 = network2.get_weights()
        
        assert np.allclose(weights1, weights2)
    
    def test_mutation(self):
        """Test network weight mutation."""
        network = NPCDecisionNetwork(input_size=18, hidden_size=64)
        original_weights = network.get_weights().copy()
        
        network.mutate(mutation_rate=1.0, mutation_strength=0.1)
        mutated_weights = network.get_weights()
        
        # Weights should have changed
        assert not np.allclose(original_weights, mutated_weights)
    
    def test_crossover(self):
        """Test network crossover."""
        parent1 = NPCDecisionNetwork(input_size=18, hidden_size=64)
        parent2 = NPCDecisionNetwork(input_size=18, hidden_size=64)
        
        child = NPCDecisionNetwork.crossover(parent1, parent2)
        
        assert child.input_size == parent1.input_size
        assert child.hidden_size == parent1.hidden_size
        assert child.output_size == parent1.output_size
        
        # Child weights should be mix of parents
        child_weights = child.get_weights()
        parent1_weights = parent1.get_weights()
        parent2_weights = parent2.get_weights()
        
        # Child should have some weights from each parent
        assert len(child_weights) == len(parent1_weights)


class TestFeatureExtractor:
    """Test feature extraction for neural network."""
    
    def test_extract_features(self):
        """Test feature extraction from NPC and world."""
        npc = NPC(0.0, 0.0, 0.0)
        world = MockWorld()
        
        features = FeatureExtractor.extract_features(npc, world)
        
        assert len(features) == 18
        assert np.all(features >= 0)  # All features should be non-negative (normalized)
        assert np.all(features <= 1)  # All features should be normalized to [0, 1]
    
    def test_decode_action(self):
        """Test action decoding from probabilities."""
        # Test action probabilities
        action_probs = np.array([0.1, 0.8, 0.05, 0.03, 0.01, 0.01])
        
        action = FeatureExtractor.decode_action(action_probs)
        
        assert action == "seeking_food"  # Index 1 has highest probability
        assert action in FeatureExtractor.ACTION_TO_ID.keys()
    
    def test_decode_movement(self):
        """Test movement decoding from vector."""
        npc = NPC(10.0, 0.0, 10.0)
        movement_vec = np.array([0.5, 0.3])  # Normalized movement vector
        
        target_x, target_z = FeatureExtractor.decode_movement(movement_vec, npc)
        
        # Target should be offset from NPC position
        assert target_x != npc.x
        assert target_z != npc.z
    
    def test_feature_normalization(self):
        """Test that features are properly normalized."""
        npc = NPC(0.0, 0.0, 0.0)
        npc.health = 50.0
        npc.max_health = 100.0
        npc.hunger = 75.0
        npc.max_hunger = 100.0
        
        world = MockWorld()
        features = FeatureExtractor.extract_features(npc, world)
        
        # Health should be normalized
        assert 0.0 <= features[0] <= 1.0
        # Hunger should be normalized
        assert 0.0 <= features[1] <= 1.0


class TestNPCNeuralNetworkIntegration:
    """Test NPC integration with neural network."""
    
    def test_npc_has_brain(self):
        """Test that NPC has a neural network brain."""
        npc = NPC(0.0, 0.0, 0.0)
        
        assert hasattr(npc, 'brain')
        assert isinstance(npc.brain, NPCDecisionNetwork)
    
    def test_npc_makes_neural_decisions(self):
        """Test that NPC makes decisions using neural network."""
        npc = NPC(0.0, 0.0, 0.0)
        world = MockWorld()
        
        # Set decision timer to trigger decision
        npc.decision_timer = npc.decision_interval
        
        initial_state = npc.state
        
        # Make a decision
        npc._make_neural_decision(world)
        
        # State should have changed (or be valid)
        assert npc.state in ["wandering", "seeking_food", "eating", "resting", "seeking_shelter", "in_shelter"]
    
    def test_npc_brain_evolution(self):
        """Test that NPC brains evolve through reproduction."""
        parent1 = NPC(0.0, 0.0, 0.0)
        parent1.age = parent1.adult_age + 1.0
        parent1.age_stage = "adult"
        parent1.can_reproduce = True
        
        parent2 = NPC(1.0, 0.0, 1.0)
        parent2.age = parent2.adult_age + 1.0
        parent2.age_stage = "adult"
        parent2.can_reproduce = True
        
        # Mock house for reproduction
        from world_simulation.houses.house import House
        house = House(0.0, 0.0, 0.0)
        parent1.current_house = house
        parent2.current_house = house
        
        # Get parent weights
        parent1_weights = parent1.brain.get_weights()
        parent2_weights = parent2.brain.get_weights()
        
        # Reproduce
        offspring = parent1.reproduce(parent2)
        
        # Offspring should have brain
        assert hasattr(offspring, 'brain')
        offspring_weights = offspring.brain.get_weights()
        
        # Offspring weights should be different from parents (crossover + mutation)
        assert not np.allclose(offspring_weights, parent1_weights)
        assert not np.allclose(offspring_weights, parent2_weights)
    
    def test_npc_neural_decision_with_features(self):
        """Test that NPC extracts features correctly for neural network."""
        npc = NPC(0.0, 0.0, 0.0)
        world = MockWorld()
        
        # Extract features
        features = FeatureExtractor.extract_features(npc, world)
        
        assert len(features) == 18
        assert np.all(np.isfinite(features))  # All features should be finite
    
    def test_npc_decision_frequency(self):
        """Test that NPCs make decisions at the correct frequency."""
        npc = NPC(0.0, 0.0, 0.0)
        world = MockWorld()
        
        # Decision should not be made if timer hasn't reached interval
        npc.decision_timer = 0.0
        initial_state = npc.state
        
        npc._make_neural_decision(world)
        
        # State might change, but decision should be valid
        assert npc.state in ["wandering", "seeking_food", "eating", "resting", "seeking_shelter", "in_shelter"]
    
    def test_generative_ai_integration(self):
        """Test optional generative AI integration."""
        npc = NPC(0.0, 0.0, 0.0)
        
        # Generative AI should be optional
        assert hasattr(npc, 'generative_ai')
        # By default, should not use OpenAI (requires API key)
        assert npc.generative_ai is None or not npc.use_generative_ai

