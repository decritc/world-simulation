"""Tests for generative AI reasoning."""

import pytest
from world_simulation.entities.generative_ai import GenerativeAIReasoner


class TestGenerativeAIReasoner:
    """Test generative AI reasoning functionality."""
    
    def test_init_without_openai(self):
        """Test initialization without OpenAI."""
        reasoner = GenerativeAIReasoner(use_openai=False)
        
        assert reasoner.use_openai == False
        assert reasoner.openai_client is None
    
    def test_init_with_openai_no_key(self):
        """Test initialization with OpenAI flag but no key."""
        reasoner = GenerativeAIReasoner(use_openai=True, api_key=None)
        
        # Should fall back to not using OpenAI if no key
        assert reasoner.use_openai == False or reasoner.openai_client is None
    
    def test_rule_based_reasoning(self):
        """Test rule-based reasoning fallback."""
        reasoner = GenerativeAIReasoner(use_openai=False)
        
        npc_state = {
            'health': 100.0,
            'max_health': 100.0,
            'hunger': 20.0,  # Low hunger
            'max_hunger': 100.0,
            'stamina': 50.0,
            'max_stamina': 100.0,
            'age_stage': 'adult',
            'in_shelter': False,
        }
        
        world_context = {
            'is_night': False,
            'nearest_food_dist': 'near',
            'nearest_shelter_dist': 'far',
        }
        
        result = reasoner.reason_about_action(npc_state, world_context)
        
        assert 'action' in result
        assert 'reasoning' in result
        assert 'confidence' in result
        assert result['action'] in ['wandering', 'seeking_food', 'eating', 'resting', 'seeking_shelter', 'stay_in_shelter']
        assert result['confidence'] > 0.0
    
    def test_rule_based_reasoning_low_health(self):
        """Test rule-based reasoning prioritizes low health."""
        reasoner = GenerativeAIReasoner(use_openai=False)
        
        npc_state = {
            'health': 10.0,  # Very low health
            'max_health': 100.0,
            'hunger': 50.0,
            'max_hunger': 100.0,
            'stamina': 50.0,
            'max_stamina': 100.0,
            'age_stage': 'adult',
            'in_shelter': False,
        }
        
        world_context = {
            'is_night': False,
            'nearest_food_dist': 'near',
            'nearest_shelter_dist': 'far',
        }
        
        result = reasoner.reason_about_action(npc_state, world_context)
        
        # Should prioritize survival (resting or seeking shelter)
        assert result['action'] in ['resting', 'seeking_shelter', 'wandering']
        assert result['confidence'] >= 0.7
    
    def test_rule_based_reasoning_nighttime(self):
        """Test rule-based reasoning handles nighttime."""
        reasoner = GenerativeAIReasoner(use_openai=False)
        
        npc_state = {
            'health': 80.0,
            'max_health': 100.0,
            'hunger': 60.0,
            'max_hunger': 100.0,
            'stamina': 70.0,
            'max_stamina': 100.0,
            'age_stage': 'adult',
            'in_shelter': False,
        }
        
        world_context = {
            'is_night': True,  # Nighttime
            'nearest_food_dist': 'near',
            'nearest_shelter_dist': 'near',
        }
        
        result = reasoner.reason_about_action(npc_state, world_context)
        
        # Should prioritize seeking shelter at night
        assert result['action'] == 'seeking_shelter'
        assert result['confidence'] >= 0.8
    
    def test_rule_based_reasoning_low_stamina(self):
        """Test rule-based reasoning handles low stamina."""
        reasoner = GenerativeAIReasoner(use_openai=False)
        
        npc_state = {
            'health': 90.0,
            'max_health': 100.0,
            'hunger': 70.0,
            'max_hunger': 100.0,
            'stamina': 15.0,  # Low stamina
            'max_stamina': 100.0,
            'age_stage': 'adult',
            'in_shelter': False,
        }
        
        world_context = {
            'is_night': False,
            'nearest_food_dist': 'near',
            'nearest_shelter_dist': 'far',
        }
        
        result = reasoner.reason_about_action(npc_state, world_context)
        
        # Should prioritize resting
        assert result['action'] == 'resting'
        assert result['confidence'] >= 0.7
    
    def test_rule_based_reasoning_normal_state(self):
        """Test rule-based reasoning for normal state."""
        reasoner = GenerativeAIReasoner(use_openai=False)
        
        npc_state = {
            'health': 90.0,
            'max_health': 100.0,
            'hunger': 80.0,  # Well fed
            'max_hunger': 100.0,
            'stamina': 80.0,  # Good stamina
            'max_stamina': 100.0,
            'age_stage': 'adult',
            'in_shelter': False,
        }
        
        world_context = {
            'is_night': False,
            'nearest_food_dist': 'near',
            'nearest_shelter_dist': 'far',
        }
        
        result = reasoner.reason_about_action(npc_state, world_context)
        
        # Should suggest wandering or exploring
        assert result['action'] in ['wandering', 'seeking_food']
        assert result['confidence'] > 0.0

