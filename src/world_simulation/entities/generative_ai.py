"""Generative AI integration for complex NPC reasoning."""

from typing import Dict, Any, Optional
import json


class GenerativeAIReasoner:
    """
    Uses generative AI (LLM) for complex decision making and reasoning.
    
    This provides more sophisticated reasoning capabilities beyond simple
    neural network decisions, allowing NPCs to make contextual decisions
    based on their situation.
    """
    
    def __init__(self, use_openai: bool = False, api_key: Optional[str] = None):
        """
        Initialize the generative AI reasoner.
        
        Args:
            use_openai: Whether to use OpenAI API (requires API key)
            api_key: OpenAI API key (if using OpenAI)
        """
        self.use_openai = use_openai
        self.api_key = api_key
        self.openai_client = None
        
        if use_openai and api_key:
            try:
                import openai
                self.openai_client = openai.OpenAI(api_key=api_key)
            except ImportError:
                print("Warning: OpenAI package not installed. Install with: pip install openai")
                self.use_openai = False
    
    def reason_about_action(self, npc_state: Dict[str, Any], world_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Use generative AI to reason about the best action.
        
        Args:
            npc_state: Dictionary with NPC state (health, hunger, stamina, etc.)
            world_context: Dictionary with world context (is_night, nearby_resources, etc.)
            
        Returns:
            Dictionary with reasoning and suggested action
        """
        if not self.use_openai or not self.openai_client:
            # Fallback to rule-based reasoning if OpenAI not available
            return self._rule_based_reasoning(npc_state, world_context)
        
        # Construct prompt for LLM
        prompt = self._construct_prompt(npc_state, world_context)
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an AI assistant helping NPCs make decisions in a survival simulation."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=150,
                temperature=0.7
            )
            
            reasoning = response.choices[0].message.content
            return self._parse_llm_response(reasoning, npc_state, world_context)
        
        except Exception as e:
            print(f"Error calling OpenAI API: {e}")
            return self._rule_based_reasoning(npc_state, world_context)
    
    def _construct_prompt(self, npc_state: Dict[str, Any], world_context: Dict[str, Any]) -> str:
        """Construct a prompt for the LLM."""
        prompt = f"""NPC State:
- Health: {npc_state.get('health', 0):.1f}/{npc_state.get('max_health', 100):.1f}
- Hunger: {npc_state.get('hunger', 0):.1f}/{npc_state.get('max_hunger', 100):.1f}
- Stamina: {npc_state.get('stamina', 0):.1f}/{npc_state.get('max_stamina', 100):.1f}
- Age Stage: {npc_state.get('age_stage', 'unknown')}
- Is Night: {world_context.get('is_night', False)}
- Nearest Food Distance: {world_context.get('nearest_food_dist', 'far')}
- Nearest Shelter Distance: {world_context.get('nearest_shelter_dist', 'far')}
- In Shelter: {npc_state.get('in_shelter', False)}

What should this NPC do? Respond with JSON: {{"action": "action_name", "reasoning": "brief explanation"}}
Available actions: wandering, seeking_food, eating, resting, seeking_shelter, stay_in_shelter"""
        return prompt
    
    def _parse_llm_response(self, response: str, npc_state: Dict[str, Any], world_context: Dict[str, Any]) -> Dict[str, Any]:
        """Parse LLM response to extract action and reasoning."""
        try:
            # Try to extract JSON from response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                result = json.loads(json_str)
                return {
                    'action': result.get('action', 'wandering'),
                    'reasoning': result.get('reasoning', 'No reasoning provided'),
                    'confidence': 0.8
                }
        except:
            pass
        
        # Fallback: try to find action name in response
        actions = ['wandering', 'seeking_food', 'eating', 'resting', 'seeking_shelter', 'stay_in_shelter']
        for action in actions:
            if action.lower() in response.lower():
                return {
                    'action': action,
                    'reasoning': response[:100],
                    'confidence': 0.6
                }
        
        return self._rule_based_reasoning(npc_state, world_context)
    
    def _rule_based_reasoning(self, npc_state: Dict[str, Any], world_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fallback rule-based reasoning when LLM is not available.
        
        This provides basic intelligent reasoning without requiring API calls.
        """
        health_ratio = npc_state.get('health', 100) / max(1.0, npc_state.get('max_health', 100))
        hunger_ratio = npc_state.get('hunger', 50) / max(1.0, npc_state.get('max_hunger', 100))
        stamina_ratio = npc_state.get('stamina', 50) / max(1.0, npc_state.get('max_stamina', 100))
        is_night = world_context.get('is_night', False)
        in_shelter = npc_state.get('in_shelter', False)
        
        # Priority-based decision making
        if health_ratio < 0.2:
            return {
                'action': 'seeking_shelter' if is_night else 'resting',
                'reasoning': 'Health is critically low, prioritizing survival',
                'confidence': 0.9
            }
        
        if hunger_ratio < 0.3:
            return {
                'action': 'seeking_food',
                'reasoning': 'Hunger is low, need to find food',
                'confidence': 0.85
            }
        
        if is_night and not in_shelter:
            return {
                'action': 'seeking_shelter',
                'reasoning': 'Night time, need shelter for safety',
                'confidence': 0.9
            }
        
        if stamina_ratio < 0.2:
            return {
                'action': 'resting',
                'reasoning': 'Stamina is low, need to rest',
                'confidence': 0.8
            }
        
        return {
            'action': 'wandering',
            'reasoning': 'Exploring and looking for opportunities',
            'confidence': 0.7
        }

