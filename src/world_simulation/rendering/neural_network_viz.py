"""Neural network visualization module."""

import pyglet
from pyglet.gl import *
import numpy as np
import math

from ..entities.npc import NPC


class NeuralNetworkVisualization:
    """Handles rendering of neural network visualizations."""
    
    def __init__(self):
        """Initialize the visualization."""
        self.nn_title_height = 30
        self.nn_info_height = 48
    
    def render(self, npc: NPC, x, y, width, height):
        """
        Render a graphical representation of the NPC's neural network.
        
        Args:
            npc: NPC instance
            x: Left position of visualization area
            y: Bottom position of visualization area
            width: Width of visualization area
            height: Height of visualization area
        """
        if not hasattr(npc, 'brain'):
            return
        
        # Setup 2D rendering
        glDisable(GL_LIGHTING)
        glDisable(GL_DEPTH_TEST)
        
        brain = npc.brain
        
        # Render title
        self._render_title(x, y, height)
        
        # Get network structure
        actual_layer_sizes = [
            brain.input_size,  # 18
            brain.fc1.out_features,  # 64
            brain.fc2.out_features,  # 64
            brain.fc3.out_features,  # 32
            6  # action outputs
        ]
        
        # Scale down for visualization
        max_neurons_to_show = 18
        scale_factor = min(1.0, max_neurons_to_show / max(actual_layer_sizes))
        layer_sizes = [max(1, int(size * scale_factor)) for size in actual_layer_sizes]
        layer_sizes = [max(3, size) if i > 0 else size for i, size in enumerate(layer_sizes)]
        layer_sizes[-1] = 6  # Always show all 6 output actions
        
        # Render network visualization
        self._render_network(npc, x, y, width, height, layer_sizes)
        
        # Render info text at bottom
        self._render_info_text(npc, x, y, width, height, layer_sizes)
    
    def _render_title(self, x, y, height):
        """Render the neural network title."""
        try:
            title_label = pyglet.text.Label(
                "Neural Network Architecture:",
                font_name='Courier New',
                font_size=11,
                x=int(x + 10),
                y=int(y + height - 15),
                anchor_x='left',
                anchor_y='top',
                color=(200, 200, 255, 255)
            )
            title_label.draw()
        except:
            pass
    
    def _render_network(self, npc: NPC, x, y, width, height, layer_sizes):
        """Render the neural network diagram."""
        brain = npc.brain
        
        num_layers = len(layer_sizes)
        layer_spacing = (width - 40) / max(1, num_layers - 1)
        neuron_radius = 3.0
        vis_height = height - 50  # Leave space for title and info
        
        # Draw connections first
        glLineWidth(0.5)
        for layer_idx in range(num_layers - 1):
            num_neurons_current = layer_sizes[layer_idx]
            num_neurons_next = layer_sizes[layer_idx + 1]
            
            layer_x = x + 20 + layer_idx * layer_spacing
            next_layer_x = x + 20 + (layer_idx + 1) * layer_spacing
            
            # Get weights
            layer_weights = self._get_layer_weights(brain, layer_idx)
            
            # Calculate neuron positions within visualization area (excluding title and info)
            vis_y_start = y + self.nn_info_height  # Start above info text
            vis_y_end = y + height - self.nn_title_height  # End below title
            
            for neuron_idx_current in range(num_neurons_current):
                neuron_y_current = vis_y_start + (neuron_idx_current + 1) * (vis_y_end - vis_y_start) / (num_neurons_current + 1)
                
                for neuron_idx_next in range(num_neurons_next):
                    neuron_y_next = vis_y_start + (neuron_idx_next + 1) * (vis_y_end - vis_y_start) / (num_neurons_next + 1)
                    
                    weight = self._get_weight(layer_weights, neuron_idx_current, neuron_idx_next)
                    self._draw_connection(layer_x, neuron_y_current, next_layer_x, neuron_y_next, weight)
        
        # Draw neurons
        for layer_idx in range(num_layers):
            num_neurons = layer_sizes[layer_idx]
            layer_x = x + 20 + layer_idx * layer_spacing
            
            # Layer label
            self._render_layer_label(layer_idx, layer_x, y, height)
            
            # Draw neurons
            vis_y_start = y + self.nn_info_height
            vis_y_end = y + height - self.nn_title_height
            for neuron_idx in range(num_neurons):
                neuron_y = vis_y_start + (neuron_idx + 1) * (vis_y_end - vis_y_start) / (num_neurons + 1)
                self._draw_neuron(layer_x, neuron_y, neuron_radius, layer_idx, num_layers)
    
    def _get_layer_weights(self, brain, layer_idx):
        """Get weights for a specific layer."""
        try:
            if layer_idx == 0:
                weights = brain.fc1.weight.data.cpu().numpy()
            elif layer_idx == 1:
                weights = brain.fc2.weight.data.cpu().numpy()
            elif layer_idx == 2:
                weights = brain.fc3.weight.data.cpu().numpy()
            elif layer_idx == 3:
                weights = brain.fc_action.weight.data.cpu().numpy()
            else:
                return None
            
            # Normalize weights
            weight_max = max(abs(weights.max()), abs(weights.min()))
            if weight_max > 0:
                weights = weights / weight_max
            return weights
        except:
            return None
    
    def _get_weight(self, layer_weights, neuron_idx_current, neuron_idx_next):
        """Get weight value for a connection."""
        if layer_weights is None:
            return 0.0
        if neuron_idx_current < layer_weights.shape[1] and neuron_idx_next < layer_weights.shape[0]:
            return layer_weights[neuron_idx_next, neuron_idx_current]
        return 0.0
    
    def _draw_connection(self, x1, y1, x2, y2, weight):
        """Draw a connection line between neurons."""
        weight_abs = abs(weight)
        if weight_abs < 0.1:
            glColor4f(0.2, 0.2, 0.2, 0.1)  # Very faint gray
        elif weight > 0:
            glColor4f(0.0, 0.6, 0.0, min(0.6, weight_abs * 0.8))  # Green for positive
        else:
            glColor4f(0.6, 0.0, 0.0, min(0.6, weight_abs * 0.8))  # Red for negative
        
        glBegin(GL_LINES)
        glVertex2f(x1, y1)
        glVertex2f(x2, y2)
        glEnd()
    
    def _render_layer_label(self, layer_idx, layer_x, y, height):
        """Render a layer label."""
        layer_names = ["Input", "Hidden1", "Hidden2", "Hidden3", "Output"]
        layer_name = layer_names[layer_idx] if layer_idx < len(layer_names) else f"Layer{layer_idx}"
        try:
            label = pyglet.text.Label(
                layer_name,
                font_name='Courier New',
                font_size=8,
                x=int(layer_x),
                y=int(y + height - 25),
                anchor_x='center',
                anchor_y='top',
                color=(255, 255, 255, 255)
            )
            label.draw()
        except:
            pass
    
    def _draw_neuron(self, x, y, radius, layer_idx, num_layers):
        """Draw a single neuron circle."""
        # Neuron color based on layer
        if layer_idx == 0:
            glColor3f(0.3, 0.7, 1.0)  # Blue for input
        elif layer_idx == num_layers - 1:
            glColor3f(1.0, 0.8, 0.3)  # Yellow for output
        else:
            glColor3f(0.5, 0.9, 0.5)  # Green for hidden
        
        # Draw neuron circle
        num_segments = 12
        glBegin(GL_TRIANGLE_FAN)
        glVertex2f(x, y)
        for i in range(num_segments + 1):
            angle = (i / num_segments) * 2 * math.pi
            glVertex2f(
                x + radius * math.cos(angle),
                y + radius * math.sin(angle)
            )
        glEnd()
        
        # Neuron border
        glColor3f(1.0, 1.0, 1.0)
        glLineWidth(1.0)
        glBegin(GL_LINE_LOOP)
        for i in range(num_segments):
            angle = (i / num_segments) * 2 * math.pi
            glVertex2f(
                x + radius * math.cos(angle),
                y + radius * math.sin(angle)
            )
        glEnd()
    
    def _render_info_text(self, npc: NPC, x, y, width, height, layer_sizes):
        """Render the neural network info text."""
        brain = npc.brain
        nn_info_height = 48  # 4 lines * 12px
        
        info_lines = [
            f"Input: {brain.input_size} features (showing {layer_sizes[0]})",
            f"Hidden: 64→64→32 (simplified visualization)",
            f"Output: 6 actions",
            f"Lines: Green=positive, Red=negative"
        ]
        
        # Position info text at bottom of NN visualization area
        # y is the bottom of the visualization area
        # Info text goes at the very bottom
        info_y = y  # Bottom of NN area
        
        try:
            for i, info_line in enumerate(info_lines):
                info_label = pyglet.text.Label(
                    info_line,
                    font_name='Courier New',
                    font_size=8,
                    x=int(x + 10),
                    y=int(info_y + (len(info_lines) - 1 - i) * 12 + 4),  # Position from bottom up
                    anchor_x='left',
                    anchor_y='bottom',
                    color=(180, 180, 180, 255),
                    width=int(width - 20)
                )
                info_label.draw()
        except:
            pass
