"""NPC detail panel rendering module."""

import pyglet
from pyglet.gl import *
import numpy as np
import math

from ..entities.npc import NPC


class DetailPanel:
    """Handles rendering of the NPC detail panel."""
    
    def __init__(self, window):
        """
        Initialize the detail panel renderer.
        
        Args:
            window: pyglet window instance
        """
        self.window = window
        self.scroll = 0
        self.max_scroll = 0
        
        # Layout constants
        self.panel_width = 420
        self.panel_padding = 20
        self.text_padding = 15
        
        # Bottom area layout (from bottom to top)
        # Bars: 3 bars with labels
        self.bar_height = 15
        self.bar_label_height = 12  # Height of label text
        self.bar_label_offset = 5  # Space between bar and label (label above bar)
        self.bar_spacing = 45  # Total space between bars (including label area)
        self.num_bars = 3
        
        # Calculate bar area height more precisely
        # Each bar section: bar (15px) + label above (12px + 5px gap = 17px) = 32px
        # Spacing between bars: bar_spacing (45px) - bar_height (15px) = 30px additional space
        # Total: 3 bars * 32px + 2 gaps * 30px = 96 + 60 = 156px
        # Plus bottom padding: 10px
        self.bar_area_height = (
            self.num_bars * (self.bar_height + self.bar_label_offset + self.bar_label_height) +
            (self.num_bars - 1) * (self.bar_spacing - self.bar_height) +
            10  # bottom padding
        )
        
        # Neural network visualization area
        self.nn_area_height = 180
        self.nn_title_height = 30  # Space for "Neural Network Architecture:" title
        self.nn_info_height = 48  # 4 lines * 12px for info text at bottom
        self.gap_between_bars_and_nn = 25  # Gap between health label and NN info
        
        # Total bottom area reserved (from bottom of panel)
        # NN visualization height includes the visualization area (180px) + title (30px)
        # Info text (48px) is at the bottom of the visualization area
        self.total_bottom_area = (
            self.bar_area_height +
            self.gap_between_bars_and_nn +
            self.nn_info_height +
            self.nn_area_height +
            self.nn_title_height
        )
    
    def render(self, npc: NPC):
        """
        Render the detail panel for the selected NPC.
        
        Args:
            npc: NPC instance to render details for
        """
        if not npc or not npc.is_alive:
            return
        
        # Panel dimensions
        panel_height = self.window.height - 40
        panel_x = self.window.width - self.panel_width - self.panel_padding
        panel_y = self.panel_padding
        
        # Setup 2D rendering
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, self.window.width, 0, self.window.height, -1, 1)
        
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()
        
        glDisable(GL_LIGHTING)
        glDisable(GL_DEPTH_TEST)
        
        # Render panel background
        self._render_background(panel_x, panel_y, self.panel_width, panel_height)
        
        # Render bottom area (bars + NN visualization) - fixed position at bottom
        self._render_bottom_area(npc, panel_x, panel_y)
        
        # Render scrollable text content (above bottom area)
        self._render_text_content(npc, panel_x, panel_y, panel_height)
        
        # Restore OpenGL state
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        
        glPopMatrix()
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
    
    def _render_background(self, x, y, width, height):
        """Render the panel background."""
        # Background
        glColor4f(0.1, 0.1, 0.15, 0.9)
        glBegin(GL_QUADS)
        glVertex2f(x, y)
        glVertex2f(x + width, y)
        glVertex2f(x + width, y + height)
        glVertex2f(x, y + height)
        glEnd()
        
        # Border
        glColor4f(0.3, 0.6, 1.0, 1.0)
        glLineWidth(2.0)
        glBegin(GL_LINE_LOOP)
        glVertex2f(x, y)
        glVertex2f(x + width, y)
        glVertex2f(x + width, y + height)
        glVertex2f(x, y + height)
        glEnd()
    
    def _render_text_content(self, npc: NPC, panel_x, panel_y, panel_height):
        """Render the scrollable text content."""
        # Prepare text lines
        lines = self._build_text_lines(npc)
        
        # Calculate text area (top area, excluding bottom fixed area)
        text_start_y = panel_y + panel_height - 30
        text_end_y = panel_y + self.total_bottom_area + 15  # Reduced gap - text closer to NN diagram
        
        # Calculate total content height
        total_text_height = sum(16 if line.strip() else 8 for line in lines)
        
        # Calculate scrollable area
        scrollable_area_height = text_start_y - text_end_y
        self.max_scroll = max(0, total_text_height - scrollable_area_height)
        
        # Clamp scroll position
        self.scroll = max(0, min(self.max_scroll, self.scroll))
        
        # Render scroll indicator if content is scrollable
        if self.max_scroll > 0:
            self._render_scroll_indicator(panel_x, panel_y, panel_height)
        
        # Render text lines with scroll
        text_y = text_start_y + self.scroll
        for line in lines:
            if text_y < text_end_y:
                break
            if text_y > text_start_y + 30:
                text_y -= 16 if line.strip() else 8
                continue
            
            if line.strip():
                self._render_text_line(line, panel_x + self.text_padding, text_y, self.panel_width - 2 * self.text_padding)
                text_y -= 16
            else:
                text_y -= 8
    
    def _build_text_lines(self, npc: NPC):
        """Build the list of text lines to display."""
        state_descriptions = {
            "wandering": "Exploring the world randomly",
            "seeking_food": "Looking for food sources",
            "eating": "Consuming fruit from a tree",
            "resting": "Resting to restore stamina",
            "seeking_shelter": "Looking for shelter (nighttime)",
            "in_shelter": "Safe inside a house"
        }
        
        state_desc = state_descriptions.get(npc.state, npc.state)
        
        health_pct = (npc.health / npc.max_health) * 100
        health_status = "Excellent" if health_pct > 80 else "Good" if health_pct > 50 else "Fair" if health_pct > 25 else "Critical"
        
        hunger_pct = (npc.hunger / npc.max_hunger) * 100
        hunger_status = "Well Fed" if hunger_pct > 80 else "Satisfied" if hunger_pct > 50 else "Hungry" if hunger_pct > 25 else "Starving"
        
        stamina_pct = (npc.stamina / npc.max_stamina) * 100 if npc.max_stamina > 0 else 0
        stamina_status = "Energetic" if stamina_pct > 80 else "Active" if stamina_pct > 50 else "Tired" if stamina_pct > 25 else "Exhausted"
        
        return [
            f"=== NPC DETAILS ===",
            f"",
            f"Name: {npc.name}",
            f"",
            f"--- Current Status ---",
            f"Status: {npc.state.replace('_', ' ').title()}",
            f"Description: {state_desc}",
            f"",
            f"--- Age & Lifecycle ---",
            f"Age: {npc.age:.1f}s / {npc.lifespan:.1f}s",
            f"Age Stage: {npc.age_stage.title()}",
            f"Adult Age Threshold: {npc.adult_age:.1f}s",
            f"Can Reproduce: {'Yes' if npc.can_reproduce else 'No'}",
            f"Reproduction Cooldown: {npc.reproduction_cooldown:.1f}s" if npc.reproduction_cooldown > 0 else "Reproduction Cooldown: Ready",
            f"",
            f"--- Health Statistics ---",
            f"Current Health: {npc.health:.1f} / {npc.max_health:.1f} ({health_pct:.1f}%)",
            f"Health Status: {health_status}",
            f"Note: Health decreases gradually over time.",
            f"Low health indicates poor condition.",
            f"",
            f"--- Hunger Statistics ---",
            f"Current Hunger: {npc.hunger:.1f} / {npc.max_hunger:.1f} ({hunger_pct:.1f}%)",
            f"Hunger Status: {hunger_status}",
            f"Note: Hunger decreases over time.",
            f"Starvation (hunger = 0) causes health loss.",
            f"Eating fruit restores hunger.",
            f"",
            f"--- Stamina Statistics ---",
            f"Current Stamina: {npc.stamina:.1f} / {npc.max_stamina:.1f} ({stamina_pct:.1f}%)",
            f"Stamina Status: {stamina_status}",
            f"Note: Stamina decreases during movement.",
            f"Resting restores stamina over time.",
            f"Low stamina slows movement speed.",
            f"",
            f"--- Physical Attributes ---",
            f"Movement Speed: {npc.speed:.2f} units/second",
            f"Body Size: {npc.size:.2f}",
            f"",
            f"--- Genetic Traits ---",
            f"Vision Range: {npc.genome.get('vision_range', 0):.1f} units",
            f"Food Preference: {npc.genome.get('food_preference', 0):.2f}",
            f"",
            f"--- Activity Statistics ---",
            f"Fruit Collected: {npc.fruit_collected}",
            f"Current Position: ({npc.x:.1f}, {npc.y:.1f}, {npc.z:.1f})",
            f"",
            f"--- Current Actions ---",
            f"In House: {'Yes' if npc.current_house else 'No'}",
            f"Target Position: ({npc.target_x:.1f}, {npc.target_z:.1f})" if npc.target_x else "Target Position: None",
            f"",
            f"--- Controls ---",
            f"Scroll mouse wheel to scroll this panel",
            f"Click elsewhere to deselect"
        ]
    
    def _render_text_line(self, text, x, y, max_width):
        """Render a single line of text."""
        try:
            label = pyglet.text.Label(
                text,
                font_name='Courier New',
                font_size=11,
                x=int(x),
                y=int(y),
                anchor_x='left',
                anchor_y='bottom',
                color=(255, 255, 255, 255),
                multiline=False,
                width=int(max_width)
            )
            label.draw()
        except:
            try:
                label = pyglet.text.Label(
                    text,
                    font_size=11,
                    x=int(x),
                    y=int(y),
                    anchor_x='left',
                    anchor_y='bottom',
                    color=(255, 255, 255, 255),
                    width=int(max_width)
                )
                label.draw()
            except:
                pass
    
    def _render_scroll_indicator(self, panel_x, panel_y, panel_height):
        """Render the scroll position indicator."""
        try:
            indicator_label = pyglet.text.Label(
                f"Scroll: {self.scroll:.0f}/{self.max_scroll:.0f}",
                font_name='Courier New',
                font_size=9,
                x=int(panel_x + self.panel_width - 10),
                y=int(panel_y + panel_height - 15),
                anchor_x='right',
                anchor_y='top',
                color=(150, 150, 150, 255)
            )
            indicator_label.draw()
        except:
            pass
    
    def _render_bottom_area(self, npc: NPC, panel_x, panel_y):
        """Render the fixed bottom area (bars + NN visualization)."""
        # Render status bars at the bottom
        health_label_top = self._render_status_bars(npc, panel_x, panel_y)
        
        # Render neural network visualization above bars
        self._render_neural_network(npc, panel_x, panel_y, health_label_top)
    
    def _render_status_bars(self, npc: NPC, panel_x, panel_y):
        """Render the status bars (Health, Hunger, Stamina)."""
        bar_x = panel_x + self.text_padding
        bar_width = self.panel_width - 2 * self.text_padding
        
        # Calculate bar positions from bottom up
        # Each bar section: bar (15px) with label above (12px + 5px gap = 17px total)
        # bar_spacing = 45px is the distance from bottom of one bar to bottom of next bar
        
        # Bottom padding to keep bars off the very bottom edge
        bottom_padding = 10
        
        # Bottom bar (Stamina)
        stamina_bar_y = panel_y + bottom_padding
        stamina_label_y = stamina_bar_y + self.bar_height + self.bar_label_offset
        
        # Middle bar (Hunger) - bar_spacing pixels above previous bar bottom
        hunger_bar_y = stamina_bar_y + self.bar_spacing
        hunger_label_y = hunger_bar_y + self.bar_height + self.bar_label_offset
        
        # Top bar (Health) - bar_spacing pixels above previous bar bottom
        health_bar_y = hunger_bar_y + self.bar_spacing
        health_label_y = health_bar_y + self.bar_height + self.bar_label_offset
        
        # Render from bottom to top to avoid overlap
        # Stamina bar
        self._render_bar("Stamina:", npc.stamina, npc.max_stamina, bar_x, stamina_bar_y, bar_width, stamina_label_y, (0.3, 0.3, 0.5), (0.2, 0.4, 1.0))
        
        # Hunger bar
        self._render_bar("Hunger:", npc.hunger, npc.max_hunger, bar_x, hunger_bar_y, bar_width, hunger_label_y, (0.5, 0.3, 0.0), (1.0, 0.8, 0.0))
        
        # Health bar
        self._render_bar("Health:", npc.health, npc.max_health, bar_x, health_bar_y, bar_width, health_label_y, (0.8, 0.0, 0.0), (0.0, 0.8, 0.0))
        
        # Return the top of the health label for NN positioning
        return health_label_y + self.bar_label_height
    
    def _render_bar(self, label_text, value, max_value, x, y, width, label_y, bg_color, fg_color):
        """Render a single status bar."""
        # Label above the bar
        try:
            label = pyglet.text.Label(
                label_text,
                font_name='Courier New',
                font_size=10,
                x=int(x),
                y=int(label_y),
                anchor_x='left',
                anchor_y='bottom',
                color=(255, 255, 255, 255)
            )
            label.draw()
        except:
            pass
        
        # Background bar
        glColor3f(*bg_color)
        glBegin(GL_QUADS)
        glVertex2f(x, y)
        glVertex2f(x + width, y)
        glVertex2f(x + width, y + self.bar_height)
        glVertex2f(x, y + self.bar_height)
        glEnd()
        
        # Foreground bar (value)
        if max_value > 0:
            fill_width = width * (value / max_value)
            glColor3f(*fg_color)
            glBegin(GL_QUADS)
            glVertex2f(x, y)
            glVertex2f(x + fill_width, y)
            glVertex2f(x + fill_width, y + self.bar_height)
            glVertex2f(x, y + self.bar_height)
            glEnd()
    
    def _render_neural_network(self, npc: NPC, panel_x, panel_y, health_label_top):
        """Render the neural network visualization."""
        if not hasattr(npc, 'brain'):
            return
        
        # NN info text goes gap_between_bars_and_nn above health label top
        # This is where the info text area starts (bottom of info text)
        nn_info_bottom = health_label_top + self.gap_between_bars_and_nn
        
        # NN visualization area: info at bottom (48px), network in middle, title at top (30px)
        # Total height is nn_area_height (180px)
        # So the visualization area starts at nn_info_bottom and goes up nn_area_height pixels
        nn_y_position = nn_info_bottom  # Bottom of entire NN visualization area
        
        # Render NN visualization (height includes title + network + info)
        from .neural_network_viz import NeuralNetworkVisualization
        viz = NeuralNetworkVisualization()
        viz.render(npc, panel_x, nn_y_position, self.panel_width, self.nn_area_height)
    
    def handle_scroll(self, x, y, scroll_x, scroll_y):
        """
        Handle mouse scroll events for the detail panel.
        
        Args:
            x, y: Mouse position
            scroll_x, scroll_y: Scroll delta
        
        Returns:
            True if scroll was handled, False otherwise
        """
        panel_x = self.window.width - self.panel_width - self.panel_padding
        if x >= panel_x:
            self.scroll -= scroll_y * 30
            self.scroll = max(0, min(self.max_scroll, self.scroll))
            return True
        return False
    
    def is_point_in_panel(self, x, y):
        """Check if a point is inside the detail panel."""
        panel_x = self.window.width - self.panel_width - self.panel_padding
        panel_height = self.window.height - 40
        panel_y = self.panel_padding
        
        return (panel_x <= x <= panel_x + self.panel_width and
                panel_y <= y <= panel_y + panel_height)