"""Historian log viewer panel module."""

import pyglet
from pyglet.gl import *
from typing import List


class HistorianLogViewer:
    """Handles rendering of the historian log viewer panel."""
    
    def __init__(self, window):
        """
        Initialize the log viewer panel.
        
        Args:
            window: pyglet window instance
        """
        self.window = window
        self.scroll = 0
        self.max_scroll = 0
        self.visible = False
        
        # Layout constants
        self.panel_width = 800
        self.panel_height = 600
        self.panel_padding = 20
        self.text_padding = 15
        self.line_height = 14
        
        # Position (centered)
        self.panel_x = 0
        self.panel_y = 0
        
        # Title
        self.title_label = None
        
    def set_visible(self, visible: bool):
        """Set visibility of the log viewer."""
        self.visible = visible
        if visible:
            self.scroll = 0  # Reset scroll when showing
    
    def toggle(self):
        """Toggle visibility of the log viewer."""
        self.set_visible(not self.visible)
    
    def handle_scroll(self, x: int, y: int, scroll_x: float, scroll_y: float):
        """
        Handle mouse scroll events over the log viewer.
        
        Args:
            x: Mouse X position
            y: Mouse Y position
            scroll_x: Scroll delta X
            scroll_y: Scroll delta Y
        """
        if not self.visible:
            return
        
        if self.is_point_in_panel(x, y):
            # Scroll up/down
            scroll_delta = int(scroll_y * 3)  # Scroll 3 lines at a time
            self.scroll = max(0, min(self.scroll - scroll_delta, self.max_scroll))
    
    def is_point_in_panel(self, x: int, y: int) -> bool:
        """
        Check if a point is inside the log viewer panel.
        
        Args:
            x: X coordinate
            y: Y coordinate (pyglet Y, 0 at bottom)
            
        Returns:
            True if point is in panel
        """
        if not self.visible:
            return False
        
        # Convert pyglet Y to screen Y (pyglet has Y=0 at bottom)
        screen_y = self.window.height - y
        
        return (self.panel_x <= x <= self.panel_x + self.panel_width and
                self.panel_y <= screen_y <= self.panel_y + self.panel_height)
    
    def update_layout(self, width: int, height: int):
        """Update panel layout based on window size."""
        # Center the panel
        self.panel_x = (width - self.panel_width) // 2
        self.panel_y = (height - self.panel_height) // 2
    
    def render(self, log_lines: List[str]):
        """
        Render the log viewer panel.
        
        Args:
            log_lines: List of log lines to display
        """
        if not self.visible:
            return
        
        width = self.window.width
        height = self.window.height
        
        # Update layout
        self.update_layout(width, height)
        
        # Calculate visible lines
        content_height = self.panel_height - (self.panel_padding * 2) - 40  # Title area
        max_visible_lines = int(content_height / self.line_height)
        
        # Calculate total content height
        total_lines = len(log_lines)
        total_content_height = total_lines * self.line_height
        
        # Update max scroll
        self.max_scroll = max(0, total_content_height - content_height)
        
        # Calculate which lines to render
        start_line = int(self.scroll / self.line_height)
        end_line = min(start_line + max_visible_lines + 1, total_lines)
        
        # Draw panel background
        glPushMatrix()
        glLoadIdentity()
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, width, 0, height, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glDisable(GL_DEPTH_TEST)
        glDisable(GL_LIGHTING)
        
        # Panel background (semi-transparent dark)
        bg_color = (0.1, 0.1, 0.15, 0.95)
        glColor4f(*bg_color)
        glBegin(GL_QUADS)
        glVertex2f(self.panel_x, self.panel_y)
        glVertex2f(self.panel_x + self.panel_width, self.panel_y)
        glVertex2f(self.panel_x + self.panel_width, self.panel_y + self.panel_height)
        glVertex2f(self.panel_x, self.panel_y + self.panel_height)
        glEnd()
        
        # Panel border
        glColor4f(0.3, 0.3, 0.4, 1.0)
        glLineWidth(2.0)
        glBegin(GL_LINE_LOOP)
        glVertex2f(self.panel_x, self.panel_y)
        glVertex2f(self.panel_x + self.panel_width, self.panel_y)
        glVertex2f(self.panel_x + self.panel_width, self.panel_y + self.panel_height)
        glVertex2f(self.panel_x, self.panel_y + self.panel_height)
        glEnd()
        
        # Title
        title_y = self.panel_y + self.panel_height - 30
        if not self.title_label:
            self.title_label = pyglet.text.Label(
                "Colony History Log",
                font_name='Arial',
                font_size=18,
                bold=True,
                color=(255, 255, 255, 255),
                x=self.panel_x + self.panel_padding,
                y=height - title_y,
                anchor_x='left',
                anchor_y='center',
                width=self.panel_width - (self.panel_padding * 2),
                multiline=False
            )
        else:
            self.title_label.x = self.panel_x + self.panel_padding
            self.title_label.y = height - title_y
        
        self.title_label.draw()
        
        # Close button hint
        close_label = pyglet.text.Label(
            "Press 'H' to close | Scroll to navigate",
            font_name='Arial',
            font_size=10,
            color=(200, 200, 200, 255),
            x=self.panel_x + self.panel_width - self.panel_padding,
            y=height - title_y,
            anchor_x='right',
            anchor_y='center',
            width=self.panel_width - (self.panel_padding * 2),
            multiline=False
        )
        close_label.draw()
        
        # Draw log lines
        content_start_y = self.panel_y + self.panel_height - 50
        content_x = self.panel_x + self.panel_padding
        
        for i, line in enumerate(log_lines[start_line:end_line]):
            line_y = content_start_y - (i * self.line_height) - (self.scroll % self.line_height)
            
            if line_y < self.panel_y + 20:  # Don't render below panel
                continue
            
            # Color code different event types
            color = (255, 255, 255, 255)  # Default white
            if "BIRTH" in line:
                color = (100, 255, 100, 255)  # Green
            elif "DEATH" in line:
                color = (255, 100, 100, 255)  # Red
            elif "REPRODUCTION" in line:
                color = (255, 200, 100, 255)  # Orange
            elif "MILESTONE" in line:
                color = (100, 200, 255, 255)  # Blue
            elif "ACHIEVEMENT" in line:
                color = (200, 100, 255, 255)  # Purple
            elif "GENERATION" in line or "COLONY SUMMARY" in line:
                color = (255, 255, 100, 255)  # Yellow
            elif line.startswith("="):
                color = (150, 150, 150, 255)  # Gray for separators
            
            label = pyglet.text.Label(
                line,
                font_name='Courier New',
                font_size=11,
                color=color,
                x=content_x,
                y=height - line_y,
                anchor_x='left',
                anchor_y='top',
                width=self.panel_width - (self.panel_padding * 2),
                multiline=True
            )
            label.draw()
        
        # Scrollbar
        if self.max_scroll > 0:
            scrollbar_width = 10
            scrollbar_x = self.panel_x + self.panel_width - scrollbar_width - 5
            scrollbar_height = content_height
            scrollbar_y = self.panel_y + 20
            
            # Background
            glColor4f(0.2, 0.2, 0.25, 1.0)
            glBegin(GL_QUADS)
            glVertex2f(scrollbar_x, scrollbar_y)
            glVertex2f(scrollbar_x + scrollbar_width, scrollbar_y)
            glVertex2f(scrollbar_x + scrollbar_width, scrollbar_y + scrollbar_height)
            glVertex2f(scrollbar_x, scrollbar_y + scrollbar_height)
            glEnd()
            
            # Thumb
            thumb_height = max(20, scrollbar_height * (content_height / total_content_height))
            thumb_position = (self.scroll / self.max_scroll) * (scrollbar_height - thumb_height)
            thumb_y = scrollbar_y + thumb_position
            
            glColor4f(0.4, 0.4, 0.5, 1.0)
            glBegin(GL_QUADS)
            glVertex2f(scrollbar_x, thumb_y)
            glVertex2f(scrollbar_x + scrollbar_width, thumb_y)
            glVertex2f(scrollbar_x + scrollbar_width, thumb_y + thumb_height)
            glVertex2f(scrollbar_x, thumb_y + thumb_height)
            glEnd()
        
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glPopMatrix()
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)

