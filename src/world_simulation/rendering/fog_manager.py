"""Fog management for atmospheric rendering."""

from pyglet.gl import *


class FogManager:
    """Manages fog rendering to hide LOD transitions and distant artifacts."""
    
    def __init__(self):
        """Initialize fog manager."""
        self.fog_start = 40.0
        self.fog_end = 180.0
        self.fog_mode = GL_LINEAR
        self.enabled = True
    
    def setup(self):
        """Setup fog parameters."""
        if self.enabled:
            glEnable(GL_FOG)
            glFogi(GL_FOG_MODE, self.fog_mode)
            glFogf(GL_FOG_START, self.fog_start)
            glFogf(GL_FOG_END, self.fog_end)
            glFogf(GL_FOG_DENSITY, 0.015)
            glHint(GL_FOG_HINT, GL_NICEST)
    
    def update_color(self, sky_r: float, sky_g: float, sky_b: float):
        """
        Update fog color to match sky color.
        
        Args:
            sky_r: Sky red component (0.0-1.0)
            sky_g: Sky green component (0.0-1.0)
            sky_b: Sky blue component (0.0-1.0)
        """
        if self.enabled:
            glEnable(GL_FOG)
            glFogfv(GL_FOG_COLOR, (GLfloat * 4)(sky_r, sky_g, sky_b, 1.0))
    
    def set_range(self, start: float, end: float):
        """
        Set fog range.
        
        Args:
            start: Distance where fog starts
            end: Distance where fog is fully opaque
        """
        self.fog_start = start
        self.fog_end = end
        if self.enabled:
            glFogf(GL_FOG_START, self.fog_start)
            glFogf(GL_FOG_END, self.fog_end)
    
    def enable(self):
        """Enable fog."""
        self.enabled = True
        glEnable(GL_FOG)
    
    def disable(self):
        """Disable fog."""
        self.enabled = False
        glDisable(GL_FOG)
