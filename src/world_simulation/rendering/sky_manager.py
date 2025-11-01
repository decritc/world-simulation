"""Sky and lighting management for day/night cycle."""

from pyglet.gl import *


class SkyManager:
    """Manages sky color and lighting based on time of day."""
    
    def __init__(self, day_length: float = 120.0):
        """
        Initialize sky manager.
        
        Args:
            day_length: Length of a day in seconds
        """
        self.day_length = day_length
    
    def calculate_sky_color(self, day_time: float) -> tuple:
        """
        Calculate sky color based on time of day.
        
        Args:
            day_time: Current time within the day
            
        Returns:
            Tuple of (r, g, b) sky color components
        """
        hour = (day_time / self.day_length) * 24.0
        
        if 6.0 <= hour < 8.0:  # Dawn
            sky_r = 0.3 + 0.2 * ((hour - 6.0) / 2.0)
            sky_g = 0.4 + 0.3 * ((hour - 6.0) / 2.0)
            sky_b = 0.5 + 0.5 * ((hour - 6.0) / 2.0)
        elif 8.0 <= hour < 18.0:  # Day
            sky_r, sky_g, sky_b = 0.5, 0.7, 1.0
        elif 18.0 <= hour < 20.0:  # Dusk
            t = (hour - 18.0) / 2.0
            sky_r = 0.5 - 0.47 * t
            sky_g = 0.7 - 0.67 * t
            sky_b = 1.0 - 0.88 * t
        else:  # Night
            sky_r, sky_g, sky_b = 0.03, 0.03, 0.12
        
        return sky_r, sky_g, sky_b
    
    def setup_lighting(self, light_intensity: float, is_night: bool):
        """
        Setup lighting based on time of day.
        
        Args:
            light_intensity: Current light intensity (0.0-1.0)
            is_night: Whether it's currently night time
        """
        ambient_intensity = 0.2 + (light_intensity - 0.3) * 0.5
        
        # Sun light intensity
        glLightfv(GL_LIGHT0, GL_POSITION, (GLfloat * 4)(30.0, 100.0, 30.0, 0.0))
        
        if is_night:
            # Cool blue-white moonlight at night
            glLightfv(GL_LIGHT0, GL_AMBIENT, (GLfloat * 4)(0.0, 0.12 * ambient_intensity, 0.3 * ambient_intensity, 1.0))
            glLightfv(GL_LIGHT0, GL_DIFFUSE, (GLfloat * 4)(0.0, 0.3 * light_intensity, 0.7 * light_intensity, 1.0))
            glLightfv(GL_LIGHT0, GL_SPECULAR, (GLfloat * 4)(0.0, 0.2 * light_intensity, 0.6 * light_intensity, 1.0))
        else:
            # Warm sunlight during day
            glLightfv(GL_LIGHT0, GL_AMBIENT, (GLfloat * 4)(0.4 * ambient_intensity, 0.4 * ambient_intensity, 0.5 * ambient_intensity, 1.0))
            glLightfv(GL_LIGHT0, GL_DIFFUSE, (GLfloat * 4)(1.0 * light_intensity, 0.95 * light_intensity, 0.8 * light_intensity, 1.0))
            glLightfv(GL_LIGHT0, GL_SPECULAR, (GLfloat * 4)(1.0, 1.0, 1.0, 1.0))
        
        # Ambient light
        if is_night:
            glLightfv(GL_LIGHT1, GL_AMBIENT, (GLfloat * 4)(0.0, 0.1 * ambient_intensity, 0.25 * ambient_intensity, 1.0))
            glLightfv(GL_LIGHT1, GL_DIFFUSE, (GLfloat * 4)(0.0, 0.15 * ambient_intensity, 0.3 * ambient_intensity, 1.0))
            glLightModelfv(GL_LIGHT_MODEL_AMBIENT, (GLfloat * 4)(0.0, 0.015 * ambient_intensity, 0.05 * ambient_intensity, 1.0))
        else:
            glLightfv(GL_LIGHT1, GL_AMBIENT, (GLfloat * 4)(0.3 * ambient_intensity, 0.4 * ambient_intensity, 0.5 * ambient_intensity, 1.0))
            glLightfv(GL_LIGHT1, GL_DIFFUSE, (GLfloat * 4)(0.3 * ambient_intensity, 0.4 * ambient_intensity, 0.5 * ambient_intensity, 1.0))
            glLightModelfv(GL_LIGHT_MODEL_AMBIENT, (GLfloat * 4)(0.2 * ambient_intensity, 0.2 * ambient_intensity, 0.25 * ambient_intensity, 1.0))
