"""Camera management for 3D rendering."""

import numpy as np
import pyglet
from pyglet.gl import *
from pyglet.gl.glu import gluPerspective


class Camera:
    """Manages camera position, rotation, and projection."""
    
    def __init__(self, width: int = 1280, height: int = 720):
        """
        Initialize the camera.
        
        Args:
            width: Window width
            height: Window height
        """
        # Camera position
        self.x = 0.0
        self.y = 50.0
        self.z = 0.0
        self.yaw = 0.0
        self.pitch = -45.0
        
        # Camera frustum
        self.fov = 70.0
        self.near_plane = 0.5  # Increased from 0.1 to prevent clipping artifacts
        self.far_plane = 500.0
        
        # Window dimensions
        self.width = width
        self.height = height
    
    def setup_projection(self):
        """Setup the projection matrix."""
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(self.fov, self.width / self.height, self.near_plane, self.far_plane)
    
    def setup_view(self):
        """Setup the view matrix."""
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        
        # Rotate camera
        glRotatef(self.pitch, 1.0, 0.0, 0.0)
        glRotatef(self.yaw, 0.0, 1.0, 0.0)
        glTranslatef(-self.x, -self.y, -self.z)
    
    def get_forward_vector(self):
        """Get the forward direction vector."""
        yaw_rad = np.radians(self.yaw)
        pitch_rad = np.radians(self.pitch)
        
        forward_x = np.sin(yaw_rad) * np.cos(pitch_rad)
        forward_y = -np.sin(pitch_rad)
        forward_z = -np.cos(yaw_rad) * np.cos(pitch_rad)
        
        return forward_x, forward_y, forward_z
    
    def get_right_vector(self):
        """Get the right direction vector."""
        yaw_rad = np.radians(self.yaw)
        right_x = np.cos(yaw_rad)
        right_z = np.sin(yaw_rad)
        return right_x, right_z
    
    def update_position(self, delta_time: float, keys: dict, keys_handler):
        """Update camera position based on keyboard input."""
        move_speed = 20.0 * delta_time
        
        # Get movement vectors
        forward_x, _, forward_z = self.get_forward_vector()
        right_x, right_z = self.get_right_vector()
        
        # Check keys
        w_pressed = keys.get(pyglet.window.key.W, False) or keys_handler[pyglet.window.key.W]
        s_pressed = keys.get(pyglet.window.key.S, False) or keys_handler[pyglet.window.key.S]
        a_pressed = keys.get(pyglet.window.key.A, False) or keys_handler[pyglet.window.key.A]
        d_pressed = keys.get(pyglet.window.key.D, False) or keys_handler[pyglet.window.key.D]
        space_pressed = keys.get(pyglet.window.key.SPACE, False) or keys_handler[pyglet.window.key.SPACE]
        shift_pressed = keys.get(pyglet.window.key.LSHIFT, False) or keys_handler[pyglet.window.key.LSHIFT]
        
        if w_pressed:
            self.x += forward_x * move_speed
            self.z += forward_z * move_speed
        if s_pressed:
            self.x -= forward_x * move_speed
            self.z -= forward_z * move_speed
        if a_pressed:
            self.x -= right_x * move_speed
            self.z -= right_z * move_speed
        if d_pressed:
            self.x += right_x * move_speed
            self.z += right_z * move_speed
        if space_pressed:
            self.y += move_speed
        if shift_pressed:
            self.y -= move_speed
        
        # Minimum height will be enforced by renderer based on terrain
        # Keep absolute minimum of 1.0 to prevent going below zero
        self.y = max(1.0, self.y)
    
    def rotate(self, dx: float, dy: float):
        """Rotate camera based on mouse movement."""
        self.yaw += dx * 0.1
        self.pitch = np.clip(self.pitch - dy * 0.1, -90.0, 90.0)
    
    def zoom(self, scroll_y: float):
        """Zoom camera."""
        self.y += scroll_y * 2.0
        # Minimum height will be enforced by renderer based on terrain
    
    def resize(self, width: int, height: int):
        """Update window dimensions."""
        self.width = width
        self.height = height
