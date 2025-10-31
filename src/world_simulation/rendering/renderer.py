"""3D rendering implementation using pyglet."""

import pyglet
from pyglet.gl import *
from pyglet.gl.glu import gluPerspective
import numpy as np
import math

from ..world.world import World
from ..entities.npc import NPC
from ..trees.tree import FruitTree


class Renderer:
    """3D renderer for the world simulation."""
    
    def __init__(self, world: World, width: int = 1280, height: int = 720):
        """
        Initialize the renderer.
        
        Args:
            world: World instance to render
            width: Window width
            height: Window height
        """
        self.world = world
        self.window = pyglet.window.Window(width, height, caption="World Simulation")
        self.window.set_exclusive_mouse(True)
        
        # Camera
        self.camera_x = 0.0
        self.camera_y = 50.0
        self.camera_z = 0.0
        self.camera_yaw = 0.0
        self.camera_pitch = -45.0
        
        # Controls
        self.keys = pyglet.window.key.KeyStateHandler()
        self.window.push_handlers(self.keys)
        
        # Setup OpenGL
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        # Lighting
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glLightfv(GL_LIGHT0, GL_POSITION, (GLfloat * 4)(0.0, 50.0, 0.0, 1.0))
        glLightfv(GL_LIGHT0, GL_AMBIENT, (GLfloat * 4)(0.3, 0.3, 0.3, 1.0))
        glLightfv(GL_LIGHT0, GL_DIFFUSE, (GLfloat * 4)(0.8, 0.8, 0.8, 1.0))
        
        # Event handlers
        @self.window.event
        def on_draw():
            self.render()
        
        @self.window.event
        def on_mouse_motion(x, y, dx, dy):
            self.camera_yaw += dx * 0.1
            self.camera_pitch = np.clip(self.camera_pitch - dy * 0.1, -90.0, 90.0)
    
    def update(self, delta_time: float):
        """Update camera based on input."""
        move_speed = 10.0 * delta_time
        
        if self.keys[pyglet.window.key.W]:
            self.camera_x += np.cos(np.radians(self.camera_yaw)) * move_speed
            self.camera_z += np.sin(np.radians(self.camera_yaw)) * move_speed
        if self.keys[pyglet.window.key.S]:
            self.camera_x -= np.cos(np.radians(self.camera_yaw)) * move_speed
            self.camera_z -= np.sin(np.radians(self.camera_yaw)) * move_speed
        if self.keys[pyglet.window.key.A]:
            self.camera_x += np.cos(np.radians(self.camera_yaw - 90)) * move_speed
            self.camera_z += np.sin(np.radians(self.camera_yaw - 90)) * move_speed
        if self.keys[pyglet.window.key.D]:
            self.camera_x += np.cos(np.radians(self.camera_yaw + 90)) * move_speed
            self.camera_z += np.sin(np.radians(self.camera_yaw + 90)) * move_speed
        if self.keys[pyglet.window.key.SPACE]:
            self.camera_y += move_speed
        if self.keys[pyglet.window.key.LSHIFT]:
            self.camera_y -= move_speed
        
        self.camera_y = max(5.0, self.camera_y)
    
    def render(self):
        """Render the world."""
        self.window.clear()
        
        # Setup camera
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(70.0, self.window.width / self.window.height, 0.1, 500.0)
        
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        
        # Rotate camera
        glRotatef(self.camera_pitch, 1.0, 0.0, 0.0)
        glRotatef(self.camera_yaw, 0.0, 1.0, 0.0)
        glTranslatef(-self.camera_x, -self.camera_y, -self.camera_z)
        
        # Render terrain
        self._render_terrain()
        
        # Render trees
        for tree in self.world.trees:
            self._render_tree(tree)
        
        # Render NPCs
        for npc in self.world.entities:
            self._render_npc(npc)
    
    def _render_terrain(self):
        """Render the terrain mesh."""
        # Render visible chunks
        glColor3f(0.2, 0.6, 0.2)
        
        # Simple ground plane for now
        glBegin(GL_QUADS)
        glVertex3f(-100, 0, -100)
        glVertex3f(100, 0, -100)
        glVertex3f(100, 0, 100)
        glVertex3f(-100, 0, 100)
        glEnd()
    
    def _render_tree(self, tree: FruitTree):
        """Render a fruit tree."""
        if not tree.is_alive:
            return
        
        glPushMatrix()
        glTranslatef(tree.x, tree.y, tree.z)
        
        # Trunk
        glColor3f(0.4, 0.2, 0.1)
        glBegin(GL_QUADS)
        trunk_height = 2.0 * tree.growth_stage
        glVertex3f(-0.1, 0, -0.1)
        glVertex3f(0.1, 0, -0.1)
        glVertex3f(0.1, trunk_height, -0.1)
        glVertex3f(-0.1, trunk_height, -0.1)
        glEnd()
        
        # Leaves/Fruit
        if tree.growth_stage >= 0.5:
            glColor3f(0.1, 0.5, 0.1)
            glPushMatrix()
            glTranslatef(0, trunk_height + 1.0, 0)
            self._draw_sphere(1.0 * tree.growth_stage, 16)
            glPopMatrix()
            
            # Fruit
            if tree.get_ripe_fruit_count() > 0:
                glColor3f(1.0, 0.0, 0.0)
                glPushMatrix()
                glTranslatef(0, trunk_height + 1.5, 0)
                self._draw_sphere(0.1, 8)
                glPopMatrix()
        
        glPopMatrix()
    
    def _render_npc(self, npc: NPC):
        """Render an NPC."""
        if not npc.is_alive:
            return
        
        glPushMatrix()
        glTranslatef(npc.x, npc.y + 0.5, npc.z)
        
        # Color based on health
        health_ratio = npc.health / npc.max_health
        glColor3f(1.0 - health_ratio, health_ratio, 0.0)
        
        # Simple cube representation
        size = 0.3 * npc.size
        glBegin(GL_QUADS)
        # Front face
        glVertex3f(-size, -size, size)
        glVertex3f(size, -size, size)
        glVertex3f(size, size, size)
        glVertex3f(-size, size, size)
        # Back face
        glVertex3f(-size, -size, -size)
        glVertex3f(-size, size, -size)
        glVertex3f(size, size, -size)
        glVertex3f(size, -size, -size)
        # Top face
        glVertex3f(-size, size, -size)
        glVertex3f(-size, size, size)
        glVertex3f(size, size, size)
        glVertex3f(size, size, -size)
        # Bottom face
        glVertex3f(-size, -size, -size)
        glVertex3f(size, -size, -size)
        glVertex3f(size, -size, size)
        glVertex3f(-size, -size, size)
        glEnd()
        
        glPopMatrix()
    
    def _draw_sphere(self, radius: float, segments: int):
        """Draw a simple sphere using triangle strips."""
        for i in range(segments):
            angle1 = i * 2 * math.pi / segments
            angle2 = (i + 1) * 2 * math.pi / segments
            
            glBegin(GL_TRIANGLE_STRIP)
            for j in range(segments // 2 + 1):
                theta = j * math.pi / (segments // 2)
                x1 = radius * math.sin(theta) * math.cos(angle1)
                y1 = radius * math.cos(theta)
                z1 = radius * math.sin(theta) * math.sin(angle1)
                x2 = radius * math.sin(theta) * math.cos(angle2)
                y2 = radius * math.cos(theta)
                z2 = radius * math.sin(theta) * math.sin(angle2)
                glVertex3f(x1, y1, z1)
                glVertex3f(x2, y2, z2)
            glEnd()
