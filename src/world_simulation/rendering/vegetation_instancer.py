"""GPU instancing system for vegetation rendering."""

import numpy as np
from typing import List, Dict, Tuple
from pyglet.gl import *
import math

from ..world.vegetation import Vegetation


class VegetationInstancer:
    """Handles GPU instanced rendering of vegetation."""
    
    def __init__(self):
        """Initialize the vegetation instancer."""
        self.instances_by_type: Dict[str, List[Vegetation]] = {
            'bush': [],
            'grass': [],
            'flower': [],
            'rock': []
        }
    
    def prepare_instances(self, vegetation_list: List[Vegetation]):
        """
        Prepare vegetation instances for rendering, grouped by type.
        
        Args:
            vegetation_list: List of Vegetation objects to render
        """
        # Clear previous instances
        for veg_type in self.instances_by_type:
            self.instances_by_type[veg_type].clear()
        
        # Group by type
        for veg in vegetation_list:
            if veg.is_alive and veg.vegetation_type in self.instances_by_type:
                self.instances_by_type[veg.vegetation_type].append(veg)
    
    def render_bush_batch(self, instances: List[Vegetation], is_night: bool, max_height: float):
        """Render all bush instances in a batch."""
        if not instances:
            return
        
        # Set material once for all bushes
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        
        if is_night:
            glColor3f(0.0, 0.08, 0.12)
        else:
            glColor3f(0.2, 0.4, 0.1)
        
        # Render all bushes using immediate mode but batched
        segments = 8
        
        # Render all instances
        for veg in instances:
            glPushMatrix()
            glTranslatef(veg.x, veg.y * max_height, veg.z)
            
            # Render sphere
            for i in range(segments):
                lat1 = math.pi * (-0.5 + i / segments)
                lat2 = math.pi * (-0.5 + (i + 1) / segments)
                
                glBegin(GL_QUAD_STRIP)
                for j in range(segments + 1):
                    lng = 2 * math.pi * j / segments
                    x1 = veg.size * math.cos(lat1) * math.cos(lng)
                    y1 = veg.size * math.sin(lat1)
                    z1 = veg.size * math.cos(lat1) * math.sin(lng)
                    x2 = veg.size * math.cos(lat2) * math.cos(lng)
                    y2 = veg.size * math.sin(lat2)
                    z2 = veg.size * math.cos(lat2) * math.sin(lng)
                    glVertex3f(x1, y1, z1)
                    glVertex3f(x2, y2, z2)
                glEnd()
            
            glPopMatrix()
    
    def render_grass_batch(self, instances: List[Vegetation], is_night: bool, max_height: float):
        """Render all grass instances in a batch."""
        if not instances:
            return
        
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        
        if is_night:
            glColor3f(0.0, 0.05, 0.08)
        else:
            glColor3f(0.15, 0.5, 0.1)
        
        # Render all grass patches
        num_blades = 3
        for veg in instances:
            glPushMatrix()
            glTranslatef(veg.x, veg.y * max_height, veg.z)
            
            for i in range(num_blades):
                angle = (i / num_blades) * 2 * math.pi
                glPushMatrix()
                glRotatef(angle * 180 / math.pi, 0, 1, 0)
                glBegin(GL_QUADS)
                glVertex3f(-veg.size * 0.1, 0, 0)
                glVertex3f(veg.size * 0.1, 0, 0)
                glVertex3f(veg.size * 0.1, veg.size * 0.5, 0)
                glVertex3f(-veg.size * 0.1, veg.size * 0.5, 0)
                glEnd()
                glPopMatrix()
            
            glPopMatrix()
    
    def render_flower_batch(self, instances: List[Vegetation], is_night: bool, max_height: float):
        """Render all flower instances in a batch."""
        if not instances:
            return
        
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        
        for veg in instances:
            glPushMatrix()
            glTranslatef(veg.x, veg.y * max_height, veg.z)
            
            # Stem
            glColor3f(0.0, 0.3, 0.0)
            glBegin(GL_QUADS)
            glVertex3f(-veg.size * 0.05, 0, 0)
            glVertex3f(veg.size * 0.05, 0, 0)
            glVertex3f(veg.size * 0.05, veg.size * 0.3, 0)
            glVertex3f(-veg.size * 0.05, veg.size * 0.3, 0)
            glEnd()
            
            # Petals (small sphere)
            if is_night:
                glColor3f(0.0, 0.06, 0.1)
            else:
                glColor3f(0.8, 0.6, 0.2)
            
            glPushMatrix()
            glTranslatef(0, veg.size * 0.3, 0)
            glScalef(veg.size * 0.2, veg.size * 0.2, veg.size * 0.2)
            
            # Simple sphere for petals
            segments = 6
            for i in range(segments):
                lat1 = math.pi * (-0.5 + i / segments)
                lat2 = math.pi * (-0.5 + (i + 1) / segments)
                glBegin(GL_QUAD_STRIP)
                for j in range(segments + 1):
                    lng = 2 * math.pi * j / segments
                    x1 = math.cos(lat1) * math.cos(lng)
                    y1 = math.sin(lat1)
                    z1 = math.cos(lat1) * math.sin(lng)
                    x2 = math.cos(lat2) * math.cos(lng)
                    y2 = math.sin(lat2)
                    z2 = math.cos(lat2) * math.sin(lng)
                    glVertex3f(x1, y1, z1)
                    glVertex3f(x2, y2, z2)
                glEnd()
            
            glPopMatrix()
            glPopMatrix()
    
    def render_rock_batch(self, instances: List[Vegetation], is_night: bool, max_height: float):
        """Render all rock instances in a batch."""
        if not instances:
            return
        
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        
        if is_night:
            glColor3f(0.05, 0.05, 0.08)
        else:
            glColor3f(0.4, 0.4, 0.35)
        
        # Render all rocks
        for veg in instances:
            glPushMatrix()
            glTranslatef(veg.x, veg.y * max_height, veg.z)
            glScalef(veg.size, veg.size, veg.size)
            
            glBegin(GL_QUADS)
            # Top
            glVertex3f(-1, 1, -1)
            glVertex3f(1, 1, -1)
            glVertex3f(1, 1, 1)
            glVertex3f(-1, 1, 1)
            # Bottom
            glVertex3f(-1, 0, 1)
            glVertex3f(1, 0, 1)
            glVertex3f(1, 0, -1)
            glVertex3f(-1, 0, -1)
            # Sides
            glVertex3f(-1, 0, -1)
            glVertex3f(-1, 1, -1)
            glVertex3f(-1, 1, 1)
            glVertex3f(-1, 0, 1)
            glVertex3f(1, 0, 1)
            glVertex3f(1, 1, 1)
            glVertex3f(1, 1, -1)
            glVertex3f(1, 0, -1)
            glVertex3f(-1, 0, -1)
            glVertex3f(1, 0, -1)
            glVertex3f(1, 1, -1)
            glVertex3f(-1, 1, -1)
            glVertex3f(-1, 0, 1)
            glVertex3f(-1, 1, 1)
            glVertex3f(1, 1, 1)
            glVertex3f(1, 0, 1)
            glEnd()
            
            glPopMatrix()
    
    def render_all(self, is_night: bool, max_height: float):
        """
        Render all prepared vegetation instances.
        
        Args:
            is_night: Whether it's nighttime
            max_height: Maximum terrain height for scaling
        """
        # Render each type in batches
        self.render_bush_batch(self.instances_by_type['bush'], is_night, max_height)
        self.render_grass_batch(self.instances_by_type['grass'], is_night, max_height)
        self.render_flower_batch(self.instances_by_type['flower'], is_night, max_height)
        self.render_rock_batch(self.instances_by_type['rock'], is_night, max_height)

