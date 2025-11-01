"""3D rendering implementation using pyglet."""

import pyglet
from pyglet.gl import *
import numpy as np
import math

from ..world.world import World
from ..entities.npc import NPC
from ..trees.tree import FruitTree
from ..houses.house import House


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
        self.window.set_exclusive_mouse(False)  # Show cursor
        self.window.set_mouse_visible(True)
        
        # Camera
        self.camera_x = 0.0
        self.camera_y = 50.0
        self.camera_z = 0.0
        self.camera_yaw = 0.0
        self.camera_pitch = -45.0
        
        # Controls - track keys manually for better reliability
        self.keys = {}  # Dictionary to track pressed keys
        self.keys_handler = pyglet.window.key.KeyStateHandler()
        self.window.push_handlers(self.keys_handler)
        
        # Mouse movement tracking
        self.mouse_x = width // 2
        self.mouse_y = height // 2
        self.last_mouse_x = self.mouse_x
        self.last_mouse_y = self.mouse_y
        self.mouse_buttons = set()  # Track pressed mouse buttons
        
        # NPC selection
        self.selected_npc = None  # Currently selected NPC
        
        # Setup OpenGL
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        # Enable color material for better color rendering
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        
        # Lighting setup - Sun (directional light)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        # Sun light from above (directional) - will be updated in render()
        glLightfv(GL_LIGHT0, GL_POSITION, (GLfloat * 4)(30.0, 100.0, 30.0, 0.0))
        
        # Ambient light from sky - will be updated in render()
        glEnable(GL_LIGHT1)
        glLightfv(GL_LIGHT1, GL_POSITION, (GLfloat * 4)(0.0, 100.0, 0.0, 0.0))
        
        # Global ambient light - will be updated in render()
        glLightModelfv(GL_LIGHT_MODEL_AMBIENT, (GLfloat * 4)(0.2, 0.2, 0.25, 1.0))
        
        # Enable smooth shading
        glShadeModel(GL_SMOOTH)
        
        # Debug log
        self.debug_log = []
        self.max_log_lines = 20
        self.debug_colors = False  # Debug flag for color testing
        
        # Event handlers
        @self.window.event
        def on_draw():
            self.render()
        
        # Add explicit key handlers to ensure keys are tracked
        @self.window.event
        def on_key_press(symbol, modifiers):
            # Track key press
            self.keys[symbol] = True
            # Press 'T' to toggle color debug mode
            if symbol == pyglet.window.key.T:
                self.debug_colors = not self.debug_colors
                if self.debug_colors:
                    self.log("Color debug mode ON - checking nighttime colors")
                else:
                    self.log("Color debug mode OFF")
        
        @self.window.event
        def on_key_release(symbol, modifiers):
            # Track key release
            self.keys[symbol] = False
        
        @self.window.event
        def on_mouse_press(x, y, button, modifiers):
            self.mouse_buttons.add(button)
            # Left click for NPC selection
            if button == pyglet.window.mouse.LEFT:
                self._pick_npc(x, y)
        
        @self.window.event
        def on_mouse_release(x, y, button, modifiers):
            self.mouse_buttons.discard(button)
        
        @self.window.event
        def on_mouse_motion(x, y, dx, dy):
            self.mouse_x = x
            self.mouse_y = y
            # Rotate camera when dragging (mouse button held)
            if self.mouse_buttons:
                self.camera_yaw += dx * 0.1
                self.camera_pitch = np.clip(self.camera_pitch - dy * 0.1, -90.0, 90.0)
        
        @self.window.event
        def on_mouse_drag(x, y, dx, dy, buttons, modifiers):
            self.mouse_x = x
            self.mouse_y = y
            self.camera_yaw += dx * 0.1
            self.camera_pitch = np.clip(self.camera_pitch - dy * 0.1, -90.0, 90.0)
        
        @self.window.event
        def on_mouse_scroll(x, y, scroll_x, scroll_y):
            # Zoom with mouse wheel
            self.camera_y += scroll_y * 2.0
            self.camera_y = max(5.0, self.camera_y)
    
    def update(self, delta_time: float):
        """Update camera based on keyboard and mouse."""
        move_speed = 20.0 * delta_time
        
        # Calculate forward and right vectors based on camera yaw
        # OpenGL rotates around Y axis: positive yaw rotates counterclockwise
        # When yaw=0: camera faces forward along +Z axis
        # Forward vector: direction camera is facing in world space
        
        yaw_rad = np.radians(self.camera_yaw)
        
        # Forward vector: when yaw=0, forward = (0, 0, -1) typically in OpenGL
        # But it depends on coordinate system convention
        # Try: forward = (sin(yaw), 0, -cos(yaw)) for OpenGL convention
        forward_x = np.sin(yaw_rad)
        forward_z = -np.cos(yaw_rad)  # Negate Z for OpenGL convention
        
        # Right vector: perpendicular to forward
        right_x = np.cos(yaw_rad)
        right_z = np.sin(yaw_rad)
        
        # Keyboard movement (WASD) - relative to camera direction
        # Use manual keys dict (updated by on_key_press/on_key_release) as primary
        # Fallback to KeyStateHandler if key not in dict
        w_pressed = self.keys.get(pyglet.window.key.W, False) or self.keys_handler[pyglet.window.key.W]
        s_pressed = self.keys.get(pyglet.window.key.S, False) or self.keys_handler[pyglet.window.key.S]
        a_pressed = self.keys.get(pyglet.window.key.A, False) or self.keys_handler[pyglet.window.key.A]
        d_pressed = self.keys.get(pyglet.window.key.D, False) or self.keys_handler[pyglet.window.key.D]
        space_pressed = self.keys.get(pyglet.window.key.SPACE, False) or self.keys_handler[pyglet.window.key.SPACE]
        shift_pressed = self.keys.get(pyglet.window.key.LSHIFT, False) or self.keys_handler[pyglet.window.key.LSHIFT]
        
        if w_pressed:
            # Forward (along camera's facing direction)
            self.camera_x += forward_x * move_speed
            self.camera_z += forward_z * move_speed
        if s_pressed:
            # Backward (opposite of camera's facing direction)
            self.camera_x -= forward_x * move_speed
            self.camera_z -= forward_z * move_speed
        if a_pressed:
            # Left (strafe left relative to camera)
            self.camera_x -= right_x * move_speed
            self.camera_z -= right_z * move_speed
        if d_pressed:
            # Right (strafe right relative to camera)
            self.camera_x += right_x * move_speed
            self.camera_z += right_z * move_speed
        if space_pressed:
            self.camera_y += move_speed
        if shift_pressed:
            self.camera_y -= move_speed
        
        self.camera_y = max(5.0, self.camera_y)
    
    def render(self):
        """Render the world."""
        # Update lighting based on time of day
        light_intensity = self.world.get_light_intensity()
        hour = (self.world.day_time / self.world.day_length) * 24.0
        is_night = hour < 6.0 or hour >= 18.0
        
        # Calculate sky color based on time
        if 6.0 <= hour < 8.0:  # Dawn
            sky_r = 0.3 + 0.2 * ((hour - 6.0) / 2.0)
            sky_g = 0.4 + 0.3 * ((hour - 6.0) / 2.0)
            sky_b = 0.5 + 0.5 * ((hour - 6.0) / 2.0)
        elif 8.0 <= hour < 18.0:  # Day
            sky_r, sky_g, sky_b = 0.5, 0.7, 1.0
        elif 18.0 <= hour < 20.0:  # Dusk
            t = (hour - 18.0) / 2.0
            # Transition from day blue to night blue-purple (no orange/red)
            sky_r = 0.5 - 0.47 * t  # Quickly reduce red (0.5 -> 0.03)
            sky_g = 0.7 - 0.67 * t  # Quickly reduce green (0.7 -> 0.03)
            sky_b = 1.0 - 0.88 * t  # Transition to deep blue (1.0 -> 0.12)
        else:  # Night
            # Deep blue-purple night sky
            sky_r, sky_g, sky_b = 0.03, 0.03, 0.12
        
        glClearColor(sky_r, sky_g, sky_b, 1.0)
        self.window.clear()
        
        # Setup camera - using pyglet's glu module
        from pyglet.gl.glu import gluPerspective
        
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(70.0, self.window.width / self.window.height, 0.1, 500.0)
        
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        
        # Rotate camera
        glRotatef(self.camera_pitch, 1.0, 0.0, 0.0)
        glRotatef(self.camera_yaw, 0.0, 1.0, 0.0)
        glTranslatef(-self.camera_x, -self.camera_y, -self.camera_z)
        
        # Update lighting based on time of day
        sun_intensity = light_intensity
        ambient_intensity = 0.2 + (light_intensity - 0.3) * 0.5
        
        # Sun light intensity
        glLightfv(GL_LIGHT0, GL_POSITION, (GLfloat * 4)(30.0, 100.0, 30.0, 0.0))
        
        # Adjust light color based on time of day
        if is_night:
            # Cool blue-white moonlight at night - ZERO red component
            glLightfv(GL_LIGHT0, GL_AMBIENT, (GLfloat * 4)(0.0, 0.12 * ambient_intensity, 0.3 * ambient_intensity, 1.0))
            glLightfv(GL_LIGHT0, GL_DIFFUSE, (GLfloat * 4)(0.0, 0.3 * sun_intensity, 0.7 * sun_intensity, 1.0))
            glLightfv(GL_LIGHT0, GL_SPECULAR, (GLfloat * 4)(0.0, 0.2 * sun_intensity, 0.6 * sun_intensity, 1.0))
        else:
            # Warm sunlight during day
            glLightfv(GL_LIGHT0, GL_AMBIENT, (GLfloat * 4)(0.4 * ambient_intensity, 0.4 * ambient_intensity, 0.5 * ambient_intensity, 1.0))
            glLightfv(GL_LIGHT0, GL_DIFFUSE, (GLfloat * 4)(1.0 * sun_intensity, 0.95 * sun_intensity, 0.8 * sun_intensity, 1.0))
            glLightfv(GL_LIGHT0, GL_SPECULAR, (GLfloat * 4)(1.0, 1.0, 1.0, 1.0))
        
        # Ambient light
        if is_night:
            # Cool blue ambient light at night - ZERO red component
            glLightfv(GL_LIGHT1, GL_AMBIENT, (GLfloat * 4)(0.0, 0.1 * ambient_intensity, 0.25 * ambient_intensity, 1.0))
            glLightfv(GL_LIGHT1, GL_DIFFUSE, (GLfloat * 4)(0.0, 0.15 * ambient_intensity, 0.3 * ambient_intensity, 1.0))
            # Very cool, ZERO red for night
            glLightModelfv(GL_LIGHT_MODEL_AMBIENT, (GLfloat * 4)(0.0, 0.015 * ambient_intensity, 0.05 * ambient_intensity, 1.0))
        else:
            # Normal ambient light during day
            glLightfv(GL_LIGHT1, GL_AMBIENT, (GLfloat * 4)(0.3 * ambient_intensity, 0.4 * ambient_intensity, 0.5 * ambient_intensity, 1.0))
            glLightfv(GL_LIGHT1, GL_DIFFUSE, (GLfloat * 4)(0.3 * ambient_intensity, 0.4 * ambient_intensity, 0.5 * ambient_intensity, 1.0))
            glLightModelfv(GL_LIGHT_MODEL_AMBIENT, (GLfloat * 4)(0.2 * ambient_intensity, 0.2 * ambient_intensity, 0.25 * ambient_intensity, 1.0))
        
        # Render terrain
        self._render_terrain()
        
        # Render houses
        for house in self.world.houses:
            self._render_house(house)
        
        # Render trees
        for tree in self.world.trees:
            self._render_tree(tree)
        
        # Render NPCs
        for npc in self.world.entities:
            self._render_npc(npc)
        
        # Render debug overlay
        self._render_debug_overlay()
        
        # Render NPC detail panel if NPC is selected (render after debug overlay for proper layering)
        if self.selected_npc and self.selected_npc.is_alive:
            self._render_npc_detail_panel()
    
    def _render_terrain(self):
        """Render the terrain mesh with colorful grass."""
        # Adjust terrain color based on time of day
        hour = (self.world.day_time / self.world.day_length) * 24.0
        is_night = hour < 6.0 or hour >= 18.0
        
        # Log color info if debug mode is on
        if self.debug_colors and is_night:
            self.log(f"Night terrain: hour={hour:.1f}, is_night={is_night}")
        
        if is_night:
            # Disable COLOR_MATERIAL temporarily to use pure material colors
            glDisable(GL_COLOR_MATERIAL)
            # Cool, darker colors for nighttime terrain - ZERO red component
            glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, (GLfloat * 4)(0.0, 0.03, 0.06, 1.0))
            glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, (GLfloat * 4)(0.0, 0.04, 0.08, 1.0))
            glMaterialfv(GL_FRONT_AND_BACK, GL_EMISSION, (GLfloat * 4)(0.0, 0.0, 0.0, 1.0))  # No emission
            glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, (GLfloat * 4)(0.0, 0.01, 0.02, 1.0))
            glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 10.0)
            
            # Set color as well (though material should override) - ZERO red
            glColor3f(0.0, 0.04, 0.08)
            
            if self.debug_colors:
                self.log(f"Night terrain color: RGB(0.0, 0.04, 0.08) - Blue={0.08:.3f}, Red=0.0")
        else:
            # Normal green grass colors for daytime
            glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, (GLfloat * 4)(0.2, 0.3, 0.15, 1.0))
            glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, (GLfloat * 4)(0.3, 0.6, 0.2, 1.0))
            glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, (GLfloat * 4)(0.1, 0.1, 0.1, 1.0))
            glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 10.0)
            glColor3f(0.2, 0.5, 0.15)
        
        # Render ground plane with gradient colors
        glBegin(GL_QUADS)
        glNormal3f(0.0, 1.0, 0.0)
        glVertex3f(-100, 0, -100)
        glVertex3f(100, 0, -100)
        glVertex3f(100, 0, 100)
        glVertex3f(-100, 0, 100)
        glEnd()
        
        # Re-enable COLOR_MATERIAL for other objects
        if is_night:
            glEnable(GL_COLOR_MATERIAL)
            glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
    
    def _render_tree(self, tree: FruitTree):
        """Render a colorful fruit tree with improved visuals."""
        if not tree.is_alive:
            return
        
        hour = (self.world.day_time / self.world.day_length) * 24.0
        is_night = hour < 6.0 or hour >= 18.0
        
        glPushMatrix()
        glTranslatef(tree.x, tree.y, tree.z)
        
        trunk_height = 2.0 * tree.growth_stage
        trunk_base_width = 0.15 * tree.growth_stage
        trunk_top_width = trunk_base_width * 0.7  # Tapered trunk
        
        # Trunk with brown material - render as a proper cylinder
        if is_night:
            # Cooler, darker trunk at night - minimal red, blue-gray tones
            glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, (GLfloat * 4)(0.0, 0.06, 0.08, 1.0))
            glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, (GLfloat * 4)(0.0, 0.08, 0.1, 1.0))
            glColor3f(0.0, 0.08, 0.1)
        else:
            # Normal brown trunk during day
            glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, (GLfloat * 4)(0.3, 0.15, 0.1, 1.0))
            glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, (GLfloat * 4)(0.5, 0.3, 0.2, 1.0))
            glColor3f(0.5, 0.3, 0.2)
        
        glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, (GLfloat * 4)(0.1, 0.1, 0.1, 1.0))
        glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 5.0)
        
        # Draw cylindrical trunk with proper normals
        import math
        trunk_segments = 12
        for i in range(trunk_segments):
            angle1 = (i / trunk_segments) * 2 * math.pi
            angle2 = ((i + 1) / trunk_segments) * 2 * math.pi
            
            # Bottom vertices
            x1_bottom = math.cos(angle1) * trunk_base_width
            z1_bottom = math.sin(angle1) * trunk_base_width
            x2_bottom = math.cos(angle2) * trunk_base_width
            z2_bottom = math.sin(angle2) * trunk_base_width
            
            # Top vertices (narrower)
            x1_top = math.cos(angle1) * trunk_top_width
            z1_top = math.sin(angle1) * trunk_top_width
            x2_top = math.cos(angle2) * trunk_top_width
            z2_top = math.sin(angle2) * trunk_top_width
            
            # Normal for this face
            normal_x = math.cos(angle1)
            normal_z = math.sin(angle1)
            
            glBegin(GL_QUADS)
            glNormal3f(normal_x, 0.0, normal_z)
            glVertex3f(x1_bottom, 0, z1_bottom)
            glVertex3f(x2_bottom, 0, z2_bottom)
            glVertex3f(x2_top, trunk_height, z2_top)
            glVertex3f(x1_top, trunk_height, z1_top)
            glEnd()
        
        # Leaves/Foliage with green material - multiple layers for better look
        if tree.growth_stage >= 0.5:
            if is_night:
                # Cool, muted colors for night - ZERO red, blue-green-gray
                glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, (GLfloat * 4)(0.0, 0.05, 0.04, 1.0))
                glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, (GLfloat * 4)(0.0, 0.08, 0.06, 1.0))
                glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, (GLfloat * 4)(0.0, 0.04, 0.03, 1.0))
                glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 20.0)
                glColor3f(0.0, 0.08, 0.06)
            else:
                # Vibrant green leaves for day
                glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, (GLfloat * 4)(0.1, 0.2, 0.1, 1.0))
                glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, (GLfloat * 4)(0.2, 0.6, 0.2, 1.0))
                glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, (GLfloat * 4)(0.2, 0.3, 0.2, 1.0))
                glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 20.0)
                glColor3f(0.1, 0.7, 0.2)
            
            # Main foliage cluster at top
            foliage_base_height = trunk_height + 0.8 * tree.growth_stage
            main_foliage_size = 1.2 * tree.growth_stage
            glPushMatrix()
            glTranslatef(0, foliage_base_height, 0)
            self._draw_sphere(main_foliage_size, 20)
            glPopMatrix()
            
            # Secondary foliage clusters for fuller look
            if tree.growth_stage >= 0.7:
                # Medium cluster slightly lower
                glPushMatrix()
                glTranslatef(0, foliage_base_height - 0.3 * tree.growth_stage, 0)
                self._draw_sphere(main_foliage_size * 0.85, 18)
                glPopMatrix()
            
            if tree.growth_stage >= 0.85:
                # Smaller clusters around the sides for natural variation
                for i in range(3):
                    angle = (i / 3.0) * 2 * math.pi
                    offset_x = math.cos(angle) * 0.4 * tree.growth_stage
                    offset_z = math.sin(angle) * 0.4 * tree.growth_stage
                    glPushMatrix()
                    glTranslatef(offset_x, foliage_base_height - 0.2 * tree.growth_stage, offset_z)
                    self._draw_sphere(main_foliage_size * 0.6, 14)
                    glPopMatrix()
            
            # Render fruit - all red, make them more visible
            ripe_count = tree.get_ripe_fruit_count()
            total_fruit = len(tree.fruit_maturity)
            
            if total_fruit > 0:
                # Bright red fruit material - make it glow at night
                if is_night:
                    # At night, make fruit glow brighter (emission-like effect)
                    glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, (GLfloat * 4)(0.8, 0.1, 0.1, 1.0))
                    glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, (GLfloat * 4)(1.0, 0.2, 0.2, 1.0))
                    glMaterialfv(GL_FRONT_AND_BACK, GL_EMISSION, (GLfloat * 4)(0.5, 0.05, 0.05, 1.0))
                    glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, (GLfloat * 4)(0.9, 0.9, 0.9, 1.0))
                    glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 60.0)
                    glColor3f(1.0, 0.2, 0.2)  # Brighter red at night
                else:
                    # Daytime - normal bright red
                    glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, (GLfloat * 4)(0.4, 0.05, 0.05, 1.0))
                    glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, (GLfloat * 4)(1.0, 0.1, 0.1, 1.0))
                    glMaterialfv(GL_FRONT_AND_BACK, GL_EMISSION, (GLfloat * 4)(0.0, 0.0, 0.0, 1.0))
                    glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, (GLfloat * 4)(0.9, 0.9, 0.9, 1.0))
                    glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 60.0)
                    glColor3f(1.0, 0.1, 0.1)  # Bright red
                
                # Render multiple fruit pieces around the tree
                fruit_positions = []
                for i, (fruit_id, maturity) in enumerate(tree.fruit_maturity.items()):
                    # Render all fruit, but brighter if ripe
                    if maturity >= 1.0:  # Ripe fruit
                        angle = (i * 2 * math.pi / max(total_fruit, 1))
                        radius_offset = 0.8 + (i % 4) * 0.2
                        x_offset = math.cos(angle) * radius_offset
                        z_offset = math.sin(angle) * radius_offset
                        y_offset = 1.0 + (i % 4) * 0.3
                        fruit_positions.append((x_offset, y_offset, z_offset, True))  # True = ripe
                    elif maturity >= 0.5:  # Growing fruit (smaller, less bright)
                        angle = (i * 2 * math.pi / max(total_fruit, 1))
                        radius_offset = 0.8 + (i % 4) * 0.2
                        x_offset = math.cos(angle) * radius_offset
                        z_offset = math.sin(angle) * radius_offset
                        y_offset = 1.0 + (i % 4) * 0.3
                        fruit_positions.append((x_offset, y_offset, z_offset, False))  # False = growing
                
                # Render all fruit pieces (up to 30 visible)
                for pos in fruit_positions[:30]:
                    glPushMatrix()
                    glTranslatef(pos[0], trunk_height + pos[1], pos[2])
                    if pos[3]:  # Ripe fruit - larger and brighter
                        self._draw_sphere(0.18, 12)
                    else:  # Growing fruit - smaller
                        glColor3f(0.8, 0.2, 0.2)
                        self._draw_sphere(0.12, 10)
                        glColor3f(1.0, 0.1, 0.1)  # Reset color
                    glPopMatrix()
        
        glPopMatrix()
    
    def _render_npc(self, npc: NPC):
        """Render an NPC with a humanoid model."""
        if not npc.is_alive:
            return
        
        hour = (self.world.day_time / self.world.day_length) * 24.0
        is_night = hour < 6.0 or hour >= 18.0
        
        glPushMatrix()
        glTranslatef(npc.x, npc.y, npc.z)
        
        # Calculate NPC facing direction based on movement
        if hasattr(npc, 'target_x') and npc.target_x is not None:
            dx = npc.target_x - npc.x
            dz = npc.target_z - npc.z
            if abs(dx) > 0.01 or abs(dz) > 0.01:
                facing_angle = math.degrees(math.atan2(dz, dx))
                glRotatef(facing_angle, 0.0, 1.0, 0.0)
        
        # Color based on multiple factors - adjust for night
        health_ratio = npc.health / npc.max_health
        hunger_ratio = npc.hunger / npc.max_hunger
        
        if is_night:
            # Cool colors at night - blue-gray tones, minimal red
            # Base color: health affects brightness, but keep cool tones
            base_r = 0.1 * (1.0 - health_ratio)  # Very low red
            base_g = 0.15 * health_ratio + 0.1  # Some green
            base_b = 0.2 * health_ratio + 0.15  # Blue dominant
            
            # Add saturation based on hunger (well-fed = brighter)
            saturation = 0.3 + hunger_ratio * 0.3
            
            # Final color with saturation - keep cool tones
            r = base_r * saturation
            g = base_g * saturation
            b = base_b * saturation
        else:
            # Normal colors during day
            base_r = 1.0 - health_ratio
            base_g = health_ratio
            base_b = npc.genome.get('stamina', 100.0) / 150.0
            
            saturation = 0.5 + hunger_ratio * 0.5
            
            r = 0.3 + base_r * saturation
            g = 0.3 + base_g * saturation
            b = 0.3 + base_b * saturation
        
        # Set material properties
        glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, (GLfloat * 4)(r * 0.3, g * 0.3, b * 0.3, 1.0))
        glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, (GLfloat * 4)(r, g, b, 1.0))
        glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, (GLfloat * 4)(0.5, 0.5, 0.5, 1.0))
        glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 30.0)
        
        glColor3f(r, g, b)
        
        # Scale based on NPC size (children are smaller)
        scale = npc.size
        if hasattr(npc, 'age_stage') and npc.age_stage == "child":
            scale *= 0.7  # Children are visually smaller
        
        # Head (sphere on top)
        glPushMatrix()
        glTranslatef(0.0, 0.6 * scale, 0.0)
        self._draw_sphere(0.15 * scale, 12)
        glPopMatrix()
        
        # Body/Torso (cube)
        body_height = 0.4 * scale
        body_width = 0.2 * scale
        body_depth = 0.15 * scale
        glPushMatrix()
        glTranslatef(0.0, 0.25 * scale, 0.0)
        self._draw_box(body_width, body_height, body_depth)
        glPopMatrix()
        
        # Left arm
        glPushMatrix()
        glTranslatef(-body_width * 1.2, 0.3 * scale, 0.0)
        self._draw_box(0.08 * scale, 0.3 * scale, 0.08 * scale)
        glPopMatrix()
        
        # Right arm
        glPushMatrix()
        glTranslatef(body_width * 1.2, 0.3 * scale, 0.0)
        self._draw_box(0.08 * scale, 0.3 * scale, 0.08 * scale)
        glPopMatrix()
        
        # Left leg
        glPushMatrix()
        glTranslatef(-body_width * 0.6, -0.2 * scale, 0.0)
        self._draw_box(0.1 * scale, 0.4 * scale, 0.1 * scale)
        glPopMatrix()
        
        # Right leg
        glPushMatrix()
        glTranslatef(body_width * 0.6, -0.2 * scale, 0.0)
        self._draw_box(0.1 * scale, 0.4 * scale, 0.1 * scale)
        glPopMatrix()
        
        glPopMatrix()
        
        # Render stat bars above NPC
        self._render_npc_stats(npc)
        
        # Highlight selected NPC with a ring/outline
        if self.selected_npc == npc:
            self._render_npc_highlight(npc)
    
    def _draw_box(self, width: float, height: float, depth: float):
        """Draw a box (cube) with proper normals."""
        w = width / 2.0
        h = height / 2.0
        d = depth / 2.0
        
        glBegin(GL_QUADS)
        
        # Front face
        glNormal3f(0.0, 0.0, 1.0)
        glVertex3f(-w, -h, d)
        glVertex3f(w, -h, d)
        glVertex3f(w, h, d)
        glVertex3f(-w, h, d)
        
        # Back face
        glNormal3f(0.0, 0.0, -1.0)
        glVertex3f(-w, -h, -d)
        glVertex3f(-w, h, -d)
        glVertex3f(w, h, -d)
        glVertex3f(w, -h, -d)
        
        # Top face
        glNormal3f(0.0, 1.0, 0.0)
        glVertex3f(-w, h, -d)
        glVertex3f(-w, h, d)
        glVertex3f(w, h, d)
        glVertex3f(w, h, -d)
        
        # Bottom face
        glNormal3f(0.0, -1.0, 0.0)
        glVertex3f(-w, -h, -d)
        glVertex3f(w, -h, -d)
        glVertex3f(w, -h, d)
        glVertex3f(-w, -h, d)
        
        # Right face
        glNormal3f(1.0, 0.0, 0.0)
        glVertex3f(w, -h, -d)
        glVertex3f(w, h, -d)
        glVertex3f(w, h, d)
        glVertex3f(w, -h, d)
        
        # Left face
        glNormal3f(-1.0, 0.0, 0.0)
        glVertex3f(-w, -h, -d)
        glVertex3f(-w, -h, d)
        glVertex3f(-w, h, d)
        glVertex3f(-w, h, -d)
        
        glEnd()
    
    def _render_npc_stats(self, npc: NPC):
        """Render health, hunger, and stamina bars above NPC."""
        if not npc.is_alive:
            return
        
        # Calculate screen position (simplified - always render in 3D space above NPC)
        bar_height = 0.8
        bar_width = 0.4
        bar_thickness = 0.05
        spacing = 0.1
        
        glPushMatrix()
        glTranslatef(npc.x, npc.y + bar_height, npc.z)
        
        # Make bars face camera (billboard effect)
        # Calculate direction to camera
        dx = self.camera_x - npc.x
        dy = self.camera_y - (npc.y + bar_height)
        dz = self.camera_z - npc.z
        
        # Rotate bars to face camera (simplified - just rotate around Y axis)
        angle = math.degrees(math.atan2(dx, dz))
        glRotatef(-angle, 0.0, 1.0, 0.0)
        
        glDisable(GL_LIGHTING)
        
        # Health bar (red)
        health_ratio = npc.health / npc.max_health
        glColor3f(1.0, 0.0, 0.0)
        glBegin(GL_QUADS)
        glVertex3f(-bar_width/2, 0, 0)
        glVertex3f(bar_width/2, 0, 0)
        glVertex3f(bar_width/2, bar_thickness, 0)
        glVertex3f(-bar_width/2, bar_thickness, 0)
        glEnd()
        
        # Health fill (green)
        glColor3f(0.0, 1.0, 0.0)
        glBegin(GL_QUADS)
        glVertex3f(-bar_width/2, 0, 0.001)
        glVertex3f(-bar_width/2 + bar_width * health_ratio, 0, 0.001)
        glVertex3f(-bar_width/2 + bar_width * health_ratio, bar_thickness, 0.001)
        glVertex3f(-bar_width/2, bar_thickness, 0.001)
        glEnd()
        
        # Hunger bar (orange)
        hunger_ratio = npc.hunger / npc.max_hunger
        glColor3f(0.5, 0.3, 0.0)
        glBegin(GL_QUADS)
        glVertex3f(-bar_width/2, spacing + bar_thickness, 0)
        glVertex3f(bar_width/2, spacing + bar_thickness, 0)
        glVertex3f(bar_width/2, spacing + bar_thickness * 2, 0)
        glVertex3f(-bar_width/2, spacing + bar_thickness * 2, 0)
        glEnd()
        
        # Hunger fill (yellow)
        glColor3f(1.0, 0.8, 0.0)
        glBegin(GL_QUADS)
        glVertex3f(-bar_width/2, spacing + bar_thickness, 0.001)
        glVertex3f(-bar_width/2 + bar_width * hunger_ratio, spacing + bar_thickness, 0.001)
        glVertex3f(-bar_width/2 + bar_width * hunger_ratio, spacing + bar_thickness * 2, 0.001)
        glVertex3f(-bar_width/2, spacing + bar_thickness * 2, 0.001)
        glEnd()
        
        # Stamina bar (gray)
        stamina_ratio = npc.stamina / npc.max_stamina
        glColor3f(0.3, 0.3, 0.3)
        glBegin(GL_QUADS)
        glVertex3f(-bar_width/2, spacing * 2 + bar_thickness * 2, 0)
        glVertex3f(bar_width/2, spacing * 2 + bar_thickness * 2, 0)
        glVertex3f(bar_width/2, spacing * 2 + bar_thickness * 3, 0)
        glVertex3f(-bar_width/2, spacing * 2 + bar_thickness * 3, 0)
        glEnd()
        
        # Stamina fill (blue)
        glColor3f(0.0, 0.5, 1.0)
        glBegin(GL_QUADS)
        glVertex3f(-bar_width/2, spacing * 2 + bar_thickness * 2, 0.001)
        glVertex3f(-bar_width/2 + bar_width * stamina_ratio, spacing * 2 + bar_thickness * 2, 0.001)
        glVertex3f(-bar_width/2 + bar_width * stamina_ratio, spacing * 2 + bar_thickness * 3, 0.001)
        glVertex3f(-bar_width/2, spacing * 2 + bar_thickness * 3, 0.001)
        glEnd()
        
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glPopMatrix()
    
    def _render_house(self, house: House):
        """Render a house."""
        if not house.is_built:
            return
        
        # Adjust house color based on time of day
        hour = (self.world.day_time / self.world.day_length) * 24.0
        is_night = hour < 6.0 or hour >= 18.0
        
        glPushMatrix()
        glTranslatef(house.x, house.y, house.z)
        
        # House base/walls
        if is_night:
            # Cool, muted colors for night - blue-gray, ZERO red
            glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, (GLfloat * 4)(0.0, 0.1, 0.12, 1.0))
            glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, (GLfloat * 4)(0.0, 0.12, 0.18, 1.0))
        else:
            # Normal warm colors for day
            glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, (GLfloat * 4)(0.4, 0.3, 0.25, 1.0))
            glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, (GLfloat * 4)(0.6, 0.5, 0.4, 1.0))
        
        glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, (GLfloat * 4)(0.1, 0.1, 0.1, 1.0))
        glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 5.0)
        
        size = 1.5
        height = 2.0
        
        # Color based on occupancy
        occupancy = len(house.current_occupants) / house.capacity
        if is_night:
            glColor3f(0.0, 0.12 + occupancy * 0.04, 0.18 + occupancy * 0.04)
        else:
            glColor3f(0.6 + occupancy * 0.2, 0.5, 0.4)
        
        # Walls (cube)
        glBegin(GL_QUADS)
        # Front
        glNormal3f(0.0, 0.0, 1.0)
        glVertex3f(-size, 0, size)
        glVertex3f(size, 0, size)
        glVertex3f(size, height, size)
        glVertex3f(-size, height, size)
        # Back
        glNormal3f(0.0, 0.0, -1.0)
        glVertex3f(-size, 0, -size)
        glVertex3f(-size, height, -size)
        glVertex3f(size, height, -size)
        glVertex3f(size, 0, -size)
        # Left
        glNormal3f(-1.0, 0.0, 0.0)
        glVertex3f(-size, 0, -size)
        glVertex3f(-size, height, -size)
        glVertex3f(-size, height, size)
        glVertex3f(-size, 0, size)
        # Right
        glNormal3f(1.0, 0.0, 0.0)
        glVertex3f(size, 0, -size)
        glVertex3f(size, 0, size)
        glVertex3f(size, height, size)
        glVertex3f(size, height, -size)
        glEnd()
        
        # Roof
        if is_night:
            # Cool, muted roof colors for night - darker, ZERO red
            glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, (GLfloat * 4)(0.0, 0.06, 0.1, 1.0))
            glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, (GLfloat * 4)(0.0, 0.08, 0.12, 1.0))
            glColor3f(0.0, 0.08, 0.12)
        else:
            # Normal warm roof colors for day
            glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, (GLfloat * 4)(0.5, 0.2, 0.15, 1.0))
            glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, (GLfloat * 4)(0.7, 0.3, 0.2, 1.0))
            glColor3f(0.7, 0.3, 0.2)
        
        glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, (GLfloat * 4)(0.1, 0.1, 0.1, 1.0))
        glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 5.0)
        
        glBegin(GL_TRIANGLES)
        # Front roof triangle
        glNormal3f(0.0, 0.7, 0.7)
        glVertex3f(-size, height, size)
        glVertex3f(size, height, size)
        glVertex3f(0, height + 1.0, 0)
        # Back roof triangle
        glNormal3f(0.0, 0.7, -0.7)
        glVertex3f(-size, height, -size)
        glVertex3f(0, height + 1.0, 0)
        glVertex3f(size, height, -size)
        glEnd()
        
        # Roof sides
        glBegin(GL_QUADS)
        glNormal3f(-0.7, 0.7, 0.0)
        glVertex3f(-size, height, size)
        glVertex3f(0, height + 1.0, 0)
        glVertex3f(-size, height, -size)
        glVertex3f(-size, height, size)
        
        glNormal3f(0.7, 0.7, 0.0)
        glVertex3f(size, height, size)
        glVertex3f(size, height, -size)
        glVertex3f(0, height + 1.0, 0)
        glVertex3f(size, height, size)
        glEnd()
        
        glPopMatrix()
    
    def _pick_npc(self, screen_x: int, screen_y: int):
        """
        Pick an NPC from screen coordinates using raycasting.
        
        Args:
            screen_x: Mouse X coordinate in screen space
            screen_y: Mouse Y coordinate in screen space
        """
        # Convert screen Y (pyglet uses bottom-left origin)
        screen_y = self.window.height - screen_y
        
        # Simple picking: project screen point to world space and find closest NPC
        # For simplicity, we'll project to the ground plane and check nearby NPCs
        # This is a simplified version - proper raycasting would be more accurate
        
        # Get viewport and projection info
        # We'll use a simple approach: check NPCs near the camera's look direction
        
        # Calculate world position from screen coordinates
        # This is simplified - for proper implementation, we'd unproject properly
        viewport = (GLint * 4)(0, 0, self.window.width, self.window.height)
        
        # Normalize screen coordinates to [-1, 1]
        norm_x = (screen_x / self.window.width) * 2.0 - 1.0
        norm_y = (screen_y / self.window.height) * 2.0 - 1.0
        
        # Calculate FOV and aspect ratio (matching gluPerspective)
        fov = 70.0  # Match the FOV used in render()
        aspect = self.window.width / self.window.height
        tan_fov = np.tan(np.radians(fov / 2.0))
        
        # Calculate camera forward direction
        yaw_rad = np.radians(self.camera_yaw)
        pitch_rad = np.radians(self.camera_pitch)
        
        forward_x = np.sin(yaw_rad) * np.cos(pitch_rad)
        forward_y = -np.sin(pitch_rad)
        forward_z = -np.cos(yaw_rad) * np.cos(pitch_rad)
        
        # Calculate right and up vectors for camera
        right_x = np.cos(yaw_rad)
        right_z = np.sin(yaw_rad)
        up_x = -np.sin(yaw_rad) * np.sin(pitch_rad)
        up_y = np.cos(pitch_rad)
        up_z = np.cos(yaw_rad) * np.sin(pitch_rad)
        
        # Calculate ray direction from camera through click point
        ray_x = forward_x + right_x * norm_x * tan_fov * aspect + up_x * norm_y * tan_fov
        ray_y = forward_y + right_z * norm_x * tan_fov * aspect + up_y * norm_y * tan_fov
        ray_z = forward_z + up_z * norm_y * tan_fov
        
        # Normalize ray direction
        ray_len = np.sqrt(ray_x*ray_x + ray_y*ray_y + ray_z*ray_z)
        if ray_len > 0.01:
            ray_x /= ray_len
            ray_y /= ray_len
            ray_z /= ray_len
        
        # Find NPC closest to ray (within reasonable distance)
        closest_npc = None
        closest_distance = float('inf')
        max_selection_distance = 150.0  # Increased range
        
        for npc in self.world.entities:
            if not npc.is_alive:
                continue
            
            # Vector from camera to NPC
            dx = npc.x - self.camera_x
            dy = npc.y - self.camera_y
            dz = npc.z - self.camera_z
            
            # Distance to NPC
            dist = np.sqrt(dx*dx + dy*dy + dz*dz)
            
            if dist > max_selection_distance:
                continue
            
            # Calculate distance from ray to NPC (point-to-line distance)
            cross_x = dy * ray_z - dz * ray_y
            cross_y = dz * ray_x - dx * ray_z
            cross_z = dx * ray_y - dy * ray_x
            dist_to_ray = np.sqrt(cross_x*cross_x + cross_y*cross_y + cross_z*cross_z)
            
            # Check if NPC is in front of camera (dot product > 0)
            dot_product = ray_x * dx + ray_y * dy + ray_z * dz
            if dot_product < 0:
                continue
            
            # NPC bounding sphere radius - increased for easier clicking
            npc_radius = max(0.8, 0.5 * npc.size)  # Minimum radius of 0.8
            
            # Check if ray passes close enough to NPC - increased tolerance
            # Use a larger tolerance based on distance (further NPCs need larger tolerance)
            tolerance = npc_radius * (3.0 + dist * 0.05)  # Scale tolerance with distance
            
            if dist_to_ray < tolerance:
                if dist < closest_distance:
                    closest_npc = npc
                    closest_distance = dist
        
        self.selected_npc = closest_npc
        if closest_npc:
            self.log(f"Selected NPC at ({closest_npc.x:.1f}, {closest_npc.y:.1f}, {closest_npc.z:.1f})")
    
    def _render_npc_highlight(self, npc: NPC):
        """Render a highlight ring around the selected NPC."""
        glDisable(GL_LIGHTING)
        glDisable(GL_DEPTH_TEST)
        
        glPushMatrix()
        glTranslatef(npc.x, npc.y, npc.z)
        
        # Draw a glowing ring around the NPC
        ring_radius = 0.6 * npc.size
        ring_height = 0.1
        
        glColor4f(1.0, 1.0, 0.0, 0.8)  # Yellow, semi-transparent
        glLineWidth(3.0)
        
        glBegin(GL_LINE_LOOP)
        import math
        segments = 32
        for i in range(segments):
            angle = (i / segments) * 2 * math.pi
            x = math.cos(angle) * ring_radius
            z = math.sin(angle) * ring_radius
            glVertex3f(x, ring_height, z)
        glEnd()
        
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glPopMatrix()
    
    def _render_npc_detail_panel(self):
        """Render a detailed information panel for the selected NPC."""
        npc = self.selected_npc
        if not npc or not npc.is_alive:
            return
        
        # Switch to 2D rendering
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, self.window.width, 0, self.window.height, -1, 1)
        
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()
        
        glDisable(GL_LIGHTING)
        glDisable(GL_DEPTH_TEST)
        
        # Panel dimensions (increased height to accommodate NN visualization)
        panel_width = 400
        panel_height = 750  # Increased from 600
        panel_x = self.window.width - panel_width - 20
        panel_y = self.window.height - panel_height - 20
        
        # Background panel (semi-transparent dark)
        glColor4f(0.1, 0.1, 0.15, 0.9)
        glBegin(GL_QUADS)
        glVertex2f(panel_x, panel_y)
        glVertex2f(panel_x + panel_width, panel_y)
        glVertex2f(panel_x + panel_width, panel_y + panel_height)
        glVertex2f(panel_x, panel_y + panel_height)
        glEnd()
        
        # Border
        glColor4f(0.3, 0.6, 1.0, 1.0)  # Blue border
        glLineWidth(2.0)
        glBegin(GL_LINE_LOOP)
        glVertex2f(panel_x, panel_y)
        glVertex2f(panel_x + panel_width, panel_y)
        glVertex2f(panel_x + panel_width, panel_y + panel_height)
        glVertex2f(panel_x, panel_y + panel_height)
        glEnd()
        
        # Prepare text content
        state_descriptions = {
            "wandering": "Exploring the world randomly",
            "seeking_food": "Looking for food sources",
            "eating": "Consuming fruit from a tree",
            "resting": "Resting to restore stamina",
            "seeking_shelter": "Looking for shelter (nighttime)",
            "in_shelter": "Safe inside a house"
        }
        
        state_desc = state_descriptions.get(npc.state, npc.state)
        
        # Health percentage
        health_pct = (npc.health / npc.max_health) * 100
        health_status = "Excellent" if health_pct > 80 else "Good" if health_pct > 50 else "Fair" if health_pct > 25 else "Critical"
        
        # Hunger percentage
        hunger_pct = (npc.hunger / npc.max_hunger) * 100
        hunger_status = "Well Fed" if hunger_pct > 80 else "Satisfied" if hunger_pct > 50 else "Hungry" if hunger_pct > 25 else "Starving"
        
        # Stamina percentage
        stamina_pct = (npc.stamina / npc.max_stamina) * 100 if npc.max_stamina > 0 else 0
        stamina_status = "Energetic" if stamina_pct > 80 else "Active" if stamina_pct > 50 else "Tired" if stamina_pct > 25 else "Exhausted"
        
        # Build text lines (organized to avoid overlap with bars)
        lines = [
            f"=== NPC DETAILS ===",
            f"",
            f"Name: {npc.name}",
            f"",
            f"Status: {npc.state.replace('_', ' ').title()}",
            f"Description: {state_desc}",
            f"",
            f"--- Age & Lifecycle ---",
            f"Age: {npc.age:.1f}s / {npc.lifespan:.1f}s",
            f"Age Stage: {npc.age_stage.title()}",
            f"Adult Age: {npc.adult_age:.1f}s",
            f"Can Reproduce: {'Yes' if npc.can_reproduce else 'No'}",
            f"Repro Cooldown: {npc.reproduction_cooldown:.1f}s" if npc.reproduction_cooldown > 0 else "Repro Cooldown: Ready",
            f"",
            f"--- Health ---",
            f"Health: {npc.health:.1f} / {npc.max_health:.1f} ({health_pct:.1f}%)",
            f"Status: {health_status}",
            f"",
            f"--- Hunger ---",
            f"Hunger: {npc.hunger:.1f} / {npc.max_hunger:.1f} ({hunger_pct:.1f}%)",
            f"Status: {hunger_status}",
            f"Note: Hunger decreases over time.",
            f"Starvation causes health loss.",
            f"",
            f"--- Stamina ---",
            f"Stamina: {npc.stamina:.1f} / {npc.max_stamina:.1f} ({stamina_pct:.1f}%)",
            f"Status: {stamina_status}",
            f"Note: Stamina decreases during",
            f"movement. Rest restores it.",
            f"",
            f"--- Physical ---",
            f"Speed: {npc.speed:.2f}",
            f"Size: {npc.size:.2f}",
            f"",
            f"--- Genetic ---",
            f"Vision: {npc.genome.get('vision_range', 0):.1f}",
            f"Food Pref: {npc.genome.get('food_preference', 0):.2f}",
            f"",
            f"--- Statistics ---",
            f"Fruit Collected: {npc.fruit_collected}",
            f"Position: ({npc.x:.1f}, {npc.y:.1f}, {npc.z:.1f})",
            f"",
            f"House: {'Yes' if npc.current_house else 'No'}",
            f"Target: ({npc.target_x:.1f}, {npc.target_z:.1f})" if npc.target_x else "Target: None",
            f"",
            f"Click elsewhere to deselect"
        ]
        
        # Render bars visually (with labels) - positioned at bottom to avoid text overlap
        bar_x = panel_x + 20
        bar_y = panel_y + 160  # Positioned at bottom area of panel (above reserved space)
        bar_width = panel_width - 40
        bar_height = 15
        bar_spacing = 50  # Increased spacing between bars
        bar_label_offset = 18  # Space above bar for label
        
        # Health bar with label
        try:
            health_label = pyglet.text.Label(
                "Health:",
                font_name='Courier New',
                font_size=10,
                x=int(bar_x),
                y=int(bar_y + bar_height + bar_label_offset),
                anchor_x='left',
                anchor_y='bottom',
                color=(255, 255, 255, 255)
            )
            health_label.draw()
        except:
            pass
        
        glColor3f(0.8, 0.0, 0.0)
        glBegin(GL_QUADS)
        glVertex2f(bar_x, bar_y)
        glVertex2f(bar_x + bar_width, bar_y)
        glVertex2f(bar_x + bar_width, bar_y + bar_height)
        glVertex2f(bar_x, bar_y + bar_height)
        glEnd()
        
        glColor3f(0.0, 0.8, 0.0)
        glBegin(GL_QUADS)
        glVertex2f(bar_x, bar_y)
        glVertex2f(bar_x + bar_width * (npc.health / npc.max_health), bar_y)
        glVertex2f(bar_x + bar_width * (npc.health / npc.max_health), bar_y + bar_height)
        glVertex2f(bar_x, bar_y + bar_height)
        glEnd()
        
        # Hunger bar with label
        bar_y -= bar_spacing
        try:
            hunger_label = pyglet.text.Label(
                "Hunger:",
                font_name='Courier New',
                font_size=10,
                x=int(bar_x),
                y=int(bar_y + bar_height + bar_label_offset),
                anchor_x='left',
                anchor_y='bottom',
                color=(255, 255, 255, 255)
            )
            hunger_label.draw()
        except:
            pass
        
        glColor3f(0.5, 0.3, 0.0)
        glBegin(GL_QUADS)
        glVertex2f(bar_x, bar_y)
        glVertex2f(bar_x + bar_width, bar_y)
        glVertex2f(bar_x + bar_width, bar_y + bar_height)
        glVertex2f(bar_x, bar_y + bar_height)
        glEnd()
        
        glColor3f(1.0, 0.8, 0.0)
        glBegin(GL_QUADS)
        glVertex2f(bar_x, bar_y)
        glVertex2f(bar_x + bar_width * (npc.hunger / npc.max_hunger), bar_y)
        glVertex2f(bar_x + bar_width * (npc.hunger / npc.max_hunger), bar_y + bar_height)
        glVertex2f(bar_x, bar_y + bar_height)
        glEnd()
        
        # Stamina bar with label
        bar_y -= bar_spacing
        try:
            stamina_label = pyglet.text.Label(
                "Stamina:",
                font_name='Courier New',
                font_size=10,
                x=int(bar_x),
                y=int(bar_y + bar_height + bar_label_offset),
                anchor_x='left',
                anchor_y='bottom',
                color=(255, 255, 255, 255)
            )
            stamina_label.draw()
        except:
            pass
        
        glColor3f(0.3, 0.3, 0.5)
        glBegin(GL_QUADS)
        glVertex2f(bar_x, bar_y)
        glVertex2f(bar_x + bar_width, bar_y)
        glVertex2f(bar_x + bar_width, bar_y + bar_height)
        glVertex2f(bar_x, bar_y + bar_height)
        glEnd()
        
        glColor3f(0.2, 0.4, 1.0)
        glBegin(GL_QUADS)
        glVertex2f(bar_x, bar_y)
        glVertex2f(bar_x + bar_width * (npc.stamina / npc.max_stamina) if npc.max_stamina > 0 else 0, bar_y)
        glVertex2f(bar_x + bar_width * (npc.stamina / npc.max_stamina) if npc.max_stamina > 0 else 0, bar_y + bar_height)
        glVertex2f(bar_x, bar_y + bar_height)
        glEnd()
        
        # Restore OpenGL state (but keep 2D matrices for text)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        
        glPopMatrix()
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        
        # Render text labels using pyglet labels - keep 2D matrices active
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, self.window.width, 0, self.window.height, -1, 1)
        
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()
        
        glDisable(GL_LIGHTING)
        glDisable(GL_DEPTH_TEST)
        
        # Calculate text area (leave bottom 350 pixels for bars and NN visualization)
        bar_area_height = 200  # Space reserved for bars at bottom
        nn_area_height = 150  # Space for neural network visualization
        total_bottom_area = bar_area_height + nn_area_height
        text_start_y = panel_y + panel_height - 20
        text_end_y = panel_y + total_bottom_area  # Stop text before bar/NN area
        
        text_y = text_start_y
        for i, line in enumerate(lines):
            # Stop rendering text if we've reached the reserved area
            if text_y < text_end_y:
                break
                
            if line.strip():
                # Create and draw label immediately
                try:
                    label = pyglet.text.Label(
                        line,
                        font_name='Courier New',
                        font_size=11,
                        x=int(panel_x + 15),
                        y=int(text_y),
                        anchor_x='left',
                        anchor_y='bottom',
                        color=(255, 255, 255, 255),
                        multiline=False
                    )
                    label.draw()
                except:
                    # Fallback: try with default font
                    try:
                        label = pyglet.text.Label(
                            line,
                            font_size=11,
                            x=int(panel_x + 15),
                            y=int(text_y),
                            anchor_x='left',
                            anchor_y='bottom',
                            color=(255, 255, 255, 255)
                        )
                        label.draw()
                    except:
                        pass  # Skip if font rendering fails
                text_y -= 16  # Slightly increased spacing
            else:
                text_y -= 6  # Slightly increased spacing for blank lines
        
        # Render neural network visualization above bars
        self._render_neural_network_visualization(npc, panel_x, panel_y + bar_area_height, panel_width, nn_area_height)
        
        # Restore OpenGL state
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        
        glPopMatrix()
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
    
    def _render_neural_network_visualization(self, npc, x, y, width, height):
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
        
        # Setup 2D rendering (already in 2D mode, but ensure lighting is off)
        glDisable(GL_LIGHTING)
        glDisable(GL_DEPTH_TEST)
        
        # Title
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
        
        # Get network structure
        brain = npc.brain
        
        # Layer sizes (simplified visualization - show subset of neurons)
        layer_sizes = [min(12, brain.input_size), 8, 8, 6, 6]  # Reduced for visibility
        
        # Calculate positions
        num_layers = len(layer_sizes)
        layer_spacing = (width - 40) / max(1, num_layers - 1)
        neuron_radius = 3.0
        vis_height = height - 50  # Leave space for title and info
        
        # Draw connections first (so neurons appear on top)
        glLineWidth(0.5)
        for layer_idx in range(num_layers - 1):
            num_neurons_current = layer_sizes[layer_idx]
            num_neurons_next = layer_sizes[layer_idx + 1]
            
            layer_x = x + 20 + layer_idx * layer_spacing
            next_layer_x = x + 20 + (layer_idx + 1) * layer_spacing
            
            # Get weights for this layer connection
            layer_weights = None
            try:
                if layer_idx == 0:
                    layer_weights = brain.fc1.weight.data.cpu().numpy()
                elif layer_idx == 1:
                    layer_weights = brain.fc2.weight.data.cpu().numpy()
                elif layer_idx == 2:
                    layer_weights = brain.fc3.weight.data.cpu().numpy()
                elif layer_idx == 3:
                    layer_weights = brain.fc_action.weight.data.cpu().numpy()
                
                if layer_weights is not None:
                    # Normalize weights for visualization
                    weight_max = max(abs(layer_weights.max()), abs(layer_weights.min()))
                    if weight_max > 0:
                        layer_weights = layer_weights / weight_max
            except:
                layer_weights = None
            
            for neuron_idx_current in range(num_neurons_current):
                neuron_y_current = y + 20 + (neuron_idx_current + 1) * vis_height / (num_neurons_current + 1)
                
                for neuron_idx_next in range(num_neurons_next):
                    neuron_y_next = y + 20 + (neuron_idx_next + 1) * vis_height / (num_neurons_next + 1)
                    
                    # Get weight strength
                    if layer_weights is not None and neuron_idx_current < layer_weights.shape[1] and neuron_idx_next < layer_weights.shape[0]:
                        weight = layer_weights[neuron_idx_next, neuron_idx_current]
                    else:
                        weight = 0.0
                    
                    # Color based on weight (green = positive, red = negative, gray = near zero)
                    weight_abs = abs(weight)
                    if weight_abs < 0.1:
                        glColor4f(0.2, 0.2, 0.2, 0.1)  # Very faint gray
                    elif weight > 0:
                        glColor4f(0.0, 0.6, 0.0, min(0.6, weight_abs * 0.8))  # Green for positive
                    else:
                        glColor4f(0.6, 0.0, 0.0, min(0.6, weight_abs * 0.8))  # Red for negative
                    
                    glBegin(GL_LINES)
                    glVertex2f(layer_x, neuron_y_current)
                    glVertex2f(next_layer_x, neuron_y_next)
                    glEnd()
        
        # Draw neurons (circles)
        for layer_idx in range(num_layers):
            num_neurons = layer_sizes[layer_idx]
            layer_x = x + 20 + layer_idx * layer_spacing
            
            # Layer label
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
            
            # Draw neurons in this layer
            for neuron_idx in range(num_neurons):
                neuron_y = y + 20 + (neuron_idx + 1) * vis_height / (num_neurons + 1)
                
                # Neuron color based on layer
                if layer_idx == 0:
                    glColor3f(0.3, 0.7, 1.0)  # Blue for input
                elif layer_idx == num_layers - 1:
                    glColor3f(1.0, 0.8, 0.3)  # Yellow for output
                else:
                    glColor3f(0.5, 0.9, 0.5)  # Green for hidden
                
                # Draw neuron circle
                import math
                num_segments = 12
                glBegin(GL_TRIANGLE_FAN)
                glVertex2f(layer_x, neuron_y)
                for i in range(num_segments + 1):
                    angle = (i / num_segments) * 2 * math.pi
                    glVertex2f(
                        layer_x + neuron_radius * math.cos(angle),
                        neuron_y + neuron_radius * math.sin(angle)
                    )
                glEnd()
                
                # Neuron border
                glColor3f(1.0, 1.0, 1.0)
                glLineWidth(1.0)
                glBegin(GL_LINE_LOOP)
                for i in range(num_segments):
                    angle = (i / num_segments) * 2 * math.pi
                    glVertex2f(
                        layer_x + neuron_radius * math.cos(angle),
                        neuron_y + neuron_radius * math.sin(angle)
                    )
                glEnd()
        
        # Add legend/info
        try:
            info_lines = [
                f"Input: {brain.input_size} features",
                f"Hidden: 64→64→32",
                f"Output: 6 actions",
                f"Lines: Green=positive, Red=negative"
            ]
            info_y = y + 5
            for i, info_line in enumerate(info_lines):
                info_label = pyglet.text.Label(
                    info_line,
                    font_name='Courier New',
                    font_size=8,
                    x=int(x + 10),
                    y=int(info_y - i * 12),
                    anchor_x='left',
                    anchor_y='top',
                    color=(180, 180, 180, 255)
                )
                info_label.draw()
        except:
            pass
    
    def _render_debug_overlay(self):
        """Render debug information overlay."""
        # Switch to 2D rendering
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, self.window.width, 0, self.window.height, -1, 1)
        
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()
        
        # Disable lighting and depth test for text
        glDisable(GL_LIGHTING)
        glDisable(GL_DEPTH_TEST)
        
        # Render text
        hour = (self.world.day_time / self.world.day_length) * 24.0
        time_str = f"Day {self.world.day_number + 1} - {hour:.1f}h ({'Night' if self.world.is_night() else 'Day'})"
        alive_count = sum(1 for npc in self.world.entities if npc.is_alive)
        
        # Calculate current color values for debugging
        is_night = hour < 6.0 or hour >= 18.0
        if is_night:
            terrain_r, terrain_g, terrain_b = 0.0, 0.04, 0.08
            sky_r, sky_g, sky_b = 0.03, 0.03, 0.12
            color_info = f"Night Colors - Terrain: RGB({terrain_r:.2f},{terrain_g:.3f},{terrain_b:.3f}) Sky: RGB({sky_r:.2f},{sky_g:.2f},{sky_b:.2f})"
        else:
            terrain_r, terrain_g, terrain_b = 0.2, 0.5, 0.15
            sky_r, sky_g, sky_b = 0.5, 0.7, 1.0
            color_info = f"Day Colors - Terrain: RGB({terrain_r:.2f},{terrain_g:.2f},{terrain_b:.2f}) Sky: RGB({sky_r:.2f},{sky_g:.2f},{sky_b:.2f})"
        
        stats = [
            f"World Simulation Debug",
            f"Time: {time_str}",
            f"NPCs Alive: {alive_count}/{len(self.world.entities)}",
            f"Trees: {len([t for t in self.world.trees if t.is_alive])}",
            f"Houses: {len(self.world.houses)}",
            f"Light Intensity: {self.world.get_light_intensity():.2f}",
            f"",
            f"Color Info: {color_info}",
            f"Press 'T' to toggle color debug logs",
            f"",
            f"NPC States:",
        ]
        
        # Count NPC states
        state_counts = {}
        for npc in self.world.entities:
            if npc.is_alive:
                state_counts[npc.state] = state_counts.get(npc.state, 0) + 1
        
        for state, count in sorted(state_counts.items()):
            stats.append(f"  {state}: {count}")
        
        # Add recent log entries
        stats.append("")
        stats.append("Recent Events:")
        for log_entry in self.debug_log[-5:]:
            stats.append(f"  {log_entry}")
        
        # Render text
        label = pyglet.text.Label(
            '\n'.join(stats),
            font_name='Courier New',
            font_size=12,
            x=10,
            y=self.window.height - 10,
            anchor_x='left',
            anchor_y='top',
            color=(255, 255, 255, 255),
            multiline=True,
            width=self.window.width - 20
        )
        label.draw()
        
        # Restore OpenGL state
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        
        glPopMatrix()
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
    
    def log(self, message: str):
        """Add a message to the debug log."""
        self.debug_log.append(message)
        if len(self.debug_log) > self.max_log_lines:
            self.debug_log.pop(0)
    
    def _draw_sphere(self, radius: float, segments: int):
        """Draw a sphere with proper normals for lighting."""
        for i in range(segments):
            angle1 = i * 2 * math.pi / segments
            angle2 = (i + 1) * 2 * math.pi / segments
            
            glBegin(GL_TRIANGLE_STRIP)
            for j in range(segments // 2 + 1):
                theta = j * math.pi / (segments // 2)
                sin_theta = math.sin(theta)
                cos_theta = math.cos(theta)
                
                # First vertex
                x1 = radius * sin_theta * math.cos(angle1)
                y1 = radius * cos_theta
                z1 = radius * sin_theta * math.sin(angle1)
                # Normal (points outward from center)
                glNormal3f(x1 / radius, y1 / radius, z1 / radius)
                glVertex3f(x1, y1, z1)
                
                # Second vertex
                x2 = radius * sin_theta * math.cos(angle2)
                y2 = radius * cos_theta
                z2 = radius * sin_theta * math.sin(angle2)
                # Normal
                glNormal3f(x2 / radius, y2 / radius, z2 / radius)
                glVertex3f(x2, y2, z2)
            glEnd()
