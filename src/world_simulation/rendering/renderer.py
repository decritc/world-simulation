"""3D rendering implementation using pyglet."""

import pyglet
from pyglet.gl import *
import numpy as np
import math

from ..world.world import World
from ..entities.npc import NPC
from ..trees.tree import FruitTree
from ..houses.house import House
from .detail_panel import DetailPanel


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
        
        # Detail panel
        self.detail_panel = DetailPanel(self.window)
        
        # FPS tracking
        self.fps_display = pyglet.window.FPSDisplay(window=self.window)
        self.frame_count = 0
        self.fps_update_time = 0.0
        self.current_fps = 0.0
        
        # Camera frustum for culling
        self.fov = 70.0
        self.near_plane = 0.1
        self.far_plane = 500.0
        
        # Terrain caching and batching
        self.terrain_display_list = None
        self.last_terrain_x = None
        self.last_terrain_z = None
        self.terrain_cache_radius = 40  # Cache terrain within 40 units
        
        # Spatial partitioning for entities
        self.entity_grid = {}  # Grid-based spatial partitioning
        self.grid_size = 20.0  # 20 unit grid cells
        
        # Performance profiling
        self.profiling = True  # Enable by default for testing
        self.profile_times = {
            'terrain': 0.0,
            'vegetation': 0.0,
            'trees': 0.0,
            'npcs': 0.0,
            'houses': 0.0,
            'overlay': 0.0,
            'total': 0.0
        }
        self.profile_frame_count = 0
        self.profile_log_interval = 120  # Log every 120 frames (2 seconds at 60fps)
        self.profile_log_file = "profiling.log"  # Log file for profiling data
        
        
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
        
        # Screenshot capture
        self.screenshot_count = 0
        import os
        self.screenshot_dir = "screenshots"
        os.makedirs(self.screenshot_dir, exist_ok=True)
        
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
            # Press 'P' to toggle performance profiling
            if symbol == pyglet.window.key.P:
                self.profiling = not self.profiling
                if self.profiling:
                    self.log("Performance profiling ON - press P again to disable")
                    # Reset profiling data
                    self.profile_times = {k: 0.0 for k in self.profile_times}
                    self.profile_frame_count = 0
                else:
                    self.log("Performance profiling OFF")
            # Press 'S' to capture screenshot
            if symbol == pyglet.window.key.S:
                self.capture_screenshot()
        
        @self.window.event
        def on_key_release(symbol, modifiers):
            # Track key release
            self.keys[symbol] = False
        
        @self.window.event
        def on_mouse_press(x, y, button, modifiers):
            self.mouse_buttons.add(button)
            # Left click for NPC selection
            if button == pyglet.window.mouse.LEFT:
                # Always try to pick NPC first (don't check panel area initially)
                clicked_npc = self._pick_npc(x, y)
                
                if clicked_npc:
                    # Successfully clicked on an NPC
                    self.selected_npc = clicked_npc
                    self.detail_panel.scroll = 0  # Reset scroll when selecting new NPC
                    # Reset screenshot flag so we capture when panel is shown
                    self._panel_screenshot_captured = False
                else:
                    # No NPC clicked - check if clicking on panel
                    if self.detail_panel.is_point_in_panel(x, y) and self.selected_npc:
                        # Clicking on panel - don't deselect
                        pass
                    else:
                        # Clicking elsewhere - deselect
                        self.selected_npc = None
                        self.detail_panel.scroll = 0
        
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
            # Check if mouse is over detail panel
            if self.detail_panel.is_point_in_panel(x, y):
                # Scroll detail panel when hovering over it
                self.detail_panel.handle_scroll(x, y, scroll_x, scroll_y)
            else:
                # Zoom with mouse wheel when not over panel
                self.camera_y += scroll_y * 2.0
                self.camera_y = max(5.0, self.camera_y)
    
    def update(self, delta_time: float):
        """Update camera based on keyboard and mouse."""
        # Update FPS
        self.frame_count += 1
        self.fps_update_time += delta_time
        if self.fps_update_time >= 0.5:  # Update FPS every 0.5 seconds
            self.current_fps = self.frame_count / self.fps_update_time
            self.frame_count = 0
            self.fps_update_time = 0.0
        
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
        import time
        frame_start = time.time()
        
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
        gluPerspective(self.fov, self.window.width / self.window.height, self.near_plane, self.far_plane)
        
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
        terrain_start = time.time()
        self._render_terrain()
        terrain_time = time.time() - terrain_start
        
        # Render vegetation (optimized with spatial partitioning)
        veg_start = time.time()
        # Build spatial grid for this frame
        veg_grid = {}
        for veg in self.world.vegetation:
            if not veg.is_alive:
                continue
            grid_x = int(veg.x / self.grid_size)
            grid_z = int(veg.z / self.grid_size)
            key = (grid_x, grid_z)
            if key not in veg_grid:
                veg_grid[key] = []
            veg_grid[key].append(veg)
        
        # Only check vegetation in nearby grid cells
        camera_grid_x = int(self.camera_x / self.grid_size)
        camera_grid_z = int(self.camera_z / self.grid_size)
        veg_count = 0
        max_veg_per_frame = 200  # High limit for quality
        
        for dx in range(-2, 3):  # Check 5x5 grid cells around camera
            for dz in range(-2, 3):
                key = (camera_grid_x + dx, camera_grid_z + dz)
                if key in veg_grid:
                    for veg in veg_grid[key]:
                        if veg_count >= max_veg_per_frame:
                            break
                        # Quick distance check
                        dist_sq = (veg.x - self.camera_x)**2 + (veg.z - self.camera_z)**2
                        if dist_sq < 10000:  # Within 100 units
                            self._render_vegetation(veg)
                            veg_count += 1
                if veg_count >= max_veg_per_frame:
                    break
            if veg_count >= max_veg_per_frame:
                break
        veg_time = time.time() - veg_start
        
        # Render houses (all within distance)
        house_start = time.time()
        for house in self.world.houses:
            dx = house.x - self.camera_x
            dz = house.z - self.camera_z
            dist_sq = dx*dx + dz*dz
            if dist_sq < 10000:  # Within 100 units
                self._render_house(house)
        house_time = time.time() - house_start
        
        # Render trees (optimized with spatial partitioning)
        tree_start = time.time()
        tree_grid = {}
        for tree in self.world.trees:
            if not tree.is_alive:
                continue
            grid_x = int(tree.x / self.grid_size)
            grid_z = int(tree.z / self.grid_size)
            key = (grid_x, grid_z)
            if key not in tree_grid:
                tree_grid[key] = []
            tree_grid[key].append(tree)
        
        tree_count = 0
        max_trees_per_frame = 100  # High limit for quality
        
        for dx in range(-2, 3):
            for dz in range(-2, 3):
                key = (camera_grid_x + dx, camera_grid_z + dz)
                if key in tree_grid:
                    for tree in tree_grid[key]:
                        if tree_count >= max_trees_per_frame:
                            break
                        dist_sq = (tree.x - self.camera_x)**2 + (tree.z - self.camera_z)**2
                        if dist_sq < 10000:  # Within 100 units
                            self._render_tree(tree)
                            tree_count += 1
                if tree_count >= max_trees_per_frame:
                    break
            if tree_count >= max_trees_per_frame:
                break
        tree_time = time.time() - tree_start
        
        # Render NPCs (optimized with spatial partitioning)
        npc_start = time.time()
        npc_grid = {}
        for npc in self.world.entities:
            if not npc.is_alive:
                continue
            grid_x = int(npc.x / self.grid_size)
            grid_z = int(npc.z / self.grid_size)
            key = (grid_x, grid_z)
            if key not in npc_grid:
                npc_grid[key] = []
            npc_grid[key].append(npc)
        
        npc_count = 0
        max_npcs_per_frame = 150  # High limit for quality
        
        for dx in range(-2, 3):
            for dz in range(-2, 3):
                key = (camera_grid_x + dx, camera_grid_z + dz)
                if key in npc_grid:
                    for npc in npc_grid[key]:
                        if npc_count >= max_npcs_per_frame:
                            break
                        dist_sq = (npc.x - self.camera_x)**2 + (npc.z - self.camera_z)**2
                        if dist_sq < 10000:  # Within 100 units
                            self._render_npc(npc)
                            npc_count += 1
                if npc_count >= max_npcs_per_frame:
                    break
            if npc_count >= max_npcs_per_frame:
                break
        npc_time = time.time() - npc_start
        
        # Render debug overlay
        overlay_start = time.time()
        self._render_debug_overlay()
        overlay_time = time.time() - overlay_start
        
        # Render NPC detail panel if NPC is selected (render after debug overlay for proper layering)
        if self.selected_npc and self.selected_npc.is_alive:
            self.detail_panel.render(self.selected_npc)
            # Auto-capture screenshot once when panel is first shown
            if not hasattr(self, '_panel_screenshot_captured'):
                self.capture_screenshot()
                self._panel_screenshot_captured = True
        
        # Update profiling data
        total_time = time.time() - frame_start
        self.profile_frame_count += 1
        
        # Update rolling averages (every 60 frames)
        if self.profile_frame_count % 60 == 0:
            alpha = 0.1  # Smoothing factor
            self.profile_times['terrain'] = self.profile_times['terrain'] * (1 - alpha) + terrain_time * alpha
            self.profile_times['vegetation'] = self.profile_times['vegetation'] * (1 - alpha) + veg_time * alpha
            self.profile_times['trees'] = self.profile_times['trees'] * (1 - alpha) + tree_time * alpha
            self.profile_times['npcs'] = self.profile_times['npcs'] * (1 - alpha) + npc_time * alpha
            self.profile_times['houses'] = self.profile_times['houses'] * (1 - alpha) + house_time * alpha
            self.profile_times['overlay'] = self.profile_times['overlay'] * (1 - alpha) + overlay_time * alpha
            self.profile_times['total'] = self.profile_times['total'] * (1 - alpha) + total_time * alpha
            
            # Log profiling data to console (every N frames for visibility)
            if self.profiling and total_time > 0 and self.profile_frame_count % self.profile_log_interval == 0:
                fps = 1.0 / total_time if total_time > 0 else 0
                avg_fps = 1.0 / self.profile_times['total'] if self.profile_times['total'] > 0 else 0
                
                report_lines = []
                report_lines.append("\n" + "="*60)
                report_lines.append(f"PERFORMANCE PROFILING REPORT (Frame {self.profile_frame_count})")
                report_lines.append("="*60)
                report_lines.append(f"Current FPS: {fps:.1f} | Average FPS (last 60 frames): {avg_fps:.1f}")
                report_lines.append(f"Target: 60 FPS | Status: {'MEETS TARGET' if fps >= 60 else 'BELOW TARGET'}")
                report_lines.append("-"*60)
                report_lines.append(f"Total Frame Time: {total_time*1000:.2f}ms ({100:.0f}%)")
                report_lines.append(f"  - Terrain:       {terrain_time*1000:.2f}ms ({terrain_time/total_time*100:.1f}%) - Avg: {self.profile_times['terrain']*1000:.2f}ms")
                report_lines.append(f"  - Vegetation:    {veg_time*1000:.2f}ms ({veg_time/total_time*100:.1f}%) - Avg: {self.profile_times['vegetation']*1000:.2f}ms")
                report_lines.append(f"  - Trees:         {tree_time*1000:.2f}ms ({tree_time/total_time*100:.1f}%) - Avg: {self.profile_times['trees']*1000:.2f}ms")
                report_lines.append(f"  - NPCs:          {npc_time*1000:.2f}ms ({npc_time/total_time*100:.1f}%) - Avg: {self.profile_times['npcs']*1000:.2f}ms")
                report_lines.append(f"  - Houses:        {house_time*1000:.2f}ms ({house_time/total_time*100:.1f}%) - Avg: {self.profile_times['houses']*1000:.2f}ms")
                report_lines.append(f"  - Overlay:       {overlay_time*1000:.2f}ms ({overlay_time/total_time*100:.1f}%) - Avg: {self.profile_times['overlay']*1000:.2f}ms")
                report_lines.append("="*60)
                
                # Identify bottlenecks
                components = [
                    ('Terrain', terrain_time, self.profile_times['terrain']),
                    ('Vegetation', veg_time, self.profile_times['vegetation']),
                    ('Trees', tree_time, self.profile_times['trees']),
                    ('NPCs', npc_time, self.profile_times['npcs']),
                    ('Houses', house_time, self.profile_times['houses']),
                    ('Overlay', overlay_time, self.profile_times['overlay']),
                ]
                components.sort(key=lambda x: x[1], reverse=True)
                report_lines.append("\nTop Bottlenecks:")
                for i, (name, current_time, avg_time) in enumerate(components[:3], 1):
                    report_lines.append(f"  {i}. {name}: {current_time*1000:.2f}ms ({current_time/total_time*100:.1f}%) - Avg: {avg_time*1000:.2f}ms")
                report_lines.append("")
                
                # Print to console
                report_text = "\n".join(report_lines)
                print(report_text)
                
                # Write to log file
                try:
                    with open(self.profile_log_file, 'a', encoding='utf-8') as f:
                        f.write(report_text + "\n")
                except:
                    pass  # Ignore file write errors
                
                # Log to debug overlay as well
                self.log(f"FPS: {fps:.1f} | Terrain: {terrain_time*1000:.1f}ms ({terrain_time/total_time*100:.0f}%)")
                self.log(f"Veg: {veg_time*1000:.1f}ms ({veg_time/total_time*100:.0f}%) | Trees: {tree_time*1000:.1f}ms ({tree_time/total_time*100:.0f}%)")
                self.log(f"NPCs: {npc_time*1000:.1f}ms ({npc_time/total_time*100:.0f}%) | Houses: {house_time*1000:.1f}ms | Overlay: {overlay_time*1000:.1f}ms")
    
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
        
        # Render terrain with high quality LOD system and caching
        terrain_size = 200  # Full render area
        base_resolution = 200  # Increased resolution for smoother terrain
        
        # Use smooth shading for better visual quality
        glShadeModel(GL_SMOOTH)
        
        # LOD zones: (min_distance, max_distance, resolution)
        lod_zones = [
            (0, 60, base_resolution),      # High detail: 0-60 units (200x200 = 40,000 quads)
            (60, 120, base_resolution // 2),  # Medium detail: 60-120 units (100x100 = 10,000 quads)
            (120, terrain_size, base_resolution // 4),  # Low detail: 120+ units (50x50 = 2,500 quads)
        ]
        
        # Check if we can reuse cached terrain display list
        camera_grid_x = int(self.camera_x / self.terrain_cache_radius)
        camera_grid_z = int(self.camera_z / self.terrain_cache_radius)
        
        # Invalidate cache if camera moved significantly
        if (self.last_terrain_x != camera_grid_x or 
            self.last_terrain_z != camera_grid_z or 
            self.terrain_display_list is None):
            
            # Delete old display list if it exists
            if self.terrain_display_list is not None:
                glDeleteLists(self.terrain_display_list, 1)
            
            # Create new display list for terrain
            self.terrain_display_list = glGenLists(1)
            glNewList(self.terrain_display_list, GL_COMPILE)
            
            # Render each LOD zone
            for zone_min, zone_max, resolution in lod_zones:
                if resolution < 4:
                    continue
                
                zone_size = zone_max - zone_min
                if zone_size <= 0:
                    continue
                
                # Calculate zone bounds
                zone_start_x = self.camera_x - zone_max / 2
                zone_start_z = self.camera_z - zone_max / 2
                step = zone_size / resolution
                
                # Pre-calculate ALL heights for this zone (single pass)
                height_grid = {}
                for z in range(resolution + 1):
                    for x in range(resolution + 1):
                        world_x = zone_start_x + x * step
                        world_z = zone_start_z + z * step
                        
                        # Quick squared distance check
                        dx = world_x - self.camera_x
                        dz = world_z - self.camera_z
                        dist_sq = dx*dx + dz*dz
                        
                        if dist_sq < zone_min*zone_min or dist_sq >= zone_max*zone_max:
                            continue
                        
                        height_grid[(x, z)] = self.world.get_height(world_x, world_z)
                
                # Batch render terrain for this LOD zone
                glBegin(GL_QUADS)
                for z in range(resolution):
                    for x in range(resolution):
                        world_x1 = zone_start_x + x * step
                        world_z1 = zone_start_z + z * step
                        
                        # Quick squared distance check
                        dx = world_x1 - self.camera_x
                        dz = world_z1 - self.camera_z
                        dist_sq = dx*dx + dz*dz
                        dist = math.sqrt(dist_sq) if dist_sq > 0 else 0
                        
                        if dist < zone_min or dist >= zone_max:
                            continue
                        
                        # Get heights at corners
                        h1 = height_grid.get((x, z))
                        h2 = height_grid.get((x + 1, z))
                        h3 = height_grid.get((x + 1, z + 1))
                        h4 = height_grid.get((x, z + 1))
                        
                        if None in (h1, h2, h3, h4):
                            continue
                        
                        # Get terrain type for color
                        terrain_type = self.world.generator.get_terrain_type(
                            h1 / self.world.generator.max_height
                        )
                        
                        # Set color based on terrain type with variety
                        if not is_night:
                            if terrain_type == 'water':
                                glColor3f(0.1, 0.2, 0.4)  # Dark blue water
                            elif terrain_type == 'sand':
                                glColor3f(0.7, 0.65, 0.5)  # Light sandy color
                            elif terrain_type == 'dirt':
                                glColor3f(0.4, 0.3, 0.2)  # Brown dirt
                            elif terrain_type == 'grass':
                                glColor3f(0.2, 0.5, 0.15)  # Green grass
                            elif terrain_type == 'hill':
                                glColor3f(0.3, 0.45, 0.2)  # Darker green hills
                            elif terrain_type == 'mountain':
                                glColor3f(0.35, 0.35, 0.3)  # Gray-brown mountains
                            else:  # snow/peak
                                glColor3f(0.9, 0.9, 0.95)  # Light gray/white snow
                        else:
                            # Night colors - darker, cooler tones
                            if terrain_type == 'water':
                                glColor3f(0.0, 0.02, 0.04)
                            elif terrain_type == 'sand':
                                glColor3f(0.02, 0.02, 0.03)
                            elif terrain_type == 'dirt':
                                glColor3f(0.01, 0.02, 0.02)
                            elif terrain_type == 'grass':
                                glColor3f(0.0, 0.04, 0.08)
                            elif terrain_type == 'hill':
                                glColor3f(0.0, 0.03, 0.06)
                            elif terrain_type == 'mountain':
                                glColor3f(0.01, 0.02, 0.03)
                            else:  # snow/peak
                                glColor3f(0.05, 0.05, 0.06)
                        
                        # Calculate smooth per-vertex normals efficiently (simplified)
                        # Use gradient-based normal calculation from height grid
                        h_right = height_grid.get((x + 1, z), h1)
                        h_left = height_grid.get((x - 1, z), h1) if x > 0 else h1
                        h_up = height_grid.get((x, z + 1), h1)
                        h_down = height_grid.get((x, z - 1), h1) if z > 0 else h1
                        
                        # Calculate normal from gradients (faster than full cross product)
                        dh_dx = (h_right - h_left) / (2 * step) if step > 0 else 0
                        dh_dz = (h_up - h_down) / (2 * step) if step > 0 else 0
                        nx = -dh_dx
                        nz = -dh_dz
                        length_inv = 1.0 / math.sqrt(nx*nx + 1.0 + nz*nz) if (nx*nx + 1.0 + nz*nz) > 0.0001 else 1.0
                        n1 = (nx * length_inv, 1.0 * length_inv, nz * length_inv)
                        
                        # Same calculation for other vertices (can optimize further)
                        h_right2 = height_grid.get((x + 1, z), h2)
                        h_left2 = height_grid.get((x - 1, z), h1) if x > 0 else h1
                        h_up2 = height_grid.get((x + 1, z + 1), h2)
                        h_down2 = height_grid.get((x + 1, z - 1), h2) if z > 0 else h2
                        dh_dx2 = (h_right2 - h_left2) / (2 * step) if step > 0 else 0
                        dh_dz2 = (h_up2 - h_down2) / (2 * step) if step > 0 else 0
                        nx2 = -dh_dx2
                        nz2 = -dh_dz2
                        length_inv2 = 1.0 / math.sqrt(nx2*nx2 + 1.0 + nz2*nz2) if (nx2*nx2 + 1.0 + nz2*nz2) > 0.0001 else 1.0
                        n2 = (nx2 * length_inv2, 1.0 * length_inv2, nz2 * length_inv2)
                        
                        h_right3 = height_grid.get((x + 1, z + 1), h3)
                        h_left3 = height_grid.get((x, z + 1), h4)
                        h_up3 = height_grid.get((x + 1, z + 2), h3) if z + 1 < resolution else h3
                        h_down3 = height_grid.get((x + 1, z), h2)
                        dh_dx3 = (h_right3 - h_left3) / (2 * step) if step > 0 else 0
                        dh_dz3 = (h_up3 - h_down3) / (2 * step) if step > 0 else 0
                        nx3 = -dh_dx3
                        nz3 = -dh_dz3
                        length_inv3 = 1.0 / math.sqrt(nx3*nx3 + 1.0 + nz3*nz3) if (nx3*nx3 + 1.0 + nz3*nz3) > 0.0001 else 1.0
                        n3 = (nx3 * length_inv3, 1.0 * length_inv3, nz3 * length_inv3)
                        
                        h_right4 = height_grid.get((x, z + 1), h4)
                        h_left4 = height_grid.get((x - 1, z + 1), h4) if x > 0 else h4
                        h_up4 = height_grid.get((x, z + 2), h4) if z + 1 < resolution else h4
                        h_down4 = height_grid.get((x, z), h1)
                        dh_dx4 = (h_right4 - h_left4) / (2 * step) if step > 0 else 0
                        dh_dz4 = (h_up4 - h_down4) / (2 * step) if step > 0 else 0
                        nx4 = -dh_dx4
                        nz4 = -dh_dz4
                        length_inv4 = 1.0 / math.sqrt(nx4*nx4 + 1.0 + nz4*nz4) if (nx4*nx4 + 1.0 + nz4*nz4) > 0.0001 else 1.0
                        n4 = (nx4 * length_inv4, 1.0 * length_inv4, nz4 * length_inv4)
                        
                        # Draw quad with per-vertex normals for smooth shading
                        glNormal3f(*n1)
                        glVertex3f(world_x1, h1, world_z1)
                        glNormal3f(*n2)
                        glVertex3f(world_x1 + step, h2, world_z1)
                        glNormal3f(*n3)
                        glVertex3f(world_x1 + step, h3, world_z1 + step)
                        glNormal3f(*n4)
                        glVertex3f(world_x1, h4, world_z1 + step)
                
                glEnd()
            
            glEndList()
            self.last_terrain_x = camera_grid_x
            self.last_terrain_z = camera_grid_z
        
        # Render cached terrain display list (much faster than immediate mode)
        glCallList(self.terrain_display_list)
        
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
            self._draw_sphere(main_foliage_size, 10)  # Reduced segments for performance
            glPopMatrix()
            
            # Secondary foliage clusters for fuller look (simplified)
            if tree.growth_stage >= 0.7:
                # Medium cluster slightly lower
                glPushMatrix()
                glTranslatef(0, foliage_base_height - 0.3 * tree.growth_stage, 0)
                self._draw_sphere(main_foliage_size * 0.85, 8)  # Reduced segments
                glPopMatrix()
            
            if tree.growth_stage >= 0.85:
                # Smaller clusters around the sides (only 2 instead of 3 for performance)
                for i in range(2):  # Reduced from 3 to 2
                    angle = (i / 2.0) * 2 * math.pi
                    offset_x = math.cos(angle) * 0.4 * tree.growth_stage
                    offset_z = math.sin(angle) * 0.4 * tree.growth_stage
                    glPushMatrix()
                    glTranslatef(offset_x, foliage_base_height - 0.2 * tree.growth_stage, offset_z)
                    self._draw_sphere(main_foliage_size * 0.6, 6)  # Reduced segments
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
                
                # Render all fruit pieces (limit to reduce overhead)
                for pos in fruit_positions[:20]:  # Reduced from 30 to 20
                    glPushMatrix()
                    glTranslatef(pos[0], trunk_height + pos[1], pos[2])
                    if pos[3]:  # Ripe fruit - larger and brighter
                        self._draw_sphere(0.18, 8)  # Reduced segments
                    else:  # Growing fruit - smaller
                        glColor3f(0.8, 0.2, 0.2)
                        self._draw_sphere(0.12, 6)  # Reduced segments
                        glColor3f(1.0, 0.1, 0.1)  # Reset color
                    glPopMatrix()
        
        glPopMatrix()
    
    def _render_npc(self, npc: NPC):
        """Render an NPC with a simplified model for performance."""
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
            base_r = 0.1 * (1.0 - health_ratio)
            base_g = 0.15 * health_ratio + 0.1
            base_b = 0.2 * health_ratio + 0.15
            saturation = 0.3 + hunger_ratio * 0.3
            r = base_r * saturation
            g = base_g * saturation
            b = base_b * saturation
        else:
            base_g = health_ratio * 0.6 + 0.3
            base_b = health_ratio * 0.7 + 0.4
            base_r = (1.0 - health_ratio) * 0.2
            hunger_factor = hunger_ratio * 0.3
            stamina_factor = npc.genome.get('stamina', 100.0) / 150.0 * 0.2
            saturation = 0.6 + hunger_ratio * 0.4
            r = base_r * saturation
            g = (base_g + hunger_factor) * saturation
            b = (base_b + stamina_factor) * saturation
        
        # Set material properties (batch set once)
        glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, (GLfloat * 4)(r * 0.3, g * 0.3, b * 0.3, 1.0))
        glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, (GLfloat * 4)(r, g, b, 1.0))
        glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, (GLfloat * 4)(0.5, 0.5, 0.5, 1.0))
        glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 30.0)
        glColor3f(r, g, b)
        
        # Scale based on NPC size (children are smaller)
        scale = npc.size
        if hasattr(npc, 'age_stage') and npc.age_stage == "child":
            scale *= 0.7
        
        # Simplified NPC model - batch render all parts together
        # Head (sphere)
        glPushMatrix()
        glTranslatef(0.0, 0.6 * scale, 0.0)
        self._draw_sphere(0.15 * scale, 8)  # Reduced segments for performance
        glPopMatrix()
        
        # Body/Torso (simplified cube - batch rendered)
        body_height = 0.4 * scale
        body_width = 0.2 * scale
        body_depth = 0.15 * scale
        glPushMatrix()
        glTranslatef(0.0, 0.25 * scale, 0.0)
        self._draw_box_fast(body_width, body_height, body_depth)
        glPopMatrix()
        
        # Arms and legs (simplified - fewer quads)
        arm_size = 0.08 * scale
        leg_size = 0.1 * scale
        
        # Left arm
        glPushMatrix()
        glTranslatef(-body_width * 1.2, 0.3 * scale, 0.0)
        self._draw_box_fast(arm_size, 0.3 * scale, arm_size)
        glPopMatrix()
        
        # Right arm
        glPushMatrix()
        glTranslatef(body_width * 1.2, 0.3 * scale, 0.0)
        self._draw_box_fast(arm_size, 0.3 * scale, arm_size)
        glPopMatrix()
        
        # Left leg
        glPushMatrix()
        glTranslatef(-body_width * 0.6, -0.2 * scale, 0.0)
        self._draw_box_fast(leg_size, 0.4 * scale, leg_size)
        glPopMatrix()
        
        # Right leg
        glPushMatrix()
        glTranslatef(body_width * 0.6, -0.2 * scale, 0.0)
        self._draw_box_fast(leg_size, 0.4 * scale, leg_size)
        glPopMatrix()
        
        glPopMatrix()
        
        # Render stat bars above NPC
        self._render_npc_stats(npc)
        
        # Highlight selected NPC with a ring/outline
        if self.selected_npc == npc:
            self._render_npc_highlight(npc)
    
    def _draw_box_fast(self, width: float, height: float, depth: float):
        """Fast box rendering using minimal quads (optimized for performance)."""
        w, h, d = width/2, height/2, depth/2
        
        # Render box with fewer quads - only visible faces
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
        
        # Left face
        glNormal3f(-1.0, 0.0, 0.0)
        glVertex3f(-w, -h, -d)
        glVertex3f(-w, -h, d)
        glVertex3f(-w, h, d)
        glVertex3f(-w, h, -d)
        
        # Right face
        glNormal3f(1.0, 0.0, 0.0)
        glVertex3f(w, -h, -d)
        glVertex3f(w, h, -d)
        glVertex3f(w, h, d)
        glVertex3f(w, -h, d)
        
        glEnd()
    
    def _draw_box(self, width: float, height: float, depth: float):
        """Draw a box (cube) with proper normals."""
        self._draw_box_fast(width, height, depth)
    
    def _render_npc_stats(self, npc: NPC):
        """Render health, hunger, and stamina bars above NPC (optimized)."""
        if not npc.is_alive:
            return
        
        # Only render stats for NPCs near camera to save performance
        dx = npc.x - self.camera_x
        dz = npc.z - self.camera_z
        dist_sq = dx*dx + dz*dz
        if dist_sq > 2500:  # Only render bars within 50 units
            return
        
        # Calculate screen position (simplified - always render in 3D space above NPC)
        bar_height = 0.8
        bar_width = 0.4
        bar_thickness = 0.05
        spacing = 0.1
        
        glPushMatrix()
        glTranslatef(npc.x, npc.y + bar_height, npc.z)
        
        # Make bars face camera (billboard effect) - simplified
        dx = self.camera_x - npc.x
        dz = self.camera_z - npc.z
        angle = math.degrees(math.atan2(dx, dz))
        glRotatef(-angle, 0.0, 1.0, 0.0)
        
        glDisable(GL_LIGHTING)
        
        # Health bar (red background)
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
        
        # Hunger bar (brown background)
        hunger_ratio = npc.hunger / npc.max_hunger
        glColor3f(0.5, 0.3, 0.0)
        glBegin(GL_QUADS)
        glVertex3f(-bar_width/2, -spacing - bar_thickness, 0)
        glVertex3f(bar_width/2, -spacing - bar_thickness, 0)
        glVertex3f(bar_width/2, -spacing, 0)
        glVertex3f(-bar_width/2, -spacing, 0)
        glEnd()
        
        # Hunger fill (yellow)
        glColor3f(1.0, 0.8, 0.0)
        glBegin(GL_QUADS)
        glVertex3f(-bar_width/2, -spacing - bar_thickness, 0.001)
        glVertex3f(-bar_width/2 + bar_width * hunger_ratio, -spacing - bar_thickness, 0.001)
        glVertex3f(-bar_width/2 + bar_width * hunger_ratio, -spacing, 0.001)
        glVertex3f(-bar_width/2, -spacing, 0.001)
        glEnd()
        
        # Stamina bar (blue background)
        stamina_ratio = npc.stamina / npc.max_stamina if npc.max_stamina > 0 else 0
        glColor3f(0.0, 0.0, 0.5)
        glBegin(GL_QUADS)
        glVertex3f(-bar_width/2, -spacing * 2 - bar_thickness * 2, 0)
        glVertex3f(bar_width/2, -spacing * 2 - bar_thickness * 2, 0)
        glVertex3f(bar_width/2, -spacing * 2 - bar_thickness, 0)
        glVertex3f(-bar_width/2, -spacing * 2 - bar_thickness, 0)
        glEnd()
        
        # Stamina fill (cyan)
        glColor3f(0.0, 0.8, 1.0)
        glBegin(GL_QUADS)
        glVertex3f(-bar_width/2, -spacing * 2 - bar_thickness * 2, 0.001)
        glVertex3f(-bar_width/2 + bar_width * stamina_ratio, -spacing * 2 - bar_thickness * 2, 0.001)
        glVertex3f(-bar_width/2 + bar_width * stamina_ratio, -spacing * 2 - bar_thickness, 0.001)
        glVertex3f(-bar_width/2, -spacing * 2 - bar_thickness, 0.001)
        glEnd()
        
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
    
    def _render_vegetation(self, veg):
        """Render a vegetation instance (bush, grass, flower, rock)."""
        hour = (self.world.day_time / self.world.day_length) * 24.0
        is_night = hour < 6.0 or hour >= 18.0
        
        glPushMatrix()
        glTranslatef(veg.x, veg.y * self.world.generator.max_height, veg.z)
        
        # Set material based on vegetation type
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        
        if veg.vegetation_type == 'bush':
            if is_night:
                glColor3f(0.0, 0.08, 0.12)
            else:
                glColor3f(0.2, 0.4, 0.1)
            # Render as small sphere
            self._render_sphere(veg.size)
        elif veg.vegetation_type == 'grass':
            if is_night:
                glColor3f(0.0, 0.05, 0.08)
            else:
                glColor3f(0.15, 0.5, 0.1)
            # Render as small cylinder
            self._render_grass_patch(veg.size)
        elif veg.vegetation_type == 'flower':
            if is_night:
                glColor3f(0.0, 0.06, 0.1)
            else:
                glColor3f(0.8, 0.6, 0.2)  # Yellow flower
            # Render as small sphere with stem
            self._render_flower(veg.size)
        elif veg.vegetation_type == 'rock':
            if is_night:
                glColor3f(0.05, 0.05, 0.08)
            else:
                glColor3f(0.4, 0.4, 0.35)
            # Render as small cube
            self._render_rock(veg.size)
        
        glPopMatrix()
    
    def _render_sphere(self, radius):
        """Render a simple sphere."""
        import math
        segments = 8
        for i in range(segments):
            lat1 = math.pi * (-0.5 + i / segments)
            lat2 = math.pi * (-0.5 + (i + 1) / segments)
            
            glBegin(GL_QUAD_STRIP)
            for j in range(segments + 1):
                lng = 2 * math.pi * j / segments
                x1 = radius * math.cos(lat1) * math.cos(lng)
                y1 = radius * math.sin(lat1)
                z1 = radius * math.cos(lat1) * math.sin(lng)
                x2 = radius * math.cos(lat2) * math.cos(lng)
                y2 = radius * math.sin(lat2)
                z2 = radius * math.cos(lat2) * math.sin(lng)
                glVertex3f(x1, y1, z1)
                glVertex3f(x2, y2, z2)
            glEnd()
    
    def _render_grass_patch(self, size):
        """Render a grass patch as small vertical quads."""
        import math
        num_blades = 3
        for i in range(num_blades):
            angle = (i / num_blades) * 2 * math.pi
            glPushMatrix()
            glRotatef(angle * 180 / math.pi, 0, 1, 0)
            glBegin(GL_QUADS)
            glVertex3f(-size * 0.1, 0, 0)
            glVertex3f(size * 0.1, 0, 0)
            glVertex3f(size * 0.1, size * 0.5, 0)
            glVertex3f(-size * 0.1, size * 0.5, 0)
            glEnd()
            glPopMatrix()
    
    def _render_flower(self, size):
        """Render a flower with petals."""
        # Stem
        glColor3f(0.0, 0.3, 0.0)
        glBegin(GL_QUADS)
        glVertex3f(-size * 0.05, 0, 0)
        glVertex3f(size * 0.05, 0, 0)
        glVertex3f(size * 0.05, size * 0.3, 0)
        glVertex3f(-size * 0.05, size * 0.3, 0)
        glEnd()
        
        # Petals (small spheres)
        hour = (self.world.day_time / self.world.day_length) * 24.0
        is_night = hour < 6.0 or hour >= 18.0
        if is_night:
            glColor3f(0.0, 0.06, 0.1)
        else:
            glColor3f(0.8, 0.6, 0.2)
        glPushMatrix()
        glTranslatef(0, size * 0.3, 0)
        self._render_sphere(size * 0.2)
        glPopMatrix()
    
    def _render_rock(self, size):
        """Render a rock as an irregular cube."""
        glBegin(GL_QUADS)
        # Top
        glVertex3f(-size, size, -size)
        glVertex3f(size, size, -size)
        glVertex3f(size, size, size)
        glVertex3f(-size, size, size)
        # Bottom
        glVertex3f(-size, 0, size)
        glVertex3f(size, 0, size)
        glVertex3f(size, 0, -size)
        glVertex3f(-size, 0, -size)
        # Sides
        glVertex3f(-size, 0, -size)
        glVertex3f(-size, size, -size)
        glVertex3f(-size, size, size)
        glVertex3f(-size, 0, size)
        glVertex3f(size, 0, size)
        glVertex3f(size, size, size)
        glVertex3f(size, size, -size)
        glVertex3f(size, 0, -size)
        glVertex3f(-size, 0, -size)
        glVertex3f(size, 0, -size)
        glVertex3f(size, size, -size)
        glVertex3f(-size, size, -size)
        glVertex3f(-size, 0, size)
        glVertex3f(-size, size, size)
        glVertex3f(size, size, size)
        glVertex3f(size, 0, size)
        glEnd()
    
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
        # Fix: properly calculate ray in camera space then transform to world space
        ray_x = forward_x + right_x * norm_x * tan_fov * aspect + up_x * norm_y * tan_fov
        ray_y = forward_y + up_y * norm_y * tan_fov
        ray_z = forward_z + right_z * norm_x * tan_fov * aspect + up_z * norm_y * tan_fov
        
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
            npc_radius = max(1.2, 0.6 * npc.size)  # Minimum radius of 1.2 for easier selection
            
            # Check if ray passes close enough to NPC - increased tolerance for easier clicking
            # Use a larger tolerance based on distance (further NPCs need larger tolerance)
            tolerance = npc_radius * (5.0 + dist * 0.1)  # Much larger tolerance for easier clicking
            
            if dist_to_ray < tolerance:
                if dist < closest_distance:
                    closest_npc = npc
                    closest_distance = dist
        
        if closest_npc:
            self.log(f"Selected NPC: {closest_npc.name} at ({closest_npc.x:.1f}, {closest_npc.y:.1f}, {closest_npc.z:.1f})")
        
        return closest_npc
    
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
    
    def _is_in_frustum(self, x: float, y: float, z: float, radius: float = 1.0) -> bool:
        """
        Check if a point (with radius) is within the camera's view frustum.
        Optimized version using squared distances to avoid sqrt.
        
        Args:
            x, y, z: World coordinates of the point
            radius: Bounding sphere radius
            
        Returns:
            True if the point is potentially visible
        """
        # Transform point to camera space
        dx = x - self.camera_x
        dy = y - self.camera_y
        dz = z - self.camera_z
        
        # Calculate squared distance from camera
        dist_sq = dx*dx + dy*dy + dz*dz
        
        # Near and far plane culling (using squared distances)
        if dist_sq < self.near_plane*self.near_plane or dist_sq > self.far_plane*self.far_plane:
            return False
        
        # Calculate camera forward direction
        yaw_rad = np.radians(self.camera_yaw)
        pitch_rad = np.radians(self.camera_pitch)
        
        forward_x = math.sin(yaw_rad) * math.cos(pitch_rad)
        forward_y = -math.sin(pitch_rad)
        forward_z = -math.cos(yaw_rad) * math.cos(pitch_rad)
        
        # Dot product to check if point is in front of camera (normalized vectors)
        # Use squared distance to avoid sqrt
        dist = math.sqrt(dist_sq)  # Only calculate sqrt once for normalization
        if dist < 0.001:
            return True  # Very close to camera, always render
        
        vec_x = dx / dist
        vec_y = dy / dist
        vec_z = dz / dist
        
        dot = vec_x * forward_x + vec_y * forward_y + vec_z * forward_z
        
        # FOV-based culling (simplified cone check)
        fov_rad = math.radians(self.fov / 2.0)
        cos_fov = math.cos(fov_rad)
        
        # If point is behind camera or outside FOV cone, cull it
        if dot < cos_fov - (radius / dist if dist > radius else 0):
            return False
        
        return True
    
    def _render_debug_overlay(self):
        """Render debug information overlay (optimized for performance)."""
        # Switch to 2D rendering
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, self.window.width, 0, self.window.height, -1, 1)
        
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()
        
        glDisable(GL_DEPTH_TEST)
        glDisable(GL_LIGHTING)
        
        # Background for debug text
        bg_y = self.window.height - 20
        bg_height = 200  # Reduced height
        
        # Show profiling info if enabled
        if self.profiling:
            bg_height = 280  # Reduced but enough for profiling
        
        glColor4f(0.0, 0.0, 0.0, 0.7)
        glBegin(GL_QUADS)
        glVertex2f(10, bg_y - bg_height)
        glVertex2f(350, bg_y - bg_height)
        glVertex2f(350, bg_y)
        glVertex2f(10, bg_y)
        glEnd()
        
        # Simplified debug text - only essential info
        time_str = f"Day {self.world.day_number}, "
        hour = (self.world.day_time / self.world.day_length) * 24.0
        time_str += f"{int(hour):02d}:{int((hour % 1) * 60):02d}"
        
        # Minimal stats - avoid expensive list comprehensions
        stats = [
            f"FPS: {self.current_fps:.1f}",
            f"Time: {time_str}",
            f"Entities: {sum(1 for e in self.world.entities if e.is_alive)}",
        ]
        
        # Add profiling stats if enabled (simplified)
        if self.profiling and self.profile_times['total'] > 0:
            avg_fps = 1.0 / self.profile_times['total'] if self.profile_times['total'] > 0 else 0
            stats.extend([
                f"Avg FPS: {avg_fps:.1f}",
                f"Terrain: {self.profile_times['terrain']*1000:.1f}ms",
                f"NPCs: {self.profile_times['npcs']*1000:.1f}ms",
                f"Overlay: {self.profile_times['overlay']*1000:.1f}ms",
            ])
        
        # Render minimal text (single label for performance)
        stats_text = '\n'.join(stats)
        label = pyglet.text.Label(
            stats_text,
            font_name='Courier New',
            font_size=12,
            x=20,
            y=bg_y - 20,
            anchor_x='left',
            anchor_y='top',
            color=(255, 255, 255, 255),
            multiline=True,
            width=330
        )
        label.draw()
        
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        
        glPopMatrix()
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
    
    def capture_screenshot(self):
        """Capture a screenshot of the current window."""
        try:
            import os
            from datetime import datetime
            
            # Get buffer and convert to image
            buffer = pyglet.image.get_buffer_manager().get_color_buffer()
            image_data = buffer.get_image_data()
            
            # Save screenshot
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"screenshot_{timestamp}_{self.screenshot_count:03d}.png"
            filepath = os.path.join(self.screenshot_dir, filename)
            
            image_data.save(filepath)
            self.screenshot_count += 1
            
            self.log(f"Screenshot saved: {filepath}")
            print(f"Screenshot saved: {filepath}")
        except Exception as e:
            self.log(f"Failed to capture screenshot: {e}")
            print(f"Failed to capture screenshot: {e}")
    
    def log(self, message: str):
        """Add a message to the debug log."""
        self.debug_log.append(message)
        if len(self.debug_log) > self.max_log_lines:
            self.debug_log.pop(0)
    
    def _draw_sphere(self, radius: float, segments: int = 12):
        """Draw a sphere (optimized - reduced segments for performance)."""
        # Use fewer segments for better performance - cap at 10
        segments = max(6, min(segments, 10))  # Cap at 10 segments for performance
        
        for i in range(segments):
            lat1 = (math.pi / 2.0) - (i * math.pi / segments)
            lat2 = (math.pi / 2.0) - ((i + 1) * math.pi / segments)
            
            glBegin(GL_QUAD_STRIP)
            for j in range(segments + 1):
                lng = 2 * math.pi * j / segments
                x1 = radius * math.cos(lat1) * math.cos(lng)
                y1 = radius * math.sin(lat1)
                z1 = radius * math.cos(lat1) * math.sin(lng)
                
                x2 = radius * math.cos(lat2) * math.cos(lng)
                y2 = radius * math.sin(lat2)
                z2 = radius * math.cos(lat2) * math.sin(lng)
                
                glNormal3f(x1/radius, y1/radius, z1/radius)
                glVertex3f(x1, y1, z1)
                glNormal3f(x2/radius, y2/radius, z2/radius)
                glVertex3f(x2, y2, z2)
            glEnd()
