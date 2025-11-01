"""3D rendering implementation using pyglet."""

import pyglet
from pyglet.gl import *
import numpy as np
import math

from ..world.world import World
from ..entities.npc import NPC
from ..entities.animal import Animal
from ..trees.tree import FruitTree
from ..houses.house import House
from .detail_panel import DetailPanel
from .camera import Camera
from .fog_manager import FogManager
from .sky_manager import SkyManager
from .performance_profiler import PerformanceProfiler
from .vegetation_instancer import VegetationInstancer
from .historian_log_viewer import HistorianLogViewer


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
        self.window = pyglet.window.Window(width, height, caption="World Simulation", resizable=True)
        self.window.set_exclusive_mouse(False)  # Show cursor
        self.window.set_mouse_visible(True)
        
        # Camera management
        self.camera = Camera(width, height)
        
        # Ensure initial camera position is above terrain
        initial_terrain_height = self.world.get_height(self.camera.x, self.camera.z)
        # Account for 1080p monitor with window bar (~30 pixels) - adjust camera height accordingly
        # Initial camera Y is 50.0, but we want it higher for 1080p display
        self.camera.y = max(self.camera.y, initial_terrain_height + 10.0)  # Increased from 8.0 for steep terrain
        # Start slightly higher for better view on 1080p monitors
        self.camera.y = max(self.camera.y, 60.0)
        
        # Fog and sky management
        self.fog_manager = FogManager()
        self.sky_manager = SkyManager(day_length=world.day_length)
        
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
        
        # Historian log viewer
        self.log_viewer = HistorianLogViewer(self.window)
        
        # Menu system
        self.menu_visible = False
        self.menu_items = [
            ("View History Log", self._toggle_log_viewer),
            ("Toggle Debug Colors", self._toggle_debug_colors),
            ("Toggle Performance Profiling", self._toggle_profiling),
        ]
        self.selected_menu_item = 0
        
        # FPS tracking
        self.fps_display = pyglet.window.FPSDisplay(window=self.window)
        self.frame_count = 0
        self.fps_update_time = 0.0
        self.current_fps = 0.0
        
        # Camera frustum for culling (now in Camera class, keeping for compatibility)
        self.fov = self.camera.fov
        self.near_plane = self.camera.near_plane
        self.far_plane = self.camera.far_plane
        
        # Terrain caching and batching
        self.terrain_display_list = None
        self.last_terrain_x = None
        self.last_terrain_z = None
        self.terrain_cache_radius = 10  # Reduced from 40 - regenerate terrain more frequently to prevent gaps
        
        # Spatial partitioning for entities
        self.entity_grid = {}  # Grid-based spatial partitioning
        self.grid_size = 20.0  # 20 unit grid cells
        
        # Cached spatial grids (updated only when entities move significantly)
        self.cached_veg_grid = None
        self.cached_tree_grid = None
        self.cached_npc_grid = None
        self.cached_animal_grid = None
        self.last_grid_update_x = None
        self.last_grid_update_z = None
        self.grid_update_threshold = 5.0  # Update grid when camera moves >5 units
        
        # GPU instancing for vegetation
        self.vegetation_instancer = VegetationInstancer()
        
        # Cached time info (calculated once per frame)
        self.cached_time_info = None
        
        # Performance profiling
        self.profiler = PerformanceProfiler(log_file="profiling.log", log_interval=60)
        
        
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
        
        # Setup fog (managed by FogManager)
        self.fog_manager.setup()
        
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
                enabled = self.profiler.toggle()
                if enabled:
                    self.log("Performance profiling ON - press P again to disable")
                else:
                    self.log("Performance profiling OFF")
            # Press 'H' to toggle historian log viewer
            if symbol == pyglet.window.key.H:
                self.log_viewer.toggle()
                if self.log_viewer.visible:
                    self.log("History log viewer opened")
                else:
                    self.log("History log viewer closed")
            # Press 'M' to toggle menu
            if symbol == pyglet.window.key.M:
                self.menu_visible = not self.menu_visible
                if self.menu_visible:
                    self.log("Menu opened")
                else:
                    self.log("Menu closed")
            # Press 'F12' to capture screenshot
            if symbol == pyglet.window.key.F12:
                self.capture_screenshot()
            # Press 'TAB' to deselect NPC (ESC closes the app by default)
            if symbol == pyglet.window.key.TAB:
                self.selected_npc = None
                self.detail_panel.scroll = 0
                self.log("NPC deselected (TAB key)")
            # Menu navigation
            if self.menu_visible:
                if symbol == pyglet.window.key.UP:
                    self.selected_menu_item = (self.selected_menu_item - 1) % len(self.menu_items)
                elif symbol == pyglet.window.key.DOWN:
                    self.selected_menu_item = (self.selected_menu_item + 1) % len(self.menu_items)
                elif symbol == pyglet.window.key.ENTER or symbol == pyglet.window.key.RETURN:
                    # Execute selected menu item
                    _, action = self.menu_items[self.selected_menu_item]
                    action()
                    self.menu_visible = False
        
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
                        # Clicking elsewhere (on terrain/empty space) - deselect
                        # Make deselection easier - always deselect when clicking empty space
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
                self.camera.rotate(dx, dy)
        
        @self.window.event
        def on_mouse_drag(x, y, dx, dy, buttons, modifiers):
            self.mouse_x = x
            self.mouse_y = y
            self.camera.rotate(dx, dy)
        
        @self.window.event
        def on_mouse_scroll(x, y, scroll_x, scroll_y):
            # Check if mouse is over detail panel
            if self.detail_panel.is_point_in_panel(x, y):
                # Scroll detail panel when hovering over it
                self.detail_panel.handle_scroll(x, y, scroll_x, scroll_y)
            elif self.log_viewer.is_point_in_panel(x, y):
                # Scroll log viewer when hovering over it
                self.log_viewer.handle_scroll(x, y, scroll_x, scroll_y)
            else:
                # Zoom with mouse wheel when not over panel
                self.camera.zoom(scroll_y)
        
        @self.window.event
        def on_resize(width, height):
            """Handle window resize."""
            # Update camera dimensions
            self.camera.resize(width, height)
            # Update viewport
            glViewport(0, 0, width, height)
            # Update mouse tracking center
            self.mouse_x = width // 2
            self.mouse_y = height // 2
            self.log(f"Window resized to {width}x{height}")
    
    def update(self, delta_time: float):
        """Update camera based on keyboard and mouse."""
        # Update FPS
        self.frame_count += 1
        self.fps_update_time += delta_time
        if self.fps_update_time >= 0.5:  # Update FPS every 0.5 seconds
            self.current_fps = self.frame_count / self.fps_update_time
            self.frame_count = 0
            self.fps_update_time = 0.0
        
        # Update camera position
        self.camera.update_position(delta_time, self.keys, self.keys_handler)
        
        # Ensure camera stays above terrain (minimum 10 units above terrain height for steep terrain)
        # Also check ahead of camera to prevent clipping when looking down at steep angles
        terrain_height = self.world.get_height(self.camera.x, self.camera.z)
        
        # Check terrain height in multiple directions to prevent clipping through cliffs
        forward_x, forward_y, forward_z = self.camera.get_forward_vector()
        right_x, right_z = self.camera.get_right_vector()
        
        # Check ahead, left, right, and diagonally for better clipping prevention
        look_ahead_distance = 8.0  # Increased from 5.0 for better cliff detection
        check_points = [
            (self.camera.x + forward_x * look_ahead_distance, self.camera.z + forward_z * look_ahead_distance),  # Forward
            (self.camera.x + forward_x * look_ahead_distance * 0.5 + right_x * look_ahead_distance * 0.5, 
             self.camera.z + forward_z * look_ahead_distance * 0.5 + right_z * look_ahead_distance * 0.5),  # Forward-right
            (self.camera.x + forward_x * look_ahead_distance * 0.5 - right_x * look_ahead_distance * 0.5, 
             self.camera.z + forward_z * look_ahead_distance * 0.5 - right_z * look_ahead_distance * 0.5),  # Forward-left
        ]
        
        # Find maximum terrain height among all check points
        max_terrain_height = terrain_height
        for check_x, check_z in check_points:
            check_height = self.world.get_height(check_x, check_z)
            max_terrain_height = max(max_terrain_height, check_height)
        
        # Use the maximum terrain height with increased clearance
        min_camera_height = max_terrain_height + 10.0  # Increased from 8.0 to prevent clipping through cliffs
        self.camera.y = max(self.camera.y, min_camera_height)
    
    def render(self):
        """Render the world."""
        import time
        frame_start = time.time()
        
        # Cache time info once per frame (used by many render functions)
        self.cached_time_info = self._get_time_info()
        hour, is_night = self.cached_time_info
        
        # Update lighting and sky based on time of day
        light_intensity = self.world.get_light_intensity()
        
        # Calculate sky color using SkyManager
        sky_r, sky_g, sky_b = self.sky_manager.calculate_sky_color(self.world.day_time)
        
        glClearColor(sky_r, sky_g, sky_b, 1.0)
        self.window.clear()
        
        # Setup camera projection and view
        self.camera.resize(self.window.width, self.window.height)
        self.camera.setup_projection()
        self.camera.setup_view()
        
        # Update fog color to match sky color
        self.fog_manager.update_color(sky_r, sky_g, sky_b)
        
        # Setup lighting using SkyManager
        self.sky_manager.setup_lighting(light_intensity, is_night)
        
        # Render terrain
        terrain_start = time.time()
        self._render_terrain()
        terrain_time = time.time() - terrain_start
        
        # Render vegetation (optimized with GPU instancing and spatial partitioning)
        veg_start = time.time()
        camera_grid_x = int(self.camera.x / self.grid_size)
        camera_grid_z = int(self.camera.z / self.grid_size)
        
        # Update cached grid only if camera moved significantly
        if (self.cached_veg_grid is None or 
            self.last_grid_update_x is None or
            abs(self.camera.x - self.last_grid_update_x) > self.grid_update_threshold or
            abs(self.camera.z - self.last_grid_update_z) > self.grid_update_threshold):
            
            # Build spatial grid
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
            self.cached_veg_grid = veg_grid
            self.last_grid_update_x = self.camera.x
            self.last_grid_update_z = self.camera.z
        else:
            veg_grid = self.cached_veg_grid
        
        # Collect visible vegetation instances
        visible_vegetation = []
        veg_count = 0
        max_veg_per_frame = 200  # Increased limit due to GPU instancing efficiency
        
        for dx in range(-2, 3):  # Check 5x5 grid cells around camera
            for dz in range(-2, 3):
                key = (camera_grid_x + dx, camera_grid_z + dz)
                if key in veg_grid:
                    for veg in veg_grid[key]:
                        if veg_count >= max_veg_per_frame:
                            break
                        # Quick distance check
                        dist_sq = (veg.x - self.camera.x)**2 + (veg.z - self.camera.z)**2
                        if dist_sq < 10000:  # Within 100 units
                            visible_vegetation.append(veg)
                            veg_count += 1
                if veg_count >= max_veg_per_frame:
                    break
            if veg_count >= max_veg_per_frame:
                break
        
        # Prepare instances for GPU instancing
        self.vegetation_instancer.prepare_instances(visible_vegetation)
        
        # Render all vegetation using GPU instancing (batched by type)
        hour, is_night = self.cached_time_info
        self.vegetation_instancer.render_all(is_night, self.world.generator.max_height)
        
        veg_time = time.time() - veg_start
        
        # Render houses (all within distance)
        house_start = time.time()
        for house in self.world.houses:
            dx = house.x - self.camera.x
            dz = house.z - self.camera.z
            dist_sq = dx*dx + dz*dz
            if dist_sq < 10000:  # Within 100 units
                self._render_house(house)
        house_time = time.time() - house_start
        
        # Render trees (optimized with spatial partitioning and caching)
        tree_start = time.time()
        
        # Update cached grid only if needed
        if self.cached_tree_grid is None or abs(self.camera.x - self.last_grid_update_x) > self.grid_update_threshold or abs(self.camera.z - self.last_grid_update_z) > self.grid_update_threshold:
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
            self.cached_tree_grid = tree_grid
        else:
            tree_grid = self.cached_tree_grid
        
        tree_count = 0
        max_trees_per_frame = 50  # Further reduced from 80 for better performance
        
        for dx in range(-2, 3):
            for dz in range(-2, 3):
                key = (camera_grid_x + dx, camera_grid_z + dz)
                if key in tree_grid:
                    for tree in tree_grid[key]:
                        if tree_count >= max_trees_per_frame:
                            break
                        dx_tree = tree.x - self.camera.x
                        dz_tree = tree.z - self.camera.z
                        dist_sq = dx_tree*dx_tree + dz_tree*dz_tree
                        if dist_sq < 10000:  # Within 100 units
                            # Calculate distance for LOD
                            dist = math.sqrt(dist_sq) if dist_sq > 0 else 0
                            self._render_tree(tree, dist)
                            tree_count += 1
                if tree_count >= max_trees_per_frame:
                    break
            if tree_count >= max_trees_per_frame:
                break
        tree_time = time.time() - tree_start
        
        # Render NPCs (optimized with spatial partitioning and caching)
        npc_start = time.time()
        
        # Update cached grid only if needed
        if self.cached_npc_grid is None or abs(self.camera.x - self.last_grid_update_x) > self.grid_update_threshold or abs(self.camera.z - self.last_grid_update_z) > self.grid_update_threshold:
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
            self.cached_npc_grid = npc_grid
        else:
            npc_grid = self.cached_npc_grid
        
        npc_count = 0
        max_npcs_per_frame = 100  # Reduced from 150 while maintaining quality
        
        for dx in range(-2, 3):
            for dz in range(-2, 3):
                key = (camera_grid_x + dx, camera_grid_z + dz)
                if key in npc_grid:
                    for npc in npc_grid[key]:
                        if npc_count >= max_npcs_per_frame:
                            break
                        dist_sq = (npc.x - self.camera.x)**2 + (npc.z - self.camera.z)**2
                        if dist_sq < 10000:  # Within 100 units
                            self._render_npc(npc)
                            npc_count += 1
                if npc_count >= max_npcs_per_frame:
                    break
            if npc_count >= max_npcs_per_frame:
                break
        npc_time = time.time() - npc_start
        
        # Render animals (optimized with spatial partitioning and caching)
        animal_start = time.time()
        
        # Update cached grid only if needed
        if self.cached_animal_grid is None or abs(self.camera.x - self.last_grid_update_x) > self.grid_update_threshold or abs(self.camera.z - self.last_grid_update_z) > self.grid_update_threshold:
            animal_grid = {}
            for animal in self.world.animals:
                if not animal.is_alive:
                    continue
                grid_x = int(animal.x / self.grid_size)
                grid_z = int(animal.z / self.grid_size)
                key = (grid_x, grid_z)
                if key not in animal_grid:
                    animal_grid[key] = []
                animal_grid[key].append(animal)
            self.cached_animal_grid = animal_grid
        else:
            animal_grid = self.cached_animal_grid
        
        animal_count = 0
        max_animals_per_frame = 40  # Reduced from 50 while maintaining quality
        
        for dx in range(-2, 3):
            for dz in range(-2, 3):
                key = (camera_grid_x + dx, camera_grid_z + dz)
                if key in animal_grid:
                    for animal in animal_grid[key]:
                        if animal_count >= max_animals_per_frame:
                            break
                        dist_sq = (animal.x - self.camera.x)**2 + (animal.z - self.camera.z)**2
                        if dist_sq < 10000:  # Within 100 units
                            self._render_animal(animal)
                            animal_count += 1
                if animal_count >= max_animals_per_frame:
                    break
            if animal_count >= max_animals_per_frame:
                break
        animal_time = time.time() - animal_start
        
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
        
        # Render log viewer if visible
        if self.log_viewer.visible:
            log_lines = self.world.historian.get_log_lines()
            self.log_viewer.render(log_lines)
        
        # Render menu if visible
        if self.menu_visible:
            self._render_menu()
        
        # Update profiling data
        total_time = time.time() - frame_start
        
        # Update profiler with current frame times
        frame_times = {
            'terrain': terrain_time,
            'vegetation': veg_time,
            'trees': tree_time,
            'npcs': npc_time,
            'animals': animal_time,
            'houses': house_time,
            'overlay': overlay_time,
        }
        self.profiler.update_frame_times(frame_times, total_time)
    
    def _get_time_info(self):
        """
        Get current time information.
        
        Returns:
            Tuple of (hour, is_night) where hour is 0-24 and is_night is boolean
        """
        hour = (self.world.day_time / self.world.day_length) * 24.0
        is_night = hour < 6.0 or hour >= 18.0
        return hour, is_night
    
    def _render_terrain(self):
        """Render the terrain mesh with colorful grass."""
        # Use cached time info
        hour, is_night = self.cached_time_info
        
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
        base_resolution = 150  # Optimized resolution (still smooth due to bilinear interpolation)
        
        # Use smooth shading for better visual quality
        glShadeModel(GL_SMOOTH)
        
        # Single unified terrain grid - no LOD zones to prevent gaps
        # All quads share the same grid alignment, ensuring seamless coverage
        zone_max = terrain_size
        resolution = base_resolution
        
        # Check if we can reuse cached terrain display list
        camera_grid_x = int(self.camera.x / self.terrain_cache_radius)
        camera_grid_z = int(self.camera.z / self.terrain_cache_radius)
        
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
            
            # Single unified terrain grid - no LOD zones to prevent gaps
            # All quads share the same grid alignment, ensuring seamless coverage
            zone_start_x = self.camera.x - zone_max
            zone_start_z = self.camera.z - zone_max
            step = (zone_max * 2) / resolution
            
            # Pre-calculate ALL heights for the entire grid
            height_grid = {}
            for z in range(resolution + 1):
                for x in range(resolution + 1):
                    world_x = zone_start_x + x * step
                    world_z = zone_start_z + z * step
                    height_grid[(x, z)] = self.world.get_height(world_x, world_z)
            
            # Render ALL quads in the unified grid
            glBegin(GL_QUADS)
            for z in range(resolution):
                for x in range(resolution):
                    world_x1 = zone_start_x + x * step
                    world_z1 = zone_start_z + z * step
                    
                    # Use pre-calculated heights from height_grid
                    # CRITICAL: Always use height_grid values - they're guaranteed to exist
                    # since we pre-calculated all of them above
                    h1 = height_grid[(x, z)]
                    h2 = height_grid[(x + 1, z)]
                    h3 = height_grid[(x + 1, z + 1)]
                    h4 = height_grid[(x, z + 1)]
                    
                    # REMOVED height smoothing - it was causing gaps at quad boundaries
                    # Heights are already smooth due to bilinear interpolation in get_height()
                    
                    # Get terrain type for color
                    terrain_type = self.world.generator.get_terrain_type(
                        h1 / self.world.generator.max_height
                    )
                    
                    # Set color based on terrain type
                    if not is_night:
                        if terrain_type == 'water':
                            glColor3f(0.1, 0.2, 0.4)
                        elif terrain_type == 'sand':
                            glColor3f(0.7, 0.65, 0.5)
                        elif terrain_type == 'dirt':
                            glColor3f(0.4, 0.3, 0.2)
                        elif terrain_type == 'grass':
                            glColor3f(0.2, 0.5, 0.15)
                        elif terrain_type == 'hill':
                            glColor3f(0.3, 0.45, 0.2)
                        elif terrain_type == 'mountain':
                            glColor3f(0.35, 0.35, 0.3)
                        else:
                            glColor3f(0.9, 0.9, 0.95)
                    else:
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
                        else:
                            glColor3f(0.05, 0.05, 0.06)
                    
                    # Calculate smooth per-vertex normals
                    # CRITICAL: Use direct dictionary access - all values guaranteed to exist
                    # Handle edge cases properly to prevent gaps
                    h_right = height_grid[(x + 1, z)]
                    h_left = height_grid[(x - 1, z)] if x > 0 else height_grid[(x, z)]
                    h_up = height_grid[(x, z + 1)]
                    h_down = height_grid[(x, z - 1)] if z > 0 else height_grid[(x, z)]
                    dh_dx = (h_right - h_left) / (2 * step) if step > 0 else 0
                    dh_dz = (h_up - h_down) / (2 * step) if step > 0 else 0
                    nx = -dh_dx
                    nz = -dh_dz
                    length_inv = 1.0 / math.sqrt(nx*nx + 1.0 + nz*nz) if (nx*nx + 1.0 + nz*nz) > 0.0001 else 1.0
                    n1 = (nx * length_inv, 1.0 * length_inv, nz * length_inv)
                    
                    h_right2 = height_grid[(x + 1, z)]
                    h_left2 = height_grid[(x - 1, z)] if x > 0 else height_grid[(x, z)]
                    h_up2 = height_grid[(x + 1, z + 1)]
                    h_down2 = height_grid[(x + 1, z - 1)] if z > 0 else height_grid[(x + 1, z)]
                    dh_dx2 = (h_right2 - h_left2) / (2 * step) if step > 0 else 0
                    dh_dz2 = (h_up2 - h_down2) / (2 * step) if step > 0 else 0
                    nx2 = -dh_dx2
                    nz2 = -dh_dz2
                    length_inv2 = 1.0 / math.sqrt(nx2*nx2 + 1.0 + nz2*nz2) if (nx2*nx2 + 1.0 + nz2*nz2) > 0.0001 else 1.0
                    n2 = (nx2 * length_inv2, 1.0 * length_inv2, nz2 * length_inv2)
                    
                    h_right3 = height_grid[(x + 1, z + 1)]
                    h_left3 = height_grid[(x, z + 1)]
                    h_up3 = height_grid[(x + 1, z + 2)] if z + 1 < resolution else height_grid[(x + 1, z + 1)]
                    h_down3 = height_grid[(x + 1, z)]
                    dh_dx3 = (h_right3 - h_left3) / (2 * step) if step > 0 else 0
                    dh_dz3 = (h_up3 - h_down3) / (2 * step) if step > 0 else 0
                    nx3 = -dh_dx3
                    nz3 = -dh_dz3
                    length_inv3 = 1.0 / math.sqrt(nx3*nx3 + 1.0 + nz3*nz3) if (nx3*nx3 + 1.0 + nz3*nz3) > 0.0001 else 1.0
                    n3 = (nx3 * length_inv3, 1.0 * length_inv3, nz3 * length_inv3)
                    
                    h_right4 = height_grid[(x, z + 1)]
                    h_left4 = height_grid[(x - 1, z + 1)] if x > 0 else height_grid[(x, z + 1)]
                    h_up4 = height_grid[(x, z + 2)] if z + 1 < resolution else height_grid[(x, z + 1)]
                    h_down4 = height_grid[(x, z)]
                    dh_dx4 = (h_right4 - h_left4) / (2 * step) if step > 0 else 0
                    dh_dz4 = (h_up4 - h_down4) / (2 * step) if step > 0 else 0
                    nx4 = -dh_dx4
                    nz4 = -dh_dz4
                    length_inv4 = 1.0 / math.sqrt(nx4*nx4 + 1.0 + nz4*nz4) if (nx4*nx4 + 1.0 + nz4*nz4) > 0.0001 else 1.0
                    n4 = (nx4 * length_inv4, 1.0 * length_inv4, nz4 * length_inv4)
                    
                    # Draw quad with per-vertex normals
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
    
    def _render_tree(self, tree: FruitTree, distance: float = 0.0):
        """Render a colorful fruit tree with improved visuals and distance-based LOD."""
        if not tree.is_alive:
            return
        
        hour, is_night = self.cached_time_info
        
        # Distance-based LOD - reduce detail for distant trees
        if distance < 20:
            # Close: full detail
            trunk_segments = 8
            main_foliage_segments = 8
            secondary_foliage_segments = 6
            side_cluster_segments = 4
            fruit_segments = 6
            render_secondary_foliage = True
            render_side_clusters = True
        elif distance < 50:
            # Medium: reduced detail
            trunk_segments = 6
            main_foliage_segments = 6
            secondary_foliage_segments = 4
            side_cluster_segments = 3
            fruit_segments = 4
            render_secondary_foliage = True
            render_side_clusters = False
        else:
            # Far: minimal detail
            trunk_segments = 4
            main_foliage_segments = 4
            secondary_foliage_segments = 3
            side_cluster_segments = 3
            fruit_segments = 3
            render_secondary_foliage = False
            render_side_clusters = False
        
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
        
        # Draw cylindrical trunk with proper normals (using LOD segments)
        import math
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
            
            # Main foliage cluster at top (always rendered)
            foliage_base_height = trunk_height + 0.8 * tree.growth_stage
            main_foliage_size = 1.2 * tree.growth_stage
            glPushMatrix()
            glTranslatef(0, foliage_base_height, 0)
            self._draw_sphere(main_foliage_size, main_foliage_segments)
            glPopMatrix()
            
            # Secondary foliage clusters for fuller look (only if close)
            if render_secondary_foliage and tree.growth_stage >= 0.7:
                # Medium cluster slightly lower
                glPushMatrix()
                glTranslatef(0, foliage_base_height - 0.3 * tree.growth_stage, 0)
                self._draw_sphere(main_foliage_size * 0.85, secondary_foliage_segments)
                glPopMatrix()
            
            if render_side_clusters and tree.growth_stage >= 0.85:
                # Smaller clusters around the sides (only 2 instead of 3 for performance)
                for i in range(2):  # Reduced from 3 to 2
                    angle = (i / 2.0) * 2 * math.pi
                    offset_x = math.cos(angle) * 0.4 * tree.growth_stage
                    offset_z = math.sin(angle) * 0.4 * tree.growth_stage
                    glPushMatrix()
                    glTranslatef(offset_x, foliage_base_height - 0.2 * tree.growth_stage, offset_z)
                    self._draw_sphere(main_foliage_size * 0.6, side_cluster_segments)
                    glPopMatrix()
            
            # Render fruit - all red, make them more visible (only if close)
            if distance < 50:  # Only render fruit for trees within 50 units
                ripe_count = tree.get_ripe_fruit_count()
                total_fruit = len(tree.fruit_maturity)
                
                if total_fruit > 0:
                    # Bright red fruit material - make it glow at night
                    if is_night:
                        # At night, make fruit glow brighter (emission-like effect)
                        # But reduce red emission to prevent red bleed onto tree
                        glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, (GLfloat * 4)(0.4, 0.05, 0.05, 1.0))
                        glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, (GLfloat * 4)(0.6, 0.1, 0.1, 1.0))
                        glMaterialfv(GL_FRONT_AND_BACK, GL_EMISSION, (GLfloat * 4)(0.2, 0.0, 0.0, 1.0))  # Reduced emission
                        glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, (GLfloat * 4)(0.5, 0.5, 0.5, 1.0))
                        glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 40.0)
                        glColor3f(0.6, 0.1, 0.1)  # Muted red at night to prevent bleed
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
                            self._draw_sphere(0.18, fruit_segments)
                        else:  # Growing fruit - smaller
                            if is_night:
                                glColor3f(0.4, 0.05, 0.05)  # Very dark red at night
                            else:
                                glColor3f(0.8, 0.2, 0.2)
                            self._draw_sphere(0.12, fruit_segments)
                            if is_night:
                                glColor3f(0.6, 0.1, 0.1)  # Reset to night fruit color
                            else:
                                glColor3f(1.0, 0.1, 0.1)  # Reset color
                        glPopMatrix()
                    
                    # Reset material properties after fruit rendering to prevent bleed
                    if is_night:
                        # Reset to leaf colors
                        glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, (GLfloat * 4)(0.0, 0.05, 0.04, 1.0))
                        glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, (GLfloat * 4)(0.0, 0.08, 0.06, 1.0))
                        glMaterialfv(GL_FRONT_AND_BACK, GL_EMISSION, (GLfloat * 4)(0.0, 0.0, 0.0, 1.0))  # No emission
                        glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, (GLfloat * 4)(0.0, 0.04, 0.03, 1.0))
                        glColor3f(0.0, 0.08, 0.06)  # Reset to leaf color
                    else:
                        # Reset to leaf colors
                        glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, (GLfloat * 4)(0.1, 0.2, 0.1, 1.0))
                        glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, (GLfloat * 4)(0.2, 0.6, 0.2, 1.0))
                        glMaterialfv(GL_FRONT_AND_BACK, GL_EMISSION, (GLfloat * 4)(0.0, 0.0, 0.0, 1.0))
                        glColor3f(0.1, 0.7, 0.2)
        
        glPopMatrix()
    
    def _render_npc(self, npc: NPC):
        """Render an NPC with a simplified model for performance."""
        if not npc.is_alive:
            return
        
        hour, is_night = self.cached_time_info
        
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
            # ZERO red component at night - only blue-gray tones
            base_r = 0.0  # No red at all
            base_g = 0.15 * health_ratio + 0.1
            base_b = 0.2 * health_ratio + 0.15
            saturation = 0.3 + hunger_ratio * 0.3
            r = 0.0  # Zero red
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
    
    def _render_animal(self, animal: Animal):
        """Render an animal."""
        if not animal.is_alive:
            return
        
        hour, is_night = self.cached_time_info
        
        # Set up colors based on species and time of day
        species_colors = {
            "deer": {
                "day": (0.6, 0.4, 0.2),  # Brown
                "night": (0.15, 0.1, 0.05)  # Dark brown
            },
            "rabbit": {
                "day": (0.8, 0.8, 0.8),  # Light gray/white
                "night": (0.2, 0.2, 0.2)  # Dark gray
            },
            "boar": {
                "day": (0.3, 0.2, 0.2),  # Dark brown/black
                "night": (0.08, 0.05, 0.05)  # Very dark
            }
        }
        
        color_key = "night" if is_night else "day"
        base_color = species_colors.get(animal.species, species_colors["deer"])[color_key]
        
        # Set material properties
        if is_night:
            glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, (GLfloat * 4)(base_color[0] * 0.3, base_color[1] * 0.3, base_color[2] * 0.3, 1.0))
            glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, (GLfloat * 4)(base_color[0] * 0.4, base_color[1] * 0.4, base_color[2] * 0.4, 1.0))
            glMaterialfv(GL_FRONT_AND_BACK, GL_EMISSION, (GLfloat * 4)(0.0, 0.0, 0.0, 1.0))
            glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, (GLfloat * 4)(0.0, 0.0, 0.0, 1.0))
            glColor3f(base_color[0] * 0.4, base_color[1] * 0.4, base_color[2] * 0.4)
        else:
            glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, (GLfloat * 4)(base_color[0] * 0.5, base_color[1] * 0.5, base_color[2] * 0.5, 1.0))
            glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, (GLfloat * 4)(base_color[0], base_color[1], base_color[2], 1.0))
            glMaterialfv(GL_FRONT_AND_BACK, GL_EMISSION, (GLfloat * 4)(0.0, 0.0, 0.0, 1.0))
            glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, (GLfloat * 4)(0.2, 0.2, 0.2, 1.0))
            glColor3f(base_color[0], base_color[1], base_color[2])
        
        glPushMatrix()
        glTranslatef(animal.x, animal.y, animal.z)
        
        scale = animal.size
        
        # Body (ellipsoid-like shape)
        glPushMatrix()
        glTranslatef(0.0, scale * 0.3, 0.0)
        self._draw_box_fast(scale * 0.4, scale * 0.3, scale * 0.2)
        glPopMatrix()
        
        # Head
        glPushMatrix()
        glTranslatef(0.0, scale * 0.5, scale * 0.15)
        self._draw_sphere(scale * 0.15, 6)
        glPopMatrix()
        
        # Legs (4 legs)
        leg_size = scale * 0.08
        leg_height = scale * 0.3
        
        # Front left leg
        glPushMatrix()
        glTranslatef(-scale * 0.15, -scale * 0.05, scale * 0.1)
        self._draw_box_fast(leg_size, leg_height, leg_size)
        glPopMatrix()
        
        # Front right leg
        glPushMatrix()
        glTranslatef(scale * 0.15, -scale * 0.05, scale * 0.1)
        self._draw_box_fast(leg_size, leg_height, leg_size)
        glPopMatrix()
        
        # Back left leg
        glPushMatrix()
        glTranslatef(-scale * 0.15, -scale * 0.05, -scale * 0.1)
        self._draw_box_fast(leg_size, leg_height, leg_size)
        glPopMatrix()
        
        # Back right leg
        glPushMatrix()
        glTranslatef(scale * 0.15, -scale * 0.05, -scale * 0.1)
        self._draw_box_fast(leg_size, leg_height, leg_size)
        glPopMatrix()
        
        # Tail (for deer and boar)
        if animal.species in ["deer", "boar"]:
            glPushMatrix()
            glTranslatef(0.0, scale * 0.2, -scale * 0.2)
            glRotatef(45, 1, 0, 0)
            self._draw_box_fast(leg_size * 0.5, leg_size * 0.5, scale * 0.15)
            glPopMatrix()
        
        # Ears (for rabbit)
        if animal.species == "rabbit":
            glPushMatrix()
            glTranslatef(-scale * 0.08, scale * 0.6, 0.0)
            glRotatef(-20, 0, 0, 1)
            self._draw_box_fast(leg_size * 0.8, scale * 0.2, leg_size * 0.3)
            glPopMatrix()
            
            glPushMatrix()
            glTranslatef(scale * 0.08, scale * 0.6, 0.0)
            glRotatef(20, 0, 0, 1)
            self._draw_box_fast(leg_size * 0.8, scale * 0.2, leg_size * 0.3)
            glPopMatrix()
        
        glPopMatrix()
    
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
        dx = npc.x - self.camera.x
        dz = npc.z - self.camera.z
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
        dx = self.camera.x - npc.x
        dz = self.camera.z - npc.z
        angle = math.degrees(math.atan2(dx, dz))
        glRotatef(-angle, 0.0, 1.0, 0.0)
        
        glDisable(GL_LIGHTING)
        
        # Health bar (red background - darker at night)
        health_ratio = npc.health / npc.max_health
        hour, is_night = self.cached_time_info
        if is_night:
            # Dark blue-gray background at night instead of red
            glColor3f(0.0, 0.02, 0.04)
        else:
            glColor3f(1.0, 0.0, 0.0)
        glBegin(GL_QUADS)
        glVertex3f(-bar_width/2, 0, 0)
        glVertex3f(bar_width/2, 0, 0)
        glVertex3f(bar_width/2, bar_thickness, 0)
        glVertex3f(-bar_width/2, bar_thickness, 0)
        glEnd()
        
        # Health fill (green during day, blue-green at night)
        if is_night:
            glColor3f(0.0, 0.3, 0.5)  # Blue-green fill at night
        else:
            glColor3f(0.0, 1.0, 0.0)  # Green fill during day
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
        """Render a house with improved model, door, and multiple colors."""
        if not house.is_built:
            return
        
        # Adjust house color based on time of day
        hour, is_night = self.cached_time_info
        
        glPushMatrix()
        glTranslatef(house.x, house.y, house.z)
        
        size = 1.5
        height = 2.0
        door_width = 0.4
        door_height = 1.5
        door_depth = 0.1
        
        # Walls with house-specific colors
        if is_night:
            # Cool, muted colors for night - apply house color but darker
            wall_r, wall_g, wall_b = house.wall_color
            glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, (GLfloat * 4)(wall_r * 0.1, wall_g * 0.15, wall_b * 0.2, 1.0))
            glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, (GLfloat * 4)(wall_r * 0.15, wall_g * 0.2, wall_b * 0.25, 1.0))
            glColor3f(wall_r * 0.15, wall_g * 0.2, wall_b * 0.25)
        else:
            # Use house-specific wall colors during day
            wall_r, wall_g, wall_b = house.wall_color
            glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, (GLfloat * 4)(wall_r * 0.6, wall_g * 0.6, wall_b * 0.6, 1.0))
            glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, (GLfloat * 4)(wall_r, wall_g, wall_b, 1.0))
            glColor3f(wall_r, wall_g, wall_b)
        
        glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, (GLfloat * 4)(0.1, 0.1, 0.1, 1.0))
        glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 5.0)
        
        # Front wall with door opening
        glBegin(GL_QUADS)
        glNormal3f(0.0, 0.0, 1.0)
        # Top section above door
        glVertex3f(-size, door_height, size)
        glVertex3f(size, door_height, size)
        glVertex3f(size, height, size)
        glVertex3f(-size, height, size)
        # Left section (left of door)
        glVertex3f(-size, 0, size)
        glVertex3f(-door_width/2, 0, size)
        glVertex3f(-door_width/2, door_height, size)
        glVertex3f(-size, door_height, size)
        # Right section (right of door)
        glVertex3f(door_width/2, 0, size)
        glVertex3f(size, 0, size)
        glVertex3f(size, door_height, size)
        glVertex3f(door_width/2, door_height, size)
        glEnd()
        
        # Back wall (solid)
        glBegin(GL_QUADS)
        glNormal3f(0.0, 0.0, -1.0)
        glVertex3f(-size, 0, -size)
        glVertex3f(-size, height, -size)
        glVertex3f(size, height, -size)
        glVertex3f(size, 0, -size)
        glEnd()
        
        # Left wall
        glBegin(GL_QUADS)
        glNormal3f(-1.0, 0.0, 0.0)
        glVertex3f(-size, 0, -size)
        glVertex3f(-size, height, -size)
        glVertex3f(-size, height, size)
        glVertex3f(-size, 0, size)
        glEnd()
        
        # Right wall
        glBegin(GL_QUADS)
        glNormal3f(1.0, 0.0, 0.0)
        glVertex3f(size, 0, -size)
        glVertex3f(size, 0, size)
        glVertex3f(size, height, size)
        glVertex3f(size, height, -size)
        glEnd()
        
        # Door frame (slightly darker than walls)
        if is_night:
            glColor3f(0.0, 0.08, 0.12)
        else:
            glColor3f(0.3, 0.25, 0.2)  # Dark brown frame
        
        glBegin(GL_QUADS)
        glNormal3f(0.0, 0.0, 1.0)
        # Door frame sides
        glVertex3f(-door_width/2 - 0.05, 0, size + 0.01)
        glVertex3f(-door_width/2, 0, size + 0.01)
        glVertex3f(-door_width/2, door_height, size + 0.01)
        glVertex3f(-door_width/2 - 0.05, door_height, size + 0.01)
        
        glVertex3f(door_width/2, 0, size + 0.01)
        glVertex3f(door_width/2 + 0.05, 0, size + 0.01)
        glVertex3f(door_width/2 + 0.05, door_height, size + 0.01)
        glVertex3f(door_width/2, door_height, size + 0.01)
        
        # Door frame top
        glVertex3f(-door_width/2 - 0.05, door_height, size + 0.01)
        glVertex3f(door_width/2 + 0.05, door_height, size + 0.01)
        glVertex3f(door_width/2 + 0.05, door_height + 0.05, size + 0.01)
        glVertex3f(-door_width/2 - 0.05, door_height + 0.05, size + 0.01)
        glEnd()
        
        # Door (wooden, darker)
        if is_night:
            glColor3f(0.0, 0.06, 0.1)
        else:
            glColor3f(0.4, 0.3, 0.2)  # Dark brown door
        
        glBegin(GL_QUADS)
        glNormal3f(0.0, 0.0, 1.0)
        glVertex3f(-door_width/2, 0, size + door_depth)
        glVertex3f(door_width/2, 0, size + door_depth)
        glVertex3f(door_width/2, door_height, size + door_depth)
        glVertex3f(-door_width/2, door_height, size + door_depth)
        glEnd()
        
        # Roof with house-specific colors
        if is_night:
            roof_r, roof_g, roof_b = house.roof_color
            glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, (GLfloat * 4)(roof_r * 0.1, roof_g * 0.1, roof_b * 0.15, 1.0))
            glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, (GLfloat * 4)(roof_r * 0.15, roof_g * 0.15, roof_b * 0.2, 1.0))
            glColor3f(roof_r * 0.15, roof_g * 0.15, roof_b * 0.2)
        else:
            roof_r, roof_g, roof_b = house.roof_color
            glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, (GLfloat * 4)(roof_r * 0.7, roof_g * 0.7, roof_b * 0.7, 1.0))
            glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, (GLfloat * 4)(roof_r, roof_g, roof_b, 1.0))
            glColor3f(roof_r, roof_g, roof_b)
        
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
        hour, is_night = self.cached_time_info
        
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
        hour, is_night = self.cached_time_info
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
        yaw_rad = np.radians(self.camera.yaw)
        pitch_rad = np.radians(self.camera.pitch)
        
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
            dx = npc.x - self.camera.x
            dy = npc.y - self.camera.y
            dz = npc.z - self.camera.z
            
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
        dx = x - self.camera.x
        dy = y - self.camera.y
        dz = z - self.camera.z
        
        # Calculate squared distance from camera
        dist_sq = dx*dx + dy*dy + dz*dz
        
        # Near and far plane culling (using squared distances)
        if dist_sq < self.near_plane*self.near_plane or dist_sq > self.far_plane*self.far_plane:
            return False
        
        # Calculate camera forward direction
        yaw_rad = np.radians(self.camera.yaw)
        pitch_rad = np.radians(self.camera.pitch)
        
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
        if self.profiler.enabled:
            # Calculate needed height based on number of stat lines
            # Base stats: 3 lines + profiling: ~27 lines = ~30 lines total
            # At 11px font size with spacing, need ~16px per line = 480px minimum
            # Add padding for safety: 520px
            bg_height = 520  # Increased height for detailed profiling info with proper padding
        
        glColor4f(0.0, 0.0, 0.0, 0.7)
        glBegin(GL_QUADS)
        glVertex2f(10, bg_y - bg_height)
        glVertex2f(400, bg_y - bg_height)  # Increased width from 350 to 400 for better text fit
        glVertex2f(400, bg_y)
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
        
        # Add profiling stats if enabled (detailed)
        if self.profiler.enabled:
            stats.extend(self.profiler.get_stats_for_overlay())
        
        # Render text (single label for performance)
        stats_text = '\n'.join(stats)
        # Calculate text height to ensure it fits within the background
        num_lines = len(stats)
        line_height = 14  # Approximate line height in pixels for font size 11
        text_height = num_lines * line_height
        
        # Adjust y position to ensure text fits within background
        text_y = bg_y - 10  # Small margin from top
        label = pyglet.text.Label(
            stats_text,
            font_name='Courier New',
            font_size=11,  # Slightly smaller font to fit more
            x=20,
            y=text_y,
            anchor_x='left',
            anchor_y='top',
            color=(255, 255, 255, 255),
            multiline=True,
            width=380  # Increased from 330 to match new frame width
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
    
    def _toggle_log_viewer(self):
        """Toggle the historian log viewer."""
        self.log_viewer.toggle()
    
    def _toggle_debug_colors(self):
        """Toggle debug color mode."""
        self.debug_colors = not self.debug_colors
    
    def _toggle_profiling(self):
        """Toggle performance profiling."""
        self.profiler.toggle()
    
    def _render_menu(self):
        """Render the menu dropdown."""
        width = self.window.width
        height = self.window.height
        
        menu_width = 250
        menu_height = len(self.menu_items) * 30 + 20
        menu_x = 10
        menu_y = height - menu_height - 10
        
        glPushMatrix()
        glLoadIdentity()
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, width, 0, height, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glDisable(GL_DEPTH_TEST)
        glDisable(GL_LIGHTING)
        
        # Background
        glColor4f(0.15, 0.15, 0.2, 0.95)
        glBegin(GL_QUADS)
        glVertex2f(menu_x, menu_y)
        glVertex2f(menu_x + menu_width, menu_y)
        glVertex2f(menu_x + menu_width, menu_y + menu_height)
        glVertex2f(menu_x, menu_y + menu_height)
        glEnd()
        
        # Border
        glColor4f(0.4, 0.4, 0.5, 1.0)
        glLineWidth(2.0)
        glBegin(GL_LINE_LOOP)
        glVertex2f(menu_x, menu_y)
        glVertex2f(menu_x + menu_width, menu_y)
        glVertex2f(menu_x + menu_width, menu_y + menu_height)
        glVertex2f(menu_x, menu_y + menu_height)
        glEnd()
        
        # Menu title
        title_label = pyglet.text.Label(
            "Menu",
            font_name='Arial',
            font_size=14,
            bold=True,
            color=(255, 255, 255, 255),
            x=menu_x + 10,
            y=height - menu_y - 15,
            anchor_x='left',
            anchor_y='top'
        )
        title_label.draw()
        
        # Menu items
        item_height = 30
        start_y = menu_y + menu_height - 35
        
        for i, (item_text, _) in enumerate(self.menu_items):
            item_y = start_y - (i * item_height)
            
            # Highlight selected item
            if i == self.selected_menu_item:
                glColor4f(0.3, 0.3, 0.4, 1.0)
                glBegin(GL_QUADS)
                glVertex2f(menu_x + 5, item_y - 5)
                glVertex2f(menu_x + menu_width - 5, item_y - 5)
                glVertex2f(menu_x + menu_width - 5, item_y + item_height - 5)
                glVertex2f(menu_x + 5, item_y + item_height - 5)
                glEnd()
            
            # Item text
            color = (255, 255, 255, 255) if i == self.selected_menu_item else (200, 200, 200, 255)
            item_label = pyglet.text.Label(
                item_text,
                font_name='Arial',
                font_size=12,
                color=color,
                x=menu_x + 15,
                y=height - item_y,
                anchor_x='left',
                anchor_y='top'
            )
            item_label.draw()
        
        # Instructions
        instr_label = pyglet.text.Label(
            "UP/DOWN: Navigate | ENTER: Select | M: Close",
            font_name='Arial',
            font_size=9,
            color=(150, 150, 150, 255),
            x=menu_x + 10,
            y=height - menu_y - menu_height + 5,
            anchor_x='left',
            anchor_y='top'
        )
        instr_label.draw()
        
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
