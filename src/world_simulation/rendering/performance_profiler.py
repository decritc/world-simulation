"""Performance profiling module for rendering performance tracking."""

import time
from typing import Dict


class PerformanceProfiler:
    """Handles performance profiling for rendering operations."""
    
    def __init__(self, log_file: str = "profiling.log", log_interval: int = 60):
        """
        Initialize the performance profiler.
        
        Args:
            log_file: Path to log file for profiling reports
            log_interval: Frames between console log outputs
        """
        self.enabled = True  # Enable by default
        self.log_file = log_file
        self.log_interval = log_interval
        
        # Average times (rolling averages updated every 60 frames)
        self.average_times: Dict[str, float] = {
            'terrain': 0.0,
            'vegetation': 0.0,
            'trees': 0.0,
            'npcs': 0.0,
            'animals': 0.0,
            'houses': 0.0,
            'overlay': 0.0,
            'total': 0.0
        }
        
        # Current frame times (updated every frame for immediate display)
        self.current_times: Dict[str, float] = {
            'terrain': 0.0,
            'vegetation': 0.0,
            'trees': 0.0,
            'npcs': 0.0,
            'animals': 0.0,
            'houses': 0.0,
            'overlay': 0.0,
            'total': 0.0
        }
        
        self.frame_count = 0
    
    def update_frame_times(self, frame_times: Dict[str, float], total_time: float):
        """
        Update profiling data with current frame times.
        
        Args:
            frame_times: Dictionary of component times (terrain, vegetation, etc.)
            total_time: Total frame time in seconds
        """
        self.frame_count += 1
        
        # Store current frame times
        self.current_times = frame_times.copy()
        self.current_times['total'] = total_time
        
        # Update rolling averages (every 60 frames)
        if self.frame_count % 60 == 0:
            alpha = 0.1  # Smoothing factor
            for key in self.average_times:
                if key in frame_times:
                    self.average_times[key] = (
                        self.average_times[key] * (1 - alpha) + 
                        frame_times[key] * alpha
                    )
            self.average_times['total'] = (
                self.average_times['total'] * (1 - alpha) + 
                total_time * alpha
            )
            
            # Log to console if enabled and interval reached
            if self.enabled and total_time > 0 and self.frame_count % self.log_interval == 0:
                self._log_report(frame_times, total_time)
    
    def _log_report(self, frame_times: Dict[str, float], total_time: float):
        """
        Generate and log a detailed performance report.
        
        Args:
            frame_times: Dictionary of component times
            total_time: Total frame time
        """
        fps = 1.0 / total_time if total_time > 0 else 0
        avg_fps = 1.0 / self.average_times['total'] if self.average_times['total'] > 0 else 0
        
        report_lines = []
        report_lines.append("\n" + "="*60)
        report_lines.append(f"PERFORMANCE PROFILING REPORT (Frame {self.frame_count})")
        report_lines.append("="*60)
        report_lines.append(f"Current FPS: {fps:.1f} | Average FPS (last 60 frames): {avg_fps:.1f}")
        report_lines.append(f"Target: 60 FPS | Status: {'MEETS TARGET' if fps >= 60 else 'BELOW TARGET'}")
        report_lines.append("-"*60)
        report_lines.append(f"Total Frame Time: {total_time*1000:.2f}ms ({100:.0f}%)")
        
        # Component details
        report_lines.append(f"  - Terrain:       {frame_times['terrain']*1000:.2f}ms ({frame_times['terrain']/total_time*100:.1f}%) - Avg: {self.average_times['terrain']*1000:.2f}ms")
        report_lines.append(f"  - Vegetation:    {frame_times['vegetation']*1000:.2f}ms ({frame_times['vegetation']/total_time*100:.1f}%) - Avg: {self.average_times['vegetation']*1000:.2f}ms")
        report_lines.append(f"  - Trees:         {frame_times['trees']*1000:.2f}ms ({frame_times['trees']/total_time*100:.1f}%) - Avg: {self.average_times['trees']*1000:.2f}ms")
        report_lines.append(f"  - NPCs:          {frame_times['npcs']*1000:.2f}ms ({frame_times['npcs']/total_time*100:.1f}%) - Avg: {self.average_times['npcs']*1000:.2f}ms")
        report_lines.append(f"  - Animals:       {frame_times['animals']*1000:.2f}ms ({frame_times['animals']/total_time*100:.1f}%) - Avg: {self.average_times['animals']*1000:.2f}ms")
        report_lines.append(f"  - Houses:        {frame_times['houses']*1000:.2f}ms ({frame_times['houses']/total_time*100:.1f}%) - Avg: {self.average_times['houses']*1000:.2f}ms")
        report_lines.append(f"  - Overlay:       {frame_times['overlay']*1000:.2f}ms ({frame_times['overlay']/total_time*100:.1f}%) - Avg: {self.average_times['overlay']*1000:.2f}ms")
        report_lines.append("="*60)
        
        # Identify bottlenecks
        components = [
            ('Terrain', frame_times['terrain'], self.average_times['terrain']),
            ('Vegetation', frame_times['vegetation'], self.average_times['vegetation']),
            ('Trees', frame_times['trees'], self.average_times['trees']),
            ('NPCs', frame_times['npcs'], self.average_times['npcs']),
            ('Animals', frame_times['animals'], self.average_times['animals']),
            ('Houses', frame_times['houses'], self.average_times['houses']),
            ('Overlay', frame_times['overlay'], self.average_times['overlay']),
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
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(report_text + "\n")
        except:
            pass  # Ignore file write errors
    
    def get_stats_for_overlay(self) -> list:
        """
        Get formatted statistics for overlay display.
        
        Returns:
            List of formatted stat strings
        """
        if not self.enabled:
            return []
        
        current = self.current_times
        current_fps = 1.0 / current['total'] if current['total'] > 0 else 0
        
        # Use averages if available, otherwise use current
        avg_total = self.average_times['total'] if self.average_times['total'] > 0 else current['total']
        avg_fps = 1.0 / avg_total if avg_total > 0 else 0
        
        stats = [
            "",
            "--- Performance Profiling ---",
            f"Current FPS: {current_fps:.1f}",
            f"Avg FPS: {avg_fps:.1f}",
            f"Target: 60 FPS | Status: {'✓ MEETS' if current_fps >= 60 else '✗ BELOW'}",
            "",
            "Current Frame Times (ms):",
            f"  Terrain:    {current['terrain']*1000:.1f}ms",
            f"  Vegetation: {current['vegetation']*1000:.1f}ms",
            f"  Trees:      {current['trees']*1000:.1f}ms",
            f"  NPCs:       {current['npcs']*1000:.1f}ms",
            f"  Animals:    {current['animals']*1000:.1f}ms",
            f"  Houses:     {current['houses']*1000:.1f}ms",
            f"  Overlay:    {current['overlay']*1000:.1f}ms",
            f"  Total:      {current['total']*1000:.1f}ms",
        ]
        
        # Calculate percentages
        total_ms = current['total'] * 1000
        if total_ms > 0:
            stats.extend([
                "",
                "Percentage of Frame Time:",
                f"  Terrain:    {current['terrain']*1000/total_ms*100:.1f}%",
                f"  Vegetation: {current['vegetation']*1000/total_ms*100:.1f}%",
                f"  Trees:      {current['trees']*1000/total_ms*100:.1f}%",
                f"  NPCs:       {current['npcs']*1000/total_ms*100:.1f}%",
                f"  Animals:    {current['animals']*1000/total_ms*100:.1f}%",
                f"  Houses:     {current['houses']*1000/total_ms*100:.1f}%",
                f"  Overlay:    {current['overlay']*1000/total_ms*100:.1f}%",
            ])
        
        return stats
    
    def reset(self):
        """Reset all profiling data."""
        for key in self.average_times:
            self.average_times[key] = 0.0
        for key in self.current_times:
            self.current_times[key] = 0.0
        self.frame_count = 0
    
    def toggle(self) -> bool:
        """
        Toggle profiling on/off.
        
        Returns:
            New enabled state
        """
        self.enabled = not self.enabled
        if self.enabled:
            self.reset()
        return self.enabled
