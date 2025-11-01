"""Historian system for tracking NPC stories and colony history."""

import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from collections import defaultdict


class Historian:
    """Centralized timekeeper that logs NPC stories and colony history."""
    
    def __init__(self, log_file: str = "colony_history.txt"):
        """
        Initialize the historian.
        
        Args:
            log_file: Path to the log file
        """
        self.log_file = log_file
        self.generation_counter = 0  # Track current generation number
        self.npc_generations: Dict[int, int] = {}  # NPC ID -> generation number
        self.npc_parents: Dict[int, Tuple[Optional[int], Optional[int]]] = {}  # NPC ID -> (parent1_id, parent2_id)
        self.npc_children: Dict[int, List[int]] = defaultdict(list)  # NPC ID -> list of child IDs
        self.generation_stats: Dict[int, Dict] = defaultdict(dict)  # Generation -> stats
        self.colony_start_time = datetime.now()
        self.npc_names: Dict[int, str] = {}  # Cache NPC names by ID
        
        # In-memory log buffer for in-app viewing
        self.log_buffer: List[str] = []  # List of log lines for display
        self.max_buffer_lines = 10000  # Maximum lines to keep in memory
        
        # Delete old file and start fresh
        if os.path.exists(self.log_file):
            os.remove(self.log_file)
        
        # Write header
        self._write_header()
        
        # Initialize log buffer with header
        self._add_to_buffer("=" * 80)
        self._add_to_buffer("COLONY HISTORY LOG")
        self._add_to_buffer("=" * 80)
        self._add_to_buffer(f"Simulation Started: {self.colony_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        self._add_to_buffer("=" * 80)
        self._add_to_buffer("")
    
    def _write_header(self):
        """Write the header to the log file."""
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("COLONY HISTORY LOG\n")
            f.write("=" * 80 + "\n")
            f.write(f"Simulation Started: {self.colony_start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")
    
    def register_npc_birth(self, npc_id: int, npc_name: str, parent1_id: Optional[int] = None, 
                          parent2_id: Optional[int] = None, world_time: float = 0.0, 
                          day_number: int = 0):
        """
        Log the birth of an NPC.
        
        Args:
            npc_id: Unique identifier for the NPC
            npc_name: Name of the NPC
            parent1_id: ID of first parent (None if initial population)
            parent2_id: ID of second parent (None if initial population)
            world_time: Current simulation time
            day_number: Current day number
        """
        # Determine generation
        if parent1_id is None and parent2_id is None:
            # Initial population - Generation 0
            generation = 0
        else:
            # Get generation from parents (should be same)
            parent1_gen = self.npc_generations.get(parent1_id, 0)
            parent2_gen = self.npc_generations.get(parent2_id, 0)
            generation = max(parent1_gen, parent2_gen) + 1
        
        self.npc_generations[npc_id] = generation
        self.npc_parents[npc_id] = (parent1_id, parent2_id)
        self.npc_names[npc_id] = npc_name  # Cache name
        
        if parent1_id is not None:
            self.npc_children[parent1_id].append(npc_id)
        if parent2_id is not None:
            self.npc_children[parent2_id].append(npc_id)
        
        # Update generation counter
        self.generation_counter = max(self.generation_counter, generation)
        
        # Log event
        timestamp = self._format_time(world_time, day_number)
        parent_info = ""
        if parent1_id is not None and parent2_id is not None:
            parent1_name = self.npc_names.get(parent1_id, f"NPC-{parent1_id}")
            parent2_name = self.npc_names.get(parent2_id, f"NPC-{parent2_id}")
            parent_info = f" (Parents: {parent1_name} & {parent2_name})"
        elif parent1_id is not None:
            parent1_name = self.npc_names.get(parent1_id, f"NPC-{parent1_id}")
            parent_info = f" (Parent: {parent1_name})"
        
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(f"[{timestamp}] BIRTH - Generation {generation}\n")
            f.write(f"  Name: {npc_name}{parent_info}\n")
            f.write(f"  NPC ID: {npc_id}\n\n")
        
        # Add to buffer
        self._add_to_buffer(f"[{timestamp}] BIRTH - Generation {generation}")
        self._add_to_buffer(f"  Name: {npc_name}{parent_info}")
        self._add_to_buffer(f"  NPC ID: {npc_id}")
        self._add_to_buffer("")
    
    def register_npc_death(self, npc_id: int, npc_name: str, age: float, cause: str,
                          world_time: float = 0.0, day_number: int = 0, 
                          fruit_collected: int = 0, animals_hunted: int = 0):
        """
        Log the death of an NPC.
        
        Args:
            npc_id: Unique identifier for the NPC
            npc_name: Name of the NPC
            age: Age at death
            cause: Cause of death (e.g., "old age", "starvation", "night exposure")
            world_time: Current simulation time
            day_number: Current day number
            fruit_collected: Total fruit collected in lifetime
            animals_hunted: Total animals hunted in lifetime
        """
        generation = self.npc_generations.get(npc_id, 0)
        timestamp = self._format_time(world_time, day_number)
        
        # Get children count
        children_count = len(self.npc_children.get(npc_id, []))
        
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(f"[{timestamp}] DEATH - Generation {generation}\n")
            f.write(f"  Name: {npc_name}\n")
            f.write(f"  Age: {age:.1f} seconds ({age/60:.1f} minutes)\n")
            f.write(f"  Cause: {cause}\n")
            f.write(f"  Achievements:\n")
            f.write(f"    - Fruit Collected: {fruit_collected}\n")
            f.write(f"    - Animals Hunted: {animals_hunted}\n")
            f.write(f"    - Offspring: {children_count}\n")
            f.write(f"  NPC ID: {npc_id}\n\n")
        
        # Add to buffer
        self._add_to_buffer(f"[{timestamp}] DEATH - Generation {generation}")
        self._add_to_buffer(f"  Name: {npc_name}")
        self._add_to_buffer(f"  Age: {age:.1f} seconds ({age/60:.1f} minutes)")
        self._add_to_buffer(f"  Cause: {cause}")
        self._add_to_buffer(f"  Achievements:")
        self._add_to_buffer(f"    - Fruit Collected: {fruit_collected}")
        self._add_to_buffer(f"    - Animals Hunted: {animals_hunted}")
        self._add_to_buffer(f"    - Offspring: {children_count}")
        self._add_to_buffer(f"  NPC ID: {npc_id}")
        self._add_to_buffer("")
    
    def register_reproduction(self, parent1_id: int, parent1_name: str,
                             parent2_id: int, parent2_name: str,
                             offspring_id: int, offspring_name: str,
                             world_time: float = 0.0, day_number: int = 0):
        """
        Log a reproduction event.
        
        Args:
            parent1_id: ID of first parent
            parent1_name: Name of first parent
            parent2_id: ID of second parent
            parent2_name: Name of second parent
            offspring_id: ID of the offspring
            offspring_name: Name of the offspring
            world_time: Current simulation time
            day_number: Current day number
        """
        timestamp = self._format_time(world_time, day_number)
        generation = self.npc_generations.get(parent1_id, 0)
        
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(f"[{timestamp}] REPRODUCTION - Generation {generation}\n")
            f.write(f"  Parents: {parent1_name} & {parent2_name}\n")
            f.write(f"  Offspring: {offspring_name}\n")
            f.write(f"  Parent IDs: {parent1_id} & {parent2_id}\n")
            f.write(f"  Offspring ID: {offspring_id}\n\n")
        
        # Add to buffer
        self._add_to_buffer(f"[{timestamp}] REPRODUCTION - Generation {generation}")
        self._add_to_buffer(f"  Parents: {parent1_name} & {parent2_name}")
        self._add_to_buffer(f"  Offspring: {offspring_name}")
        self._add_to_buffer(f"  Parent IDs: {parent1_id} & {parent2_id}")
        self._add_to_buffer(f"  Offspring ID: {offspring_id}")
        self._add_to_buffer("")
    
    def register_milestone(self, npc_id: int, npc_name: str, milestone: str,
                           world_time: float = 0.0, day_number: int = 0, 
                           details: Optional[str] = None):
        """
        Log a milestone event.
        
        Args:
            npc_id: Unique identifier for the NPC
            npc_name: Name of the NPC
            milestone: Type of milestone (e.g., "reached_adult", "first_hunt", "first_fruit")
            world_time: Current simulation time
            day_number: Current day number
            details: Additional details about the milestone
        """
        generation = self.npc_generations.get(npc_id, 0)
        timestamp = self._format_time(world_time, day_number)
        
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(f"[{timestamp}] MILESTONE - Generation {generation}\n")
            f.write(f"  Name: {npc_name}\n")
            f.write(f"  Milestone: {milestone}\n")
            if details:
                f.write(f"  Details: {details}\n")
            f.write(f"  NPC ID: {npc_id}\n\n")
        
        # Add to buffer
        self._add_to_buffer(f"[{timestamp}] MILESTONE - Generation {generation}")
        self._add_to_buffer(f"  Name: {npc_name}")
        self._add_to_buffer(f"  Milestone: {milestone}")
        if details:
            self._add_to_buffer(f"  Details: {details}")
        self._add_to_buffer(f"  NPC ID: {npc_id}")
        self._add_to_buffer("")
    
    def register_achievement(self, npc_id: int, npc_name: str, achievement: str,
                            value: float, world_time: float = 0.0, day_number: int = 0):
        """
        Log an achievement event.
        
        Args:
            npc_id: Unique identifier for the NPC
            npc_name: Name of the NPC
            achievement: Type of achievement (e.g., "fruit_collected", "animals_hunted")
            value: Value of the achievement
            world_time: Current simulation time
            day_number: Current day number
        """
        generation = self.npc_generations.get(npc_id, 0)
        timestamp = self._format_time(world_time, day_number)
        
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(f"[{timestamp}] ACHIEVEMENT - Generation {generation}\n")
            f.write(f"  Name: {npc_name}\n")
            f.write(f"  Achievement: {achievement}\n")
            f.write(f"  Value: {value}\n")
            f.write(f"  NPC ID: {npc_id}\n\n")
        
        # Add to buffer
        self._add_to_buffer(f"[{timestamp}] ACHIEVEMENT - Generation {generation}")
        self._add_to_buffer(f"  Name: {npc_name}")
        self._add_to_buffer(f"  Achievement: {achievement}")
        self._add_to_buffer(f"  Value: {value}")
        self._add_to_buffer(f"  NPC ID: {npc_id}")
        self._add_to_buffer("")
    
    def generate_generation_summary(self, generation: int, world_time: float, day_number: int,
                                   population: List, alive_count: int, dead_count: int):
        """
        Generate a summary for a generation.
        
        Args:
            generation: Generation number
            world_time: Current simulation time
            day_number: Current day number
            population: List of all NPCs in this generation
            alive_count: Number of alive NPCs
            dead_count: Number of dead NPCs
        """
        if not population:
            return
        
        # Calculate statistics
        ages = [npc.age for npc in population]
        lifespans = [npc.age for npc in population if not npc.is_alive]
        fruit_totals = [npc.fruit_collected for npc in population]
        hunt_totals = [npc.animals_hunted for npc in population]
        reproduction_counts = [len(self.npc_children.get(id(npc), [])) for npc in population]
        
        timestamp = self._format_time(world_time, day_number)
        
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write("\n" + "=" * 80 + "\n")
            f.write(f"GENERATION {generation} SUMMARY\n")
            f.write("=" * 80 + "\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Population Size: {len(population)}\n")
            f.write(f"Alive: {alive_count}\n")
            f.write(f"Deceased: {dead_count}\n")
            
            if ages:
                f.write(f"\nAge Statistics:\n")
                f.write(f"  Average Age: {sum(ages) / len(ages):.1f} seconds\n")
                f.write(f"  Max Age: {max(ages):.1f} seconds\n")
                f.write(f"  Min Age: {min(ages):.1f} seconds\n")
            
            if lifespans:
                f.write(f"\nLifespan Statistics (deceased):\n")
                f.write(f"  Average Lifespan: {sum(lifespans) / len(lifespans):.1f} seconds\n")
                f.write(f"  Max Lifespan: {max(lifespans):.1f} seconds\n")
                f.write(f"  Min Lifespan: {min(lifespans):.1f} seconds\n")
            
            if fruit_totals:
                f.write(f"\nFruit Collection:\n")
                f.write(f"  Total Fruit Collected: {sum(fruit_totals)}\n")
                f.write(f"  Average per NPC: {sum(fruit_totals) / len(fruit_totals):.1f}\n")
                f.write(f"  Max Collected: {max(fruit_totals)}\n")
            
            if hunt_totals:
                f.write(f"\nHunting Statistics:\n")
                f.write(f"  Total Animals Hunted: {sum(hunt_totals)}\n")
                f.write(f"  Average per NPC: {sum(hunt_totals) / len(hunt_totals):.1f}\n")
                f.write(f"  Max Hunted: {max(hunt_totals)}\n")
            
            if reproduction_counts:
                f.write(f"\nReproduction Statistics:\n")
                f.write(f"  Total Offspring: {sum(reproduction_counts)}\n")
                f.write(f"  Average Offspring per NPC: {sum(reproduction_counts) / len(reproduction_counts):.1f}\n")
                f.write(f"  Max Offspring: {max(reproduction_counts)}\n")
                reproducing_npcs = sum(1 for count in reproduction_counts if count > 0)
                f.write(f"  NPCs with Offspring: {reproducing_npcs}/{len(population)}\n")
            
            f.write("\n" + "=" * 80 + "\n\n")
        
        # Add summary to buffer
        self._add_to_buffer("")
        self._add_to_buffer("=" * 80)
        self._add_to_buffer(f"GENERATION {generation} SUMMARY")
        self._add_to_buffer("=" * 80)
        self._add_to_buffer(f"Timestamp: {timestamp}")
        self._add_to_buffer(f"Population Size: {len(population)}")
        self._add_to_buffer(f"Alive: {alive_count}")
        self._add_to_buffer(f"Deceased: {dead_count}")
        
        if ages:
            self._add_to_buffer("")
            self._add_to_buffer("Age Statistics:")
            self._add_to_buffer(f"  Average Age: {sum(ages) / len(ages):.1f} seconds")
            self._add_to_buffer(f"  Max Age: {max(ages):.1f} seconds")
            self._add_to_buffer(f"  Min Age: {min(ages):.1f} seconds")
        
        if lifespans:
            self._add_to_buffer("")
            self._add_to_buffer("Lifespan Statistics (deceased):")
            self._add_to_buffer(f"  Average Lifespan: {sum(lifespans) / len(lifespans):.1f} seconds")
            self._add_to_buffer(f"  Max Lifespan: {max(lifespans):.1f} seconds")
            self._add_to_buffer(f"  Min Lifespan: {min(lifespans):.1f} seconds")
        
        if fruit_totals:
            self._add_to_buffer("")
            self._add_to_buffer("Fruit Collection:")
            self._add_to_buffer(f"  Total Fruit Collected: {sum(fruit_totals)}")
            self._add_to_buffer(f"  Average per NPC: {sum(fruit_totals) / len(fruit_totals):.1f}")
            self._add_to_buffer(f"  Max Collected: {max(fruit_totals)}")
        
        if hunt_totals:
            self._add_to_buffer("")
            self._add_to_buffer("Hunting Statistics:")
            self._add_to_buffer(f"  Total Animals Hunted: {sum(hunt_totals)}")
            self._add_to_buffer(f"  Average per NPC: {sum(hunt_totals) / len(hunt_totals):.1f}")
            self._add_to_buffer(f"  Max Hunted: {max(hunt_totals)}")
        
        if reproduction_counts:
            self._add_to_buffer("")
            self._add_to_buffer("Reproduction Statistics:")
            self._add_to_buffer(f"  Total Offspring: {sum(reproduction_counts)}")
            self._add_to_buffer(f"  Average Offspring per NPC: {sum(reproduction_counts) / len(reproduction_counts):.1f}")
            self._add_to_buffer(f"  Max Offspring: {max(reproduction_counts)}")
            reproducing_npcs = sum(1 for count in reproduction_counts if count > 0)
            self._add_to_buffer(f"  NPCs with Offspring: {reproducing_npcs}/{len(population)}")
        
        self._add_to_buffer("")
        self._add_to_buffer("=" * 80)
        self._add_to_buffer("")
    
    def generate_colony_summary(self, world_time: float, day_number: int,
                                all_npcs: List, alive_npcs: List):
        """
        Generate an overall colony summary.
        
        Args:
            world_time: Current simulation time
            day_number: Current day number
            all_npcs: List of all NPCs (alive and dead)
            alive_npcs: List of currently alive NPCs
        """
        timestamp = self._format_time(world_time, day_number)
        
        # Calculate overall statistics
        total_npcs = len(all_npcs)
        alive_count = len(alive_npcs)
        dead_count = total_npcs - alive_count
        
        # Generation distribution
        gen_counts = defaultdict(int)
        for npc in all_npcs:
            gen = self.npc_generations.get(id(npc), 0)
            gen_counts[gen] += 1
        
        # Population growth rate
        current_gen = max(gen_counts.keys()) if gen_counts else 0
        previous_gen_count = gen_counts.get(current_gen - 1, 0)
        current_gen_count = gen_counts.get(current_gen, 0)
        growth_rate = ((current_gen_count - previous_gen_count) / previous_gen_count * 100) if previous_gen_count > 0 else 0
        
        # Survival metrics
        survival_rate = (alive_count / total_npcs * 100) if total_npcs > 0 else 0
        
        # Overall achievements
        total_fruit = sum(npc.fruit_collected for npc in all_npcs)
        total_hunts = sum(npc.animals_hunted for npc in all_npcs)
        total_offspring = sum(len(self.npc_children.get(id(npc), [])) for npc in all_npcs)
        
        # Average lifespan (for deceased NPCs)
        deceased_npcs = [npc for npc in all_npcs if not npc.is_alive]
        avg_lifespan = sum(npc.age for npc in deceased_npcs) / len(deceased_npcs) if deceased_npcs else 0
        
        # Success/failure assessment
        success_factors = []
        if growth_rate > 0:
            success_factors.append("Growing Population")
        else:
            success_factors.append("Declining Population")
        
        if avg_lifespan > 300:  # More than 5 minutes
            success_factors.append("Good Lifespan")
        else:
            success_factors.append("Short Lifespan")
        
        if total_offspring > total_npcs * 0.5:
            success_factors.append("High Reproduction Rate")
        else:
            success_factors.append("Low Reproduction Rate")
        
        if survival_rate > 50:
            success_factors.append("High Survival Rate")
        else:
            success_factors.append("Low Survival Rate")
        
        success_count = sum(1 for factor in success_factors if "Good" in factor or "High" in factor or "Growing" in factor)
        overall_status = "SUCCESS" if success_count >= 3 else "STABLE" if success_count >= 2 else "FAILURE"
        
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write("\n" + "=" * 80 + "\n")
            f.write("COLONY SUMMARY\n")
            f.write("=" * 80 + "\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Simulation Time: {world_time:.1f} seconds ({world_time/60:.1f} minutes)\n")
            f.write(f"Day Number: {day_number}\n")
            f.write(f"Overall Status: {overall_status}\n\n")
            
            f.write(f"Population Statistics:\n")
            f.write(f"  Total NPCs (All Time): {total_npcs}\n")
            f.write(f"  Currently Alive: {alive_count}\n")
            f.write(f"  Deceased: {dead_count}\n")
            f.write(f"  Survival Rate: {survival_rate:.1f}%\n")
            f.write(f"  Current Generation: {current_gen}\n")
            f.write(f"  Population Growth Rate: {growth_rate:.1f}%\n\n")
            
            f.write(f"Generation Distribution:\n")
            for gen in sorted(gen_counts.keys()):
                f.write(f"  Generation {gen}: {gen_counts[gen]} NPCs\n")
            
            f.write(f"\nColony Achievements:\n")
            f.write(f"  Total Fruit Collected: {total_fruit}\n")
            f.write(f"  Total Animals Hunted: {total_hunts}\n")
            f.write(f"  Total Offspring Produced: {total_offspring}\n")
            f.write(f"  Average Lifespan: {avg_lifespan:.1f} seconds ({avg_lifespan/60:.1f} minutes)\n\n")
            
            f.write(f"Success Factors:\n")
            for factor in success_factors:
                f.write(f"  - {factor}\n")
            
            f.write("\n" + "=" * 80 + "\n\n")
        
        # Add summary to buffer
        self._add_to_buffer("")
        self._add_to_buffer("=" * 80)
        self._add_to_buffer("COLONY SUMMARY")
        self._add_to_buffer("=" * 80)
        self._add_to_buffer(f"Timestamp: {timestamp}")
        self._add_to_buffer(f"Simulation Time: {world_time:.1f} seconds ({world_time/60:.1f} minutes)")
        self._add_to_buffer(f"Day Number: {day_number}")
        self._add_to_buffer(f"Overall Status: {overall_status}")
        self._add_to_buffer("")
        self._add_to_buffer("Population Statistics:")
        self._add_to_buffer(f"  Total NPCs (All Time): {total_npcs}")
        self._add_to_buffer(f"  Currently Alive: {alive_count}")
        self._add_to_buffer(f"  Deceased: {dead_count}")
        self._add_to_buffer(f"  Survival Rate: {survival_rate:.1f}%")
        self._add_to_buffer(f"  Current Generation: {current_gen}")
        self._add_to_buffer(f"  Population Growth Rate: {growth_rate:.1f}%")
        self._add_to_buffer("")
        self._add_to_buffer("Generation Distribution:")
        for gen in sorted(gen_counts.keys()):
            self._add_to_buffer(f"  Generation {gen}: {gen_counts[gen]} NPCs")
        self._add_to_buffer("")
        self._add_to_buffer("Colony Achievements:")
        self._add_to_buffer(f"  Total Fruit Collected: {total_fruit}")
        self._add_to_buffer(f"  Total Animals Hunted: {total_hunts}")
        self._add_to_buffer(f"  Total Offspring Produced: {total_offspring}")
        self._add_to_buffer(f"  Average Lifespan: {avg_lifespan:.1f} seconds ({avg_lifespan/60:.1f} minutes)")
        self._add_to_buffer("")
        self._add_to_buffer("Success Factors:")
        for factor in success_factors:
            self._add_to_buffer(f"  - {factor}")
        self._add_to_buffer("")
        self._add_to_buffer("=" * 80)
        self._add_to_buffer("")
    
    def _add_to_buffer(self, line: str):
        """Add a line to the log buffer."""
        self.log_buffer.append(line)
        # Limit buffer size to prevent memory issues
        if len(self.log_buffer) > self.max_buffer_lines:
            # Remove oldest entries (keep most recent)
            self.log_buffer = self.log_buffer[-self.max_buffer_lines:]
    
    def get_log_lines(self) -> List[str]:
        """
        Get all log lines for display.
        
        Returns:
            List of log lines
        """
        return self.log_buffer.copy()
    
    def _format_time(self, world_time: float, day_number: int) -> str:
        """Format time for logging."""
        minutes = int(world_time / 60)
        seconds = int(world_time % 60)
        return f"Day {day_number} | {minutes:02d}:{seconds:02d}"
    
    def _get_npc_name_from_id(self, npc_id: int) -> str:
        """Get NPC name from ID (for parent references)."""
        return self.npc_names.get(npc_id, f"NPC-{npc_id}")
    
    def get_generation(self, npc_id: int) -> int:
        """Get the generation number for an NPC."""
        return self.npc_generations.get(npc_id, 0)
    
    def get_parents(self, npc_id: int) -> Tuple[Optional[int], Optional[int]]:
        """Get the parent IDs for an NPC."""
        return self.npc_parents.get(npc_id, (None, None))

