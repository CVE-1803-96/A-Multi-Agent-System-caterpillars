import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle, Rectangle
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional
import math

@dataclass
class Caterpillar:
    def __init__(self, agent_id: int, position: Tuple[float, float]):
        self.id = agent_id
        self.position = np.array(position, dtype=float)
        self.at_target = False
        self.at_tree = False
        self.color = 'green'
        self.movement_history = [tuple(position)]
    
    def distance_to(self, target: np.ndarray) -> float:
        return np.linalg.norm(self.position - target)
    
    def move_toward(self, target: np.ndarray, step_size: float = 0.5):
        
        direction = target - self.position
        distance = np.linalg.norm(direction)
        
        if distance > 0:
            
            noise_angle = random.uniform(-0.5, 0.5)  
            direction_normalized = direction / distance
            
            
            cos_angle = np.cos(noise_angle)
            sin_angle = np.sin(noise_angle)
            new_direction = np.array([
                direction_normalized[0] * cos_angle - direction_normalized[1] * sin_angle,
                direction_normalized[0] * sin_angle + direction_normalized[1] * cos_angle
            ])
            
            
            movement = new_direction * step_size * random.uniform(0.7, 1.3)
            self.position += movement
            self.movement_history.append(tuple(self.position.copy()))

class NonOptimizedCaterpillarSimulation:
    def __init__(self, 
                 num_caterpillars: int = 15,
                 tree_position: Tuple[float, float] = (2, 2),
                 target_spot: Tuple[float, float] = (8, 8)):
        
        self.caterpillars: List[Caterpillar] = []
        self.tree_position = np.array(tree_position)
        self.target_spot = np.array(target_spot)
        self.rain_active = False
        self.steps = 0
        self.rain_start_step = 0
        self.rain_duration = 0
        self.metrics = {
            'at_target': [],
            'at_tree': [],
            'moving': [],
            'total_distance_traveled': []
        }
        
        
        self.initialize_caterpillars(num_caterpillars)
    
    def initialize_caterpillars(self, num_caterpillars: int):
        
        for i in range(num_caterpillars):
            
            angle = random.uniform(0, 2 * math.pi)
            distance = random.uniform(0.5, 2.0)
            x = self.tree_position[0] + distance * math.cos(angle)
            y = self.tree_position[1] + distance * math.sin(angle)
            
            caterpillar = Caterpillar(i, (x, y))
            self.caterpillars.append(caterpillar)
    
    def toggle_rain(self, duration: int = 20):
        """Activate or deactivate rain randomly"""
        if not self.rain_active and random.random() < 0.05: 
            self.rain_active = True
            self.rain_start_step = self.steps
            self.rain_duration = duration
            print(f"Rain started at step {self.steps}")
        elif self.rain_active and (self.steps - self.rain_start_step) >= self.rain_duration:
            self.rain_active = False
            print(f"Rain stopped at step {self.steps}")
    
    def step(self):
        
        self.steps += 1
        
        
        self.toggle_rain()
        
        
        caterpillars_moving = 0
        
        for caterpillar in self.caterpillars:
            if caterpillar.at_target or caterpillar.at_tree:
                continue
                
            caterpillars_moving += 1
            
            if self.rain_active:
                
                if caterpillar.distance_to(self.tree_position) > 0.5:
                    caterpillar.move_toward(self.tree_position, step_size=0.3)
                else:
                    caterpillar.at_tree = True
                    caterpillar.color = 'blue'
            else:
                
                if caterpillar.distance_to(self.target_spot) > 0.5:
                    caterpillar.move_toward(self.target_spot, step_size=0.4)
                else:
                    caterpillar.at_target = True
                    caterpillar.color = 'red'
        
        
        at_target = sum(1 for c in self.caterpillars if c.at_target)
        at_tree = sum(1 for c in self.caterpillars if c.at_tree)
        total_distance = sum(
            sum(np.linalg.norm(np.array(c.movement_history[i+1]) - np.array(c.movement_history[i])) 
                for i in range(len(c.movement_history)-1))
            for c in self.caterpillars
        )
        
        self.metrics['at_target'].append(at_target)
        self.metrics['at_tree'].append(at_tree)
        self.metrics['moving'].append(caterpillars_moving)
        self.metrics['total_distance_traveled'].append(total_distance)
        
        
        all_done = all(c.at_target or c.at_tree for c in self.caterpillars)
        return all_done

class NonOptimizedVisualization:
    def __init__(self, simulation: NonOptimizedCaterpillarSimulation):
        self.simulation = simulation
        self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(1, 3, figsize=(18, 6))
        self.setup_plots()
    
    def setup_plots(self):
        
        self.ax1.set_xlim(0, 10)
        self.ax1.set_ylim(0, 10)
        self.ax1.set_aspect('equal')
        self.ax1.set_title('Non-Optimized Caterpillar Movement')
        self.ax1.set_xlabel('X Position')
        self.ax1.set_ylabel('Y Position')
        self.ax1.grid(True, alpha=0.3)
        
        
        self.ax2.set_xlim(0, 200)
        self.ax2.set_ylim(0, len(self.simulation.caterpillars) + 2)
        self.ax2.set_title('Caterpillar Status Over Time')
        self.ax2.set_xlabel('Simulation Steps')
        self.ax2.set_ylabel('Number of Caterpillars')
        self.ax2.grid(True, alpha=0.3)
        
        
        self.ax3.set_xlim(0, 200)
        self.ax3.set_ylim(0, 100)
        self.ax3.set_title('Total Distance Traveled')
        self.ax3.set_xlabel('Simulation Steps')
        self.ax3.set_ylabel('Total Distance')
        self.ax3.grid(True, alpha=0.3)
    
    def update(self, frame):
        
        all_done = self.simulation.step()
        
        
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()
        self.setup_plots()
        
        
        tree_circle = Circle(self.simulation.tree_position, 0.4, color='green', alpha=0.7)
        self.ax1.add_patch(tree_circle)
        self.ax1.text(self.simulation.tree_position[0], self.simulation.tree_position[1] + 0.6, 
                     'Tree', ha='center', va='center', fontweight='bold', color='darkgreen')
        
        
        target_rect = Rectangle((self.simulation.target_spot[0]-0.3, self.simulation.target_spot[1]-0.3), 
                               0.6, 0.6, color='brown', alpha=0.7)
        self.ax1.add_patch(target_rect)
        self.ax1.text(self.simulation.target_spot[0], self.simulation.target_spot[1] + 0.6, 
                     'Target', ha='center', va='center', fontweight='bold', color='brown')
        
        
        if self.simulation.rain_active:
            rain_text = self.ax1.text(5, 9.5, 'RAIN!', ha='center', va='center', 
                                    fontsize=20, fontweight='bold', color='blue',
                                    bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.7))
            
           
            for _ in range(20):
                x = random.uniform(0, 10)
                y = random.uniform(0, 10)
                self.ax1.plot([x, x], [y, y-0.2], 'b-', alpha=0.5)
        
        
        for caterpillar in self.simulation.caterpillars:
            
            if len(caterpillar.movement_history) > 1:
                path_x, path_y = zip(*caterpillar.movement_history)
                self.ax1.plot(path_x, path_y, 'gray', alpha=0.3, linewidth=1)
            
            
            caterpillar_circle = Circle(caterpillar.position, 0.15, color=caterpillar.color, alpha=0.8)
            self.ax1.add_patch(caterpillar_circle)
            
            
            self.ax1.text(caterpillar.position[0], caterpillar.position[1] - 0.3, 
                         str(caterpillar.id), ha='center', va='center', fontsize=8)
        
        
        if len(self.simulation.metrics['at_target']) > 0:
            steps = range(len(self.simulation.metrics['at_target']))
            self.ax2.plot(steps, self.simulation.metrics['at_target'], 'r-', linewidth=2, label='At Target')
            self.ax2.plot(steps, self.simulation.metrics['at_tree'], 'b-', linewidth=2, label='At Tree')
            self.ax2.plot(steps, self.simulation.metrics['moving'], 'g-', linewidth=2, label='Moving')
            self.ax2.legend()
            self.ax2.set_xlim(0, max(200, len(steps)))
            max_caterpillars = len(self.simulation.caterpillars)
            self.ax2.set_ylim(0, max_caterpillars + 1)
        
        
        if len(self.simulation.metrics['total_distance_traveled']) > 0:
            steps = range(len(self.simulation.metrics['total_distance_traveled']))
            self.ax3.plot(steps, self.simulation.metrics['total_distance_traveled'], 'purple', linewidth=2)
            self.ax3.set_xlim(0, max(200, len(steps)))
            max_dist = max(self.simulation.metrics['total_distance_traveled']) * 1.1
            self.ax3.set_ylim(0, max(50, max_dist))
        
        
        info_text = f'Step: {self.simulation.steps}\n'
        info_text += f'Rain: {"Yes" if self.simulation.rain_active else "No"}\n'
        info_text += f'At Target: {self.simulation.metrics["at_target"][-1]}\n'
        info_text += f'At Tree: {self.simulation.metrics["at_tree"][-1]}\n'
        info_text += f'Moving: {self.simulation.metrics["moving"][-1]}\n'
        info_text += f'Total Distance: {self.simulation.metrics["total_distance_traveled"][-1]:.1f}'
        
        self.ax1.text(0.02, 0.98, info_text, transform=self.ax1.transAxes,
                     verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        if all_done:
            self.ax1.text(0.5, 0.5, 'SIMULATION COMPLETE!', transform=self.ax1.transAxes,
                         ha='center', va='center', fontsize=16, fontweight='bold',
                         bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.9))
        
        return []

def run_non_optimized_simulation():
    
    print("Non-Optimized Caterpillar Movement Simulation")
    print("=" * 50)
    print("Features:")
    print("• Random initialization near tree")
    print("• Random rain events that force return to tree")
    print("• Non-optimized random movement patterns")
    print("• Color coding: Green=Moving, Red=At Target, Blue=At Tree")
    print("• Movement paths shown in gray")
    print()
    
    
    sim = NonOptimizedCaterpillarSimulation(
        num_caterpillars=15,
        tree_position=(2, 2),
        target_spot=(8, 8)
    )
    
    
    viz = NonOptimizedVisualization(sim)
    
    
    ani = animation.FuncAnimation(
        viz.fig, viz.update, frames=300, interval=400, blit=False, repeat=False
    )
    
    plt.tight_layout()
    plt.show()
    
    return sim, ani

def analyze_efficiency(simulations: List[NonOptimizedCaterpillarSimulation]):
    """Analyze the efficiency of non-optimized movement"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    for i, sim in enumerate(simulations):
        color = colors[i % len(colors)]
        steps = range(len(sim.metrics['at_target']))
        
        
        total_caterpillars = len(sim.caterpillars)
        completion_rate = [sim.metrics['at_target'][j] / total_caterpillars * 100 for j in steps]
        ax1.plot(steps, completion_rate, color=color, linewidth=2, label=f'Run {i+1}')
        
        
        ax2.plot(steps, sim.metrics['total_distance_traveled'], color=color, linewidth=2, label=f'Run {i+1}')
        
        
        rain_events = []
        current_rain = False
        for step in steps:
            
            if step % 50 == 25:  
                rain_events.append(step)
        
        for rain_step in rain_events:
            ax3.axvline(x=rain_step, color=color, alpha=0.3, linestyle='--')
        
        
        moving_ratio = [sim.metrics['moving'][j] / total_caterpillars for j in steps]
        ax4.plot(steps, moving_ratio, color=color, linewidth=2, label=f'Run {i+1}')
    
    ax1.set_title('Target Completion Rate Over Time')
    ax1.set_xlabel('Simulation Steps')
    ax1.set_ylabel('Completion Rate (%)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.set_title('Total Distance Traveled')
    ax2.set_xlabel('Simulation Steps')
    ax2.set_ylabel('Total Distance')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    ax3.set_title('Simulated Rain Events')
    ax3.set_xlabel('Simulation Steps')
    ax3.set_ylabel('Rain Events')
    ax3.grid(True, alpha=0.3)
    
    ax4.set_title('Movement Activity Ratio')
    ax4.set_xlabel('Simulation Steps')
    ax4.set_ylabel('Moving Caterpillars Ratio')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def calculate_efficiency_metrics(simulation: NonOptimizedCaterpillarSimulation):
    """Calculate various efficiency metrics for the simulation"""
    total_steps = simulation.steps
    total_caterpillars = len(simulation.caterpillars)
    
    
    at_target = simulation.metrics['at_target'][-1]
    at_tree = simulation.metrics['at_tree'][-1]
    
    
    total_distance = simulation.metrics['total_distance_traveled'][-1]
    avg_distance_per_caterpillar = total_distance / total_caterpillars
    
    
    completion_rate = (at_target / total_caterpillars) * 100
    efficiency_ratio = at_target / total_steps if total_steps > 0 else 0
    
    print("\n" + "="*60)
    print("NON-OPTIMIZED SIMULATION EFFICIENCY ANALYSIS")
    print("="*60)
    print(f"Total simulation steps: {total_steps}")
    print(f"Total caterpillars: {total_caterpillars}")
    print(f"Caterpillars reached target: {at_target} ({completion_rate:.1f}%)")
    print(f"Caterpillars returned to tree: {at_tree}")
    print(f"Total distance traveled: {total_distance:.2f} units")
    print(f"Average distance per caterpillar: {avg_distance_per_caterpillar:.2f} units")
    print(f"Efficiency ratio (targets/steps): {efficiency_ratio:.3f}")
    print("="*60)

if __name__ == "__main__":
    
    print("Starting non-optimized caterpillar simulation...")
    simulation, animation_obj = run_non_optimized_simulation()
    
    
    calculate_efficiency_metrics(simulation)
    
    
    print("\nRunning comparative analysis with multiple simulations...")
    simulations = []
    for i in range(3):
        print(f"Running simulation {i+1}...")
        sim = NonOptimizedCaterpillarSimulation(num_caterpillars=10)
        for _ in range(150):  
            sim.step()
            if all(c.at_target or c.at_tree for c in sim.caterpillars):
                break
        simulations.append(sim)
    
    analyze_efficiency(simulations)
    
    
    print("\nDETAILED ANALYSIS OF NON-OPTIMIZED MOVEMENT:")
    print("• Movement includes random directional noise")
    print("• Step sizes vary randomly")
    print("• Rain events interrupt progress and force return to tree")
    print("• No coordination between caterpillars")
    print("• Each caterpillar acts independently")
    print("• Inefficient paths due to random movement patterns")