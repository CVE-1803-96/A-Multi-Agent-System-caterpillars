import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional
import math

@dataclass
class Pheromone:
    position: Tuple[float, float]
    strength: float = 1.0
    color: str = "red"

class Caterpillar:
    def __init__(self, agent_id: int, position: Tuple[float, float], is_leader: bool = False, is_super_leader: bool = False):
        self.id = agent_id
        self.position = np.array(position, dtype=float)
        self.is_leader = is_leader
        self.is_super_leader = is_super_leader
        self.energy_consumed = 0.0
        self.heading = 0.0  # Angle in radians
        self.color = 'blue'
        if self.is_super_leader:
            self.color = 'gold'
        elif self.is_leader:
            self.color = 'green'
    
    def distance_to(self, target: np.ndarray) -> float:
        return np.linalg.norm(self.position - target)
    
    def turn_toward(self, target: np.ndarray):
        
        direction = target - self.position
        self.heading = np.arctan2(direction[1], direction[0])
    
    def move_forward(self, distance: float):
        
        dx = distance * np.cos(self.heading)
        dy = distance * np.sin(self.heading)
        self.position += np.array([dx, dy])
        self.energy_consumed += 0.5

class CaterpillarSimulation:
    def __init__(self, num_caterpillars: int = 10, num_super_leaders: int = 1, target_point: Tuple[float, float] = (8, 8)):
        self.caterpillars: List[Caterpillar] = []
        self.pheromones: List[Pheromone] = []
        self.target_point = np.array(target_point)
        self.desired_distance = 2.0
        self.steps = 0
        self.energy_history = []
        self.distance_history = []
        
       
        start_x, start_y = 1, 5
        spacing = 0.5
        
        for i in range(num_caterpillars):
            is_super_leader = (i == 0 and num_super_leaders > 0)
            is_leader = (i < num_super_leaders) and not is_super_leader
            
            caterpillar = Caterpillar(
                i, 
                (start_x + i * spacing, start_y),
                is_leader=is_leader,
                is_super_leader=is_super_leader
            )
            self.caterpillars.append(caterpillar)
    
    def get_previous_turtle(self, current_id: int) -> Optional[Caterpillar]:
       
        if current_id == 0:
            return None
        for caterpillar in self.caterpillars:
            if caterpillar.id == current_id - 1:
                return caterpillar
        return None
    
    def get_super_leader(self) -> Optional[Caterpillar]:
        
        for caterpillar in self.caterpillars:
            if caterpillar.is_super_leader:
                return caterpillar
        return None
    
    def step(self):
        
        total_energy = 0
        avg_distance = 0
        
        for caterpillar in self.caterpillars:
            distance_to_target = caterpillar.distance_to(self.target_point)
            
            if distance_to_target > 0.5:
                
                caterpillar.turn_toward(self.target_point)
                caterpillar.move_forward(1.0)
                
                
                self.pheromones.append(Pheromone(
                    position=tuple(caterpillar.position.copy()),
                    color="red"
                ))
                
            elif caterpillar.is_leader or caterpillar.is_super_leader:
                
                super_leader = self.get_super_leader()
                if super_leader and caterpillar.id != super_leader.id:
                    caterpillar.turn_toward(super_leader.position)
                    current_distance = caterpillar.distance_to(super_leader.position)
                    
                    if current_distance > self.desired_distance:
                        caterpillar.move_forward(1.5)
                    else:
                        caterpillar.move_forward(0.5)
                else:
                    
                    caterpillar.move_forward(0.1)
                    
            else:
                
                previous_turtle = self.get_previous_turtle(caterpillar.id)
                if previous_turtle:
                    caterpillar.turn_toward(previous_turtle.position)
                    current_distance = caterpillar.distance_to(previous_turtle.position)
                    
                    if current_distance > self.desired_distance:
                        caterpillar.move_forward(1.5)
                    else:
                        caterpillar.move_forward(0.5)
            
            total_energy += caterpillar.energy_consumed
            avg_distance += distance_to_target
        
        
        self.energy_history.append(total_energy)
        self.distance_history.append(avg_distance / len(self.caterpillars))
        self.steps += 1
        
        
        self.pheromones = [p for p in self.pheromones if p.strength > 0.1]
        for p in self.pheromones:
            p.strength *= 0.95

class Visualization:
    def __init__(self, simulation: CaterpillarSimulation):
        self.simulation = simulation
        self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(1, 3, figsize=(15, 5))
        self.setup_plots()
    
    def setup_plots(self):
        
        
        self.ax1.set_xlim(0, 10)
        self.ax1.set_ylim(0, 10)
        self.ax1.set_aspect('equal')
        self.ax1.set_title('Caterpillar Movement Simulation')
        self.ax1.set_xlabel('X Position')
        self.ax1.set_ylabel('Y Position')
        self.ax1.grid(True, alpha=0.3)
        
        
        self.ax2.set_xlim(0, 100)
        self.ax2.set_ylim(0, 50)
        self.ax2.set_title('Total Energy Consumption')
        self.ax2.set_xlabel('Simulation Steps')
        self.ax2.set_ylabel('Energy Consumed')
        self.ax2.grid(True, alpha=0.3)
        
        
        self.ax3.set_xlim(0, 100)
        self.ax3.set_ylim(0, 10)
        self.ax3.set_title('Average Distance to Target')
        self.ax3.set_xlabel('Simulation Steps')
        self.ax3.set_ylabel('Distance')
        self.ax3.grid(True, alpha=0.3)
    
    def update(self, frame):
        
        self.simulation.step()
        
        
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()
        self.setup_plots()
        
        
        target_circle = Circle(self.simulation.target_point, 0.3, color='red', alpha=0.7)
        self.ax1.add_patch(target_circle)
        self.ax1.text(self.simulation.target_point[0], self.simulation.target_point[1] + 0.5, 
                     'Target', ha='center', va='center', fontweight='bold')
        
        
        for pheromone in self.simulation.pheromones:
            pheromone_circle = Circle(pheromone.position, 0.1, 
                                    color=pheromone.color, alpha=pheromone.strength * 0.5)
            self.ax1.add_patch(pheromone_circle)
        
        
        for i, caterpillar in enumerate(self.simulation.caterpillars):
            # Draw caterpillar
            caterpillar_circle = Circle(caterpillar.position, 0.2, 
                                      color=caterpillar.color, alpha=0.8)
            self.ax1.add_patch(caterpillar_circle)
            
            
            dx = 0.3 * np.cos(caterpillar.heading)
            dy = 0.3 * np.sin(caterpillar.heading)
            self.ax1.arrow(caterpillar.position[0], caterpillar.position[1], 
                          dx, dy, head_width=0.1, head_length=0.1, 
                          fc=caterpillar.color, ec=caterpillar.color)
            
            
            if i > 0:
                prev_caterpillar = self.simulation.caterpillars[i-1]
                self.ax1.plot([prev_caterpillar.position[0], caterpillar.position[0]],
                            [prev_caterpillar.position[1], caterpillar.position[1]],
                            'gray', alpha=0.5, linestyle='--')
            
            
            label = f'{caterpillar.id}'
            if caterpillar.is_super_leader:
                label += ' (SL)'
            elif caterpillar.is_leader:
                label += ' (L)'
            self.ax1.text(caterpillar.position[0], caterpillar.position[1] - 0.4, 
                         label, ha='center', va='center', fontsize=8)
        
        
        if len(self.simulation.energy_history) > 0:
            self.ax2.plot(range(len(self.simulation.energy_history)), 
                         self.simulation.energy_history, 'b-', linewidth=2)
            self.ax2.set_xlim(0, max(100, len(self.simulation.energy_history)))
            self.ax2.set_ylim(0, max(50, max(self.simulation.energy_history) * 1.1))
        
        
        if len(self.simulation.distance_history) > 0:
            self.ax3.plot(range(len(self.simulation.distance_history)), 
                         self.simulation.distance_history, 'r-', linewidth=2)
            self.ax3.set_xlim(0, max(100, len(self.simulation.distance_history)))
            self.ax3.set_ylim(0, max(10, max(self.simulation.distance_history) * 1.1))
        
        
        info_text = f'Step: {self.simulation.steps}\n'
        info_text += f'Total Energy: {self.simulation.energy_history[-1]:.1f}\n'
        info_text += f'Avg Distance: {self.simulation.distance_history[-1]:.2f}'
        self.ax1.text(0.02, 0.98, info_text, transform=self.ax1.transAxes,
                     verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        return []

def run_simulation():
    
    
    sim = CaterpillarSimulation(
        num_caterpillars=8,
        num_super_leaders=1,
        target_point=(8, 8)
    )
    
    
    viz = Visualization(sim)
    
    
    ani = animation.FuncAnimation(
        viz.fig, viz.update, frames=200, interval=500, blit=False, repeat=True
    )
    
    plt.tight_layout()
    plt.show()
    
    return sim, ani

def analyze_performance(simulations: List[CaterpillarSimulation]):
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    for i, sim in enumerate(simulations):
        color = colors[i % len(colors)]
        
        
        ax1.plot(range(len(sim.energy_history)), sim.energy_history, 
                color=color, linewidth=2, label=f'Run {i+1}')
        
        
        ax2.plot(range(len(sim.distance_history)), sim.distance_history,
                color=color, linewidth=2, label=f'Run {i+1}')
    
    ax1.set_title('Energy Consumption Comparison')
    ax1.set_xlabel('Simulation Steps')
    ax1.set_ylabel('Total Energy Consumed')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.set_title('Distance to Target Comparison')
    ax2.set_xlabel('Simulation Steps')
    ax2.set_ylabel('Average Distance')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("Caterpillar Movement Simulation")
    print("=" * 40)
    print("Legend:")
    print("• Gold circles: Super Leaders")
    print("• Green circles: Leaders") 
    print("• Blue circles: Followers")
    print("• Red circles: Target point")
    print("• Fading red dots: Pheromones")
    print("• Dashed lines: Connections between caterpillars")
    print("• Arrows: Movement direction")
    print()
    
    
    print("Running simulation...")
    simulation, animation_obj = run_simulation()
    
    
    print("Running performance analysis...")
    simulations = []
    for i in range(3):
        sim = CaterpillarSimulation(num_caterpillars=6, num_super_leaders=1)
        for _ in range(100):  
            sim.step()
        simulations.append(sim)
    
    analyze_performance(simulations)
    
    
    print("\nFinal Simulation Statistics:")
    print(f"Total simulation steps: {simulation.steps}")
    print(f"Final total energy consumed: {simulation.energy_history[-1]:.2f}")
    print(f"Final average distance to target: {simulation.distance_history[-1]:.2f}")
    
    leader_count = sum(1 for c in simulation.caterpillars if c.is_leader or c.is_super_leader)
    print(f"Number of leaders: {leader_count}")
    print(f"Number of pheromones deposited: {len(simulation.pheromones)}")