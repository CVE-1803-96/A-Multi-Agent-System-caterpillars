import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle, Rectangle
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional
import math
import pandas as pd
from time import time

@dataclass
class OptimizedCaterpillar:
    
    def __init__(self, agent_id: int, position: Tuple[float, float], is_leader: bool = False, is_super_leader: bool = False):
        self.id = agent_id
        self.position = np.array(position, dtype=float)
        self.is_leader = is_leader
        self.is_super_leader = is_super_leader
        self.energy_consumed = 0.0
        self.heading = 0.0
        self.color = 'blue'
        if self.is_super_leader:
            self.color = 'gold'
        elif self.is_leader:
            self.color = 'green'
        self.reached_target = False
        self.movement_history = [tuple(position)]
    
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
        self.movement_history.append(tuple(self.position.copy()))

class OptimizedSimulation:
    
    def __init__(self, num_caterpillars: int = 10, num_super_leaders: int = 1, target_point: Tuple[float, float] = (8, 8)):
        self.caterpillars: List[OptimizedCaterpillar] = []
        self.target_point = np.array(target_point)
        self.desired_distance = 2.0
        self.steps = 0
        self.energy_history = []
        self.distance_history = []
        self.completion_time = None
        self.total_energy_at_completion = 0
        
        
        start_x, start_y = 1, 5
        spacing = 0.5
        
        for i in range(num_caterpillars):
            is_super_leader = (i == 0 and num_super_leaders > 0)
            is_leader = (i < num_super_leaders) and not is_super_leader
            
            caterpillar = OptimizedCaterpillar(
                i, 
                (start_x + i * spacing, start_y),
                is_leader=is_leader,
                is_super_leader=is_super_leader
            )
            self.caterpillars.append(caterpillar)
    
    def get_previous_turtle(self, current_id: int) -> Optional[OptimizedCaterpillar]:
        if current_id == 0:
            return None
        for caterpillar in self.caterpillars:
            if caterpillar.id == current_id - 1:
                return caterpillar
        return None
    
    def get_super_leader(self) -> Optional[OptimizedCaterpillar]:
        for caterpillar in self.caterpillars:
            if caterpillar.is_super_leader:
                return caterpillar
        return None
    
    def step(self):
        total_energy = 0
        avg_distance = 0
        caterpillars_at_target = 0
        
        for caterpillar in self.caterpillars:
            if caterpillar.reached_target:
                caterpillars_at_target += 1
                continue
                
            distance_to_target = caterpillar.distance_to(self.target_point)
            
            if distance_to_target > 0.5:
                caterpillar.turn_toward(self.target_point)
                caterpillar.move_forward(1.0)
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
            
            
            if caterpillar.distance_to(self.target_point) <= 0.5:
                caterpillar.reached_target = True
                caterpillars_at_target += 1
            
            total_energy += caterpillar.energy_consumed
            avg_distance += distance_to_target
        
        
        self.energy_history.append(total_energy)
        self.distance_history.append(avg_distance / len(self.caterpillars))
        self.steps += 1
        
        
        if caterpillars_at_target == len(self.caterpillars) and self.completion_time is None:
            self.completion_time = self.steps
            self.total_energy_at_completion = total_energy
        
        return caterpillars_at_target == len(self.caterpillars)

@dataclass
class NonOptimizedCaterpillar:
    
    def __init__(self, agent_id: int, position: Tuple[float, float]):
        self.id = agent_id
        self.position = np.array(position, dtype=float)
        self.at_target = False
        self.at_tree = False
        self.color = 'green'
        self.energy_consumed = 0.0
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
            self.energy_consumed += 0.5
            self.movement_history.append(tuple(self.position.copy()))

class NonOptimizedSimulation:
    
    def __init__(self, 
                 num_caterpillars: int = 15,
                 tree_position: Tuple[float, float] = (2, 2),
                 target_spot: Tuple[float, float] = (8, 8),
                 rain_enabled: bool = True):
        
        self.caterpillars: List[NonOptimizedCaterpillar] = []
        self.tree_position = np.array(tree_position)
        self.target_spot = np.array(target_spot)
        self.rain_active = False
        self.rain_enabled = rain_enabled
        self.steps = 0
        self.rain_start_step = 0
        self.rain_duration = 0
        self.completion_time = None
        self.total_energy_at_completion = 0
        
        self.metrics = {
            'at_target': [],
            'at_tree': [],
            'moving': [],
            'total_energy': [],
            'avg_distance': []
        }
        
        self.initialize_caterpillars(num_caterpillars)
    
    def initialize_caterpillars(self, num_caterpillars: int):
        for i in range(num_caterpillars):
            angle = random.uniform(0, 2 * math.pi)
            distance = random.uniform(0.5, 2.0)
            x = self.tree_position[0] + distance * math.cos(angle)
            y = self.tree_position[1] + distance * math.sin(angle)
            
            caterpillar = NonOptimizedCaterpillar(i, (x, y))
            self.caterpillars.append(caterpillar)
    
    def toggle_rain(self, duration: int = 20):
        if self.rain_enabled and not self.rain_active and random.random() < 0.05:
            self.rain_active = True
            self.rain_start_step = self.steps
            self.rain_duration = duration
        elif self.rain_active and (self.steps - self.rain_start_step) >= self.rain_duration:
            self.rain_active = False
    
    def step(self):
        self.steps += 1
        self.toggle_rain()
        
        caterpillars_moving = 0
        total_energy = 0
        avg_distance = 0
        caterpillars_at_target = 0
        
        for caterpillar in self.caterpillars:
            if caterpillar.at_target or caterpillar.at_tree:
                if caterpillar.at_target:
                    caterpillars_at_target += 1
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
                    caterpillars_at_target += 1
            
            total_energy += caterpillar.energy_consumed
            avg_distance += caterpillar.distance_to(self.target_spot)
        
        
        at_target = sum(1 for c in self.caterpillars if c.at_target)
        at_tree = sum(1 for c in self.caterpillars if c.at_tree)
        
        self.metrics['at_target'].append(at_target)
        self.metrics['at_tree'].append(at_tree)
        self.metrics['moving'].append(caterpillars_moving)
        self.metrics['total_energy'].append(total_energy)
        self.metrics['avg_distance'].append(avg_distance / len(self.caterpillars))
        
        
        if caterpillars_at_target == len(self.caterpillars) and self.completion_time is None:
            self.completion_time = self.steps
            self.total_energy_at_completion = total_energy
        
        return caterpillars_at_target == len(self.caterpillars)

class ComparativeAnalysis:
    
    def __init__(self, num_trials: int = 10, max_steps: int = 300):
        self.num_trials = num_trials
        self.max_steps = max_steps
        self.results = []
    
    def run_comparison(self):
        
        print("Running comparative analysis...")
        print("=" * 60)
        
        for trial in range(self.num_trials):
            print(f"Trial {trial + 1}/{self.num_trials}")
            
            
            optimized_sim = OptimizedSimulation(num_caterpillars=10, num_super_leaders=1)
            optimized_time = None
            optimized_energy = None
            
            for step in range(self.max_steps):
                done = optimized_sim.step()
                if done:
                    optimized_time = optimized_sim.completion_time
                    optimized_energy = optimized_sim.total_energy_at_completion
                    break
            
            if optimized_time is None:
                optimized_time = self.max_steps
                optimized_energy = optimized_sim.energy_history[-1]
            
            
            non_optimized_no_rain = NonOptimizedSimulation(
                num_caterpillars=10, 
                rain_enabled=False
            )
            non_opt_no_rain_time = None
            non_opt_no_rain_energy = None
            
            for step in range(self.max_steps):
                done = non_optimized_no_rain.step()
                if done:
                    non_opt_no_rain_time = non_optimized_no_rain.completion_time
                    non_opt_no_rain_energy = non_optimized_no_rain.total_energy_at_completion
                    break
            
            if non_opt_no_rain_time is None:
                non_opt_no_rain_time = self.max_steps
                non_opt_no_rain_energy = non_optimized_no_rain.metrics['total_energy'][-1]
            
            
            non_optimized_rain = NonOptimizedSimulation(
                num_caterpillars=10, 
                rain_enabled=True
            )
            non_opt_rain_time = None
            non_opt_rain_energy = None
            
            for step in range(self.max_steps):
                done = non_optimized_rain.step()
                if done:
                    non_opt_rain_time = non_optimized_rain.completion_time
                    non_opt_rain_energy = non_optimized_rain.total_energy_at_completion
                    break
            
            if non_opt_rain_time is None:
                non_opt_rain_time = self.max_steps
                non_opt_rain_energy = non_optimized_rain.metrics['total_energy'][-1]
            
            
            self.results.append({
                'trial': trial + 1,
                'optimized_time': optimized_time,
                'optimized_energy': optimized_energy,
                'non_optimized_no_rain_time': non_opt_no_rain_time,
                'non_optimized_no_rain_energy': non_opt_no_rain_energy,
                'non_optimized_rain_time': non_opt_rain_time,
                'non_optimized_rain_energy': non_opt_rain_energy
            })
        
        return self.analyze_results()
    
    def analyze_results(self):
        
        df = pd.DataFrame(self.results)
        
        
        stats = {
            'Method': ['Optimized', 'Non-Optimized (No Rain)', 'Non-Optimized (With Rain)'],
            'Avg_Time': [
                df['optimized_time'].mean(),
                df['non_optimized_no_rain_time'].mean(),
                df['non_optimized_rain_time'].mean()
            ],
            'Std_Time': [
                df['optimized_time'].std(),
                df['non_optimized_no_rain_time'].std(),
                df['non_optimized_rain_time'].std()
            ],
            'Avg_Energy': [
                df['optimized_energy'].mean(),
                df['non_optimized_no_rain_energy'].mean(),
                df['non_optimized_rain_energy'].mean()
            ],
            'Std_Energy': [
                df['optimized_energy'].std(),
                df['non_optimized_no_rain_energy'].std(),
                df['non_optimized_rain_energy'].std()
            ],
            'Efficiency_Ratio': [
                (df['optimized_time'].mean() / df['optimized_energy'].mean()) if df['optimized_energy'].mean() > 0 else 0,
                (df['non_optimized_no_rain_time'].mean() / df['non_optimized_no_rain_energy'].mean()) if df['non_optimized_no_rain_energy'].mean() > 0 else 0,
                (df['non_optimized_rain_time'].mean() / df['non_optimized_rain_energy'].mean()) if df['non_optimized_rain_energy'].mean() > 0 else 0
            ]
        }
        
        stats_df = pd.DataFrame(stats)
        
        
        self.create_comparison_plots(df, stats_df)
        
        return df, stats_df
    
    def create_comparison_plots(self, df, stats_df):
        """Create comprehensive comparison plots"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        
        methods = ['Optimized', 'Non-Optimized\n(No Rain)', 'Non-Optimized\n(With Rain)']
        times = [df['optimized_time'].mean(), df['non_optimized_no_rain_time'].mean(), df['non_optimized_rain_time'].mean()]
        time_errors = [df['optimized_time'].std(), df['non_optimized_no_rain_time'].std(), df['non_optimized_rain_time'].std()]
        
        bars1 = ax1.bar(methods, times, yerr=time_errors, capsize=5, color=['green', 'orange', 'red'], alpha=0.7)
        ax1.set_title('Time to Reach Target (Steps)', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Steps')
        ax1.grid(True, alpha=0.3)
        
        
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 5,
                    f'{height:.1f}', ha='center', va='bottom', fontweight='bold')
        
        
        energies = [df['optimized_energy'].mean(), df['non_optimized_no_rain_energy'].mean(), df['non_optimized_rain_energy'].mean()]
        energy_errors = [df['optimized_energy'].std(), df['non_optimized_no_rain_energy'].std(), df['non_optimized_rain_energy'].std()]
        
        bars2 = ax2.bar(methods, energies, yerr=energy_errors, capsize=5, color=['green', 'orange', 'red'], alpha=0.7)
        ax2.set_title('Total Energy Consumed', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Energy Units')
        ax2.grid(True, alpha=0.3)
        
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 2,
                    f'{height:.1f}', ha='center', va='bottom', fontweight='bold')
        
        
        colors = ['green', 'orange', 'red']
        for i, method in enumerate(methods):
            ax3.scatter(stats_df['Avg_Time'][i], stats_df['Avg_Energy'][i], 
                       s=200, color=colors[i], label=method, alpha=0.7)
            ax3.errorbar(stats_df['Avg_Time'][i], stats_df['Avg_Energy'][i],
                        xerr=stats_df['Std_Time'][i], yerr=stats_df['Std_Energy'][i],
                        color=colors[i], alpha=0.5)
        
        ax3.set_xlabel('Time (Steps)')
        ax3.set_ylabel('Energy Consumed')
        ax3.set_title('Time vs Energy Efficiency', fontsize=14, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        
        efficiency = stats_df['Efficiency_Ratio']
        bars4 = ax4.bar(methods, efficiency, color=['green', 'orange', 'red'], alpha=0.7)
        ax4.set_title('Efficiency Ratio (Time/Energy)', fontsize=14, fontweight='bold')
        ax4.set_ylabel('Efficiency Ratio\n(Lower is Better)')
        ax4.grid(True, alpha=0.3)
        
        for bar in bars4:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.show()
        
        
        self.print_detailed_stats(stats_df)
    
    def print_detailed_stats(self, stats_df):
        
        print("\n" + "="*80)
        print("COMPREHENSIVE COMPARISON RESULTS")
        print("="*80)
        
        for _, row in stats_df.iterrows():
            print(f"\n{row['Method']}:")
            print(f"  Average Time: {row['Avg_Time']:.2f} ± {row['Std_Time']:.2f} steps")
            print(f"  Average Energy: {row['Avg_Energy']:.2f} ± {row['Std_Energy']:.2f} units")
            print(f"  Efficiency Ratio: {row['Efficiency_Ratio']:.3f} (time/energy)")
        
        
        opt_time = stats_df[stats_df['Method'] == 'Optimized']['Avg_Time'].values[0]
        no_rain_time = stats_df[stats_df['Method'] == 'Non-Optimized (No Rain)']['Avg_Time'].values[0]
        rain_time = stats_df[stats_df['Method'] == 'Non-Optimized (With Rain)']['Avg_Time'].values[0]
        
        opt_energy = stats_df[stats_df['Method'] == 'Optimized']['Avg_Energy'].values[0]
        no_rain_energy = stats_df[stats_df['Method'] == 'Non-Optimized (No Rain)']['Avg_Energy'].values[0]
        rain_energy = stats_df[stats_df['Method'] == 'Non-Optimized (With Rain)']['Avg_Energy'].values[0]
        
        print("\n" + "="*80)
        print("PERFORMANCE IMPROVEMENT ANALYSIS")
        print("="*80)
        print(f"\nOptimized vs Non-Optimized (No Rain):")
        print(f"  Time Improvement: {((no_rain_time - opt_time) / no_rain_time * 100):.1f}% faster")
        print(f"  Energy Efficiency: {((no_rain_energy - opt_energy) / no_rain_energy * 100):.1f}% less energy")
        
        print(f"\nOptimized vs Non-Optimized (With Rain):")
        print(f"  Time Improvement: {((rain_time - opt_time) / rain_time * 100):.1f}% faster")
        print(f"  Energy Efficiency: {((rain_energy - opt_energy) / rain_energy * 100):.1f}% less energy")
        
        print(f"\nKey Insights:")
        print("• Organized movement (optimized) significantly reduces both time and energy consumption")
        print("• Rain events in non-optimized movement dramatically decrease efficiency")
        print("• Leadership hierarchy in optimized movement enables coordinated, efficient paths")
        print("• Random movement patterns in non-optimized approach lead to wasted energy")

def run_side_by_side_demonstration():
    
    print("Running side-by-side demonstration...")
    
    
    optimized_sim = OptimizedSimulation(num_caterpillars=8, num_super_leaders=1)
    non_optimized_sim = NonOptimizedSimulation(num_caterpillars=8, rain_enabled=True)
    
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    def update(frame):
        ax1.clear()
        ax2.clear()
        
        
        opt_done = optimized_sim.step()
        non_opt_done = non_optimized_sim.step()
        
        
        ax1.set_xlim(0, 10)
        ax1.set_ylim(0, 10)
        ax1.set_aspect('equal')
        ax1.set_title('Optimized Movement (Algorithm 1)', fontweight='bold')
        ax1.set_xlabel('X Position')
        ax1.set_ylabel('Y Position')
        ax1.grid(True, alpha=0.3)
        
        
        target_circle = Circle(optimized_sim.target_point, 0.3, color='red', alpha=0.7)
        ax1.add_patch(target_circle)
        
       
        for i, caterpillar in enumerate(optimized_sim.caterpillars):
            caterpillar_circle = Circle(caterpillar.position, 0.2, color=caterpillar.color, alpha=0.8)
            ax1.add_patch(caterpillar_circle)
            
            if i > 0 and not caterpillar.reached_target:
                prev_caterpillar = optimized_sim.caterpillars[i-1]
                ax1.plot([prev_caterpillar.position[0], caterpillar.position[0]],
                        [prev_caterpillar.position[1], caterpillar.position[1]],
                        'gray', alpha=0.5, linestyle='--')
        
        
        ax2.set_xlim(0, 10)
        ax2.set_ylim(0, 10)
        ax2.set_aspect('equal')
        ax2.set_title('Non-Optimized Movement (Algorithm 2)', fontweight='bold')
        ax2.set_xlabel('X Position')
        ax2.set_ylabel('Y Position')
        ax2.grid(True, alpha=0.3)
        
        
        tree_circle = Circle(non_optimized_sim.tree_position, 0.4, color='green', alpha=0.7)
        ax2.add_patch(tree_circle)
        target_rect = Rectangle((non_optimized_sim.target_spot[0]-0.3, non_optimized_sim.target_spot[1]-0.3), 
                               0.6, 0.6, color='brown', alpha=0.7)
        ax2.add_patch(target_rect)
        
        
        if non_optimized_sim.rain_active:
            ax2.text(5, 9.5, 'RAIN!', ha='center', va='center', 
                    fontsize=16, fontweight='bold', color='blue')
        
        
        for caterpillar in non_optimized_sim.caterpillars:
            
            if len(caterpillar.movement_history) > 1:
                path_x, path_y = zip(*caterpillar.movement_history)
                ax2.plot(path_x, path_y, 'gray', alpha=0.3, linewidth=1)
            
            
            caterpillar_circle = Circle(caterpillar.position, 0.15, color=caterpillar.color, alpha=0.8)
            ax2.add_patch(caterpillar_circle)
        
        
        opt_info = f'Optimized\nSteps: {optimized_sim.steps}\nEnergy: {optimized_sim.energy_history[-1]:.1f}\nAt Target: {sum(1 for c in optimized_sim.caterpillars if c.reached_target)}'
        non_opt_info = f'Non-Optimized\nSteps: {non_optimized_sim.steps}\nEnergy: {non_optimized_sim.metrics["total_energy"][-1]:.1f}\nAt Target: {non_optimized_sim.metrics["at_target"][-1]}\nAt Tree: {non_optimized_sim.metrics["at_tree"][-1]}'
        
        ax1.text(0.02, 0.98, opt_info, transform=ax1.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        ax2.text(0.02, 0.98, non_opt_info, transform=ax2.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
        
        if opt_done and non_opt_done:
            ax1.text(0.5, 0.5, 'COMPLETE!', transform=ax1.transAxes, ha='center', va='center',
                    fontsize=20, fontweight='bold', bbox=dict(boxstyle='round', facecolor='yellow'))
            ax2.text(0.5, 0.5, 'COMPLETE!', transform=ax2.transAxes, ha='center', va='center',
                    fontsize=20, fontweight='bold', bbox=dict(boxstyle='round', facecolor='yellow'))
        
        return []
    
    ani = animation.FuncAnimation(fig, update, frames=200, interval=400, blit=False, repeat=False)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("CATERPILLAR MOVEMENT ALGORITHMS COMPARISON")
    print("=" * 60)
    print("Comparing Optimized (Algorithm 1) vs Non-Optimized (Algorithm 2)")
    print("Metrics: Time to reach target and Energy consumption")
    print()
    
    
    run_side_by_side_demonstration()
    
    
    print("\n" + "="*60)
    print("RUNNING COMPREHENSIVE STATISTICAL ANALYSIS")
    print("="*60)
    
    comparator = ComparativeAnalysis(num_trials=15, max_steps=250)
    results_df, stats_df = comparator.run_comparison()
    
    
    results_df.to_csv('caterpillar_comparison_results.csv', index=False)
    stats_df.to_csv('caterpillar_comparison_stats.csv', index=False)
    
    print("\nResults saved to 'caterpillar_comparison_results.csv' and 'caterpillar_comparison_stats.csv'")