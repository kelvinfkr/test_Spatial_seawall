"""
Large-Scale Spatial Seawall Optimization Experiments

Supports:
- 100+ coastal sectors
- Multiple optimization algorithms (Proposed, GA, PSO, Uniform baseline)
- Real and synthetic data
- Comprehensive result analysis and comparison
"""

import os
import json
import numpy as np
from datetime import datetime
from typing import Dict, Tuple, List

# Optional imports for visualization (not required for core functionality)
try:
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from data_generator import ExperimentConfig
from seawall_model import (
    SurgeDistribution, CostFunction, DamageFunction,
    SpatialSeawallOptimizer, PairwiseRelativeHeight
)
from baseline_algorithms import AlgorithmComparator


class LargeScaleExperimentRunner:
    """Manages large-scale optimization experiments with multiple algorithms"""

    def __init__(self,
                 num_sectors: int = 100,
                 num_events: int = 1000,
                 data_dir: str = '../data',
                 results_dir: str = '../results'):
        """
        Initialize large-scale experiment runner.

        Args:
            num_sectors: Number of coastal sectors (default 100)
            num_events: Number of flood simulation events
            data_dir: Directory for data files
            results_dir: Directory for results
        """
        self.num_sectors = num_sectors
        self.num_events = num_events
        self.data_dir = data_dir
        self.results_dir = results_dir

        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)

    def generate_large_scale_data(self, regenerate: bool = False) -> Dict:
        """
        Generate or load data for large number of sectors.

        Args:
            regenerate: Force regeneration of data

        Returns:
            Dictionary with all data
        """
        params_file = f'{self.data_dir}/large_scale_params.json'

        if os.path.exists(params_file) and not regenerate:
            print(f"Loading existing {self.num_sectors}-sector data...")
            with open(params_file, 'r') as f:
                params_data = json.load(f)

            floods_file = f'{self.data_dir}/large_scale_floods.npz'
            floods_data = np.load(floods_file)
            events = floods_data['events']

            return {
                'flood_events': events,
                'surge_parameters': params_data['surge_parameters'],
                'cost_parameters': params_data['cost_parameters'],
                'damage_parameters': params_data['damage_parameters'],
                'config': params_data['config']
            }
        else:
            print(f"Generating new {self.num_sectors}-sector data...")
            print(f"  Sectors: {self.num_sectors}")
            print(f"  Flood events: {self.num_events}")
            print(f"  Coastline: 50 miles ({self.num_sectors*0.5} miles per sector)")

            config = ExperimentConfig(
                num_sectors=self.num_sectors,
                num_events=self.num_events
            )
            data = config.generate_full_experiment_data(seed=42)

            # Save with appropriate filenames
            filenames = config.save_experiment_data(
                data, prefix=f'{self.data_dir}/large_scale'
            )

            return data

    def setup_optimizer(self, data: Dict) -> Tuple[SpatialSeawallOptimizer, Dict]:
        """Create optimizer from data"""
        surge_params = data['surge_parameters']
        cost_params = data['cost_parameters']
        damage_params = data['damage_parameters']

        surges = [
            SurgeDistribution(p['mean'], p['std'], p['location_id'])
            for p in surge_params
        ]

        costs = [
            CostFunction(p['a'], p['b'], p['c'])
            for p in cost_params
        ]

        damage = DamageFunction(damage_params['alpha'], damage_params['beta'])

        optimizer = SpatialSeawallOptimizer(
            num_sectors=self.num_sectors,
            surge_distributions=surges,
            cost_functions=costs,
            damage_func=damage,
            max_gradient=0.1
        )

        return optimizer, {
            'surges': surges,
            'costs': costs,
            'damage': damage
        }

    def create_cost_function(self, models: Dict, events: np.ndarray):
        """Create cost evaluation function for algorithms"""
        def evaluate_cost(heights: np.ndarray) -> float:
            # Construction costs
            construction = sum(
                models['costs'][i].cost(heights[i])
                for i in range(self.num_sectors)
            )

            # Expected damage
            total_damage = 0
            for event in events:
                max_surge = np.max(event)
                max_idx = np.argmax(event)
                damage = models['damage'].damage(max_surge, heights[max_idx])
                total_damage += damage

            expected_damage = total_damage / events.shape[0]

            return construction + expected_damage

        return evaluate_cost

    def run_proposed_method(self, data: Dict, optimizer: SpatialSeawallOptimizer,
                           h_range: Tuple[float, float] = (12, 20),
                           step: float = 1.0) -> Dict:
        """Run proposed pairwise optimization method"""
        print("\n" + "="*70)
        print("ALGORITHM 1: PROPOSED PAIRWISE OPTIMIZATION METHOD")
        print("="*70)

        import time
        start_time = time.time()

        events = data['flood_events']
        optimal_h_mid, min_cost, optimal_heights = optimizer.find_optimal_middle_height(
            events, h_min=h_range[0], h_max=h_range[1], step=step
        )

        elapsed = time.time() - start_time

        print(f"\nOptimization completed in {elapsed:.2f} seconds")
        print(f"  Optimal middle height: {optimal_h_mid:.1f} ft")
        print(f"  Height range: {np.min(optimal_heights):.1f} - {np.max(optimal_heights):.1f} ft")
        print(f"  Total cost: ${min_cost:,.0f}")

        return {
            'name': 'Proposed Pairwise Method',
            'heights': optimal_heights,
            'total_cost': min_cost,
            'computation_time': elapsed,
            'middle_height': optimal_h_mid,
            'algorithm_type': 'analytical'
        }

    def run_baseline_algorithms(self, data: Dict, models: Dict) -> Dict:
        """Run baseline algorithms (GA, PSO, Uniform)"""
        print("\n" + "="*70)
        print("ALGORITHM 2-4: BASELINE METAHEURISTIC AND BASELINE ALGORITHMS")
        print("="*70)

        events = data['flood_events']
        cost_func = self.create_cost_function(models, events)

        comparator = AlgorithmComparator(cost_func)

        # GA parameters for large problem
        ga_results = comparator.run_genetic_algorithm(
            self.num_sectors,
            population_size=100,
            generations=200,
            mutation_rate=0.15
        )
        print(f"  GA: {ga_results.total_cost:,.0f} in {ga_results.computation_time:.2f}s")

        # PSO parameters
        pso_results = comparator.run_pso(
            self.num_sectors,
            num_particles=50,
            iterations=200
        )
        print(f"  PSO: {pso_results.total_cost:,.0f} in {pso_results.computation_time:.2f}s")

        # Uniform baseline
        uniform_results = comparator.run_uniform_baseline(self.num_sectors)
        print(f"  Uniform: {uniform_results.total_cost:,.0f}")

        return comparator.results

    def calculate_comparison_metrics(self, proposed_cost: float,
                                    baseline_costs: Dict) -> Dict:
        """Calculate comparison metrics"""
        uniform_cost = baseline_costs['Uniform'].total_cost

        # Compute improvements
        metrics = {
            'proposed_cost': proposed_cost,
            'uniform_cost': uniform_cost,
            'savings_vs_uniform': uniform_cost - proposed_cost,
            'savings_percentage': 100 * (uniform_cost - proposed_cost) / uniform_cost,
            'baseline_comparison': {}
        }

        for name, result in baseline_costs.items():
            if name != 'Uniform':
                metrics['baseline_comparison'][name] = {
                    'cost': result.total_cost,
                    'cost_difference': result.total_cost - proposed_cost,
                    'computation_time': result.computation_time,
                    'speedup': result.computation_time / 0.1  # Reference time
                }

        return metrics

    def analyze_spatial_distribution(self, heights: np.ndarray) -> Dict:
        """Analyze spatial pattern of heights"""
        height_diffs = np.diff(heights)

        return {
            'mean_height': float(np.mean(heights)),
            'std_height': float(np.std(heights)),
            'min_height': float(np.min(heights)),
            'max_height': float(np.max(heights)),
            'height_range': float(np.max(heights) - np.min(heights)),
            'max_gradient': float(np.max(np.abs(height_diffs))),
            'mean_gradient': float(np.mean(np.abs(height_diffs))),
            'spatial_smoothness': float(np.sum(height_diffs**2) / len(height_diffs))
        }

    def run_full_experiment(self, regenerate_data: bool = False) -> Dict:
        """
        Execute complete large-scale experiment.

        Returns:
            Comprehensive results dictionary
        """
        print("\n" + "="*70)
        print("LARGE-SCALE SPATIAL SEAWALL OPTIMIZATION EXPERIMENT")
        print("="*70)
        print(f"Configuration: {self.num_sectors} sectors, {self.num_events} flood events")

        # Data generation
        print("\n[1/4] Generating/loading data...")
        data = self.generate_large_scale_data(regenerate=regenerate_data)

        # Setup
        print("[2/4] Setting up optimizer...")
        optimizer, models = self.setup_optimizer(data)

        # Run proposed method
        print("[3/4] Running optimization algorithms...")
        proposed_result = self.run_proposed_method(data, optimizer)

        # Run baselines
        baseline_results = self.run_baseline_algorithms(data, models)

        # Analysis
        print("[4/4] Analyzing results...")

        # Metrics
        metrics = self.calculate_comparison_metrics(
            proposed_result['total_cost'],
            baseline_results
        )

        # Spatial analysis
        spatial_analysis = {
            'proposed': self.analyze_spatial_distribution(proposed_result['heights']),
            'ga': self.analyze_spatial_distribution(baseline_results['GA'].heights),
            'pso': self.analyze_spatial_distribution(baseline_results['PSO'].heights),
            'uniform': self.analyze_spatial_distribution(baseline_results['Uniform'].heights)
        }

        results = {
            'timestamp': datetime.now().isoformat(),
            'configuration': {
                'num_sectors': self.num_sectors,
                'num_events': self.num_events,
                'coastline_miles': self.num_sectors * 0.5,
                'sector_length_miles': 0.5
            },
            'proposed_method': proposed_result,
            'baseline_results': {
                'GA': {
                    'total_cost': float(baseline_results['GA'].total_cost),
                    'computation_time': float(baseline_results['GA'].computation_time),
                    'mean_height': float(np.mean(baseline_results['GA'].heights))
                },
                'PSO': {
                    'total_cost': float(baseline_results['PSO'].total_cost),
                    'computation_time': float(baseline_results['PSO'].computation_time),
                    'mean_height': float(np.mean(baseline_results['PSO'].heights))
                },
                'Uniform': {
                    'total_cost': float(baseline_results['Uniform'].total_cost),
                    'computation_time': float(baseline_results['Uniform'].computation_time),
                    'uniform_height': float(baseline_results['Uniform'].heights[0])
                }
            },
            'comparison_metrics': metrics,
            'spatial_analysis': spatial_analysis,
            'cost_breakdown': {
                'proposed': {
                    'construction': float(sum(models['costs'][i].cost(proposed_result['heights'][i])
                                             for i in range(self.num_sectors))),
                    'expected_damage': float(proposed_result['total_cost'] -
                                           sum(models['costs'][i].cost(proposed_result['heights'][i])
                                               for i in range(self.num_sectors)))
                }
            }
        }

        return results, baseline_results, proposed_result, data, models

    def save_results(self, results: Dict, filename: str = None) -> str:
        """Save results to JSON"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f'{self.results_dir}/seawall_optimization_{self.num_sectors}sectors_{timestamp}.json'

        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)

        return filename

    def print_summary(self, results: Dict):
        """Print comprehensive summary"""
        print("\n" + "="*70)
        print("RESULTS SUMMARY AND COMPARISON")
        print("="*70)

        print("\nProposed Method (Pairwise Optimization):")
        print(f"  Total Cost: ${results['proposed_method']['total_cost']:,.0f}")
        print(f"  Computation Time: {results['proposed_method']['computation_time']:.3f}s")
        print(f"  Mean Height: {results['spatial_analysis']['proposed']['mean_height']:.2f} ft")
        print(f"  Height Range: {results['spatial_analysis']['proposed']['min_height']:.2f} - "
              f"{results['spatial_analysis']['proposed']['max_height']:.2f} ft")

        print("\nBaseline Algorithms:")
        print(f"  GA Cost: ${results['baseline_results']['GA']['total_cost']:,.0f} "
              f"({results['baseline_results']['GA']['computation_time']:.2f}s)")
        print(f"  PSO Cost: ${results['baseline_results']['PSO']['total_cost']:,.0f} "
              f"({results['baseline_results']['PSO']['computation_time']:.2f}s)")
        print(f"  Uniform Cost: ${results['baseline_results']['Uniform']['total_cost']:,.0f}")

        print("\nSavings vs Uniform Design:")
        print(f"  Absolute: ${results['comparison_metrics']['savings_vs_uniform']:,.0f}")
        print(f"  Percentage: {results['comparison_metrics']['savings_percentage']:.2f}%")

        print("\nCost Breakdown (Proposed):")
        print(f"  Construction: ${results['cost_breakdown']['proposed']['construction']:,.0f}")
        print(f"  Expected Damage: ${results['cost_breakdown']['proposed']['expected_damage']:,.0f}")

        print("\n" + "="*70)


def main():
    """Main entry point"""
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)

    # Run for different scales
    scales = [20, 100]

    for num_sectors in scales:
        print(f"\n\n{'#'*70}")
        print(f"# EXPERIMENT: {num_sectors} SECTORS")
        print(f"{'#'*70}")

        runner = LargeScaleExperimentRunner(
            num_sectors=num_sectors,
            num_events=1000,
            data_dir=os.path.join(project_dir, 'data'),
            results_dir=os.path.join(project_dir, 'results')
        )

        results, baseline_results, proposed_result, data, models = runner.run_full_experiment(
            regenerate_data=(num_sectors==20)  # Regenerate only for first run
        )

        runner.print_summary(results)
        results_file = runner.save_results(results)
        print(f"\nResults saved to: {results_file}")


if __name__ == '__main__':
    main()
