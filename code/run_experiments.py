"""
Main Experiment Runner

Executes the spatial seawall optimization experiments and generates results
for the research paper.
"""

import os
import json
import numpy as np
from datetime import datetime
from typing import Dict, Tuple, List

from data_generator import ExperimentConfig, generate_baseline_experiment
from seawall_model import (
    SurgeDistribution, CostFunction, DamageFunction,
    SpatialSeawallOptimizer
)


class ExperimentRunner:
    """Runs optimization experiments and collects results."""

    def __init__(self, data_dir: str = '../data', results_dir: str = '../results'):
        self.data_dir = data_dir
        self.results_dir = results_dir

        # Create directories if needed
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)

    def load_or_generate_data(self, regenerate: bool = False) -> Dict:
        """Load experiment data or generate if needed."""
        params_file = f'{self.data_dir}/baseline_params.json'

        if os.path.exists(params_file) and not regenerate:
            print("Loading existing experiment data...")
            with open(params_file, 'r') as f:
                params_data = json.load(f)

            floods_file = f'{self.data_dir}/baseline_floods.npz'
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
            print("Generating new experiment data...")
            filenames, data = generate_baseline_experiment(output_dir=self.data_dir)
            return data

    def setup_optimizer(self, data: Dict) -> SpatialSeawallOptimizer:
        """Create optimizer from experiment data."""
        surge_params = data['surge_parameters']
        cost_params = data['cost_parameters']
        damage_params = data['damage_parameters']

        # Create surge distributions
        surges = [
            SurgeDistribution(p['mean'], p['std'], p['location_id'])
            for p in surge_params
        ]

        # Create cost functions
        costs = [
            CostFunction(p['a'], p['b'], p['c'])
            for p in cost_params
        ]

        # Create damage function
        damage = DamageFunction(damage_params['alpha'], damage_params['beta'])

        # Create optimizer
        optimizer = SpatialSeawallOptimizer(
            num_sectors=data['config']['num_sectors'],
            surge_distributions=surges,
            cost_functions=costs,
            damage_func=damage,
            max_gradient=0.1  # 10ft per 0.1 mile as per paper
        )

        return optimizer

    def run_optimization(self, data: Dict, h_range: Tuple[float, float] = (15, 22),
                       step: float = 1.0) -> Dict:
        """
        Run the spatial seawall optimization.

        Args:
            data: Experiment data
            h_range: Range of middle heights to test
            step: Step size for height search

        Returns:
            Optimization results
        """
        print("\n" + "="*60)
        print("SPATIAL SEAWALL OPTIMIZATION EXPERIMENT")
        print("="*60)

        # Setup optimizer
        optimizer = self.setup_optimizer(data)
        events = data['flood_events']

        print(f"\nConfiguration:")
        print(f"  Coastal sectors: {data['config']['num_sectors']}")
        print(f"  Coastline length: {data['config']['total_coastline_miles']} miles")
        print(f"  Flood scenarios: {events.shape[0]}")
        print(f"  Search range: {h_range[0]:.0f} - {h_range[1]:.0f} ft")
        print(f"  Step size: {step} ft")

        # Run optimization
        print("\nRunning optimization search...")
        optimal_h_mid, min_cost, optimal_heights = optimizer.find_optimal_middle_height(
            events, h_min=h_range[0], h_max=h_range[1], step=step
        )

        results = {
            'timestamp': datetime.now().isoformat(),
            'configuration': {
                'num_sectors': data['config']['num_sectors'],
                'num_events': events.shape[0],
                'coastline_miles': data['config']['total_coastline_miles'],
                'search_range': list(h_range),
                'search_step': step
            },
            'optimal_solution': {
                'middle_height_ft': float(optimal_h_mid),
                'total_cost': float(min_cost),
                'sector_heights_ft': optimal_heights.tolist()
            }
        }

        return results

    def analyze_results(self, data: Dict, results: Dict) -> Dict:
        """
        Perform analysis on optimization results.

        Returns:
            Analysis summary
        """
        optimizer = self.setup_optimizer(data)
        optimal_heights = np.array(results['optimal_solution']['sector_heights_ft'])
        events = data['flood_events']

        # Calculate construction costs
        cost_params = data['cost_parameters']
        construction_costs = [
            cost_params[i]['a'] +
            cost_params[i]['b'] * optimal_heights[i] +
            cost_params[i]['c'] * optimal_heights[i]**2
            for i in range(len(cost_params))
        ]
        total_construction_cost = sum(construction_costs)

        # Calculate expected damage
        damage_params = data['damage_parameters']
        damage_func = DamageFunction(damage_params['alpha'], damage_params['beta'])

        total_damage = 0
        for event in events:
            max_surge = np.max(event)
            max_idx = np.argmax(event)
            damage = damage_func.damage(max_surge, optimal_heights[max_idx])
            total_damage += damage

        expected_damage = total_damage / events.shape[0]

        # Statistics
        height_stats = {
            'mean': float(np.mean(optimal_heights)),
            'std': float(np.std(optimal_heights)),
            'min': float(np.min(optimal_heights)),
            'max': float(np.max(optimal_heights)),
            'range': float(np.max(optimal_heights) - np.min(optimal_heights))
        }

        # Height differences (gradient analysis)
        height_diffs = np.diff(optimal_heights)
        gradient_stats = {
            'max_diff_ft': float(np.max(np.abs(height_diffs))),
            'mean_abs_diff': float(np.mean(np.abs(height_diffs))),
            'compliance_with_constraint': all(np.abs(height_diffs) <= 0.1)  # 10ft per 0.1 mile
        }

        analysis = {
            'cost_breakdown': {
                'construction_cost': float(total_construction_cost),
                'expected_annual_damage': float(expected_damage),
                'total_cost': float(total_construction_cost + expected_damage)
            },
            'spatial_distribution': {
                'height_statistics': height_stats,
                'gradient_analysis': gradient_stats
            },
            'coverage_analysis': {
                'mean_protection_height': float(np.mean(optimal_heights)),
                'most_exposed_sector': int(np.argmax(optimal_heights)),
                'most_protected_sector': int(np.argmin(optimal_heights))
            }
        }

        return analysis

    def save_results(self, results: Dict, analysis: Dict, filename: str = None) -> str:
        """Save results to JSON file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f'{self.results_dir}/seawall_optimization_{timestamp}.json'

        output = {
            'results': results,
            'analysis': analysis
        }

        with open(filename, 'w') as f:
            json.dump(output, f, indent=2)

        return filename

    def print_summary(self, results: Dict, analysis: Dict):
        """Print summary of results."""
        opt = results['optimal_solution']
        cost = analysis['cost_breakdown']
        spatial = analysis['spatial_distribution']

        print("\n" + "="*60)
        print("OPTIMIZATION RESULTS SUMMARY")
        print("="*60)

        print(f"\nOptimal Solution:")
        print(f"  Middle sector height: {opt['middle_height_ft']:.1f} ft")
        print(f"  Height range: {spatial['height_statistics']['min']:.1f} - "
              f"{spatial['height_statistics']['max']:.1f} ft")
        print(f"  Mean height: {spatial['height_statistics']['mean']:.1f} ft")
        print(f"  Max gradient: {spatial['gradient_analysis']['max_diff_ft']:.2f} ft/sector")

        print(f"\nCost Analysis:")
        print(f"  Construction cost: ${cost['construction_cost']:,.0f}")
        print(f"  Expected annual damage: ${cost['expected_annual_damage']:,.0f}")
        print(f"  Total cost: ${cost['total_cost']:,.0f}")

        print(f"\nCoverage Analysis:")
        coverage = analysis['coverage_analysis']
        print(f"  Most protected sector: {coverage['most_protected_sector']}")
        print(f"  Most exposed sector: {coverage['most_exposed_sector']}")

        print("\n" + "="*60)

    def run_full_experiment(self, regenerate_data: bool = False) -> Dict:
        """
        Run the complete experiment pipeline.

        Returns:
            Complete results dict
        """
        # Load or generate data
        data = self.load_or_generate_data(regenerate=regenerate_data)

        # Run optimization
        results = self.run_optimization(data)

        # Analyze results
        analysis = self.analyze_results(data, results)

        # Print summary
        self.print_summary(results, analysis)

        # Save results
        results_file = self.save_results(results, analysis)
        print(f"\nResults saved to: {results_file}")

        return {
            'data': data,
            'results': results,
            'analysis': analysis,
            'results_file': results_file
        }


def main():
    """Main entry point for experiment runner."""
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)

    runner = ExperimentRunner(
        data_dir=os.path.join(project_dir, 'data'),
        results_dir=os.path.join(project_dir, 'results')
    )

    # Run full experiment
    experiment_results = runner.run_full_experiment(regenerate_data=False)

    return experiment_results


if __name__ == '__main__':
    main()
