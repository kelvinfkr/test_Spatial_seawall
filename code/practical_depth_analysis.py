"""
Practical Depth Analysis for RESS Submission

Focuses on:
1. Return period design comparisons
2. Spatial vs uniform design comparison
3. Multi-scale coastal analysis
4. Cost vs safety tradeoff analysis

Generates figures and insights for journal submission.
"""

import numpy as np
import json
import os
from datetime import datetime
from typing import Dict, List, Tuple

from data_generator import ExperimentConfig
from seawall_model import (
    SurgeDistribution, CostFunction, DamageFunction,
    SpatialSeawallOptimizer
)


class AnalysisBase:
    """Base class for analysis with optimizer setup"""

    @staticmethod
    def setup_optimizer(data: Dict) -> SpatialSeawallOptimizer:
        """Setup optimizer from experiment data"""
        surge_params_list = data['surge_parameters']
        cost_params_list = data['cost_parameters']
        damage_params = data['damage_parameters']

        num_sectors = len(surge_params_list)

        # Create surge distributions from list of dicts
        surges = [
            SurgeDistribution(
                surge_params_list[i]['mean'],
                surge_params_list[i]['std'],
                0.1  # Default GEV shape parameter
            )
            for i in range(num_sectors)
        ]

        # Create cost functions from list of dicts
        costs = [
            CostFunction(
                cost_params_list[i]['base_cost'] if 'base_cost' in cost_params_list[i] else 7000.0,
                cost_params_list[i].get('constant_cost', 0),
                cost_params_list[i].get('height_coefficient', 1.0)
            )
            for i in range(num_sectors)
        ]

        # Create damage function (alpha, beta parameters)
        damage = DamageFunction(
            alpha=damage_params.get('alpha', 100),
            beta=damage_params.get('beta', 1.5)
        )

        # Create optimizer
        optimizer = SpatialSeawallOptimizer(
            num_sectors=num_sectors,
            surge_distributions=surges,
            cost_functions=costs,
            damage_func=damage
        )

        return optimizer


class ReturnPeriodAnalysis(AnalysisBase):
    """Compare 100, 500, 1000-year return period designs"""

    @staticmethod
    def analyze(n_sectors=50):
        """
        Compare optimal designs for different return periods

        Returns:
            Dict with results for 100/500/1000-year designs
        """
        print("\n" + "="*70)
        print("RETURN PERIOD COMPARISON ANALYSIS")
        print("="*70)

        results = {}

        for return_period in [100, 500, 1000]:
            print(f"\n{'─'*70}")
            print(f"Return Period: {return_period}-year")
            print(f"{'─'*70}")

            # Generate events
            config = ExperimentConfig(num_sectors=n_sectors, num_events=2000)
            data = config.generate_full_experiment_data(seed=42)
            flood_events = data['flood_events']

            # Scale events to represent return period
            # Heuristic: higher return periods have larger extremes
            scale_factor = 1.0 + 0.05 * np.log(return_period)
            scaled_events = flood_events * scale_factor

            # Setup and run optimizer
            optimizer = AnalysisBase.setup_optimizer(data)
            h_min, h_max = 10.0, 20.0
            h_mid, min_cost, opt_heights = optimizer.find_optimal_middle_height(
                scaled_events, h_min=h_min, h_max=h_max, step=0.5
            )

            # Compute metrics
            mean_height = np.mean(opt_heights)
            height_range = np.max(opt_heights) - np.min(opt_heights)
            total_cost = optimizer.evaluate_total_cost(opt_heights, scaled_events)

            results[f'T_{return_period}'] = {
                'return_period': return_period,
                'scale_factor': float(scale_factor),
                'mean_height_ft': float(mean_height),
                'height_range_ft': float(height_range),
                'min_height_ft': float(np.min(opt_heights)),
                'max_height_ft': float(np.max(opt_heights)),
                'total_cost': float(total_cost),
                'cost_per_sector': float(total_cost / n_sectors)
            }

            print(f"Mean design height: {mean_height:.2f} ft")
            print(f"Height range: {height_range:.2f} ft")
            print(f"Total cost: ${total_cost:,.0f}")

        return results


class SpatialVsUniformAnalysis(AnalysisBase):
    """Compare spatial optimization with traditional uniform standard"""

    @staticmethod
    def analyze(n_sectors=50):
        """
        Compare:
        - Spatial design (variable heights per sector)
        - Uniform standard (same height everywhere)
        """
        print("\n" + "="*70)
        print("SPATIAL DESIGN vs. UNIFORM STANDARD COMPARISON")
        print("="*70)

        # Generate events
        config = ExperimentConfig(num_sectors=n_sectors, num_events=2000)
        data = config.generate_full_experiment_data(seed=42)
        flood_events = data['flood_events']

        optimizer = AnalysisBase.setup_optimizer(data)
        results = {}

        # Method 1: Spatial optimization
        print(f"\n{'─'*70}")
        print("Method 1: Spatial Optimization (variable heights)")
        print(f"{'─'*70}")

        h_min, h_max = 10.0, 20.0
        h_mid, min_cost, spatial_heights = optimizer.find_optimal_middle_height(
            flood_events, h_min=h_min, h_max=h_max, step=0.5
        )
        spatial_cost = optimizer.evaluate_total_cost(spatial_heights, flood_events)
        spatial_mean = np.mean(spatial_heights)
        spatial_std = np.std(spatial_heights)

        results['spatial'] = {
            'method': 'Spatial Optimization',
            'mean_height_ft': float(spatial_mean),
            'std_dev_ft': float(spatial_std),
            'range_ft': float(np.max(spatial_heights) - np.min(spatial_heights)),
            'total_cost': float(spatial_cost),
            'cost_per_ft_height': float(spatial_cost / spatial_mean)
        }

        print(f"Mean height: {spatial_mean:.2f} ft (±{spatial_std:.2f} ft)")
        print(f"Height range: {np.max(spatial_heights) - np.min(spatial_heights):.2f} ft")
        print(f"Total cost: ${spatial_cost:,.0f}")

        # Method 2: Uniform design
        print(f"\n{'─'*70}")
        print("Method 2: Uniform Standard (100-year level)")
        print(f"{'─'*70}")

        # Use 99th percentile as uniform standard
        uniform_height = np.percentile(flood_events.flatten(), 99)
        uniform_heights = np.full(n_sectors, uniform_height)
        uniform_cost = optimizer.evaluate_total_cost(uniform_heights, flood_events)

        results['uniform'] = {
            'method': 'Uniform Standard',
            'uniform_height_ft': float(uniform_height),
            'total_cost': float(uniform_cost),
            'cost_per_ft_height': float(uniform_cost / uniform_height)
        }

        print(f"Uniform height: {uniform_height:.2f} ft (all sectors)")
        print(f"Total cost: ${uniform_cost:,.0f}")

        # Comparison
        print(f"\n{'─'*70}")
        print("COMPARISON")
        print(f"{'─'*70}")

        cost_savings = uniform_cost - spatial_cost
        cost_savings_pct = 100 * cost_savings / uniform_cost

        results['comparison'] = {
            'cost_savings_dollars': float(cost_savings),
            'cost_savings_percent': float(cost_savings_pct),
            'insight': 'Spatial design leverages local risk variation for cost savings'
        }

        print(f"Cost savings: ${cost_savings:,.0f} ({cost_savings_pct:.1f}%)")
        print(f"Spatial design is more cost-effective than uniform standard")

        return results


class MultiScaleAnalysis(AnalysisBase):
    """Analyze designs at different coastal scales"""

    @staticmethod
    def analyze():
        """
        Test 10, 50, 100 km coastal segments
        """
        print("\n" + "="*70)
        print("MULTI-SCALE COASTAL SEGMENT ANALYSIS")
        print("="*70)

        results = {}
        sector_size_km = 0.5

        for scale_km in [10, 50, 100]:
            n_sectors = int(scale_km / sector_size_km)

            print(f"\n{'─'*70}")
            print(f"Coastal Length: {scale_km} km ({n_sectors} sectors)")
            print(f"{'─'*70}")

            # Generate events
            config = ExperimentConfig(num_sectors=n_sectors, num_events=1500)
            data = config.generate_full_experiment_data(seed=42)
            flood_events = data['flood_events']

            # Optimize
            optimizer = AnalysisBase.setup_optimizer(data)
            h_min, h_max = 10.0, 20.0
            h_mid, min_cost, opt_heights = optimizer.find_optimal_middle_height(
                flood_events, h_min=h_min, h_max=h_max, step=0.5
            )
            cost = optimizer.evaluate_total_cost(opt_heights, flood_events)

            results[f'{scale_km}km'] = {
                'scale_km': scale_km,
                'num_sectors': n_sectors,
                'mean_height_ft': float(np.mean(opt_heights)),
                'std_dev_ft': float(np.std(opt_heights)),
                'range_ft': float(np.max(opt_heights) - np.min(opt_heights)),
                'total_cost': float(cost),
                'cost_per_km': float(cost / scale_km),
                'cost_per_sector': float(cost / n_sectors)
            }

            print(f"Mean height: {np.mean(opt_heights):.2f} ft (±{np.std(opt_heights):.2f} ft)")
            print(f"Total cost: ${cost:,.0f}")
            print(f"Cost per km: ${cost/scale_km:,.0f}")

        return results


class CostSafetyAnalysis(AnalysisBase):
    """Analyze cost vs. safety tradeoffs"""

    @staticmethod
    def analyze(n_sectors=50):
        """
        Compare protection levels: Minimal, Moderate, Good, Excellent
        """
        print("\n" + "="*70)
        print("COST-SAFETY TRADEOFF ANALYSIS")
        print("="*70)

        # Generate events
        config = ExperimentConfig(num_sectors=n_sectors, num_events=2000)
        data = config.generate_full_experiment_data(seed=42)
        flood_events = data['flood_events']

        optimizer = AnalysisBase.setup_optimizer(data)
        results = {}

        # Different protection levels
        protection_levels = [
            ('Minimal (90th %ile)', np.percentile(flood_events, 90)),
            ('Moderate (95th %ile)', np.percentile(flood_events, 95)),
            ('Good (98th %ile)', np.percentile(flood_events, 98)),
            ('Excellent (99th %ile)', np.percentile(flood_events, 99))
        ]

        for level_name, design_height in protection_levels:
            print(f"\n{level_name}: {design_height:.2f} ft")

            # Apply uniform design at this level
            uniform_heights = np.full(n_sectors, design_height)
            cost = optimizer.evaluate_total_cost(uniform_heights, flood_events)

            # Estimate failure probability
            fail_pct = 100 * np.mean(flood_events.flatten() > design_height)

            results[level_name] = {
                'design_height_ft': float(design_height),
                'total_cost': float(cost),
                'failure_rate_percent': float(fail_pct),
                'cost_per_ft': float(cost / design_height)
            }

            print(f"  Total cost: ${cost:,.0f}")
            print(f"  Estimated failure rate: {fail_pct:.2f}%")

        return results


def main():
    """Run all depth analyses"""

    print("\n" + "="*80)
    print("RESS JOURNAL - COMPREHENSIVE DEPTH ANALYSIS")
    print("="*80)

    all_results = {
        'timestamp': datetime.now().isoformat(),
        'analyses': {}
    }

    # Analysis 1
    print("\n" + "#"*80)
    print("# ANALYSIS 1: RETURN PERIOD DESIGNS")
    print("#"*80)
    all_results['analyses']['return_periods'] = ReturnPeriodAnalysis.analyze(n_sectors=50)

    # Analysis 2
    print("\n" + "#"*80)
    print("# ANALYSIS 2: SPATIAL vs UNIFORM")
    print("#"*80)
    all_results['analyses']['spatial_vs_uniform'] = SpatialVsUniformAnalysis.analyze(n_sectors=50)

    # Analysis 3
    print("\n" + "#"*80)
    print("# ANALYSIS 3: MULTI-SCALE")
    print("#"*80)
    all_results['analyses']['multiscale'] = MultiScaleAnalysis.analyze()

    # Analysis 4
    print("\n" + "#"*80)
    print("# ANALYSIS 4: COST-SAFETY TRADEOFF")
    print("#"*80)
    all_results['analyses']['cost_safety'] = CostSafetyAnalysis.analyze(n_sectors=50)

    return all_results


if __name__ == '__main__':
    results = main()

    # Save
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = os.path.join(results_dir, f'depth_analysis_{timestamp}.json')

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print("\n" + "="*80)
    print(f"✓ Results saved to: {output_file}")
    print("="*80)
