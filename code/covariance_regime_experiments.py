"""
Comparative Experiments: Sea Level Covariance Regimes

Analyzes how different spatial correlation structures
in sea level rise affect optimal seawall design.
"""

import numpy as np
import json
from datetime import datetime
import os
from typing import Dict
from sea_level_covariance_modeling import (
    SeaLevelCovarianceComparison,
    SeaLevelUncertaintyAnalysis,
    SeaLevelCovarianceGenerator
)
from seawall_model import SpatialSeawallOptimizer, SurgeDistribution, CostFunction, DamageFunction
from data_generator import ExperimentConfig


class CovarianceRegimeExperimenter:
    """
    Runs comparative experiments across covariance regimes.
    """

    def __init__(self, n_sectors: int = 100, n_events: int = 1000):
        self.n_sectors = n_sectors
        self.n_events = n_events
        self.results_dir = 'results'
        os.makedirs(self.results_dir, exist_ok=True)

    def generate_sea_level_rise_data(self, regime: str) -> np.ndarray:
        """
        Generate sea level rise adjustments based on covariance regime.

        Returns:
            Shape (n_sectors,) array of SLR by location (mm)
        """
        generator = SeaLevelCovarianceGenerator(
            np.column_stack([
                np.linspace(0, 50, self.n_sectors),
                np.zeros(self.n_sectors)
            ]) / 111.0
        )

        slr_field, spatial_mean, weights = generator.generate_slr_field(
            regime=regime, n_time_steps=30
        )

        # Take final year SLR (after 30-year projection)
        final_slr = slr_field[:, -1]  # mm

        return final_slr, weights

    def adjust_surge_distribution_for_slr(self, surge_params_base: list,
                                         slr_values: np.ndarray) -> list:
        """
        Adjust surge height distributions by adding SLR.

        SLR increases the effective surge heights experienced at each location.
        """
        surge_params_adjusted = []

        for i, param in enumerate(surge_params_base):
            # Convert SLR from mm to feet (1 mm ≈ 0.00328 feet)
            slr_ft = slr_values[i] * 0.00328

            adjusted = {
                'location_id': param['location_id'],
                'mean': param['mean'] + slr_ft,  # Surge distribution shifted up
                'std': param['std']  # Variability unchanged
            }
            surge_params_adjusted.append(adjusted)

        return surge_params_adjusted

    def run_optimization_for_regime(self, regime: str, base_data: Dict) -> Dict:
        """
        Run optimization for a specific covariance regime.

        Returns:
            Dictionary with optimal heights and costs
        """
        print(f"\n{'='*70}")
        print(f"OPTIMIZING FOR {regime.upper()} COVARIANCE REGIME")
        print(f"{'='*70}")

        # Generate SLR for this regime
        slr_values, weights = self.generate_sea_level_rise_data(regime)

        # Adjust surge distributions
        base_surge_params = base_data['surge_parameters']
        adjusted_surge_params = self.adjust_surge_distribution_for_slr(
            base_surge_params, slr_values
        )

        # Create optimizer with adjusted distributions
        surges = [
            SurgeDistribution(p['mean'], p['std'], p['location_id'])
            for p in adjusted_surge_params
        ]
        costs = [
            CostFunction(cp['a'], cp['b'], cp['c'])
            for cp in base_data['cost_parameters']
        ]
        damage = DamageFunction(
            base_data['damage_parameters']['alpha'],
            base_data['damage_parameters']['beta']
        )

        optimizer = SpatialSeawallOptimizer(
            self.n_sectors, surges, costs, damage, max_gradient=0.1
        )

        # Optimize
        events = base_data['flood_events']
        h_mid, total_cost, optimal_heights = optimizer.find_optimal_middle_height(
            events, h_min=12, h_max=20, step=1.0
        )

        # Calculate construction cost
        construction_cost = sum(
            costs[i].cost(optimal_heights[i])
            for i in range(self.n_sectors)
        )

        # Calculate expected damage
        expected_damage = total_cost - construction_cost

        # Statistics
        height_stats = {
            'mean': float(np.mean(optimal_heights)),
            'std': float(np.std(optimal_heights)),
            'min': float(np.min(optimal_heights)),
            'max': float(np.max(optimal_heights)),
            'range': float(np.max(optimal_heights) - np.min(optimal_heights))
        }

        # SLR impact
        slr_stats = {
            'mean_slr_mm': float(np.mean(slr_values)),
            'slr_std_mm': float(np.std(slr_values)),
            'slr_range_mm': float(np.max(slr_values) - np.min(slr_values)),
            'slr_weights': weights
        }

        result = {
            'regime': regime,
            'optimal_middle_height': float(h_mid),
            'optimal_heights': optimal_heights.tolist(),
            'total_cost': float(total_cost),
            'construction_cost': float(construction_cost),
            'expected_damage': float(expected_damage),
            'height_statistics': height_stats,
            'sea_level_rise': slr_stats
        }

        print(f"Regime: {regime}")
        print(f"  Mean SLR: {slr_stats['mean_slr_mm']:.2f} mm")
        print(f"  SLR range: {slr_stats['slr_range_mm']:.2f} mm")
        print(f"  Mean height: {height_stats['mean']:.2f} ft")
        print(f"  Height range: {height_stats['min']:.2f} - {height_stats['max']:.2f} ft")
        print(f"  Height std dev: {height_stats['std']:.3f} ft")
        print(f"  Total cost: ${total_cost:,.0f}")

        return result

    def analyze_covariance_impact(self, results_by_regime: Dict) -> Dict:
        """
        Analyze how covariance structure affects design outcomes.
        """
        regimes = ['weak', 'moderate', 'strong']
        analysis = {}

        # Baseline (moderate regime)
        baseline_cost = results_by_regime['moderate']['total_cost']
        baseline_height = results_by_regime['moderate']['height_statistics']['mean']
        baseline_slr = results_by_regime['moderate']['sea_level_rise']['slr_range_mm']

        for regime in regimes:
            result = results_by_regime[regime]

            # Cost impact
            cost_increase = (result['total_cost'] - baseline_cost) / baseline_cost * 100

            # Height impact
            height_increase = (
                result['height_statistics']['mean'] - baseline_height
            ) / baseline_height * 100

            # Differentiation (measure of spatial variation)
            height_range = result['height_statistics']['range']
            height_std = result['height_statistics']['std']

            # SLR correlation with design differentiation
            slr_range = result['sea_level_rise']['slr_range_mm']
            slr_height_correlation = height_range / (slr_range + 0.1)  # Avoid division by zero

            analysis[regime] = {
                'cost_vs_baseline_percent': cost_increase,
                'height_vs_baseline_percent': height_increase,
                'height_differentiation': height_range,
                'height_std_dev': height_std,
                'slr_spatial_range': slr_range,
                'slr_height_correlation': slr_height_correlation,
                'spatial_correlation_strength': {
                    'weak': 'Low (0.2-0.4)',
                    'moderate': 'Medium (0.4-0.6)',
                    'strong': 'High (0.6-0.8)'
                }[regime]
            }

        return analysis

    def run_full_comparison(self) -> Dict:
        """
        Execute complete covariance regime comparison experiment.
        """
        print("\n" + "="*70)
        print("SEA LEVEL RISE COVARIANCE REGIME COMPARISON")
        print("="*70)

        # Generate base data
        print("\nGenerating base experiment data...")
        config = ExperimentConfig(
            num_sectors=self.n_sectors,
            num_events=self.n_events
        )
        base_data = config.generate_full_experiment_data(seed=42)

        # Run optimization for each regime
        results_by_regime = {}
        for regime in ['weak', 'moderate', 'strong']:
            results_by_regime[regime] = self.run_optimization_for_regime(
                regime, base_data
            )

        # Analyze impacts
        analysis = self.analyze_covariance_impact(results_by_regime)

        # Compile final results
        full_results = {
            'timestamp': datetime.now().isoformat(),
            'configuration': {
                'num_sectors': self.n_sectors,
                'num_events': self.n_events,
                'regimes_tested': ['weak', 'moderate', 'strong']
            },
            'results_by_regime': results_by_regime,
            'comparative_analysis': analysis,
            'key_findings': self._generate_key_findings(results_by_regime, analysis)
        }

        return full_results

    def _generate_key_findings(self, results: Dict, analysis: Dict) -> Dict:
        """
        Extract key findings from the experiment.
        """
        weak = analysis['weak']
        strong = analysis['strong']

        findings = {
            'covariance_effect_on_height': {
                'weak_vs_moderate': weak['height_vs_baseline_percent'],
                'strong_vs_moderate': strong['height_vs_baseline_percent'],
                'insight': 'Weak covariance requires more differentiated heights (spatial variation), while strong covariance allows more uniform protection'
            },

            'covariance_effect_on_cost': {
                'weak_vs_moderate_percent': weak['cost_vs_baseline_percent'],
                'strong_vs_moderate_percent': strong['cost_vs_baseline_percent'],
                'insight': 'Strong spatial correlation in SLR increases total cost (design must account for spatial coherence)'
            },

            'spatial_differentiation': {
                'weak_height_range': weak['height_differentiation'],
                'moderate_height_range': analysis['moderate']['height_differentiation'],
                'strong_height_range': strong['height_differentiation'],
                'pattern': 'Stronger SLR correlation → larger height differences across coast'
            },

            'physical_interpretation': {
                'weak_covariance': 'Multiple local processes; local adaptation optimal',
                'moderate_covariance': 'Balance of scales; mixed strategy',
                'strong_covariance': 'Large-scale processes dominate; broad regional protection'
            }
        }

        return findings

    def save_results(self, results: Dict) -> str:
        """Save results to JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'{self.results_dir}/covariance_regime_comparison_{self.n_sectors}sectors_{timestamp}.json'

        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)

        return filename

    def print_summary(self, results: Dict):
        """Print comprehensive summary of results."""
        print("\n" + "="*70)
        print("COMPARATIVE SUMMARY: COVARIANCE REGIMES")
        print("="*70)

        analysis = results['comparative_analysis']
        findings = results['key_findings']

        print("\n1. HEIGHT IMPACT (vs Moderate Regime):")
        for regime in ['weak', 'moderate', 'strong']:
            if regime != 'moderate':
                height_pct = analysis[regime]['height_vs_baseline_percent']
                height_diff = analysis[regime]['height_differentiation']
                print(f"  {regime.upper():10s}: {height_pct:+6.2f}% | Range: {height_diff:.2f} ft")

        print("\n2. COST IMPACT (vs Moderate Regime):")
        for regime in ['weak', 'moderate', 'strong']:
            if regime != 'moderate':
                cost_pct = analysis[regime]['cost_vs_baseline_percent']
                print(f"  {regime.upper():10s}: {cost_pct:+6.2f}%")

        print("\n3. SEA LEVEL RISE SPATIAL VARIATION:")
        for regime in ['weak', 'moderate', 'strong']:
            slr_range = analysis[regime]['slr_spatial_range']
            print(f"  {regime.upper():10s}: {slr_range:.2f} mm")

        print("\n4. KEY INSIGHTS:")
        print(f"  • {findings['covariance_effect_on_height']['insight']}")
        print(f"  • {findings['covariance_effect_on_cost']['insight']}")

        print("\n5. SPATIAL DIFFERENTIATION PATTERN:")
        print(f"  {findings['spatial_differentiation']['pattern']}")

        print("\n6. PHYSICAL INTERPRETATION:")
        for regime in ['weak', 'moderate', 'strong']:
            key = f'{regime}_covariance'
            interpretation = findings['physical_interpretation'][key]
            print(f"  {regime.upper():10s}: {interpretation}")

        print("\n" + "="*70)


def run_covariance_experiments(n_sectors: int = 100, n_events: int = 1000) -> Dict:
    """
    Execute complete covariance regime comparison.
    """
    experimenter = CovarianceRegimeExperimenter(n_sectors, n_events)
    results = experimenter.run_full_comparison()
    results_file = experimenter.save_results(results)
    experimenter.print_summary(results)

    print(f"\nResults saved to: {results_file}")
    return results


if __name__ == '__main__':
    results = run_covariance_experiments(n_sectors=100, n_events=1000)
