"""
Sensitivity Analysis and Extreme Scenario Comparison for RESS Submission

Implements:
1. Parametric sensitivity analysis (GEV, Hüsler-Reiss, cost parameters)
2. Return period comparisons (100, 500, 1000-year)
3. Cost-safety Pareto frontier analysis
4. Multi-scale coastal segment analysis
"""

import numpy as np
import json
from typing import Dict, Tuple, List
from datetime import datetime
from seawall_model import (
    SurgeDistribution, CostFunction, DamageFunction,
    PairwiseRelativeHeight, SpatialSeawallOptimizer
)
from data_generator import FloodEventGenerator, ExperimentConfig


class ParameterSensitivityAnalyzer:
    """Analyze sensitivity of optimal design to key parameters"""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.base_optimizer = SpatialSeawallOptimizer(config)

    def sensitivity_to_gev_shape(self, xi_range=[-0.2, -0.1, 0, 0.1, 0.2]):
        """
        Analyze sensitivity to GEV shape parameter ξ

        Physical interpretation:
        - ξ > 0 (Fréchet): Heavy tail, extreme risks
        - ξ = 0 (Gumbel): Exponential tail
        - ξ < 0 (Weibull): Bounded tail, limited extremes
        """
        print("\n" + "="*70)
        print("GEV SHAPE PARAMETER (ξ) SENSITIVITY ANALYSIS")
        print("="*70)

        results = {}
        baseline_height = None
        baseline_cost = None

        for xi in xi_range:
            # Modify GEV shape parameter
            modified_config = self.config
            modified_config.gev_shape = xi

            # Generate events with modified distribution
            generator = FloodEventGenerator(modified_config)
            events = generator.generate_surge_events()

            # Run optimization
            optimizer = SpatialSeawallOptimizer(modified_config)
            opt_heights = optimizer.optimize_pairwise(events)

            # Compute metrics
            mean_height = np.mean(opt_heights)
            cost = self.base_optimizer.compute_total_cost(opt_heights, events)

            if baseline_height is None:
                baseline_height = mean_height
                baseline_cost = cost

            results[f'xi_{xi:.2f}'] = {
                'xi': xi,
                'mean_height_ft': mean_height,
                'height_change_percent': 100 * (mean_height - baseline_height) / baseline_height,
                'total_cost': cost,
                'cost_change_percent': 100 * (cost - baseline_cost) / baseline_cost,
                'physical_meaning': self._interpret_xi(xi)
            }

            print(f"\nξ = {xi:+.2f} ({self._interpret_xi(xi)}):")
            print(f"  Mean height: {mean_height:.2f} ft ({results[f'xi_{xi:.2f}']['height_change_percent']:+.2f}%)")
            print(f"  Total cost: ${cost:,.0f} ({results[f'xi_{xi:.2f}']['cost_change_percent']:+.2f}%)")

        return results

    def sensitivity_to_husler_reiss_lambda(self, lambda_range=[0.3, 0.5, 0.7, 0.9]):
        """
        Analyze sensitivity to Hüsler-Reiss extremal dependence coefficient λ

        Physical interpretation:
        - λ near 0: Nearly independent extremes (weak correlation)
        - λ = 0.5: Moderate spatial dependence
        - λ near 1: Strong spatial correlation (unified risk)
        """
        print("\n" + "="*70)
        print("HÜSLER-REISS SPATIAL DEPENDENCE (λ) SENSITIVITY ANALYSIS")
        print("="*70)

        results = {}
        baseline_height = None
        baseline_cost = None

        for lam in lambda_range:
            # Modify Hüsler-Reiss parameter
            modified_config = self.config
            modified_config.husler_reiss_lambda = lam

            # Generate events with modified correlation
            generator = FloodEventGenerator(modified_config)
            events = generator.generate_surge_events()

            # Run optimization
            optimizer = SpatialSeawallOptimizer(modified_config)
            opt_heights = optimizer.optimize_pairwise(events)

            # Compute metrics
            mean_height = np.mean(opt_heights)
            height_std = np.std(opt_heights)
            cost = self.base_optimizer.compute_total_cost(opt_heights, events)

            if baseline_height is None:
                baseline_height = mean_height
                baseline_cost = cost

            results[f'lambda_{lam:.2f}'] = {
                'lambda': lam,
                'mean_height_ft': mean_height,
                'height_std_ft': height_std,
                'height_differentiation': np.max(opt_heights) - np.min(opt_heights),
                'mean_height_change_percent': 100 * (mean_height - baseline_height) / baseline_height,
                'total_cost': cost,
                'cost_change_percent': 100 * (cost - baseline_cost) / baseline_cost,
                'correlation_regime': self._interpret_lambda(lam)
            }

            print(f"\nλ = {lam:.2f} ({self._interpret_lambda(lam)}):")
            print(f"  Mean height: {mean_height:.2f} ft ({results[f'lambda_{lam:.2f}']['mean_height_change_percent']:+.2f}%)")
            print(f"  Height variation (std): {height_std:.3f} ft")
            print(f"  Height differentiation: {results[f'lambda_{lam:.2f}']['height_differentiation']:.2f} ft")
            print(f"  Total cost: ${cost:,.0f} ({results[f'lambda_{lam:.2f}']['cost_change_percent']:+.2f}%)")

        return results

    def sensitivity_to_construction_cost(self, cost_multipliers=[0.8, 0.9, 1.0, 1.1, 1.2]):
        """
        Analyze sensitivity to construction cost parameter

        Tests how construction cost inflation affects design heights and total cost
        """
        print("\n" + "="*70)
        print("CONSTRUCTION COST SENSITIVITY ANALYSIS")
        print("="*70)

        results = {}
        baseline_height = None
        baseline_total_cost = None

        for multiplier in cost_multipliers:
            # Modify cost multiplier
            modified_config = self.config
            modified_config.cost_per_foot_per_sector = (
                modified_config.cost_per_foot_per_sector * multiplier
            )

            # Generate events
            generator = FloodEventGenerator(modified_config)
            events = generator.generate_surge_events()

            # Run optimization
            optimizer = SpatialSeawallOptimizer(modified_config)
            opt_heights = optimizer.optimize_pairwise(events)

            # Compute metrics
            mean_height = np.mean(opt_heights)
            total_cost = self.base_optimizer.compute_total_cost(opt_heights, events)

            if baseline_height is None:
                baseline_height = mean_height
                baseline_total_cost = total_cost

            results[f'cost_mult_{multiplier:.2f}'] = {
                'multiplier': multiplier,
                'unit_cost': modified_config.cost_per_foot_per_sector,
                'mean_height_ft': mean_height,
                'height_change_percent': 100 * (mean_height - baseline_height) / baseline_height,
                'total_cost': total_cost,
                'cost_change_percent': 100 * (total_cost - baseline_total_cost) / baseline_total_cost
            }

            print(f"\nCost multiplier: {multiplier:.2f}x")
            print(f"  Unit cost: ${modified_config.cost_per_foot_per_sector:,.0f}/ft")
            print(f"  Mean height: {mean_height:.2f} ft ({results[f'cost_mult_{multiplier:.2f}']['height_change_percent']:+.2f}%)")
            print(f"  Total cost: ${total_cost:,.0f} ({results[f'cost_mult_{multiplier:.2f}']['cost_change_percent']:+.2f}%)")

        return results

    @staticmethod
    def _interpret_xi(xi):
        if xi < -0.1:
            return "Weibull (bounded tail)"
        elif xi < 0.05:
            return "Gumbel (exponential tail)"
        else:
            return "Fréchet (heavy tail)"

    @staticmethod
    def _interpret_lambda(lam):
        if lam < 0.4:
            return "Weak spatial correlation"
        elif lam < 0.7:
            return "Moderate spatial correlation"
        else:
            return "Strong spatial correlation"


class ReturnPeriodComparison:
    """Compare optimal designs for different return periods (100, 500, 1000-year)"""

    def __init__(self, config: ExperimentConfig):
        self.config = config

    def compare_return_periods(self, return_periods=[100, 500, 1000], n_events=5000):
        """
        Compare optimal designs across different return periods

        Return period T relates to annual probability p: p = 1/T
        GEV return level: z_T = μ + (σ/ξ) * {[(-ln(1-p))^(-ξ)] - 1}
        """
        print("\n" + "="*70)
        print("RETURN PERIOD COMPARISON ANALYSIS")
        print("="*70)

        results = {}

        for T in return_periods:
            print(f"\n{'='*70}")
            print(f"RETURN PERIOD: {T}-YEAR ({1/T*100:.2f}% annual probability)")
            print(f"{'='*70}")

            # Calculate GEV return level for this period
            p = 1 / T
            gev_return_level = self._calculate_gev_return_level(p)

            # Generate events scaled to this return period
            generator = FloodEventGenerator(self.config)
            events = generator.generate_surge_events(n_events=n_events)

            # Scale events to represent return period
            mean_event = np.mean(events)
            scaled_events = events + (gev_return_level - mean_event)

            # Run optimization
            optimizer = SpatialSeawallOptimizer(self.config)
            opt_heights = optimizer.optimize_pairwise(scaled_events)

            # Compute metrics
            mean_height = np.mean(opt_heights)
            height_range = np.max(opt_heights) - np.min(opt_heights)
            cost = optimizer.compute_total_cost(opt_heights, scaled_events)
            damage = np.mean([optimizer.damage_function.compute_expected_damage(h, e.max())
                            for h, e in zip(opt_heights, scaled_events.T)])

            results[f'T_{T}'] = {
                'return_period': T,
                'annual_probability_percent': 1/T * 100,
                'gev_return_level_ft': gev_return_level,
                'mean_height_ft': mean_height,
                'height_range_ft': height_range,
                'total_cost': cost,
                'expected_annual_damage': damage,
                'cost_benefit_ratio': cost / damage if damage > 0 else float('inf')
            }

            print(f"GEV Return Level: {gev_return_level:.2f} ft")
            print(f"Mean optimal height: {mean_height:.2f} ft")
            print(f"Height range: {height_range:.2f} ft")
            print(f"Total cost: ${cost:,.0f}")
            print(f"Expected annual damage: ${damage:,.0f}")
            print(f"Cost-benefit ratio: {results[f'T_{T}']['cost_benefit_ratio']:.2f}")

        return results

    def _calculate_gev_return_level(self, p, mu=8.5, sigma=2.2, xi=0.1):
        """Calculate GEV return level for exceedance probability p"""
        if xi == 0:
            return mu - sigma * np.log(-np.log(1 - p))
        else:
            return mu + (sigma / xi) * (((-np.log(1 - p)) ** (-xi)) - 1)


class ParetoFrontierAnalysis:
    """Analyze cost-safety Pareto frontier"""

    def __init__(self, config: ExperimentConfig):
        self.config = config

    def generate_pareto_frontier(self, target_pf_values=None):
        """
        Generate Pareto frontier by varying target failure probability

        Shows trade-off between cost and safety (reliability)
        """
        if target_pf_values is None:
            # Create range from 10^-2 to 10^-4 (1% to 0.01%)
            target_pf_values = [10**(-i/2) for i in range(4, 9)]

        print("\n" + "="*70)
        print("COST-SAFETY PARETO FRONTIER ANALYSIS")
        print("="*70)

        results = {
            'target_pf_values': [],
            'mean_costs': [],
            'mean_heights': [],
            'reliability_index': []
        }

        for pf_target in target_pf_values:
            print(f"\nTarget failure probability: {pf_target:.2e} ({pf_target*100:.4f}%)")

            # Generate events
            generator = FloodEventGenerator(self.config)
            events = generator.generate_surge_events()

            # Adjust design to meet target Pf
            optimizer = SpatialSeawallOptimizer(self.config)

            # Iterative design adjustment
            heights = np.full(self.config.n_sectors, 14.0)  # Initial guess

            for iteration in range(5):  # Max 5 iterations
                # Estimate current system Pf
                current_pf = self._estimate_system_pf(heights, events)

                if abs(current_pf - pf_target) / pf_target < 0.1:  # Within 10%
                    break

                # Adjust heights proportionally
                height_adjustment = self._height_adjustment_for_pf(
                    current_pf, pf_target, heights
                )
                heights += height_adjustment
                heights = np.clip(heights, 10.0, 20.0)  # Bounds

            # Compute final metrics
            final_pf = self._estimate_system_pf(heights, events)
            cost = optimizer.compute_total_cost(heights, events)
            mean_height = np.mean(heights)
            beta = -np.log(final_pf)  # Approximate reliability index

            results['target_pf_values'].append(pf_target)
            results['mean_costs'].append(cost)
            results['mean_heights'].append(mean_height)
            results['reliability_index'].append(beta)

            print(f"  Mean height: {mean_height:.2f} ft")
            print(f"  Achieved Pf: {final_pf:.2e}")
            print(f"  Total cost: ${cost:,.0f}")
            print(f"  Reliability index (β): {beta:.2f}")

        return results

    @staticmethod
    def _estimate_system_pf(heights, events):
        """Estimate system failure probability as max of component Pf"""
        component_pf = np.mean(events > heights[:, np.newaxis], axis=1)
        return np.max(component_pf)

    @staticmethod
    def _height_adjustment_for_pf(current_pf, target_pf, heights):
        """Calculate height adjustment to reach target Pf"""
        ratio = target_pf / (current_pf + 1e-10)
        if ratio < 1:
            return -(np.log(1/ratio) / 2) * np.ones_like(heights)
        else:
            return np.log(ratio / 2) * np.ones_like(heights)


class MultiScaleAnalysis:
    """Analyze coastal segment at different scales (10, 50, 100, 500 km)"""

    def __init__(self, base_config: ExperimentConfig):
        self.base_config = base_config

    def analyze_coastal_scales(self, scales_km=[10, 50, 100, 500], sector_size_km=0.5):
        """
        Analyze how coastal segment length affects optimal design

        Tests: Do longer coasts have different design characteristics?
        """
        print("\n" + "="*70)
        print("MULTI-SCALE COASTAL SEGMENT ANALYSIS")
        print("="*70)

        results = {}

        for scale_km in scales_km:
            n_sectors = int(scale_km / sector_size_km)

            print(f"\n{'='*70}")
            print(f"COASTAL SCALE: {scale_km} km ({n_sectors} sectors @ {sector_size_km} km each)")
            print(f"{'='*70}")

            # Modify config for this scale
            modified_config = self.base_config
            modified_config.n_sectors = n_sectors

            # Generate events
            generator = FloodEventGenerator(modified_config)
            events = generator.generate_surge_events()

            # Run optimization
            optimizer = SpatialSeawallOptimizer(modified_config)
            opt_heights = optimizer.optimize_pairwise(events)
            cost = optimizer.compute_total_cost(opt_heights, events)

            # Compute metrics
            mean_height = np.mean(opt_heights)
            height_std = np.std(opt_heights)
            cost_per_km = cost / scale_km
            cost_per_sector = cost / n_sectors

            results[f'scale_{scale_km}km'] = {
                'scale_km': scale_km,
                'num_sectors': n_sectors,
                'mean_height_ft': mean_height,
                'height_std_ft': height_std,
                'height_differentiation_ft': np.max(opt_heights) - np.min(opt_heights),
                'total_cost': cost,
                'cost_per_km': cost_per_km,
                'cost_per_sector': cost_per_sector,
                'cost_efficiency_per_km': cost / (scale_km * mean_height)  # $/km/ft
            }

            print(f"Sectors: {n_sectors}")
            print(f"Mean height: {mean_height:.2f} ft (±{height_std:.2f} ft)")
            print(f"Height differentiation: {results[f'scale_{scale_km}km']['height_differentiation_ft']:.2f} ft")
            print(f"Total cost: ${cost:,.0f}")
            print(f"Cost per km: ${cost_per_km:,.0f}")
            print(f"Cost per sector: ${cost_per_sector:,.0f}")

        return results


def run_comprehensive_sensitivity_analysis(n_events=1000, n_sectors=100):
    """Execute all sensitivity and scenario analyses"""

    print("\n" + "="*80)
    print(" "*20 + "RESS JOURNAL DEPTH ANALYSIS")
    print(" "*15 + "Sensitivity and Scenario Comparison Study")
    print("="*80)

    config = ExperimentConfig(
        num_sectors=n_sectors,
        num_events=n_events
    )

    all_results = {
        'timestamp': datetime.now().isoformat(),
        'configuration': {
            'n_sectors': config.n_sectors,
            'n_events': n_events
        }
    }

    # 1. Parameter Sensitivity
    print("\n" + "#"*80)
    print("# 1. PARAMETER SENSITIVITY ANALYSIS")
    print("#"*80)

    sensitivity = ParameterSensitivityAnalyzer(config)
    all_results['gev_shape_sensitivity'] = sensitivity.sensitivity_to_gev_shape()
    all_results['husler_reiss_sensitivity'] = sensitivity.sensitivity_to_husler_reiss_lambda()
    all_results['cost_sensitivity'] = sensitivity.sensitivity_to_construction_cost()

    # 2. Return Period Comparison
    print("\n" + "#"*80)
    print("# 2. RETURN PERIOD COMPARISON (100, 500, 1000-YEAR)")
    print("#"*80)

    return_period = ReturnPeriodComparison(config)
    all_results['return_period_comparison'] = return_period.compare_return_periods()

    # 3. Pareto Frontier
    print("\n" + "#"*80)
    print("# 3. COST-SAFETY PARETO FRONTIER")
    print("#"*80)

    pareto = ParetoFrontierAnalysis(config)
    all_results['pareto_frontier'] = pareto.generate_pareto_frontier()

    # 4. Multi-scale Analysis
    print("\n" + "#"*80)
    print("# 4. MULTI-SCALE COASTAL SEGMENT ANALYSIS")
    print("#"*80)

    multiscale = MultiScaleAnalysis(config)
    all_results['multiscale_analysis'] = multiscale.analyze_coastal_scales()

    return all_results


if __name__ == '__main__':
    results = run_comprehensive_sensitivity_analysis(n_events=1000)

    # Save results
    import os
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = os.path.join(results_dir, f'sensitivity_analysis_{timestamp}.json')

    with open(output_file, 'w') as f:
        # Convert numpy types to native Python for JSON
        json.dump(results, f, indent=2, default=str)

    print("\n" + "="*80)
    print(f"Results saved to: {output_file}")
    print("="*80)
