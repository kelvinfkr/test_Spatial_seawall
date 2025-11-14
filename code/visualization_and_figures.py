"""
Publication-Quality Visualization for RESS Journal

Generates all figures for sensitivity analysis, return period comparison,
Pareto frontier, multi-scale analysis, and coastal design comparisons.

Exports to EPS/PDF for journal submission.
"""

import numpy as np
import json
import os
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.gridspec import GridSpec
    import matplotlib.ticker as mticker
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available. Cannot generate figures.")


class PublicationVisualizer:
    """Create publication-quality figures for RESS submission"""

    def __init__(self, dpi=300, font_size=11):
        self.dpi = dpi
        self.font_size = font_size
        if HAS_MATPLOTLIB:
            plt.rcParams['font.size'] = font_size
            plt.rcParams['font.family'] = 'Times New Roman'
            plt.rcParams['figure.dpi'] = dpi

    def plot_sensitivity_analysis(self, results_dict, output_dir='../paper/figures'):
        """
        Create 3-panel figure showing parameter sensitivity

        Panel A: GEV shape parameter (ξ) sensitivity
        Panel B: Hüsler-Reiss dependence (λ) sensitivity
        Panel C: Construction cost sensitivity
        """
        if not HAS_MATPLOTLIB:
            print("Skipping visualization - matplotlib not available")
            return

        os.makedirs(output_dir, exist_ok=True)

        fig = plt.figure(figsize=(14, 5))
        gs = GridSpec(1, 3, figure=fig, hspace=0.3, wspace=0.35)

        # Panel A: GEV Shape Sensitivity
        ax_a = fig.add_subplot(gs[0, 0])
        gev_data = results_dict.get('gev_shape_sensitivity', {})

        if gev_data:
            xi_values = [float(k.split('_')[1]) for k in sorted(gev_data.keys())]
            height_changes = [gev_data[f'xi_{xi:.2f}']['height_change_percent'] for xi in xi_values]
            cost_changes = [gev_data[f'xi_{xi:.2f}']['cost_change_percent'] for xi in xi_values]

            ax_a_twin = ax_a.twinx()

            line1 = ax_a.plot(xi_values, height_changes, 'o-', color='#2E86AB', linewidth=2.5,
                             markersize=8, label='Height change')
            line2 = ax_a_twin.plot(xi_values, cost_changes, 's--', color='#A23B72', linewidth=2.5,
                                   markersize=8, label='Cost change')

            ax_a.set_xlabel('GEV Shape Parameter (ξ)', fontsize=11, fontweight='bold')
            ax_a.set_ylabel('Height Change (%)', color='#2E86AB', fontsize=11, fontweight='bold')
            ax_a_twin.set_ylabel('Cost Change (%)', color='#A23B72', fontsize=11, fontweight='bold')
            ax_a.tick_params(axis='y', labelcolor='#2E86AB')
            ax_a_twin.tick_params(axis='y', labelcolor='#A23B72')
            ax_a.grid(True, alpha=0.3)
            ax_a.axvline(x=0, color='gray', linestyle=':', alpha=0.5)
            ax_a.axhline(y=0, color='gray', linestyle=':', alpha=0.5)

            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax_a.legend(lines, labels, loc='upper left', fontsize=10)
            ax_a.set_title('(A) GEV Shape Parameter Sensitivity', fontweight='bold', fontsize=11)

        # Panel B: Hüsler-Reiss Dependence Sensitivity
        ax_b = fig.add_subplot(gs[0, 1])
        hr_data = results_dict.get('husler_reiss_sensitivity', {})

        if hr_data:
            lambda_values = sorted([float(k.split('_')[1]) for k in hr_data.keys()])
            mean_heights = [hr_data[f'lambda_{lam:.2f}']['mean_height_ft'] for lam in lambda_values]
            height_diffs = [hr_data[f'lambda_{lam:.2f}']['height_differentiation'] for lam in lambda_values]

            ax_b_twin = ax_b.twinx()

            line1 = ax_b.plot(lambda_values, mean_heights, 'D-', color='#06A77D', linewidth=2.5,
                             markersize=8, label='Mean height')
            line2 = ax_b_twin.plot(lambda_values, height_diffs, '^--', color='#F18F01', linewidth=2.5,
                                   markersize=8, label='Height differentiation')

            ax_b.set_xlabel('Hüsler-Reiss Dependence (λ)', fontsize=11, fontweight='bold')
            ax_b.set_ylabel('Mean Height (ft)', color='#06A77D', fontsize=11, fontweight='bold')
            ax_b_twin.set_ylabel('Height Differentiation (ft)', color='#F18F01', fontsize=11, fontweight='bold')
            ax_b.tick_params(axis='y', labelcolor='#06A77D')
            ax_b_twin.tick_params(axis='y', labelcolor='#F18F01')
            ax_b.grid(True, alpha=0.3)
            ax_b.set_title('(B) Spatial Dependence Sensitivity', fontweight='bold', fontsize=11)

            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax_b.legend(lines, labels, loc='best', fontsize=10)

        # Panel C: Construction Cost Sensitivity
        ax_c = fig.add_subplot(gs[0, 2])
        cost_data = results_dict.get('cost_sensitivity', {})

        if cost_data:
            multipliers = sorted([float(k.split('_')[2]) for k in cost_data.keys()])
            total_costs = [cost_data[f'cost_mult_{m:.2f}']['total_cost'] for m in multipliers]
            mean_heights = [cost_data[f'cost_mult_{m:.2f}']['mean_height_ft'] for m in multipliers]

            ax_c_twin = ax_c.twinx()

            line1 = ax_c.plot(multipliers, np.array(total_costs)/1e6, 'o-', color='#C1121F', linewidth=2.5,
                             markersize=8, label='Total cost')
            line2 = ax_c_twin.plot(multipliers, mean_heights, 's--', color='#003049', linewidth=2.5,
                                   markersize=8, label='Mean height')

            ax_c.set_xlabel('Cost Multiplier', fontsize=11, fontweight='bold')
            ax_c.set_ylabel('Total Cost ($ millions)', color='#C1121F', fontsize=11, fontweight='bold')
            ax_c_twin.set_ylabel('Mean Height (ft)', color='#003049', fontsize=11, fontweight='bold')
            ax_c.tick_params(axis='y', labelcolor='#C1121F')
            ax_c_twin.tick_params(axis='y', labelcolor='#003049')
            ax_c.grid(True, alpha=0.3)
            ax_c.axvline(x=1.0, color='gray', linestyle=':', alpha=0.5, label='Baseline')
            ax_c.set_title('(C) Construction Cost Sensitivity', fontweight='bold', fontsize=11)

            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax_c.legend(lines, labels, loc='best', fontsize=10)

        plt.suptitle('Parameter Sensitivity Analysis for Spatial Seawall Design',
                     fontsize=13, fontweight='bold', y=1.02)

        # Save figure
        output_file = os.path.join(output_dir, 'Figure_1_Sensitivity.eps')
        plt.savefig(output_file, format='eps', dpi=self.dpi, bbox_inches='tight')
        print(f"✓ Saved: {output_file}")

        plt.close()

    def plot_return_period_analysis(self, results_dict, output_dir='../paper/figures'):
        """
        Create 2-panel figure comparing 100, 500, 1000-year designs

        Panel A: Design heights across return periods
        Panel B: Cost-safety tradeoff
        """
        if not HAS_MATPLOTLIB:
            return

        os.makedirs(output_dir, exist_ok=True)

        fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(12, 5))

        rp_data = results_dict.get('return_period_comparison', {})

        if rp_data:
            return_periods = sorted([int(k.split('_')[1]) for k in rp_data.keys()])
            mean_heights = [rp_data[f'T_{T}']['mean_height_ft'] for T in return_periods]
            total_costs = [rp_data[f'T_{T}']['total_cost'] for T in return_periods]
            annual_damages = [rp_data[f'T_{T}']['expected_annual_damage'] for T in return_periods]

            # Panel A: Heights vs Return Period
            colors_a = plt.cm.Blues(np.linspace(0.4, 0.9, len(return_periods)))
            bars_a = ax_a.bar(range(len(return_periods)), mean_heights, color=colors_a,
                             edgecolor='black', linewidth=1.5, width=0.6)

            # Add value labels on bars
            for i, (bar, height) in enumerate(zip(bars_a, mean_heights)):
                ax_a.text(bar.get_x() + bar.get_width()/2, height + 0.2,
                         f'{height:.2f} ft', ha='center', va='bottom', fontsize=10, fontweight='bold')

            ax_a.set_ylabel('Mean Design Height (ft)', fontsize=11, fontweight='bold')
            ax_a.set_xlabel('Return Period (years)', fontsize=11, fontweight='bold')
            ax_a.set_xticks(range(len(return_periods)))
            ax_a.set_xticklabels([f'{T}-year' for T in return_periods])
            ax_a.grid(True, alpha=0.3, axis='y')
            ax_a.set_title('(A) Design Heights by Return Period', fontweight='bold', fontsize=11)
            ax_a.set_ylim(0, max(mean_heights) * 1.15)

            # Panel B: Cost vs Safety
            colors_b = plt.cm.Reds(np.linspace(0.4, 0.9, len(return_periods)))
            x_pos = np.arange(len(return_periods))
            width = 0.35

            bars_cost = ax_b.bar(x_pos - width/2, np.array(total_costs)/1e6, width,
                                label='Total cost', color=colors_b[0], edgecolor='black', linewidth=1.5)

            ax_b_twin = ax_b.twinx()
            bars_damage = ax_b_twin.bar(x_pos + width/2, np.array(annual_damages)/1e3, width,
                                       label='Expected annual damage', color='#FF6B6B',
                                       edgecolor='black', linewidth=1.5)

            ax_b.set_ylabel('Total Cost ($ millions)', fontsize=11, fontweight='bold', color=colors_b[0])
            ax_b_twin.set_ylabel('Expected Annual Damage ($1000s)', fontsize=11, fontweight='bold', color='#FF6B6B')
            ax_b.set_xlabel('Return Period (years)', fontsize=11, fontweight='bold')
            ax_b.set_xticks(x_pos)
            ax_b.set_xticklabels([f'{T}-year' for T in return_periods])
            ax_b.tick_params(axis='y', labelcolor=colors_b[0])
            ax_b_twin.tick_params(axis='y', labelcolor='#FF6B6B')
            ax_b.grid(True, alpha=0.3, axis='y')
            ax_b.set_title('(B) Cost vs. Expected Annual Damage', fontweight='bold', fontsize=11)

            # Legends
            lines1, labels1 = ax_b.get_legend_handles_labels()
            lines2, labels2 = ax_b_twin.get_legend_handles_labels()
            ax_b.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10)

        plt.suptitle('Return Period Comparison for 100, 500, and 1000-Year Designs',
                     fontsize=13, fontweight='bold', y=1.00)

        # Save
        output_file = os.path.join(output_dir, 'Figure_2_Return_Periods.eps')
        plt.savefig(output_file, format='eps', dpi=self.dpi, bbox_inches='tight')
        print(f"✓ Saved: {output_file}")

        plt.close()

    def plot_pareto_frontier(self, results_dict, output_dir='../paper/figures'):
        """
        Create Pareto frontier showing cost-safety tradeoff

        Shows how system cost varies with target reliability
        """
        if not HAS_MATPLOTLIB:
            return

        os.makedirs(output_dir, exist_ok=True)

        fig, ax = plt.subplots(figsize=(10, 7))

        pareto_data = results_dict.get('pareto_frontier', {})

        if pareto_data and 'mean_costs' in pareto_data:
            costs = np.array(pareto_data['mean_costs']) / 1e6
            heights = np.array(pareto_data['mean_heights'])
            pf_values = np.array(pareto_data['target_pf_values'])

            # Create scatter plot with color gradient
            scatter = ax.scatter(costs, heights, c=-np.log10(pf_values), cmap='viridis',
                               s=200, edgecolors='black', linewidth=1.5, alpha=0.8,
                               vmin=0, vmax=4)

            # Connect points
            sorted_idx = np.argsort(costs)
            ax.plot(costs[sorted_idx], heights[sorted_idx], 'k--', alpha=0.3, linewidth=1.5)

            # Add annotations
            for i, (cost, height, pf) in enumerate(zip(costs, heights, pf_values)):
                ax.annotate(f'{pf*100:.3f}%', (cost, height),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=9, alpha=0.7)

            # Colorbar
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('-log₁₀(Target Pf)', fontsize=11, fontweight='bold')

            ax.set_xlabel('Total Cost ($ millions)', fontsize=12, fontweight='bold')
            ax.set_ylabel('Mean Design Height (ft)', fontsize=12, fontweight='bold')
            ax.set_title('Cost-Safety Pareto Frontier\n(Optimal designs for different reliability targets)',
                        fontsize=12, fontweight='bold', pad=15)
            ax.grid(True, alpha=0.3)

            # Add annotation box
            textstr = 'Moving right: higher cost\nMoving up: taller walls\nColor: target failure probability'
            ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # Save
        output_file = os.path.join(output_dir, 'Figure_3_Pareto_Frontier.eps')
        plt.savefig(output_file, format='eps', dpi=self.dpi, bbox_inches='tight')
        print(f"✓ Saved: {output_file}")

        plt.close()

    def plot_multiscale_analysis(self, results_dict, output_dir='../paper/figures'):
        """
        Create 2-panel figure showing scale effects

        Panel A: Design characteristics vs. coastal length
        Panel B: Cost efficiency at different scales
        """
        if not HAS_MATPLOTLIB:
            return

        os.makedirs(output_dir, exist_ok=True)

        fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(12, 5))

        ms_data = results_dict.get('multiscale_analysis', {})

        if ms_data:
            scales = sorted([int(k.split('_')[1]) for k in ms_data.keys()])
            mean_heights = [ms_data[f'scale_{s}km']['mean_height_ft'] for s in scales]
            height_stds = [ms_data[f'scale_{s}km']['height_std_ft'] for s in scales]
            cost_per_km = [ms_data[f'scale_{s}km']['cost_per_km']/1e3 for s in scales]

            # Panel A: Heights and variation
            ax_a.errorbar(scales, mean_heights, yerr=height_stds, fmt='o-',
                         color='#2E86AB', ecolor='#A23B72', elinewidth=2,
                         markersize=10, linewidth=2.5, capsize=5, label='Mean height ± std dev')

            ax_a.set_xlabel('Coastal Segment Length (km)', fontsize=11, fontweight='bold')
            ax_a.set_ylabel('Design Height (ft)', fontsize=11, fontweight='bold')
            ax_a.set_title('(A) Design Heights at Different Coastal Scales', fontweight='bold', fontsize=11)
            ax_a.grid(True, alpha=0.3)
            ax_a.legend(fontsize=10)
            ax_a.set_xscale('log')

            # Panel B: Cost efficiency
            ax_b.plot(scales, cost_per_km, 'D-', color='#F18F01', linewidth=2.5,
                     markersize=10, label='Cost per km')
            ax_b.fill_between(scales, cost_per_km, alpha=0.3, color='#F18F01')

            ax_b.set_xlabel('Coastal Segment Length (km)', fontsize=11, fontweight='bold')
            ax_b.set_ylabel('Cost per km ($1000s)', fontsize=11, fontweight='bold')
            ax_b.set_title('(B) Cost Efficiency at Different Scales', fontweight='bold', fontsize=11)
            ax_b.grid(True, alpha=0.3)
            ax_b.legend(fontsize=10)
            ax_b.set_xscale('log')

        plt.suptitle('Multi-Scale Coastal Segment Analysis',
                     fontsize=13, fontweight='bold', y=1.00)

        # Save
        output_file = os.path.join(output_dir, 'Figure_4_Multiscale.eps')
        plt.savefig(output_file, format='eps', dpi=self.dpi, bbox_inches='tight')
        print(f"✓ Saved: {output_file}")

        plt.close()

    def create_all_figures(self, results_file, output_dir='../paper/figures'):
        """Generate all figures from analysis results"""

        print("\n" + "="*70)
        print("GENERATING PUBLICATION-QUALITY FIGURES")
        print("="*70)

        # Load results
        with open(results_file, 'r') as f:
            results = json.load(f)

        print(f"\nLoading results from: {results_file}")

        # Create figures
        print("\nGenerating figures:")
        self.plot_sensitivity_analysis(results, output_dir)
        self.plot_return_period_analysis(results, output_dir)
        self.plot_pareto_frontier(results, output_dir)
        self.plot_multiscale_analysis(results, output_dir)

        print("\n" + "="*70)
        print("All figures saved to:", output_dir)
        print("="*70)


if __name__ == '__main__':
    # Example usage
    import sys

    if len(sys.argv) > 1:
        results_file = sys.argv[1]
    else:
        # Find latest results file
        results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
        result_files = sorted(Path(results_dir).glob('sensitivity_analysis_*.json'))
        if result_files:
            results_file = str(result_files[-1])
        else:
            print("No results file found. Run sensitivity_and_scenario_analysis.py first.")
            sys.exit(1)

    visualizer = PublicationVisualizer(dpi=300)
    visualizer.create_all_figures(results_file)
