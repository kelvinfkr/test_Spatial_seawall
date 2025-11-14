"""
Data Generation Module for Spatial Seawall Experiments

Generates synthetic flood events, surge distributions, and coastal parameters
for testing the spatial seawall optimization algorithm.
"""

import numpy as np
from scipy.stats import norm, multivariate_normal
import json
from typing import Dict, Tuple, List


class FloodEventGenerator:
    """Generates realistic synthetic flood events with spatial correlation."""

    def __init__(self, num_sectors: int, num_events: int = 5000,
                 base_mean: float = 8.0, base_std: float = 2.5,
                 spatial_correlation: float = 0.7, seed: int = 42):
        """
        Initialize flood event generator.

        Args:
            num_sectors: Number of coastal sectors
            num_events: Number of flood events to generate
            base_mean: Base mean surge height (ft)
            base_std: Base standard deviation of surge height (ft)
            spatial_correlation: Spatial correlation between adjacent sectors
            seed: Random seed for reproducibility
        """
        self.num_sectors = num_sectors
        self.num_events = num_events
        self.base_mean = base_mean
        self.base_std = base_std
        self.spatial_correlation = spatial_correlation
        self.seed = seed

        np.random.seed(seed)

    def generate_surge_distribution_params(self) -> List[Dict[str, float]]:
        """
        Generate mean and std parameters for each sector.

        Creates spatial variation in surge heights (e.g., some sectors more
        exposed than others due to geography).

        Returns:
            List of dicts with 'mean' and 'std' for each sector
        """
        params = []
        middle_idx = self.num_sectors // 2

        # Create a peak in exposure at the middle sector
        for i in range(self.num_sectors):
            # Distance from middle
            distance = abs(i - middle_idx)

            # Increase exposure toward middle (coastal bay effect)
            exposure_factor = 1.0 + 0.5 * np.exp(-distance / 3.0)

            mean = self.base_mean * exposure_factor
            std = self.base_std * (0.8 + 0.4 * exposure_factor)

            params.append({
                'location_id': i,
                'mean': mean,
                'std': std
            })

        return params

    def generate_correlated_surges(self) -> np.ndarray:
        """
        Generate spatially correlated flood events.

        Uses multivariate normal with exponential decay correlation structure.

        Returns:
            Array of shape (num_events, num_sectors) with surge heights
        """
        # Build correlation matrix with exponential decay
        corr_matrix = np.zeros((self.num_sectors, self.num_sectors))
        for i in range(self.num_sectors):
            for j in range(self.num_sectors):
                distance = abs(i - j)
                corr_matrix[i, j] = self.spatial_correlation ** distance

        # Get distribution parameters
        params = self.generate_surge_distribution_params()
        means = np.array([p['mean'] for p in params])
        stds = np.array([p['std'] for p in params])

        # Build covariance matrix: Cov(i,j) = std_i * std_j * corr(i,j)
        cov_matrix = np.outer(stds, stds) * corr_matrix

        # Generate multivariate normal samples
        events = multivariate_normal.rvs(mean=means, cov=cov_matrix,
                                        size=self.num_events)

        # Ensure non-negative surge heights
        events = np.maximum(events, 0)

        return events, params

    def add_climate_change_trend(self, events: np.ndarray, trend_rate: float = 0.05) -> np.ndarray:
        """
        Add climate change trend to flood events.

        Models increasing sea levels due to climate change.

        Args:
            events: Original flood events
            trend_rate: Rate of increase (fraction of base mean per decade)

        Returns:
            Events with climate change trend
        """
        # Simulate 2-3 decades of climate change
        decades = 2.5
        increase = self.base_mean * trend_rate * decades

        # Apply trend (more to later events to simulate time series)
        time_indices = np.linspace(0, 1, self.num_events)
        trend = increase * time_indices

        events_with_trend = events + trend[:, np.newaxis]

        return events_with_trend

    def save_events(self, events: np.ndarray, params: List[Dict],
                    filename: str = 'flood_events.npz'):
        """Save generated events and parameters to file."""
        np.savez(filename, events=events, params=json.dumps(params))
        return filename


class CoastalGeometryGenerator:
    """Generates coastal geometry and construction cost parameters."""

    def __init__(self, num_sectors: int, total_length_miles: float = 5.0,
                 base_unit_cost: float = 100.0):
        """
        Initialize coastal geometry generator.

        Args:
            num_sectors: Number of coastal sectors
            total_length_miles: Total coastline length in miles
            base_unit_cost: Base unit construction cost ($/ft of height)
        """
        self.num_sectors = num_sectors
        self.sector_length = total_length_miles / num_sectors
        self.base_unit_cost = base_unit_cost

    def generate_cost_parameters(self) -> List[Dict[str, float]]:
        """
        Generate construction cost parameters for each sector.

        Accounts for varying site conditions, material availability, etc.

        Returns:
            List of dicts with cost function parameters for each sector
        """
        params = []
        middle_idx = self.num_sectors // 2

        for i in range(self.num_sectors):
            # Vary unit costs based on location
            distance_from_middle = abs(i - middle_idx)

            # Higher costs at ends due to access, lower in middle
            cost_factor = 1.0 + 0.3 * (distance_from_middle / middle_idx)

            # Linear cost coefficient ($/ft of height per sector)
            b = self.base_unit_cost * cost_factor

            # Quadratic cost coefficient (increasing marginal costs)
            c = self.base_unit_cost * 0.5 * cost_factor

            params.append({
                'location_id': i,
                'position_miles': i * self.sector_length,
                'a': 0,      # Fixed cost
                'b': b,      # Linear cost
                'c': c       # Quadratic cost (marginal increase)
            })

        return params

    def generate_damage_parameters(self) -> Dict[str, float]:
        """
        Generate damage function parameters.

        Returns:
            Dict with 'alpha' and 'beta' for damage function D(d) = alpha * d^beta
        """
        return {
            'alpha': 100.0,  # Damage scaling
            'beta': 1.5      # Damage exponent (convex damage function)
        }


class ExperimentConfig:
    """Manages experiment configuration."""

    def __init__(self, num_sectors: int = 20, num_events: int = 5000):
        self.num_sectors = num_sectors
        self.num_events = num_events

        # Generators
        self.flood_gen = FloodEventGenerator(num_sectors, num_events)
        self.geometry_gen = CoastalGeometryGenerator(num_sectors)

    def generate_full_experiment_data(self, seed: int = 42) -> Dict:
        """
        Generate complete experiment data including floods, costs, and damage.

        Returns:
            Dict with all necessary data for running optimization
        """
        np.random.seed(seed)

        # Generate flood events
        events, surge_params = self.flood_gen.generate_correlated_surges()
        events = self.flood_gen.add_climate_change_trend(events)

        # Generate cost parameters
        cost_params = self.geometry_gen.generate_cost_parameters()

        # Damage parameters
        damage_params = self.geometry_gen.generate_damage_parameters()

        return {
            'flood_events': events,
            'surge_parameters': surge_params,
            'cost_parameters': cost_params,
            'damage_parameters': damage_params,
            'config': {
                'num_sectors': self.num_sectors,
                'num_events': self.num_events,
                'total_coastline_miles': 5.0,
                'sector_length_miles': 5.0 / self.num_sectors
            }
        }

    def save_experiment_data(self, data: Dict, prefix: str = 'experiment') -> Dict:
        """
        Save experiment data to files.

        Args:
            data: Experiment data dict
            prefix: Prefix for output files

        Returns:
            Dict with filenames
        """
        filenames = {}

        # Save flood events
        flood_file = f'{prefix}_floods.npz'
        np.savez(flood_file, events=data['flood_events'])
        filenames['floods'] = flood_file

        # Save parameters as JSON
        params_file = f'{prefix}_params.json'
        params_data = {
            'surge_parameters': data['surge_parameters'],
            'cost_parameters': data['cost_parameters'],
            'damage_parameters': data['damage_parameters'],
            'config': data['config']
        }
        with open(params_file, 'w') as f:
            json.dump(params_data, f, indent=2)
        filenames['params'] = params_file

        return filenames


def generate_baseline_experiment(output_dir: str = '../data') -> Dict:
    """
    Generate baseline experiment data for the paper.

    Args:
        output_dir: Directory to save data

    Returns:
        Dict with paths to generated files
    """
    print("Generating baseline experiment data...")
    print(f"  Sectors: 20")
    print(f"  Flood events: 5000")
    print(f"  Coastline: 5 miles")

    config = ExperimentConfig(num_sectors=20, num_events=5000)
    data = config.generate_full_experiment_data(seed=42)

    filenames = config.save_experiment_data(data, prefix=f'{output_dir}/baseline')

    print(f"  ✓ Saved to {output_dir}/baseline_*")

    return filenames, data
