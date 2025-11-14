"""
Sea Level Rise Covariance Structure Modeling

Implements spatial covariance structures for sea level rise,
accounting for different physical drivers and spatial scales.

Key drivers:
1. Thermal expansion: High spatial correlation (1000+ km scale)
2. Glacier/ice sheet melt: Regional correlation (100-500 km)
3. Groundwater depletion: Local scale (10-100 km)
4. Glacial isostatic adjustment (GIA): Very long scale (global)
"""

import numpy as np
from scipy.spatial.distance import cdist
from scipy.special import gamma, kv
from typing import Dict, Tuple, List
from dataclasses import dataclass

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


@dataclass
class CovarianceParameters:
    """Parameters for spatial covariance models"""
    range: float          # Correlation length scale (km)
    sill: float          # Total variance
    nugget: float        # Measurement error variance
    name: str            # Model name


class SpatialCovarianceModels:
    """
    Various spatial covariance models for sea level rise.

    Standard geostatistical models used in climate science.
    """

    @staticmethod
    def exponential_covariance(distance: np.ndarray,
                              range_param: float = 500.0,
                              sill: float = 1.0) -> np.ndarray:
        """
        Exponential covariance model.

        C(d) = sill * exp(-d / range)

        Model: Fast decorrelation, suitable for local processes
        Use case: Groundwater, local subsidence
        """
        return sill * np.exp(-distance / range_param)

    @staticmethod
    def gaussian_covariance(distance: np.ndarray,
                           range_param: float = 500.0,
                           sill: float = 1.0) -> np.ndarray:
        """
        Gaussian (squared exponential) covariance model.

        C(d) = sill * exp(-(d/range)²)

        Model: Smooth, very smooth behavior near origin
        Use case: Thermal expansion (smooth spatial field)
        """
        return sill * np.exp(-(distance / range_param)**2)

    @staticmethod
    def matern_covariance(distance: np.ndarray,
                         range_param: float = 500.0,
                         sill: float = 1.0,
                         nu: float = 1.5) -> np.ndarray:
        """
        Matérn covariance model (generalization of exponential/gaussian).

        C(d) = sill * (2^(1-ν)/Γ(ν)) * (d/range)^ν * K_ν(d/range)

        Parameters:
        - ν=0.5: Exponential model
        - ν=∞: Gaussian model
        - ν=1.5: Whittle-Matérn (common in climate)

        Model: Flexible, realistic for many geophysical processes
        Use case: Ice sheet melting (intermediate smoothness)
        """
        # Avoid division by zero
        distance = np.maximum(distance, 1e-10)

        # Matérn formula
        scaled_dist = distance / range_param
        coef = (2**(1-nu)) / (gamma(nu))

        # Bessel function K_nu
        cov = sill * coef * (scaled_dist**nu) * kv(nu, scaled_dist)

        return cov

    @staticmethod
    def power_law_covariance(distance: np.ndarray,
                            range_param: float = 500.0,
                            sill: float = 1.0,
                            power: float = 1.0) -> np.ndarray:
        """
        Power-law covariance model (for long-range dependence).

        C(d) = sill * (1 + (d/range)^power)^(-1)

        Model: Power-law decay, long-range dependence
        Use case: Large-scale climate modes (GIA, ocean circulation)
        """
        return sill / (1 + (distance / range_param)**power)

    @staticmethod
    def composite_covariance(distance: np.ndarray,
                            weights: Dict[str, float],
                            params: Dict[str, CovarianceParameters]) -> np.ndarray:
        """
        Composite covariance from multiple drivers.

        C_total(d) = w_thermal * C_thermal(d) + w_glacier * C_glacier(d) + w_local * C_local(d)

        This is a LINEAR COMBINATION of different processes,
        each with its own spatial scale and magnitude.

        Args:
            distance: Pairwise distances
            weights: Relative contributions (sum to 1.0)
            params: Parameters for each component

        Returns:
            Combined covariance matrix
        """
        if not np.isclose(sum(weights.values()), 1.0):
            raise ValueError("Weights must sum to 1.0")

        total_cov = np.zeros_like(distance, dtype=float)

        for process_name, weight in weights.items():
            p = params[process_name]

            if process_name == 'thermal_expansion':
                # High spatial correlation (smooth)
                cov = SpatialCovarianceModels.gaussian_covariance(
                    distance, p.range, p.sill
                )
            elif process_name == 'ice_sheet_melt':
                # Intermediate correlation (Matérn)
                cov = SpatialCovarianceModels.matern_covariance(
                    distance, p.range, p.sill, nu=1.5
                )
            elif process_name == 'groundwater_depletion':
                # Local correlation (exponential)
                cov = SpatialCovarianceModels.exponential_covariance(
                    distance, p.range, p.sill
                )
            elif process_name == 'gia':
                # Global smooth trend (power-law)
                cov = SpatialCovarianceModels.power_law_covariance(
                    distance, p.range, p.sill
                )
            else:
                cov = np.zeros_like(distance)

            total_cov += weight * cov

        return total_cov


class SeaLevelCovarianceGenerator:
    """
    Generates sea level rise fields with specified covariance structure.

    Supports multiple covariance regimes representing different
    physical interpretations of spatial correlation in SLR.
    """

    def __init__(self, locations: np.ndarray):
        """
        Initialize sea level covariance generator.

        Args:
            locations: Array of shape (n_sites, 2) with (lat, lon) or (x, y)
        """
        self.locations = locations
        self.n_sites = locations.shape[0]
        self.distances = self._compute_distances()

    def _compute_distances(self) -> np.ndarray:
        """Compute pairwise Euclidean distances in km"""
        # Assuming locations are in degrees, convert to km
        # 1 degree ≈ 111 km
        locations_km = self.locations * 111.0
        distances = cdist(locations_km, locations_km, metric='euclidean')
        return distances

    def create_covariance_matrix(self, regime: str = 'moderate') -> np.ndarray:
        """
        Create sea level covariance matrix for different regimes.

        Args:
            regime: 'weak', 'moderate', or 'strong' correlation

        Returns:
            Covariance matrix of shape (n_sites, n_sites)
        """
        if regime == 'weak':
            # Low correlation: Multiple local processes dominate
            weights = {
                'thermal_expansion': 0.3,
                'ice_sheet_melt': 0.2,
                'groundwater_depletion': 0.4,
                'gia': 0.1
            }
            params = {
                'thermal_expansion': CovarianceParameters(200, 0.3, 0.05, 'thermal'),
                'ice_sheet_melt': CovarianceParameters(150, 0.2, 0.05, 'ice'),
                'groundwater_depletion': CovarianceParameters(50, 0.4, 0.1, 'groundwater'),
                'gia': CovarianceParameters(1000, 0.1, 0.05, 'gia')
            }

        elif regime == 'moderate':
            # Balanced: Mix of scales
            weights = {
                'thermal_expansion': 0.4,
                'ice_sheet_melt': 0.35,
                'groundwater_depletion': 0.15,
                'gia': 0.1
            }
            params = {
                'thermal_expansion': CovarianceParameters(400, 0.4, 0.05, 'thermal'),
                'ice_sheet_melt': CovarianceParameters(300, 0.35, 0.05, 'ice'),
                'groundwater_depletion': CovarianceParameters(80, 0.15, 0.05, 'groundwater'),
                'gia': CovarianceParameters(1500, 0.1, 0.05, 'gia')
            }

        elif regime == 'strong':
            # High correlation: Large-scale processes dominate
            weights = {
                'thermal_expansion': 0.55,
                'ice_sheet_melt': 0.3,
                'groundwater_depletion': 0.05,
                'gia': 0.1
            }
            params = {
                'thermal_expansion': CovarianceParameters(600, 0.55, 0.05, 'thermal'),
                'ice_sheet_melt': CovarianceParameters(500, 0.3, 0.05, 'ice'),
                'groundwater_depletion': CovarianceParameters(100, 0.05, 0.05, 'groundwater'),
                'gia': CovarianceParameters(2000, 0.1, 0.05, 'gia')
            }

        else:
            raise ValueError(f"Unknown regime: {regime}")

        # Generate composite covariance
        cov_matrix = SpatialCovarianceModels.composite_covariance(
            self.distances, weights, params
        )

        # Add nugget effect (measurement error)
        nugget = 0.05
        np.fill_diagonal(cov_matrix, np.diag(cov_matrix) + nugget)

        return cov_matrix, weights, params

    def generate_slr_field(self, regime: str = 'moderate',
                          mean_trend: float = 3.5,
                          n_time_steps: int = 30) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate spatially correlated sea level rise time series.

        Args:
            regime: Covariance structure ('weak', 'moderate', 'strong')
            mean_trend: Mean SLR rate (mm/year)
            n_time_steps: Number of years (default 30-year record)

        Returns:
            (time_series, spatial_means)
            - time_series: Shape (n_sites, n_time_steps)
            - spatial_means: Shape (n_time_steps,) - average SLR over all sites
        """
        cov_matrix, weights, params = self.create_covariance_matrix(regime)

        # Generate multivariate normal samples
        try:
            L = np.linalg.cholesky(cov_matrix)
        except np.linalg.LinAlgError:
            # Add regularization if not positive definite
            cov_matrix += np.eye(self.n_sites) * 1e-6
            L = np.linalg.cholesky(cov_matrix)

        # Generate spatial anomalies for each year
        spatial_anomalies = L @ np.random.randn(self.n_sites, n_time_steps)

        # Add temporal trend
        times = np.arange(n_time_steps)
        temporal_trend = mean_trend * times[np.newaxis, :]  # Shape: (1, n_time_steps)

        # Combine trend and anomalies
        slr_field = temporal_trend + spatial_anomalies

        # Spatial mean
        spatial_mean = np.mean(slr_field, axis=0)

        return slr_field, spatial_mean, weights

    def compute_spatial_correlation(self, regime: str = 'moderate') -> float:
        """
        Compute average spatial correlation coefficient for regime.

        Useful summary statistic for comparing regimes.
        """
        cov_matrix, _, _ = self.create_covariance_matrix(regime)

        # Get off-diagonal elements (exclude diagonal)
        n = cov_matrix.shape[0]
        mask = ~np.eye(n, dtype=bool)

        # Normalize to correlation (assuming variance 1)
        correlations = cov_matrix[mask]
        mean_correlation = np.mean(correlations)

        return mean_correlation


class SeaLevelCovarianceComparison:
    """
    Comprehensive comparison of sea level covariance regimes
    and their impact on optimal seawall design.
    """

    def __init__(self, n_sites: int = 100):
        """
        Initialize comparison framework.

        Args:
            n_sites: Number of coastal locations
        """
        self.n_sites = n_sites
        # Create synthetic coastal locations (along 50 km coast)
        self.locations = np.column_stack([
            np.linspace(0, 50, n_sites),  # x-coordinates (km)
            np.zeros(n_sites)              # y-coordinates (on coast)
        ]) / 111.0  # Convert to degrees

        self.generator = SeaLevelCovarianceGenerator(self.locations)

    def analyze_regime(self, regime: str, n_years: int = 30) -> Dict:
        """
        Analyze single covariance regime.

        Returns:
            Dictionary with statistics
        """
        # Generate SLR field
        slr_field, spatial_mean, weights = self.generator.generate_slr_field(
            regime=regime, n_time_steps=n_years
        )

        # Compute statistics
        spatial_std = np.std(slr_field, axis=1)  # Std at each location
        mean_spatial_variation = np.mean(np.std(slr_field, axis=0))  # Variation over time

        # Correlation
        correlation = self.generator.compute_spatial_correlation(regime)

        # Range of SLR across sites (at end of period)
        final_slr = slr_field[:, -1]
        slr_range = final_slr.max() - final_slr.min()

        return {
            'regime': regime,
            'mean_slr': np.mean(spatial_mean[-1]),
            'slr_spatial_variation': slr_range,
            'spatial_correlation': correlation,
            'mean_spatial_std': np.mean(spatial_std),
            'weights': weights,
            'slr_field': slr_field,
            'spatial_mean': spatial_mean
        }

    def compare_all_regimes(self) -> Dict:
        """
        Compare all three covariance regimes.
        """
        results = {}

        for regime in ['weak', 'moderate', 'strong']:
            print(f"Analyzing {regime} covariance regime...")
            results[regime] = self.analyze_regime(regime)

        return results

    def impact_on_seawall_design(self, results: Dict) -> Dict:
        """
        Analyze how covariance regimes affect optimal seawall heights.

        Key insight: Strong spatial correlation → More uniform heights
                     Weak spatial correlation → More differentiated heights
        """
        impacts = {}

        for regime, data in results.items():
            slr_field = data['slr_field']
            final_slr = slr_field[:, -1]

            # Optimal seawall heights would roughly match the final SLR
            # (assuming other factors equal)
            heights = final_slr + 15  # Add base height

            impacts[regime] = {
                'mean_height': np.mean(heights),
                'height_std': np.std(heights),
                'height_range': (heights.min(), heights.max()),
                'height_differentiation': heights.max() - heights.min(),
                'uniform_height_equiv': np.mean(heights),
                'cost_difference_if_uniform': np.sum((heights - np.mean(heights))**2) * 0.1
            }

        return impacts


class SeaLevelUncertaintyAnalysis:
    """
    Quantify uncertainty in sea level covariance structure
    and propagate to seawall design decisions.
    """

    def __init__(self, n_sites: int = 100):
        self.n_sites = n_sites
        self.locations = np.column_stack([
            np.linspace(0, 50, n_sites),
            np.zeros(n_sites)
        ]) / 111.0

    def uncertainty_in_correlation_structure(self) -> Dict:
        """
        Estimate uncertainty in inferred covariance regime
        from limited observations (e.g., 30-year tide gauge records).

        Bootstrap approach: Resample from observations, recompute correlation
        """
        generator = SeaLevelCovarianceGenerator(self.locations)

        n_bootstrap = 100
        regimes = ['weak', 'moderate', 'strong']
        correlation_samples = {regime: [] for regime in regimes}

        for regime in regimes:
            for _ in range(n_bootstrap):
                # Generate synthetic "observed" field
                slr_field, _, _ = generator.generate_slr_field(
                    regime=regime, n_time_steps=30
                )

                # Compute empirical correlation
                correlation_matrix = np.corrcoef(slr_field)
                mask = ~np.eye(self.n_sites, dtype=bool)
                mean_corr = np.mean(np.abs(correlation_matrix[mask]))

                correlation_samples[regime].append(mean_corr)

        # Summarize uncertainty
        results = {}
        for regime in regimes:
            samples = np.array(correlation_samples[regime])
            results[regime] = {
                'mean': np.mean(samples),
                'std': np.std(samples),
                'ci_lower': np.percentile(samples, 2.5),
                'ci_upper': np.percentile(samples, 97.5)
            }

        return results

    def robust_design_under_covariance_uncertainty(self) -> Dict:
        """
        Design seawall system that is robust to uncertainty
        in underlying covariance structure.

        Conservative approach: Design for worst-case (weakest spatial correlation)
        """
        comparison = SeaLevelCovarianceComparison(self.n_sites)
        results = comparison.compare_all_regimes()
        impacts = comparison.impact_on_seawall_design(results)

        # Robust design: Take max over all regimes
        robust_design = {
            'mean_height': max([impacts[r]['mean_height'] for r in ['weak', 'moderate', 'strong']]),
            'height_range_max': max([impacts[r]['height_differentiation'] for r in ['weak', 'moderate', 'strong']]),
            'cost_premium': (
                max([impacts[r]['cost_difference_if_uniform'] for r in ['weak', 'moderate', 'strong']])
            )
        }

        return robust_design
