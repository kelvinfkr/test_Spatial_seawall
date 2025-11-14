"""
Real Data Processing Module for Spatial Seawall Optimization

This module provides utilities to fetch, process, and convert real-world
hurricane and storm surge data into formats suitable for the seawall
optimization model.

Data Sources:
- NOAA CO-OPS Tide Gauge API
- NOAA HURDAT2 Hurricane Database
- Extreme Value Analysis (GEV fitting)

Author: Generated for Spatial Seawall Project
Date: November 2024
"""

import numpy as np
import pandas as pd
import json
from scipy import stats
from typing import Dict, List, Tuple, Optional
import warnings

# Optional imports (install as needed)
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    warnings.warn("requests not available. Install with: pip install requests")


class NOAATideGaugeAPI:
    """
    Fetch historical water level data from NOAA CO-OPS API.

    API Documentation: https://api.tidesandcurrents.noaa.gov/api/prod/
    """

    BASE_URL = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"

    # Example Gulf Coast stations (Louisiana to Texas)
    GULF_COAST_STATIONS = {
        '8761724': {'name': 'Grand Isle, LA', 'lat': 29.263, 'lon': -89.957},
        '8764227': {'name': 'Shell Beach, LA', 'lat': 29.868, 'lon': -89.673},
        '8760922': {'name': 'Pilots Station East, LA', 'lat': 28.932, 'lon': -89.407},
        '8768094': {'name': 'Calcasieu Pass, LA', 'lat': 29.768, 'lon': -93.343},
        '8771450': {'name': 'Galveston Pier 21, TX', 'lat': 29.310, 'lon': -94.793},
        '8770570': {'name': 'Sabine Pass North, TX', 'lat': 29.728, 'lon': -93.870},
    }

    def __init__(self, station_id: str):
        """
        Initialize API for a specific tide gauge station.

        Args:
            station_id: NOAA CO-OPS station ID (e.g., '8761724' for Grand Isle)
        """
        if not REQUESTS_AVAILABLE:
            raise ImportError("requests library required. Install: pip install requests")

        self.station_id = station_id
        self.station_info = self.GULF_COAST_STATIONS.get(station_id, {})

    def fetch_hourly_water_levels(self,
                                   begin_date: str,
                                   end_date: str,
                                   datum: str = 'MLLW',
                                   units: str = 'english') -> pd.DataFrame:
        """
        Fetch hourly water level data.

        Args:
            begin_date: Start date in format YYYYMMDD
            end_date: End date in format YYYYMMDD
            datum: Vertical datum (MLLW, MSL, NAVD, etc.)
            units: 'english' (feet) or 'metric' (meters)

        Returns:
            DataFrame with columns: timestamp, water_level, flags
        """
        params = {
            'product': 'hourly_height',
            'station': self.station_id,
            'begin_date': begin_date,
            'end_date': end_date,
            'datum': datum,
            'units': units,
            'time_zone': 'gmt',
            'format': 'json',
            'application': 'spatial_seawall_research'
        }

        print(f"Fetching data from station {self.station_id} "
              f"({self.station_info.get('name', 'Unknown')})...")
        print(f"  Date range: {begin_date} to {end_date}")

        try:
            response = requests.get(self.BASE_URL, params=params, timeout=60)
            response.raise_for_status()
            data = response.json()

            if 'data' not in data:
                raise ValueError(f"No data returned. Error: {data.get('error', 'Unknown')}")

            # Parse into DataFrame
            df = pd.DataFrame(data['data'])
            df['timestamp'] = pd.to_datetime(df['t'])
            df['water_level'] = pd.to_numeric(df['v'], errors='coerce')

            # Handle flags and quality indicators
            df['flags'] = df.get('f', '')
            df['quality'] = df.get('q', '')

            print(f"  Retrieved {len(df)} hourly observations")

            return df[['timestamp', 'water_level', 'flags', 'quality']]

        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Failed to fetch data: {e}")

    def fetch_high_low_water(self,
                             begin_date: str,
                             end_date: str,
                             datum: str = 'MLLW',
                             units: str = 'english') -> pd.DataFrame:
        """
        Fetch high and low water levels (tidal extremes).

        This can be useful for identifying storm surge peaks.
        """
        params = {
            'product': 'high_low',
            'station': self.station_id,
            'begin_date': begin_date,
            'end_date': end_date,
            'datum': datum,
            'units': units,
            'time_zone': 'gmt',
            'format': 'json',
            'application': 'spatial_seawall_research'
        }

        response = requests.get(self.BASE_URL, params=params, timeout=60)
        response.raise_for_status()
        data = response.json()

        if 'data' not in data:
            raise ValueError(f"No data returned: {data}")

        df = pd.DataFrame(data['data'])
        df['timestamp'] = pd.to_datetime(df['t'])
        df['water_level'] = pd.to_numeric(df['v'], errors='coerce')
        df['type'] = df['ty']  # 'H' for high, 'L' for low

        return df[['timestamp', 'water_level', 'type']]


class ExtremeValueAnalyzer:
    """
    Perform extreme value analysis on water level data.

    Fits Generalized Extreme Value (GEV) distribution to annual maxima
    and calculates return periods for coastal flood risk assessment.
    """

    def __init__(self, water_levels: pd.Series, timestamps: pd.Series):
        """
        Initialize with time series of water levels.

        Args:
            water_levels: Water level measurements (e.g., in feet)
            timestamps: Corresponding timestamps
        """
        self.df = pd.DataFrame({
            'timestamp': timestamps,
            'water_level': water_levels
        })
        self.df = self.df.dropna()
        self.df['year'] = self.df['timestamp'].dt.year

    def extract_annual_maxima(self) -> pd.Series:
        """
        Extract annual maximum water levels (Block Maxima method).

        Returns:
            Series of annual maximum water levels
        """
        annual_max = self.df.groupby('year')['water_level'].max()
        print(f"Extracted {len(annual_max)} annual maxima from {annual_max.index.min()} "
              f"to {annual_max.index.max()}")
        return annual_max

    def extract_peaks_over_threshold(self, threshold: float) -> pd.Series:
        """
        Extract peaks over threshold (POT method).

        Args:
            threshold: Threshold value for peak extraction

        Returns:
            Series of peak values exceeding threshold
        """
        peaks = self.df[self.df['water_level'] > threshold]['water_level']
        print(f"Extracted {len(peaks)} peaks over threshold {threshold:.2f}")
        return peaks

    def fit_gev(self, data: Optional[pd.Series] = None) -> Dict[str, float]:
        """
        Fit Generalized Extreme Value distribution to data.

        Args:
            data: Data to fit (if None, uses annual maxima)

        Returns:
            Dictionary with GEV parameters and fit diagnostics
        """
        if data is None:
            data = self.extract_annual_maxima()

        # Fit GEV using scipy
        # Note: scipy uses floc parameter. For extreme values, typically use maximum likelihood
        shape, loc, scale = stats.genextreme.fit(data)

        # Calculate goodness of fit metrics
        ks_statistic, ks_pvalue = stats.kstest(
            data,
            lambda x: stats.genextreme.cdf(x, shape, loc, scale)
        )

        print("\n=== GEV Fit Results ===")
        print(f"Shape parameter (ξ): {shape:.4f}")
        print(f"Location parameter (μ): {loc:.4f}")
        print(f"Scale parameter (σ): {scale:.4f}")
        print(f"KS test p-value: {ks_pvalue:.4f}")

        if shape < -0.5:
            print("⚠ Warning: Heavy negative tail (shape < -0.5). Check data quality.")
        elif shape > 0.5:
            print("⚠ Warning: Heavy positive tail (shape > 0.5). May indicate non-stationarity.")

        return {
            'shape': shape,
            'location': loc,
            'scale': scale,
            'n_observations': len(data),
            'ks_statistic': ks_statistic,
            'ks_pvalue': ks_pvalue,
            'data_min': float(data.min()),
            'data_max': float(data.max()),
            'data_mean': float(data.mean()),
            'data_std': float(data.std())
        }

    def calculate_return_levels(self,
                                 gev_params: Dict[str, float],
                                 return_periods: List[int] = [2, 5, 10, 25, 50, 100, 500]
                                 ) -> Dict[int, float]:
        """
        Calculate return levels for given return periods.

        Args:
            gev_params: GEV parameters from fit_gev()
            return_periods: List of return periods in years

        Returns:
            Dictionary mapping return period to return level
        """
        shape = gev_params['shape']
        loc = gev_params['location']
        scale = gev_params['scale']

        return_levels = {}
        print("\n=== Return Levels ===")

        for T in return_periods:
            # Exceedance probability: p = 1 - 1/T
            p = 1 - 1/T

            # Inverse CDF (quantile function)
            z_T = stats.genextreme.ppf(p, shape, loc, scale)
            return_levels[T] = float(z_T)

            print(f"{T:4d}-year: {z_T:6.2f} ft")

        return return_levels

    def generate_qq_plot_data(self, gev_params: Dict[str, float]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate data for Q-Q plot (for visualization).

        Returns:
            (theoretical_quantiles, empirical_quantiles)
        """
        data = self.extract_annual_maxima().values
        shape = gev_params['shape']
        loc = gev_params['location']
        scale = gev_params['scale']

        # Empirical quantiles
        sorted_data = np.sort(data)

        # Theoretical quantiles
        n = len(data)
        probabilities = np.arange(1, n + 1) / (n + 1)
        theoretical = stats.genextreme.ppf(probabilities, shape, loc, scale)

        return theoretical, sorted_data


class SpatialCorrelationEstimator:
    """
    Estimate spatial correlation structure between multiple locations.
    """

    def __init__(self, stations_data: Dict[str, pd.DataFrame]):
        """
        Initialize with data from multiple stations.

        Args:
            stations_data: Dict mapping station_id to DataFrame with annual maxima
        """
        self.stations_data = stations_data
        self.station_ids = list(stations_data.keys())

    def calculate_correlation_matrix(self) -> np.ndarray:
        """
        Calculate empirical correlation matrix between stations.

        Returns:
            Correlation matrix (n_stations x n_stations)
        """
        # Combine all station data
        df_all = pd.DataFrame({
            station_id: data['water_level']
            for station_id, data in self.stations_data.items()
        })

        corr_matrix = df_all.corr().values

        print("\n=== Spatial Correlation Matrix ===")
        print(corr_matrix)

        return corr_matrix

    def fit_exponential_decay(self,
                              distances: np.ndarray,
                              correlations: np.ndarray) -> Dict[str, float]:
        """
        Fit exponential decay model: ρ(d) = ρ₀^(d/d₀)

        Args:
            distances: Pairwise distances between stations (miles)
            correlations: Pairwise correlations

        Returns:
            Dictionary with decay parameters
        """
        # Linearize: log(ρ) = (d/d₀) * log(ρ₀)
        # Filter out negative/zero correlations
        valid_mask = correlations > 0

        if not np.any(valid_mask):
            raise ValueError("No positive correlations to fit")

        log_corr = np.log(correlations[valid_mask])
        dist = distances[valid_mask]

        # Linear regression
        slope, intercept = np.polyfit(dist, log_corr, 1)

        # Back-transform
        rho_0 = np.exp(intercept)
        d_0 = -1 / slope

        print("\n=== Exponential Decay Model ===")
        print(f"ρ(d) = {rho_0:.4f}^(d/{d_0:.2f})")
        print(f"ρ₀ (baseline correlation): {rho_0:.4f}")
        print(f"d₀ (decay length scale): {d_0:.2f} miles")

        return {
            'rho_0': float(rho_0),
            'd_0': float(d_0),
            'slope': float(slope),
            'intercept': float(intercept)
        }


class RealDataIntegrator:
    """
    Integrate real data into the seawall optimization model format.
    """

    def __init__(self, num_sectors: int = 20, coastline_length_miles: float = 5.0):
        """
        Initialize for a coastal segment.

        Args:
            num_sectors: Number of coastal sectors
            coastline_length_miles: Total length of coastline
        """
        self.num_sectors = num_sectors
        self.coastline_length = coastline_length_miles
        self.sector_length = coastline_length_miles / num_sectors

    def create_surge_parameters(self,
                                 gev_params_by_location: List[Dict],
                                 spatial_correlation: Dict) -> Dict:
        """
        Create surge distribution parameters in model format.

        Args:
            gev_params_by_location: List of GEV parameters for each sector
            spatial_correlation: Spatial correlation parameters

        Returns:
            Dictionary ready for seawall optimization model
        """
        surge_params = []

        for i, gev_params in enumerate(gev_params_by_location):
            surge_params.append({
                'location_id': i,
                'position_miles': i * self.sector_length,
                'distribution_type': 'GEV',
                'shape': gev_params['shape'],
                'location': gev_params['location'],
                'scale': gev_params['scale'],
                'mean': gev_params['data_mean'],
                'std': gev_params['data_std'],
                'n_years': gev_params['n_observations']
            })

        return {
            'surge_parameters': surge_params,
            'spatial_correlation': spatial_correlation,
            'units': 'feet',
            'datum': 'MLLW',
            'data_source': 'NOAA_tide_gauges'
        }

    def generate_correlated_gev_samples(self,
                                        surge_params: List[Dict],
                                        correlation_matrix: np.ndarray,
                                        n_samples: int = 5000,
                                        seed: int = 42) -> np.ndarray:
        """
        Generate spatially-correlated samples from GEV distributions.

        Uses Gaussian copula approach:
        1. Generate correlated normal samples
        2. Transform to uniform via normal CDF
        3. Transform to GEV via inverse GEV CDF

        Args:
            surge_params: GEV parameters for each location
            correlation_matrix: Spatial correlation structure
            n_samples: Number of synthetic events
            seed: Random seed

        Returns:
            Array of shape (n_samples, n_sectors) with surge heights
        """
        np.random.seed(seed)
        n_sectors = len(surge_params)

        # Step 1: Generate correlated normal samples
        mean = np.zeros(n_sectors)
        cov = correlation_matrix

        normal_samples = np.random.multivariate_normal(mean, cov, size=n_samples)

        # Step 2: Transform to uniform [0, 1] via normal CDF
        uniform_samples = stats.norm.cdf(normal_samples)

        # Step 3: Transform to GEV marginals
        gev_samples = np.zeros((n_samples, n_sectors))

        for i, params in enumerate(surge_params):
            shape = params['shape']
            loc = params['location']
            scale = params['scale']

            gev_samples[:, i] = stats.genextreme.ppf(
                uniform_samples[:, i],
                shape, loc, scale
            )

        # Ensure non-negative surge heights
        gev_samples = np.maximum(gev_samples, 0)

        print(f"\nGenerated {n_samples} correlated surge events for {n_sectors} sectors")
        print(f"  Mean surge height: {gev_samples.mean():.2f} ft")
        print(f"  Max surge height: {gev_samples.max():.2f} ft")
        print(f"  Std deviation: {gev_samples.std():.2f} ft")

        return gev_samples

    def save_to_model_format(self,
                            surge_events: np.ndarray,
                            surge_params: List[Dict],
                            cost_params: List[Dict],
                            damage_params: Dict,
                            output_dir: str = '../data') -> Dict[str, str]:
        """
        Save processed real data in the format expected by seawall_model.py.

        Args:
            surge_events: Generated flood events
            surge_params: Surge distribution parameters
            cost_params: Cost function parameters
            damage_params: Damage function parameters
            output_dir: Directory to save files

        Returns:
            Dictionary with paths to saved files
        """
        import os

        # Create output directory if needed
        os.makedirs(output_dir, exist_ok=True)

        # Save flood events
        floods_file = os.path.join(output_dir, 'real_data_floods.npz')
        np.savez(floods_file, events=surge_events)

        # Save parameters
        params_file = os.path.join(output_dir, 'real_data_params.json')
        params_data = {
            'surge_parameters': surge_params,
            'cost_parameters': cost_params,
            'damage_parameters': damage_params,
            'config': {
                'num_sectors': self.num_sectors,
                'num_events': surge_events.shape[0],
                'total_coastline_miles': self.coastline_length,
                'sector_length_miles': self.sector_length,
                'data_source': 'real_NOAA_data',
                'analysis_date': pd.Timestamp.now().isoformat()
            }
        }

        with open(params_file, 'w') as f:
            json.dump(params_data, f, indent=2)

        print(f"\n=== Saved Real Data Files ===")
        print(f"Floods: {floods_file}")
        print(f"Parameters: {params_file}")

        return {
            'floods': floods_file,
            'params': params_file
        }


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

def example_fetch_and_analyze():
    """
    Example workflow: fetch data, perform extreme value analysis, generate events.
    """
    print("=" * 70)
    print("EXAMPLE: Fetch Real Data and Perform Extreme Value Analysis")
    print("=" * 70)

    # Example: Grand Isle, LA tide gauge
    station_id = '8761724'

    # Initialize API
    api = NOAATideGaugeAPI(station_id)

    # Fetch 10 years of hourly data (2014-2023)
    try:
        df = api.fetch_hourly_water_levels(
            begin_date='20140101',
            end_date='20231231'
        )

        # Extreme value analysis
        analyzer = ExtremeValueAnalyzer(df['water_level'], df['timestamp'])

        # Fit GEV to annual maxima
        gev_params = analyzer.fit_gev()

        # Calculate return levels
        return_levels = analyzer.calculate_return_levels(gev_params)

        # Generate Q-Q plot data (for visualization)
        theoretical, empirical = analyzer.generate_qq_plot_data(gev_params)

        print("\n✓ Analysis complete!")
        print(f"\n100-year surge height: {return_levels[100]:.2f} ft")

    except Exception as e:
        print(f"\n⚠ Example failed (likely due to API access): {e}")
        print("\nTo run this example, ensure you have internet access and:")
        print("  pip install requests pandas numpy scipy")


def example_generate_synthetic_events():
    """
    Example: Generate synthetic correlated surge events from real distributions.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE: Generate Synthetic Events from Real Distributions")
    print("=" * 70)

    # Assume we have GEV parameters from multiple locations
    # (In practice, these would come from actual data analysis)

    n_sectors = 20
    integrator = RealDataIntegrator(num_sectors=n_sectors, coastline_length_miles=5.0)

    # Example GEV parameters (these should come from real analysis)
    gev_params_list = []
    for i in range(n_sectors):
        # Simulate spatial variation in exposure
        exposure_factor = 1.0 + 0.5 * np.exp(-(i - 10)**2 / 20)

        gev_params_list.append({
            'shape': -0.15,
            'location': 3.0 * exposure_factor,
            'scale': 1.5 * exposure_factor,
            'data_mean': 3.5 * exposure_factor,
            'data_std': 2.0 * exposure_factor,
            'n_observations': 30
        })

    # Spatial correlation matrix (exponential decay)
    rho_0 = 0.7
    correlation_matrix = np.zeros((n_sectors, n_sectors))
    for i in range(n_sectors):
        for j in range(n_sectors):
            distance = abs(i - j) * 0.25  # miles
            correlation_matrix[i, j] = rho_0 ** (distance / 1.0)

    # Generate correlated events
    surge_events = integrator.generate_correlated_gev_samples(
        gev_params_list,
        correlation_matrix,
        n_samples=5000,
        seed=42
    )

    print(f"\n✓ Generated {surge_events.shape[0]} events for {surge_events.shape[1]} sectors")
    print(f"\nEvent statistics:")
    print(f"  Mean across all events and sectors: {surge_events.mean():.2f} ft")
    print(f"  Maximum surge height observed: {surge_events.max():.2f} ft")
    print(f"  95th percentile: {np.percentile(surge_events, 95):.2f} ft")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("REAL DATA PROCESSOR FOR SPATIAL SEAWALL OPTIMIZATION")
    print("=" * 70)

    print("\nThis module provides tools to:")
    print("  1. Fetch historical water level data from NOAA")
    print("  2. Perform extreme value analysis (GEV fitting)")
    print("  3. Estimate spatial correlation")
    print("  4. Generate synthetic correlated surge events")
    print("  5. Save data in format for optimization model")

    print("\n" + "-" * 70)

    # Run examples (comment out if you don't have requests installed)
    example_fetch_and_analyze()
    example_generate_synthetic_events()

    print("\n" + "=" * 70)
    print("For detailed instructions, see: data/REAL_DATA_SOURCES.md")
    print("=" * 70 + "\n")
