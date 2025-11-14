"""
Spatial Seawall Design Optimization Model

Implementation of the spatial seawall optimization theory from the paper:
"Spatial Seawall Design Theory" by Kairui Feng

This module contains the core mathematical models and algorithms for optimizing
spatial seawall heights along a coastline.
"""

import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize_scalar
from typing import Tuple, Dict, List


class SurgeDistribution:
    """
    Represents the spatial surge height distribution for a coastal segment.

    Attributes:
        mean: Mean sea level surge height
        std: Standard deviation of surge height
        location_id: Identifier for the location
    """

    def __init__(self, mean: float, std: float, location_id: int):
        self.mean = mean
        self.std = std
        self.location_id = location_id

    def pdf(self, s: float) -> float:
        """Probability density function of surge height"""
        return norm.pdf(s, self.mean, self.std)

    def cdf(self, s: float) -> float:
        """Cumulative distribution function"""
        return norm.cdf(s, self.mean, self.std)

    def survival(self, s: float) -> float:
        """Survival function P(S > s)"""
        return 1 - self.cdf(s)


class CostFunction:
    """
    Construction cost function for seawall.

    Assumes quadratic cost function: C(h) = a + b*h + c*h^2
    """

    def __init__(self, a: float = 0, b: float = 100, c: float = 50):
        """
        Initialize cost function parameters.

        Args:
            a: Fixed cost (intercept)
            b: Linear cost coefficient (unit cost per foot)
            c: Quadratic cost coefficient (increasing marginal cost)
        """
        self.a = a
        self.b = b
        self.c = c

    def cost(self, h: float) -> float:
        """Total construction cost for seawall height h"""
        return self.a + self.b * h + self.c * h ** 2

    def marginal_cost(self, h: float) -> float:
        """Marginal cost (derivative) of seawall height"""
        return self.b + 2 * self.c * h

    def efficiency(self, h: float, marginal_damage: float) -> float:
        """
        Cost efficiency: marginal expected damage prevented / marginal construction cost

        Args:
            h: Seawall height
            marginal_damage: Marginal expected damage prevented

        Returns:
            Cost efficiency ratio
        """
        mc = self.marginal_cost(h)
        if mc <= 0:
            return 0
        return marginal_damage / mc


class DamageFunction:
    """
    Damage function for inundation.

    Assumes damage increases with inundation depth: D(s, h) = 0 if s <= h,
    otherwise D(s, h) = alpha * (s - h)^beta
    """

    def __init__(self, alpha: float = 100, beta: float = 1.5):
        """
        Initialize damage function parameters.

        Args:
            alpha: Damage scaling coefficient
            beta: Damage exponent (determines convexity)
        """
        self.alpha = alpha
        self.beta = beta

    def damage(self, surge: float, height: float) -> float:
        """
        Calculate damage for given surge and seawall height.

        Args:
            surge: Surge water level
            height: Seawall height

        Returns:
            Damage amount
        """
        if surge <= height:
            return 0
        inundation_depth = surge - height
        return self.alpha * (inundation_depth ** self.beta)

    def marginal_damage(self, height: float, distribution: SurgeDistribution) -> float:
        """
        Calculate marginal expected damage prevented by increasing height.

        Marginal damage ≈ E[D'(h)] = damage(h) * pdf(h)

        Args:
            height: Current seawall height
            distribution: Surge height distribution

        Returns:
            Marginal expected damage
        """
        # Damage at the threshold
        damage_at_h = self.alpha * (self.beta) * (1.0 ** (self.beta - 1))
        # Probability of exceedance
        prob_exceed = distribution.survival(height)

        return damage_at_h * prob_exceed


class PairwiseRelativeHeight:
    """
    Implements the pairwise comparison model from the paper.

    Calculates the optimal relative height between two adjacent seawall sectors
    based on Equation 15: h_i = h_j + G_ij^(-1)(CE_i(h_i) / CE_j(h_j))
    """

    def __init__(self,
                 surge_i: SurgeDistribution,
                 surge_j: SurgeDistribution,
                 cost_i: CostFunction,
                 cost_j: CostFunction,
                 damage_func: DamageFunction,
                 correlation_coef: float = 0.7):
        """
        Initialize pairwise model for two locations i and j.

        Args:
            surge_i: Surge distribution at location i
            surge_j: Surge distribution at location j
            cost_i: Cost function at location i
            cost_j: Cost function at location j
            damage_func: Damage function
            correlation_coef: Spatial correlation coefficient between locations
        """
        self.surge_i = surge_i
        self.surge_j = surge_j
        self.cost_i = cost_i
        self.cost_j = cost_j
        self.damage = damage_func
        self.rho = correlation_coef  # Correlation coefficient

    def relative_risk_measure(self, delta_h: float, h_i: float, h_j: float) -> float:
        """
        Calculate G_ij(Δh) from Equation 13.

        Represents the relative risk measure between two locations.

        Args:
            delta_h: Height difference (h_i - h_j)
            h_i: Seawall height at location i
            h_j: Seawall height at location j

        Returns:
            Risk measure value
        """
        # Spatial correlation creates conditional distributions
        # s_i | s_j ~ N(s_j + δ(s_j), σ_i^2)
        # We use simplified model: δ ≈ ρ * (σ_i/σ_j) * (s - mean)

        std_ratio = self.surge_i.std / self.surge_j.std

        # Conditional probabilities with correlation
        # P(s_j < h_j | s_i = h_i)
        delta_mean_i = self.rho * std_ratio * (h_i - self.surge_i.mean)
        delta_mean_j = self.rho * std_ratio * (h_j - self.surge_j.mean)

        term1 = norm.cdf((h_j - self.surge_j.mean - delta_mean_i) / self.surge_j.std)
        term2 = norm.cdf((h_i - self.surge_i.mean - delta_mean_j) / self.surge_i.std)

        if term2 <= 0:
            return 1e10

        return term1 / term2

    def cost_efficiency_ratio(self, h_i: float, h_j: float) -> float:
        """
        Calculate CE_i(h_i) / CE_j(h_j) from Equation 14.

        Args:
            h_i: Seawall height at location i
            h_j: Seawall height at location j

        Returns:
            Cost efficiency ratio
        """
        marg_damage_i = self.damage.marginal_damage(h_i, self.surge_i)
        marg_damage_j = self.damage.marginal_damage(h_j, self.surge_j)

        ce_i = self.cost_i.efficiency(h_i, marg_damage_i)
        ce_j = self.cost_j.efficiency(h_j, marg_damage_j)

        if ce_j <= 0:
            return 1

        return ce_i / ce_j

    def optimal_height_difference(self, h_j: float) -> float:
        """
        Compute h_i = h_j + G_ij^(-1)(CE_i/CE_j) from Equation 15.

        Args:
            h_j: Seawall height at location j (reference point)

        Returns:
            Optimal seawall height at location i
        """
        def equation_15(h_i):
            # We need to find h_i such that:
            # G_ij(h_i - h_j) = CE_i(h_i) / CE_j(h_j)

            delta_h = h_i - h_j
            if delta_h < -20 or delta_h > 20:  # Reasonable bounds
                return 1e10

            lhs = self.relative_risk_measure(delta_h, h_i, h_j)
            rhs = self.cost_efficiency_ratio(h_i, h_j)

            return (lhs - rhs) ** 2

        # Search for optimal height near h_j
        result = minimize_scalar(equation_15, bounds=(h_j - 5, h_j + 15), method='bounded')

        return result.x


class SpatialSeawallOptimizer:
    """
    Implements the numerical scheme for solving the spatial seawall problem.

    Algorithm from Section 4 of the paper:
    1. Start with initial height at middle sector
    2. Use pairwise relationships to compute heights for adjacent sectors
    3. Enforce gradient constraints (max 10ft per 0.1 mile)
    4. Evaluate total expected cost
    5. Iterate to find optimal initial height
    """

    def __init__(self,
                 num_sectors: int,
                 surge_distributions: List[SurgeDistribution],
                 cost_functions: List[CostFunction],
                 damage_func: DamageFunction,
                 max_gradient: float = 0.1):
        """
        Initialize spatial seawall optimizer.

        Args:
            num_sectors: Number of coastal sectors
            surge_distributions: List of surge distributions for each sector
            cost_functions: List of cost functions for each sector
            damage_func: Damage function
            max_gradient: Maximum allowed height change per sector (ft/0.1mile)
        """
        self.num_sectors = num_sectors
        self.surges = surge_distributions
        self.costs = cost_functions
        self.damage = damage_func
        self.max_gradient = max_gradient

    def compute_spatial_heights(self, h_middle: float) -> np.ndarray:
        """
        Compute optimal seawall heights for all sectors given middle height.

        Args:
            h_middle: Seawall height at middle sector (index num_sectors//2)

        Returns:
            Array of optimal heights for all sectors
        """
        heights = np.zeros(self.num_sectors)
        middle_idx = self.num_sectors // 2
        heights[middle_idx] = h_middle

        # Compute heights moving right from middle
        for i in range(middle_idx + 1, self.num_sectors):
            pairwise = PairwiseRelativeHeight(
                self.surges[i-1], self.surges[i],
                self.costs[i-1], self.costs[i],
                self.damage
            )
            h_i = pairwise.optimal_height_difference(heights[i-1])

            # Apply gradient constraint
            max_h = heights[i-1] + self.max_gradient
            min_h = heights[i-1] - self.max_gradient
            h_i = np.clip(h_i, min_h, max_h)

            heights[i] = h_i

        # Compute heights moving left from middle
        for i in range(middle_idx - 1, -1, -1):
            pairwise = PairwiseRelativeHeight(
                self.surges[i], self.surges[i+1],
                self.costs[i], self.costs[i+1],
                self.damage
            )
            h_i = pairwise.optimal_height_difference(heights[i+1])

            # Apply gradient constraint
            max_h = heights[i+1] + self.max_gradient
            min_h = heights[i+1] - self.max_gradient
            h_i = np.clip(h_i, min_h, max_h)

            heights[i] = h_i

        return heights

    def evaluate_total_cost(self, heights: np.ndarray, flood_events: np.ndarray) -> float:
        """
        Evaluate total expected cost for given seawall heights and flood events.

        Args:
            heights: Array of seawall heights for each sector
            flood_events: Array of shape (num_events, num_sectors) with surge heights

        Returns:
            Total expected cost (construction + expected damage)
        """
        num_events = flood_events.shape[0]

        # Construction cost
        construction_cost = sum(self.costs[i].cost(heights[i]) for i in range(self.num_sectors))

        # Expected damage
        total_damage = 0
        for event_idx in range(num_events):
            surges = flood_events[event_idx, :]
            # Damage dominated by maximum surge (assumption from paper)
            max_surge = np.max(surges)
            max_idx = np.argmax(surges)
            damage = self.damage.damage(max_surge, heights[max_idx])
            total_damage += damage

        expected_damage = total_damage / num_events

        return construction_cost + expected_damage

    def find_optimal_middle_height(self, flood_events: np.ndarray,
                                   h_min: float = 15, h_max: float = 22,
                                   step: float = 1.0) -> Tuple[float, float, np.ndarray]:
        """
        Search for optimal middle seawall height.

        Args:
            flood_events: Array of flood events
            h_min: Minimum height to test
            h_max: Maximum height to test
            step: Height increment

        Returns:
            Tuple of (optimal_middle_height, minimum_cost, optimal_heights)
        """
        heights_to_test = np.arange(h_min, h_max + step, step)
        costs = []
        all_heights = []

        for h_mid in heights_to_test:
            spatial_heights = self.compute_spatial_heights(h_mid)
            total_cost = self.evaluate_total_cost(spatial_heights, flood_events)
            costs.append(total_cost)
            all_heights.append(spatial_heights)

        optimal_idx = np.argmin(costs)
        optimal_h_mid = heights_to_test[optimal_idx]
        min_cost = costs[optimal_idx]
        optimal_heights = all_heights[optimal_idx]

        return optimal_h_mid, min_cost, optimal_heights
