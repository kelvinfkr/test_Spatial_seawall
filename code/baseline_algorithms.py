"""
Baseline Optimization Algorithms for Spatial Seawall Design

Implements multiple baseline algorithms for comparison:
1. Genetic Algorithm (GA)
2. Particle Swarm Optimization (PSO)
3. Uniform Design (constant height across all sectors)
4. Greedy Algorithm (simple heuristic)
"""

import numpy as np
from typing import Tuple, Dict, List, Callable
from dataclasses import dataclass
import json


@dataclass
class AlgorithmResult:
    """Container for algorithm results"""
    name: str
    heights: np.ndarray
    total_cost: float
    construction_cost: float
    expected_damage: float
    convergence_history: List[float]
    computation_time: float
    metadata: Dict


class GeneticAlgorithm:
    """
    Genetic Algorithm for spatial seawall optimization.

    Population-based evolutionary algorithm that evolves seawall height
    configurations over generations.
    """

    def __init__(self,
                 cost_function: Callable,
                 population_size: int = 50,
                 generations: int = 100,
                 mutation_rate: float = 0.1,
                 crossover_rate: float = 0.8):
        """
        Initialize Genetic Algorithm.

        Args:
            cost_function: Function that evaluates total cost for height vector
            population_size: Number of individuals in population
            generations: Number of generations to evolve
            mutation_rate: Probability of mutation per gene
            crossover_rate: Probability of crossover
        """
        self.cost_func = cost_function
        self.pop_size = population_size
        self.generations = generations
        self.mut_rate = mutation_rate
        self.cross_rate = crossover_rate
        self.convergence_history = []

    def initialize_population(self, num_sectors: int,
                             h_min: float = 10, h_max: float = 20) -> np.ndarray:
        """
        Initialize population with random heights.

        Args:
            num_sectors: Number of coastal sectors
            h_min: Minimum allowable height
            h_max: Maximum allowable height

        Returns:
            Population array of shape (population_size, num_sectors)
        """
        population = np.random.uniform(h_min, h_max,
                                      size=(self.pop_size, num_sectors))
        return population

    def evaluate_fitness(self, population: np.ndarray) -> np.ndarray:
        """
        Evaluate fitness (negative cost) for each individual.

        Args:
            population: Population array

        Returns:
            Fitness array (lower cost = higher fitness)
        """
        fitness = np.zeros(self.pop_size)
        for i in range(self.pop_size):
            fitness[i] = self.cost_func(population[i])
        return fitness

    def selection(self, population: np.ndarray,
                 fitness: np.ndarray) -> np.ndarray:
        """
        Tournament selection.

        Args:
            population: Current population
            fitness: Fitness values

        Returns:
            Selected parents
        """
        selected = []
        for _ in range(self.pop_size):
            # Tournament: pick 3 random individuals, return best
            tournament_idx = np.random.choice(self.pop_size, size=3, replace=False)
            winner_idx = tournament_idx[np.argmin(fitness[tournament_idx])]
            selected.append(population[winner_idx].copy())
        return np.array(selected)

    def crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Two-point crossover.

        Args:
            parent1: First parent
            parent2: Second parent

        Returns:
            Two offspring
        """
        if np.random.rand() > self.cross_rate:
            return parent1.copy(), parent2.copy()

        num_sectors = len(parent1)
        point1 = np.random.randint(0, num_sectors)
        point2 = np.random.randint(point1, num_sectors)

        child1 = parent1.copy()
        child2 = parent2.copy()

        child1[point1:point2] = parent2[point1:point2]
        child2[point1:point2] = parent1[point1:point2]

        return child1, child2

    def mutate(self, individual: np.ndarray,
              h_min: float = 10, h_max: float = 20) -> np.ndarray:
        """
        Gaussian mutation.

        Args:
            individual: Individual to mutate
            h_min: Minimum height
            h_max: Maximum height

        Returns:
            Mutated individual
        """
        mutated = individual.copy()
        for i in range(len(mutated)):
            if np.random.rand() < self.mut_rate:
                # Add Gaussian noise
                noise = np.random.normal(0, 0.5)
                mutated[i] = np.clip(mutated[i] + noise, h_min, h_max)
        return mutated

    def optimize(self, num_sectors: int,
                h_min: float = 10, h_max: float = 20) -> np.ndarray:
        """
        Run genetic algorithm optimization.

        Args:
            num_sectors: Number of coastal sectors
            h_min: Minimum height
            h_max: Maximum height

        Returns:
            Optimal height configuration
        """
        # Initialize population
        population = self.initialize_population(num_sectors, h_min, h_max)

        # Evolution loop
        for gen in range(self.generations):
            # Evaluate fitness
            fitness = self.evaluate_fitness(population)
            self.convergence_history.append(np.min(fitness))

            # Selection
            parents = self.selection(population, fitness)

            # Create offspring via crossover and mutation
            offspring = []
            for i in range(0, self.pop_size, 2):
                child1, child2 = self.crossover(parents[i], parents[i+1 if i+1 < self.pop_size else 0])
                child1 = self.mutate(child1, h_min, h_max)
                child2 = self.mutate(child2, h_min, h_max)
                offspring.extend([child1, child2])

            # Elitism: keep best individual
            best_idx = np.argmin(fitness)
            population = np.array(offspring[:self.pop_size])
            population[0] = parents[best_idx]

        # Return best solution
        fitness = self.evaluate_fitness(population)
        best_idx = np.argmin(fitness)
        return population[best_idx]


class ParticleSwarmOptimizer:
    """
    Particle Swarm Optimization for spatial seawall design.

    Population-based metaheuristic inspired by bird flocking behavior.
    """

    def __init__(self,
                 cost_function: Callable,
                 num_particles: int = 30,
                 iterations: int = 100,
                 w: float = 0.7,
                 c1: float = 1.5,
                 c2: float = 1.5):
        """
        Initialize PSO.

        Args:
            cost_function: Function that evaluates total cost
            num_particles: Number of particles
            iterations: Number of iterations
            w: Inertia weight
            c1: Cognitive parameter
            c2: Social parameter
        """
        self.cost_func = cost_function
        self.num_particles = num_particles
        self.iterations = iterations
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.convergence_history = []

    def optimize(self, num_sectors: int,
                h_min: float = 10, h_max: float = 20) -> np.ndarray:
        """
        Run PSO optimization.

        Args:
            num_sectors: Number of sectors
            h_min: Minimum height
            h_max: Maximum height

        Returns:
            Optimal height configuration
        """
        # Initialize particles and velocities
        positions = np.random.uniform(h_min, h_max,
                                     size=(self.num_particles, num_sectors))
        velocities = np.random.uniform(-1, 1,
                                      size=(self.num_particles, num_sectors))

        # Initialize personal and global best
        costs = np.array([self.cost_func(pos) for pos in positions])
        pbest_positions = positions.copy()
        pbest_costs = costs.copy()

        gbest_idx = np.argmin(pbest_costs)
        gbest_position = pbest_positions[gbest_idx].copy()
        gbest_cost = pbest_costs[gbest_idx]

        # Iteration loop
        for iteration in range(self.iterations):
            for i in range(self.num_particles):
                # Update velocity
                r1 = np.random.rand(num_sectors)
                r2 = np.random.rand(num_sectors)

                velocities[i] = (self.w * velocities[i] +
                               self.c1 * r1 * (pbest_positions[i] - positions[i]) +
                               self.c2 * r2 * (gbest_position - positions[i]))

                # Update position
                positions[i] += velocities[i]
                positions[i] = np.clip(positions[i], h_min, h_max)

                # Evaluate and update best
                cost = self.cost_func(positions[i])
                if cost < pbest_costs[i]:
                    pbest_costs[i] = cost
                    pbest_positions[i] = positions[i].copy()

                if cost < gbest_cost:
                    gbest_cost = cost
                    gbest_position = positions[i].copy()

            self.convergence_history.append(gbest_cost)

        return gbest_position


class UniformDesignBaseline:
    """Baseline: uniform seawall height across all sectors"""

    @staticmethod
    def compute_optimal_uniform_height(cost_func: Callable,
                                       num_sectors: int,
                                       h_min: float = 10,
                                       h_max: float = 20,
                                       step: float = 0.1) -> Tuple[float, float]:
        """
        Find optimal uniform height by grid search.

        Args:
            cost_func: Cost evaluation function
            num_sectors: Number of sectors
            h_min: Minimum height
            h_max: Maximum height
            step: Grid search step

        Returns:
            (optimal_height, minimum_cost)
        """
        heights_to_test = np.arange(h_min, h_max + step, step)
        costs = []

        for h in heights_to_test:
            heights = np.full(num_sectors, h)
            cost = cost_func(heights)
            costs.append(cost)

        optimal_idx = np.argmin(costs)
        optimal_h = heights_to_test[optimal_idx]
        min_cost = costs[optimal_idx]

        return optimal_h, min_cost


class GreedyAlgorithm:
    """
    Greedy heuristic: optimize each sector independently.

    Approximate algorithm that serves as a lower bound for comparison.
    """

    def __init__(self,
                 cost_func: Callable,
                 damage_func: Callable):
        """
        Initialize greedy algorithm.

        Args:
            cost_func: Function returning cost for sector heights
            damage_func: Function returning damage for sector heights
        """
        self.cost_func = cost_func
        self.damage_func = damage_func

    def optimize_sector(self, sector_idx: int, num_sectors: int,
                       h_min: float = 10, h_max: float = 20,
                       step: float = 0.5) -> float:
        """
        Optimize height for single sector (ignoring interactions).

        Args:
            sector_idx: Index of sector to optimize
            num_sectors: Total number of sectors
            h_min: Minimum height
            h_max: Maximum height
            step: Grid search step

        Returns:
            Optimal height for this sector
        """
        heights_to_test = np.arange(h_min, h_max + step, step)
        best_cost = float('inf')
        best_h = h_min

        for h in heights_to_test:
            # Evaluate cost and damage for this sector only
            sector_cost = self.cost_func(sector_idx, h)
            sector_damage = self.damage_func(sector_idx, h)
            total = sector_cost + sector_damage

            if total < best_cost:
                best_cost = total
                best_h = h

        return best_h

    def optimize(self, num_sectors: int) -> np.ndarray:
        """
        Optimize all sectors independently.

        Args:
            num_sectors: Number of sectors

        Returns:
            Height configuration
        """
        heights = np.zeros(num_sectors)
        for i in range(num_sectors):
            heights[i] = self.optimize_sector(i, num_sectors)
        return heights


class AlgorithmComparator:
    """
    Compare multiple optimization algorithms on the same problem.
    """

    def __init__(self, cost_func: Callable):
        """
        Initialize comparator.

        Args:
            cost_func: Cost evaluation function
        """
        self.cost_func = cost_func
        self.results = {}

    def run_genetic_algorithm(self, num_sectors: int,
                            **kwargs) -> AlgorithmResult:
        """Run GA and store results"""
        import time
        start = time.time()

        ga = GeneticAlgorithm(self.cost_func, **kwargs)
        heights = ga.optimize(num_sectors)

        elapsed = time.time() - start
        cost = self.cost_func(heights)

        result = AlgorithmResult(
            name="Genetic Algorithm",
            heights=heights,
            total_cost=cost,
            construction_cost=0,  # To be filled
            expected_damage=0,    # To be filled
            convergence_history=ga.convergence_history,
            computation_time=elapsed,
            metadata={'population_size': kwargs.get('population_size', 50),
                     'generations': kwargs.get('generations', 100)}
        )
        self.results['GA'] = result
        return result

    def run_pso(self, num_sectors: int,
               **kwargs) -> AlgorithmResult:
        """Run PSO and store results"""
        import time
        start = time.time()

        pso = ParticleSwarmOptimizer(self.cost_func, **kwargs)
        heights = pso.optimize(num_sectors)

        elapsed = time.time() - start
        cost = self.cost_func(heights)

        result = AlgorithmResult(
            name="Particle Swarm Optimization",
            heights=heights,
            total_cost=cost,
            construction_cost=0,
            expected_damage=0,
            convergence_history=pso.convergence_history,
            computation_time=elapsed,
            metadata={'num_particles': kwargs.get('num_particles', 30),
                     'iterations': kwargs.get('iterations', 100)}
        )
        self.results['PSO'] = result
        return result

    def run_uniform_baseline(self, num_sectors: int) -> AlgorithmResult:
        """Run uniform height baseline"""
        import time
        start = time.time()

        h_optimal, min_cost = UniformDesignBaseline.compute_optimal_uniform_height(
            self.cost_func, num_sectors)

        elapsed = time.time() - start
        heights = np.full(num_sectors, h_optimal)

        result = AlgorithmResult(
            name="Uniform Design",
            heights=heights,
            total_cost=min_cost,
            construction_cost=0,
            expected_damage=0,
            convergence_history=[min_cost],  # Single evaluation
            computation_time=elapsed,
            metadata={'uniform_height': h_optimal}
        )
        self.results['Uniform'] = result
        return result

    def compare_all(self, num_sectors: int,
                   ga_params: Dict = None,
                   pso_params: Dict = None) -> Dict:
        """
        Run all algorithms and return comparison results.

        Args:
            num_sectors: Number of sectors
            ga_params: Parameters for GA
            pso_params: Parameters for PSO

        Returns:
            Dictionary with all results
        """
        if ga_params is None:
            ga_params = {'population_size': 50, 'generations': 100}
        if pso_params is None:
            pso_params = {'num_particles': 30, 'iterations': 100}

        print(f"Comparing algorithms on {num_sectors}-sector problem...")

        # Run all algorithms
        print("  Running Genetic Algorithm...")
        self.run_genetic_algorithm(num_sectors, **ga_params)

        print("  Running Particle Swarm Optimization...")
        self.run_pso(num_sectors, **pso_params)

        print("  Running Uniform Design baseline...")
        self.run_uniform_baseline(num_sectors)

        # Summary
        summary = {}
        for name, result in self.results.items():
            summary[name] = {
                'total_cost': float(result.total_cost),
                'computation_time': float(result.computation_time),
                'mean_height': float(np.mean(result.heights)),
                'height_range': (float(np.min(result.heights)),
                               float(np.max(result.heights))),
                'convergence_iterations': len(result.convergence_history)
            }

        return summary
